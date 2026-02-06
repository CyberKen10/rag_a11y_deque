from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import List

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from .retrieval import RetrievedChunk


LOW_CONFIDENCE_THRESHOLD = 0.05
HF_INFERENCE_URL_TEMPLATES = (
    "https://router.huggingface.co/hf-inference/models/{model}",
    "https://api-inference.huggingface.co/models/{model}",
)
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"


def _build_context(retrieved: List[RetrievedChunk]) -> str:
    blocks = []
    for item in retrieved:
        blocks.append(
            "\n".join(
                [
                    f"FUENTE: {item.chunk.source_file}",
                    f"SECCION: {item.chunk.section}",
                    f"SCORE: {item.score:.3f}",
                    "TEXTO:",
                    item.chunk.text,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def _build_evidence_lines(retrieved: List[RetrievedChunk]) -> List[str]:
    evidence_lines = []
    for item in retrieved:
        snippet = item.chunk.text[:420].replace("\n", " ").strip()
        evidence_lines.append(
            f"- ({item.score:.3f}) [{item.chunk.source_file} | {item.chunk.section}] {snippet}"
        )
    return evidence_lines


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9áéíóúñü-]", "", token.lower())


def _extract_query_tokens(question: str) -> set[str]:
    tokens: set[str] = set()
    for raw in question.split():
        token = _normalize_token(raw)
        if not token:
            continue
        if len(token) > 2:
            tokens.add(token)
        if "-" in token:
            parts = [part for part in token.split("-") if len(part) > 2]
            tokens.update(parts)
    return tokens


def _is_aria_live_question(question: str) -> bool:
    q = question.lower()
    return "aria-live" in q or "aria live" in q


def _extract_aria_live_guidance(retrieved: List[RetrievedChunk]) -> List[str]:
    hints: List[str] = []
    for item in retrieved:
        for sentence in _sentence_candidates(item.chunk.text):
            lower = sentence.lower()
            if "aria live" in lower or "aria-live" in lower:
                hints.append(sentence)
            elif "alerta" in lower and ("anunciar" in lower or "dom" in lower):
                hints.append(sentence)
    deduped = list(dict.fromkeys(hints))
    return deduped[:3]


def _sentence_candidates(text: str) -> List[str]:
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    return [s.strip() for s in sentences if s.strip()]


def _build_local_grounded_answer(question: str, retrieved: List[RetrievedChunk]) -> str:
    if _is_aria_live_question(question):
        aria_live_hints = _extract_aria_live_guidance(retrieved)
        if aria_live_hints:
            return "\n".join(f"- {line}" for line in aria_live_hints)

    query_tokens = _extract_query_tokens(question)
    ranked_sentences: List[tuple[float, str]] = []

    for item in retrieved[:5]:
        for sentence in _sentence_candidates(item.chunk.text):
            sentence_tokens = {
                token
                for raw in sentence.split()
                if (token := _normalize_token(raw))
            }
            overlap = len(query_tokens & sentence_tokens)
            has_exact_phrase = "aria-live" in sentence.lower() and "aria-live" in question.lower()
            bonus = 3 if has_exact_phrase else 0
            score = item.score + overlap * 0.07 + bonus
            ranked_sentences.append((score, sentence))

    ranked_sentences.sort(key=lambda x: x[0], reverse=True)
    best = [sentence for _, sentence in ranked_sentences[:3] if sentence]

    if not best:
        top_chunk = retrieved[0].chunk
        return (
            f"Según el documento '{top_chunk.source_file}' (sección '{top_chunk.section}'): \n\n"
            f"{top_chunk.text[:500]}..."
        )

    return "\n".join(f"- {line}" for line in best)


def _extract_generated_text(body: object) -> str:
    if isinstance(body, list) and body:
        first = body[0]
        if isinstance(first, dict) and "generated_text" in first:
            return str(first["generated_text"]).strip()

    if isinstance(body, dict):
        if "generated_text" in body:
            return str(body["generated_text"]).strip()
        if "error" in body:
            raise RuntimeError(f"Hugging Face devolvió error: {body['error']}")

    raise RuntimeError("Respuesta no reconocida de Hugging Face Inference API.")


def _answer_with_huggingface(
    question: str,
    retrieved: List[RetrievedChunk],
    model: str,
    api_key: str,
) -> str:
    context = _build_context(retrieved)
    prompt = (
        "Eres un asistente de accesibilidad. Responde EXCLUSIVAMENTE con información del CONTEXTO.\n"
        "Si la respuesta no está en el contexto, responde exactamente: 'No encontrado en la base documental'.\n"
        "No inventes información. Responde en español, claro y breve.\n\n"
        f"PREGUNTA:\n{question}\n\n"
        f"CONTEXTO:\n{context}\n\n"
        "RESPUESTA:"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 220,
            "temperature": 0.1,
            "return_full_text": False,
        },
        "options": {
            "wait_for_model": True,
            "use_cache": False,
        },
    }

    base_override = os.getenv("HF_INFERENCE_BASE_URL")
    endpoints: List[str]
    if base_override:
        endpoints = [base_override.rstrip("/") + f"/models/{model}"]
    else:
        endpoints = [template.format(model=model) for template in HF_INFERENCE_URL_TEMPLATES]

    errors: List[str] = []
    for endpoint in endpoints:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
            return _extract_generated_text(body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            errors.append(f"{endpoint} -> HTTP {exc.code}: {detail}")
            if exc.code == 410:
                continue
            raise RuntimeError(f"Error HTTP de Hugging Face ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            errors.append(f"{endpoint} -> conexión fallida: {exc.reason}")

    joined_errors = " | ".join(errors) if errors else "sin detalles"
    raise RuntimeError(f"No se pudo usar Hugging Face Router/Inference: {joined_errors}")


def _answer_with_grok(
    question: str,
    retrieved: List[RetrievedChunk],
) -> str:
    """Responde usando la API gratuita de Grok."""
    context = _build_context(retrieved)
    messages = [
        {
            "role": "system",
            "content": "Eres un asistente de accesibilidad. Responde EXCLUSIVAMENTE con información del CONTEXTO proporcionado. Si la respuesta no está en el contexto, responde exactamente: 'No encontrado en la base documental'. No inventes información. Responde en español, claro y breve."
        },
        {
            "role": "user",
            "content": f"PREGUNTA:\n{question}\n\nCONTEXTO:\n{context}\n\nRESPUESTA:"
        }
    ]

    payload = {
        "messages": messages,
        "model": "grok-beta",
        "stream": False,
        "temperature": 0.1
    }

    req = urllib.request.Request(
        GROK_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))

        if "choices" in body and body["choices"]:
            return body["choices"][0]["message"]["content"].strip()
        else:
            raise RuntimeError("Respuesta inesperada de Grok API")

    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Error HTTP de Grok ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se pudo conectar con Grok: {exc.reason}") from exc


def _answer_with_together(
    question: str,
    retrieved: List[RetrievedChunk],
) -> str:
    """Responde usando la API de Together AI."""
    context = _build_context(retrieved)
    messages = [
        {
            "role": "system",
            "content": "Eres un asistente de accesibilidad. Responde EXCLUSIVAMENTE con información del CONTEXTO proporcionado. Si la respuesta no está en el contexto, responde exactamente: 'No encontrado en la base documental'. No inventes información. Responde en español, claro y breve."
        },
        {
            "role": "user",
            "content": f"PREGUNTA:\n{question}\n\nCONTEXTO:\n{context}\n\nRESPUESTA:"
        }
    ]

    payload = {
        "model": "meta-llama/Llama-2-7b-chat-hf",  # Modelo gratuito disponible en Together AI
        "messages": messages,
        "max_tokens": 220,
        "temperature": 0.1,
    }

    req = urllib.request.Request(
        TOGETHER_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer ",  # API gratuita, no necesita key
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))

        if "choices" in body and body["choices"]:
            return body["choices"][0]["message"]["content"].strip()
        else:
            raise RuntimeError("Respuesta inesperada de Together AI API")

    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Error HTTP de Together AI ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se pudo conectar con Together AI: {exc.reason}") from exc


def build_grounded_answer(question: str, retrieved: List[RetrievedChunk], model: str, api_key: str | None) -> str:
    if not retrieved:
        return "No encontré información en tu base de conocimiento para responder."

    if retrieved[0].score < LOW_CONFIDENCE_THRESHOLD:
        return (
            "No tengo evidencia suficiente en los documentos para responder con confianza. "
            "Prueba reformular la pregunta o agregar más documentación."
        )

    effective_api_key = api_key or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")

    # Intentar primero con API de Hugging Face y, si no está disponible, usar síntesis local.
    answer_origin = "Hugging Face Inference API + RAG"
    try:
        if effective_api_key:
            llm_answer = _answer_with_huggingface(
                question,
                retrieved,
                model=model,
                api_key=effective_api_key,
            )
        else:
            raise RuntimeError("No API key available")
    except Exception as e:
        print(f"Hugging Face falló ({e}), usando síntesis local basada en documentos...")
        llm_answer = _build_local_grounded_answer(question, retrieved)
        answer_origin = "Síntesis local + RAG"
    evidence_lines = _build_evidence_lines(retrieved)

    return (
        f"Respuesta (generada con {answer_origin}):\n"
        f"{llm_answer}\n\n"
        "Trazabilidad (fuentes usadas):\n"
        + "\n".join(evidence_lines)
    )

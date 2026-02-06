from __future__ import annotations

import json
import os
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
HF_INFERENCE_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{model}"
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

    req = urllib.request.Request(
        HF_INFERENCE_URL_TEMPLATE.format(model=model),
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
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Error HTTP de Hugging Face ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se pudo conectar con Hugging Face: {exc.reason}") from exc

    return _extract_generated_text(body)


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

    # Intentar primero con API de Hugging Face, luego Together AI, luego Grok, finalmente respuesta simple
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
        print(f"Hugging Face falló ({e}), intentando con Together AI...")
        try:
            llm_answer = _answer_with_together(
                question,
                retrieved,
            )
        except Exception as e2:
            print(f"Together AI también falló ({e2}), intentando con Grok...")
            try:
                llm_answer = _answer_with_grok(
                    question,
                    retrieved,
                )
            except Exception as e3:
                print(f"Todas las APIs fallaron ({e3}), usando respuesta simple basada en documentos...")
                # Usar respuesta simple basada en los documentos recuperados
                if retrieved:
                    # Tomar el chunk más relevante y extraer información
                    top_chunk = retrieved[0].chunk
                    llm_answer = f"Según el documento '{top_chunk.source_file}' (sección '{top_chunk.section}'):\n\n{top_chunk.text[:500]}..."
                else:
                    llm_answer = "No encontré información relevante en los documentos para responder a tu pregunta."
    evidence_lines = _build_evidence_lines(retrieved)

    return (
        "Respuesta (generada con Hugging Face Inference API + RAG):\n"
        f"{llm_answer}\n\n"
        "Trazabilidad (fuentes usadas):\n"
        + "\n".join(evidence_lines)
    )

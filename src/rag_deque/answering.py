from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import List

from .retrieval import RetrievedChunk


LOW_CONFIDENCE_THRESHOLD = 0.05
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


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


def _answer_with_mistral(question: str, retrieved: List[RetrievedChunk], model: str, api_key: str) -> str:
    context = _build_context(retrieved)
    system_prompt = (
        "Eres un asistente de accesibilidad. Responde EXCLUSIVAMENTE con información del CONTEXTO. "
        "Si la respuesta no está en el contexto, responde exactamente: 'No encontrado en la base documental'. "
        "No inventes información. Responde en español, de forma clara y breve."
    )
    user_prompt = f"PREGUNTA:\n{question}\n\nCONTEXTO:\n{context}"

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    req = urllib.request.Request(
        MISTRAL_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Error HTTP de Mistral AI ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se pudo conectar con Mistral AI: {exc.reason}") from exc

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError("Mistral AI no devolvió respuestas (choices vacío).")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, list):
        text_parts = [c.get("text", "") for c in content if isinstance(c, dict)]
        content = "\n".join([t for t in text_parts if t])

    return str(content).strip()


def build_grounded_answer(question: str, retrieved: List[RetrievedChunk], model: str, api_key: str | None) -> str:
    if not retrieved:
        return "No encontré información en tu base de conocimiento para responder."

    if retrieved[0].score < LOW_CONFIDENCE_THRESHOLD:
        return (
            "No tengo evidencia suficiente en los documentos para responder con confianza. "
            "Prueba reformular la pregunta o agregar más documentación."
        )

    effective_api_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not effective_api_key:
        raise RuntimeError(
            "Falta API key de Mistral AI. Define MISTRAL_API_KEY o usa --mistral-api-key."
        )

    llm_answer = _answer_with_mistral(
        question,
        retrieved,
        model=model,
        api_key=effective_api_key,
    )
    evidence_lines = _build_evidence_lines(retrieved)

    return (
        "Respuesta (generada con Mistral AI + RAG):\n"
        f"{llm_answer}\n\n"
        "Trazabilidad (fuentes usadas):\n"
        + "\n".join(evidence_lines)
    )

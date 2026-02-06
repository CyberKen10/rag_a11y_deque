from __future__ import annotations

import argparse
from pathlib import Path

from .answering import build_grounded_answer
from .retrieval import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Hace preguntas al índice RAG local.")
    parser.add_argument("--index-dir", default=Path("data/index"), type=Path)
    parser.add_argument("--question", required=True, type=str)
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument(
        "--mistral-model",
        default="mistral-small-latest",
        type=str,
        help="Modelo de Mistral AI (API externa con plan gratuito y límites).",
    )
    parser.add_argument(
        "--mistral-api-key",
        default=None,
        type=str,
        help="API key de Mistral AI (opcional si usas variable de entorno MISTRAL_API_KEY).",
    )
    args = parser.parse_args()

    retriever = Retriever(args.index_dir)
    retrieved = retriever.search(args.question, top_k=args.top_k)
    print(
        build_grounded_answer(
            args.question,
            retrieved,
            model=args.mistral_model,
            api_key=args.mistral_api_key,
        )
    )


if __name__ == "__main__":
    main()

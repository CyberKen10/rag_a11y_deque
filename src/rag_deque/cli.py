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
        "--hf-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        type=str,
        help="Modelo remoto de Hugging Face Inference API (sin instalación local).",
    )
    parser.add_argument(
        "--hf-api-key",
        default=None,
        type=str,
        help="API key de Hugging Face (opcional si usas HF_API_KEY/HUGGINGFACE_API_KEY).",
    )
    args = parser.parse_args()

    retriever = Retriever(args.index_dir)
    retrieved = retriever.search(args.question, top_k=args.top_k)
    print(
        build_grounded_answer(
            args.question,
            retrieved,
            model=args.hf_model,
            api_key=args.hf_api_key,
        )
    )


if __name__ == "__main__":
    main()

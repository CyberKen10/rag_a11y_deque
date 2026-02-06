from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List
import xml.etree.ElementTree as ET
import zipfile

from .models import Chunk

ALLOWED_EXTENSIONS = {".docx"}

# Modelo de embeddings: all-MiniLM-L6-v2 (más pequeño y rápido, solo inglés pero funciona bien)
# Alternativas: "paraphrase-multilingual-MiniLM-L12-v2" (multilenguaje, más pesado)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def iter_docx_files(input_dir: Path) -> Iterable[Path]:
    for file_path in sorted(input_dir.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
            yield file_path


def extract_text_from_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml_content = zf.read("word/document.xml")

    root = ET.fromstring(xml_content)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    paragraphs: List[str] = []
    for p in root.findall(".//w:p", ns):
        texts = [t.text for t in p.findall(".//w:t", ns) if t.text]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def split_into_chunks(text: str, max_chars: int = 600, overlap_chars: int = 100) -> List[str]:
    """Divide texto en chunks con overlap para no perder contexto en los bordes."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current, current_len = [], 0

    for p in paragraphs:
        if current and current_len + len(p) + 1 > max_chars:
            chunks.append("\n".join(current))
            # Overlap: mantener los últimos párrafos que quepan en overlap_chars
            overlap, overlap_len = [], 0
            for prev_p in reversed(current):
                if overlap_len + len(prev_p) + 1 > overlap_chars:
                    break
                overlap.insert(0, prev_p)
                overlap_len += len(prev_p) + 1
            current, current_len = overlap + [p], overlap_len + len(p) + 1
        else:
            current.append(p)
            current_len += len(p) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def build_index(input_dir: Path, output_dir: Path) -> None:
    """
    Indexa documentos usando embeddings semánticos (sentence-transformers).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers no está instalado. Ejecuta: pip install sentence-transformers"
        )

    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Modelo cargado.")

    all_chunks: List[Chunk] = []
    for docx_file in iter_docx_files(input_dir):
        text = extract_text_from_docx(docx_file)
        for i, chunk_text in enumerate(split_into_chunks(text)):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{docx_file.stem}-{i}",
                    source_file=str(docx_file),
                    section="Documento",
                    text=chunk_text,
                )
            )

    if not all_chunks:
        raise ValueError("No se encontraron archivos .docx con contenido para indexar.")

    print(f"Generando embeddings para {len(all_chunks)} chunks...")
    chunk_texts = [c.text for c in all_chunks]
    embeddings = model.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)
    print("Embeddings generados.")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar chunks
    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    # Guardar embeddings (como lista de listas para JSON)
    import numpy as np
    with (output_dir / "embeddings.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": EMBEDDING_MODEL_NAME,
                "embeddings": embeddings.tolist(),
                "dimension": embeddings.shape[1],
            },
            f,
            ensure_ascii=False,
        )

    print(f"Índice guardado en: {output_dir}")
    print(f"  - {len(all_chunks)} chunks")
    print(f"  - Dimensión de embeddings: {embeddings.shape[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa documentos Word (.docx) para RAG con embeddings.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Carpeta con .docx")
    parser.add_argument("--output-dir", default=Path("data/index"), type=Path)
    args = parser.parse_args()
    build_index(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

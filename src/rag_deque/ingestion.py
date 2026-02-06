from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List
import xml.etree.ElementTree as ET

from .models import Chunk

ALLOWED_EXTENSIONS = {".docx"}
TOKEN_RE = re.compile(r"[a-záéíóúñ0-9]{2,}", re.IGNORECASE)


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


def split_into_chunks(text: str, max_chars: int = 900) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current, current_len = [], 0

    for p in paragraphs:
        if current and current_len + len(p) + 1 > max_chars:
            chunks.append("\n".join(current))
            current, current_len = [p], len(p)
        else:
            current.append(p)
            current_len += len(p) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    vec = {term: (freq / total) * idf.get(term, 0.0) for term, freq in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def build_index(input_dir: Path, output_dir: Path) -> None:
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

    tokenized_docs = [tokenize(c.text) for c in all_chunks]
    doc_freq: Counter = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    n_docs = len(tokenized_docs)
    idf = {term: math.log((1 + n_docs) / (1 + df)) + 1 for term, df in doc_freq.items()}
    vectors = [tfidf_vector(tokens, idf) for tokens in tokenized_docs]

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    with (output_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump({"idf": idf, "vectors": vectors}, f, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa documentos Word (.docx) para RAG.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Carpeta con .docx")
    parser.add_argument("--output-dir", default=Path("data/index"), type=Path)
    args = parser.parse_args()
    build_index(args.input_dir, args.output_dir)
    print(f"Índice generado en: {args.output_dir}")


if __name__ == "__main__":
    main()

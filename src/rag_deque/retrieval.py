from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .ingestion import tfidf_vector, tokenize
from .models import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


class Retriever:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self._load()

    def _load(self) -> None:
        with (self.index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
            self.chunks = [Chunk(**json.loads(line)) for line in f]

        with (self.index_dir / "index.json").open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.idf = payload["idf"]
        self.vectors = payload["vectors"]

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        qv = tfidf_vector(tokenize(query), self.idf)
        scored = [RetrievedChunk(chunk=chunk, score=cosine_sparse(qv, vec)) for chunk, vec in zip(self.chunks, self.vectors)]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

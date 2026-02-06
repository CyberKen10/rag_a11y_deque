from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .models import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self._load()

    def _load(self) -> None:
        """Carga chunks y embeddings del índice."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            raise ImportError(
                "sentence-transformers no está instalado. Ejecuta: pip install sentence-transformers numpy"
            )

        # Cargar chunks
        with (self.index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
            self.chunks = [Chunk(**json.loads(line)) for line in f]

        # Cargar embeddings
        with (self.index_dir / "embeddings.json").open("r", encoding="utf-8") as f:
            payload = json.load(f)
            self.model_name = payload["model"]
            self.embeddings = np.array(payload["embeddings"], dtype=np.float32)

        # Cargar modelo para queries
        print(f"Cargando modelo de embeddings: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("Modelo cargado.")

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calcula similitud de coseno entre dos vectores."""
        import numpy as np
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Busca chunks relevantes usando similitud semántica (embeddings).
        """
        import numpy as np

        # Generar embedding de la query
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]

        # Calcular similitudes con todos los chunks
        scored: List[RetrievedChunk] = []
        for i, chunk in enumerate(self.chunks):
            similarity = self._cosine_similarity(query_embedding, self.embeddings[i])
            scored.append(RetrievedChunk(chunk=chunk, score=similarity))

        # Ordenar por score descendente
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

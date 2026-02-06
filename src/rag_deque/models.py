from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    section: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

"""Small serializable types used by the standalone LooGLE runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ChunkRecord:
    doc_id: str
    chunk_id: str
    chunk_index: int
    raw_text: str
    token_start: int | None = None
    token_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalExample:
    query_id: str
    doc_id: str
    question: str
    gold_chunk_ids: list[str] = field(default_factory=list)
    silver_chunk_ids: list[str] = field(default_factory=list)
    silver_chunk_groups: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalResult:
    query_id: str
    doc_id: str
    question: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_indices: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)
    silver_chunk_ids: list[str] = field(default_factory=list)
    silver_chunk_groups: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

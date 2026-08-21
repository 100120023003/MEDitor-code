from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentRecord:
    doc_id: str
    source: str
    title: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    chunk_index: int
    token_count: int
    char_start: int
    char_end: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalHit:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    score: float
    rank: int
    retriever: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CorpusManifest:
    corpus_name: str
    version: str
    created_at: str
    doc_count: int
    chunk_count: int
    sources: List[str]
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

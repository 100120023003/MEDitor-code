from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .io_utils import ensure_parent_dir, iter_jsonl, load_jsonl, write_jsonl
from .schema import ChunkRecord, DocumentRecord


_TOKEN_RE = re.compile(r"\S+")


@dataclass
class ChunkingConfig:
    chunk_size: int = 384
    chunk_overlap: int = 96
    min_tokens: int = 32


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _token_spans(text: str) -> List[Tuple[int, int]]:
    return [match.span() for match in _TOKEN_RE.finditer(text or "")]


def chunk_document(doc: DocumentRecord, config: ChunkingConfig) -> List[ChunkRecord]:
    text = str(doc.text or "")
    spans = _token_spans(text)
    if not spans:
        return []

    chunk_size = max(1, int(config.chunk_size))
    overlap = max(0, min(int(config.chunk_overlap), chunk_size - 1))
    step = max(1, chunk_size - overlap)
    chunks: List[ChunkRecord] = []

    for chunk_idx, start_token in enumerate(range(0, len(spans), step)):
        end_token = min(start_token + chunk_size, len(spans))
        if end_token - start_token < int(config.min_tokens) and chunks:
            break
        char_start = spans[start_token][0]
        char_end = spans[end_token - 1][1]
        chunk_text = _normalize_space(text[char_start:char_end])
        if not chunk_text:
            continue
        chunk_id = f"{doc.doc_id}#chunk{chunk_idx:04d}"
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                source=doc.source,
                title=_normalize_space(doc.title),
                text=chunk_text,
                chunk_index=chunk_idx,
                token_count=end_token - start_token,
                char_start=char_start,
                char_end=char_end,
                meta=dict(doc.meta or {}),
            )
        )
        if end_token >= len(spans):
            break
    return chunks


def chunk_documents(docs: Sequence[DocumentRecord], config: ChunkingConfig) -> List[ChunkRecord]:
    chunks: List[ChunkRecord] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, config))
    return chunks


def chunk_documents_file(
    documents_path: str,
    output_chunks_path: str,
    config: ChunkingConfig,
) -> List[ChunkRecord]:
    chunks: List[ChunkRecord] = []
    ensure_parent_dir(output_chunks_path)
    with open(output_chunks_path, "w", encoding="utf-8") as f:
        for row in iter_jsonl(documents_path):
            doc = DocumentRecord(**row)
            doc_chunks = chunk_document(doc, config)
            chunks.extend(doc_chunks)
            for chunk in doc_chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return chunks

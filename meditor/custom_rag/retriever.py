from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .schema import RetrievalHit


def render_options(options: Optional[Dict[str, str]]) -> str:
    if not options:
        return ""
    lines = []
    for key in sorted(options.keys()):
        value = str(options[key]).strip()
        if value:
            lines.append(f"{key}. {value}")
    return "\n".join(lines)


def build_retrieval_query(
    question: str,
    options: Optional[Dict[str, str]] = None,
    mode: str = "question_only",
) -> str:
    mode = str(mode or "question_only").strip().lower()
    if mode == "question_plus_options" and options:
        options_text = render_options(options)
        if options_text:
            return f"{str(question or '').strip()}\n\nOptions:\n{options_text}".strip()
    return str(question or "").strip()


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievalHit]],
    top_k: int = 32,
    rrf_k: int = 60,
) -> List[RetrievalHit]:
    fused: Dict[str, RetrievalHit] = {}
    fused_scores: Dict[str, float] = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list, start=1):
            key = str(hit.chunk_id)
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (float(rrf_k) + rank))
            if key not in fused:
                fused[key] = RetrievalHit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    source=hit.source,
                    title=hit.title,
                    text=hit.text,
                    score=0.0,
                    rank=0,
                    retriever="rrf",
                    meta=dict(hit.meta or {}),
                )

    ranked_ids = sorted(fused_scores.keys(), key=lambda key: fused_scores[key], reverse=True)[: max(1, int(top_k))]
    merged: List[RetrievalHit] = []
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        item = fused[chunk_id]
        item.score = float(fused_scores[chunk_id])
        item.rank = rank
        merged.append(item)
    return merged


class HybridRetriever:
    def __init__(self, bm25=None, dense=None) -> None:
        self.bm25 = bm25
        self.dense = dense

    def retrieve(
        self,
        question: str,
        options: Optional[Dict[str, str]] = None,
        top_k: int = 32,
        sparse_k: int = 64,
        dense_k: int = 64,
        query_mode: str = "question_only",
        sparse_query_mode: Optional[str] = None,
        dense_query_mode: Optional[str] = None,
        fusion: str = "rrf",
        rrf_k: int = 60,
    ) -> Tuple[List[RetrievalHit], Dict[str, object]]:
        sparse_mode = str(sparse_query_mode or query_mode or "question_only").strip().lower()
        dense_mode = str(dense_query_mode or query_mode or "question_only").strip().lower()
        sparse_query = build_retrieval_query(question, options=options, mode=sparse_mode)
        dense_query = build_retrieval_query(question, options=options, mode=dense_mode)
        sparse_hits = self.bm25.search(sparse_query, top_k=sparse_k) if self.bm25 is not None else []
        dense_hits = self.dense.search(dense_query, top_k=dense_k) if self.dense is not None else []

        if fusion == "rrf" and sparse_hits and dense_hits:
            hits = reciprocal_rank_fusion([sparse_hits, dense_hits], top_k=top_k, rrf_k=rrf_k)
        elif dense_hits and not sparse_hits:
            hits = dense_hits[: max(1, int(top_k))]
        else:
            hits = sparse_hits[: max(1, int(top_k))]

        trace = {
            "query": sparse_query if sparse_mode == dense_mode else "",
            "query_mode": query_mode,
            "sparse_query_mode": sparse_mode,
            "dense_query_mode": dense_mode,
            "sparse_query": sparse_query,
            "dense_query": dense_query,
            "fusion": fusion,
            "sparse_hits": [hit.to_dict() for hit in sparse_hits[: min(8, len(sparse_hits))]],
            "dense_hits": [hit.to_dict() for hit in dense_hits[: min(8, len(dense_hits))]],
            "final_hits": [hit.to_dict() for hit in hits[: min(8, len(hits))]],
        }
        return hits, trace

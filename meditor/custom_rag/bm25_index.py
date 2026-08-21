from __future__ import annotations

import math
import os
import pickle
import re
from collections import Counter, defaultdict
from typing import Dict, List

from .io_utils import ensure_dir, load_json, load_jsonl, resolve_index_data_path, write_json
from .schema import RetrievalHit


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(str(text or ""))]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)

    def build(self, chunks_path: str, index_dir: str) -> Dict[str, object]:
        ensure_dir(index_dir)
        chunks = load_jsonl(chunks_path)
        postings: Dict[str, List[List[float]]] = defaultdict(list)
        doc_lengths: List[int] = []

        for idx, row in enumerate(chunks):
            terms = _tokenize(f"{row.get('title', '')} {row.get('text', '')}")
            tf = Counter(terms)
            doc_lengths.append(len(terms))
            for term, freq in tf.items():
                postings[term].append([idx, int(freq)])

        total_docs = len(chunks)
        avgdl = (sum(doc_lengths) / total_docs) if total_docs else 0.0
        for term, plist in postings.items():
            doc_freq = len(plist)
            idf = math.log(1.0 + ((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))) if total_docs else 0.0
            postings[term] = [[doc_idx, freq, idf] for doc_idx, freq in plist]

        meta = {
            "chunks_path": chunks_path,
            "doc_count": total_docs,
            "avgdl": avgdl,
            "k1": self.k1,
            "b": self.b,
        }
        with open(os.path.join(index_dir, "postings.pkl"), "wb") as f:
            pickle.dump(dict(postings), f, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(os.path.join(index_dir, "meta.json"), meta)
        return meta


class BM25Searcher:
    def __init__(self, index_dir: str) -> None:
        self.index_dir = index_dir
        self.meta = load_json(os.path.join(index_dir, "meta.json"))
        with open(os.path.join(index_dir, "postings.pkl"), "rb") as f:
            self.postings: Dict[str, List[List[float]]] = pickle.load(f)
        chunks_path = resolve_index_data_path(index_dir, str(self.meta.get("chunks_path", "")), "chunks.jsonl")
        self.meta["resolved_chunks_path"] = chunks_path
        self.chunks = load_jsonl(chunks_path)
        self.doc_lengths = [len(_tokenize(f"{row.get('title', '')} {row.get('text', '')}")) for row in self.chunks]
        self.avgdl = float(self.meta.get("avgdl", 0.0) or 0.0)
        self.k1 = float(self.meta.get("k1", 1.5) or 1.5)
        self.b = float(self.meta.get("b", 0.75) or 0.75)

    def search(self, query: str, top_k: int = 32) -> List[RetrievalHit]:
        terms = _tokenize(query)
        if not terms:
            return []

        scores: Dict[int, float] = defaultdict(float)
        for term in terms:
            plist = self.postings.get(term)
            if not plist:
                continue
            for doc_idx, freq, idf in plist:
                doc_len = self.doc_lengths[int(doc_idx)] if int(doc_idx) < len(self.doc_lengths) else 0
                denom = float(freq) + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl)) if self.avgdl else 1.0
                score = float(idf) * ((float(freq) * (self.k1 + 1.0)) / max(1e-12, denom))
                scores[int(doc_idx)] += score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: max(1, int(top_k))]
        hits: List[RetrievalHit] = []
        for rank, (doc_idx, score) in enumerate(ranked, start=1):
            row = self.chunks[doc_idx]
            hits.append(
                RetrievalHit(
                    chunk_id=str(row["chunk_id"]),
                    doc_id=str(row["doc_id"]),
                    source=str(row["source"]),
                    title=str(row.get("title", "")),
                    text=str(row.get("text", "")),
                    score=float(score),
                    rank=rank,
                    retriever="bm25",
                    meta=dict(row.get("meta") or {}),
                )
            )
        return hits

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from .io_utils import ensure_dir, load_json, load_jsonl, resolve_index_data_path, resolve_model_ref, write_json
from .schema import RetrievalHit


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def _is_medcpt_model(model_ref: str) -> bool:
    text = str(model_ref or "")
    return "medcpt" in text.lower() or "medcpt" in os.path.basename(text).lower()


def _build_corpus_inputs(chunks: List[Dict[str, Any]], model_name: str) -> List[Any]:
    if _is_medcpt_model(model_name):
        return [
            [str(row.get("title", "") or "").strip(), str(row.get("text", "") or "").strip()]
            for row in chunks
        ]
    return [f"{row.get('title', '')}\n{row.get('text', '')}".strip() for row in chunks]


def _build_query_inputs(query: str, model_name: str) -> List[Any]:
    return [str(query or "").strip()]


def _load_sentence_transformer_classes():
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer

    class _ClsSentenceTransformer(SentenceTransformer):
        def _load_auto_model(self, model_name_or_path, *args, **kwargs):
            print(
                f"No sentence-transformers model found with name {model_name_or_path}. "
                "Creating a new one with CLS pooling."
            )
            token = kwargs.get("token", None)
            cache_folder = kwargs.get("cache_folder", None)
            revision = kwargs.get("revision", None)
            trust_remote_code = kwargs.get("trust_remote_code", False)
            local_files_only = kwargs.get("local_files_only", False)

            transformer_kwargs = {
                "token": token,
                "cache_dir": cache_folder,
                "revision": revision,
                "trust_remote_code": trust_remote_code,
                "local_files_only": local_files_only,
            }
            transformer_kwargs = {k: v for k, v in transformer_kwargs.items() if v not in {None, False}}
            transformer_model = Transformer(model_name_or_path, **transformer_kwargs)
            try:
                pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), "cls")
            except TypeError:
                pooling_model = Pooling(
                    transformer_model.get_word_embedding_dimension(),
                    pooling_mode="cls",
                )
            return [transformer_model, pooling_model]

    return SentenceTransformer, _ClsSentenceTransformer


def _load_encoder(model_name: str, device: Optional[str]):
    try:
        SentenceTransformer, ClsSentenceTransformer = _load_sentence_transformer_classes()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("sentence-transformers is required to build/query dense indexes.") from exc

    if _is_medcpt_model(model_name):
        return ClsSentenceTransformer(model_name, device=device)
    return SentenceTransformer(model_name, device=device)


def _validate_device(device: Optional[str]) -> None:
    requested = str(device or "").strip().lower()
    if not requested.startswith("cuda"):
        return
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required when device=cuda is requested.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            f"torch={torch.__version__}, torch_cuda={torch.version.cuda}. "
            "Check your NVIDIA driver / CUDA runtime compatibility or switch to --device cpu."
        )


def _resolve_query_model_ref(value: str, env_names: List[str]) -> str:
    resolved = resolve_model_ref(value)
    if resolved and os.path.exists(resolved):
        return resolved
    if "medcpt" not in str(value or "").lower():
        return resolved
    for env_name in env_names:
        env_value = str(os.environ.get(env_name, "") or "").strip()
        if not env_value:
            continue
        env_resolved = resolve_model_ref(env_value)
        if env_resolved and os.path.exists(env_resolved):
            print(
                f"[WARN] Dense index model path is missing: {resolved}. "
                f"Using {env_name}={env_resolved}.",
                flush=True,
            )
            return env_resolved
    return resolved


class DenseIndex:
    def build(
        self,
        chunks_path: str,
        index_dir: str,
        model_name: str,
        query_model_name: Optional[str] = None,
        batch_size: int = 64,
        device: Optional[str] = None,
        normalize: bool = True,
        use_faiss: bool = True,
    ) -> Dict[str, object]:
        resolved_model_name = resolve_model_ref(model_name)
        resolved_query_model_name = resolve_model_ref(query_model_name or model_name)

        _validate_device(device)
        ensure_dir(index_dir)
        chunks = load_jsonl(chunks_path)
        texts = _build_corpus_inputs(chunks, resolved_model_name)
        encoder = _load_encoder(resolved_model_name, device=device)
        embeddings = encoder.encode(
            texts,
            batch_size=max(1, int(batch_size)),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=bool(normalize),
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if normalize:
            embeddings = _normalize_rows(embeddings)
        np.save(os.path.join(index_dir, "embeddings.npy"), embeddings)

        faiss_written = False
        if bool(use_faiss):
            try:
                import faiss

                index = faiss.IndexFlatIP(int(embeddings.shape[1]))
                index.add(embeddings)
                faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
                faiss_written = True
            except Exception:
                faiss_written = False

        meta = {
            "chunks_path": chunks_path,
            "model_name": resolved_model_name,
            "query_model_name": resolved_query_model_name,
            "batch_size": int(batch_size),
            "normalize": bool(normalize),
            "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "faiss": bool(faiss_written),
            "pooling": "cls" if _is_medcpt_model(resolved_model_name) else "default",
            "corpus_input_format": "title_text_pair" if _is_medcpt_model(resolved_model_name) else "joined_text",
        }
        write_json(os.path.join(index_dir, "meta.json"), meta)
        return meta


class DenseSearcher:
    def __init__(self, index_dir: str, device: Optional[str] = None) -> None:
        self.index_dir = index_dir
        self.meta = load_json(os.path.join(index_dir, "meta.json"))
        chunks_path = resolve_index_data_path(index_dir, str(self.meta.get("chunks_path", "")), "chunks.jsonl")
        self.meta["resolved_chunks_path"] = chunks_path
        self.chunks = load_jsonl(chunks_path)
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy")).astype(np.float32)
        self.normalize = bool(self.meta.get("normalize", True))
        self.model_name = _resolve_query_model_ref(
            str(self.meta["model_name"]),
            ["MEDCPT_ARTICLE_MODEL", "MEDCPT_ARTICLE_ENCODER"],
        )
        self.query_model_name = _resolve_query_model_ref(
            str(self.meta.get("query_model_name") or self.model_name),
            ["MEDCPT_QUERY_MODEL", "MEDCPT_QUERY_ENCODER"],
        )
        self.device = device
        self._encoder = None
        self._faiss_index = None
        if bool(self.meta.get("faiss", False)):
            try:
                import faiss

                self._faiss_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
            except Exception:
                self._faiss_index = None
        _validate_device(self.device)

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = _load_encoder(self.query_model_name, device=self.device)
        return self._encoder

    def search(self, query: str, top_k: int = 32) -> List[RetrievalHit]:
        encoder = self._get_encoder()
        query_vec = encoder.encode(
            _build_query_inputs(query, self.query_model_name),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        query_vec = np.asarray(query_vec, dtype=np.float32)
        if self.normalize:
            query_vec = _normalize_rows(query_vec)

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_vec, max(1, int(top_k)))
            score_list = scores[0].tolist()
            index_list = indices[0].tolist()
        else:
            score_vec = np.matmul(self.embeddings, query_vec[0])
            top_indices = np.argsort(score_vec)[::-1][: max(1, int(top_k))]
            score_list = score_vec[top_indices].tolist()
            index_list = top_indices.tolist()

        hits: List[RetrievalHit] = []
        for rank, (idx, score) in enumerate(zip(index_list, score_list), start=1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            row = self.chunks[idx]
            hits.append(
                RetrievalHit(
                    chunk_id=str(row["chunk_id"]),
                    doc_id=str(row["doc_id"]),
                    source=str(row["source"]),
                    title=str(row.get("title", "")),
                    text=str(row.get("text", "")),
                    score=float(score),
                    rank=rank,
                    retriever="dense",
                    meta=dict(row.get("meta") or {}),
                )
            )
        return hits

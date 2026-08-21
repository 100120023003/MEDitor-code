from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .chunker import ChunkingConfig, chunk_document
from .cleaning import clean_document, default_cleaning_config, normalize_text, safe_doc_id, text_fingerprint
from .io_utils import count_jsonl_rows, ensure_dir, file_stem, iter_jsonl, load_jsonl, progress_bar, write_json, write_jsonl
from .schema import ChunkRecord, CorpusManifest, DocumentRecord

_TOKEN_RE = re.compile(r"\b\w+\b")


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _concat_title_text(title: str, text: str) -> str:
    title = _normalize_space(title)
    text = _normalize_space(text)
    if not title:
        return text
    if not text:
        return title
    if title.endswith((".", "?", "!")):
        return f"{title} {text}"
    return f"{title}. {text}"


def _suffix_index(value: str, default: int) -> int:
    match = re.search(r"(\d+)$", str(value or ""))
    return int(match.group(1)) if match else default


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(str(text or "")))


def _routing_meta(corpus_name: str, source: str) -> Dict[str, Any]:
    source = _normalize_space(source or "")
    corpus_name = _normalize_space(corpus_name or source or "custom_corpus")
    return {
        "corpus_name": corpus_name,
        "routing_arm": corpus_name,
        "routeable": True,
        "source_type": source or "generic",
    }


def _merge_overlapping_segments(
    segments: Sequence[str],
    min_overlap_chars: int = 48,
    max_overlap_probe: int = 400,
) -> Tuple[str, Dict[str, int]]:
    cleaned_segments = [normalize_text(segment) for segment in segments if normalize_text(segment)]
    if not cleaned_segments:
        return "", {"segments": 0, "overlap_joins": 0, "plain_joins": 0, "overlap_chars_saved": 0}

    merged = cleaned_segments[0]
    stats = {"segments": len(cleaned_segments), "overlap_joins": 0, "plain_joins": 0, "overlap_chars_saved": 0}
    for segment in cleaned_segments[1:]:
        if segment in merged:
            stats["overlap_joins"] += 1
            stats["overlap_chars_saved"] += len(segment)
            continue
        best = 0
        probe_limit = min(len(merged), len(segment), int(max_overlap_probe))
        for overlap in range(probe_limit, int(min_overlap_chars) - 1, -1):
            if merged.endswith(segment[:overlap]):
                best = overlap
                break
        if best > 0:
            merged = f"{merged}{segment[best:]}"
            stats["overlap_joins"] += 1
            stats["overlap_chars_saved"] += best
        else:
            merged = f"{merged} {segment}"
            stats["plain_joins"] += 1
    return normalize_text(merged), stats


def _write_corpus_outputs(
    *,
    output_dir: str,
    corpus_name: str,
    version: str,
    source: str,
    documents: Sequence[DocumentRecord],
    rejected_rows: Sequence[Dict[str, Any]],
    stats: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    ensure_dir(output_dir)
    documents_path = os.path.join(output_dir, "documents.jsonl")
    rejected_path = os.path.join(output_dir, "rejected.jsonl")
    stats_path = os.path.join(output_dir, "stats.json")
    manifest_path = os.path.join(output_dir, "manifest.json")

    write_jsonl(documents_path, [doc.to_dict() for doc in documents])
    write_jsonl(rejected_path, rejected_rows)
    write_json(stats_path, stats)
    manifest_metadata = dict(metadata or {})
    manifest_metadata.setdefault("routing", _routing_meta(corpus_name=corpus_name, source=source))
    manifest = CorpusManifest(
        corpus_name=corpus_name,
        version=version,
        created_at=_timestamp(),
        doc_count=len(documents),
        chunk_count=0,
        sources=[source],
        chunk_size=None,
        chunk_overlap=None,
        metadata=manifest_metadata,
    )
    write_json(manifest_path, manifest.to_dict())
    return {
        "documents_path": documents_path,
        "rejected_path": rejected_path,
        "stats_path": stats_path,
        "manifest_path": manifest_path,
    }


def _list_jsonl_files(root_dir: str) -> List[str]:
    return sorted(fname for fname in os.listdir(root_dir) if fname.endswith(".jsonl"))


def _merge_counter(target: Counter[str], source: Dict[str, int]) -> None:
    for key, value in source.items():
        target[key] += int(value)


def _pack_textbook_segments(
    segments: Sequence[Dict[str, Any]],
    *,
    target_doc_tokens: int,
    max_doc_tokens: int,
    max_segments_per_doc: int,
) -> List[List[Dict[str, Any]]]:
    if not segments:
        return []

    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = int(seg.get("token_count", 0) or 0)
        if current and (
            len(current) >= int(max_segments_per_doc)
            or current_tokens >= int(max_doc_tokens)
            or (current_tokens >= int(target_doc_tokens) and current_tokens + seg_tokens > int(max_doc_tokens))
        ):
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(seg)
        current_tokens += seg_tokens

    if current:
        groups.append(current)

    return groups


def _run_textbook_import_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _textbook_import_job(
        chunk_dir=str(payload["chunk_dir"]),
        fname=str(payload["fname"]),
        source=str(payload["source"]),
        corpus_name=str(payload["corpus_name"]),
        config_payload=dict(payload["config_payload"]),
        merge_min_overlap_chars=int(payload["merge_min_overlap_chars"]),
        pack_mode=str(payload["pack_mode"]),
        target_doc_tokens=int(payload["target_doc_tokens"]),
        max_doc_tokens=int(payload["max_doc_tokens"]),
        max_segments_per_doc=int(payload["max_segments_per_doc"]),
    )


def _textbook_import_job(
    *,
    chunk_dir: str,
    fname: str,
    source: str,
    corpus_name: str,
    config_payload: Dict[str, Any],
    merge_min_overlap_chars: int,
    pack_mode: str,
    target_doc_tokens: int,
    max_doc_tokens: int,
    max_segments_per_doc: int,
) -> Dict[str, Any]:
    config = default_cleaning_config("textbooks")
    config.min_chars = int(config_payload["min_chars"])
    config.min_tokens = int(config_payload["min_tokens"])
    config.min_alpha_ratio = float(config_payload["min_alpha_ratio"])
    config.max_digit_ratio = float(config_payload["max_digit_ratio"])
    config.min_unique_token_ratio = float(config_payload["min_unique_token_ratio"])
    config.dedupe_by_text = bool(config_payload["dedupe_by_text"])
    config.drop_leading_title = bool(config_payload["drop_leading_title"])

    fpath = os.path.join(chunk_dir, fname)
    rows = load_jsonl(fpath)
    if not rows:
        return {
            "documents": [],
            "rejected_rows": [],
            "raw_segment_count": 0,
        }

    ordered_rows = sorted(
        enumerate(rows),
        key=lambda item: _suffix_index(item[1].get("id"), item[0]),
    )
    book_name = file_stem(fname)
    title = _normalize_space(rows[0].get("title") or book_name or file_stem(fname))
    segments: List[Dict[str, Any]] = []
    raw_segment_count = 0
    for row_idx, row in ordered_rows:
        text = normalize_text(row.get("content", row.get("contents", "")))
        if not text:
            continue
        raw_segment_count += 1
        raw_id = str(row.get("id") or f"{book_name}_{row_idx}")
        segments.append(
            {
                "segment_id": raw_id,
                "text": text,
                "token_count": _token_count(text),
            }
        )

    if not segments:
        return {
            "documents": [],
            "rejected_rows": [],
            "raw_segment_count": 0,
        }

    normalized_mode = _normalize_space(pack_mode).lower() or "sequential"
    if normalized_mode == "legacy_merge":
        segment_groups = [segments]
    elif normalized_mode == "snippet":
        segment_groups = [[segment] for segment in segments]
    else:
        segment_groups = _pack_textbook_segments(
            segments,
            target_doc_tokens=target_doc_tokens,
            max_doc_tokens=max_doc_tokens,
            max_segments_per_doc=max_segments_per_doc,
        )

    accepted_documents: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    route_meta = _routing_meta(corpus_name=corpus_name, source=source)

    for doc_idx, group in enumerate(segment_groups):
        merged_text, merge_stats = _merge_overlapping_segments(
            [str(item["text"]) for item in group],
            min_overlap_chars=merge_min_overlap_chars,
        )
        doc_id = safe_doc_id(source, f"{book_name}#doc{doc_idx:04d}", f"{book_name}#doc{doc_idx:04d}")
        cleaned = clean_document(title=book_name, text=merged_text, config=config)
        meta: Dict[str, Any] = {
            "imported_from": "medrag_chunk_jsonl",
            "source_type": "textbooks",
            "book_name": book_name,
            "original_file": fname,
            "segment_count": len(group),
            "segment_start_id": str(group[0]["segment_id"]),
            "segment_end_id": str(group[-1]["segment_id"]),
            "segment_ids": [str(item["segment_id"]) for item in group],
            "packed_document_index": doc_idx,
            "pack_mode": normalized_mode,
            "target_doc_tokens": int(target_doc_tokens),
            "max_doc_tokens": int(max_doc_tokens),
            "max_segments_per_doc": int(max_segments_per_doc),
            "merge_stats": merge_stats,
            "cleaning_version": "v2",
        }
        meta.update(route_meta)
        meta.update(cleaned.meta)

        if not cleaned.accepted:
            rejected_rows.append(
                {
                    "doc_id": doc_id,
                    "source": source,
                    "title": cleaned.title or book_name,
                    "text": cleaned.text,
                    "reason": cleaned.reason,
                    "meta": meta,
                }
            )
            continue

        accepted_documents.append(
            {
                "doc_id": doc_id,
                "source": source,
                "title": cleaned.title or book_name,
                "text": cleaned.text,
                "meta": meta,
                "fingerprint": text_fingerprint(cleaned.title or book_name, cleaned.text),
            }
        )

    return {
        "documents": accepted_documents,
        "rejected_rows": rejected_rows,
        "raw_segment_count": raw_segment_count,
    }


def _pubmed_import_job(
    *,
    chunk_dir: str,
    fname: str,
    source: str,
    corpus_name: str,
    config_payload: Dict[str, Any],
) -> Dict[str, Any]:
    config = default_cleaning_config("pubmed")
    config.min_chars = int(config_payload["min_chars"])
    config.min_tokens = int(config_payload["min_tokens"])
    config.min_alpha_ratio = float(config_payload["min_alpha_ratio"])
    config.max_digit_ratio = float(config_payload["max_digit_ratio"])
    config.min_unique_token_ratio = float(config_payload["min_unique_token_ratio"])
    config.dedupe_by_text = bool(config_payload["dedupe_by_text"])
    config.drop_leading_title = bool(config_payload["drop_leading_title"])

    accepted_documents: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    raw_row_count = 0

    fpath = os.path.join(chunk_dir, fname)
    for line_no, row in enumerate(load_jsonl(fpath)):
        raw_row_count += 1
        raw_title = _normalize_space(row.get("title", ""))
        raw_text = normalize_text(row.get("content", row.get("contents", "")))
        raw_id = row.get("id") or f"{file_stem(fname)}:{line_no:06d}"
        doc_id = safe_doc_id(source, str(raw_id), f"{file_stem(fname)}:{line_no:06d}")
        cleaned = clean_document(title=raw_title, text=raw_text, config=config)
        meta: Dict[str, Any] = {
            "imported_from": "medrag_chunk_jsonl",
            "source_type": "pubmed",
            "original_file": fname,
            "original_line_no": line_no,
            "original_id": str(raw_id),
            "cleaning_version": "v1",
        }
        meta.update(_routing_meta(corpus_name=corpus_name, source=source))
        if str(raw_id).upper().startswith("PMID:"):
            meta["pmid"] = str(raw_id).split(":", 1)[1]
        meta.update(cleaned.meta)
        if not cleaned.accepted:
            rejected_rows.append(
                {
                    "doc_id": doc_id,
                    "source": source,
                    "title": cleaned.title or raw_title,
                    "text": cleaned.text,
                    "reason": cleaned.reason,
                    "meta": meta,
                }
            )
            continue
        accepted_documents.append(
            {
                "doc_id": doc_id,
                "source": source,
                "title": cleaned.title or raw_title,
                "text": cleaned.text,
                "meta": meta,
                "fingerprint": text_fingerprint(cleaned.title or raw_title, cleaned.text),
            }
        )
    return {
        "documents": accepted_documents,
        "rejected_rows": rejected_rows,
        "raw_row_count": raw_row_count,
    }


def _run_pubmed_import_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _pubmed_import_job(
        chunk_dir=str(payload["chunk_dir"]),
        fname=str(payload["fname"]),
        source=str(payload["source"]),
        corpus_name=str(payload["corpus_name"]),
        config_payload=dict(payload["config_payload"]),
    )


def _stream_chunks_from_documents(
    documents_path: str,
    chunks_path: str,
    config: ChunkingConfig,
    *,
    desc: str = "Build chunks",
    show_progress: bool = True,
) -> Tuple[int, int, List[str]]:
    ensure_dir(os.path.dirname(chunks_path) or ".")
    doc_count = 0
    chunk_count = 0
    sources = set()
    total_docs = count_jsonl_rows(documents_path)
    progress = progress_bar(total=total_docs, desc=desc, unit="doc", disable=not show_progress)
    with open(chunks_path, "w", encoding="utf-8") as fout:
        for row in iter_jsonl(documents_path):
            doc = DocumentRecord(**row)
            doc_count += 1
            sources.add(doc.source)
            doc_chunks = chunk_document(doc, config)
            chunk_count += len(doc_chunks)
            for chunk in doc_chunks:
                fout.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            progress.update(1)
            if doc_count == 1 or doc_count % 200 == 0:
                progress.set_postfix(docs=doc_count, chunks=chunk_count, refresh=False)
    progress.set_postfix(docs=doc_count, chunks=chunk_count, refresh=False)
    progress.close()
    return doc_count, chunk_count, sorted(sources)


def import_medrag_chunks(
    chunk_dir: str,
    output_dir: str,
    corpus_name: str,
    source_name: Optional[str] = None,
) -> Dict[str, str]:
    ensure_dir(output_dir)
    source = _normalize_space(source_name or corpus_name or "medrag_import")
    documents: List[DocumentRecord] = []
    chunks: List[ChunkRecord] = []

    for fname in sorted(os.listdir(chunk_dir)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(chunk_dir, fname)
        base = file_stem(fname)
        for line_no, row in enumerate(load_jsonl(fpath)):
            title = _normalize_space(row.get("title", ""))
            content = _normalize_space(row.get("content", row.get("contents", "")))
            if not title and not content:
                continue
            doc_id = str(row.get("id") or f"{source}:{base}:{line_no:06d}")
            doc = DocumentRecord(
                doc_id=doc_id,
                source=source,
                title=title,
                text=_concat_title_text(title, content),
                meta={
                    "imported_from": "medrag_chunk_jsonl",
                    "original_file": fname,
                    "original_line_no": line_no,
                    "prechunked": True,
                    **_routing_meta(corpus_name=corpus_name, source=source),
                },
            )
            chunk = ChunkRecord(
                chunk_id=f"{doc_id}#chunk0000",
                doc_id=doc_id,
                source=source,
                title=title,
                text=content or doc.text,
                chunk_index=0,
                token_count=max(1, len((content or doc.text).split())),
                char_start=0,
                char_end=len(content or doc.text),
                meta=dict(doc.meta),
            )
            documents.append(doc)
            chunks.append(chunk)

    documents_path = os.path.join(output_dir, "documents.jsonl")
    chunks_path = os.path.join(output_dir, "chunks.jsonl")
    manifest_path = os.path.join(output_dir, "manifest.json")
    write_jsonl(documents_path, [doc.to_dict() for doc in documents])
    write_jsonl(chunks_path, [chunk.to_dict() for chunk in chunks])
    manifest = CorpusManifest(
        corpus_name=corpus_name,
        version="v0-import-medrag",
        created_at=_timestamp(),
        doc_count=len(documents),
        chunk_count=len(chunks),
        sources=[source],
        chunk_size=None,
        chunk_overlap=None,
        metadata={"format": "medrag_chunk_jsonl", "chunk_dir": chunk_dir},
    )
    write_json(manifest_path, manifest.to_dict())
    return {
        "documents_path": documents_path,
        "chunks_path": chunks_path,
        "manifest_path": manifest_path,
    }


def import_medrag_textbooks(
    chunk_dir: str,
    output_dir: str,
    corpus_name: str,
    source_name: str = "textbooks",
    *,
    min_chars: Optional[int] = None,
    min_tokens: Optional[int] = None,
    min_alpha_ratio: Optional[float] = None,
    merge_min_overlap_chars: int = 48,
    no_dedupe: bool = False,
    show_progress: bool = True,
    workers: int = 1,
    pack_mode: str = "sequential",
    target_doc_tokens: int = 1536,
    max_doc_tokens: int = 2048,
    max_segments_per_doc: int = 12,
) -> Dict[str, str]:
    source = _normalize_space(source_name or "textbooks")
    config = default_cleaning_config("textbooks")
    if min_chars is not None:
        config.min_chars = int(min_chars)
    if min_tokens is not None:
        config.min_tokens = int(min_tokens)
    if min_alpha_ratio is not None:
        config.min_alpha_ratio = float(min_alpha_ratio)
    if no_dedupe:
        config.dedupe_by_text = False

    documents: List[DocumentRecord] = []
    rejected_rows: List[Dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()
    raw_segment_count = 0
    seen_fingerprints = set()

    chunk_files = _list_jsonl_files(chunk_dir)
    progress = progress_bar(
        chunk_files,
        total=len(chunk_files),
        desc=f"Import textbooks:{corpus_name}",
        unit="file",
        disable=not show_progress,
    )
    config_payload = {
        "min_chars": config.min_chars,
        "min_tokens": config.min_tokens,
        "min_alpha_ratio": config.min_alpha_ratio,
        "max_digit_ratio": config.max_digit_ratio,
        "min_unique_token_ratio": config.min_unique_token_ratio,
        "dedupe_by_text": config.dedupe_by_text,
        "drop_leading_title": config.drop_leading_title,
    }

    if int(workers) > 1 and len(chunk_files) > 1:
        jobs = [
            {
                "chunk_dir": chunk_dir,
                "fname": fname,
                "source": source,
                "corpus_name": corpus_name,
                "config_payload": config_payload,
                "merge_min_overlap_chars": merge_min_overlap_chars,
                "pack_mode": pack_mode,
                "target_doc_tokens": target_doc_tokens,
                "max_doc_tokens": max_doc_tokens,
                "max_segments_per_doc": max_segments_per_doc,
            }
            for fname in chunk_files
        ]
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            iterator = ex.map(_run_textbook_import_job, jobs)
            for result in iterator:
                raw_segment_count += int(result.get("raw_segment_count", 0) or 0)
                for rejected in result.get("rejected_rows", []) or []:
                    reason = str(rejected.get("reason", "") or "")
                    reject_reasons[reason] += 1
                    rejected_rows.append(rejected)
                for document in result.get("documents", []) or []:
                    fingerprint = str(document.pop("fingerprint", "") or "")
                    if config.dedupe_by_text and fingerprint in seen_fingerprints:
                        reject_reasons["duplicate_text"] += 1
                        rejected_rows.append(
                            {
                                "doc_id": document["doc_id"],
                                "source": document["source"],
                                "title": document["title"],
                                "text": document["text"],
                                "reason": "duplicate_text",
                                "meta": document["meta"],
                            }
                        )
                    else:
                        seen_fingerprints.add(fingerprint)
                        documents.append(DocumentRecord(**document))
                progress.update(1)
                progress.set_postfix(
                    accepted=len(documents),
                    rejected=len(rejected_rows),
                    segments=raw_segment_count,
                    refresh=False,
                )
    else:
        for fname in progress:
            result = _textbook_import_job(
                chunk_dir=chunk_dir,
                fname=fname,
                source=source,
                corpus_name=corpus_name,
                config_payload=config_payload,
                merge_min_overlap_chars=merge_min_overlap_chars,
                pack_mode=pack_mode,
                target_doc_tokens=target_doc_tokens,
                max_doc_tokens=max_doc_tokens,
                max_segments_per_doc=max_segments_per_doc,
            )
            raw_segment_count += int(result.get("raw_segment_count", 0) or 0)
            for rejected in result.get("rejected_rows", []) or []:
                reason = str(rejected.get("reason", "") or "")
                reject_reasons[reason] += 1
                rejected_rows.append(rejected)
            for document in result.get("documents", []) or []:
                fingerprint = str(document.pop("fingerprint", "") or "")
                if config.dedupe_by_text and fingerprint in seen_fingerprints:
                    reject_reasons["duplicate_text"] += 1
                    rejected_rows.append(
                        {
                            "doc_id": document["doc_id"],
                            "source": document["source"],
                            "title": document["title"],
                            "text": document["text"],
                            "reason": "duplicate_text",
                            "meta": document["meta"],
                        }
                    )
                else:
                    seen_fingerprints.add(fingerprint)
                    documents.append(DocumentRecord(**document))
            progress.set_postfix(
                accepted=len(documents),
                rejected=len(rejected_rows),
                segments=raw_segment_count,
                refresh=False,
            )
        progress.set_postfix(
            accepted=len(documents),
            rejected=len(rejected_rows),
            segments=raw_segment_count,
            refresh=False,
        )
    progress.set_postfix(
        accepted=len(documents),
        rejected=len(rejected_rows),
        segments=raw_segment_count,
        refresh=False,
    )
    progress.close()

    accepted_chars = [len(doc.text) for doc in documents]
    accepted_tokens = [int(doc.meta.get("token_count", 0)) for doc in documents]
    stats = {
        "corpus_name": corpus_name,
        "source": source,
        "source_type": "textbooks",
        "raw_book_count": len(chunk_files),
        "raw_segment_count": raw_segment_count,
        "accepted_documents": len(documents),
        "rejected_documents": len(rejected_rows),
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "avg_char_count": round(sum(accepted_chars) / len(accepted_chars), 2) if accepted_chars else 0.0,
        "avg_token_count": round(sum(accepted_tokens) / len(accepted_tokens), 2) if accepted_tokens else 0.0,
    }
    metadata = {
        "format": "owned_documents_v1",
        "import_kind": "medrag_textbooks",
        "chunk_dir": chunk_dir,
        "routing": _routing_meta(corpus_name=corpus_name, source=source),
        "cleaning_config": {
            "min_chars": config.min_chars,
            "min_tokens": config.min_tokens,
            "min_alpha_ratio": config.min_alpha_ratio,
            "dedupe_by_text": config.dedupe_by_text,
            "merge_min_overlap_chars": merge_min_overlap_chars,
            "pack_mode": pack_mode,
            "target_doc_tokens": target_doc_tokens,
            "max_doc_tokens": max_doc_tokens,
            "max_segments_per_doc": max_segments_per_doc,
        },
    }
    return _write_corpus_outputs(
        output_dir=output_dir,
        corpus_name=corpus_name,
        version="v1-textbooks-clean",
        source=source,
        documents=documents,
        rejected_rows=rejected_rows,
        stats=stats,
        metadata=metadata,
    )


def import_medrag_pubmed(
    chunk_dir: str,
    output_dir: str,
    corpus_name: str,
    source_name: str = "pubmed",
    *,
    min_chars: Optional[int] = None,
    min_tokens: Optional[int] = None,
    min_alpha_ratio: Optional[float] = None,
    no_dedupe: bool = False,
    show_progress: bool = True,
    workers: int = 1,
) -> Dict[str, str]:
    source = _normalize_space(source_name or "pubmed")
    config = default_cleaning_config("pubmed")
    if min_chars is not None:
        config.min_chars = int(min_chars)
    if min_tokens is not None:
        config.min_tokens = int(min_tokens)
    if min_alpha_ratio is not None:
        config.min_alpha_ratio = float(min_alpha_ratio)
    if no_dedupe:
        config.dedupe_by_text = False

    documents: List[DocumentRecord] = []
    rejected_rows: List[Dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()
    raw_row_count = 0
    seen_fingerprints = set()

    chunk_files = _list_jsonl_files(chunk_dir)
    progress = progress_bar(
        chunk_files,
        total=len(chunk_files),
        desc=f"Import pubmed:{corpus_name}",
        unit="file",
        disable=not show_progress,
    )
    config_payload = {
        "min_chars": config.min_chars,
        "min_tokens": config.min_tokens,
        "min_alpha_ratio": config.min_alpha_ratio,
        "max_digit_ratio": config.max_digit_ratio,
        "min_unique_token_ratio": config.min_unique_token_ratio,
        "dedupe_by_text": config.dedupe_by_text,
        "drop_leading_title": config.drop_leading_title,
    }
    if int(workers) > 1 and len(chunk_files) > 1:
        jobs = [
            {
                "chunk_dir": chunk_dir,
                "fname": fname,
                "source": source,
                "corpus_name": corpus_name,
                "config_payload": config_payload,
            }
            for fname in chunk_files
        ]
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            iterator = ex.map(_run_pubmed_import_job, jobs)
            for result in iterator:
                raw_row_count += int(result.get("raw_row_count", 0) or 0)
                for rejected in result.get("rejected_rows", []) or []:
                    reason = str(rejected.get("reason", "") or "")
                    reject_reasons[reason] += 1
                    rejected_rows.append(rejected)
                for document in result.get("documents", []) or []:
                    fingerprint = str(document.pop("fingerprint", "") or "")
                    if config.dedupe_by_text and fingerprint in seen_fingerprints:
                        reject_reasons["duplicate_text"] += 1
                        rejected_rows.append(
                            {
                                "doc_id": document["doc_id"],
                                "source": document["source"],
                                "title": document["title"],
                                "text": document["text"],
                                "reason": "duplicate_text",
                                "meta": document["meta"],
                            }
                        )
                    else:
                        seen_fingerprints.add(fingerprint)
                        documents.append(DocumentRecord(**document))
                progress.update(1)
                progress.set_postfix(
                    accepted=len(documents),
                    rejected=len(rejected_rows),
                    rows=raw_row_count,
                    refresh=False,
                )
    else:
        for fname in progress:
            result = _pubmed_import_job(
                chunk_dir=chunk_dir,
                fname=fname,
                source=source,
                corpus_name=corpus_name,
                config_payload=config_payload,
            )
            raw_row_count += int(result.get("raw_row_count", 0) or 0)
            for rejected in result.get("rejected_rows", []) or []:
                reason = str(rejected.get("reason", "") or "")
                reject_reasons[reason] += 1
                rejected_rows.append(rejected)
            for document in result.get("documents", []) or []:
                fingerprint = str(document.pop("fingerprint", "") or "")
                if config.dedupe_by_text and fingerprint in seen_fingerprints:
                    reject_reasons["duplicate_text"] += 1
                    rejected_rows.append(
                        {
                            "doc_id": document["doc_id"],
                            "source": document["source"],
                            "title": document["title"],
                            "text": document["text"],
                            "reason": "duplicate_text",
                            "meta": document["meta"],
                        }
                    )
                else:
                    seen_fingerprints.add(fingerprint)
                    documents.append(DocumentRecord(**document))
            progress.set_postfix(
                accepted=len(documents),
                rejected=len(rejected_rows),
                rows=raw_row_count,
                refresh=False,
            )
    progress.set_postfix(
        accepted=len(documents),
        rejected=len(rejected_rows),
        rows=raw_row_count,
        refresh=False,
    )
    progress.close()

    accepted_chars = [len(doc.text) for doc in documents]
    accepted_tokens = [int(doc.meta.get("token_count", 0)) for doc in documents]
    stats = {
        "corpus_name": corpus_name,
        "source": source,
        "source_type": "pubmed",
        "raw_row_count": raw_row_count,
        "accepted_documents": len(documents),
        "rejected_documents": len(rejected_rows),
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "avg_char_count": round(sum(accepted_chars) / len(accepted_chars), 2) if accepted_chars else 0.0,
        "avg_token_count": round(sum(accepted_tokens) / len(accepted_tokens), 2) if accepted_tokens else 0.0,
    }
    metadata = {
        "format": "owned_documents_v1",
        "import_kind": "medrag_pubmed",
        "chunk_dir": chunk_dir,
        "routing": _routing_meta(corpus_name=corpus_name, source=source),
        "cleaning_config": {
            "min_chars": config.min_chars,
            "min_tokens": config.min_tokens,
            "min_alpha_ratio": config.min_alpha_ratio,
            "dedupe_by_text": config.dedupe_by_text,
        },
    }
    return _write_corpus_outputs(
        output_dir=output_dir,
        corpus_name=corpus_name,
        version="v1-pubmed-clean",
        source=source,
        documents=documents,
        rejected_rows=rejected_rows,
        stats=stats,
        metadata=metadata,
    )


def build_chunks_from_documents(
    documents_path: str,
    output_dir: str,
    chunk_size: int = 384,
    chunk_overlap: int = 96,
    min_tokens: int = 32,
    corpus_name: str = "custom_corpus",
    show_progress: bool = True,
) -> Dict[str, str]:
    ensure_dir(output_dir)
    config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap, min_tokens=min_tokens)
    chunks_path = os.path.join(output_dir, "chunks.jsonl")
    manifest_path = os.path.join(output_dir, "manifest.json")
    doc_count, chunk_count, sources = _stream_chunks_from_documents(
        documents_path=documents_path,
        chunks_path=chunks_path,
        config=config,
        desc=f"Build chunks:{corpus_name}",
        show_progress=show_progress,
    )
    manifest = CorpusManifest(
        corpus_name=corpus_name,
        version="v1-custom",
        created_at=_timestamp(),
        doc_count=doc_count,
        chunk_count=chunk_count,
        sources=sources,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata={
            "documents_path": documents_path,
            "min_tokens": min_tokens,
            "routing": _routing_meta(corpus_name=corpus_name, source=sources[0] if len(sources) == 1 else "mixed"),
        },
    )
    write_json(manifest_path, manifest.to_dict())
    return {"chunks_path": chunks_path, "manifest_path": manifest_path}


def build_corpus_registry(corpus_root: str, output_path: str) -> Dict[str, Any]:
    ensure_dir(os.path.dirname(output_path) or ".")
    entries: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(corpus_root)):
        corpus_dir = os.path.join(corpus_root, name)
        if not os.path.isdir(corpus_dir):
            continue
        manifest_path = os.path.join(corpus_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            continue
        manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
        metadata = dict(manifest.get("metadata", {}) or {})
        routing = dict(metadata.get("routing", {}) or {})
        entry = {
            "corpus_name": str(manifest.get("corpus_name", name) or name),
            "path": corpus_dir,
            "documents_path": os.path.join(corpus_dir, "documents.jsonl") if os.path.exists(os.path.join(corpus_dir, "documents.jsonl")) else "",
            "chunks_path": os.path.join(corpus_dir, "chunks.jsonl") if os.path.exists(os.path.join(corpus_dir, "chunks.jsonl")) else "",
            "bm25_dir": os.path.join(corpus_dir, "indexes", "bm25") if os.path.isdir(os.path.join(corpus_dir, "indexes", "bm25")) else "",
            "dense_dir": os.path.join(corpus_dir, "indexes", "medcpt") if os.path.isdir(os.path.join(corpus_dir, "indexes", "medcpt")) else "",
            "doc_count": int(manifest.get("doc_count", 0) or 0),
            "chunk_count": int(manifest.get("chunk_count", 0) or 0),
            "sources": list(manifest.get("sources", []) or []),
            "routeable": bool(routing.get("routeable", True)),
            "routing_arm": str(routing.get("routing_arm", manifest.get("corpus_name", name)) or manifest.get("corpus_name", name)),
            "source_type": str(routing.get("source_type", (manifest.get("sources", ["generic"]) or ["generic"])[0]) or "generic"),
            "version": str(manifest.get("version", "") or ""),
        }
        entries.append(entry)

    registry = {
        "created_at": _timestamp(),
        "corpus_root": corpus_root,
        "entries": entries,
    }
    write_json(output_path, registry)
    return registry

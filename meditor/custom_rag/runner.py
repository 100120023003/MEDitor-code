from __future__ import annotations

import argparse
import json
from typing import Dict, Optional

from .bm25_index import BM25Index, BM25Searcher
from .corpus_builder import (
    build_corpus_registry,
    build_chunks_from_documents,
    import_medrag_chunks,
    import_medrag_pubmed,
    import_medrag_textbooks,
)
from .io_utils import write_json
from .prompting import build_base_aligned_messages
from .retriever import HybridRetriever


def _parse_options_json(text: str) -> Optional[Dict[str, str]]:
    if not text:
        return None
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("--options_json must decode to a JSON object.")
    return {str(k): str(v) for k, v in obj.items()}


def cmd_import_medrag(args: argparse.Namespace) -> None:
    out = import_medrag_chunks(
        chunk_dir=args.chunk_dir,
        output_dir=args.output_dir,
        corpus_name=args.corpus_name,
        source_name=args.source_name,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_import_medrag_textbooks(args: argparse.Namespace) -> None:
    out = import_medrag_textbooks(
        chunk_dir=args.chunk_dir,
        output_dir=args.output_dir,
        corpus_name=args.corpus_name,
        source_name=args.source_name,
        min_chars=args.min_chars,
        min_tokens=args.min_tokens,
        min_alpha_ratio=args.min_alpha_ratio,
        merge_min_overlap_chars=args.merge_min_overlap_chars,
        no_dedupe=args.no_dedupe,
        show_progress=not args.no_progress,
        workers=args.workers,
        pack_mode=args.pack_mode,
        target_doc_tokens=args.target_doc_tokens,
        max_doc_tokens=args.max_doc_tokens,
        max_segments_per_doc=args.max_segments_per_doc,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_import_medrag_pubmed(args: argparse.Namespace) -> None:
    out = import_medrag_pubmed(
        chunk_dir=args.chunk_dir,
        output_dir=args.output_dir,
        corpus_name=args.corpus_name,
        source_name=args.source_name,
        min_chars=args.min_chars,
        min_tokens=args.min_tokens,
        min_alpha_ratio=args.min_alpha_ratio,
        no_dedupe=args.no_dedupe,
        show_progress=not args.no_progress,
        workers=args.workers,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_build_chunks(args: argparse.Namespace) -> None:
    out = build_chunks_from_documents(
        documents_path=args.documents,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_tokens=args.min_tokens,
        corpus_name=args.corpus_name,
        show_progress=not args.no_progress,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_build_corpus_registry(args: argparse.Namespace) -> None:
    out = build_corpus_registry(
        corpus_root=args.corpus_root,
        output_path=args.output_json,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_build_bm25(args: argparse.Namespace) -> None:
    meta = BM25Index(k1=args.k1, b=args.b).build(chunks_path=args.chunks, index_dir=args.index_dir)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def cmd_build_dense(args: argparse.Namespace) -> None:
    from .dense_index import DenseIndex

    meta = DenseIndex().build(
        chunks_path=args.chunks,
        index_dir=args.index_dir,
        model_name=args.model_name,
        query_model_name=args.query_model_name,
        batch_size=args.batch_size,
        device=args.device,
        normalize=not args.no_normalize,
        use_faiss=not args.no_faiss,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def cmd_inspect(args: argparse.Namespace) -> None:
    bm25 = BM25Searcher(args.bm25_dir) if args.bm25_dir else None
    if args.dense_dir:
        from .dense_index import DenseSearcher

        dense = DenseSearcher(args.dense_dir, device=args.device)
    else:
        dense = None
    retriever = HybridRetriever(bm25=bm25, dense=dense)
    options = _parse_options_json(args.options_json)
    hits, trace = retriever.retrieve(
        question=args.question,
        options=options,
        top_k=args.top_k,
        sparse_k=args.sparse_k,
        dense_k=args.dense_k,
        query_mode=args.query_mode,
        fusion=args.fusion,
        rrf_k=args.rrf_k,
    )
    messages = build_base_aligned_messages(
        question=args.question,
        options=options,
        hits=hits,
        allowed_choices=list(options.keys()) if options else None,
        max_snippets=args.prompt_snippets,
        max_chars=args.prompt_chars,
    )
    result = {
        "trace": trace,
        "messages": messages,
    }
    if args.out_json:
        write_json(args.out_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build and inspect MEDitor retrieval indexes.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("import-medrag-chunks")
    sp.add_argument("--chunk_dir", required=True)
    sp.add_argument("--output_dir", required=True)
    sp.add_argument("--corpus_name", required=True)
    sp.add_argument("--source_name", default="")
    sp.set_defaults(func=cmd_import_medrag)

    sp = sub.add_parser("import-medrag-textbooks")
    sp.add_argument("--chunk_dir", required=True)
    sp.add_argument("--output_dir", required=True)
    sp.add_argument("--corpus_name", required=True)
    sp.add_argument("--source_name", default="textbooks")
    sp.add_argument("--min_chars", type=int, default=None)
    sp.add_argument("--min_tokens", type=int, default=None)
    sp.add_argument("--min_alpha_ratio", type=float, default=None)
    sp.add_argument("--merge_min_overlap_chars", type=int, default=48)
    sp.add_argument("--no_dedupe", action="store_true")
    sp.add_argument("--workers", type=int, default=1)
    sp.add_argument(
        "--pack_mode",
        default="sequential",
        choices=["sequential", "snippet", "legacy_merge"],
        help="How to turn official textbook snippets into owned documents before rechunking.",
    )
    sp.add_argument("--target_doc_tokens", type=int, default=1536)
    sp.add_argument("--max_doc_tokens", type=int, default=2048)
    sp.add_argument("--max_segments_per_doc", type=int, default=12)
    sp.add_argument("--no_progress", action="store_true")
    sp.set_defaults(func=cmd_import_medrag_textbooks)

    sp = sub.add_parser("import-medrag-pubmed")
    sp.add_argument("--chunk_dir", required=True)
    sp.add_argument("--output_dir", required=True)
    sp.add_argument("--corpus_name", required=True)
    sp.add_argument("--source_name", default="pubmed")
    sp.add_argument("--min_chars", type=int, default=None)
    sp.add_argument("--min_tokens", type=int, default=None)
    sp.add_argument("--min_alpha_ratio", type=float, default=None)
    sp.add_argument("--no_dedupe", action="store_true")
    sp.add_argument("--workers", type=int, default=1)
    sp.add_argument("--no_progress", action="store_true")
    sp.set_defaults(func=cmd_import_medrag_pubmed)

    sp = sub.add_parser("build-chunks")
    sp.add_argument("--documents", required=True)
    sp.add_argument("--output_dir", required=True)
    sp.add_argument("--corpus_name", default="custom_corpus")
    sp.add_argument("--chunk_size", type=int, default=384)
    sp.add_argument("--chunk_overlap", type=int, default=96)
    sp.add_argument("--min_tokens", type=int, default=32)
    sp.add_argument("--no_progress", action="store_true")
    sp.set_defaults(func=cmd_build_chunks)

    sp = sub.add_parser("build-corpus-registry")
    sp.add_argument("--corpus_root", required=True)
    sp.add_argument("--output_json", required=True)
    sp.set_defaults(func=cmd_build_corpus_registry)

    sp = sub.add_parser("build-bm25")
    sp.add_argument("--chunks", required=True)
    sp.add_argument("--index_dir", required=True)
    sp.add_argument("--k1", type=float, default=1.5)
    sp.add_argument("--b", type=float, default=0.75)
    sp.set_defaults(func=cmd_build_bm25)

    sp = sub.add_parser("build-dense")
    sp.add_argument("--chunks", required=True)
    sp.add_argument("--index_dir", required=True)
    sp.add_argument("--model_name", required=True)
    sp.add_argument("--query_model_name", default="")
    sp.add_argument("--batch_size", type=int, default=64)
    sp.add_argument("--device", default=None)
    sp.add_argument("--no_normalize", action="store_true")
    sp.add_argument("--no_faiss", action="store_true")
    sp.set_defaults(func=cmd_build_dense)

    sp = sub.add_parser("inspect")
    sp.add_argument("--question", required=True)
    sp.add_argument("--options_json", default="")
    sp.add_argument("--bm25_dir", default="")
    sp.add_argument("--dense_dir", default="")
    sp.add_argument("--device", default=None)
    sp.add_argument("--top_k", type=int, default=8)
    sp.add_argument("--sparse_k", type=int, default=32)
    sp.add_argument("--dense_k", type=int, default=32)
    sp.add_argument("--query_mode", default="question_only", choices=["question_only", "question_plus_options"])
    sp.add_argument("--fusion", default="rrf", choices=["rrf", "sparse_only", "dense_only"])
    sp.add_argument("--rrf_k", type=int, default=60)
    sp.add_argument("--prompt_snippets", type=int, default=6)
    sp.add_argument("--prompt_chars", type=int, default=6000)
    sp.add_argument("--out_json", default="")
    sp.set_defaults(func=cmd_inspect)
    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "fusion"):
        if args.fusion == "sparse_only":
            args.dense_dir = ""
            args.fusion = "rrf"
        elif args.fusion == "dense_only":
            args.bm25_dir = ""
            args.fusion = "rrf"
    args.func(args)


if __name__ == "__main__":
    main()

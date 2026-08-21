from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class _NullProgress:
    def __init__(self, iterable=None) -> None:
        self._iterable = iterable

    def __iter__(self):
        if self._iterable is None:
            return iter(())
        return iter(self._iterable)

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def write(self, text: str) -> None:
        print(text, flush=True)

    def close(self) -> None:
        return None


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in iter_jsonl(path):
        rows.append(row)
    return rows


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def count_jsonl_rows(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def progress_bar(
    iterable=None,
    *,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
    disable: bool = False,
):
    if disable:
        return _NullProgress(iterable)
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)
    except Exception:
        return _NullProgress(iterable)


def expand_env_placeholders(text: str) -> str:
    raw = str(text or "")

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2)
        value = os.environ.get(key)
        if value not in {None, ""}:
            return value
        if default is not None:
            return default
        return match.group(0)

    expanded = _ENV_VAR_PATTERN.sub(_replace, raw)
    expanded = os.path.expandvars(expanded)
    expanded = os.path.expanduser(expanded)
    return expanded


def resolve_model_ref(value: str) -> str:
    expanded = expand_env_placeholders(value)
    if expanded and os.path.exists(expanded):
        return os.path.realpath(expanded)
    return expanded


def resolve_index_data_path(index_dir: str, stored_path: str, filename: str) -> str:
    """Resolve a path stored in an index meta file after moving the corpus.

    Older custom RAG indexes stored absolute `chunks.jsonl` paths. When the
    corpus directory is copied to a new server those paths become stale even
    though the index itself is valid. Prefer the stored path when it exists;
    otherwise fall back to the corpus directory inferred from
    `<corpus>/indexes/<index_name>`.
    """
    raw = expand_env_placeholders(str(stored_path or "")).strip()
    candidates: List[str] = []
    if raw:
        candidates.append(raw)
        if not os.path.isabs(raw):
            candidates.append(os.path.join(index_dir, raw))

    index_abs = os.path.abspath(index_dir)
    corpus_dir = os.path.abspath(os.path.join(index_abs, os.pardir, os.pardir))
    candidates.extend(
        [
            os.path.join(corpus_dir, filename),
            os.path.join(index_abs, os.pardir, filename),
            os.path.join(index_abs, filename),
        ]
    )

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.exists(normalized):
            return os.path.realpath(normalized)

    return raw or os.path.join(corpus_dir, filename)

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List


_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\b\w+\b")
_TITLE_PREFIX_RE_TEMPLATE = r"^(?:{title}[\s:;,.!\-]+)+"
_TEXTBOOKS_PATTERNS = [
    re.compile(r"\bcopyright(?:\s+©)?\s+\d{4}[^.]{0,160}\.?", re.IGNORECASE),
    re.compile(r"\ball rights reserved\b[^.]{0,120}\.?", re.IGNORECASE),
    re.compile(r"\bisbn(?:-1[03])?:?\s*[0-9xX\-\s]{8,}\b", re.IGNORECASE),
    re.compile(r"\baccess provided by\b[^.]{0,120}\.?", re.IGNORECASE),
    re.compile(r"\bprinted in the united states of america\b[^.]{0,80}\.?", re.IGNORECASE),
]
_PUBMED_PATTERNS = [
    re.compile(r"\bcopyright information\b[^.]{0,160}\.?", re.IGNORECASE),
]
_STRUCTURED_ABSTRACT_RE = re.compile(
    r"\b(background|objective|methods?|results?|conclusion|conclusions)\s*:",
    re.IGNORECASE,
)
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._:-]+")


@dataclass
class CleaningConfig:
    source_type: str
    min_chars: int = 120
    min_tokens: int = 24
    min_alpha_ratio: float = 0.55
    max_digit_ratio: float = 0.35
    min_unique_token_ratio: float = 0.12
    dedupe_by_text: bool = True
    drop_leading_title: bool = True


@dataclass
class CleanResult:
    accepted: bool
    title: str
    text: str
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("\u200b", " ").replace("\ufeff", " ").replace("\x00", " ")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def safe_doc_id(source: str, raw_id: str, fallback: str) -> str:
    candidate = normalize_text(raw_id or fallback).replace(" ", "_")
    candidate = _SAFE_ID_RE.sub("_", candidate).strip("_")
    if not candidate:
        candidate = _SAFE_ID_RE.sub("_", fallback).strip("_") or "doc"
    if source and not candidate.startswith(f"{source}:"):
        return f"{source}:{candidate}"
    return candidate


def text_fingerprint(title: str, text: str) -> str:
    normalized = f"{normalize_text(title).lower()}\n{normalize_text(text).lower()}"
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def default_cleaning_config(source_type: str) -> CleaningConfig:
    source_type = normalize_text(source_type).lower() or "generic"
    if source_type == "textbooks":
        return CleaningConfig(
            source_type=source_type,
            min_chars=200,
            min_tokens=40,
            min_alpha_ratio=0.58,
            max_digit_ratio=0.25,
            min_unique_token_ratio=0.10,
        )
    if source_type == "pubmed":
        return CleaningConfig(
            source_type=source_type,
            min_chars=80,
            min_tokens=20,
            min_alpha_ratio=0.55,
            max_digit_ratio=0.35,
            min_unique_token_ratio=0.10,
        )
    return CleaningConfig(source_type=source_type)


def _strip_leading_title(title: str, text: str) -> str:
    title = normalize_text(title)
    text = normalize_text(text)
    if not title or not text:
        return text
    prefix = re.escape(title)
    return re.sub(_TITLE_PREFIX_RE_TEMPLATE.format(title=prefix), "", text, count=1, flags=re.IGNORECASE).strip()


def _apply_source_patterns(text: str, source_type: str) -> str:
    patterns = _TEXTBOOKS_PATTERNS if source_type == "textbooks" else _PUBMED_PATTERNS if source_type == "pubmed" else []
    for pattern in patterns:
        text = pattern.sub(" ", text)
    return normalize_text(text)


def _quality_metrics(text: str) -> Dict[str, Any]:
    text = normalize_text(text)
    non_space_chars = max(1, len(re.sub(r"\s+", "", text)))
    alpha_chars = sum(ch.isalpha() for ch in text)
    digit_chars = sum(ch.isdigit() for ch in text)
    tokens = [tok.lower() for tok in _TOKEN_RE.findall(text)]
    unique_ratio = (len(set(tokens)) / len(tokens)) if tokens else 0.0
    return {
        "char_count": len(text),
        "token_count": len(tokens),
        "alpha_ratio": alpha_chars / non_space_chars,
        "digit_ratio": digit_chars / non_space_chars,
        "unique_token_ratio": unique_ratio,
    }


def clean_document(title: str, text: str, config: CleaningConfig) -> CleanResult:
    title = normalize_text(title)
    text = normalize_text(text)
    if config.drop_leading_title:
        text = _strip_leading_title(title, text)
    text = _apply_source_patterns(text, config.source_type)

    metrics = _quality_metrics(text)
    flags: List[str] = []
    if config.source_type == "pubmed" and _STRUCTURED_ABSTRACT_RE.search(text):
        flags.append("structured_abstract")
    if metrics["token_count"] >= 4000:
        flags.append("long_document")

    if not text:
        return CleanResult(accepted=False, title=title, text=text, reason="empty_after_clean", meta=metrics)
    if metrics["char_count"] < int(config.min_chars):
        return CleanResult(accepted=False, title=title, text=text, reason="too_short_chars", meta=metrics)
    if metrics["token_count"] < int(config.min_tokens):
        return CleanResult(accepted=False, title=title, text=text, reason="too_short_tokens", meta=metrics)
    if metrics["alpha_ratio"] < float(config.min_alpha_ratio):
        return CleanResult(accepted=False, title=title, text=text, reason="low_alpha_ratio", meta=metrics)
    if metrics["digit_ratio"] > float(config.max_digit_ratio):
        return CleanResult(accepted=False, title=title, text=text, reason="high_digit_ratio", meta=metrics)
    if metrics["token_count"] >= 50 and metrics["unique_token_ratio"] < float(config.min_unique_token_ratio):
        return CleanResult(accepted=False, title=title, text=text, reason="repetitive_text", meta=metrics)

    meta: Dict[str, Any] = dict(metrics)
    if flags:
        meta["quality_flags"] = flags
    return CleanResult(accepted=True, title=title, text=text, meta=meta)

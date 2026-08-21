from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .schema import RetrievalHit


def render_options(options: Optional[Dict[str, str]]) -> str:
    if not options:
        return ""
    lines: List[str] = []
    for key in sorted(options.keys()):
        value = str(options[key]).strip()
        if value:
            lines.append(f"{key}. {value}")
    return "\n".join(lines)


def render_snippets(
    hits: Sequence[RetrievalHit],
    max_snippets: int = 6,
    max_chars: int = 6000,
) -> str:
    blocks: List[str] = []
    total_chars = 0
    for rank, hit in enumerate(hits[: max(1, int(max_snippets))], start=1):
        title = str(hit.title or "").strip()
        text = str(hit.text or "").strip()
        block = f"[{rank}] {title}\n{text}".strip()
        if not block:
            continue
        if blocks and (total_chars + len(block)) > int(max_chars):
            break
        blocks.append(block)
        total_chars += len(block)
    return "\n\n".join(blocks)


def build_base_aligned_messages(
    question: str,
    options: Optional[Dict[str, str]],
    hits: Sequence[RetrievalHit],
    allowed_choices: Optional[Sequence[str]] = None,
    system: str = "You are a helpful assistant.",
    max_snippets: int = 6,
    max_chars: int = 6000,
) -> List[Dict[str, str]]:
    allowed = [str(x).strip().upper() for x in (allowed_choices or ["A", "B", "C", "D"]) if str(x).strip()]
    choices_list = ", ".join(allowed)
    parts = [
        f"Question:\n{str(question or '').strip()}",
        f"Options:\n{render_options(options)}".strip(),
    ]
    snippets_text = render_snippets(hits, max_snippets=max_snippets, max_chars=max_chars)
    if snippets_text:
        parts.append(f"Relevant Medical References:\n{snippets_text}")
    parts.append("Use the references only when they are relevant. If they are noisy or off-topic, ignore them.")
    parts.append(
        f"Please choose the best option and conclude with exactly one line in the format: Final Answer: X (where X is one of {choices_list})."
    )
    user_prompt = "\n\n".join([part for part in parts if part.strip()])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]


def build_unified_rag_messages(
    *,
    model_cfg: Dict[str, Any],
    example: Dict[str, Any],
    cfg: Dict[str, Any],
    hits: Sequence[RetrievalHit],
    max_snippets: int = 6,
    max_chars: int = 6000,
) -> List[Dict[str, str]]:
    try:
        from benchio.unified_eval_by_config_0312 import build_messages as build_unified_messages
        from benchio.unified_eval_by_config_0312 import get_allowed_choices
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Unified benchmark prompting requires the optional benchio package."
        ) from exc

    msgs = [
        {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
        for m in build_unified_messages(model_cfg, example, cfg)
    ]

    snippets_text = render_snippets(hits, max_snippets=max_snippets, max_chars=max_chars)
    if not snippets_text:
        return msgs

    ds = str(example.get("dataset", "") or "").strip().lower()
    allowed = get_allowed_choices(example, cfg, ds)
    choices = "|".join(allowed)
    choices_list = ", ".join(allowed)

    suffix = model_cfg.get("user_suffix") or ""
    if "{choices" in suffix:
        suffix = suffix.format(choices=choices, choices_list=choices_list)
    suffix = suffix.strip()

    refs_block = (
        "Relevant Medical References:\n"
        f"{snippets_text}\n\n"
        "Use the references only when they are relevant. "
        "If they are noisy or off-topic, ignore them."
    ).strip()

    for idx in range(len(msgs) - 1, -1, -1):
        if msgs[idx].get("role") != "user":
            continue
        content = str(msgs[idx].get("content", "") or "").rstrip()
        if suffix and content.endswith(suffix):
            prefix = content[: -len(suffix)].rstrip()
            parts = [part for part in [prefix, refs_block, suffix] if part]
            msgs[idx]["content"] = "\n\n".join(parts).strip()
        else:
            parts = [part for part in [content, refs_block] if part]
            msgs[idx]["content"] = "\n\n".join(parts).strip()
        return msgs

    msgs.append({"role": "user", "content": refs_block})
    return msgs

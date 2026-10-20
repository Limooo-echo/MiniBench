"""Validate provider counts without inferring missing tokens or altering raw usage.

Reasoning/cache counts are breakdowns, not additional billable tokens. Partial
reports may supply a known total, but never an invented input/output breakdown.
"""
from __future__ import annotations

from typing import Any

STANDARD_TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def assess_token_usage(
    usage: object, *, reasoning_chars: int = 0, content_chars: int = 0,
) -> dict[str, Any]:
    issues: list[str] = []
    values: dict[str, int] = {}
    if not isinstance(usage, dict):
        return {"status": "missing", "complete": False, "issues": ["usage_missing"],
                "tokens": None, "total_tokens": None}

    def collect(node: dict, path: str = "") -> None:
        for raw_key, value in node.items():
            key = str(raw_key)
            if key in STANDARD_TOKEN_KEYS or key.endswith("_tokens"):
                if key == "audio_tokens" and path in {"prompt_tokens_details", "completion_tokens_details"}:
                    key = path.split("_", 1)[0] + "_audio_tokens"
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    issues.append(f"invalid_token_count:{key}")
                elif key in values and values[key] != value:
                    issues.append(f"conflicting_token_count:{key}")
                else:
                    # Some compatible endpoints repeat details at the top level.
                    # Identical aliases must not be added twice.
                    values[key] = value
            elif isinstance(value, dict):
                collect(value, key)

    collect(usage)
    prompt, completion, total = (values.get(k) for k in STANDARD_TOKEN_KEYS)
    if all(v is not None for v in (prompt, completion, total)) and total != prompt + completion:
        issues.append("total_tokens_not_prompt_plus_completion")
    if total is not None:
        for key in ("prompt_tokens", "completion_tokens"):
            if values.get(key, 0) > total:
                issues.append(f"count_exceeds_total:{key}")
    # A missing parent is not a reported zero. However, total plus the other
    # parent supplies a strict upper bound for consistency checks. Do not add
    # this inferred bound to values: a partial report must remain partial.
    prompt_bound = prompt if prompt is not None else (
        total - completion if total is not None and completion is not None else total)
    completion_bound = completion if completion is not None else (
        total - prompt if total is not None and prompt is not None else total)
    parents = {
        "reasoning_tokens": completion_bound,
        "accepted_prediction_tokens": completion_bound,
        "rejected_prediction_tokens": completion_bound,
        "completion_audio_tokens": completion_bound,
        "cached_tokens": prompt_bound, "prompt_audio_tokens": prompt_bound,
        "prompt_cache_hit_tokens": prompt_bound,
        "prompt_cache_miss_tokens": prompt_bound,
    }
    for key, parent in parents.items():
        if parent is None:
            parent = total
        if parent is not None and values.get(key, 0) > parent:
            issues.append(f"count_exceeds_parent:{key}")
    hit, miss = values.get("prompt_cache_hit_tokens"), values.get("prompt_cache_miss_tokens")
    if prompt is not None and hit is not None and miss is not None and hit + miss != prompt:
        issues.append("prompt_cache_breakdown_mismatch")
    elif prompt_bound is not None and hit is not None and miss is not None and hit + miss > prompt_bound:
        issues.append("prompt_cache_breakdown_exceeds_parent")
    if reasoning_chars and (completion_bound == 0 or total == 0 or values.get("reasoning_tokens") == 0):
        issues.append("nonempty_reasoning_but_zero_reported_tokens")
    if content_chars and (completion_bound == 0 or total == 0):
        issues.append("nonempty_content_but_zero_reported_tokens")
    if not any(k in values for k in STANDARD_TOKEN_KEYS):
        issues.append("standard_token_counts_missing")
    issues = list(dict.fromkeys(issues))
    invalid = bool(issues)
    complete = not invalid and all(k in values for k in STANDARD_TOKEN_KEYS)
    return {
        "status": "invalid" if invalid else "valid" if complete else "partial",
        "complete": complete, "issues": issues,
        "tokens": None if invalid else values,
        "total_tokens": None if invalid else total,
    }


def known_total_tokens(usage: object) -> int | None:
    """Accept a coherent reported total, including a total-only response."""
    return assess_token_usage(usage)["total_tokens"]

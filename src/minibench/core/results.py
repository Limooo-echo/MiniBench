from __future__ import annotations

from typing import Any, Iterable


def summarize_boolean_results(
    results: Iterable[Any],
    *,
    success_attr: str,
    success_key: str,
    rate_key: str,
) -> dict[str, Any]:
    items = list(results)
    total = len(items)
    success_count = sum(1 for item in items if bool(getattr(item, success_attr)))
    by_tag: dict[str, dict[str, int | float]] = {}

    for result in items:
        for tag in getattr(result, "tags", ()):
            bucket = by_tag.setdefault(
                tag,
                {"total": 0, success_key: 0, rate_key: 0.0},
            )
            bucket["total"] = int(bucket["total"]) + 1
            bucket[success_key] = int(bucket[success_key]) + int(
                bool(getattr(result, success_attr))
            )

    for bucket in by_tag.values():
        bucket[rate_key] = int(bucket[success_key]) / int(bucket["total"])

    return {
        "total": total,
        success_key: success_count,
        rate_key: success_count / total if total else 0.0,
        "by_tag": by_tag,
    }

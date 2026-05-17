from __future__ import annotations

import json
import re
from typing import Any


DEFAULT_EXTRACTORS = (
    r'(?is)"answer"\s*:\s*"([^"]+)"',
    r"(?im)^answer\s*[:=]\s*(.+)$",
    r"(?im)^final answer\s*[:=]\s*(.+)$",
)


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip("`'\" ")


def _first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def extract_answer(output: str, extractors: tuple[str, ...] = ()) -> tuple[str | None, str]:
    parsed = _first_json_object(output)
    if parsed is not None and "answer" in parsed:
        return normalize_answer(parsed["answer"]), "json.answer"

    for pattern in tuple(extractors) + DEFAULT_EXTRACTORS:
        match = re.search(pattern, output)
        if match:
            group = match.groupdict().get("answer") if match.groupdict() else match.group(1)
            return normalize_answer(group), f"regex:{pattern}"

    return None, "none"


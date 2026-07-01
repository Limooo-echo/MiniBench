from __future__ import annotations

import json
import re
from typing import Any


DEFAULT_EXTRACTORS = (
    r'(?is)"answer"\s*:\s*"([^"]+)"',
    r"(?im)^answer\s*[:=]\s*([ABCD])\b",
    r"(?im)^final answer\s*[:=]\s*([ABCD])\b",
    r"(?im)\b(?:option|choice)\s*[:=]\s*([ABCD])\b",
)


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip("`'\" ")


def normalize_choice(value: Any) -> str | None:
    text = normalize_answer(value).upper()
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    if len(text) >= 1 and text[0] in {"A", "B", "C", "D"}:
        return text[0]
    return None


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
        choice = normalize_choice(parsed["answer"])
        return choice, "json.answer" if choice else "json.answer.invalid_choice"

    for pattern in tuple(extractors) + DEFAULT_EXTRACTORS:
        match = re.search(pattern, output)
        if match:
            group = match.groupdict().get("answer") if match.groupdict() else match.group(1)
            choice = normalize_choice(group)
            return choice, f"regex:{pattern}" if choice else f"regex.invalid_choice:{pattern}"

    return None, "none"

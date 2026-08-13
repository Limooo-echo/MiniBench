from __future__ import annotations

import base64
from dataclasses import dataclass
import mimetypes
from pathlib import Path
import random
from typing import Any, Literal, Sequence


SUPPORTED_IMAGE_MIME_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/gif"}
)


@dataclass(frozen=True)
class ImageAttachment:
    """One image supplied to a multimodal model call.

    Exactly one of ``path`` and ``data`` is required.  Keeping raw image handling
    here prevents task-specific evaluators from duplicating MIME and base64 logic.
    """

    path: Path | None = None
    data: bytes | None = None
    mime_type: str | None = None
    detail: Literal["auto", "low", "high"] = "high"

    def __post_init__(self) -> None:
        if (self.path is None) == (self.data is None):
            raise ValueError("ImageAttachment requires exactly one of path or data")
        if self.detail not in {"auto", "low", "high"}:
            raise ValueError("image detail must be auto, low, or high")
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))
        if self.mime_type is not None and self.mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            raise ValueError(f"unsupported image MIME type: {self.mime_type!r}")

    def resolved(self) -> tuple[bytes, str]:
        if self.path is not None:
            if not self.path.is_file():
                raise RuntimeError(f"image file does not exist: {self.path}")
            data = self.path.read_bytes()
            extension_type, _encoding = mimetypes.guess_type(self.path.name)
        else:
            assert self.data is not None
            data = self.data
            extension_type = None

        sniffed_type = _sniff_image_mime_type(data)
        if sniffed_type is None:
            raise RuntimeError("unsupported image bytes or image format")
        mime_type = self.mime_type or sniffed_type or extension_type
        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            raise RuntimeError(f"unsupported image MIME type: {mime_type!r}")
        if sniffed_type is not None and mime_type != sniffed_type:
            raise RuntimeError(
                f"image MIME type {mime_type!r} does not match bytes {sniffed_type!r}"
            )
        return data, mime_type

    def data_url(self) -> str:
        data, mime_type = self.resolved()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def content_part(self) -> dict[str, object]:
        return {
            "type": "image_url",
            "image_url": {"url": self.data_url(), "detail": self.detail},
        }


def _sniff_image_mime_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def summarize_paired_modes(
    results: Sequence[Any],
    *,
    baseline_mode: str = "text",
    bootstrap_samples: int = 2000,
    seed: int = 20260813,
) -> dict[str, Any]:
    """Summarize paired input modes using task success as the primary score."""

    by_mode: dict[str, list[Any]] = {}
    paired: dict[tuple[str, str], float] = {}
    for result in results:
        mode = str(_field(result, "input_mode"))
        source_id = str(_field(result, "source_task_id"))
        score = float(bool(_field(result, "success")))
        by_mode.setdefault(mode, []).append(result)
        key = (source_id, mode)
        if key in paired:
            raise ValueError(f"duplicate paired result for {source_id}/{mode}")
        paired[key] = score

    mode_summary = {
        mode: {
            "total": len(selected),
            "success": sum(int(bool(_field(item, "success"))) for item in selected),
            "success_rate": (
                sum(int(bool(_field(item, "success"))) for item in selected)
                / len(selected)
                if selected
                else 0.0
            ),
        }
        for mode, selected in sorted(by_mode.items())
    }

    gaps: dict[str, dict[str, Any]] = {}
    if baseline_mode in by_mode:
        baseline_ids = {
            str(_field(item, "source_task_id")) for item in by_mode[baseline_mode]
        }
        for mode in sorted(set(by_mode) - {baseline_mode}):
            mode_ids = {str(_field(item, "source_task_id")) for item in by_mode[mode]}
            source_ids = sorted(baseline_ids & mode_ids)
            differences = [
                paired[(source_id, baseline_mode)] - paired[(source_id, mode)]
                for source_id in source_ids
            ]
            gap = sum(differences) / len(differences) if differences else None
            low, high = _bootstrap_mean_ci(
                differences,
                samples=bootstrap_samples,
                seed=seed + sum(ord(char) for char in mode),
            )
            gaps[mode] = {
                "baseline_mode": baseline_mode,
                "paired_total": len(differences),
                "visual_gap": gap,
                "ci95": [low, high] if low is not None else None,
            }

    return {"by_input_mode": mode_summary, "visual_gap": gaps}


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    low_index = max(0, int(0.025 * samples) - 1)
    high_index = min(samples - 1, int(0.975 * samples))
    return estimates[low_index], estimates[high_index]


def _field(record: Any, name: str) -> Any:
    if isinstance(record, dict):
        return record[name]
    return getattr(record, name)

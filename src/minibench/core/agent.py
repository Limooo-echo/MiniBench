from __future__ import annotations

import base64
from dataclasses import dataclass
import mimetypes
from pathlib import Path
from typing import Any, Protocol


class Agent:
    name = "base"

    def generate(self, prompt: str, task: Any) -> str:
        raise NotImplementedError

class ChatClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
        image_data_url: str | None = None,
    ) -> str:
        raise NotImplementedError


def task_image_data_url(task: Any) -> str | None:
    """Load a task's local raster image as an API-ready data URL."""
    image_path = getattr(task, "image_path", None)
    if image_path is None:
        return None
    path = Path(image_path)
    if not path.is_file():
        raise RuntimeError(f"task image does not exist: {path}")
    mime_type, _encoding = mimetypes.guess_type(path.name)
    if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise RuntimeError(
            f"unsupported multimodal task image type {mime_type!r}: {path}"
        )
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@dataclass(frozen=True)
class ReasoningConfig:
    samples: int = 3
    reasoning_temperature: float = 0.0
    final_temperature: float = 0.0
    max_reasoning_tokens: int = 512
    final_max_tokens: int = 64

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError("samples must be at least 1")
        if self.max_reasoning_tokens < 1:
            raise ValueError("max_reasoning_tokens must be at least 1")
        if self.final_max_tokens < 1:
            raise ValueError("final_max_tokens must be at least 1")

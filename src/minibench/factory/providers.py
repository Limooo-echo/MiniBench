from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
import socket
from time import perf_counter, sleep
from typing import Any, Sequence
import urllib.error
import urllib.request
import warnings

from minibench.core.agent import Agent, ChatMessage, CompletionResult, MessagePhase
from minibench.core.metrics import empty_token_usage, extract_token_usage
from minibench.core.multimodal import ImageAttachment
from minibench.core.prompts import FINAL_ANSWER_SYSTEM_PROMPT


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key_env: str
    default_model: str | None


RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 429})


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code in RETRYABLE_HTTP_STATUS_CODES or 500 <= status_code <= 599


def _retry_after_seconds(error: urllib.error.HTTPError) -> float | None:
    headers = error.headers
    if headers is None:
        return None
    raw_value = headers.get("Retry-After")
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


PROVIDERS = {
    "deepseek": ProviderConfig(
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-v4-flash",
    ),
    "qwen": ProviderConfig(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        default_model="qwen3.8-max",
    ),
    "qwen-intl": ProviderConfig(
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        default_model="qwen3.8-max",
    ),
    "qwen-us": ProviderConfig(
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        default_model="qwen3.8-max",
    ),
    "siliconflow": ProviderConfig(
        base_url="https://api.siliconflow.cn/v1",
        api_key_env="SILICONFLOW_API_KEY",
        default_model=None,
    ),
}


class OpenAICompatibleClient:
    """Transport client for OpenAI-compatible Chat Completions endpoints."""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str,
        temperature: float = 0.0,
        max_tokens: int = 64,
        timeout: int = 60,
        json_mode: bool = False,
        extra_body: dict[str, object] | None = None,
        default_system_prompt: str | None = None,
        max_retries: int = 0,
        retry_initial_backoff_seconds: float = 1.0,
        retry_max_backoff_seconds: float = 30.0,
    ):
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ValueError("max_retries must be an integer")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_initial_backoff_seconds < 0:
            raise ValueError("retry_initial_backoff_seconds must be non-negative")
        if retry_max_backoff_seconds < retry_initial_backoff_seconds:
            raise ValueError(
                "retry_max_backoff_seconds must be greater than or equal to "
                "retry_initial_backoff_seconds"
            )
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.json_mode = json_mode
        self.extra_body = extra_body or {}
        self.default_system_prompt = default_system_prompt
        self.max_retries = max_retries
        self.retry_initial_backoff_seconds = float(retry_initial_backoff_seconds)
        self.retry_max_backoff_seconds = float(retry_max_backoff_seconds)
        self._model_elapsed_seconds = 0.0
        self._llm_calls = 0
        self._usage_missing_calls = 0
        self._token_usage = empty_token_usage()

    @property
    def endpoint(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def build_payload(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
        images: Sequence[ImageAttachment] = (),
    ) -> dict[str, object]:
        return self.build_messages_payload(
            [{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            images=images,
        )

    def build_messages_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
        images: Sequence[ImageAttachment] = (),
    ) -> dict[str, object]:
        normalized_messages = _validate_messages(messages)
        if not any(message["role"] == "system" for message in normalized_messages):
            normalized_messages.insert(
                0,
                {"role": "system", "content": self._system_prompt(system_prompt)},
            )
        elif system_prompt is not None:
            raise ValueError(
                "system_prompt cannot be combined with messages that already include "
                "a system role"
            )

        if images:
            _attach_images_to_last_user_message(normalized_messages, images)

        payload: dict[str, object] = {
            "model": self.model,
            "messages": normalized_messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        use_json_mode = self.json_mode if json_mode is None else json_mode
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.extra_body)
        return payload

    def _system_prompt(self, phase_prompt: str | None = None) -> str:
        prompts = [
            prompt for prompt in (self.default_system_prompt, phase_prompt) if prompt
        ]
        if prompts:
            return "\n\n".join(prompts)
        return FINAL_ANSWER_SYSTEM_PROMPT

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
        images: Sequence[ImageAttachment] = (),
    ) -> str:
        return self.complete_result(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            images=images,
        ).content

    def complete_result(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
        images: Sequence[ImageAttachment] = (),
    ) -> CompletionResult:
        return self.complete_messages_result(
            [{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            images=images,
        )

    def complete_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
        images: Sequence[ImageAttachment] = (),
    ) -> str:
        return self.complete_messages_result(
            messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            images=images,
        ).content

    def complete_messages_result(
        self,
        messages: Sequence[ChatMessage],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
        images: Sequence[ImageAttachment] = (),
    ) -> CompletionResult:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set ${self.api_key_env} before using "
                f"{self.name}."
            )

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(
                self.build_messages_payload(
                    messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    images=images,
                )
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for retry_index in range(self.max_retries + 1):
            try:
                return self._complete_request_once(request)
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if retry_index < self.max_retries and _is_retryable_http_status(
                    exc.code
                ):
                    self._sleep_before_retry(retry_index + 1, exc)
                    continue
                raise RuntimeError(
                    f"{self.name} request failed with HTTP {exc.code}: {error_body}"
                ) from exc
            except urllib.error.URLError as exc:
                if retry_index < self.max_retries:
                    self._sleep_before_retry(retry_index + 1)
                    continue
                raise RuntimeError(f"{self.name} request failed: {exc.reason}") from exc
            except (TimeoutError, socket.timeout) as exc:
                if retry_index < self.max_retries:
                    self._sleep_before_retry(retry_index + 1)
                    continue
                raise RuntimeError(f"{self.name} request timed out: {exc}") from exc
        raise AssertionError("retry loop terminated without returning or raising")

    def _complete_request_once(
        self, request: urllib.request.Request
    ) -> CompletionResult:
        started_at = perf_counter()
        elapsed_seconds: float | None = None
        usage: object = None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Unexpected chat completion response object: {raw}")
            usage = payload.get("usage")
            try:
                choice = payload["choices"][0]
                message = choice["message"]
                content = _message_text(message)
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"Unexpected chat completion response: {raw}"
                ) from exc

            if not isinstance(choice, dict) or not isinstance(message, dict):
                raise RuntimeError(f"Unexpected chat completion response: {raw}")

            finish_reason_value = choice.get("finish_reason")
            finish_reason = (
                finish_reason_value if isinstance(finish_reason_value, str) else None
            )
            if finish_reason == "length":
                raise RuntimeError(
                    "OpenAI-compatible response was truncated "
                    "(finish_reason=length). Increase max_tokens or reduce the "
                    "requested output size."
                )

            if not isinstance(content, str):
                raise RuntimeError(f"Unexpected message content in response: {raw}")

            if not content.strip():
                message_keys = ", ".join(sorted(str(key) for key in message.keys()))
                raise RuntimeError(
                    "OpenAI-compatible response had empty message content "
                    f"(finish_reason={finish_reason}, message_keys=[{message_keys}]). "
                    "Try increasing --max-tokens, disabling provider thinking mode via "
                    "--extra-body-json, or using a non-reasoning/chat model."
                )

            elapsed_seconds = perf_counter() - started_at
            model = payload.get("model")
            response_id = payload.get("id")
            return CompletionResult(
                content=content,
                reasoning=_message_reasoning(message),
                finish_reason=finish_reason,
                usage=usage if isinstance(usage, dict) else None,
                model=model if isinstance(model, str) else None,
                response_id=response_id if isinstance(response_id, str) else None,
                elapsed_seconds=elapsed_seconds,
            )
        finally:
            if elapsed_seconds is None:
                elapsed_seconds = perf_counter() - started_at
            self._record_completion_metrics(elapsed_seconds, usage)

    def _sleep_before_retry(
        self,
        retry_number: int,
        http_error: urllib.error.HTTPError | None = None,
    ) -> None:
        delay = min(
            self.retry_max_backoff_seconds,
            self.retry_initial_backoff_seconds * (2 ** (retry_number - 1)),
        )
        if http_error is not None:
            retry_after = _retry_after_seconds(http_error)
            if retry_after is not None:
                delay = max(
                    delay,
                    min(self.retry_max_backoff_seconds, retry_after),
                )
        if delay > 0:
            sleep(delay)

    def metrics_snapshot(self) -> dict[str, Any]:
        return {
            "model_elapsed_seconds": self._model_elapsed_seconds,
            "llm_calls": self._llm_calls,
            "usage_missing_calls": self._usage_missing_calls,
            "token_usage": dict(self._token_usage),
        }

    def _record_completion_metrics(
        self,
        elapsed_seconds: float,
        usage: object,
    ) -> None:
        self._model_elapsed_seconds += elapsed_seconds
        self._llm_calls += 1
        token_usage = extract_token_usage(usage)
        if token_usage is None:
            self._usage_missing_calls += 1
            return
        for key, value in token_usage.items():
            self._token_usage[key] = self._token_usage.get(key, 0) + value


class OpenAICompatibleAgent(OpenAICompatibleClient, Agent):
    """Deprecated combined transport/agent kept for Python API compatibility."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "OpenAICompatibleAgent is deprecated; compose OpenAICompatibleClient "
            "with PassthroughAgent or another agent architecture instead",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    def generate(self, prompt: str, task: Any) -> str:
        return self.complete(prompt)

    def generate_multimodal(
        self,
        prompt: str,
        task: Any,
        *,
        images: Sequence[ImageAttachment],
    ) -> str:
        return self.complete(prompt, images=images)

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        task: Any,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
        images: Sequence[ImageAttachment] = (),
    ) -> str:
        return self.complete_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            images=images,
        )

    def generate_messages_for_phase(
        self,
        messages: Sequence[ChatMessage],
        task: Any,
        *,
        phase: MessagePhase,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
    ) -> str:
        if phase not in ("intermediate", "final"):
            raise ValueError(
                f"Unsupported message phase {phase!r}; expected 'intermediate' or 'final'"
            )
        return self.generate_messages(
            messages,
            task,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )


def _content_part_to_text(part: object) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""
    text = part.get("text")
    if isinstance(text, str):
        return text
    content = part.get("content")
    if isinstance(content, str):
        return content
    return ""


def _validate_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    if not messages:
        raise ValueError("messages must not be empty")
    normalized: list[ChatMessage] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"message {index} role must be system, user, or assistant")
        if not isinstance(content, str):
            raise ValueError(f"message {index} content must be a string")
        normalized.append({"role": role, "content": content})
    return normalized


def _attach_images_to_last_user_message(
    messages: list[ChatMessage],
    images: Sequence[ImageAttachment],
) -> None:
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        text = message["content"]
        if not isinstance(text, str):
            raise ValueError("cannot attach images to a non-text user message")
        message["content"] = [
            {"type": "text", "text": text},
            *(image.content_part() for image in images),
        ]
        return
    raise ValueError("images require at least one user message")


def _content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_part_to_text(part) for part in content)
    return ""


def _message_text(message: object) -> str:
    if not isinstance(message, dict):
        return ""
    return _content_to_text(message.get("content"))


def _message_reasoning(message: object) -> str | None:
    if not isinstance(message, dict):
        return None
    for key in ("reasoning_content", "reasoning", "reasoning_details"):
        reasoning = _content_to_text(message.get(key))
        if reasoning.strip():
            return reasoning
    return None


def resolve_provider(
    provider: str,
    *,
    model: str | None,
    base_url: str | None,
    api_key_env: str | None,
) -> tuple[str, str, str]:
    if provider == "generic":
        if not model or not base_url or not api_key_env:
            raise ValueError(
                "generic provider requires --model, --base-url, and --api-key-env"
            )
        return model, base_url, api_key_env

    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider}")

    config = PROVIDERS[provider]
    if not model and not config.default_model:
        raise ValueError(f"{provider} provider requires --model")
    return (
        model or config.default_model,
        base_url or config.base_url,
        api_key_env or config.api_key_env,
    )

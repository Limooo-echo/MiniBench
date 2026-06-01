from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
import urllib.error
import urllib.request

from minibench.agents.base import Agent
from minibench.agents.prompts import FINAL_ANSWER_SYSTEM_PROMPT


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key_env: str
    default_model: str | None


PROVIDERS = {
    "deepseek": ProviderConfig(
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-v4-flash",
    ),
    "qwen": ProviderConfig(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        default_model="qwen3.6-plus",
    ),
    "qwen-intl": ProviderConfig(
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        default_model="qwen3.6-plus",
    ),
    "qwen-us": ProviderConfig(
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        default_model="qwen3.6-plus",
    ),
    "siliconflow": ProviderConfig(
        base_url="https://api.siliconflow.cn/v1",
        api_key_env="SILICONFLOW_API_KEY",
        default_model=None,
    ),
}


class OpenAICompatibleAgent(Agent):
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
    ):
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.json_mode = json_mode
        self.extra_body = extra_body or {}
        self.default_system_prompt = default_system_prompt

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
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        system_prompt
                        or self.default_system_prompt
                        or FINAL_ANSWER_SYSTEM_PROMPT
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        use_json_mode = self.json_mode if json_mode is None else json_mode
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.extra_body)
        return payload

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
    ) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set ${self.api_key_env} before using "
                f"{self.name}."
            )

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(
                self.build_payload(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{self.name} request failed with HTTP {exc.code}: {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{self.name} request failed: {exc.reason}") from exc

        payload = json.loads(raw)
        try:
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat completion response: {raw}") from exc

        if isinstance(content, list):
            content = "".join(
                _content_part_to_text(part)
                for part in content
            )

        if content is None:
            content = ""

        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected message content in response: {raw}")

        if not content.strip():
            finish_reason = choice.get("finish_reason")
            message_keys = ", ".join(sorted(str(key) for key in message.keys()))
            raise RuntimeError(
                "OpenAI-compatible response had empty message content "
                f"(finish_reason={finish_reason}, message_keys=[{message_keys}]). "
                "Try increasing --max-tokens, disabling provider thinking mode via "
                "--extra-body-json, or using a non-reasoning/chat model."
            )

        return content

    def generate(self, prompt: str, task: Any) -> str:
        return self.complete(prompt)


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

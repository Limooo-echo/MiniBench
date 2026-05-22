from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from minibench.dataset import Task


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


DEFAULT_MULTIPLE_CHOICE_SYSTEM_PROMPT = (
    "You are answering a multiple-choice benchmark. "
    'Return exactly one JSON object like {"answer":"A"}.'
)


class Agent:
    name = "base"

    def generate(self, prompt: str, task: Any) -> str:
        raise NotImplementedError


class OracleAgent(Agent):
    name = "oracle"

    def generate(self, prompt: str, task: Task) -> str:
        return json.dumps({"answer": task.correct_option}, ensure_ascii=False)


class NoisyAgent(Agent):
    name = "noisy"

    def generate(self, prompt: str, task: Task) -> str:
        answer = task.correct_option
        return f"I worked it out. answer: {answer}"


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
        system_prompt: str | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.json_mode = json_mode
        self.extra_body = extra_body or {}
        self.system_prompt = system_prompt or DEFAULT_MULTIPLE_CHOICE_SYSTEM_PROMPT

    @property
    def endpoint(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def build_payload(self, prompt: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        payload.update(self.extra_body)
        return payload

    def generate(self, prompt: str, task: Any) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set ${self.api_key_env} before using "
                f"{self.name}."
            )

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(self.build_payload(prompt)).encode("utf-8"),
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
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
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


class PredictionFileAgent(Agent):
    name = "prediction-file"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.outputs = self._load_outputs()

    def _load_outputs(self) -> dict[str, str]:
        outputs: dict[str, str] = {}

        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                record = json.loads(line)
                task_id = record.get("task_id") or record.get("id")
                output = record.get("raw_output") or record.get("output")

                if not isinstance(task_id, str) or not isinstance(output, str):
                    raise ValueError(
                        f"{self.path}:{line_number}: expected task_id and raw_output"
                    )

                outputs[task_id] = output

        return outputs

    def generate(self, prompt: str, task: Any) -> str:
        task_id = getattr(task, "id", None)
        if not isinstance(task_id, str):
            raise ValueError("prediction-file agent requires task.id")

        if task_id not in self.outputs:
            raise KeyError(f"prediction file has no output for task {task_id}")

        return self.outputs[task_id]


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


def make_agent(
    name: str,
    predictions: str | Path | None = None,
    *,
    provider: str = "generic",
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 64,
    timeout: int = 60,
    json_mode: bool = False,
    extra_body: dict[str, object] | None = None,
    system_prompt: str | None = None,
) -> Agent:
    if predictions:
        return PredictionFileAgent(predictions)

    if name == "oracle":
        return OracleAgent()

    if name == "noisy":
        return NoisyAgent()

    if name == "openai-compatible":
        resolved_model, resolved_base_url, resolved_api_key_env = resolve_provider(
            provider,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
        )

        return OpenAICompatibleAgent(
            model=resolved_model,
            base_url=resolved_base_url,
            api_key_env=resolved_api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            json_mode=json_mode,
            extra_body=extra_body,
            system_prompt=system_prompt,
        )

    raise ValueError(f"unknown agent: {name}")

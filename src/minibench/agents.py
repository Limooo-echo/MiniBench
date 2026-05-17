from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
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


class Agent:
    name = "base"

    def generate(self, prompt: str, task: Task) -> str:
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
    ):
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.json_mode = json_mode
        self.extra_body = extra_body or {}

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
                    "content": (
                        "You are answering a multiple-choice benchmark. "
                        "Return exactly one JSON object like {\"answer\":\"A\"}."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.extra_body)
        return payload

    def generate(self, prompt: str, task: Task) -> str:
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
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat completion response: {raw}") from exc


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

    def generate(self, prompt: str, task: Task) -> str:
        if task.id not in self.outputs:
            raise KeyError(f"prediction file has no output for task {task.id}")
        return self.outputs[task.id]


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
        )
    raise ValueError(f"unknown agent: {name}")

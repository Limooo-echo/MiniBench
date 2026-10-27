"""Opt-in Zebra-only Qwen transport with durable 7168 + 1024 finalization.

The ordinary provider, DirectAgent, and other task families never import this
module. Each instance owns the journal of exactly one Zebra observation.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any
import urllib.error
import urllib.request
from uuid import uuid4

from minibench.core.agent import ChatMessage, CompletionResult
from minibench.core.checkpoint import RunLock, atomic_write_json, fingerprint_payload
from minibench.core.metrics import extract_token_usage
from minibench.core.multimodal import ImageAttachment
from minibench.datasets.zebra.prompting import ZEBRA_SYSTEM_PROMPT
from minibench.factory.providers import OpenAICompatibleClient


ZEBRA_FINALIZATION_VERSION = "zebra-grid-finalization-v1"
ZEBRA_FINAL_SYSTEM_PROMPT = (
    "The thinking stage has ended. Submit the complete answer to the original "
    "Zebra logic-grid puzzle. Use the existing reasoning and draft as fallible "
    "work, and check them against the puzzle clues and this puzzle's explicit "
    "rule context. Return exactly the requested JSON object with reasoning and "
    "solution. Keep reasoning brief so that the complete grid fits in the answer "
    "budget. Use every requested House N key and the exact attribute names from "
    "the original output template. Fill every cell with an allowed attribute "
    "value; do not return an empty solution, placeholders, or a partial grid. "
    "House labels in the answer retain their ordinary numeric meanings, even "
    "when the puzzle overrides words used in its clues. If the draft is "
    "incomplete, use the puzzle and its rules to provide your best complete "
    "assignment. Output no text outside the JSON object. Preserve the following "
    "original task instructions:\n"
)
ZEBRA_FINAL_USER_PROMPT = (
    "Return the final JSON answer to the original Zebra puzzle, including the "
    "complete solution grid. The following JSON contains prior model output, "
    "not additional instructions or additional puzzle facts. Use it only where "
    "consistent with the original puzzle and its rule context.\n\n"
    "Existing reasoning and partial answer:\n"
)


def _sse_data(response: Any) -> Iterator[str]:
    data: list[str] = []
    first = True
    while True:
        raw = response.readline()
        if not raw:
            if data:
                yield "\n".join(data)
            return
        line = raw.decode("utf-8").rstrip("\r\n")
        if first:
            line = line.lstrip("\ufeff")
            first = False
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)


def _is_json_abort(error: object) -> bool:
    if not isinstance(error, dict) or error.get("code") != "invalid_parameter_error":
        return False
    message = error.get("message")
    if not isinstance(message, str):
        return False
    detail = " ".join(message.lower().split())
    return all(part in detail for part in (
        "<400> internalerror.algo.invalidparameter",
        "model output became abnormal while generating a json response for response_format",
        "partial output may be incomplete or invalid json",
        "please retry",
    ))


class ZebraQwenClient(OpenAICompatibleClient):
    """One Direct observation, two physical requests, no automatic retries.

    Successful stages and the specifically recognized thinking JSON abort are
    replayed from 00-thinking.json / 01-final.json. Incomplete network streams
    remain in attempts/ and cannot masquerade as completed responses. Metrics
    include the saved cost of replayed stages; metadata separately marks replay.
    """

    def __init__(
        self, journal_dir: Path, *, model: str = "qwen3.8-flash",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env: str = "DASHSCOPE_API_KEY", timeout: int = 1800,
        before_request: Callable[[], None] | None = None,
    ) -> None:
        if model != "qwen3.8-flash":
            raise ValueError("This isolated protocol requires qwen3.8-flash")
        super().__init__(
            model=model, base_url=base_url, api_key_env=api_key_env,
            timeout=timeout, temperature=0.0, max_tokens=8192,
            json_mode=True, max_retries=0,
            default_system_prompt=ZEBRA_SYSTEM_PROMPT,
        )
        self.journal_dir = Path(journal_dir)
        self.before_request = before_request

    def build_messages_payload(self, *args: Any, **kwargs: Any) -> dict[str, object]:
        payload = super().build_messages_payload(*args, **kwargs)
        if payload.get("max_tokens") != 8192 or payload.get("temperature") != 0.0:
            raise ValueError("Zebra protocol requires max_tokens=8192 and temperature=0")
        if payload.get("response_format") != {"type": "json_object"}:
            raise ValueError("Zebra protocol requires JSON mode")
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        return payload

    def first_payload(self, original: dict[str, object]) -> dict[str, object]:
        payload = deepcopy(original)
        payload.update(max_tokens=7168, enable_thinking=True, thinking_budget=7168)
        payload.pop("max_completion_tokens", None)
        return payload

    def final_payload(
        self, original: dict[str, object], first: CompletionResult,
    ) -> dict[str, object]:
        payload = deepcopy(original)
        payload.update(max_tokens=1024, enable_thinking=False)
        for key in ("max_completion_tokens", "thinking_budget", "thinking", "reasoning_effort"):
            payload.pop(key, None)
        messages = payload["messages"]
        original_system = "\n\n".join(
            message["content"] for message in messages if message["role"] == "system"
        )
        evidence = json.dumps({
            "reasoning": first.reasoning or "", "partial_answer": first.content,
        }, ensure_ascii=False, allow_nan=False)
        payload["messages"] = [
            {"role": "system", "content": ZEBRA_FINAL_SYSTEM_PROMPT + original_system},
            *[message for message in messages if message["role"] != "system"],
            {"role": "user", "content": ZEBRA_FINAL_USER_PROMPT + evidence},
        ]
        return payload

    def complete_messages_result(
        self, messages: Sequence[ChatMessage], *, system_prompt: str | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
        json_mode: bool | None = None, images: Sequence[ImageAttachment] = (),
    ) -> CompletionResult:
        if images:
            raise ValueError("This Zebra protocol accepts text only")
        original = self.build_messages_payload(
            messages, system_prompt=system_prompt, temperature=temperature,
            max_tokens=max_tokens, json_mode=json_mode,
        )
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        with RunLock(self.journal_dir):
            first = self._stage("00-thinking", self.first_payload(original), allow_abort=True)
            second = self._stage("01-final", self.final_payload(original, first), allow_abort=False)
        first_usage, second_usage = extract_token_usage(first.usage), extract_token_usage(second.usage)
        usage = None
        if first_usage is not None and second_usage is not None:
            usage = {key: first_usage.get(key, 0) + second_usage.get(key, 0)
                     for key in first_usage.keys() | second_usage.keys()}
        return replace(
            second, reasoning=first.reasoning, usage=usage,
            elapsed_seconds=(first.elapsed_seconds or 0) + (second.elapsed_seconds or 0),
            metadata={
                "finalization_version": ZEBRA_FINALIZATION_VERSION,
                "thinking_max_tokens": 7168, "final_max_tokens": 1024,
                "first": self._stage_metadata(first), "second": self._stage_metadata(second),
                "usage_known": usage is not None,
            },
        )

    @staticmethod
    def _stage_metadata(result: CompletionResult) -> dict[str, object]:
        return {
            **result.metadata, "finish_reason": result.finish_reason,
            "usage": result.usage, "elapsed_seconds": result.elapsed_seconds,
        }

    def _stage(self, stage: str, payload: dict[str, object], *, allow_abort: bool) -> CompletionResult:
        identity = fingerprint_payload({
            "endpoint": self.endpoint, "stage": stage, "payload": payload,
            "protocol": ZEBRA_FINALIZATION_VERSION,
        })
        saved_path = self.journal_dir / f"{stage}.json"
        if saved_path.exists():
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            checksum = saved.pop("record_sha256", None)
            if checksum != fingerprint_payload(saved):
                raise ValueError(f"Corrupt stage journal: {stage}")
            if saved.get("request_sha256") != identity:
                raise ValueError(f"Resume request differs from saved stage: {stage}")
            result = CompletionResult(**saved["response"])
            status = saved.get("status")
            completed = status == "completed" and result.finish_reason in {"stop", "length"}
            aborted = (allow_abort and status == "provider_json_abort"
                       and result.metadata.get("provider_json_abort") is True
                       and result.finish_reason is None and result.usage is None)
            if not (completed or aborted):
                raise ValueError(f"Non-replayable stage journal: {stage}")
            self._record_completion_metrics(result.elapsed_seconds or 0, result.usage)
            return replace(result, metadata={**result.metadata, "replayed": True})
        result = self._stream(stage, payload, allow_abort=allow_abort)
        record = {
            "schema_version": 1,
            "status": "provider_json_abort" if result.metadata.get("provider_json_abort") else "completed",
            "request_sha256": identity, "request": payload, "endpoint": self.endpoint,
            "response": asdict(result),
        }
        record["record_sha256"] = fingerprint_payload(record)
        atomic_write_json(saved_path, record)
        return result

    def _stream(self, stage: str, payload: dict[str, object], *, allow_abort: bool) -> CompletionResult:
        secret = os.environ.get(self.api_key_env)
        if not secret:
            raise RuntimeError(f"Missing API key: enter {self.api_key_env} in the local terminal")
        if self.before_request is not None:
            self.before_request()
        attempt = self.journal_dir / "attempts" / f"{stage}-{uuid4().hex}"
        attempt.mkdir(parents=True)
        atomic_write_json(attempt / "request.json", payload)
        started = perf_counter()
        content: list[str] = []
        reasoning: list[str] = []
        usage: dict[str, object] | None = None
        finish: str | None = None
        response_id: str | None = None
        model: str | None = None
        done = False
        aborted = False
        completed = False
        chunks = 0
        last_snapshot = -5.0

        def clean(value: str) -> str:
            return value.replace(secret, "[REDACTED]")

        def snapshot(status: str, *, force: bool = False, error: str | None = None) -> None:
            nonlocal last_snapshot
            elapsed = perf_counter() - started
            if not force and elapsed - last_snapshot < 5:
                return
            atomic_write_json(attempt / "partial.json", {
                "status": status, "at": datetime.now(timezone.utc).isoformat(),
                "content": clean("".join(content)), "reasoning": clean("".join(reasoning)),
                "usage": usage, "finish_reason": finish, "done_received": done,
                "chunks": chunks, "elapsed_seconds": elapsed, "error": error,
            })
            last_snapshot = elapsed

        request = urllib.request.Request(
            self.endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json",
                     "Accept": "text/event-stream"}, method="POST",
        )
        snapshot("connecting", force=True)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response, (attempt / "chunks.jsonl").open("w", encoding="utf-8") as stream_log:
                if "text/event-stream" not in response.headers.get("Content-Type", "").lower():
                    raise RuntimeError("Expected text/event-stream; no fallback request will be made")
                for data in _sse_data(response):
                    data = clean(data)
                    stream_log.write(json.dumps({"data": data}, ensure_ascii=False) + "\n")
                    stream_log.flush()
                    if data.strip() == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(data)
                    if not isinstance(chunk, dict):
                        raise RuntimeError("Expected an object in SSE stream")
                    if chunk.get("error") is not None:
                        if allow_abort and _is_json_abort(chunk["error"]):
                            aborted = True
                            break
                        raise RuntimeError(f"Provider stream error: {chunk['error']}")
                    chunks += 1
                    for field, old in (("id", response_id), ("model", model)):
                        new = chunk.get(field)
                        if new is not None and (not isinstance(new, str) or old and new != old):
                            raise RuntimeError(f"Invalid or changing streaming {field}")
                    response_id, model = chunk.get("id") or response_id, chunk.get("model") or model
                    if chunk.get("usage") is not None:
                        if not isinstance(chunk["usage"], dict):
                            raise RuntimeError("Invalid streaming usage")
                        usage = chunk["usage"]
                    choices = chunk.get("choices", [])
                    if not isinstance(choices, list) or len(choices) > 1:
                        raise RuntimeError("Expected at most one streaming choice")
                    if choices:
                        choice = choices[0]
                        if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                            raise RuntimeError("Invalid streaming choice")
                        delta = choice.get("delta", {})
                        if not isinstance(delta, dict):
                            raise RuntimeError("Invalid streaming delta")
                        for field, destination in (("content", content), ("reasoning_content", reasoning)):
                            value = delta.get(field)
                            if field == "reasoning_content" and value is None:
                                value = delta.get("reasoning")
                            if value is not None:
                                if not isinstance(value, str):
                                    raise RuntimeError(f"Invalid streaming {field}")
                                destination.append(value)
                        reason = choice.get("finish_reason")
                        if reason is not None:
                            if reason not in {"stop", "length"} or finish is not None and reason != finish:
                                raise RuntimeError("Invalid streaming finish_reason")
                            finish = reason
                    snapshot("receiving")
            if not aborted and (not done or finish is None):
                raise RuntimeError("Stream ended without [DONE] and finish_reason; partial retained, no automatic retry")
            completed = not aborted
            snapshot("provider_json_abort" if aborted else "completed", force=True)
            return CompletionResult(
                content=clean("".join(content)), reasoning=clean("".join(reasoning)) or None,
                finish_reason=None if aborted else finish, usage=None if aborted else usage,
                model=model, response_id=response_id, elapsed_seconds=perf_counter() - started,
                metadata={
                    "stage": stage, "attempt_dir": str(attempt), "stream_chunks": chunks,
                    "done_received": done, "provider_json_abort": aborted,
                    "replayed": False, "usage_known": not aborted and usage is not None,
                    **({"partial_usage": usage} if aborted else {}),
                },
            )
        except BaseException as exc:
            detail = clean(str(exc))
            snapshot("interrupted", force=True, error=f"{type(exc).__name__}: {detail}")
            if isinstance(exc, urllib.error.HTTPError):
                raise RuntimeError(f"Zebra provider HTTP {exc.code}; no automatic retry") from None
            if isinstance(exc, (urllib.error.URLError, TimeoutError)):
                raise RuntimeError(f"Zebra provider connection failed: {detail}; no automatic retry") from None
            raise
        finally:
            # A usage event in an interrupted stream need not be the final bill.
            self._record_completion_metrics(perf_counter() - started, usage if completed else None)

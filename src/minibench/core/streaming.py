"""Durable streaming and successful-call replay for isolated cloud workers."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any
import urllib.request

from minibench.core.agent import CompletionResult
from minibench.core.checkpoint import atomic_write_json, fingerprint_payload
from minibench.factory.providers import OpenAICompatibleClient

def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def safe_error(exc: BaseException, secret: str | None = None) -> str:
    text = str(exc)
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return text


def sse_data(response):
    """Decode SSE framing, including comments, CRLF and multiline data."""
    data = []
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


class JournalClient(OpenAICompatibleClient):
    """Replay successful calls in an unfinished observation without rebilling."""

    def __init__(self, journal: Path, callback=None, *, gate=None, account="default", omit_max_tokens=True, **kwargs):
        super().__init__(**kwargs)
        self.journal = journal
        self.callback = callback
        self.gate = gate
        self.account = account
        self.omit_max_tokens = omit_max_tokens
        self.index = 0
        self.journal.mkdir(parents=True, exist_ok=True)

    def build_messages_payload(self, *args, **kwargs):
        payload = super().build_messages_payload(*args, **kwargs)
        if self.omit_max_tokens:
            payload.pop("max_tokens", None)
            payload.pop("max_completion_tokens", None)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        return payload

    def _stream_completion_once(self, request, attempt: Path, index: int):
        """Persist every chunk; never score or replay an incomplete stream."""
        started = perf_counter()
        content = []
        reasoning = []
        usage = None
        finish_reason = None
        response_id = None
        model = None
        chunks = 0
        content_chars = 0
        reasoning_chars = 0
        done = False
        last_snapshot = 0.0
        last_report = 0.0

        def snapshot(status, *, force=False):
            nonlocal last_snapshot, last_report
            elapsed = perf_counter() - started
            if force or elapsed - last_snapshot >= 5 or chunks == 1:
                atomic_write_json(attempt / "partial.json", {
                    "at": now(), "status": status, "call": index,
                    "chunks": chunks, "content": "".join(content),
                    "reasoning": "".join(reasoning), "usage": usage,
                    "finish_reason": finish_reason, "done_received": done,
                    "response_id": response_id, "model": model,
                    "elapsed_seconds": elapsed,
                })
                last_snapshot = elapsed
            if self.callback and (force or chunks == 1 or elapsed - last_report >= 10):
                self.callback({
                    "kind": "stream_progress", "call": index, "at": now(),
                    "chunks": chunks, "content_chars": content_chars,
                    "reasoning_chars": reasoning_chars, "usage": usage,
                    "elapsed_seconds": elapsed, "stream_status": status,
                    "attempt_dir": str(attempt),
                })
                last_report = elapsed

        request.add_header("Accept", "text/event-stream")
        snapshot("connecting", force=True)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                media_type = response.headers.get("Content-Type", "").lower()
                if "text/event-stream" not in media_type:
                    raise RuntimeError(
                        "Streaming request did not return text/event-stream; "
                        "refusing a silent transport fallback or second request."
                    )
                snapshot("receiving", force=True)
                for data in sse_data(response):
                    if data.strip() == "[DONE]":
                        append_json(attempt / "chunks.jsonl", {"at": now(), "done": True})
                        done = True
                        break
                    chunk = json.loads(data)
                    if not isinstance(chunk, dict):
                        raise RuntimeError("Expected an object in the completion stream")
                    # Never persist a credential echoed in a server error.
                    secret = os.environ.get(self.api_key_env)
                    if secret:
                        chunk = json.loads(json.dumps(chunk, ensure_ascii=False).replace(secret, "[REDACTED]"))
                    append_json(attempt / "chunks.jsonl", {"at": now(), "chunk": chunk})
                    if chunk.get("error") is not None:
                        raise RuntimeError(f"Server stream error: {chunk['error']}")
                    chunks += 1
                    for field, previous in (("id", response_id), ("model", model)):
                        value = chunk.get(field)
                        if value is not None and not isinstance(value, str):
                            raise RuntimeError(f"Unexpected streaming {field}")
                        if value and previous and value != previous:
                            raise RuntimeError(f"Streaming {field} changed within a request")
                    response_id = chunk.get("id") or response_id
                    model = chunk.get("model") or model
                    if chunk.get("usage") is not None:
                        if not isinstance(chunk["usage"], dict):
                            raise RuntimeError("Unexpected streaming token usage")
                        # These are cumulative request totals, not per-chunk charges.
                        usage = chunk["usage"]
                    choices = chunk.get("choices", [])
                    if not isinstance(choices, list) or len(choices) > 1:
                        raise RuntimeError("Unexpected streaming choices")
                    if choices:
                        choice = choices[0]
                        if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                            raise RuntimeError("Unexpected streaming choice index")
                        delta = choice.get("delta", {})
                        if not isinstance(delta, dict):
                            raise RuntimeError("Unexpected streaming delta")
                        text = delta.get("content")
                        if text is not None:
                            if not isinstance(text, str):
                                raise RuntimeError("Unexpected streaming content")
                            content.append(text)
                            content_chars += len(text)
                        for field in ("reasoning_content", "reasoning"):
                            text = delta.get(field)
                            if text is not None:
                                if not isinstance(text, str):
                                    raise RuntimeError("Unexpected streaming reasoning")
                                reasoning.append(text)
                                reasoning_chars += len(text)
                                break
                        reason = choice.get("finish_reason")
                        if reason is not None:
                            if not isinstance(reason, str):
                                raise RuntimeError("Unexpected streaming finish_reason")
                            finish_reason = reason
                    snapshot("receiving")
            if not done or finish_reason is None:
                raise RuntimeError(
                    "Stream ended without [DONE] and a finish_reason; "
                    "partial output retained, not scored, and not automatically retried."
                )
            if finish_reason != "stop":
                raise RuntimeError(
                    f"Stream did not finish normally (finish_reason={finish_reason}); "
                    "not scored and not automatically retried."
                )
            answer = "".join(content)
            if not answer.strip():
                raise RuntimeError("Completed stream has no final answer; not scored")
            snapshot("completed", force=True)
            return CompletionResult(
                content=answer, reasoning="".join(reasoning) or None,
                finish_reason=finish_reason, usage=usage,
                model=model, response_id=response_id,
                elapsed_seconds=perf_counter() - started,
                metadata={"transport": "sse", "stream_chunks": chunks,
                          "attempt_dir": str(attempt), "usage_known": usage is not None},
            )
        except BaseException:
            snapshot("interrupted", force=True)
            raise
        finally:
            self._record_completion_metrics(perf_counter() - started, usage)

    def _complete_request_once(self, request):
        index = self.index
        self.index += 1
        payload = json.loads(request.data)
        identity = fingerprint_payload({"endpoint": self.endpoint, "account": self.account, "payload": payload})
        saved = self.journal / f"{index:04d}.json"
        if saved.exists():
            entry = json.loads(saved.read_text(encoding="utf-8"))
            if entry["request_sha256"] != identity:
                raise RuntimeError(
                    "Resume request differs from its saved call; refusing "
                    "to silently change or rebill the unfinished observation."
                )
            result = CompletionResult(**entry["response"])
            self._record_completion_metrics(result.elapsed_seconds or 0, result.usage)
            if self.callback:
                self.callback({"kind": "replayed_call", "call": index,
                               "usage": result.usage, "at": now()})
            return result
        # A manual resume creates a fresh attempt; incomplete streams are evidence,
        # never prompts or successful replay entries. No automatic retries occur.
        number = 0
        while True:
            attempt = self.journal / "attempts" / f"{index:04d}-{number:04d}"
            try:
                attempt.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                number += 1
        request_path = self.journal / f"{index:04d}.request.json"
        if request_path.exists():
            previous = json.loads(request_path.read_text())
            atomic_write_json(attempt / "previous_request.json", previous)
        atomic_write_json(attempt / "request.json", payload)
        atomic_write_json(request_path, payload)
        if self.callback:
            self.callback({"kind": "request_queued", "call": index, "at": now()})
        try:
            if self.gate is not None:
                self.gate.acquire(self.account)
            result = None
            try:
                at = now()
                atomic_write_json(attempt / "started.json", {"at": at, "call": index})
                if self.callback:
                    self.callback({"kind": "request_started", "call": index, "at": at})
                result = self._stream_completion_once(request, attempt, index)
                # Commit BEFORE returning quota reservation or invoking Agent.
                atomic_write_json(saved, {"at": now(), "request_sha256": identity,
                                          "response": asdict(result)})
            finally:
                if self.gate is not None:
                    self.gate.release(self.account, result.usage if result else None)
        except BaseException as exc:
            append_json(self.journal / "failed_attempts.jsonl", {
                "at": now(), "call": index, "request_sha256": identity,
                "error_type": type(exc).__name__,
                "error": safe_error(exc, os.environ.get(self.api_key_env)),
                "attempt_dir": str(attempt),
                "note": "A lost server response may already have consumed quota.",
            })
            raise
        if self.callback:
            self.callback({"kind": "new_call", "call": index,
                           "usage": result.usage, "at": now(),
                           "reasoning_present": bool(result.reasoning)})
        return result


@contextmanager
def replay_h2_engine(directory: Path):
    """Reproduce already observed opponent moves when resuming a partial game."""
    from minibench.datasets.xiangqi import history
    original = history._pikafish_analysis
    index = 0
    directory.mkdir(parents=True, exist_ok=True)

    def analyse(engine, board, side, *, perspective_side, depth):
        nonlocal index
        identity = fingerprint_payload({
            "board": board.board, "side": side,
            "perspective_side": perspective_side, "depth": depth,
        })
        path = directory / f"{index:04d}.json"
        index += 1
        if path.exists():
            entry = json.loads(path.read_text(encoding="utf-8"))
            if identity != entry["request_sha256"]:
                raise RuntimeError("H2 engine replay position differs from its checkpoint")
            return tuple(entry["response"])
        result = original(engine, board, side,
                          perspective_side=perspective_side, depth=depth)
        atomic_write_json(path, {"request_sha256": identity, "response": result})
        return result

    history._pikafish_analysis = analyse
    try:
        yield
    finally:
        history._pikafish_analysis = original

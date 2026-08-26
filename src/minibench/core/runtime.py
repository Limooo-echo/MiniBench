from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from time import perf_counter
from typing import Any, Callable, Literal, Sequence
from uuid import uuid4

from minibench.core.agent import ChatClient, ChatMessage, CompletionResult
from minibench.core.multimodal import ImageAttachment


TraceMode = Literal["off", "summary", "full"]

__all__ = [
    "AgentRun",
    "AgentRuntime",
    "ExecutionBudget",
    "ExecutionBudgetExceeded",
    "StageResult",
    "StrictJSONObjectError",
    "TraceMode",
    "parse_single_json_object",
]

_FORMAT_REPAIR_SYSTEM_PROMPT = (
    "You repair benchmark response formatting. Return exactly one valid JSON "
    "object and no markdown or commentary. Preserve the intended answer."
)


class ExecutionBudgetExceeded(RuntimeError):
    """Raised before a call that would exceed the configured logical-call budget."""


class StrictJSONObjectError(ValueError):
    """Raised when a response is not exactly one JSON object."""


@dataclass
class ExecutionBudget:
    """Mutable logical-call budget, reset when an outermost span begins."""

    max_calls: int | None = None
    max_total_tokens: int | None = None
    calls_used: int = field(default=0, init=False)
    total_tokens_used: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_calls is not None and (
            isinstance(self.max_calls, bool)
            or not isinstance(self.max_calls, int)
            or self.max_calls < 0
        ):
            raise ValueError("max_calls must be a non-negative integer or None")
        if self.max_total_tokens is not None and (
            isinstance(self.max_total_tokens, bool)
            or not isinstance(self.max_total_tokens, int)
            or self.max_total_tokens < 1
        ):
            raise ValueError("max_total_tokens must be a positive integer or None")

    @property
    def remaining_calls(self) -> int | None:
        if self.max_calls is None:
            return None
        return self.max_calls - self.calls_used

    @property
    def remaining_tokens(self) -> int | None:
        if self.max_total_tokens is None:
            return None
        return max(0, self.max_total_tokens - self.total_tokens_used)

    def reset(self) -> None:
        self.calls_used = 0
        self.total_tokens_used = 0

    def reserve_call(self) -> int:
        if self.max_calls is not None and self.calls_used >= self.max_calls:
            raise ExecutionBudgetExceeded(
                f"execution call budget exhausted ({self.calls_used}/{self.max_calls})"
            )
        if (
            self.max_total_tokens is not None
            and self.total_tokens_used >= self.max_total_tokens
        ):
            raise ExecutionBudgetExceeded(
                "execution token budget exhausted "
                f"({self.total_tokens_used}/{self.max_total_tokens})"
            )
        self.calls_used += 1
        return self.calls_used

    def record_tokens(self, total_tokens: int) -> None:
        if total_tokens > 0:
            self.total_tokens_used += total_tokens


@dataclass(frozen=True)
class StageResult:
    stage_name: str
    call_index: int
    prompt_sha256: str
    elapsed_seconds: float
    model_elapsed_seconds: float
    llm_calls: int
    usage_missing_calls: int
    token_usage: dict[str, int]
    parent_id: str | None = None
    finish_reason: str | None = None
    model: str | None = None
    response_id: str | None = None
    json_valid: bool | None = None
    is_format_repair: bool = False
    image_count: int = 0
    error: str | None = None
    prompt: str | None = None
    system_prompt: str | None = None
    output: str | None = None
    reasoning: str | None = None
    parsed_json: dict[str, Any] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_name": self.stage_name,
            "call_index": self.call_index,
            "prompt_sha256": self.prompt_sha256,
            "elapsed_seconds": self.elapsed_seconds,
            "model_elapsed_seconds": self.model_elapsed_seconds,
            "llm_calls": self.llm_calls,
            "usage_missing_calls": self.usage_missing_calls,
            "token_usage": dict(self.token_usage),
            "parent_id": self.parent_id,
            "finish_reason": self.finish_reason,
            "model": self.model,
            "response_id": self.response_id,
            "json_valid": self.json_valid,
            "is_format_repair": self.is_format_repair,
            "image_count": self.image_count,
            "error": self.error,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "output": self.output,
            "reasoning": self.reasoning,
            "parsed_json": (
                dict(self.parsed_json) if self.parsed_json is not None else None
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentRun:
    span_id: str
    name: str
    trace_mode: TraceMode
    status: Literal["completed", "failed", "stopped"]
    elapsed_seconds: float
    metrics: dict[str, object]
    stages: tuple[StageResult, ...]
    budget: dict[str, int | None]
    output_sha256: str | None = None
    output: str | None = None
    error: str | None = None
    stop_reason: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        metrics = dict(self.metrics)
        metrics["token_usage"] = dict(metrics.get("token_usage") or {})
        return {
            "span_id": self.span_id,
            "name": self.name,
            "trace_mode": self.trace_mode,
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": metrics,
            "stages": [stage.to_dict() for stage in self.stages],
            "budget": dict(self.budget),
            "output_sha256": self.output_sha256,
            "output": self.output,
            "error": self.error,
            "stop_reason": self.stop_reason,
            "metadata": dict(self.metadata),
        }


@dataclass
class _RuntimeSpan:
    span_id: str
    name: str
    started_at: float
    metrics_start: dict[str, object]
    stage_start: int
    budget_calls_start: int
    metadata: dict[str, object]


@dataclass(frozen=True)
class _ObservedCompletion:
    result: CompletionResult
    stage_name: str
    call_index: int
    prompt_text: str
    prompt_sha256: str
    system_prompt: str | None
    image_count: int
    is_format_repair: bool
    elapsed_seconds: float
    metrics: dict[str, object]
    metadata: dict[str, object]


class AgentRuntime:
    """Execution, tracing, budgeting, and output-contract layer for agent stages.

    The wrapper itself conforms to the existing string-returning ChatClient
    surface. New algorithms may call complete_result() for metadata and strict
    JSON handling, while legacy algorithms can use it as an ordinary client.
    """

    _is_agent_runtime = True

    def __init__(
        self,
        client: ChatClient,
        *,
        budget: ExecutionBudget | None = None,
        trace_mode: TraceMode = "summary",
        prompt_version: str | None = None,
    ):
        if trace_mode not in ("off", "summary", "full"):
            raise ValueError("trace_mode must be 'off', 'summary', or 'full'")
        self.client = client
        self.budget = budget or ExecutionBudget()
        self.trace_mode = trace_mode
        self.prompt_version = prompt_version
        self.last_run: AgentRun | None = None
        self._spans: list[_RuntimeSpan] = []
        self._stages: list[StageResult] = []
        self._metrics = _empty_model_metrics()

    @property
    def active_span_id(self) -> str | None:
        return self._spans[-1].span_id if self._spans else None

    def annotate_run(self, **metadata: object) -> None:
        """Attach algorithm metadata to the active run span."""

        if not self._spans:
            raise RuntimeError("cannot annotate an agent run when no span is active")
        self._spans[-1].metadata.update(metadata)

    def metrics_snapshot(self) -> dict[str, object]:
        snapshot = dict(self._metrics)
        snapshot["token_usage"] = dict(self._metrics["token_usage"])
        return snapshot

    def begin_span(
        self,
        name: str = "agent-run",
        *,
        metadata: dict[str, object] | None = None,
    ) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("span name must be a non-empty string")
        if not self._spans:
            self.budget.reset()
            self._stages.clear()
        span = _RuntimeSpan(
            span_id=uuid4().hex,
            name=name,
            started_at=perf_counter(),
            metrics_start=self.metrics_snapshot(),
            stage_start=len(self._stages),
            budget_calls_start=self.budget.calls_used,
            metadata=dict(metadata or {}),
        )
        self._spans.append(span)
        return span.span_id

    def end_span(
        self,
        span_id: str | None = None,
        *,
        output: str | None = None,
        error: str | None = None,
        stop_reason: str | None = None,
    ) -> AgentRun:
        if not self._spans:
            raise RuntimeError("cannot end an agent runtime span when none is active")
        span = self._spans[-1]
        if span_id is not None and span_id != span.span_id:
            raise RuntimeError(
                f"runtime spans must end in LIFO order; active span is {span.span_id}"
            )
        self._spans.pop()
        metrics = _model_metrics_delta(span.metrics_start, self.metrics_snapshot())
        logical_calls = self.budget.calls_used - span.budget_calls_start
        metrics["logical_calls"] = logical_calls
        metrics["runtime_elapsed_seconds"] = round(perf_counter() - span.started_at, 6)
        stages = (
            tuple(self._stages[span.stage_start :]) if self.trace_mode != "off" else ()
        )
        run = AgentRun(
            span_id=span.span_id,
            name=span.name,
            trace_mode=self.trace_mode,
            status=(
                "failed"
                if error is not None
                else ("completed" if stop_reason in (None, "completed") else "stopped")
            ),
            elapsed_seconds=float(metrics["runtime_elapsed_seconds"]),
            metrics=metrics,
            stages=stages,
            budget={
                "max_calls": self.budget.max_calls,
                "calls_used": logical_calls,
                "remaining_calls": self.budget.remaining_calls,
                "max_total_tokens": self.budget.max_total_tokens,
                "total_tokens_used": self.budget.total_tokens_used,
                "remaining_tokens": self.budget.remaining_tokens,
            },
            output_sha256=_text_sha256(output) if output is not None else None,
            output=output if self.trace_mode == "full" else None,
            error=error,
            metadata=dict(span.metadata),
            stop_reason=stop_reason,
        )
        self.last_run = run
        return run

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
        images: Sequence[ImageAttachment] = (),
        stage_name: str = "completion",
        strict_json: bool = False,
        repair_format: bool = True,
        stage_metadata: dict[str, object] | None = None,
    ) -> str:
        return self.complete_result(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            images=images,
            stage_name=stage_name,
            strict_json=strict_json,
            repair_format=repair_format,
            stage_metadata=stage_metadata,
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
        stage_name: str = "completion",
        strict_json: bool = False,
        repair_format: bool = True,
        stage_metadata: dict[str, object] | None = None,
    ) -> CompletionResult:
        observed = self._invoke_text(
            prompt,
            stage_name=stage_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            images=images,
            metadata=stage_metadata,
        )
        return self._validate_or_repair(
            observed,
            strict_json=strict_json,
            repair_format=repair_format,
            repair_context=prompt,
            max_tokens=max_tokens,
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
        stage_name: str = "completion",
        strict_json: bool = False,
        repair_format: bool = True,
        stage_metadata: dict[str, object] | None = None,
    ) -> str:
        return self.complete_messages_result(
            messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            images=images,
            stage_name=stage_name,
            strict_json=strict_json,
            repair_format=repair_format,
            stage_metadata=stage_metadata,
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
        stage_name: str = "completion",
        strict_json: bool = False,
        repair_format: bool = True,
        stage_metadata: dict[str, object] | None = None,
    ) -> CompletionResult:
        serialized = _serialize_messages(messages)
        observed = self._invoke_messages(
            messages,
            prompt_text=serialized,
            stage_name=stage_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            images=images,
            metadata=stage_metadata,
        )
        return self._validate_or_repair(
            observed,
            strict_json=strict_json,
            repair_format=repair_format,
            repair_context=serialized,
            max_tokens=max_tokens,
        )

    @staticmethod
    def parse_single_json_object(content: str) -> dict[str, Any]:
        return parse_single_json_object(content)

    def _invoke_text(
        self,
        prompt: str,
        *,
        stage_name: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool | None,
        images: Sequence[ImageAttachment],
        is_format_repair: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> _ObservedCompletion:
        def invoke() -> CompletionResult:
            rich_complete = getattr(self.client, "complete_result", None)
            if callable(rich_complete):
                value = rich_complete(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    images=images,
                )
            else:
                value = self.client.complete(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    images=images,
                )
            return _coerce_completion_result(value)

        return self._invoke(
            invoke,
            prompt_text=prompt,
            stage_name=stage_name,
            system_prompt=system_prompt,
            image_count=len(images),
            is_format_repair=is_format_repair,
            metadata=metadata,
        )

    def _invoke_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        prompt_text: str,
        stage_name: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool | None,
        images: Sequence[ImageAttachment],
        metadata: dict[str, object] | None = None,
    ) -> _ObservedCompletion:
        def invoke() -> CompletionResult:
            rich_complete = getattr(self.client, "complete_messages_result", None)
            if callable(rich_complete):
                value = rich_complete(
                    messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    images=images,
                )
            else:
                value = self.client.complete_messages(
                    messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    images=images,
                )
            return _coerce_completion_result(value)

        return self._invoke(
            invoke,
            prompt_text=prompt_text,
            stage_name=stage_name,
            system_prompt=system_prompt,
            image_count=len(images),
            is_format_repair=False,
            metadata=metadata,
        )

    def _invoke(
        self,
        invoke: Callable[[], CompletionResult],
        *,
        prompt_text: str,
        stage_name: str,
        system_prompt: str | None,
        image_count: int,
        is_format_repair: bool,
        metadata: dict[str, object] | None,
    ) -> _ObservedCompletion:
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValueError("stage_name must be a non-empty string")
        call_index = self.budget.reserve_call()
        before = _metrics_snapshot(self.client)
        started_at = perf_counter()
        try:
            result = invoke()
            if not isinstance(result.content, str):
                raise TypeError("completion result content must be a string")
        except BaseException as exc:
            elapsed = perf_counter() - started_at
            after = _metrics_snapshot(self.client)
            metrics = _observed_call_metrics(
                before,
                after,
                result=None,
                elapsed_seconds=elapsed,
            )
            self._record_metrics(metrics)
            self._record_failed_stage(
                stage_name=stage_name,
                call_index=call_index,
                prompt_text=prompt_text,
                system_prompt=system_prompt,
                image_count=image_count,
                is_format_repair=is_format_repair,
                elapsed_seconds=elapsed,
                metrics=metrics,
                error=f"{type(exc).__name__}: {exc}",
                metadata=metadata,
            )
            raise
        elapsed = perf_counter() - started_at
        after = _metrics_snapshot(self.client)
        metrics = _observed_call_metrics(
            before,
            after,
            result=result,
            elapsed_seconds=elapsed,
        )
        self._record_metrics(metrics)
        return _ObservedCompletion(
            result=result,
            stage_name=stage_name,
            call_index=call_index,
            prompt_text=prompt_text,
            prompt_sha256=_text_sha256(prompt_text),
            system_prompt=system_prompt,
            image_count=image_count,
            is_format_repair=is_format_repair,
            elapsed_seconds=elapsed,
            metrics=metrics,
            metadata=dict(metadata or {}),
        )

    def _validate_or_repair(
        self,
        observed: _ObservedCompletion,
        *,
        strict_json: bool,
        repair_format: bool,
        repair_context: str,
        max_tokens: int | None,
    ) -> CompletionResult:
        if not strict_json:
            self._record_successful_stage(observed, json_valid=None, parsed=None)
            return observed.result

        try:
            parsed = parse_single_json_object(observed.result.content)
        except StrictJSONObjectError as exc:
            self._record_successful_stage(
                observed,
                json_valid=False,
                parsed=None,
                error=str(exc),
            )
            if not repair_format:
                raise
        else:
            self._record_successful_stage(observed, json_valid=True, parsed=parsed)
            return replace(observed.result, parsed_json=parsed)

        repair_prompt = _format_repair_prompt(
            repair_context,
            observed.result.content,
        )
        repaired = self._invoke_text(
            repair_prompt,
            stage_name=f"{observed.stage_name}.format_repair",
            system_prompt=_FORMAT_REPAIR_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=True,
            images=(),
            is_format_repair=True,
            metadata={"repairs_stage": observed.stage_name},
        )
        try:
            parsed = parse_single_json_object(repaired.result.content)
        except StrictJSONObjectError as exc:
            self._record_successful_stage(
                repaired,
                json_valid=False,
                parsed=None,
                error=str(exc),
            )
            raise StrictJSONObjectError(
                "format repair did not produce exactly one JSON object"
            ) from exc
        self._record_successful_stage(repaired, json_valid=True, parsed=parsed)
        return replace(
            repaired.result,
            parsed_json=parsed,
            format_repaired=True,
        )

    def _record_successful_stage(
        self,
        observed: _ObservedCompletion,
        *,
        json_valid: bool | None,
        parsed: dict[str, Any] | None,
        error: str | None = None,
    ) -> None:
        if self.trace_mode == "off":
            return
        metadata = dict(observed.result.metadata)
        metadata.update(observed.metadata)
        if self.prompt_version is not None:
            metadata.setdefault("prompt_version", self.prompt_version)
        full = self.trace_mode == "full"
        self._stages.append(
            StageResult(
                stage_name=observed.stage_name,
                call_index=observed.call_index,
                prompt_sha256=observed.prompt_sha256,
                elapsed_seconds=round(observed.elapsed_seconds, 6),
                model_elapsed_seconds=float(observed.metrics["model_elapsed_seconds"]),
                llm_calls=int(observed.metrics["llm_calls"]),
                usage_missing_calls=int(observed.metrics["usage_missing_calls"]),
                token_usage=dict(observed.metrics["token_usage"]),
                parent_id=(
                    str(metadata["parent_id"])
                    if isinstance(metadata.get("parent_id"), str)
                    else None
                ),
                finish_reason=observed.result.finish_reason,
                model=observed.result.model,
                response_id=observed.result.response_id,
                json_valid=json_valid,
                is_format_repair=observed.is_format_repair,
                image_count=observed.image_count,
                error=error,
                prompt=observed.prompt_text if full else None,
                system_prompt=observed.system_prompt if full else None,
                output=observed.result.content if full else None,
                reasoning=observed.result.reasoning if full else None,
                parsed_json=dict(parsed) if full and parsed is not None else None,
                metadata=metadata,
            )
        )

    def _record_failed_stage(
        self,
        *,
        stage_name: str,
        call_index: int,
        prompt_text: str,
        system_prompt: str | None,
        image_count: int,
        is_format_repair: bool,
        elapsed_seconds: float,
        metrics: dict[str, object],
        error: str,
        metadata: dict[str, object] | None,
    ) -> None:
        if self.trace_mode == "off":
            return
        full = self.trace_mode == "full"
        stage_metadata = dict(metadata or {})
        if self.prompt_version is not None:
            stage_metadata.setdefault("prompt_version", self.prompt_version)
        self._stages.append(
            StageResult(
                stage_name=stage_name,
                call_index=call_index,
                prompt_sha256=_text_sha256(prompt_text),
                elapsed_seconds=round(elapsed_seconds, 6),
                model_elapsed_seconds=float(metrics["model_elapsed_seconds"]),
                llm_calls=int(metrics["llm_calls"]),
                usage_missing_calls=int(metrics["usage_missing_calls"]),
                token_usage=dict(metrics["token_usage"]),
                parent_id=(
                    str(stage_metadata["parent_id"])
                    if isinstance(stage_metadata.get("parent_id"), str)
                    else None
                ),
                json_valid=None,
                is_format_repair=is_format_repair,
                image_count=image_count,
                error=error,
                prompt=prompt_text if full else None,
                system_prompt=system_prompt if full else None,
                metadata=stage_metadata,
            )
        )

    def _record_metrics(self, metrics: dict[str, object]) -> None:
        self._metrics["model_elapsed_seconds"] = float(
            self._metrics["model_elapsed_seconds"]
        ) + float(metrics["model_elapsed_seconds"])
        self._metrics["llm_calls"] = int(self._metrics["llm_calls"]) + int(
            metrics["llm_calls"]
        )
        self._metrics["usage_missing_calls"] = int(
            self._metrics["usage_missing_calls"]
        ) + int(metrics["usage_missing_calls"])
        cumulative_usage = self._metrics["token_usage"]
        assert isinstance(cumulative_usage, dict)
        for key, value in dict(metrics["token_usage"]).items():
            cumulative_usage[key] = int(cumulative_usage.get(key, 0)) + int(value)

        stage_usage = dict(metrics["token_usage"])
        self.budget.record_tokens(int(stage_usage.get("total_tokens", 0)))


def parse_single_json_object(content: str) -> dict[str, Any]:
    """Parse content only when its complete non-whitespace value is one object."""

    if not isinstance(content, str):
        raise TypeError("JSON response content must be a string")
    stripped = content.strip()
    if not stripped:
        raise StrictJSONObjectError("response is empty")
    try:
        value, end = json.JSONDecoder(
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        ).raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise StrictJSONObjectError("response is not valid JSON") from exc
    if end != len(stripped):
        raise StrictJSONObjectError(
            "response contains text or another value outside the JSON object"
        )
    if not isinstance(value, dict):
        raise StrictJSONObjectError("response JSON value must be an object")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise StrictJSONObjectError(
                f"response JSON object contains duplicate key {key!r}"
            )
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise StrictJSONObjectError(
        f"response contains non-standard JSON constant {value!r}"
    )


def _format_repair_prompt(original_context: str, malformed_response: str) -> str:
    payload = json.dumps(
        {
            "original_request": original_context,
            "response_to_repair": malformed_response,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "Rewrite response_to_repair so it satisfies the output contract in "
        "original_request. Preserve the intended answer. Return exactly one "
        "JSON object and nothing else.\n\n"
        f"Repair input:\n{payload}"
    )


def _serialize_messages(messages: Sequence[ChatMessage]) -> str:
    return json.dumps(
        list(messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _coerce_completion_result(value: object) -> CompletionResult:
    if isinstance(value, CompletionResult):
        return value
    if isinstance(value, str):
        return CompletionResult(content=value)
    raise TypeError(
        "complete_result() must return CompletionResult; legacy complete() "
        "must return str"
    )


def _metrics_snapshot(target: object) -> dict[str, object] | None:
    snapshot_method = getattr(target, "metrics_snapshot", None)
    if callable(snapshot_method):
        raw = snapshot_method()
        if not isinstance(raw, dict):
            return None
        usage = {
            str(key): int(value)
            for key, value in dict(raw.get("token_usage") or {}).items()
            if not isinstance(value, bool) and isinstance(value, int)
        }
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage.setdefault(key, 0)
        return {
            "model_elapsed_seconds": float(raw.get("model_elapsed_seconds", 0.0)),
            "llm_calls": int(raw.get("llm_calls", 0)),
            "usage_missing_calls": int(raw.get("usage_missing_calls", 0)),
            "token_usage": usage,
        }
    nested = getattr(target, "client", None)
    if nested is not None and nested is not target:
        return _metrics_snapshot(nested)
    return None


def _observed_call_metrics(
    before: dict[str, object] | None,
    after: dict[str, object] | None,
    *,
    result: CompletionResult | None,
    elapsed_seconds: float,
) -> dict[str, object]:
    has_client_delta = before is not None and after is not None
    metrics = (
        _model_metrics_delta(before, after)
        if has_client_delta
        else _empty_model_metrics()
    )
    client_calls = int(metrics["llm_calls"])
    missing_client_delta = not has_client_delta or client_calls < 1
    if client_calls < 1:
        metrics["llm_calls"] = 1

    completion_usage = _completion_token_usage(result)
    current_usage = dict(metrics["token_usage"])
    if completion_usage is not None and (
        missing_client_delta or not any(current_usage.values())
    ):
        metrics["token_usage"] = completion_usage
        metrics["usage_missing_calls"] = max(0, int(metrics["llm_calls"]) - 1)
    elif missing_client_delta:
        metrics["usage_missing_calls"] = 1

    model_elapsed = float(metrics["model_elapsed_seconds"])
    if model_elapsed <= 0.0:
        if result is not None and result.elapsed_seconds is not None:
            model_elapsed = float(result.elapsed_seconds)
        else:
            model_elapsed = elapsed_seconds
        metrics["model_elapsed_seconds"] = round(model_elapsed, 6)
    return metrics


def _completion_token_usage(
    result: CompletionResult | None,
) -> dict[str, int] | None:
    if result is None or result.usage is None:
        return None
    from minibench.core.metrics import extract_token_usage

    return extract_token_usage(result.usage)


def _empty_model_metrics() -> dict[str, object]:
    return {
        "model_elapsed_seconds": 0.0,
        "llm_calls": 0,
        "usage_missing_calls": 0,
        "token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _model_metrics_delta(
    before: dict[str, object],
    after: dict[str, object],
) -> dict[str, object]:
    before_usage = dict(before.get("token_usage") or {})
    after_usage = dict(after.get("token_usage") or {})
    usage_keys = (
        set(before_usage)
        | set(after_usage)
        | {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }
    )
    return {
        "model_elapsed_seconds": round(
            float(after.get("model_elapsed_seconds", 0.0))
            - float(before.get("model_elapsed_seconds", 0.0)),
            6,
        ),
        "llm_calls": int(after.get("llm_calls", 0)) - int(before.get("llm_calls", 0)),
        "usage_missing_calls": int(after.get("usage_missing_calls", 0))
        - int(before.get("usage_missing_calls", 0)),
        "token_usage": {
            key: int(after_usage.get(key, 0)) - int(before_usage.get(key, 0))
            for key in sorted(usage_keys)
        },
    }


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()

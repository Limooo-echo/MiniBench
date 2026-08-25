"""Crash-safe checkpoint storage for one-stroke experiment runs.

``predictions.jsonl`` is the authoritative progress record.  It is replaced
atomically after every completed work item, so a stale ``run_state.json`` can
always be reconstructed without repeating a durable model call.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

from minibench.core.checkpoint import (
    RunLock,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    fingerprint_payload,
    read_json_object,
    read_jsonl_objects,
)
from minibench.core.metrics import summary_metrics_line


MANIFEST_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
WorkKey = tuple[str, str]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_name() -> str:
    return "one-stroke-" + datetime.now(timezone.utc).strftime(
        "%Y%m%d-%H%M%S-%f"
    )


def validate_run_name(run_name: str | None, *, on_existing: str) -> str:
    if on_existing not in {"error", "resume"}:
        raise ValueError("run.on_existing must be error or resume")
    if run_name is not None and (
        not isinstance(run_name, str) or not run_name.strip()
    ):
        raise ValueError("run.run_name must be a non-empty string or null")
    if isinstance(run_name, str) and (
        run_name in {".", ".."}
        or Path(run_name).is_absolute()
        or Path(run_name).name != run_name
        or "/" in run_name
        or "\\" in run_name
    ):
        raise ValueError("run.run_name must be a single directory name")
    if on_existing == "resume" and run_name is None:
        raise ValueError("run.run_name is required when run.on_existing=resume")
    return run_name or default_run_name()


def sidecar_lock_name(run_name: str) -> str:
    digest = hashlib.sha256(run_name.encode("utf-8")).hexdigest()[:16]
    return f".one-stroke-lock-{digest}"


def git_metadata(repository_root: Path) -> dict[str, Any]:
    """Return reproducibility metadata without making Git availability fatal."""

    try:
        head = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {
            "head": None,
            "dirty": None,
            "changed_entries": None,
            "status_sha256": None,
        }
    entries = [line for line in status.splitlines() if line]
    return {
        "head": head or None,
        "dirty": bool(entries),
        "changed_entries": len(entries),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def make_manifest(
    *,
    resolved_config: Mapping[str, Any],
    task_path: Path,
    task_sha256: str,
    model_identity: Mapping[str, Any],
    work_plan: Sequence[WorkKey],
    input_assets: Sequence[Mapping[str, Any]],
    dataset_profile: Mapping[str, Any],
    interpretation_warnings: Sequence[Mapping[str, Any]],
    fingerprint_inputs: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    created_at = utc_timestamp()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "family": "one_stroke",
        "created_at": created_at,
        "resolved_config": dict(resolved_config),
        "task_data": {
            "path": str(task_path.resolve()),
            "sha256": task_sha256,
        },
        "model_identity": dict(model_identity),
        "git": git_metadata(repository_root),
        "work_plan": [
            {"task_id": task_id, "mode": mode}
            for task_id, mode in work_plan
        ],
        "input_assets": [dict(asset) for asset in input_assets],
        "dataset_profile": dict(dataset_profile),
        "interpretation_warnings": [
            dict(warning) for warning in interpretation_warnings
        ],
        "fingerprint_inputs": dict(fingerprint_inputs),
        "fingerprint": fingerprint_payload(fingerprint_inputs),
    }


class OneStrokeCheckpointRun:
    """Own one checkpointed run directory for the duration of evaluation."""

    _ALLOWED_ARTIFACTS = {
        "manifest.json",
        "predictions.jsonl",
        "results.json",
        "summary.txt",
        "run_state.json",
    }

    def __init__(
        self,
        *,
        output_dir: str | Path,
        run_name: str,
        on_existing: str,
        manifest: Mapping[str, Any],
        work_plan: Sequence[WorkKey],
        result_key: Callable[[Any], WorkKey],
        result_from_dict: Callable[[dict[str, Any]], Any],
        summarize: Callable[[list[Any]], dict[str, Any]],
    ) -> None:
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.run_dir = self.output_dir / run_name
        self.on_existing = on_existing
        self.expected_manifest = dict(manifest)
        self.work_plan = tuple(work_plan)
        if len(set(self.work_plan)) != len(self.work_plan):
            raise ValueError("one-stroke work plan contains duplicate keys")
        self.plan_index = {
            key: index for index, key in enumerate(self.work_plan)
        }
        self.result_key = result_key
        self.result_from_dict = result_from_dict
        self.summarize = summarize
        self.results: list[Any] = []
        self.completed_keys: set[WorkKey] = set()
        self.current_key: WorkKey | None = None
        self.created_at = str(manifest["created_at"])
        self.resume_count = 0
        self._lock: RunLock | None = None

    @property
    def planned_total(self) -> int:
        return len(self.work_plan)

    def __enter__(self) -> "OneStrokeCheckpointRun":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RunLock(
            self.output_dir,
            lock_name=sidecar_lock_name(self.run_name),
        )
        self._lock.__enter__()
        try:
            self._open_or_resume()
        except BaseException:
            self._lock.__exit__(None, None, None)
            self._lock = None
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        lock = self._lock
        self._lock = None
        if lock is not None:
            lock.__exit__(exc_type, exc, traceback)

    def _open_or_resume(self) -> None:
        if self.run_dir.exists():
            if self.run_dir.is_symlink() or not self.run_dir.is_dir():
                raise ValueError(
                    f"one-stroke run path must be a real directory: {self.run_dir}"
                )
            if self.on_existing == "error":
                raise FileExistsError(
                    f"run directory already exists: {self.run_dir}; choose a new "
                    "run_name or set run.on_existing=resume"
                )
            self._resume()
            return

        self.run_dir.mkdir()
        atomic_write_json(self.run_dir / "manifest.json", self.expected_manifest)
        self.results = []
        self.completed_keys = set()
        self._write_snapshot()
        self._write_state(status="running")

    def _resume(self) -> None:
        entries = list(self.run_dir.iterdir())
        unexpected = {path.name for path in entries} - self._ALLOWED_ARTIFACTS
        if unexpected:
            rendered = ", ".join(sorted(unexpected))
            raise ValueError(
                "cannot resume one-stroke run with unexpected file(s): " + rendered
            )
        invalid = {
            path.name for path in entries if path.is_symlink() or not path.is_file()
        }
        if invalid:
            rendered = ", ".join(sorted(invalid))
            raise ValueError(
                "cannot resume one-stroke run with non-file artifact(s): " + rendered
            )
        actual_manifest = self._validate_manifest()
        state = self._read_state()
        if state["status"] == "completed":
            raise ValueError(
                f"one-stroke run is already completed: {self.run_dir}; "
                "choose a new run_name"
            )
        if state["created_at"] != actual_manifest["created_at"]:
            raise ValueError(
                f"one-stroke state created_at does not match manifest: {self.run_dir}"
            )
        self.created_at = str(actual_manifest["created_at"])
        self.resume_count = int(state["resume_count"]) + 1
        self.results = self._load_predictions()
        self.completed_keys = {self.result_key(result) for result in self.results}
        self.current_key = None
        self._write_snapshot()
        self._write_state(status="running")

    def _validate_manifest(self) -> dict[str, Any]:
        path = self.run_dir / "manifest.json"
        if not path.is_file():
            raise ValueError(
                f"cannot resume one-stroke run without manifest.json: {self.run_dir}"
            )
        actual = read_json_object(path)
        if actual.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported one-stroke manifest schema in {path}")
        if actual.get("family") != "one_stroke":
            raise ValueError(f"run manifest is not for one_stroke: {path}")
        if not isinstance(actual.get("created_at"), str) or not actual["created_at"]:
            raise ValueError(f"invalid one-stroke manifest created_at in {path}")
        inputs = actual.get("fingerprint_inputs")
        if (
            not isinstance(inputs, dict)
            or fingerprint_payload(inputs) != actual.get("fingerprint")
        ):
            raise ValueError(f"corrupt one-stroke fingerprint in {path}")
        if actual.get("fingerprint") != self.expected_manifest.get("fingerprint"):
            raise ValueError(
                f"one-stroke resume fingerprint mismatch for {self.run_dir}; "
                "use a new run_name"
            )
        for field in (
            "task_data",
            "model_identity",
            "work_plan",
            "input_assets",
            "dataset_profile",
            "interpretation_warnings",
        ):
            if actual.get(field) != self.expected_manifest.get(field):
                raise ValueError(
                    f"corrupt one-stroke manifest {field} in {path}"
                )
        return actual

    def _read_state(self) -> dict[str, Any]:
        path = self.run_dir / "run_state.json"
        if not path.is_file():
            raise ValueError(
                f"cannot resume one-stroke run without run_state.json: {self.run_dir}"
            )
        state = read_json_object(path)
        if state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported one-stroke state schema in {path}")
        if state.get("status") not in {"running", "interrupted", "completed"}:
            raise ValueError(f"invalid one-stroke run status in {path}")
        if not isinstance(state.get("created_at"), str) or not state["created_at"]:
            raise ValueError(f"invalid one-stroke created_at in {path}")
        resume_count = state.get("resume_count")
        if (
            isinstance(resume_count, bool)
            or not isinstance(resume_count, int)
            or resume_count < 0
        ):
            raise ValueError(f"invalid one-stroke resume_count in {path}")
        return state

    def _load_predictions(self) -> list[Any]:
        path = self.run_dir / "predictions.jsonl"
        if not path.is_file():
            raise ValueError(
                f"cannot resume one-stroke run without predictions.jsonl: "
                f"{self.run_dir}"
            )
        results: list[Any] = []
        seen: set[WorkKey] = set()
        for raw_result in read_jsonl_objects(path):
            try:
                result = self.result_from_dict(raw_result)
                key = self.result_key(result)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid one-stroke prediction in {path}: {exc}"
                ) from exc
            if key not in self.plan_index:
                raise ValueError(
                    "predictions.jsonl contains unknown one-stroke work item: "
                    f"{key[0]}/{key[1]}"
                )
            if key in seen:
                raise ValueError(
                    "predictions.jsonl contains duplicate one-stroke work item: "
                    f"{key[0]}/{key[1]}"
                )
            seen.add(key)
            results.append(result)
        results.sort(key=lambda result: self.plan_index[self.result_key(result)])
        return results

    def start_work_item(self, key: WorkKey) -> None:
        if key not in self.plan_index:
            raise ValueError(
                f"one-stroke evaluator started unknown work item: {key[0]}/{key[1]}"
            )
        if key in self.completed_keys:
            raise ValueError(
                f"one-stroke evaluator restarted completed work item: {key[0]}/{key[1]}"
            )
        self.current_key = key
        self._write_state(status="running")

    def record_result(self, result: Any) -> None:
        key = self.result_key(result)
        if key not in self.plan_index:
            raise ValueError(
                f"one-stroke evaluator returned unknown work item: {key[0]}/{key[1]}"
            )
        if key in self.completed_keys:
            raise ValueError(
                f"one-stroke evaluator returned duplicate work item: {key[0]}/{key[1]}"
            )
        candidate = [*self.results, result]
        candidate.sort(key=lambda item: self.plan_index[self.result_key(item)])
        self._write_snapshot(candidate)
        self.results[:] = candidate
        self.completed_keys.add(key)
        self.current_key = None
        self._write_state(status="running")

    def mark_interrupted(self, exc: BaseException) -> None:
        # Reload the authoritative file in case interruption occurred between
        # the atomic predictions replacement and in-memory bookkeeping.
        try:
            self.results = self._load_predictions()
            self.completed_keys = {
                self.result_key(result) for result in self.results
            }
            if self.current_key in self.completed_keys:
                self.current_key = None
        except (OSError, ValueError):
            pass
        self._write_snapshot(error=f"{type(exc).__name__}: {exc}")
        self._write_state(
            status="interrupted",
            error={"type": type(exc).__name__, "message": str(exc)},
        )

    def mark_completed(self) -> dict[str, Any]:
        missing = set(self.work_plan) - self.completed_keys
        if missing:
            rendered = ", ".join(
                f"{task_id}/{mode}" for task_id, mode in sorted(missing)
            )
            raise RuntimeError(
                "cannot complete one-stroke run with missing work item(s): "
                + rendered
            )
        self.current_key = None
        summary = self._write_snapshot()
        self._write_state(status="completed")
        return summary

    def _write_snapshot(
        self,
        results: Sequence[Any] | None = None,
        *,
        error: str | None = None,
    ) -> dict[str, Any]:
        selected = list(self.results if results is None else results)
        atomic_write_jsonl(
            self.run_dir / "predictions.jsonl",
            (asdict(result) for result in selected),
        )
        summary = self.summarize(selected)
        atomic_write_json(self.run_dir / "results.json", summary)
        metrics = summary.get("metrics", {})
        text = (
            f"total={summary['total']} success={summary['success']} "
            f"success_rate={summary['success_rate']:.3f}\n"
        )
        if summary.get("rule_ignore_rate") is not None:
            text += (
                f"rule_ignore_rate={summary['rule_ignore_rate']:.3f} "
                f"({summary['rule_ignore_count']}/"
                f"{summary['rule_ignore_denominator']})\n"
            )
        text += summary_metrics_line(metrics)
        if error is not None:
            text += f"status=interrupted error={error}\n"
        atomic_write_text(self.run_dir / "summary.txt", text)
        return summary

    def _write_state(
        self,
        *,
        status: str,
        error: Mapping[str, str] | None = None,
    ) -> None:
        atomic_write_json(
            self.run_dir / "run_state.json",
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "status": status,
                "planned_total": self.planned_total,
                "completed_total": len(self.results),
                "remaining_total": self.planned_total - len(self.results),
                "created_at": self.created_at,
                "resume_count": self.resume_count,
                "current_work_key": (
                    {
                        "task_id": self.current_key[0],
                        "mode": self.current_key[1],
                    }
                    if self.current_key is not None
                    else None
                ),
                "error": dict(error) if error is not None else None,
                "updated_at": utc_timestamp(),
            },
        )


def one_stroke_result_from_dict(raw: dict[str, Any]) -> Any:
    """Rehydrate a stored result while restoring tuple-typed dataclass fields."""

    from minibench.datasets.one_stroke.evaluation import OneStrokeInstanceResult

    payload = dict(raw)
    for field in (
        "rule_types",
        "conversation",
        "tags",
        "history_protocol_reasons",
    ):
        value = payload.get(field)
        if isinstance(value, list):
            payload[field] = tuple(value)
    return OneStrokeInstanceResult(**payload)

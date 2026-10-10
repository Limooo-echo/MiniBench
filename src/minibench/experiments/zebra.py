"""Zebra checkpoint orchestration, resume validation and durable state."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minibench.core.checkpoint import (
    RunLock,
    atomic_write_json,
    fingerprint_payload,
    read_json_object,
    read_jsonl_objects,
    sha256_file,
)
from minibench.experiments.metadata import _plain_config, _utc_timestamp
from minibench.experiments.registry import TaskFamilySpec
from minibench.factory.agents import make_agent_from_config


_ZEBRA_MANIFEST_SCHEMA_VERSION = 1
_ZEBRA_STATE_SCHEMA_VERSION = 1


def _zebra_sidecar_lock_name(run_name: str) -> str:
    digest = hashlib.sha256(run_name.encode("utf-8")).hexdigest()[:16]
    return f".zebra-lock-{digest}"


def _zebra_fingerprint_inputs(
    config: dict[str, Any],
    *,
    task_path: Path,
    plan_keys: list[tuple[str, str]],
) -> dict[str, Any]:
    agent_config = config["agent"]
    provider_config = config.get("provider", {})
    prediction_path = agent_config.get("predictions")
    if prediction_path:
        model_identity: dict[str, Any] = {
            "kind": "prediction_file",
            "sha256": sha256_file(Path(prediction_path)),
        }
    else:
        from minibench.factory.providers import resolve_provider

        resolved_model, resolved_base_url, _ = resolve_provider(
            str(provider_config.get("name", "generic")),
            model=provider_config.get("model"),
            base_url=provider_config.get("base_url"),
            api_key_env=provider_config.get("api_key_env"),
        )
        model_identity = {
            "kind": "model",
            "model": resolved_model,
            "base_url": resolved_base_url,
        }
    semantic_config = {
        section: _plain_config(config.get(section, {}))
        for section in ("task", "agent", "provider", "evaluation")
    }
    return {
        "family": "zebra",
        "data_sha256": sha256_file(task_path),
        "work_plan": [
            {"task_id": task_id, "mode": mode} for task_id, mode in plan_keys
        ],
        "config": semantic_config,
        "model_identity": model_identity,
        "code_sha256": _zebra_code_sha256(),
    }


def _zebra_code_sha256() -> str:
    package_root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((package_root / "experiments").glob("*.py")),
        package_root / "core" / "agent.py",
        package_root / "core" / "checkpoint.py",
        package_root / "core" / "metrics.py",
        package_root / "core" / "prompts.py",
        package_root / "core" / "runtime.py",
        package_root / "factory" / "agents.py",
        package_root / "factory" / "providers.py",
        package_root / "datasets" / "zebra" / "dataset.py",
        package_root / "datasets" / "zebra" / "evaluation.py",
        package_root / "datasets" / "zebra" / "prompting.py",
        *sorted((package_root / "agents").glob("*.py")),
    ]
    file_hashes = {
        path.relative_to(package_root).as_posix(): sha256_file(path)
        for path in paths
    }
    return fingerprint_payload(file_hashes)


def _run_checkpointed_zebra_experiment(
    spec: TaskFamilySpec,
    tasks: list[Any],
    config: dict[str, Any],
    task_path: Path,
    evaluation_config: dict[str, Any],
    run_config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    from minibench.datasets.zebra.evaluation import (
        plan_zebra_work_items,
        write_zebra_snapshot,
        zebra_result_key,
    )

    memory_modes = evaluation_config.get(
        "memory_modes",
        ("incremental_state", "deferred_reasoning"),
    )
    if isinstance(memory_modes, str):
        memory_modes = (memory_modes,)
    selected_modes = tuple(memory_modes)
    work_plan = plan_zebra_work_items(tasks, selected_modes)
    plan_keys = [
        (task.id, mode if mode is not None else "single")
        for task, mode in work_plan
    ]
    plan_index = {key: index for index, key in enumerate(plan_keys)}
    planned_total = len(plan_keys)

    on_existing = str(run_config.get("on_existing", "error"))
    if on_existing not in {"error", "resume"}:
        raise ValueError("run.on_existing must be error or resume")
    configured_run_name = run_config.get("run_name")
    if configured_run_name is not None and (
        not isinstance(configured_run_name, str) or not configured_run_name.strip()
    ):
        raise ValueError("run.run_name must be a non-empty string or null")
    if isinstance(configured_run_name, str) and (
        configured_run_name in {".", ".."}
        or Path(configured_run_name).is_absolute()
        or Path(configured_run_name).name != configured_run_name
        or "/" in configured_run_name
        or "\\" in configured_run_name
    ):
        raise ValueError("run.run_name must be a single directory name")
    if on_existing == "resume" and configured_run_name is None:
        raise ValueError("run.run_name is required when run.on_existing=resume")
    run_name = configured_run_name or (
        "zebra-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    )
    output_dir = Path(run_config.get("output_dir", "runs"))
    run_dir = output_dir / run_name

    fingerprint_inputs = _zebra_fingerprint_inputs(
        config,
        task_path=task_path,
        plan_keys=plan_keys,
    )
    manifest = {
        "schema_version": _ZEBRA_MANIFEST_SCHEMA_VERSION,
        "family": "zebra",
        "fingerprint": fingerprint_payload(fingerprint_inputs),
        "fingerprint_inputs": fingerprint_inputs,
        "created_at": _utc_timestamp(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(
        output_dir,
        lock_name=_zebra_sidecar_lock_name(run_name),
    ):
        fresh_run = False
        if run_dir.exists():
            if run_dir.is_symlink() or not run_dir.is_dir():
                raise ValueError(f"Zebra run path must be a real directory: {run_dir}")
            if on_existing == "error":
                raise FileExistsError(
                    f"run directory already exists: {run_dir}; choose a new run_name "
                    "or set run.on_existing=resume"
                )
            existing_names = {path.name for path in run_dir.iterdir()}
            if "manifest.json" not in existing_names:
                unexplained = existing_names
                if unexplained:
                    rendered = ", ".join(sorted(unexplained))
                    raise ValueError(
                        f"cannot resume uninitialized Zebra run with unexpected "
                        f"file(s): {rendered}"
                    )
                fresh_run = True
        else:
            run_dir.mkdir()
            fresh_run = True

        if fresh_run:
            state_created_at = manifest["created_at"]
            resume_count = 0
            atomic_write_json(run_dir / "manifest.json", manifest)
            completed_results: list[Any] = []
            write_zebra_snapshot(
                run_dir,
                completed_results,
                planned_total=planned_total,
                run_status="running",
            )
            _write_zebra_state(
                run_dir,
                status="running",
                planned_total=planned_total,
                completed_total=0,
                created_at=state_created_at,
                resume_count=resume_count,
            )
        else:
            allowed_names = {
                "manifest.json",
                "predictions.jsonl",
                "results.json",
                "summary.txt",
                "run_state.json",
            }
            unexpected = {
                path.name for path in run_dir.iterdir()
            } - allowed_names
            if unexpected:
                rendered = ", ".join(sorted(unexpected))
                raise ValueError(
                    f"cannot resume Zebra run with unexpected file(s): {rendered}"
                )
            invalid_entries = {
                path.name
                for path in run_dir.iterdir()
                if path.is_symlink() or not path.is_file()
            }
            if invalid_entries:
                rendered = ", ".join(sorted(invalid_entries))
                raise ValueError(
                    f"cannot resume Zebra run with non-file artifact(s): {rendered}"
                )
            actual_manifest = _validate_zebra_manifest(run_dir, manifest)
            predictions_path = run_dir / "predictions.jsonl"
            if not predictions_path.is_file():
                later_artifacts = existing_names & {
                    "results.json",
                    "summary.txt",
                    "run_state.json",
                }
                if later_artifacts:
                    rendered = ", ".join(sorted(later_artifacts))
                    raise ValueError(
                        "corrupt Zebra run: predictions.jsonl is missing while "
                        f"later artifact(s) exist: {rendered}"
                    )
            state_path = run_dir / "run_state.json"
            if state_path.is_file():
                state = _read_zebra_state(run_dir)
                if state["status"] == "completed":
                    raise ValueError(
                        f"Zebra run is already completed: {run_dir}; "
                        "choose a new run_name"
                    )
                state_created_at = state["created_at"]
                if state_created_at != actual_manifest["created_at"]:
                    raise ValueError(
                        f"Zebra state created_at does not match manifest: {run_dir}"
                    )
                resume_count = state["resume_count"] + 1
            else:
                state_created_at = actual_manifest["created_at"]
                resume_count = 1

            if predictions_path.is_file():
                completed_results = _load_zebra_predictions(
                    run_dir,
                    tasks=tasks,
                    work_plan=work_plan,
                    plan_index=plan_index,
                )
            else:
                completed_results = []
            write_zebra_snapshot(
                run_dir,
                completed_results,
                planned_total=planned_total,
                run_status="running",
            )
            _write_zebra_state(
                run_dir,
                status="running",
                planned_total=planned_total,
                completed_total=len(completed_results),
                created_at=state_created_at,
                resume_count=resume_count,
            )

        completed_keys = {zebra_result_key(result) for result in completed_results}
        if completed_keys == set(plan_keys):
            summary = write_zebra_snapshot(
                run_dir,
                completed_results,
                planned_total=planned_total,
                run_status="completed",
            )
            _write_zebra_state(
                run_dir,
                status="completed",
                planned_total=planned_total,
                completed_total=len(completed_results),
                created_at=state_created_at,
                resume_count=resume_count,
            )
            return run_dir, summary

        current_key: tuple[str, str] | None = None

        def work_item_started(key: tuple[str, str]) -> None:
            nonlocal current_key
            current_key = key
            _write_zebra_state(
                run_dir,
                status="running",
                planned_total=planned_total,
                completed_total=len(completed_results),
                created_at=state_created_at,
                resume_count=resume_count,
                current_work_key=key,
            )

        def checkpoint_result(result: Any) -> None:
            nonlocal current_key
            key = zebra_result_key(result)
            if key not in plan_index:
                raise ValueError(
                    f"Zebra evaluator returned unknown work item: {key[0]}/{key[1]}"
                )
            if key in completed_keys:
                raise ValueError(
                    f"Zebra evaluator returned duplicate work item: {key[0]}/{key[1]}"
                )
            candidate_results = [*completed_results, result]
            candidate_results.sort(
                key=lambda item: plan_index[zebra_result_key(item)]
            )
            write_zebra_snapshot(
                run_dir,
                candidate_results,
                planned_total=planned_total,
                run_status="running",
            )
            completed_results[:] = candidate_results
            completed_keys.add(key)
            current_key = None
            _write_zebra_state(
                run_dir,
                status="running",
                planned_total=planned_total,
                completed_total=len(completed_results),
                created_at=state_created_at,
                resume_count=resume_count,
            )

        try:
            agent = make_agent_from_config(
                config["agent"],
                config.get("provider", {}),
                system_prompt=spec.system_prompt,
            )
            final_max_tokens = evaluation_config.get("final_max_tokens")
            spec.evaluate_tasks(
                tasks,
                agent,
                memory_modes=selected_modes,
                state_max_tokens=int(evaluation_config.get("state_max_tokens", 512)),
                ack_max_tokens=int(evaluation_config.get("ack_max_tokens", 32)),
                final_max_tokens=(
                    int(final_max_tokens) if final_max_tokens is not None else None
                ),
                show_progress=bool(evaluation_config.get("show_progress", False)),
                skip_keys=completed_keys,
                on_work_item_start=work_item_started,
                on_result=checkpoint_result,
            )
            missing_keys = set(plan_keys) - completed_keys
            if missing_keys:
                rendered = ", ".join(
                    f"{task_id}/{mode}" for task_id, mode in sorted(missing_keys)
                )
                raise RuntimeError(
                    f"Zebra evaluator returned without completing work item(s): {rendered}"
                )
        except (Exception, KeyboardInterrupt) as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            try:
                completed_results = _load_zebra_predictions(
                    run_dir,
                    tasks=tasks,
                    work_plan=work_plan,
                    plan_index=plan_index,
                )
                durable_keys = {
                    zebra_result_key(result) for result in completed_results
                }
                if current_key in durable_keys:
                    current_key = None
            except (OSError, ValueError):
                pass
            write_zebra_snapshot(
                run_dir,
                completed_results,
                planned_total=planned_total,
                run_status="interrupted",
                error=error_text,
            )
            _write_zebra_state(
                run_dir,
                status="interrupted",
                planned_total=planned_total,
                completed_total=len(completed_results),
                created_at=state_created_at,
                resume_count=resume_count,
                current_work_key=current_key,
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise RuntimeError(
                "zebra evaluation failed after "
                f"{len(completed_results)}/{planned_total} results; partial results "
                f"saved to {run_dir}: {exc}"
            ) from exc

        completed_results.sort(key=lambda item: plan_index[zebra_result_key(item)])
        summary = write_zebra_snapshot(
            run_dir,
            completed_results,
            planned_total=planned_total,
            run_status="completed",
        )
        _write_zebra_state(
            run_dir,
            status="completed",
            planned_total=planned_total,
            completed_total=len(completed_results),
            created_at=state_created_at,
            resume_count=resume_count,
        )
        return run_dir, summary


def _validate_zebra_manifest(
    run_dir: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"cannot resume Zebra run without manifest.json: {run_dir}")
    actual = read_json_object(manifest_path)
    if actual.get("schema_version") != _ZEBRA_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported Zebra manifest schema in {manifest_path}")
    if actual.get("family") != "zebra":
        raise ValueError(f"run manifest is not for Zebra: {manifest_path}")
    if not isinstance(actual.get("created_at"), str) or not actual["created_at"]:
        raise ValueError(f"invalid Zebra manifest created_at in {manifest_path}")
    actual_inputs = actual.get("fingerprint_inputs")
    if (
        not isinstance(actual_inputs, dict)
        or fingerprint_payload(actual_inputs) != actual.get("fingerprint")
    ):
        raise ValueError(f"corrupt Zebra fingerprint in {manifest_path}")
    if actual.get("fingerprint") != expected["fingerprint"]:
        raise ValueError(
            f"Zebra resume fingerprint mismatch for {run_dir}; use a new run_name"
        )
    return actual


def _read_zebra_state(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / "run_state.json"
    if not state_path.is_file():
        raise ValueError(f"cannot resume Zebra run without run_state.json: {run_dir}")
    state = read_json_object(state_path)
    if state.get("schema_version") != _ZEBRA_STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported Zebra state schema in {state_path}")
    if state.get("status") not in {"running", "interrupted", "completed"}:
        raise ValueError(f"invalid Zebra run status in {state_path}")
    if not isinstance(state.get("created_at"), str) or not state["created_at"]:
        raise ValueError(f"invalid Zebra created_at in {state_path}")
    resume_count = state.get("resume_count")
    if (
        isinstance(resume_count, bool)
        or not isinstance(resume_count, int)
        or resume_count < 0
    ):
        raise ValueError(f"invalid Zebra resume_count in {state_path}")
    return state


def _load_zebra_predictions(
    run_dir: Path,
    *,
    tasks: list[Any],
    work_plan: list[tuple[Any, str | None]],
    plan_index: dict[tuple[str, str], int],
) -> list[Any]:
    from minibench.datasets.zebra.evaluation import (
        score_zebra_output,
        zebra_result_from_dict,
        zebra_result_key,
    )

    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise ValueError(
            f"cannot resume Zebra run without predictions.jsonl: {run_dir}"
        )
    tasks_by_id = {task.id: task for task in tasks}
    expected_modes = {
        (
            task.id,
            mode if mode is not None else "single",
        ): mode
        for task, mode in work_plan
    }
    results: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for raw_result in read_jsonl_objects(predictions_path):
        result = zebra_result_from_dict(raw_result)
        key = zebra_result_key(result)
        if key not in plan_index:
            raise ValueError(
                f"predictions.jsonl contains unknown Zebra work item: "
                f"{key[0]}/{key[1]}"
            )
        if key in seen:
            raise ValueError(
                f"predictions.jsonl contains duplicate Zebra work item: "
                f"{key[0]}/{key[1]}"
            )
        seen.add(key)
        task = tasks_by_id[result.task_id]
        expected_mode = expected_modes[key]
        expected_metadata = {
            "source_id": task.source_id,
            "variant": task.variant,
            "size": task.size,
            "difficulty": task.difficulty,
            "capability": task.capability,
            "rule_mode": task.rule_mode,
            "memory_mode": expected_mode,
            "tags": task.tags,
        }
        for field, expected_value in expected_metadata.items():
            if getattr(result, field) != expected_value:
                raise ValueError(
                    f"predictions.jsonl has mismatched {field} for "
                    f"{key[0]}/{key[1]}"
                )
        rescored = score_zebra_output(task, result.raw_output)
        for field in (
            "success",
            "score",
            "correct_cells",
            "total_cells",
            "cell_accuracy",
            "parsed",
            "no_answer",
            "reasoning",
        ):
            if getattr(result, field) != rescored[field]:
                raise ValueError(
                    f"predictions.jsonl has inconsistent {field} for "
                    f"{key[0]}/{key[1]}"
                )
        if expected_mode is None and result.conversation:
            raise ValueError(
                f"predictions.jsonl has a conversation for direct work item "
                f"{key[0]}/{key[1]}"
            )
        results.append(result)
    results.sort(key=lambda result: plan_index[zebra_result_key(result)])
    return results


def _write_zebra_state(
    run_dir: Path,
    *,
    status: str,
    planned_total: int,
    completed_total: int,
    created_at: str,
    resume_count: int,
    current_work_key: tuple[str, str] | None = None,
    error: dict[str, str] | None = None,
) -> None:
    atomic_write_json(
        run_dir / "run_state.json",
        {
            "schema_version": _ZEBRA_STATE_SCHEMA_VERSION,
            "status": status,
            "planned_total": planned_total,
            "completed_total": completed_total,
            "remaining_total": planned_total - completed_total,
            "created_at": created_at,
            "resume_count": resume_count,
            "current_work_key": (
                {
                    "task_id": current_work_key[0],
                    "mode": current_work_key[1],
                }
                if current_work_key is not None
                else None
            ),
            "error": error,
            "updated_at": _utc_timestamp(),
        },
    )

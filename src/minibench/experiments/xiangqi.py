"""Frozen Xiangqi selection, durable results and reproducibility artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import strftime
from typing import Any

import yaml

from minibench.core.checkpoint import (
    atomic_write_json,
    fingerprint_payload,
    read_json_object,
    sha256_file,
)
from minibench.experiments.dispatch import _evaluate
from minibench.experiments.metadata import (
    _git_is_dirty,
    _git_revision,
    _model_request_metadata,
    _plain_config,
)
from minibench.experiments.registry import TaskFamilySpec
from minibench.factory.agents import make_agent_from_config


def _select_frozen_xiangqi_tasks(
    tasks: list[Any], task_config: dict[str, Any], data_path: Path, family: str
) -> list[Any]:
    """Use one hashed roster across architectures, without sampling it again."""
    from minibench.datasets.xiangqi.schema import load_records

    if not family.startswith("xiangqi-"):
        raise ValueError("task.selection is currently supported only for Xiangqi")
    selection = task_config["selection"]
    if (not isinstance(selection, dict) or not isinstance(selection.get("path"), str)
            or not isinstance(selection.get("sha256"), str)):
        raise ValueError("task.selection needs path and sha256")
    if (task_config.get("task_ids") or task_config.get("limit") is not None
            or (task_config.get("sampling") or {}).get("enabled")):
        raise ValueError("task.selection cannot be combined with task_ids, limit, or sampling")
    path = Path(selection["path"])
    if sha256_file(path) != selection["sha256"]:
        raise ValueError("task.selection sha256 mismatch")
    records = load_records(path, expected_family=family)
    originals = {r["id"]: r for r in load_records(data_path, expected_family=family)}
    if not records:
        raise ValueError("task.selection must be nonempty")
    for record in records:
        if originals.get(record["id"]) != record:
            raise ValueError(f"task.selection differs from the source dataset: {record['id']}")
    indexed = {str(t["id"] if isinstance(t, dict) else t.id): t for t in tasks}
    return [indexed[r["id"]] for r in records]


def _run_checkpointed_xiangqi_experiment(
    spec: TaskFamilySpec, tasks: list[Any], config: dict[str, Any], data_path: Path,
    family: str, evaluation_config: dict[str, Any], run_config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Persist inputs before model setup and fsync each completed observation."""
    if not tasks:
        raise ValueError("Xiangqi evaluation requires at least one selected task")
    name = run_config.get("run_name") or f"{family}-{strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(run_config.get("output_dir", "runs")) / name
    # Reusing a name must never truncate an earlier run, including an interrupted one.
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions = run_dir / "predictions.jsonl"
    predictions.touch(exist_ok=False)
    completed: list[Any] = []
    identities: set[tuple[str, str]] = set()
    state = {
        "version": 1, "family": family, "status": "preparing",
        "selected_tasks": len(tasks), "completed_results": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    def save_state(status: str, **extra: Any) -> None:
        state.update(status=status, completed_results=len(completed),
                     updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        atomic_write_json(run_dir / "run_state.json", state)

    save_state("preparing")
    try:
        selected_path = _write_xiangqi_selected_tasks(
            run_dir, tasks=tasks, data_path=data_path, family=family
        )
        _write_xiangqi_run_metadata(
            run_dir, config=config, data_path=data_path, family=family,
            selected_tasks_path=selected_path,
        )
        save_state("running")
        agent = make_agent_from_config(
            config["agent"], config.get("provider", {}), system_prompt=spec.system_prompt
        )

        def persist(result: Any) -> None:
            record = asdict(result) if is_dataclass(result) else dict(result)
            task_id = str(record.get("task_id", record.get("id", "")))
            mode = str(record.get("input_mode") or record.get("history_mode") or "single")
            identity = task_id, mode
            if not task_id or identity in identities:
                raise ValueError(f"Invalid or duplicate Xiangqi checkpoint result: {identity}")
            line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            with predictions.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            identities.add(identity)
            completed.append(result)
            save_state("running", last_task_id=task_id, last_mode=mode)

        results = _evaluate(spec, tasks, agent, family, evaluation_config, on_result=persist)
        if len(results) != len(completed):
            raise RuntimeError("Xiangqi evaluator returned results without checkpoint callbacks")
        # Writers retain their native summaries while leaving the durable JSONL intact.
        spec.write_run(results, run_dir.parent, run_dir.name, write_predictions=False)
        summary = spec.summarize(results)
        save_state("completed")
        return run_dir, summary
    except BaseException as exc:
        save_state("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "error",
                   error_type=type(exc).__name__, error=str(exc))
        raise


def _sample_xiangqi_tasks(
    tasks: list[Any],
    *,
    family: str,
    count: int,
    seed: int,
    strategy: str = "family-default",
) -> list[Any]:
    if not family.startswith("xiangqi-"):
        return tasks[:count]
    from minibench.datasets.xiangqi.schema import sample_records

    indexed: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for task in tasks:
        if isinstance(task, dict):
            record = {key: value for key, value in task.items() if key != "board"}
        else:
            record = {
                "schema_version": task.schema_version,
                "id": task.id,
                "family": task.family,
                "fen": task.fen,
                "agent_color": task.agent_color,
                "goal": "checkmate",
                "max_plies": task.max_steps,
                "difficulty": task.difficulty,
                "piece_count": sum(value != 0 for row in task.board for value in row),
                "oracle": task.oracle,
                "tags": list(task.tags),
            }
        indexed[record["id"]] = task
        records.append(record)
    selected = sample_records(records, count=count, seed=seed, strategy=strategy)
    return [indexed[record["id"]] for record in selected]


def _write_xiangqi_run_metadata(
    run_dir: Path,
    *,
    config: dict[str, Any],
    data_path: Path,
    family: str,
    selected_tasks_path: Path,
) -> None:
    from minibench.datasets.xiangqi.multimodal import (
        M2_PROTOCOL_VERSION,
        XIANGQI_RENDERER_VERSION,
    )
    from minibench.datasets.xiangqi.runtime import PROTOCOL_VERSION
    from minibench.datasets.xiangqi.schema import SCHEMA_VERSION

    data_bytes = data_path.read_bytes()
    dependencies: dict[str, str] = {}
    for distribution in (
        "Pillow", "matplotlib", "networkx", "numpy", "PyYAML",
        "mahjong", "datasets",
    ):
        try:
            dependencies[distribution] = version(distribution)
        except PackageNotFoundError:
            dependencies[distribution] = "not-installed"
    resolved_config = yaml.safe_dump(
        _plain_config(config), sort_keys=False, allow_unicode=True
    ).encode("utf-8")
    metadata = {
        "family": family,
        "protocol": {
            "version": config["task"].get(
                "prompt_version",
                M2_PROTOCOL_VERSION if family == "xiangqi-multimodal" else PROTOCOL_VERSION,
            ),
            "evaluation": _plain_config(config.get("evaluation") or {}),
        },
        "schema_version": SCHEMA_VERSION,
        "renderer_version": XIANGQI_RENDERER_VERSION,
        "data_file": data_path.as_posix(),
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "selected_tasks_file": selected_tasks_path.name,
        "selected_tasks_sha256": sha256_file(selected_tasks_path),
        "selected_record_count": sum(
            1
            for line in selected_tasks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        "resolved_config_sha256": hashlib.sha256(resolved_config).hexdigest(),
        "dependencies": dependencies,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": _git_revision(Path.cwd()),
            "git_dirty": _git_is_dirty(Path.cwd()),
        },
        "model_request": _model_request_metadata(config.get("provider", {})),
    }
    package_root = Path(__file__).resolve().parents[1]
    code_hashes = {path.relative_to(package_root).as_posix(): sha256_file(path)
                   for path in sorted(package_root.rglob("*.py"))}
    metadata["runtime"]["source_files_sha256"] = code_hashes
    metadata["runtime"]["source_sha256"] = fingerprint_payload(code_hashes)
    selection = (config.get("task") or {}).get("selection")
    if selection is not None:
        metadata["selection_file"] = str(selection["path"])
        metadata["selection_sha256"] = selection["sha256"]
    for provenance_path in (data_path.parent / "provenance.json", data_path.parent.parent / "provenance.json"):
        if provenance_path.is_file():
            provenance = read_json_object(provenance_path)
            if provenance.get("release_id"):
                metadata["release_id"] = provenance["release_id"]
                metadata["protocol"]["release_id"] = provenance["release_id"]
                metadata["dataset_provenance_sha256"] = sha256_file(provenance_path)
                break
    evaluation = config.get("evaluation") or {}
    pikafish_path = evaluation.get("pikafish_path")
    uses_pikafish = family in {
        "xiangqi-mate-in-one",
        "xiangqi-history",
        "xiangqi-multimodal",
    } and bool(evaluation.get("verify_with_pikafish", True))
    if pikafish_path or uses_pikafish:
        from minibench.datasets.xiangqi.engines.pikafish import (
            pikafish_fingerprint,
            resolve_pikafish_executable,
        )

        try:
            executable = resolve_pikafish_executable(
                pikafish_path, start_dir=Path.cwd()
            )
            metadata["pikafish"] = pikafish_fingerprint(executable)
        except (FileNotFoundError, RuntimeError, ValueError):
            # Static/prediction-only tests may intentionally skip engine setup.
            metadata["pikafish"] = None
    else:
        metadata["pikafish"] = None
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(_plain_config(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_xiangqi_selected_tasks(
    run_dir: Path,
    *,
    tasks: list[Any],
    data_path: Path,
    family: str,
) -> Path:
    """Persist the exact sampled records used by a Xiangqi run."""
    from minibench.datasets.xiangqi.schema import load_records

    selected_ids = [
        str(task["id"] if isinstance(task, dict) else task.id) for task in tasks
    ]
    records = load_records(data_path, expected_family=family)
    by_id = {record["id"]: record for record in records}
    missing = [task_id for task_id in selected_ids if task_id not in by_id]
    if missing:
        raise ValueError(
            "selected Xiangqi tasks are absent from the source dataset: "
            + ", ".join(missing)
        )
    path = run_dir / "selected_tasks.jsonl"
    path.write_text(
        "".join(
            json.dumps(by_id[task_id], ensure_ascii=False, sort_keys=True) + "\n"
            for task_id in selected_ids
        ),
        encoding="utf-8",
    )
    return path

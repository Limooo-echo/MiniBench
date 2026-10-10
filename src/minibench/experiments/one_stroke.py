"""One-stroke run planning, input fingerprints and checkpoint orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from minibench.core.checkpoint import fingerprint_payload, sha256_file
from minibench.experiments.metadata import _plain_config
from minibench.experiments.registry import TaskFamilySpec
from minibench.experiments.selection import _one_stroke_selected_modes
from minibench.factory.agents import make_agent_from_config


def _one_stroke_model_identity(config: dict[str, Any]) -> dict[str, Any]:
    agent_config = config["agent"]
    provider_config = config.get("provider", {})
    prediction_path = agent_config.get("predictions")
    if prediction_path:
        return {
            "kind": "prediction_file",
            "provider": None,
            "model": None,
            "sha256": sha256_file(Path(prediction_path)),
        }

    from minibench.factory.providers import resolve_provider

    provider_name = str(provider_config.get("name", "generic"))
    resolved_model, resolved_base_url, _ = resolve_provider(
        provider_name,
        model=provider_config.get("model"),
        base_url=provider_config.get("base_url"),
        api_key_env=provider_config.get("api_key_env"),
    )
    return {
        "kind": "model",
        "provider": provider_name,
        "model": resolved_model,
        "base_url": resolved_base_url,
    }


def _one_stroke_fingerprint_inputs(
    config: dict[str, Any],
    *,
    task_sha256: str,
    model_identity: dict[str, Any],
    work_plan: tuple[tuple[str, str], ...],
    input_assets: list[dict[str, str]],
    dataset_profile: dict[str, Any],
) -> dict[str, Any]:
    semantic_config = {
        section: _plain_config(config.get(section, {}))
        for section in ("task", "agent", "provider", "evaluation")
    }
    return {
        "family": "one_stroke",
        "data_sha256": task_sha256,
        "work_plan": [
            {"task_id": task_id, "mode": mode}
            for task_id, mode in work_plan
        ],
        "input_asset_hashes": [
            {
                "task_id": asset["task_id"],
                "sha256": asset["sha256"],
            }
            for asset in input_assets
        ],
        "config": semantic_config,
        "model_identity": model_identity,
        "dataset_profile": dataset_profile,
        "code_sha256": _one_stroke_code_sha256(),
    }


def _one_stroke_input_assets(tasks: list[Any]) -> list[dict[str, str]]:
    assets: list[dict[str, str]] = []
    for task in tasks:
        if task.capability != "multimodal":
            continue
        if task.image_path is None:
            raise ValueError(f"{task.id}: multimodal tasks require an image")
        path = Path(task.image_path).resolve()
        assets.append(
            {"task_id": task.id, "path": str(path), "sha256": sha256_file(path)}
        )
    return sorted(assets, key=lambda asset: asset["task_id"])


def _one_stroke_dataset_profile(
    tasks: list[Any],
    *,
    work_plan: tuple[tuple[str, str], ...],
    input_modes: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from minibench.datasets.one_stroke.dataset import (
        has_one_stroke_solution,
        one_stroke_edge_ids,
        simulate_one_stroke_history,
    )

    def counts(field: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for task in tasks:
            key = str(getattr(task, field))
            result[key] = result.get(key, 0) + 1
        return dict(sorted(result.items()))

    history_tasks = [task for task in tasks if task.capability == "history_memory"]
    multimodal_tasks = [task for task in tasks if task.capability == "multimodal"]
    tasks_by_id = {task.id: task for task in tasks}
    protocol_generation_total = sum(
        len(tasks_by_id[task_id].history_events) + 1
        if mode.startswith("memory:")
        else 1
        for task_id, mode in work_plan
    )
    base_solution_flags = [
        has_one_stroke_solution(
            task.vertices,
            task.edges,
            start=task.start,
            end=task.end,
        )
        for task in tasks
    ]
    history_completion_flags: list[bool] = []
    for task in history_tasks:
        state = simulate_one_stroke_history(task)
        remaining_ids = set(state.remaining_edge_ids)
        remaining_edges = tuple(
            edge
            for edge_id, edge in zip(one_stroke_edge_ids(task.edges), task.edges)
            if edge_id in remaining_ids
        )
        history_completion_flags.append(
            has_one_stroke_solution(
                task.vertices,
                remaining_edges,
                start=state.current_vertex,
                end=task.end,
            )
        )
    negative_history_count = sum(int(not value) for value in history_completion_flags)
    visual_gap_not_estimable = bool(multimodal_tasks) and not {
        "text", "image"
    }.issubset(input_modes)
    profile = {
        "task_total": len(tasks),
        "work_item_total": len(work_plan),
        "protocol_generation_total": protocol_generation_total,
        "protocol_generation_semantics": {
            "kind": "minimum_protocol_generate_calls",
            "ordinary_work_item_calls": 1,
            "history_work_item_calls": "history_event_count + 1 final call",
            "reasoning_wrapper_may_add_internal_final_calls": True,
        },
        "by_capability": counts("capability"),
        "by_difficulty": counts("difficulty"),
        "solution_exists_semantics": "task_base_graph",
        "by_solution_exists": {
            "true": sum(int(value) for value in base_solution_flags),
            "false": sum(int(not value) for value in base_solution_flags),
        },
        "history": {
            "task_total": len(history_tasks),
            "negative_history_count": negative_history_count,
            "no_negative_history_cases": (
                negative_history_count == 0 if history_tasks else None
            ),
        },
        "multimodal": {
            "task_total": len(multimodal_tasks),
            "selected_input_modes": list(input_modes),
            "visual_gap_not_estimable": visual_gap_not_estimable,
        },
    }

    warnings: list[dict[str, Any]] = [
        {
            "code": "cross_track_raw_scores_not_attributable",
            "category": "interpretation",
            "applies_to": ["direct", "history", "multimodal"],
            "condition": "model_identity_or_task_fingerprint_differs",
            "comparison_keys": [
                "model_identity",
                "fingerprint_inputs.data_sha256",
                "fingerprint_inputs.input_asset_hashes",
            ],
            "message": (
                "Do not attribute raw score differences across direct, history, "
                "and multimodal tracks to the capability alone when model "
                "identity or dataset fingerprint differs."
            ),
        }
    ]
    if history_tasks:
        warnings.append(
            {
                "code": "history_transcript_context_not_persistent_memory",
                "category": "design",
                "applies_to": ["history"],
                "message": (
                    "History tasks measure state tracking inside the supplied "
                    "conversation transcript, not persistent memory across sessions."
                ),
            }
        )
        if negative_history_count == 0:
            warnings.append(
                {
                    "code": "history_no_negative_history_cases",
                    "category": "design",
                    "applies_to": ["history"],
                    "message": (
                        "All loaded history tasks are structurally completable; "
                        "there are no negative history cases."
                    ),
                }
            )
    if multimodal_tasks:
        warnings.append(
            {
                "code": "multimodal_report_path_transcription_and_joint",
                "category": "interpretation",
                "applies_to": ["multimodal"],
                "required_score_views": [
                    "multimodal_path_score",
                    "multimodal_transcription_score",
                    "multimodal_joint_score",
                ],
                "message": (
                    "Interpret multimodal tasks using separate path, "
                    "graph-transcription, and joint scores; no single "
                    "component is a substitute for all three."
                ),
            }
        )
    if visual_gap_not_estimable:
        warnings.append(
            {
                "code": "multimodal_visual_gap_not_estimable",
                "category": "design",
                "applies_to": ["multimodal"],
                "selected_input_modes": list(input_modes),
                "message": (
                    "A paired text/image visual gap requires both text "
                    "and image results from this run."
                ),
            }
        )
    return profile, warnings


def _one_stroke_code_sha256() -> str:
    package_root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((package_root / "experiments").glob("*.py")),
        package_root / "core" / "agent.py",
        package_root / "core" / "checkpoint.py",
        package_root / "core" / "metrics.py",
        package_root / "core" / "multimodal.py",
        package_root / "core" / "prompts.py",
        package_root / "core" / "runtime.py",
        package_root / "core" / "one_stroke_checkpoint.py",
        package_root / "factory" / "agents.py",
        package_root / "factory" / "providers.py",
        package_root / "datasets" / "one_stroke" / "dataset.py",
        package_root / "datasets" / "one_stroke" / "evaluation.py",
        package_root / "datasets" / "one_stroke" / "multimodal.py",
        package_root / "datasets" / "one_stroke" / "prompting.py",
        *sorted((package_root / "agents").glob("*.py")),
    ]
    file_hashes = {
        path.relative_to(package_root).as_posix(): sha256_file(path)
        for path in paths
    }
    return fingerprint_payload(file_hashes)


def _run_checkpointed_one_stroke_experiment(
    spec: TaskFamilySpec,
    tasks: list[Any],
    config: dict[str, Any],
    task_path: Path,
    evaluation_config: dict[str, Any],
    run_config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    from minibench.core.one_stroke_checkpoint import (
        OneStrokeCheckpointRun,
        make_manifest,
        one_stroke_result_from_dict,
        validate_run_name,
    )
    from minibench.datasets.one_stroke.evaluation import (
        one_stroke_result_key,
        plan_one_stroke_work_items,
    )

    memory_modes, input_modes = _one_stroke_selected_modes(
        evaluation_config
    )
    work_plan = tuple(
        plan_one_stroke_work_items(
            tasks,
            memory_modes=memory_modes,
            input_modes=input_modes,
        )
    )
    for key in work_plan:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not all(isinstance(value, str) and value for value in key)
        ):
            raise ValueError(
                "one-stroke work planner must return (task_id, mode_key) pairs"
            )

    on_existing = str(run_config.get("on_existing", "error"))
    run_name = validate_run_name(
        run_config.get("run_name"),
        on_existing=on_existing,
    )
    model_identity = _one_stroke_model_identity(config)
    task_sha256 = sha256_file(task_path)
    input_assets = _one_stroke_input_assets(tasks)
    dataset_profile, interpretation_warnings = _one_stroke_dataset_profile(
        tasks,
        work_plan=work_plan,
        input_modes=input_modes,
    )
    fingerprint_inputs = _one_stroke_fingerprint_inputs(
        config,
        task_sha256=task_sha256,
        model_identity=model_identity,
        work_plan=work_plan,
        input_assets=input_assets,
        dataset_profile=dataset_profile,
    )
    repository_root = Path(__file__).resolve().parents[3]
    manifest = make_manifest(
        resolved_config=_plain_config(config),
        task_path=task_path,
        task_sha256=task_sha256,
        model_identity=model_identity,
        work_plan=work_plan,
        input_assets=input_assets,
        dataset_profile=dataset_profile,
        interpretation_warnings=interpretation_warnings,
        fingerprint_inputs=fingerprint_inputs,
        repository_root=repository_root,
    )

    with OneStrokeCheckpointRun(
        output_dir=run_config.get("output_dir", "runs"),
        run_name=run_name,
        on_existing=on_existing,
        manifest=manifest,
        work_plan=work_plan,
        result_key=one_stroke_result_key,
        result_from_dict=one_stroke_result_from_dict,
        summarize=spec.summarize,
    ) as checkpoint:
        if checkpoint.completed_keys == set(work_plan):
            return checkpoint.run_dir, checkpoint.mark_completed()

        try:
            # The run directory, manifest, state, and empty predictions snapshot
            # all exist before constructing an agent that can make API calls.
            agent = make_agent_from_config(
                config["agent"],
                config.get("provider", {}),
                system_prompt=spec.system_prompt,
            )
            final_max_tokens = evaluation_config.get("final_max_tokens")
            spec.evaluate_tasks(
                tasks,
                agent,
                prompt_variant=evaluation_config.get(
                    "prompt_variant", "baseline"
                ),
                memory_modes=memory_modes,
                input_modes=input_modes,
                state_max_tokens=int(
                    evaluation_config.get("state_max_tokens", 512)
                ),
                ack_max_tokens=int(
                    evaluation_config.get("ack_max_tokens", 32)
                ),
                final_max_tokens=(
                    int(final_max_tokens)
                    if final_max_tokens is not None
                    else None
                ),
                show_progress=bool(
                    evaluation_config.get("show_progress", False)
                ),
                skip_keys=set(checkpoint.completed_keys),
                on_work_item_start=checkpoint.start_work_item,
                on_result=checkpoint.record_result,
            )
            missing = set(work_plan) - checkpoint.completed_keys
            if missing:
                rendered = ", ".join(
                    f"{task_id}/{mode}" for task_id, mode in sorted(missing)
                )
                raise RuntimeError(
                    "one-stroke evaluator returned without completing work "
                    f"item(s): {rendered}"
                )
        except (Exception, KeyboardInterrupt) as exc:
            checkpoint.mark_interrupted(exc)
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise RuntimeError(
                "one-stroke evaluation failed after "
                f"{len(checkpoint.results)}/{checkpoint.planned_total} results; "
                f"partial results saved to {checkpoint.run_dir}. To resume with "
                "the unchanged semantic config, set "
                f"run.run_name={checkpoint.run_name!r} and "
                f"run.on_existing='resume': {exc}"
            ) from exc

        return checkpoint.run_dir, checkpoint.mark_completed()

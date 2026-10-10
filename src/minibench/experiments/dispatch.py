"""Forward each family's supported parameters to its evaluator."""
from __future__ import annotations

from typing import Any, Callable

from minibench.core.agent import Agent
from minibench.experiments.registry import TaskFamilySpec
from minibench.experiments.selection import _one_stroke_selected_modes


def _evaluate(
    spec: TaskFamilySpec,
    tasks: list[Any],
    agent: Agent,
    family: str,
    evaluation_config: dict[str, Any],
    on_result: Callable[[Any], None] | None = None,
) -> list[Any]:
    if family.startswith("xiangqi-"):
        from minibench.datasets.xiangqi.runtime import validate_evaluation_config

        validate_evaluation_config(family, evaluation_config)
    if family == "xiangqi-mate-in-one":
        return spec.evaluate_tasks(
            tasks,
            agent,
            on_result=on_result,
            pikafish_path=evaluation_config.get("pikafish_path"),
            pikafish_binary_sha256=evaluation_config.get("pikafish_binary_sha256"),
            pikafish_nnue_sha256=evaluation_config.get("pikafish_nnue_sha256"),
            pikafish_depth=evaluation_config.get("pikafish_depth", 8),
            pikafish_timeout=evaluation_config.get("pikafish_timeout", 60.0),
        )
    if family == "xiangqi-rule-variants":
        return spec.evaluate_tasks(
            tasks,
            agent,
            on_result=on_result,
            oracle_depth=int(evaluation_config.get("oracle_depth", 3)),
        )
    if family == "xiangqi-history":
        return spec.evaluate_tasks(
            tasks,
            agent,
            on_result=on_result,
            history_mode=evaluation_config.get("history_mode", "move-history-only"),
            pikafish_path=evaluation_config.get("pikafish_path"),
            pikafish_binary_sha256=evaluation_config.get("pikafish_binary_sha256"),
            pikafish_nnue_sha256=evaluation_config.get("pikafish_nnue_sha256"),
            pikafish_depth=int(evaluation_config.get("pikafish_depth", 16)),
            pikafish_timeout=float(evaluation_config.get("pikafish_timeout", 60.0)),
        )
    if family == "xiangqi-multimodal":
        modes = evaluation_config.get(
            "input_modes",
            ("text", "chinese-piece-image", "latin-piece-image"),
        )
        return spec.evaluate_tasks(
            tasks,
            agent,
            on_result=on_result,
            modes=tuple(modes),
            pikafish_path=evaluation_config.get("pikafish_path"),
            pikafish_binary_sha256=evaluation_config.get("pikafish_binary_sha256"),
            pikafish_nnue_sha256=evaluation_config.get("pikafish_nnue_sha256"),
            pikafish_depth=int(evaluation_config.get("pikafish_depth", 8)),
            pikafish_timeout=float(
                evaluation_config.get("pikafish_timeout", 60.0)
            ),
            verify_with_pikafish=bool(
                evaluation_config.get("verify_with_pikafish", True)
            ),
            step_dir=evaluation_config.get("step_dir"),
        )
    if family == "one_stroke":
        memory_modes, input_modes = _one_stroke_selected_modes(evaluation_config)
        final_max_tokens = evaluation_config.get("final_max_tokens")
        return spec.evaluate_tasks(
            tasks,
            agent,
            prompt_variant=evaluation_config.get("prompt_variant", "baseline"),
            memory_modes=tuple(memory_modes),
            input_modes=tuple(input_modes),
            state_max_tokens=int(evaluation_config.get("state_max_tokens", 512)),
            ack_max_tokens=int(evaluation_config.get("ack_max_tokens", 32)),
            final_max_tokens=(
                int(final_max_tokens) if final_max_tokens is not None else None
            ),
            show_progress=bool(evaluation_config.get("show_progress", False)),
        )
    if family == "zebra":
        memory_modes = evaluation_config.get(
            "memory_modes",
            ("incremental_state", "deferred_reasoning"),
        )
        if isinstance(memory_modes, str):
            memory_modes = (memory_modes,)
        final_max_tokens = evaluation_config.get("final_max_tokens")
        return spec.evaluate_tasks(
            tasks,
            agent,
            memory_modes=tuple(memory_modes),
            state_max_tokens=int(evaluation_config.get("state_max_tokens", 512)),
            ack_max_tokens=int(evaluation_config.get("ack_max_tokens", 32)),
            final_max_tokens=(
                int(final_max_tokens) if final_max_tokens is not None else None
            ),
            show_progress=bool(evaluation_config.get("show_progress", False)),
        )
    if family == "mahjong_solo":
        return spec.evaluate_tasks(
            tasks,
            agent,
            observation_mode=evaluation_config.get("observation_mode", "full-hand"),
            show_progress=bool(evaluation_config.get("show_progress", False)),
            on_result=on_result,
        )
    if family == "mahjong_rule_variants":
        return spec.evaluate_tasks(
            tasks,
            agent,
            observation_mode=evaluation_config.get("observation_mode", "full-hand"),
            show_progress=bool(evaluation_config.get("show_progress", False)),
            on_result=on_result,
        )
    if family == "mahjong":
        input_modes = evaluation_config.get("input_modes")
        if isinstance(input_modes, str):
            input_modes = (input_modes,)
        return spec.evaluate_tasks(
            tasks,
            agent,
            input_modes=(tuple(input_modes) if input_modes is not None else None),
            show_progress=bool(evaluation_config.get("show_progress", False)),
            on_result=on_result,
        )
    return spec.evaluate_tasks(tasks, agent)

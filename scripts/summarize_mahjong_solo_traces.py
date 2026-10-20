from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Mahjong Solo diagnostic success, terminal actions, and "
            "saved internal reasoning-stage outputs."
        )
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    return parser.parse_args()


def _load_predictions(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing predictions file: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _metric_completion_tokens(metrics: Any) -> int | None:
    if not isinstance(metrics, dict):
        return None
    usage = metrics.get("token_usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("completion_tokens")
    return int(value) if isinstance(value, (int, float)) else None


def _trace_stages(trace: dict[str, Any]) -> list[tuple[str, str, Any]]:
    """Read the runtime's saved stages for every supported agent architecture."""
    stages = trace.get("stages", [])
    if not isinstance(stages, list):
        return []
    return [
        (str(stage.get("stage_name", "unknown")),
         str(stage.get("output") or ""), stage)
        for stage in stages
        if isinstance(stage, dict)
    ]


def summarize_run(run_dir: Path) -> dict[str, Any]:
    predictions = _load_predictions(run_dir)
    actions = Counter(
        str(action.get("action"))
        for prediction in predictions
        for action in prediction.get("agent_actions", [])
        if isinstance(action, dict)
    )
    stage_totals: dict[str, dict[str, int]] = {}
    trace_count = 0
    for prediction in predictions:
        traces = prediction.get("reasoning_traces", [])
        if not isinstance(traces, list):
            continue
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            trace_count += 1
            for stage, output, metrics in _trace_stages(trace):
                item = stage_totals.setdefault(
                    stage,
                    {"calls": 0, "characters": 0, "completion_tokens": 0},
                )
                item["calls"] += 1
                item["characters"] += len(output)
                completion_tokens = _metric_completion_tokens(metrics)
                if completion_tokens is not None:
                    item["completion_tokens"] += completion_tokens

    stage_summary = {
        stage: {
            **totals,
            "average_characters": (
                totals["characters"] / totals["calls"] if totals["calls"] else 0.0
            ),
            "average_completion_tokens": (
                totals["completion_tokens"] / totals["calls"]
                if totals["calls"]
                else 0.0
            ),
        }
        for stage, totals in stage_totals.items()
    }
    success = sum(bool(prediction.get("success")) for prediction in predictions)
    return {
        "run_dir": str(run_dir),
        "tasks": len(predictions),
        "success": success,
        "success_rate": success / len(predictions) if predictions else 0.0,
        "actions": dict(sorted(actions.items())),
        "saved_reasoning_traces": trace_count,
        "stages": stage_summary,
    }


def main() -> int:
    args = _parse_args()
    summaries = [summarize_run(run_dir) for run_dir in args.run_dirs]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

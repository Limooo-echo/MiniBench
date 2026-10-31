"""Recompute accuracy and paired comparisons for the frozen 1028 Zebra run.

Standard-library only. The experiment directory is always read-only; derived
tables are written to a separate analysis directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
TREATMENTS = ("thinking_off", "thinking_on")
MODES = ("deferred_reasoning", "incremental_state", "incremental_state_pro")
SIZE_GROUPS = {
    "small": {"2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3", "4*2"},
    "medium": {"3*4", "3*5", "3*6", "4*3", "4*4", "5*2", "6*2"},
    "large": {"4*5", "5*3", "4*6", "5*4", "6*3"},
    "x-large": {"5*5", "6*4", "5*6", "6*5", "6*6"},
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def size_group(size: str) -> str:
    return next(group for group, sizes in SIZE_GROUPS.items() if size in sizes)


def wilson(success: int, total: int) -> tuple[float | None, float | None]:
    if not total:
        return None, None
    z = 1.959963984540054
    probability = success / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    half = z * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total)) / denominator
    return max(0, center - half), min(1, center + half)


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def strict_rescore(task: dict, output: str) -> dict:
    # Independently reproduce the benchmark's extraction and strict key matching.
    stack, start, last_json = [], None, None
    for index, character in enumerate(output):
        if character == "{":
            stack.append(index)
            if start is None:
                start = index
        elif character == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                last_json = output[start:index + 1]
                start = None
    payload = None
    if last_json is not None:
        try:
            candidate = json.loads(last_json.replace("\n", ""))
            if isinstance(candidate, dict):
                payload = candidate
        except json.JSONDecodeError:
            pass
    solution = payload.get("solution") if payload else None
    parsed = isinstance(solution, dict)
    header = task["solution"]["header"]
    expected = task["solution"]["rows"]
    total_cells = len(expected) * (len(header) - 1)
    correct = 0
    if parsed:
        for house, values in enumerate(expected, 1):
            predicted = solution.get(f"House {house}")
            if not isinstance(predicted, dict):
                continue
            for attribute, truth in zip(header[1:], values[1:]):
                value = predicted.get(attribute)
                if isinstance(value, list):
                    value = value[0] if value else None
                if value is None or isinstance(value, (dict, list)):
                    continue
                correct += str(value).lower().strip() == str(truth).lower().strip()
    return {"success": parsed and correct == total_cells, "parsed": parsed,
            "no_answer": not parsed, "correct_cells": correct, "total_cells": total_cells,
            "cell_accuracy": correct / total_cells}


def load_records(batch: Path, tasks: dict[str, dict]) -> list[dict]:
    records = []
    for treatment in TREATMENTS:
        found = set()
        for source in read_jsonl(batch / treatment / "predictions.jsonl"):
            key = source["task_id"], source["memory_mode"]
            if key in found:
                raise ValueError(f"duplicate work item: {treatment}/{key}")
            found.add(key)
            if key[0] not in tasks or key[1] not in MODES:
                raise ValueError(f"unexpected work item: {treatment}/{key}")
            task = tasks[key[0]]
            rescored = strict_rescore(task, source["raw_output"])
            for field, expected in rescored.items():
                if source[field] != expected:
                    raise ValueError(f"independent score mismatch: {treatment}/{key}/{field}")
            if source["size"] != task["size"] or source["source_id"] != task["source_id"]:
                raise ValueError(f"prediction metadata mismatch: {treatment}/{key}")
            metrics = source.get("metrics", {})
            completion = source["history_trace"][-1]["completion_traces"][-1]
            group = size_group(task["size"])
            expected_difficulty = "easy" if group == "small" else "medium" if group == "medium" else "hard"
            if source["difficulty"] != expected_difficulty:
                raise ValueError("difficulty label differs from frozen definition")
            clue_band = next((tag.split(":", 1)[1] for tag in task["tags"] if tag.startswith("clue-band:")), "unspecified")
            record = {
                "treatment": treatment, "mode": source["memory_mode"],
                "task_id": task["id"], "source_id": task["source_id"], "size": task["size"],
                "size_group": group, "difficulty": expected_difficulty,
                "clues": len(task["clue_turns"]), "clue_band": clue_band,
                **rescored, "wrong": not rescored["success"],
                "final_finish_reason": completion.get("finish_reason"),
                "raw_output_characters": len(source["raw_output"]),
                "model_elapsed_seconds": metrics.get("model_elapsed_seconds"),
                "task_elapsed_seconds": metrics.get("task_elapsed_seconds"),
                "llm_calls": metrics.get("llm_calls"),
                "usage_missing_calls": metrics.get("usage_missing_calls"),
                "token_usage": metrics.get("token_usage", {}),
            }
            if record["llm_calls"] != len(task["clue_turns"]) + 1:
                raise ValueError("model-call count differs from frozen task plan")
            records.append(record)
        expected_keys = {(task_id, mode) for task_id in tasks for mode in MODES}
        if found != expected_keys or len(found) != 300:
            raise ValueError(f"{treatment}: incomplete task/mode roster")
    if len(records) != 600:
        raise ValueError("expected 600 instances")
    return records


def summarize(records: list[dict]) -> dict:
    count = len(records)
    success = sum(record["success"] for record in records)
    parsed = sum(record["parsed"] for record in records)
    cells = sum(record["correct_cells"] for record in records)
    total_cells = sum(record["total_cells"] for record in records)
    token_usage = Counter()
    for record in records:
        token_usage.update(record["token_usage"])
    low, high = wilson(success, count)
    answer = {
        "n": count, "success": success, "wrong": count - success,
        "puzzle_accuracy": success / count,
        "wilson_95_low": low, "wilson_95_high": high,
        "parsed": parsed, "parsed_wrong": parsed - success,
        "no_answer": count - parsed, "no_answer_rate": (count - parsed) / count,
        "puzzle_accuracy_given_parsed": success / parsed if parsed else None,
        "correct_cells": cells, "total_cells": total_cells,
        "cell_accuracy": cells / total_cells,
        "final_finish_reason": dict(Counter(record["final_finish_reason"] for record in records)),
        "final_length": sum(record["final_finish_reason"] == "length" for record in records),
        "llm_calls": sum(record["llm_calls"] for record in records),
        "usage_missing_calls": sum(record["usage_missing_calls"] for record in records),
        "token_usage": dict(sorted(token_usage.items())),
    }
    for field in ("model_elapsed_seconds", "task_elapsed_seconds"):
        values = [record[field] for record in records if isinstance(record[field], (int, float))]
        answer[field + "_recorded_n"] = len(values)
        answer[field + "_sum"] = sum(values)
        answer[field + "_mean"] = statistics.mean(values) if values else None
        answer[field + "_median"] = statistics.median(values) if values else None
        answer[field + "_p90"] = quantile(values, .9)
    return answer


def paired(left: list[dict], right: list[dict], seed: int, bootstrap_samples: int) -> dict:
    left_map = {record["task_id"]: record for record in left}
    right_map = {record["task_id"]: record for record in right}
    if set(left_map) != set(right_map) or len(left_map) != 100:
        raise ValueError("paired comparison does not contain identical 100-task rosters")
    ids = sorted(left_map)
    both_correct, both_wrong, right_only, left_only = [], [], [], []
    differences = []
    for task_id in ids:
        lcorrect, rcorrect = left_map[task_id]["success"], right_map[task_id]["success"]
        target = both_correct if lcorrect and rcorrect else right_only if rcorrect else left_only if lcorrect else both_wrong
        target.append(left_map[task_id]["source_id"])
        differences.append(int(rcorrect) - int(lcorrect))
    discordant = len(left_only) + len(right_only)
    probability = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(len(left_only), len(right_only)) + 1)) / (2 ** discordant)) if discordant else 1.
    random_generator = random.Random(seed)
    draws = [sum(random_generator.choices(differences, k=100)) / 100 for _ in range(bootstrap_samples)]
    return {
        "n": 100, "left_correct": sum(row["success"] for row in left),
        "right_correct": sum(row["success"] for row in right),
        "both_correct": len(both_correct), "both_wrong": len(both_wrong),
        "right_only_correct": len(right_only), "left_only_correct": len(left_only),
        "right_minus_left_accuracy": sum(differences) / 100,
        "paired_bootstrap_95_low": quantile(draws, .025),
        "paired_bootstrap_95_high": quantile(draws, .975),
        "mcnemar_exact_two_sided_p": probability,
        "right_only_correct_ids": right_only, "left_only_correct_ids": left_only,
        "both_correct_ids": both_correct, "both_wrong_ids": both_wrong,
    }


def csv_rows(path: Path, rows: list[dict]) -> None:
    flattened = []
    for original in rows:
        row = {}
        for key, value in original.items():
            if key == "token_usage":
                row.update({"tokens_" + token: amount for token, amount in value.items()})
            else:
                row[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        flattened.append(row)
    fieldnames = list(dict.fromkeys(key for row in flattened for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path, nargs="?", default=ROOT / "runs/1028_zebra_proexp_direct")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    batch = args.batch_dir.resolve()
    output = (args.output_dir or batch.with_name(batch.name + "_analysis")).resolve()
    if output == batch or output.is_relative_to(batch):
        raise ValueError("analysis output must be outside the raw experiment directory")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    roster = list(read_jsonl(batch / "history_eval.jsonl"))
    tasks = {task["id"]: task for task in roster}
    if len(tasks) != len(roster) or len(tasks) != 100:
        raise ValueError("expected 100 unique source puzzles")
    records = load_records(batch, tasks)
    indexed = {(treatment, mode): [row for row in records if row["treatment"] == treatment and row["mode"] == mode]
               for treatment in TREATMENTS for mode in MODES}
    groups = [{"treatment": treatment, "mode": mode, **summarize(indexed[treatment, mode])}
              for treatment in TREATMENTS for mode in MODES]
    by_size, by_difficulty, by_band = [], [], []
    for (treatment, mode), rows in indexed.items():
        for size in sorted({row["size"] for row in rows}, key=lambda s: tuple(map(int, s.split("*")))):
            by_size.append({"treatment": treatment, "mode": mode, "size": size,
                            **summarize([row for row in rows if row["size"] == size])})
        for category_type, categories in (("difficulty", ("easy", "medium", "hard")), ("size_group", tuple(SIZE_GROUPS))):
            for category in categories:
                by_difficulty.append({"treatment": treatment, "mode": mode, "category_type": category_type,
                                      "category": category, **summarize([row for row in rows if row[category_type] == category])})
        for band in sorted({row["clue_band"] for row in rows}):
            by_band.append({"treatment": treatment, "mode": mode, "clue_band": band,
                            **summarize([row for row in rows if row["clue_band"] == band])})
    comparisons = []
    for treatment in TREATMENTS:
        for left_mode, right_mode in combinations(MODES, 2):
            comparisons.append({"comparison_type": "history_modes", "left_treatment": treatment,
                                "left_mode": left_mode, "right_treatment": treatment, "right_mode": right_mode,
                                **paired(indexed[treatment, left_mode], indexed[treatment, right_mode],
                                         20261028 + len(comparisons), args.bootstrap_samples)})
    for mode in MODES:
        comparisons.append({"comparison_type": "thinking", "left_treatment": "thinking_off", "left_mode": mode,
                            "right_treatment": "thinking_on", "right_mode": mode,
                            **paired(indexed["thinking_off", mode], indexed["thinking_on", mode],
                                     20261028 + len(comparisons), args.bootstrap_samples)})
    running = 0.
    for position, comparison in enumerate(sorted(comparisons, key=lambda row: row["mcnemar_exact_two_sided_p"])):
        running = max(running, min(1., (len(comparisons) - position) * comparison["mcnemar_exact_two_sided_p"]))
        comparison["mcnemar_holm_adjusted_p_9_tests"] = running
    matrix = []
    by_instance = {(row["task_id"], row["treatment"], row["mode"]): row for row in records}
    for task in roster:
        row = {"task_id": task["id"], "source_id": task["source_id"], "size": task["size"],
               "size_group": size_group(task["size"]), "clues": len(task["clue_turns"]),
               "total_cells": len(task["solution"]["rows"]) * (len(task["solution"]["header"]) - 1)}
        for treatment in TREATMENTS:
            for mode in MODES:
                instance = by_instance[task["id"], treatment, mode]
                for key in ("success", "correct_cells", "parsed", "no_answer", "final_finish_reason"):
                    row[f"{treatment}__{mode}__{key}"] = instance[key]
        matrix.append(row)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(), "batch_dir": str(batch),
        "data_sha256": hashlib.sha256((batch / "history_eval.jsonl").read_bytes()).hexdigest(),
        "validation": {"unique_input_tasks": len(tasks), "instances": len(records), "independent_rescore_mismatches": 0,
                       "calls": sum(record["llm_calls"] for record in records), "identical_roster_in_all_six_groups": True,
                       "input_tasks_by_size": dict(sorted(Counter(task["size"] for task in roster).items())),
                       "input_tasks_by_size_group": dict(Counter(size_group(task["size"]) for task in roster))},
        "definitions": {
            "puzzle_accuracy": "Strictly correct complete solution / all 100 assigned puzzles, including no-answer cases.",
            "cell_accuracy": "Correct attribute cells / all assigned attribute cells, including no-answer cases; house-index column excluded.",
            "parsed": "Top-level extracted solution is a dictionary; this does not guarantee complete or valid keys.",
            "puzzle_accuracy_given_parsed": "Strictly correct puzzles / parsed puzzles, with changing denominator shown explicitly.",
            "wilson_95": "Wilson binomial interval; generalization requires assumptions beyond this fixed benchmark sample.",
            "pairwise_tests": "Exploratory exact two-sided McNemar tests; Holm correction across all nine comparisons.",
            "paired_bootstrap": f"Percentile interval, {args.bootstrap_samples} resamples of the paired 100 source tasks; fixed seeds starting 20261028.",
            "elapsed": "Model/task durations are summed across overlapping workers and are NOT experiment wall-clock time.",
            "difficulty": "Frozen labels: small=easy, medium=medium, large+x-large=hard. Actual task counts are computed from the input roster.",
            "size_subgroups": "There are 25 size cells with 3 to 5 source puzzles each; subgroup rates are descriptive and imprecise.",
        },
        "groups": groups, "by_size": by_size, "by_difficulty": by_difficulty,
        "by_clue_band": by_band, "pairwise_comparisons": comparisons,
    }
    (output / "statistics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_rows(output / "group_accuracy.csv", groups)
    csv_rows(output / "size_accuracy.csv", by_size)
    csv_rows(output / "difficulty_accuracy.csv", by_difficulty)
    csv_rows(output / "paired_task_matrix.csv", matrix)
    csv_rows(output / "pairwise_comparisons.csv", comparisons)
    csv_rows(output / "per_instance_accuracy.csv", records)
    print(json.dumps({"validation": result["validation"], "groups": groups,
                      "pairwise_comparisons": [{key: value for key, value in row.items() if not key.endswith("_ids")}
                                               for row in comparisons], "output_dir": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

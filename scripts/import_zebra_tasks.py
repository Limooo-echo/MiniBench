from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Callable, Iterable

from minibench.datasets.zebra.dataset import difficulty_for_size


PUBLIC_DATASET = "WildEval/ZebraLogic"
SUBSET = "grid_mode"
SPLIT = "test"


def is_masked_solution(solution: object) -> bool:
    if not isinstance(solution, dict):
        return True
    rows = solution.get("rows")
    if not isinstance(rows, list) or not rows:
        return True
    cells = [cell for row in rows if isinstance(row, list) for cell in row[1:]]
    return not cells or all(str(cell).strip() == "___" for cell in cells)


def convert_official_records(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for record in records:
        task_id = str(record["id"])
        solution = record.get("solution")
        if is_masked_solution(solution):
            raise ValueError(
                f"{task_id}: the selected Zebra dataset contains masked solutions; "
                "use WildEval/ZebraLogic grid_mode for a scoreable export."
            )
        converted.append(
            {
                "id": task_id,
                "size": str(record["size"]),
                "puzzle": str(record["puzzle"]),
                "solution": solution,
                "capability": "direct",
                "rule_context": None,
                "clue_turns": [],
                "tags": ["source:WildEval/ZebraLogic"],
            }
        )
    return converted


def select_smoke_records(
    records: Iterable[dict[str, Any]],
    *,
    per_difficulty: int,
    seed: int,
) -> list[dict[str, Any]]:
    if per_difficulty < 1:
        raise ValueError("per_difficulty must be positive")
    grouped = {"easy": [], "medium": [], "hard": []}
    for record in records:
        grouped[difficulty_for_size(str(record["size"]))].append(record)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for difficulty in ("easy", "medium", "hard"):
        candidates = grouped[difficulty]
        rng.shuffle(candidates)
        if len(candidates) < per_difficulty:
            raise ValueError(
                f"not enough {difficulty} records: need {per_difficulty}, "
                f"found {len(candidates)}"
            )
        selected.extend(candidates[:per_difficulty])
    return selected


def load_official_records(
    load_dataset: Callable[..., Iterable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records = load_dataset(PUBLIC_DATASET, SUBSET, split=SPLIT)
    return convert_official_records(records)


def write_records(
    records: Iterable[dict[str, Any]],
    output: str | Path,
    *,
    overwrite: bool,
) -> int:
    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite")
    materialized = list(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in materialized:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(materialized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import scoreable ZebraLogicBench grid tasks into MiniBench JSONL."
    )
    parser.add_argument("--output", type=Path, default=Path("data/zebra/tasks.jsonl"))
    parser.add_argument(
        "--smoke-per-difficulty",
        type=int,
        default=None,
        help="Select N deterministic records for each easy/medium/hard bucket.",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install project dependencies first: pip install -e .") from exc
    try:
        records = load_official_records(load_dataset)
        if args.smoke_per_difficulty is not None:
            records = select_smoke_records(
                records,
                per_difficulty=args.smoke_per_difficulty,
                seed=args.seed,
            )
        count = write_records(records, args.output, overwrite=args.overwrite)
    except (FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(f"Zebra import failed: {exc}") from exc
    print(
        json.dumps(
            {"output": str(args.output), "records": count},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

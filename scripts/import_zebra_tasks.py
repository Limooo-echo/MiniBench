from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
from typing import Any, Callable, Iterable

from minibench.datasets.zebra.dataset import ZEROEVAL_SIZE_GROUPS, difficulty_for_size


PUBLIC_DATASET = "WildEval/ZebraLogic"
SUBSET = "grid_mode"
SPLIT = "test"
DIFFICULTIES = ("easy", "medium", "hard")
CLUE_BANDS = ("low", "middle", "high")
CLUE_LINE = re.compile(r"(?m)^\s*\d+\.\s+\S")


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
    grouped = {difficulty: [] for difficulty in DIFFICULTIES}
    for record in records:
        grouped[difficulty_for_size(str(record["size"]))].append(record)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for difficulty in DIFFICULTIES:
        candidates = grouped[difficulty]
        rng.shuffle(candidates)
        if len(candidates) < per_difficulty:
            raise ValueError(
                f"not enough {difficulty} records: need {per_difficulty}, "
                f"found {len(candidates)}"
            )
        selected.extend(candidates[:per_difficulty])
    return selected


def count_clues(puzzle: str) -> int:
    """Count the numbered clue lines in an official Zebra puzzle."""
    return len(CLUE_LINE.findall(puzzle))


def _balanced_targets(labels: list[str], total: int, rng: random.Random) -> dict[str, int]:
    if total < len(labels):
        raise ValueError(f"cannot cover {len(labels)} strata with only {total} records")
    base, remainder = divmod(total, len(labels))
    shuffled = list(labels)
    rng.shuffle(shuffled)
    return {
        label: base + int(label in set(shuffled[:remainder]))
        for label in labels
    }


def _records_by_clue_band(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Split one grid size into within-size clue-count thirds."""
    ordered = sorted(records, key=lambda item: (count_clues(str(item["puzzle"])), str(item["id"])))
    banded = {band: [] for band in CLUE_BANDS}
    for index, record in enumerate(ordered):
        band_index = min(len(CLUE_BANDS) - 1, index * len(CLUE_BANDS) // len(ordered))
        banded[CLUE_BANDS[band_index]].append(record)
    return banded


def select_evaluation_records(
    records: Iterable[dict[str, Any]],
    *,
    per_difficulty: int,
    seed: int,
    exclude_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Select a reproducible, size-balanced evaluation set.

    Each difficulty covers every official grid size as evenly as possible. Within
    each size, records are ranked by clue count and divided into low/middle/high
    thirds; the final selection balances those three bands as well.
    """
    excluded = set(exclude_ids)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        difficulty: {} for difficulty in DIFFICULTIES
    }
    for record in records:
        if str(record["id"]) in excluded:
            continue
        difficulty = difficulty_for_size(str(record["size"]))
        grouped[difficulty].setdefault(str(record["size"]), []).append(record)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for difficulty in DIFFICULTIES:
        size_groups = grouped[difficulty]
        sizes = sorted(size_groups)
        if not sizes:
            raise ValueError(f"no {difficulty} records available")
        size_targets = _balanced_targets(sizes, per_difficulty, rng)
        band_targets = _balanced_targets(list(CLUE_BANDS), per_difficulty, rng)
        banded = {
            size: _records_by_clue_band(size_groups[size])
            for size in sizes
        }

        size_order = list(sizes)
        rng.shuffle(size_order)
        allocation = {
            size: {band: 0 for band in CLUE_BANDS}
            for size in sizes
        }
        remaining = dict(band_targets)
        for size in size_order:
            used_bands: set[str] = set()
            for _ in range(size_targets[size]):
                available = [
                    band
                    for band in CLUE_BANDS
                    if remaining[band] > 0
                    and len(banded[size][band]) > allocation[size][band]
                ]
                fresh = [band for band in available if band not in used_bands]
                choices = fresh or available
                if not choices:
                    raise ValueError(
                        f"cannot satisfy clue-band quotas for {difficulty}/{size}"
                    )
                largest_need = max(remaining[band] for band in choices)
                tied = sorted(
                    band for band in choices if remaining[band] == largest_need
                )
                band = rng.choice(tied)
                allocation[size][band] += 1
                remaining[band] -= 1
                used_bands.add(band)

        if any(remaining.values()):
            raise ValueError(f"could not balance clue bands for {difficulty}: {remaining}")

        for size in sizes:
            for band in CLUE_BANDS:
                candidates = sorted(banded[size][band], key=lambda item: str(item["id"]))
                rng.shuffle(candidates)
                for candidate in candidates[: allocation[size][band]]:
                    chosen = dict(candidate)
                    tags = list(chosen.get("tags", []))
                    for tag in (
                        "split:test",
                        "benchmark:eval",
                        "sampling:balanced-size-clue-band",
                        f"selection-seed:{seed}",
                        f"clue-band:{band}",
                    ):
                        if tag not in tags:
                            tags.append(tag)
                    chosen["tags"] = tags
                    selected.append(chosen)

    return selected


def load_official_records(
    load_dataset: Callable[..., Iterable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records = load_dataset(PUBLIC_DATASET, SUBSET, split=SPLIT)
    return convert_official_records(records)


def load_local_parquet_records(
    read_table: Callable[[str | Path], Any],
    source_parquet: str | Path,
) -> list[dict[str, Any]]:
    source_path = Path(source_parquet).resolve()
    if not source_path.is_file():
        raise ValueError(f"local Parquet source does not exist: {source_path}")
    table = read_table(source_path)
    return convert_official_records(table.to_pylist())


def load_excluded_ids(paths: Iterable[str | Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        task_path = Path(path)
        with task_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    task_id = raw["id"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"{task_path}:{line_number}: cannot read task id"
                    ) from exc
                excluded.add(str(task_id))
    return excluded


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
    parser.add_argument("--output", type=Path, default=None)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--smoke-per-difficulty",
        type=int,
        default=None,
        help="Select N deterministic records for each easy/medium/hard bucket.",
    )
    selection.add_argument(
        "--evaluation-per-difficulty",
        type=int,
        default=None,
        help=(
            "Select N records per difficulty, balanced over grid size and "
            "within-size clue-count thirds."
        ),
    )
    parser.add_argument(
        "--source-parquet",
        type=Path,
        default=None,
        help="Read an already-downloaded grid_mode Parquet file instead of Hugging Face.",
    )
    parser.add_argument(
        "--exclude-task-file",
        type=Path,
        action="append",
        default=[],
        help="Exclude ids found in a JSONL task file; may be passed more than once.",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.source_parquet is None:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise SystemExit(
                    "Install project dependencies first: pip install -e ."
                ) from exc
            records = load_official_records(load_dataset)
        else:
            try:
                from pyarrow.parquet import read_table
            except ImportError as exc:
                raise SystemExit(
                    "Reading --source-parquet requires pyarrow; install project dependencies first."
                ) from exc
            records = load_local_parquet_records(read_table, args.source_parquet)
        if args.smoke_per_difficulty is not None:
            records = select_smoke_records(
                records,
                per_difficulty=args.smoke_per_difficulty,
                seed=args.seed,
            )
        elif args.evaluation_per_difficulty is not None:
            records = select_evaluation_records(
                records,
                per_difficulty=args.evaluation_per_difficulty,
                seed=args.seed,
                exclude_ids=load_excluded_ids(args.exclude_task_file),
            )
        output = args.output or Path(
            "data/zebra/eval.jsonl"
            if args.evaluation_per_difficulty is not None
            else "data/zebra/tasks.jsonl"
        )
        count = write_records(records, output, overwrite=args.overwrite)
    except (FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(f"Zebra import failed: {exc}") from exc
    print(
        json.dumps(
            {"output": str(output), "records": count, "seed": args.seed},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

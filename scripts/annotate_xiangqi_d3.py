"""Annotate D3 with exact mate sets and model-independent structural strata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minibench.datasets.xiangqi.mate_in_one import d3_position_features
from minibench.datasets.xiangqi.schema import fen_to_board

WEIGHTS = {"choice_pressure": 0.50, "near_miss_rate": 0.30, "state_load": 0.20}
LABEL_COUNTS = (("easy", 80), ("medium", 85), ("hard", 85))


def _percentile_ranks(values: list[float]) -> list[float]:
    """Return tie-aware percentile ranks without assuming a feature scale."""
    if len(values) == 1:
        return [0.0]
    ranks = []
    for value in values:
        below = sum(other < value for other in values)
        equal = sum(other == value for other in values)
        ranks.append((below + 0.5 * (equal - 1)) / (len(values) - 1))
    return ranks


def annotate(records: list[dict]) -> list[dict]:
    if len(records) != sum(count for _, count in LABEL_COUNTS):
        raise ValueError("D3 label counts must cover the complete 250-record corpus")
    for record in records:
        board, color = fen_to_board(record["fen"])
        features = d3_position_features(board, 1 if color == "red" else -1)
        if features["mate_move_count"] < 1:
            raise ValueError(f"{record['id']}: not a mate-in-one position")
        # D3 consumes exactly one agent action. Keep the persisted horizon in
        # sync with the evaluator rather than carrying the legacy four-ply cap.
        record["max_plies"] = 1
        legal = int(features["legal_move_count"])
        mates = int(features["mate_move_count"])
        near_misses = int(features["non_mating_check_count"])
        record["d3_analysis"] = {
            "version": "d3-structural-v2",
            **features,
            "difficulty_factors": {
                "choice_pressure": (legal - mates) / legal,
                "near_miss_rate": near_misses / (near_misses + mates),
                "state_load": int(features["piece_count"]),
            },
        }
    for key in WEIGHTS:
        values = [float(record["d3_analysis"]["difficulty_factors"][key]) for record in records]
        for record, rank in zip(records, _percentile_ranks(values)):
            record["d3_analysis"]["difficulty_factors"][key + "_percentile"] = round(rank, 6)
    for record in records:
        factors = record["d3_analysis"]["difficulty_factors"]
        record["d3_analysis"]["difficulty_score"] = round(sum(
            WEIGHTS[key] * factors[key + "_percentile"] for key in WEIGHTS
        ), 6)
    ranked = sorted(records, key=lambda record: (record["d3_analysis"]["difficulty_score"], record["id"]))
    offset = 0
    for label, count in LABEL_COUNTS:
        for record in ranked[offset:offset + count]:
            record["difficulty"] = label
            record["tags"] = sorted({
                tag for tag in record.get("tags", [])
                if not tag.startswith("difficulty:")
            } | {f"difficulty:{label}", "d3", "mate-in-one"})
        offset += count
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/xiangqi/mate_in_one/tasks.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()
    if args.in_place == (args.output is not None):
        parser.error("choose exactly one of --in-place or --output")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    annotate(records)
    destination = args.input if args.in_place else args.output
    destination.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    print(json.dumps({"total": len(records), "difficulty": {label: sum(record["difficulty"] == label for record in records) for label, _ in LABEL_COUNTS}}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

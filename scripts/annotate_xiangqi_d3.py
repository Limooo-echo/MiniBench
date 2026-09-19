"""Annotate D3 with exact mate sets and model-independent structural strata."""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minibench.datasets.xiangqi.mate_in_one import d3_position_features
from minibench.datasets.xiangqi.schema import fen_to_board
from minibench.datasets.xiangqi import reference
from minibench.datasets.xiangqi.validation import validate_position, compare_standard_legal_moves, validate_pv, VALIDATION_VERSION

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
    baseline_path = ROOT / "data/xiangqi/legacy/xiangqi-corpus-rebuild-2026-08-31/mate_in_one/tasks.jsonl"
    baseline = {r["id"]: r for r in (json.loads(line) for line in baseline_path.read_text(encoding="utf-8").splitlines())} if baseline_path.exists() else {}
    for record in records:
        identity = hashlib.sha256((record["fen"] + "|" + record["goal"]).encode()).hexdigest()
        previous = baseline.get(record["id"])
        changed = (record.get("identity_sha256") not in (None, identity)
                   or (previous is not None and (previous["fen"],previous["goal"]) != (record["fen"],record["goal"])))
        if changed:
            record["replaces_id"] = record["id"]
            record["id"] = "xiangqi-mate-in-one-r1-" + identity[:16]
            record.pop("source_id", None)
        record["identity_sha256"] = identity
        board, color = fen_to_board(record["fen"])
        side = 1 if color == "red" else -1
        reasons = validate_position(board, side)
        comparison = compare_standard_legal_moves(board, side)
        if reasons or not comparison["valid"]:
            raise ValueError(f"{record['id']}: {reasons + comparison['reasons']}")
        independent_mates = []
        for move in reference.legal_moves(board, side):
            child = reference.apply_move(board, move)
            child_comparison = compare_standard_legal_moves(child, -side)
            if not child_comparison["valid"]:
                raise ValueError(f"{record['id']} after {move.to_uci()}: {child_comparison['reasons']}")
            if reference.terminal_status(child, -side) == "checkmate":
                proof = validate_pv(record["fen"], [move.to_uci()])
                if not proof["valid"]:
                    raise ValueError(f"{record['id']}: {proof['reasons']}")
                independent_mates.append(move.to_uci())
        features = d3_position_features(board, side)
        if sorted(independent_mates) != sorted(features["mate_moves_uci"]):
            raise ValueError(f"{record['id']}: independent mate sets disagree")
        record["release_id"] = "xiangqi-2026-09-19-r1"
        record["source_id"] = record.get("source_id", record["id"])
        record["validation"] = {"version": VALIDATION_VERSION, "position_valid": True,
                                "standard_moves": "cchess==1.25.5 plus independent reference",
                                "all_mate_answers_verified": True}
        record["oracle"]["best_move_uci"] = sorted(independent_mates)[0] if independent_mates else None
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

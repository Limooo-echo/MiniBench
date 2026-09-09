from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minibench.datasets.xiangqi.engines.pikafish import (
    PikafishEngine,
    pikafish_fingerprint,
)
from minibench.datasets.xiangqi.schema import fen_to_board, validate_record
from minibench.datasets.xiangqi.variants.board import VariantBoard


SOURCE_REPOSITORY = "https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset"
SOURCE_REVISION = "368a47a947773dd8692c026e286dd19b6277b993"
SOURCE_LICENSE = "CC-BY-4.0"
TARGETS = {"short": 80, "medium": 80, "long": 90}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mate_band(mate: int) -> str | None:
    if 2 <= mate <= 3:
        return "short"
    if 4 <= mate <= 5:
        return "medium"
    if 6 <= mate <= 12:
        return "long"
    return None


def interleave_by_mate(items: list[dict], rng: random.Random) -> list[dict]:
    groups: dict[int, list[dict]] = {}
    for item in items:
        groups.setdefault(int(item["screening"]["mate_in_moves"]), []).append(item)
    for group in groups.values():
        rng.shuffle(group)
    ordered = []
    while any(groups.values()):
        for mate in sorted(groups):
            if groups[mate]:
                ordered.append(groups[mate].pop())
    return ordered


def pv_is_legal(fen: str, pv: list[str]) -> bool:
    board, color = fen_to_board(fen)
    position = VariantBoard(board, [])
    side = 1 if color == "red" else -1
    for uci in pv:
        move = next((item for item in position.legal_moves(side) if item.to_uci() == uci), None)
        if move is None:
            return False
        position.apply(move)
        side = -side
        if position.find_general(side) is None:
            break
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--depth-a", type=int, default=16)
    parser.add_argument("--depth-b", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--pikafish",
        type=Path,
        default=None,
        help="Pikafish executable; otherwise use the standard resolver.",
    )
    args = parser.parse_args()

    dedup: dict[str, dict] = {}
    for path in args.pool:
        for item in read_jsonl(path):
            current = dedup.get(item["fen"])
            depth = int(item.get("screening", {}).get("depth") or 0)
            current_depth = int((current or {}).get("screening", {}).get("depth") or 0)
            if current is None or depth > current_depth:
                dedup[item["fen"]] = item

    candidates: dict[str, list[dict]] = {band: [] for band in TARGETS}
    for item in dedup.values():
        band = mate_band(int(item["screening"]["mate_in_moves"]))
        if band is not None:
            candidates[band].append(item)
    rng = random.Random(args.seed)
    for band, items in candidates.items():
        items.sort(key=lambda item: (item["source_file"], int(item["ply_index"]), item["fen"]))
        candidates[band] = interleave_by_mate(items, rng)

    from minibench.datasets.xiangqi.engines.pikafish import (
        resolve_pikafish_executable,
    )

    executable = resolve_pikafish_executable(args.pikafish, start_dir=ROOT)
    verified: dict[str, list[dict]] = {band: [] for band in TARGETS}
    attempted_fens: set[str] = set()
    used_sources: set[str] = set()
    failures = Counter()
    validations: list[dict] = []

    def try_candidate(engine: PikafishEngine, item: dict, band: str, *, allow_source_reuse: bool) -> bool:
        if item["fen"] in attempted_fens:
            return False
        if not allow_source_reuse and item["source_file"] in used_sources:
            return False
        attempted_fens.add(item["fen"])
        try:
            first = engine.analyze_fen(item["fen"], depth=args.depth_a)
            second = engine.analyze_fen(item["fen"], depth=args.depth_b)
        except Exception:
            failures["engine_error"] += 1
            return False
        if first.score_kind != "mate" or second.score_kind != "mate":
            failures["not_mate"] += 1
            return False
        if first.score != second.score:
            failures["depth_disagreement"] += 1
            return False
        mate_moves = int(second.score)
        if mate_band(mate_moves) != band:
            failures["bucket_changed"] += 1
            return False
        if not second.pv or second.pv[0] != second.bestmove or not pv_is_legal(item["fen"], list(second.pv)):
            failures["invalid_pv"] += 1
            return False
        payload = dict(item)
        payload["verified"] = {
            "mate_in_moves": mate_moves,
            "mate_in_plies": 2 * mate_moves - 1,
            "best_move_uci": second.bestmove,
            "pv": list(second.pv),
            "depth_a": first.depth,
            "depth_b": second.depth,
            "source_reused": item["source_file"] in used_sources,
        }
        verified[band].append(payload)
        used_sources.add(item["source_file"])
        validations.append({
            "fen": item["fen"],
            "source_file": item["source_file"],
            "source_ply": item["ply_index"],
            "mate_in_moves": mate_moves,
            "mate_in_plies": 2 * mate_moves - 1,
            "depth_a": first.depth,
            "depth_b": second.depth,
            "best_move_uci": second.bestmove,
            "pv_legal": True,
            "source_reused": payload["verified"]["source_reused"],
        })
        total = sum(len(items) for items in verified.values())
        if total % 10 == 0:
            print(
                f"verified={total}/250 buckets={ {key: len(value) for key, value in verified.items()} } "
                f"failures={dict(failures)}",
                file=sys.stderr,
                flush=True,
            )
        return True

    with PikafishEngine(executable, timeout=args.timeout) as engine:
        # Rare long buckets go first so source uniqueness is spent where it matters.
        for band in ("long", "medium", "short"):
            target = TARGETS[band]
            for item in candidates[band]:
                if len(verified[band]) >= target:
                    break
                try_candidate(engine, item, band, allow_source_reuse=False)
        # If necessary, allow a different position from an already used source game.
        for band in ("long", "medium", "short"):
            target = TARGETS[band]
            for item in candidates[band]:
                if len(verified[band]) >= target:
                    break
                try_candidate(engine, item, band, allow_source_reuse=True)

    shortages = {band: target - len(verified[band]) for band, target in TARGETS.items() if len(verified[band]) < target}
    if shortages:
        raise RuntimeError(f"insufficient verified H2 candidates: {shortages}; failures={dict(failures)}")

    records = []
    for band in ("short", "medium", "long"):
        for item in verified[band]:
            mate_moves = item["verified"]["mate_in_moves"]
            difficulty = mate_band(mate_moves)
            board, color = fen_to_board(item["fen"])
            record = {
                "schema_version": 2,
                "id": f"xiangqi-history-{len(records) + 1:04d}",
                "family": "xiangqi-history",
                "fen": item["fen"],
                "agent_color": color,
                "goal": "checkmate",
                # One extra agent turn beyond the shortest engine line.
                "max_plies": 2 * mate_moves + 1,
                "difficulty": difficulty,
                "piece_count": sum(value != 0 for row in board for value in row),
                "oracle": {
                    "best_move_uci": item["verified"]["best_move_uci"],
                    "mate_in_plies": 2 * mate_moves - 1,
                    "evaluation_cp": None,
                },
                "tags": sorted({
                    "ccpd",
                    f"mate-band:{difficulty}",
                    f"mate-in-moves:{mate_moves}",
                    f"source-kind:{item['source_kind']}",
                }),
                "source": item.get("source") or {
                    "repository": SOURCE_REPOSITORY,
                    "revision": SOURCE_REVISION,
                    "license": SOURCE_LICENSE,
                    "file": item["source_file"],
                    "ply_index": item["ply_index"],
                    "game_plies": item["game_plies"],
                    "game_result": item["game_result"],
                },
                "h2_analysis": {
                    "version": "h2-pikafish-double-depth-v1",
                    "mate_in_moves": mate_moves,
                    "shortest_mate_plies": 2 * mate_moves - 1,
                    "verification_depths": [args.depth_a, args.depth_b],
                    "principal_variation_uci": item["verified"]["pv"],
                    "source_reused": item["verified"]["source_reused"],
                },
            }
            validate_record(record, expected_family="xiangqi-history")
            records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    report = {
        "total": len(records),
        "difficulty": dict(sorted(Counter(record["difficulty"] for record in records).items())),
        "mate_in_moves": dict(sorted(Counter(record["h2_analysis"]["mate_in_moves"] for record in records).items())),
        "unique_fens": len({record["fen"] for record in records}),
        "unique_source_games": len({record["source"]["file"] for record in records}),
        "reused_source_positions": sum(record["h2_analysis"]["source_reused"] for record in records),
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_license": SOURCE_LICENSE,
        "pikafish": pikafish_fingerprint(executable),
        "verification_depths": [args.depth_a, args.depth_b],
        "failures": dict(sorted(failures.items())),
        "validations": validations,
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "validations"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

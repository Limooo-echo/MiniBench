"""Mine C2 exclusively from legally replayed CCPD positions; never pad shortages."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from minibench.core.checkpoint import atomic_write_json
from minibench.datasets.xiangqi.schema import RULESETS, fen_to_board, internal_rules
from minibench.datasets.xiangqi.variants.board import VariantBoard
from minibench.datasets.xiangqi.variants.rules import Rule
from minibench.datasets.xiangqi.variants.search import score_moves
from minibench.datasets.xiangqi import reference
from minibench.datasets.xiangqi.validation import validate_position, compare_standard_legal_moves

FOCI = ("horse-no-leg-block", "chariot-no-center", "soldier-free-retreat")
RULES = {name: [Rule.from_dict(item) for item in internal_rules({"ruleset": name})] for name in RULESETS}
SOURCE_REVISION = "368a47a947773dd8692c026e286dd19b6277b993"


def analyse(board, ruleset):
    position = VariantBoard(board, RULES[ruleset])
    # The side that just moved cannot already be in check under this condition.
    if position._is_in_check(-1):
        return None
    scored = score_moves(position, 1, 3, 1)
    if len(scored) < 2 or scored[0][1] - scored[1][1] < .5:
        return None
    return {"legal": len(scored), "best": scored[0][0].to_uci(), "score": scored[0][1],
            "margin": scored[0][1] - scored[1][1],
            "scores": {move.to_uci(): value for move, value in scored}}


def screen(payload):
    item, needed, need_control = payload
    board, color = fen_to_board(item["fen"])
    if color != "red" or validate_position(board, 1):
        return None
    standard = analyse(board, "standard")
    if standard is None:
        return None
    analysis = {"standard": standard}
    # Reject uninformative candidates before the remaining conditions; acceptance
    # still requires exact unique-optimum checks in all four conditions.
    for ruleset in needed:
        value = analyse(board, ruleset)
        if value is None:
            return None
        analysis[ruleset] = value
    if not need_control and not any(analysis[name]["best"] != standard["best"] for name in needed):
        return None
    for ruleset in RULESETS:
        if ruleset in analysis: continue
        value = analyse(board, ruleset)
        if value is None: return None
        analysis[ruleset] = value
    changed = [focus for focus in FOCI if analysis[focus]["best"] != standard["best"]]
    return {"board": board, "source": item, "analysis": analysis, "changed": changed}


def independently_verify(item):
    board = item["board"]
    standard = compare_standard_legal_moves(board, 1)
    if not standard["valid"]:
        raise ValueError(standard["reasons"])
    for name in RULESETS:
        scores = {move.to_uci(): value for move, value in reference.score_moves(board, 1, 3, 1, RULES[name])}
        expected = item["analysis"][name]["scores"]
        if scores.keys() != expected.keys() or any(abs(scores[move] - expected[move]) > 1e-6 for move in scores):
            raise ValueError(f"independent depth-3 scores disagree: {name}")
    item["verification"] = {"standard": "cchess==1.25.5 plus independent reference",
                            "all_rules": "independent unpruned depth-3 search", "tolerance": 1e-6}


class InvalidSource(ValueError):
    pass


def verify_source(item, source_root):
    from extract_ccpd_positions import replay
    source = item["source"]
    path = source_root / source["source_file"]
    try:
        positions, _ = replay(path)
    except (ValueError, OSError) as exc:
        raise InvalidSource(str(exc)) from exc
    ply = int(source["ply_index"])
    if positions[ply]["fen"] != source["fen"]:
        raise InvalidSource("source replay FEN mismatch")
    # Verify every edge leading to the selected position, including the source start.
    for index, position in enumerate(positions[:ply+1]):
        board, color = fen_to_board(position["fen"])
        side = 1 if color == "red" else -1
        reasons = validate_position(board, side)
        if reasons:
            raise InvalidSource(f"source ply {index}: {reasons}")
        compared = compare_standard_legal_moves(board, side)
        if not compared["valid"]:
            raise ValueError(f"source ply {index}: {compared['reasons']}")
        if index < ply and position["played_move_uci"] not in compared["move_sets"]["reference"]:
            raise InvalidSource(f"source illegal move at ply {index}")
    source["repository"] = "https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset"
    source["revision"] = SOURCE_REVISION
    source["license"] = "CC-BY-4.0"
    source["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    source["replayed_plies_verified"] = ply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=10000)
    parser.add_argument("--max-pieces", type=int, default=18)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--revalidate-resume", action="store_true", help="Recheck saved candidates after generator/rule code changes")
    args = parser.parse_args()
    dedup = {}
    for line in args.positions.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        if row["side_to_move"] != "red": continue
        board, _ = fen_to_board(row["fen"])
        if sum(bool(x) for rank in board for x in rank) > args.max_pieces: continue
        key = row["fen"].split()[0]
        dedup.setdefault(key, row)
    candidates = sorted(dedup.values(), key=lambda r: (r["source_file"], r["ply_index"], r["fen"]))
    random.Random(args.seed).shuffle(candidates)
    candidates = candidates[:args.max_candidates]
    accepted = {focus: [] for focus in FOCI}
    accepted["standard-controls"] = []
    used, rejected = set(), []
    report = {"seed": args.seed, "depth": 3, "unique_margin_min": .5, "tolerance": 1e-6,
              "positions_sha256": hashlib.sha256(args.positions.read_bytes()).hexdigest(),
              "source_revision": SOURCE_REVISION, "available_unique": len(dedup),
              "max_pieces": args.max_pieces, "candidate_limit": args.max_candidates,
              "rejections": rejected}
    report["code_sha256"] = {relative: hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() for relative in (
        "scripts/generate_xiangqi_c2_candidates.py", "scripts/extract_ccpd_positions.py",
        "src/minibench/datasets/xiangqi/variants/board.py", "src/minibench/datasets/xiangqi/variants/search.py",
        "src/minibench/datasets/xiangqi/variants/rules.py", "src/minibench/datasets/xiangqi/reference.py",
        "src/minibench/datasets/xiangqi/validation.py")}
    state_path = args.output.with_suffix(".state.json")
    resumed_scanned = 0
    if args.resume:
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            previous, accepted = state["report"], state["candidates"]
        else:
            if not args.revalidate_resume:
                raise ValueError("legacy checkpoint requires --revalidate-resume")
            previous = json.loads(args.report.read_text(encoding="utf-8"))
            accepted = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("blocked_disagreement"):
            raise ValueError("independent disagreement blocked this run; repair and regenerate into new output paths")
        if previous["counts"] != {k:len(v) for k,v in accepted.items()}:
            raise ValueError("inconsistent checkpoint counts; refuse to skip unsaved candidates")
        if previous.get("code_sha256") != report["code_sha256"] and not args.revalidate_resume:
            raise ValueError("code changed; use --revalidate-resume to recheck every saved candidate")
        for key in ("seed", "positions_sha256", "max_pieces", "candidate_limit"):
            if report[key] != previous[key]:
                raise ValueError(f"cannot resume with changed {key}")
        if args.revalidate_resume:
            for focus, group in accepted.items():
                for item in group:
                    verify_source(item, args.source)
                    if focus == "standard-controls":
                        expected = item["analysis"]["scores"]
                        actual = {m.to_uci():v for m,v in reference.score_moves(item["board"],1,3)}
                        if actual != expected: raise ValueError("saved control failed independent revalidation")
                    else:
                        independently_verify(item)
        used = {tuple(map(tuple, item["board"])) for group in accepted.values() for item in group}
        rejected[:] = previous["rejections"]
        resumed_scanned = previous["scanned"]
    elif args.output.exists() or args.report.exists():
        raise FileExistsError("candidate output already exists; use --resume or a new path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def checkpoint(scanned):
        report.update(scanned=scanned, counts={k: len(v) for k,v in accepted.items()},
                      ready=not report.get("blocked_disagreement", False) and all(len(accepted[k]) == (10 if k == "standard-controls" else 20) for k in accepted))
        # This single atomic object is the resume source of truth. Human-facing
        # report/candidate projections can be reconstructed if interrupted.
        atomic_write_json(state_path, {"report":report,"candidates":accepted})
        atomic_write_json(args.output, accepted)
        atomic_write_json(args.report, report)
    if all(len(accepted[k]) == (10 if k == "standard-controls" else 20) for k in accepted):
        checkpoint(resumed_scanned)
        return 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # Bounded batches keep early completion from scheduling the whole corpus.
        scanned = resumed_scanned
        for offset in range(resumed_scanned, len(candidates), args.workers * 2):
            needed = tuple(k for k in FOCI if len(accepted[k]) < 20)
            need_control = len(accepted["standard-controls"]) < 10
            batch = [(r, needed, need_control) for r in candidates[offset:offset+args.workers*2]]
            for item in pool.map(screen, batch):
                scanned += 1
                if item is not None:
                    focus = next((k for k in FOCI if k in item["changed"] and len(accepted[k]) < 20), None)
                    if focus is None and len(accepted["standard-controls"]) < 10:
                        focus = "standard-controls"
                    if focus:
                        key = tuple(tuple(row) for row in item["board"])
                        if key not in used:
                            try:
                                verify_source(item, args.source)
                                independently_verify(item)
                            except InvalidSource as exc:
                                rejected.append({"source": item["source"], "reason": str(exc), "stage": "source_replay"})
                                checkpoint(scanned)
                                continue
                            except ValueError as exc:
                                # A rules disagreement blocks publication, even if another candidate could fill its place.
                                rejected.append({"source": item["source"], "reason": str(exc), "stage": "independent_disagreement"})
                                report["blocked_disagreement"] = True
                                checkpoint(scanned)
                                raise
                            if focus == "standard-controls": item["analysis"] = item["analysis"]["standard"]
                            accepted[focus].append(item)
                            used.add(key)
                            print(json.dumps({"scanned":scanned,"accepted": {k:len(v) for k,v in accepted.items()}}), flush=True)
                            checkpoint(scanned)
                if all(len(accepted[k]) == (10 if k == "standard-controls" else 20) for k in accepted):
                    checkpoint(scanned)
                    return 0
            if scanned % 100 < args.workers*2:
                print(f"scanned={scanned}/{len(candidates)} counts={ {k:len(v) for k,v in accepted.items()} }", flush=True)
                checkpoint(scanned)
    checkpoint(scanned)
    raise SystemExit("insufficient independently verified C2 candidates; release blocked; see report")

if __name__ == "__main__":
    raise SystemExit(main())

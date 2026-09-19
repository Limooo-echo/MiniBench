"""Audit and rebuild H2 with two independently replayed complete mate lines.

The engine scores are reference distances, not a proof of a shortest forced mate.
Only a complete legal PV ending in checkmate passes. Every source game is used
at most once, and failed candidates remain inspectable in the validation report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from minibench.core.checkpoint import atomic_write_json, atomic_write_jsonl, fingerprint_payload, sha256_file
from minibench.datasets.xiangqi.engines.pikafish import PikafishEngine, pikafish_fingerprint, resolve_pikafish_executable
from minibench.datasets.xiangqi.schema import board_to_fen, fen_to_board, validate_record
from minibench.datasets.xiangqi.validation import VALIDATION_VERSION, calibrate_cchess, validate_position, validate_pv

SOURCE_REPOSITORY = "https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset"
SOURCE_REVISION = "368a47a947773dd8692c026e286dd19b6277b993"
SOURCE_LICENSE = "CC-BY-4.0"
TARGETS = {"short": 80, "medium": 80, "long": 90}
GENERATOR_VERSION = "h2-pikafish-double-depth-v2"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pool_fingerprints(pools: list[Path]) -> list[dict]:
    """Snapshot immutable candidate inputs separately from engine proof cache keys."""
    return [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in pools]


def assert_frozen_pools(pools: list[Path], expected: list[dict]) -> None:
    if pool_fingerprints(pools) != expected:
        raise RuntimeError("candidate pool changed during generation; output not published")


def assert_no_cross_family_collisions(records: list[dict], excluded_fens: set[str]) -> None:
    collisions = [record["id"] for record in records if record["fen"] in excluded_fens]
    if collisions:
        raise RuntimeError(f"cross_family_fen_overlap; output not published: {collisions}")


def cached_candidate_usable(result: dict | None, needed_bands: set[str]) -> bool:
    if not result:
        return False
    # Complete two-depth evidence is independent of the bands currently needed.
    if len(result.get("runs", [])) == 2:
        return True
    first = (result.get("runs") or [result.get("screening", {})])[0]
    if first.get("score_kind") == "mate" and mate_band(first.get("score", 0)) in needed_bands:
        return False
    return True


def mate_band(mate: int) -> str | None:
    if 2 <= mate <= 3:
        return "short"
    if 4 <= mate <= 5:
        return "medium"
    if 6 <= mate <= 12:
        return "long"
    return None


def canonical_fen(fen: str) -> str:
    board, color = fen_to_board(fen)
    return board_to_fen(board, active_color=color)


def packed_analysis(result, depth: int) -> dict:
    scoring_lines = [line for line in result.info_lines if " score " in line]
    last = scoring_lines[-1] if scoring_lines else ""
    return {"requested_depth": depth, "actual_depth": result.depth,
            "score_kind": result.score_kind, "score": result.score,
            "score_is_bound": "lowerbound" in last or "upperbound" in last,
            "best_move_uci": result.bestmove, "principal_variation_uci": list(result.pv)}


def assess_depth_pair(fen: str, runs: list[dict]) -> list[str]:
    """Validate the two full PVs and the distance claimed by each engine result."""
    reasons = []
    for index, run in enumerate(runs):
        label = f"depth_{run['requested_depth']}"
        if run["actual_depth"] is None or run["actual_depth"] < run["requested_depth"]:
            reasons.append(label + ":insufficient_search_depth")
        if run["score_is_bound"]:
            reasons.append(label + ":bound_only_score")
        if run["score_kind"] != "mate" or run["score"] <= 0:
            reasons.append(label + ":not_positive_mate")
        pv = run["principal_variation_uci"]
        validation = validate_pv(fen, pv, require_checkmate=True)
        run["pv_validation"] = validation
        if not validation["valid"]:
            reasons.extend(label + ":" + value for value in validation["reasons"])
        if not pv or pv[0] != run["best_move_uci"]:
            reasons.append(label + ":bestmove_pv_mismatch")
        if run["score_kind"] == "mate" and run["score"] > 0:
            expected = 2 * run["score"] - 1
            if len(pv) != expected or validation["plies"] != expected:
                reasons.append(label + ":reference_length_mismatch")
    if len(runs) != 2:
        reasons.append("missing_depth_result")
    elif (runs[0]["score_kind"], runs[0]["score"]) != (runs[1]["score_kind"], runs[1]["score"]):
        reasons.append("depth_score_disagreement")
    elif mate_band(runs[1]["score"]) is None:
        reasons.append("reference_distance_outside_h2_bands")
    return reasons


class Verifier:
    def __init__(self, executable: Path, depths: tuple[int, int], timeout: float):
        self.executable, self.depths, self.timeout = executable, depths, timeout
        self.local = threading.local()
        self.engines = []
        self.lock = threading.Lock()

    def engine(self):
        if not hasattr(self.local, "engine"):
            self.local.engine = PikafishEngine(self.executable, timeout=self.timeout)
            self.local.engine.start()
            with self.lock:
                self.engines.append(self.local.engine)
        return self.local.engine

    def close(self):
        for engine in self.engines:
            engine.close()

    def check(self, item: dict, needed_bands: set[str] | None = None) -> dict:
        result = {"fen": item["fen"], "source_file": item["source_file"],
                  "source_ply": item["ply_index"], "valid": False, "reasons": [], "runs": []}
        try:
            board, color = fen_to_board(item["fen"])
            result["reasons"] = validate_position(board, 1 if color == "red" else -1)
            if result["reasons"]:
                return result
            engine = self.engine()
            if needed_bands is not None:
                screen = packed_analysis(engine.analyze_fen(item["fen"], depth=8), 8)
                result["screening"] = screen
                # A depth-8 search often has no mate score for an 11+ ply line.
                # Centipawn advantage is only a candidate filter; acceptance
                # still requires both complete depth-16/20 checkmate lines.
                promising_mate = (screen["score_kind"] == "mate" and screen["score"] > 0
                                  and mate_band(screen["score"]) in needed_bands)
                promising_advantage = (screen["score_kind"] == "cp" and screen["score"] >= 300)
                if not (promising_mate or promising_advantage):
                    result["reasons"] = ["screening_not_promising_for_required_bands"]
                    return result
            first = packed_analysis(engine.analyze_fen(item["fen"], depth=self.depths[0]), self.depths[0])
            result["runs"] = [first]
            if needed_bands is not None and (first["score_kind"] != "mate"
                    or mate_band(first["score"]) not in needed_bands):
                result["reasons"] = ["depth_16_candidate_not_in_required_mate_band"]
                return result
            result["runs"].append(packed_analysis(
                engine.analyze_fen(item["fen"], depth=self.depths[1]), self.depths[1]))
            result["reasons"] = assess_depth_pair(item["fen"], result["runs"])
            result["valid"] = not result["reasons"]
            if result["valid"]:
                result["reference_mate_moves"] = result["runs"][-1]["score"]
                result["reference_mate_plies"] = 2 * result["reference_mate_moves"] - 1
                result["band"] = mate_band(result["reference_mate_moves"])
        except Exception as exc:
            result["reasons"].append(f"validation_error:{type(exc).__name__}:{exc}")
        return result


def existing_item(record: dict) -> dict:
    source = record["source"]
    return {"fen": record["fen"], "source_file": source["file"], "ply_index": source["ply_index"],
            "game_plies": source.get("game_plies"), "game_result": source.get("game_result"),
            "source": source, "source_kind": next((t.split(":", 1)[1] for t in record["tags"]
                                                    if t.startswith("source-kind:")), "unknown"),
            "existing_record": record}


def make_record(item: dict, verification: dict, identifier: str) -> dict:
    board, color = fen_to_board(item["fen"])
    mate = verification["reference_mate_moves"]
    identity = {"fen": item["fen"], "goal": "checkmate", "max_plies": 2 * mate + 1, "agent_color": color}
    identity_hash = fingerprint_payload(identity)[:10]
    previous = item.get("existing_record")
    changed_identity = previous is not None and any(previous.get(key) != value for key, value in identity.items())
    if changed_identity:
        identifier = "xiangqi-history-r1-revised-" + previous["id"].rsplit("-", 1)[-1] + "-" + identity_hash
    elif identifier.startswith("xiangqi-history-r1-") and identifier.removeprefix("xiangqi-history-r1-").isdigit():
        identifier += "-" + identity_hash
    source = item.get("source") or {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION,
        "license": SOURCE_LICENSE, "file": item["source_file"], "ply_index": item["ply_index"],
        "game_plies": item.get("game_plies"), "game_result": item.get("game_result")}
    record = {"schema_version": 2, "release_id": "xiangqi-2026-09-19-r1", "id": identifier, "family": "xiangqi-history", "fen": item["fen"],
        "agent_color": color, "goal": "checkmate", "max_plies": 2 * mate + 1,
        "difficulty": verification["band"], "piece_count": sum(bool(v) for row in board for v in row),
        "oracle": {"best_move_uci": verification["runs"][-1]["best_move_uci"],
                   "mate_in_plies": verification["reference_mate_plies"], "evaluation_cp": None},
        "tags": sorted({"ccpd", f"mate-band:{verification['band']}", f"mate-in-moves:{mate}",
                        f"source-kind:{item.get('source_kind', 'unknown')}"}),
        "source": source, "source_id": "ccpd:" + source["file"],
        "h2_analysis": {"version": GENERATOR_VERSION, "mate_in_moves": mate,
            "reference_mate_plies": verification["reference_mate_plies"],
            "verification_depths": [run["actual_depth"] for run in verification["runs"]],
            "requested_depths": [run["requested_depth"] for run in verification["runs"]],
            "principal_variation_uci": verification["runs"][-1]["principal_variation_uci"],
            "validation_runs": verification["runs"], "source_reused": False,
            "interpretation": "Two engine reference lines end in independently verified checkmate; not a shortest-mate proof."}}
    if changed_identity:
        record["replaces_id"] = previous["id"]
    validate_record(record, expected_family="xiangqi-history")
    return record


def candidate_items(existing: list[dict], source_root: Path | None, pools: list[Path], seed: int,
                    used_sources: set[str], failures: list[dict]):
    # Prefer nearby earlier positions from source games whose former record failed.
    if source_root is not None:
        sys.path.insert(0, str(ROOT / "scripts"))
        from extract_ccpd_positions import replay, source_kind
        ordered_existing = sorted(existing, key=lambda record: ({"long": 0, "medium": 1, "short": 2}[record["difficulty"]], record["id"]))
        for record in ordered_existing:
            source = record["source"]
            if source["file"] in used_sources:
                continue
            path = source_root / source["file"]
            try:
                positions, game_result = replay(path)
                source_ply = int(source["ply_index"])
                for position in reversed(positions[max(0, source_ply - 24):source_ply]):
                    if position["side_to_move"] != record["agent_color"]:
                        continue
                    yield {**position, "fen": canonical_fen(position["fen"]),
                           "source_file": source["file"], "source_kind": source_kind(path),
                           "game_plies": len(positions), "game_result": game_result}
            except Exception as exc:
                failures.append({"stage": "source_replay", "source_file": source["file"],
                                 "reasons": [f"{type(exc).__name__}:{exc}"]})
    for pool in pools:
        groups = defaultdict(list)
        with pool.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    source_file, fen = item["source_file"], item["fen"]
                except (ValueError, KeyError) as exc:
                    raise ValueError(f"invalid frozen pool {pool}:{line_number}") from exc
                if source_file in used_sources:
                    continue
                try:
                    item["fen"] = canonical_fen(fen)
                except ValueError as exc:
                    failures.append({"stage": "pool_position", "pool": str(pool), "line": line_number,
                                     "source_file": source_file, "fen": fen,
                                     "reasons": [f"invalid_position:{exc}"]})
                    continue
                groups[source_file].append(item)
        sources = sorted(groups)
        random.Random(seed).shuffle(sources)
        # Curated tactical/endgame sources are more likely to contain a verifiable
        # long mate than ordinary games ending in resignation.
        source_priority = {"kill-tactic": 0, "endgame": 1, "fullgame-tactics": 2}
        sources.sort(key=lambda source: source_priority.get(groups[source][0].get("source_kind"), 3))
        # Round-robin games avoids spending the budget on many adjacent states of one game.
        queues = [deque(sorted(groups[source], key=lambda item: (
            abs(int(item.get("plies_from_end", 14)) - 14), -int(item["ply_index"])))) for source in sources]
        while queues:
            for queue in queues:
                yield queue.popleft()
            queues = [queue for queue in queues if queue]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, default=ROOT / "data/xiangqi/history/tasks.jsonl")
    parser.add_argument("--pool", type=Path, action="append", default=[])
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--exclude-dataset", type=Path, action="append", default=None,
                        help="Exclude every FEN from this dataset; defaults to current D3 and C2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--depth-a", type=int, default=16)
    parser.add_argument("--depth-b", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--pikafish", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("require workers>=1")
    if (args.depth_a, args.depth_b) != (16, 20):
        parser.error("r1 H2 requires exactly depth-a=16 and depth-b=20")
    frozen_pools = pool_fingerprints(args.pool)
    excluded_paths = args.exclude_dataset if args.exclude_dataset is not None else [
        ROOT / "data/xiangqi/mate_in_one/tasks.jsonl", ROOT / "data/xiangqi/rule_variants/tasks.jsonl"]
    excluded_inputs = pool_fingerprints(excluded_paths)
    excluded_fens = {canonical_fen(record["fen"]) for path in excluded_paths for record in read_jsonl(path)}
    calibration = calibrate_cchess()
    if not calibration["valid"]:
        raise RuntimeError(f"independent validator calibration failed: {calibration}")
    executable = resolve_pikafish_executable(args.pikafish, start_dir=ROOT).resolve()
    engine_fingerprint = pikafish_fingerprint(executable)
    # Fail once on setup faults rather than misclassifying every candidate.
    with PikafishEngine(executable, timeout=args.timeout) as preflight:
        preflight.ready()
    configuration = {"generator": GENERATOR_VERSION, "depths": [args.depth_a, args.depth_b],
        "validation_version": VALIDATION_VERSION, "engine": engine_fingerprint,
        "validator_sha256": sha256_file(ROOT / "src/minibench/datasets/xiangqi/validation.py"),
        "reference_sha256": sha256_file(ROOT / "src/minibench/datasets/xiangqi/reference.py"),
        "board_sha256": sha256_file(ROOT / "src/minibench/datasets/xiangqi/variants/board.py")}
    config_hash = fingerprint_payload(configuration)
    checkpoint = args.checkpoint or args.report.with_suffix(".checkpoint.jsonl")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if checkpoint.exists():
        for entry in read_jsonl(checkpoint):
            if entry.get("configuration_sha256") == config_hash:
                cached[entry["fen"]] = entry
    old = read_jsonl(args.existing)
    old_input_sha256 = sha256_file(args.existing)
    selection_configuration = {"seed": args.seed, "existing_input_sha256": old_input_sha256,
                               "pool_inputs": frozen_pools, "excluded_datasets": excluded_inputs,
                               "candidate_order": "old-upstream-then-pools-in-order-v1"}
    selection_hash = fingerprint_payload(selection_configuration)
    selection_history = checkpoint.with_suffix(".selection-configurations.jsonl")
    with selection_history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"sha256": selection_hash, "configuration": selection_configuration}, ensure_ascii=False) + "\n")
    selected, attempts, replay_failures, selection_exclusions = [], [], [], []
    by_band, used_sources, tried = Counter(), set(), set()
    verifier = Verifier(executable, (args.depth_a, args.depth_b), args.timeout)

    def remember(item, result, stage):
        result = {**result, "configuration_sha256": config_hash, "stage": stage,
                  "selection_configuration_sha256": selection_hash,
                  "task_id": (item.get("existing_record") or {}).get("id")}
        if item["fen"] not in cached or cached[item["fen"]].get("runs") != result.get("runs"):
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
            cached[item["fen"]] = result
        tried.add(item["fen"])
        attempts.append(result)
        return result

    def report(status):
        output = {"version": GENERATOR_VERSION, "status": status, "total": len(selected),
            "release_id": "xiangqi-2026-09-19-r1", "generator_sha256": sha256_file(Path(__file__)),
            "targets": TARGETS, "seed": args.seed, "workers": args.workers,
            "selection_configuration": selection_configuration, "selection_configuration_sha256": selection_hash,
            "excluded_datasets": excluded_inputs, "selection_exclusions": selection_exclusions,
            "excluded_existing_ids": [r["task_id"] for r in selection_exclusions if r["stage"] == "existing"],
            "screening": {"depth": 8, "candidate_min_cp": 300},
            "difficulty": dict(by_band), "unique_fens": len({r["fen"] for r in selected}),
            "unique_source_games": len(used_sources), "reused_source_positions": 0,
            "existing_input_sha256": old_input_sha256,
            "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION,
            "source_license": SOURCE_LICENSE, "pikafish": engine_fingerprint,
            "verification_depths": [args.depth_a, args.depth_b], "configuration_sha256": config_hash,
            "independent_calibration": calibration, "existing_audited": sum(a["stage"] == "existing" for a in attempts),
            "existing_valid": sum(a["stage"] == "existing" and a["valid"] for a in attempts),
            "retained_existing_ids": [r["id"] for r in selected if not r["id"].startswith("xiangqi-history-r1-")],
            "replacement_ids": [r["id"] for r in selected if r["id"].startswith("xiangqi-history-r1-")],
            "revised_goal_ids": [{"old_id": r["replaces_id"], "new_id": r["id"]} for r in selected if "replaces_id" in r],
            "failure_counts": dict(Counter(reason for a in attempts if not a["valid"] for reason in a["reasons"])),
            "failures": [a for a in attempts if not a["valid"]] + replay_failures,
            "validations": [{"task_id": r["id"], "fen": r["fen"], "source_file": r["source"]["file"],
                             **r["h2_analysis"]} for r in selected], "attempts": attempts,
            "interpretation": "Reference engine distances and two complete checked PVs, not a proof of shortest forced mate."}
        atomic_write_json(args.report, output)
        return output

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            items = [existing_item(record) for record in old]
            def audit(item):
                return cached.get(item["fen"]) or verifier.check(item)
            for item, raw in zip(items, executor.map(audit, items)):
                result = remember(item, raw, "existing")
                if item["fen"] in excluded_fens:
                    selection_exclusions.append({"stage": "existing", "task_id": item["existing_record"]["id"],
                        "fen": item["fen"], "reason": "cross_family_fen_overlap"})
                elif (result["valid"] and by_band[result["band"]] < TARGETS[result["band"]]
                        and item["source_file"] not in used_sources):
                    selected.append(make_record(item, result, item["existing_record"]["id"]))
                    by_band[result["band"]] += 1
                    used_sources.add(item["source_file"])
                if len(attempts) % 10 == 0:
                    print(f"audit={len(attempts)}/{len(old)} retained={dict(by_band)}", flush=True)
                    report("auditing")
            report("audit_complete")
            if args.audit_only:
                return 0
            assert_frozen_pools(args.pool, frozen_pools)
            assert_frozen_pools(excluded_paths, excluded_inputs)
            candidates = candidate_items(old, args.source_root, args.pool, args.seed, used_sources, replay_failures)
            replacements = 0
            exhausted = False
            while any(by_band[band] < count for band, count in TARGETS.items()):
                needed = {band for band, count in TARGETS.items() if by_band[band] < count}
                batch, batch_fens = [], set()
                while len(batch) < args.workers * 2:
                    try:
                        item = next(candidates)
                    except StopIteration:
                        exhausted = True
                        break
                    if item["source_file"] in used_sources or item["fen"] in tried or item["fen"] in batch_fens:
                        continue
                    if item["fen"] in excluded_fens:
                        selection_exclusions.append({"stage": "candidate", "source_file": item["source_file"],
                            "fen": item["fen"], "reason": "cross_family_fen_overlap"})
                        tried.add(item["fen"])
                        continue
                    batch.append(item); batch_fens.add(item["fen"])
                if not batch:
                    break
                def check_candidate(item):
                    prior = cached.get(item["fen"])
                    return prior if cached_candidate_usable(prior, needed) else verifier.check(item, needed)
                before_selected = len(selected)
                for item, raw in zip(batch, executor.map(check_candidate, batch)):
                    result = remember(item, raw, "candidate")
                    if (result["valid"] and by_band[result["band"]] < TARGETS[result["band"]]
                            and item["source_file"] not in used_sources):
                        replacements += 1
                        selected.append(make_record(item, result, f"xiangqi-history-r1-{replacements:04d}"))
                        by_band[result["band"]] += 1
                        used_sources.add(item["source_file"])
                if len(selected) != before_selected or (len(attempts)-len(old)) % 80 == 0:
                    print(f"candidates={len(attempts)-len(old)} selected={dict(by_band)} replacements={replacements}", flush=True)
                    report("replacing")
                if exhausted:
                    break
    finally:
        verifier.close()
    assert_frozen_pools(args.pool, frozen_pools)
    assert_frozen_pools(excluded_paths, excluded_inputs)
    assert_no_cross_family_collisions(selected, excluded_fens)
    shortages = {band: count - by_band[band] for band, count in TARGETS.items() if by_band[band] < count}
    if shortages:
        report("insufficient_candidates")
        raise RuntimeError(f"insufficient independently verified H2 candidates: {shortages}; old output preserved")
    selected.sort(key=lambda record: (list(TARGETS).index(record["difficulty"]), record["id"]))
    atomic_write_jsonl(args.output, selected)
    final = report("complete")
    print(json.dumps({key: final[key] for key in ("status", "total", "difficulty", "unique_source_games", "existing_valid")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
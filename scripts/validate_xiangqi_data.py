"""Independent, fail-closed release gate for all four Xiangqi datasets.

No engine or model is called. Stored answers and engine PVs are rechecked from
board geometry using production rules, a separate reference implementation and
cchess. A partial/--skip-search audit cannot certify a complete release.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from minibench.core.checkpoint import atomic_write_json
from minibench.datasets.xiangqi import reference
from minibench.datasets.xiangqi.schema import (
    RULESETS, XIANGQI_FAMILIES, board_to_fen, fen_to_board, internal_rules,
    sample_records, validate_record,
)
from minibench.datasets.xiangqi.validation import (
    VALIDATION_VERSION, _cchess_in_check, calibrate_cchess,
    compare_standard_legal_moves, validate_position, validate_pv,
)
from minibench.datasets.xiangqi.variants.board import VariantBoard
from minibench.datasets.xiangqi.variants.rules import Rule
from minibench.datasets.xiangqi.variants.search import score_moves

RELEASE_ID = "xiangqi-2026-09-19-r1"
GATE_VERSION = "xiangqi-release-gate-v1"
FAMILY_DIRS = {
    "xiangqi-mate-in-one": "mate_in_one", "xiangqi-history": "history",
    "xiangqi-rule-variants": "rule_variants", "xiangqi-multimodal": "multimodal",
}
FOCI = tuple(name for name in RULESETS if name != "standard")
ALIASES = dict(zip(("d3", "c2", "h2", "m2"), XIANGQI_FAMILIES))
TOLERANCE = 1e-6
CCPD_REPOSITORY = "https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset"
CCPD_REVISION = "368a47a947773dd8692c026e286dd19b6277b993"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_number(left: Any, right: Any) -> bool:
    return (isinstance(left, (int, float)) and not isinstance(left, bool)
            and isinstance(right, (int, float)) and not isinstance(right, bool)
            and math.isfinite(left) and math.isfinite(right)
            and abs(left - right) <= TOLERANCE)


def _root_check(board, side, reasons):
    comparison = compare_standard_legal_moves(board, side)
    if not comparison["valid"]:
        reasons.extend(comparison["reasons"])
    return comparison


def _mate_details(record, board, side, reasons):
    legal = reference.legal_moves(board, side)
    mates, checks = [], 0
    for move in legal:
        child = reference.apply_move(board, move)
        compared = compare_standard_legal_moves(child, -side)
        if not compared["valid"]:
            reasons.extend(f"after_{move.to_uci()}:{why}" for why in compared["reasons"])
            continue
        ref_terminal = reference.terminal_status(child, -side)
        prod_terminal = VariantBoard(child, []).terminal_status(-side)
        child_fen = board_to_fen(child, active_color="red" if side < 0 else "black")
        external_terminal = (
            "ongoing" if compared["move_sets"]["cchess"]
            else "checkmate" if _cchess_in_check(child_fen) else "stalemate"
        )
        if len({ref_terminal, prod_terminal, external_terminal}) != 1:
            reasons.append(f"after_{move.to_uci()}:terminal_disagreement")
        if ref_terminal == "checkmate":
            mates.append(move.to_uci())
        elif reference.in_check(child, -side):
            checks += 1
    mates.sort()
    analysis_key = "d3_analysis" if record["family"] == "xiangqi-mate-in-one" else "m2_analysis"
    analysis = record.get(analysis_key, {})
    if analysis.get("mate_moves_uci") != mates:
        reasons.append("stored_mate_set_disagrees_with_independent_enumeration")
    if not mates:
        reasons.append("no_mate_in_one")
    if analysis.get("mate_move_count") != len(mates):
        reasons.append("mate_move_count_mismatch")
    if record["oracle"].get("best_move_uci") not in mates:
        reasons.append("oracle_move_is_not_mate_in_one")
    if record["oracle"].get("mate_in_plies") != 1:
        reasons.append("oracle_mate_distance_must_be_one")
    if record.get("max_plies") != 1:
        reasons.append("one_move_horizon_required")
    if analysis_key == "d3_analysis":
        for key, value in (("legal_move_count", len(legal)), ("non_mating_check_count", checks)):
            if analysis.get(key) != value:
                reasons.append(f"d3_{key}_mismatch")
    return {"legal_move_count": len(legal), "mate_moves_uci": mates,
            "non_mating_check_count": checks, "all_replies_independently_checked": not reasons}


def _h2_details(record, reasons):
    analysis = record.get("h2_analysis", {})
    moves = analysis.get("mate_in_moves")
    if not isinstance(moves, int) or isinstance(moves, bool) or not 2 <= moves <= 12:
        reasons.append("invalid_reference_mate_moves")
        return {}
    plies = 2 * moves - 1
    expected_band = "short" if moves <= 3 else "medium" if moves <= 5 else "long"
    if record.get("difficulty") != expected_band:
        reasons.append("reference_distance_band_mismatch")
    if analysis.get("reference_mate_plies") != plies or record["oracle"].get("mate_in_plies") != plies:
        reasons.append("reference_mate_plies_mismatch")
    if record.get("max_plies") != plies + 2:
        reasons.append("h2_horizon_mismatch")
    if analysis.get("source_reused") is not False:
        reasons.append("source_reuse_must_be_false")
    if analysis.get("requested_depths") != [16, 20]:
        reasons.append("h2_requires_requested_depths_16_20")
    source_file = record.get("source", {}).get("file")
    if not isinstance(source_file, str) or not source_file:
        reasons.append("missing_h2_source_file")
    elif record.get("source_id") != "ccpd:" + source_file:
        reasons.append("h2_source_id_mismatch")
    runs = analysis.get("validation_runs")
    if not isinstance(runs, list) or len(runs) != 2:
        reasons.append("h2_requires_two_validation_runs")
        return {"reference_mate_plies": plies}
    results = []
    actual_depths = []
    for index, (run, depth) in enumerate(zip(runs, (16, 20))):
        if not isinstance(run, dict):
            reasons.append(f"depth_{depth}:invalid_validation_run")
            continue
        prefix = f"depth_{depth}:"
        actual = run.get("actual_depth")
        actual_depths.append(actual)
        if run.get("requested_depth") != depth:
            reasons.append(prefix + "requested_depth_mismatch")
        if not isinstance(actual, int) or isinstance(actual, bool) or actual < depth:
            reasons.append(prefix + "actual_depth_too_low")
        if run.get("score_kind") != "mate" or run.get("score") != moves:
            reasons.append(prefix + "reference_score_mismatch")
        if run.get("score_is_bound") is not False:
            reasons.append(prefix + "bound_or_unspecified_score")
        pv = run.get("principal_variation_uci")
        proof = validate_pv(record["fen"], pv, require_checkmate=True)
        results.append({"requested_depth": depth, "recomputed_validation": proof})
        if not proof["valid"] or not proof["independent_verified"]:
            reasons.extend(prefix + why for why in proof["reasons"])
            if not proof["independent_verified"]:
                reasons.append(prefix + "independent_verification_missing")
        if not isinstance(pv, list) or len(pv) != plies or proof["plies"] != plies:
            reasons.append(prefix + "reference_length_mismatch")
        if not pv or run.get("best_move_uci") != pv[0]:
            reasons.append(prefix + "bestmove_pv_mismatch")
        stored_proof = run.get("pv_validation", {})
        if (stored_proof.get("valid") is not True or stored_proof.get("terminal") != "checkmate"
                or stored_proof.get("plies") != plies or stored_proof.get("independent_verified") is not True):
            reasons.append(prefix + "stored_validation_metadata_invalid")
    if analysis.get("verification_depths") != actual_depths:
        reasons.append("actual_depth_metadata_mismatch")
    if isinstance(runs[-1], dict):
        if analysis.get("principal_variation_uci") != runs[-1].get("principal_variation_uci"):
            reasons.append("primary_pv_is_not_depth_20_pv")
        if record["oracle"].get("best_move_uci") != runs[-1].get("best_move_uci"):
            reasons.append("oracle_is_not_depth_20_bestmove")
    return {"reference_mate_plies": plies, "runs": results,
            "claim_limit": "Two complete engine reference PVs; not a proof over every defence or a shortest-mate proof."}


def _c2_details(record, board, side, reasons, skip_search):
    active = internal_rules(record)
    production_rules = [Rule.from_dict(rule) for rule in active]
    production_board = VariantBoard(board, production_rules)
    legal = {move.to_uci() for move in production_board.legal_moves(side)}
    independent = reference.legal_uci_moves(board, side, active)
    if legal != independent:
        reasons.append("active_rule_legal_moves_disagree")
    if reference.in_check(board, -side, active):
        reasons.append("active_rules_check_nonmoving_general")
    analysis = record.get("c2_analysis", {})
    if analysis.get("oracle_depth") != 3 or record.get("max_plies") != 1:
        reasons.append("c2_requires_depth_three_one_action")
    if record.get("agent_color") != "red":
        reasons.append("c2_release_requires_red_to_move")
    if analysis.get("legal_move_count") != len(independent):
        reasons.append("c2_legal_move_count_mismatch")
    if record["oracle"].get("best_move_uci") not in independent:
        reasons.append("c2_oracle_is_illegal")
    details = {"legal_move_count": len(independent), "search_verified": False}
    if skip_search:
        details["skipped"] = "depth-three production/reference root utility comparison"
        return details
    scored = reference.score_moves(board, side, 3, side, active)
    production = score_moves(production_board, side, 3, side)
    scores = {move.to_uci(): value for move, value in scored}
    prod_scores = {move.to_uci(): value for move, value in production}
    if scores.keys() != prod_scores.keys() or any(not _same_number(value, prod_scores.get(move)) for move, value in scores.items()):
        reasons.append("all_root_utilities_disagree")
    best_set = reference.best_moves(scored)
    if len(scored) < 2 or len(best_set) != 1:
        reasons.append("c2_requires_unique_best_and_alternative")
    if scored:
        best_move, best_value = scored[0]
        if record["oracle"].get("best_move_uci") != best_move.to_uci():
            reasons.append("c2_oracle_best_move_mismatch")
        if not _same_number(analysis.get("best_value"), best_value):
            reasons.append("c2_best_value_mismatch")
    margin = side * (scored[0][1] - scored[1][1]) if len(scored) > 1 else 0.0
    if margin < 0.5 - TOLERANCE:
        reasons.append("c2_unique_margin_below_half")
    if not _same_number(analysis.get("unique_best_margin"), margin):
        reasons.append("c2_margin_metadata_mismatch")
    details.update(search_verified=not reasons, all_root_utilities=scores,
                   best_moves=sorted(best_set), unique_best_margin=margin)
    return details


def audit_record(record: dict, family: str, *, skip_search: bool = False) -> dict[str, Any]:
    reasons = []
    report = {"id": record.get("id", "<missing>"), "valid": False, "reasons": reasons, "details": {}}
    try:
        validate_record(record, expected_family=family)
        if record.get("release_id") != RELEASE_ID:
            reasons.append("release_id_mismatch")
        board, color = fen_to_board(record["fen"])
        side = 1 if color == "red" else -1
        comparison = _root_check(board, side, reasons)
        report["details"]["standard_legal_moves_verified"] = comparison["valid"] and comparison["independent_verified"]
        if not comparison["valid"]:
            return report
        if family in {"xiangqi-mate-in-one", "xiangqi-multimodal"}:
            details = _mate_details(record, board, side, reasons)
        elif family == "xiangqi-history":
            details = _h2_details(record, reasons)
        else:
            details = _c2_details(record, board, side, reasons, skip_search)
        report["details"].update(details)
    except Exception as exc:
        reasons.append(f"validation_error:{type(exc).__name__}:{exc}")
    report["valid"] = not reasons
    return report


def _audit_payload(payload):
    family, line, record, skip_search = payload
    return {"line": line, **audit_record(record, family, skip_search=skip_search)}


def audit_collection(family: str, records: Sequence[dict]) -> list[str]:
    reasons = []
    if len(records) != 250:
        reasons.append(f"expected_250_records_got_{len(records)}")
    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)):
        reasons.append("duplicate_task_ids")
    fens = [record.get("fen") for record in records]
    if family != "xiangqi-rule-variants" and len(fens) != len(set(fens)):
        reasons.append("duplicate_fens_within_family")
    counts = Counter(record.get("difficulty") for record in records)
    expected = {"short": 80, "medium": 80, "long": 90} if family == "xiangqi-history" else {"easy": 80, "medium": 85, "hard": 85}
    if family != "xiangqi-rule-variants" and dict(counts) != expected:
        reasons.append(f"difficulty_counts_mismatch:{dict(counts)}")
    if family == "xiangqi-history":
        sources = [record.get("source", {}).get("file") for record in records]
        if not all(isinstance(source, str) and source for source in sources) or len(sources) != len(set(sources)):
            reasons.append("h2_source_games_must_be_unique")
    if family == "xiangqi-rule-variants":
        groups = defaultdict(list)
        for record in records:
            groups[record.get("scenario_id")].append(record)
        pairs, controls, strata = 0, 0, Counter()
        fen_scenarios = defaultdict(set)
        for scenario, group in groups.items():
            for record in group:
                fen_scenarios[record.get("fen")].add(scenario)
            if len(group) == 1:
                controls += 1
                control = group[0]
                if control.get("ruleset") != "standard" or control.get("design_stratum") != "standard-control":
                    reasons.append(f"{scenario}:invalid_standard_control")
                analysis = control.get("c2_analysis", {})
                if (analysis.get("focus_ruleset") != "standard" or analysis.get("focus_changes_best_move") is not False
                        or analysis.get("standard_best_uci") != control.get("oracle", {}).get("best_move_uci")):
                    reasons.append(f"{scenario}:invalid_control_metadata")
                continue
            pairs += 1
            by_rule = {record.get("ruleset"): record for record in group}
            if len(group) != 4 or set(by_rule) != set(RULESETS) or len({record.get("fen") for record in group}) != 1:
                reasons.append(f"{scenario}:incomplete_or_mismatched_four_rule_pair")
                continue
            labels = {record.get("design_stratum") for record in group}
            if len(labels) != 1 or not labels <= set(FOCI):
                reasons.append(f"{scenario}:invalid_design_stratum")
                continue
            focus = next(iter(labels))
            strata[focus] += 1
            standard_best = by_rule["standard"]["oracle"]["best_move_uci"]
            if by_rule[focus]["oracle"]["best_move_uci"] == standard_best:
                reasons.append(f"{scenario}:focus_does_not_change_best_move")
            for record in group:
                analysis = record.get("c2_analysis", {})
                if (analysis.get("standard_best_uci") != standard_best
                        or analysis.get("focus_ruleset") != focus
                        or analysis.get("focus_changes_best_move") is not True):
                    reasons.append(f"{record.get('id')}:counterfactual_metadata_mismatch")
        if pairs != 60 or controls != 10:
            reasons.append(f"c2_requires_60_four_rule_pairs_and_10_controls_got_{pairs}_{controls}")
        if dict(strata) != {focus: 20 for focus in FOCI}:
            reasons.append(f"c2_requires_20_scenarios_per_focus:{dict(strata)}")
        if any(len(scenarios) != 1 for scenarios in fen_scenarios.values()):
            reasons.append("c2_same_fen_reused_across_scenarios")
    return reasons


def audit_sampling(family: str, records: Sequence[dict]) -> dict[str, Any]:
    reasons, probes = [], []
    sizes = (1, 10, 60) if family == "xiangqi-rule-variants" else (1, 30, 250)
    for count in sizes:
        try:
            first = sample_records(records, count=count, seed=42)
            repeated = sample_records(records, count=count, seed=42)
            ids = [item["id"] for item in first]
            if ids != [item["id"] for item in repeated] or len(ids) != len(set(ids)):
                reasons.append(f"sample_{count}:not_reproducible_or_duplicate_ids")
            expected_size = count * 4 if family == "xiangqi-rule-variants" else count
            if len(ids) != expected_size or not set(ids) <= {item["id"] for item in records}:
                reasons.append(f"sample_{count}:wrong_membership_or_size")
            if family == "xiangqi-rule-variants":
                grouped = defaultdict(list)
                for record in first:
                    grouped[record["scenario_id"]].append(record)
                if len(grouped) != count or any(
                    len(group) != 4 or {item["ruleset"] for item in group} != set(RULESETS)
                    or len({item["fen"] for item in group}) != 1 for group in grouped.values()
                ):
                    reasons.append(f"sample_{count}:incomplete_paired_scenarios")
            probes.append({"count": count, "selected_records": len(ids), "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
        except Exception as exc:
            reasons.append(f"sample_{count}:{type(exc).__name__}:{exc}")
    return {"valid": not reasons, "reasons": reasons, "seed": 42, "probes": probes}


def audit_m2_mapping(d3: Sequence[dict], m2: Sequence[dict]) -> list[str]:
    reasons = []
    sources = {record["id"]: record for record in d3}
    linked = [record.get("source_task_id") for record in m2]
    if len(linked) != len(set(linked)) or set(linked) != set(sources):
        reasons.append("m2_must_map_one_to_one_onto_all_d3_tasks")
    for record in m2:
        source = sources.get(record.get("source_task_id"))
        if source is None:
            reasons.append(f"{record.get('id')}:unknown_d3_source")
            continue
        for key in ("fen", "agent_color", "goal", "max_plies", "difficulty", "piece_count", "oracle", "source_id"):
            if record.get(key) != source.get(key):
                reasons.append(f"{record['id']}:d3_{key}_mapping_mismatch")
        if record.get("m2_analysis", {}).get("mate_moves_uci") != source.get("d3_analysis", {}).get("mate_moves_uci"):
            reasons.append(f"{record['id']}:d3_mate_set_mapping_mismatch")
    return reasons




def _parse_source_pgn(path: Path):
    # The extractor is used only to decode PGN notation, never as a legality oracle.
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("xiangqi_pgn_text_parser", ROOT / "scripts/extract_ccpd_positions.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.replay(path)


def audit_c2_sources(records: Sequence[dict], data_root: Path) -> dict[str, Any]:
    report = {"valid": False, "reasons": [], "inputs": {}, "records": [], "source_count": 0}
    groups = defaultdict(list)
    source_root = (data_root / "sources/ccpd").resolve()
    for record in records:
        result = {"id": record.get("id"), "valid": False, "reasons": []}
        report["records"].append(result)
        reasons = result["reasons"]
        try:
            source = record["source"]
            filename = source["source_file"]
            if (not isinstance(filename, str) or not filename or "\\" in filename
                    or PurePosixPath(filename).is_absolute() or ".." in PurePosixPath(filename).parts):
                raise ValueError("source_file must be a safe source-relative POSIX path")
            path = (source_root / filename).resolve()
            if source_root not in path.parents:
                raise ValueError("source path escapes archived CCPD directory")
            ply = source["ply_index"]
            if not isinstance(ply, int) or isinstance(ply, bool) or ply < 0:
                raise ValueError("invalid source ply index")
            for key, expected in (("repository", CCPD_REPOSITORY), ("revision", CCPD_REVISION), ("license", "CC-BY-4.0")):
                if source.get(key) != expected:
                    reasons.append(f"source_{key}_mismatch")
            if source.get("replayed_plies_verified") != ply:
                reasons.append("source_verified_prefix_length_mismatch")
            if record.get("source_id") != "ccpd:" + filename:
                reasons.append("source_id_mismatch")
            groups[filename].append((record, result, ply, path))
        except Exception as exc:
            reasons.append(f"source_metadata_error:{type(exc).__name__}:{exc}")
    for filename, entries in groups.items():
        path = entries[0][3]
        try:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            report["inputs"][filename] = {"path": str(path), "sha256": digest}
            positions, _ = _parse_source_pgn(path)
            if not positions:
                raise ValueError("PGN contains no parsed positions")
            max_ply = min(max(entry[2] for entry in entries), len(positions) - 1)
            cumulative, prefix_reasons = [], []
            previous_child = previous_side = None
            for index, position in enumerate(positions[:max_ply + 1]):
                board, color = fen_to_board(position["fen"])
                side = 1 if color == "red" else -1
                # FEN reparsing reassigns the two same-kind piece IDs in scan
                # order, so compare piece types/sides through canonical FEN.
                if previous_child is not None and (board_to_fen(board) != board_to_fen(previous_child) or side != -previous_side):
                    cumulative.append(f"ply_{index}:parsed_transition_disagrees_with_independent_move")
                cumulative.extend(f"ply_{index}:{why}" for why in validate_position(board, side))
                compared = compare_standard_legal_moves(board, side)
                if not compared["valid"] or not compared["independent_verified"]:
                    cumulative.extend(f"ply_{index}:{why}" for why in compared["reasons"])
                    if not compared["independent_verified"]:
                        cumulative.append(f"ply_{index}:independent_verification_missing")
                prefix_reasons.append(list(cumulative))
                previous_child = previous_side = None
                if index < max_ply:
                    move = position.get("played_move_uci")
                    independent = {item.to_uci(): item for item in reference.legal_moves(board, side)}
                    if move not in independent or move not in compared["move_sets"]["cchess"]:
                        cumulative.append(f"ply_{index}:source_move_illegal")
                    else:
                        previous_child = reference.apply_move(board, independent[move])
                        previous_side = side
            for record, result, ply, _ in entries:
                reasons = result["reasons"]
                source = record["source"]
                if source.get("file_sha256") != digest:
                    reasons.append("source_file_hash_mismatch")
                if ply >= len(positions):
                    reasons.append("source_ply_out_of_range")
                    continue
                reasons.extend(prefix_reasons[ply])
                target = positions[ply]
                if record.get("fen") != target["fen"] or source.get("fen") != target["fen"]:
                    reasons.append("source_replayed_fen_mismatch")
                if source.get("side_to_move") != target["side_to_move"]:
                    reasons.append("source_side_to_move_mismatch")
                if source.get("played_move_uci") != target.get("played_move_uci"):
                    reasons.append("source_played_move_metadata_mismatch")
                result.update(source_file=filename, verified_prefix_plies=ply)
        except Exception as exc:
            for _, result, _, _ in entries:
                result["reasons"].append(f"source_replay_error:{type(exc).__name__}:{exc}")
    for result in report["records"]:
        result["valid"] = not result["reasons"]
        report["reasons"].extend(f"{result['id']}:{why}" for why in result["reasons"])
    report["source_count"] = len(report["inputs"])
    report["valid"] = not report["reasons"]
    return report


def audit_frozen_samples(data_root: Path, datasets: dict[str, Sequence[dict]], inputs: dict) -> dict[str, Any]:
    """Check the committed rosters, not just the sampler's ability to draw them.

    Absence of a manifest is a permissible pre-freeze state, but never a full
    release audit. Once present, any inconsistent/missing artifact fails closed.
    Stored paths are compared to canonical paths and are never followed.
    """
    report = {"status": "not_frozen", "valid": False, "complete": False,
              "reasons": [], "inputs": {}, "samples": {}}
    manifest_path = data_root / "evaluation_samples/manifest.json"
    if not manifest_path.is_file():
        report["reasons"].append("frozen_manifest_missing")
        return report
    if set(datasets) != set(XIANGQI_FAMILIES):
        report["status"] = "partial_family_audit"
        report["reasons"].append("all_four_families_required_for_frozen_sample_audit")
        return report
    report["status"] = "checked"
    reasons = report["reasons"]
    try:
        raw = manifest_path.read_bytes()
        report["inputs"]["manifest.json"] = {"path": str(manifest_path.resolve()),
                                                 "sha256": hashlib.sha256(raw).hexdigest()}
        manifest = json.loads(raw)
        if manifest.get("release_id") != RELEASE_ID:
            reasons.append("frozen_release_id_mismatch")
        if manifest.get("protocol_version") != "xiangqi-reasoning-v2":
            reasons.append("frozen_protocol_version_mismatch")
        if set(manifest.get("samples", {})) != {"smoke", "formal"}:
            reasons.append("frozen_requires_smoke_and_formal_samples")
        sources = {family: {record["id"]: record for record in rows} for family, rows in datasets.items()}
        for size, seed, count, c2count in (("smoke", 20260909, 6, 3), ("formal", 42, 30, 10)):
            sample = manifest.get("samples", {}).get(size, {})
            current = {"valid": False, "reasons": [], "tasks": {}}
            report["samples"][size] = current
            local = current["reasons"]
            if sample.get("seed") != seed:
                local.append("seed_mismatch")
            if set(sample.get("tasks", {})) != set(ALIASES):
                local.append("requires_all_four_sample_families")
            expected = {key: sample_records(datasets[ALIASES[key]], count=c2count if key == "c2" else count,
                                           seed=seed, strategy="paired-stratified" if key == "c2" else "stratified")
                        for key in ("d3", "c2", "h2")}
            m2_sources = {row.get("source_task_id"): row for row in datasets[ALIASES["m2"]]}
            expected["m2"] = [m2_sources[row["id"]] for row in expected["d3"]]
            selected = {}
            for key, family in ALIASES.items():
                file_key = f"{size}/{key}.jsonl"
                path = data_root / "evaluation_samples" / file_key
                meta = sample.get("tasks", {}).get(key, {})
                task_reasons = []
                task = {"valid": False, "reasons": task_reasons}
                current["tasks"][key] = task
                try:
                    raw = path.read_bytes()
                    digest = hashlib.sha256(raw).hexdigest()
                    report["inputs"][file_key] = {"path": str(path.resolve()), "sha256": digest}
                    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
                    if any(not isinstance(row, dict) for row in rows):
                        raise ValueError("sample records must be JSON objects")
                    selected[key] = rows
                    ids = [row["id"] for row in rows]
                    task.update(record_count=len(rows), task_ids=ids, sha256=digest)
                    required_count = c2count * 4 if key == "c2" else count
                    if len(rows) != required_count or len(ids) != len(set(ids)):
                        task_reasons.append("wrong_count_or_duplicate_ids")
                    if meta.get("path") != "data/xiangqi/evaluation_samples/" + file_key:
                        task_reasons.append("manifest_path_mismatch")
                    if meta.get("sha256") != digest:
                        task_reasons.append("manifest_sample_hash_mismatch")
                    if meta.get("dataset_sha256") != inputs.get(family, {}).get("sha256"):
                        task_reasons.append("manifest_dataset_hash_mismatch")
                    if meta.get("record_count") != len(rows) or meta.get("task_ids") != ids:
                        task_reasons.append("manifest_roster_metadata_mismatch")
                    for row in rows:
                        if sources[family].get(row["id"]) != row:
                            task_reasons.append(f"{row['id']}:sample_record_differs_from_dataset")
                    if ids != [row["id"] for row in expected[key]]:
                        task_reasons.append("frozen_roster_disagrees_with_seeded_selection")
                    if key == "c2":
                        groups = defaultdict(list)
                        for row in rows:
                            groups[row.get("scenario_id")].append(row)
                        if len(groups) != c2count or any(
                            len(group) != 4 or {row.get("ruleset") for row in group} != set(RULESETS)
                            or len({row.get("fen") for row in group}) != 1
                            or len({row.get("design_stratum") for row in group}) != 1
                            for group in groups.values()
                        ):
                            task_reasons.append("incomplete_or_mismatched_paired_scenarios")
                        task["scenario_count"] = len(groups)
                except Exception as exc:
                    task_reasons.append(f"sample_validation_error:{type(exc).__name__}:{exc}")
                task["valid"] = not task_reasons
                local.extend(f"{key}:{why}" for why in task_reasons)
            if {"d3", "m2"} <= selected.keys():
                local.extend(audit_m2_mapping(selected["d3"], selected["m2"]))
                if [row.get("source_task_id") for row in selected["m2"]] != [row.get("id") for row in selected["d3"]]:
                    local.append("d3_m2_sample_order_or_sources_differ")
            current["valid"] = not local
            reasons.extend(f"{size}:{why}" for why in local)
        report["complete"] = set(report["inputs"]) == {"manifest.json", *(f"{size}/{key}.jsonl" for size in ("smoke", "formal") for key in ALIASES)}
    except Exception as exc:
        reasons.append(f"frozen_sample_validation_error:{type(exc).__name__}:{exc}")
    report["valid"] = report["complete"] and not reasons
    return report


def audit_release(
    data_root: Path, *, families: Sequence[str] = XIANGQI_FAMILIES,
    skip_search: bool = False, workers: int = 1,
) -> dict[str, Any]:
    families = tuple(dict.fromkeys(families))
    if workers < 1 or not families or set(families) - set(XIANGQI_FAMILIES):
        raise ValueError("select supported families and workers >= 1")
    needed = list(families)
    if "xiangqi-multimodal" in needed and "xiangqi-mate-in-one" not in needed:
        needed.append("xiangqi-mate-in-one")
    rows, inputs, errors = {}, {}, {}
    for family in needed:
        path = data_root / FAMILY_DIRS[family] / "tasks.jsonl"
        rows[family], errors[family] = [], []
        try:
            raw = path.read_bytes()
            inputs[family] = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
            for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("record must be a JSON object")
                    rows[family].append((number, record))
                except (ValueError, TypeError) as exc:
                    errors[family].append(f"line_{number}:invalid_json_record:{exc}")
        except (OSError, UnicodeError) as exc:
            errors[family].append(f"input_read_failed:{exc}")
    result: dict[str, Any] = {"gate_version": GATE_VERSION, "validation_version": VALIDATION_VERSION,
        "release_id": RELEASE_ID, "valid": False, "full": not skip_search and set(families) == set(XIANGQI_FAMILIES),
        "release_ready": False, "skip_search": skip_search, "families_requested": list(families),
        "inputs": inputs, "independent_calibration": calibrate_cchess(), "families": {}, "reasons": [],
        "validator_hashes": {str(path.relative_to(ROOT)): _hash(path) for path in (
            Path(__file__).resolve(), ROOT / "scripts/extract_ccpd_positions.py", ROOT / "src/minibench/datasets/xiangqi/reference.py",
            ROOT / "src/minibench/datasets/xiangqi/validation.py", ROOT / "src/minibench/datasets/xiangqi/schema.py",
            ROOT / "src/minibench/datasets/xiangqi/variants/board.py", ROOT / "src/minibench/datasets/xiangqi/variants/search.py")},
        "limitations": ["PV checks validate the two supplied lines, not all possible defences or shortest mate distance.",
                        "Structural position constraints do not independently prove game-history reachability."]}
    if not result["independent_calibration"]["valid"]:
        result["reasons"].append("cchess_calibration_failed")
        return result
    for family in families:
        records = [record for _, record in rows[family]]
        family_reasons = list(errors[family])
        try:
            family_reasons.extend(audit_collection(family, records))
        except Exception as exc:
            family_reasons.append(f"collection_validation_error:{type(exc).__name__}:{exc}")
        payloads = [(family, line, record, skip_search) for line, record in rows[family]]
        if workers == 1:
            checked = [_audit_payload(payload) for payload in payloads]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                checked = list(pool.map(_audit_payload, payloads))
        sampling = audit_sampling(family, records)
        family_reasons.extend(sampling["reasons"])
        if any(not record["valid"] for record in checked):
            family_reasons.append("one_or_more_records_failed")
        result["families"][family] = {"valid": not family_reasons, "record_count": len(records),
            "failed_record_count": sum(not record["valid"] for record in checked),
            "reasons": family_reasons, "sampling": sampling, "records": checked}
    if "xiangqi-multimodal" in families:
        d3, m2 = ( [record for _, record in rows[family]] for family in ("xiangqi-mate-in-one", "xiangqi-multimodal") )
        result["reasons"].extend(errors["xiangqi-mate-in-one"])
        try:
            result["reasons"].extend(audit_m2_mapping(d3, m2))
        except Exception as exc:
            result["reasons"].append(f"m2_mapping_error:{type(exc).__name__}:{exc}")
    if set(families) == set(XIANGQI_FAMILIES):
        by_fen = defaultdict(set)
        all_ids = []
        for family in families:
            for _, record in rows[family]:
                if isinstance(record.get("fen"), str):
                    by_fen[record["fen"]].add(family)
                else:
                    result["reasons"].append(f"{family}:invalid_fen_type")
                all_ids.append(str(record.get("id")))
        if len(all_ids) != len(set(all_ids)):
            result["reasons"].append("duplicate_task_ids_across_families")
        unexpected = [fen for fen, owners in by_fen.items()
                      if len(owners) > 1 and owners != {"xiangqi-mate-in-one", "xiangqi-multimodal"}]
        if unexpected:
            result["reasons"].append(f"unexpected_cross_family_fen_overlaps:{len(unexpected)}")
            result["unexpected_cross_family_fens"] = unexpected
    if "xiangqi-rule-variants" in families:
        source_audit = audit_c2_sources([record for _, record in rows["xiangqi-rule-variants"]], data_root)
        result["c2_sources"] = source_audit
        result["reasons"].extend("c2_sources:" + why for why in source_audit["reasons"])
    else:
        source_audit = {"inputs": {}}
    frozen = audit_frozen_samples(data_root, {family: [record for _, record in values] for family, values in rows.items()}, inputs)
    result["frozen_samples"] = frozen
    if frozen["status"] == "checked" and not frozen["valid"]:
        result["reasons"].extend("frozen_samples:" + why for why in frozen["reasons"])
    result["full"] = result["full"] and frozen["valid"] and frozen["complete"]
    audited_inputs = {**inputs,
        **{"frozen_samples/" + key: info for key, info in frozen["inputs"].items()},
        **{"c2_sources/" + key: info for key, info in source_audit["inputs"].items()}}
    for family, info in audited_inputs.items():
        try:
            if _hash(Path(info["path"])) != info["sha256"]:
                result["reasons"].append(f"{family}:input_changed_during_audit")
        except OSError:
            result["reasons"].append(f"{family}:input_disappeared_during_audit")
    result["valid"] = not result["reasons"] and all(item["valid"] for item in result["families"].values())
    result["release_ready"] = result["valid"] and result["full"]
    return result


def _family_arg(value: str) -> str:
    value = ALIASES.get(value.lower(), value)
    if value not in XIANGQI_FAMILIES:
        raise argparse.ArgumentTypeError(f"unknown family {value!r}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/xiangqi")
    parser.add_argument("--families", type=_family_arg, nargs="+", default=list(XIANGQI_FAMILIES))
    parser.add_argument("--skip-search", action="store_true", help="Skip C2 depth-three utility comparisons; never certifies a full release.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Write the complete per-record report and input/code hashes.")
    args = parser.parse_args()
    report = audit_release(args.data_root, families=args.families, skip_search=args.skip_search, workers=args.workers)
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps({"valid": report["valid"], "full": report["full"], "release_ready": report["release_ready"],
        "reasons": report["reasons"],
        "frozen_samples": {key: report.get("frozen_samples", {}).get(key) for key in ("status", "valid", "complete", "reasons")},
        "families": {family: {key: values[key] for key in (
            "valid", "record_count", "failed_record_count", "reasons")} for family, values in report["families"].items()},
        "report": str(args.output) if args.output else None}, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

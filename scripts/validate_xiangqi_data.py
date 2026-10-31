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
    shortest = record.get("shortest_mate_moves", moves)
    expected_band = "short" if shortest <= 3 else "medium" if shortest <= 5 else "long"
    if record.get("difficulty") != expected_band:
        reasons.append("reference_distance_band_mismatch")
    if (analysis.get("reference_mate_plies") != plies
            or ("shortest_proof" not in record and record["oracle"].get("mate_in_plies") != plies)):
        reasons.append("reference_mate_plies_mismatch")
    horizon = 2*shortest-1
    allowed_horizons = ({horizon} if "shortest_proof" in record else {horizon, horizon+4})
    if record.get("max_plies") not in allowed_horizons:
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
        if "shortest_proof" not in record and record["oracle"].get("best_move_uci") != runs[-1].get("best_move_uci"):
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
    for move in production_board.legal_moves(side):
        trial = production_board.copy()
        trial.apply(move)
        if (trial.find_general(-side) is not None and trial._is_in_check(-side)
                and not trial.legal_moves(-side)):
            reasons.append("c2_active_rule_has_mate_in_one")
            break
    analysis = record.get("c2_analysis", {})
    pv = analysis.get("principal_variation_uci", [])
    reference_plies = analysis.get("reference_mate_plies")
    if analysis.get("search_depth") != 3:
        reasons.append("c2_requires_registered_depth_three_policy")
    expected_horizon = (2*record["shortest_mate_moves"]-1
                        if "shortest_mate_moves" in record
                        else reference_plies+analysis.get("max_plies_extension", 4))
    if (not isinstance(reference_plies, int) or reference_plies < 3
            or record.get("max_plies") != expected_horizon):
        reasons.append("c2_invalid_multi_ply_horizon")
    if not pv or ("shortest_proof" not in record and record["oracle"].get("best_move_uci") != pv[0]):
        reasons.append("c2_oracle_reference_line_mismatch")

    from minibench.datasets.xiangqi.variants.mate_search import replay_strict_mate_line
    production_proof = replay_strict_mate_line(
        production_board.copy(), side, pv, attacker_side=side
    )
    production_state = production_board.copy()
    state = [row[:] for row in board]
    turn = side
    independent_reason = None
    legal_counts = []
    production_legal_counts = []
    for index, uci in enumerate(pv):
        production_moves = {
            move.to_uci(): move for move in production_state.legal_moves(turn)
        }
        independent_moves = {
            move.to_uci(): move for move in reference.legal_moves(state, turn, active)
        }
        production_legal_counts.append(len(production_moves))
        legal_counts.append(len(independent_moves))
        if set(production_moves) != set(independent_moves):
            reasons.append(f"c2_active_rule_legal_moves_disagree_at_ply:{index}")
        production_move = production_moves.get(uci)
        independent_move = independent_moves.get(uci)
        if production_move is None or independent_move is None:
            independent_reason = f"illegal_ply_{index}:{uci}"
            break
        production_state.apply(production_move)
        state = reference.apply_move(state, independent_move)
        if production_state.board != state:
            reasons.append(f"c2_active_rule_board_disagrees_at_ply:{index}")
        turn = -turn
    independent_terminal = (
        reference.terminal_status(state, turn, active)
        if independent_reason is None else "ongoing"
    )
    if (not production_proof["valid"]
            or production_proof["terminal"] != "checkmate"
            or production_proof["plies"] != reference_plies):
        reasons.append("c2_production_reference_line_not_strict_mate")
    if independent_reason is not None or independent_terminal != "checkmate" or turn != -side:
        reasons.append("c2_independent_reference_line_not_strict_mate")

    focus = analysis.get("focus_ruleset")
    recomputed_sensitive = False
    sensitivity_replay_valid = focus in FOCI
    if sensitivity_replay_valid:
        standard_state = VariantBoard(board, [])
        focus_rules = internal_rules({"ruleset": focus})
        focus_state = VariantBoard(
            board, [Rule.from_dict(rule) for rule in focus_rules]
        )
        sensitivity_turn = side
        for index, uci in enumerate(pv):
            standard_moves = {
                move.to_uci(): move
                for move in standard_state.legal_moves(sensitivity_turn)
            }
            focus_moves = {
                move.to_uci(): move
                for move in focus_state.legal_moves(sensitivity_turn)
            }
            recomputed_sensitive = (
                recomputed_sensitive or set(standard_moves) != set(focus_moves)
            )
            if uci not in standard_moves or uci not in focus_moves:
                # A rule-specific mating line may be illegal under the other
                # condition. That is rule impact, not an invalid paired puzzle.
                recomputed_sensitive = True
                break
            standard_state.apply(standard_moves[uci])
            focus_state.apply(focus_moves[uci])
            sensitivity_turn = -sensitivity_turn
    if (
        sensitivity_replay_valid
        and analysis.get("rule_sensitive_along_reference")
        != recomputed_sensitive
    ):
        reasons.append("c2_rule_sensitive_label_mismatch")
    stored = analysis.get("verification", {})
    for key in ("production", "independent"):
        proof = stored.get(key, {}) if isinstance(stored, dict) else {}
        if (proof.get("valid") is not True or proof.get("terminal") != "checkmate"
                or proof.get("plies") != reference_plies):
            reasons.append(f"c2_stored_{key}_proof_invalid")
    details = {
        "initial_legal_move_count": len(independent),
        "reference_mate_plies": reference_plies,
        "reference_line": pv,
        "production_legal_counts_along_reference": production_legal_counts,
        "legal_counts_along_reference": legal_counts,
        "production_replay": production_proof,
        "independent_terminal": independent_terminal,
        "recomputed_rule_sensitive_along_reference": recomputed_sensitive,
        "claim_limit": "Validated complete mating line; not a shortest-mate or all-defences proof.",
    }
    if skip_search:
        details["skipped"] = "none; C2 full-line replay is mandatory and inexpensive"
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
    allowed_sizes = (100, 200, 250) if family == "xiangqi-rule-variants" else (100, 250)
    if len(records) not in allowed_sizes:
        reasons.append(f"unexpected_record_count:{len(records)}")
    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)):
        reasons.append("duplicate_task_ids")
    fens = [record.get("fen") for record in records]
    if family != "xiangqi-rule-variants" and len(fens) != len(set(fens)):
        reasons.append("duplicate_fens_within_family")
    counts = Counter(record.get("difficulty") for record in records)
    if len(records) == 100:
        new_shortest = family == "xiangqi-history" and all("shortest_proof" in r for r in records)
        expected = ({"short":50,"medium":50} if new_shortest else
                    {"short":34,"medium":33,"long":33} if family == "xiangqi-history" else
                    {"easy":34,"medium":33,"hard":33})
    else:
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
        pairs, strata = 0, Counter()
        fen_scenarios = defaultdict(set)
        source_scenarios = defaultdict(set)
        within_five = 0
        sensitive = Counter()
        for scenario, group in groups.items():
            for record in group:
                fen_scenarios[record.get("fen")].add(scenario)
                source_scenarios[record.get("source_task_id")].add(scenario)
                if record.get("c2_analysis", {}).get("mate_in_moves", 99) <= 5:
                    within_five += 1
            pairs += 1
            by_rule = {record.get("ruleset"): record for record in group}
            labels = {record.get("design_stratum") for record in group}
            if len(labels) != 1 or not labels <= set(FOCI):
                reasons.append(f"{scenario}:invalid_design_stratum")
                continue
            focus = next(iter(labels))
            if (len(group) != 2 or set(by_rule) != {"standard", focus}
                    or len({record.get("fen") for record in group}) != 1
                    or len({record.get("source_task_id") for record in group}) != 1
                    ):
                reasons.append(f"{scenario}:incomplete_or_mismatched_standard_focus_pair")
                continue
            strata[focus] += 1
            if by_rule[focus].get("c2_analysis", {}).get("rule_sensitive_along_reference"):
                sensitive[focus] += 1
            for record in group:
                analysis = record.get("c2_analysis", {})
                if (analysis.get("focus_ruleset") != focus
                        or analysis.get("pair_rulesets") != ["standard", focus]
                        or analysis.get("source_task_id") != record.get("source_task_id")):
                    reasons.append(f"{record.get('id')}:paired_metadata_mismatch")
        target_pairs = len(records) // 2
        if pairs != target_pairs:
            reasons.append(f"c2_requires_{target_pairs}_standard_focus_pairs_got_{pairs}")
        expected_strata = ({"horse-no-leg-block": 17, "chariot-no-center": 17,
                            "soldier-free-retreat": 16} if len(records) == 100 else
                           dict(strata) if len(records) == 200 else
                           {"horse-no-leg-block": 42, "chariot-no-center": 42,
                            "soldier-free-retreat": 41})
        if dict(strata) != expected_strata:
            reasons.append(f"c2_focus_counts_mismatch:{dict(strata)}")
        if len(records) == 200:
            if set(strata) != set(FOCI) or any(not 20 <= strata[focus] <= 40 for focus in FOCI):
                reasons.append(f"c2_each_rule_requires_20_to_40_pairs:{dict(strata)}")
            baseline = Counter(r.get("difficulty") for r in records if r.get("ruleset") == "standard")
            if dict(baseline) != {"short": 50, "medium": 50}:
                reasons.append(f"c2_standard_requires_50_short_50_medium:{dict(baseline)}")
        if within_five <= len(records) // 2:
            reasons.append(f"c2_requires_majority_within_five_moves:{within_five}")
        if any(sensitive[focus] < (4 if len(records) == 100 else 8) for focus in FOCI):
            reasons.append(f"c2_insufficient_rule_sensitive_reference_lines:{dict(sensitive)}")
        if any(len(scenarios) != 1 for scenarios in fen_scenarios.values()):
            reasons.append("c2_same_fen_reused_across_scenarios")
        if any(len(scenarios) != 1 for scenarios in source_scenarios.values()):
            reasons.append("c2_h2_source_reused_across_scenarios")
    return reasons


def audit_c2_pair_impact(records, data_root):
    from minibench.datasets.xiangqi.rule_effects import paired_impact
    groups = defaultdict(list)
    reasons, counts = [], Counter()
    for row in records:
        groups[row["scenario_id"]].append(row)
    for scenario, pair in groups.items():
        if len(pair) != 2 or not any("paired_rule_impact" in row for row in pair):
            continue
        try:
            standard = next(r for r in pair if r["ruleset"] == "standard")
            variant = next(r for r in pair if r["ruleset"] != "standard")
            trees = [json.loads((data_root / "solution_trees/c2" / f"{row['id']}.json").read_text()) for row in (standard, variant)]
            impact = paired_impact(*trees)
            if any(row.get("paired_rule_impact") != impact for row in pair):
                reasons.append(f"{scenario}:paired_rule_impact_mismatch")
            counts.update(key for key, value in impact.items() if value is True)
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            reasons.append(f"{scenario}:paired_rule_impact_unavailable:{type(exc).__name__}")
    return {"valid": not reasons, "reasons": reasons, "impact_counts": dict(counts)}


def audit_sampling(family: str, records: Sequence[dict]) -> dict[str, Any]:
    reasons, probes = [], []
    sizes = (1, 10, len(records)//2) if family == "xiangqi-rule-variants" else (1, 30, len(records))
    for count in sizes:
        try:
            first = sample_records(records, count=count, seed=42)
            repeated = sample_records(records, count=count, seed=42)
            ids = [item["id"] for item in first]
            if ids != [item["id"] for item in repeated] or len(ids) != len(set(ids)):
                reasons.append(f"sample_{count}:not_reproducible_or_duplicate_ids")
            expected_size = count * 2 if family == "xiangqi-rule-variants" else count
            if len(ids) != expected_size or not set(ids) <= {item["id"] for item in records}:
                reasons.append(f"sample_{count}:wrong_membership_or_size")
            if family == "xiangqi-rule-variants":
                grouped = defaultdict(list)
                for record in first:
                    grouped[record["scenario_id"]].append(record)
                if len(grouped) != count or any(
                    len(group) != 2
                    or {item["ruleset"] for item in group}
                       != {"standard", group[0]["design_stratum"]}
                    or len({item["fen"] for item in group}) != 1
                    for group in grouped.values()
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
    history_path = data_root / FAMILY_DIRS["xiangqi-history"] / "tasks.jsonl"
    try:
        raw = history_path.read_bytes()
        report["inputs"]["history/tasks.jsonl"] = {
            "path": str(history_path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        history = {
            row["id"]: row
            for row in (json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip())
        }
        evidence_path = data_root / "sources/c2_h2_origins.jsonl"
        if evidence_path.is_file():
            evidence_raw = evidence_path.read_bytes()
            report["inputs"]["sources/c2_h2_origins.jsonl"] = {
                "path": str(evidence_path.resolve()), "sha256": hashlib.sha256(evidence_raw).hexdigest()}
            seen = set()
            for line in evidence_raw.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                source = json.loads(line)
                validate_record(source, expected_family="xiangqi-history")
                if source["id"] in seen or (source["id"] in history and history[source["id"]] != source):
                    raise ValueError("duplicate or inconsistent C2 source evidence")
                seen.add(source["id"])
                history[source["id"]] = source
    except Exception as exc:
        report["reasons"].append(f"history_source_read_error:{type(exc).__name__}:{exc}")
        return report

    scenarios_by_source = defaultdict(set)
    for record in records:
        result = {"id": record.get("id"), "valid": False, "reasons": []}
        report["records"].append(result)
        reasons = result["reasons"]
        try:
            validate_record(record,expected_family="xiangqi-rule-variants")
        except (ValueError,TypeError,KeyError) as exc:
            reasons.append(f"record_validation_failed:{type(exc).__name__}:{exc}")
        source_task_id = record.get("source_task_id")
        source = history.get(source_task_id)
        if source is None:
            reasons.append("unknown_h2_source_task_id")
        else:
            scenarios_by_source[source_task_id].add(record.get("scenario_id"))
            expected_pv = source.get("h2_analysis", {}).get("principal_variation_uci")
            analysis = record.get("c2_analysis", {})
            mapped_keys = ("fen", "agent_color", "piece_count", "source", "source_id")
            if "shortest_proof" not in record:
                mapped_keys += ("difficulty",)
            for key in mapped_keys:
                if record.get(key) != source.get(key):
                    reasons.append(f"h2_{key}_mapping_mismatch")
            if "shortest_proof" not in record and analysis.get("principal_variation_uci") != expected_pv:
                reasons.append("h2_principal_variation_mapping_mismatch")
            if "shortest_proof" not in record and record.get("oracle", {}).get("best_move_uci") != source.get("oracle", {}).get("best_move_uci"):
                reasons.append("h2_bestmove_mapping_mismatch")
            if "shortest_proof" not in record and record.get("oracle", {}).get("mate_in_plies") != source.get("oracle", {}).get("mate_in_plies"):
                reasons.append("h2_reference_distance_mapping_mismatch")
        result["valid"] = not reasons
        report["reasons"].extend(f"{record.get('id')}:{why}" for why in reasons)
    if any(len(scenarios) != 1 for scenarios in scenarios_by_source.values()):
        report["reasons"].append("h2_source_reused_across_c2_scenarios")
    report["source_count"] = len(scenarios_by_source)
    expected_source_count = len({r.get("scenario_id") for r in records})
    if report["source_count"] != expected_source_count:
        report["reasons"].append(f"expected_{expected_source_count}_unique_h2_sources_got_{report['source_count']}")
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
        if manifest.get("protocol_version") not in {"xiangqi-reasoning-v2", "xiangqi-shortest-tree-v1", "xiangqi-shortest-tree-strict-n-v2"}:
            reasons.append("frozen_protocol_version_mismatch")
        if set(manifest.get("samples", {})) != {"smoke", "formal"}:
            reasons.append("frozen_requires_smoke_and_formal_samples")
        sources = {family: {record["id"]: record for record in rows} for family, rows in datasets.items()}
        formal_new = manifest.get("protocol_version") in {"xiangqi-shortest-tree-v1", "xiangqi-shortest-tree-strict-n-v2"}
        for size, seed, count, c2count in (("smoke",20260909,6,3),
                                          ("formal",42,100 if formal_new else 30,len(datasets[ALIASES["c2"]])//2 if formal_new else 10)):
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
            if size=="formal" and formal_new:
                expected={key:list(datasets[ALIASES[key]]) for key in ("d3","c2","h2")}
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
                    required_count = c2count * 2 if key == "c2" else count
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
                            len(group) != 2
                            or {row.get("ruleset") for row in group}
                               != {"standard", group[0].get("design_stratum")}
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
    if "xiangqi-rule-variants" in needed and "xiangqi-history" not in needed:
        needed.append("xiangqi-history")
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
        "static_data_ready": False, "runtime_policy_certified": None,
        "experiment_ready": False, "release_ready": False,
        "skip_search": skip_search, "families_requested": list(families),
        "inputs": inputs, "independent_calibration": calibrate_cchess(), "families": {}, "reasons": [],
        "validator_hashes": {str(path.relative_to(ROOT)): _hash(path) for path in (
            Path(__file__).resolve(), ROOT / "scripts/extract_ccpd_positions.py", ROOT / "src/minibench/datasets/xiangqi/reference.py",
            ROOT / "src/minibench/datasets/xiangqi/validation.py", ROOT / "src/minibench/datasets/xiangqi/schema.py",
            ROOT / "src/minibench/datasets/xiangqi/variants/board.py", ROOT / "src/minibench/datasets/xiangqi/variants/search.py",
            ROOT / "src/minibench/datasets/xiangqi/variants/mate_search.py",
            ROOT / "src/minibench/datasets/xiangqi/variants/certification.py",
            ROOT / "src/minibench/datasets/xiangqi/solution_tree.py",
            ROOT / "src/minibench/datasets/xiangqi/tree_verification.py",
            ROOT / "src/minibench/datasets/xiangqi/tree_certificate.py",
            ROOT / "src/minibench/datasets/xiangqi/rule_effects.py",
            ROOT / "scripts/verify_xiangqi_mate.py",
            ROOT / "src/minibench/datasets/xiangqi/rule_variants.py")},
        "limitations": ["PV checks validate the supplied lines, not all possible defences or shortest mate distance.",
                        "Structural position constraints do not independently prove game-history reachability."]}
    if not result["independent_calibration"]["valid"]:
        result["reasons"].append("cchess_calibration_failed")
        return result
    for family in families:
        records = [record for _, record in rows[family]]
        family_reasons = list(errors[family])
        try:
            family_reasons.extend(audit_collection(family, records))
            if family == "xiangqi-rule-variants":
                family_reasons.extend(audit_c2_pair_impact(records, data_root)["reasons"])
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
        allowed_overlaps = (
            {"xiangqi-mate-in-one", "xiangqi-multimodal"},
            {"xiangqi-history", "xiangqi-rule-variants"},
        )
        unexpected = [fen for fen, owners in by_fen.items()
                      if len(owners) > 1 and owners not in allowed_overlaps]
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
    result["static_data_ready"] = result["valid"] and result["full"]
    if "xiangqi-history" in families:
        history_records=[r for _,r in rows["xiangqi-history"]]
        if history_records and all("shortest_proof" in r for r in history_records):
            checked=audit_shortest_certificates(data_root,history_records,"h2")
            result["h2_shortest_certificates"]=checked
            if not checked["valid"]:
                result["reasons"].append("h2_shortest_certification_failed")
                result["valid"]=result["static_data_ready"]=False
    if "xiangqi-rule-variants" in families:
        certificate = audit_runtime_certificates(data_root, [r for _, r in rows["xiangqi-rule-variants"]], workers=workers)
        result["c2_runtime_certificates"] = certificate
        result["runtime_policy_certified"] = certificate["valid"]
        if not certificate["valid"]:
            result["limitations"].append("C2 runtime-policy certification failed or is absent: " + "; ".join(certificate["reasons"]))
            if rows["xiangqi-rule-variants"] and all(
                record.get("shortest_proof") for _, record in rows["xiangqi-rule-variants"]
            ):
                result["reasons"].append("c2_independent_shortest_certification_failed")
                result["valid"] = result["static_data_ready"] = False
    result["experiment_ready"] = (
        result["static_data_ready"]
        and result["runtime_policy_certified"] is not False
    )
    result["release_ready"] = result["experiment_ready"]
    return result


def _certificate_payload(payload):
    from minibench.datasets.xiangqi.variants.certification import verify_winning_line
    record, line = payload
    return record["id"], verify_winning_line(record, line)


def audit_shortest_certificates(data_root,records,folder):
    from minibench.datasets.xiangqi.solution_tree import validate_tree_record
    from minibench.datasets.xiangqi.tree_certificate import validate_saved_tree_certificate
    family_folder = "history" if folder == "h2" else "rule_variants"
    independent = validate_saved_tree_certificate(
        records, data_path=data_root/family_folder/"tasks.jsonl",
        tree_dir=data_root/"solution_trees"/folder)
    report={"valid":False,"reasons":list(independent["reasons"]),
            "record_count":len(records),"records":{},
            "independent_verification": independent,
            "policy":"content-bound independent shortest/branch proof plus positive replay; finite-depth fallback is not a proof"}
    for record in records:
        path=data_root/"solution_trees"/folder/f"{record['id']}.json"
        try:
            checked=validate_tree_record(record,path)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            checked={"valid":False,"reasons":[type(exc).__name__+":"+str(exc)]}
        report["records"][record["id"]]=checked
        if not checked["valid"]:
            report["reasons"].append(f"{record['id']}:tree_certificate_failed")
    report["valid"]=not report["reasons"]
    return report


def audit_runtime_certificates(data_root, records, *, workers=4):
    if records and all("shortest_proof" in r for r in records):
        return audit_shortest_certificates(data_root,records,"c2")
    from minibench.datasets.xiangqi.variants.certification import certificate_header_valid
    path = data_root / "rule_variants/runtime_certificates.json"
    report = {"valid": False, "reasons": [], "record_count": len(records), "records": {}}
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        report["sha256"] = _hash(path)
        digest = _hash(data_root / "rule_variants/tasks.jsonl")
        if not certificate_header_valid(bundle, records, digest):
            report["reasons"].append("certificate_header_or_inputs_mismatch")
            return report
        payloads = [(r, bundle["tasks"][r["id"]]["moves_uci"]) for r in records]
        if workers == 1:
            checked = map(_certificate_payload, payloads)
            report["records"] = dict(checked)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                report["records"] = dict(pool.map(_certificate_payload, payloads))
        if not all(v["valid"] for v in report["records"].values()):
            report["reasons"].append("one_or_more_runtime_witnesses_failed")
        report["valid"] = not report["reasons"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["reasons"].append(f"certificate_unavailable_or_invalid:{type(exc).__name__}")
    return report


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
    print(json.dumps({"valid": report["valid"], "full": report["full"],
        "static_data_ready": report["static_data_ready"],
        "runtime_policy_certified": report["runtime_policy_certified"],
        "experiment_ready": report["experiment_ready"], "release_ready": report["release_ready"],
        "reasons": report["reasons"],
        "frozen_samples": {key: report.get("frozen_samples", {}).get(key) for key in ("status", "valid", "complete", "reasons")},
        "families": {family: {key: values[key] for key in (
            "valid", "record_count", "failed_record_count", "reasons")} for family, values in report["families"].items()},
        "report": str(args.output) if args.output else None}, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

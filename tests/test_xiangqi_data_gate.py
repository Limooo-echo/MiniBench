from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.datasets.xiangqi.schema import (
    FAMILY_PATHS, RULESETS, fen_to_board, load_records, sample_records,
)
from minibench.datasets.xiangqi.validation import validate_pv
from minibench.datasets.xiangqi.variants.board import VariantBoard
from minibench.datasets.xiangqi.variants.search import score_moves
from scripts.validate_xiangqi_data import (
    ALIASES, CCPD_REPOSITORY, CCPD_REVISION, FAMILY_DIRS, RELEASE_ID, _parse_source_pgn,
    audit_c2_sources, audit_collection, audit_frozen_samples, audit_m2_mapping, audit_record, audit_release, main,
)

HAS_CCHESS = importlib.util.find_spec("cchess") is not None


def d3_fixture():
    return {"schema_version": 2, "release_id": RELEASE_ID,
        "id": "xiangqi-mate-in-one-fixture", "source_id": "fixture:mate", "family": "xiangqi-mate-in-one",
        "fen": "3k5/1R7/1N2R4/9/9/9/9/9/9/5K3 w - - 0 1", "agent_color": "red",
        "goal": "checkmate", "max_plies": 1, "difficulty": "easy", "piece_count": 5, "tags": [],
        "oracle": {"best_move_uci": "b8b9", "mate_in_plies": 1, "evaluation_cp": None},
        "d3_analysis": {"version": "d3-structural-v2", "legal_move_count": 30,
            "mate_moves_uci": ["b8b9", "b8d8"], "mate_move_count": 2,
            "non_mating_check_count": 2, "piece_count": 5, "difficulty_score": 0.0,
            "difficulty_factors": {
                "choice_pressure": 28 / 30, "near_miss_rate": 0.5, "state_load": 5,
                "choice_pressure_percentile": 0.0, "near_miss_rate_percentile": 0.0,
                "state_load_percentile": 0.0}}}


def h2_fixture():
    fen = "3a1k3/4a4/4C3b/p5R1p/1n7/9/P3P3P/7R1/4K4/2rC1rBc1 b - - 0 1"
    pv = ["f0e0", "e1d1", "c0d0"]
    proof = validate_pv(fen, pv)
    board, _ = fen_to_board(fen)
    runs = [{"requested_depth": depth, "actual_depth": depth, "score_kind": "mate", "score": 2,
             "score_is_bound": False, "best_move_uci": pv[0], "principal_variation_uci": list(pv),
             "pv_validation": dict(proof)} for depth in (16, 20)]
    return {"schema_version": 2, "release_id": RELEASE_ID, "family": "xiangqi-history",
        "id": "xiangqi-history-fixture", "fen": fen, "agent_color": "black", "goal": "checkmate",
        "max_plies": 7, "difficulty": "short", "piece_count": sum(bool(v) for row in board for v in row),
        "tags": [], "source": {"file": "fixture.pgn"}, "source_id": "ccpd:fixture.pgn",
        "oracle": {"best_move_uci": pv[0], "mate_in_plies": 3, "evaluation_cp": None},
        "h2_analysis": {"version": "h2-pikafish-double-depth-v2",
            "mate_in_moves": 2, "reference_mate_plies": 3, "requested_depths": [16, 20],
            "verification_depths": [16, 20], "principal_variation_uci": list(pv), "validation_runs": runs,
            "source_reused": False, "interpretation": "fixture reference lines"}}


def c2_fixture():
    return deepcopy(load_records(FAMILY_PATHS["xiangqi-rule-variants"])[0])



def freeze_fixture(root):
    """Small canonical pools for roster tests; chess semantics are tested above."""
    data = {family: [] for family in ALIASES.values()}
    for number in range(30):
        d3 = d3_fixture()
        d3.update(id=f"d3-{number:02}", difficulty=("easy", "medium", "hard")[number % 3])
        m2 = deepcopy(d3)
        m2.update(id=f"m2-{number:02}", family=ALIASES["m2"], source_task_id=d3["id"])
        m2["m2_analysis"] = {"mate_moves_uci": d3["d3_analysis"]["mate_moves_uci"]}
        h2 = {"id": f"h2-{number:02}", "family": ALIASES["h2"], "difficulty": ("short", "medium", "long")[number % 3]}
        data[ALIASES["d3"]].append(d3)
        data[ALIASES["m2"]].append(m2)
        data[ALIASES["h2"]].append(h2)
    foci = [name for name in RULESETS if name != "standard"]
    for number in range(12):
        focus = foci[number % 3]
        for rule in ("standard", focus):
            data[ALIASES["c2"]].append({"id": f"c2-{number:02}-{rule}", "family": ALIASES["c2"],
                "scenario_id": f"scenario-{number:02}", "ruleset": rule, "fen": f"position-{number}",
                "design_stratum": focus})
    inputs = {}
    for family, rows in data.items():
        path = root / FAMILY_DIRS[family] / "tasks.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        inputs[family] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"release_id": RELEASE_ID, "protocol_version": "xiangqi-reasoning-v2", "samples": {}}
    for size, seed, count, pairs in (("smoke", 20260909, 6, 3), ("formal", 42, 30, 10)):
        selected = {key: sample_records(data[ALIASES[key]], count=pairs if key == "c2" else count, seed=seed)
                    for key in ("d3", "c2", "h2")}
        m2_sources = {row["source_task_id"]: row for row in data[ALIASES["m2"]]}
        selected["m2"] = [m2_sources[row["id"]] for row in selected["d3"]]
        sample = {"seed": seed, "tasks": {}}
        for key, rows in selected.items():
            path = root / "evaluation_samples" / size / f"{key}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            sample["tasks"][key] = {"path": f"data/xiangqi/evaluation_samples/{size}/{key}.jsonl",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "dataset_sha256": inputs[ALIASES[key]]["sha256"],
                "record_count": len(rows), "task_ids": [row["id"] for row in rows]}
        manifest["samples"][size] = sample
    (root / "evaluation_samples/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return data, inputs, manifest


class FrozenXiangqiSampleTests(unittest.TestCase):
    def test_actual_frozen_rosters_and_hashes_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, inputs, _ = freeze_fixture(root)
            report = audit_frozen_samples(root, data, inputs)
            self.assertTrue(report["valid"], report)
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["inputs"]), 9)
            self.assertEqual(report["samples"]["smoke"]["tasks"]["c2"]["scenario_count"], 3)
            self.assertEqual(report["samples"]["formal"]["tasks"]["c2"]["record_count"], 20)

    def test_sample_content_and_manifest_tampering_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, inputs, manifest = freeze_fixture(root)
            path = root / "evaluation_samples/smoke/d3.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["oracle"]["best_move_uci"] = "a0a1"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            # Updating the declared hash still cannot bless a changed answer.
            manifest["samples"]["smoke"]["tasks"]["d3"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest["samples"]["formal"]["tasks"]["h2"]["dataset_sha256"] = "0" * 64
            (root / "evaluation_samples/manifest.json").write_text(json.dumps(manifest))
            report = audit_frozen_samples(root, data, inputs)
            self.assertFalse(report["valid"])
            self.assertTrue(any("sample_record_differs_from_dataset" in reason for reason in report["reasons"]))
            self.assertIn("formal:h2:manifest_dataset_hash_mismatch", report["reasons"])

    def test_missing_pair_and_desynchronised_d3_m2_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, inputs, _ = freeze_fixture(root)
            path = root / "evaluation_samples/smoke/c2.jsonl"
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n")
            path = root / "evaluation_samples/formal/m2.jsonl"
            path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
            report = audit_frozen_samples(root, data, inputs)
            self.assertFalse(report["valid"])
            self.assertIn("smoke:c2:incomplete_or_mismatched_paired_scenarios", report["reasons"])
            self.assertIn("formal:d3_m2_sample_order_or_sources_differ", report["reasons"])

    def test_unfrozen_data_cannot_certify_even_if_chess_checks_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key, family in ALIASES.items():
                path = root / FAMILY_DIRS[family] / "tasks.jsonl"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"id": key, "fen": key, "family": family}) + "\n")
            with patch("scripts.validate_xiangqi_data.calibrate_cchess", return_value={"valid": True}), \
                 patch("scripts.validate_xiangqi_data.audit_collection", return_value=[]), \
                 patch("scripts.validate_xiangqi_data.audit_sampling", return_value={"valid": True, "reasons": []}), \
                 patch("scripts.validate_xiangqi_data._audit_payload", return_value={"valid": True}), \
                 patch("scripts.validate_xiangqi_data.audit_m2_mapping", return_value=[]),                  patch("scripts.validate_xiangqi_data.audit_c2_sources", return_value={"valid": True, "reasons": [], "inputs": {}}):
                report = audit_release(root)
            self.assertTrue(report["valid"], report)
            self.assertFalse(report["full"])
            self.assertFalse(report["static_data_ready"])
            self.assertFalse(report["runtime_policy_certified"])
            self.assertFalse(report["experiment_ready"])
            self.assertFalse(report["release_ready"])
            self.assertEqual(report["frozen_samples"]["status"], "not_frozen")




def c2_source_fixture(root):
    records = load_records(FAMILY_PATHS["xiangqi-rule-variants"])
    scenario = records[0]["scenario_id"]
    pair = [deepcopy(record) for record in records if record["scenario_id"] == scenario]
    source_id = pair[0]["source_task_id"]
    source = next(
        deepcopy(record)
        for record in (load_records(FAMILY_PATHS["xiangqi-history"])
                       + load_records(Path("data/xiangqi/sources/c2_h2_origins.jsonl")))
        if record["id"] == source_id
    )
    path = root / "history/tasks.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(source) + "\n", encoding="utf-8")
    return pair, source


class XiangqiSourceGateTests(unittest.TestCase):
    def test_c2_rows_map_exactly_to_one_audited_h2_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair, _ = c2_source_fixture(root)
            report = audit_c2_sources(pair, root)
            # Source mapping is independent of the collection-size gate.
            self.assertTrue(report["valid"], report)
            self.assertEqual(report["source_count"], 1)
            self.assertTrue(all(item["valid"] for item in report["records"]))
            self.assertEqual(report["reasons"], [])

    def test_source_evidence_sidecar_is_verified_and_duplicates_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair, source = c2_source_fixture(root)
            (root / 'history/tasks.jsonl').write_text('', encoding='utf-8')
            evidence = root / 'sources/c2_h2_origins.jsonl'
            evidence.parent.mkdir(parents=True)
            row = json.dumps(source) + '\n'
            evidence.write_text(row, encoding='utf-8')
            report = audit_c2_sources(pair, root)
            self.assertTrue(report['valid'], report)
            self.assertIn('sources/c2_h2_origins.jsonl', report['inputs'])
            evidence.write_text(row + row, encoding='utf-8')
            report = audit_c2_sources(pair, root)
            self.assertFalse(report['valid'])
            self.assertTrue(any('duplicate or inconsistent C2 source evidence' in r for r in report['reasons']))

    def test_changed_position_or_reference_line_breaks_h2_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair, _ = c2_source_fixture(root)
            pair[0]["fen"] = pair[0]["fen"] + " "
            pair[1]["c2_analysis"]["principal_variation_uci"][0] = "a0a0"
            report = audit_c2_sources(pair, root)
            self.assertFalse(report["valid"])
            reasons = [why for item in report["records"] for why in item["reasons"]]
            self.assertIn("h2_fen_mapping_mismatch", reasons)
            self.assertTrue(any(why.startswith("record_validation_failed:") for why in reasons))

    def test_unknown_or_missing_h2_source_cannot_be_certified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair, _ = c2_source_fixture(root)
            pair[0]["source_task_id"] = "xiangqi-history-missing"
            report = audit_c2_sources(pair, root)
            self.assertFalse(report["valid"])
            self.assertIn("unknown_h2_source_task_id", report["records"][0]["reasons"])
            (root / "history/tasks.jsonl").unlink()
            report = audit_c2_sources(pair, root)
            self.assertFalse(report["valid"])
            self.assertTrue(any("history_source_read_error" in why for why in report["reasons"]))



@unittest.skipUnless(HAS_CCHESS, "install .[xiangqi-generation] to run the independent release gate")
class XiangqiDataGateTests(unittest.TestCase):
    def test_d3_gate_recomputes_complete_mate_set(self):
        record = d3_fixture()
        good = audit_record(record, record["family"])
        self.assertTrue(good["valid"], good)
        record["d3_analysis"]["mate_moves_uci"] = ["b8b9"]
        record["d3_analysis"]["mate_move_count"] = 1
        # Even internally consistent difficulty metadata cannot hide a missing
        # true mating move from the independent board enumeration.
        factors = record["d3_analysis"]["difficulty_factors"]
        factors["choice_pressure"] = 29 / 30
        factors["near_miss_rate"] = 2 / 3
        bad = audit_record(record, record["family"])
        self.assertFalse(bad["valid"])
        self.assertIn("stored_mate_set_disagrees_with_independent_enumeration", bad["reasons"])

    def test_h2_gate_replays_both_depths_instead_of_trusting_stored_flags(self):
        record = h2_fixture()
        good = audit_record(record, record["family"])
        self.assertTrue(good["valid"], good)
        record["h2_analysis"]["validation_runs"][1]["principal_variation_uci"].pop()
        bad = audit_record(record, record["family"])
        self.assertFalse(bad["valid"])
        self.assertIn("depth_20:pv_not_checkmate:ongoing", bad["reasons"])
        self.assertIn("depth_20:reference_length_mismatch", bad["reasons"])

    def test_c2_gate_replays_the_complete_rule_aware_mating_line(self):
        record = c2_fixture()
        good = audit_record(record, record["family"])
        self.assertTrue(good["valid"], good)
        self.assertEqual(good["details"]["production_replay"]["terminal"], "checkmate")
        self.assertEqual(
            good["details"]["production_legal_counts_along_reference"],
            good["details"]["legal_counts_along_reference"],
        )
        record["c2_analysis"]["principal_variation_uci"][-1] = "a0a0"
        bad = audit_record(record, record["family"])
        self.assertFalse(bad["valid"])
        self.assertTrue(any("replay" in reason or "schema" in reason for reason in bad["reasons"]))
        light = audit_record(record, record["family"], skip_search=True)
        self.assertFalse(light["valid"])

    def test_c2_gate_recomputes_rule_sensitivity_instead_of_trusting_label(self):
        record = c2_fixture()
        record["c2_analysis"]["rule_sensitive_along_reference"] = not record[
            "c2_analysis"
        ]["rule_sensitive_along_reference"]
        bad = audit_record(record, record["family"])
        self.assertFalse(bad["valid"])
        self.assertIn("c2_rule_sensitive_label_mismatch", bad["reasons"])

    def test_m2_mapping_rejects_changes_to_position_or_answer(self):
        source = d3_fixture()
        modal = deepcopy(source)
        modal.update(family="xiangqi-multimodal", id="xiangqi-multimodal-fixture", source_task_id=source["id"])
        modal["m2_analysis"] = {"mate_moves_uci": source["d3_analysis"]["mate_moves_uci"]}
        self.assertEqual(audit_m2_mapping([source], [modal]), [])
        modal["fen"] = modal["fen"].replace("5K3", "4K4")
        self.assertTrue(any("fen_mapping_mismatch" in why for why in audit_m2_mapping([source], [modal])))

    def test_collection_rejects_reused_sources_and_scenario_fens(self):
        source = h2_fixture()
        other = deepcopy(source)
        other["id"] += "-2"
        self.assertIn("h2_source_games_must_be_unique", audit_collection(source["family"], [source, other]))
        all_c2 = load_records(FAMILY_PATHS["xiangqi-rule-variants"])
        first_id = all_c2[0]["scenario_id"]
        pair = [deepcopy(item) for item in all_c2 if item["scenario_id"] == first_id]
        duplicate = deepcopy(pair)
        for item in duplicate:
            item["id"] += "-duplicate"
            item["scenario_id"] += "-duplicate"
        reasons = audit_collection(pair[0]["family"], pair + duplicate)
        self.assertIn("c2_same_fen_reused_across_scenarios", reasons)

    def test_partial_gate_records_hashes_and_cannot_certify_release(self):
        record = c2_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rule_variants").mkdir()
            path = root / "rule_variants/tasks.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            report = audit_release(root, families=(record["family"],), skip_search=True)
            self.assertFalse(report["full"])
            self.assertFalse(report["static_data_ready"])
            self.assertFalse(report["runtime_policy_certified"])
            self.assertFalse(report["experiment_ready"])
            self.assertFalse(report["release_ready"])
            self.assertEqual(len(report["inputs"][record["family"]]["sha256"]), 64)
            self.assertEqual(report["families"][record["family"]]["record_count"], 1)
            with patch("sys.argv", ["validate_xiangqi_data.py", "--data-root", str(root), "--families", "c2", "--skip-search", "--workers", "1"]), redirect_stdout(StringIO()):
                self.assertEqual(main(), 1)

    def test_malformed_record_is_reported_without_aborting_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "history").mkdir()
            (root / "history/tasks.jsonl").write_text('{"id": [], "fen": [], "source": "bad"}\n', encoding="utf-8")
            report = audit_release(root, families=("xiangqi-history",))
            self.assertFalse(report["valid"])
            self.assertEqual(report["families"]["xiangqi-history"]["failed_record_count"], 1)


if __name__ == "__main__":
    unittest.main()

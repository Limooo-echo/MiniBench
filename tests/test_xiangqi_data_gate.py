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

from minibench.datasets.xiangqi.schema import RULESETS, fen_to_board, sample_records
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
            "difficulty_factors": {key: 0.0 for key in (
                "choice_pressure", "near_miss_rate", "state_load", "choice_pressure_percentile",
                "near_miss_rate_percentile", "state_load_percentile")}}}


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
        "max_plies": 5, "difficulty": "short", "piece_count": sum(bool(v) for row in board for v in row),
        "tags": [], "source": {"file": "fixture.pgn"}, "source_id": "ccpd:fixture.pgn",
        "oracle": {"best_move_uci": pv[0], "mate_in_plies": 3, "evaluation_cp": None},
        "h2_analysis": {"mate_in_moves": 2, "reference_mate_plies": 3, "requested_depths": [16, 20],
            "verification_depths": [16, 20], "principal_variation_uci": list(pv), "validation_runs": runs,
            "source_reused": False}}


def c2_fixture():
    fen = "3k5/9/9/9/9/Rr7/9/9/9/4K4 w - - 0 1"
    board, _ = fen_to_board(fen)
    scored = score_moves(VariantBoard(board, []), 1, 3, 1)
    return {"schema_version": 2, "release_id": RELEASE_ID, "family": "xiangqi-rule-variants",
        "id": "xiangqi-rule-variants-fixture", "fen": fen, "agent_color": "red", "goal": "best-move-under-rule",
        "max_plies": 1, "difficulty": "control", "piece_count": 4, "tags": [], "ruleset": "standard", "rules": [],
        "scenario_id": "xiangqi-rule-scenario-fixture", "design_stratum": "standard-control",
        "oracle": {"best_move_uci": scored[0][0].to_uci(), "mate_in_plies": None, "evaluation_cp": None},
        "c2_analysis": {"oracle_depth": 3, "legal_move_count": len(scored), "best_value": scored[0][1],
            "unique_best_margin": scored[0][1]-scored[1][1], "standard_best_uci": scored[0][0].to_uci(),
            "focus_ruleset": "standard", "focus_changes_best_move": False}}



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
        for rule in RULESETS:
            data[ALIASES["c2"]].append({"id": f"c2-{number:02}-{rule}", "family": ALIASES["c2"],
                "scenario_id": f"scenario-{number:02}", "ruleset": rule, "fen": f"position-{number}",
                "design_stratum": foci[number % 3]})
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
            self.assertEqual(report["samples"]["formal"]["tasks"]["c2"]["record_count"], 40)

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
            self.assertFalse(report["release_ready"])
            self.assertEqual(report["frozen_samples"]["status"], "not_frozen")




def c2_source_fixture(root):
    filename = "Dataset/fixture.pgn"
    path = root / "sources/ccpd" / filename
    path.parent.mkdir(parents=True)
    initial = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    text = f'[FEN "{initial}"]\n[Result "*"]\n1. 炮二平五 馬８進７\n2. 馬二進三 車９平８\n3. 車一平二 *\n'
    path.write_bytes(text.encode("big5"))
    positions, _ = _parse_source_pgn(path)
    source = {**positions[4], "source_file": filename, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "replayed_plies_verified": 4, "repository": CCPD_REPOSITORY, "revision": CCPD_REVISION, "license": "CC-BY-4.0"}
    return {"id": "c2-source-fixture", "fen": positions[4]["fen"], "source": source, "source_id": "ccpd:" + filename}, positions


@unittest.skipUnless(HAS_CCHESS, "install cchess for PGN notation decoding")
class XiangqiSourceGateTests(unittest.TestCase):
    def test_archive_hash_and_independent_prefix_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, positions = c2_source_fixture(root)
            other = deepcopy(record)
            other["id"] += "-paired"
            with patch("scripts.validate_xiangqi_data._parse_source_pgn", wraps=_parse_source_pgn) as parser:
                report = audit_c2_sources([record, other], root)
            self.assertTrue(report["valid"], report)
            self.assertEqual(parser.call_count, 1)
            self.assertEqual(report["source_count"], 1)
            record["source"]["file_sha256"] = "0" * 64
            record["source"]["revision"] = "wrong-revision"
            report = audit_c2_sources([record], root)
            self.assertFalse(report["valid"])
            self.assertIn("source_file_hash_mismatch", report["records"][0]["reasons"])
            self.assertIn("source_revision_mismatch", report["records"][0]["reasons"])

    def test_source_parser_is_not_trusted_as_legality_or_transition_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, positions = c2_source_fixture(root)
            corrupted = deepcopy(positions)
            corrupted[0]["played_move_uci"] = "a0a9"
            with patch("scripts.validate_xiangqi_data._parse_source_pgn", return_value=(corrupted, "*")):
                report = audit_c2_sources([record], root)
            self.assertFalse(report["valid"])
            self.assertIn("ply_0:source_move_illegal", report["records"][0]["reasons"])
            corrupted = deepcopy(positions)
            corrupted[1]["fen"] = positions[0]["fen"].replace(" w ", " b ")
            with patch("scripts.validate_xiangqi_data._parse_source_pgn", return_value=(corrupted, "*")):
                report = audit_c2_sources([record], root)
            self.assertFalse(report["valid"])
            self.assertIn("ply_1:parsed_transition_disagrees_with_independent_move", report["records"][0]["reasons"])

    def test_missing_or_escaping_source_cannot_be_certified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, _ = c2_source_fixture(root)
            record["source"]["source_file"] = "../outside.pgn"
            report = audit_c2_sources([record], root)
            self.assertFalse(report["valid"])
            self.assertEqual(report["inputs"], {})
            record["source"]["source_file"] = "missing.pgn"
            report = audit_c2_sources([record], root)
            self.assertFalse(report["valid"])
            self.assertTrue(any("FileNotFoundError" in why for why in report["reasons"]))



@unittest.skipUnless(HAS_CCHESS, "install .[xiangqi-generation] to run the independent release gate")
class XiangqiDataGateTests(unittest.TestCase):
    def test_d3_gate_recomputes_complete_mate_set(self):
        record = d3_fixture()
        good = audit_record(record, record["family"])
        self.assertTrue(good["valid"], good)
        record["d3_analysis"]["mate_moves_uci"] = ["b8b9"]
        record["d3_analysis"]["mate_move_count"] = 1
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

    def test_c2_gate_recomputes_depth_three_utilities(self):
        record = c2_fixture()
        good = audit_record(record, record["family"])
        self.assertTrue(good["valid"], good)
        self.assertTrue(good["details"]["search_verified"])
        record["c2_analysis"]["best_value"] += 1
        bad = audit_record(record, record["family"])
        self.assertIn("c2_best_value_mismatch", bad["reasons"])
        light = audit_record(record, record["family"], skip_search=True)
        self.assertFalse(light["details"]["search_verified"])
        self.assertIn("skipped", light["details"])

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
        scenario = c2_fixture()
        other = deepcopy(scenario)
        other["id"] += "-2"
        other["scenario_id"] += "-2"
        self.assertIn("c2_same_fen_reused_across_scenarios", audit_collection(scenario["family"], [scenario, other]))

    def test_partial_gate_records_hashes_and_cannot_certify_release(self):
        record = c2_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rule_variants").mkdir()
            path = root / "rule_variants/tasks.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            report = audit_release(root, families=(record["family"],), skip_search=True)
            self.assertFalse(report["full"])
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

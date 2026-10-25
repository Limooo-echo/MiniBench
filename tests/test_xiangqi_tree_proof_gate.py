"""Saved independent proofs are bound to data, trees and checker source bytes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from minibench.datasets.xiangqi import tree_certificate as certificate
from minibench.datasets.xiangqi import history, rule_variants
from minibench.datasets.xiangqi.dataset import xiangqi_task_from_dict
from minibench.datasets.xiangqi.schema import runtime_dict
from minibench.datasets.xiangqi.solution_tree import TreeSession
from minibench.datasets.xiangqi.tree_verification import VERIFICATION_VERSION


class IndependentTreeProofGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "data/xiangqi"
        self.rows = {}
        self.results = {}
        for family, folder in certificate.FAMILY_FOLDERS.items():
            original = certificate.ROOT / "data/xiangqi" / folder / "tasks.jsonl"
            rows = [json.loads(line) for line in original.read_text().splitlines()[:2]]
            self.rows[family] = rows
            path = self.root / folder / "tasks.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            tree_folder = certificate.TREE_FOLDERS[family]
            destination = self.root / "solution_trees" / tree_folder
            destination.mkdir(parents=True)
            for row in rows:
                source = certificate.ROOT / row["shortest_proof"]["tree_path"]
                target = destination / source.name
                shutil.copyfile(source, target)
                tree = json.loads(target.read_text())
                self.results[row["id"]] = {
                    "status": "valid", "valid": True,
                    "version": VERIFICATION_VERSION,
                    "computed_shortest_attacker_moves": row["shortest_mate_moves"],
                    "optimal_root_moves_uci": sorted(tree["nodes"][tree["root"]]["edges"]),
                    "nodes_verified": len(tree["nodes"]),
                    "proof_complete": True, "coverage_verified": True,
                    "checks": {"root_shortest": True, "optimal_root_set": True,
                               "all_nodes_and_edges": True},
                    "record_sha256": certificate.record_sha256(row),
                    "tree_sha256": certificate.sha256_file(target),
                    "verification_source_sha256": certificate.verifier_fingerprint(),
                }
        self.report_path = self.root / certificate.REPORT_FILENAME
        self.write_report()

    def write_report(self):
        report = certificate.build_verification_report(self.root, self.results)
        self.report_path.write_text(json.dumps(report, ensure_ascii=False))
        return report

    def check(self, family="xiangqi-history", records=None):
        return certificate.validate_saved_tree_certificate(
            self.rows[family] if records is None else records,
            data_path=self.root / certificate.FAMILY_FOLDERS[family] / "tasks.jsonl",
            tree_dir=self.root / "solution_trees" / certificate.TREE_FOLDERS[family])

    def test_complete_bound_report_passes_both_families(self):
        report = self.write_report()
        self.assertEqual(report["summary"], {"total": 4, "verified": 4, "invalid": 0, "unknown": 0})
        for family in self.rows:
            self.assertTrue(self.check(family)["valid"])

    def test_budget_unknown_false_stays_unknown_and_cannot_pass(self):
        row = self.rows["xiangqi-history"][0]
        self.results[row["id"]].update(status="unknown", valid=False)
        report = self.write_report()
        self.assertEqual(report["records"][row["id"]]["status"], "unknown")
        self.assertEqual(report["summary"]["invalid"], 0)
        self.assertFalse(report["valid"])
        self.assertFalse(self.check()["valid"])

    def test_positive_status_without_full_proof_transcript_is_unknown(self):
        row = self.rows["xiangqi-history"][0]
        self.results[row["id"]].pop("coverage_verified")
        report = self.write_report()
        self.assertEqual(report["records"][row["id"]]["status"], "unknown")
        self.assertFalse(self.check()["valid"])

    def test_invalid_search_result_cannot_pass(self):
        row = self.rows["xiangqi-history"][0]
        self.results[row["id"]].update(status="invalid", valid=False)
        report = self.write_report()
        self.assertEqual(report["summary"]["invalid"], 1)
        self.assertFalse(self.check()["valid"])

    def test_old_search_cannot_receive_a_new_source_fingerprint(self):
        row = self.rows["xiangqi-history"][0]
        self.results[row["id"]]["verification_source_sha256"] = "0" * 64
        report = self.write_report()
        self.assertEqual(report["records"][row["id"]]["status"], "unknown")
        self.assertFalse(self.check()["valid"])

    def test_source_change_makes_saved_proof_stale(self):
        changed = certificate.verifier_hashes()
        changed[certificate.VERIFIER_FILES[0]] = "0" * 64
        with patch.object(certificate, "verifier_hashes", return_value=changed):
            self.assertFalse(self.check()["valid"])

    def test_dataset_bytes_change_makes_saved_proof_stale(self):
        path = self.root / "history/tasks.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        self.assertFalse(self.check()["valid"])

    def test_tree_bytes_change_makes_saved_proof_stale(self):
        row = self.rows["xiangqi-history"][0]
        path = self.root / "solution_trees/h2" / f"{row['id']}.json"
        path.write_bytes(path.read_bytes() + b"\n")
        self.assertFalse(self.check()["valid"])

    def test_runtime_projection_is_valid_but_changed_identity_is_not(self):
        row = self.rows["xiangqi-history"][0]
        projection = {key: row[key] for key in ("id", "fen", "oracle", "max_plies", "shortest_proof")}
        self.assertTrue(self.check(records=[projection])["valid"])
        projection["max_plies"] += 2
        self.assertFalse(self.check(records=[projection])["valid"])

    def test_missing_report_or_roster_entry_refuses(self):
        report = json.loads(self.report_path.read_text())
        report["records"].pop(next(iter(report["records"])))
        self.report_path.write_text(json.dumps(report))
        self.assertFalse(self.check()["valid"])
        self.report_path.unlink()
        self.assertFalse(self.check()["valid"])

    def test_unselected_unknown_does_not_erase_selected_proof(self):
        first, second = self.rows["xiangqi-history"]
        del self.results[second["id"]]
        report = self.write_report()
        self.assertFalse(report["complete"])
        self.assertTrue(self.check(records=[first])["valid"])
        self.assertFalse(self.check()["valid"])

    def test_portable_frozen_copy_keeps_identity(self):
        target = Path(self.temporary.name) / "suite/inputs/data/xiangqi"
        shutil.copytree(self.root, target)
        self.assertTrue(certificate.validate_saved_tree_certificate(
            self.rows["xiangqi-history"], data_path=target/"history/tasks.jsonl",
            tree_dir=target/"solution_trees/h2")["valid"])

    def test_single_tree_session_refuses_unknown_before_using_the_tree(self):
        row = self.rows["xiangqi-history"][0]
        self.results[row["id"]].update(status="unknown", valid=None)
        self.write_report()
        with self.assertRaisesRegex(ValueError, "Independent shortest-tree verification required"):
            TreeSession(row, self.root/"solution_trees/h2")

    def test_all_h2_and_c2_rows_are_preflighted_before_any_agent_call(self):
        for family in self.rows:
            rows = self.rows[family]
            self.results[rows[1]["id"]].update(status="unknown", valid=None)
            self.write_report()
            agent = Mock()
            with self.subTest(family=family), self.assertRaises(ValueError):
                if family == "xiangqi-history":
                    history.evaluate_history_tasks([xiangqi_task_from_dict(row) for row in rows], agent,
                                                   solution_tree_dir=self.root/"solution_trees/h2")
                else:
                    rule_variants.evaluate_rule_variant_tasks([runtime_dict(row) for row in rows], agent,
                                                             solution_tree_dir=self.root/"solution_trees/c2")
            agent.generate.assert_not_called()
            agent.generate_messages.assert_not_called()


if __name__ == "__main__":
    unittest.main()

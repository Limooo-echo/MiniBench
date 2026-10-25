"""Durable independent-proof CLI never promotes skipped or unknown work."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.core.checkpoint import atomic_write_json, atomic_write_jsonl
from scripts import verify_xiangqi_solution_trees as cli


class IndependentTreeCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.rows = []
        for family, folder in cli.FAMILY_FOLDERS.items():
            source = cli.ROOT / "data/xiangqi" / folder / "tasks.jsonl"
            record = next(json.loads(line) for line in source.read_text().splitlines()
                          if json.loads(line)["shortest_mate_moves"] == 2)
            raw = (cli.ROOT / record["shortest_proof"]["tree_path"]).read_bytes()
            tree = self.data / "solution_trees" / cli.TREE_FOLDERS[family] / f"{record['id']}.json"
            tree.parent.mkdir(parents=True)
            tree.write_bytes(raw)
            atomic_write_jsonl(self.data / folder / "tasks.jsonl", [record])
            self.rows.append(record)
        self.output = self.root / "report.json"

    def run_audit(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return cli.run_verification(
                self.data, self.output, families=set(cli.FAMILY_FOLDERS),
                seconds=10, workers=1, **kwargs)

    def test_two_small_real_trees_and_resume_do_not_recompute_verified_work(self):
        report = self.run_audit()
        self.assertEqual(report["summary"], {"total": 2, "verified": 2, "invalid": 0, "unknown": 0})
        self.assertTrue(report["complete"])
        with patch.object(cli, "verify_one", side_effect=AssertionError("must reuse")):
            resumed = self.run_audit(resume=True)
        self.assertTrue(resumed["valid"])

    def test_time_budget_remains_unknown_not_invalid_or_pass(self):
        with redirect_stdout(io.StringIO()):
            report = cli.run_verification(self.data, self.output,
                families=set(cli.FAMILY_FOLDERS), seconds=0.000001, workers=1)
        self.assertEqual(report["summary"]["unknown"], 2)
        self.assertEqual(report["summary"]["invalid"], 0)
        self.assertFalse(report["valid"])

    def test_rehashed_missing_strategy_edge_is_invalid(self):
        record = deepcopy(self.rows[0])
        tree_path = self.data / "solution_trees" / cli.TREE_FOLDERS[record["family"]] / f"{record['id']}.json"
        tree = json.loads(tree_path.read_text())
        edges = tree["nodes"][tree["root"]]["edges"]
        del edges[next(iter(edges))]
        atomic_write_json(tree_path, tree)
        record["shortest_proof"]["tree_sha256"] = hashlib.sha256(tree_path.read_bytes()).hexdigest()
        atomic_write_jsonl(self.data / cli.FAMILY_FOLDERS[record["family"]] / "tasks.jsonl", [record])
        report = self.run_audit()
        self.assertEqual(report["records"][record["id"]]["status"], "invalid")
        self.assertFalse(report["valid"])

    def test_existing_report_is_not_overwritten_without_resume(self):
        atomic_write_json(self.output, {"sentinel": True})
        with self.assertRaises(FileExistsError):
            self.run_audit()
        self.assertEqual(json.loads(self.output.read_text()), {"sentinel": True})

    def test_h2_loader_preserves_certified_shortest_distance(self):
        from minibench.datasets.xiangqi.dataset import xiangqi_task_from_dict
        record = next(row for row in self.rows if row['family'] == 'xiangqi-history')
        task = xiangqi_task_from_dict(record)
        self.assertEqual(task.shortest_mate_moves, record['shortest_mate_moves'])
        self.assertEqual(task.shortest_proof, record['shortest_proof'])

    def test_contradictory_formal_record_is_invalid_not_a_verified_search(self):
        record = deepcopy(self.rows[0])
        record['agent_color'] = 'black' if record['agent_color'] == 'red' else 'red'
        atomic_write_jsonl(self.data / cli.FAMILY_FOLDERS[record['family']] / 'tasks.jsonl', [record])
        report = self.run_audit()
        self.assertEqual(report['records'][record['id']]['status'], 'invalid')
        self.assertFalse(report['valid'])

    def test_resume_retries_unknown_with_new_ordering_without_promoting_it(self):
        with redirect_stdout(io.StringIO()):
            first = cli.run_verification(self.data, self.output,
                families=set(cli.FAMILY_FOLDERS), seconds=0.000001, workers=1)
        self.assertEqual(first['summary']['unknown'], 2)
        self.assertFalse(first['valid'])
        completed = self.run_audit(resume=True, ordering_mode='checks-history')
        self.assertTrue(completed['valid'])
        self.assertEqual(completed['summary']['verified'], 2)
        for item in completed['records'].values():
            proof = item['verification']
            self.assertEqual(proof['ordering_mode'], 'checks-history')
            self.assertEqual(proof['verification_parameters']['ordering_mode'], 'checks-history')


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_zebra_paired as runner
from minibench.core.agent import CompletionResult
from minibench.core.checkpoint import atomic_write_json
from minibench.datasets.zebra.dataset import zebra_task_from_dict
from minibench.datasets.zebra.evaluation import expected_solution_table


def item():
    record = json.loads((ROOT / "data/zebra/eval.jsonl").read_text().splitlines()[0])
    record.setdefault("source_id", record["id"])
    return {"key": "offline-fixture", "mode": "direct", "record": record}


class FakeClient:
    output = ""
    calls = 0

    def __init__(self, **kwargs):
        pass

    def complete_result(self, *args, **kwargs):
        type(self).calls += 1
        return CompletionResult(content=self.output, finish_reason="stop")

    def metrics_snapshot(self):
        return {"llm_calls": 1, "usage_missing_calls": 1,
                "model_elapsed_seconds": 0, "token_usage": {}}


class ZebraPairedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.config = runner.load_config(ROOT / "config/experiments/zebra_paired_qwen38.yaml")
        FakeClient.calls = 0

    def execute(self, directory, output):
        FakeClient.output = output
        with patch.object(runner, "ZebraQwenClient", FakeClient):
            return runner.execute_item(item(), self.config, directory, "frozen", lambda: None)

    def test_correct_answer_is_saved_and_restored_without_calling_model(self):
        task = zebra_task_from_dict(item()["record"])
        output = json.dumps({"reasoning": "Test fixture.", "solution": expected_solution_table(task)})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = self.execute(path, output)
            self.assertTrue(result["result"]["success"])
            self.assertTrue(result["diagnostics"]["schema_complete"])
            restored = runner.restore_results([item()], path, "frozen")
            self.assertEqual(restored[item()["key"]]["raw_output"], output)
            self.assertEqual(FakeClient.calls, 1)

    def test_bad_json_is_scored_without_repair_or_aborting_run(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute(Path(directory), '{"solution":')
            self.assertFalse(result["result"]["success"])
            self.assertFalse(result["result"]["parsed"])
            self.assertTrue(result["diagnostics"]["strict_json_error"])
            self.assertEqual(result["raw_output"], '{"solution":')
            self.assertEqual(FakeClient.calls, 1)

    def test_parsed_empty_solution_is_distinguished_from_complete_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute(Path(directory), '{"reasoning":"", "solution":{}}')
            self.assertTrue(result["result"]["parsed"])
            self.assertEqual(result["result"]["correct_cells"], 0)
            self.assertFalse(result["diagnostics"]["schema_complete"])
            summary = runner.save_summary(Path(directory), [item()], {item()["key"]: result}, "running")
            self.assertEqual(summary["conditions"]["direct"]["parsed_zero_cells"], 1)

    def test_resume_rejects_different_run_and_corrupted_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = self.execute(path, '{}')
            with self.assertRaisesRegex(ValueError, "identity changed"):
                runner.restore_results([item()], path, "changed")
            result["result"]["success"] = True
            atomic_write_json(path / "jobs" / item()["key"] / "result.json", result)
            with self.assertRaisesRegex(ValueError, "score mismatch"):
                runner.restore_results([item()], path, "frozen")

    def test_transport_failure_is_not_saved_as_an_incorrect_model_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(FakeClient, "complete_result", side_effect=RuntimeError("offline disconnect")):
                with self.assertRaisesRegex(RuntimeError, "disconnect"):
                    self.execute(Path(directory), '{}')
            self.assertFalse((Path(directory) / "jobs" / item()["key"] / "result.json").exists())


if __name__ == "__main__":
    unittest.main()

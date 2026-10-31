import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("zebra_history_batch", ROOT / "scripts/run_zebra_history_batch.py")
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


class ZebraHistoryBatchTests(unittest.TestCase):
    def test_pair_constructs_direct_agents_without_api_calls(self):
        configs = batch.load_pair()
        self.assertEqual(list(configs), ["thinking_off", "thinking_on"])
        self.assertTrue(all(c["agent"]["name"] == "direct" for c in configs.values()))

    def run_case(self, key, failure=False):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "batch"
            configs = batch.load_pair()
            evidence = {"code_sha256": batch._zebra_code_sha256(), "data_sha256":
                        batch.hashlib.sha256((ROOT / configs["thinking_off"]["task"]["path"]).read_bytes()).hexdigest()}
            def fake_run(config):
                if failure:
                    raise RuntimeError("injected transport failure")
                return target / config["run"]["run_name"], {"by_memory_mode": {}}
            with patch.object(batch.sys.stdin, "isatty", return_value=True), patch.object(
                batch.getpass, "getpass", return_value=key,
            ), patch.object(batch, "run_family_experiment", side_effect=fake_run) as run, patch.dict(
                os.environ, {}, clear=False,
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = batch.run_batch(target, configs, evidence)
                self.assertNotIn("DASHSCOPE_API_KEY", os.environ)
            state = json.loads((target / "batch-status.json").read_text())
            for path in target.rglob("*"):
                if path.is_file() and path.suffix in {".json", ".yaml", ".log"}:
                    self.assertNotIn("unit-test-hidden-key-12345", path.read_text())
            return code, state, run.call_count

    def test_single_hidden_entry_runs_both_and_records_completion(self):
        code, state, calls = self.run_case("unit-test-hidden-key-12345")
        self.assertEqual((code, calls, state["status"]), (0, 2, "completed"))
        self.assertEqual(state["completed_treatments"], ["thinking_off", "thinking_on"])

    def test_empty_key_never_calls_model(self):
        code, state, calls = self.run_case("")
        self.assertEqual((code, calls, state["status"]), (1, 0, "not_started"))

    def test_first_failure_stops_second_treatment(self):
        code, state, calls = self.run_case("unit-test-hidden-key-12345", failure=True)
        self.assertEqual((code, calls, state["status"]), (1, 1, "interrupted"))
        self.assertEqual(state["current_treatment"], "thinking_off")


if __name__ == "__main__":
    unittest.main()

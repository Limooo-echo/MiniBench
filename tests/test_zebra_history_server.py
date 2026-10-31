"""Offline raw HTTP journal and server scheduling regression checks."""
from concurrent.futures import Future
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from io import BytesIO, StringIO
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_zebra_history_server as server
from minibench.core.checkpoint import atomic_write_json, sha256_file
from minibench.datasets.zebra.evaluation import evaluate_zebra_work_item, zebra_result_key
from minibench.datasets.zebra.state_validation import initial_candidate_state
from minibench.experiments.zebra import _zebra_code_sha256
from minibench.factory.providers import OpenAICompatibleClient
from test_zebra_history_pro import ProposalAgent, completion_response, final_output, history_task


def raw_response():
    return json.dumps({
        "id": "offline-response", "model": "qwen3.8-flash",
        "choices": [{"message": {"role": "assistant", "content": '{"acknowledged":1}',
                                  "reasoning_content": "native reasoning retained"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }).encode()


def offline_result(task, mode, config):
    ev = config["evaluation"]
    return asdict(evaluate_zebra_work_item(task, ProposalAgent([initial_candidate_state(task)]), mode,
        state_max_tokens=ev["state_max_tokens"], ack_max_tokens=ev["ack_max_tokens"],
        final_max_tokens=ev["final_max_tokens"]))


class RawJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"DASHSCOPE_API_KEY": "OFFLINE-SECRET-DO-NOT-RECORD"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def client(self, *, thinking=True):
        client = OpenAICompatibleClient(model="qwen3.8-flash", base_url="https://offline.invalid/v1",
            api_key_env="DASHSCOPE_API_KEY", json_mode=True, extra_body={"enable_thinking": thinking})
        journal = server.RawCallJournal(client, self.root, task_id="task1", mode="incremental_state",
            treatment="thinking_on", planned_calls=1)
        client._complete_request_once = journal.complete_request_once
        return client, journal

    def test_exact_body_response_reasoning_and_replay_preserve_metrics(self):
        client, journal = self.client()
        with patch("urllib.request.urlopen", return_value=BytesIO(raw_response())) as transport:
            with client.capture_completion_traces() as traces:
                first = client.complete_messages_result([{"role": "user", "content": "clue"}])
        self.assertEqual(transport.call_count, 1)
        sent = transport.call_args.args[0]
        call = self.root / "calls/0000"
        self.assertEqual(call.with_suffix(".request.raw").read_bytes(), sent.data)
        self.assertEqual(json.loads(call.with_suffix(".request.json").read_text()), json.loads(sent.data))
        self.assertEqual(call.with_suffix(".response.raw").read_bytes(), raw_response())
        self.assertEqual(first.reasoning, "native reasoning retained")
        self.assertEqual(traces[0]["reasoning_content"], first.reasoning)
        for path in self.root.rglob("*"):
            if path.is_file():
                self.assertNotIn("OFFLINE-SECRET-DO-NOT-RECORD", path.read_text())
                self.assertNotIn("Authorization", path.read_text())
        resumed, second_journal = self.client()
        with patch("urllib.request.urlopen", side_effect=AssertionError("no paid replay")) as transport:
            with resumed.capture_completion_traces() as replayed:
                second = resumed.complete_messages_result([{"role": "user", "content": "clue"}])
        self.assertEqual(first, second)
        self.assertEqual(traces, replayed)
        self.assertEqual(client.metrics_snapshot(), resumed.metrics_snapshot())
        self.assertEqual(second_journal.index, 1)
        transport.assert_not_called()

    def test_changed_thinking_cannot_reuse_another_call(self):
        client, _ = self.client()
        with patch("urllib.request.urlopen", return_value=BytesIO(raw_response())):
            client.complete("clue")
        resumed, _ = self.client(thinking=False)
        with patch("urllib.request.urlopen") as transport:
            with self.assertRaisesRegex(ValueError, "resumed request differs"):
                resumed.complete("clue")
        transport.assert_not_called()

    def test_received_response_without_record_is_reparsed_without_api(self):
        client, _ = self.client()
        with patch("urllib.request.urlopen", return_value=BytesIO(raw_response())):
            first = client.complete("clue")
        (self.root / "calls/0000.record.json").unlink()
        (self.root / "calls/0000.response.json").unlink()
        resumed, _ = self.client()
        with patch("urllib.request.urlopen", side_effect=AssertionError("no duplicate charge")) as transport:
            second = resumed.complete("clue")
        self.assertEqual(first, second)
        transport.assert_not_called()
        self.assertTrue((self.root / "calls/0000.response.json").exists())
        third_client, _ = self.client()
        with patch("urllib.request.urlopen", side_effect=AssertionError("no duplicate charge")) as transport:
            self.assertEqual(third_client.complete("clue"), first)
        transport.assert_not_called()

    def test_http_error_body_retained_for_original_parser(self):
        client, _ = self.client()
        error = HTTPError("https://offline.invalid", 429, "limited", None, BytesIO(b'{"error":"limited"}'))
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'HTTP 429:.*limited'):
                client.complete("clue")
        self.assertEqual(len(list((self.root / "calls").glob("*.error-*.json"))), 1)
        self.assertFalse((self.root / "calls/0000.record.json").exists())

    def test_real_direct_history_prefix_replay_in_all_three_modes(self):
        task = history_task()
        config = yaml.safe_load((ROOT / "config/experiments/zebra_history_thinking_on.yaml").read_text())
        for mode in config["evaluation"]["memory_modes"]:
            with self.subTest(mode=mode):
                proposal = '{"acknowledged":1}' if mode == "deferred_reasoning" else json.dumps(initial_candidate_state(task))
                responses = [completion_response(proposal), completion_response(proposal), completion_response(final_output())]
                with patch("urllib.request.urlopen", side_effect=responses) as transport:
                    first = server.run_job(self.root, "thinking_on", task, mode, config)
                self.assertTrue(first["success"])
                self.assertEqual(transport.call_count, 3)
                with patch("urllib.request.urlopen", side_effect=AssertionError("prefix must replay")) as transport:
                    second = server.run_job(self.root, "thinking_on", task, mode, config)
                self.assertTrue(second["success"])
                self.assertEqual(first["raw_output"], second["raw_output"])
                self.assertEqual(first["metrics"]["token_usage"], second["metrics"]["token_usage"])
                self.assertEqual(first["metrics"]["model_elapsed_seconds"], second["metrics"]["model_elapsed_seconds"])
                transport.assert_not_called()


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.batch = Path(self.temp.name)
        (self.batch / "configs").mkdir()
        tasks = [history_task()]
        data = self.batch / "history_eval.jsonl"
        data.write_text("".join(json.dumps(asdict(t)) + "\n" for t in tasks), encoding="utf-8")
        atomic_write_json(self.batch / "batch-manifest.json", {
            "code_sha256": _zebra_code_sha256(), "data_sha256": sha256_file(data)})
        for treatment in ("thinking_off", "thinking_on"):
            config = yaml.safe_load((ROOT / f"config/experiments/zebra_history_{treatment}.yaml").read_text())
            (self.batch / "configs" / f"{treatment}.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    def test_six_jobs_interleave_treatments_and_use_sixteen_capacity(self):
        coordinator = server.Coordinator(self.batch, 16)
        self.addCleanup(coordinator.stack.close)
        submissions = []

        class FakePool:
            def __init__(self, **kwargs):
                if kwargs["max_workers"] != 16:
                    raise AssertionError("expected sixteen workers")
                if kwargs["mp_context"].get_start_method() != "spawn":
                    raise AssertionError("expected spawn")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def submit(self, fn, batch, treatment, task, mode, config):
                submissions.append((treatment, task.id, mode))
                future = Future()
                future.set_result(offline_result(task, mode, config))
                return future

        with patch.object(server, "ProcessPoolExecutor", FakePool), patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline"}), redirect_stdout(StringIO()):
            result = coordinator.run()
        self.assertEqual(result, 0)
        self.assertEqual([s[0] for s in submissions], ["thinking_off", "thinking_on"] * 3)
        self.assertEqual(len(set(submissions)), 6)
        coordinator.stack.close()
        resumed = server.Coordinator(self.batch, 16)
        self.addCleanup(resumed.stack.close)
        with patch.object(server, "ProcessPoolExecutor", side_effect=AssertionError("completed work must not rerun")), redirect_stdout(StringIO()):
            self.assertEqual(resumed.run(), 0)
        status = json.loads((self.batch / "parallel-status.json").read_text())
        self.assertEqual(status["completed_total"], 6)
        self.assertEqual(status["workers"], 16)

    def test_failure_drains_already_submitted_jobs(self):
        coordinator = server.Coordinator(self.batch, 16)
        self.addCleanup(coordinator.stack.close)
        submissions = []

        class FakePool:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def submit(self, fn, batch, treatment, task, mode, config):
                submissions.append((treatment, task.id, mode))
                future = Future()
                if len(submissions) == 1:
                    future.set_exception(RuntimeError("injected transport failure"))
                else:
                    future.set_result(offline_result(task, mode, config))
                return future

        def harvest_one(futures, **kwargs):
            first = next(iter(futures))
            return {first}, set(futures) - {first}

        with patch.object(server, "ProcessPoolExecutor", FakePool), patch.object(server, "wait", harvest_one), patch.dict(
            os.environ, {"DASHSCOPE_API_KEY": "offline"}), redirect_stdout(StringIO()):
            self.assertEqual(coordinator.run(), 1)
        self.assertEqual(len(submissions), 6)
        status = json.loads((self.batch / "parallel-status.json").read_text())
        self.assertEqual(status["status"], "interrupted")
        self.assertEqual(status["completed_total"], 5)
        self.assertEqual(len(status["failures"]), 1)
        self.assertEqual(len(list((self.batch / "parallel-journal").rglob("*.json"))), 5)


if __name__ == "__main__":
    unittest.main()

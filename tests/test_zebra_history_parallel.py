"""Offline regression checks for the hot parallel handoff runner."""
from concurrent.futures import Future
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("zebra_history_parallel", ROOT / "scripts/resume_zebra_history_parallel.py")
parallel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parallel)

from minibench.core.checkpoint import RunLock, atomic_write_json, sha256_file
from minibench.datasets.zebra.evaluation import evaluate_zebra_work_item, zebra_result_key
from minibench.datasets.zebra.state_validation import initial_candidate_state
from minibench.experiments.zebra import _zebra_code_sha256, _zebra_sidecar_lock_name
from test_zebra_history_pro import ProposalAgent, completion_response, final_output, history_task


def offline_result(task, mode, config):
    evaluation = config["evaluation"]
    return asdict(evaluate_zebra_work_item(
        task, ProposalAgent([initial_candidate_state(task)]), mode,
        state_max_tokens=evaluation["state_max_tokens"],
        ack_max_tokens=evaluation["ack_max_tokens"],
        final_max_tokens=evaluation["final_max_tokens"],
    ))


class ZebraHistoryParallelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.batch = Path(self.temp.name)
        (self.batch / "configs").mkdir()
        self.tasks = [history_task(), replace(history_task(), id="history-pro-second")]
        data = self.batch / "history_eval.jsonl"
        data.write_text("".join(json.dumps(asdict(t)) + "\n" for t in self.tasks), encoding="utf-8")
        atomic_write_json(self.batch / "batch-manifest.json", {
            "code_sha256": _zebra_code_sha256(), "data_sha256": sha256_file(data),
        })
        for treatment in ("thinking_off", "thinking_on"):
            config = yaml.safe_load((ROOT / f"config/experiments/zebra_history_{treatment}.yaml").read_text())
            config["provider"]["api_key_env"] = "ZEBRA_PARALLEL_OFFLINE_KEY"
            (self.batch / "configs" / f"{treatment}.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    def coordinator(self):
        coordinator = parallel.Coordinator(self.batch, 4)
        self.addCleanup(coordinator.stack.close)
        return coordinator

    def seed(self, coordinator, treatment, count):
        from minibench.datasets.zebra.evaluation import zebra_result_from_dict
        coordinator.acquire(treatment)
        ctx = coordinator.contexts[treatment]
        for task, mode in ctx["plan"][:count]:
            row = zebra_result_from_dict(offline_result(task, mode, ctx["config"]))
            ctx["results"][zebra_result_key(row)] = row
        coordinator.snapshot(treatment)

    def test_run_job_real_direct_with_mock_transport_in_all_modes(self):
        coordinator = self.coordinator()
        config = coordinator.contexts["thinking_off"]["config"]
        task = self.tasks[0]
        for mode in config["evaluation"]["memory_modes"]:
            with self.subTest(mode=mode):
                proposal = '{"ack":true}' if mode == "deferred_reasoning" else json.dumps(initial_candidate_state(task))
                responses = [completion_response(proposal), completion_response(proposal), completion_response(final_output())]
                with patch.dict(os.environ, {"ZEBRA_PARALLEL_OFFLINE_KEY": "offline-test-only"}), patch(
                    "minibench.factory.providers.urllib.request.urlopen", side_effect=responses,
                ) as transport:
                    result = parallel.run_job(task, mode, config)
                self.assertTrue(result["success"])
                self.assertEqual(result["memory_mode"], mode)
                self.assertEqual(transport.call_count, 3)
                self.assertEqual(len(result["history_trace"]), 3)

    def test_journal_recovery_skips_completed_work_and_persists_completion(self):
        original = self.coordinator()
        self.seed(original, "thinking_off", 5)
        ctx = original.contexts["thinking_off"]
        task, mode = ctx["plan"][-1]
        atomic_write_json(self.batch / "parallel-journal/thinking_off/last.json", offline_result(task, mode, ctx["config"]))
        original.stack.close()
        resumed = self.coordinator()
        self.assertTrue(resumed.acquire("thinking_off"))
        self.assertEqual(len(resumed.contexts["thinking_off"]["results"]), 6)
        self.assertEqual(list(resumed.pending()), [])
        state = json.loads((self.batch / "thinking_off/run_state.json").read_text())
        self.assertEqual((state["status"], state["completed_total"]), ("completed", 6))
        self.assertEqual(len((self.batch / "thinking_off/predictions.jsonl").read_text().splitlines()), 6)

    def test_legacy_journal_allows_absent_legality_but_checks_saved_scores(self):
        coordinator = self.coordinator()
        ctx = coordinator.contexts["thinking_off"]
        task, mode = ctx["plan"][0]
        saved = offline_result(task, mode, ctx["config"])
        for field in ("is_legal", "legality_reasons"):
            saved.pop(field)
        path = self.batch / "parallel-journal/thinking_off/legacy.json"
        atomic_write_json(path, saved)
        before = path.read_bytes()
        coordinator.recover_journal("thinking_off")
        recovered = ctx["results"][(task.id, mode)]
        self.assertTrue(recovered.is_legal)
        self.assertEqual(path.read_bytes(), before)
        coordinator.recover_journal("thinking_off")
        self.assertEqual(ctx["results"][(task.id, mode)], recovered)
        saved["success"] = not saved["success"]
        atomic_write_json(path, saved)
        with self.assertRaisesRegex(ValueError, "journal scores"):
            coordinator.recover_journal("thinking_off")

    def test_journal_rejects_tampered_legality_when_present(self):
        coordinator = self.coordinator()
        ctx = coordinator.contexts["thinking_off"]
        task, mode = ctx["plan"][0]
        saved = offline_result(task, mode, ctx["config"])
        saved["is_legal"] = not saved["is_legal"]
        atomic_write_json(self.batch / "parallel-journal/thinking_off/tampered.json", saved)
        with self.assertRaisesRegex(ValueError, "journal scores"):
            coordinator.recover_journal("thinking_off")

    def run_with_fake_pool(self, coordinator, *, first_failure=False, release_old=None):
        submissions, capacities = [], []

        class FakePool:
            def __init__(self, **kwargs):
                if kwargs["mp_context"].get_start_method() != "spawn":
                    raise AssertionError("workers must spawn without inheriting the coordinator's locks")
                if kwargs["max_workers"] != 4:
                    raise AssertionError("expected a four-worker pool")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, fn, task, mode, config):
                future = Future()
                submissions.append((task.id, mode))
                if first_failure and len(submissions) == 1:
                    future.set_exception(RuntimeError("injected transport failure"))
                else:
                    future.set_result(offline_result(task, mode, config))
                return future

        def fake_wait(futures, **kwargs):
            capacities.append(len(futures))
            if len(capacities) == 1 and release_old is not None:
                release_old()
            # Harvest one at a time to exercise refilling and draining.
            first = next(iter(futures))
            return {first}, set(futures) - {first}

        with patch.object(parallel, "ProcessPoolExecutor", FakePool), patch.object(parallel, "wait", fake_wait), patch.object(
            parallel.sys.stdin, "isatty", return_value=True,
        ), patch.object(parallel.getpass, "getpass", return_value="offline-test-only"), patch.dict(os.environ, {}, clear=False), redirect_stdout(io.StringIO()):
            result = coordinator.run()
        return result, submissions, capacities

    def test_real_external_lock_reserves_one_slot_then_releases_four(self):
        original = self.coordinator()
        self.seed(original, "thinking_off", 6)
        original.stack.close()
        old_lock = RunLock(self.batch, lock_name=_zebra_sidecar_lock_name("thinking_off"))
        old_lock.__enter__()
        self.addCleanup(old_lock.__exit__, None, None, None)
        before = (self.batch / "thinking_off/predictions.jsonl").read_bytes()
        coordinator = self.coordinator()
        result, submissions, capacities = self.run_with_fake_pool(
            coordinator, release_old=lambda: old_lock.__exit__(None, None, None),
        )
        self.assertEqual(result, 0)
        self.assertEqual(capacities[:2], [3, 4])
        self.assertLessEqual(max(capacities), 4)
        self.assertEqual(len(submissions), 6)
        self.assertEqual(len(set(submissions)), 6)
        self.assertEqual((self.batch / "thinking_off/predictions.jsonl").read_bytes(), before)
        final = json.loads((self.batch / "parallel-status.json").read_text())
        self.assertEqual(final["completed_total"], 12)
        self.assertEqual(final["status"], "completed")

    def test_worker_failure_drains_successes_without_submitting_more(self):
        coordinator = self.coordinator()
        result, submissions, capacities = self.run_with_fake_pool(coordinator, first_failure=True)
        self.assertEqual(result, 1)
        self.assertEqual(len(submissions), 4)
        self.assertEqual(capacities, [4, 3, 2, 1])
        final = json.loads((self.batch / "parallel-status.json").read_text())
        self.assertEqual(final["status"], "interrupted")
        self.assertEqual(final["completed_total"], 3)
        self.assertEqual(len(final["failures"]), 1)
        self.assertEqual(len(list((self.batch / "parallel-journal").rglob("*.json"))), 3)


if __name__ == "__main__":
    unittest.main()

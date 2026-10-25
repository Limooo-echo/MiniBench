"""Offline cloud concurrency, account limits, plan and resume regressions."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import multiprocessing as mp
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import yaml

from minibench import cloud
from minibench.core.checkpoint import atomic_write_json, fingerprint_payload
from minibench.core.rate_limit import RateGate


class ShortestPreflightTests(unittest.TestCase):
    def test_shortest_preflight_cannot_be_bypassed_and_precedes_credentials(self):
        for family in ('xiangqi-history','xiangqi-rule-variants'):
            record={'id':'fixture','shortest_proof':{'tree_complete':True}}
            exp={'family':family,'source':'inputs/tasks.jsonl','task_ids':['fixture'],
                 'configs':{'direct':{'evaluation':{'solution_tree_dir':'trees'}}}}
            plan={'jobs':[{'key':'not-completed'}],'experiments':{'fixture':exp}}
            with self.subTest(family=family),tempfile.TemporaryDirectory() as temporary,\
                    patch.object(cloud,'completed_jobs',return_value={}),\
                    patch('minibench.datasets.xiangqi.schema.load_records',return_value=[record]),\
                    patch('minibench.datasets.xiangqi.solution_tree.TreeSession',
                          side_effect=ValueError('invalid complete tree')) as tree_session,\
                    patch.object(cloud,'execute_job') as paid_job:
                # Missing independent evidence must block before even the cheap
                # tree loader, credentials, or any model work is reached.
                with self.assertRaisesRegex(ValueError,'Independent shortest-tree verification required'):
                    cloud.run(Path(temporary),plan,allow_uncertified_c2=True)
                tree_session.assert_not_called()
                paid_job.assert_not_called()
                exp['configs']['direct']['evaluation']={}
                with self.assertRaisesRegex(ValueError,'require solution_tree_dir'):
                    cloud.run(Path(temporary),plan,allow_uncertified_c2=True)
                paid_job.assert_not_called()


class PoolCleanupTests(unittest.TestCase):
    def test_stalled_worker_has_bounded_cleanup(self):
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
                                   initializer=cloud._worker_initializer)
        future = pool.submit(time.sleep, 60)
        # Ensure the real worker is running before exercising the watchdog.
        deadline = time.monotonic() + 5
        while not future.running() and time.monotonic() < deadline:
            time.sleep(.01)
        processes = list(pool._processes.values())
        started = time.monotonic()
        forced = cloud._shutdown_pool(pool, failed=True, grace_seconds=.25)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(len(forced), 1)
        self.assertTrue(all(not p.is_alive() for p in processes))


class RateLimitTests(unittest.TestCase):
    def test_hard_inflight_and_rolling_rpm_and_account_isolation(self):
        gate = RateGate({"a": {"max_in_flight": 2, "requests_per_minute": 2},
                         "b": {"max_in_flight": 1}}, window_seconds=.15)
        starts, active, high = [], [0], [0]
        lock = threading.Lock()
        def call():
            gate.acquire("a")
            with lock:
                starts.append(time.monotonic())
                active[0] += 1
                high[0] = max(high[0], active[0])
            time.sleep(.02)
            with lock:
                active[0] -= 1
            gate.release("a", {"total_tokens": 7})
        threads = [threading.Thread(target=call) for _ in range(6)]
        for t in threads:
            t.start()
        gate.acquire("b")
        gate.release("b", {"total_tokens": 1})
        for t in threads:
            t.join(timeout=3)
            self.assertFalse(t.is_alive())
        self.assertEqual(high[0], 2)
        self.assertGreaterEqual(sorted(starts)[2] - sorted(starts)[0], .14)
        self.assertEqual(gate.snapshot()["a"]["accounted_or_reserved_tokens"], 42)
        self.assertEqual(gate.snapshot()["b"]["network_calls"], 1)

    def test_unknown_usage_keeps_reservation_and_budget_stops(self):
        gate = RateGate({"a": {"max_in_flight": 1, "reserve_tokens": 10, "total_token_budget": 10}})
        gate.acquire("a")
        gate.release("a", None)
        self.assertEqual(gate.snapshot()["a"]["unknown_usage_calls"], 1)
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            gate.acquire("a")

    def test_observed_tokens_reconcile_and_overshoots_are_not_hidden(self):
        gate = RateGate({"a": {"reserve_tokens": 10, "total_token_budget": 30}})
        gate.acquire("a")
        gate.release("a", {"total_tokens": 20})
        self.assertEqual(gate.snapshot()["a"]["reservation_overshoots"], 1)
        gate.restore("a", {"total_tokens": 5})
        with self.assertRaises(RuntimeError):
            gate.acquire("a")

    def test_cancel_wakes_waiters_and_invalid_limits_fail(self):
        gate = RateGate({"a": {"max_in_flight": 1}})
        gate.acquire("a")
        errors = []
        def wait():
            try:
                gate.acquire("a")
            except RuntimeError:
                errors.append(True)
        t = threading.Thread(target=wait)
        t.start()
        gate.cancel()
        t.join(timeout=2)
        self.assertEqual(errors, [True])
        for limits in ({"max_in_flight": True}, {"tokens_per_minute": 1},
                       {"requests_per_minute": 0}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                RateGate({"a": limits})

    def test_restart_preserves_recent_rpm_events(self):
        from minibench.core.streaming import now
        gate = RateGate({"a": {"requests_per_minute": 1}}, window_seconds=.15)
        gate.restore_start("a", now())
        started = time.monotonic()
        gate.acquire("a")
        self.assertGreaterEqual(time.monotonic() - started, .13)
        gate.release("a", {"total_tokens": 1})


class CloudPlanTests(unittest.TestCase):
    def fixture(self, root):
        # Use the real current datasets, but freeze only small existing selections.
        raw = yaml.safe_load((cloud.ROOT / "config/cloud/xiangqi-pilot.yaml").read_text())
        raw["engine_binding"] = "reference"
        raw["accounts"]["qwen"]["provider"]["api_key_env"] = "UNIT_TEST_CLOUD_KEY"
        config = root / "cloud.yaml"
        config.write_text(yaml.safe_dump(raw))
        suite = root / "suite"
        return suite, cloud.prepare(config, suite)

    def test_plan_freezes_ids_modes_configs_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite, plan = self.fixture(root)
            self.assertEqual(plan["planned_jobs"], 30)  # 6*(D3+H2x2+C2+M2-image)
            self.assertEqual(cloud.verify_plan(suite), plan)
            self.assertTrue(all(len(e["task_ids"]) == 6 for e in plan["experiments"].values()))
            self.assertEqual({j["mode"] for j in plan["jobs"] if j["experiment"] == "m2"},
                             {"chinese-piece-image"})
            selected = json.loads((suite / "samples/c2.json").read_text())
            groups = {}
            for row in selected:
                groups.setdefault(row["scenario_id"], []).append(row["ruleset"])
            self.assertEqual(len(groups), 3)
            self.assertTrue(all(len(g) == 2 and "standard" in g for g in groups.values()))
            source = suite / plan["files"][0]["path"]
            source.write_text(source.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                cloud.verify_plan(suite)

    def test_completed_checkpoint_is_authority_and_errors_are_not_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite, plan = self.fixture(Path(tmp))
            job = plan["jobs"][0]
            path = suite / "jobs" / job["key"] / "result.json"
            value = {"job": job, "plan_sha256": plan["plan_sha256"],
                     "records": [{"status": "ok", "success": False}], "result_type": None}
            value["sha256"] = fingerprint_payload(value)
            atomic_write_json(path, value)
            self.assertEqual(set(cloud.completed_jobs(suite, plan)), {job["key"]})
            value["records"][0]["status"] = "error"
            value.pop("sha256")
            value["sha256"] = fingerprint_payload(value)
            atomic_write_json(path, value)
            with self.assertRaisesRegex(ValueError, "Invalid completed checkpoint"):
                cloud.completed_jobs(suite, plan)

    def test_report_and_resume_include_failed_usage_and_missing_started_attempt(self):
        from minibench.core.streaming import now
        with tempfile.TemporaryDirectory() as tmp:
            suite, plan = self.fixture(Path(tmp))
            job = plan['jobs'][0]
            calls = suite / 'jobs' / job['key'] / 'calls'
            failed = calls / 'attempts/0000-0000'
            good = calls / 'attempts/0000-0001'
            lost = calls / 'attempts/0001-0000'
            for directory in (failed, good, lost):
                atomic_write_json(directory / 'started.json', {'at': now()})
            atomic_write_json(failed / 'partial.json', {'at': now(), 'status': 'interrupted',
                                                       'usage': {'total_tokens': 10944}})
            atomic_write_json(calls / '0000.json', {'at': now(), 'response': {
                'usage': {'total_tokens': 7}, 'metadata': {'attempt_dir': str(good)},
            }})
            # An unstarted queued attempt must not become an unknown charge.
            atomic_write_json(calls / 'attempts/0002-0000/request.json', {'messages': []})
            usage = cloud.report(suite, plan, {})['usage']
            self.assertEqual(usage['started_attempts'], 3)
            self.assertEqual(usage['saved_successful_calls'], 1)
            self.assertEqual(usage['known_failed_tokens'], 10944)
            self.assertEqual(usage['known_total_tokens'], 10951)
            self.assertEqual(usage['attempts_without_usage'], 1)
            gate = RateGate(plan['limits'])
            cloud.restore_accounting(suite, plan, gate)
            account = next(iter(gate.snapshot().values()))
            self.assertEqual(account['network_calls'], 3)
            self.assertEqual(account['unknown_usage_calls'], 1)
            self.assertEqual(account['accounted_or_reserved_tokens'], 10951 + 32768)

    def test_code_or_plan_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite, plan = self.fixture(Path(tmp))
            with patch.object(cloud, "code_hash", return_value="changed"):
                with self.assertRaisesRegex(ValueError, "Source code changed"):
                    cloud.verify_plan(suite)
            plan["agents"] = ["cot"]
            atomic_write_json(suite / "plan.json", plan)
            with self.assertRaisesRegex(ValueError, "plan integrity"):
                cloud.verify_plan(suite)

    def test_comparison_modes_cover_c2_pairs_and_missing_is_not_zero(self):
        from minibench.visualization.xiangqi_suite import comparison_rows
        with tempfile.TemporaryDirectory() as tmp:
            suite, plan = self.fixture(Path(tmp))
            rows = comparison_rows(suite)
            self.assertTrue(rows)
            self.assertTrue(all(r["value"] is None and r["completed"] == 0 for r in rows))
            c2 = [r for r in rows if r["task"] == "c2"]
            self.assertEqual({r["mode"] for r in c2},
                             {"all", "standard", "horse-no-leg-block", "chariot-no-center", "soldier-free-retreat"})
            self.assertTrue(all(r["metric"] != "avg_cp_loss" for r in c2))
            self.assertEqual(sum(r["planned"] for r in c2 if r["metric"] == "mate_within_shortest_rate" and r["mode"] != "all"), 6)

    def test_credentials_are_rejected_before_preparation(self):
        for value in ({"api_key": "private"}, {"extra_body": {"headers": {}}},
                      {"unknown": "sk-test-not-a-real-key"}):
            with self.assertRaises(ValueError):
                cloud.reject_secrets(value)

    def test_headless_plot_exports_without_missing_font_glyphs(self):
        import warnings
        from minibench.visualization.xiangqi_suite import plot_suite
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            plan = {"agents": ["direct", "cot"], "experiments": {
                "d3": {"family": "xiangqi-mate-in-one", "condition_counts": {"native": 2}},
            }}
            atomic_write_json(suite / "plan.json", plan)
            atomic_write_json(suite / "results/d3/direct/results.json", {"mate_rate": .5, "legality_rate": 1.0})
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                charts = plot_suite(suite)
            self.assertEqual(len(list(charts.glob("*.png"))), 2)
            self.assertEqual(len(list(charts.glob("*.svg"))), 2)
            self.assertTrue((charts / "comparison.csv").is_file())

    def test_c2_changed_tree_is_explicitly_blocked_without_paid_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite, plan = self.fixture(Path(tmp))
            config=plan["experiments"]["c2"]["configs"]["direct"]
            directory=suite/config["evaluation"]["solution_tree_dir"]
            certificate=directory/f"{plan['experiments']['c2']['task_ids'][0]}.json"
            certificate.write_text(certificate.read_text()+"\n")
            with self.assertRaisesRegex(ValueError, "independent_tree_bytes_changed"), patch.object(cloud, "execute_job") as execute:
                cloud.run(suite, plan)
            execute.assert_not_called()

    def test_formal_agents_and_portable_other_task_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = yaml.safe_load((cloud.ROOT / "config/cloud/xiangqi-all-agents.yaml").read_text())
            raw["engine_binding"] = "reference"
            cfg = root / "formal.yaml"
            cfg.write_text(yaml.safe_dump(raw))
            plan = cloud.prepare(cfg, root / "formal")
            self.assertEqual(plan["planned_jobs"], 4500)
            self.assertEqual(len(plan["agents"]), 9)
            raw["agents"] = ["direct"]
            raw["pilot_count"] = 1
            raw["experiments"] = [
                {"id": name, "config": f"config/experiments/{name}.yaml", "account": "qwen"}
                for name in ["one_stroke_multimodal", "mahjong_multimodal", "mahjong_rule_variants", "zebra"]
            ]
            cfg.write_text(yaml.safe_dump(raw))
            suite = root / "other"
            plan = cloud.prepare(cfg, suite)
            cloud.verify_plan(suite)
            self.assertEqual(plan["planned_jobs"], 4)
            for name in ["one_stroke_multimodal", "mahjong_multimodal"]:
                exp = plan["experiments"][name]
                tasks = cloud.get_task_family_spec(exp["family"]).load_tasks(suite / exp["source"])
                self.assertTrue(tasks[0].image_path.is_file())
                self.assertTrue(tasks[0].image_path.is_relative_to(suite))

    def test_certified_c2_passes_normal_gate_before_any_api_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = yaml.safe_load((cloud.ROOT / 'config/cloud/xiangqi-pilot.yaml').read_text())
            raw['experiments'] = [e for e in raw['experiments'] if e['id'] == 'c2']
            raw['engine_binding'] = 'reference'
            raw['accounts']['qwen']['provider']['api_key_env'] = 'UNIT_TEST_UNUSED_KEY'
            config = root / 'cloud.yaml'
            config.write_text(yaml.safe_dump(raw))
            suite = root / 'suite'
            plan = cloud.prepare(config, suite)
            with patch.dict(os.environ, {'UNIT_TEST_UNUSED_KEY': ''}), patch.object(cloud, 'execute_job') as execute:
                with self.assertRaisesRegex(ValueError, 'Set transient environment variable'):
                    cloud.run(suite, plan)
                execute.assert_not_called()


class CloudHTTPIntegrationTests(unittest.TestCase):
    def test_spawn_workers_and_resume_without_new_requests(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from minibench.datasets.xiangqi.schema import load_records, FAMILY_PATHS
        active, high, calls = [0], [0], []
        fail_on = set()
        lock = threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    active[0] += 1
                    high[0] = max(high[0], active[0])
                    calls.append(body)
                    number = len(calls)
                if number in fail_on:
                    self.send_error(503, "simulated interruption")
                    with lock:
                        active[0] -= 1
                    return
                time.sleep(.15)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunk = {"id": "local-test", "model": "fixture",
                         "choices": [{"index": 0, "delta": {"content": '{"move":"a0a0"}'}, "finish_reason": "stop"}]}
                end = {"choices": [], "usage": {"total_tokens": 7}}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\ndata: " + json.dumps(end) + "\n\ndata: [DONE]\n\n").encode())
                self.wfile.flush()
                with lock:
                    active[0] -= 1
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"UNIT_TEST_CLOUD_KEY": "local-only-sentinel"}):
            root = Path(tmp)
            raw = yaml.safe_load((cloud.ROOT / "config/cloud/xiangqi-pilot.yaml").read_text())
            raw["experiments"] = [e for e in raw["experiments"] if e["id"] == "c2"]
            raw["engine_binding"] = "reference"
            raw["accounts"]["qwen"]["provider"].update(
                base_url=f"http://127.0.0.1:{server.server_port}/v1", model="fixture", api_key_env="UNIT_TEST_CLOUD_KEY")
            raw["accounts"]["qwen"]["limits"]["max_in_flight"] = 2
            cfg = root / "cloud.yaml"
            cfg.write_text(yaml.safe_dump(raw))
            suite = root / "suite"
            plan = cloud.prepare(cfg, suite)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cloud.run(suite, plan, workers=8, allow_uncertified_c2=True), 0)
                before = len(calls)
                self.assertEqual(cloud.run(suite, plan, workers=16, allow_uncertified_c2=True), 0)
            self.assertEqual(before, 6)
            self.assertEqual(len(calls), before)
            self.assertLessEqual(high[0], 2)
            self.assertEqual(len(cloud.completed_jobs(suite, plan)), 6)
            summary = cloud.report(suite, plan, cloud.completed_jobs(suite, plan))
            self.assertEqual(summary["usage"]["known_success_tokens"], 42)
            self.assertTrue(all("max_tokens" not in body for body in calls))
            for p in suite.rglob("*.json"):
                self.assertNotIn("local-only-sentinel", p.read_text())

            # An actual new 16-worker run, not merely an already-complete resume.
            raw["pilot_count"] = 18
            raw["workers"] = 16
            raw["accounts"]["qwen"]["limits"]["max_in_flight"] = 4
            cfg.write_text(yaml.safe_dump(raw))
            suite16 = root / "suite16"
            plan16 = cloud.prepare(cfg, suite16)
            previous = len(calls)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cloud.run(suite16, plan16, allow_uncertified_c2=True), 0)
            self.assertEqual(len(calls) - previous, 18)
            self.assertEqual(len(cloud.completed_jobs(suite16, plan16)), 18)
            self.assertLessEqual(high[0], 4)

            # One completed result, then HTTP failure: no fault is scored done;
            # explicit manual resume skips the first job without rebilling it.
            raw["pilot_count"] = 6
            raw["workers"] = 1
            cfg.write_text(yaml.safe_dump(raw))
            failed_suite = root / "failed"
            failed_plan = cloud.prepare(cfg, failed_suite)
            previous = len(calls)
            fail_on.add(previous + 2)
            with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                cloud.run(failed_suite, failed_plan, allow_uncertified_c2=True)
            self.assertEqual(len(cloud.completed_jobs(failed_suite, failed_plan)), 1)
            self.assertEqual(json.loads((failed_suite / "state.json").read_text())["status"], "error")
            fail_on.clear()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cloud.run(failed_suite, failed_plan, allow_uncertified_c2=True), 0)
            self.assertEqual(len(calls) - previous, 7)  # 6 saved successes + 1 failure
            self.assertEqual(len(cloud.completed_jobs(failed_suite, failed_plan)), 6)
            summary = cloud.report(failed_suite, failed_plan, cloud.completed_jobs(failed_suite, failed_plan))
            self.assertEqual(summary["usage"]["saved_successful_calls"], 6)
            self.assertEqual(summary["usage"]["unfinished_attempts"], 1)

            # Parallel failure with active AND idle workers must exit promptly,
            # with no manager/pool child left behind or failed job scored done.
            raw["workers"] = 8
            cfg.write_text(yaml.safe_dump(raw))
            parallel_suite = root / "parallel-failure"
            parallel_plan = cloud.prepare(cfg, parallel_suite)
            children_before = {p.pid for p in mp.active_children()}
            fail_on.add(len(calls) + 2)
            started = time.monotonic()
            with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                cloud.run(parallel_suite, parallel_plan, allow_uncertified_c2=True)
            self.assertLess(time.monotonic() - started, 20)
            state = json.loads((parallel_suite / "state.json").read_text())
            self.assertEqual(state["status"], "error")
            self.assertEqual(state["active_jobs"], [])
            self.assertTrue(all(a["active_calls"] == 0 for a in state["accounts"].values()))
            self.assertEqual(state["forced_worker_pids"], [])
            self.assertEqual({p.pid for p in mp.active_children()}, children_before)
            summary = json.loads((parallel_suite / "summary.json").read_text())
            self.assertEqual(summary["usage"]["known_total_tokens"],
                             summary["usage"]["known_success_tokens"] + summary["usage"]["known_failed_tokens"])
            self.assertGreaterEqual(summary["usage"]["attempts_without_usage"], 1)


if __name__ == "__main__":
    unittest.main()

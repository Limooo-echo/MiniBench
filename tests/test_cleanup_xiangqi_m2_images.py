"""Offline safety and orchestration tests; never call a model or live suite."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from minibench.core.checkpoint import RunLock, RunLockedError, atomic_write_json, fingerprint_payload, sha256_file
from scripts.maintenance import cleanup_xiangqi_m2_images as tool


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.suite = Path(self.temp.name) / "suite"
        self.suite.mkdir()
        self.png = b"generated PNG fixture"
        self.plan = self.fixture()

    def write_plan(self):
        self.plan.pop("plan_sha256", None)
        self.plan["planned_jobs"] = len(self.plan["jobs"])
        self.plan["plan_sha256"] = fingerprint_payload(self.plan)
        atomic_write_json(self.suite / "plan.json", self.plan)

    def fixture(self):
        plan = {"version": "minibench-cloud-v1", "agents": ["direct"], "experiments": {}, "jobs": [], "files": []}
        for name, family, modes in [
            ("d3", "xiangqi-mate-in-one", ["native"]),
            ("h2", "xiangqi-history", ["full-state", "move-history-only"]),
            ("c2", "xiangqi-rule-variants", ["native"]),
            ("m2", "xiangqi-multimodal", ["text", "chinese-piece-image"]),
        ]:
            plan["experiments"][name] = {"family": family, "task_ids": [name + "-0001"],
                                         "record_count": 1, "configs": {"direct": {}},
                                         "condition_counts": {m: 1 for m in modes} if name != "c2" else {"all": 1, "standard": 1}}
            for mode in modes:
                identity = [name, "direct", name + "-0001", mode]
                plan["jobs"].append(dict(zip(["experiment", "agent", "task_id", "mode"], identity), key=fingerprint_payload(identity)))
        asset = self.suite / "inputs" / "original.png"
        asset.parent.mkdir()
        asset.write_bytes(b"original input asset must remain")
        plan["files"].append({"path": "inputs/original.png", "sha256": sha256_file(asset)})
        self.plan = plan
        self.write_plan()
        for job in plan["jobs"]:
            directory = self.suite / "jobs" / job["key"]
            atomic_write_json(directory / "state.json", {"status": "completed"})
            row = {"task_id": job["task_id"], "status": "ok", "success": False, "is_legal": False,
                   "diagnostics": [], "steps": [], "input_mode": job["mode"]}
            if job["mode"] == "chinese-piece-image":
                image = directory / "images" / job["task_id"] / "chinese-piece-image_step00.png"
                image.parent.mkdir(parents=True)
                image.write_bytes(self.png)
                row["steps"] = [{"input_image_path": str(image), "input_image_sha256": sha256_file(image)}]
            result = {"job": job, "plan_sha256": plan["plan_sha256"], "records": [row]}
            result["sha256"] = fingerprint_payload(result)
            atomic_write_json(directory / "result.json", result)
            atomic_write_json(directory / "calls" / "0000.request.json", {"image_url": "encoded image remains"})
        atomic_write_json(self.suite / "state.json", {"status": "completed", "completed_jobs": len(plan["jobs"]),
                          "planned_jobs": len(plan["jobs"]), "active_jobs": [], "accounts": {"qwen": {"active_calls": 0}}})
        charts = self.suite / "charts"
        charts.mkdir()
        (charts / "comparison.png").write_bytes(b"plot must remain")
        return plan

    def image(self):
        return next(self.suite.glob("jobs/*/images/*/*.png"))

    def image_job(self):
        return next(j for j in self.plan["jobs"] if j["mode"] == "chinese-piece-image")

    def edit_state(self, **values):
        path = self.suite / "state.json"
        state = json.loads(path.read_text())
        state.update(values)
        atomic_write_json(path, state)

    def edit_result(self, job, mutate):
        path = self.suite / "jobs" / job["key"] / "result.json"
        result = json.loads(path.read_text())
        result.pop("sha256")
        mutate(result)
        result["sha256"] = fingerprint_payload(result)
        atomic_write_json(path, result)

    def test_preview_is_non_destructive(self):
        value = tool.cleanup(self.suite)
        self.assertEqual(value["candidate_files"], 1)
        self.assertTrue(self.image().exists())
        self.assertFalse((self.suite / tool.RECEIPT_NAME).exists())

    def test_apply_only_removes_verified_m2_png_and_is_idempotent(self):
        paths = [p for p in self.suite.rglob("*") if p.is_file()]
        protected = {p: p.read_bytes() for p in paths if p != self.image()}
        value = tool.cleanup(self.suite, apply=True)
        self.assertEqual(value["deleted_this_run"], 1)
        self.assertEqual(len(list(self.suite.glob("jobs/*/images/*/*.png"))), 0)
        for path, content in protected.items():
            self.assertEqual(path.read_bytes(), content)
        receipt = json.loads((self.suite / tool.RECEIPT_NAME).read_text())
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["deleted_files"], 1)
        again = tool.cleanup(self.suite, apply=True)
        self.assertEqual(again["deleted_this_run"], 0)
        self.assertEqual(again["previously_removed_files"], 1)

    def test_failure_pause_or_stale_counter_refuses(self):
        original = json.loads((self.suite / "state.json").read_text())
        for change in [{"status": "error"}, {"status": "running"}, {"status": "interrupted"},
                       {"completed_jobs": 5}, {"active_jobs": ["h2"]}, {"accounts": {"qwen": {"active_calls": 1}}}]:
            with self.subTest(change=change):
                atomic_write_json(self.suite / "state.json", {**original, **change})
                with self.assertRaises(ValueError):
                    tool.cleanup(self.suite, apply=True)
                self.assertTrue(self.image().exists())

    def test_only_one_family_or_missing_group_refuses(self):
        self.plan["experiments"].pop("c2")
        self.plan["jobs"] = [j for j in self.plan["jobs"] if j["experiment"] != "c2"]
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "must all be included"):
            tool.cleanup(self.suite, apply=True)

    def test_missing_job_despite_complete_counter_refuses(self):
        h2 = next(j for j in self.plan["jobs"] if j["mode"] == "move-history-only")
        (self.suite / "jobs" / h2["key"] / "result.json").unlink()
        with self.assertRaises(FileNotFoundError):
            tool.cleanup(self.suite, apply=True)
        self.assertTrue(self.image().exists())

    def test_plan_cannot_drop_a_declared_condition(self):
        self.plan["jobs"] = [j for j in self.plan["jobs"] if j["mode"] != "move-history-only"]
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "every planned"):
            tool.cleanup(self.suite, apply=True)

    def test_an_unfinished_declared_agent_refuses(self):
        self.plan["agents"].append("cot")
        for experiment in self.plan["experiments"].values():
            experiment["configs"]["cot"] = {}
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "every planned"):
            tool.cleanup(self.suite, apply=True)
        self.assertTrue(self.image().exists())

    def test_bad_plan_digest_refuses(self):
        self.plan["planned_jobs"] = 123
        atomic_write_json(self.suite / "plan.json", self.plan)
        with self.assertRaisesRegex(ValueError, "Integrity mismatch"):
            tool.cleanup(self.suite, apply=True)

    def test_bad_result_digest_or_diagnostic_refuses(self):
        job = self.image_job()
        self.edit_result(job, lambda r: r["records"][0].update(diagnostics=[{"error": "engine failure"}]))
        with self.assertRaisesRegex(ValueError, "evaluation failure"):
            tool.cleanup(self.suite, apply=True)
        self.edit_result(job, lambda r: r["records"][0].update(diagnostics=[]))
        p = self.suite / "jobs" / job["key"] / "result.json"
        r = json.loads(p.read_text());r["sha256"] = "bad";atomic_write_json(p, r)
        with self.assertRaisesRegex(ValueError, "Integrity mismatch"):
            tool.cleanup(self.suite, apply=True)

    def test_infrastructure_reason_refuses(self):
        self.edit_result(self.image_job(), lambda r: r["records"][0].update(reasons=["pikafish_error: failed"]))
        with self.assertRaisesRegex(ValueError, "infrastructure failure"):
            tool.cleanup(self.suite, apply=True)

    def test_changed_or_unknown_image_refuses_before_delete(self):
        image = self.image()
        extra = image.with_name("unknown.png");extra.write_bytes(self.png)
        with self.assertRaisesRegex(ValueError, "Unexpected PNG"):
            tool.cleanup(self.suite, apply=True)
        self.assertTrue(image.exists());extra.unlink()
        image.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "Generated image changed"):
            tool.cleanup(self.suite, apply=True)
        self.assertTrue(image.exists())

    def test_missing_image_without_cleanup_receipt_refuses(self):
        self.image().unlink()
        with self.assertRaisesRegex(ValueError, "without a matching"):
            tool.cleanup(self.suite, apply=True)

    def test_pending_receipt_recovers_a_crash_after_unlink(self):
        tool.cleanup(self.suite, apply=True)
        p = self.suite / tool.RECEIPT_NAME
        receipt = json.loads(p.read_text());receipt["status"] = "in_progress";receipt["files"][0]["deleted"] = False
        atomic_write_json(p, receipt)
        self.assertEqual(tool.cleanup(self.suite, apply=True)["deleted_this_run"], 0)
        self.assertTrue(json.loads(p.read_text())["files"][0]["deleted"])

    def test_active_run_lock_refuses(self):
        with RunLock(self.suite):
            with self.assertRaises(RunLockedError):
                tool.cleanup(self.suite, apply=True)
        self.assertTrue(self.image().exists())

    def test_symlink_file_directory_or_receipt_refuses(self):
        image = self.image()
        external = Path(self.temp.name) / "outside.png"
        external.write_bytes(self.png)
        image.unlink();image.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "symlink"):
            tool.cleanup(self.suite, apply=True)
        self.assertTrue(external.exists())
        image.unlink();image.write_bytes(self.png)
        folder = image.parent;real_folder = folder.with_name("original-folder");folder.rename(real_folder)
        folder.symlink_to(real_folder, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            tool.cleanup(self.suite, apply=True)
        folder.unlink();real_folder.rename(folder)
        (self.suite / tool.RECEIPT_NAME).symlink_to(external)
        with self.assertRaisesRegex(ValueError, "symlink"):
            tool.cleanup(self.suite, apply=True)
        self.assertEqual(external.read_bytes(), self.png)

    def test_cli_nonzero_for_incomplete_and_zero_for_preview(self):
        self.assertEqual(tool.main(["--suite-dir", str(self.suite)]), 0)
        self.edit_state(status="interrupted")
        self.assertEqual(tool.main(["--suite-dir", str(self.suite), "--apply"]), 2)


class ShellIntegrationTests(unittest.TestCase):
    def test_cleanup_is_after_run_and_plot_and_not_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);suite = root / "suite";suite.mkdir();(suite / "plan.json").write_text("{}")
            stub = root / "python-stub"
            stub.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                            "with open(os.environ['TEST_CALLS'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                            "sys.exit(9 if os.environ.get('TEST_FAIL_PHASE') in sys.argv[1:] else 0)\n")
            stub.chmod(0o700)
            log = root / "calls.jsonl"
            env = {**os.environ, "MINIBENCH_PYTHON": str(stub), "TEST_CALLS": str(log)}
            cmd = ["bash", str(tool.ROOT / "scripts/cloud_run.sh"), "unused.yaml", str(suite), "--workers", "8"]
            for phase in ["none", "run", "plot"]:
                if log.exists():log.unlink()
                proc = subprocess.run(cmd + ["--cleanup-m2-images"], env={**env, "TEST_FAIL_PHASE": phase}, capture_output=True)
                calls = [json.loads(l) for l in log.read_text().splitlines()]
                self.assertIn("run", calls[0]);self.assertNotIn("--cleanup-m2-images", calls[0])
                if phase == "none":
                    self.assertEqual(proc.returncode, 0);self.assertEqual(len(calls), 3)
                    self.assertIn("plot", calls[1]);self.assertIn("--apply", calls[2])
                else:
                    self.assertEqual(proc.returncode, 9);self.assertEqual(len(calls), 1 if phase == "run" else 2)
            log.unlink()
            proc = subprocess.run(cmd, env=env, capture_output=True)
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(len(log.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()

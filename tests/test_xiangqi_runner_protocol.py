"""Offline checks for frozen smoke preparation and durable manual-browser runs."""
from __future__ import annotations
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import yaml

from scripts import run_xiangqi_smoke as smoke, xiangqi_web_test as web
from minibench.core.multimodal import ImageAttachment
from minibench.factory.experiments import get_task_family_spec, _evaluate


class RunnerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        seed = json.loads((smoke.ROOT / "data/xiangqi/mate_in_one/tasks.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.families = {}
        for key, family in (("d3", "xiangqi-mate-in-one"), ("h2", "xiangqi-history")):
            record = deepcopy(seed)
            record.update(id=family + "-fixture", family=family)
            if key == "h2":
                record.update(difficulty="short", max_plies=5)
            other = deepcopy(record)
            other["id"] += "-other"
            source, selected, config_path = self.root / f"{key}-source.jsonl", self.root / f"{key}-selected.jsonl", self.root / f"{key}.yaml"
            source.write_text(json.dumps(other) + "\n" + json.dumps(record) + "\n")
            selected.write_text(json.dumps(record) + "\n")
            config = {"task": {"family": family, "path": str(source), "sampling": {"enabled": False},
                               "selection": {"path": str(selected), "sha256": smoke._sha256(selected)}},
                      "agent": {"name": "direct"},
                      "provider": {"name": "qwen", "model": "fixture-model", "api_key_env": "NEVER_CALL_A_MODEL"},
                      "evaluation": {"history_mode": "paired", "pikafish_depth": 8,
                                     "pikafish_binary_sha256": "a" * 64, "pikafish_nnue_sha256": "b" * 64},
                      "run": {"output_dir": "runs", "run_name": key}}
            config_path.write_text(yaml.safe_dump(config))
            self.families[key] = family, config_path, source, selected
        self.suite = self.root / "suite"
        self.engine = self.root / "engine.exe"
        self.engine.write_bytes(b"unused-engine")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(smoke, "FAMILIES", self.families))
        self.stack.enter_context(patch.object(smoke, "pikafish_fingerprint", return_value={"binary_sha256": "a" * 64, "eval_file_sha256": "b" * 64}))
        self.stack.enter_context(patch.object(web, "pikafish_fingerprint", return_value={"binary_sha256": "a" * 64, "eval_file_sha256": "b" * 64}))

    def prepare(self):
        return smoke.prepare_suite(self.suite, ["d3", "h2"], pikafish_path=self.engine)

    def test_prepare_only_needs_no_credential_and_keeps_full_dataset(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(smoke, "run_experiment") as run:
            status = smoke.main(["--tasks", "d3,h2", "--output-dir", str(self.suite), "--pikafish-path", str(self.engine), "--prepare-only"])
        self.assertEqual(status, 0)
        run.assert_not_called()
        plan = json.loads((self.suite / "smoke_plan.json").read_text())
        for key in ("d3", "h2"):
            config = json.loads((self.suite / "configs" / f"{key}.json").read_text())
            self.assertEqual(config["task"]["path"], str(self.families[key][2]))
            self.assertEqual(config["task"]["selection"]["sha256"], smoke._sha256(self.families[key][3]))
            self.assertEqual(plan["tasks"][key]["selected_record_count"], 1)
            self.assertEqual(config["evaluation"]["pikafish_depth"], 16 if key == "h2" else 8)
            self.assertEqual(config["evaluation"]["pikafish_binary_sha256"], "a" * 64)
        self.assertEqual(json.loads((self.suite / "suite_results.json").read_text())["status"], "prepared")

    def test_complete_suite_plan_exists_before_first_model_setup(self):
        calls = []
        def run(config):
            self.assertTrue((self.suite / "smoke_plan.json").is_file())
            self.assertTrue((self.suite / "configs/h2.json").is_file())
            self.assertTrue((self.suite / "samples/h2.jsonl").is_file())
            calls.append(config["task"]["family"])
            return {"total": 1}
        with patch.dict(os.environ, {"NEVER_CALL_A_MODEL": "private-test-sentinel"}), patch.object(smoke, "run_experiment", side_effect=run):
            status = smoke.main(["--tasks", "d3,h2", "--output-dir", str(self.suite), "--pikafish-path", str(self.engine)])
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("private-test-sentinel", (self.suite / "configs/d3.json").read_text())

    def test_existing_suite_is_never_truncated(self):
        self.prepare()
        before = (self.suite / "smoke_plan.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual((self.suite / "smoke_plan.json").read_bytes(), before)

    def test_frozen_hash_and_source_record_changes_are_rejected(self):
        family, cfg, source, selected = self.families["d3"]
        original = cfg.read_text()
        config = yaml.safe_load(original)
        config["task"]["selection"]["sha256"] = "0" * 64
        cfg.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.prepare()
        cfg.write_text(original)
        self.suite = self.root / "suite-second"
        record = json.loads(selected.read_text())
        record["tags"] = sorted(record["tags"] + ["changed"])
        selected.write_text(json.dumps(record) + "\n")
        config = yaml.safe_load(original)
        config["task"]["selection"]["sha256"] = smoke._sha256(selected)
        cfg.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, "differs from source"):
            self.prepare()

    def test_web_preparation_keeps_h2_depth_and_actual_config(self):
        self.prepare()
        session = self.root / "web"
        prepared = web.prepare_web_session(self.suite, session, ["d3", "h2"])
        config = prepared["h2"][2]
        self.assertEqual(config["evaluation"]["pikafish_depth"], 16)
        self.assertEqual(config["evaluation"]["pikafish_nnue_sha256"], "b" * 64)
        metadata = json.loads((session / "h2/results/run_metadata.json").read_text())
        self.assertEqual(metadata["engine"]["binary_sha256"], "a" * 64)
        self.assertEqual(len(prepared["h2"][1]), 1)
        with self.assertRaises(FileExistsError):
            web.prepare_web_session(self.suite, session, ["h2"])

    def test_web_interruption_retains_each_completed_result(self):
        self.prepare()
        session = self.root / "web"
        def evaluator(*args, on_result, **kwargs):
            record = {"task_id": "completed", "status": "ok", "success": True, "input_mode": "text"}
            on_result(record)
            saved = session / "d3/results/predictions.jsonl"
            self.assertEqual(json.loads(saved.read_text()), record)
            self.assertTrue((session / "h2/results/resolved_config.json").is_file())
            raise KeyboardInterrupt()
        with patch.object(web, "_evaluate", side_effect=evaluator), patch("builtins.input", side_effect=AssertionError("no web calls")):
            status = web.main(["--suite-dir", str(self.suite), "--tasks", "d3,h2", "--output-dir", str(session)])
        self.assertEqual(status, 1)
        state = json.loads((session / "d3/results/run_state.json").read_text())
        self.assertEqual((state["status"], state["completed_results"]), ("stopped_by_user", 1))
        self.assertEqual(len((session / "d3/results/predictions.jsonl").read_text().splitlines()), 1)

    def test_web_summary_writer_cannot_overwrite_checkpoint(self):
        self.prepare()
        session = self.root / "web"
        spec = get_task_family_spec("xiangqi-mate-in-one")
        record = {"task_id": "completed", "status": "ok", "success": True}
        def evaluator(*args, on_result, **kwargs):
            on_result(record)
            return [record]
        def writer(results, output_dir, run_name, *, write_predictions=True):
            self.assertFalse(write_predictions)
            self.assertEqual(json.loads((Path(output_dir) / run_name / "predictions.jsonl").read_text()), record)
            return Path(output_dir) / run_name
        with patch.object(web, "get_task_family_spec", return_value=replace(spec, summarize=lambda r: {"total": len(r)}, write_run=writer)), patch.object(web, "_evaluate", side_effect=evaluator):
            status = web.main(["--suite-dir", str(self.suite), "--tasks", "d3", "--output-dir", str(session)])
        self.assertEqual(status, 0)

    def test_factory_forwards_pinned_engine_hashes_and_h2_default_depth(self):
        for family in ("xiangqi-mate-in-one", "xiangqi-history", "xiangqi-multimodal"):
            with self.subTest(family=family):
                evaluate = Mock(return_value=[])
                spec = replace(get_task_family_spec(family), evaluate_tasks=evaluate)
                _evaluate(spec, [], object(), family, {"pikafish_binary_sha256": "a" * 64, "pikafish_nnue_sha256": "b" * 64})
                kwargs = evaluate.call_args.kwargs
                self.assertEqual(kwargs["pikafish_binary_sha256"], "a" * 64)
                self.assertEqual(kwargs["pikafish_nnue_sha256"], "b" * 64)
                self.assertEqual(kwargs["pikafish_depth"], 16 if family == "xiangqi-history" else 8)

    def test_manual_reset_does_not_overwrite_images_or_reset_usage(self):
        agent = web.ManualWebAgent(self.root / "manual")
        with patch("builtins.input", return_value='{"move":"a0a1"}'), patch("builtins.print"):
            for _ in range(2):
                agent.reset()
                agent.generate_multimodal("prompt", {"id": "same"}, images=[ImageAttachment(data=b"png", mime_type="image/png")])
        self.assertEqual(len(list(agent.image_dir.glob("*.png"))), 2)
        self.assertEqual(agent.metrics_snapshot()["llm_calls"], 2)


if __name__ == "__main__":
    unittest.main()

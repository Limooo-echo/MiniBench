import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.datasets.one_stroke.dataset import load_one_stroke_tasks
from minibench.datasets.one_stroke.evaluation import plan_one_stroke_work_items
from minibench.factory.experiments import (
    _one_stroke_dataset_profile,
    _one_stroke_input_assets,
    run_family_experiment,
)


def one_stroke_record(task_id):
    return {
        "id": task_id,
        "difficulty": "easy",
        "vertices": ["A", "B", "C"],
        "edges": [["A", "B"], ["B", "C"]],
        "start": "A",
        "end": "C",
        "solution_path": ["A", "B", "C"],
        "tags": ["one-stroke", "difficulty:easy", "source:resume-test"],
    }


def write_tasks(path, task_ids):
    path.write_text(
        "".join(json.dumps(one_stroke_record(task_id)) + "\n" for task_id in task_ids),
        encoding="utf-8",
    )


def experiment_config(root, tasks_path, *, run_name="one-stroke-resume"):
    return {
        "task": {
            "family": "one_stroke",
            "path": str(tasks_path),
            "limit": None,
            "task_ids": [],
        },
        "agent": {"name": "openai-compatible"},
        "provider": {
            "name": "generic",
            "model": "unit-model",
            "base_url": "https://unit.invalid",
            "api_key_env": "UNIT_TEST_API_KEY",
        },
        "evaluation": {
            "prompt_variant": "baseline",
            "memory_modes": ["incremental_state", "step_history_only"],
            "input_modes": ["image"],
        },
        "run": {
            "output_dir": str(root / "runs"),
            "run_name": run_name,
            "on_existing": "resume",
        },
    }


class DirectAgent:
    def __init__(self, *, fail_after=None, required_artifacts=None):
        self.fail_after = fail_after
        self.required_artifacts = required_artifacts
        self.task_ids = []

    def generate(self, prompt, task):
        if self.required_artifacts is not None:
            for name in (
                "manifest.json",
                "run_state.json",
                "predictions.jsonl",
            ):
                if not (self.required_artifacts / name).is_file():
                    raise AssertionError(f"missing pre-call artifact: {name}")
        self.task_ids.append(task.id)
        if self.fail_after is not None and len(self.task_ids) > self.fail_after:
            raise RuntimeError("injected one-stroke failure")
        return json.dumps({"path": ["A", "B", "C"]})


class HistoryAgent:
    def __init__(self, *, fail_on_call=None):
        self.fail_on_call = fail_on_call
        self.calls = []

    def generate_messages_for_phase(
        self,
        messages,
        task,
        *,
        phase,
        max_tokens=None,
        json_mode=None,
    ):
        self.calls.append(phase)
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("injected history failure")
        if phase == "final":
            return json.dumps({"path": ["B", "C", "A"]})
        if "complete intermediate state" in messages[-1]["content"]:
            return json.dumps(
                {
                    "current_vertex": "B",
                    "used_edges": ["e01"],
                    "remaining_edges": ["e02", "e03"],
                }
            )
        return json.dumps({"step": 1})


class OneStrokeResumeTests(unittest.TestCase):
    def test_interruption_is_atomic_and_resume_only_runs_missing_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(tasks_path, ["task-0", "task-1"])
            config = experiment_config(root, tasks_path)
            run_dir = root / "runs" / "one-stroke-resume"
            first_agent = DirectAgent(
                fail_after=1,
                required_artifacts=run_dir,
            )

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=first_agent,
            ):
                with self.assertRaisesRegex(RuntimeError, "1/2 results") as raised:
                    run_family_experiment(config)

            self.assertIn("run.run_name='one-stroke-resume'", str(raised.exception))
            self.assertIn("run.on_existing='resume'", str(raised.exception))

            self.assertEqual(first_agent.task_ids, ["task-0", "task-1"])
            predictions = [
                json.loads(line)
                for line in (run_dir / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["task_id"] for row in predictions], ["task-0"])
            state = json.loads(
                (run_dir / "run_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["completed_total"], 1)
            self.assertEqual(state["current_work_key"]["task_id"], "task-1")

            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["family"], "one_stroke")
            self.assertEqual(manifest["task_data"]["path"], str(tasks_path.resolve()))
            self.assertEqual(len(manifest["task_data"]["sha256"]), 64)
            self.assertEqual(manifest["model_identity"]["provider"], "generic")
            self.assertEqual(manifest["model_identity"]["model"], "unit-model")
            self.assertEqual(
                [item["task_id"] for item in manifest["work_plan"]],
                ["task-0", "task-1"],
            )
            self.assertIn("resolved_config", manifest)
            self.assertIn("head", manifest["git"])
            self.assertEqual(len(manifest["fingerprint"]), 64)
            self.assertEqual(manifest["input_assets"], [])
            self.assertEqual(
                manifest["fingerprint_inputs"]["input_asset_hashes"],
                [],
            )
            self.assertEqual(manifest["dataset_profile"]["task_total"], 2)
            self.assertEqual(manifest["dataset_profile"]["work_item_total"], 2)
            self.assertEqual(
                manifest["dataset_profile"]["by_capability"],
                {"direct": 2},
            )
            warning_codes = {
                warning["code"]
                for warning in manifest["interpretation_warnings"]
            }
            self.assertIn(
                "cross_track_raw_scores_not_attributable",
                warning_codes,
            )

            second_agent = DirectAgent()
            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=second_agent,
            ):
                resumed_dir, summary = run_family_experiment(config)

            self.assertEqual(resumed_dir, run_dir)
            self.assertEqual(second_agent.task_ids, ["task-1"])
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["success"], 2)
            final_state = json.loads(
                (run_dir / "run_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(final_state["status"], "completed")
            self.assertEqual(final_state["completed_total"], 2)
            self.assertEqual(final_state["resume_count"], 1)
            self.assertTrue((run_dir / "results.json").is_file())
            self.assertTrue((run_dir / "summary.txt").is_file())

    def test_history_resume_retries_only_incomplete_memory_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "history.jsonl"
            record = {
                "id": "history-task",
                "capability": "history_memory",
                "difficulty": "easy",
                "vertices": ["A", "B", "C"],
                "edges": [["A", "B"], ["B", "C"], ["C", "A"]],
                "start": "A",
                "end": "A",
                "solution_path": ["A", "B", "C", "A"],
                "history_events": [
                    {
                        "action": "move",
                        "edge_id": "e01",
                        "from": "A",
                        "to": "B",
                    }
                ],
                "tags": ["one-stroke", "difficulty:easy"],
            }
            tasks_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            config = experiment_config(root, tasks_path, run_name="history-resume")
            first_agent = HistoryAgent(fail_on_call=3)

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=first_agent,
            ):
                with self.assertRaisesRegex(RuntimeError, "1/2 results"):
                    run_family_experiment(config)

            run_dir = root / "runs" / "history-resume"
            state = json.loads(
                (run_dir / "run_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                state["current_work_key"],
                {"task_id": "history-task", "mode": "memory:step_history_only"},
            )
            first_prediction = json.loads(
                (run_dir / "predictions.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(first_prediction["memory_mode"], "incremental_state")

            resumed_agent = HistoryAgent()
            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=resumed_agent,
            ):
                _, summary = run_family_experiment(config)

            self.assertEqual(resumed_agent.calls, ["intermediate", "final"])
            self.assertEqual(summary["total"], 2)
            saved = [
                json.loads(line)
                for line in (run_dir / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [row["memory_mode"] for row in saved],
                ["incremental_state", "step_history_only"],
            )

    def test_fingerprint_mismatch_refuses_before_agent_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(tasks_path, ["task-0", "task-1"])
            config = experiment_config(root, tasks_path, run_name="mismatch")

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=DirectAgent(fail_after=1),
            ):
                with self.assertRaises(RuntimeError):
                    run_family_experiment(config)

            run_dir = root / "runs" / "mismatch"
            before = {
                name: (run_dir / name).read_bytes()
                for name in (
                    "manifest.json",
                    "run_state.json",
                    "predictions.jsonl",
                )
            }
            changed = {
                **config,
                "provider": {**config["provider"], "temperature": 0.5},
            }
            with patch(
                "minibench.factory.experiments.make_agent_from_config"
            ) as make_agent:
                with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                    run_family_experiment(changed)
                make_agent.assert_not_called()

            self.assertEqual(
                {name: (run_dir / name).read_bytes() for name in before},
                before,
            )

    def test_completed_run_is_rejected_before_agent_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(tasks_path, ["task-0"])
            config = experiment_config(root, tasks_path, run_name="completed")

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=DirectAgent(),
            ):
                run_family_experiment(config)

            with patch(
                "minibench.factory.experiments.make_agent_from_config"
            ) as make_agent:
                with self.assertRaisesRegex(ValueError, "already completed"):
                    run_family_experiment(config)
                make_agent.assert_not_called()

    def test_null_run_name_is_resolved_before_first_model_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(tasks_path, ["task-0"])
            config = experiment_config(root, tasks_path)
            config["run"] = {
                "output_dir": str(root / "runs"),
                "run_name": None,
                "on_existing": "error",
            }

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=DirectAgent(),
            ):
                run_dir, _ = run_family_experiment(config)

            self.assertRegex(run_dir.name, r"^one-stroke-\d{8}-\d{6}-\d{6}$")
            self.assertTrue((run_dir / "manifest.json").is_file())
            self.assertTrue((run_dir / "run_state.json").is_file())

    def test_dataset_profiles_capture_track_design_limits(self):
        direct_tasks = load_one_stroke_tasks("data/one_stroke/direct.jsonl")
        self.assertEqual(_one_stroke_input_assets(direct_tasks), [])
        direct_plan = plan_one_stroke_work_items(direct_tasks)
        direct_profile, _ = _one_stroke_dataset_profile(
            direct_tasks,
            work_plan=direct_plan,
            input_modes=("image",),
        )
        self.assertEqual(direct_profile["protocol_generation_total"], 30)

        history_tasks = load_one_stroke_tasks("data/one_stroke/history.jsonl")
        self.assertEqual(_one_stroke_input_assets(history_tasks), [])
        history_plan = plan_one_stroke_work_items(
            history_tasks,
            memory_modes=("incremental_state", "step_history_only"),
        )
        history_profile, history_warnings = _one_stroke_dataset_profile(
            history_tasks,
            work_plan=history_plan,
            input_modes=("image",),
        )
        self.assertEqual(history_profile["work_item_total"], 60)
        self.assertEqual(history_profile["protocol_generation_total"], 652)
        self.assertEqual(history_profile["history"]["negative_history_count"], 0)
        self.assertTrue(history_profile["history"]["no_negative_history_cases"])
        history_codes = {warning["code"] for warning in history_warnings}
        self.assertIn("history_no_negative_history_cases", history_codes)
        self.assertIn("history_transcript_context_not_persistent_memory", history_codes)

        multimodal_tasks = load_one_stroke_tasks("data/one_stroke/multimodal.jsonl")
        multimodal_assets = _one_stroke_input_assets(multimodal_tasks)
        self.assertEqual(len(multimodal_assets), 30)
        self.assertEqual(
            len({asset["task_id"] for asset in multimodal_assets}),
            30,
        )
        self.assertTrue(all(len(asset["sha256"]) == 64 for asset in multimodal_assets))
        self.assertTrue(
            all(Path(asset["path"]).is_absolute() for asset in multimodal_assets)
        )
        multimodal_plan = plan_one_stroke_work_items(
            multimodal_tasks,
            input_modes=("image",),
        )
        multimodal_profile, multimodal_warnings = _one_stroke_dataset_profile(
            multimodal_tasks,
            work_plan=multimodal_plan,
            input_modes=("image",),
        )
        self.assertEqual(multimodal_profile["protocol_generation_total"], 30)
        self.assertEqual(
            multimodal_profile["multimodal"]["selected_input_modes"],
            ["image"],
        )
        self.assertTrue(
            multimodal_profile["multimodal"]["visual_gap_not_estimable"]
        )
        multimodal_codes = {warning["code"] for warning in multimodal_warnings}
        self.assertIn("multimodal_visual_gap_not_estimable", multimodal_codes)
        self.assertIn("multimodal_report_path_transcription_and_joint", multimodal_codes)
        score_warning = next(
            warning
            for warning in multimodal_warnings
            if warning["code"] == "multimodal_report_path_transcription_and_joint"
        )
        self.assertEqual(
            score_warning["required_score_views"],
            ["multimodal_path_score", "multimodal_transcription_score", "multimodal_joint_score"],
        )
        multimodal_ablation_plan = plan_one_stroke_work_items(
            multimodal_tasks,
            input_modes=("text", "image"),
        )
        multimodal_ablation_profile, _ = _one_stroke_dataset_profile(
            multimodal_tasks,
            work_plan=multimodal_ablation_plan,
            input_modes=("text", "image"),
        )
        self.assertEqual(multimodal_ablation_profile["protocol_generation_total"], 60)

    def test_changed_multimodal_asset_refuses_resume_before_agent_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "image.png"
            image_path.write_bytes(b"image-v1")
            tasks_path = root / "multimodal.jsonl"
            record = {
                "id": "multimodal-asset-task",
                "source_task_id": "direct-source-task",
                "capability": "multimodal",
                "difficulty": "easy",
                "vertices": ["A", "B", "C"],
                "edges": [["A", "B"], ["B", "C"]],
                "start": "A",
                "end": "C",
                "solution_path": ["A", "B", "C"],
                "image": image_path.name,
                "tags": ["one-stroke", "difficulty:easy"],
            }
            tasks_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            config = experiment_config(root, tasks_path, run_name="asset-change")
            # Hash every declared image even though this interrupted run selects
            # only the text presentation of the multimodal task.
            config["evaluation"]["input_modes"] = ["text"]

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=DirectAgent(fail_after=0),
            ):
                with self.assertRaisesRegex(RuntimeError, "0/1 results"):
                    run_family_experiment(config)

            run_dir = root / "runs" / "asset-change"
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["input_assets"]), 1)
            self.assertEqual(
                len(manifest["fingerprint_inputs"]["input_asset_hashes"]),
                1,
            )
            before = {
                name: (run_dir / name).read_bytes()
                for name in (
                    "manifest.json",
                    "run_state.json",
                    "predictions.jsonl",
                )
            }
            image_path.write_bytes(b"image-v2")

            with patch(
                "minibench.factory.experiments.make_agent_from_config"
            ) as make_agent:
                with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                    run_family_experiment(config)
                make_agent.assert_not_called()

            self.assertEqual(
                {name: (run_dir / name).read_bytes() for name in before},
                before,
            )


if __name__ == "__main__":
    unittest.main()

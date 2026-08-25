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
            "rule_modes": ["full"],
            "input_modes": ["challenge_image"],
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

    def test_dataset_profiles_capture_a2_a3_a4_design_limits(self):
        a1_tasks = load_one_stroke_tasks("data/one_stroke/a1_direct.jsonl")
        self.assertEqual(_one_stroke_input_assets(a1_tasks), [])
        a1_plan = plan_one_stroke_work_items(a1_tasks)
        a1_profile, _ = _one_stroke_dataset_profile(
            a1_tasks,
            work_plan=a1_plan,
            input_modes=("challenge_image",),
        )
        self.assertEqual(a1_profile["protocol_generation_total"], 30)

        a2_tasks = load_one_stroke_tasks(
            "data/one_stroke/a2_rule_condition.jsonl"
        )
        self.assertEqual(_one_stroke_input_assets(a2_tasks), [])
        a2_main_plan = plan_one_stroke_work_items(
            a2_tasks,
            rule_modes=("full",),
        )
        a2_main_profile, _ = _one_stroke_dataset_profile(
            a2_tasks,
            work_plan=a2_main_plan,
            input_modes=("challenge_image",),
        )
        self.assertEqual(a2_main_profile["protocol_generation_total"], 30)
        self.assertEqual(
            a2_main_profile["solution_exists_semantics"],
            "task_base_graph",
        )
        self.assertEqual(
            a2_main_profile["by_solution_exists"],
            {"true": 30, "false": 0},
        )
        self.assertEqual(
            a2_main_profile["rule_condition"]["by_selected_mode"]["full"],
            {
                "task_total": 30,
                "constrained_solvable": 24,
                "constrained_unsolvable": 6,
            },
        )
        a2_plan = plan_one_stroke_work_items(
            a2_tasks,
            rule_modes=(
                "full",
                "standard",
                "drop_key_rule",
                "conflicting_rule",
            ),
        )
        a2_profile, a2_warnings = _one_stroke_dataset_profile(
            a2_tasks,
            work_plan=a2_plan,
            input_modes=("challenge_image",),
        )
        self.assertEqual(a2_profile["task_total"], 30)
        self.assertEqual(a2_profile["work_item_total"], 120)
        self.assertEqual(a2_profile["protocol_generation_total"], 120)
        self.assertEqual(
            a2_profile["rule_condition"][
                "standard_drop_key_rule_same_task_count"
            ],
            10,
        )
        self.assertIn(
            "a2_standard_drop_key_rule_equivalent",
            {warning["code"] for warning in a2_warnings},
        )

        a3_tasks = load_one_stroke_tasks("data/one_stroke/a3_history.jsonl")
        self.assertEqual(_one_stroke_input_assets(a3_tasks), [])
        a3_plan = plan_one_stroke_work_items(
            a3_tasks,
            memory_modes=("incremental_state", "step_history_only"),
        )
        a3_profile, a3_warnings = _one_stroke_dataset_profile(
            a3_tasks,
            work_plan=a3_plan,
            input_modes=("challenge_image",),
        )
        self.assertEqual(a3_profile["work_item_total"], 60)
        self.assertEqual(a3_profile["protocol_generation_total"], 652)
        self.assertEqual(a3_profile["history"]["negative_history_count"], 0)
        self.assertTrue(a3_profile["history"]["no_negative_history_cases"])
        a3_codes = {warning["code"] for warning in a3_warnings}
        self.assertIn("a3_no_negative_history_cases", a3_codes)
        self.assertIn("a3_transcript_context_not_persistent_memory", a3_codes)

        a4_tasks = load_one_stroke_tasks("data/one_stroke/a4_multimodal.jsonl")
        a4_assets = _one_stroke_input_assets(a4_tasks)
        self.assertEqual(len(a4_assets), 60)
        self.assertEqual(
            len({(asset["task_id"], asset["variant"]) for asset in a4_assets}),
            60,
        )
        self.assertTrue(all(len(asset["sha256"]) == 64 for asset in a4_assets))
        self.assertTrue(
            all(Path(asset["path"]).is_absolute() for asset in a4_assets)
        )
        a4_plan = plan_one_stroke_work_items(
            a4_tasks,
            input_modes=("challenge_image",),
        )
        a4_profile, a4_warnings = _one_stroke_dataset_profile(
            a4_tasks,
            work_plan=a4_plan,
            input_modes=("challenge_image",),
            input_assets=a4_assets,
        )
        self.assertEqual(a4_profile["protocol_generation_total"], 30)
        self.assertEqual(
            a4_profile["multimodal"]["selected_input_modes"],
            ["challenge_image"],
        )
        self.assertTrue(
            a4_profile["multimodal"]["visual_gap_not_estimable"]
        )
        self.assertEqual(
            a4_profile["multimodal"][
                "clear_challenge_identical_task_count"
            ],
            10,
        )
        a4_codes = {warning["code"] for warning in a4_warnings}
        self.assertIn("a4_visual_gap_not_estimable", a4_codes)
        self.assertIn("a4_report_path_transcription_and_joint", a4_codes)
        self.assertIn("a4_clear_challenge_identical_assets", a4_codes)
        score_warning = next(
            warning
            for warning in a4_warnings
            if warning["code"] == "a4_report_path_transcription_and_joint"
        )
        self.assertEqual(
            score_warning["required_score_views"],
            ["a4_path_score", "a4_transcription_score", "a4_joint_score"],
        )
        a4_ablation_plan = plan_one_stroke_work_items(
            a4_tasks,
            input_modes=("text", "clear_image", "challenge_image"),
        )
        a4_ablation_profile, _ = _one_stroke_dataset_profile(
            a4_tasks,
            work_plan=a4_ablation_plan,
            input_modes=("text", "clear_image", "challenge_image"),
            input_assets=a4_assets,
        )
        self.assertEqual(a4_ablation_profile["protocol_generation_total"], 90)

    def test_changed_multimodal_asset_refuses_resume_before_agent_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clear_path = root / "clear.png"
            challenge_path = root / "challenge.png"
            clear_path.write_bytes(b"clear-image-v1")
            challenge_path.write_bytes(b"challenge-image-v1")
            tasks_path = root / "multimodal.jsonl"
            record = {
                "id": "a4-asset-task",
                "source_task_id": "a1-source-task",
                "capability": "multimodal",
                "difficulty": "easy",
                "vertices": ["A", "B", "C"],
                "edges": [["A", "B"], ["B", "C"]],
                "start": "A",
                "end": "C",
                "solution_path": ["A", "B", "C"],
                "image_variants": {
                    "clear": clear_path.name,
                    "challenge": challenge_path.name,
                },
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
            self.assertEqual(len(manifest["input_assets"]), 2)
            self.assertEqual(
                len(manifest["fingerprint_inputs"]["input_asset_hashes"]),
                2,
            )
            before = {
                name: (run_dir / name).read_bytes()
                for name in (
                    "manifest.json",
                    "run_state.json",
                    "predictions.jsonl",
                )
            }
            challenge_path.write_bytes(b"challenge-image-v2")

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

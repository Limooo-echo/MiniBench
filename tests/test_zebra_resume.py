import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.core.checkpoint import RunLock, RunLockedError
from minibench.factory.experiments import (
    _zebra_sidecar_lock_name,
    run_family_experiment,
)


def zebra_record(
    task_id,
    *,
    capability="direct",
    clue_turns=None,
):
    return {
        "id": task_id,
        "size": "2*2",
        "puzzle": "There are two houses. Alice is left of Bob. Alice drinks tea.",
        "solution": {
            "header": ["House", "Name", "Drink"],
            "rows": [["1", "Alice", "tea"], ["2", "Bob", "milk"]],
        },
        "capability": capability,
        "rule_context": None,
        "clue_turns": list(clue_turns or []),
        "tags": ["source:resume-test"],
    }


def correct_output():
    return json.dumps(
        {
            "reasoning": "resume-test",
            "solution": {
                "House 1": {"Name": "Alice", "Drink": "tea"},
                "House 2": {"Name": "Bob", "Drink": "milk"},
            },
        }
    )


def write_tasks(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def experiment_config(root, tasks_path, *, evaluation=None, run_name="zebra-resume"):
    return {
        "task": {
            "family": "zebra",
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
        "evaluation": dict(evaluation or {}),
        "run": {
            "output_dir": str(root / "runs"),
            "run_name": run_name,
            "on_existing": "resume",
        },
    }


class DirectAgent:
    def __init__(self, *, fail_after=None, interrupt=False):
        self.fail_after = fail_after
        self.interrupt = interrupt
        self.task_ids = []

    def generate(self, prompt, task):
        self.task_ids.append(task.id)
        if self.interrupt:
            raise KeyboardInterrupt()
        if self.fail_after is not None and len(self.task_ids) > self.fail_after:
            raise RuntimeError("injected failure")
        return correct_output()


class HistoryAgent:
    def __init__(self, *, fail_on_call=None):
        self.fail_on_call = fail_on_call
        self.phases = []

    def generate_messages_for_phase(
        self,
        messages,
        task,
        *,
        phase,
        temperature=None,
        max_tokens=None,
        json_mode=None,
    ):
        self.phases.append(phase)
        if self.fail_on_call is not None and len(self.phases) == self.fail_on_call:
            raise RuntimeError("history failure")
        if phase == "final":
            return correct_output()
        return '{"acknowledged":true}'


class ZebraResumeTests(unittest.TestCase):
    def test_precreated_empty_directory_with_legacy_lock_is_safely_initialized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(tasks_path, [zebra_record("empty-dir-task")])
            config = experiment_config(root, tasks_path, run_name="precreated")
            run_dir = root / "runs" / "precreated"
            run_dir.mkdir(parents=True)
            (run_dir / ".run.lock").write_text("0", encoding="utf-8")
            agent = DirectAgent()

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=agent,
            ):
                completed_dir, summary = run_family_experiment(config)

            self.assertEqual(completed_dir, run_dir)
            self.assertEqual(agent.task_ids, ["empty-dir-task"])
            self.assertEqual(summary["run_status"], "completed")
            self.assertTrue((run_dir / "manifest.json").is_file())
            self.assertTrue((run_dir / "predictions.jsonl").is_file())
            self.assertTrue((run_dir / "run_state.json").is_file())

    def test_manifest_only_initialization_gap_is_resumed_before_agent_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(tasks_path, [zebra_record("manifest-gap-task")])
            config = experiment_config(root, tasks_path, run_name="manifest-gap")

            with patch(
                "minibench.datasets.zebra.evaluation.write_zebra_snapshot",
                side_effect=OSError("initial snapshot failed"),
            ), patch(
                "minibench.factory.experiments.make_agent_from_config"
            ) as make_agent:
                with self.assertRaisesRegex(OSError, "initial snapshot failed"):
                    run_family_experiment(config)
                make_agent.assert_not_called()

            run_dir = root / "runs" / "manifest-gap"
            self.assertTrue((run_dir / "manifest.json").is_file())
            self.assertFalse((run_dir / "predictions.jsonl").exists())
            self.assertFalse((run_dir / "run_state.json").exists())
            resumed_agent = DirectAgent()

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=resumed_agent,
            ):
                _, summary = run_family_experiment(config)

            self.assertEqual(resumed_agent.task_ids, ["manifest-gap-task"])
            self.assertEqual(summary["run_status"], "completed")
            state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["resume_count"], 1)

    def test_missing_predictions_with_interrupted_state_is_corruption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(tasks_path, [zebra_record("missing-predictions-task")])
            config = experiment_config(
                root,
                tasks_path,
                run_name="missing-predictions",
            )

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=DirectAgent(interrupt=True),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_family_experiment(config)

            run_dir = root / "runs" / "missing-predictions"
            (run_dir / "predictions.jsonl").unlink()
            state_before = (run_dir / "run_state.json").read_bytes()

            with patch(
                "minibench.factory.experiments.make_agent_from_config"
            ) as make_agent:
                with self.assertRaisesRegex(ValueError, "predictions.jsonl is missing"):
                    run_family_experiment(config)
                make_agent.assert_not_called()

            self.assertEqual(
                (run_dir / "run_state.json").read_bytes(),
                state_before,
            )
            self.assertFalse((run_dir / "predictions.jsonl").exists())

    def test_missing_state_with_authoritative_predictions_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(tasks_path, [zebra_record("state-gap-task")])
            config = experiment_config(root, tasks_path, run_name="state-gap")

            with patch(
                "minibench.factory.experiments._write_zebra_state",
                side_effect=OSError("initial state failed"),
            ), patch(
                "minibench.factory.experiments.make_agent_from_config"
            ) as make_agent:
                with self.assertRaisesRegex(OSError, "initial state failed"):
                    run_family_experiment(config)
                make_agent.assert_not_called()

            run_dir = root / "runs" / "state-gap"
            self.assertTrue((run_dir / "predictions.jsonl").is_file())
            self.assertFalse((run_dir / "run_state.json").exists())
            resumed_agent = DirectAgent()

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=resumed_agent,
            ):
                _, summary = run_family_experiment(config)

            self.assertEqual(resumed_agent.task_ids, ["state-gap-task"])
            self.assertEqual(summary["run_status"], "completed")
            state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["resume_count"], 1)
            self.assertEqual(state["remaining_total"], 0)

    def test_sidecar_lock_is_held_before_run_directory_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(tasks_path, [zebra_record("locked-task")])
            run_name = "locked-run"
            config = experiment_config(root, tasks_path, run_name=run_name)
            output_dir = root / "runs"
            output_dir.mkdir()

            with RunLock(
                output_dir,
                lock_name=_zebra_sidecar_lock_name(run_name),
            ), patch(
                "minibench.factory.experiments.make_agent_from_config"
            ) as make_agent:
                with self.assertRaises(RunLockedError):
                    run_family_experiment(config)
                make_agent.assert_not_called()

            self.assertFalse((output_dir / run_name).exists())

    def test_uninitialized_directory_with_unknown_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(tasks_path, [zebra_record("unknown-file-task")])
            config = experiment_config(root, tasks_path, run_name="unknown-file")
            run_dir = root / "runs" / "unknown-file"
            run_dir.mkdir(parents=True)
            (run_dir / "mystery.txt").write_text("unknown", encoding="utf-8")

            with patch(
                "minibench.factory.experiments.make_agent_from_config"
            ) as make_agent:
                with self.assertRaisesRegex(ValueError, "unexpected file"):
                    run_family_experiment(config)
                make_agent.assert_not_called()

    def test_runner_rejects_unsafe_run_name_even_without_config_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(tasks_path, [zebra_record("safe-task")])
            config = experiment_config(root, tasks_path, run_name="../escape")

            with patch(
                "minibench.factory.experiments.make_agent_from_config"
            ) as make_agent:
                with self.assertRaisesRegex(ValueError, "single directory name"):
                    run_family_experiment(config)
                make_agent.assert_not_called()

            self.assertFalse((root / "escape").exists())

    def test_failed_checkpoint_does_not_commit_non_durable_result(self):
        from minibench.datasets.zebra.evaluation import write_zebra_snapshot

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(tasks_path, [zebra_record("retry-task")])
            config = experiment_config(root, tasks_path, run_name="atomic-failure")
            snapshot_calls = 0

            def fail_first_result_snapshot(*args, **kwargs):
                nonlocal snapshot_calls
                snapshot_calls += 1
                if snapshot_calls == 2:
                    raise OSError("injected checkpoint failure")
                return write_zebra_snapshot(*args, **kwargs)

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=DirectAgent(),
            ), patch(
                "minibench.datasets.zebra.evaluation.write_zebra_snapshot",
                side_effect=fail_first_result_snapshot,
            ):
                with self.assertRaisesRegex(RuntimeError, "0/1 results"):
                    run_family_experiment(config)

            run_dir = root / "runs" / "atomic-failure"
            self.assertEqual(
                (run_dir / "predictions.jsonl").read_text(encoding="utf-8"),
                "",
            )
            state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["completed_total"], 0)

            resumed_agent = DirectAgent()
            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=resumed_agent,
            ):
                run_family_experiment(config)

            self.assertEqual(resumed_agent.task_ids, ["retry-task"])

    def test_predictions_remain_authoritative_if_derived_summary_write_fails(self):
        from minibench.datasets.zebra.evaluation import atomic_write_json

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(tasks_path, [zebra_record("durable-task")])
            config = experiment_config(root, tasks_path, run_name="derived-failure")
            derived_writes = 0

            def fail_second_derived_write(*args, **kwargs):
                nonlocal derived_writes
                derived_writes += 1
                if derived_writes == 2:
                    raise OSError("injected derived summary failure")
                return atomic_write_json(*args, **kwargs)

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=DirectAgent(),
            ), patch(
                "minibench.datasets.zebra.evaluation.atomic_write_json",
                side_effect=fail_second_derived_write,
            ):
                with self.assertRaisesRegex(RuntimeError, "1/1 results"):
                    run_family_experiment(config)

            run_dir = root / "runs" / "derived-failure"
            state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["completed_total"], 1)
            self.assertIsNone(state["current_work_key"])
            self.assertEqual(
                json.loads(
                    (run_dir / "predictions.jsonl").read_text(encoding="utf-8")
                )["task_id"],
                "durable-task",
            )

            with patch(
                "minibench.factory.experiments.make_agent_from_config"
            ) as make_agent:
                _, summary = run_family_experiment(config)
                make_agent.assert_not_called()
            self.assertEqual(summary["run_status"], "completed")

    def test_resume_skips_durable_items_sorts_results_and_rejects_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(
                tasks_path,
                [zebra_record(f"task-{index}") for index in range(3)],
            )
            config = experiment_config(root, tasks_path)
            first_agent = DirectAgent(fail_after=2)

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=first_agent,
            ):
                with self.assertRaisesRegex(RuntimeError, "2/3 results"):
                    run_family_experiment(config)

            run_dir = root / "runs" / "zebra-resume"
            manifest_before = (run_dir / "manifest.json").read_bytes()
            state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["completed_total"], 2)
            self.assertEqual(state["remaining_total"], 1)
            self.assertEqual(state["resume_count"], 0)
            self.assertEqual(
                state["current_work_key"],
                {"task_id": "task-2", "mode": "single"},
            )

            prediction_lines = (
                run_dir / "predictions.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            (run_dir / "predictions.jsonl").write_text(
                "\n".join(reversed(prediction_lines)) + "\n",
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text("derived file is not authoritative")
            second_agent = DirectAgent()

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=second_agent,
            ):
                resumed_dir, summary = run_family_experiment(config)

            self.assertEqual(resumed_dir, run_dir)
            self.assertEqual(second_agent.task_ids, ["task-2"])
            self.assertEqual(summary["run_status"], "completed")
            saved_results = [
                json.loads(line)
                for line in (run_dir / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [result["task_id"] for result in saved_results],
                ["task-0", "task-1", "task-2"],
            )
            final_state = json.loads(
                (run_dir / "run_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(final_state["status"], "completed")
            self.assertEqual(final_state["completed_total"], 3)
            self.assertEqual(final_state["remaining_total"], 0)
            self.assertEqual(final_state["resume_count"], 1)
            self.assertEqual(final_state["created_at"], state["created_at"])
            self.assertEqual((run_dir / "manifest.json").read_bytes(), manifest_before)

            with patch(
                "minibench.factory.experiments.make_agent_from_config"
            ) as make_agent:
                with self.assertRaisesRegex(ValueError, "already completed"):
                    run_family_experiment(config)
                make_agent.assert_not_called()

    def test_history_resume_retries_only_the_incomplete_task_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "history.jsonl"
            write_tasks(
                tasks_path,
                [
                    zebra_record(
                        "history-task",
                        capability="history_memory",
                        clue_turns=["Only clue."],
                    )
                ],
            )
            config = experiment_config(
                root,
                tasks_path,
                evaluation={
                    "memory_modes": [
                        "incremental_state",
                        "deferred_reasoning",
                    ],
                    "state_max_tokens": 16,
                    "ack_max_tokens": 4,
                    "final_max_tokens": 64,
                },
                run_name="history-resume",
            )
            first_agent = HistoryAgent(fail_on_call=3)

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=first_agent,
            ):
                with self.assertRaisesRegex(RuntimeError, "1/2 results"):
                    run_family_experiment(config)

            run_dir = root / "runs" / "history-resume"
            state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
            self.assertEqual(
                state["current_work_key"],
                {"task_id": "history-task", "mode": "deferred_reasoning"},
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
                run_family_experiment(config)

            self.assertEqual(resumed_agent.phases, ["intermediate", "final"])
            saved = [
                json.loads(line)
                for line in (run_dir / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [result["memory_mode"] for result in saved],
                ["incremental_state", "deferred_reasoning"],
            )

    def test_fingerprint_mismatch_refuses_without_modifying_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(tasks_path, [zebra_record("task-0"), zebra_record("task-1")])
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
                for name in ("manifest.json", "run_state.json", "predictions.jsonl")
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
                {
                    name: (run_dir / name).read_bytes()
                    for name in before
                },
                before,
            )

    def test_keyboard_interrupt_records_interrupted_current_work_item(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            write_tasks(tasks_path, [zebra_record("interrupt-task")])
            config = experiment_config(root, tasks_path, run_name="interrupt")

            with patch(
                "minibench.factory.experiments.make_agent_from_config",
                return_value=DirectAgent(interrupt=True),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_family_experiment(config)

            run_dir = root / "runs" / "interrupt"
            state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["completed_total"], 0)
            self.assertEqual(
                state["current_work_key"],
                {"task_id": "interrupt-task", "mode": "single"},
            )
            self.assertEqual(state["error"]["type"], "KeyboardInterrupt")
            self.assertEqual(
                (run_dir / "predictions.jsonl").read_text(encoding="utf-8"),
                "",
            )


if __name__ == "__main__":
    unittest.main()

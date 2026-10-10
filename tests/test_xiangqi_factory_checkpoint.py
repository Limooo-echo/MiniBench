"""Offline durability and common-roster checks for the Xiangqi factory path."""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from minibench.core.checkpoint import sha256_file
from minibench.factory.experiments import (
    TaskFamilySpec, _evaluate, _run_checkpointed_xiangqi_experiment,
    _select_frozen_xiangqi_tasks,
)


@dataclass
class SavedResult:
    task_id: str
    success: bool | None = True
    input_mode: str = "text"
    status: str = "ok"


class XiangqiFactoryCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "tasks.jsonl"
        self.tasks = [{"id": "task-1"}, {"id": "task-2"}]
        self.data.write_text("".join(json.dumps(t) + "\n" for t in self.tasks))
        self.run_config = {"output_dir": str(self.root), "run_name": "trial"}
        self.config = {"task": {"family": "xiangqi-multimodal"}, "agent": {"name": "static"},
                       "evaluation": {"verify_with_pikafish": False}, "run": self.run_config}
        self.run_dir = self.root / "trial"
        self.writer = Mock(side_effect=self.write_summary)
        self.spec = TaskFamilySpec(self.data, lambda path: self.tasks, Mock(),
                                   lambda results: {"total": len(results)}, self.writer)

    def write_summary(self, results, output_dir, run_name, *, write_predictions=True):
        self.assertFalse(write_predictions)
        directory = Path(output_dir) / run_name
        (directory / "results.json").write_text(json.dumps({"total": len(results)}))
        (directory / "summary.txt").write_text("finished\n")
        return directory

    def write_selection(self, directory, **kwargs):
        path = directory / "selected_tasks.jsonl"
        path.write_text(self.data.read_text())
        return path

    def write_metadata(self, directory, **kwargs):
        (directory / "resolved_config.yaml").write_text("fixture: true\n")
        (directory / "run_metadata.json").write_text("{}\n")

    def create_agent(self, *args, **kwargs):
        for name in ("selected_tasks.jsonl", "resolved_config.yaml", "run_metadata.json", "run_state.json"):
            self.assertTrue((self.run_dir / name).is_file(), name)
        return Mock()

    def run_experiment(self, evaluator):
        with patch("minibench.factory.experiments._write_xiangqi_selected_tasks", self.write_selection), \
             patch("minibench.factory.experiments._write_xiangqi_run_metadata", self.write_metadata), \
             patch("minibench.factory.experiments.make_agent_from_config", self.create_agent), \
             patch("minibench.factory.experiments._evaluate", evaluator):
            return _run_checkpointed_xiangqi_experiment(
                self.spec, self.tasks, self.config, self.data, "xiangqi-multimodal",
                self.config["evaluation"], self.run_config)

    def test_interruption_preserves_completed_mode_and_frozen_inputs(self):
        def evaluator(*args, on_result, **kwargs):
            on_result(SavedResult("task-1", input_mode="chinese-piece-image"))
            saved = (self.run_dir / "predictions.jsonl").read_text().splitlines()
            self.assertEqual(len(saved), 1)
            self.assertEqual(json.loads(saved[0])["task_id"], "task-1")
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.run_experiment(evaluator)
        state = json.loads((self.run_dir / "run_state.json").read_text())
        self.assertEqual((state["status"], state["completed_results"]), ("interrupted", 1))
        before = (self.run_dir / "predictions.jsonl").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_experiment(evaluator)
        self.assertEqual((self.run_dir / "predictions.jsonl").read_bytes(), before)
        self.writer.assert_not_called()

    def test_all_modes_are_durable_and_summary_writer_cannot_truncate_them(self):
        results = [SavedResult("task-1", input_mode="chinese-piece-image"),
                   SavedResult("task-1", input_mode="latin-piece-image")]
        def evaluator(*args, on_result, **kwargs):
            for result in results:
                on_result(result)
            return results
        directory, summary = self.run_experiment(evaluator)
        saved = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual([r["input_mode"] for r in saved], [r.input_mode for r in results])
        self.assertEqual(summary["total"], 2)
        state = json.loads((directory / "run_state.json").read_text())
        self.assertEqual((state["status"], state["completed_results"]), ("completed", 2))
        self.writer.assert_called_once()

    def test_runtime_exception_marks_error_without_erasing_previous_result(self):
        def evaluator(*args, on_result, **kwargs):
            on_result(SavedResult("task-1"))
            raise RuntimeError("fixture engine disconnected")
        with self.assertRaisesRegex(RuntimeError, "engine disconnected"):
            self.run_experiment(evaluator)
        state = json.loads((self.run_dir / "run_state.json").read_text())
        self.assertEqual(state["status"], "error")
        self.assertEqual(state["completed_results"], 1)
        self.assertEqual(len((self.run_dir / "predictions.jsonl").read_text().splitlines()), 1)

    def test_callback_is_forwarded_to_each_xiangqi_evaluator(self):
        callback = Mock()
        for family in ("xiangqi-mate-in-one", "xiangqi-history", "xiangqi-rule-variants", "xiangqi-multimodal"):
            with self.subTest(family=family):
                self.spec.evaluate_tasks.reset_mock()
                _evaluate(self.spec, [], Mock(), family, {}, on_result=callback)
                self.assertIs(self.spec.evaluate_tasks.call_args.kwargs["on_result"], callback)

    def test_frozen_roster_preserves_order_and_rejects_resampling_or_changed_gold(self):
        roster = self.root / "selected.jsonl"
        roster.write_text(json.dumps(self.tasks[1]) + "\n")
        config = {"selection": {"path": str(roster), "sha256": sha256_file(roster)}}
        def load_records(path, **kwargs):
            return [json.loads(line) for line in Path(path).read_text().splitlines()]
        with patch("minibench.datasets.xiangqi.schema.load_records", load_records):
            chosen = _select_frozen_xiangqi_tasks(self.tasks, config, self.data, "xiangqi-history")
            self.assertEqual(chosen, [self.tasks[1]])
            for extra in ({"sampling": {"enabled": True}}, {"limit": 1}, {"task_ids": ["task-2"]}):
                with self.assertRaisesRegex(ValueError, "cannot be combined"):
                    _select_frozen_xiangqi_tasks(self.tasks, {**config, **extra}, self.data, "xiangqi-history")
            roster.write_text(json.dumps({"id": "task-2", "gold": "modified"}) + "\n")
            config["selection"]["sha256"] = sha256_file(roster)
            with self.assertRaisesRegex(ValueError, "differs from the source dataset"):
                _select_frozen_xiangqi_tasks(self.tasks, config, self.data, "xiangqi-history")


if __name__ == "__main__":
    unittest.main()
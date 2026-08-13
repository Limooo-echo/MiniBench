import json
from pathlib import Path
import tempfile
import unittest

import yaml

from minibench.evaluate import run_config
from minibench.factory.config import validate_experiment_config
from minibench.factory.experiments import get_task_family_spec


class EvaluateConfigTests(unittest.TestCase):
    def test_run_config_writes_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "runs"
            tasks_path = Path(tmpdir) / "zebra.jsonl"
            tasks_path.write_text(
                json.dumps(
                    {
                        "id": "zebra-unit",
                        "size": "2*2",
                        "puzzle": "Two houses and two attributes.",
                        "solution": {
                            "header": ["House", "Name", "Drink"],
                            "rows": [["1", "Alice", "tea"], ["2", "Bob", "milk"]],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            predictions_path = Path(tmpdir) / "predictions.jsonl"
            predictions_path.write_text(
                json.dumps(
                    {
                        "task_id": "zebra-unit",
                        "raw_output": json.dumps(
                            {
                                "reasoning": "unit",
                                "solution": {
                                    "House 1": {"Name": "Alice", "Drink": "tea"},
                                    "House 2": {"Name": "Bob", "Drink": "milk"},
                                },
                            }
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = Path(tmpdir) / "experiment.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "task": {
                            "family": "zebra",
                            "path": str(tasks_path),
                            "limit": 1,
                            "task_ids": [],
                        },
                        "agent": {
                            "name": "openai-compatible",
                            "predictions": str(predictions_path),
                        },
                        "provider": {"name": "generic"},
                        "run": {
                            "output_dir": str(output_dir),
                            "run_name": "unit-run",
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = run_config(config_path)

            run_dir = Path(result["run_dir"])
            self.assertEqual(result["total"], 1)
            self.assertEqual(result["success"], 1)
            self.assertTrue((run_dir / "predictions.jsonl").exists())
            self.assertTrue((run_dir / "results.json").exists())
            saved = json.loads(
                (run_dir / "results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["puzzle_accuracy"], 1.0)
            self.assertIn("metrics", saved)
            self.assertEqual(saved["metrics"]["total"]["llm_calls"], 0)
            prediction = json.loads(
                (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertIn("metrics", prediction)
            self.assertIn("task_elapsed_seconds", prediction["metrics"])

    def test_run_config_passes_one_stroke_prompt_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "runs"
            predictions_path = Path(tmpdir) / "predictions.jsonl"
            predictions_path.write_text(
                json.dumps(
                    {
                        "task_id": "os-path-001",
                        "raw_output": '{"path":["A","B","C","D"]}',
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = Path(tmpdir) / "experiment.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "task": {
                            "family": "one_stroke",
                            "path": "data/one_stroke/tasks.jsonl",
                            "limit": 1,
                            "task_ids": [],
                        },
                        "agent": {
                            "name": "openai-compatible",
                            "predictions": str(predictions_path),
                        },
                        "provider": {"name": "generic"},
                        "evaluation": {"prompt_variant": "euler_theorem"},
                        "run": {
                            "output_dir": str(output_dir),
                            "run_name": "one-stroke-unit-run",
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = run_config(config_path)

            run_dir = Path(result["run_dir"])
            saved_prediction = json.loads(
                (run_dir / "predictions.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(result["success"], 1)
            self.assertEqual(saved_prediction["prompt_variant"], "euler_theorem")

    def test_run_config_expands_one_stroke_rule_modes(self):
        from minibench.datasets.one_stroke.dataset import load_one_stroke_tasks
        from minibench.datasets.one_stroke.rules import (
            find_constrained_one_stroke_path,
            rules_for_mode,
        )

        task = load_one_stroke_tasks(
            "data/one_stroke/a2_rule_condition.jsonl"
        )[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "runs"
            predictions_path = Path(tmpdir) / "predictions.jsonl"
            raw_outputs = []
            for mode in ("full", "conflicting_rule"):
                constraints = rules_for_mode(
                    task.rule_constraints,
                    task.key_rule_id,
                    task.conflicting_rule,
                    mode,
                )
                oracle = find_constrained_one_stroke_path(
                    task.vertices, task.edges, constraints=constraints
                )
                raw_outputs.append(
                    json.dumps({"path": oracle[0], "edge_path": oracle[1]})
                )
            predictions_path.write_text(
                json.dumps({"task_id": task.id, "raw_outputs": raw_outputs}) + "\n",
                encoding="utf-8",
            )
            config_path = Path(tmpdir) / "a2.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "task": {
                            "family": "one_stroke",
                            "path": "data/one_stroke/a2_rule_condition.jsonl",
                            "limit": 1,
                            "task_ids": [],
                        },
                        "agent": {
                            "name": "openai-compatible",
                            "predictions": str(predictions_path),
                        },
                        "provider": {"name": "generic"},
                        "evaluation": {
                            "rule_modes": ["full", "conflicting_rule"]
                        },
                        "run": {
                            "output_dir": str(output_dir),
                            "run_name": "a2-unit-run",
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = run_config(config_path)

            self.assertEqual(result["total"], 2)
            self.assertEqual(result["success"], 2)
            self.assertEqual(
                set(result["by_rule_mode"]), {"full", "conflicting_rule"}
            )

    def test_invalid_task_family_reports_clear_error(self):
        with self.assertRaisesRegex(ValueError, "task.family must be one of"):
            validate_experiment_config(
                {
                    "task": {"family": "not-a-family"},
                    "agent": {"name": "openai-compatible"},
                    "provider": {"name": "generic"},
                    "run": {"output_dir": "runs"},
                }
            )

    def test_missing_section_reports_clear_error(self):
        with self.assertRaisesRegex(ValueError, "missing required section: provider"):
            validate_experiment_config(
                {
                    "task": {"family": "one_stroke"},
                    "agent": {"name": "openai-compatible"},
                    "run": {"output_dir": "runs"},
                }
            )

    def test_task_family_specs_are_available(self):
        self.assertEqual(
            get_task_family_spec("zebra").default_path,
            Path("data/zebra/tasks.jsonl"),
        )


if __name__ == "__main__":
    unittest.main()

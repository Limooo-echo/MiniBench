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
            config_path = Path(tmpdir) / "experiment.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "task": {
                            "family": "multiple_choice",
                            "path": "data/multiple_choice/tasks.jsonl",
                            "limit": 2,
                            "task_ids": [],
                        },
                        "agent": {"name": "oracle"},
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
            self.assertEqual(result["total"], 2)
            self.assertEqual(result["correct"], 2)
            self.assertTrue((run_dir / "predictions.jsonl").exists())
            self.assertTrue((run_dir / "results.json").exists())
            saved = json.loads(
                (run_dir / "results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["accuracy"], 1.0)

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

    def test_invalid_task_family_reports_clear_error(self):
        with self.assertRaisesRegex(ValueError, "task.family must be one of"):
            validate_experiment_config(
                {
                    "task": {"family": "not-a-family"},
                    "agent": {"name": "oracle"},
                    "provider": {"name": "generic"},
                    "run": {"output_dir": "runs"},
                }
            )

    def test_missing_section_reports_clear_error(self):
        with self.assertRaisesRegex(ValueError, "missing required section: provider"):
            validate_experiment_config(
                {
                    "task": {"family": "multiple_choice"},
                    "agent": {"name": "oracle"},
                    "run": {"output_dir": "runs"},
                }
            )

    def test_task_family_specs_are_available(self):
        self.assertEqual(
            get_task_family_spec("multiple_choice").default_path,
            Path("data/multiple_choice/tasks.jsonl"),
        )


if __name__ == "__main__":
    unittest.main()

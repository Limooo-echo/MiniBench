import json
from pathlib import Path
import tempfile
import unittest

import yaml

from minibench.evaluate import run_config
from minibench.factory.config import load_experiment_config, validate_experiment_config
from minibench.factory.experiments import get_task_family_spec


class EvaluateConfigTests(unittest.TestCase):
    def test_visual_configs_use_qwen_3_8_max(self):
        visual_configs = (
            "mahjong_multimodal.yaml",
            "mahjong_multimodal_ablation.yaml",
            "one_stroke_multimodal.yaml",
            "one_stroke_multimodal_ablation.yaml",
            "xiangqi_multimodal.yaml",
        )

        for filename in visual_configs:
            with self.subTest(filename=filename):
                config = load_experiment_config(Path("config/experiments") / filename)
                self.assertEqual(config["provider"]["name"], "qwen")
                self.assertEqual(config["provider"]["model"], "qwen3.8-max")
                self.assertEqual(
                    config["provider"]["extra_body"],
                    {"enable_thinking": False},
                )

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
            saved = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["puzzle_accuracy"], 1.0)
            self.assertIn("metrics", saved)
            self.assertEqual(saved["metrics"]["total"]["llm_calls"], 0)
            prediction = json.loads(
                (run_dir / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
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

    def test_one_stroke_rejects_removed_rule_and_image_modes(self):
        evaluations = (
            {"rule_mode": "full"},
            {"rule_modes": ["full"]},
            {"input_modes": ["clear_image"]},
            {"input_modes": ["challenge_image"]},
        )
        for evaluation in evaluations:
            with self.subTest(evaluation=evaluation):
                with self.assertRaises(ValueError):
                    validate_experiment_config(
                        {
                            "task": {"family": "one_stroke"},
                            "agent": {"name": "passthrough"},
                            "provider": {"name": "generic"},
                            "evaluation": evaluation,
                            "run": {"output_dir": "runs"},
                        }
                    )

    def test_run_config_expands_one_stroke_multimodal_input_modes(self):
        from minibench.datasets.one_stroke.dataset import load_one_stroke_tasks

        task = load_one_stroke_tasks("data/one_stroke/multimodal.jsonl")[0]
        raw_output = json.dumps(
            {
                "recognized_vertices": list(task.vertices),
                "recognized_edges": [list(edge) for edge in task.edges],
                "solvable": task.solution_exists,
                "path": list(task.solution_path),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions = Path(tmpdir) / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"task_id": task.id, "raw_outputs": [raw_output] * 2})
                + "\n",
                encoding="utf-8",
            )
            config = Path(tmpdir) / "multimodal.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "task": {
                            "family": "one_stroke",
                            "path": "data/one_stroke/multimodal.jsonl",
                            "limit": 1,
                            "task_ids": [],
                        },
                        "agent": {
                            "name": "openai-compatible",
                            "predictions": str(predictions),
                        },
                        "provider": {"name": "generic"},
                        "evaluation": {
                            "input_modes": [
                                "text",
                                "image",
                            ]
                        },
                        "run": {
                            "output_dir": str(Path(tmpdir) / "runs"),
                            "run_name": "multimodal-unit-run",
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = run_config(config)

        self.assertEqual(result["total"], 2)
        self.assertEqual(result["success"], 2)
        self.assertEqual(
            set(result["by_input_mode"]),
            {"text", "image"},
        )

    def test_cli_accepts_multimodal_all_modes(self):
        from minibench.cli import build_parser

        one_stroke = build_parser().parse_args(
            ["evaluate-one-stroke", "--input-mode", "all"]
        )
        mahjong = build_parser().parse_args(["evaluate-mahjong", "--input-mode", "all"])
        self.assertEqual(one_stroke.input_mode, "all")
        self.assertEqual(mahjong.input_mode, "all")

    def test_cli_one_stroke_defaults_to_canonical_direct(self):
        from minibench.cli import build_parser

        args = build_parser().parse_args(["evaluate-one-stroke"])

        self.assertEqual(
            args.one_stroke_tasks,
            Path("data/one_stroke/direct.jsonl"),
        )
        self.assertEqual(args.max_tokens, 1024)

    def test_one_stroke_openai_compatible_rejects_reasoning_only_fields(self):
        field_values = {
            "samples": 3,
            "reasoning_temperature": 0.7,
            "final_temperature": 0.0,
            "max_reasoning_tokens": 512,
        }
        for field, value in field_values.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"agent\.{field}.*would be ignored",
                ):
                    validate_experiment_config(
                        {
                            "task": {"family": "one_stroke"},
                            "agent": {
                                "name": "openai-compatible",
                                field: value,
                            },
                            "provider": {"name": "generic"},
                            "run": {"output_dir": "runs"},
                        }
                    )

    def test_samples_are_supported_only_by_sc_and_best_of_n(self):
        for agent_name in ("self-consistency", "best-of-n"):
            with self.subTest(agent_name=agent_name):
                config = validate_experiment_config(
                    {
                        "task": {"family": "zebra"},
                        "agent": {
                            "name": agent_name,
                            "samples": 3,
                            "reasoning_temperature": 0.7,
                            "final_temperature": 0.0,
                            "max_reasoning_tokens": 512,
                        },
                        "provider": {"name": "generic"},
                        "run": {"output_dir": "runs"},
                    }
                )
                self.assertEqual(config["agent"]["samples"], 3)

        for agent_name in ("openai-compatible", "cot"):
            with self.subTest(agent_name=agent_name):
                with self.assertRaisesRegex(
                    ValueError,
                    r"agent\.samples.*would be ignored",
                ):
                    validate_experiment_config(
                        {
                            "task": {"family": "zebra"},
                            "agent": {
                                "name": agent_name,
                                "samples": 3,
                            },
                            "provider": {"name": "generic"},
                            "run": {"output_dir": "runs"},
                        }
                    )

    def test_one_stroke_configs_are_canonical_and_retry_safe(self):
        unsupported_fields = {
            "samples",
            "reasoning_temperature",
            "final_temperature",
            "max_reasoning_tokens",
        }
        paths = sorted(Path("config/experiments").glob("one_stroke*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path):
                config = load_experiment_config(path)
                self.assertTrue(unsupported_fields.isdisjoint(config["agent"]))
                self.assertEqual(config["provider"]["max_retries"], 3)
                self.assertEqual(
                    config["provider"]["retry_initial_backoff_seconds"],
                    1.0,
                )
                self.assertEqual(
                    config["provider"]["retry_max_backoff_seconds"],
                    30.0,
                )

        alias = load_experiment_config("config/experiments/one_stroke.yaml")
        canonical = load_experiment_config("config/experiments/one_stroke_direct.yaml")
        self.assertEqual(alias, canonical)
        self.assertEqual(canonical["provider"]["max_tokens"], 1024)
        self.assertIsNone(canonical["run"]["run_name"])
        self.assertEqual(canonical["run"]["on_existing"], "error")

        theorem = load_experiment_config(
            "config/experiments/one_stroke_euler_theorem.yaml"
        )
        self.assertEqual(theorem["task"], canonical["task"])
        self.assertEqual(theorem["agent"], canonical["agent"])
        self.assertEqual(theorem["provider"], canonical["provider"])
        self.assertEqual(theorem["evaluation"]["prompt_variant"], "euler_theorem")

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

    def test_run_on_existing_defaults_to_error(self):
        config = validate_experiment_config(
            {
                "task": {"family": "zebra"},
                "agent": {"name": "openai-compatible"},
                "provider": {"name": "generic"},
                "run": {"output_dir": "runs"},
            }
        )

        self.assertEqual(config["run"]["on_existing"], "error")

    def test_resume_requires_safe_explicit_run_name(self):
        base = {
            "task": {"family": "zebra"},
            "agent": {"name": "openai-compatible"},
            "provider": {"name": "generic"},
        }
        with self.assertRaisesRegex(ValueError, "run.run_name is required"):
            validate_experiment_config({**base, "run": {"on_existing": "resume"}})
        for run_name in ("../escape", "nested/run", "nested\\run", ".", ".."):
            with self.subTest(run_name=run_name):
                with self.assertRaisesRegex(ValueError, "single directory name"):
                    validate_experiment_config(
                        {
                            **base,
                            "run": {
                                "on_existing": "resume",
                                "run_name": run_name,
                            },
                        }
                    )

    def test_provider_retry_fields_are_validated(self):
        base = {
            "task": {"family": "zebra"},
            "agent": {"name": "openai-compatible"},
            "run": {"output_dir": "runs"},
        }
        valid = validate_experiment_config(
            {
                **base,
                "provider": {
                    "name": "generic",
                    "max_retries": 3,
                    "retry_initial_backoff_seconds": 1.0,
                    "retry_max_backoff_seconds": 30.0,
                },
            }
        )
        self.assertEqual(valid["provider"]["max_retries"], 3)

        invalid_providers = (
            {"name": "generic", "max_retries": True},
            {"name": "generic", "max_retries": -1},
            {"name": "generic", "retry_initial_backoff_seconds": -0.1},
            {"name": "generic", "retry_max_backoff_seconds": float("inf")},
            {"name": "generic", "retry_max_backoff_seconds": 0},
            {"name": "generic", "retry_initial_backoff_seconds": 31},
            {
                "name": "generic",
                "retry_initial_backoff_seconds": 2.0,
                "retry_max_backoff_seconds": 1.0,
            },
        )
        for provider in invalid_providers:
            with self.subTest(provider=provider):
                with self.assertRaises(ValueError):
                    validate_experiment_config({**base, "provider": provider})

    def test_zebra_history_config_enables_bounded_retry_and_resume(self):
        config = load_experiment_config("config/experiments/zebra_history.yaml")

        self.assertEqual(config["provider"]["max_retries"], 3)
        self.assertEqual(config["provider"]["retry_initial_backoff_seconds"], 1.0)
        self.assertEqual(config["provider"]["retry_max_backoff_seconds"], 30.0)
        self.assertEqual(config["run"]["run_name"], "zebra-history-deferred_0823")
        self.assertEqual(config["run"]["on_existing"], "resume")

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

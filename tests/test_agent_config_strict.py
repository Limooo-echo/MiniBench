from pathlib import Path
import unittest

from minibench.factory.config import (
    load_experiment_config,
    validate_experiment_config,
)


def experiment(agent):
    return {
        "task": {"family": "zebra"},
        "agent": dict(agent),
        "provider": {"name": "generic"},
        "run": {"output_dir": "runs"},
    }


class StrictAgentConfigTests(unittest.TestCase):
    def test_all_checked_in_experiment_yamls_validate(self):
        paths = sorted(Path("config/experiments").glob("*.yaml"))
        self.assertTrue(paths)

        for path in paths:
            with self.subTest(path=path):
                config = load_experiment_config(path)
                self.assertIn("name", config["agent"])

    def test_each_architecture_accepts_its_documented_fields(self):
        cases = (
            {"name": "passthrough", "max_tokens": 128},
            {
                "name": "direct",
                "prompt_version": "v1",
                "trace": "off",
                "max_llm_calls": 2,
                "max_total_tokens": 100,
                "max_format_repairs": 0,
                "final_temperature": 0.1,
            },
            {
                "name": "cot",
                "reasoning_temperature": 0.8,
                "final_temperature": 0.1,
                "max_reasoning_tokens": 256,
            },
            {
                "name": "self-consistency",
                "samples": 5,
                "reasoning_temperature": 0.8,
            },
            {
                "name": "best-of-n",
                "samples": 5,
                "selection_seed": -7,
            },
            {
                "name": "tot",
                "max_depth": 4,
                "branching_factor": 3,
                "beam_width": 2,
                "max_search_nodes": 40,
            },
            {
                "name": "plan-then-solve",
                "reasoning_temperature": 0.8,
                "final_temperature": 0.1,
                "max_reasoning_tokens": 256,
            },
            {
                "name": "critic-refine",
                "reasoning_temperature": 0.8,
                "final_temperature": 0.1,
                "max_reasoning_tokens": 256,
            },
            {
                "name": "least-to-most",
                "max_subproblems": 6,
            },
        )

        for agent in cases:
            with self.subTest(agent=agent["name"]):
                validated = validate_experiment_config(experiment(agent))
                self.assertEqual(validated["agent"]["name"], agent["name"])

    def test_samples_are_exclusive_to_sc_and_best_of_n(self):
        unsupported = (
            "passthrough",
            "direct",
            "cot",
            "tot",
            "plan-then-solve",
            "critic-refine",
            "least-to-most",
        )
        for name in unsupported:
            with self.subTest(agent=name):
                with self.assertRaisesRegex(
                    ValueError, r"agent\.samples.*would be ignored"
                ):
                    validate_experiment_config(experiment({"name": name, "samples": 3}))

        for name in ("self-consistency", "best-of-n"):
            with self.subTest(agent=name):
                validated = validate_experiment_config(
                    experiment({"name": name, "samples": 2})
                )
                self.assertEqual(validated["agent"]["samples"], 2)

    def test_sampled_agents_reject_too_few_samples_and_zero_temperature(self):
        for name in ("self-consistency", "best-of-n"):
            with self.subTest(agent=name, invalid="samples"):
                with self.assertRaisesRegex(ValueError, "at least 2"):
                    validate_experiment_config(experiment({"name": name, "samples": 1}))
            with self.subTest(agent=name, invalid="temperature"):
                with self.assertRaisesRegex(ValueError, "must be non-zero"):
                    validate_experiment_config(
                        experiment(
                            {
                                "name": name,
                                "reasoning_temperature": 0.0,
                            }
                        )
                    )

    def test_v2_only_architectures_reject_v1(self):
        for name in ("best-of-n", "tot", "least-to-most"):
            with self.subTest(agent=name):
                with self.assertRaisesRegex(ValueError, "requires prompt_version=v2"):
                    validate_experiment_config(
                        experiment(
                            {
                                "name": name,
                                "prompt_version": "v1",
                            }
                        )
                    )

    def test_old_alias_keeps_passthrough_validation_semantics(self):
        with self.assertWarnsRegex(FutureWarning, "passthrough"):
            with self.assertRaisesRegex(
                ValueError, r"agent\.samples.*agent.name=passthrough"
            ):
                validate_experiment_config(
                    experiment(
                        {
                            "name": "openai-compatible",
                            "samples": 3,
                        }
                    )
                )

    def test_invalid_runtime_and_search_values_are_rejected(self):
        cases = (
            ("direct", "prompt_version", "v3"),
            ("direct", "trace", "verbose"),
            ("direct", "max_llm_calls", 0),
            ("direct", "max_total_tokens", True),
            ("direct", "max_format_repairs", 2),
            ("best-of-n", "selection_seed", True),
            ("tot", "max_depth", 0),
            ("tot", "branching_factor", 1.5),
            ("tot", "beam_width", False),
            ("tot", "max_search_nodes", 0),
            ("least-to-most", "max_subproblems", 0),
        )
        for name, field, value in cases:
            with self.subTest(agent=name, field=field):
                with self.assertRaises(ValueError):
                    validate_experiment_config(experiment({"name": name, field: value}))

    def test_optional_budget_fields_accept_null(self):
        validated = validate_experiment_config(
            experiment(
                {
                    "name": "tot",
                    "max_llm_calls": None,
                    "max_total_tokens": None,
                    "max_search_nodes": None,
                }
            )
        )

        self.assertIsNone(validated["agent"]["max_llm_calls"])
        self.assertIsNone(validated["agent"]["max_total_tokens"])
        self.assertIsNone(validated["agent"]["max_search_nodes"])

    def test_missing_agent_name_defaults_to_direct(self):
        validated = validate_experiment_config(experiment({}))

        self.assertEqual(validated["agent"]["name"], "direct")


if __name__ == "__main__":
    unittest.main()

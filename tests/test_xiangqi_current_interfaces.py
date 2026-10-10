"""Cleanup guards: old formats/options must fail before any paid model call."""
from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from minibench.cli import build_parser
from minibench.commands.xiangqi import _xiangqi_config_with_overrides
from minibench.datasets.xiangqi.dataset import load_xiangqi_tasks, xiangqi_task_from_dict
from minibench.datasets.xiangqi.runtime import validate_evaluation_config
from minibench.datasets.xiangqi.schema import FAMILY_PATHS, load_records, runtime_dict
from minibench.factory.config import load_experiment_config, validate_experiment_config


class CurrentInterfaceTests(unittest.TestCase):
    def test_old_commands_and_horizon_overrides_are_removed(self):
        for argv in (
            ["evaluate-xiangqi"], ["generate-xiangqi-pikafish"],
            ["generate-xiangqi-capture"], ["generate-ccpd-endgames"],
            ["migrate-xiangqi-v2"],
            ["run-task", "xiangqi-rule-variants", "--search-depth", "3"],
            ["run-task", "xiangqi-multimodal", "--max-plies", "20"],
        ):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    build_parser().parse_args(argv)
                self.assertEqual(error.exception.code, 2)

    def test_legacy_board_input_is_not_silently_loaded(self):
        with self.assertRaises(ValueError):
            xiangqi_task_from_dict({"id": "old", "board": [[0] * 9 for _ in range(10)]})

    def test_duplicate_task_ids_are_rejected_by_typed_loader(self):
        record = load_records(FAMILY_PATHS["xiangqi-mate-in-one"])[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            path.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_xiangqi_tasks(path)

    def test_runtime_projection_preserves_current_goal_and_horizon(self):
        for family, path in FAMILY_PATHS.items():
            record = load_records(path, expected_family=family)[0]
            runtime = runtime_dict(record)
            self.assertEqual(runtime["goal"], record["goal"])
            self.assertEqual(runtime["max_plies"], record["max_plies"])
            self.assertNotIn("max_steps", runtime)
            task = xiangqi_task_from_dict(record)
            self.assertEqual(task.max_steps, record["max_plies"])
            self.assertEqual(task.agent_color, record["agent_color"])

    def test_removed_config_fields_fail_at_config_load_boundary(self):
        config = load_experiment_config("config/experiments/xiangqi_rule_variants.yaml")
        for field in ("search_depth", "opponent_depth", "optimal_depth", "max_steps", "max_plies"):
            with self.subTest(field=field):
                bad = deepcopy(config)
                bad["evaluation"][field] = 3
                with self.assertRaisesRegex(ValueError, "unsupported evaluation fields"):
                    validate_experiment_config(bad)

    def test_only_fixed_c2_oracle_depth_is_accepted(self):
        for value in (2, 4, True, 3.0, "3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_evaluation_config("xiangqi-rule-variants", {"oracle_depth": value})

    def test_cli_cannot_silently_apply_another_familys_option(self):
        parser = build_parser()
        args = parser.parse_args(["run-task", "xiangqi-mate-in-one", "--oracle-depth", "3"])
        with self.assertRaisesRegex(SystemExit, "not supported"):
            _xiangqi_config_with_overrides(args, args.task_name)
        args = parser.parse_args(["run-suite", "--tasks", "xiangqi-mate-in-one,xiangqi-rule-variants", "--oracle-depth", "3"])
        config = _xiangqi_config_with_overrides(args, "xiangqi-mate-in-one")
        self.assertNotIn("oracle_depth", config["evaluation"])
        config = _xiangqi_config_with_overrides(args, "xiangqi-rule-variants")
        self.assertEqual(config["evaluation"]["oracle_depth"], 3)

    def test_formal_and_smoke_configs_remain_loadable_and_frozen(self):
        paths = list(Path("config/experiments").glob("xiangqi_*.yaml"))
        paths += list(Path("config/experiments/xiangqi_smoke").glob("*.yaml"))
        self.assertEqual(len(paths), 8)
        for path in paths:
            with self.subTest(path=path):
                config = load_experiment_config(path)
                self.assertNotIn("sampling", config["task"])
                self.assertIn("selection", config["task"])
                self.assertNotIn("max_plies", config["evaluation"])

    def test_cli_keeps_frozen_config_free_of_inactive_sampling_options(self):
        args = build_parser().parse_args(["run-task", "xiangqi-mate-in-one", "--agent", "cot"])
        config = _xiangqi_config_with_overrides(args, args.task_name)
        self.assertNotIn("sampling", config["task"])
        self.assertIn("selection", config["task"])


if __name__ == "__main__":
    unittest.main()

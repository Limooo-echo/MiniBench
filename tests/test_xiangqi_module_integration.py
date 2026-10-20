"""Protect the current Xiangqi protocol across orchestration module moves."""
from dataclasses import replace
import unittest
from unittest.mock import Mock

from minibench.cli import build_parser
from minibench.commands.xiangqi import _xiangqi_config_with_overrides
from minibench.datasets.xiangqi.mate_in_one import MATE_IN_ONE_SYSTEM_PROMPT
from minibench.experiments.dispatch import _evaluate
from minibench.experiments.metadata import _model_request_metadata
from minibench.experiments.registry import get_task_family_spec


class XiangqiModuleIntegrationTests(unittest.TestCase):
    def forwarded(self, family, config):
        evaluator = Mock(return_value=[])
        spec = replace(get_task_family_spec(family), evaluate_tasks=evaluator)
        callback = Mock()
        _evaluate(spec, [], object(), family, config, on_result=callback)
        self.assertIs(evaluator.call_args.kwargs["on_result"], callback)
        return evaluator.call_args.kwargs

    def test_deadline_and_pricing_reach_all_four_evaluators(self):
        pricing = {"currency": "USD", "input_per_million": 1, "output_per_million": 2}
        for family in ("xiangqi-mate-in-one", "xiangqi-history",
                       "xiangqi-rule-variants", "xiangqi-multimodal"):
            with self.subTest(family=family):
                kwargs = self.forwarded(family, {
                    "final_move_timeout_seconds": 123, "pricing": pricing,
                })
                self.assertEqual(kwargs["final_move_timeout_seconds"], 123)
                self.assertIs(kwargs["pricing"], pricing)

    def test_h2_and_c2_receive_complete_tree_configuration(self):
        for family in ("xiangqi-history", "xiangqi-rule-variants"):
            with self.subTest(family=family):
                kwargs = self.forwarded(family, {"solution_tree_dir": "data/xiangqi/solution_trees"})
                self.assertEqual(kwargs["solution_tree_dir"], "data/xiangqi/solution_trees")
                self.assertEqual(kwargs["final_move_timeout_seconds"], 900)
                if family == "xiangqi-rule-variants":
                    self.assertEqual(kwargs["search_depth"], 3)
                    self.assertNotIn("oracle_depth", kwargs)

    def test_m2_defaults_to_chinese_images_with_d3_system_prompt(self):
        spec = get_task_family_spec("xiangqi-multimodal")
        self.assertEqual(spec.system_prompt, MATE_IN_ONE_SYSTEM_PROMPT)
        self.assertEqual(self.forwarded("xiangqi-multimodal", {})["modes"],
                         ("chinese-piece-image",))

    def test_cli_c2_search_depth_uses_current_evaluation_field(self):
        args = build_parser().parse_args(["run-task", "c2", "--search-depth", "3"])
        config = _xiangqi_config_with_overrides(args, "xiangqi-rule-variants")
        self.assertEqual(config["evaluation"]["search_depth"], 3)
        self.assertNotIn("oracle_depth", config["evaluation"])

    def test_request_metadata_preserves_omit_setting_without_credentials(self):
        metadata = _model_request_metadata({
            "name": "generic", "model": "fixture", "omit_max_tokens": True,
            "api_key": "not-a-real-credential", "api_key_env": "FIXTURE_KEY",
        })
        self.assertTrue(metadata["omit_max_tokens"])
        self.assertNotIn("api_key", metadata)
        self.assertNotIn("api_key_env", metadata)


if __name__ == "__main__":
    unittest.main()

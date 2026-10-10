"""Current module boundaries, CLI ownership and resume fingerprint coverage."""
from importlib.util import find_spec
from pathlib import Path
import unittest
from unittest.mock import patch

from minibench.cli import build_parser
from minibench.experiments import one_stroke, zebra
from minibench.experiments.registry import TASK_FAMILIES, get_task_family_spec
from minibench.factory.config import load_experiment_config


class ProjectLayoutTests(unittest.TestCase):
    def test_registry_keeps_all_current_task_families(self):
        self.assertEqual(set(TASK_FAMILIES), {
            "one_stroke", "zebra", "mahjong", "mahjong_solo",
            "mahjong_rule_variants", "mahjong_riichi",
            "xiangqi-mate-in-one", "xiangqi-rule-variants",
            "xiangqi-history", "xiangqi-multimodal",
        })
        for family in TASK_FAMILIES:
            with self.subTest(family=family):
                spec = get_task_family_spec(family)
                self.assertTrue(spec.default_path.is_file())
                for method in (spec.load_tasks, spec.evaluate_tasks, spec.summarize, spec.write_run):
                    self.assertTrue(callable(method))

    def test_cli_handlers_are_owned_by_their_command_groups(self):
        parser = build_parser()
        cases = {
            "score-suite": (["--manifest", "manifest.json", "--output", "report"], "scoring"),
            "run-task": (["xiangqi-mate-in-one"], "xiangqi"),
            "run-suite": (["--tasks", "xiangqi-history"], "xiangqi"),
            "inspect-xiangqi": (["--task", "xiangqi-history", "--id", "test"], "xiangqi"),
            "build-xiangqi-gallery": (["--output", "gallery.html"], "xiangqi"),
            "evaluate-one-stroke": ([], "static"),
            "evaluate-zebra": ([], "static"),
            "evaluate-mahjong": ([], "mahjong"),
            "evaluate-mahjong-solo": ([], "mahjong"),
            "evaluate-mahjong-rules": ([], "mahjong"),
            "evaluate-mahjong-riichi": ([], "mahjong"),
            "generate-mahjong-static": ([], "mahjong"),
            "generate-mahjong-visual": ([], "mahjong"),
            "generate-mahjong-solo": ([], "mahjong"),
        }
        for command, (arguments, owner) in cases.items():
            with self.subTest(command=command):
                args = parser.parse_args([command, *arguments])
                self.assertEqual(args.func.__module__, f"minibench.commands.{owner}")
        args = parser.parse_args(["run-config", "experiment.yaml"])
        self.assertEqual(args.func.__module__, "minibench.cli")

    def test_all_shipped_experiment_configs_use_current_agents(self):
        paths = sorted(Path("config/experiments").rglob("*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path):
                config = load_experiment_config(path)
                self.assertNotEqual(config["agent"]["name"], "openai-compatible")

    def test_old_orchestration_module_is_not_a_compatibility_shell(self):
        self.assertIsNone(find_spec("minibench.factory.experiments"))

    def test_resume_fingerprints_cover_all_extracted_orchestration_files(self):
        filenames = sorted(path.name for path in Path(one_stroke.__file__).parent.glob("*.py"))
        for module, calculate in (
            (one_stroke, one_stroke._one_stroke_code_sha256),
            (zebra, zebra._zebra_code_sha256),
        ):
            for filename in filenames:
                with self.subTest(runner=module.__name__, filename=filename):
                    def original(path):
                        return path.as_posix()

                    def changed(path):
                        suffix = ":changed" if path.parent.name == "experiments" and path.name == filename else ""
                        return path.as_posix() + suffix

                    with patch.object(module, "sha256_file", side_effect=original):
                        before = calculate()
                    with patch.object(module, "sha256_file", side_effect=changed):
                        after = calculate()
                    self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()

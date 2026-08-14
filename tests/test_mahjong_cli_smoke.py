import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.cli import build_parser
from minibench.datasets.mahjong.dataset import mahjong_task_from_dict
from minibench.datasets.mahjong_rule_variants.dataset import MahjongRuleVariantTask


class OfflineMahjongAgent:
    def generate(self, prompt, task):
        if isinstance(task, MahjongRuleVariantTask):
            return json.dumps({"action": "tsumo"})
        return json.dumps(
            {
                "hand": list(task.hand),
                "visible_tiles": list(task.visible_tiles),
                "winning_tiles": ["E"],
            }
        )

    def generate_multimodal(self, prompt, task, *, images):
        return self.generate(prompt, task)


def static_wait_task():
    return mahjong_task_from_dict(
        {
            "id": "smoke-static",
            "goal": "winning_tiles",
            "hand": [
                "1m", "2m", "3m", "4m", "5m", "6m", "7p",
                "8p", "9p", "2s", "3s", "4s", "E",
            ],
            "tags": ["easy", "task:winning_tiles"],
        }
    )


def standard_rule_task():
    return MahjongRuleVariantTask(
        id="smoke-rule--standard",
        source_task_id="smoke-rule",
        channel="standard",
        seed=1,
        initial_hand=(
            "1m", "2m", "3m", "4p", "5p", "6p", "7s",
            "8s", "9s", "E", "E", "E", "N",
        ),
        wall=("N",),
        max_draws=1,
        round_wind="E",
        seat_wind="E",
        tags=("mahjong", "solo-draw-discard", "rule-channel:standard"),
    )


class MahjongCliOfflineSmokeTests(unittest.TestCase):
    def test_readme_static_command_runs_offline(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "minibench.datasets.mahjong.dataset.load_mahjong_tasks",
            return_value=[static_wait_task()],
        ), patch(
            "minibench.cli._make_cli_agent",
            return_value=OfflineMahjongAgent(),
        ):
            args = build_parser().parse_args(
                [
                    "evaluate-mahjong",
                    "--output-dir",
                    tmpdir,
                    "--run-name",
                    "static-smoke",
                ]
            )
            self.assertEqual(args.func(args), 0)

    def test_readme_rule_and_history_commands_run_offline(self):
        for observation_mode in ("full-hand", "history-only"):
            with tempfile.TemporaryDirectory() as tmpdir, patch(
                "minibench.datasets.mahjong_rule_variants.dataset."
                "load_mahjong_rule_variant_tasks",
                return_value=[standard_rule_task()],
            ), patch(
                "minibench.cli._make_cli_agent",
                return_value=OfflineMahjongAgent(),
            ):
                args = build_parser().parse_args(
                    [
                        "evaluate-mahjong-rules",
                        "--rule-channel",
                        "standard",
                        "--observation-mode",
                        observation_mode,
                        "--output-dir",
                        tmpdir,
                        "--run-name",
                        f"{observation_mode}-smoke",
                    ]
                )
                self.assertEqual(args.func(args), 0)

    def test_readme_visual_command_runs_paired_modes_offline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "visual.png"
            image_path.write_bytes(b"offline-smoke")
            visual_task = mahjong_task_from_dict(
                {
                    "id": "smoke-visual",
                    "goal": "winning_tiles",
                    "hand": list(static_wait_task().hand),
                    "visible_tiles": ["1p", "2p"],
                    "image": "visual.png",
                    "tags": [
                        "easy",
                        "task:winning_tiles",
                        "visual",
                        "visible:2",
                    ],
                },
                source_path=Path(tmpdir) / "tasks.jsonl",
            )
            with patch(
                "minibench.datasets.mahjong.dataset.load_mahjong_tasks",
                return_value=[visual_task],
            ), patch(
                "minibench.cli._make_cli_agent",
                return_value=OfflineMahjongAgent(),
            ):
                args = build_parser().parse_args(
                    [
                        "evaluate-mahjong",
                        "--input-mode",
                        "all",
                        "--output-dir",
                        tmpdir,
                        "--run-name",
                        "visual-smoke",
                    ]
                )
                self.assertEqual(args.func(args), 0)

            results = json.loads(
                (Path(tmpdir) / "visual-smoke" / "results.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(results["planned_total"], 2)
            self.assertEqual(results["completed_total"], 2)


if __name__ == "__main__":
    unittest.main()

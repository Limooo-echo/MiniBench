import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.cli import build_parser
from minibench.datasets.mahjong.dataset import mahjong_task_from_dict


class OfflineMahjongAgent:
    def generate(self, prompt, task):
        answer = {"winning_tiles": ["E"]}
        if task.task_id == "smoke-visual":
            answer.update(
                {
                    "hand": list(static_wait_task().hand),
                }
            )
        return json.dumps(answer)

    def generate_multimodal(self, prompt, task, *, images):
        return self.generate(prompt, task)

    def generate_messages(
        self,
        messages,
        task,
        *,
        temperature=None,
        max_tokens=None,
        json_mode=None,
    ):
        prompt = str(messages[-1]["content"]) if messages else ""
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

    def test_readme_independent_odd_rule_command_runs_offline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = Path(tmpdir) / "odd.jsonl"
            data.write_text(json.dumps({
                "id": "odd", "goal": "winning_tiles", "channel": "cyclic_odd_tile_substitution",
                "hand": "1s 2m 3s 4p 5m 6p 7p 8s 9p E E E N".split(),
            }) + "\n", encoding="utf-8")
            predictions = Path(tmpdir) / "answers.jsonl"
            predictions.write_text(json.dumps({"task_id": "odd",
                "raw_output": '{"winning_tiles":["N"]}'}) + "\n", encoding="utf-8")
            output = Path(tmpdir) / "runs"
            args = build_parser().parse_args([
                "evaluate-mahjong-rules", "--rule-channel", "cyclic_odd_tile_substitution",
                "--mahjong-rule-tasks", str(data), "--predictions", str(predictions),
                "--output-dir", str(output), "--run-name", "static-rule-smoke",
            ])
            self.assertEqual(args.func(args), 0)
            summary = json.loads((output / "static-rule-smoke/results.json").read_text(encoding="utf-8"))
            self.assertEqual((summary["total"], summary["success"]), (1, 1))

    def test_readme_visual_command_runs_image_only_offline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "visual.png"
            image_path.write_bytes(b"offline-smoke")
            visual_task = mahjong_task_from_dict(
                {
                    "id": "smoke-visual",
                    "goal": "winning_tiles",
                    "hand": list(static_wait_task().hand),
                    "image": "visual.png",
                    "tags": [
                        "easy",
                        "task:winning_tiles",
                        "visual",
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
                        "image",
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
            self.assertEqual(results["planned_total"], 1)
            self.assertEqual(results["completed_total"], 1)


if __name__ == "__main__":
    unittest.main()

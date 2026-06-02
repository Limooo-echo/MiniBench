import json
from pathlib import Path
import tempfile
import unittest

from minibench.cli import build_parser


class StaticAgentCliTests(unittest.TestCase):
    def test_static_commands_accept_reasoning_agents(self):
        parser = build_parser()

        cases = [
            ["evaluate-xiangqi", "--agent", "cot"],
            ["evaluate-one-stroke", "--agent", "tot"],
            ["evaluate-mahjong", "--agent", "critic-refine"],
        ]
        for argv in cases:
            with self.subTest(command=argv[0], agent=argv[-1]):
                args = parser.parse_args(argv)

                self.assertEqual(args.agent, argv[-1])

    def test_xiangqi_battle_tasks_reject_reasoning_agents(self):
        task = {
            "id": "unit-pikafish",
            "board": [
                [0, 0, 0, 0, -1, 0, 0, 0, -8],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -12, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 12, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [8, 0, 0, 0, 1, 0, 0, 0, 0],
            ],
            "side_to_move": "ally",
            "agent_side": "ally",
            "opponent": "pikafish",
            "max_steps": 16,
            "goal": "agent_win",
            "tags": ["xiangqi", "environment"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "xiangqi_hard.jsonl"
            path.write_text(json.dumps(task) + "\n", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "evaluate-xiangqi",
                    "--xiangqi-tasks",
                    str(path),
                    "--agent",
                    "cot",
                ]
            )

            with self.assertRaises(SystemExit) as raised:
                args.func(args)

        self.assertIn("only supported for static Xiangqi", str(raised.exception))


if __name__ == "__main__":
    unittest.main()

import importlib.util
import unittest
from pathlib import Path

from minibench.datasets.xiangqi.dataset import load_xiangqi_tasks, xiangqi_task_from_dict
from minibench.datasets.xiangqi.engines.pikafish import (
    board_to_pikafish_fen,
    square_to_uci,
    uci_to_square,
)


XIANGQI_ENV_AVAILABLE = importlib.util.find_spec("gym_xiangqi") is not None


class XiangqiPikafishTests(unittest.TestCase):
    def test_board_to_pikafish_fen(self):
        board = [
            [0, 0, 0, 8, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ]

        expected_board = "/".join(["3Rk4"] + ["9"] * 8 + ["4K4"])
        self.assertEqual(
            board_to_pikafish_fen(board, side_to_move="ally"),
            f"{expected_board} w - - 0 1",
        )
        self.assertEqual(
            board_to_pikafish_fen(board, side_to_move="enemy"),
            f"{expected_board} b - - 0 1",
        )

    def test_square_coordinate_round_trip(self):
        self.assertEqual(square_to_uci(7, 1), "b2")
        self.assertEqual(uci_to_square("b2"), (7, 1))
        self.assertEqual(square_to_uci(0, 8), "i9")
        self.assertEqual(uci_to_square("i9"), (0, 8))

    def test_dataset_accepts_survival_goal(self):
        task = xiangqi_task_from_dict(
            {
                "id": "unit-survival",
                "board": [[0] * 9 for _ in range(10)],
                "side_to_move": "ally",
                "agent_side": "ally",
                "opponent": "pikafish",
                "max_steps": 2,
                "goal": "agent_survive",
                "tags": ["xiangqi"],
            }
        )

        self.assertEqual(task.goal, "agent_survive")

    def test_hard_dataset_has_three_categories(self):
        tasks = load_xiangqi_tasks(Path("data/xiangqi/hard_tasks.jsonl"))
        categories = {
            tag
            for task in tasks
            for tag in task.tags
            if tag.startswith("category:")
        }

        self.assertEqual(len(tasks), 9)
        self.assertEqual(
            categories,
            {
                "category:tactical-win",
                "category:advantage-play",
                "category:survival-defense",
            },
        )
        for category in categories:
            self.assertEqual(
                sum(category in task.tags for task in tasks),
                3,
            )

    @unittest.skipUnless(XIANGQI_ENV_AVAILABLE, "gym-xiangqi is not installed")
    def test_strict_actions_filter_moves_that_leave_general_in_check(self):
        from minibench.datasets.xiangqi.env import (
            make_xiangqi_env_from_board,
            strict_legal_actions,
        )

        board = [
            [0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -12, 0, 0, 0, 0],
            [0, 0, 0, 0, 12, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [8, 0, 0, 0, 1, 0, 0, 0, -8],
        ]
        env = make_xiangqi_env_from_board(board, side_to_move="ally")

        try:
            actions = strict_legal_actions(env)

            self.assertNotIn(92731, actions)
            self.assertEqual(actions, [7726])
        finally:
            env.close()

    @unittest.skipUnless(XIANGQI_ENV_AVAILABLE, "gym-xiangqi is not installed")
    def test_prompt_uses_strict_legal_actions(self):
        from minibench.datasets.xiangqi.env import make_xiangqi_env_from_board
        from minibench.datasets.xiangqi.prompting import build_xiangqi_prompt

        task = xiangqi_task_from_dict(
            {
                "id": "unit-filter-prompt",
                "board": [
                    [0, 0, 0, 0, -1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, -12, 0, 0, 0, 0],
                    [0, 0, 0, 0, 12, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [8, 0, 0, 0, 1, 0, 0, 0, -8],
                ],
                "side_to_move": "ally",
                "agent_side": "ally",
                "opponent": "pikafish",
                "max_steps": 2,
                "goal": "agent_survive",
                "tags": ["xiangqi"],
            }
        )
        env = make_xiangqi_env_from_board(task.board, side_to_move=task.side_to_move)

        try:
            prompt = build_xiangqi_prompt(task, env, [])

            self.assertIn("7726: GENERAL#1", prompt)
            self.assertNotIn("92731: SOLDIER_1#12", prompt)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

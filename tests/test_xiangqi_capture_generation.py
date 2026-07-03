import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from minibench.datasets.xiangqi.simple_capture_generation import (
    generate_xiangqi_capture_tasks,
    winning_actions_for_board,
)


XIANGQI_ENV_AVAILABLE = importlib.util.find_spec("gym_xiangqi") is not None


@unittest.skipUnless(XIANGQI_ENV_AVAILABLE, "gym-xiangqi is not installed")
class XiangqiCaptureGenerationTests(unittest.TestCase):
    def test_generates_unique_one_move_capture_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "tasks.jsonl"

            summary = generate_xiangqi_capture_tasks(
                output=output,
                count=8,
                seed=123,
                piece_types=("rook", "cannon", "horse", "soldier"),
                difficulties=("easy", "medium"),
                overwrite=True,
                progress_interval=0,
            )

            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(summary["generated"], 8)
            self.assertEqual(len(records), 8)

            for record in records:
                self.assertEqual(record["goal"], "capture_enemy_general")
                self.assertEqual(record["max_steps"], 1)
                winning = winning_actions_for_board(record["board"])
                self.assertEqual(len(winning), 1)
                self.assertEqual(record["answer"]["action"], winning[0].action)


if __name__ == "__main__":
    unittest.main()

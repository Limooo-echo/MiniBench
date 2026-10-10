from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.accept_xiangqi_protocol import CorrectAgent, replay_history, run_acceptance
from tests.test_xiangqi_runtime_errors import task_fixture, board_fixture


class ProtocolAcceptanceTests(unittest.TestCase):
    def test_refuses_existing_output_before_any_engine_work(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run_acceptance(Path(directory), Path("engine-not-used"), ["d3"])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_scripted_h2_agent_replays_identical_full_and_message_history(self):
        class Engine:
            def __init__(self):
                self.seen = []
            def analyze_fen(self, fen, *, depth):
                self.seen.append((fen, depth))
                return SimpleNamespace(bestmove="a7a8")
        engine = Engine()
        agent = CorrectAgent(engine)
        task = task_fixture(board=board_fixture(with_pawn=True), family="xiangqi-history")
        full = agent.generate("Move history:\nagent: a8a7\npikafish: a6a5", task)
        historical = agent.generate_messages([
            {"role": "system", "content": "test"},
            {"role": "user", "content": "initial board"},
            {"role": "assistant", "content": '{"move":"a8a7"}'},
            {"role": "user", "content": "Previous turn: You played a8a7; the opponent replied with a6a5."},
        ], task)
        self.assertEqual(json.loads(full), {"move": "a7a8"})
        self.assertEqual(full, historical)
        self.assertEqual(engine.seen[0], engine.seen[1])
        self.assertEqual(engine.seen[0][1], 20)
        self.assertIn(" w ", engine.seen[0][0])

    def test_scripted_agent_rejects_corrupt_replay(self):
        with self.assertRaisesRegex(ValueError, "illegal move"):
            replay_history(board_fixture(), 1, ["a0a1"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import accept_xiangqi_protocol as acceptance
from scripts.accept_xiangqi_protocol import CorrectAgent, replay_history, run_acceptance
from minibench.datasets.xiangqi.schema import runtime_dict
from minibench.datasets.xiangqi.solution_tree import TreeSession
from tests.test_xiangqi_runtime_errors import task_fixture, board_fixture


class ProtocolAcceptanceTests(unittest.TestCase):
    def test_refuses_existing_output_before_any_engine_work(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run_acceptance(Path(directory), Path("engine-not-used"), ["d3"])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_scripted_h2_agent_supports_fresh_lagged_requests_and_current_transcription(self):
        from tests.test_xiangqi_h2_lagged import fixture, LINE, transcription
        from minibench.datasets.xiangqi import history
        from minibench.datasets.xiangqi.engines.pikafish import board_to_pikafish_fen
        from minibench.datasets.xiangqi.text_encoding import board_state_text
        task, states, _ = fixture()
        answers = {board_to_pikafish_fen(board, side_to_move="ally"): LINE[2*i]
                   for i, board in enumerate(states)}
        class Engine:
            def __init__(self):
                self.seen = []
            def analyze_fen(self, fen, *, depth):
                self.seen.append((fen, depth))
                return SimpleNamespace(bestmove=answers[fen])
        engine = Engine()
        full, lagged = CorrectAgent(engine), CorrectAgent(engine)
        full_answers = []
        for turn in range(3):
            moves = "\n".join(f"{'agent' if i % 2 == 0 else 'pikafish'}: {uci}"
                              for i, uci in enumerate(LINE[:2*turn]))
            full_answers.append(json.loads(full.generate("Move history:\n" + moves, task)))
        for turn in range(3):
            anchor = states[max(0, turn-1)]
            options = ({"agent_uci": LINE[2*turn-2], "opponent_uci": LINE[2*turn-1]}
                       if turn else {})
            messages = history._lagged_state_messages(task, board_state_text(anchor),
                                                     turn_number=turn+1, **options)
            reply = json.loads(lagged.generate_messages(messages, task))
            self.assertEqual(reply, full_answers[turn])
            self.assertEqual(reply["pieces"], transcription(states[turn]))
            self.assertEqual(reply["move"], LINE[2*turn])
        self.assertEqual(engine.seen[:3], engine.seen[3:])
        self.assertTrue(all(depth == 20 for _, depth in engine.seen))
        lagged.reset()
        self.assertEqual(lagged._lagged_moves, [])

    def test_failure_control_does_not_parse_its_own_intentionally_invalid_answer(self):
        agent = acceptance.FixedAgent("intentionally malformed")
        answer = agent.generate_messages([], task_fixture())
        self.assertEqual(answer, "intentionally malformed")
        self.assertEqual(agent.calls, 1)

    def test_scripted_agent_rejects_corrupt_replay(self):
        with self.assertRaisesRegex(ValueError, "illegal move"):
            replay_history(board_fixture(), 1, ["a0a1"])

    def test_current_c2_acceptance_uses_complete_tree_without_legacy_certificate(self):
        """The submitted tree corpus must work after retiring depth-3 witnesses."""
        original_read_text = Path.read_text
        def forbid_legacy(path, *args, **kwargs):
            if path.name == "runtime_certificates.json":
                raise AssertionError("Current acceptance must not read a legacy witness")
            return original_read_text(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(acceptance, "pikafish_fingerprint", return_value={
                 "binary_sha256": "offline", "eval_file_sha256": "offline"}), \
             patch.object(acceptance, "PikafishEngine"), \
             patch.object(Path, "read_text", forbid_legacy):
            report = run_acceptance(Path(directory) / "acceptance", Path("not-used"), ["c2"])
            self.assertEqual(report["status"], "passed", report)
            self.assertTrue(report["checks"]["c2.certified_shortest_protocol"])
            self.assertEqual(report["result_count"], 3)

    def test_tree_agent_replays_actual_defence_instead_of_historical_pv(self):
        record, _ = acceptance._read_first("c2")
        directory = acceptance.ROOT / "data/xiangqi/solution_trees/c2"
        session = TreeSession(record, directory)
        runtime = runtime_dict(record)
        agent = CorrectAgent(tree_records={record["id"]: record},
                             solution_tree_dirs={record["family"]: directory})
        first = json.loads(agent.generate("Move history:\n(initial position)", runtime))["move"]
        root = session.tree["nodes"][session.tree["root"]]
        self.assertIn(first, root["edges"])
        defender = session.tree["nodes"][root["edges"][first]]
        reply = min(defender["edges"])
        raw = agent.generate(f"Move history:\nagent: {first}\ndepth-opponent: {reply}", runtime)
        next_node = session.tree["nodes"][defender["edges"][reply]]
        self.assertIn(json.loads(raw)["move"], next_node["edges"])


if __name__ == "__main__":
    unittest.main()

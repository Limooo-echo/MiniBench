"""H2 rolling one-round reconstruction and true-board recognition checks."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from minibench.datasets.xiangqi import history as h2
from minibench.datasets.xiangqi.schema import fen_to_board
from minibench.datasets.xiangqi.text_encoding import board_state_text, FILES, piece_symbol
from minibench.datasets.xiangqi.variants.board import VariantBoard
from tests.test_xiangqi_runtime_errors import task_fixture, no_engine


FEN = "3R5/C1c1ak3/4b4/2p1p3p/1r7/2P6/P3P3P/2N1B4/4K2R1/2BA1r3 w - - 0 1"
LINE = ("h1h8", "f8f7", "d9f9", "e8f9", "h8f8")


def key(board):
    return tuple(tuple(row) for row in board)


def transcription(board):
    return {f"{FILES[c]}{9-r}": piece_symbol(piece)
            for r, rank in enumerate(board) for c, piece in enumerate(rank) if piece}


def fixture():
    board, _ = fen_to_board(FEN)
    task = task_fixture(board=board, fen=FEN, max_steps=5,
                        shortest_mate_moves=3, oracle={"mate_in_plies": 5})
    runtime = VariantBoard(board, [])
    states, moves = [], {}
    for index, uci in enumerate(LINE):
        moves[key(runtime.board)] = uci
        if index % 2 == 0:
            states.append(deepcopy(runtime.board))
        side = 1 if index % 2 == 0 else -1
        move = next(move for move in runtime.legal_moves(side) if move.to_uci() == uci)
        runtime.apply(move)
    assert not runtime.legal_moves(-1) and runtime._is_in_check(-1)
    return task, states, moves


class ReplayAgent:
    def __init__(self, states, *, wrong_round=None, missing_pieces=False,
                 malformed_round=None, outage_round=None):
        self.states = states
        self.wrong_round, self.missing_pieces = wrong_round, missing_pieces
        self.malformed_round, self.outage_round = malformed_round, outage_round
        self.prompts, self.messages = [], []
        self.index = 0

    def reset(self):
        self.index = 0

    def output(self):
        index = self.index
        self.index += 1
        if index == self.outage_round:
            raise ConnectionError("offline model fixture outage")
        if index == self.malformed_round:
            return "not a JSON move"
        payload = {"move": LINE[2*index]}
        if not self.missing_pieces:
            board = self.states[0] if index == self.wrong_round else self.states[index]
            payload["pieces"] = transcription(board)
        return json.dumps(payload)

    def generate(self, prompt, task):
        self.prompts.append(prompt)
        return self.output()

    def generate_messages(self, messages, task, **kwargs):
        self.messages.append(deepcopy(messages))
        return self.output()


def run(mode="move-history-only", **agent_options):
    task, states, moves = fixture()
    agent = ReplayAgent(states, **agent_options)
    def analysis(engine, board, side, *, perspective_side, depth):
        return moves[key(board.board)], 0.0
    with no_engine(h2), patch.object(h2, "_pikafish_analysis", side_effect=analysis):
        result = h2.evaluate_history_tasks([task], agent, history_mode=mode)[0]
    return result, agent, states


class H2LaggedStateTests(unittest.TestCase):
    def test_prompt_states_both_palaces_without_giving_a_solution(self):
        text = h2.HISTORY_SYSTEM_PROMPT
        self.assertIn("files d/e/f and ranks 0/1/2", text)
        self.assertIn("files d/e/f and ranks 7/8/9", text)
        self.assertIn("EVERY hypothetical reply", text)
        self.assertIn("an advisor moves one square diagonally", text)
        self.assertNotIn("f0e0", text)
        self.assertNotIn("b1b0", text)

    def test_first_request_is_only_initial_board_without_an_invented_previous_pair(self):
        task, states, _ = fixture()
        messages = h2._lagged_state_messages(task, "INITIAL_BOARD_MARKER", turn_number=1)
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        prompt = messages[1]["content"]
        self.assertIn("INITIAL_BOARD_MARKER", prompt)
        self.assertIn('"pieces"', prompt)
        self.assertIn('"move"', prompt)
        self.assertNotIn(LINE[0], prompt)
        self.assertNotIn(LINE[1], prompt)

    def test_later_request_contains_only_supplied_lagged_board_and_previous_pair(self):
        task, _, _ = fixture()
        messages = h2._lagged_state_messages(task, "PREVIOUS_ROUND_START", agent_uci=LINE[2],
                                           opponent_uci=LINE[3], turn_number=3)
        self.assertEqual(len(messages), 2)
        prompt = messages[1]["content"]
        self.assertIn("PREVIOUS_ROUND_START", prompt)
        self.assertIn(LINE[2], prompt)
        self.assertIn(LINE[3], prompt)
        self.assertNotIn(LINE[0], prompt)
        self.assertNotIn(LINE[1], prompt)

    def test_real_three_turn_game_uses_correct_rolling_snapshot_and_no_accumulated_chat(self):
        result, agent, states = run()
        self.assertTrue(result.success)
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(agent.messages), 3)
        for messages in agent.messages:
            self.assertEqual([m["role"] for m in messages], ["system", "user"])
        prompts = [messages[-1]["content"] for messages in agent.messages]
        expected = (states[0], states[0], states[1])
        for prompt, board in zip(prompts, expected):
            self.assertIn(board_state_text(board), prompt)
        self.assertNotIn(board_state_text(states[1]), prompts[1])
        self.assertNotIn(board_state_text(states[2]), prompts[2])
        self.assertNotIn(board_state_text(states[0]), prompts[2])
        self.assertIn(LINE[0], prompts[1])
        self.assertIn(LINE[1], prompts[1])
        self.assertIn(LINE[2], prompts[2])
        self.assertIn(LINE[3], prompts[2])
        self.assertNotIn(LINE[0], prompts[2])
        self.assertNotIn(LINE[1], prompts[2])
        self.assertTrue(result.recognition_exact)
        self.assertEqual(result.outcome_labels["recognition_exact"], 1)
        self.assertEqual(result.outcome_labels["mate_given_recognition_and_legal"], 1)
        self.assertEqual([s["recognition_exact"] for s in result.steps if s["actor"] == "agent"],
                         [True, True, True])

    def test_full_state_control_uses_current_board_and_same_transcription_output(self):
        result, agent, states = run("full-state")
        self.assertTrue(result.success)
        self.assertTrue(result.recognition_exact)
        self.assertEqual(len(agent.prompts), 3)
        for number, (prompt, board) in enumerate(zip(agent.prompts, states), start=1):
            self.assertIn(f"This is your move number {number}.", prompt)
            self.assertIn(board_state_text(board), prompt)
            self.assertIn('"pieces"', prompt)
            self.assertIn('"move"', prompt)

    def test_wrong_reconstruction_does_not_rewrite_true_board_or_hide_lucky_mate(self):
        result, agent, states = run(wrong_round=1)
        self.assertTrue(result.success)
        self.assertFalse(result.recognition_exact)
        self.assertEqual(result.outcome_labels["mate"], 1)
        self.assertEqual(result.outcome_labels["legal"], 1)
        self.assertEqual(result.outcome_labels["recognition_exact"], 0)
        self.assertEqual(result.outcome_labels["mate_given_recognition_and_legal"], "NA")
        decisions = [s for s in result.steps if s["actor"] == "agent"]
        self.assertEqual([s["recognition_exact"] for s in decisions], [True, False, True])
        self.assertEqual([s["is_legal"] for s in decisions], [True, True, True])
        self.assertEqual(result.legality_rate, 1.0)
        self.assertEqual([s["uci"] for s in result.steps], list(LINE))

    def test_move_only_legacy_response_is_model_recognition_failure_not_infrastructure_error(self):
        result, _, _ = run(missing_pieces=True)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.success)
        self.assertFalse(result.recognition_exact)
        self.assertTrue(all(s["recognition_exact"] is False for s in result.steps if s["actor"] == "agent"))

    def test_malformed_second_response_fails_reconstruction_and_move_quality(self):
        result, _, _ = run(malformed_round=1)
        self.assertEqual(result.status, "invalid")
        self.assertFalse(result.success)
        self.assertFalse(result.recognition_exact)
        decisions = [s for s in result.steps if s["actor"] == "agent"]
        self.assertTrue(decisions[0]["recognition_exact"])
        self.assertFalse(decisions[-1]["recognition_exact"])
        self.assertFalse(decisions[-1]["is_legal"])

    def test_provider_outage_keeps_whole_episode_recognition_unknown(self):
        result, _, _ = run(outage_round=1)
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.success)
        self.assertIsNone(result.recognition_exact)
        self.assertEqual(result.outcome_labels["recognition_exact"], "NA")
        self.assertEqual(result.outcome_labels["mate_given_recognition_and_legal"], "NA")


    def test_current_board_transcription_parser_is_independent_of_move_format(self):
        self.assertEqual(h2._extract_history_uci('{"pieces":{},"move":"H5H9"}'), "h5h9")
        self.assertEqual(h2._extract_history_uci('{"pieces":"bad transcription","move":"h5h9"}'), "h5h9")
        self.assertIsNone(h2._extract_history_uci('{"pieces":{},"move":"h5h9","analysis":"extra"}'))

    def test_full_state_control_uses_the_same_strict_gate_for_lucky_mates(self):
        result, _, _ = run("full-state", wrong_round=1)
        self.assertTrue(result.success)
        self.assertFalse(result.recognition_exact)
        self.assertEqual(result.outcome_labels["mate"], 1)
        self.assertNotIn("recognition_exact", result.outcome_labels)
        self.assertNotIn("mate_given_recognition_and_legal", result.outcome_labels)

if __name__ == "__main__":
    unittest.main()

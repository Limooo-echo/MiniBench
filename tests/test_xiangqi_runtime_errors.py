"""Offline protocol regressions: malformed answers, outages and strict mate goals."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.agents.direct import DirectAgent
from minibench.core.agent import ReasoningConfig
from minibench.core.runtime import StrictJSONObjectError
from minibench.datasets.xiangqi.runtime import extract_strict_uci_move
from minibench.datasets.xiangqi.dataset import XiangqiTask
from minibench.datasets.xiangqi import mate_in_one as d3, history as h2, multimodal as m2, rule_variants as c2
from minibench.datasets.xiangqi.variants.board import VariantBoard
from minibench.datasets.xiangqi.variants.mate_search import MoveValue


def board_fixture(*, with_pawn=False):
    # R a8-d8 is mate: N b7 protects d8; N d7 covers the only escape e9.
    # R a8-a7 instead stalemates Black unless Black still has pawn a6.
    board = [[0] * 9 for _ in range(10)]
    board[0][3], board[9][4] = -1, 1
    board[1][0], board[2][1], board[2][3] = 8, 6, 7
    if with_pawn:
        board[3][0] = -12
    return board


def task_fixture(**changes):
    task = XiangqiTask(id="runtime-unit", board=board_fixture(), side_to_move="ally", agent_side="ally",
                       max_steps=3, tags=(), difficulty="short", oracle={}, schema_version=2,
                       family="xiangqi-history", fen="", agent_color="red")
    return replace(task, **changes)


def dict_fixture(**changes):
    return {
        "id": "runtime-unit",
        "scenario_id": "runtime-scenario",
        "board": board_fixture(),
        "rules": [],
        "ruleset": "standard",
        "agent_color": "red",
        "side_to_move": "ally",
        "max_plies": 3,
        "difficulty": "short",
        "oracle": {"mate_in_plies": 1},
        **changes,
    }


class FixedAgent:
    def __init__(self, output='{"move":"a8d8"}', error=None):
        self.output, self.error = output, error
        self.calls = self.resets = 0
        self.messages = []

    def reset(self):
        self.resets += 1

    def generate(self, prompt, task):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.output

    def generate_multimodal(self, prompt, task, *, images):
        return self.generate(prompt, task)

    def generate_messages(self, messages, task, **kwargs):
        self.messages.append([dict(message) for message in messages])
        return self.generate("", task)


class ResponseClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.options = []

    def complete(self, prompt, **kwargs):
        self.calls += 1
        self.options.append(kwargs)
        return next(self.responses)

    def complete_messages(self, messages, **kwargs):
        return self.complete("", **kwargs)


@contextmanager
def no_engine(module):
    with patch.object(module, "resolve_pikafish_executable", side_effect=RuntimeError("offline diagnostic engine")):
        yield


class RuntimeErrorsTests(unittest.TestCase):
    def test_move_output_requires_exact_single_field_json(self):
        self.assertEqual(extract_strict_uci_move('{"move":"H5H9"}'), "h5h9")
        for raw in (
            "h5h9",
            '{"answer":"h5h9"}',
            '{"move":"h5h9","analysis":"extra"}',
            '{"move":"h5h9 trailing"}',
            '```json\n{"move":"h5h9"}\n```',
        ):
            with self.subTest(raw=raw):
                self.assertIsNone(extract_strict_uci_move(raw))

    def test_d3_engine_start_failure_does_not_change_exact_success(self):
        agent = FixedAgent()
        received = []
        with no_engine(d3):
            result = d3.evaluate_mate_in_one_tasks([task_fixture()], agent, on_result=received.append)[0]
        self.assertEqual((result.status, result.success, result.goal_achieved), ("ok", True, True))
        self.assertIsNone(result.cp_before)
        self.assertIsNone(result.cp_loss)
        self.assertEqual(result.diagnostics[0]["stage"], "diagnostic_engine_start")
        self.assertEqual(received, [result])
        self.assertEqual(agent.calls, 1)

    def test_d3_accepts_every_mate_not_just_a_single_oracle_answer(self):
        mates = d3.enumerate_mate_in_one_moves(board_fixture(), 1)
        self.assertGreater(len(mates), 1)
        for move in mates:
            with self.subTest(move=move), no_engine(d3):
                result = d3.evaluate_mate_in_one_tasks([task_fixture()], FixedAgent(json.dumps({"move": move})))[0]
            self.assertTrue(result.success)

    def test_d3_timeout_is_missing_but_bad_answer_is_zero(self):
        with no_engine(d3):
            good = d3.evaluate_mate_in_one_tasks([task_fixture()], FixedAgent())[0]
            missing = d3.evaluate_mate_in_one_tasks([task_fixture()], FixedAgent(error=TimeoutError("timeout")))[0]
            bad = d3.evaluate_mate_in_one_tasks([task_fixture()], FixedAgent("bad answer"))[0]
        self.assertEqual(missing.status, "error")
        self.assertIsNone(missing.success)
        self.assertIsNone(missing.goal_achieved)
        self.assertEqual(missing.error["stage"], "model")
        self.assertEqual((bad.status, bad.success), ("invalid", False))
        summary = d3.summarize_mate_in_one([good, missing, bad])
        self.assertEqual((summary["total"], summary["evaluated"], summary["missing"], summary["invalid"]), (3, 2, 1, 1))
        self.assertEqual(summary["mate_rate"], .5)

    def test_complete_response_is_preserved(self):
        raw = json.dumps({"move": "a8d8", "padding": "evidence " * 150})
        with no_engine(d3):
            result = d3.evaluate_mate_in_one_tasks([task_fixture()], FixedAgent(raw))[0]
        self.assertEqual(result.raw_output, raw)

    def test_h2_both_modes_call_once_on_model_timeout(self):
        agent = FixedAgent(error=TimeoutError("provider failed"))
        received = []
        with no_engine(h2):
            results = h2.evaluate_history_tasks([task_fixture()], agent, history_mode="paired", on_result=received.append)
        self.assertEqual(agent.calls, 2)
        self.assertEqual(agent.resets, 2)
        self.assertEqual(received, results)
        self.assertTrue(all(r.status == "error" and r.success is None and r.error["stage"] == "model" for r in results))
        summary = h2.summarize_history(results)
        self.assertIsNone(summary["mate_rate"])
        self.assertEqual(summary["paired_comparison"]["paired_total"], 0)

    def test_h2_refuses_to_drop_history_for_unsupported_agent(self):
        class TextOnly:
            def generate(self, *_args):
                raise AssertionError("flattened fallback must not run")
        with no_engine(h2):
            result = h2.evaluate_history_tasks([task_fixture()], TextOnly())[0]
        self.assertEqual(result.error["stage"], "configuration")
        self.assertIn("generate_messages", result.error["reason"])
        self.assertIsNone(result.success)

    def test_h2_stalemate_is_a_valid_failure(self):
        with no_engine(h2):
            result = h2.evaluate_history_tasks([task_fixture()], FixedAgent('{"move":"a8a7"}'))[0]
        self.assertEqual((result.status, result.success), ("ok", False))
        self.assertEqual(result.termination_reason, "stalemate")

    def test_h2_reference_horizon_is_a_hard_total_cap_without_extension(self):
        class SequenceAgent(FixedAgent):
            def __init__(self):
                super().__init__()
                self.outputs = iter(('{"move":"a8a7"}', '{"move":"a7a8"}', '{"move":"a8d8"}'))

            def generate(self, *_args):
                self.calls += 1
                return next(self.outputs)

        def analysis(_engine, board, side, **_kwargs):
            if side == 1:
                return "a8d8", 0.0
            return ("a6a5", 0.0) if board.board[3][0] else ("a5a4", 0.0)

        task = task_fixture(
            board=board_fixture(with_pawn=True), max_steps=5,
            oracle={"mate_in_plies": 3},
        )
        agent = SequenceAgent()
        with no_engine(h2), patch.object(h2, "_pikafish_analysis", side_effect=analysis):
            result = h2.evaluate_history_tasks([task], agent)[0]
        self.assertEqual(agent.calls, 2)
        self.assertEqual(len(result.steps), 3)
        self.assertIsNone(result.checkmate_ply)
        self.assertFalse(result.success)
        self.assertFalse(result.goal_achieved)
        self.assertFalse(result.extended_mate_success)
        self.assertFalse(result.mate_within_prescribed_steps)
        self.assertEqual(result.termination_reason, "max_steps_reached")
        self.assertEqual(result.reference_mate_plies, 3)
        self.assertIsNone(result.first_move_optimal_rate)
        summary = h2.summarize_history([result])
        self.assertEqual(summary["mate_rate"], 0.0)
        self.assertEqual(summary["mate_within_prescribed_rate"], 0.0)

    def test_h2_unavailable_required_opponent_is_missing(self):
        with no_engine(h2):
            result = h2.evaluate_history_tasks([task_fixture(board=board_fixture(with_pawn=True))], FixedAgent('{"move":"a8a7"}'))[0]
        self.assertEqual((result.status, result.success), ("error", None))
        self.assertEqual(result.error["stage"], "opponent_engine")
        self.assertEqual(result.steps[0]["uci"], "a8a7")

    def test_h2_complete_history_reaches_second_turn(self):
        class SequenceAgent(FixedAgent):
            def generate(self, *_args):
                self.calls += 1
                return '{"move":"a8a7"}' if self.calls == 1 else '{"move":"a7a8"}'
        agent = SequenceAgent()
        def analysis(_engine, _board, side, **_kwargs):
            return ("a6a5", 0.0) if side == -1 else ("a8d8", 0.0)
        with no_engine(h2), patch.object(h2, "_pikafish_analysis", side_effect=analysis):
            result = h2.evaluate_history_tasks([task_fixture(board=board_fixture(with_pawn=True))], agent)[0]
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(agent.messages), 2)
        self.assertEqual([m["role"] for m in agent.messages[1]], ["system", "user", "assistant", "user"])
        self.assertEqual(agent.messages[1][1], agent.messages[0][1])
        self.assertIn("a6a5", agent.messages[1][-1]["content"])

    def test_m2_no_extra_image_answer_and_reset_each_condition(self):
        agent = FixedAgent("first invalid answer")
        received = []
        with patch.object(m2, "render_board_png", return_value=b"image"):
            results = m2.evaluate_xiangqi_multimodal_tasks([dict_fixture()], agent, modes=["text","chinese-piece-image"], verify_with_pikafish=False, on_result=received.append)
        self.assertEqual(agent.calls, 2)
        self.assertEqual(agent.resets, 2)
        self.assertEqual(received, results)
        self.assertTrue(all(r["status"] == "invalid" and r["success"] is False for r in results))
        self.assertEqual(m2.summarize_xiangqi_multimodal(results)["by_input_mode"]["text"]["mate_rate"], 0)

    def test_m2_diagnostic_start_failure_does_not_suppress_exact_score(self):
        with no_engine(m2):
            result = m2.evaluate_xiangqi_multimodal_tasks([dict_fixture()], FixedAgent(), modes=["text"])[0]
        self.assertEqual((result["status"], result["success"]), ("ok", True))
        self.assertIsNone(result["engine_cp_loss"])

    def test_black_side_d3_and_m2_share_exact_checkmate_and_correct_prompt(self):
        black_board = [[-piece for piece in row] for row in reversed(board_fixture())]
        task = task_fixture(board=black_board, side_to_move="enemy", agent_side="enemy", agent_color="black")
        raw = '{"move":"a1d1"}'
        with no_engine(d3):
            direct = d3.evaluate_mate_in_one_tasks([task], FixedAgent(raw))[0]
        self.assertTrue(direct.success)
        self.assertIn("checkmate Red in exactly this one Black move", d3._build_mate_in_one_prompt(task, VariantBoard(black_board, [])))
        with patch.object(m2, "render_board_png", return_value=b"image"):
            results = m2.evaluate_xiangqi_multimodal_tasks([dict_fixture(board=black_board, agent_color="black", side_to_move="enemy")], FixedAgent(raw), verify_with_pikafish=False)
        self.assertTrue(all(result["success"] for result in results))
        for mode in m2.XIANGQI_MULTIMODAL_INPUT_MODES:
            prompt = m2._build_multimodal_prompt(VariantBoard(black_board, []), mode, "", agent_side=-1)
            self.assertIn("checkmate Red in exactly this one Black move", prompt)

    def test_h2_prompts_state_same_budget_without_claiming_proven_force(self):
        task = task_fixture()
        full = h2._full_state_prompt(task, VariantBoard(task.board, []), [])
        history = h2._initial_move_history_messages(task, "initial")[1]["content"]
        for prompt in (full, history):
            self.assertIn("hard total limit of 3 half-moves", prompt)
            self.assertIn("Stalemate is not checkmate", prompt)
            self.assertNotIn("A forced checkmate exists", prompt)

    def test_m2_timeout_excluded_from_paired_denominator(self):
        class ImageOutage(FixedAgent):
            def generate_multimodal(self, *_args, **_kwargs):
                raise TimeoutError("images unavailable")
        with patch.object(m2, "render_board_png", return_value=b"image"):
            results = m2.evaluate_xiangqi_multimodal_tasks([dict_fixture()], ImageOutage(), verify_with_pikafish=False)
        summary = m2.summarize_xiangqi_multimodal(results)
        self.assertEqual(summary["missing"], 1)
        self.assertIsNone(summary["by_input_mode"]["chinese-piece-image"]["mate_rate"])
        self.assertIsNone(summary["by_input_mode"]["chinese-piece-image"]["legality_rate"])

    def test_wrong_engine_hash_preserves_d3_and_m2_exact_scores_without_startup(self):
        actual = {"binary_sha256": "a" * 64, "eval_file_sha256": "b" * 64}
        with patch("minibench.datasets.xiangqi.engines.pikafish.pikafish_fingerprint", return_value=actual):
            for module in (d3, m2):
                with self.subTest(module=module.__name__), patch.object(module, "resolve_pikafish_executable", return_value="unused"), patch.object(module, "PikafishEngine") as engine:
                    if module is d3:
                        result = asdict(d3.evaluate_mate_in_one_tasks([task_fixture()], FixedAgent(), pikafish_binary_sha256="c" * 64)[0])
                    else:
                        result = m2.evaluate_xiangqi_multimodal_tasks([dict_fixture()], FixedAgent(), modes=["text"], pikafish_nnue_sha256="c" * 64)[0]
                    engine.assert_not_called()
                    self.assertEqual((result["status"], result["success"]), ("ok", True))
                    self.assertIn("fingerprint mismatch", result["diagnostics"][0]["reason"])

    def test_wrong_h2_nnue_hash_is_missing_when_opponent_is_required(self):
        actual = {"binary_sha256": "a" * 64, "eval_file_sha256": "b" * 64}
        with patch("minibench.datasets.xiangqi.engines.pikafish.pikafish_fingerprint", return_value=actual), patch.object(h2, "resolve_pikafish_executable", return_value="unused"), patch.object(h2, "PikafishEngine") as engine:
            results = h2.evaluate_history_tasks([task_fixture(board=board_fixture(with_pawn=True))], FixedAgent('{"move":"a8a7"}'), history_mode="paired", pikafish_binary_sha256="a" * 64, pikafish_nnue_sha256="c" * 64)
        engine.assert_not_called()
        self.assertTrue(all(r.status == "error" and r.success is None for r in results))
        self.assertTrue(all(r.error["stage"] == "opponent_engine" for r in results))
        self.assertTrue(all("fingerprint mismatch" in r.diagnostics[0]["reason"] for r in results))

    def test_c2_mate_success_is_separate_from_loss_and_search_depth_is_fixed(self):
        legal = VariantBoard(board_fixture(), []).legal_moves(1)
        mate_uci = d3.enumerate_mate_in_one_moves(board_fixture(), 1)
        mates = [move for move in legal if move.to_uci() in mate_uci]
        self.assertGreaterEqual(len(mates), 2)
        wrong_move = next(move for move in legal if move.to_uci() not in mate_uci)
        scored = [
            MoveValue(mates[0], 20.0000005),
            MoveValue(mates[1], 20.0),
            MoveValue(wrong_move, 10.0),
        ]
        with patch.object(c2, "score_moves_cpe", return_value=scored) as oracle:
            first = c2.evaluate_rule_variant_task(
                dict_fixture(max_plies=1),
                FixedAgent(json.dumps({"move": mates[0].to_uci()})),
            )
            second = c2.evaluate_rule_variant_task(
                dict_fixture(max_plies=1),
                FixedAgent(json.dumps({"move": mates[1].to_uci()})),
            )
            wrong = c2.evaluate_rule_variant_task(
                dict_fixture(max_plies=1),
                FixedAgent(json.dumps({"move": wrong_move.to_uci()})),
            )
        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertTrue(first.steps[0]["is_optimal"])
        self.assertTrue(second.steps[0]["is_optimal"])
        self.assertEqual(first.avg_variant_loss, 0.0)
        self.assertEqual(second.avg_variant_loss, 0.0)
        self.assertEqual((wrong.status, wrong.success), ("ok", False))
        self.assertEqual(wrong.legality_rate, 1.0)
        self.assertEqual(wrong.avg_variant_loss, 10.0)
        self.assertEqual(oracle.call_args.kwargs["depth"], 3)
        with self.assertRaisesRegex(ValueError, "search_depth=3"):
            c2.evaluate_rule_variant_task(dict_fixture(), FixedAgent(), search_depth=2)

    def test_c2_judge_and_model_errors_are_missing(self):
        with patch.object(c2, "score_moves_cpe", side_effect=RuntimeError("judge crashed")):
            judge = c2.evaluate_rule_variant_task(dict_fixture(), FixedAgent())
        provider = c2.evaluate_rule_variant_task(
            dict_fixture(), FixedAgent(error=TimeoutError("timeout"))
        )
        self.assertEqual(judge.error["stage"], "judge")
        self.assertEqual(provider.error["stage"], "model")
        self.assertIsNone(c2.summarize_rule_variants([judge, provider])["mate_rate"])

    def test_architecture_format_repair_failure_is_invalid_in_every_family(self):
        for family in ("d3", "h2", "c2", "m2"):
            with self.subTest(family=family), ExitStack() as stack:
                client = ResponseClient(["original malformed " * 40, "repair malformed " * 40])
                agent = DirectAgent(client, ReasoningConfig(max_format_repairs=1))
                if family == "d3":
                    stack.enter_context(no_engine(d3))
                    result = asdict(d3.evaluate_mate_in_one_tasks([task_fixture()], agent)[0])
                    raw = result["raw_output"]
                elif family == "h2":
                    stack.enter_context(no_engine(h2))
                    result = asdict(h2.evaluate_history_tasks([task_fixture()], agent)[0])
                    raw = result["steps"][0]["raw_output"]
                elif family == "c2":
                    result = asdict(c2.evaluate_rule_variant_task(dict_fixture(), agent))
                    raw = result["steps"][0]["raw_output"]
                else:
                    result = m2.evaluate_xiangqi_multimodal_tasks([dict_fixture()], agent, modes=["text"], verify_with_pikafish=False)[0]
                    raw = result["steps"][0]["raw_output"]
                self.assertEqual((result["status"], result["success"]), ("invalid", False))
                self.assertEqual(result["error"]["stage"], "format")
                self.assertEqual(raw, "repair malformed " * 40)
                self.assertEqual(result["error"]["raw_outputs"], ["original malformed " * 40, "repair malformed " * 40])
                self.assertEqual(client.calls, 2)
                self.assertEqual(result["metrics"]["llm_calls"], 2)

    def test_same_architecture_generation_options_for_text_and_images(self):
        client = ResponseClient(['{"move":"a8d8"}'] * 3)
        agent = DirectAgent(client)
        with patch.object(m2, "render_board_png", return_value=b"image"):
            results = m2.evaluate_xiangqi_multimodal_tasks([dict_fixture()], agent, modes=["text","chinese-piece-image"], verify_with_pikafish=False)
        self.assertTrue(all(r["success"] for r in results))
        self.assertEqual(len(client.options), 2)
        for key in ("temperature", "max_tokens", "json_mode"):
            self.assertEqual(len({options[key] for options in client.options}), 1)

    def test_diagnostic_failure_after_move_does_not_relabel_legal_wrong_answer(self):
        class FakeEngine:
            def __init__(self, *_args, **_kwargs):
                self.calls = 0
            def start(self):
                pass
            def close(self):
                pass
            def bestmove_for_fen(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("diagnostic crashed after move")
                return "a8d8", ["info score mate 1"]
            def analyze_fen(self, *_args, **_kwargs):
                from minibench.datasets.xiangqi.engines.pikafish import PikafishAnalysis
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("diagnostic crashed after move")
                return PikafishAnalysis("unused", "a8d8", "mate", 1, 8, ("a8d8",), ())
        wrong = FixedAgent('{"move":"a8a7"}')
        with patch.object(d3, "resolve_pikafish_executable", return_value="fake"), patch.object(d3, "PikafishEngine", FakeEngine), patch.object(d3, "_ensure_engine_alive"):
            direct = d3.evaluate_mate_in_one_tasks([task_fixture()], wrong)[0]
        with patch.object(m2, "resolve_pikafish_executable", return_value="fake"), patch.object(m2, "PikafishEngine", FakeEngine):
            image = m2.evaluate_xiangqi_multimodal_tasks([dict_fixture()], wrong, modes=["text"])[0]
        self.assertEqual((direct.status, direct.success), ("ok", False))
        self.assertEqual((image["status"], image["success"]), ("ok", False))
        self.assertIsNone(direct.cp_loss)
        self.assertIsNone(image["engine_cp_loss"])
        self.assertEqual(direct.diagnostics[-1]["stage"], "diagnostic_engine_after")
        self.assertEqual(image["diagnostics"][-1]["stage"], "diagnostic_engine_after")

    def test_h2_checkmate_does_not_require_a_diagnostic_engine(self):
        with no_engine(h2):
            result = h2.evaluate_history_tasks([task_fixture()], FixedAgent())[0]
        self.assertEqual((result.status, result.success), ("ok", True))
        self.assertEqual(result.termination_reason, "agent_checkmated_opponent")
        self.assertIsNone(result.avg_cp_loss)

    def test_architecture_without_format_repair_still_retains_invalid_raw(self):
        raw = "unformatted " * 80
        client = ResponseClient([raw])
        agent = DirectAgent(client, ReasoningConfig(max_format_repairs=0))
        with no_engine(d3):
            result = d3.evaluate_mate_in_one_tasks([task_fixture()], agent)[0]
        self.assertEqual((result.status, result.success), ("invalid", False))
        self.assertEqual(result.raw_output, raw)
        self.assertEqual(result.error["raw_outputs"], [raw])
        self.assertEqual(client.calls, 1)

    def test_error_only_writers_and_checkpoint_preservation(self):
        failure = FixedAgent(error=TimeoutError("offline"))
        with no_engine(d3), no_engine(h2):
            datasets = [
                (d3.write_mate_in_one_run, d3.evaluate_mate_in_one_tasks([task_fixture()], failure)),
                (h2.write_history_run, h2.evaluate_history_tasks([task_fixture()], failure)),
                (c2.write_rule_variants_run, c2.evaluate_rule_variant_tasks([dict_fixture()], failure)),
                (m2.write_xiangqi_multimodal_run, m2.evaluate_xiangqi_multimodal_tasks([dict_fixture()], failure, modes=["text"], verify_with_pikafish=False)),
            ]
        with tempfile.TemporaryDirectory() as directory:
            for index, (writer, results) in enumerate(datasets):
                run_dir = Path(directory) / str(index)
                run_dir.mkdir()
                prediction = run_dir / "predictions.jsonl"
                prediction.write_text("checkpoint retained\n")
                writer(results, directory, str(index), write_predictions=False)
                self.assertEqual(prediction.read_text(), "checkpoint retained\n")
                saved = json.loads((run_dir / "results.json").read_text())
                self.assertEqual(saved["missing"], 1)


if __name__ == "__main__":
    unittest.main()

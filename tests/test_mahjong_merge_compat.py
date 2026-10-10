"""Public-entrypoint regressions for the Mahjong/runtime merge boundary."""
from dataclasses import asdict
import json
from pathlib import Path
import unittest

from minibench.agents import (
    CoTAgent, CriticRefineAgent, DirectAgent, PlanThenSolveAgent, TreeOfThoughtAgent,
)
from minibench.core.agent import ReasoningConfig
from minibench.datasets.mahjong_rule_variants.evaluation import (
    evaluate_mahjong_rule_variant_task, summarize_mahjong_rule_variants,
)
from minibench.datasets.mahjong_rule_variants.rules import RED_DRAGON_WILDCARD
from minibench.datasets.mahjong_solo.evaluation import (
    evaluate_mahjong_solo_task, summarize_mahjong_solo,
)
from minibench.factory.config import load_experiment_config
from minibench.scoring.adapters import score_record
from tests.test_mahjong_rule_variants import make_task
from tests.test_mahjong_solo import delayed_tsumo_task, tsumo_task
from tests.test_reasoning_agents import FakeClient


class FailingClient(FakeClient):
    def _next(self):
        value = super()._next()
        if isinstance(value, Exception):
            raise value
        return value


class MahjongMergeCompatibilityTests(unittest.TestCase):
    def families(self, task):
        variant = make_task(RED_DRAGON_WILDCARD, task.initial_hand,
                            task.wall[:task.max_draws])
        return (
            (task, evaluate_mahjong_solo_task, summarize_mahjong_solo),
            (variant, evaluate_mahjong_rule_variant_task, summarize_mahjong_rule_variants),
        )

    def test_all_current_mahjong_configs_validate_after_strict_agent_merge(self):
        paths = sorted(Path("config/experiments").glob("mahjong*.yaml"))
        self.assertEqual(len(paths), 5)
        for path in paths:
            with self.subTest(path=path):
                config = load_experiment_config(path)
                self.assertEqual(config["agent"]["prompt_version"], "v1")
                self.assertNotIn("samples", config["agent"])

    def test_all_architectures_keep_stage_outputs_in_both_evaluation_entrypoints(self):
        final = '{"action":"tsumo"}'
        proposal = json.dumps({"thought": "four melds and a pair", "state_summary": "win",
                               "terminal": True, "proposed_answer": {"action": "tsumo"}})
        evaluation = json.dumps({"evaluations": [{"node_id": "n0001", "score": 1,
                                 "valid": True, "terminal": True, "reason": "complete"}]})
        cases = (
            (DirectAgent, [final], "v1"),
            (CoTAgent, ["complete hand", final], "v1"),
            (PlanThenSolveAgent, ["inspect melds", "complete hand", final], "v1"),
            (CriticRefineAgent, ["complete hand", "valid shape", final], "v1"),
            (TreeOfThoughtAgent, [proposal, evaluation, final], "v2"),
        )
        for task, evaluate, _ in self.families(tsumo_task()):
            for mode in ("full-hand", "history-only"):
                for agent_type, responses, version in cases:
                    with self.subTest(task=task.id, mode=mode, agent=agent_type.name):
                        config = ReasoningConfig(prompt_version=version, max_depth=1,
                                                 branching_factor=1, beam_width=1)
                        client = FakeClient(responses)
                        agent = agent_type(client, config)
                        result = evaluate(task, agent, observation_mode=mode)
                        self.assertTrue(result.success)
                        self.assertEqual(result.metrics["llm_calls"], len(responses))
                        self.assertEqual(result.process_score, 1)
                        self.assertEqual(len(result.reasoning_traces), 1)
                        self.assertEqual([stage["output"] for stage in
                                          result.reasoning_traces[0]["stages"]], responses)
                        self.assertEqual(agent.config, config)
                        self.assertEqual(agent.runtime.trace_mode, "summary")
                        self.assertIsNone(agent.runtime.active_span_id)

    def test_invalid_final_is_an_action_retry_not_api_error_or_hidden_format_repair(self):
        for task, evaluate, summarize in self.families(tsumo_task()):
            for mode in ("full-hand", "history-only"):
                with self.subTest(task=task.id, mode=mode):
                    client = FakeClient(["complete", "not JSON", "complete", '{"action":"tsumo"}'])
                    agent = CoTAgent(client, ReasoningConfig(prompt_version="v1"))
                    result = evaluate(task, agent, observation_mode=mode)
                    summary = summarize([result])
                    self.assertTrue(result.success)
                    self.assertEqual(result.first_attempt_outputs, ["not JSON"])
                    self.assertEqual(result.evaluation_status, "completed")
                    self.assertEqual(summary["api_error_total"], 0)
                    self.assertEqual(summary["strict_first_attempt_success"], 0)
                    self.assertEqual(summary["retry_corrected_success"], 1)
                    self.assertEqual((result.correct_steps, result.total_steps), (1, 2))
                    self.assertEqual(result.process_score, .5)
                    self.assertEqual(len(client.calls), 4)
                    self.assertFalse(any(stage["is_format_repair"] for trace in
                                         result.reasoning_traces for stage in trace["stages"]))
                    if mode == "history-only":
                        self.assertEqual(result.conversation[1]["content"], "{}")

    def test_api_error_excludes_the_whole_episode_from_both_process_scores(self):
        for task, evaluate, summarize in self.families(delayed_tsumo_task()):
            for mode in ("full-hand", "history-only"):
                with self.subTest(task=task.id, mode=mode):
                    client = FailingClient(["discard 9s", '{"action":"discard","tile":"9s"}',
                                            RuntimeError("simulated transport timeout")])
                    agent = CoTAgent(client, ReasoningConfig(prompt_version="v1"))
                    result = evaluate(task, agent, observation_mode=mode)
                    summary = summarize([result])
                    self.assertIsNone(result.success)
                    self.assertIsNone(result.score)
                    self.assertIsNone(result.process_score)
                    self.assertIsNone(summary["average_process_score"])
                    self.assertIsNone(summary["shanten_optimal_discard_rate"])
                    self.assertEqual(summary["api_error_total"], 1)
                    self.assertEqual(summary["evaluated_total"], 0)
                    self.assertEqual(result.metrics["llm_calls"], 3)
                    offline = score_record("mahjong_solo", "H", asdict(result),
                                           asdict(task), mode="history-only")
                    self.assertIsNone(offline["y"])
                    self.assertIsNone(offline["p"])
                    self.assertIsNone(agent.runtime.active_span_id)

    def test_legacy_cot_history_keeps_current_turn_and_conversation_instructions(self):
        client = FakeClient(["discard 9s", '{"action":"discard","tile":"9s"}',
                             "complete hand", '{"action":"tsumo"}'])
        result = evaluate_mahjong_solo_task(delayed_tsumo_task(),
            CoTAgent(client, ReasoningConfig(prompt_version="v1")), observation_mode="history-only")
        self.assertTrue(result.success)
        self.assertIn("current turn", client.calls[2]["messages"][-1]["content"])
        self.assertIn("this conversation", client.calls[3]["messages"][-1]["content"])
        self.assertNotIn("Current hand", client.calls[2]["messages"][-1]["content"])

    def test_offline_scorer_cannot_overwrite_strict_winning_tile_failures(self):
        task = {"id": "strict", "goal": "winning_tiles", "hand": list(tsumo_task().initial_hand)}
        for payload in ({"winning_tiles": ["E", "E"]},
                        {"winning_tiles": ["E", 3]},
                        {"winning_tiles": ["E"], "notes": "memory"}):
            with self.subTest(payload=payload):
                scored = score_record("mahjong", "D", {"raw_output": json.dumps(payload)}, task, mode="text")
                self.assertEqual(scored["y"], 0)


if __name__ == "__main__":
    unittest.main()

"""Public-entrypoint regressions for the Mahjong/runtime merge boundary."""
from dataclasses import asdict
import json
from pathlib import Path
import unittest

from minibench.agents import (
    CoTAgent, CriticRefineAgent, DirectAgent, PlanThenSolveAgent, TreeOfThoughtAgent,
)
from minibench.core.agent import ReasoningConfig
from minibench.datasets.mahjong_solo.evaluation import (
    evaluate_mahjong_solo_task, summarize_mahjong_solo,
)
from minibench.factory.config import load_experiment_config
from minibench.scoring.adapters import score_record
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
        return (
            (task, evaluate_mahjong_solo_task, summarize_mahjong_solo),
        )

    def test_all_current_mahjong_configs_validate_after_strict_agent_merge(self):
        paths = sorted(Path("config/experiments").glob("mahjong*.yaml"))
        self.assertTrue({"mahjong.yaml", "mahjong_solo.yaml", "mahjong_history_only.yaml",
                         "mahjong_multimodal.yaml", "mahjong_rule_variants.yaml"}
                        .issubset({path.name for path in paths}))
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
                        self.assertNotIn("process_score", asdict(result))
                        self.assertEqual(len(result.reasoning_traces), 1)
                        self.assertEqual([stage["output"] for stage in
                                          result.reasoning_traces[0]["stages"]], responses)
                        self.assertEqual(agent.config, config)
                        self.assertEqual(agent.runtime.trace_mode, "summary")
                        self.assertIsNone(agent.runtime.active_span_id)

    def test_invalid_final_fails_once_without_api_error_or_hidden_format_repair(self):
        for task, evaluate, summarize in self.families(tsumo_task()):
            for mode in ("full-hand", "history-only"):
                with self.subTest(task=task.id, mode=mode):
                    client = FakeClient(["complete", "not JSON", "complete", '{"action":"tsumo"}'])
                    agent = CoTAgent(client, ReasoningConfig(prompt_version="v1"))
                    result = evaluate(task, agent, observation_mode=mode)
                    summary = summarize([result])
                    self.assertFalse(result.success)
                    self.assertEqual(result.first_attempt_outputs, ["not JSON"])
                    self.assertEqual(result.evaluation_status, "completed")
                    self.assertEqual(summary["api_error_total"], 0)
                    self.assertEqual(summary["strict_first_attempt_success"], 0)
                    self.assertEqual(summary["retry_corrected_success"], 0)
                    self.assertNotIn("process_score", asdict(result))
                    self.assertEqual(len(client.calls), 2)
                    self.assertFalse(any(stage["is_format_repair"] for trace in
                                         result.reasoning_traces for stage in trace["stages"]))
                    reply = next(m["content"] for m in result.conversation if m["role"] == "assistant")
                    self.assertIn("complete", reply)
                    self.assertNotIn("not JSON", reply)
                    self.assertTrue(reply.rstrip().endswith("{}"))

    def test_reasoning_architectures_carry_only_model_outputs_between_turns(self):
        discard = '{"action":"discard","tile":"9s"}'
        win = '{"action":"tsumo"}'
        proposals = [json.dumps({"thought": f"MODEL_MEMORY_{turn}", "state_summary": "tracked hand",
                    "terminal": True, "proposed_answer": json.loads(answer)})
                    for turn, answer in ((1, discard), (2, win))]
        evaluation = json.dumps({"evaluations": [{"node_id": "n0001", "score": 1,
            "valid": True, "terminal": True, "reason": "valid action"}]})
        cases = (
            (CoTAgent, ["MODEL_MEMORY_1", discard, "MODEL_MEMORY_2", win], 2, "v1"),
            (PlanThenSolveAgent, ["plan", "MODEL_MEMORY_1", discard,
                                 "plan", "MODEL_MEMORY_2", win], 3, "v1"),
            (CriticRefineAgent, ["MODEL_MEMORY_1", "checked", discard,
                                "MODEL_MEMORY_2", "checked", win], 3, "v1"),
            (TreeOfThoughtAgent, [proposals[0], evaluation, discard,
                                 proposals[1], evaluation, win], 3, "v2"),
        )
        for mode in ("full-hand", "history-only"):
            for agent_type, responses, calls_per_turn, version in cases:
                with self.subTest(mode=mode, agent=agent_type.name):
                    client = FakeClient(responses)
                    agent = agent_type(client, ReasoningConfig(prompt_version=version,
                        max_depth=1, branching_factor=1, beam_width=1))
                    result = evaluate_mahjong_solo_task(delayed_tsumo_task(), agent,
                                                       observation_mode=mode)
                    self.assertTrue(result.success)
                    messages = client.calls[calls_per_turn]["messages"]
                    self.assertEqual([m["role"] for m in messages], ["system", "assistant", "user"])
                    self.assertIn("MODEL_MEMORY_1", messages[1]["content"])
                    self.assertIn('"tile": "9s"', messages[1]["content"])
                    self.assertNotIn("Initial concealed hand", json.dumps(messages))

    def test_offline_scoring_distinguishes_old_hand_reports_from_new_action_only_logs(self):
        task = tsumo_task()
        reported = json.dumps({"current_hand": list(task.initial_hand) + ["E"], "action": "tsumo"})
        legacy = {"success": True, "observation_mode": "history-only",
                  "raw_outputs": [reported], "hand_reconstruction_steps": []}
        current = {"success": True, "observation_mode": "history-only",
                   "raw_outputs": [reported, '{"action":"tsumo"}']}
        single_attempt = {"success": False, "observation_mode": "history-only",
                          "raw_outputs": ["not JSON"], "max_action_attempts": 1}
        for record, outcome in ((legacy, 1.0), (current, 1.0), (single_attempt, 0.0)):
            with self.subTest(legacy="hand_reconstruction_steps" in record):
                scored = score_record("mahjong_solo", "H", record,
                                      json.loads(json.dumps(asdict(task))), mode="history-only")
                self.assertEqual(scored["y"], outcome)
                self.assertEqual(scored["p"], 0.0)

    def test_api_error_excludes_the_whole_episode_from_success_and_shanten_metrics(self):
        for task, evaluate, summarize in self.families(delayed_tsumo_task()):
            for mode in ("full-hand", "history-only"):
                with self.subTest(task=task.id, mode=mode):
                    client = FailingClient(["discard 9s", '{"action":"discard","tile":"9s"}',
                                            RuntimeError("simulated transport timeout")])
                    agent = CoTAgent(client, ReasoningConfig(prompt_version="v1"))
                    result = evaluate(task, agent, observation_mode=mode)
                    summary = summarize([result])
                    self.assertIsNone(result.success)
                    self.assertNotIn("score", asdict(result))
                    self.assertNotIn("process_score", asdict(result))
                    self.assertNotIn("average_process_score", summary)
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

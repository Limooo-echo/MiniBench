from copy import deepcopy
from dataclasses import asdict, replace
from io import BytesIO
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import URLError

from minibench.agents.passthrough import PassthroughAgent
from minibench.agents.direct import DirectAgent
from minibench.core.agent import ReasoningConfig
from minibench.datasets.zebra.dataset import zebra_task_from_dict
from minibench.datasets.zebra.evaluation import (
    evaluate_zebra_tasks,
    summarize_zebra,
    zebra_result_from_dict,
)
from minibench.datasets.zebra.state_validation import initial_candidate_state
from minibench.datasets.zebra.validation import ValidationError
from minibench.factory.providers import OpenAICompatibleClient


def history_task():
    return zebra_task_from_dict({
        "id": "history-pro-unit",
        "size": "2*2",
        "puzzle": (
            "There are 2 houses, numbered 1 to 2 from left to right, as seen from across the street. "
            "Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:\n"
            " - Each person has a unique name: `Alice`, `Bob`\n"
            " - Each person has a unique favorite drink: `tea`, `milk`"
        ),
        "solution": {
            "header": ["House", "Name", "Drink"],
            "rows": [["1", "Alice", "tea"], ["2", "Bob", "milk"]],
        },
        "capability": "history_memory",
        "clue_turns": [
            "The person who likes milk is in the second house.",
            "Alice is in the first house.",
        ],
        "tags": [],
    })


def final_output():
    return json.dumps({"solution": {
        "House 1": {"Name": "Alice", "Drink": "tea"},
        "House 2": {"Name": "Bob", "Drink": "milk"},
    }})


def completion_response(content, finish_reason="stop"):
    return BytesIO(json.dumps({
        "id": "offline-test-response",
        "model": "offline-test-model",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }).encode())


class ProposalAgent:
    def __init__(self, proposals):
        self.proposals = list(proposals)
        self.calls = []
        self.intermediate_count = 0

    def generate_messages_for_phase(self, messages, task, *, phase, **options):
        self.calls.append({"phase": phase, "messages": deepcopy(list(messages)), **options})
        if phase == "final":
            return final_output()
        proposal = self.proposals[self.intermediate_count % len(self.proposals)]
        self.intermediate_count += 1
        return proposal if isinstance(proposal, str) else json.dumps(proposal)


class ZebraHistoryProTests(unittest.TestCase):
    def setUp(self):
        self.task = history_task()
        self.initial = initial_candidate_state(self.task)
        self.safe = deepcopy(self.initial)
        self.safe["domains"]["Drink"] = {"tea": [1], "milk": [2]}
        self.wrong = deepcopy(self.safe)
        self.wrong["domains"]["Name"] = {"Alice": [2], "Bob": [1]}

    def run_mode(self, mode, proposals):
        agent = ProposalAgent(proposals)
        result = evaluate_zebra_tasks([self.task], agent, memory_modes=(mode,))[0]
        return result, agent

    def test_default_three_conditions_have_equal_model_rounds(self):
        agent = ProposalAgent([self.initial])
        results = evaluate_zebra_tasks([self.task], agent)
        self.assertEqual([r.memory_mode for r in results], [
            "deferred_reasoning", "incremental_state", "incremental_state_pro",
        ])
        self.assertEqual(len(agent.calls), 9)
        self.assertTrue(all(result.success for result in results))

    def test_safe_trajectory_has_identical_prompts_and_contexts_in_both_state_groups(self):
        plain, plain_agent = self.run_mode("incremental_state", [self.safe, self.initial])
        pro, pro_agent = self.run_mode("incremental_state_pro", [self.safe, self.initial])
        self.assertEqual(plain_agent.calls, pro_agent.calls)
        self.assertEqual([t["decision"] for t in plain.history_trace[:-1]], ["accept", "accept"])
        self.assertEqual([t["decision"] for t in pro.history_trace[:-1]], ["accept", "accept"])

    def test_schema_valid_but_unsupported_update_is_accepted_only_without_gate(self):
        plain, _ = self.run_mode("incremental_state", [self.wrong, self.initial])
        pro, pro_agent = self.run_mode("incremental_state_pro", [self.wrong, self.initial])
        self.assertEqual(plain.history_trace[0]["decision"], "accept")
        self.assertEqual(plain.history_trace[0]["validation"]["status"], "not_checked")
        self.assertEqual(plain.history_trace[0]["state_after"], self.wrong)
        self.assertEqual(pro.history_trace[0]["decision"], "reject")
        self.assertEqual(pro.history_trace[0]["validation"]["status"], "not_entailed")
        self.assertEqual(pro.history_trace[0]["state_after"], self.initial)
        self.assertNotIn(json.dumps(self.wrong, separators=(',', ':')), pro_agent.calls[1]["messages"][-1]["content"])
        self.assertEqual(pro.history_trace[1]["state_before"], self.initial)
        self.assertEqual(len(pro_agent.calls), 3)

    def test_wrong_last_update_never_enters_final_context_and_clues_survive(self):
        result, agent = self.run_mode("incremental_state_pro", [self.safe, self.wrong])
        self.assertEqual(result.history_trace[0]["decision"], "accept")
        self.assertEqual(result.history_trace[1]["decision"], "reject")
        self.assertEqual(result.history_trace[-1]["state_before"], self.safe)
        final_context = agent.calls[-1]["messages"][-1]["content"]
        self.assertIn(json.dumps(self.safe, separators=(',', ':')), final_context)
        self.assertNotIn(json.dumps(self.wrong, separators=(',', ':')), final_context)
        for clue in self.task.clue_turns:
            self.assertIn(clue, final_context)
        self.assertEqual(len(result.conversation), 3)
        self.assertEqual(list(result.conversation[:-1]), agent.calls[-1]["messages"])

    def test_each_request_contains_only_received_clues(self):
        _, agent = self.run_mode("incremental_state_pro", [self.initial])
        first_context = json.dumps(agent.calls[0]["messages"])
        second_context = json.dumps(agent.calls[1]["messages"])
        self.assertIn(self.task.clue_turns[0], first_context)
        self.assertNotIn(self.task.clue_turns[1], first_context)
        self.assertTrue(all(clue in second_context for clue in self.task.clue_turns))

    def test_malformed_proposal_is_rejected_identically_and_not_retried(self):
        for mode in ("incremental_state", "incremental_state_pro"):
            with self.subTest(mode=mode):
                result, agent = self.run_mode(mode, ['{"domains":', self.safe])
                first = result.history_trace[0]
                self.assertEqual(first["schema_status"], "invalid")
                self.assertEqual(first["decision"], "reject")
                self.assertEqual(first["state_after"], self.initial)
                self.assertEqual(first["validation"]["status"], "not_checked")
                self.assertEqual(result.history_trace[1]["state_before"], self.initial)
                self.assertEqual(len(agent.calls), 3)

    def test_unguarded_group_can_restore_wrongly_deleted_positions(self):
        result, _ = self.run_mode("incremental_state", [self.wrong, self.initial])
        self.assertEqual(result.history_trace[1]["state_before"], self.wrong)
        self.assertEqual(result.history_trace[1]["decision"], "accept")
        self.assertEqual(result.history_trace[1]["state_after"], self.initial)
        first_audit = result.history_trace[0]["elimination_audit"]
        self.assertTrue(first_audit["gold_excluded"])
        self.assertEqual(sum(item["is_gold"] for item in first_audit["removed_candidates"]), 2)
        self.assertFalse(result.history_trace[1]["elimination_audit"]["gold_excluded"])
        self.assertEqual(result.history_trace[1]["elimination_audit"]["removed_candidates"], [])

    def test_both_groups_audit_gold_without_exposing_it_to_model(self):
        for mode in ("incremental_state", "incremental_state_pro"):
            with self.subTest(mode=mode):
                result, agent = self.run_mode(mode, [self.wrong, self.safe])
                audit = result.history_trace[0]["elimination_audit"]
                self.assertTrue(audit["gold_excluded"])
                self.assertTrue(audit["is_legal"])
                self.assertEqual(len(audit["excluded_gold"]), 2)
                self.assertNotIn("excluded_gold", json.dumps(agent.calls))
                summary = summarize_zebra([result])["state_updates"][mode]
                self.assertEqual(summary["gold_exclusion_checked"], 2)
                self.assertEqual(summary["gold_excluded_proposals"], 1)

    def test_correct_guess_can_be_unentailed_without_excluding_gold(self):
        guess = deepcopy(self.safe)
        guess["domains"]["Name"] = {"Alice": [1], "Bob": [2]}
        result, _ = self.run_mode("incremental_state_pro", [guess])
        first = result.history_trace[0]
        self.assertEqual(first["validation"]["status"], "not_entailed")
        self.assertFalse(first["elimination_audit"]["gold_excluded"])
        self.assertEqual(first["decision"], "reject")

    def test_multiple_candidates_are_legal_but_two_confirmed_values_conflict(self):
        conflict = deepcopy(self.initial)
        conflict["domains"]["Name"] = {"Alice": [1], "Bob": [1]}
        result, _ = self.run_mode("incremental_state", [self.initial, conflict])
        self.assertTrue(result.history_trace[0]["elimination_audit"]["is_legal"])
        self.assertFalse(result.history_trace[1]["elimination_audit"]["is_legal"])
        self.assertEqual(result.history_trace[1]["decision"], "accept")

    def test_plain_gold_audit_does_not_require_solver(self):
        with patch("minibench.datasets.zebra.evaluation.PrefixStateValidator", side_effect=RuntimeError("missing z3")):
            result, _ = self.run_mode("incremental_state", [self.wrong, self.initial])
        self.assertEqual(result.history_trace[0]["validation"]["status"], "not_checked")
        self.assertTrue(result.history_trace[0]["elimination_audit"]["gold_excluded"])
        self.assertEqual(result.history_trace[0]["decision"], "accept")

    def test_unknown_verification_keeps_old_state_and_distinct_reason(self):
        unknown = {"status": "solver_unknown", "accepted": False, "reason": "timeout", "phase": "entailment"}
        with patch("minibench.datasets.zebra.evaluation.PrefixStateValidator") as validator:
            validator.return_value.validate.return_value = unknown
            result, agent = self.run_mode("incremental_state_pro", [self.safe])
        self.assertEqual(len(agent.calls), 3)
        for turn in result.history_trace[:-1]:
            self.assertEqual(turn["decision"], "reject")
            self.assertEqual(turn["validation"]["status"], "solver_unknown")
            self.assertEqual(turn["state_after"], self.initial)

    def test_invalid_prefix_preflight_stops_before_any_condition_calls_model(self):
        task = replace(self.task, clue_turns=(
            "Alice is in the first house.", "Alice is in the second house.",
        ))
        agent = ProposalAgent([self.initial])
        with self.assertRaisesRegex(ValidationError, "unsatisfiable"):
            evaluate_zebra_tasks([task], agent)
        self.assertEqual(agent.calls, [])

    def test_missing_solver_stops_before_even_deferred_model_calls(self):
        agent = ProposalAgent([self.initial])
        with patch("minibench.datasets.zebra.evaluation.PrefixStateValidator", side_effect=RuntimeError("missing z3")):
            with self.assertRaisesRegex(RuntimeError, "missing z3"):
                evaluate_zebra_tasks([self.task], agent)
        self.assertEqual(agent.calls, [])

    def test_history_trace_round_trips_and_legacy_records_are_readable(self):
        result, _ = self.run_mode("incremental_state_pro", [self.safe])
        payload = json.loads(json.dumps(asdict(result)))
        restored = zebra_result_from_dict(payload)
        self.assertEqual(restored.history_trace, result.history_trace)
        payload.pop("history_trace")
        payload.pop("is_legal")
        payload.pop("legality_reasons")
        legacy = zebra_result_from_dict(payload)
        self.assertEqual(legacy.history_trace, ())
        self.assertIsNone(legacy.is_legal)

    def test_summary_separates_rejections_and_uses_paired_tasks(self):
        plain, _ = self.run_mode("incremental_state", [self.wrong, self.safe])
        pro, _ = self.run_mode("incremental_state_pro", [self.wrong, self.safe])
        summary = summarize_zebra([plain, pro])
        self.assertEqual(summary["state_updates"]["incremental_state"]["accepted"], 2)
        self.assertEqual(summary["state_updates"]["incremental_state_pro"]["not_entailed"], 1)
        self.assertEqual(summary["state_updates"]["incremental_state_pro"]["rejected"], 1)
        self.assertEqual(summary["incremental_pro_paired"]["paired_tasks"], 1)
        self.assertEqual(summary["incremental_pro_paired"]["both_correct"], 1)
        self.assertEqual(summary["incremental_pro_paired"]["pro_minus_incremental_accuracy"], 0)

    def real_client(self):
        return OpenAICompatibleClient(
            model="offline-test-model", base_url="https://offline.invalid/v1",
            api_key_env="ZEBRA_TEST_API_KEY", max_retries=0,
        )

    def test_truncated_response_is_a_rejected_proposal_without_retry(self):
        client = self.real_client()
        responses = [
            completion_response('{"domains":', "length"),
            completion_response(json.dumps(self.safe)),
            completion_response(final_output()),
        ]
        with patch.dict(os.environ, {"ZEBRA_TEST_API_KEY": "offline-test-only"}), patch(
            "minibench.factory.providers.urllib.request.urlopen", side_effect=responses,
        ) as request:
            result = evaluate_zebra_tasks(
                [self.task], PassthroughAgent(client), memory_modes=("incremental_state_pro",),
            )[0]
        self.assertEqual(request.call_count, 3)
        self.assertTrue(result.success)
        self.assertEqual(result.history_trace[0]["decision"], "reject")
        self.assertEqual(result.history_trace[0]["completion_traces"][0]["finish_reason"], "length")
        self.assertFalse(client.allow_truncated_content)

    def test_empty_response_is_recorded_as_rejected_state_or_final_no_answer(self):
        for empty_content in ("", None, [], [{"type": "text", "text": ""}]):
            with self.subTest(content=empty_content):
                client = self.real_client()
                responses = [
                    completion_response(empty_content), completion_response(json.dumps(self.safe)),
                    completion_response(empty_content),
                ]
                with patch.dict(os.environ, {"ZEBRA_TEST_API_KEY": "offline-test-only"}), patch(
                    "minibench.factory.providers.urllib.request.urlopen", side_effect=responses,
                ) as request:
                    result = evaluate_zebra_tasks(
                        [self.task], PassthroughAgent(client), memory_modes=("incremental_state",),
                    )[0]
                self.assertEqual(request.call_count, 3)
                self.assertTrue(result.no_answer)
                self.assertEqual(result.history_trace[0]["decision"], "reject")
                self.assertEqual(result.history_trace[0]["raw_proposal"], "")
                self.assertEqual(result.history_trace[0]["completion_traces"][0]["content"], empty_content)
                self.assertEqual(result.history_trace[1]["decision"], "accept")
                self.assertEqual(result.history_trace[-1]["raw_output"], "")
                self.assertFalse(client.allow_truncated_content)

    def test_transport_failure_still_stops_and_restores_client_policy(self):
        client = self.real_client()
        with patch.dict(os.environ, {"ZEBRA_TEST_API_KEY": "offline-test-only"}), patch(
            "minibench.factory.providers.urllib.request.urlopen", side_effect=URLError("offline failure"),
        ) as request:
            with self.assertRaisesRegex(RuntimeError, "offline failure"):
                evaluate_zebra_tasks([self.task], PassthroughAgent(client), memory_modes=("incremental_state",))
        self.assertEqual(request.call_count, 1)
        self.assertFalse(client.allow_truncated_content)

    def test_direct_strict_json_truncated_final_is_scored_without_format_retry(self):
        client = self.real_client()
        invalid_final = '{"solution":'
        responses = [
            completion_response(json.dumps(self.safe)),
            completion_response(json.dumps(self.safe)),
            completion_response(invalid_final, "length"),
        ]
        agent = DirectAgent(client, ReasoningConfig(max_format_repairs=0))
        with patch.dict(os.environ, {"ZEBRA_TEST_API_KEY": "offline-test-only"}), patch(
            "minibench.factory.providers.urllib.request.urlopen", side_effect=responses,
        ) as request:
            result = evaluate_zebra_tasks(
                [self.task], agent, memory_modes=("incremental_state_pro",),
            )[0]
        self.assertEqual(request.call_count, 3)
        self.assertTrue(result.no_answer)
        self.assertFalse(result.success)
        self.assertEqual(result.raw_output, invalid_final)
        self.assertEqual(result.history_trace[-1]["raw_output"], invalid_final)
        self.assertEqual(result.history_trace[-1]["completion_traces"][0]["finish_reason"], "length")
        self.assertFalse(client.allow_truncated_content)


if __name__ == "__main__":
    unittest.main()

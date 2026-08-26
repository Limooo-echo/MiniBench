from copy import deepcopy
import json
import unittest

from minibench.agents import (
    BestOfNAgent,
    CoTAgent,
    CriticRefineAgent,
    DirectAgent,
    LeastToMostAgent,
    PlanThenSolveAgent,
    SelfConsistencyAgent,
    TreeOfThoughtAgent,
)
from minibench.core.agent import ReasoningConfig
from minibench.core.multimodal import ImageAttachment
from minibench.core.runtime import StrictJSONObjectError


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def _next(self):
        if not self.responses:
            raise AssertionError("fake client ran out of responses")
        return self.responses.pop(0)

    def complete(self, prompt, **options):
        self.calls.append(
            {
                "kind": "prompt",
                "prompt": prompt,
                "options": options,
            }
        )
        return self._next()

    def complete_messages(self, messages, **options):
        self.calls.append(
            {
                "kind": "messages",
                "messages": deepcopy(list(messages)),
                "options": options,
            }
        )
        return self._next()


def image():
    return ImageAttachment(
        data=b"\x89PNG\r\n\x1a\n",
        mime_type="image/png",
    )


def history():
    return [
        {"role": "system", "content": "Domain rules"},
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Current question"},
    ]


def reasoning_config(**overrides):
    values = {
        "reasoning_temperature": 0.8,
        "final_temperature": 0.1,
        "max_reasoning_tokens": 128,
        "final_max_tokens": 32,
    }
    values.update(overrides)
    return ReasoningConfig(**values)


def one_level_tot_responses(answer=1):
    return [
        json.dumps(
            {
                "thought": "finish the task",
                "state_summary": "ready",
                "terminal": True,
                "proposed_answer": {"answer": answer},
            }
        ),
        json.dumps(
            {
                "evaluations": [
                    {
                        "node_id": "n0001",
                        "score": 0.9,
                        "valid": True,
                        "terminal": True,
                        "reason": "complete",
                    }
                ]
            }
        ),
        json.dumps({"answer": answer}),
    ]


class ExistingReasoningTopologyTests(unittest.TestCase):
    def test_existing_agent_call_topologies_temperatures_and_tokens(self):
        cases = (
            (
                DirectAgent,
                ['{"answer":"direct"}'],
                1,
                [0.1],
                [32],
            ),
            (
                CoTAgent,
                ['reasoning {"answer":"cot"}', '{"answer":"cot"}'],
                2,
                [0.8, 0.1],
                [128, 32],
            ),
            (
                PlanThenSolveAgent,
                [
                    '{"steps":["inspect"],"risks":[]}',
                    '{"solution":"done"}',
                    '{"answer":"plan"}',
                ],
                3,
                [0.8, 0.8, 0.1],
                [128, 128, 32],
            ),
            (
                CriticRefineAgent,
                [
                    '{"answer":"draft"}',
                    '{"is_valid":false,"issues":["x"],"corrections":["y"]}',
                    '{"answer":"refined"}',
                ],
                3,
                [0.8, 0.1, 0.1],
                [128, 128, 32],
            ),
        )
        attachment = image()
        for agent_type, responses, expected_calls, temperatures, token_limits in cases:
            with self.subTest(agent=agent_type.__name__):
                client = FakeClient(responses)
                agent = agent_type(client, reasoning_config())

                output = agent.generate_multimodal(
                    "Question",
                    object(),
                    images=[attachment],
                )

                self.assertEqual(len(client.calls), expected_calls)
                self.assertTrue(output.startswith('{"answer"'))
                self.assertEqual(
                    [call["options"]["temperature"] for call in client.calls],
                    temperatures,
                )
                self.assertEqual(
                    [call["options"]["max_tokens"] for call in client.calls],
                    token_limits,
                )
                self.assertTrue(
                    all(
                        call["options"]["images"] == [attachment]
                        for call in client.calls
                    )
                )
                self.assertEqual(len(agent.last_run.stages), expected_calls)

    def test_existing_agents_preserve_history_and_transform_only_current_turn(self):
        cases = (
            (DirectAgent, ['{"answer":"direct"}'], 1),
            (
                CoTAgent,
                ['reasoning {"answer":"cot"}', '{"answer":"cot"}'],
                2,
            ),
            (
                PlanThenSolveAgent,
                [
                    '{"steps":["x"],"risks":[]}',
                    '{"solution":"s"}',
                    '{"answer":"plan"}',
                ],
                3,
            ),
            (
                CriticRefineAgent,
                [
                    '{"answer":"draft"}',
                    '{"is_valid":true,"issues":[],"corrections":[]}',
                    '{"answer":"refined"}',
                ],
                3,
            ),
        )
        for agent_type, responses, expected_calls in cases:
            with self.subTest(agent=agent_type.__name__):
                messages = history()
                original = deepcopy(messages)
                client = FakeClient(responses)
                agent = agent_type(client, reasoning_config())

                output = agent.generate_messages(
                    messages,
                    object(),
                    temperature=0.2,
                    max_tokens=44,
                    json_mode=False,
                )

                self.assertTrue(output.startswith('{"answer"'))
                self.assertEqual(messages, original)
                self.assertEqual(len(client.calls), expected_calls)
                self.assertTrue(
                    all(call["kind"] == "messages" for call in client.calls)
                )
                for call in client.calls:
                    prepared = call["messages"]
                    self.assertEqual(prepared[1]["content"], "Earlier question")
                    self.assertNotEqual(prepared[-1]["content"], "Current question")
                final_options = client.calls[-1]["options"]
                self.assertEqual(final_options["temperature"], 0.2)
                self.assertEqual(final_options["max_tokens"], 44)
                self.assertFalse(final_options["json_mode"])

    def test_intermediate_phase_is_one_raw_history_call_for_every_architecture(self):
        agent_cases = (
            (DirectAgent, reasoning_config()),
            (CoTAgent, reasoning_config()),
            (SelfConsistencyAgent, reasoning_config(samples=2)),
            (BestOfNAgent, reasoning_config(samples=2)),
            (
                TreeOfThoughtAgent,
                reasoning_config(max_depth=1, branching_factor=1, beam_width=1),
            ),
            (PlanThenSolveAgent, reasoning_config()),
            (CriticRefineAgent, reasoning_config()),
            (LeastToMostAgent, reasoning_config()),
        )
        for agent_type, config in agent_cases:
            with self.subTest(agent=agent_type.__name__):
                messages = history()
                original = deepcopy(messages)
                client = FakeClient(["intermediate"])
                agent = agent_type(client, config)

                output = agent.generate_messages_for_phase(
                    messages,
                    object(),
                    phase="intermediate",
                    temperature=0.3,
                    max_tokens=77,
                    json_mode=False,
                )

                self.assertEqual(output, "intermediate")
                self.assertEqual(messages, original)
                self.assertEqual(len(client.calls), 1)
                self.assertEqual(client.calls[0]["messages"], original)
                self.assertEqual(client.calls[0]["options"]["temperature"], 0.3)
                self.assertEqual(client.calls[0]["options"]["max_tokens"], 77)
                self.assertFalse(client.calls[0]["options"]["json_mode"])


class SelfConsistencyTests(unittest.TestCase):
    def test_programmatic_vote_canonicalizes_key_order_and_whitespace(self):
        client = FakeClient(
            [
                'path A\n{"b":2, "a":1}',
                'path B\n{ "a": 1, "b": 2 }',
                'path C\n{"answer":"other"}',
            ]
        )
        agent = SelfConsistencyAgent(client, reasoning_config(samples=3))

        output = agent.generate("Question", object())

        self.assertEqual(json.loads(output), {"a": 1, "b": 2})
        self.assertEqual(len(client.calls), 3)
        self.assertFalse(agent.last_run.metadata["tie"])
        self.assertAlmostEqual(agent.last_run.metadata["consensus"], 2 / 3)

    def test_tie_is_stable_and_invalid_candidates_are_ignored(self):
        client = FakeClient(
            [
                'first {"answer":"A"}',
                "not json",
                'third {"answer":"B"}',
            ]
        )
        agent = SelfConsistencyAgent(client, reasoning_config(samples=3))

        output = agent.generate("Question", object())

        self.assertEqual(json.loads(output), {"answer": "A"})
        self.assertTrue(agent.last_run.metadata["tie"])
        self.assertEqual(agent.last_run.metadata["valid_samples"], 2)
        self.assertEqual(agent.last_run.metadata["selected_sample"], 1)
        self.assertEqual(len(client.calls), 3)

    def test_all_invalid_candidates_get_exactly_one_low_temperature_repair(self):
        client = FakeClient(["bad A", "bad B", '{"answer":"repaired"}'])
        agent = SelfConsistencyAgent(client, reasoning_config(samples=2))

        output = agent.generate("Question", object())

        self.assertEqual(json.loads(output), {"answer": "repaired"})
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(client.calls[-1]["options"]["temperature"], 0.0)
        self.assertTrue(client.calls[-1]["options"]["json_mode"])

    def test_all_invalid_without_repairs_fails_after_n_calls(self):
        client = FakeClient(["bad A", "bad B"])
        config = reasoning_config(samples=2, max_format_repairs=0)
        agent = SelfConsistencyAgent(client, config)

        with self.assertRaisesRegex(StrictJSONObjectError, "all self-consistency"):
            agent.generate("Question", object())

        self.assertEqual(len(client.calls), 2)


class BestOfNTests(unittest.TestCase):
    def test_seeded_judge_selects_id_without_rewriting_candidate(self):
        judgement = json.dumps(
            {
                "selected_id": "candidate-2",
                "scores": [
                    {
                        "candidate_id": "candidate-1",
                        "score": 0.2,
                        "reason": "weak",
                    },
                    {
                        "candidate_id": "candidate-2",
                        "score": 0.9,
                        "reason": "best",
                    },
                    {
                        "candidate_id": "candidate-3",
                        "score": 0.4,
                        "reason": "partial",
                    },
                ],
            }
        )
        client = FakeClient(
            [
                'path 1 {"answer":1}',
                'path 2 {"answer":2}',
                'path 3 {"answer":3}',
                judgement,
            ]
        )
        agent = BestOfNAgent(client, reasoning_config(samples=3, selection_seed=42))

        output = agent.generate("Question", object())

        self.assertEqual(json.loads(output), {"answer": 2})
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(
            agent.last_run.metadata["displayed_order"],
            ["candidate-2", "candidate-1", "candidate-3"],
        )
        judge_prompt = client.calls[-1]["prompt"]
        self.assertIn('"selected_id"', judge_prompt)
        self.assertIn("never rewrite", judge_prompt.lower())

    def test_selected_candidate_format_gets_one_repair(self):
        judgement = json.dumps(
            {
                "selected_id": "candidate-1",
                "scores": [
                    {
                        "candidate_id": "candidate-1",
                        "score": 1.0,
                        "reason": "best",
                    },
                    {
                        "candidate_id": "candidate-2",
                        "score": 0.0,
                        "reason": "weak",
                    },
                ],
            }
        )
        client = FakeClient(
            [
                "answer one without JSON",
                'path {"answer":2}',
                judgement,
                '{"answer":1}',
            ]
        )
        agent = BestOfNAgent(client, reasoning_config(samples=2))

        output = agent.generate("Question", object())

        self.assertEqual(json.loads(output), {"answer": 1})
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(client.calls[-1]["options"]["temperature"], 0.0)


class TreeOfThoughtTests(unittest.TestCase):
    def test_two_level_beam_search_tracks_parents_prunes_and_stops_terminal(self):
        responses = [
            '{"thought":"a","state_summary":"a","terminal":false,"proposed_answer":null}',
            '{"thought":"b","state_summary":"b","terminal":false,"proposed_answer":null}',
            json.dumps(
                {
                    "evaluations": [
                        {
                            "node_id": "n0001",
                            "score": 0.2,
                            "valid": True,
                            "terminal": False,
                            "reason": "weak",
                        },
                        {
                            "node_id": "n0002",
                            "score": 0.9,
                            "valid": True,
                            "terminal": False,
                            "reason": "strong",
                        },
                    ]
                }
            ),
            '{"thought":"b1","state_summary":"done","terminal":true,"proposed_answer":{"answer":1}}',
            '{"thought":"b2","state_summary":"open","terminal":false,"proposed_answer":null}',
            json.dumps(
                {
                    "evaluations": [
                        {
                            "node_id": "n0003",
                            "score": 0.8,
                            "valid": True,
                            "terminal": True,
                            "reason": "complete",
                        },
                        {
                            "node_id": "n0004",
                            "score": 0.7,
                            "valid": True,
                            "terminal": False,
                            "reason": "unfinished",
                        },
                    ]
                }
            ),
            '{"answer":1}',
        ]
        client = FakeClient(responses)
        config = reasoning_config(
            max_depth=3,
            branching_factor=2,
            beam_width=1,
        )
        agent = TreeOfThoughtAgent(client, config)

        output = agent.generate("Question", object())

        self.assertEqual(json.loads(output), {"answer": 1})
        self.assertEqual(len(client.calls), 7)
        metadata = agent.last_run.metadata
        self.assertEqual(metadata["selected_node_id"], "n0003")
        self.assertEqual(metadata["stop_reason"], "all_beam_nodes_terminal")
        nodes = {node["node_id"]: node for node in metadata["nodes"]}
        self.assertEqual(nodes["n0003"]["parent_id"], "n0002")
        self.assertEqual(nodes["n0004"]["parent_id"], "n0002")
        self.assertNotIn(
            "n0001", {nodes["n0003"]["parent_id"], nodes["n0004"]["parent_id"]}
        )
        proposal_stages = [
            stage
            for stage in agent.last_run.stages
            if stage.stage_name.startswith("tot.propose")
        ]
        self.assertEqual(proposal_stages[-1].parent_id, "n0002")

    def test_call_budget_reserves_the_finalizer(self):
        client = FakeClient(['{"answer":"root"}'])
        config = reasoning_config(
            max_depth=3,
            branching_factor=3,
            beam_width=2,
            max_llm_calls=1,
        )
        agent = TreeOfThoughtAgent(client, config)

        output = agent.generate("Question", object())

        self.assertEqual(json.loads(output), {"answer": "root"})
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(agent.last_run.metadata["selected_node_id"], "n0000")
        self.assertEqual(agent.last_run.metadata["stop_reason"], "llm_call_budget")

    def test_history_and_images_use_the_same_tree_search(self):
        attachment = image()
        image_client = FakeClient(one_level_tot_responses())
        config = reasoning_config(max_depth=1, branching_factor=1, beam_width=1)
        image_agent = TreeOfThoughtAgent(image_client, config)

        image_agent.generate_multimodal("Question", object(), images=[attachment])

        self.assertEqual(len(image_client.calls), 3)
        self.assertTrue(
            all(
                call["options"]["images"] == [attachment] for call in image_client.calls
            )
        )

        messages = history()
        original = deepcopy(messages)
        message_client = FakeClient(one_level_tot_responses())
        message_agent = TreeOfThoughtAgent(message_client, config)

        output = message_agent.generate_messages(messages, object())

        self.assertEqual(json.loads(output), {"answer": 1})
        self.assertEqual(messages, original)
        self.assertTrue(
            all(call["kind"] == "messages" for call in message_client.calls)
        )


class LeastToMostTests(unittest.TestCase):
    def test_subproblem_limit_and_sequential_accumulation(self):
        client = FakeClient(
            [
                '{"subproblems":["first","second","third"]}',
                '{"subproblem":"first","result":"r1","reason":"ok"}',
                '{"subproblem":"second","result":"r2","reason":"uses r1"}',
                '{"answer":"done"}',
            ]
        )
        config = reasoning_config(max_subproblems=2)
        agent = LeastToMostAgent(client, config)

        output = agent.generate("Question", object())

        self.assertEqual(json.loads(output), {"answer": "done"})
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(agent.last_run.metadata["subproblems"], ["first", "second"])
        self.assertIn("r1", client.calls[2]["prompt"])

    def test_empty_decomposition_falls_back_to_original_problem(self):
        client = FakeClient(
            [
                '{"subproblems":[]}',
                '{"subproblem":"Question","result":"r","reason":"ok"}',
                '{"answer":"done"}',
            ]
        )
        agent = LeastToMostAgent(client, reasoning_config())

        output = agent.generate("Question", object())

        self.assertEqual(json.loads(output), {"answer": "done"})
        self.assertEqual(agent.last_run.metadata["subproblems"], ["Question"])

    def test_multimodal_and_history_paths_preserve_context(self):
        attachment = image()
        responses = [
            '{"subproblems":["one"]}',
            '{"subproblem":"one","result":"r","reason":"ok"}',
            '{"answer":"done"}',
        ]
        image_client = FakeClient(list(responses))
        image_agent = LeastToMostAgent(image_client, reasoning_config())
        image_agent.generate_multimodal("Question", object(), images=[attachment])
        self.assertTrue(
            all(
                call["options"]["images"] == [attachment] for call in image_client.calls
            )
        )

        messages = history()
        original = deepcopy(messages)
        message_client = FakeClient(list(responses))
        message_agent = LeastToMostAgent(message_client, reasoning_config())
        output = message_agent.generate_messages(messages, object())
        self.assertEqual(json.loads(output), {"answer": "done"})
        self.assertEqual(messages, original)
        self.assertTrue(
            all(call["kind"] == "messages" for call in message_client.calls)
        )


if __name__ == "__main__":
    unittest.main()

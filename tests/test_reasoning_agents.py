from copy import deepcopy
import unittest

from minibench.agents import (
    CoTAgent,
    CriticRefineAgent,
    DirectAgent,
    PlanThenSolveAgent,
    ReasoningConfig,
    SelfConsistencyAgent,
    TreeOfThoughtAgent,
)
from minibench.core.metrics import finish_task_metrics, start_task_metrics


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(
        self,
        prompt,
        *,
        system_prompt=None,
        temperature=None,
        max_tokens=None,
        json_mode=None,
        images=(),
    ):
        self.calls.append(
            {
                "kind": "prompt",
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
                "images": images,
            }
        )
        if not self.responses:
            raise AssertionError("fake client ran out of responses")
        return self.responses.pop(0)

    def complete_messages(
        self,
        messages,
        *,
        system_prompt=None,
        temperature=None,
        max_tokens=None,
        json_mode=None,
        images=(),
    ):
        self.calls.append(
            {
                "kind": "messages",
                "messages": deepcopy(list(messages)),
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
                "images": images,
            }
        )
        if not self.responses:
            raise AssertionError("fake client ran out of responses")
        return self.responses.pop(0)


class MetricsClient(FakeClient):
    def __init__(self, responses):
        super().__init__(responses)
        self.model_elapsed_seconds = 0.0
        self.llm_calls = 0
        self.usage_missing_calls = 0
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def complete(self, *args, **kwargs):
        output = super().complete(*args, **kwargs)
        self.llm_calls += 1
        self.model_elapsed_seconds += 0.25
        self.token_usage["prompt_tokens"] += 7
        self.token_usage["completion_tokens"] += 3
        self.token_usage["total_tokens"] += 10
        return output

    def metrics_snapshot(self):
        return {
            "model_elapsed_seconds": self.model_elapsed_seconds,
            "llm_calls": self.llm_calls,
            "usage_missing_calls": self.usage_missing_calls,
            "token_usage": dict(self.token_usage),
        }


def sample_task():
    return object()


def message_agent_cases(config):
    return [
        (DirectAgent, config, 1),
        (CoTAgent, config, 2),
        (SelfConsistencyAgent, config, config.samples + 1),
        (TreeOfThoughtAgent, config, config.samples + 1),
        (CriticRefineAgent, config, 3),
        (PlanThenSolveAgent, config, 3),
    ]


class ReasoningAgentTests(unittest.TestCase):
    def test_every_reasoning_stage_receives_images(self):
        from minibench.core.multimodal import ImageAttachment

        image = ImageAttachment(data=b"\x89PNG\r\n\x1a\n", mime_type="image/png")
        cases = [
            (DirectAgent, ReasoningConfig(), 1),
            (CoTAgent, ReasoningConfig(), 2),
            (SelfConsistencyAgent, ReasoningConfig(samples=2), 3),
            (TreeOfThoughtAgent, ReasoningConfig(samples=2), 3),
            (CriticRefineAgent, ReasoningConfig(), 3),
            (PlanThenSolveAgent, ReasoningConfig(), 3),
        ]
        for agent_type, config, call_count in cases:
            with self.subTest(agent=agent_type.__name__):
                client = FakeClient(["draft"] * (call_count - 1) + ['{"ok":true}'])
                agent = agent_type(client, config)
                agent.generate_multimodal("Question", object(), images=[image])
                self.assertEqual(len(client.calls), call_count)
                self.assertTrue(
                    all(call["images"] == [image] for call in client.calls)
                )

    def test_cot_finalizes_non_choice_json_schema(self):
        client = FakeClient(["The pair wait is E.", '{"winning_tiles":["E"]}'])
        agent = CoTAgent(client, ReasoningConfig())

        output = agent.generate(
            'Return {"winning_tiles":["E"]} for this Mahjong task.',
            object(),
        )

        self.assertEqual(output, '{"winning_tiles":["E"]}')
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(client.calls[-1]["json_mode"])
        self.assertIn("schema requested", client.calls[-1]["prompt"])

    def test_self_consistency_uses_generic_judge(self):
        client = FakeClient(
            ["candidate C", "candidate B", "candidate C", '{"answer":"C"}']
        )
        agent = SelfConsistencyAgent(client, ReasoningConfig(samples=3))

        output = agent.generate("Question prompt", sample_task())

        self.assertEqual(output, '{"answer":"C"}')
        self.assertEqual(len(client.calls), 4)
        self.assertTrue(client.calls[-1]["json_mode"])
        self.assertIn("Candidate solutions:", client.calls[-1]["prompt"])

    def test_tree_of_thought_generates_candidates_and_judges(self):
        client = FakeClient(
            [
                "Candidate says A",
                "Candidate says C",
                "Candidate says B",
                '{"answer":"C"}',
            ]
        )
        agent = TreeOfThoughtAgent(client, ReasoningConfig(samples=3))

        output = agent.generate("Question prompt", sample_task())

        self.assertEqual(output, '{"answer":"C"}')
        self.assertEqual(len(client.calls), 4)
        self.assertTrue(client.calls[-1]["json_mode"])
        self.assertIn("Candidate solutions:", client.calls[-1]["prompt"])

    def test_critic_refine_returns_refined_answer(self):
        client = FakeClient(
            [
                '{"answer":"A"}',
                "The draft ignores the clue; C is better.",
                '{"answer":"C"}',
            ]
        )
        agent = CriticRefineAgent(client, ReasoningConfig())

        output = agent.generate("Question prompt", sample_task())

        self.assertEqual(output, '{"answer":"C"}')
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(client.calls[-1]["json_mode"])
        self.assertIn("Critique:", client.calls[-1]["prompt"])

    def test_generate_messages_runs_full_architecture_without_mutating_history(self):
        config = ReasoningConfig(
            samples=2,
            reasoning_temperature=0.81,
            final_temperature=0.17,
            max_reasoning_tokens=123,
            final_max_tokens=45,
        )
        messages = [
            {"role": "system", "content": "Base system"},
            {"role": "user", "content": "Remember clue one."},
            {"role": "assistant", "content": "Clue noted."},
            {"role": "user", "content": "Return the final answer."},
        ]
        original_messages = deepcopy(messages)

        for agent_type, agent_config, call_count in message_agent_cases(config):
            with self.subTest(agent=agent_type.__name__):
                client = FakeClient(["hidden"] * (call_count - 1) + ["final"])
                agent = agent_type(client, agent_config)

                output = agent.generate_messages(
                    messages,
                    object(),
                    temperature=0.23,
                    max_tokens=77,
                    json_mode=False,
                )

                self.assertEqual(output, "final")
                self.assertEqual(len(client.calls), call_count)
                self.assertEqual(messages, original_messages)
                self.assertTrue(all(call["kind"] == "messages" for call in client.calls))
                for call in client.calls:
                    self.assertIsNone(call["system_prompt"])
                    merged_system = call["messages"][0]["content"]
                    self.assertTrue(merged_system.startswith("Base system\n\n"))
                    self.assertEqual(merged_system.count("Base system"), 1)
                    self.assertEqual(call["messages"][1:3], messages[1:3])
                    self.assertNotEqual(
                        call["messages"][-1]["content"],
                        messages[-1]["content"],
                    )
                self.assertEqual(client.calls[-1]["temperature"], 0.23)
                self.assertEqual(client.calls[-1]["max_tokens"], 77)
                self.assertFalse(client.calls[-1]["json_mode"])

    def test_final_phase_runs_full_architecture_with_default_visible_options(self):
        config = ReasoningConfig(
            samples=2,
            reasoning_temperature=0.81,
            final_temperature=0.17,
            max_reasoning_tokens=123,
            final_max_tokens=45,
        )
        messages = [
            {"role": "user", "content": "Remember clue one."},
            {"role": "assistant", "content": "Clue noted."},
            {"role": "user", "content": "Return the final answer."},
        ]
        original_messages = deepcopy(messages)

        for agent_type, agent_config, call_count in message_agent_cases(config):
            with self.subTest(agent=agent_type.__name__):
                client = FakeClient(["hidden"] * (call_count - 1) + ["final"])
                agent = agent_type(client, agent_config)

                output = agent.generate_messages_for_phase(
                    messages,
                    object(),
                    phase="final",
                )

                self.assertEqual(output, "final")
                self.assertEqual(len(client.calls), call_count)
                self.assertEqual(messages, original_messages)
                for call in client.calls:
                    self.assertEqual(call["kind"], "messages")
                    self.assertIsNotNone(call["system_prompt"])
                    self.assertEqual(call["messages"][:2], messages[:2])
                    self.assertFalse(
                        any(
                            message["role"] == "system"
                            for message in call["messages"]
                        )
                    )
                final_call = client.calls[-1]
                self.assertEqual(final_call["temperature"], 0.17)
                self.assertEqual(final_call["max_tokens"], 45)
                self.assertTrue(final_call["json_mode"])

                hidden_calls = client.calls[:-1]
                if agent_type is CriticRefineAgent:
                    self.assertEqual(
                        [call["temperature"] for call in hidden_calls],
                        [0.81, 0.17],
                    )
                else:
                    self.assertTrue(
                        all(call["temperature"] == 0.81 for call in hidden_calls)
                    )
                self.assertTrue(
                    all(call["max_tokens"] == 123 for call in hidden_calls)
                )
                self.assertTrue(
                    all(call["json_mode"] is False for call in hidden_calls)
                )

    def test_intermediate_phase_is_one_raw_history_call(self):
        config = ReasoningConfig(samples=2)
        messages = [
            {"role": "system", "content": "Base system"},
            {"role": "user", "content": "New clue."},
        ]
        original_messages = deepcopy(messages)

        for agent_type, agent_config, _ in message_agent_cases(config):
            with self.subTest(agent=agent_type.__name__):
                client = FakeClient(["acknowledged"])
                agent = agent_type(client, agent_config)

                output = agent.generate_messages_for_phase(
                    messages,
                    object(),
                    phase="intermediate",
                    temperature=0.33,
                    max_tokens=21,
                    json_mode=False,
                )

                self.assertEqual(output, "acknowledged")
                self.assertEqual(len(client.calls), 1)
                call = client.calls[0]
                self.assertEqual(call["kind"], "messages")
                self.assertEqual(call["messages"], original_messages)
                self.assertIsNone(call["system_prompt"])
                self.assertEqual(call["temperature"], 0.33)
                self.assertEqual(call["max_tokens"], 21)
                self.assertFalse(call["json_mode"])
                self.assertEqual(messages, original_messages)

    def test_invalid_message_phase_fails_before_client_call(self):
        config = ReasoningConfig(samples=2)
        messages = [{"role": "user", "content": "Question"}]

        for agent_type, agent_config, _ in message_agent_cases(config):
            with self.subTest(agent=agent_type.__name__):
                client = FakeClient([])
                agent = agent_type(client, agent_config)

                with self.assertRaisesRegex(ValueError, "Unsupported message phase"):
                    agent.generate_messages_for_phase(
                        messages,
                        object(),
                        phase="unexpected",
                    )

                self.assertEqual(client.calls, [])

    def test_candidate_message_branches_start_from_same_history(self):
        config = ReasoningConfig(samples=2)
        messages = [
            {"role": "system", "content": "Base system"},
            {"role": "user", "content": "Question"},
        ]

        consistency_client = FakeClient(["candidate one", "candidate two", "final"])
        SelfConsistencyAgent(consistency_client, config).generate_messages(
            messages,
            object(),
        )
        self.assertEqual(
            consistency_client.calls[0]["messages"],
            consistency_client.calls[1]["messages"],
        )
        self.assertNotIn(
            "candidate one",
            consistency_client.calls[1]["messages"][-1]["content"],
        )

        tree_client = FakeClient(["branch one", "branch two", "final"])
        TreeOfThoughtAgent(tree_client, config).generate_messages(messages, object())
        self.assertEqual(
            tree_client.calls[0]["messages"][:-1],
            tree_client.calls[1]["messages"][:-1],
        )
        self.assertNotIn(
            "branch one",
            tree_client.calls[1]["messages"][-1]["content"],
        )

    def test_reasoning_agent_metrics_include_nested_client_calls(self):
        client = MetricsClient(["Reasoning says C.", '{"answer":"C"}'])
        agent = CoTAgent(client, ReasoningConfig())
        metrics_start = start_task_metrics(agent)

        output = agent.generate("Question prompt", sample_task())
        metrics = finish_task_metrics(agent, metrics_start)

        self.assertEqual(output, '{"answer":"C"}')
        self.assertEqual(metrics["llm_calls"], 2)
        self.assertEqual(metrics["model_elapsed_seconds"], 0.5)
        self.assertEqual(metrics["token_usage"]["prompt_tokens"], 14)
        self.assertEqual(metrics["token_usage"]["completion_tokens"], 6)
        self.assertEqual(metrics["token_usage"]["total_tokens"], 20)
        self.assertTrue(metrics["usage_available"])


if __name__ == "__main__":
    unittest.main()

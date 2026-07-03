import unittest

from minibench.agents import (
    CoTAgent,
    CriticRefineAgent,
    ReasoningConfig,
    SelfConsistencyAgent,
    TreeOfThoughtAgent,
)
from minibench.datasets.multiple_choice.dataset import Task


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
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            }
        )
        if not self.responses:
            raise AssertionError("fake client ran out of responses")
        return self.responses.pop(0)


def sample_task():
    return Task(
        id="unit-task",
        question="Pick C.",
        options={"A": "A", "B": "B", "C": "C", "D": "D"},
        correct_option="C",
        answer_extractors=(),
        prompt_constraints=(),
        tags=(),
    )


class ReasoningAgentTests(unittest.TestCase):
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

    def test_self_consistency_uses_majority_vote(self):
        client = FakeClient(["answer: C", "answer: B", "answer: C"])
        agent = SelfConsistencyAgent(client, ReasoningConfig(samples=3))

        output = agent.generate("Question prompt", sample_task())

        self.assertEqual(output, '{"answer": "C"}')
        self.assertEqual(len(client.calls), 3)

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


if __name__ == "__main__":
    unittest.main()

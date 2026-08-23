import unittest

from minibench.agents import (
    CoTAgent,
    CriticRefineAgent,
    DirectAgent,
    PlanThenSolveAgent,
    SelfConsistencyAgent,
    TreeOfThoughtAgent,
)
from minibench.factory.agents import make_agent, make_agent_from_config
from minibench.factory.providers import OpenAICompatibleAgent


class AgentFactoryTests(unittest.TestCase):
    def test_creates_openai_compatible_agent(self):
        agent = make_agent("openai-compatible", provider="deepseek")

        self.assertIsInstance(agent, OpenAICompatibleAgent)

    def test_creates_reasoning_agents(self):
        cases = {
            "direct": DirectAgent,
            "cot": CoTAgent,
            "self-consistency": SelfConsistencyAgent,
            "tot": TreeOfThoughtAgent,
            "plan-then-solve": PlanThenSolveAgent,
            "critic-refine": CriticRefineAgent,
        }

        for name, expected_type in cases.items():
            with self.subTest(agent=name):
                agent = make_agent(name, provider="deepseek")

                self.assertIsInstance(agent, expected_type)
                self.assertTrue(callable(getattr(agent, "generate_messages", None)))
                self.assertTrue(
                    callable(getattr(agent, "generate_messages_for_phase", None))
                )

    def test_wires_retry_config_to_openai_compatible_agent(self):
        agent = make_agent_from_config(
            {"name": "openai-compatible"},
            {
                "name": "deepseek",
                "max_retries": 3,
                "retry_initial_backoff_seconds": 1.5,
                "retry_max_backoff_seconds": 12.0,
            },
        )

        self.assertIsInstance(agent, OpenAICompatibleAgent)
        self.assertEqual(agent.max_retries, 3)
        self.assertEqual(agent.retry_initial_backoff_seconds, 1.5)
        self.assertEqual(agent.retry_max_backoff_seconds, 12.0)

    def test_wires_retry_config_to_reasoning_agent_client(self):
        agent = make_agent_from_config(
            {"name": "cot"},
            {
                "name": "deepseek",
                "max_retries": 2,
                "retry_initial_backoff_seconds": 2.0,
                "retry_max_backoff_seconds": 8.0,
            },
        )

        self.assertIsInstance(agent, CoTAgent)
        self.assertEqual(agent.client.max_retries, 2)
        self.assertEqual(agent.client.retry_initial_backoff_seconds, 2.0)
        self.assertEqual(agent.client.retry_max_backoff_seconds, 8.0)


if __name__ == "__main__":
    unittest.main()

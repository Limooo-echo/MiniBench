import unittest

from minibench.agents import (
    CoTAgent,
    CriticRefineAgent,
    DirectAgent,
    PlanThenSolveAgent,
    SelfConsistencyAgent,
    TreeOfThoughtAgent,
)
from minibench.factory.agents import make_agent
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


if __name__ == "__main__":
    unittest.main()

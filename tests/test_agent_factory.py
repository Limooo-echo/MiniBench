import unittest

from minibench.agents import (
    CoTAgent,
    CriticRefineAgent,
    DirectAgent,
    OpenAICompatibleAgent,
    PlanThenSolveAgent,
    SelfConsistencyAgent,
    TreeOfThoughtAgent,
    make_agent,
)


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

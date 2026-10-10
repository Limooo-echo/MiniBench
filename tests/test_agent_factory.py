import unittest

from minibench.agents import (
    BestOfNAgent,
    CoTAgent,
    CriticRefineAgent,
    DirectAgent,
    LeastToMostAgent,
    PassthroughAgent,
    PlanThenSolveAgent,
    SelfConsistencyAgent,
    TreeOfThoughtAgent,
)
from minibench.factory.agents import AGENT_NAMES, make_agent, make_agent_from_config
from minibench.factory.providers import OpenAICompatibleClient


EXPECTED_AGENT_NAMES = (
    "passthrough",
    "direct",
    "cot",
    "self-consistency",
    "best-of-n",
    "tot",
    "plan-then-solve",
    "critic-refine",
    "least-to-most",
)


class AgentFactoryTests(unittest.TestCase):
    def test_creates_passthrough_agent_with_openai_compatible_client(self):
        agent = make_agent("passthrough", provider="deepseek")

        self.assertIsInstance(agent, PassthroughAgent)
        self.assertIsInstance(agent.client, OpenAICompatibleClient)

    def test_openai_compatible_name_is_a_deprecated_passthrough_alias(self):
        with self.assertWarnsRegex(FutureWarning, "passthrough"):
            agent = make_agent("openai-compatible", provider="deepseek")

        self.assertIsInstance(agent, PassthroughAgent)
        self.assertIsInstance(agent.client, OpenAICompatibleClient)

    def test_public_agent_names_are_exact_and_exclude_deprecated_alias(self):
        self.assertEqual(AGENT_NAMES, EXPECTED_AGENT_NAMES)
        self.assertNotIn("openai-compatible", AGENT_NAMES)

    def test_factory_defaults_to_direct(self):
        agent = make_agent(provider="deepseek")

        self.assertIsInstance(agent, DirectAgent)

    def test_creates_all_reasoning_agents(self):
        cases = {
            "direct": DirectAgent,
            "cot": CoTAgent,
            "self-consistency": SelfConsistencyAgent,
            "best-of-n": BestOfNAgent,
            "tot": TreeOfThoughtAgent,
            "plan-then-solve": PlanThenSolveAgent,
            "critic-refine": CriticRefineAgent,
            "least-to-most": LeastToMostAgent,
        }

        for name, expected_type in cases.items():
            with self.subTest(agent=name):
                agent = make_agent(name, provider="deepseek")

                self.assertIsInstance(agent, expected_type)
                self.assertIsInstance(agent.client, OpenAICompatibleClient)
                self.assertTrue(callable(getattr(agent, "generate_messages", None)))
                self.assertTrue(
                    callable(getattr(agent, "generate_messages_for_phase", None))
                )

    def test_wires_retry_config_to_passthrough_client(self):
        agent = make_agent_from_config(
            {"name": "passthrough"},
            {
                "name": "deepseek",
                "max_retries": 3,
                "retry_initial_backoff_seconds": 1.5,
                "retry_max_backoff_seconds": 12.0,
            },
        )

        self.assertIsInstance(agent, PassthroughAgent)
        self.assertEqual(agent.client.max_retries, 3)
        self.assertEqual(agent.client.retry_initial_backoff_seconds, 1.5)
        self.assertEqual(agent.client.retry_max_backoff_seconds, 12.0)

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

    def test_wires_runtime_and_tot_search_config(self):
        agent = make_agent_from_config(
            {
                "name": "tot",
                "prompt_version": "v2",
                "trace": "full",
                "max_tokens": 96,
                "max_llm_calls": 25,
                "max_total_tokens": 4000,
                "max_format_repairs": 0,
                "reasoning_temperature": 0.9,
                "final_temperature": 0.1,
                "max_reasoning_tokens": 768,
                "max_depth": 4,
                "branching_factor": 5,
                "beam_width": 3,
                "max_search_nodes": 48,
            },
            {"name": "deepseek"},
        )

        self.assertIsInstance(agent, TreeOfThoughtAgent)
        self.assertEqual(agent.config.prompt_version, "v2")
        self.assertEqual(agent.config.trace, "full")
        self.assertEqual(agent.config.final_max_tokens, 96)
        self.assertEqual(agent.config.max_llm_calls, 25)
        self.assertEqual(agent.config.max_total_tokens, 4000)
        self.assertEqual(agent.config.max_format_repairs, 0)
        self.assertEqual(agent.config.reasoning_temperature, 0.9)
        self.assertEqual(agent.config.final_temperature, 0.1)
        self.assertEqual(agent.config.max_reasoning_tokens, 768)
        self.assertEqual(agent.config.max_depth, 4)
        self.assertEqual(agent.config.branching_factor, 5)
        self.assertEqual(agent.config.beam_width, 3)
        self.assertEqual(agent.config.max_search_nodes, 48)
        self.assertEqual(agent.runtime.budget.max_calls, 25)
        self.assertEqual(agent.runtime.budget.max_total_tokens, 4000)

    def test_wires_best_of_n_and_least_to_most_fields(self):
        best = make_agent_from_config(
            {
                "name": "best-of-n",
                "samples": 7,
                "selection_seed": 123,
            },
            {"name": "deepseek"},
        )
        least = make_agent_from_config(
            {
                "name": "least-to-most",
                "max_subproblems": 6,
            },
            {"name": "deepseek"},
        )

        self.assertIsInstance(best, BestOfNAgent)
        self.assertEqual(best.config.samples, 7)
        self.assertEqual(best.config.selection_seed, 123)
        self.assertIsInstance(least, LeastToMostAgent)
        self.assertEqual(least.config.max_subproblems, 6)

    def test_v2_only_agents_reject_v1(self):
        for name in ("best-of-n", "tot", "least-to-most"):
            with self.subTest(agent=name):
                with self.assertRaisesRegex(ValueError, "v2"):
                    make_agent(
                        name,
                        provider="deepseek",
                        prompt_version="v1",
                    )

    def test_sampled_agents_require_two_samples_and_nonzero_temperature(self):
        for name in ("self-consistency", "best-of-n"):
            with self.subTest(agent=name, invalid="samples"):
                with self.assertRaisesRegex(ValueError, "samples >= 2"):
                    make_agent(name, provider="deepseek", samples=1)
            with self.subTest(agent=name, invalid="temperature"):
                with self.assertRaisesRegex(
                    ValueError, "non-zero reasoning_temperature"
                ):
                    make_agent(
                        name,
                        provider="deepseek",
                        reasoning_temperature=0.0,
                    )


if __name__ == "__main__":
    unittest.main()

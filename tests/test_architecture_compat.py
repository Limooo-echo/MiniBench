import unittest


class ArchitectureCompatTests(unittest.TestCase):
    def test_factory_agent_path_exports_make_agent(self):
        from minibench.agents.passthrough import PassthroughAgent
        from minibench.factory.agents import make_agent

        self.assertIsInstance(
            make_agent("passthrough", provider="deepseek"),
            PassthroughAgent,
        )

    def test_deprecated_python_agent_is_still_constructible(self):
        from minibench.factory.providers import (
            OpenAICompatibleAgent,
            OpenAICompatibleClient,
        )

        with self.assertWarnsRegex(FutureWarning, "deprecated"):
            agent = OpenAICompatibleAgent(
                model="unit-model",
                base_url="https://example.invalid/v1",
                api_key_env="UNIT_KEY",
            )

        self.assertIsInstance(agent, OpenAICompatibleClient)
        self.assertTrue(callable(agent.generate))

    def test_new_agent_architectures_are_publicly_exported(self):
        from minibench.agents import (
            BestOfNAgent,
            LeastToMostAgent,
            PassthroughAgent,
            TreeOfThoughtAgent,
        )

        self.assertEqual(BestOfNAgent.name, "best-of-n")
        self.assertEqual(LeastToMostAgent.name, "least-to-most")
        self.assertEqual(PassthroughAgent.name, "passthrough")
        self.assertEqual(TreeOfThoughtAgent.name, "tot")

    def test_new_dataset_paths_reexport_existing_loaders(self):
        from minibench.datasets.one_stroke.dataset import load_one_stroke_tasks
        from minibench.datasets.xiangqi.engines.pikafish import PikafishEngine

        self.assertGreater(
            len(load_one_stroke_tasks("data/one_stroke/tasks.jsonl")),
            0,
        )
        self.assertEqual(PikafishEngine.__name__, "PikafishEngine")


if __name__ == "__main__":
    unittest.main()

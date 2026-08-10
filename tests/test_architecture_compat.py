import unittest


class ArchitectureCompatTests(unittest.TestCase):
    def test_factory_agent_path_exports_make_agent(self):
        from minibench.factory.agents import make_agent
        from minibench.factory.providers import OpenAICompatibleAgent

        self.assertIsInstance(
            make_agent("openai-compatible", provider="deepseek"),
            OpenAICompatibleAgent,
        )

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

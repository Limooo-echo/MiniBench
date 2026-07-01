import unittest


class ArchitectureCompatTests(unittest.TestCase):
    def test_factory_agent_path_exports_make_agent(self):
        from minibench.factory.agents import make_agent
        from minibench.agents.simple import OracleAgent

        self.assertIsInstance(make_agent("oracle"), OracleAgent)

    def test_new_dataset_paths_reexport_existing_loaders(self):
        from minibench.datasets.multiple_choice.dataset import load_tasks
        from minibench.datasets.xiangqi.engines.pikafish import PikafishEngine

        self.assertGreater(len(load_tasks("data/multiple_choice/tasks.jsonl")), 0)
        self.assertEqual(PikafishEngine.__name__, "PikafishEngine")

if __name__ == "__main__":
    unittest.main()

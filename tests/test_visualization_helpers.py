"""Visualization utilities must not leak fonts or leave figures open."""
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt


class VisualizationHelperTests(unittest.TestCase):
    def test_importing_helpers_does_not_change_global_plot_parameters(self):
        before = dict(plt.rcParams)
        for name in ("plot_results", "mahjong_vis", "onestroke_vis"):
            module = importlib.import_module(f"minibench.visualization.{name}")
            importlib.reload(module)
        self.assertEqual(dict(plt.rcParams), before)

    def test_accuracy_plot_accepts_current_agents_and_closes_figure(self):
        from minibench.visualization.plot_results import AGENT_DISPLAY_NAMES, plot_accuracy_comparison
        from minibench.factory.agents import AGENT_NAMES

        self.assertTrue(set(AGENT_NAMES).issubset(AGENT_DISPLAY_NAMES))
        self.assertNotIn("openai-compatible", AGENT_DISPLAY_NAMES)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("cot", "best-of-n", "least-to-most"):
                run = root / "runs" / f"test-{name}"
                run.mkdir(parents=True)
                (run / "results.json").write_text(json.dumps({"success_rate": 0.5}), encoding="utf-8")
            before = dict(plt.rcParams)
            figures = plt.get_fignums()
            previous = Path.cwd()
            try:
                os.chdir(root)
                plot_accuracy_comparison(root / "runs", "comparison.png")
            finally:
                os.chdir(previous)
            self.assertTrue((root / "comparison.png").is_file())
            self.assertEqual(plt.get_fignums(), figures)
            self.assertEqual(dict(plt.rcParams), before)

    def test_task_plots_keep_settings_and_accept_filename_only_outputs(self):
        from minibench.visualization.mahjong_vis import draw_mahjong_hand
        from minibench.visualization.onestroke_vis import draw_one_stroke_task
        from minibench.visualization.plot_results import plot_task_results

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            before = dict(plt.rcParams)
            figures = plt.get_fignums()
            previous = Path.cwd()
            try:
                os.chdir(root)
                draw_mahjong_hand({"id": "hand", "hand": ["1m", "2p", "3s", "E"]}, "hand.png")
                draw_one_stroke_task({"id": "graph", "vertices": [1, 2], "edges": [[1, 2]], "start": 1, "end": 2}, "graph.png")
                plot_task_results(runs, "tasks.png")
            finally:
                os.chdir(previous)
            for filename in ("hand.png", "graph.png", "tasks.png"):
                self.assertTrue((root / filename).is_file())
            self.assertEqual(dict(plt.rcParams), before)
            self.assertEqual(plt.get_fignums(), figures)


if __name__ == "__main__":
    unittest.main()

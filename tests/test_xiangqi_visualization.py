"""Current-schema board/GIF rendering and missing-aware diagnostic charts."""
from math import isnan
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import matplotlib

from minibench.datasets.xiangqi.multimodal import render_board_png
from minibench.datasets.xiangqi.schema import FAMILY_PATHS, fen_to_board, load_records
from minibench.datasets.xiangqi.variants.board import VariantBoard
from minibench.visualization.history_vis import (
    DIFFS, _agent_mode_values, visualize_history_comparison, visualize_history_results,
)
from minibench.visualization.mate_in_one_vis import (
    visualize_mate_in_one_comparison, visualize_mate_in_one_results,
)
from minibench.visualization.xiangqi_metrics import metric_mean
from minibench.visualization.xiangqi_vis import (
    draw_xiangqi_board, render_xiangqi_trajectory, visualize_xiangqi_by_id,
)


class XiangqiVisualizationTests(unittest.TestCase):
    def test_board_and_trajectory_share_current_renderer(self):
        record = load_records(FAMILY_PATHS["xiangqi-mate-in-one"])[0]
        board, _ = fen_to_board(record["fen"])
        moved = VariantBoard(board, [])
        moved.apply(moved.legal_moves(1)[0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = draw_xiangqi_board(record, root / "task.png")
            self.assertEqual(destination.read_bytes(), render_board_png(board, "chinese-piece-image"))
            self.assertTrue(visualize_xiangqi_by_id(record["id"], FAMILY_PATHS[record["family"]], root))
            self.assertFalse(visualize_xiangqi_by_id("not-a-task", FAMILY_PATHS[record["family"]], root))
            gif = render_xiangqi_trajectory(record["id"], [board, moved.board], root)
            with Image.open(gif) as replay:
                self.assertEqual(replay.n_frames, 2)
            self.assertEqual((gif.parent / "step_00.png").read_bytes(), destination.read_bytes())

    def test_old_board_only_record_is_not_a_second_rendering_format(self):
        with self.assertRaises(ValueError):
            draw_xiangqi_board({"id": "old", "board": [[0] * 9 for _ in range(10)]}, "unused.png")

    def test_missing_evidence_is_nan_not_false_or_zero(self):
        results = [SimpleNamespace(status="ok", success=True),
                   SimpleNamespace(status="invalid", success=False),
                   SimpleNamespace(status="error", success=None)]
        self.assertEqual(metric_mean(results, "success"), 0.5)
        self.assertTrue(isnan(metric_mean(results[-1:], "success")))
        self.assertEqual(DIFFS, ("short", "medium", "long"))
        _, values = _agent_mode_values({"agent": {"full-state": results[-1:]}}, "success")
        self.assertTrue(all(isnan(value) for group in values.values() for value in group))

    def test_current_diagnostic_charts_handle_missing_results(self):
        result = SimpleNamespace(
            status="error", task_id="unit", difficulty="easy", history_mode="full-state",
            normalized_score=None, legality_score=0.0, correctness_score=None,
            quality_score=None, cp_loss=None, success=None, legality_rate=None, avg_cp_loss=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            style_before = dict(matplotlib.rcParams)
            visualize_mate_in_one_results([result], root / "d3")
            self.assertEqual(len(list((root / "d3/visualizations").glob("*.png"))), 6)
            visualize_mate_in_one_comparison({"agent": [result]}, root / "d3-comparison")
            visualize_mate_in_one_comparison({}, root / "d3-empty-comparison")
            result.difficulty = "medium"
            visualize_history_results([result], root / "h2")
            visualize_history_comparison({"agent": {"full-state": [result]}}, root / "comparison")
            self.assertEqual(len(list(root.rglob("*.png"))), 12)
            self.assertEqual(dict(matplotlib.rcParams), style_before)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.datasets.xiangqi.multimodal import (
    board_to_compact,
    evaluate_m2_tasks,
    render_board,
    render_board_png,
    summarize_m2,
)


class FirstMoveAgent:
    def __init__(self):
        self.text_calls = 0
        self.image_calls = 0

    def generate(self, prompt, task):
        self.text_calls += 1
        return "1"

    def generate_multimodal(self, prompt, task, *, images):
        self.image_calls += 1
        self.last_images = images
        return "1"


def sample_m2_task():
    return {
        "id": "m2-unit",
        "board": [
            [0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -2, 0, 0, 0, 0],
            [0, 0, 0, 0, -4, 0, 0, 8, 0],
            [-12, 0, 10, 0, 0, 0, 0, 0, -12],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 6],
            [12, 0, 0, 0, 0, 0, 0, 0, 12],
            [4, 0, 0, 0, 0, 0, 8, 0, 4],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 2, 0, 0, 0],
        ],
    }


class XiangqiMultimodalTests(unittest.TestCase):
    def test_renderer_supports_bytes_and_legacy_base64(self):
        board = sample_m2_task()["board"]
        for mode in ("img_cn", "img_ab"):
            png = render_board_png(board, mode)
            self.assertTrue(png.startswith(b"\x89PNG"))
            self.assertEqual(base64.b64decode(render_board(board, mode)), png)
        compact = board_to_compact(board)
        self.assertEqual(len(compact.splitlines()), 10)

    @patch("minibench.datasets.xiangqi.multimodal.score_moves", return_value=[])
    def test_three_modes_use_shared_agent_and_write_step_images(self, _score_moves):
        agent = FirstMoveAgent()
        with tempfile.TemporaryDirectory() as directory:
            results = evaluate_m2_tasks(
                [sample_m2_task()],
                agent,
                modes=("text", "img_cn", "img_ab"),
                max_steps=1,
                step_dir=directory,
            )
            written = list(Path(directory).rglob("*.png"))
        self.assertEqual(len(results), 3)
        self.assertEqual(agent.text_calls, 1)
        self.assertEqual(agent.image_calls, 2)
        self.assertEqual(len(written), 2)
        self.assertTrue(all("mode" in result and "steps" in result for result in results))
        summary = summarize_m2(results)
        self.assertEqual(set(summary["by_input_mode"]), {"text", "img_cn", "img_ab"})
        self.assertEqual(summary["visual_gap"]["img_cn"]["paired_total"], 1)
        self.assertIn("metrics", summary)

    def test_script_wrapper_has_no_private_http_stack(self):
        source = Path("scripts/m2/eval.py").read_text(encoding="utf-8")
        self.assertNotIn("urllib.request", source)
        self.assertIn("evaluate_m2_tasks", source)
        self.assertIn("M2_API_KEY", source)
        self.assertIn("M2_BASE_URL", source)
        self.assertIn("M2_MODEL", source)
        self.assertIn("M2_THINKING", source)
        self.assertIn("M2_OPP_DEPTH", source)


if __name__ == "__main__":
    unittest.main()

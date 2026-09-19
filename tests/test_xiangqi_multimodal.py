from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

from minibench.datasets.xiangqi.multimodal import (
    _build_multimodal_prompt,
    _extract_uci,
    board_to_compact,
    evaluate_xiangqi_multimodal_tasks,
    render_board,
    render_board_png,
    summarize_xiangqi_multimodal,
)
from minibench.datasets.xiangqi.variants.board import VariantBoard


class FirstMoveAgent:
    def __init__(self):
        self.text_calls = 0
        self.image_calls = 0

    def _first_legal_uci(self, task):
        return VariantBoard(task["board"], []).legal_moves(1)[0].to_uci()

    def generate(self, prompt, task):
        self.text_calls += 1
        return json.dumps({"move": self._first_legal_uci(task)})

    def generate_multimodal(self, prompt, task, *, images):
        self.image_calls += 1
        self.last_images = images
        return json.dumps({"move": self._first_legal_uci(task)})


def sample_multimodal_task():
    return {
        "id": "xiangqi-multimodal-unit",
        # R a8-d8 mates: N b7 protects d8 and N d7 controls e9.
        "board": [
            [0, 0, 0, -1, 0, 0, 0, 0, 0],
            [8, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 6, 0, 7, 0, 0, 0, 0, 0],
            [-12, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ],
    }


class XiangqiMultimodalTests(unittest.TestCase):
    def test_prompt_requires_free_uci_and_no_candidate_list(self):
        board = VariantBoard(sample_multimodal_task()["board"], [])
        prompt = _build_multimodal_prompt(board, "text", "")
        self.assertIn('{"move": "<uci_move>"}', prompt)
        self.assertIn("rank 0 is the bottom row", prompt)
        self.assertIn("    a b c d e f g h i", prompt)
        self.assertIn("Black pieces: k@d9", prompt)
        self.assertIn("Red pieces:", prompt)
        self.assertNotIn("Candidate moves", prompt)
        self.assertEqual(_extract_uci('{"move": "H5H9"}'), "h5h9")
        self.assertIsNone(_extract_uci('{"action": 3}'))

    def test_renderer_supports_bytes_and_legacy_base64(self):
        board = sample_multimodal_task()["board"]
        for mode in ("chinese-piece-image", "latin-piece-image"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                png = render_board_png(board, mode)
            self.assertFalse(
                any("missing from font" in str(item.message) for item in caught)
            )
            self.assertTrue(png.startswith(b"\x89PNG"))
            self.assertEqual(base64.b64decode(render_board(board, mode)), png)
        compact = board_to_compact(board)
        self.assertEqual(len(compact.splitlines()), 10)
        self.assertEqual(compact.splitlines()[0], "...k.....")
        self.assertIn("R", compact)
        self.assertIn("p", compact)

    @patch("minibench.datasets.xiangqi.multimodal.score_moves", return_value=[])
    def test_three_modes_use_shared_agent_and_write_step_images(self, _score_moves):
        agent = FirstMoveAgent()
        with tempfile.TemporaryDirectory() as directory:
            results = evaluate_xiangqi_multimodal_tasks(
                [sample_multimodal_task()],
                agent,
                modes=("text", "chinese-piece-image", "latin-piece-image"),
                max_steps=1,
                verify_with_pikafish=False,
                step_dir=directory,
            )
            written = list(Path(directory).rglob("*.png"))
        self.assertEqual(len(results), 3)
        self.assertEqual(agent.text_calls, 1)
        self.assertEqual(agent.image_calls, 2)
        self.assertEqual(len(written), 2)
        self.assertTrue(all("mode" in result and "steps" in result for result in results))
        summary = summarize_xiangqi_multimodal(results)
        self.assertEqual(
            set(summary["by_input_mode"]),
            {"text", "chinese-piece-image", "latin-piece-image"},
        )
        self.assertEqual(
            summary["visual_gap"]["chinese-piece-image"]["paired_total"], 1
        )
        self.assertIn("metrics", summary)

    def test_thin_wrapper_delegates_to_unified_cli(self):
        source = Path("scripts/run_task.py").read_text(encoding="utf-8")
        self.assertIn("minibench.cli", source)
        self.assertNotIn("urllib.request", source)


if __name__ == "__main__":
    unittest.main()

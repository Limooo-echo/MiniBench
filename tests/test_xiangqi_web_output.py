"""Console output from the manual web bridge stays concise and task-focused."""
from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "xiangqi_web_test",
    Path(__file__).resolve().parents[1] / "scripts/xiangqi_web_test.py",
)
web = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(web)


class XiangqiWebOutputTests(unittest.TestCase):
    def capture(self, callback, *args):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            callback(*args)
        return stream.getvalue().strip()

    def test_h2_item_reports_only_shortest_window_and_move_trace(self):
        record = {
            "task_id": "h2-example",
            "history_mode": "full-state",
            "success": False,
            # Archived fields must not re-enable removed console metrics.
            "extended_mate_success": True,
            "mate_within_prescribed_steps": False,
            "mate_within_shortest_steps": False,
            "mate_within_shortest_plus_two_steps": True,
            "legality_rate": 1.0,
            "avg_cp_loss": 12.5,
            "steps": [
                {"actor": "agent", "uci": "a0a1"},
                {"actor": "pikafish", "uci": "a9a8"},
            ],
        }
        output = self.capture(web._print_item_result, "h2", record)
        self.assertIn("mate_shortest=no legal=100.0%", output)
        self.assertNotIn("plus2", output)
        self.assertIn("cp_loss=12.5", output)
        self.assertNotIn("first_optimal", output)
        self.assertIn("moves=agent:a0a1,pikafish:a9a8", output)

    def test_one_ply_items_do_not_print_cp_loss(self):
        record = {"task_id": "d3-example", "success": True, "is_legal": True, "agent_uci": "a0a1"}
        output = self.capture(web._print_item_result, "d3", record)
        self.assertIn("mate=yes legal=100.0%", output)
        self.assertNotIn("cp_loss", output)

    def test_c2_reports_shortest_window_and_cpe_without_extension(self):
        record = {"task_id": "c2-example", "ruleset": "standard",
                  "mate_within_shortest_steps": True,
                  "mate_within_shortest_plus_two_steps": True,
                  "legality_rate": 1.0, "avg_variant_loss": 8.0,
                  "steps": [{"actor": "agent", "uci": "a0a1"},
                            {"actor": "depth-opponent", "uci": "a9a8"}]}
        output = self.capture(web._print_item_result, "c2", record)
        self.assertIn("mate_shortest=yes legal=100.0% variant_loss=8.0", output)
        self.assertIn("moves=agent:a0a1,depth-opponent:a9a8", output)
        self.assertNotIn("plus2", output)
        summary = self.capture(web._print_family_summary, "c2", {
            "mate_within_shortest_rate": .5, "mate_within_shortest_plus_two_rate": .8,
            "legality_rate": .9, "avg_variant_loss": 8.0,
        })
        self.assertEqual(summary, "SUMMARY C2 mate_shortest=50.0% legal=90.0% variant_loss=8.0")

    def test_h2_summary_is_compact(self):
        summary = {
            "mate_rate": 0.3,
            "mate_within_prescribed_rate": 0.2,
            "mate_within_shortest_rate": 0.2,
            "mate_within_shortest_plus_two_rate": 0.3,
            "legality_rate": 0.8,
            "avg_cp_loss": 321.0,
        }
        output = self.capture(web._print_family_summary, "h2", summary)
        self.assertEqual(
            output,
            "SUMMARY H2 mate_shortest=20.0% "
            "legal=80.0% cp_loss=321.0",
        )


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import sys
import unittest

from minibench.datasets.mahjong.api import score_closed_hand
from minibench.datasets.mahjong_riichi.ai import ExternalMahjongAI
from minibench.datasets.mahjong_riichi.dataset import load_mahjong_riichi_tasks
from minibench.datasets.mahjong_riichi.evaluation import (
    evaluate_mahjong_riichi_tasks,
    extract_riichi_action,
    legal_call_options,
    legal_closed_kan_tiles,
    riichi_discards,
    score_riichi_seats,
    summarize_mahjong_riichi,
)
from minibench.datasets.mahjong_riichi.prompting import build_mahjong_riichi_prompt


class SimpleRiichiAgent:
    def generate(self, prompt, task):
        if "Choose whether to call the latest discard" in prompt:
            return json.dumps({"action": "pass"})
        hand_line = next(
            line for line in prompt.splitlines()
            if line.startswith("Your current concealed hand")
        )
        hand = hand_line.split(": ", 1)[1].split()
        riichi_line = next(
            line for line in prompt.splitlines()
            if line.startswith("Riichi legal discards: ")
        )
        riichi_discards_text = riichi_line.split(": ", 1)[1]
        if riichi_discards_text != "(none)":
            discard = riichi_discards_text.split()[0]
            return json.dumps({"action": "riichi", "discard": discard})
        return json.dumps({"action": "discard", "tile": hand[0]})


class MahjongRiichiTests(unittest.TestCase):
    def test_loads_riichi_tasks(self):
        tasks = load_mahjong_riichi_tasks()

        data_path = Path("data/mahjong_riichi/tasks.jsonl")
        expected_count = sum(
            1 for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        self.assertGreater(expected_count, 0)
        self.assertEqual(len(tasks), expected_count)
        self.assertEqual(len({task.id for task in tasks}), expected_count)
        self.assertEqual(len({task.seed for task in tasks}), expected_count)

    def test_extracts_riichi_action(self):
        self.assertEqual(
            extract_riichi_action('{"action":"riichi","discard":"5M"}'),
            {"action": "riichi", "discard": "5m"},
        )
        self.assertEqual(
            extract_riichi_action('{"action":"chi","tiles":["3M","4m","5m"]}'),
            {"action": "chi", "tiles": ["3m", "4m", "5m"]},
        )
        self.assertEqual(
            extract_riichi_action('{"action":"pon","tile":"P"}'),
            {"action": "pon", "tile": "P"},
        )

    def test_scores_closed_tsumo(self):
        score = score_closed_hand(
            [
                "1m",
                "2m",
                "3m",
                "4m",
                "5m",
                "6m",
                "7p",
                "8p",
                "9p",
                "2s",
                "3s",
                "4s",
                "E",
                "E",
            ],
            win_tile="E",
            is_tsumo=True,
        )

        self.assertIsNotNone(score)
        self.assertIn("Menzen Tsumo", score["yaku"])

    def test_scores_open_meld_hand(self):
        score = score_closed_hand(
            [
                "2m",
                "3m",
                "4m",
                "5m",
                "6m",
                "7m",
                "2p",
                "2p",
                "2s",
                "3s",
                "4s",
                "P",
                "P",
                "P",
            ],
            win_tile="7m",
            is_tsumo=False,
            melds=[
                {
                    "type": "pon",
                    "tiles": ["P", "P", "P"],
                    "called_tile": "P",
                    "opened": True,
                    "who": 0,
                    "from_who": 1,
                }
            ],
        )

        self.assertIsNotNone(score)
        self.assertIn("Yakuhai (haku)", score["yaku"])

    def test_riichi_discards_identify_tenpai_discards(self):
        discards = riichi_discards(
            [
                "1m",
                "2m",
                "3m",
                "4m",
                "5m",
                "6m",
                "7p",
                "8p",
                "9p",
                "3s",
                "4s",
                "5s",
                "9s",
                "1m",
            ]
        )

        self.assertIn("9s", discards)

    def test_legal_calls_include_chi_pon_and_kan(self):
        chi_options = legal_call_options(
            ["2m", "4m", "P", "P", "P"],
            caller_seat=1,
            discarder_seat=0,
            discard="3m",
        )
        self.assertIn(("chi", ("2m", "3m", "4m")), [(o.action, o.tiles) for o in chi_options])

        honor_options = legal_call_options(
            ["P", "P", "P", "2m", "4m"],
            caller_seat=2,
            discarder_seat=0,
            discard="P",
        )
        self.assertIn("pon", [option.action for option in honor_options])
        self.assertIn("kan", [option.action for option in honor_options])

    def test_legal_closed_kan_tiles(self):
        self.assertEqual(
            legal_closed_kan_tiles(["5m", "5m", "5m", "5m", "P"]),
            ["5m"],
        )

    def test_scores_all_four_seats(self):
        self.assertEqual(
            score_riichi_seats(
                winner_seat=0,
                win_type="tsumo",
                loser_seat=None,
                final_scores=[29800, 23400, 23400, 23400],
                reasons=["seat0_tsumo:5m"],
            ),
            [1.0, 0.25, 0.25, 0.25],
        )
        self.assertEqual(
            score_riichi_seats(
                winner_seat=None,
                win_type=None,
                loser_seat=None,
                final_scores=[25000, 25000, 25000, 25000],
                reasons=["max_draws_reached"],
            ),
            [0.5, 0.5, 0.5, 0.5],
        )
        self.assertEqual(
            score_riichi_seats(
                winner_seat=1,
                win_type="ron",
                loser_seat=0,
                final_scores=[24000, 26000, 25000, 25000],
                reasons=["seat1_ron:N:from_seat0"],
            ),
            [0.0, 1.0, 0.375, 0.375],
        )

    def test_prompt_includes_strategy_hints(self):
        task = load_mahjong_riichi_tasks()[0]
        prompt = build_mahjong_riichi_prompt(
            task,
            draw_number=3,
            drawn_tile="E",
            hand=["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "C", "N"],
            discards=[["1p"], ["E", "9m"], ["4s"], ["N"]],
            melds=[[], [{"type": "pon", "tiles": ["E", "E", "E"], "opened": True}], [], []],
            riichi_declared=[False, True, False, False],
            scores=[25000, 25000, 25000, 25000],
            remaining_tiles=50,
            can_riichi_discards=[],
            legal_closed_kan_tiles=[],
        )

        self.assertIn("Benchmark objective:", prompt)
        self.assertIn("Legal discard tiles:", prompt)
        self.assertIn("Discard shanten hints", prompt)
        self.assertIn("Defense hints:", prompt)
        self.assertIn("genbutsu", prompt)

    def test_riichi_table_runs_with_agent(self):
        task = load_mahjong_riichi_tasks()[0]
        result = evaluate_mahjong_riichi_tasks([task], SimpleRiichiAgent())[0]

        self.assertTrue(result.reasons)
        self.assertGreaterEqual(len(result.draws), len(result.agent_actions))
        self.assertEqual(len(result.final_scores), 4)
        self.assertEqual(len(result.melds), 4)
        self.assertEqual(len(result.seat_scores), 4)

        summary = summarize_mahjong_riichi([result])
        self.assertIn("seat_total_scores", summary)
        self.assertIn("seat0", summary["seat_total_scores"])

    def test_external_mahjong_ai_oneshot_protocol(self):
        command = [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "request = json.load(sys.stdin); "
                "actions = request.get('legal_actions') or [{'action': 'pass'}]; "
                "print(json.dumps(actions[0]))"
            ),
        ]
        ai = ExternalMahjongAI(command, mode="oneshot", timeout=5)

        response = ai.choose(
            {
                "decision": "turn",
                "legal_actions": [
                    {"action": "discard", "tile": "1m", "discard": "1m"},
                ],
            }
        )

        self.assertEqual(response.action["action"], "discard")
        self.assertEqual(response.action["tile"], "1m")

    def test_external_mahjong_ai_stdio_protocol(self):
        command = [
            sys.executable,
            "-c",
            (
                "import json, sys\n"
                "for line in sys.stdin:\n"
                "    request = json.loads(line)\n"
                "    actions = request.get('legal_actions') or [{'action': 'pass'}]\n"
                "    print(json.dumps(actions[0]), flush=True)\n"
            ),
        ]
        ai = ExternalMahjongAI(command, mode="stdio", timeout=5)
        try:
            first = ai.choose({"legal_actions": [{"action": "pass"}]})
            second = ai.choose(
                {
                    "legal_actions": [
                        {"action": "discard", "tile": "9s", "discard": "9s"},
                    ]
                }
            )
        finally:
            ai.close()

        self.assertEqual(first.action, {"action": "pass"})
        self.assertEqual(second.action["tile"], "9s")

    def test_riichi_table_runs_with_external_opponents(self):
        task = load_mahjong_riichi_tasks()[0]
        command = [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "request = json.load(sys.stdin); "
                "events = request.get('mjai_events') or []; "
                "assert events and events[0].get('type') == 'start_game'; "
                "actions = request.get('legal_actions') or [{'action': 'pass'}]; "
                "print(json.dumps(actions[0]))"
            ),
        ]

        result = evaluate_mahjong_riichi_tasks(
            [task],
            SimpleRiichiAgent(),
            opponent="external",
            mahjong_ai_command=command,
            mahjong_ai_mode="oneshot",
            mahjong_ai_timeout=5,
        )[0]

        self.assertTrue(result.reasons)
        self.assertEqual(len(result.final_scores), 4)
        self.assertEqual(len(result.seat_scores), 4)


if __name__ == "__main__":
    unittest.main()

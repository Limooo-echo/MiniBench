"""Check the exact oracle against independent short-wall trajectory search."""
from functools import lru_cache
import json
import random
import unittest

from minibench.datasets.mahjong_rule_variants.rules import (
    RULE_CHANNELS, STANDARD_RULES, CYCLIC_SEQUENCES,
    is_rule_variant_winning_hand,
)
from minibench.datasets.mahjong_solo.dataset import mahjong_solo_task_from_dict
from minibench.datasets.mahjong_solo.oracle import optimal_actions, score_episode
from minibench.scoring.adapters import score_record


HAND = "1m 1m 1m 2m 3m 4m 2p 3p 4p 2s 3s 4s E".split()


def brute_actions(hand, wall, channel):
    @lru_cache(maxsize=None)
    def earliest(thirteen, future):
        if not future:
            return None
        fourteen = [*thirteen, future[0]]
        if is_rule_variant_winning_hand(fourteen, channel):
            return 1
        scores = []
        for discard in set(fourteen):
            remaining = list(fourteen)
            remaining.remove(discard)
            value = earliest(tuple(sorted(remaining)), future[1:])
            if value is not None:
                scores.append(1 + value)
        return min(scores) if scores else None

    if is_rule_variant_winning_hand(hand, channel):
        return [{"action": "tsumo"}], 0
    candidates = {}
    for discard in set(hand):
        remaining = list(hand)
        remaining.remove(discard)
        candidates[discard] = earliest(tuple(sorted(remaining)), tuple(wall))
    finite = [value for value in candidates.values() if value is not None]
    best = min(finite) if finite else None
    return ([{"action": "discard", "tile": tile} for tile, value in candidates.items()
             if best is not None and value == best], best)


class MahjongOracleTests(unittest.TestCase):
    def task(self, wall=("9s", "E"), **extra):
        return mahjong_solo_task_from_dict({"id": "oracle", "initial_hand": HAND,
                                            "wall": list(wall), "max_draws": len(wall), **extra})

    def test_complete_tied_answer_sets_match_independent_trajectory_search(self):
        rng = random.Random(20261009)
        fixtures = ["1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N N".split(),
                    "8m 9m 1m 2p 3p 4p 5s 6s 7s E E E N N".split(),
                    "1m 2m C 4p 5p 6p 7s 8s 9s E E E N N".split(),
                    "1m 1m 2m 2m 3p 3p 4p 4p 5s 5s 6s 6s E E".split(),
                    "1m 9m 1p 9p 1s 9s E S W N P F C C".split()]
        for winning in fixtures:
            hand = list(winning)
            removed = hand.pop(rng.randrange(14))
            extra = "5m" if hand.count("5m") < 4 else "6m"
            hand.append(extra)
            for wall in ((removed,), ("8p", removed)):
                for channel in RULE_CHANNELS:
                    with self.subTest(hand=hand, wall=wall, channel=channel):
                        answers, best = brute_actions(hand, wall, channel)
                        actual = optimal_actions(hand, wall, channel=channel)
                        self.assertEqual(actual["earliest_win_in_draws"], best)
                        self.assertEqual({tuple(sorted(a.items())) for a in actual["actions"]},
                                         {tuple(sorted(a.items())) for a in answers})

    def test_required_final_draw_cannot_be_ignored(self):
        hand = [*HAND, "9s"]
        self.assertEqual(optimal_actions(hand, ("8s",))["actions"], [])

    def test_answer_set_updates_after_a_suboptimal_discard(self):
        outputs = [json.dumps({"action": "discard", "tile": "1m"}),
                   json.dumps({"action": "discard", "tile": "9s"})]
        result = score_episode(self.task(), outputs)
        self.assertFalse(result["steps"][0]["correct"])
        self.assertEqual(result["steps"][0]["optimal_actions"], [{"action": "discard", "tile": "9s"}])
        self.assertEqual(result["steps"][1]["optimal_actions"], [])
        self.assertEqual(result["process_score"], 0)

    def test_invalid_attempts_count_and_success_overrides_process(self):
        outputs = ["not json", json.dumps({"action": "discard", "tile": "9s"}),
                   json.dumps({"action": "tsumo"})]
        record = {"success": False, "raw_outputs": outputs}
        task = {"id": "oracle", "initial_hand": HAND, "wall": ["9s", "E"], "max_draws": 2}
        result = score_record("mahjong_solo", "H", record, task, mode="history-only")
        self.assertEqual((result["y"], result["p"]), (1, 2 / 3))
        self.assertEqual(result["evidence"]["total_steps"], 3)
        self.assertEqual([step["draw_number"] for step in result["evidence"]["steps"]], [1, 1, 2])

    def test_failure_process_is_membership_ratio_without_heuristic_credit(self):
        outputs = [json.dumps({"action": "discard", "tile": "9s"}),
                   json.dumps({"action": "discard", "tile": "E"})]
        result = score_episode(self.task(), outputs)
        self.assertEqual((result["success"], result["process_score"]), (False, .5))
        self.assertEqual((result["correct_steps"], result["total_steps"]), (1, 2))

    def test_empty_wall_continuation_and_empty_log_have_zero_process(self):
        self.assertEqual(optimal_actions([*HAND, "9s"], ())["actions"], [])
        result = score_episode(self.task(), [])
        self.assertEqual((result["process_score"], result["total_steps"]), (0, 0))

    def test_variant_adapter_uses_active_rules_and_recomputes_outcome(self):
        hand = "8m 9m 1m 2p 3p 4p 5s 6s 7s E E E N".split()
        task = {"id": "rule", "initial_hand": hand, "wall": ["N"], "max_draws": 1,
                "channel": CYCLIC_SEQUENCES}
        result = score_record("mahjong_rule_variants", "R", {"success": False,
            "raw_outputs": [json.dumps({"action": "tsumo"})]}, task, mode="full-hand")
        self.assertEqual((result["y"], result["p"]), (1, 1))
        task["channel"] = STANDARD_RULES
        result = score_record("mahjong_rule_variants", "R", {"success": True,
            "raw_outputs": [json.dumps({"action": "tsumo"})] * 3}, task, mode="full-hand")
        self.assertEqual((result["y"], result["p"]), (0, 0))

    def test_wall_information_is_not_mutated(self):
        hand, wall = [*HAND, "9s"], ("E",)
        original = list(hand)
        self.assertEqual(optimal_actions(hand, wall)["actions"], [{"action": "discard", "tile": "9s"}])
        self.assertEqual(hand, original)

    def test_truncated_action_log_is_missing_process_evidence(self):
        task = {"id": "oracle", "initial_hand": HAND, "wall": ["9s", "E"], "max_draws": 2}
        result = score_record("mahjong_solo", "H", {"success": False,
            "raw_outputs": [json.dumps({"action": "discard", "tile": "9s"})]},
            task, mode="history-only")
        self.assertEqual((result["y"], result["p"], result["status"]), (0, None, "missing"))
        self.assertIn("incomplete_action_log", result["reasons"])

    def test_image_and_text_share_discard_utility_and_tenpai_scoring(self):
        for goal in ("tenpai_discard", "max_wait_discard", "max_ukeire_discard"):
            with self.subTest(goal=goal):
                task = {"id": "discard", "hand": [*HAND, "9s"], "goal": goal, "tags": ["visual"]}
                record = {"raw_output": json.dumps({"discard": "9s"})}
                text = score_record("mahjong", "D", record, task, mode="text")
                image = score_record("mahjong", "V", record, task, mode="image")
                self.assertEqual((text["y"], text["p"]), (image["y"], image["p"]))
                self.assertEqual(text["y"], 1)

    def test_nontrivial_ties_all_receive_credit(self):
        hand = "1m 2m 3m 4p 5p 6p 7s 8s 9s E E N S W".split()
        optimum = optimal_actions(hand, ("N", "E"))
        self.assertEqual({a["tile"] for a in optimum["actions"]}, {"S", "W"})


if __name__ == "__main__":
    unittest.main()

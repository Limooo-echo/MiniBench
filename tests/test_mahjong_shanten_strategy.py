import json
import random
from dataclasses import replace
from types import SimpleNamespace
import unittest

from minibench.datasets.mahjong.api import index_to_tile
from minibench.datasets.mahjong_rule_variants.rules import (
    CYCLIC_ONE_TILE_SUBSTITUTION,
    MODIFIED_RULES,
    RED_DRAGON_WILDCARD,
    calculate_rule_variant_shanten,
    channel_for_rules,
    is_rule_variant_winning_hand,
)
from minibench.datasets.mahjong_rule_variants.evaluation import (
    evaluate_mahjong_rule_variant_task,
    summarize_mahjong_rule_variants,
)
from minibench.datasets.mahjong_solo.evaluation import (
    evaluate_mahjong_solo_task,
    summarize_mahjong_solo,
)
from minibench.datasets.mahjong_solo.shanten_strategy import (
    mean_episode_shanten_discard_rate,
)
from tests.test_mahjong_rule_variants import make_task
from tests.test_mahjong_solo import (
    MessageSequenceMahjongAgent,
    SequenceMahjongAgent,
    delayed_tsumo_task,
)


class MahjongShantenStrategyTests(unittest.TestCase):
    def test_rule_channels_apply_live_ukeire_tiebreak_under_selected_rules(self):
        for channel in (
            RED_DRAGON_WILDCARD,
            CYCLIC_ONE_TILE_SUBSTITUTION,
            channel_for_rules(MODIFIED_RULES),
        ):
            task = make_task(
                channel,
                "2p 3p 4p 5p 6p 7p 7s 8s 9s 2m 3m 4m 5m".split(),
                ["E"],
            )
            # E leaves six ordinary winning copies; 2m/5m leave three.
            # Under the wildcard rule, C adds four copies to both choices.
            for discard, expected_rate in (("E", 1.0), ("2m", 0.0)):
                with self.subTest(channel=channel, discard=discard):
                    result = evaluate_mahjong_rule_variant_task(
                        task, SequenceMahjongAgent([{"action": "discard", "tile": discard}])
                    )
                    self.assertEqual(result.shanten_optimal_steps[0]["optimal_discards"], ["E"])
                    self.assertEqual(result.shanten_optimal_discard_rate, expected_rate)

    def test_live_ukeire_deducts_prior_discards_and_preserves_all_best_ties(self):
        task = replace(
            delayed_tsumo_task(),
            initial_hand=tuple("2m 2m 2m 2m 5m 5m 5m 5m 2p 3p 4p 7s 8s".split()),
            wall=tuple("9s 5p 6p 7p 3m 4m E".split()),
            max_draws=7,
        )
        for mode in ("full-hand", "history-only"):
            for last_discard, expected_hit in (("2m", True), ("5m", True), ("E", False)):
                with self.subTest(mode=mode, discard=last_discard):
                    agent_type = (
                        MessageSequenceMahjongAgent
                        if mode == "history-only" else SequenceMahjongAgent
                    )
                    result = evaluate_mahjong_solo_task(
                        task, agent_type([
                            {"action": "discard", "tile": tile}
                            for tile in ["2m", "2m", "2m", "5m", "5m", "5m", last_discard]
                        ]), observation_mode=mode,
                    )
                    # All 2m/5m copies are now public: E has zero live
                    # winning copies; discarding 2m or 5m still waits on E.
                    step = result.shanten_optimal_steps[-1]
                    self.assertEqual(step["draw_number"], 7)
                    self.assertEqual(step["optimal_discards"], ["2m", "5m"])
                    self.assertEqual(step["hit"], expected_hit)

    def test_minimum_shanten_ties_require_maximum_live_ukeire(self):
        # Discarding E waits on 2m/5m (six live copies); discarding
        # either 2m or 5m instead waits only on E (three live copies).
        task = replace(
            delayed_tsumo_task(),
            initial_hand=tuple("1p 2p 3p 4p 5p 6p 7s 8s 9s 2m 3m 4m 5m".split()),
            wall=("E",),
            max_draws=1,
        )
        for mode in ("full-hand", "history-only"):
            for discard, expected_rate in (("E", 1.0), ("2m", 0.0)):
                with self.subTest(mode=mode, discard=discard):
                    agent_type = (
                        MessageSequenceMahjongAgent
                        if mode == "history-only" else SequenceMahjongAgent
                    )
                    result = evaluate_mahjong_solo_task(
                        task, agent_type([{"action": "discard", "tile": discard}]),
                        observation_mode=mode,
                    )
                    self.assertEqual(result.shanten_optimal_steps[0]["optimal_discards"], ["E"])
                    self.assertEqual(result.shanten_optimal_discard_rate, expected_rate)

    def test_run_average_weights_tasks_equally_and_skips_empty_episodes(self):
        results = [
            SimpleNamespace(shanten_optimal_discard_rate=1.0),
            SimpleNamespace(shanten_optimal_discard_rate=1 / 3),
            SimpleNamespace(shanten_optimal_discard_rate=None),
        ]
        self.assertAlmostEqual(mean_episode_shanten_discard_rate(results), 2 / 3)

    def test_first_attempt_is_scored_and_winning_turn_is_excluded(self):
        good = evaluate_mahjong_solo_task(
            delayed_tsumo_task(),
            SequenceMahjongAgent(
                [{"action": "discard", "tile": "9s"}, {"action": "tsumo"}]
            ),
        )
        retried = evaluate_mahjong_solo_task(
            delayed_tsumo_task(),
            SequenceMahjongAgent(
                [
                    {"action": "tsumo"},
                    {"action": "discard", "tile": "9s"},
                    {"action": "tsumo"},
                ]
            ),
        )
        self.assertTrue(good.success)
        self.assertTrue(retried.success)
        self.assertEqual(good.shanten_optimal_discard_rate, 1.0)
        self.assertEqual(retried.shanten_optimal_discard_rate, 0.0)
        self.assertEqual(good.shanten_assessed_step_count, 1)
        self.assertEqual(retried.shanten_assessed_step_count, 1)
        self.assertIn("9s", good.shanten_optimal_steps[0]["optimal_discards"])
        self.assertEqual(
            summarize_mahjong_solo([good, retried])["shanten_optimal_discard_rate"],
            0.5,
        )

    def test_history_only_uses_same_oracle_without_exposing_it(self):
        agent = MessageSequenceMahjongAgent(
            [{"action": "discard", "tile": "9s"}, {"action": "tsumo"}]
        )
        result = evaluate_mahjong_solo_task(
            delayed_tsumo_task(), agent, observation_mode="history-only"
        )
        self.assertEqual(result.shanten_optimal_discard_rate, 1.0)
        self.assertNotIn("optimal_discards", json.dumps(agent.conversations))

    def test_all_rule_channels_score_using_their_own_shanten(self):
        for channel in (
            RED_DRAGON_WILDCARD,
            CYCLIC_ONE_TILE_SUBSTITUTION,
            channel_for_rules(MODIFIED_RULES),
        ):
            with self.subTest(channel=channel):
                task = make_task(
                    channel,
                    "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N".split(),
                    ["9m"],
                )
                result = evaluate_mahjong_rule_variant_task(
                    task, SequenceMahjongAgent([{"action": "discard", "tile": "9m"}])
                )
                self.assertEqual(result.shanten_optimal_discard_rate, 1.0)
                self.assertIn("9m", result.shanten_optimal_steps[0]["optimal_discards"])
                summary = summarize_mahjong_rule_variants([result])
                self.assertEqual(summary["shanten_optimal_discard_rate"], 1.0)
                self.assertEqual(
                    summary["by_channel"][channel]["shanten_optimal_discard_rate"],
                    1.0,
                )

    def test_wildcard_shanten_zero_matches_actual_waits(self):
        physical_tiles = [index_to_tile(index) for index in range(34) for _ in range(4)]
        randomizer = random.Random(71)
        for channel in (
            RED_DRAGON_WILDCARD,
            channel_for_rules(MODIFIED_RULES),
        ):
            for _ in range(100):
                randomizer.shuffle(physical_tiles)
                hand = physical_tiles[:13]
                waiting = any(
                    is_rule_variant_winning_hand([*hand, tile], channel)
                    for tile in set(physical_tiles[13:])
                )
                self.assertEqual(
                    calculate_rule_variant_shanten(hand, channel) == 0,
                    waiting,
                )


if __name__ == "__main__":
    unittest.main()

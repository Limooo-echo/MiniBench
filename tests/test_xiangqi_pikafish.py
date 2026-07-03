import importlib.util
import unittest
from pathlib import Path

from minibench.datasets.xiangqi.dataset import load_xiangqi_tasks, xiangqi_task_from_dict
from minibench.datasets.xiangqi.engines.pikafish import (
    PikafishAnalysis,
    board_to_pikafish_fen,
    parse_pikafish_analysis,
    square_to_uci,
    uci_to_square,
)
from minibench.datasets.xiangqi.task_generation import (
    extract_fen_from_pgn,
    fen_to_position,
    task_record_from_fen,
)
from minibench.datasets.xiangqi.move_scoring import (
    analysis_value_cp,
    engine_score_scale,
    grade_engine_score,
    score_from_loss_cp,
)


XIANGQI_ENV_AVAILABLE = importlib.util.find_spec("gym_xiangqi") is not None


class XiangqiPikafishTests(unittest.TestCase):
    def test_board_to_pikafish_fen(self):
        board = [
            [0, 0, 0, 8, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ]

        expected_board = "/".join(["3Rk4"] + ["9"] * 8 + ["4K4"])
        self.assertEqual(
            board_to_pikafish_fen(board, side_to_move="ally"),
            f"{expected_board} w - - 0 1",
        )
        self.assertEqual(
            board_to_pikafish_fen(board, side_to_move="enemy"),
            f"{expected_board} b - - 0 1",
        )

    def test_square_coordinate_round_trip(self):
        self.assertEqual(square_to_uci(7, 1), "b2")
        self.assertEqual(uci_to_square("b2"), (7, 1))
        self.assertEqual(square_to_uci(0, 8), "i9")
        self.assertEqual(uci_to_square("i9"), (0, 8))

    def test_parse_pikafish_analysis_score(self):
        analysis = parse_pikafish_analysis(
            "9/9/9/9/9/9/9/9/9/9 w - - 0 1",
            "a0a1",
            (
                "info depth 1 score cp 12 pv a0a1",
                "info depth 8 score mate 3 pv a0a1 a9a8",
            ),
        )

        self.assertEqual(analysis.score_kind, "mate")
        self.assertEqual(analysis.score, 3)
        self.assertEqual(analysis.depth, 8)
        self.assertEqual(analysis.pv, ("a0a1", "a9a8"))

    def test_engine_score_value_uses_requested_perspective(self):
        analysis = PikafishAnalysis(
            fen="4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1",
            bestmove="e0e1",
            score_kind="cp",
            score=120,
            depth=8,
            pv=("e0e1",),
            info_lines=(),
        )

        self.assertEqual(
            analysis_value_cp(
                analysis,
                side_to_move="ally",
                perspective_side="ally",
            ),
            120,
        )
        self.assertEqual(
            analysis_value_cp(
                analysis,
                side_to_move="enemy",
                perspective_side="ally",
            ),
            -120,
        )

    def test_mate_scores_are_ordered_as_large_cp_values(self):
        mate_in_3 = PikafishAnalysis(
            fen="4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1",
            bestmove="e0e1",
            score_kind="mate",
            score=3,
            depth=8,
            pv=("e0e1",),
            info_lines=(),
        )
        mate_in_5 = PikafishAnalysis(
            fen=mate_in_3.fen,
            bestmove="e0e1",
            score_kind="mate",
            score=5,
            depth=8,
            pv=("e0e1",),
            info_lines=(),
        )

        self.assertGreater(
            analysis_value_cp(
                mate_in_3,
                side_to_move="ally",
                perspective_side="ally",
            ),
            analysis_value_cp(
                mate_in_5,
                side_to_move="ally",
                perspective_side="ally",
            ),
        )
        self.assertLess(
            analysis_value_cp(
                mate_in_3,
                side_to_move="enemy",
                perspective_side="ally",
            ),
            0,
        )

    def test_move_score_from_centipawn_loss(self):
        self.assertEqual(score_from_loss_cp(0, loss_cap_cp=600), 1.0)
        self.assertEqual(score_from_loss_cp(30, loss_cap_cp=600), 1.0)
        self.assertEqual(score_from_loss_cp(600, loss_cap_cp=600), 0.0)
        self.assertAlmostEqual(
            score_from_loss_cp(315, loss_cap_cp=600),
            0.5,
        )

    def test_engine_score_grade_scale(self):
        self.assertEqual(grade_engine_score(None), None)
        self.assertEqual(grade_engine_score(0.97), "near_engine")
        self.assertEqual(grade_engine_score(0.88), "strong")
        self.assertEqual(grade_engine_score(0.72), "playable")
        self.assertEqual(grade_engine_score(0.50), "weak")
        self.assertEqual(grade_engine_score(0.25), "poor")
        self.assertEqual(grade_engine_score(0.10), "blunder_prone")
        self.assertEqual(engine_score_scale()[0]["grade"], "near_engine")

    @unittest.skipUnless(XIANGQI_ENV_AVAILABLE, "gym-xiangqi is not installed")
    def test_xiangqi_summary_reports_median_engine_losses(self):
        from minibench.datasets.xiangqi.evaluation import summarize_xiangqi

        results = [
            self._scored_result(
                "xq-unit-001",
                engine_score=1.0,
                engine_avg_loss_cp=20.0,
                move_losses=[10, 30],
            ),
            self._scored_result(
                "xq-unit-002",
                engine_score=0.5,
                engine_avg_loss_cp=200.0,
                move_losses=[200, 99_000],
            ),
        ]

        summary = summarize_xiangqi(results)

        self.assertEqual(summary["engine_average_score"], 0.75)
        self.assertEqual(summary["engine_average_grade"], "playable")
        self.assertEqual(summary["engine_score_summary"], "0.750 -> playable")
        self.assertEqual(summary["engine_raw_average_loss_cp"], 110.0)
        self.assertEqual(summary["engine_median_loss_cp"], 110.0)
        self.assertEqual(summary["engine_per_move_median_loss_cp"], 115.0)
        self.assertNotIn("engine_average_loss_cp", summary)

    @unittest.skipUnless(XIANGQI_ENV_AVAILABLE, "gym-xiangqi is not installed")
    def test_xiangqi_summary_text_uses_human_readable_engine_score(self):
        import tempfile

        from minibench.datasets.xiangqi.evaluation import write_xiangqi_run

        results = [
            self._scored_result(
                "xq-unit-001",
                engine_score=1.0,
                engine_avg_loss_cp=20.0,
                move_losses=[10, 30],
            ),
            self._scored_result(
                "xq-unit-002",
                engine_score=0.5,
                engine_avg_loss_cp=200.0,
                move_losses=[200, 99_000],
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_xiangqi_run(results, tmpdir, "xiangqi-summary-unit")
            summary_text = (run_dir / "summary.txt").read_text(encoding="utf-8")

        self.assertIn("engine_average_score=0.750 -> playable", summary_text)
        self.assertIn("median_loss_cp=110.0", summary_text)
        self.assertIn("per_move_median_loss_cp=115.0", summary_text)
        self.assertNotIn("engine_average_loss_cp=", summary_text)

    def _scored_result(
        self,
        task_id: str,
        *,
        engine_score: float,
        engine_avg_loss_cp: float,
        move_losses: list[int],
    ):
        from minibench.datasets.xiangqi.evaluation import (
            XiangqiAgentMoveScore,
            XiangqiInstanceResult,
        )

        return XiangqiInstanceResult(
            task_id=task_id,
            success=True,
            score=1.0,
            engine_score=engine_score,
            engine_grade=grade_engine_score(engine_score),
            engine_avg_loss_cp=engine_avg_loss_cp,
            raw_outputs=[],
            actions=[],
            engine_moves=[],
            fen_history=[],
            agent_move_scores=[
                XiangqiAgentMoveScore(
                    step=index,
                    action=index,
                    uci_move="a0a1",
                    bestmove="a0a1",
                    matched_bestmove=True,
                    before_score_kind="cp",
                    before_score=0,
                    before_value_cp=0,
                    after_score_kind="cp",
                    after_score=-loss,
                    after_value_cp=-loss,
                    loss_cp=loss,
                    move_score=1.0,
                    move_grade="near_engine",
                )
                for index, loss in enumerate(move_losses, start=1)
            ],
            reasons=["agent_survived_step_limit"],
            tags=("xiangqi",),
        )

    def test_fen_to_position_assigns_unique_piece_ids(self):
        position = fen_to_position(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/"
            "P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        )

        self.assertEqual(position.side_to_move, "ally")
        self.assertEqual(position.board[0], [-8, -6, -4, -2, -1, -3, -5, -7, -9])
        self.assertEqual(position.board[9], [8, 6, 4, 2, 1, 3, 5, 7, 9])
        self.assertEqual(position.board[6], [12, 0, 13, 0, 14, 0, 15, 0, 16])

    def test_extract_fen_from_pgn_bytes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.pgn"
            path.write_bytes(
                b'[Game "Chinese Chess"]\n'
                b'[FEN "4k4/9/9/9/9/9/9/9/9/4K4 b - - 0 1"]\n'
                b"\xff\xfe bad non-utf8 movetext\n"
            )

            self.assertEqual(
                extract_fen_from_pgn(path),
                "4k4/9/9/9/9/9/9/9/9/4K4 b - - 0 1",
            )

    @unittest.skipUnless(XIANGQI_ENV_AVAILABLE, "gym-xiangqi is not installed")
    def test_task_record_from_fen_matches_loader_schema(self):
        analysis = PikafishAnalysis(
            fen="4k4/9/9/9/9/9/9/9/9/R3K4 w - - 0 1",
            bestmove="a0a9",
            score_kind="mate",
            score=1,
            depth=8,
            pv=("a0a9",),
            info_lines=("info depth 8 score mate 1 pv a0a9",),
        )
        record = task_record_from_fen(
            task_id="unit-ccpd-tactic",
            fen=analysis.fen,
            source_file="Dataset/sample.pgn",
            source_kind="endgame",
            category="tactical-win",
            analysis=analysis,
            max_steps=16,
        )
        task = xiangqi_task_from_dict(record)

        self.assertEqual(task.goal, "agent_win")
        self.assertEqual(task.opponent, "pikafish")
        self.assertIn("category:tactical-win", task.tags)

    def test_dataset_accepts_survival_goal(self):
        task = xiangqi_task_from_dict(
            {
                "id": "unit-survival",
                "board": [[0] * 9 for _ in range(10)],
                "side_to_move": "ally",
                "agent_side": "ally",
                "opponent": "pikafish",
                "max_steps": 2,
                "goal": "agent_survive",
                "tags": ["xiangqi"],
            }
        )

        self.assertEqual(task.goal, "agent_survive")

    def test_hard_dataset_has_three_categories(self):
        tasks = load_xiangqi_tasks(Path("data/xiangqi/hard_tasks.jsonl"))
        categories = {
            tag
            for task in tasks
            for tag in task.tags
            if tag.startswith("category:")
        }

        self.assertEqual(len(tasks), 9)
        self.assertEqual(
            categories,
            {
                "category:tactical-win",
                "category:advantage-play",
                "category:survival-defense",
            },
        )
        for category in categories:
            self.assertEqual(
                sum(category in task.tags for task in tasks),
                3,
            )

    @unittest.skipUnless(XIANGQI_ENV_AVAILABLE, "gym-xiangqi is not installed")
    def test_strict_actions_filter_moves_that_leave_general_in_check(self):
        from minibench.datasets.xiangqi.env import (
            make_xiangqi_env_from_board,
            strict_legal_actions,
        )

        board = [
            [0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -12, 0, 0, 0, 0],
            [0, 0, 0, 0, 12, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [8, 0, 0, 0, 1, 0, 0, 0, -8],
        ]
        env = make_xiangqi_env_from_board(board, side_to_move="ally")

        try:
            actions = strict_legal_actions(env)

            self.assertNotIn(92731, actions)
            self.assertEqual(actions, [7726])
        finally:
            env.close()

    @unittest.skipUnless(XIANGQI_ENV_AVAILABLE, "gym-xiangqi is not installed")
    def test_prompt_uses_strict_legal_actions(self):
        from minibench.datasets.xiangqi.env import make_xiangqi_env_from_board
        from minibench.datasets.xiangqi.prompting import build_xiangqi_prompt

        task = xiangqi_task_from_dict(
            {
                "id": "unit-filter-prompt",
                "board": [
                    [0, 0, 0, 0, -1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, -12, 0, 0, 0, 0],
                    [0, 0, 0, 0, 12, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [8, 0, 0, 0, 1, 0, 0, 0, -8],
                ],
                "side_to_move": "ally",
                "agent_side": "ally",
                "opponent": "pikafish",
                "max_steps": 2,
                "goal": "agent_survive",
                "tags": ["xiangqi"],
            }
        )
        env = make_xiangqi_env_from_board(task.board, side_to_move=task.side_to_move)

        try:
            prompt = build_xiangqi_prompt(task, env, [])

            self.assertIn("7726: GENERAL#1", prompt)
            self.assertNotIn("92731: SOLDIER_1#12", prompt)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

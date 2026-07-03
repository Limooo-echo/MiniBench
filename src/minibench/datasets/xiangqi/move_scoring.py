from __future__ import annotations

from minibench.datasets.xiangqi.engines.pikafish import PikafishAnalysis


MATE_SCORE_CP = 100_000
MATE_PLY_PENALTY_CP = 1_000
FREE_LOSS_CP = 30

ENGINE_SCORE_SCALE = (
    {
        "grade": "near_engine",
        "min_score": 0.95,
        "max_score": 1.00,
        "approx_avg_loss_cp": "0-60",
        "meaning": "near-best play",
    },
    {
        "grade": "strong",
        "min_score": 0.85,
        "max_score": 0.95,
        "approx_avg_loss_cp": "60-115",
        "meaning": "strong play with small inaccuracies",
    },
    {
        "grade": "playable",
        "min_score": 0.70,
        "max_score": 0.85,
        "approx_avg_loss_cp": "115-200",
        "meaning": "playable but misses some improvements",
    },
    {
        "grade": "weak",
        "min_score": 0.45,
        "max_score": 0.70,
        "approx_avg_loss_cp": "200-345",
        "meaning": "weak play with frequent mistakes",
    },
    {
        "grade": "poor",
        "min_score": 0.20,
        "max_score": 0.45,
        "approx_avg_loss_cp": "345-485",
        "meaning": "poor play with large mistakes",
    },
    {
        "grade": "blunder_prone",
        "min_score": 0.00,
        "max_score": 0.20,
        "approx_avg_loss_cp": "485+",
        "meaning": "major blunders or losing tactical swings",
    },
)


def analysis_value_cp(
    analysis: PikafishAnalysis,
    *,
    side_to_move: str,
    perspective_side: str,
) -> int:
    if analysis.score_kind == "cp":
        value = analysis.score
    elif analysis.score_kind == "mate":
        sign = 1 if analysis.score > 0 else -1 if analysis.score < 0 else 0
        value = sign * max(
            MATE_SCORE_CP - abs(analysis.score) * MATE_PLY_PENALTY_CP,
            MATE_PLY_PENALTY_CP,
        )
    else:
        raise ValueError(f"unsupported Pikafish score kind: {analysis.score_kind}")

    return value if side_to_move == perspective_side else -value


def score_from_loss_cp(loss_cp: int, *, loss_cap_cp: int) -> float:
    if loss_cap_cp <= FREE_LOSS_CP:
        raise ValueError("loss_cap_cp must be greater than 30")
    if loss_cp <= FREE_LOSS_CP:
        return 1.0
    if loss_cp >= loss_cap_cp:
        return 0.0
    return 1.0 - ((loss_cp - FREE_LOSS_CP) / (loss_cap_cp - FREE_LOSS_CP))


def grade_engine_score(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 0.95:
        return "near_engine"
    if score >= 0.85:
        return "strong"
    if score >= 0.70:
        return "playable"
    if score >= 0.45:
        return "weak"
    if score >= 0.20:
        return "poor"
    return "blunder_prone"


def engine_score_scale() -> list[dict[str, object]]:
    return [dict(item) for item in ENGINE_SCORE_SCALE]

from __future__ import annotations

from gym_xiangqi.utils import action_space_to_move
from gym_xiangqi.constants import PIECE_ID_TO_NAME

from minibench.xiangqi_dataset import XiangqiTask
from minibench.xiangqi_env import strict_legal_actions, turn_to_side


FILES = "abcdefghi"


XIANGQI_SYSTEM_PROMPT = """You are playing a Xiangqi endgame benchmark.
You must choose exactly one legal action from the provided legal action list.
Return exactly one JSON object with schema {"action": 123}.
Do not include markdown fences or explanations."""


def board_to_text(board) -> str:
    lines = []
    for r, row in enumerate(board):
        cells = " ".join(f"{int(x):>3}" for x in row)
        lines.append(f"row {r}: {cells}")
    return "\n".join(lines)


def format_action(action: int) -> str:
    piece_id, start, end = action_space_to_move(action)
    piece_name = PIECE_ID_TO_NAME[piece_id]
    uci = _square_to_uci(int(start[0]), int(start[1])) + _square_to_uci(
        int(end[0]),
        int(end[1]),
    )
    return f"{action}: {piece_name}#{piece_id} {tuple(start)} -> {tuple(end)} uci={uci}"


def _square_to_uci(row: int, col: int) -> str:
    return f"{FILES[col]}{9 - row}"


def _goal_guidance(task: XiangqiTask) -> str:
    tags = set(task.tags)

    if "category:tactical-win" in tags or task.goal == "agent_win":
        return (
            "Success condition: force a win. Prefer forcing checks, captures, "
            "and mating threats over passive survival moves."
        )
    if "category:advantage-play" in tags:
        return (
            "Success condition: preserve the advantage and survive the move "
            "horizon. Avoid allowing counterplay against your general."
        )
    if "category:survival-defense" in tags or task.goal == "agent_survive":
        return (
            "Success condition: survive the move horizon. Prioritize king "
            "safety, blocking checks, and avoiding immediate tactical losses."
        )

    return "Success condition: satisfy the stated goal while keeping your general safe."


def build_xiangqi_prompt(task: XiangqiTask, env, history: list[str]) -> str:
    actions = strict_legal_actions(env)
    action_lines = "\n".join(format_action(a) for a in actions)

    history_text = "\n".join(history) if history else "(none)"
    current_side = turn_to_side(env)

    return f"""{XIANGQI_SYSTEM_PROMPT}

Task ID: {task.id}
Goal: {task.goal}
Task tags: {", ".join(task.tags)}
Agent side: {task.agent_side}
Opponent: {task.opponent}
Side to move: {current_side}
Max steps: {task.max_steps}
{_goal_guidance(task)}

Board encoding:
- positive number = ally piece
- negative number = enemy piece
- 0 = empty
- 1 = General
- 8/9 = Chariot
- 10/11 = Cannon
- 6/7 = Horse
- 12-16 = Soldier

Current board:
{board_to_text(env.state)}

Move history:
{history_text}

Safe legal actions:
These actions have already been filtered to remove moves that leave your own
general immediately capturable. Choose one action id from this list.
{action_lines}

Return exactly:
{{"action": one_integer_from_the_legal_action_list}}
"""

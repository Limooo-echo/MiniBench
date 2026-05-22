from __future__ import annotations

from gym_xiangqi.utils import action_space_to_move
from gym_xiangqi.constants import PIECE_ID_TO_NAME

from minibench.xiangqi_dataset import XiangqiTask
from minibench.xiangqi_env import legal_actions, turn_to_side


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
    return f"{action}: {piece_name}#{piece_id} {tuple(start)} -> {tuple(end)}"


def build_xiangqi_prompt(task: XiangqiTask, env, history: list[str]) -> str:
    actions = legal_actions(env)
    action_lines = "\n".join(format_action(a) for a in actions)

    history_text = "\n".join(history) if history else "(none)"
    current_side = turn_to_side(env)

    return f"""{XIANGQI_SYSTEM_PROMPT}

Task ID: {task.id}
Goal: {task.goal}
Agent side: {task.agent_side}
Opponent: {task.opponent}
Side to move: {current_side}
Max steps: {task.max_steps}

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

Legal actions:
{action_lines}

Return exactly:
{{"action": one_integer_from_the_legal_action_list}}
"""

from __future__ import annotations

import json

from minibench.datasets.zebra.dataset import ZebraTask


# Adapted and modified for MiniBench from WildEval/ZeroEval
# src/templates/ZEBRA_GRID.py and src/_TEMPLATES.py under Apache-2.0.
# See THIRD_PARTY_NOTICES.md.
ZEBRA_SYSTEM_PROMPT = (
    "You solve Zebra logic grid puzzles. Follow every clue exactly and return "
    "the requested JSON object without markdown fences or trailing commentary."
)

ZEBRA_EXAMPLE = """# Example Puzzle

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {"Name": "Arnold", "Drink": "tea"},
        "House 2": {"Name": "Peter", "Drink": "water"},
        "House 3": {"Name": "Eric", "Drink": "milk"}
    }
}"""


def solution_json_template(task: ZebraTask) -> dict[str, object]:
    columns = task.solution.header
    return {
        "reasoning": "___",
        "solution": {
            f"House {index}": {column: "___" for column in columns[1:]}
            for index in range(1, len(task.solution.rows) + 1)
        },
    }


def final_solution_instruction(task: ZebraTask) -> str:
    template = json.dumps(solution_json_template(task), indent=4, ensure_ascii=False)
    return (
        "Now solve the puzzle using all information provided. Present your "
        "reasoning and complete solution in exactly this JSON shape:\n\n"
        f"{template}"
    )


def build_zebra_prompt(task: ZebraTask) -> str:
    sections = [ZEBRA_EXAMPLE, "# Puzzle to Solve", task.puzzle]
    if task.rule_context:
        sections.extend(
            [
                "# Additional Rule Context",
                task.rule_context,
                "Apply this rule context together with every puzzle clue.",
            ]
        )
    sections.extend(["# Instruction", final_solution_instruction(task)])
    return "\n\n".join(sections)


def history_system_prompt(task: ZebraTask, mode: str) -> str:
    if mode in {"incremental_state", "incremental_state_pro"}:
        protocol = (
            "During intermediate turns, return a complete structured candidate state as JSON "
            'with exactly the shape {"domains": {attribute: {value: [house_numbers]}}}. '
            "Include every attribute and every value from the background, with a "
            "nonempty list of distinct integer house numbers from 1 to the number "
            "of houses. A list contains the positions still possible for that value. "
            "Only remove a position when the clues seen so far rule it out. "
            "Do not guess a single position just because it is possible. "
            "You may enlarge candidate lists and revise earlier decisions to correct "
            "mistakes. The supplied previous state is a fallible working note; all "
            "original clues remain authoritative. Return only the domains object, "
            "without reasoning text, extra fields, or a final solution grid. "
            "When explicitly asked to solve the puzzle at the final turn, instead "
            "return the complete solution in the final request's JSON shape."
        )
    elif mode == "deferred_reasoning":
        protocol = (
            "After each clue, do not reason, infer, summarize, or maintain a "
            "candidate state. Only acknowledge that clue in the required JSON."
        )
    else:
        raise ValueError(f"unknown Zebra history mode: {mode}")
    return "\n\n".join(
        [
            ZEBRA_SYSTEM_PROMPT,
            "This is a multi-turn Zebra puzzle. The background is:",
            task.puzzle,
            protocol,
        ]
    )


def history_clue_prompt(task: ZebraTask, mode: str, index: int) -> str:
    clue = task.clue_turns[index - 1]
    total = len(task.clue_turns)
    if mode in {"incremental_state", "incremental_state_pro"}:
        instruction = (
            "Update the complete structured domains state using the clues seen "
            "so far. Keep all positions that have not been ruled out."
        )
    elif mode == "deferred_reasoning":
        instruction = f'Return only {{"acknowledged":{index}}}.'
    else:
        raise ValueError(f"unknown Zebra history mode: {mode}")
    return f"Clue {index} of {total}:\n{clue}\n\n{instruction}"


def structured_history_prompt(
    task: ZebraTask,
    clue_count: int,
    state: dict[str, object],
    *,
    final: bool = False,
) -> str:
    """Rebuild the same clean, prefix-only context for both state conditions."""
    clues = "\n".join(
        f"{index}. {clue}"
        for index, clue in enumerate(task.clue_turns[:clue_count], start=1)
    )
    instruction = (
        final_solution_instruction(task)
        if final else
        "Return the complete updated domains JSON only. You may revise or enlarge "
        "previous candidate lists; do not output the final grid yet."
    )
    return (
        f"Clues received so far ({clue_count} of {len(task.clue_turns)}):\n{clues}\n\n"
        "Previous committed candidate state (a fallible working note):\n"
        f"{json.dumps(state, ensure_ascii=False, separators=(',', ':'))}\n\n"
        f"{instruction}"
    )

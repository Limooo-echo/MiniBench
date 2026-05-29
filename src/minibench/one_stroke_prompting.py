from __future__ import annotations

from minibench.one_stroke_dataset import OneStrokeTask


ONE_STROKE_SYSTEM_PROMPT = (
    "You solve one-stroke graph puzzles. Return exactly one JSON object with "
    'the schema {"path":["A","B","C"]}. The path must visit every listed edge '
    "exactly once, may repeat vertices when needed, and must not include "
    "markdown or commentary."
)


def build_one_stroke_prompt(task: OneStrokeTask) -> str:
    lines = [
        "Solve this one-stroke graph puzzle.",
        "",
        "Rules:",
        "- Move along one listed undirected edge at a time.",
        "- Use every edge exactly once.",
        "- You may revisit a vertex, but you may not reuse an edge.",
        "- Return only JSON: {\"path\":[\"A\",\"B\"]}.",
        "",
        f"Vertices: {', '.join(task.vertices)}",
        "Edges:",
    ]
    for index, (a, b) in enumerate(task.edges, start=1):
        lines.append(f"{index}. {a}-{b}")
    if task.start is not None:
        lines.append(f"Required start vertex: {task.start}")
    if task.end is not None:
        lines.append(f"Required end vertex: {task.end}")
    lines.extend(
        [
            "",
            "Your answer must contain len(edges)+1 vertex names in order.",
        ]
    )
    return "\n".join(lines)

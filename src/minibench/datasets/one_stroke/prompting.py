from __future__ import annotations

from collections import Counter

from minibench.datasets.one_stroke.dataset import (
    OneStrokeHistoryEvent,
    OneStrokeTask,
    one_stroke_edge_ids,
)


ONE_STROKE_PROMPT_VARIANTS = ("baseline", "euler_theorem")

ONE_STROKE_SYSTEM_PROMPT = (
    "You solve one-stroke graph puzzles. Return exactly one JSON object with "
    'either the schema {"path":["A","B","C"]} when a one-stroke path exists, '
    'or {"solvable":false} when no such path exists. A path must visit every '
    "listed edge exactly once, may repeat vertices when needed, and must not "
    "include markdown or commentary. If no valid path exists, do not invent or "
    "guess a path; return exactly {\"solvable\":false}."
)

ONE_STROKE_MEMORY_MODES = ("incremental_state", "step_history_only")


def build_one_stroke_prompt(
    task: OneStrokeTask,
    *,
    prompt_variant: str = "baseline",
) -> str:
    if prompt_variant not in ONE_STROKE_PROMPT_VARIANTS:
        choices = ", ".join(ONE_STROKE_PROMPT_VARIANTS)
        raise ValueError(
            f"unknown one-stroke prompt variant {prompt_variant!r}: {choices}"
        )

    lines = [
        "Solve this one-stroke graph puzzle, or determine that it has no solution.",
        "",
        "Rules:",
        "- Move along one listed undirected edge at a time.",
        "- Use every edge exactly once.",
        "- You may revisit a vertex, but you may not reuse an edge.",
        "- If a one-stroke path exists, return only JSON: {\"path\":[\"A\",\"B\"]}.",
        "- If no one-stroke path exists, return only JSON: {\"solvable\":false}.",
        "- Do not force a path for an unsolvable graph. A guessed path that repeats, "
        "skips, or invents edges is wrong; use {\"solvable\":false} instead.",
        "",
    ]
    if prompt_variant == "euler_theorem":
        edge_count = len(task.edges)
        vertex_count = edge_count + 1
        degrees = _degree_table(task)
        odd_vertices = [vertex for vertex in task.vertices if degrees[vertex] % 2 == 1]
        odd_text = ", ".join(odd_vertices) if odd_vertices else "none"
        degree_text = ", ".join(
            f"{vertex}={degrees[vertex]}" for vertex in task.vertices
        )
        component_count = _non_isolated_component_count(task)
        lines.extend(
            [
                "Useful theorem and checklist:",
                "- First compute every vertex degree. A vertex with odd degree is an "
                "odd-degree vertex.",
                "- Check connectivity among vertices that touch at least one edge. If "
                "those non-isolated vertices are not connected, return "
                '{"solvable":false}.',
                "- If there are 0 odd-degree vertices, a connected graph has an Euler "
                "circuit. The path may start at any non-isolated vertex and must end "
                "at the same vertex. If a required start or end is given, use that "
                "required vertex as both the start and the end.",
                "- If there are exactly 2 odd-degree vertices, a connected graph has "
                "an Euler trail. The path must start at one odd-degree vertex and "
                "end at the other odd-degree vertex. Any required start/end must "
                "match those two odd-degree vertices.",
                "- If there are any other number of odd-degree vertices, return "
                '{"solvable":false}.',
                "",
                "Computed graph facts for this puzzle:",
                f"- Degree table: {degree_text}.",
                f"- Odd-degree vertices ({len(odd_vertices)}): {odd_text}.",
                f"- Non-isolated connected components: {component_count}.",
                "- Use these computed facts; do not recount them differently.",
                "",
                f"- This puzzle has exactly {edge_count} listed edges, so a path "
                f"answer must contain exactly {vertex_count} vertices and exactly "
                f"{edge_count} edge-steps.",
                "- Treat A-B and B-A as the same undirected edge. Never use both "
                "directions of the same listed edge.",
                "- Before answering, internally audit each consecutive pair in your "
                "path: it must be a listed edge, no listed edge may be repeated, "
                "no listed edge may be missing, and the final path uses every "
                "listed edge exactly once.",
                "- Do not add extra steps to return to the start unless that final "
                "step uses the last unused listed edge. Do not output a partial or "
                "overlong path.",
                "",
            ]
        )
    lines.extend(
        [
            f"Vertices: {', '.join(task.vertices)}",
            "Edges:",
        ]
    )
    for index, (a, b) in enumerate(task.edges, start=1):
        lines.append(f"{index}. {a}-{b}")
    if task.start is not None:
        lines.append(f"Required start vertex: {task.start}")
    if task.end is not None:
        lines.append(f"Required end vertex: {task.end}")
    lines.extend(
        [
            "",
            f"A path answer must contain exactly {len(task.edges) + 1} vertex names "
            f"in order, because there are exactly {len(task.edges)} listed edges.",
        ]
    )
    return "\n".join(lines)


def history_system_prompt(task: OneStrokeTask, memory_mode: str) -> str:
    if memory_mode not in ONE_STROKE_MEMORY_MODES:
        choices = ", ".join(ONE_STROKE_MEMORY_MODES)
        raise ValueError(f"unknown one-stroke memory mode {memory_mode!r}: {choices}")
    lines = [
        "You are participating in a multi-turn one-stroke graph task.",
        "The graph is undirected. Each edge has a unique ID and may be used once.",
        "Vertices may be revisited. Undo events make their edge unused again.",
        "Track the event history exactly. Do not assume any unreported move.",
        "At the final turn, continue from the current vertex and use every remaining "
        "edge exactly once.",
        "",
        f"Task ID: {task.id}",
        f"Initial vertex: {task.start}",
        f"Required final vertex: {task.end if task.end is not None else 'not fixed'}",
        f"Vertices: {', '.join(task.vertices)}",
        "Static edge list:",
    ]
    for edge_id, (a, b) in zip(one_stroke_edge_ids(task.edges), task.edges):
        lines.append(f"- {edge_id}: {a}-{b}")
    lines.extend(["", _memory_mode_instruction(memory_mode)])
    return "\n".join(lines)


def history_event_prompt(
    task: OneStrokeTask,
    memory_mode: str,
    step_number: int,
) -> str:
    if memory_mode not in ONE_STROKE_MEMORY_MODES:
        choices = ", ".join(ONE_STROKE_MEMORY_MODES)
        raise ValueError(f"unknown one-stroke memory mode {memory_mode!r}: {choices}")
    if not 1 <= step_number <= len(task.history_events):
        raise ValueError(f"history step out of range: {step_number}")
    event = task.history_events[step_number - 1]
    if event.action == "move":
        action_text = (
            f"A move occurred: {event.from_vertex} -> {event.to_vertex} "
            f"using {event.edge_id}. That edge is now used."
        )
    else:
        action_text = (
            f"The latest move was undone: {event.from_vertex} -> {event.to_vertex} "
            f"along {event.edge_id}. That edge is unused again."
        )
    incident = _incident_edge_text(task, event.to_vertex)
    lines = [
        f"Step {step_number} of {len(task.history_events)}.",
        action_text,
        f"Current vertex: {event.to_vertex}",
        f"Original incident edges at {event.to_vertex}: {incident}",
        "The incident list is static and does not indicate which edges are already used.",
    ]
    if memory_mode == "incremental_state":
        lines.extend(
            [
                "Record the complete intermediate state now. Return only JSON:",
                '{"current_vertex":"A","used_edges":["e01"],'
                '"remaining_edges":["e02"]}',
            ]
        )
    else:
        lines.append(f'Return only JSON: {{"step":{step_number}}}')
    return "\n".join(lines)


def history_final_prompt(task: OneStrokeTask) -> str:
    return "\n".join(
        [
            "The event history is complete.",
            "Continue from the current vertex using every edge that remains unused "
            "exactly once. Do not include already-used steps in the returned path.",
            "If a completion exists, return only JSON using the schema "
            '{"path":["CURRENT","NEXT","FINAL"]}. The first vertex must be the '
            "current vertex.",
            "If no completion exists, return only JSON: {\"solvable\":false}.",
        ]
    )


def _memory_mode_instruction(memory_mode: str) -> str:
    if memory_mode == "incremental_state":
        return (
            "After each event, explicitly record the current vertex plus the complete "
            "used-edge and remaining-edge sets. Your own record will stay in chat "
            "history for the final decision."
        )
    return (
        "After each event, acknowledge only its step number. Do not write an "
        "intermediate state, edge set, summary, or reasoning. The raw sequence of "
        "step events will stay in chat history for the final decision."
    )


def _incident_edge_text(task: OneStrokeTask, vertex: str) -> str:
    items = []
    for edge_id, (a, b) in zip(one_stroke_edge_ids(task.edges), task.edges):
        if vertex in {a, b}:
            items.append(f"{edge_id}({a}-{b})")
    return ", ".join(items)


def _degree_table(task: OneStrokeTask) -> Counter[str]:
    degrees = Counter[str]({vertex: 0 for vertex in task.vertices})
    for a, b in task.edges:
        degrees[a] += 1
        degrees[b] += 1
    return degrees


def _non_isolated_component_count(task: OneStrokeTask) -> int:
    adjacency = {vertex: set[str]() for vertex in task.vertices}
    active = set[str]()
    for a, b in task.edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
        active.add(a)
        active.add(b)
    count = 0
    visited = set[str]()
    for vertex in sorted(active):
        if vertex in visited:
            continue
        count += 1
        stack = [vertex]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(sorted(adjacency[current] - visited))
    return count

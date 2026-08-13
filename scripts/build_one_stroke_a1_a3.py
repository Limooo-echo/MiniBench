#!/usr/bin/env python3
"""Build the deterministic MiniBench 2.0 A1/A3 one-stroke task sets."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "one_stroke"


def _canonical(edge: tuple[str, str]) -> tuple[str, str]:
    a, b = edge
    return (a, b) if a <= b else (b, a)


def _edge_lists(text: str) -> list[list[str]]:
    return [token.split("-") for token in text.split()]


def _euler_path(
    vertices: list[str],
    edges: list[list[str]],
    start: str | None,
) -> list[str] | None:
    adjacency: dict[str, list[tuple[int, str]]] = {vertex: [] for vertex in vertices}
    degree = Counter[str]()
    for edge_index, (a, b) in enumerate(edges):
        adjacency[a].append((edge_index, b))
        adjacency[b].append((edge_index, a))
        degree[a] += 1
        degree[b] += 1
    odd = sorted(vertex for vertex in vertices if degree[vertex] % 2)
    if len(odd) not in {0, 2}:
        return None
    if start is None:
        start = odd[0] if odd else next(vertex for vertex in vertices if degree[vertex])
    if odd and start not in odd:
        return None
    used = [False] * len(edges)
    positions = {vertex: 0 for vertex in vertices}
    stack = [start]
    reverse_path: list[str] = []
    while stack:
        current = stack[-1]
        candidates = adjacency[current]
        while positions[current] < len(candidates) and used[candidates[positions[current]][0]]:
            positions[current] += 1
        if positions[current] == len(candidates):
            reverse_path.append(stack.pop())
            continue
        edge_index, neighbor = candidates[positions[current]]
        used[edge_index] = True
        stack.append(neighbor)
    if not all(used):
        return None
    return list(reversed(reverse_path))


def _direct_record(
    task_id: str,
    difficulty: str,
    vertices: str,
    edge_text: str,
    *,
    solvable: bool,
    start: str | None = None,
    end: str | None = None,
    source: str = "handcrafted",
) -> dict[str, object]:
    vertex_list = vertices.split()
    edges = _edge_lists(edge_text)
    path = _euler_path(vertex_list, edges, start)
    if solvable != (path is not None and (end is None or path[-1] == end)):
        raise ValueError(f"bad direct specification: {task_id}")
    return {
        "id": task_id,
        "capability": "direct",
        "difficulty": difficulty,
        "vertices": vertex_list,
        "edges": edges,
        "start": start,
        "end": end,
        "solution_exists": solvable,
        "solution_path": path if solvable else None,
        "tags": [
            "one-stroke",
            "benchmark:a1",
            "capability:direct",
            f"difficulty:{difficulty}",
            f"solution:{'yes' if solvable else 'no'}",
            f"source:{source}",
        ],
    }


def build_a1_records() -> list[dict[str, object]]:
    specs = [
        # Easy: 4-6 vertices, a required start, seven solvable and three adversarial.
        ("a1-easy-01", "easy", "A B C D", "A-B B-C C-D", True, "A", "D", "graph-atlas-inspired"),
        ("a1-easy-02", "easy", "A B C D", "A-B B-C C-D D-A", True, "A", "A", "graph-atlas-inspired"),
        ("a1-easy-03", "easy", "A B C D", "A-B B-C C-A C-D", True, "D", "C", "graph-atlas-inspired"),
        ("a1-easy-04", "easy", "A B C D E", "A-B B-C C-D D-E E-A", True, "A", "A", "graph-atlas-inspired"),
        ("a1-easy-05", "easy", "A B C D E", "A-B B-C C-D D-A B-E E-C", True, "B", "C", "graph-atlas-inspired"),
        ("a1-easy-06", "easy", "A B C D E", "A-B B-C C-A C-D D-E E-C", True, "C", "C", "graph-atlas-inspired"),
        ("a1-easy-07", "easy", "A B C D E F", "A-B B-C C-D D-A C-E E-F F-C", True, "C", "C", "graph-atlas-inspired"),
        ("a1-easy-08", "easy", "A B C D", "A-B A-C A-D B-C B-D C-D", False, "A", None, "graph-atlas-inspired"),
        ("a1-easy-09", "easy", "A B C D E", "A-B A-C A-D A-E", False, "A", None, "graph-atlas-inspired"),
        ("a1-easy-10", "easy", "A B C D E F", "A-B B-C C-A D-E E-F F-D", False, "A", None, "graph-atlas-inspired"),
        # Medium: 7-9 vertices, bridge traps and locally plausible wrong branches.
        ("a1-medium-01", "medium", "A B C D E F G", "A-B B-C C-A C-D D-E E-C E-F F-G G-E", True, None, None, "handcrafted"),
        ("a1-medium-02", "medium", "A B C D E F G H", "A-B B-C C-A C-D D-E E-C E-F F-G G-E G-H", True, "H", "G", "handcrafted"),
        ("a1-medium-03", "medium", "A B C D E F G H", "A-B B-D A-C C-E E-D A-F F-G G-H H-D", True, None, None, "handcrafted"),
        ("a1-medium-04", "medium", "A B C D E F G", "A-B B-C C-A C-D D-E E-F F-G G-D", True, None, None, "handcrafted"),
        ("a1-medium-05", "medium", "A B C D E F G H", "A-B B-C C-D D-E E-F F-G G-H H-A A-E", True, None, None, "handcrafted"),
        ("a1-medium-06", "medium", "A B C D E F G H I", "A-B B-C C-D D-E E-F F-G G-H H-I I-A A-D D-G G-A", True, "A", "A", "handcrafted"),
        ("a1-medium-07", "medium", "A B C D E F G H I", "A-B B-C C-A C-D D-E E-F F-C F-G G-H H-I I-F", True, "C", "C", "handcrafted"),
        ("a1-medium-08", "medium", "A B C D E F G", "A-B B-C C-D D-E E-F F-G G-A A-D B-E", False, None, None, "handcrafted"),
        ("a1-medium-09", "medium", "A B C D E F G H", "A-B B-C C-D D-A E-F F-G G-H H-E", False, None, None, "handcrafted"),
        ("a1-medium-10", "medium", "A B C D E F G", "A-B A-C A-D A-E A-F A-G", False, None, None, "handcrafted"),
        # Hard: 9-12 vertices, parallel edges and unsolvable contrasts.
        ("a1-hard-01", "hard", "A B C D E F G H I", "A-B B-C C-D D-E E-F F-G G-H H-I I-A A-D D-G G-A B-E E-H H-B", True, "A", "A", "cocos-inspired-handcrafted"),
        ("a1-hard-02", "hard", "A B C D E F G H I J", "A-B B-C C-J A-D D-E E-F F-J A-G G-H H-I I-J", True, None, None, "cocos-inspired-handcrafted"),
        ("a1-hard-03", "hard", "A B C D E F G H I J", "A-B B-C C-D D-E E-F F-G G-H H-I I-J J-A A-F A-F", True, "A", "A", "cocos-inspired-handcrafted"),
        ("a1-hard-04", "hard", "A B C D E F G H I J K", "A-B B-C C-D D-E E-A F-G G-H H-I I-J J-K K-F E-F", True, None, None, "handcrafted"),
        ("a1-hard-05", "hard", "A B C D E F G H I J K L", "A-B B-C C-D D-E E-F F-G G-H H-I I-J J-K K-L L-A A-D D-G G-A C-H H-K K-C", True, "A", "A", "handcrafted"),
        ("a1-hard-06", "hard", "A B C D E F G H I", "A-B B-C C-D D-E E-F F-G G-H H-I I-A A-E B-G B-G", True, None, None, "cocos-inspired-handcrafted"),
        ("a1-hard-07", "hard", "A B C D E F G H I J", "A-B B-C C-A A-D D-E E-F F-A A-G G-H H-I I-J J-A", True, "A", "A", "handcrafted"),
        ("a1-hard-08", "hard", "A B C D E F G H I", "A-B B-C C-D D-E E-F F-G G-H H-I I-A A-D B-F", False, None, None, "handcrafted"),
        ("a1-hard-09", "hard", "A B C D E F G H I J", "A-B B-C C-D D-E E-A F-G G-H H-I I-J J-F", False, None, None, "handcrafted"),
        ("a1-hard-10", "hard", "A B C D E F G H I", "A-B B-C C-D D-E E-F F-G G-H H-I I-A A-D B-F C-H", False, None, None, "handcrafted"),
    ]
    return [
        _direct_record(
            task_id,
            difficulty,
            vertices,
            edge_text,
            solvable=solvable,
            start=start,
            end=end,
            source=source,
        )
        for task_id, difficulty, vertices, edge_text, solvable, start, end, source in specs
    ]


def _cycle(vertices: list[str]) -> list[list[str]]:
    return [
        [vertices[index], vertices[(index + 1) % len(vertices)]]
        for index in range(len(vertices))
    ]


def _path_edge_ids(path: list[str], edges: list[list[str]]) -> list[str]:
    used = set[int]()
    answer: list[str] = []
    width = max(2, len(str(len(edges))))
    for a, b in zip(path, path[1:]):
        target = _canonical((a, b))
        edge_index = next(
            index
            for index, edge in enumerate(edges)
            if index not in used and _canonical(tuple(edge)) == target
        )
        used.add(edge_index)
        answer.append(f"e{edge_index + 1:0{width}d}")
    return answer


def _history_events(
    path: list[str],
    edges: list[list[str]],
    event_count: int,
    *,
    include_undo: bool,
) -> list[dict[str, str]]:
    path_ids = _path_edge_ids(path, edges)
    correct_moves = event_count - (2 if include_undo else 0)
    if not 1 <= correct_moves < len(edges):
        raise ValueError("history must leave at least one edge for final completion")
    events: list[dict[str, str]] = []
    used = set[str]()
    injection_index: int | None = None
    wrong_edge: tuple[str, str] | None = None
    if include_undo:
        width = max(2, len(str(len(edges))))
        for prefix_length in range(3, correct_moves):
            current = path[prefix_length]
            used_prefix = set(path_ids[:prefix_length])
            next_id = path_ids[prefix_length]
            for edge_index, (a, b) in enumerate(edges):
                edge_id = f"e{edge_index + 1:0{width}d}"
                if edge_id in used_prefix or edge_id == next_id or current not in {a, b}:
                    continue
                neighbor = b if a == current else a
                injection_index = prefix_length
                wrong_edge = (edge_id, neighbor)
                break
            if wrong_edge is not None:
                break
        if wrong_edge is None:
            raise ValueError("unable to inject a reversible wrong move")

    for path_index in range(correct_moves):
        if include_undo and path_index == injection_index and wrong_edge is not None:
            edge_id, neighbor = wrong_edge
            current = path[path_index]
            events.append(
                {"action": "move", "edge_id": edge_id, "from": current, "to": neighbor}
            )
            events.append(
                {"action": "undo", "edge_id": edge_id, "from": neighbor, "to": current}
            )
        edge_id = path_ids[path_index]
        if edge_id in used:
            raise ValueError("solution edge IDs must be unique")
        used.add(edge_id)
        events.append(
            {
                "action": "move",
                "edge_id": edge_id,
                "from": path[path_index],
                "to": path[path_index + 1],
            }
        )
    if len(events) != event_count:
        raise ValueError("wrong history event count")
    return events


def _history_record(difficulty: str, number: int) -> dict[str, object]:
    offsets = {"easy": 0, "medium": 1, "hard": 2}
    base_nodes = {"easy": 4, "medium": 7, "hard": 9}[difficulty]
    node_span = {"easy": 3, "medium": 3, "hard": 4}[difficulty]
    node_count = base_nodes + ((number - 1) % node_span)
    vertices = [chr(ord("A") + index) for index in range(node_count)]
    edges = _cycle(vertices)
    extra_cycle_count = {"easy": 1, "medium": 2, "hard": 4}[difficulty]
    offset = offsets[difficulty] + number
    for cycle_index in range(extra_cycle_count):
        a = vertices[(offset + cycle_index * 2) % node_count]
        b = vertices[(offset + cycle_index * 2 + 1) % node_count]
        c = vertices[(offset + cycle_index * 2 + 2) % node_count]
        edges.extend([[a, b], [b, c], [c, a]])
    if number % 2 == 0:
        chord = [vertices[0], vertices[node_count // 2]]
        edges.append(chord)
        start, end = chord
    else:
        start = vertices[0]
        end = start
    path = _euler_path(vertices, edges, start)
    if path is None or path[-1] != end:
        raise ValueError(f"failed to construct history graph {difficulty}-{number}")
    target = {
        "easy": 4 + ((number - 1) % 3),
        "medium": 7 + ((number - 1) % 6),
        "hard": 12 + ((number - 1) % 9),
    }[difficulty]
    events = _history_events(
        path,
        edges,
        target,
        include_undo=difficulty == "hard",
    )
    return {
        "id": f"a3-{difficulty}-{number:02d}",
        "capability": "history_memory",
        "difficulty": difficulty,
        "vertices": vertices,
        "edges": edges,
        "start": start,
        "end": end,
        "solution_exists": True,
        "solution_path": path,
        "history_events": events,
        "tags": [
            "one-stroke",
            "benchmark:a3",
            "capability:history-memory",
            f"difficulty:{difficulty}",
            f"history-steps:{len(events)}",
            "history-comparison:incremental-vs-step-only",
            "source:dataset-inspired-handcrafted",
            *( ["history:error-recovery"] if difficulty == "hard" else [] ),
        ],
    }


def build_a3_records() -> list[dict[str, object]]:
    return [
        _history_record(difficulty, number)
        for difficulty in ("easy", "medium", "hard")
        for number in range(1, 11)
    ]


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    a1 = build_a1_records()
    a3 = build_a3_records()
    if len(a1) != 30 or len(a3) != 30:
        raise ValueError("A1 and A3 must each contain exactly 30 tasks")
    _write_jsonl(OUTPUT_DIR / "a1_direct.jsonl", a1)
    _write_jsonl(OUTPUT_DIR / "a3_history.jsonl", a3)
    print(f"wrote {len(a1)} A1 tasks and {len(a3)} A3 tasks to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

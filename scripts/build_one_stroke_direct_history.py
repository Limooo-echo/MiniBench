#!/usr/bin/env python3
"""Build the deterministic MiniBench direct/history one-stroke task sets."""

from __future__ import annotations

from collections import Counter
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Iterable

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "one_stroke"
SEED = 20261013
DEFAULT_ATLAS = OUTPUT_DIR / "sources/atlas.dat.gz"
sys.path.insert(0, str(ROOT / "src"))
from minibench.datasets.one_stroke.dataset import (  # noqa: E402
    one_stroke_edge_ids,
    one_stroke_task_from_dict,
    simulate_one_stroke_history,
)
from minibench.datasets.one_stroke.multimodal import one_stroke_layout_issues  # noqa: E402


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
            "benchmark:direct",
            "capability:direct",
            f"difficulty:{difficulty}",
            f"solution:{'yes' if solvable else 'no'}",
            f"source:{source}",
        ],
    }


def legacy_direct_records() -> list[dict[str, object]]:
    specs = [
        # Easy: 4-6 vertices, a required start, seven solvable and three adversarial.
        ("direct-easy-01", "easy", "A B C D", "A-B B-C C-D", True, "A", "D", "graph-atlas-inspired"),
        ("direct-easy-02", "easy", "A B C D", "A-B B-C C-D D-A", True, "A", "A", "graph-atlas-inspired"),
        ("direct-easy-03", "easy", "A B C D", "A-B B-C C-A C-D", True, "D", "C", "graph-atlas-inspired"),
        ("direct-easy-04", "easy", "A B C D E", "A-B B-C C-D D-E E-A", True, "A", "A", "graph-atlas-inspired"),
        ("direct-easy-05", "easy", "A B C D E", "A-B B-C C-D D-A B-E E-C", True, "B", "C", "graph-atlas-inspired"),
        ("direct-easy-06", "easy", "A B C D E", "A-B B-C C-A C-D D-E E-C", True, "C", "C", "graph-atlas-inspired"),
        ("direct-easy-07", "easy", "A B C D E F", "A-B B-C C-D D-A C-E E-F F-C", True, "C", "C", "graph-atlas-inspired"),
        ("direct-easy-08", "easy", "A B C D", "A-B A-C A-D B-C B-D C-D", False, "A", None, "graph-atlas-inspired"),
        ("direct-easy-09", "easy", "A B C D E", "A-B A-C A-D A-E", False, "A", None, "graph-atlas-inspired"),
        ("direct-easy-10", "easy", "A B C D E F", "A-B B-C C-A D-E E-F F-D", False, "A", None, "graph-atlas-inspired"),
        # Medium: 7-9 vertices, bridge traps and locally plausible wrong branches.
        ("direct-medium-01", "medium", "A B C D E F G", "A-B B-C C-A C-D D-E E-C E-F F-G G-E", True, None, None, "handcrafted"),
        ("direct-medium-02", "medium", "A B C D E F G H", "A-B B-C C-A C-D D-E E-C E-F F-G G-E G-H", True, "H", "G", "handcrafted"),
        ("direct-medium-03", "medium", "A B C D E F G H", "A-B B-D A-C C-E E-D A-F F-G G-H H-D", True, None, None, "handcrafted"),
        ("direct-medium-04", "medium", "A B C D E F G", "A-B B-C C-A C-D D-E E-F F-G G-D", True, None, None, "handcrafted"),
        ("direct-medium-05", "medium", "A B C D E F G H", "A-B B-C C-D D-E E-F F-G G-H H-A A-E", True, None, None, "handcrafted"),
        ("direct-medium-06", "medium", "A B C D E F G H I", "A-B B-C C-D D-E E-F F-G G-H H-I I-A A-D D-G G-A", True, "A", "A", "handcrafted"),
        ("direct-medium-07", "medium", "A B C D E F G H I", "A-B B-C C-A C-D D-E E-F F-C F-G G-H H-I I-F", True, "C", "C", "handcrafted"),
        ("direct-medium-08", "medium", "A B C D E F G", "A-B B-C C-D D-E E-F F-G G-A A-D B-E", False, None, None, "handcrafted"),
        ("direct-medium-09", "medium", "A B C D E F G H", "A-B B-C C-D D-A E-F F-G G-H H-E", False, None, None, "handcrafted"),
        ("direct-medium-10", "medium", "A B C D E F G", "A-B A-C A-D A-E A-F A-G", False, None, None, "handcrafted"),
        # Hard: 9-12 vertices, parallel edges and unsolvable contrasts.
        ("direct-hard-01", "hard", "A B C D E F G H I", "A-B B-C C-D D-E E-F F-G G-H H-I I-A A-D D-G G-A B-E E-H H-B", True, "A", "A", "cocos-inspired-handcrafted"),
        ("direct-hard-02", "hard", "A B C D E F G H I J", "A-B B-C C-J A-D D-E E-F F-J A-G G-H H-I I-J", True, None, None, "cocos-inspired-handcrafted"),
        ("direct-hard-03", "hard", "A B C D E F G H I J", "A-B B-C C-D D-E E-F F-G G-H H-I I-J J-A A-F A-F", True, "A", "A", "cocos-inspired-handcrafted"),
        ("direct-hard-04", "hard", "A B C D E F G H I J K", "A-B B-C C-D D-E E-A F-G G-H H-I I-J J-K K-F E-F", True, None, None, "handcrafted"),
        ("direct-hard-05", "hard", "A B C D E F G H I J K L", "A-B B-C C-D D-E E-F F-G G-H H-I I-J J-K K-L L-A A-D D-G G-A C-H H-K K-C", True, "A", "A", "handcrafted"),
        ("direct-hard-06", "hard", "A B C D E F G H I", "A-B B-C C-D D-E E-F F-G G-H H-I I-A A-E B-G B-G", True, None, None, "cocos-inspired-handcrafted"),
        ("direct-hard-07", "hard", "A B C D E F G H I J", "A-B B-C C-A A-D D-E E-F F-A A-G G-H H-I I-J J-A", True, "A", "A", "handcrafted"),
        ("direct-hard-08", "hard", "A B C D E F G H I", "A-B B-C C-D D-E E-F F-G G-H H-I I-A A-D B-F", False, None, None, "handcrafted"),
        ("direct-hard-09", "hard", "A B C D E F G H I J", "A-B B-C C-D D-E E-A F-G G-H H-I I-J J-F", False, None, None, "handcrafted"),
        ("direct-hard-10", "hard", "A B C D E F G H I", "A-B B-C C-D D-E E-F F-G G-H H-I I-A A-D B-F C-H", False, None, None, "handcrafted"),
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
        "id": f"history-{difficulty}-{number:02d}",
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
            "benchmark:history",
            "capability:history-memory",
            f"difficulty:{difficulty}",
            f"history-steps:{len(events)}",
            "history-comparison:incremental-vs-step-only",
            "source:dataset-inspired-handcrafted",
            *( ["history:error-recovery"] if difficulty == "hard" else [] ),
        ],
    }


def legacy_history_records() -> list[dict[str, object]]:
    return [
        _history_record(difficulty, number)
        for difficulty in ("easy", "medium", "hard")
        for number in range(1, 11)
    ]


def graph_for_record(raw: dict[str, object]) -> nx.MultiGraph:
    """Represent multiplicity and required endpoint roles for isomorphism."""
    graph = nx.MultiGraph()
    for vertex in raw["vertices"]:
        graph.add_node(vertex, role=(vertex == raw.get("start"), vertex == raw.get("end")))
    graph.add_edges_from(raw["edges"])
    return graph


def graph_signature(graph: nx.MultiGraph) -> tuple[object, ...]:
    return (
        len(graph), graph.number_of_edges(),
        tuple(sorted((graph.degree(v), graph.nodes[v]["role"]) for v in graph)),
        tuple(sorted(Counter(tuple(sorted((a, b))) for a, b in graph.edges()).values())),
        tuple(sorted(len(c) for c in nx.connected_components(graph))),
    )


def graphs_isomorphic(a: nx.MultiGraph, b: nx.MultiGraph) -> bool:
    return graph_signature(a) == graph_signature(b) and nx.is_isomorphic(
        a, b, node_match=lambda x, y: x["role"] == y["role"]
    )


def _distinct(raw: dict[str, object], seen: list[nx.MultiGraph]) -> bool:
    graph = graph_for_record(raw)
    if any(graphs_isomorphic(graph, previous) for previous in seen):
        return False
    seen.append(graph)
    return True


def visual_issues(raw: dict[str, object]) -> list[str]:
    render_raw = {**raw, "id": "multimodal-" + str(raw["id"]).removeprefix("direct-")}
    return one_stroke_layout_issues(one_stroke_task_from_dict(render_raw))


def _atlas_records(path: Path) -> list[tuple[int, nx.Graph]]:
    """Read the pinned BSD-3-Clause archive, never an unlicensed level source."""
    records: list[tuple[int, nx.Graph]] = []
    graph: nx.Graph | None = None
    index = 0
    with gzip.open(path, "rt", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("GRAPH"):
                if graph is not None:
                    records.append((index, graph))
                index = int(line.split()[1])
                graph = nx.Graph()
            elif line.startswith("NODES"):
                assert graph is not None
                graph.add_nodes_from(range(int(line.split()[1])))
            elif line.strip():
                assert graph is not None
                graph.add_edge(*map(int, line.split()))
    if graph is not None:
        records.append((index, graph))
    return records


def _from_edges(
    task_id: str, difficulty: str, vertices: list[str], edges: list[list[str]],
    *, start: str | None, end: str | None, provenance: dict[str, object],
) -> dict[str, object]:
    probe = {"vertices": vertices, "edges": edges, "start": start, "end": end}
    facts = _networkx_facts(probe)
    path = _euler_path(vertices, edges, start) if facts["solution_exists"] else None
    raw = {
        "id": task_id, "capability": "direct", "difficulty": difficulty,
        **probe, "solution_exists": facts["solution_exists"], "solution_path": path,
        "tags": ["one-stroke", "benchmark:direct", "capability:direct",
                 f"difficulty:{difficulty}", f"solution:{'yes' if path else 'no'}",
                 f"source:{provenance['kind']}"],
        "provenance": provenance,
    }
    validate_record(raw)
    return raw


def _networkx_facts(raw: dict[str, object]) -> dict[str, object]:
    graph = nx.MultiGraph()
    graph.add_nodes_from(raw["vertices"])
    graph.add_edges_from(raw["edges"])
    active = graph.subgraph([v for v in graph if graph.degree(v)]).copy()
    connected = bool(active) and nx.is_connected(active)
    odd = sorted(v for v in graph if graph.degree(v) % 2)
    start, end = raw.get("start"), raw.get("end")
    reason = None
    if not connected:
        reason = "disconnected_edge_components"
    elif len(odd) not in (0, 2):
        reason = "odd_degree_count"
    elif start is not None and graph.degree(start) == 0:
        reason = "isolated_start"
    elif end is not None and graph.degree(end) == 0:
        reason = "isolated_end"
    elif len(odd) == 2 and ((start is not None and start not in odd) or (end is not None and end not in odd)):
        reason = "endpoint_not_odd"
    elif start is not None and end is not None and ((not odd and start != end) or (odd and start == end)):
        reason = "endpoint_pair_incompatible"
    exists = reason is None
    # NetworkX's Euler routine independently checks trail existence and consumes
    # the multiedges. The explicit end constraint is not part of its API.
    witness = None
    if exists:
        source = start if start is not None else (odd[0] if odd else sorted(active)[0])
        trail = list(nx.eulerian_path(active, source=source, keys=True))
        witness = [source, *(b for _, b, _ in trail)]
        if end is not None and witness[-1] != end:
            if start is None:
                witness.reverse()
            if witness[-1] != end:
                raise ValueError("independent Euler witness does not meet required end")
    return {
        "solution_exists": exists, "connected_on_edges": connected,
        "edge_components": nx.number_connected_components(active) if active else 0,
        "degrees": dict(sorted(graph.degree())), "odd_vertices": odd,
        "negative_proof": reason, "networkx_witness": witness,
    }


def validate_record(raw: dict[str, object]) -> dict[str, object]:
    """Cross-check the production loader against an independent library oracle."""
    task = one_stroke_task_from_dict(raw)
    facts = _networkx_facts(raw)
    if task.solution_exists != facts["solution_exists"]:
        raise ValueError(f"{task.id}: loader/NetworkX disagreement")
    if task.solution_exists:
        if task.solution_path is None:
            raise ValueError(f"{task.id}: missing full-path witness")
        traversed = Counter(_canonical(e) for e in zip(task.solution_path, task.solution_path[1:]))
        if traversed != Counter(_canonical(e) for e in task.edges):
            raise ValueError(f"{task.id}: oracle does not consume exact multiset")
    if task.capability == "history_memory":
        # Independently replay every move and strict LIFO undo rather than only
        # checking the final production state.
        edge_map = dict(zip(one_stroke_edge_ids(task.edges), task.edges))
        current = task.start
        stack: list[tuple[str, str, str]] = []
        used: set[str] = set()
        for event in task.history_events:
            if event.from_vertex != current or _canonical(edge_map[event.edge_id]) != _canonical((event.from_vertex, event.to_vertex)):
                raise ValueError(f"{task.id}: illegal event endpoints")
            if event.action == "move":
                if event.edge_id in used:
                    raise ValueError(f"{task.id}: reused event edge")
                used.add(event.edge_id)
                stack.append((event.edge_id, event.from_vertex, event.to_vertex))
            else:
                if not stack or stack.pop() != (event.edge_id, event.to_vertex, event.from_vertex):
                    raise ValueError(f"{task.id}: non-LIFO undo")
                used.remove(event.edge_id)
            current = event.to_vertex
        state = simulate_one_stroke_history(task)
        if current != state.current_vertex or used != set(state.used_edge_ids):
            raise ValueError(f"{task.id}: independent history state disagreement")
        residual = {"vertices": list(task.vertices), "edges": [list(e) for i, e in edge_map.items() if i not in used], "start": current, "end": task.end}
        residual_facts = _networkx_facts(residual)
        if not residual_facts["solution_exists"]:
            raise ValueError(f"{task.id}: no residual completion")
        facts["history"] = {"events_valid": True, "lifo_valid": True, "event_count": len(task.history_events), "current_vertex": current, "used_edge_ids": sorted(used), "remaining_edge_ids": list(state.remaining_edge_ids), "completion_path": residual_facts["networkx_witness"]}
    return {"id": task.id, "difficulty": task.difficulty, "production_loader_valid": True, "oracle_multiset_valid": task.solution_exists, **facts}


def _random_edges(rng: random.Random, difficulty: str, *, history: bool, solvable: bool) -> tuple[list[str], list[list[str]], str | None, str | None]:
    bounds = {"easy": (4, 6), "medium": (7, 9), "hard": (9, 12)}
    n = rng.randint(*bounds[difficulty])
    vertices = [chr(65 + i) for i in range(n)]
    order = vertices[:]
    rng.shuffle(order)
    edges = _cycle(order)
    existing = Counter(_canonical(tuple(e)) for e in edges)
    cycles = rng.randint(1, 3 if difficulty == "easy" else 4 if difficulty == "medium" else 5)
    for _ in range(cycles):
        for _ in range(100):
            triangle = rng.sample(vertices, 3)
            cycle = _cycle(triangle)
            if (history or difficulty == "hard"):
                allowed = all(existing[_canonical(tuple(e))] < 2 for e in cycle)
            else:
                allowed = all(not existing[_canonical(tuple(e))] for e in cycle)
            if allowed:
                edges.extend(cycle)
                existing.update(_canonical(tuple(e)) for e in cycle)
                break
    # Circuits and open trails vary structure as well as endpoint roles.
    chord = rng.sample(vertices, 2)
    if rng.choice((True, False)) and (history or difficulty == "hard" or not existing[_canonical(tuple(chord))]):
        edges.append(chord)
        start, end = chord
    else:
        start = rng.choice(vertices)
        end = start
    if not solvable:
        if rng.choice((True, False)):
            # Endpoint contrast: valid unconstrained graph, invalid fixed pair.
            end = rng.choice([v for v in vertices if v != end])
        else:
            # Ordinary connected graphs with more than two odd vertices.
            a, b = rng.sample(vertices, 2)
            if difficulty == "hard" or not existing[_canonical((a, b))]:
                edges.append([a, b])
            a, b = rng.sample(vertices, 2)
            if difficulty == "hard" or not existing[_canonical((a, b))]:
                edges.append([a, b])
            start = end = None
    return vertices, edges, start, end


def build_direct_records(*, total: int = 100, seed: int = SEED, atlas_path: Path = DEFAULT_ATLAS) -> list[dict[str, object]]:
    if total != 100:
        raise ValueError("formal release --total must be 100")
    records = legacy_direct_records()
    seen = [graph_for_record(r) for r in records]
    rng = random.Random(seed)
    atlas = _atlas_records(atlas_path)
    atlas_hash = hashlib.sha256(atlas_path.read_bytes()).hexdigest()
    # Keep all existing 30 records byte-equivalent at the record level.
    for difficulty, yes_goal in (("easy", 24), ("medium", 23), ("hard", 23)):
        number = 11
        for wanted, goal in ((True, yes_goal), (False, 10)):
            count = sum(r["difficulty"] == difficulty and r["solution_exists"] == wanted for r in records)
            candidates: list[tuple[list[str], list[list[str]], str | None, str | None, dict[str, object]]] = []
            if difficulty == "easy":
                shuffled = [item for item in atlas if 4 <= len(item[1]) <= 6 and all(item[1].degree(v) for v in item[1])]
                rng.shuffle(shuffled)
                for index, graph in shuffled:
                    vertices = [chr(65 + i) for i in range(len(graph))]
                    edges = [[vertices[a], vertices[b]] for a, b in sorted(graph.edges())]
                    odd = [vertices[v] for v in graph if graph.degree(v) % 2]
                    starts = vertices if not odd else odd
                    for start in starts:
                        end = next((v for v in odd if v != start), start)
                        candidates.append((vertices, edges, start, end if wanted else None, {"kind": "networkx-graph-atlas", "atlas_index": index, "atlas_sha256": atlas_hash, "license": "BSD-3-Clause", "seed": seed}))
            attempts = 0
            while count < goal:
                attempts += 1
                if attempts > 50000:
                    raise RuntimeError(f"candidate supply exhausted: {difficulty}/{wanted}")
                if candidates:
                    vertices, edges, start, end, provenance = candidates.pop(0)
                else:
                    vertices, edges, start, end = _random_edges(rng, difficulty, history=False, solvable=wanted)
                    provenance = {"kind": "seeded-cycle-chord-generator", "seed": seed, "candidate_attempt": attempts}
                raw = _from_edges(f"direct-{difficulty}-{number:02d}", difficulty, vertices, edges, start=start, end=end, provenance=provenance)
                if raw["solution_exists"] != wanted or visual_issues(raw) or not _distinct(raw, seen):
                    continue
                records.append(raw)
                count += 1
                number += 1
    return sorted(records, key=lambda r: (("easy", "medium", "hard").index(r["difficulty"]), r["id"]))


def build_history_records(*, total: int = 100, seed: int = SEED) -> list[dict[str, object]]:
    if total != 100:
        raise ValueError("formal release --total must be 100")
    records = [r for r in legacy_history_records() if r["id"] != "history-easy-09"]
    # Baseline graph-isomorphic histories remain; only new candidates are
    # compared against every baseline and previously accepted new graph.
    seen = [graph_for_record(r) for r in records]
    rng = random.Random(seed)
    for difficulty, goal in (("easy", 34), ("medium", 33), ("hard", 33)):
        number = 11
        attempts = 0
        while sum(r["difficulty"] == difficulty for r in records) < goal:
            attempts += 1
            if attempts > 50000:
                raise RuntimeError(f"history candidate supply exhausted: {difficulty}")
            vertices, edges, start, end = _random_edges(rng, difficulty, history=True, solvable=True)
            raw = _from_edges(f"history-{difficulty}-{number:02d}", difficulty, vertices, edges, start=start, end=end, provenance={"kind": "seeded-cycle-chord-generator", "seed": seed, "candidate_attempt": attempts})
            target = rng.randint(*{"easy": (4, 6), "medium": (7, 12), "hard": (12, 20)}[difficulty])
            if target - (2 if difficulty == "hard" else 0) >= len(edges):
                continue
            try:
                events = _history_events(raw["solution_path"], edges, target, include_undo=difficulty == "hard")
            except ValueError:
                continue
            raw.update(capability="history_memory", history_events=events)
            raw["tags"] = ["one-stroke", "benchmark:history", "capability:history-memory", f"difficulty:{difficulty}", f"history-steps:{target}", "history-comparison:incremental-vs-step-only", "source:seeded-cycle-chord-generator", *( ["history:error-recovery"] if difficulty == "hard" else [] )]
            validate_record(raw)
            if not _distinct(raw, seen):
                continue
            records.append(raw)
            number += 1
    return sorted(records, key=lambda r: (("easy", "medium", "hard").index(r["difficulty"]), r["id"]))


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )


def copy_frozen_legacy(output_dir: Path) -> None:
    """Preserve archived input bytes so old result fingerprints stay valid."""
    source = OUTPUT_DIR / "legacy/pre-20261013"
    destination = output_dir / "legacy/pre-20261013"
    for name in ("direct.jsonl", "history.jsonl", "multimodal.jsonl"):
        archived = source / name
        if not archived.is_file():
            raise FileNotFoundError(f"missing frozen pre-expansion archive: {archived}")
        if archived.resolve() != (destination / name).resolve():
            destination.mkdir(parents=True, exist_ok=True)
            (destination / name).write_bytes(archived.read_bytes())
    for archived in sorted((source / "images").glob("*.png")):
        target = destination / "images" / archived.name
        if archived.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archived.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    args = parser.parse_args()
    direct = build_direct_records(total=args.total, seed=args.seed, atlas_path=args.atlas)
    history = build_history_records(total=args.total, seed=args.seed)
    validation = {"direct": [validate_record(r) for r in direct], "history": [validate_record(r) for r in history]}
    for raw, proof in zip(direct, validation["direct"]):
        proof["visual_clearance"] = (
            {"node_node_minimum": 0.19, "node_edge_minimum": 0.10, "issues": visual_issues(raw)}
            if "provenance" in raw else {"preserved_original_image": True}
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "direct.jsonl", direct)
    _write_jsonl(args.output_dir / "history.jsonl", history)
    report = {"release": "20261013", "seed": args.seed, "networkx_version": nx.__version__, "atlas_path": str(args.atlas), "atlas_sha256": hashlib.sha256(args.atlas.read_bytes()).hexdigest(), "atlas_license": "BSD-3-Clause", "original_direct_preserved": 30, "original_history_preserved": 29, "retired_history_ids": ["history-easy-09"], "retirement_reason": "Exact graph+events duplicate of history-easy-03", "quotas": {"direct": dict(Counter((r["difficulty"] + (":yes" if r["solution_exists"] else ":no")) for r in direct)), "history": dict(Counter(r["difficulty"] for r in history))}, "validation": validation}
    (args.output_dir / "validation-20261013.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    copy_frozen_legacy(args.output_dir)
    print(json.dumps({"output": str(args.output_dir), "direct": len(direct), "history": len(history), "quotas": report["quotas"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

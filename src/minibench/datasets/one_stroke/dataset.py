from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OneStrokeHistoryEvent:
    action: str
    edge_id: str
    from_vertex: str
    to_vertex: str


@dataclass(frozen=True)
class OneStrokeHistoryState:
    current_vertex: str
    used_edge_ids: tuple[str, ...]
    remaining_edge_ids: tuple[str, ...]


@dataclass(frozen=True)
class OneStrokeTask:
    id: str
    vertices: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    start: str | None
    end: str | None
    solution_exists: bool
    solution_path: tuple[str, ...] | None
    capability: str
    difficulty: str
    history_events: tuple[OneStrokeHistoryEvent, ...]
    tags: tuple[str, ...]


def default_one_stroke_tasks_path() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "one_stroke" / "tasks.jsonl"


def _require_string_list(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{raw.get('id', '<unknown>')}: {key} must be a list of strings")
    return tuple(value)


def _optional_vertex(raw: dict[str, Any], key: str, vertices: set[str]) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or value not in vertices:
        raise ValueError(f"{raw.get('id', '<unknown>')}: {key} must be a known vertex")
    return value


def _optional_solution_path(
    raw: dict[str, Any],
    vertices: set[str],
) -> tuple[str, ...] | None:
    value = raw.get("solution_path")
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            f"{raw.get('id', '<unknown>')}: solution_path must be a list of strings"
        )
    unknown = sorted(set(value) - vertices)
    if unknown:
        raise ValueError(
            f"{raw.get('id', '<unknown>')}: solution_path references unknown vertices "
            f"{', '.join(unknown)}"
        )
    return tuple(value)


def has_one_stroke_solution(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    *,
    start: str | None = None,
    end: str | None = None,
) -> bool:
    return _has_euler_trail(vertices, edges, start=start, end=end)


def _has_euler_trail(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    *,
    start: str | None,
    end: str | None,
) -> bool:
    degree = Counter[str]()
    seen = set[str]()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
        seen.add(a)
        seen.add(b)

    if edges and not _is_connected_on_edges(vertices, edges, seen):
        return False

    odd = {vertex for vertex, count in degree.items() if count % 2 == 1}
    if len(odd) not in {0, 2}:
        return False

    if start is not None and degree[start] == 0 and edges:
        return False
    if end is not None and degree[end] == 0 and edges:
        return False

    if len(odd) == 2:
        if start is not None and start not in odd:
            return False
        if end is not None and end not in odd:
            return False
        return start is None or end is None or start != end
    return start is None or end is None or start == end


def _is_connected_on_edges(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    seen: set[str],
) -> bool:
    adjacency = {vertex: set[str]() for vertex in vertices}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    stack = [next(iter(seen))]
    visited = set[str]()
    while stack:
        vertex = stack.pop()
        if vertex in visited:
            continue
        visited.add(vertex)
        stack.extend(adjacency[vertex] - visited)
    return seen <= visited


def one_stroke_task_from_dict(raw: dict[str, Any]) -> OneStrokeTask:
    if not isinstance(raw.get("id"), str) or not raw["id"]:
        raise ValueError("task id must be a non-empty string")

    vertices = _require_string_list(raw, "vertices")
    if not vertices:
        raise ValueError(f"{raw['id']}: vertices must not be empty")
    if len(set(vertices)) != len(vertices):
        raise ValueError(f"{raw['id']}: vertices must be unique")
    vertex_set = set(vertices)

    raw_edges = raw.get("edges")
    if not isinstance(raw_edges, list) or not raw_edges:
        raise ValueError(f"{raw['id']}: edges must be a non-empty list")

    edges: list[tuple[str, str]] = []
    for index, item in enumerate(raw_edges, start=1):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(vertex, str) for vertex in item)
        ):
            raise ValueError(f"{raw['id']}: edge {index} must be [from, to]")
        a, b = item
        if a not in vertex_set or b not in vertex_set:
            raise ValueError(f"{raw['id']}: edge {index} references an unknown vertex")
        if a == b:
            raise ValueError(f"{raw['id']}: edge {index} must not be a self-loop")
        edges.append((a, b))

    edge_tuple = tuple(edges)
    start = _optional_vertex(raw, "start", vertex_set)
    end = _optional_vertex(raw, "end", vertex_set)

    solution_exists = raw.get("solution_exists", True)
    if not isinstance(solution_exists, bool):
        raise ValueError(f"{raw['id']}: solution_exists must be true or false")

    actual_solution_exists = has_one_stroke_solution(
        vertices,
        edge_tuple,
        start=start,
        end=end,
    )
    if solution_exists and not actual_solution_exists:
        raise ValueError(f"{raw['id']}: graph has no one-stroke solution")
    if not solution_exists and actual_solution_exists:
        raise ValueError(
            f"{raw['id']}: marked solution_exists=false but graph has a one-stroke solution"
        )

    solution_path = _optional_solution_path(raw, vertex_set)
    if solution_path is not None:
        if not solution_exists:
            raise ValueError(
                f"{raw['id']}: solution_path must be null when solution_exists=false"
            )
        _validate_solution_path(
            raw["id"],
            edge_tuple,
            solution_path,
            start=start,
            end=end,
        )

    tags = _require_string_list(raw, "tags")
    capability = raw.get("capability", "direct")
    if not isinstance(capability, str) or capability not in {
        "direct",
        "history_memory",
    }:
        raise ValueError(
            f"{raw['id']}: capability must be direct or history_memory"
        )
    difficulty = raw.get("difficulty") or _tag_value(tags, "difficulty:") or "unknown"
    if not isinstance(difficulty, str):
        raise ValueError(f"{raw['id']}: difficulty must be a string")
    history_events = _parse_history_events(raw, vertex_set)

    task = OneStrokeTask(
        id=raw["id"],
        vertices=vertices,
        edges=edge_tuple,
        start=start,
        end=end,
        solution_exists=solution_exists,
        solution_path=solution_path,
        capability=capability,
        difficulty=difficulty,
        history_events=history_events,
        tags=tags,
    )
    if capability == "direct" and history_events:
        raise ValueError(f"{raw['id']}: direct tasks must not define history_events")
    if capability == "history_memory":
        if not solution_exists:
            raise ValueError(f"{raw['id']}: history tasks must be solvable")
        if start is None:
            raise ValueError(f"{raw['id']}: history tasks require a start vertex")
        if not history_events:
            raise ValueError(f"{raw['id']}: history tasks require history_events")
        state = simulate_one_stroke_history(task)
        remaining_edges = tuple(
            edge
            for edge_id, edge in zip(one_stroke_edge_ids(task.edges), task.edges)
            if edge_id in set(state.remaining_edge_ids)
        )
        if not has_one_stroke_solution(
            task.vertices,
            remaining_edges,
            start=state.current_vertex,
            end=task.end,
        ):
            raise ValueError(
                f"{raw['id']}: history leaves no valid one-stroke completion"
            )
    return task


def _parse_history_events(
    raw: dict[str, Any],
    vertices: set[str],
) -> tuple[OneStrokeHistoryEvent, ...]:
    value = raw.get("history_events", [])
    if not isinstance(value, list):
        raise ValueError(f"{raw['id']}: history_events must be a list")
    events: list[OneStrokeHistoryEvent] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{raw['id']}: history event {index} must be an object")
        action = item.get("action")
        edge_id = item.get("edge_id")
        from_vertex = item.get("from")
        to_vertex = item.get("to")
        if action not in {"move", "undo"}:
            raise ValueError(
                f"{raw['id']}: history event {index} action must be move or undo"
            )
        if not isinstance(edge_id, str) or not edge_id:
            raise ValueError(
                f"{raw['id']}: history event {index} edge_id must be a string"
            )
        if (
            not isinstance(from_vertex, str)
            or not isinstance(to_vertex, str)
            or from_vertex not in vertices
            or to_vertex not in vertices
        ):
            raise ValueError(
                f"{raw['id']}: history event {index} references an unknown vertex"
            )
        events.append(
            OneStrokeHistoryEvent(
                action=action,
                edge_id=edge_id,
                from_vertex=from_vertex,
                to_vertex=to_vertex,
            )
        )
    return tuple(events)


def one_stroke_edge_ids(
    edges: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    width = max(2, len(str(len(edges))))
    return tuple(f"e{index:0{width}d}" for index in range(1, len(edges) + 1))


def simulate_one_stroke_history(task: OneStrokeTask) -> OneStrokeHistoryState:
    if task.start is None:
        raise ValueError(f"{task.id}: cannot simulate history without a start vertex")
    edge_map = dict(zip(one_stroke_edge_ids(task.edges), task.edges))
    current = task.start
    used_stack: list[tuple[str, str, str]] = []
    used = set[str]()
    for index, event in enumerate(task.history_events, start=1):
        if event.edge_id not in edge_map:
            raise ValueError(
                f"{task.id}: history event {index} references unknown edge {event.edge_id}"
            )
        if event.from_vertex != current:
            raise ValueError(
                f"{task.id}: history event {index} starts at {event.from_vertex}, "
                f"but current vertex is {current}"
            )
        edge = edge_map[event.edge_id]
        if _canonical_edge(edge) != _canonical_edge(
            (event.from_vertex, event.to_vertex)
        ):
            raise ValueError(
                f"{task.id}: history event {index} does not traverse {event.edge_id}"
            )
        if event.action == "move":
            if event.edge_id in used:
                raise ValueError(
                    f"{task.id}: history event {index} reuses {event.edge_id}"
                )
            used.add(event.edge_id)
            used_stack.append(
                (event.edge_id, event.from_vertex, event.to_vertex)
            )
        else:
            if not used_stack:
                raise ValueError(f"{task.id}: history event {index} has nothing to undo")
            edge_id, previous_from, previous_to = used_stack[-1]
            if (
                edge_id != event.edge_id
                or previous_to != event.from_vertex
                or previous_from != event.to_vertex
            ):
                raise ValueError(
                    f"{task.id}: history event {index} must undo the latest move"
                )
            used_stack.pop()
            used.remove(event.edge_id)
        current = event.to_vertex
    all_edge_ids = one_stroke_edge_ids(task.edges)
    return OneStrokeHistoryState(
        current_vertex=current,
        used_edge_ids=tuple(edge_id for edge_id in all_edge_ids if edge_id in used),
        remaining_edge_ids=tuple(
            edge_id for edge_id in all_edge_ids if edge_id not in used
        ),
    )


def _tag_value(tags: tuple[str, ...], prefix: str) -> str | None:
    for tag in tags:
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return None


def _validate_solution_path(
    task_id: str,
    edges: tuple[tuple[str, str], ...],
    path: tuple[str, ...],
    *,
    start: str | None,
    end: str | None,
) -> None:
    expected_length = len(edges) + 1
    if len(path) != expected_length:
        raise ValueError(
            f"{task_id}: solution_path length must be {expected_length}, got {len(path)}"
        )
    if start is not None and path[0] != start:
        raise ValueError(f"{task_id}: solution_path does not start at {start}")
    if end is not None and path[-1] != end:
        raise ValueError(f"{task_id}: solution_path does not end at {end}")

    available_edges = Counter(_canonical_edge(edge) for edge in edges)
    used_edges = Counter(_canonical_edge(edge) for edge in zip(path, path[1:]))
    if used_edges != available_edges:
        raise ValueError(f"{task_id}: solution_path does not use every edge once")


def _canonical_edge(edge: tuple[str, str]) -> tuple[str, str]:
    a, b = edge
    return (a, b) if a <= b else (b, a)


def load_one_stroke_tasks(path: str | Path | None = None) -> list[OneStrokeTask]:
    task_path = Path(path) if path else default_one_stroke_tasks_path()
    tasks: list[OneStrokeTask] = []
    with task_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{task_path}:{line_number}: invalid JSON") from exc
            tasks.append(one_stroke_task_from_dict(raw))
    if not tasks:
        raise ValueError(f"{task_path} contains no tasks")
    return tasks

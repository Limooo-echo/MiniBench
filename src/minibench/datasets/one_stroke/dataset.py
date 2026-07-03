from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OneStrokeTask:
    id: str
    vertices: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    start: str | None
    end: str | None
    solution_exists: bool
    solution_path: tuple[str, ...] | None
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
        return (start is None or start in odd) and (end is None or end in odd)
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

    return OneStrokeTask(
        id=raw["id"],
        vertices=vertices,
        edges=edge_tuple,
        start=start,
        end=end,
        solution_exists=solution_exists,
        solution_path=solution_path,
        tags=_require_string_list(raw, "tags"),
    )


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

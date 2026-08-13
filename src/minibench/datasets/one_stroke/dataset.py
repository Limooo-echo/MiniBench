from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from minibench.datasets.one_stroke.rules import (
    OneStrokeRule,
    find_constrained_one_stroke_path,
    parse_one_stroke_rule,
    rules_for_mode,
    rules_are_semantically_equivalent,
    rules_form_conflicting_pair,
    validate_edge_path,
)


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
    solution_edge_path: tuple[str, ...] | None
    capability: str
    difficulty: str
    history_events: tuple[OneStrokeHistoryEvent, ...]
    rule_constraints: tuple[OneStrokeRule, ...]
    key_rule_id: str | None
    conflicting_rule: OneStrokeRule | None
    source_task_id: str | None
    image_variants: dict[str, Path]
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


def _optional_solution_edge_path(
    raw: dict[str, Any],
    edge_ids: tuple[str, ...],
) -> tuple[str, ...] | None:
    value = raw.get("solution_edge_path")
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            f"{raw.get('id', '<unknown>')}: solution_edge_path must be a list of strings"
        )
    unknown = sorted(set(value) - set(edge_ids))
    if unknown:
        raise ValueError(
            f"{raw.get('id', '<unknown>')}: solution_edge_path references unknown "
            f"edge IDs {', '.join(unknown)}"
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


def one_stroke_task_from_dict(
    raw: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> OneStrokeTask:
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
    edge_ids = one_stroke_edge_ids(edge_tuple)
    start = _optional_vertex(raw, "start", vertex_set)
    end = _optional_vertex(raw, "end", vertex_set)

    capability = raw.get("capability", "direct")
    if not isinstance(capability, str) or capability not in {
        "direct",
        "rule_condition",
        "history_memory",
        "multimodal",
    }:
        raise ValueError(
            f"{raw['id']}: capability must be direct, rule_condition, "
            "history_memory, or multimodal"
        )

    rule_constraints = _parse_rule_constraints(
        raw,
        vertices,
        edge_ids,
        edge_tuple,
    )
    key_rule_id = raw.get("key_rule_id")
    if key_rule_id is not None and not isinstance(key_rule_id, str):
        raise ValueError(f"{raw['id']}: key_rule_id must be a string or null")
    raw_conflicting_rule = raw.get("conflicting_rule")
    conflicting_rule = (
        parse_one_stroke_rule(
            raw_conflicting_rule,
            task_id=raw["id"],
            vertices=vertices,
            edge_ids=edge_ids,
            edge_count=len(edge_tuple),
            field="conflicting_rule",
        )
        if raw_conflicting_rule is not None
        else None
    )
    if conflicting_rule is not None:
        _validate_rule_edge_semantics(raw["id"], conflicting_rule, edge_ids, edge_tuple)

    solution_exists = raw.get("solution_exists", True)
    if not isinstance(solution_exists, bool):
        raise ValueError(f"{raw['id']}: solution_exists must be true or false")

    standard_solution_exists = has_one_stroke_solution(
        vertices,
        edge_tuple,
        start=start,
        end=end,
    )
    if capability == "rule_condition":
        if not standard_solution_exists:
            raise ValueError(f"{raw['id']}: rule-condition base graph must be solvable")
        actual_solution_exists = (
            find_constrained_one_stroke_path(
                vertices,
                edge_tuple,
                start=start,
                end=end,
                constraints=rule_constraints,
            )
            is not None
        )
        if solution_exists != actual_solution_exists:
            raise ValueError(
                f"{raw['id']}: solution_exists does not match full rule constraints"
            )
    else:
        if solution_exists and not standard_solution_exists:
            raise ValueError(f"{raw['id']}: graph has no one-stroke solution")
        if not solution_exists and standard_solution_exists:
            raise ValueError(
                f"{raw['id']}: marked solution_exists=false but graph has a "
                "one-stroke solution"
            )

    solution_path = _optional_solution_path(raw, vertex_set)
    solution_edge_path = _optional_solution_edge_path(raw, edge_ids)
    if solution_path is not None:
        if not solution_exists:
            raise ValueError(
                f"{raw['id']}: solution_path must be null when solution_exists=false"
            )
        if capability == "rule_condition":
            if solution_edge_path is None:
                raise ValueError(
                    f"{raw['id']}: rule-condition solution requires solution_edge_path"
                )
            valid, reasons = validate_edge_path(
                vertices,
                edge_tuple,
                solution_path,
                solution_edge_path,
                start=start,
                end=end,
                constraints=rule_constraints,
            )
            if not valid:
                raise ValueError(
                    f"{raw['id']}: invalid constrained solution: {', '.join(reasons)}"
                )
        else:
            _validate_solution_path(
                raw["id"],
                edge_tuple,
                solution_path,
                start=start,
                end=end,
            )
    elif solution_edge_path is not None:
        raise ValueError(f"{raw['id']}: solution_edge_path requires solution_path")
    if solution_exists and solution_path is None and capability == "rule_condition":
        raise ValueError(f"{raw['id']}: solvable rule-condition task requires an oracle")

    tags = _require_string_list(raw, "tags")
    difficulty = raw.get("difficulty") or _tag_value(tags, "difficulty:") or "unknown"
    if not isinstance(difficulty, str):
        raise ValueError(f"{raw['id']}: difficulty must be a string")
    history_events = _parse_history_events(raw, vertex_set)
    source_task_id = raw.get("source_task_id")
    if source_task_id is not None and (
        not isinstance(source_task_id, str) or not source_task_id
    ):
        raise ValueError(f"{raw['id']}: source_task_id must be a non-empty string")
    image_variants = _parse_image_variants(raw, base_dir=base_dir)

    task = OneStrokeTask(
        id=raw["id"],
        vertices=vertices,
        edges=edge_tuple,
        start=start,
        end=end,
        solution_exists=solution_exists,
        solution_path=solution_path,
        solution_edge_path=solution_edge_path,
        capability=capability,
        difficulty=difficulty,
        history_events=history_events,
        rule_constraints=rule_constraints,
        key_rule_id=key_rule_id,
        conflicting_rule=conflicting_rule,
        source_task_id=source_task_id,
        image_variants=image_variants,
        tags=tags,
    )
    if capability != "history_memory" and history_events:
        raise ValueError(f"{raw['id']}: only history tasks may define history_events")
    if capability != "rule_condition" and (
        rule_constraints or key_rule_id is not None or conflicting_rule is not None
    ):
        raise ValueError(
            f"{raw['id']}: rule fields require capability rule_condition"
        )
    if capability == "multimodal":
        if source_task_id is None:
            raise ValueError(f"{raw['id']}: multimodal tasks require source_task_id")
        if set(image_variants) != {"clear", "challenge"}:
            raise ValueError(
                f"{raw['id']}: multimodal image_variants must contain clear and challenge"
            )
    elif source_task_id is not None or image_variants:
        raise ValueError(
            f"{raw['id']}: source_task_id and image_variants require capability multimodal"
        )
    if capability == "rule_condition":
        _validate_rule_task_modes(task)
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


def _parse_image_variants(
    raw: dict[str, Any],
    *,
    base_dir: Path | None,
) -> dict[str, Path]:
    value = raw.get("image_variants", {})
    if not isinstance(value, dict):
        raise ValueError(f"{raw['id']}: image_variants must be an object")
    variants: dict[str, Path] = {}
    for name, raw_path in value.items():
        if not isinstance(name, str) or not isinstance(raw_path, str) or not raw_path:
            raise ValueError(
                f"{raw['id']}: image_variants must map names to non-empty paths"
            )
        path = Path(raw_path)
        if base_dir is not None and not path.is_absolute():
            path = (base_dir / path).resolve()
        if base_dir is not None and not path.is_file():
            raise ValueError(f"{raw['id']}: image file does not exist: {path}")
        variants[name] = path
    return variants


def _parse_rule_constraints(
    raw: dict[str, Any],
    vertices: tuple[str, ...],
    edge_ids: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> tuple[OneStrokeRule, ...]:
    value = raw.get("rule_constraints", [])
    if not isinstance(value, list):
        raise ValueError(f"{raw['id']}: rule_constraints must be a list")
    rules = tuple(
        parse_one_stroke_rule(
            item,
            task_id=raw["id"],
            vertices=vertices,
            edge_ids=edge_ids,
            edge_count=len(edges),
            field=f"rule_constraints[{index}]",
        )
        for index, item in enumerate(value, start=1)
    )
    ids = [rule.id for rule in rules]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{raw['id']}: rule IDs must be unique")
    for rule in rules:
        _validate_rule_edge_semantics(raw["id"], rule, edge_ids, edges)
    return rules


def _validate_rule_edge_semantics(
    task_id: str,
    rule: OneStrokeRule,
    edge_ids: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> None:
    if rule.type != "directed_edge":
        return
    edge_map = dict(zip(edge_ids, edges))
    assert rule.edge_id is not None
    expected = _canonical_edge(edge_map[rule.edge_id])
    actual = _canonical_edge((rule.from_vertex or "", rule.to_vertex or ""))
    if expected != actual:
        raise ValueError(
            f"{task_id}: directed rule {rule.id} endpoints do not match {rule.edge_id}"
        )


def _validate_rule_task_modes(task: OneStrokeTask) -> None:
    if not task.rule_constraints:
        raise ValueError(f"{task.id}: rule-condition tasks require rule_constraints")
    if task.key_rule_id not in {rule.id for rule in task.rule_constraints}:
        raise ValueError(f"{task.id}: key_rule_id must reference a full constraint")
    if task.conflicting_rule is None:
        raise ValueError(f"{task.id}: rule-condition tasks require conflicting_rule")
    if task.conflicting_rule.id in {rule.id for rule in task.rule_constraints}:
        raise ValueError(f"{task.id}: conflicting rule ID must be unique")
    key_rule = next(rule for rule in task.rule_constraints if rule.id == task.key_rule_id)
    if not rules_form_conflicting_pair(key_rule, task.conflicting_rule):
        raise ValueError(
            f"{task.id}: conflicting_rule must be a true logical reverse of the key rule"
        )
    remaining_rules = tuple(
        rule for rule in task.rule_constraints if rule.id != task.key_rule_id
    )
    if any(
        rules_are_semantically_equivalent(rule, task.conflicting_rule)
        for rule in remaining_rules
    ):
        raise ValueError(
            f"{task.id}: conflicting_rule must not duplicate a remaining rule"
        )
    for mode in ("standard", "drop_key_rule", "conflicting_rule"):
        constraints = rules_for_mode(
            task.rule_constraints,
            task.key_rule_id,
            task.conflicting_rule,
            mode,
        )
        if find_constrained_one_stroke_path(
            task.vertices,
            task.edges,
            start=task.start,
            end=task.end,
            constraints=constraints,
        ) is None:
            raise ValueError(f"{task.id}: {mode} mode must have a valid oracle path")
    combined = (*task.rule_constraints, task.conflicting_rule)
    if find_constrained_one_stroke_path(
        task.vertices,
        task.edges,
        start=task.start,
        end=task.end,
        constraints=combined,
    ) is not None:
        raise ValueError(
            f"{task.id}: key and conflicting rules must not be jointly satisfiable"
        )


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
            tasks.append(one_stroke_task_from_dict(raw, base_dir=task_path.parent))
    if not tasks:
        raise ValueError(f"{task_path} contains no tasks")
    return tasks

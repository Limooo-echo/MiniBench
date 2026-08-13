from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence


ONE_STROKE_RULE_MODES = (
    "full",
    "standard",
    "drop_key_rule",
    "conflicting_rule",
)

ONE_STROKE_RULE_TYPES = (
    "start_vertex",
    "end_vertex",
    "first_edge",
    "last_edge",
    "directed_edge",
    "edge_before",
    "vertex_at_step",
    "adjacent_edges",
    "nonconsecutive_edges",
    "edge_step_window",
)


@dataclass(frozen=True)
class OneStrokeRule:
    id: str
    type: str
    vertex: str | None = None
    edge_id: str | None = None
    before_edge_id: str | None = None
    after_edge_id: str | None = None
    edge_ids: tuple[str, str] | None = None
    from_vertex: str | None = None
    to_vertex: str | None = None
    step: int | None = None
    min_step: int | None = None
    max_step: int | None = None


def parse_one_stroke_rule(
    raw: object,
    *,
    task_id: str,
    vertices: Sequence[str],
    edge_ids: Sequence[str],
    edge_count: int,
    field: str,
) -> OneStrokeRule:
    if not isinstance(raw, dict):
        raise ValueError(f"{task_id}: {field} must be an object")
    rule_id = raw.get("id")
    rule_type = raw.get("type")
    if not isinstance(rule_id, str) or not rule_id:
        raise ValueError(f"{task_id}: {field}.id must be a non-empty string")
    if rule_type not in ONE_STROKE_RULE_TYPES:
        choices = ", ".join(ONE_STROKE_RULE_TYPES)
        raise ValueError(f"{task_id}: {field}.type must be one of {choices}")

    vertex_set = set(vertices)
    edge_id_set = set(edge_ids)

    def vertex(key: str) -> str:
        value = raw.get(key)
        if not isinstance(value, str) or value not in vertex_set:
            raise ValueError(f"{task_id}: {field}.{key} must be a known vertex")
        return value

    def edge_id(key: str) -> str:
        value = raw.get(key)
        if not isinstance(value, str) or value not in edge_id_set:
            raise ValueError(f"{task_id}: {field}.{key} must be a known edge ID")
        return value

    if rule_type in {"start_vertex", "end_vertex"}:
        return OneStrokeRule(id=rule_id, type=rule_type, vertex=vertex("vertex"))
    if rule_type in {"first_edge", "last_edge"}:
        return OneStrokeRule(id=rule_id, type=rule_type, edge_id=edge_id("edge_id"))
    if rule_type == "directed_edge":
        selected_edge = edge_id("edge_id")
        from_vertex = vertex("from")
        to_vertex = vertex("to")
        return OneStrokeRule(
            id=rule_id,
            type=rule_type,
            edge_id=selected_edge,
            from_vertex=from_vertex,
            to_vertex=to_vertex,
        )
    if rule_type == "edge_before":
        before = edge_id("before_edge_id")
        after = edge_id("after_edge_id")
        if before == after:
            raise ValueError(f"{task_id}: {field} must reference two different edges")
        return OneStrokeRule(
            id=rule_id,
            type=rule_type,
            before_edge_id=before,
            after_edge_id=after,
        )
    if rule_type in {"adjacent_edges", "nonconsecutive_edges"}:
        pair = raw.get("edge_ids")
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(item, str) and item in edge_id_set for item in pair)
        ):
            raise ValueError(
                f"{task_id}: {field}.edge_ids must contain two known edge IDs"
            )
        if pair[0] == pair[1]:
            raise ValueError(f"{task_id}: {field} must reference two different edges")
        return OneStrokeRule(
            id=rule_id,
            type=rule_type,
            edge_ids=(pair[0], pair[1]),
        )
    if rule_type == "vertex_at_step":
        step = raw.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= edge_count:
            raise ValueError(
                f"{task_id}: {field}.step must be between 0 and {edge_count}"
            )
        return OneStrokeRule(
            id=rule_id,
            type=rule_type,
            vertex=vertex("vertex"),
            step=step,
        )
    minimum = raw.get("min_step")
    maximum = raw.get("max_step")
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, int)
        or not isinstance(maximum, int)
        or not 1 <= minimum <= maximum <= edge_count
    ):
        raise ValueError(
            f"{task_id}: {field} step window must satisfy "
            f"1 <= min_step <= max_step <= {edge_count}"
        )
    return OneStrokeRule(
        id=rule_id,
        type=rule_type,
        edge_id=edge_id("edge_id"),
        min_step=minimum,
        max_step=maximum,
    )


def rules_for_mode(
    constraints: Sequence[OneStrokeRule],
    key_rule_id: str | None,
    conflicting_rule: OneStrokeRule | None,
    mode: str,
) -> tuple[OneStrokeRule, ...]:
    if mode not in ONE_STROKE_RULE_MODES:
        choices = ", ".join(ONE_STROKE_RULE_MODES)
        raise ValueError(f"unknown one-stroke rule mode {mode!r}; choose {choices}")
    if mode == "standard":
        return ()
    if mode == "full":
        return tuple(constraints)
    without_key = tuple(rule for rule in constraints if rule.id != key_rule_id)
    if mode == "drop_key_rule":
        return without_key
    if conflicting_rule is None:
        raise ValueError("conflicting_rule mode requires a replacement rule")
    return (*without_key, conflicting_rule)


def rules_form_conflicting_pair(
    key_rule: OneStrokeRule,
    replacement: OneStrokeRule,
) -> bool:
    if key_rule.type in {"start_vertex", "end_vertex"}:
        return key_rule.type == replacement.type and key_rule.vertex != replacement.vertex
    if key_rule.type in {"first_edge", "last_edge"}:
        return key_rule.type == replacement.type and key_rule.edge_id != replacement.edge_id
    if key_rule.type == "directed_edge":
        return (
            replacement.type == "directed_edge"
            and key_rule.edge_id == replacement.edge_id
            and key_rule.from_vertex == replacement.to_vertex
            and key_rule.to_vertex == replacement.from_vertex
        )
    if key_rule.type == "edge_before":
        return (
            replacement.type == "edge_before"
            and key_rule.before_edge_id == replacement.after_edge_id
            and key_rule.after_edge_id == replacement.before_edge_id
        )
    if key_rule.type == "vertex_at_step":
        return (
            replacement.type == "vertex_at_step"
            and key_rule.step == replacement.step
            and key_rule.vertex != replacement.vertex
        )
    if key_rule.type in {"adjacent_edges", "nonconsecutive_edges"}:
        return (
            {key_rule.type, replacement.type}
            == {"adjacent_edges", "nonconsecutive_edges"}
            and key_rule.edge_ids is not None
            and replacement.edge_ids is not None
            and set(key_rule.edge_ids) == set(replacement.edge_ids)
        )
    if key_rule.type == "edge_step_window":
        return (
            replacement.type == "edge_step_window"
            and key_rule.edge_id == replacement.edge_id
            and key_rule.min_step is not None
            and key_rule.max_step is not None
            and replacement.min_step is not None
            and replacement.max_step is not None
            and (
                key_rule.max_step < replacement.min_step
                or replacement.max_step < key_rule.min_step
            )
        )
    return False


def rules_are_semantically_equivalent(
    first: OneStrokeRule,
    second: OneStrokeRule,
) -> bool:
    if first.type != second.type:
        return False
    if first.type in {"adjacent_edges", "nonconsecutive_edges"}:
        return (
            first.edge_ids is not None
            and second.edge_ids is not None
            and set(first.edge_ids) == set(second.edge_ids)
        )
    return (
        first.vertex,
        first.edge_id,
        first.before_edge_id,
        first.after_edge_id,
        first.from_vertex,
        first.to_vertex,
        first.step,
        first.min_step,
        first.max_step,
    ) == (
        second.vertex,
        second.edge_id,
        second.before_edge_id,
        second.after_edge_id,
        second.from_vertex,
        second.to_vertex,
        second.step,
        second.min_step,
        second.max_step,
    )


def validate_edge_path(
    vertices: Sequence[str],
    edges: Sequence[tuple[str, str]],
    path: Sequence[str],
    edge_path: Sequence[str],
    *,
    start: str | None = None,
    end: str | None = None,
    constraints: Sequence[OneStrokeRule] = (),
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    edge_ids = _edge_ids(edges)
    edge_map = dict(zip(edge_ids, edges))
    expected_steps = len(edges)

    if len(path) != expected_steps + 1:
        reasons.append(
            f"wrong_path_length:expected={expected_steps + 1},actual={len(path)}"
        )
    if len(edge_path) != expected_steps:
        reasons.append(
            f"wrong_edge_path_length:expected={expected_steps},actual={len(edge_path)}"
        )
    unknown_vertices = sorted(set(path) - set(vertices))
    if unknown_vertices:
        reasons.append(f"unknown_vertices:{','.join(unknown_vertices)}")
    unknown_edges = sorted(set(edge_path) - set(edge_ids))
    if unknown_edges:
        reasons.append(f"unknown_edge_ids:{','.join(unknown_edges)}")

    counts = Counter(edge_path)
    duplicate_edges = sorted(edge_id for edge_id, count in counts.items() if count > 1)
    if duplicate_edges:
        reasons.append(f"reused_edge_ids:{','.join(duplicate_edges)}")
    missing_edges = sorted(set(edge_ids) - set(edge_path))
    if missing_edges:
        reasons.append(f"missing_edge_ids:{','.join(missing_edges)}")

    if start is not None and path and path[0] != start:
        reasons.append(f"wrong_start:expected={start},actual={path[0]}")
    if end is not None and path and path[-1] != end:
        reasons.append(f"wrong_end:expected={end},actual={path[-1]}")

    for step, edge_id in enumerate(edge_path, start=1):
        if step >= len(path) or edge_id not in edge_map:
            continue
        actual = _canonical((path[step - 1], path[step]))
        expected = _canonical(edge_map[edge_id])
        if actual != expected:
            reasons.append(
                f"edge_path_mismatch:{step}:{edge_id}:"
                f"{path[step - 1]}-{path[step]}"
            )

    reasons.extend(rule_violation_reasons(path, edge_path, constraints))
    return not reasons, reasons


def rule_violation_reasons(
    path: Sequence[str],
    edge_path: Sequence[str],
    constraints: Sequence[OneStrokeRule],
) -> list[str]:
    reasons: list[str] = []
    positions = {edge_id: index for index, edge_id in enumerate(edge_path, start=1)}
    for rule in constraints:
        ok = _rule_is_satisfied(rule, path, edge_path, positions)
        if not ok:
            reasons.append(f"rule_violation:{rule.id}:{rule.type}")
    return reasons


def find_constrained_one_stroke_path(
    vertices: Sequence[str],
    edges: Sequence[tuple[str, str]],
    *,
    start: str | None = None,
    end: str | None = None,
    constraints: Sequence[OneStrokeRule] = (),
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    edge_ids = _edge_ids(edges)
    edge_map = dict(zip(edge_ids, edges))
    adjacency: dict[str, list[tuple[str, str]]] = {vertex: [] for vertex in vertices}
    for edge_id, (a, b) in zip(edge_ids, edges):
        adjacency[a].append((edge_id, b))
        adjacency[b].append((edge_id, a))
    for items in adjacency.values():
        items.sort()

    required_start = _single_required_value(
        [start, *(rule.vertex for rule in constraints if rule.type == "start_vertex")]
    )
    required_end = _single_required_value(
        [end, *(rule.vertex for rule in constraints if rule.type == "end_vertex")]
    )
    if required_start is _CONFLICT or required_end is _CONFLICT:
        return None

    step_zero_vertices = [
        rule.vertex
        for rule in constraints
        if rule.type == "vertex_at_step" and rule.step == 0
    ]
    checkpoint_start = _single_required_value(step_zero_vertices)
    if checkpoint_start is _CONFLICT:
        return None
    if checkpoint_start is not None:
        if required_start is not None and required_start != checkpoint_start:
            return None
        required_start = checkpoint_start

    active = sorted(vertex for vertex in vertices if adjacency[vertex])
    if required_start is not None:
        starts = [required_start]
    else:
        degree = Counter[str]()
        for a, b in edges:
            degree[a] += 1
            degree[b] += 1
        odd = sorted(vertex for vertex in vertices if degree[vertex] % 2)
        starts = odd if len(odd) == 2 else active

    used: set[str] = set()
    path: list[str] = []
    edge_path: list[str] = []

    def search(current: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        step = len(edge_path)
        if step == len(edges):
            if required_end is not None and current != required_end:
                return None
            ok, _ = validate_edge_path(
                vertices,
                edges,
                path,
                edge_path,
                start=start,
                end=end,
                constraints=constraints,
            )
            if ok:
                return tuple(path), tuple(edge_path)
            return None

        remaining_ids = set(edge_ids) - used
        if not _remaining_euler_feasible(
            current,
            remaining_ids,
            edge_map,
            required_end,
        ):
            return None

        for edge_id, neighbor in adjacency[current]:
            if edge_id in used:
                continue
            edge_path.append(edge_id)
            path.append(neighbor)
            used.add(edge_id)
            if _partial_rules_possible(path, edge_path, constraints, len(edges)):
                result = search(neighbor)
                if result is not None:
                    return result
            used.remove(edge_id)
            path.pop()
            edge_path.pop()
        return None

    for candidate in starts:
        if candidate not in adjacency or not adjacency[candidate]:
            continue
        path[:] = [candidate]
        edge_path.clear()
        used.clear()
        if not _partial_rules_possible(path, edge_path, constraints, len(edges)):
            continue
        result = search(candidate)
        if result is not None:
            return result
    return None


def _partial_rules_possible(
    path: Sequence[str],
    edge_path: Sequence[str],
    constraints: Sequence[OneStrokeRule],
    edge_count: int,
) -> bool:
    step = len(edge_path)
    positions = {edge_id: index for index, edge_id in enumerate(edge_path, start=1)}
    for rule in constraints:
        if rule.type == "start_vertex" and path[0] != rule.vertex:
            return False
        if rule.type == "end_vertex" and step == edge_count and path[-1] != rule.vertex:
            return False
        if rule.type == "first_edge" and step >= 1 and edge_path[0] != rule.edge_id:
            return False
        if rule.type == "last_edge":
            if rule.edge_id in positions and positions[rule.edge_id] != edge_count:
                return False
            if step == edge_count and edge_path[-1] != rule.edge_id:
                return False
        if rule.type == "directed_edge" and rule.edge_id in positions:
            index = positions[rule.edge_id]
            if path[index - 1] != rule.from_vertex or path[index] != rule.to_vertex:
                return False
        if rule.type == "edge_before":
            before = positions.get(rule.before_edge_id or "")
            after = positions.get(rule.after_edge_id or "")
            if after is not None and before is None:
                return False
            if before is not None and after is not None and before >= after:
                return False
        if rule.type == "vertex_at_step":
            assert rule.step is not None
            if rule.step <= step and path[rule.step] != rule.vertex:
                return False
        if rule.type == "adjacent_edges":
            assert rule.edge_ids is not None
            first = positions.get(rule.edge_ids[0])
            second = positions.get(rule.edge_ids[1])
            if first is not None and second is not None and abs(first - second) != 1:
                return False
            known = first if second is None else second if first is None else None
            if known is not None and step > known and step >= known + 1:
                return False
        if rule.type == "nonconsecutive_edges":
            assert rule.edge_ids is not None
            first = positions.get(rule.edge_ids[0])
            second = positions.get(rule.edge_ids[1])
            if first is not None and second is not None and abs(first - second) == 1:
                return False
        if rule.type == "edge_step_window" and rule.edge_id in positions:
            position = positions[rule.edge_id]
            if position < (rule.min_step or 1) or position > (rule.max_step or edge_count):
                return False
    return True


def _rule_is_satisfied(
    rule: OneStrokeRule,
    path: Sequence[str],
    edge_path: Sequence[str],
    positions: dict[str, int],
) -> bool:
    if rule.type == "start_vertex":
        return bool(path) and path[0] == rule.vertex
    if rule.type == "end_vertex":
        return bool(path) and path[-1] == rule.vertex
    if rule.type == "first_edge":
        return bool(edge_path) and edge_path[0] == rule.edge_id
    if rule.type == "last_edge":
        return bool(edge_path) and edge_path[-1] == rule.edge_id
    if rule.type == "directed_edge":
        index = positions.get(rule.edge_id or "")
        return (
            index is not None
            and index < len(path)
            and path[index - 1] == rule.from_vertex
            and path[index] == rule.to_vertex
        )
    if rule.type == "edge_before":
        before = positions.get(rule.before_edge_id or "")
        after = positions.get(rule.after_edge_id or "")
        return before is not None and after is not None and before < after
    if rule.type == "vertex_at_step":
        return rule.step is not None and rule.step < len(path) and path[rule.step] == rule.vertex
    if rule.type in {"adjacent_edges", "nonconsecutive_edges"}:
        assert rule.edge_ids is not None
        first = positions.get(rule.edge_ids[0])
        second = positions.get(rule.edge_ids[1])
        if first is None or second is None:
            return False
        distance = abs(first - second)
        return distance == 1 if rule.type == "adjacent_edges" else distance != 1
    if rule.type == "edge_step_window":
        position = positions.get(rule.edge_id or "")
        return (
            position is not None
            and rule.min_step is not None
            and rule.max_step is not None
            and rule.min_step <= position <= rule.max_step
        )
    raise ValueError(f"unknown one-stroke rule type: {rule.type}")


def _remaining_euler_feasible(
    current: str,
    remaining_ids: set[str],
    edge_map: dict[str, tuple[str, str]],
    required_end: str | None,
) -> bool:
    if not remaining_ids:
        return required_end is None or current == required_end
    adjacency: dict[str, set[str]] = {}
    degree = Counter[str]()
    active = set[str]()
    for edge_id in remaining_ids:
        a, b = edge_map[edge_id]
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
        degree[a] += 1
        degree[b] += 1
        active.update((a, b))
    if current not in active:
        return False
    visited: set[str] = set()
    stack = [current]
    while stack:
        vertex = stack.pop()
        if vertex in visited:
            continue
        visited.add(vertex)
        stack.extend(adjacency.get(vertex, set()) - visited)
    if not active <= visited:
        return False
    odd = {vertex for vertex in active if degree[vertex] % 2}
    if not odd:
        return required_end is None or required_end == current
    if len(odd) != 2 or current not in odd:
        return False
    other = next(vertex for vertex in odd if vertex != current)
    return required_end is None or required_end == other


def _edge_ids(edges: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    width = max(2, len(str(len(edges))))
    return tuple(f"e{index:0{width}d}" for index in range(1, len(edges) + 1))


def _canonical(edge: tuple[str, str]) -> tuple[str, str]:
    a, b = edge
    return (a, b) if a <= b else (b, a)


_CONFLICT = object()


def _single_required_value(values: Sequence[str | None]) -> str | None | object:
    selected = {value for value in values if value is not None}
    if len(selected) > 1:
        return _CONFLICT
    return next(iter(selected)) if selected else None

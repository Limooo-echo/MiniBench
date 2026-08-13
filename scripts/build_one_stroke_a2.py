#!/usr/bin/env python3
"""Build the deterministic MiniBench 2.0 A2 one-stroke Core set."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minibench.datasets.one_stroke.rules import (  # noqa: E402
    OneStrokeRule,
    find_constrained_one_stroke_path,
    rule_violation_reasons,
    rules_for_mode,
)


OUTPUT = ROOT / "data" / "one_stroke" / "a2_rule_condition.jsonl"


def _cycle() -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    return (
        ("A", "B", "C", "D", "E"),
        (("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")),
    )


def _path() -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    return (
        ("A", "B", "C", "D", "E"),
        (("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")),
    )


def _petals(count: int) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    vertices = ["A"]
    edges: list[tuple[str, str]] = []
    for index in range(count):
        first = chr(ord("B") + index * 2)
        second = chr(ord("C") + index * 2)
        vertices.extend((first, second))
        edges.extend((("A", first), (first, second), (second, "A")))
    return tuple(vertices), tuple(edges)


def _rule(rule_id: str, rule_type: str, **kwargs: object) -> OneStrokeRule:
    return OneStrokeRule(id=rule_id, type=rule_type, **kwargs)


def _serialize_rule(rule: OneStrokeRule) -> dict[str, object]:
    raw = asdict(rule)
    result: dict[str, object] = {}
    for key, value in raw.items():
        if value is None:
            continue
        output_key = {"from_vertex": "from", "to_vertex": "to"}.get(key, key)
        result[output_key] = list(value) if isinstance(value, tuple) else value
    return result


def _record(
    task_id: str,
    difficulty: str,
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    constraints: tuple[OneStrokeRule, ...],
    conflicting_rule: OneStrokeRule,
    *,
    key_rule_id: str = "r01",
) -> dict[str, object]:
    full_oracle = find_constrained_one_stroke_path(
        vertices,
        edges,
        constraints=constraints,
    )
    standard_oracle = find_constrained_one_stroke_path(vertices, edges)
    if standard_oracle is None:
        raise ValueError(f"{task_id}: base graph must be solvable")
    conflict_constraints = rules_for_mode(
        constraints,
        key_rule_id,
        conflicting_rule,
        "conflicting_rule",
    )
    conflicting_oracle = find_constrained_one_stroke_path(
        vertices,
        edges,
        constraints=conflict_constraints,
    )
    if conflicting_oracle is None:
        raise ValueError(f"{task_id}: reverse-rule world must be solvable")
    if find_constrained_one_stroke_path(
        vertices,
        edges,
        constraints=(*constraints, conflicting_rule),
    ) is not None:
        raise ValueError(f"{task_id}: key and reverse rules must conflict")

    standard_violations = rule_violation_reasons(
        standard_oracle[0],
        standard_oracle[1],
        constraints,
    )
    if not standard_violations:
        raise ValueError(f"{task_id}: deterministic standard oracle must ignore A2 rules")

    solution_exists = full_oracle is not None
    rule_types = sorted({rule.type for rule in constraints})
    return {
        "id": task_id,
        "capability": "rule_condition",
        "difficulty": difficulty,
        "vertices": list(vertices),
        "edges": [list(edge) for edge in edges],
        "start": None,
        "end": None,
        "solution_exists": solution_exists,
        "solution_path": list(full_oracle[0]) if full_oracle else None,
        "solution_edge_path": list(full_oracle[1]) if full_oracle else None,
        "rule_constraints": [_serialize_rule(rule) for rule in constraints],
        "key_rule_id": key_rule_id,
        "conflicting_rule": _serialize_rule(conflicting_rule),
        "tags": [
            "one-stroke",
            "benchmark:a2",
            "capability:rule-condition",
            f"difficulty:{difficulty}",
            f"solution:{'yes' if solution_exists else 'no'}",
            "base-solution:yes",
            "standard-oracle:violates-full-rules",
            "conflicting-rule:verified-reverse",
            "source:deterministic-handcrafted",
            *(f"rule:{rule_type}" for rule_type in rule_types),
        ],
    }


def build_easy_records() -> list[dict[str, object]]:
    cycle_vertices, cycle_edges = _cycle()
    path_vertices, path_edges = _path()
    specs = [
        (
            cycle_vertices,
            cycle_edges,
            _rule("r01", "start_vertex", vertex="B"),
            _rule("r99", "start_vertex", vertex="A"),
        ),
        (
            cycle_vertices,
            cycle_edges,
            _rule("r01", "end_vertex", vertex="B"),
            _rule("r99", "end_vertex", vertex="A"),
        ),
        (
            cycle_vertices,
            cycle_edges,
            _rule("r01", "first_edge", edge_id="e05"),
            _rule("r99", "first_edge", edge_id="e01"),
        ),
        (
            cycle_vertices,
            cycle_edges,
            _rule("r01", "last_edge", edge_id="e01"),
            _rule("r99", "last_edge", edge_id="e05"),
        ),
        (
            cycle_vertices,
            cycle_edges,
            _rule(
                "r01",
                "directed_edge",
                edge_id="e01",
                from_vertex="B",
                to_vertex="A",
            ),
            _rule(
                "r99",
                "directed_edge",
                edge_id="e01",
                from_vertex="A",
                to_vertex="B",
            ),
        ),
        (
            cycle_vertices,
            cycle_edges,
            _rule("r01", "start_vertex", vertex="C"),
            _rule("r99", "start_vertex", vertex="A"),
        ),
        (
            cycle_vertices,
            cycle_edges,
            _rule("r01", "end_vertex", vertex="D"),
            _rule("r99", "end_vertex", vertex="A"),
        ),
        (
            cycle_vertices,
            cycle_edges,
            _rule("r01", "first_edge", edge_id="e03"),
            _rule("r99", "first_edge", edge_id="e01"),
        ),
        (
            path_vertices,
            path_edges,
            _rule("r01", "start_vertex", vertex="B"),
            _rule("r99", "start_vertex", vertex="A"),
        ),
        (
            path_vertices,
            path_edges,
            _rule("r01", "end_vertex", vertex="C"),
            _rule("r99", "end_vertex", vertex="E"),
        ),
    ]
    return [
        _record(
            f"a2-easy-{index:02d}",
            "easy",
            vertices,
            edges,
            (key_rule,),
            conflict,
        )
        for index, (vertices, edges, key_rule, conflict) in enumerate(specs, start=1)
    ]


def build_medium_records() -> list[dict[str, object]]:
    vertices, edges = _petals(3)
    specs: list[tuple[tuple[OneStrokeRule, ...], OneStrokeRule]] = [
        (
            (
                _rule("r01", "edge_before", before_edge_id="e04", after_edge_id="e01"),
                _rule("r02", "vertex_at_step", vertex="A", step=3),
            ),
            _rule("r99", "edge_before", before_edge_id="e01", after_edge_id="e04"),
        ),
        (
            (
                _rule("r01", "vertex_at_step", vertex="C", step=1),
                _rule("r02", "edge_before", before_edge_id="e01", after_edge_id="e04"),
            ),
            _rule("r99", "vertex_at_step", vertex="B", step=1),
        ),
        (
            (
                _rule("r01", "nonconsecutive_edges", edge_ids=("e03", "e04")),
                _rule("r02", "vertex_at_step", vertex="A", step=3),
            ),
            _rule("r99", "adjacent_edges", edge_ids=("e03", "e04")),
        ),
        (
            (
                _rule("r01", "adjacent_edges", edge_ids=("e01", "e04")),
                _rule("r02", "vertex_at_step", vertex="C", step=1),
            ),
            _rule("r99", "nonconsecutive_edges", edge_ids=("e01", "e04")),
        ),
        (
            (
                _rule("r01", "edge_before", before_edge_id="e07", after_edge_id="e04"),
                _rule("r02", "vertex_at_step", vertex="A", step=3),
            ),
            _rule("r99", "edge_before", before_edge_id="e04", after_edge_id="e07"),
        ),
        (
            (
                _rule("r01", "vertex_at_step", vertex="F", step=4),
                _rule("r02", "edge_before", before_edge_id="e01", after_edge_id="e04"),
            ),
            _rule("r99", "vertex_at_step", vertex="D", step=4),
        ),
        (
            (
                _rule("r01", "nonconsecutive_edges", edge_ids=("e06", "e07")),
                _rule("r02", "edge_before", before_edge_id="e01", after_edge_id="e04"),
            ),
            _rule("r99", "adjacent_edges", edge_ids=("e06", "e07")),
        ),
        (
            (
                _rule("r01", "adjacent_edges", edge_ids=("e06", "e07")),
                _rule("r02", "vertex_at_step", vertex="C", step=1),
            ),
            _rule("r99", "nonconsecutive_edges", edge_ids=("e06", "e07")),
        ),
        (
            (
                _rule("r01", "edge_before", before_edge_id="e01", after_edge_id="e04"),
                _rule("r02", "vertex_at_step", vertex="D", step=1),
            ),
            _rule("r99", "edge_before", before_edge_id="e04", after_edge_id="e01"),
        ),
        (
            (
                _rule("r01", "edge_before", before_edge_id="e04", after_edge_id="e01"),
                _rule("r02", "vertex_at_step", vertex="B", step=1),
            ),
            _rule("r99", "edge_before", before_edge_id="e01", after_edge_id="e04"),
        ),
    ]
    return [
        _record(
            f"a2-medium-{index:02d}",
            "medium",
            vertices,
            edges,
            constraints,
            conflict,
        )
        for index, (constraints, conflict) in enumerate(specs, start=1)
    ]


def _hard_constraints(
    key_petal: int,
    window_petal: int,
    before_petals: tuple[int, int],
    *,
    impossible: bool,
) -> tuple[tuple[OneStrokeRule, ...], OneStrokeRule]:
    key_edge_number = (key_petal - 1) * 3 + 1
    key_edge = f"e{key_edge_number:02d}"
    first = chr(ord("B") + (key_petal - 1) * 2)
    window_edge_number = (window_petal - 1) * 3 + 2
    before_edge = f"e{(before_petals[0] - 1) * 3 + 1:02d}"
    after_edge = f"e{(before_petals[1] - 1) * 3 + 1:02d}"
    rules = [
        _rule(
            "r01",
            "directed_edge",
            edge_id=key_edge,
            from_vertex=first,
            to_vertex="A",
        ),
        _rule(
            "r02",
            "edge_step_window",
            edge_id=f"e{window_edge_number:02d}",
            min_step=(window_petal - 1) * 3 + 1,
            max_step=window_petal * 3,
        ),
        _rule("r03", "edge_before", before_edge_id=before_edge, after_edge_id=after_edge),
        _rule("r04", "adjacent_edges", edge_ids=("e10", "e11")),
        _rule("r05", "nonconsecutive_edges", edge_ids=("e02", "e04")),
    ]
    conflict = _rule(
        "r99",
        "directed_edge",
        edge_id=key_edge,
        from_vertex="A",
        to_vertex=first,
    )
    if impossible:
        checkpoint_vertex, checkpoint_step = (
            ("C", 3) if key_petal == 1 else ("F", 6)
        )
        rules.append(
            _rule(
                "r06",
                "vertex_at_step",
                vertex=checkpoint_vertex,
                step=checkpoint_step,
            )
        )
    return tuple(rules), conflict


def build_hard_records() -> list[dict[str, object]]:
    vertices, edges = _petals(4)
    settings = [
        (1, 2, (3, 4), False),
        (2, 3, (1, 4), False),
        (3, 2, (1, 4), False),
        (4, 3, (1, 3), False),
        (1, 3, (2, 4), False),
        (2, 2, (1, 3), False),
        (3, 3, (1, 4), False),
        (4, 2, (1, 3), False),
        (1, 2, (3, 4), True),
        (3, 3, (1, 4), True),
    ]
    records = []
    for index, (key, window, ordering, impossible) in enumerate(settings, start=1):
        constraints, conflict = _hard_constraints(
            key,
            window,
            ordering,
            impossible=impossible,
        )
        records.append(
            _record(
                f"a2-hard-{index:02d}",
                "hard",
                vertices,
                edges,
                constraints,
                conflict,
            )
        )
    return records


def build_a2_records() -> list[dict[str, object]]:
    records = [*build_easy_records(), *build_medium_records(), *build_hard_records()]
    if len(records) != 30:
        raise ValueError("A2 must contain exactly 30 tasks")
    for difficulty in ("easy", "medium", "hard"):
        selected = [record for record in records if record["difficulty"] == difficulty]
        counts = Counter(bool(record["solution_exists"]) for record in selected)
        if len(selected) != 10 or counts != Counter({True: 8, False: 2}):
            raise ValueError(f"bad A2 quota for {difficulty}: {counts}")
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def main() -> None:
    records = build_a2_records()
    _write_jsonl(OUTPUT, records)
    print(f"wrote {len(records)} A2 tasks to {OUTPUT}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minibench.datasets.one_stroke.dataset import (  # noqa: E402
    has_one_stroke_solution,
    one_stroke_task_from_dict,
)
from minibench.datasets.one_stroke.evaluation import (  # noqa: E402
    validate_one_stroke_path,
)


Edge = tuple[str, str]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 1:
        raise SystemExit("--count must be positive")
    if args.min_vertices < 2 or args.max_vertices < args.min_vertices:
        raise SystemExit("vertex bounds must satisfy 2 <= min <= max")

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} already exists; pass --overwrite to replace it")

    rng = random.Random(args.seed)
    records = generate_records(
        rng,
        count=args.count,
        min_vertices=args.min_vertices,
        max_vertices=args.max_vertices,
        disconnected_ratio=args.disconnected_ratio,
        connected_unsolvable_ratio=args.connected_unsolvable_ratio,
        prefix=args.prefix,
    )
    if args.shuffle:
        rng.shuffle(records)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = Counter(_category(record) for record in records)
    print(
        json.dumps(
            {
                "output": str(output),
                "count": len(records),
                "seed": args.seed,
                "summary": dict(sorted(summary.items())),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one-stroke graph benchmark tasks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/one_stroke/generated_50.jsonl"),
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--min-vertices", type=int, default=4)
    parser.add_argument("--max-vertices", type=int, default=9)
    parser.add_argument("--disconnected-ratio", type=float, default=0.2)
    parser.add_argument("--connected-unsolvable-ratio", type=float, default=0.2)
    parser.add_argument("--prefix", default="os-gen")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    return parser


def generate_records(
    rng: random.Random,
    *,
    count: int,
    min_vertices: int,
    max_vertices: int,
    disconnected_ratio: float,
    connected_unsolvable_ratio: float,
    prefix: str,
) -> list[dict[str, Any]]:
    disconnected_count = round(count * disconnected_ratio)
    connected_unsolvable_count = round(count * connected_unsolvable_ratio)
    solvable_count = count - disconnected_count - connected_unsolvable_count
    if solvable_count < 0:
        raise ValueError("ratios leave a negative solvable task count")

    records: list[dict[str, Any]] = []
    for index in range(1, disconnected_count + 1):
        records.append(
            make_disconnected_task(
                rng,
                task_id=f"{prefix}-disconnected-{index:03d}",
                min_vertices=min_vertices,
                max_vertices=max_vertices,
            )
        )
    for index in range(1, connected_unsolvable_count + 1):
        records.append(
            make_connected_unsolvable_task(
                rng,
                task_id=f"{prefix}-connected-no-euler-{index:03d}",
                min_vertices=min_vertices,
                max_vertices=max_vertices,
            )
        )
    for index in range(1, solvable_count + 1):
        records.append(
            make_solvable_task(
                rng,
                task_id=f"{prefix}-solvable-{index:03d}",
                min_vertices=min_vertices,
                max_vertices=max_vertices,
            )
        )
    return records


def make_disconnected_task(
    rng: random.Random,
    *,
    task_id: str,
    min_vertices: int,
    max_vertices: int,
) -> dict[str, Any]:
    for _ in range(10_000):
        vertex_count = rng.randint(max(4, min_vertices), max_vertices)
        vertices = labels(vertex_count)
        split = rng.randint(2, vertex_count - 2)
        left = vertices[:split]
        right = vertices[split:]
        edges = [
            *random_connected_graph(rng, left),
            *random_connected_graph(rng, right),
        ]
        record = base_record(
            task_id,
            vertices,
            edges,
            solution_exists=False,
            start=None,
            end=None,
            solution_path=None,
            tags=[
                "one-stroke",
                "generated",
                "connectivity:disconnected",
                "solution:no",
                "euler:none",
            ],
        )
        return validate_record(record)
    raise RuntimeError("failed to generate disconnected task")


def make_connected_unsolvable_task(
    rng: random.Random,
    *,
    task_id: str,
    min_vertices: int,
    max_vertices: int,
) -> dict[str, Any]:
    for _ in range(10_000):
        vertices = labels(rng.randint(min_vertices, max_vertices))
        edges = random_connected_graph(rng, vertices)
        odd = odd_vertices(edges)
        if len(odd) in {0, 2}:
            continue
        record = base_record(
            task_id,
            vertices,
            edges,
            solution_exists=False,
            start=None,
            end=None,
            solution_path=None,
            tags=[
                "one-stroke",
                "generated",
                "connectivity:connected",
                "solution:no",
                "euler:none",
                f"odd:{len(odd)}",
            ],
        )
        return validate_record(record)
    raise RuntimeError("failed to generate connected unsolvable task")


def make_solvable_task(
    rng: random.Random,
    *,
    task_id: str,
    min_vertices: int,
    max_vertices: int,
) -> dict[str, Any]:
    for _ in range(10_000):
        vertices = labels(rng.randint(min_vertices, max_vertices))
        edges = random_connected_graph(rng, vertices)
        odd = odd_vertices(edges)
        if len(odd) not in {0, 2}:
            continue
        start = odd[0] if len(odd) == 2 else first_edge_vertex(vertices, edges)
        solution_path = hierholzer_path(vertices, edges, start=start)
        end = solution_path[-1]
        kind = "trail" if len(odd) == 2 else "circuit"
        record = base_record(
            task_id,
            vertices,
            edges,
            solution_exists=True,
            start=solution_path[0],
            end=end,
            solution_path=solution_path,
            tags=[
                "one-stroke",
                "generated",
                "connectivity:connected",
                "solution:yes",
                f"euler:{kind}",
                f"odd:{len(odd)}",
            ],
        )
        return validate_record(record)
    raise RuntimeError("failed to generate solvable task")


def base_record(
    task_id: str,
    vertices: list[str],
    edges: list[Edge],
    *,
    solution_exists: bool,
    start: str | None,
    end: str | None,
    solution_path: list[str] | None,
    tags: list[str],
) -> dict[str, Any]:
    tags = [
        *tags,
        f"vertices:{len(vertices)}",
        f"edges:{len(edges)}",
        f"difficulty:{difficulty(len(vertices), len(edges))}",
    ]
    return {
        "id": task_id,
        "vertices": vertices,
        "edges": [[a, b] for a, b in sorted(edges)],
        "start": start,
        "end": end,
        "solution_exists": solution_exists,
        "solution_path": solution_path,
        "tags": tags,
    }


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    task = one_stroke_task_from_dict(record)
    if task.solution_exists:
        if task.solution_path is None:
            raise ValueError(f"{task.id}: missing solution_path")
        ok, reasons = validate_one_stroke_path(task, list(task.solution_path))
        if not ok:
            raise ValueError(f"{task.id}: invalid generated solution {reasons}")
    elif has_one_stroke_solution(task.vertices, task.edges, start=task.start, end=task.end):
        raise ValueError(f"{task.id}: generated unsolvable task is actually solvable")
    return record


def labels(count: int) -> list[str]:
    if count > 26:
        raise ValueError("labels only support up to 26 vertices")
    return [chr(ord("A") + index) for index in range(count)]


def random_connected_graph(rng: random.Random, vertices: list[str]) -> list[Edge]:
    order = vertices[:]
    rng.shuffle(order)
    edges: set[Edge] = set()
    for index, vertex in enumerate(order[1:], start=1):
        edges.add(canonical_edge(vertex, rng.choice(order[:index])))

    remaining = sorted(set(all_edges(vertices)) - edges)
    extra_limit = min(len(remaining), max(1, len(vertices)))
    extra_count = rng.randint(0, extra_limit)
    edges.update(rng.sample(remaining, extra_count))
    return sorted(edges)


def all_edges(vertices: list[str]) -> list[Edge]:
    return [
        canonical_edge(a, b)
        for index, a in enumerate(vertices)
        for b in vertices[index + 1 :]
    ]


def odd_vertices(edges: list[Edge]) -> list[str]:
    degree = Counter[str]()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    return sorted(vertex for vertex, count in degree.items() if count % 2 == 1)


def first_edge_vertex(vertices: list[str], edges: list[Edge]) -> str:
    if edges:
        return edges[0][0]
    return vertices[0]


def hierholzer_path(
    vertices: list[str],
    edges: list[Edge],
    *,
    start: str,
) -> list[str]:
    adjacency: dict[str, list[tuple[str, int]]] = {vertex: [] for vertex in vertices}
    for index, (a, b) in enumerate(edges):
        adjacency[a].append((b, index))
        adjacency[b].append((a, index))

    used_edges: set[int] = set()
    stack = [start]
    path: list[str] = []
    while stack:
        vertex = stack[-1]
        while adjacency[vertex] and adjacency[vertex][-1][1] in used_edges:
            adjacency[vertex].pop()
        if not adjacency[vertex]:
            path.append(stack.pop())
            continue
        neighbor, edge_index = adjacency[vertex].pop()
        if edge_index in used_edges:
            continue
        used_edges.add(edge_index)
        stack.append(neighbor)

    path.reverse()
    if len(used_edges) != len(edges) or len(path) != len(edges) + 1:
        raise ValueError("Hierholzer failed to consume every edge")
    return path


def difficulty(vertex_count: int, edge_count: int) -> str:
    if edge_count <= vertex_count:
        return "easy"
    if edge_count <= vertex_count + 3:
        return "medium"
    return "hard"


def canonical_edge(a: str, b: str) -> Edge:
    return (a, b) if a <= b else (b, a)


def _category(record: dict[str, Any]) -> str:
    tags = set(record["tags"])
    if "connectivity:disconnected" in tags:
        return "disconnected"
    if "solution:no" in tags:
        return "connected_no_euler"
    return "solvable"


if __name__ == "__main__":
    raise SystemExit(main())

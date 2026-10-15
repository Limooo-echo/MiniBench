from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import networkx as nx
from PIL import Image, ImageDraw

from minibench.datasets.one_stroke.dataset import OneStrokeTask


ONE_STROKE_RENDERER_VERSION = "multimodal-v3"
ONE_STROKE_RENDER_SEED = 20260813


def one_stroke_layout_issues(
    task: OneStrokeTask, *, node_clearance: float = 0.19, edge_clearance: float = 0.10,
) -> list[str]:
    """Detect ambiguous edge/node overlap without changing the v3 renderer.

    Arc3 edges are quadratic Beziers. Sampling the unshrunk curve is
    conservative for nonincident nodes and matches the renderer's radii.
    New dataset candidates use their eventual multimodal IDs, since layouts
    are keyed by ID. Original released graphs are intentionally grandfathered.
    """
    positions = _layout(task)
    issues: list[str] = []
    for index, a in enumerate(task.vertices):
        for b in task.vertices[index + 1:]:
            if math.dist(positions[a], positions[b]) < node_clearance:
                issues.append(f"nodes:{a}:{b}")
    counts = Counter(_canonical_edge(edge) for edge in task.edges)
    seen: defaultdict[tuple[str, str], int] = defaultdict(int)
    for a, b in task.edges:
        key = _canonical_edge((a, b))
        index = seen[key]
        seen[key] += 1
        rad = 0.0 if counts[key] == 1 else 0.22 * (index - (counts[key] - 1) / 2)
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        cx, cy = (x1 + x2) / 2 + rad * (y2 - y1), (y1 + y2) / 2 - rad * (x2 - x1)
        samples = [
            ((1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2,
             (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2)
            for t in (step / 256 for step in range(257))
        ]
        for vertex in task.vertices:
            if vertex not in (a, b) and min(math.dist(positions[vertex], point) for point in samples) < edge_clearance:
                issues.append(f"edge:{a}:{b}:{index}:behind:{vertex}")
    return issues


def render_one_stroke_input_png(
    task: OneStrokeTask,
) -> bytes:
    """Render an Agent-facing graph image without IDs, answers, or path hints."""

    positions = _layout(task)

    figure, axis = plt.subplots(figsize=(7.2, 7.2), dpi=120)
    figure.patch.set_facecolor("#faf9f6")
    axis.set_facecolor("#faf9f6")
    axis.set_aspect("equal")
    axis.axis("off")

    _draw_edges(axis, task.edges, positions)
    for vertex in task.vertices:
        x, y = positions[vertex]
        axis.add_patch(
            Circle(
                (x, y),
                radius=0.085,
                facecolor="white",
                edgecolor="#111827",
                linewidth=2.5,
                zorder=5,
            )
        )
        axis.text(
            x,
            y,
            vertex,
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            color="#111827",
            zorder=6,
        )

    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-1.25, 1.25)
    figure.tight_layout(pad=0.15)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    return buffer.getvalue()


def write_one_stroke_input_png(
    task: OneStrokeTask,
    output: str | Path,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_one_stroke_input_png(task))
    return path


def write_contact_sheet(
    image_paths: Sequence[str | Path],
    output: str | Path,
    *,
    columns: int = 5,
    thumbnail_size: int = 240,
) -> Path:
    paths = [Path(path) for path in image_paths]
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new(
        "RGB",
        (columns * thumbnail_size, rows * (thumbnail_size + 28)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumbnail_size - 8, thumbnail_size - 8))
        x = (index % columns) * thumbnail_size + (thumbnail_size - image.width) // 2
        y = (index // columns) * (thumbnail_size + 28)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + thumbnail_size + 4), path.stem, fill="black")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=True)
    return output_path


def _layout(task: OneStrokeTask) -> dict[str, tuple[float, float]]:
    graph = nx.Graph()
    graph.add_nodes_from(task.vertices)
    graph.add_edges_from(task.edges)
    if nx.number_connected_components(graph) > 1:
        raw = nx.circular_layout(graph, scale=0.9)
    else:
        raw = nx.spring_layout(
            graph,
            seed=_stable_seed(task.id, "layout") % (2**32),
            iterations=250,
            k=max(0.55, 1.7 / math.sqrt(len(task.vertices))),
            scale=0.92,
        )
    return {
        vertex: (float(raw[vertex][0]), float(raw[vertex][1]))
        for vertex in task.vertices
    }


def _draw_edges(
    axis: plt.Axes,
    edges: tuple[tuple[str, str], ...],
    positions: dict[str, tuple[float, float]],
) -> None:
    counts = Counter(_canonical_edge(edge) for edge in edges)
    seen: defaultdict[tuple[str, str], int] = defaultdict(int)
    for edge in edges:
        key = _canonical_edge(edge)
        index = seen[key]
        seen[key] += 1
        count = counts[key]
        if count == 1:
            radii = [0.0]
        else:
            radii = [0.22 * (item - (count - 1) / 2) for item in range(count)]
        patch = FancyArrowPatch(
            positions[edge[0]],
            positions[edge[1]],
            arrowstyle="-",
            connectionstyle=f"arc3,rad={radii[index]}",
            linewidth=3.2,
            color="#243447",
            shrinkA=12,
            shrinkB=12,
            zorder=2,
        )
        axis.add_patch(patch)


def _stable_seed(task_id: str, purpose: str) -> int:
    digest = hashlib.sha256(
        f"{ONE_STROKE_RENDER_SEED}:{task_id}:{purpose}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _canonical_edge(edge: tuple[str, str]) -> tuple[str, str]:
    a, b = edge
    return (a, b) if a <= b else (b, a)

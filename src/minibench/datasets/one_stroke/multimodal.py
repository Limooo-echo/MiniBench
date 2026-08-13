from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import math
from pathlib import Path
import random
from typing import Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import networkx as nx
from PIL import Image, ImageDraw, ImageFilter

from minibench.datasets.one_stroke.dataset import OneStrokeTask


ONE_STROKE_RENDERER_VERSION = "a4-v1"
ONE_STROKE_RENDER_SEED = 20260813


def render_one_stroke_input_png(
    task: OneStrokeTask,
    *,
    variant: Literal["clear", "challenge"],
) -> bytes:
    """Render an Agent-facing graph image without IDs, answers, or path hints."""

    if variant not in {"clear", "challenge"}:
        raise ValueError("variant must be clear or challenge")
    effective_variant = "clear" if task.difficulty == "easy" else variant
    seed = _stable_seed(task.id, effective_variant)
    rng = random.Random(seed)
    positions = _layout(task, effective_variant, rng)

    figure, axis = plt.subplots(figsize=(7.2, 7.2), dpi=120)
    figure.patch.set_facecolor("#faf9f6")
    axis.set_facecolor("#faf9f6")
    axis.set_aspect("equal")
    axis.axis("off")

    if effective_variant == "challenge":
        _draw_background_noise(axis, task.difficulty, rng)
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
    png = buffer.getvalue()
    if effective_variant == "challenge" and task.difficulty == "hard":
        image = Image.open(BytesIO(png)).convert("RGB")
        image = image.filter(ImageFilter.GaussianBlur(radius=0.45))
        output = BytesIO()
        image.save(output, format="PNG", optimize=True)
        png = output.getvalue()
    return png


def write_one_stroke_input_png(
    task: OneStrokeTask,
    output: str | Path,
    *,
    variant: Literal["clear", "challenge"],
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_one_stroke_input_png(task, variant=variant))
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


def _layout(
    task: OneStrokeTask,
    variant: str,
    rng: random.Random,
) -> dict[str, tuple[float, float]]:
    if variant == "clear":
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

    order = list(task.vertices)
    rng.shuffle(order)
    radius = 0.73 if task.difficulty == "medium" else 0.61
    positions: dict[str, tuple[float, float]] = {}
    for index, vertex in enumerate(order):
        angle = 2 * math.pi * index / len(order) + 0.23
        radial_jitter = rng.uniform(-0.07, 0.07)
        positions[vertex] = (
            (radius + radial_jitter) * math.cos(angle),
            (radius + radial_jitter) * math.sin(angle),
        )
    return positions


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


def _draw_background_noise(
    axis: plt.Axes,
    difficulty: str,
    rng: random.Random,
) -> None:
    count = 18 if difficulty == "medium" else 42
    for _ in range(count):
        x = rng.uniform(-1.12, 1.12)
        y = rng.uniform(-1.12, 1.12)
        size = rng.uniform(4.0, 10.0)
        axis.scatter([x], [y], s=size, color="#9ca3af", alpha=0.18, zorder=0)


def _stable_seed(task_id: str, variant: str) -> int:
    digest = hashlib.sha256(
        f"{ONE_STROKE_RENDER_SEED}:{task_id}:{variant}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _canonical_edge(edge: tuple[str, str]) -> tuple[str, str]:
    a, b = edge
    return (a, b) if a <= b else (b, a)

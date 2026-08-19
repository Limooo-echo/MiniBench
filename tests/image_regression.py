from __future__ import annotations

import json
from hashlib import sha256
import math
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

from PIL import Image, ImageChops


DEFAULT_RMS_LIMIT = 2.5
DEFAULT_CHANGED_PIXEL_LIMIT = 0.01
DEFAULT_CHANNEL_DELTA = 8


def canonical_rgba(path: str | Path) -> tuple[tuple[int, int], bytes]:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        return rgba.size, rgba.tobytes()


def assert_png_deterministic(testcase: Any, first: str | Path, second: str | Path) -> None:
    first_size, first_pixels = canonical_rgba(first)
    second_size, second_pixels = canonical_rgba(second)
    testcase.assertEqual(first_size, second_size)
    testcase.assertEqual(first_pixels, second_pixels)


def assert_png_visually_equal(
    testcase: Any,
    actual: str | Path,
    expected: str | Path,
    *,
    artifact_name: str,
    rms_limit: float = DEFAULT_RMS_LIMIT,
    changed_pixel_limit: float = DEFAULT_CHANGED_PIXEL_LIMIT,
    channel_delta: int = DEFAULT_CHANNEL_DELTA,
) -> None:
    actual_path = Path(actual)
    expected_path = Path(expected)
    with Image.open(actual_path) as actual_image, Image.open(expected_path) as expected_image:
        actual_mode = actual_image.mode
        expected_mode = expected_image.mode
        actual_rgb = actual_image.convert("RGB")
        expected_rgb = expected_image.convert("RGB")
        same_size = actual_rgb.size == expected_rgb.size
        same_mode = actual_mode == expected_mode
        difference = _difference_image(actual_rgb, expected_rgb)
        if not same_size or not same_mode:
            _write_failure_artifacts(
                artifact_name,
                actual_path,
                expected_path,
                difference,
                {
                    "actual_mode": actual_mode,
                    "expected_mode": expected_mode,
                    "actual_size": actual_rgb.size,
                    "expected_size": expected_rgb.size,
                    "environment": renderer_environment(),
                },
            )
        testcase.assertEqual(actual_mode, expected_mode)
        testcase.assertEqual(actual_rgb.size, expected_rgb.size)
        histogram = difference.histogram()
        channel_samples = actual_rgb.width * actual_rgb.height * 3
        squared_error = sum(
            count * ((bucket % 256) ** 2)
            for bucket, count in enumerate(histogram)
        )
        rms = math.sqrt(squared_error / channel_samples) if channel_samples else 0.0
        raw_difference = difference.tobytes()
        changed_pixels = sum(
            max(raw_difference[index : index + 3]) > channel_delta
            for index in range(0, len(raw_difference), 3)
        )
        pixel_count = actual_rgb.width * actual_rgb.height
        changed_fraction = changed_pixels / pixel_count if pixel_count else 0.0

        if rms > rms_limit or changed_fraction > changed_pixel_limit:
            _write_failure_artifacts(
                artifact_name,
                actual_path,
                expected_path,
                difference,
                {
                    "rms": rms,
                    "rms_limit": rms_limit,
                    "changed_fraction": changed_fraction,
                    "changed_pixel_limit": changed_pixel_limit,
                    "channel_delta": channel_delta,
                    "environment": renderer_environment(),
                },
            )
        testcase.assertLessEqual(rms, rms_limit, f"{artifact_name}: RGB RMS={rms:.4f}")
        testcase.assertLessEqual(
            changed_fraction,
            changed_pixel_limit,
            f"{artifact_name}: changed pixel fraction={changed_fraction:.4%}",
        )


def renderer_environment() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    for module_name in ("PIL", "matplotlib", "networkx", "numpy"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - diagnostic only
            versions[module_name] = f"unavailable: {exc}"
    try:
        from minibench.assets.fonts import font_path

        for weight in ("regular", "bold"):
            path = font_path(bold=weight == "bold")
            digest = sha256(path.read_bytes()).hexdigest()
            versions[f"font_{weight}"] = f"{path.name} sha256={digest}"
    except Exception as exc:  # pragma: no cover - diagnostic only
        versions["fonts"] = f"unavailable: {exc}"
    return versions


def _difference_image(actual: Image.Image, expected: Image.Image) -> Image.Image:
    if actual.size == expected.size:
        return ImageChops.difference(actual, expected)
    width = max(actual.width, expected.width)
    height = max(actual.height, expected.height)
    actual_canvas = Image.new("RGB", (width, height), "white")
    expected_canvas = Image.new("RGB", (width, height), "white")
    actual_canvas.paste(actual, (0, 0))
    expected_canvas.paste(expected, (0, 0))
    return ImageChops.difference(actual_canvas, expected_canvas)


def _write_failure_artifacts(
    artifact_name: str,
    actual: Path,
    expected: Path,
    difference: Image.Image,
    report: dict[str, Any],
) -> None:
    artifact_root = os.environ.get("CI_ARTIFACT_DIR")
    if not artifact_root:
        return
    directory = Path(artifact_root) / artifact_name
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(actual, directory / "actual.png")
    shutil.copy2(expected, directory / "expected.png")
    difference.save(directory / "diff.png", format="PNG")
    (directory / "environment.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

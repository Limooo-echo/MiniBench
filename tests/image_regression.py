from __future__ import annotations

import json
from hashlib import sha256
import math
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Sequence

from PIL import Image, ImageChops, ImageDraw, ImageFilter


DEFAULT_RMS_LIMIT = 2.5
DEFAULT_CHANGED_PIXEL_LIMIT = 0.01
DEFAULT_CHANNEL_DELTA = 8

PixelRegion = tuple[int, int, int, int]
TextRegion = tuple[str, PixelRegion, tuple[int, int, int]]


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
    ignored_regions: Sequence[PixelRegion] = (),
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
        comparison_difference, compared_pixels = _mask_difference_regions(
            difference,
            ignored_regions,
        )
        histogram = comparison_difference.histogram()
        channel_samples = compared_pixels * 3
        squared_error = sum(
            count * ((bucket % 256) ** 2)
            for bucket, count in enumerate(histogram)
        )
        rms = math.sqrt(squared_error / channel_samples) if channel_samples else 0.0
        raw_difference = comparison_difference.tobytes()
        changed_pixels = sum(
            max(raw_difference[index : index + 3]) > channel_delta
            for index in range(0, len(raw_difference), 3)
        )
        changed_fraction = changed_pixels / compared_pixels if compared_pixels else 0.0

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
                    "ignored_regions": [list(region) for region in ignored_regions],
                    "compared_pixels": compared_pixels,
                    "environment": renderer_environment(),
                },
            )
        testcase.assertLessEqual(rms, rms_limit, f"{artifact_name}: RGB RMS={rms:.4f}")
        testcase.assertLessEqual(
            changed_fraction,
            changed_pixel_limit,
            f"{artifact_name}: changed pixel fraction={changed_fraction:.4%}",
        )


def assert_text_regions_similar(
    testcase: Any,
    actual: str | Path,
    expected: str | Path,
    *,
    regions: Sequence[TextRegion],
    artifact_name: str,
    foreground_threshold: int = 16,
    dilation_radius: int = 2,
    bbox_tolerance: int = 4,
    foreground_count_tolerance: float = 0.15,
    outside_fraction_limit: float = 0.03,
) -> None:
    """Compare text geometry while ignoring platform-specific anti-aliasing.

    Each region is segmented against its known solid background. The binary
    glyph masks must occupy similar bounds and remain within a small dilation
    of one another, so changed or missing text still fails without requiring
    identical FreeType edge pixels on Windows and Linux.
    """
    actual_path = Path(actual)
    expected_path = Path(expected)
    with Image.open(actual_path) as actual_image, Image.open(expected_path) as expected_image:
        actual_rgb = actual_image.convert("RGB")
        expected_rgb = expected_image.convert("RGB")
        testcase.assertEqual(actual_rgb.size, expected_rgb.size)
        difference = _difference_image(actual_rgb, expected_rgb)
        metrics: list[dict[str, Any]] = []

        for name, region, background in regions:
            _validate_region(region, actual_rgb.size)
            actual_mask = _foreground_mask(
                actual_rgb.crop(region),
                background,
                threshold=foreground_threshold,
            )
            expected_mask = _foreground_mask(
                expected_rgb.crop(region),
                background,
                threshold=foreground_threshold,
            )
            actual_bbox = actual_mask.getbbox()
            expected_bbox = expected_mask.getbbox()
            actual_pixels = actual_mask.histogram()[255]
            expected_pixels = expected_mask.histogram()[255]
            size_denominator = max(actual_pixels, expected_pixels, 1)
            count_delta = abs(actual_pixels - expected_pixels) / size_denominator

            filter_size = dilation_radius * 2 + 1
            actual_dilated = actual_mask.filter(ImageFilter.MaxFilter(filter_size))
            expected_dilated = expected_mask.filter(ImageFilter.MaxFilter(filter_size))
            actual_outside = ImageChops.subtract(actual_mask, expected_dilated).histogram()[255]
            expected_outside = ImageChops.subtract(expected_mask, actual_dilated).histogram()[255]
            actual_outside_fraction = actual_outside / max(actual_pixels, 1)
            expected_outside_fraction = expected_outside / max(expected_pixels, 1)

            bbox_delta = None
            if actual_bbox is not None and expected_bbox is not None:
                bbox_delta = max(
                    abs(actual_value - expected_value)
                    for actual_value, expected_value in zip(actual_bbox, expected_bbox)
                )
            metrics.append(
                {
                    "name": name,
                    "region": list(region),
                    "actual_bbox": list(actual_bbox) if actual_bbox is not None else None,
                    "expected_bbox": list(expected_bbox) if expected_bbox is not None else None,
                    "bbox_delta": bbox_delta,
                    "actual_foreground_pixels": actual_pixels,
                    "expected_foreground_pixels": expected_pixels,
                    "foreground_count_delta": count_delta,
                    "actual_outside_fraction": actual_outside_fraction,
                    "expected_outside_fraction": expected_outside_fraction,
                }
            )

        failed = any(
            metric["actual_bbox"] is None
            or metric["expected_bbox"] is None
            or metric["bbox_delta"] > bbox_tolerance
            or metric["foreground_count_delta"] > foreground_count_tolerance
            or metric["actual_outside_fraction"] > outside_fraction_limit
            or metric["expected_outside_fraction"] > outside_fraction_limit
            for metric in metrics
        )
        if failed:
            _write_failure_artifacts(
                artifact_name,
                actual_path,
                expected_path,
                difference,
                {
                    "text_regions": metrics,
                    "foreground_threshold": foreground_threshold,
                    "dilation_radius": dilation_radius,
                    "bbox_tolerance": bbox_tolerance,
                    "foreground_count_tolerance": foreground_count_tolerance,
                    "outside_fraction_limit": outside_fraction_limit,
                    "environment": renderer_environment(),
                },
            )

        for metric in metrics:
            name = metric["name"]
            testcase.assertIsNotNone(metric["actual_bbox"], f"{name}: rendered text is missing")
            testcase.assertIsNotNone(metric["expected_bbox"], f"{name}: golden text is missing")
            testcase.assertLessEqual(
                metric["bbox_delta"],
                bbox_tolerance,
                f"{name}: text bounds moved by {metric['bbox_delta']} pixels",
            )
            testcase.assertLessEqual(
                metric["foreground_count_delta"],
                foreground_count_tolerance,
                f"{name}: text foreground area changed by "
                f"{metric['foreground_count_delta']:.2%}",
            )
            testcase.assertLessEqual(
                metric["actual_outside_fraction"],
                outside_fraction_limit,
                f"{name}: actual glyph shape diverged by "
                f"{metric['actual_outside_fraction']:.2%}",
            )
            testcase.assertLessEqual(
                metric["expected_outside_fraction"],
                outside_fraction_limit,
                f"{name}: golden glyph shape diverged by "
                f"{metric['expected_outside_fraction']:.2%}",
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


def _mask_difference_regions(
    difference: Image.Image,
    ignored_regions: Sequence[PixelRegion],
) -> tuple[Image.Image, int]:
    masked = difference.copy()
    comparison_mask = Image.new("L", difference.size, 255)
    masked_draw = ImageDraw.Draw(masked)
    mask_draw = ImageDraw.Draw(comparison_mask)
    for region in ignored_regions:
        _validate_region(region, difference.size)
        left, top, right, bottom = region
        inclusive_region = (left, top, right - 1, bottom - 1)
        masked_draw.rectangle(inclusive_region, fill="black")
        mask_draw.rectangle(inclusive_region, fill=0)
    compared_pixels = comparison_mask.histogram()[255]
    if compared_pixels == 0:
        raise ValueError("ignored_regions exclude every image pixel")
    return masked, compared_pixels


def _foreground_mask(
    image: Image.Image,
    background: tuple[int, int, int],
    *,
    threshold: int,
) -> Image.Image:
    raw = image.convert("RGB").tobytes()
    foreground = bytearray(len(raw) // 3)
    for source_index in range(0, len(raw), 3):
        pixel = raw[source_index : source_index + 3]
        foreground[source_index // 3] = (
            255
            if max(abs(pixel[channel] - background[channel]) for channel in range(3))
            > threshold
            else 0
        )
    return Image.frombytes("L", image.size, bytes(foreground))


def _validate_region(region: PixelRegion, image_size: tuple[int, int]) -> None:
    left, top, right, bottom = region
    width, height = image_size
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"invalid image region {region} for size {image_size}")


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

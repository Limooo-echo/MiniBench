from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minibench.datasets.one_stroke.dataset import one_stroke_task_from_dict
from minibench.datasets.one_stroke.multimodal import (
    ONE_STROKE_RENDERER_VERSION,
    ONE_STROKE_RENDER_SEED,
    write_contact_sheet,
    write_one_stroke_input_png,
)
from scripts.build_one_stroke_direct_history import legacy_direct_records


def build_multimodal_dataset(
    source: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
    contact_sheet: str | Path | None = None,
) -> dict[str, object]:
    source_path = Path(source)
    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"output already exists: {output_path}; pass --overwrite")
    source_records = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not source_records:
        raise ValueError("Multimodal requires a non-empty direct collection")
    if len({raw["id"] for raw in source_records}) != len(source_records):
        raise ValueError("Multimodal source contains duplicate direct IDs")

    image_root = output_path.parent / "images"
    legacy_by_id = {raw["id"]: raw for raw in legacy_direct_records()}
    legacy_images = Path(__file__).resolve().parents[1] / "data/one_stroke/images"
    records: list[dict[str, object]] = []
    image_paths: list[Path] = []
    for raw in source_records:
        source_id = str(raw["id"])
        if not source_id.startswith("direct-"):
            raise ValueError(f"unexpected direct id: {source_id}")
        task_id = "multimodal-" + source_id.removeprefix("direct-")
        record = {
            key: value
            for key, value in raw.items()
            if key not in {"id", "capability", "tags"}
        }
        record.update(
            {
                "id": task_id,
                "source_task_id": source_id,
                "capability": "multimodal",
                "image": f"images/{task_id}.png",
                "renderer_version": ONE_STROKE_RENDERER_VERSION,
                "render_seed": ONE_STROKE_RENDER_SEED,
                "tags": [
                    "one-stroke",
                    "benchmark:multimodal",
                    "capability:multimodal",
                    f"difficulty:{raw['difficulty']}",
                    f"solution:{'yes' if raw['solution_exists'] else 'no'}",
                    "source:direct-paired",
                ],
            }
        )
        render_task = one_stroke_task_from_dict(
            {
                **record,
                "image": None,
                "source_task_id": None,
                "capability": "direct",
            }
        )
        image_path = image_root / f"{task_id}.png"
        legacy = legacy_by_id.get(source_id)
        original_image = legacy_images / f"{task_id}.png"
        if legacy is not None and original_image.is_file() and all(
            raw.get(key) == legacy.get(key)
            for key in ("vertices", "edges", "start", "end", "solution_exists", "solution_path")
        ):
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(original_image.read_bytes())
        else:
            write_one_stroke_input_png(render_task, image_path)
        image_paths.append(image_path)
        records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    if contact_sheet is not None:
        write_contact_sheet(image_paths, contact_sheet)
        # Small, readable sheets cover every image for human visual QA.
        for index in range(0, len(image_paths), 20):
            page = Path(contact_sheet).with_name(
                f"{Path(contact_sheet).stem}-{index // 20 + 1:02d}.png"
            )
            write_contact_sheet(image_paths[index:index + 20], page, columns=4, thumbnail_size=320)
    report = {
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "records": [{"id": raw["id"], "source_task_id": raw["source_task_id"],
                     "image": raw["image"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for raw, path in zip(records, image_paths)],
    }
    (output_path.parent / "multimodal-validation-20261013.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "source": str(source_path),
        "output": str(output_path),
        "tasks": len(records),
        "images": len(records),
        "renderer_version": ONE_STROKE_RENDERER_VERSION,
        "seed": ONE_STROKE_RENDER_SEED,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paired one-stroke multimodal images.")
    parser.add_argument("--source", type=Path, default=Path("data/one_stroke/direct.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/one_stroke/multimodal.jsonl"))
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_multimodal_dataset(
                args.source,
                args.output,
                overwrite=args.overwrite,
                contact_sheet=args.contact_sheet,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

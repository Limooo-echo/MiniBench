from __future__ import annotations

import argparse
import json
from pathlib import Path

from minibench.datasets.one_stroke.dataset import one_stroke_task_from_dict
from minibench.datasets.one_stroke.multimodal import (
    ONE_STROKE_RENDERER_VERSION,
    ONE_STROKE_RENDER_SEED,
    write_contact_sheet,
    write_one_stroke_input_png,
)


def build_a4_dataset(
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
    if len(source_records) != 30:
        raise ValueError("A4 must be derived from exactly 30 A1 records")

    image_root = output_path.parent / "a4_images"
    records: list[dict[str, object]] = []
    challenge_paths: list[Path] = []
    for raw in source_records:
        source_id = str(raw["id"])
        if not source_id.startswith("a1-"):
            raise ValueError(f"unexpected A1 id: {source_id}")
        task_id = "a4-" + source_id.removeprefix("a1-")
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
                "image_variants": {
                    "clear": f"a4_images/clear/{task_id}.png",
                    "challenge": f"a4_images/challenge/{task_id}.png",
                },
                "renderer_version": ONE_STROKE_RENDERER_VERSION,
                "render_seed": ONE_STROKE_RENDER_SEED,
                "tags": [
                    "one-stroke",
                    "benchmark:a4",
                    "capability:multimodal",
                    f"difficulty:{raw['difficulty']}",
                    f"solution:{'yes' if raw['solution_exists'] else 'no'}",
                    "source:a1-paired",
                ],
            }
        )
        render_task = one_stroke_task_from_dict(
            {
                **record,
                "image_variants": {},
                "source_task_id": None,
                "capability": "direct",
            }
        )
        for variant in ("clear", "challenge"):
            image_path = image_root / variant / f"{task_id}.png"
            write_one_stroke_input_png(render_task, image_path, variant=variant)
            if variant == "challenge":
                challenge_paths.append(image_path)
        records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    if contact_sheet is not None:
        write_contact_sheet(challenge_paths, contact_sheet)
    return {
        "source": str(source_path),
        "output": str(output_path),
        "tasks": len(records),
        "images": len(records) * 2,
        "renderer_version": ONE_STROKE_RENDERER_VERSION,
        "seed": ONE_STROKE_RENDER_SEED,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paired one-stroke A4 images.")
    parser.add_argument("--source", type=Path, default=Path("data/one_stroke/a1_direct.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/one_stroke/a4_multimodal.jsonl"))
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_a4_dataset(
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

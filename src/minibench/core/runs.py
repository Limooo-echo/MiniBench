from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import strftime
from typing import Any, Iterable


def timestamped_run_dir(
    output_dir: str | Path,
    *,
    run_name: str | None = None,
    prefix: str | None = None,
) -> Path:
    root = Path(output_dir)
    if run_name:
        name = run_name
    elif prefix:
        name = f"{prefix}-{strftime('%Y%m%d-%H%M%S')}"
    else:
        name = strftime("%Y%m%d-%H%M%S")
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "__dataclass_fields__"):
                payload = asdict(record)
            else:
                payload = record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_summary_artifacts(
    run_dir: str | Path,
    *,
    results: Iterable[Any],
    summary: dict[str, Any],
    summary_line: str,
) -> Path:
    run_path = Path(run_dir)
    write_jsonl(run_path / "predictions.jsonl", results)
    (run_path / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_path / "summary.txt").write_text(summary_line, encoding="utf-8")
    return run_path

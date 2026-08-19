from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from minibench.datasets.xiangqi.schema import FAMILY_PATHS, XIANGQI_FAMILIES, load_records


def default_mapping_path() -> Path:
    return Path(__file__).resolve().parents[4] / "data/xiangqi/migration_v1_to_v2.json"


def load_mapping(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path else default_mapping_path()
    return json.loads(source.read_text(encoding="utf-8"))


def _canonical_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for family in XIANGQI_FAMILIES:
        for record in load_records(FAMILY_PATHS[family], expected_family=family):
            records[record["id"]] = record
    return records


def migrate_xiangqi_v2(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dry_run: bool = False,
    mapping_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(input_path)
    destination = Path(output_path)
    if not source.exists():
        raise ValueError(f"input does not exist: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("in-place migration is forbidden")
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    mapping = load_mapping(mapping_path)
    canonical = _canonical_records()
    report: dict[str, Any] = {
        "input": str(source),
        "output": str(destination),
        "dry_run": dry_run,
        "converted": 0,
        "unchanged": 0,
        "unrecognized": [],
        "files": [],
    }

    if source.is_dir():
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source)
            output_file = destination / relative
            if path.suffix.lower() in {".json", ".jsonl"}:
                _migrate_file(path, output_file, mapping, canonical, report, dry_run)
            else:
                report["files"].append({"path": relative.as_posix(), "status": "copied"})
                if not dry_run:
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, output_file)
        report_path = destination / "migration-report.json"
    else:
        _migrate_file(source, destination, mapping, canonical, report, dry_run)
        report_path = destination.parent / "migration-report.json"

    report["unrecognized"] = sorted(set(report["unrecognized"]))
    if not dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report


def _migrate_file(
    source: Path,
    destination: Path,
    mapping: dict[str, Any],
    canonical: dict[str, dict[str, Any]],
    report: dict[str, Any],
    dry_run: bool,
) -> None:
    if source.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        migrated = [_rewrite_value(row, mapping, canonical, report) for row in rows]
        serialized = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in migrated
        )
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        migrated = _rewrite_value(payload, mapping, canonical, report)
        serialized = json.dumps(migrated, indent=2, ensure_ascii=False) + "\n"
    report["files"].append({"path": source.name, "status": "converted"})
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized, encoding="utf-8", newline="\n")


def _rewrite_value(
    value: Any,
    mapping: dict[str, Any],
    canonical: dict[str, dict[str, Any]],
    report: dict[str, Any],
    *,
    key: str | None = None,
) -> Any:
    if key in {"raw_output", "raw", "response", "model_output", "prompt"}:
        report["unchanged"] += 1
        return value
    if isinstance(value, list):
        return [_rewrite_value(item, mapping, canonical, report) for item in value]
    if not isinstance(value, dict):
        return _rewrite_scalar(value, key, mapping, report)

    old_id = value.get("id")
    if "board" in value and isinstance(old_id, str):
        new_id = mapping["task_ids"].get(old_id)
        if new_id in canonical:
            report["converted"] += 1
            return dict(canonical[new_id])

    rewritten: dict[str, Any] = {}
    for old_key, item in value.items():
        if old_key in {"raw_output", "raw", "response", "model_output", "prompt"}:
            rewritten[old_key] = item
            report["unchanged"] += 1
            continue
        new_key = old_key
        if old_key == "group":
            new_key = "ruleset"
        elif old_key == "variant":
            if item is None:
                report["converted"] += 1
                continue
            new_key = "ruleset"
        rewritten[new_key] = _rewrite_value(
            item, mapping, canonical, report, key=new_key
        )
        if new_key != old_key:
            report["converted"] += 1
    if rewritten == value:
        report["unchanged"] += 1
    return rewritten


def _rewrite_scalar(
    value: Any,
    key: str | None,
    mapping: dict[str, Any],
    report: dict[str, Any],
) -> Any:
    if not isinstance(value, str):
        return value
    if key in {"id", "task_id", "source_task_id"}:
        if value in mapping["task_ids"]:
            report["converted"] += 1
            return mapping["task_ids"][value]
        if value in mapping["scenario_ids"]:
            report["converted"] += 1
            return mapping["scenario_ids"][value]
        if value.startswith(("d3-", "c2-", "h2-", "vis-")):
            report["unrecognized"].append(value)
        return value
    if key in {"family", "task", "task_name"}:
        for legacy, replacement in mapping["family_names"].items():
            if value.upper() == legacy:
                report["converted"] += 1
                return replacement
    enum_group = {
        "history_mode": "history_mode",
        "input_mode": "input_mode",
        "mode": "input_mode",
        "ruleset": "ruleset",
    }.get(key)
    if enum_group is not None:
        replacement = mapping["enums"][enum_group].get(value)
        if replacement is not None:
            report["converted"] += int(replacement != value)
            return replacement
    return value

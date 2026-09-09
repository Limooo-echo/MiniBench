"""Read a frozen experiment roster and join it to existing result files."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

DEFAULT_WEIGHTS = {
    "zebra": {"D": .35, "R": .40, "H": .35, "V": 0.0},
    "one_stroke": {"D": .30, "R": 0.0, "H": .35, "V": .40},
    "xiangqi": {"D": .20, "R": .35, "H": .20, "V": .35},
    "mahjong": {"D": .15, "R": .25, "H": .10, "V": .25},
}
# Modes are measurement conditions, not architecture hyperparameters.
CONDITIONS = {
    ("zebra", "D"): ("zebra", ("single",)),
    ("zebra", "R"): ("zebra", ("single",)),
    ("zebra", "H"): ("zebra", ("deferred_reasoning",)),
    ("one_stroke", "D"): ("one_stroke", ("single",)),
    ("one_stroke", "H"): ("one_stroke", ("incremental_state", "step_history_only")),
    ("one_stroke", "V"): ("one_stroke", ("image",)),
    ("xiangqi_mate_in_one", "D"): ("xiangqi", ("single",)),
    ("xiangqi_rule_variants", "R"): ("xiangqi", ("single",)),
    ("xiangqi_history", "H"): ("xiangqi", ("move-history-only",)),
    ("xiangqi_multimodal", "V"): ("xiangqi", ("chinese-piece-image", "latin-piece-image")),
    ("mahjong", "D"): ("mahjong", ("text",)),
    ("mahjong", "V"): ("mahjong", ("image",)),
    ("mahjong_solo", "H"): ("mahjong", ("history-only",)),
    ("mahjong_rule_variants", "R"): ("mahjong", ("full-hand",)),
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_document(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a per-item object")
        result.append(value)
    return result


def load_weights(path: Path | None) -> dict:
    config = read_document(path) if path else {"weights": DEFAULT_WEIGHTS}
    weights = config.get("weights")
    if not isinstance(weights, dict) or set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError("weights must contain exactly zebra, one_stroke, xiangqi, mahjong")
    for task, columns in weights.items():
        if not isinstance(columns, dict) or set(columns) != set("DRHV"):
            raise ValueError(f"{task}: weights must contain D, R, H, V")
        for dim, value in columns.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid weight: {task}/{dim}")
            if (value > 0) != (DEFAULT_WEIGHTS[task][dim] > 0):
                raise ValueError("v1 weights must preserve the declared participating tasks")
    for dim in "DRHV":
        if not math.isclose(sum(w[dim] for w in weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"{dim} weights must sum to one")
    cap = config.get("partial_credit_cap", .25)
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or not 0 <= cap <= 1:
        raise ValueError("partial_credit_cap must be between zero and one")
    samples = config.get("bootstrap_samples", 2000)
    seed = config.get("seed", 20260909)
    if type(samples) is not int or samples < 0 or type(seed) is not int or seed < 0:
        raise ValueError("bootstrap_samples and seed must be nonnegative integers")
    return {"weights": weights, "partial_credit_cap": float(cap), "bootstrap_samples": samples, "seed": seed}


def _named(entries: Any, label: str) -> dict[str, dict]:
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{label} must be a nonempty list")
    result = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"].strip():
            raise ValueError(f"{label}: every entry needs a nonempty id")
        if entry["id"] in result:
            raise ValueError(f"{label}: duplicate id {entry['id']}")
        result[entry["id"]] = entry
    return result


def _path(base: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a nonempty file path")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _difficulty(task: dict, family: str) -> str:
    if task.get("difficulty"):
        return str(task["difficulty"])
    if family == "zebra" and task.get("size"):
        from minibench.datasets.zebra.dataset import difficulty_for_size
        return difficulty_for_size(task["size"])
    tags = task.get("tags", [])
    labels = {str(t).split(":", 1)[1] for t in tags if str(t).startswith("difficulty:")}
    labels.update(t for t in tags if t in {"easy", "medium", "hard"})
    if len(labels) > 1:
        raise ValueError(f"Conflicting difficulty labels: {task['id']}")
    return next(iter(labels), "unspecified")


def _tasks(path: Path, family: str, dim: str) -> tuple[list[dict], list[dict]]:
    raw = read_jsonl(path)
    if family == "mahjong_rule_variants":
        from minibench.datasets.mahjong_rule_variants.rules import RULE_CHANNELS
        raw = [dict(t, id=f"{t['id']}--{channel}", source_task_id=t["id"], channel=channel)
               for t in raw for channel in RULE_CHANNELS if channel != "standard"]
    selected, excluded = [], []
    for task in raw:
        if not isinstance(task.get("id"), str):
            raise ValueError(f"Dataset item has no id: {path}")
        reason = None
        cap = task.get("capability", "direct")
        if family == "zebra":
            expected = {"D": {"direct", "direct_reasoning"}, "H": {"history_memory", "history", "memory"},
                        "R": {"rule_condition", "rule"}}[dim]
            if cap not in expected:
                reason = "different_capability"
            if dim == "R" and task.get("rule_mode") != "temporary_codebook":
                reason = "not_temporary_codebook"
        if family == "one_stroke":
            expected = {"D": {"direct"}, "H": {"history_memory"}, "V": {"multimodal"}}[dim]
            if cap not in expected:
                reason = "different_capability"
        if family == "xiangqi_rule_variants" and (task.get("ruleset") == "standard" or not task.get("rules")):
            reason = "standard_rule_control"
        if reason:
            excluded.append({"task_id": task["id"], "reason": reason})
        else:
            selected.append(task)
    return selected, excluded


def _group(task: dict, family: str, dim: str) -> str:
    if family == "mahjong":
        return str(task.get("goal", "unspecified"))
    if family == "one_stroke":
        if type(task.get("solution_exists")) is not bool:
            raise ValueError(f"{task['id']}: solution_exists must be boolean")
        return "solvable" if task["solution_exists"] else "unsolvable"
    if family == "xiangqi_rule_variants":
        return str(task.get("ruleset") or object_hash(task["rules"]))
    if family == "mahjong_rule_variants":
        return task["channel"]
    return "all"


def _record_mode(record: dict, family: str, dim: str) -> str | None:
    cap = record.get("capability")
    expected_caps = {"D": {"direct", "direct_reasoning"}, "R": {"rule", "rule_condition"},
                     "H": {"history_memory", "history", "memory"}, "V": {"multimodal"}}
    if cap is not None and cap not in expected_caps[dim]:
        return None
    if dim == "V":
        return record.get("input_mode")
    if dim == "H":
        if record.get("input_mode") not in (None, "text"):
            return None
        return record.get("memory_mode") or record.get("history_mode") or record.get("observation_mode")
    if record.get("input_mode") not in (None, "text"):
        return None
    if any(record.get(k) not in (None, "single") for k in ("memory_mode", "history_mode")):
        return None
    if dim == "D" and record.get("rule_mode") is not None:
        return None
    if dim == "D" and record.get("ruleset") not in (None, "standard"):
        return None
    if family == "zebra" and dim == "R" and record.get("rule_mode") not in (None, "temporary_codebook"):
        return None
    if family == "mahjong":
        return record.get("input_mode")
    if family == "mahjong_rule_variants":
        return record.get("observation_mode")
    return "single"


def _provenance_reasons(run: dict, experiment: dict, profile: dict, architecture: dict,
                        dataset_hash: str) -> list[str]:
    provenance = run.get("provenance") or {}
    problems = []
    if provenance.get("verified") is not True or not provenance.get("evidence"):
        problems.append("provenance_not_verified")
    required = {"dataset_sha256": dataset_hash, "model": profile["model"],
                "agent_config": architecture.get("experiment_configs", {}).get(experiment["id"], architecture["config"]),
                "protocol": experiment["protocol"]}
    for name, expected in required.items():
        if provenance.get(name) != expected:
            problems.append(f"provenance_mismatch:{name}")
    return problems


def load_suite(path: Path) -> dict:
    """Expand planned slots before inspecting answers; missing slots retain weight."""
    from minibench.scoring.adapters import score_record

    path = path.resolve()
    manifest = read_document(path)
    if manifest.get("version") != 1 or not isinstance(manifest.get("suite_id"), str):
        raise ValueError("Manifest requires version: 1 and suite_id")
    profiles = _named(manifest.get("profiles"), "profiles")
    architectures = _named(manifest.get("architectures"), "architectures")
    for profile in profiles.values():
        model = profile.get("model")
        if not isinstance(model, dict) or not model.get("name") or not model.get("provider"):
            raise ValueError("Every profile needs model.provider and model.name")
    for arch in architectures.values():
        if not isinstance(arch.get("config"), dict) or arch.get("default_config") is not True:
            raise ValueError("Every architecture requires config and default_config: true")
    experiments = _named(manifest.get("experiments"), "experiments")
    for architecture in architectures.values():
        overrides = architecture.get("experiment_configs", {})
        if not isinstance(overrides, dict) or set(overrides) - set(experiments) or any(not isinstance(v, dict) for v in overrides.values()):
            raise ValueError("experiment_configs must map known experiment ids to complete agent config objects")
    rows, sources, exclusions, warnings = [], [], [], []
    protected = [path]
    seen_cells = set()
    for exp_id, exp in experiments.items():
        family, dim = exp.get("family"), exp.get("dimension")
        if (family, dim) not in CONDITIONS:
            raise ValueError(f"Unsupported v1 condition: {family}/{dim}")
        task_family, modes = CONDITIONS[family, dim]
        if (task_family, dim) in seen_cells:
            raise ValueError(f"Only one frozen experiment per task/dimension: {task_family}/{dim}")
        seen_cells.add((task_family, dim))
        if "modes" in exp and tuple(exp["modes"]) != modes:
            raise ValueError(f"{exp_id}: v1 modes are fixed to {modes}")
        if not isinstance(exp.get("protocol"), dict) or not exp["protocol"]:
            raise ValueError(f"{exp_id}: declare the original evaluation protocol")
        dataset = exp.get("dataset", {})
        data_path = _path(path.parent, dataset.get("path"))
        data_hash = file_hash(data_path)
        if dataset.get("sha256") != data_hash:
            raise ValueError(f"{exp_id}: dataset sha256 mismatch; do not rescore against changed gold data")
        protected.append(data_path)
        tasks, excluded = _tasks(data_path, family, dim)
        source_tasks = {t["id"]: t for t in read_jsonl(data_path)}
        source_tasks.update({t["id"]: t for t in tasks})
        exclusions.extend(dict(experiment=exp_id, **item) for item in excluded)
        selected_ids = exp.get("task_ids")
        if selected_ids is not None:
            if not isinstance(selected_ids, list) or len(selected_ids) != len(set(selected_ids)):
                raise ValueError(f"{exp_id}: task_ids must be unique")
            missing = set(selected_ids) - {t["id"] for t in tasks}
            if missing:
                raise ValueError(f"{exp_id}: unknown or ineligible task_ids: {sorted(missing)}")
            tasks = [t for t in tasks if t["id"] in set(selected_ids)]
        if not tasks or len({t["id"] for t in tasks}) != len(tasks):
            raise ValueError(f"{exp_id}: selected dataset must be nonempty with unique ids")
        tasks.sort(key=lambda t: t["id"])
        sources.append({"experiment": exp_id, "path": str(data_path), "sha256": data_hash,
                        "selected_tasks": len(tasks), "modes": list(modes)})
        runs = exp.get("runs", [])
        if not isinstance(runs, list):
            raise ValueError(f"{exp_id}: runs must be a list")
        run_index: dict[tuple, list] = {}
        seen_runs = set()
        seen_prediction_files = set()
        for run in runs:
            if run.get("role", "main") not in {"main", "diagnostic"}:
                raise ValueError(f"{exp_id}: run role must be main or diagnostic")
            key = (run.get("profile"), run.get("architecture"))
            if key[0] not in profiles or key[1] not in architectures:
                raise ValueError(f"{exp_id}: unknown profile or architecture in run")
            repeat = str(run.get("repeat", "1"))
            if (*key, repeat) in seen_runs:
                raise ValueError(f"{exp_id}: duplicate run/repeat {key}/{repeat}")
            seen_runs.add((*key, repeat))
            prediction_identity = (*key, str(_path(path.parent, run.get("predictions"))))
            if prediction_identity in seen_prediction_files:
                raise ValueError(f"{exp_id}: same predictions file cannot represent multiple repeats")
            seen_prediction_files.add(prediction_identity)
            run_index.setdefault(key, []).append(run)
        for profile_id, profile in sorted(profiles.items()):
            for arch_id, arch in sorted(architectures.items()):
                planned_runs = run_index.get((profile_id, arch_id), [None])
                if all(r is not None and r.get("role") == "diagnostic" for r in planned_runs):
                    planned_runs = [None, *planned_runs]
                for run in sorted(planned_runs, key=lambda r: str(r.get("repeat", "1")) if r else "1"):
                    records, problems, provenance = {}, [], {}
                    repeat = str(run.get("repeat", "1")) if run else "1"
                    if run is not None:
                        pred_path = _path(path.parent, run.get("predictions"))
                        protected.append(pred_path)
                        problems = _provenance_reasons(run, exp, profile, arch, data_hash)
                        provenance = {"predictions": str(pred_path), "declared": run.get("provenance", {}),
                                      "problems": problems}
                        if pred_path.exists():
                            pred_hash = file_hash(pred_path)
                            provenance["predictions_sha256"] = pred_hash
                            if run.get("sha256") and run["sha256"] != pred_hash:
                                raise ValueError(f"{pred_path}: predictions sha256 mismatch")
                            sources.append({"experiment": exp_id, "path": str(pred_path), "sha256": pred_hash})
                            for record in read_jsonl(pred_path):
                                tid = record.get("task_id")
                                if tid is None and family == "xiangqi_rule_variants":
                                    tid = record.get("id")
                                if not isinstance(tid, str):
                                    raise ValueError(f"{pred_path}: expected per-item task_id, not a summary")
                                mode = _record_mode(record, family, dim)
                                if run.get("role") == "diagnostic" or tid not in {t['id'] for t in tasks} or mode not in modes:
                                    observed_mode = mode or record.get("input_mode") or record.get("memory_mode") or record.get("history_mode") or record.get("observation_mode") or "unknown"
                                    diagnostic = score_record(family, dim, record, source_tasks.get(tid), mode=observed_mode)
                                    exclusions.append({"experiment": exp_id, "profile": profile_id,
                                                       "architecture": arch_id, "task_id": tid, "repeat_id": repeat,
                                                       "mode": observed_mode, "reason": "outside_frozen_task_or_mode", "source": str(pred_path),
                                                       "provenance_verified": not problems, "diagnostic_result": diagnostic})
                                    continue
                                if (tid, mode) in records:
                                    raise ValueError(f"{pred_path}: duplicate task/mode {tid}/{mode}")
                                records[tid, mode] = record
                        else:
                            warnings.append(f"Missing predictions file: {pred_path}")
                    if run is not None and run.get("role") == "diagnostic":
                        continue
                    for task in tasks:
                        difficulty = _difficulty(task, family)
                        source = str(task.get("source_id") or task.get("source_task_id") or task.get("scenario_id") or task["id"])
                        for mode in modes:
                            record = records.get((task["id"], mode))
                            value = {"y": None, "p": None, "status": "missing", "reasons": ["missing_result"],
                                     "evidence": {}, "diagnostics": {}, "metrics": {}}
                            if record is not None:
                                value = score_record(family, dim, record, task, mode=mode)
                                if problems:
                                    value["diagnostics"] = dict(value.get("diagnostics", {}), unverified_y=value["y"], unverified_p=value["p"])
                                    value.update(y=None, p=None, status="unverified", reasons=[*value.get("reasons", []), *problems])
                            rows.append({"profile": profile_id, "architecture": arch_id, "experiment": exp_id,
                                         "task": task_family, "family": family, "dimension": dim, "mode": mode,
                                         "group": _group(task, family, dim), "difficulty": difficulty,
                                         "item_id": task["id"], "source_id": f"{task_family}:{source}",
                                         "source_stratum": task_family, "repeat_id": repeat,
                                         "provenance": dict(provenance, dataset=str(data_path), dataset_sha256=data_hash), **value})
    return {"manifest": manifest, "manifest_sha256": file_hash(path), "rows": rows,
            "sources": sources, "exclusions": exclusions, "warnings": sorted(set(warnings)),
            "protected_paths": protected}

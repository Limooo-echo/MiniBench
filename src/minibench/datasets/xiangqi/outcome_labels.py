"""Seven Xiangqi outcome labels with explicit applicability and missing values.

The labels are presentation data derived from saved execution evidence. They do
not replace true-board judging, rewrite historical outcomes or change an oracle.
"""
from __future__ import annotations

import math
from typing import Any, Iterable
from pathlib import Path

from minibench.core.checkpoint import atomic_write_json, atomic_write_jsonl

from minibench.core.token_usage import assess_token_usage


NA = "NA"
OUTCOME_LABELS_VERSION = "xiangqi-seven-outcome-labels-v1"
LABEL_KEYS = ("mate", "legal", "recognition_exact", "strict_success", "tokens", "money", "time")
QUALITY_KEYS = LABEL_KEYS[:4]
RESOURCE_KEYS = LABEL_KEYS[4:]
_FAILURE_MOVES = {"TIMEOUT", "PARSE_FAIL", "EMPTY"}


def _get(record: Any, key: str, default=None):
    return record.get(key, default) if isinstance(record, dict) else getattr(record, key, default)


def _boolean(value):
    return int(value) if isinstance(value, bool) else NA


def _agent_steps(record):
    steps = _get(record, "steps")
    if not isinstance(steps, (list, tuple)):
        return None
    if any(not isinstance(step, dict) for step in steps):
        return None
    return [step for step in steps if step.get("actor") == "agent"]


def _mate(record):
    for key in ("success", "goal_achieved"):
        value = _get(record, key)
        if isinstance(value, bool):
            return int(value)
    return NA


def _legal(record, *, multimove):
    if _get(record, "status") == "invalid":
        return 0
    steps = _agent_steps(record)
    if multimove:
        # Rounded mean legality is insufficient for a whole-game binary label.
        if steps:
            flags = [step.get("is_legal") for step in steps]
            markers = [step.get("uci") for step in steps]
            if any(flag is False for flag in flags) or any(
                    isinstance(move, str) and move in _FAILURE_MOVES for move in markers):
                return 0
            return 1 if all(flag is True for flag in flags) else NA
        return _boolean(_get(record, "all_agent_moves_legal"))
    value = _get(record, "is_legal")
    if isinstance(value, bool):
        return int(value)
    if steps:
        return _boolean(steps[0].get("is_legal"))
    value = _get(record, "legality_rate")
    if not isinstance(value, bool) and isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    return NA


def _recognition(record):
    value = _get(record, "recognition_exact")
    if isinstance(value, bool):
        return int(value)
    steps = _agent_steps(record)
    if not steps:
        return NA
    flags = [step.get("recognition_exact") for step in steps]
    if any(flag is False for flag in flags):
        return 0
    return 1 if all(flag is True for flag in flags) else NA


def _joint(*values):
    if any(value == 0 for value in values):
        return 0
    return 1 if all(value == 1 for value in values) else NA


def _nonnegative(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def _metrics(record):
    metrics = _get(record, "metrics")
    return metrics if isinstance(metrics, dict) else {}


def _resources(record):
    metrics = _metrics(record)
    usage = metrics.get("token_usage")
    missing = metrics.get("usage_missing_calls", 0)
    valid_missing = type(missing) is int and missing == 0
    calls = metrics.get("llm_calls")
    calls_valid = calls is None or (type(calls) is int and calls > 0)
    assessment = assess_token_usage(usage)
    total = assessment["total_tokens"]
    tokens = (total if valid_missing and calls_valid and metrics.get("usage_complete") is not False
              and metrics.get("usage_available") is not False and type(total) is int and total > 0
              else NA)
    cost = metrics.get("cost")
    cost = cost if isinstance(cost, dict) else {}
    amount, currency = cost.get("amount"), cost.get("currency")
    money = (amount if cost.get("complete") is True and _nonnegative(amount)
             and isinstance(currency, str) and currency.strip() else NA)
    elapsed = metrics.get("task_elapsed_seconds")
    elapsed = elapsed if _nonnegative(elapsed) else NA
    return {"tokens": tokens, "money": money, "time": elapsed}


def outcome_labels(record: Any, *, recognition_applicable: bool, multimove: bool) -> dict[str, Any]:
    """Return exactly seven labels; unsupported or unknown fields are ``NA``."""
    if not isinstance(recognition_applicable, bool) or not isinstance(multimove, bool):
        raise TypeError("recognition_applicable and multimove must be booleans")
    if _get(record, "status", "ok") == "error":
        quality = dict.fromkeys(QUALITY_KEYS, NA)
    else:
        mate = _mate(record)
        legal = _legal(record, multimove=multimove)
        recognition = _recognition(record) if recognition_applicable else NA
        strict = _joint(recognition, legal, mate) if recognition_applicable else NA
        quality = {"mate": mate, "legal": legal,
                   "recognition_exact": recognition, "strict_success": strict}
    return {**quality, **_resources(record)}


def _conditional(labels, *, applicable, recognition):
    eligible = successes = unknown = 0
    if applicable:
        for item in labels:
            gate = _joint(item["recognition_exact"], item["legal"]) if recognition else item["legal"]
            if gate == NA:
                unknown += 1
            elif gate == 1:
                eligible += 1
                if item["mate"] == NA:
                    unknown += 1
                else:
                    successes += item["mate"]
    rate = successes / eligible if applicable and eligible and not unknown else NA
    return rate, successes, eligible, unknown


def summarize_outcome_labels(results: Iterable[Any], *, recognition_applicable: bool,
                             multimove: bool) -> dict[str, Any]:
    """Aggregate quality by mean and resources by sum, retaining valid counts.

    Infrastructure failures do not enter quality denominators, but their saved
    request resource use remains billable. Unknown values prevent a partial
    subtotal from masquerading as a complete aggregate.
    """
    if not isinstance(recognition_applicable, bool) or not isinstance(multimove, bool):
        raise TypeError("recognition_applicable and multimove must be booleans")
    rows = list(results)
    all_labels = [outcome_labels(row, recognition_applicable=recognition_applicable,
                                 multimove=multimove) for row in rows]
    quality_labels = [labels for row, labels in zip(rows, all_labels)
                      if _get(row, "status", "ok") != "error"]
    labels, counts = {}, {}
    currencies = {_metrics(row)["cost"]["currency"]
                  for row, item in zip(rows, all_labels) if item["money"] != NA}
    for key in LABEL_KEYS:
        values = [item[key] for item in (quality_labels if key in QUALITY_KEYS else all_labels)]
        known = [value for value in values if value != NA]
        unknown = len(values) - len(known)
        applicable = recognition_applicable if key in {"recognition_exact", "strict_success"} else True
        subtotal = (sum(known) if key in QUALITY_KEYS or key == "tokens" else math.fsum(known)) if known else NA
        mixed_money = key == "money" and len(currencies) > 1
        if mixed_money:
            subtotal = NA
        if not applicable or not known or unknown or mixed_money:
            aggregate = NA
        else:
            aggregate = subtotal / len(known) if key in QUALITY_KEYS else subtotal
        labels[key] = aggregate
        counts[key] = {"applicable": applicable, "known_count": len(known),
                       "unknown_count": unknown, "known_subtotal": subtotal}
    conditional = _conditional(quality_labels, applicable=recognition_applicable, recognition=True)
    legal_only = _conditional(quality_labels,
                              applicable=multimove, recognition=False)
    return {"outcome_labels_summary": {
        "version": OUTCOME_LABELS_VERSION,
        "labels": labels,
        "counts": counts,
        "record_count": len(rows),
        "scorable_count": len(quality_labels),
        "infrastructure_error_count": len(rows) - len(quality_labels),
        "recognition_applicable": recognition_applicable,
        "multimove": multimove,
        "conditional_success_rate": conditional[0],
        "conditional_success_count": conditional[1],
        "conditional_eligible_count": conditional[2],
        "conditional_unknown_count": conditional[3],
        "conditional_basis": "exact_recognition_and_legal" if recognition_applicable else "not_applicable",
        "mate_given_all_legal_rate": legal_only[0],
        "all_legal_success_count": legal_only[1],
        "all_legal_eligible_count": legal_only[2],
        "all_legal_unknown_count": legal_only[3],
        "money_currency": next(iter(currencies)) if len(currencies) == 1 else NA,
        "time_basis": "sum of observed per-task wall seconds; not concurrent suite wall time",
    }}


def _record_group(record):
    for key in ("history_mode", "input_mode", "ruleset"):
        value = _get(record, key)
        if isinstance(value, str) and value:
            return value
    return "native"


def _record_id(record):
    for key in ("task_id", "id", "source_task_id"):
        value = _get(record, key)
        if isinstance(value, str) and value:
            return value
    return NA


def _compact_summary(summary):
    return {
        "labels": summary["labels"],
        "conditional_accuracy": {
            "rate": summary["conditional_success_rate"],
            "success_count": summary["conditional_success_count"],
            "eligible_count": summary["conditional_eligible_count"],
            "unknown_count": summary["conditional_unknown_count"],
            "basis": summary["conditional_basis"],
        },
    }


# Public output is task-specific. The seven-field evidence view above remains
# available to historical readers and independent diagnostic calculations.
TASK_LABELS_VERSION = "xiangqi-task-outcome-labels-v2"
TREE_LABEL = "solution_tree_prefix_rate"
CONDITIONAL_LABEL = "mate_given_recognition_and_legal"
PUBLIC_LABEL_KEYS = ("mate", "legal", "recognition_exact", CONDITIONAL_LABEL,
                     TREE_LABEL, "tokens", "money", "time")


def task_label_keys(task: str, group: str | None = None) -> tuple[str, ...]:
    task = task.lower()
    resources = ("tokens", "money", "time")
    if task == "d3":
        return ("mate", "legal", *resources)
    if task == "h2":
        quality = ("mate", "legal")
        if group == "move-history-only":
            quality += ("recognition_exact", CONDITIONAL_LABEL)
        return (*quality, TREE_LABEL, *resources)
    if task == "c2":
        return ("mate", "legal", TREE_LABEL, *resources)
    if task == "m2":
        return ("mate", "legal", "recognition_exact", CONDITIONAL_LABEL, *resources)
    raise ValueError(f"unsupported Xiangqi task: {task}")


def task_outcome_labels(record: Any, *, task: str) -> dict[str, Any]:
    """Public per-question vector. A failed conditional gate is NA, not failure.

    The conditional numerator/denominator are also retained by summaries, so
    dropping a misrecognized/illegal sample cannot silently improve raw mate.
    """
    task = task.lower()
    group = _record_group(record)
    base = outcome_labels(record, recognition_applicable=task != "c2",
                          multimove=task in {"h2", "c2"})
    gate = _joint(base["recognition_exact"], base["legal"])
    base[CONDITIONAL_LABEL] = base["mate"] if gate == 1 else NA
    tree = _get(record, TREE_LABEL)
    base[TREE_LABEL] = (tree if _get(record, "status", "ok") != "error"
                       and _nonnegative(tree) and tree <= 1 else NA)
    return {key: base[key] for key in task_label_keys(task, group)}


def summarize_task_outcomes(results: Iterable[Any], *, task: str,
                            group: str | None = None) -> dict[str, Any]:
    rows = list(results)
    task = task.lower()
    if task == "h2" and group is None:
        modes = {_record_group(row) for row in rows}
        group = next(iter(modes)) if len(modes) == 1 else None
    primary = summarize_outcome_labels(rows, recognition_applicable=task != "c2",
                                      multimove=task in {"h2", "c2"})["outcome_labels_summary"]
    from .tree_prefix_metrics import summarize_tree_prefix
    prefix = summarize_tree_prefix(rows) if task in {"h2", "c2"} else {}
    return project_task_summary(primary, task=task, group=group, prefix=prefix)


def project_task_summary(primary: dict[str, Any], *, task: str, group: str | None = None,
                         prefix: dict[str, Any] | None = None) -> dict[str, Any]:
    """Project saved summary evidence without fabricating per-question records."""
    task = task.lower()
    prefix = prefix or {}
    values, counts = dict(primary["labels"]), dict(primary["counts"])
    conditional_applicable = task == "m2" or task == "h2" and group == "move-history-only"
    values[CONDITIONAL_LABEL] = primary["conditional_success_rate"] if conditional_applicable else NA
    counts[CONDITIONAL_LABEL] = {
        "applicable": conditional_applicable,
        "known_count": primary["conditional_eligible_count"] if conditional_applicable else 0,
        "unknown_count": primary["conditional_unknown_count"] if conditional_applicable else 0,
        "known_subtotal": primary["conditional_success_count"] if conditional_applicable else NA,
    }
    values[TREE_LABEL] = prefix.get(TREE_LABEL) if prefix.get(TREE_LABEL) is not None else NA
    counts[TREE_LABEL] = {
        "applicable": task in {"h2", "c2"},
        "known_count": prefix.get("solution_tree_prefix_rate_eligible", 0),
        "unknown_count": prefix.get("solution_tree_prefix_rate_missing", 0),
        "known_subtotal": NA,
        "aggregation": "macro_mean_of_per_episode_ratios",
    }
    keys = task_label_keys(task, group)
    return {
        **primary, "version": TASK_LABELS_VERSION,
        "labels": {key: values[key] for key in keys},
        "counts": {key: counts[key] for key in keys},
        "conditional_success_rate": primary["conditional_success_rate"] if conditional_applicable else NA,
        "conditional_success_count": primary["conditional_success_count"] if conditional_applicable else NA,
        "conditional_eligible_count": primary["conditional_eligible_count"] if conditional_applicable else NA,
        "conditional_unknown_count": primary["conditional_unknown_count"] if conditional_applicable else NA,
        "conditional_basis": "exact_recognition_and_legal" if conditional_applicable else "not_applicable",
    }


def write_outcome_files(run_dir, results, summary, *, task: str,
                        recognition_applicable: bool, multimove: bool,
                        groups=None):
    """Write current task/group vectors; keep historical evidence in raw results."""
    rows = list(results)
    directory = Path(run_dir)
    values = summarize_task_outcomes(rows, task=task)
    group_values = {}
    for name in groups or {}:
        selected = [row for row in rows if name == "all" or _record_group(row) == name
                    or name == "variant" and isinstance(_get(row, "ruleset"), str)
                    and _get(row, "ruleset") != "standard"]
        group_values[name] = summarize_task_outcomes(selected, task=task, group=name)
    atomic_write_jsonl(directory / "labels.jsonl", (
        {"task_id": _record_id(row), "group": _record_group(row),
         "labels": task_outcome_labels(row, task=task)} for row in rows
    ))
    complete = {"task": task, "summary": values, "groups": group_values}
    atomic_write_json(directory / "labels_summary.json", complete)
    compact_groups = {key: _compact_summary(value) for key, value in group_values.items()}
    # Contrasting groups are never pooled into a single headline capability rate.
    compact = ({"task": task, "groups": compact_groups} if task in {"h2", "c2"}
               else {"task": task, **_compact_summary(values), "groups": compact_groups})
    atomic_write_json(directory / "summary.txt", compact)
    return complete

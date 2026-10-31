"""Task/condition/metric comparisons for the resumable Xiangqi suite.

Missing results are displayed as N/A, never silently converted to zero.
Task-specific primary labels use 0-1 quality values and unscaled resource amounts.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from minibench.datasets.xiangqi.legal_episode_metrics import CONDITIONAL_METRIC
from minibench.datasets.xiangqi.outcome_labels import task_outcome_labels, summarize_task_outcomes, task_label_keys, PUBLIC_LABEL_KEYS, CONDITIONAL_LABEL, project_task_summary

# Bundled CJK fonts are chess-glyph subsets, not general plot fonts. Use ASCII
# labels and Matplotlib's bundled DejaVu on headless servers; board fonts stay
# unchanged. The Chinese metric descriptions live in docs/cloud-run.md.
NAMES = {"passthrough": "Passthrough", "direct": "Direct", "cot": "CoT",
         "tot": "ToT", "best-of-n": "Best-of-N", "least-to-most": "Least-to-Most",
         "plan-then-solve": "Plan-then-Solve", "critic-refine": "Critic-Refine",
         "self-consistency": "Self-Consistency"}
TREE_PREFIX_METRIC = "solution_tree_prefix_rate"

LABEL_KEYS = PUBLIC_LABEL_KEYS
QUALITY_KEYS = tuple(key for key in LABEL_KEYS if key not in {"tokens", "money", "time"})
METRICS = {
    "mate": ("Checkmate success (0-1)", True),
    "legal": ("Whole-episode legal success (0-1)", True),
    "recognition_exact": ("Exact current-board reconstruction (0-1)", True),
    "strict_success": ("Exact reconstruction, legal play and checkmate (0-1)", True),
    "tokens": ("Total reported tokens", False),
    "money": ("Estimated cost (configured currency)", False),
    "time": ("Sum of observed episode time (seconds)", False),
    TREE_PREFIX_METRIC: ("Mean in-tree attacker prefix / certified shortest moves", True),
    "conditional_success_rate": ("Checkmate given exact reconstruction and legal play", True),
    CONDITIONAL_LABEL: ("Checkmate given exact reconstruction and legal play", True),
    "mate_given_all_legal_rate": ("Checkmate given a wholly legal episode", True),
    CONDITIONAL_METRIC: ("Shortest-bound checkmate given a wholly legal episode", True),
}

PERCEPTION_METRICS = {
    "mate_rate_given_recognition_and_legal": ("conditional_numerator", "conditional_denominator"),
    "strict_joint_success_rate": ("strict_joint_numerator", "strict_joint_denominator"),
}


def _metric_samples(bucket: dict, metric: str) -> dict:
    """A conditional rate's denominator is not the run completion count."""
    if metric == TREE_PREFIX_METRIC:
        return {"numerator": None,
                "denominator": bucket.get("solution_tree_prefix_rate_eligible"),
                "eligible_count": bucket.get("solution_tree_prefix_rate_eligible"),
                "prefix_missing_count": bucket.get("solution_tree_prefix_rate_missing"),
                "unknown_gate_count": None, "unknown_joint_count": None,
                "unknown_legality_count": None, "unknown_shortest_count": None}
    if metric == CONDITIONAL_METRIC:
        counts = bucket.get("all_legal_episodes") or {}
        return {"numerator": counts.get("conditional_numerator"),
                "denominator": counts.get("conditional_denominator"),
                "eligible_count": counts.get("all_legal_count"),
                "unknown_gate_count": None, "unknown_joint_count": None,
                "unknown_legality_count": counts.get("unknown_legality_count"),
                "unknown_shortest_count": counts.get("unknown_shortest_count")}
    counts = bucket.get("perception_gating") or {}
    keys = PERCEPTION_METRICS.get(metric)
    return {
        "numerator": counts.get(keys[0]) if keys else None,
        "denominator": counts.get(keys[1]) if keys else None,
        "eligible_count": counts.get("recognition_and_legal_count") if keys else None,
        "unknown_gate_count": counts.get("unknown_gate_count") if keys else None,
        "unknown_joint_count": counts.get("unknown_joint_count") if keys else None,
        "unknown_legality_count": None, "unknown_shortest_count": None,
    }


def _bar_label(row: dict) -> str:
    label = NAMES.get(row["agent"], row["agent"]) + f"\n{row['completed']}/{row['planned']} completed"
    if row["metric"] in LABEL_KEYS or row["metric"] in {"conditional_success_rate", "mate_given_all_legal_rate"}:
        n, d = row.get("numerator"), row.get("denominator")
        unknown = row.get("unknown_gate_count")
        label += f"\nknown {d if d is not None else 'NA'}; unknown {unknown if unknown is not None else 'NA'}"
        if n is not None and d is not None and row["metric"] in QUALITY_KEYS:
            label += f"; {n}/{d}"
    if row["metric"] == TREE_PREFIX_METRIC:
        eligible, missing = row.get("eligible_count"), row.get("prefix_missing_count")
        label += (f"\nmacro mean; eligible {eligible if eligible is not None else 'N/A'}; "
                  f"missing {missing if missing is not None else 'N/A'}")
    if row["metric"] == CONDITIONAL_METRIC:
        n, d = row.get("numerator"), row.get("denominator")
        ratio = f"{n}/{d}" if n is not None and d is not None else "N/A"
        ul, us = row.get("unknown_legality_count"), row.get("unknown_shortest_count")
        incomplete = ul is None or us is None or ul > 0 or us > 0
        label += f"\n{'known subtotal' if incomplete else 'score'} {ratio}; all-legal {d if d is not None else 'N/A'}"
        if incomplete:
            label += f"\nunknown legality {ul if ul is not None else 'N/A'}; shortest {us if us is not None else 'N/A'}"
    if row["metric"] in PERCEPTION_METRICS:
        numerator, denominator = row.get("numerator"), row.get("denominator")
        ratio = f"{numerator}/{denominator}" if numerator is not None and denominator is not None else "N/A"
        eligible = row.get("eligible_count")
        unknown_gate, unknown_joint = row.get("unknown_gate_count"), row.get("unknown_joint_count")
        gate_incomplete = unknown_gate is None or unknown_gate > 0
        joint_incomplete = unknown_joint is None or unknown_joint > 0
        incomplete = gate_incomplete if row["metric"] == "mate_rate_given_recognition_and_legal" else joint_incomplete
        prefix = "known subtotal" if incomplete else "score"
        eligible_label = "known eligible" if gate_incomplete else "eligible"
        label += f"\n{prefix} {ratio}; {eligible_label} {eligible if eligible is not None else 'N/A'}"
        if gate_incomplete or joint_incomplete:
            label += (f"\nunknown gate {unknown_gate if unknown_gate is not None else 'N/A'}; "
                      f"joint {unknown_joint if unknown_joint is not None else 'N/A'}")
    return label


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _task_key(family):
    return {"xiangqi-mate-in-one": "d3", "xiangqi-history": "h2",
            "xiangqi-rule-variants": "c2", "xiangqi-multimodal": "m2"}[family]


def _primary_summary(bucket, family, records=(), mode=None):
    if records:
        return summarize_task_outcomes(list(records), task=_task_key(family), group=mode)
    stored = bucket.get("outcome_labels_summary")
    if isinstance(stored, dict) and isinstance(stored.get("labels"), dict):
        return project_task_summary(stored, task=_task_key(family), group=mode, prefix=bucket)
    return summarize_task_outcomes([], task=_task_key(family), group=mode)


def _physical_labels(labels, actual):
    labels = dict(labels)
    if actual is None:
        return labels
    tokens = (actual.get("token_usage") or {}).get("total_tokens")
    unknown = actual.get("attempts_with_unknown_usage", 0) or actual.get("usage_missing_calls", 0)
    labels["tokens"] = tokens if type(tokens) is int and tokens > 0 and not unknown else "NA"
    cost = actual.get("cost") or {}
    amount = _number(cost.get("amount"))
    labels["money"] = (cost["amount"] if cost.get("complete") is True and amount is not None and amount >= 0
                       and isinstance(cost.get("currency"), str) and cost["currency"].strip() else "NA")
    # Time is the observed episode duration sum, not model-only physical latency.
    return labels


def _physical_group(physical, task, agent, mode, family):
    groups = physical.get(task, {}).get(agent, {})
    if not groups:
        return None
    if mode in groups:
        return groups[mode]
    if family == "xiangqi-rule-variants" and mode == "all" and {"standard", "variant"} <= groups.keys():
        parts = [_physical_labels({}, groups[key]) for key in ("standard", "variant")]
        tokens = sum(part["tokens"] for part in parts) if all(part["tokens"] != "NA" for part in parts) else None
        currencies = {(groups[key].get("cost") or {}).get("currency") for key in ("standard", "variant")}
        known_money = all(part["money"] != "NA" for part in parts) and len(currencies) == 1
        return {"token_usage": {"total_tokens": tokens}, "cost": {
            "amount": math.fsum(part["money"] for part in parts) if known_money else None,
            "currency": next(iter(currencies)) if len(currencies) == 1 else None,
            "complete": known_money}}
    # The physical ledger exists but cannot attribute billing at this finer grain.
    # Falling back to replayed logical usage here would fabricate fresh charges.
    return {"token_usage": {}, "cost": {"complete": False}, "granularity_unavailable": True}


def _records(directory):
    path = directory / "predictions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _matches(record, family, mode, native):
    if family == "xiangqi-rule-variants":
        return (mode == "all" or record.get("ruleset") == mode
                or mode == "variant" and record.get("ruleset") != "standard")
    return (record.get("history_mode") or record.get("input_mode") or native) == mode


def _entries(suite):
    plan = json.loads((suite / "plan.json").read_text(encoding="utf-8"))
    entries = plan.get("experiments", plan.get("tasks", {}))
    native = "native" if "experiments" in plan else "single"
    for task, entry in entries.items():
        family = entry.get("family") or {"d3": "xiangqi-mate-in-one", "h2": "xiangqi-history",
            "c2": "xiangqi-rule-variants", "m2": "xiangqi-multimodal"}.get(task, "")
        if not family.startswith("xiangqi-"):
            continue
        for agent in plan["agents"]:
            directory = suite / "results" / task / agent
            path = directory / "results.json"
            summary = json.loads(path.read_text()) if path.exists() else {}
            yield plan, task, entry, family, agent, native, summary, _records(directory)


def _buckets(summary, family, native):
    if family == "xiangqi-history":
        return summary.get("by_mode", {})
    if family == "xiangqi-multimodal":
        return summary.get("by_input_mode", {})
    if family == "xiangqi-rule-variants":
        return {"all": summary, **summary.get("by_ruleset", {}), **summary.get("by_condition", {})}
    return {native: summary} if summary else {}


def comparison_rows(suite: Path) -> list[dict]:
    rows = []
    physical_path = suite / "summary.json"
    physical = json.loads(physical_path.read_text()).get("by_experiment_agent_condition_resources", {}) if physical_path.exists() else {}
    for plan, task, entry, family, agent, native, summary, records in _entries(suite):
        counts = entry.get("condition_counts") or {mode: entry["record_count"] for mode in entry["modes"]}
        for mode, planned in counts.items():
            bucket = _buckets(summary, family, native).get(mode, {})
            selected = [row for row in records if _matches(row, family, mode, native)]
            primary = _primary_summary(bucket, family, selected, mode)
            labels = dict(primary.get("labels") or {})
            actual = _physical_group(physical, task, agent, mode, family)
            labels = _physical_labels(labels, actual)
            metrics = list(task_label_keys(_task_key(family), mode))
            if family in {"xiangqi-history", "xiangqi-rule-variants"}:
                metrics += [CONDITIONAL_METRIC]
            for metric in metrics:
                samples = {"numerator": None, "denominator": None, "eligible_count": None,
                           "unknown_gate_count": None, "unknown_joint_count": None,
                           "unknown_legality_count": None, "unknown_shortest_count": None,
                           "prefix_missing_count": None}
                if metric == TREE_PREFIX_METRIC:
                    samples.update(_metric_samples(bucket, metric))
                    value = bucket.get(metric)
                elif metric in LABEL_KEYS:
                    value = labels.get(metric)
                    count = (primary.get("counts") or {}).get(metric, {})
                    samples.update(denominator=count.get("known_count"), eligible_count=count.get("known_count"),
                                   unknown_gate_count=count.get("unknown_count"))
                    if metric in QUALITY_KEYS:
                        samples["numerator"] = count.get("known_subtotal")
                else:
                    is_legal = metric in {"mate_given_all_legal_rate", CONDITIONAL_METRIC}
                    prefix = "all_legal" if is_legal else "conditional"
                    samples.update(numerator=primary.get(prefix + "_success_count"),
                                   denominator=primary.get(prefix + "_eligible_count"),
                                   eligible_count=primary.get(prefix + "_eligible_count"),
                                   unknown_gate_count=primary.get(prefix + "_unknown_count"))
                    value = primary.get("mate_given_all_legal_rate" if is_legal else metric)
                    if is_legal:
                        samples.update(unknown_legality_count=primary.get("all_legal_unknown_count"), unknown_shortest_count=0)
                rows.append({"task": task, "mode": mode, "metric": metric, "agent": agent,
                             "value": _number(value), "completed": len(selected), "planned": planned, **samples})
    return rows


def outcome_label_rows(suite: Path) -> list[dict]:
    """One immutable exported label vector per saved prediction, never per bucket."""
    rows = []
    for plan, task, entry, family, agent, native, summary, records in _entries(suite):
        for record in records:
            group = ("standard" if record.get("ruleset") == "standard" else "variant") if family == "xiangqi-rule-variants" else record.get("history_mode") or record.get("input_mode") or native
            labels = task_outcome_labels(record, task=_task_key(family))
            rows.append({"task": task, "agent": agent, "group": group,
                         "task_id": record.get("task_id") or record.get("id"),
                         "ruleset": record.get("ruleset"), "labels": labels})
    return rows


def outcome_summary_rows(suite: Path) -> list[dict]:
    """Primary groups expose task-specific labels plus separately named conditional diagnostics."""
    physical_path = suite / "summary.json"
    physical = json.loads(physical_path.read_text()).get("by_experiment_agent_condition_resources", {}) if physical_path.exists() else {}
    rows = []
    for plan, task, entry, family, agent, native, summary, records in _entries(suite):
        buckets = _buckets(summary, family, native)
        if family == "xiangqi-history":
            modes = ("full-state", "move-history-only")
        elif family == "xiangqi-rule-variants":
            modes = ("standard", "variant")
        elif family == "xiangqi-multimodal":
            modes = tuple(buckets) or tuple(entry.get("condition_counts") or entry.get("modes") or ())
        else:
            modes = (native,)
        for mode in modes:
            selected = [row for row in records if _matches(row, family, mode, native)]
            bucket = buckets.get(mode, {})
            counts = entry.get("condition_counts") or {}
            if mode not in buckets and not selected and mode not in counts:
                continue
            primary = _primary_summary(bucket, family, selected, mode)
            labels = dict(primary["labels"])
            actual = _physical_group(physical, task, agent, mode, family)
            labels = _physical_labels(labels, actual)
            rows.append({"task": task, "agent": agent, "group": mode, "labels": labels,
                         "counts": primary.get("counts", {}),
                         "counts_basis": "saved_episode_records; physical token/cost totals use physical_resource_counts",
                         "physical_resource_counts": ({key: actual[key] for key in (
                             "physical_attempts", "attempts_with_usage", "attempts_with_unknown_usage",
                             "usage_missing_calls", "known_token_usage", "token_usage") if key in actual}
                             if actual is not None else {}),
                         "money_currency": ((actual.get("cost") or {}).get("currency") if actual is not None else primary.get("money_currency", "NA")) or "NA",
                         **{key: primary.get(key, "NA") for key in (
                             "conditional_success_rate", "conditional_success_count", "conditional_eligible_count", "conditional_unknown_count",
                             "conditional_basis", "mate_given_all_legal_rate", "all_legal_success_count", "all_legal_eligible_count", "all_legal_unknown_count")},
                         "resource_basis": "physical_API_tokens_and_cost; episode_time" if actual is not None else "saved_episode_metrics"})
    return rows


def condition_metric_rows(suite: Path) -> list[dict]:
    """One row per H2/C2 primary group, keeping physical billing separate."""
    plan = json.loads((suite / "plan.json").read_text())
    physical_path = suite / "summary.json"
    physical = json.loads(physical_path.read_text()).get("by_experiment_agent_condition_resources", {}) if physical_path.exists() else {}
    rows = []
    for task, entry in plan.get("experiments", {}).items():
        family = entry.get("family")
        if family not in {"xiangqi-history", "xiangqi-rule-variants"}:
            continue
        modes = ("full-state", "move-history-only") if family == "xiangqi-history" else ("standard", "variant")
        for agent in plan["agents"]:
            path = suite / "results" / task / agent / "results.json"
            summary = json.loads(path.read_text()) if path.exists() else {}
            buckets = summary.get("by_mode" if family == "xiangqi-history" else "by_condition", {})
            counts = entry.get("condition_counts", {})
            for mode in modes:
                planned = counts.get(mode)
                if mode == "variant" and planned is None:
                    planned = sum(value for key, value in counts.items() if key not in {"all", "standard", "variant"})
                if not planned and mode not in buckets:
                    continue
                bucket = buckets.get(mode, {})
                logical = (bucket.get("metrics") or {}).get("total", {})
                actual = physical.get(task, {}).get(agent, {}).get(mode)
                usage = actual if actual is not None else logical
                cost = usage.get("cost", {})
                model_time = (actual.get("known_model_seconds")
                              if actual is not None and not actual.get("attempts_with_unknown_time")
                              else None if actual is not None else logical.get("model_elapsed_seconds"))
                gate = bucket.get("all_legal_episodes", {})
                rows.append({
                    "task": task, "agent": agent, "condition": mode,
                    "completed": bucket.get("total"), "planned": planned,
                    "evaluated": bucket.get("evaluated"), "missing": bucket.get("missing"),
                    "mate_within_shortest_rate": bucket.get("mate_within_shortest_rate"),
                    "legality_rate": bucket.get("legality_rate"),
                    TREE_PREFIX_METRIC: bucket.get(TREE_PREFIX_METRIC),
                    "solution_tree_prefix_rate_eligible": bucket.get("solution_tree_prefix_rate_eligible"),
                    "solution_tree_prefix_rate_missing": bucket.get("solution_tree_prefix_rate_missing"),
                    "solution_tree_prefix_known_macro_mean": (bucket.get("solution_tree_prefix_metrics") or {}).get("known_macro_mean"),
                    CONDITIONAL_METRIC: bucket.get(CONDITIONAL_METRIC),
                    "conditional_numerator": gate.get("conditional_numerator"),
                    "conditional_denominator": gate.get("conditional_denominator"),
                    "total_tokens": (usage.get("token_usage") or {}).get("total_tokens"),
                    "known_total_tokens": (usage.get("known_token_usage") or {}).get("total_tokens"),
                    "model_elapsed_seconds": model_time,
                    "episode_elapsed_seconds": logical.get("task_elapsed_seconds"),
                    "estimated_cost": cost.get("amount"), "known_estimated_cost": cost.get("known_amount"),
                    "currency": cost.get("currency"), "cost_complete": cost.get("complete"),
                    "resource_basis": "physical_API_attempts" if actual is not None else "saved_episode_metrics",
                    "time_basis": "model request sum and observed episode duration sum; not concurrent wall time",
                })
    return rows


def plot_suite(suite: str | Path) -> Path:
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "axes.unicode_minus": False}):
        return _plot_suite(suite)


def _plot_suite(suite: str | Path) -> Path:
    suite = Path(suite)
    output = suite / "charts"
    output.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(suite)
    # The task-specific tables use NA; diagnostic comparison CSV keeps blank unavailable values.
    from minibench.core.checkpoint import atomic_write_json, atomic_write_text
    import io
    atomic_write_json(output / "comparison.json", rows)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["task", "mode", "metric", "agent", "value", "completed", "planned",
                                               "numerator", "denominator", "eligible_count",
                                               "unknown_gate_count", "unknown_joint_count",
                                               "unknown_legality_count", "unknown_shortest_count",
                                               "prefix_missing_count"])
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(output / "comparison.csv", buffer.getvalue())
    conditions = condition_metric_rows(suite)
    atomic_write_json(output / "condition-metrics.json", conditions)
    condition_buffer = io.StringIO()
    if conditions:
        condition_writer = csv.DictWriter(condition_buffer, fieldnames=list(conditions[0]))
        condition_writer.writeheader()
        condition_writer.writerows(conditions)
    atomic_write_text(output / "condition-metrics.csv", condition_buffer.getvalue())
    label_rows = outcome_label_rows(suite)
    atomic_write_json(output / "outcome-labels.json", label_rows)
    label_buffer = io.StringIO()
    label_writer = csv.DictWriter(label_buffer, fieldnames=["task", "agent", "group", "task_id", "ruleset", *LABEL_KEYS])
    label_writer.writeheader()
    for row in label_rows:
        label_writer.writerow({**{k: v for k, v in row.items() if k != "labels"}, **row["labels"]})
    atomic_write_text(output / "outcome-labels.csv", label_buffer.getvalue())
    primary_rows = outcome_summary_rows(suite)
    atomic_write_json(output / "outcome-summary.json", primary_rows)
    primary_buffer = io.StringIO()
    flat_rows = [{**{k: v for k, v in row.items() if k not in {"labels", "counts"}}, **row["labels"],
                  "counts": json.dumps(row["counts"], sort_keys=True)} for row in primary_rows]
    if flat_rows:
        primary_writer = csv.DictWriter(primary_buffer, fieldnames=list(dict.fromkeys(key for row in flat_rows for key in row)))
        primary_writer.writeheader()
        primary_writer.writerows(flat_rows)
    atomic_write_text(output / "outcome-summary.csv", primary_buffer.getvalue())
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["mode"], row["metric"]), []).append(row)
    for (task, mode, metric), values in groups.items():
        title, percent = METRICS[metric]
        fig, ax = plt.subplots(figsize=(15, 6))
        positions = list(range(len(values)))
        colors = plt.get_cmap("tab10").colors
        for index, row in enumerate(values):
            value = row["value"]
            if value is None:
                ax.text(index, 0, "NA", ha="center", va="bottom", color="#777")
                continue
            displayed = float(value)
            ax.bar(index, displayed, color=colors[index % len(colors)], width=.65)
            ax.annotate(f"{displayed:.3f}" if percent else f"{displayed:.1f}",
                        (index, displayed), ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points", fontsize=9)
        labels = [_bar_label(row) for row in values]
        ax.set_xticks(positions, labels, rotation=20, ha="right")
        ax.set_xlim(-.6, len(values) - .4)
        ax.set_ylabel(title)
        ax.set_title(f"{task.upper()} / {mode} - {title}")
        if percent:
            ax.set_ylim(0, 1.12)
            ax.set_yticks([0, .2, .4, .6, .8, 1.0])
        else:
            ax.set_ylim(bottom=0)
            ax.margins(y=.2)
        ax.grid(axis="y", linestyle="--", alpha=.3)
        ax.set_axisbelow(True)
        fig.text(.01, .01, "Below each bar: completed/planned; gated metrics also show numerator/denominator and eligible tasks.",
                 fontsize=9, color="#555")
        fig.tight_layout(rect=(0, .04, 1, 1))
        stem = output / f"{task}-{mode}-{metric}"
        fig.savefig(stem.with_suffix(".png"), dpi=160)
        fig.savefig(stem.with_suffix(".svg"))
        plt.close(fig)
    return output

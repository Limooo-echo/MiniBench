"""Deterministic, offline aggregation of a *predeclared* evaluation suite.

The caller supplies every expected observation, including missing slots.  No
group, repeat, or task weight is inferred from the subset of successful runs.
This module deliberately knows nothing about task runners or language models.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import math
from numbers import Real
from typing import Any

import numpy as np


DIMENSIONS = ("D", "R", "H", "V")
TASKS = ("zebra", "one_stroke", "xiangqi", "mahjong")


def _unit(value: Any, name: str, *, binary: bool = False) -> float | None:
    if value is None:
        return None
    if not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number or null")
    number = float(value)
    if not 0 <= number <= 1 or (binary and number not in (0.0, 1.0)):
        raise ValueError(f"{name} must be {'0 or 1' if binary else 'in [0, 1]'}")
    return number


def _score(y: float | None, p: float | None, cap: float) -> float | None:
    # Missing partial evidence never erases a known complete success or a
    # computable strict score.  A failed answer with missing evidence is not 0.
    if y is None:
        return None
    if y == 1 or cap == 0:
        return y
    return None if p is None else cap * p


def _validate_weights(weights: dict) -> dict[str, dict[str, float]]:
    if set(weights) != set(TASKS):
        raise ValueError(f"weights must contain exactly {', '.join(TASKS)}")
    result = {}
    for task in TASKS:
        if set(weights[task]) != set(DIMENSIONS):
            raise ValueError(f"{task} weights must contain D, R, H, V")
        result[task] = {}
        for dimension in DIMENSIONS:
            value = _unit(weights[task][dimension], f"{task}.{dimension}")
            if value is None:
                raise ValueError("weights cannot be null")
            result[task][dimension] = value
    for dimension in DIMENSIONS:
        if not math.isclose(sum(result[t][dimension] for t in TASKS), 1.0,
                            abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"{dimension} task weights must sum to 1")
    return result


def _prepare(rows: list[dict], weights: dict, cap: float) -> list[dict]:
    prepared = []
    identities = set()
    source_strata: dict[tuple[str, str], str] = {}
    fields = ("profile", "architecture", "task", "dimension", "mode", "group",
              "difficulty", "item_id", "repeat_id")
    for original in rows:
        row = dict(original)
        for field in fields:
            if field not in row or row[field] is None or str(row[field]) == "":
                raise ValueError(f"each expected row needs {field}")
            row[field] = str(row[field])
        if row["task"] not in TASKS or row["dimension"] not in DIMENSIONS:
            raise ValueError("unknown task or dimension")
        identity = tuple(row[field] for field in fields)
        if identity in identities:
            raise ValueError(f"duplicate expected observation: {identity}")
        identities.add(identity)
        row["y"] = _unit(row.get("y"), "y", binary=True)
        row["p"] = _unit(row.get("p"), "p")
        source = row.get("source_id")
        row["source_id"] = str(source) if source is not None and str(source) else None
        stratum = row.get("source_stratum")
        row["source_stratum"] = str(stratum) if stratum is not None else ":".join(
            (row["task"], row["group"], row["difficulty"]))
        if row["source_id"] is not None:
            key = row["profile"], row["source_id"]
            previous = source_strata.setdefault(key, row["source_stratum"])
            if previous != row["source_stratum"]:
                raise ValueError(
                    f"source {key} appears in different source_stratum values; "
                    "all variants of the same original must share a stratum")
        row["score"] = _score(row["y"], row["p"], cap)
        row["strict_score"] = row["y"]
        prepared.append(row)

    # Tree leaves are items.  Explicit expected repeats are the final level.
    trees: dict[tuple, dict] = defaultdict(dict)
    for row in prepared:
        key = row["profile"], row["architecture"], row["task"], row["dimension"]
        tree = trees[key]
        for level in ("mode", "group", "difficulty", "item_id"):
            tree = tree.setdefault(row[level], {})
        tree[row["repeat_id"]] = row

    def assign(node: dict, weight: float, depth: int, axis_weight: float) -> None:
        child_weight = weight / len(node)
        if depth == 4:
            sources = {r["source_id"] for r in node.values() if r["source_id"] is not None}
            if len(sources) > 1:
                raise ValueError("repeats of one item must share source_id")
            for row in node.values():
                row["within_task_weight"] = child_weight
                row["axis_weight"] = child_weight * axis_weight
                row["contribution"] = (None if row["score"] is None else
                                       100 * row["axis_weight"] * row["score"])
                row["strict_contribution"] = (None if row["y"] is None else
                                              100 * row["axis_weight"] * row["y"])
        else:
            for child in node.values():
                assign(child, child_weight, depth + 1, axis_weight)

    for (_, _, task, dimension), tree in trees.items():
        assign(tree, 1.0, 0, weights[task][dimension])
    return sorted(prepared, key=lambda row: tuple(row[field] for field in fields))


def _cell(rows: list[dict], *, cap: float) -> dict:
    coverage = strict_coverage = total = strict_total = 0.0
    available = strict_available = 0
    for row in rows:
        value = _score(row["y"], row["p"], cap)
        weight = row["within_task_weight"]
        if value is not None:
            coverage += weight
            total += weight * value
            available += 1
        if row["y"] is not None:
            strict_coverage += weight
            strict_total += weight * row["y"]
            strict_available += 1
    complete = bool(rows) and available == len(rows)
    strict_complete = bool(rows) and strict_available == len(rows)
    return {
        "score": min(1.0, max(0.0, total)) if complete else None,
        "strict_score": min(1.0, max(0.0, strict_total)) if strict_complete else None,
        "coverage": 1.0 if complete else min(1.0, coverage),
        "strict_coverage": 1.0 if strict_complete else min(1.0, strict_coverage),
        "expected_slots": len(rows),
        "available_slots": available,
        "strict_available_slots": strict_available,
        "expected_items": len({(r["mode"], r["group"], r["difficulty"], r["item_id"])
                               for r in rows}),
    }


def _axis(cells: dict, weights: dict, dimension: str) -> dict:
    selected = [(weights[t][dimension], cells[t, dimension]) for t in TASKS
                if weights[t][dimension] > 0]
    return {
        "score": (min(100.0, 100 * math.fsum(w * c["score"] for w, c in selected))
                  if all(c["score"] is not None for _, c in selected) else None),
        "strict_score": (min(100.0, 100 * math.fsum(w * c["strict_score"] for w, c in selected))
                         if all(c["strict_score"] is not None for _, c in selected) else None),
        "coverage": min(1.0, sum(w * c["coverage"] for w, c in selected)),
        "strict_coverage": min(1.0, sum(w * c["strict_coverage"] for w, c in selected)),
        "missing_tasks": [t for t in TASKS if weights[t][dimension] > 0
                          and cells[t, dimension]["score"] is None],
        "ci95": None,
        "strict_ci95": None,
    }


def _signature(rows: list[dict], dimension: str, weights: dict) -> tuple:
    fields = ("task", "mode", "group", "difficulty", "item_id", "source_id",
              "source_stratum", "repeat_id")
    return tuple(sorted(tuple(str(r.get(f)) for f in fields) for r in rows
                        if r["dimension"] == dimension and weights[r["task"]][dimension] > 0))


def _interval(values: np.ndarray, required: int) -> tuple[list[float] | None, int]:
    valid = values[np.isfinite(values)]
    if len(valid) < required:
        return None, int(len(valid))
    return [float(x) for x in np.quantile(valid, [0.025, 0.975])], int(len(valid))


def _bootstrap(profile_rows: list[dict], architectures: dict, weights: dict,
               cap: float, samples: int, rng: np.random.Generator) -> dict:
    """Paired ordinary cluster bootstrap, with fixed upper-level weights.

    Each source is sampled once per draw, and its multiplicity applies to all
    modes, variants, repeats and architectures.  Leaf means are recomputed;
    mode/group/difficulty weights never depend on resampled observation counts.
    """
    # Split source strata by their complete fixed-leaf support.  This keeps
    # rare leaves present without silently conditioning on nonempty draws.
    source_patterns: dict[str, dict[str, set[tuple]]] = defaultdict(lambda: defaultdict(set))
    source_strata = {}
    for row in profile_rows:
        if row["source_id"] is not None:
            source = row["source_id"]
            source_strata[source] = row["source_stratum"]
            source_patterns[source][row["architecture"]].add(tuple(row[k] for k in
                ("task", "dimension", "mode", "group", "difficulty")))
    strata: dict[tuple, set[str]] = defaultdict(set)
    inconsistent_sources = set()
    for source, patterns in source_patterns.items():
        signatures = {tuple(sorted(pattern)) for pattern in patterns.values()}
        if len(patterns) != len(architectures) or len(signatures) != 1:
            inconsistent_sources.add(source)
        signature = tuple(sorted(set().union(*patterns.values())))
        strata[source_strata[source], signature].add(source)
    multipliers = {}
    stratum_sizes = {}
    for stratum, sources in sorted(strata.items()):
        ordered = sorted(sources)
        for source in sources:
            stratum_sizes[source] = len(sources)
        if samples > 0:
            matrix = rng.multinomial(len(ordered), [1 / len(ordered)] * len(ordered), size=samples)
            for index, source in enumerate(ordered):
                multipliers[source] = matrix[:, index]

    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    for architecture, output in architectures.items():
        rows = [r for r in profile_rows if r["architecture"] == architecture]
        for dimension in DIMENSIONS:
            axis = output["dimensions"][dimension]
            relevant = [r for r in rows if r["dimension"] == dimension
                        and weights[r["task"]][dimension] > 0]
            for field, cap_value, ci_field in (("score", cap, "ci95"),
                                              ("strict_score", 0.0, "strict_ci95")):
                reason_key = "uncertainty_reason" if field == "score" else "strict_uncertainty_reason"
                count_key = "bootstrap_valid_samples" if field == "score" else "strict_bootstrap_valid_samples"
                axis[count_key] = 0
                if samples == 0:
                    axis[reason_key] = "bootstrap_disabled"
                    continue
                if axis[field] is None:
                    axis[reason_key] = "incomplete_coverage"
                    continue
                if any(r["source_id"] is None for r in relevant):
                    axis[reason_key] = "missing_source_identity"
                    continue
                if any(r["source_id"] in inconsistent_sources for r in relevant):
                    axis[reason_key] = "source_leaf_support_differs_between_architectures"
                    continue
                if any(stratum_sizes[r["source_id"]] < 2 for r in relevant):
                    axis[reason_key] = "fewer_than_two_sources_in_a_sampling_stratum"
                    continue
                leaves: dict[tuple, list[dict]] = defaultdict(list)
                for row in relevant:
                    leaves[(row["task"], row["mode"], row["group"], row["difficulty"])].append(row)
                if any(len({r["source_id"] for r in leaf}) < 2 for leaf in leaves.values()):
                    axis[reason_key] = "fewer_than_two_sources_in_a_leaf"
                    continue
                scores = np.zeros(samples, dtype=float)
                for leaf in leaves.values():
                    items: dict[str, list[dict]] = defaultdict(list)
                    for row in leaf:
                        items[row["item_id"]].append(row)
                    numerator = np.zeros(samples, dtype=float)
                    denominator = np.zeros(samples, dtype=float)
                    for item in items.values():
                        if len({r["source_id"] for r in item}) != 1:
                            raise ValueError("repeats of one item must share source_id")
                        multiplicity = multipliers[item[0]["source_id"]]
                        value = sum(_score(r["y"], r["p"], cap_value) for r in item) / len(item)
                        numerator += multiplicity * value
                        denominator += multiplicity
                    leaf_weight = sum(r["axis_weight"] for r in leaf)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        scores += 100 * leaf_weight * numerator / denominator
                scores = np.clip(scores, 0.0, 100.0)
                interval, valid = _interval(scores, max(2, samples))
                axis[ci_field] = interval
                axis[count_key] = valid
                axis[reason_key] = None if interval is not None else "too_few_valid_cluster_draws"
                arrays[architecture, dimension, field] = scores

    comparisons = []
    for left, right in combinations(sorted(architectures), 2):
        left_rows = [r for r in profile_rows if r["architecture"] == left]
        right_rows = [r for r in profile_rows if r["architecture"] == right]
        for dimension in DIMENSIONS:
            matched = _signature(left_rows, dimension, weights) == _signature(right_rows, dimension, weights)
            comparison = {"left": left, "right": right, "dimension": dimension,
                          "matched_expected_observations": matched,
                          "difference": None, "ci95": None,
                          "strict_difference": None, "strict_ci95": None}
            if not matched:
                comparison["reason"] = "different_expected_observations"
            else:
                for field, difference_key, ci_key in (("score", "difference", "ci95"),
                                                      ("strict_score", "strict_difference", "strict_ci95")):
                    a = architectures[left]["dimensions"][dimension][field]
                    b = architectures[right]["dimensions"][dimension][field]
                    if a is not None and b is not None:
                        comparison[difference_key] = a - b
                    a_draws = arrays.get((left, dimension, field))
                    b_draws = arrays.get((right, dimension, field))
                    if a_draws is not None and b_draws is not None:
                        comparison[ci_key], _ = _interval(a_draws - b_draws,
                                                         max(2, samples))
            comparisons.append(comparison)
    return {"paired_comparisons": comparisons}


def _weight_scenarios(weights: dict, dimension: str) -> list[tuple[str, dict[str, float]]]:
    base = {task: weights[task][dimension] for task in TASKS}
    positive = [task for task in TASKS if base[task] > 0]
    scenarios = [("main", base), ("equal", {task: (1 / len(positive) if task in positive else 0.0)
                                           for task in TASKS})]
    for task in positive:
        for factor in (0.8, 1.2):
            changed = dict(base)
            changed[task] *= factor
            total = sum(changed.values())
            scenarios.append((f"{task}_x{factor:g}", {t: v / total for t, v in changed.items()}))
    return scenarios


def _sensitivity(profile_rows: list[dict], architectures: dict, weights: dict,
                 main_cap: float) -> dict:
    caps = sorted({0.0, 0.25, 0.5, main_cap})
    cells = {}
    for architecture in architectures:
        for task in TASKS:
            for dimension in DIMENSIONS:
                subset = [r for r in profile_rows if r["architecture"] == architecture
                          and r["task"] == task and r["dimension"] == dimension]
                for cap in caps:
                    cells[architecture, task, dimension, cap] = _cell(subset, cap=cap)
    scenarios = []
    ranges: dict[str, dict] = {architecture: {} for architecture in architectures}
    pairwise = []
    for dimension in DIMENSIONS:
        dimension_scenarios = []
        for name, task_weights in _weight_scenarios(weights, dimension):
            for cap in caps:
                scores = {}
                for architecture in architectures:
                    relevant = [(weight, cells[architecture, task, dimension, cap]["score"])
                                for task, weight in task_weights.items() if weight > 0]
                    scores[architecture] = (min(100.0, 100 * math.fsum(w * s for w, s in relevant))
                                            if all(s is not None for _, s in relevant) else None)
                scenario = {"dimension": dimension, "weight_scheme": name,
                            "partial_credit_cap": cap, "weights": task_weights, "scores": scores}
                scenarios.append(scenario)
                dimension_scenarios.append(scenario)
        for architecture in architectures:
            values = [s["scores"][architecture] for s in dimension_scenarios
                      if s["scores"][architecture] is not None]
            ranges[architecture][dimension] = {"min": min(values) if values else None,
                                                "max": max(values) if values else None,
                                                "available_scenarios": len(values),
                                                "total_scenarios": len(dimension_scenarios)}
        for left, right in combinations(sorted(architectures), 2):
            a_rows = [r for r in profile_rows if r["architecture"] == left]
            b_rows = [r for r in profile_rows if r["architecture"] == right]
            matched = _signature(a_rows, dimension, weights) == _signature(b_rows, dimension, weights)
            a = architectures[left]["dimensions"][dimension]["score"]
            b = architectures[right]["dimensions"][dimension]["score"]
            comparison = {"left": left, "right": right, "dimension": dimension,
                          "matched_expected_observations": matched,
                          "baseline_difference": a - b if matched and a is not None and b is not None else None,
                          "rank_reversal": None, "order_changed": None,
                          "reversal_scenarios": [], "order_change_scenarios": []}
            if matched and a is not None and b is not None:
                baseline = 0 if math.isclose(a, b, abs_tol=1e-10, rel_tol=0) else (1 if a > b else -1)
                comparison["rank_reversal"] = False
                comparison["order_changed"] = False
                for scenario in dimension_scenarios:
                    first, second = scenario["scores"][left], scenario["scores"][right]
                    if first is None or second is None:
                        continue
                    order = 0 if math.isclose(first, second, abs_tol=1e-10, rel_tol=0) else (1 if first > second else -1)
                    label = {"weight_scheme": scenario["weight_scheme"],
                             "partial_credit_cap": scenario["partial_credit_cap"]}
                    if baseline * order == -1:
                        comparison["rank_reversal"] = True
                        comparison["reversal_scenarios"].append(label)
                    if order != baseline:
                        comparison["order_changed"] = True
                        comparison["order_change_scenarios"].append(label)
            pairwise.append(comparison)
    return {"ranges": ranges, "pairwise": pairwise, "scenarios": scenarios}


def aggregate_suite(rows: list[dict], weights: dict, *, partial_credit_cap: float = .25,
                    bootstrap_samples: int = 2000, seed: int = 20260909) -> dict:
    """Score every predeclared observation and aggregate within each profile.

    ``y=None`` means the full outcome is unknown; ``p=None`` means partial
    evidence is unavailable.  The caller has already mapped infrastructure and
    provenance failures to null.  Status labels are retained, never guessed at.
    Unknown scores do not redistribute their fixed weight to other rows.
    """
    validated = _validate_weights(weights)
    cap = _unit(partial_credit_cap, "partial_credit_cap")
    if cap is None:
        raise ValueError("partial_credit_cap cannot be null")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    prepared = _prepare(rows, validated, cap)
    profiles = {}
    rng = np.random.default_rng(seed)
    for profile in sorted({r["profile"] for r in prepared}):
        profile_rows = [r for r in prepared if r["profile"] == profile]
        architectures = {}
        for architecture in sorted({r["architecture"] for r in profile_rows}):
            arch_rows = [r for r in profile_rows if r["architecture"] == architecture]
            cells = {}
            task_scores = []
            for task in TASKS:
                for dimension in DIMENSIONS:
                    subset = [r for r in arch_rows if r["task"] == task and r["dimension"] == dimension]
                    cell = _cell(subset, cap=cap)
                    cells[task, dimension] = cell
                    task_scores.append({"profile": profile, "architecture": architecture,
                                        "task": task, "dimension": dimension,
                                        "weight": validated[task][dimension], **cell})
            dimensions = {d: _axis(cells, validated, d) for d in DIMENSIONS}
            architectures[architecture] = {
                "vector": [dimensions[d]["score"] for d in DIMENSIONS],
                "strict_vector": [dimensions[d]["strict_score"] for d in DIMENSIONS],
                "dimensions": dimensions, "task_scores": task_scores,
            }
        bootstrap = _bootstrap(profile_rows, architectures, validated, cap, bootstrap_samples, rng)
        profiles[profile] = {"architectures": architectures, **bootstrap,
                             "sensitivity": _sensitivity(profile_rows, architectures, validated, cap)}
    return {
        "dimension_order": list(DIMENSIONS),
        "partial_credit_cap": cap,
        "weights": validated,
        "profiles": profiles,
        "item_scores": prepared,
        "sensitivity": {"weight_schemes": ["main", "equal", "one_positive_weight_x0.8_or_x1.2_then_normalize"],
                        "partial_credit_caps": sorted({0.0, 0.25, 0.5, cap}),
                        "results_location": "profiles.<profile>.sensitivity"},
        "uncertainty": {
            "method": "paired_stratified_original_source_cluster_bootstrap",
            "samples": bootstrap_samples, "seed": seed, "confidence": 0.95,
            "minimum_sources_per_leaf": 2,
            "minimum_valid_draw_fraction": 1.0,
            "source_stratum_refinement": "fixed_leaf_support_signature",
            "note": "Intervals describe sampled original tasks, not model run-to-run stability. "
                    "Modes, variants and repeats of one source share the same bootstrap multiplicity.",
        },
    }

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys
from typing import Any

from minibench.datasets.xiangqi.engines.pikafish import (
    PikafishAnalysis,
    PikafishEngine,
    PikafishError,
    resolve_pikafish_executable,
)
from minibench.datasets.xiangqi.task_generation import CcpdFenRecord
from minibench.datasets.xiangqi.task_generation import (
    GeneratedXiangqiTask,
    classify_analysis,
    fen_to_position,
    iter_ccpd_fens,
    task_record_from_fen,
    write_jsonl,
)


CATEGORIES = ("tactical-win", "advantage-play", "survival-defense")
TACTICAL_SOURCE_KINDS = {"endgame", "kill-tactic", "fullgame-tactics", "midgame", "unknown"}


def generate_ccpd_pikafish_tasks(
    *,
    ccpd_root: Path,
    output: Path,
    per_category: int = 20,
    seed: int = 20260702,
    prefix: str = "xq-ccpd",
    max_candidates: int | None = None,
    pikafish_path: str | Path | None = None,
    pikafish_eval_file: str | Path | None = None,
    pikafish_depth: int = 8,
    pikafish_timeout: float = 30.0,
    tactical_mate_max: int = 8,
    advantage_cp: int = 500,
    survival_cp: int = 500,
    tactical_max_steps: int = 16,
    survival_max_steps: int = 12,
    allow_partial: bool = False,
    overwrite: bool = False,
    shuffle: bool = True,
    include_matches: bool = False,
    progress_interval: int | None = None,
    start_dir: Path | None = None,
) -> dict[str, Any]:
    if per_category < 1:
        raise ValueError("per_category must be positive")
    if output.exists() and not overwrite:
        raise ValueError(f"{output} already exists; pass overwrite=True to replace it")
    if not ccpd_root.exists():
        raise ValueError(f"CCPD root does not exist: {ccpd_root}")

    rng = random.Random(seed)
    sources = list(iter_ccpd_fens(ccpd_root))
    source_counts = Counter(source.source_kind for source in sources)
    if not include_matches:
        sources = [source for source in sources if source.source_kind != "match"]
    used_source_counts = Counter(source.source_kind for source in sources)
    if shuffle:
        rng.shuffle(sources)
    if max_candidates is not None:
        sources = sources[:max_candidates]

    executable = resolve_pikafish_executable(pikafish_path, start_dir=start_dir or Path.cwd())
    buckets: dict[str, list[GeneratedXiangqiTask]] = {category: [] for category in CATEGORIES}
    accepted_fens: set[str] = set()
    bad_fens: set[str] = set()
    analysis_cache: dict[str, PikafishAnalysis] = {}
    scanned = 0
    analyzed = 0
    skipped = Counter()
    source_plans = category_source_plans(sources)

    with PikafishEngine(
        executable,
        timeout=pikafish_timeout,
        eval_file=pikafish_eval_file,
    ) as engine:
        for target_category in CATEGORIES:
            for source in source_plans[target_category]:
                if len(buckets[target_category]) >= per_category:
                    break
                if source.fen in accepted_fens:
                    skipped["duplicate_accepted_fen"] += 1
                    continue
                if source.fen in bad_fens:
                    skipped["known_bad_fen"] += 1
                    continue
                scanned += 1

                try:
                    fen_to_position(source.fen)
                except ValueError:
                    bad_fens.add(source.fen)
                    skipped["bad_fen"] += 1
                    continue

                analysis = analysis_cache.get(source.fen)
                if analysis is None:
                    try:
                        analysis = engine.analyze_fen(source.fen, depth=pikafish_depth)
                    except (PikafishError, ValueError):
                        bad_fens.add(source.fen)
                        skipped["engine_error"] += 1
                        continue
                    analysis_cache[source.fen] = analysis
                    analyzed += 1
                    if progress_interval and analyzed % progress_interval == 0:
                        print(
                            f"[generate-xiangqi-pikafish] analyzed={analyzed} "
                            f"scanned={scanned} generated="
                            f"{ {key: len(value) for key, value in buckets.items()} }",
                            file=sys.stderr,
                        )

                category = classify_analysis(
                    analysis,
                    tactical_mate_max=tactical_mate_max,
                    advantage_cp=advantage_cp,
                    survival_cp=-abs(survival_cp),
                )
                if category != target_category:
                    skipped[f"{target_category}_miss"] += 1
                    continue

                task_id = f"{prefix}-{category_slug(category)}-{len(buckets[category]) + 1:03d}"
                max_steps = tactical_max_steps if category == "tactical-win" else survival_max_steps
                try:
                    record = task_record_from_fen(
                        task_id=task_id,
                        fen=source.fen,
                        source_file=source.source_file,
                        source_kind=source.source_kind,
                        category=category,
                        analysis=analysis,
                        max_steps=max_steps,
                    )
                except ValueError:
                    bad_fens.add(source.fen)
                    skipped["invalid_task"] += 1
                    continue

                buckets[category].append(GeneratedXiangqiTask(record=record, analysis=analysis))
                accepted_fens.add(source.fen)

    flat_records = [item.record for category in CATEGORIES for item in buckets[category]]
    summary = build_summary(
        output=output,
        buckets=buckets,
        scanned=scanned,
        analyzed=analyzed,
        skipped=skipped,
        per_category=per_category,
        seed=seed,
        pikafish_depth=pikafish_depth,
        tactical_mate_max=tactical_mate_max,
        advantage_cp=advantage_cp,
        survival_cp=-abs(survival_cp),
        source_counts=source_counts,
        used_source_counts=used_source_counts,
        include_matches=include_matches,
    )
    if len(flat_records) < per_category * len(CATEGORIES) and not allow_partial:
        raise RuntimeError(f"failed to generate the requested balanced task set: {summary}")

    write_jsonl(flat_records, output)
    return summary


def category_slug(category: str) -> str:
    return {
        "tactical-win": "tactic",
        "advantage-play": "advantage",
        "survival-defense": "survival",
    }[category]


def category_source_plans(
    sources: list[CcpdFenRecord],
) -> dict[str, list[CcpdFenRecord]]:
    tactical_sources = [
        source for source in sources if source.source_kind in TACTICAL_SOURCE_KINDS
    ]
    return {
        "tactical-win": tactical_sources,
        "advantage-play": sources,
        "survival-defense": sources,
    }


def build_summary(
    *,
    output: Path,
    buckets: dict[str, list[GeneratedXiangqiTask]],
    scanned: int,
    analyzed: int,
    skipped: Counter,
    per_category: int,
    seed: int,
    pikafish_depth: int,
    tactical_mate_max: int,
    advantage_cp: int,
    survival_cp: int,
    source_counts: Counter,
    used_source_counts: Counter,
    include_matches: bool,
) -> dict[str, Any]:
    return {
        "output": str(output),
        "per_category": per_category,
        "generated": {category: len(buckets[category]) for category in CATEGORIES},
        "total": sum(len(items) for items in buckets.values()),
        "scanned": scanned,
        "analyzed": analyzed,
        "seed": seed,
        "pikafish_depth": pikafish_depth,
        "thresholds": {
            "tactical_mate_max": tactical_mate_max,
            "advantage_cp": advantage_cp,
            "survival_cp": survival_cp,
        },
        "include_matches": include_matches,
        "source_counts": dict(sorted(source_counts.items())),
        "used_source_counts": dict(sorted(used_source_counts.items())),
        "skipped": dict(sorted(skipped.items())),
    }

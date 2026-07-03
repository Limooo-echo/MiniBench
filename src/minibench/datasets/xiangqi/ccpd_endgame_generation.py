from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys
from typing import Any

from minibench.datasets.xiangqi.engines.pikafish import (
    PikafishEngine,
    PikafishError,
    resolve_pikafish_executable,
)
from minibench.datasets.xiangqi.task_generation import (
    endgame_score_bucket,
    endgame_task_record_from_fen,
    fen_to_position,
    iter_ccpd_fens,
    write_jsonl,
)


def generate_ccpd_endgame_tasks(
    *,
    ccpd_root: Path,
    output: Path,
    limit: int | None = None,
    seed: int = 20260702,
    prefix: str = "xq-ccpd-endgame",
    pikafish_path: str | Path | None = None,
    pikafish_eval_file: str | Path | None = None,
    pikafish_depth: int = 8,
    pikafish_timeout: float = 30.0,
    max_steps: int = 16,
    engine_label: bool = False,
    validate_actions: bool = False,
    overwrite: bool = False,
    shuffle: bool = False,
    progress_interval: int | None = None,
    start_dir: Path | None = None,
) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if output.exists() and not overwrite:
        raise ValueError(f"{output} already exists; pass overwrite=True to replace it")
    if not ccpd_root.exists():
        raise ValueError(f"CCPD root does not exist: {ccpd_root}")

    sources = [source for source in iter_ccpd_fens(ccpd_root) if source.source_kind == "endgame"]
    total_endgames = len(sources)
    if not sources:
        raise ValueError(f"found no CCPD endgame FEN records under {ccpd_root}")

    if shuffle:
        random.Random(seed).shuffle(sources)
    if limit is not None:
        sources = sources[:limit]

    records: list[dict[str, object]] = []
    skipped = Counter()
    label_counts = Counter()

    if engine_label:
        executable = resolve_pikafish_executable(pikafish_path, start_dir=start_dir or Path.cwd())
        with PikafishEngine(
            executable,
            timeout=pikafish_timeout,
            eval_file=pikafish_eval_file,
        ) as engine:
            _convert_sources(
                sources=sources,
                records=records,
                skipped=skipped,
                label_counts=label_counts,
                prefix=prefix,
                max_steps=max_steps,
                engine=engine,
                pikafish_depth=pikafish_depth,
                validate_actions=validate_actions,
                progress_interval=progress_interval,
            )
    else:
        _convert_sources(
            sources=sources,
            records=records,
            skipped=skipped,
            label_counts=label_counts,
            prefix=prefix,
            max_steps=max_steps,
            engine=None,
            pikafish_depth=pikafish_depth,
            validate_actions=validate_actions,
            progress_interval=progress_interval,
        )

    if not records:
        raise RuntimeError(
            "failed to convert any CCPD endgame tasks; "
            f"skipped={dict(sorted(skipped.items()))}"
        )

    write_jsonl(records, output)
    return {
        "output": str(output),
        "total_endgames": total_endgames,
        "candidates": len(sources),
        "converted": len(records),
        "seed": seed,
        "pikafish_depth": pikafish_depth,
        "max_steps": max_steps,
        "engine_label": engine_label,
        "validate_actions": validate_actions,
        "goal": "agent_survive",
        "category": "endgame-play",
        "label_counts": dict(sorted(label_counts.items())),
        "skipped": dict(sorted(skipped.items())),
    }


def _convert_sources(
    *,
    sources: list[Any],
    records: list[dict[str, object]],
    skipped: Counter,
    label_counts: Counter,
    prefix: str,
    max_steps: int,
    engine: PikafishEngine | None,
    pikafish_depth: int,
    validate_actions: bool,
    progress_interval: int | None,
) -> None:
    total = len(sources)
    for index, source in enumerate(sources, start=1):
        try:
            fen_to_position(source.fen)
        except ValueError:
            skipped["bad_fen"] += 1
            _maybe_print_progress(index, total, len(records), label_counts, progress_interval)
            continue

        analysis = None
        if engine is not None:
            try:
                analysis = engine.analyze_fen(source.fen, depth=pikafish_depth)
            except (PikafishError, ValueError):
                skipped["engine_error"] += 1
                _maybe_print_progress(index, total, len(records), label_counts, progress_interval)
                continue

        task_id = f"{prefix}-{len(records) + 1:03d}"
        try:
            record = endgame_task_record_from_fen(
                task_id=task_id,
                fen=source.fen,
                source_file=source.source_file,
                source_kind=source.source_kind,
                analysis=analysis,
                max_steps=max_steps,
                validate=validate_actions,
            )
        except ValueError:
            skipped["invalid_task"] += 1
            _maybe_print_progress(index, total, len(records), label_counts, progress_interval)
            continue

        records.append(record)
        if analysis is None:
            label_counts["unlabeled"] += 1
        else:
            label_counts[endgame_score_bucket(analysis)] += 1
        _maybe_print_progress(index, total, len(records), label_counts, progress_interval)
    if progress_interval:
        print(file=sys.stderr)


def _maybe_print_progress(
    index: int,
    total: int,
    converted: int,
    label_counts: Counter,
    progress_interval: int | None,
) -> None:
    if not progress_interval:
        return
    if index != total and index % progress_interval != 0:
        return
    width = 28
    filled = int(width * index / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    labels = ",".join(f"{key}:{value}" for key, value in sorted(label_counts.items()))
    print(
        f"\r[generate-ccpd-endgames] [{bar}] {index}/{total} "
        f"converted={converted} labels={labels}",
        file=sys.stderr,
        end="",
        flush=True,
    )

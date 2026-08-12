from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Iterable

from minibench.datasets.zebra.dataset import difficulty_for_size


DERIVATION_SEED = 20260810
CLUE_MARKER = "\n\n## Clues:\n"
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth")
CODEBOOK_PHRASES = (
    "somewhere to the left of",
    "somewhere to the right of",
    "directly left of",
    "directly right of",
    "two houses between",
    "one house between",
    "not next to",
    "next to",
    *(f"in the {ordinal} house" for ordinal in ORDINALS),
)
COUNTERFACTUAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "directly_left_of": ("directly left of",),
    "directly_right_of": ("directly right of",),
    "somewhere_left_of": ("somewhere to the left of",),
    "somewhere_right_of": ("somewhere to the right of",),
    "next_to": ("next to",),
    "one_house_between": ("one house between",),
    "two_houses_between": ("two houses between",),
    "ordinal_house_reference": tuple(
        f"in the {ordinal} house" for ordinal in ORDINALS
    ),
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


def split_puzzle(puzzle: str) -> tuple[str, list[str]]:
    if puzzle.count(CLUE_MARKER) != 1:
        raise ValueError("expected exactly one standard '## Clues:' section")
    background, clue_block = puzzle.split(CLUE_MARKER)
    clues: list[str] = []
    for expected_index, line in enumerate(clue_block.splitlines(), start=1):
        prefix = f"{expected_index}. "
        if not line.startswith(prefix) or not line[len(prefix) :].strip():
            raise ValueError(f"invalid or non-sequential clue line: {line!r}")
        clues.append(line[len(prefix) :])
    if not clues:
        raise ValueError("puzzle contains no clues")
    trailing_newline = "\n" if clue_block.endswith("\n") else ""
    if rebuild_puzzle(background, clues) + trailing_newline != puzzle:
        raise ValueError("clue split is not exactly reversible")
    return background, clues


def rebuild_puzzle(background: str, clues: Iterable[str]) -> str:
    numbered = "\n".join(
        f"{index}. {clue}" for index, clue in enumerate(clues, start=1)
    )
    return f"{background}{CLUE_MARKER}{numbered}"


def derive_codebook_record(
    source: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    puzzle = _required_text(source, "puzzle")
    present = [phrase for phrase in CODEBOOK_PHRASES if phrase in puzzle]
    if not present:
        raise ValueError(f"{source['id']}: no supported codebook phrase")
    rng = random.Random(f"zebra-codebook:{seed}:{source['id']}")
    shuffled = list(present)
    rng.shuffle(shuffled)
    token_by_phrase = {
        phrase: f"[R{index}]" for index, phrase in enumerate(shuffled, start=1)
    }

    transformed = puzzle
    mapping: list[dict[str, Any]] = []
    for phrase in sorted(present, key=len, reverse=True):
        token = token_by_phrase[phrase]
        occurrences = transformed.count(phrase)
        if occurrences < 1:
            raise ValueError(f"{source['id']}: overlapping codebook phrase {phrase!r}")
        transformed = transformed.replace(phrase, token)
        mapping.append(
            {"token": token, "meaning": phrase, "occurrences": occurrences}
        )
    mapping.sort(key=lambda item: int(str(item["token"])[2:-1]))
    restored = transformed
    for item in mapping:
        restored = restored.replace(str(item["token"]), str(item["meaning"]))
    if restored != puzzle:
        raise ValueError(f"{source['id']}: codebook transformation is not reversible")

    rule_lines = [
        "The bracketed tokens below are temporary textual macros for this puzzle only.",
        "Expand each token to its quoted meaning before applying the clues.",
        "Do not reuse a token meaning from any other puzzle.",
        "",
        *(f'- {item["token"]} = "{item["meaning"]}"' for item in mapping),
    ]
    return {
        **_base_variant(source, "temporary_codebook", seed),
        "puzzle": transformed,
        "solution": source["solution"],
        "capability": "rule_condition",
        "rule_mode": "temporary_codebook",
        "rule_context": "\n".join(rule_lines),
        "rule_mapping": mapping,
        "clue_turns": [],
        "tags": _variant_tags(source, "temporary-codebook"),
    }


def derive_history_record(
    source: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    background, clues = split_puzzle(_required_text(source, "puzzle"))
    return {
        **_base_variant(source, "history_memory", seed),
        "puzzle": background,
        "solution": source["solution"],
        "capability": "history_memory",
        "rule_mode": None,
        "rule_context": None,
        "clue_turns": clues,
        "clue_order": "official",
        "tags": _variant_tags(source, "history-memory"),
    }


def derive_counterfactual_candidates(
    sources: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, int]] = {}
    candidates: list[dict[str, Any]] = []
    for source in sources:
        difficulty = difficulty_for_size(_required_text(source, "size"))
        counts.setdefault(difficulty, {})
        puzzle = _required_text(source, "puzzle")
        _, clues = split_puzzle(puzzle)
        house_count = int(str(source["size"]).split("*", 1)[0])
        eligible = [
            operator
            for operator, phrases in COUNTERFACTUAL_PATTERNS.items()
            if any(phrase in puzzle for phrase in phrases)
            and not (operator == "next_to" and house_count < 3)
        ]
        if not eligible:
            raise ValueError(f"{source['id']}: no counterfactual candidate rule")
        smallest_count = min(counts[difficulty].get(operator, 0) for operator in eligible)
        least_used = sorted(
            operator
            for operator in eligible
            if counts[difficulty].get(operator, 0) == smallest_count
        )
        rng = random.Random(f"zebra-counterfactual:{seed}:{source['id']}")
        operator = rng.choice(least_used)
        counts[difficulty][operator] = counts[difficulty].get(operator, 0) + 1
        context, original_semantics, new_semantics = _counterfactual_context(
            operator,
            house_count,
        )
        patterns = COUNTERFACTUAL_PATTERNS[operator]
        affected = [
            index
            for index, clue in enumerate(clues, start=1)
            if any(pattern in clue for pattern in patterns)
        ]
        if not affected:
            raise ValueError(f"{source['id']}: selected rule affects no clues")
        candidates.append(
            {
                **_base_variant(source, "counterfactual_semantics", seed),
                "puzzle": puzzle,
                "solution": None,
                "original_solution": source["solution"],
                "capability": "rule_condition",
                "rule_mode": "counterfactual_semantics",
                "rule_context": context,
                "counterfactual_rule": {
                    "operator": operator,
                    "original_semantics": original_semantics,
                    "new_semantics": new_semantics,
                    "affected_clue_indices": affected,
                },
                "clue_turns": [],
                "validation_status": "pending_manual_review",
                "tags": _variant_tags(
                    source,
                    "counterfactual-candidate",
                    extra=("not-scoreable", f"rule-operator:{operator}"),
                ),
            }
        )
    return candidates


def _counterfactual_context(
    operator: str,
    house_count: int,
) -> tuple[str, str, str]:
    definitions = {
        "directly_left_of": (
            "house(B) - house(A) = 1",
            (
                "house(B) - house(A) = 2"
                if house_count >= 3
                else "house(A) - house(B) = 1"
            ),
            (
                '"A is directly left of B" means A is exactly two houses to the left of B.'
                if house_count >= 3
                else '"A is directly left of B" means A is directly right of B.'
            ),
        ),
        "directly_right_of": (
            "house(A) - house(B) = 1",
            (
                "house(A) - house(B) = 2"
                if house_count >= 3
                else "house(B) - house(A) = 1"
            ),
            (
                '"A is directly right of B" means A is exactly two houses to the right of B.'
                if house_count >= 3
                else '"A is directly right of B" means A is directly left of B.'
            ),
        ),
        "somewhere_left_of": (
            "house(A) < house(B)",
            "house(A) > house(B)",
            '"A is somewhere to the left of B" means A is somewhere to the right of B.',
        ),
        "somewhere_right_of": (
            "house(A) > house(B)",
            "house(A) < house(B)",
            '"A is somewhere to the right of B" means A is somewhere to the left of B.',
        ),
        "next_to": (
            "abs(house(A) - house(B)) = 1",
            "abs(house(A) - house(B)) = 2",
            (
                '"A is next to B" means exactly one house lies between A and B. '
                '"A is not next to B" means their house-number distance is not 2.'
            ),
        ),
        "one_house_between": (
            "abs(house(A) - house(B)) = 2",
            "abs(house(A) - house(B)) = 1",
            '"There is one house between A and B" means A and B are adjacent.',
        ),
        "two_houses_between": (
            "abs(house(A) - house(B)) = 3",
            "abs(house(A) - house(B)) = 2",
            '"There are two houses between A and B" means exactly one house lies between them.',
        ),
    }
    if operator == "ordinal_house_reference":
        mappings = [
            f'"{ORDINALS[index]} house" means House {(index + 1) % house_count + 1}'
            for index in range(house_count)
        ]
        original = ", ".join(
            f"{ORDINALS[index]}=House {index + 1}" for index in range(house_count)
        )
        changed = ", ".join(
            f"{ORDINALS[index]}=House {(index + 1) % house_count + 1}"
            for index in range(house_count)
        )
        body = "; ".join(mappings) + "."
    else:
        original, changed, body = definitions[operator]
    context = "\n".join(
        [
            "Temporary counterfactual rule override (pending solution validation):",
            body,
            "Apply this override to every matching clue in this puzzle.",
            "All other relation phrases retain their ordinary meanings.",
            "House labels in the requested answer always retain their ordinary numeric meanings.",
        ]
    )
    return context, original, changed


def _base_variant(
    source: dict[str, Any],
    variant: str,
    seed: int,
) -> dict[str, Any]:
    source_id = _required_text(source, "id")
    return {
        "id": f"{source_id}__{variant}",
        "source_id": source_id,
        "variant": variant,
        "derivation_seed": seed,
        "size": _required_text(source, "size"),
    }


def _variant_tags(
    source: dict[str, Any],
    variant: str,
    *,
    extra: tuple[str, ...] = (),
) -> list[str]:
    tags = list(source.get("tags", []))
    for tag in (f"variant:{variant}", *extra):
        if tag not in tags:
            tags.append(tag)
    return tags


def _required_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"record has invalid {key!r}")
    return value


def write_jsonl(
    records: Iterable[dict[str, Any]],
    path: str | Path,
    *,
    overwrite: bool,
) -> int:
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite")
    materialized = list(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in materialized:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(materialized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive paired Zebra rule and history variants from a frozen source set."
    )
    parser.add_argument("--source", type=Path, default=Path("data/zebra/eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/zebra"))
    parser.add_argument("--seed", type=int, default=DERIVATION_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sources = load_jsonl(args.source)
        source_ids = [_required_text(source, "id") for source in sources]
        if len(source_ids) != 45 or len(set(source_ids)) != 45:
            raise ValueError("the frozen Zebra source set must contain 45 unique ids")
        codebook = [derive_codebook_record(source, seed=args.seed) for source in sources]
        history = [derive_history_record(source, seed=args.seed) for source in sources]
        counterfactual = derive_counterfactual_candidates(sources, seed=args.seed)
        outputs = {
            "rule_codebook_eval.jsonl": codebook,
            "history_eval.jsonl": history,
            "rule_counterfactual_candidates.jsonl": counterfactual,
        }
        for filename in outputs:
            path = args.output_dir / filename
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"{path} already exists; pass --overwrite")
        counts = {
            filename: write_jsonl(
                records,
                args.output_dir / filename,
                overwrite=args.overwrite,
            )
            for filename, records in outputs.items()
        }
    except (FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(f"Zebra derivation failed: {exc}") from exc
    print(json.dumps({"seed": args.seed, "records": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

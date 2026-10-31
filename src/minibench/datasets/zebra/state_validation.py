"""Strict structured Zebra states and clue-prefix entailment checks.

Every domain is an over-approximation: retaining an extra position is safe;
removing a position needs a proof from the clues already received. Nothing in
this module uses a task's solution, and solver models are never returned.
"""
from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from minibench.datasets.zebra.dataset import ZebraTask
from minibench.datasets.zebra.validation import (
    ParsedPuzzle,
    ValidationError,
    build_solver,
    parse_background,
    parse_puzzle,
)


STATE_SCHEMA_VERSION = "zebra-candidate-domains-v1"
STATE_VALIDATOR_VERSION = "zebra-prefix-entailment-v1"


class StateValidationError(ValueError):
    """The proposed state does not follow the complete public-domain schema."""


def _background(task: ZebraTask) -> ParsedPuzzle:
    if task.rule_context or task.rule_mode:
        raise ValidationError("structured Zebra history does not support rule overrides")
    return parse_background(task.puzzle, expected_size=task.size)


def initial_candidate_state(task: ZebraTask) -> dict[str, Any]:
    parsed = _background(task)
    return {
        "domains": {
            header: {value: list(range(1, parsed.houses + 1)) for value in values}
            for header, values in parsed.domains
        }
    }


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StateValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_state(parsed: ParsedPuzzle, raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            payload = json.loads(raw, object_pairs_hook=_unique_json_object)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise StateValidationError("state must be one complete JSON object") from exc
    else:
        payload = raw
    if not isinstance(payload, dict) or set(payload) != {"domains"}:
        raise StateValidationError("state must contain exactly the domains field")
    domains = payload["domains"]
    if not isinstance(domains, dict) or set(domains) != {h for h, _ in parsed.domains}:
        raise StateValidationError("state must contain every declared attribute, with no extras")
    normalized: dict[str, dict[str, list[int]]] = {}
    for header, values in parsed.domains:
        candidates = domains[header]
        if not isinstance(candidates, dict) or set(candidates) != set(values):
            raise StateValidationError(f"{header}: state must contain every declared value, with no extras")
        normalized[header] = {}
        for value in values:
            positions = candidates[value]
            if not isinstance(positions, list) or not positions:
                raise StateValidationError(f"{header}/{value}: candidates must be a nonempty list")
            if any(type(position) is not int or not 1 <= position <= parsed.houses
                   for position in positions):
                raise StateValidationError(f"{header}/{value}: invalid house number")
            if len(set(positions)) != len(positions):
                raise StateValidationError(f"{header}/{value}: repeated house number")
            normalized[header][value] = sorted(positions)
    return {"domains": normalized}


def parse_candidate_state(task: ZebraTask, raw: str | dict[str, Any]) -> dict[str, Any]:
    """Validate syntax and public entities only; this requires no Z3 dependency.

    Return a fresh normalized object. Candidate domains may grow as well as
    shrink: the unguarded comparison group must be able to correct mistakes.
    """
    return _parse_state(_background(task), raw)


class PrefixStateValidator:
    def __init__(self, task: ZebraTask, *, timeout_ms: int = 10000):
        if type(timeout_ms) is not int or timeout_ms < 1:
            raise ValueError("state verifier timeout_ms must be a positive integer")
        try:
            import z3
        except ImportError as exc:
            raise RuntimeError(
                "incremental_state_pro requires z3-solver; install the "
                "zebra-generation optional dependency before starting the run"
            ) from exc
        self._z3 = z3
        self.task = task
        self.timeout_ms = timeout_ms
        self.background = _background(task)

    def _prefix(self, clue_count: int) -> ParsedPuzzle:
        if type(clue_count) is not int or not 0 <= clue_count <= len(self.task.clue_turns):
            raise ValueError("clue_count must be an integer within the visible clue prefix")
        if clue_count == 0:
            return self.background
        puzzle = self.task.puzzle + "\n\n## Clues:\n" + "\n".join(
            f"{i}. {clue}" for i, clue in enumerate(self.task.clue_turns[:clue_count], 1)
        )
        return parse_puzzle(puzzle, expected_size=self.task.size)

    def validate(self, state: dict[str, Any], clue_count: int) -> dict[str, Any]:
        started = perf_counter()
        candidate = _parse_state(self.background, state)
        parsed = self._prefix(clue_count)
        solver, positions, _ = build_solver(parsed, timeout_ms=self.timeout_ms)
        z3 = self._z3
        prefix_result = solver.check()
        if prefix_result == z3.unsat:
            raise ValidationError(f"Zebra clue prefix {clue_count} is unsatisfiable")
        result: dict[str, Any] = {
            "status": "solver_unknown",
            "accepted": False,
            "clue_count": clue_count,
            "phase": "prefix",
            "validator_version": STATE_VALIDATOR_VERSION,
            "gold_used": False,
            "timeout_ms": self.timeout_ms,
        }
        if prefix_result == z3.unknown:
            result["reason"] = solver.reason_unknown()
        else:
            restrictions = [
                z3.Or(*(positions[header, value] == house for house in houses))
                for header, values in candidate["domains"].items()
                for value, houses in values.items()
            ]
            solver.add(z3.Not(z3.And(*restrictions)))
            entailment_result = solver.check()
            result["phase"] = "entailment"
            if entailment_result == z3.unsat:
                result.update(status="entailed", accepted=True)
            elif entailment_result == z3.sat:
                result.update(status="not_entailed", reason="candidate excludes a position still allowed by the visible clues")
            else:
                result["reason"] = solver.reason_unknown()
        result["elapsed_seconds"] = perf_counter() - started
        return result

    def preflight_prefixes(self) -> dict[str, Any]:
        """Validate this task's prefixes before any model work, without answers.

        This is dataset preflight, not online feedback: no solved states or
        future-clue conclusions are retained or exposed to the model.
        """
        started = perf_counter()
        initial = initial_candidate_state(self.task)
        for count in range(0, len(self.task.clue_turns) + 1):
            result = self.validate(initial, count)
            if not result["accepted"]:
                raise ValidationError(
                    f"Zebra prefix {count} preflight could not prove satisfiability: "
                    f"{result['status']} ({result.get('reason', '')})"
                )
        return {
            "status": "verified_prefixes",
            "prefixes": len(self.task.clue_turns) + 1,
            "elapsed_seconds": perf_counter() - started,
            "gold_used": False,
            "validator_version": STATE_VALIDATOR_VERSION,
        }

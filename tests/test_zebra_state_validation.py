from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import unittest
from unittest.mock import Mock, patch

from minibench.datasets.zebra.dataset import zebra_task_from_dict
from minibench.datasets.zebra.state_validation import (
    PrefixStateValidator,
    StateValidationError,
    initial_candidate_state,
    parse_candidate_state,
)
from minibench.datasets.zebra.validation import ValidationError, build_solver


HAS_Z3 = importlib.util.find_spec("z3") is not None


def history_task(clues=None):
    return zebra_task_from_dict({
        "id": "state-fixture",
        "size": "3*2",
        "capability": "history_memory",
        "puzzle": (
            "There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. "
            "Each house is occupied by a different person. Each house has a unique attribute for each "
            "of the following characteristics:\n"
            " - Each person has a unique name: `Alice`, `Bob`, `Eric`\n"
            " - Each person has a favorite color: `red`, `green`, `blue`"
        ),
        "clue_turns": clues or [
            "Alice is not in the third house.",
            "Alice is in the first house.",
            "Bob is in the second house.",
            "The person who loves red is Alice.",
            "The person who loves green is Bob.",
        ],
        "solution": {
            "header": ["House", "Name", "Color"],
            "rows": [["1", "Alice", "red"], ["2", "Bob", "green"], ["3", "Eric", "blue"]],
        },
    })


class CandidateStateSchemaTests(unittest.TestCase):
    def test_initial_state_uses_only_background_and_is_not_aliased(self):
        task = replace(history_task(), solution=object())
        state = initial_candidate_state(task)
        self.assertEqual(state["domains"]["Name"]["Alice"], [1, 2, 3])
        state["domains"]["Name"]["Alice"].remove(3)
        self.assertEqual(state["domains"]["Name"]["Bob"], [1, 2, 3])
        self.assertEqual(initial_candidate_state(task)["domains"]["Name"]["Alice"], [1, 2, 3])

    def test_roundtrip_normalizes_and_copies_candidates(self):
        task = history_task()
        state = initial_candidate_state(task)
        state["domains"]["Name"]["Alice"] = [2, 1]
        normalized = parse_candidate_state(task, state)
        self.assertEqual(normalized["domains"]["Name"]["Alice"], [1, 2])
        self.assertEqual(parse_candidate_state(task, json.dumps(state)), normalized)
        normalized["domains"]["Name"]["Alice"].append(3)
        self.assertEqual(state["domains"]["Name"]["Alice"], [2, 1])

    def test_candidate_domains_can_expand_to_recover_from_an_error(self):
        task = history_task()
        smaller = initial_candidate_state(task)
        smaller["domains"]["Name"]["Alice"] = [2]
        self.assertEqual(parse_candidate_state(task, smaller), smaller)
        self.assertEqual(parse_candidate_state(task, initial_candidate_state(task)), initial_candidate_state(task))

    def test_invalid_candidates_are_rejected(self):
        task = history_task()
        for positions in ([], [0], [4], [True], [1.0], ["1"], [1, 1], None, "1", {"1": True}):
            state = initial_candidate_state(task)
            state["domains"]["Name"]["Alice"] = positions
            with self.subTest(positions=positions), self.assertRaises(StateValidationError):
                parse_candidate_state(task, state)

    def test_missing_unknown_and_free_text_fields_are_rejected(self):
        task = history_task()
        baseline = initial_candidate_state(task)
        bad = []
        missing_attribute = deepcopy(baseline)
        del missing_attribute["domains"]["Color"]
        bad.append(missing_attribute)
        missing_value = deepcopy(baseline)
        del missing_value["domains"]["Name"]["Alice"]
        bad.append(missing_value)
        unknown_value = deepcopy(baseline)
        unknown_value["domains"]["Name"]["Mallory"] = [1]
        bad.append(unknown_value)
        extra_text = deepcopy(baseline)
        extra_text["reasoning"] = "Alice is certainly at house 2"
        bad.extend((extra_text, {}, [], None, '{"candidate_state":"Alice is at 2"}'))
        for state in bad:
            with self.subTest(state=state), self.assertRaises(StateValidationError):
                parse_candidate_state(task, state)

    def test_duplicate_json_and_trailing_prose_are_rejected(self):
        task = history_task()
        raw = json.dumps(initial_candidate_state(task))
        for state in ('{"domains":{},"domains":{}}', raw + '\nBecause I think so.', '```json\n' + raw + '\n```'):
            with self.subTest(state=state), self.assertRaises(StateValidationError):
                parse_candidate_state(task, state)


@unittest.skipUnless(HAS_Z3, "z3-solver optional dependency is not installed")
class PrefixStateValidatorTests(unittest.TestCase):
    def test_initial_and_sound_partial_states_pass_multi_solution_prefix(self):
        task = history_task()
        checker = PrefixStateValidator(task)
        initial = initial_candidate_state(task)
        self.assertTrue(checker.validate(initial, 0)["accepted"])
        self.assertTrue(checker.validate(initial, 1)["accepted"])
        partial = deepcopy(initial)
        partial["domains"]["Name"]["Alice"] = [1, 2]
        verdict = checker.validate(partial, 1)
        self.assertEqual(verdict["status"], "entailed")
        self.assertTrue(verdict["accepted"])
        self.assertFalse(verdict["gold_used"])

    def test_compatible_but_unsupported_guess_is_rejected(self):
        import z3
        task = history_task()
        checker = PrefixStateValidator(task)
        for guessed_house in (1, 2):
            proposal = initial_candidate_state(task)
            proposal["domains"]["Name"]["Alice"] = [guessed_house]
            solver, positions, _ = build_solver(checker._prefix(1))
            solver.add(positions["Name", "Alice"] == guessed_house)
            self.assertEqual(solver.check(), z3.sat)
            verdict = checker.validate(proposal, 1)
            self.assertEqual(verdict["status"], "not_entailed")
            self.assertFalse(verdict["accepted"])
        # The later clue proves house 1; the same proposal can then be committed.
        proposal["domains"]["Name"]["Alice"] = [1]
        self.assertTrue(checker.validate(proposal, 2)["accepted"])

    def test_one_unsupported_domain_rejects_whole_proposal(self):
        task = history_task()
        proposal = initial_candidate_state(task)
        proposal["domains"]["Name"]["Alice"] = [1, 2]
        proposal["domains"]["Color"]["red"] = [1]
        self.assertEqual(PrefixStateValidator(task).validate(proposal, 1)["status"], "not_entailed")

    def test_gold_and_future_clues_do_not_participate(self):
        task = replace(history_task(), solution=object(), clue_turns=(
            "Alice is not in the third house.",
            "This future clue is deliberately not supported by the parser.",
        ))
        checker = PrefixStateValidator(task)
        initial = initial_candidate_state(task)
        self.assertTrue(checker.validate(initial, 1)["accepted"])
        with self.assertRaises(ValidationError):
            checker.validate(initial, 2)

    def test_inconsistent_prefix_is_a_dataset_error_not_a_candidate_error(self):
        task = history_task(["Alice is in the first house.", "Alice is in the second house."])
        with self.assertRaisesRegex(ValidationError, "unsatisfiable"):
            PrefixStateValidator(task).validate(initial_candidate_state(task), 2)

    def test_unknown_and_timeout_never_accept_and_are_not_wrong_state_labels(self):
        import z3
        task = history_task()
        checker = PrefixStateValidator(task)
        _, positions, _ = build_solver(checker._prefix(1))
        for checks, phase in (([z3.unknown], "prefix"), ([z3.sat, z3.unknown], "entailment")):
            solver = Mock()
            solver.check.side_effect = checks
            solver.reason_unknown.return_value = "timeout"
            with patch("minibench.datasets.zebra.state_validation.build_solver", return_value=(solver, positions, [])):
                result = checker.validate(initial_candidate_state(task), 1)
            self.assertEqual(result["status"], "solver_unknown")
            self.assertEqual(result["reason"], "timeout")
            self.assertEqual(result["phase"], phase)
            self.assertFalse(result["accepted"])

    def test_preflight_and_count_validation(self):
        task = history_task()
        checker = PrefixStateValidator(task)
        self.assertEqual(checker.preflight_prefixes()["prefixes"], len(task.clue_turns) + 1)
        for count in (-1, True, 1.0, len(task.clue_turns) + 1):
            with self.subTest(count=count), self.assertRaises(ValueError):
                checker.validate(initial_candidate_state(task), count)
        for timeout in (0, -1, True, 1.5):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                PrefixStateValidator(task, timeout_ms=timeout)


if __name__ == "__main__":
    unittest.main()

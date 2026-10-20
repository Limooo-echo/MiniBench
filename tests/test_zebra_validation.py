from __future__ import annotations

from copy import deepcopy
import importlib.util
from itertools import permutations
import json
from pathlib import Path
import unittest

from minibench.datasets.zebra.validation import (Clause, ParsedPuzzle, ValidationError,
    applicable_operators, counterfactual_context, evaluate_clause, parse_puzzle,
    solve_unique, verify_record)

ROOT = Path(__file__).resolve().parents[1]
HAS_Z3 = importlib.util.find_spec("z3") is not None


def fixture(clues: list[str]) -> dict:
    return {"id": "independent-fixture", "size": "3*2", "puzzle":
        "There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. "
        "Each house is occupied by a different person. Each house has a unique attribute for each "
        "of the following characteristics:\n"
        " - Each person has a unique name: `Alice`, `Bob`, `Eric`\n"
        " - Each person has a favorite color: `red`, `green`, `blue`\n\n## Clues:\n" +
        "\n".join(f"{i}. {c}" for i, c in enumerate(clues, 1)),
        "solution": {"header": ["House", "Name", "Color"],
            "rows": [["1", "Alice", "red"], ["2", "Bob", "green"], ["3", "Eric", "blue"]]}}


class ZebraParserTests(unittest.TestCase):
    def test_exact_entities_with_internal_is_are_not_split(self):
        row = fixture(["The person whose favorite color is red is Alice."])
        clause = parse_puzzle(row["puzzle"]).clauses[0]
        self.assertEqual(clause.operator, "equal")
        self.assertEqual(clause.left, ("Color", "red"))
        self.assertEqual(clause.right, ("Name", "Alice"))

    def test_rejects_unknown_entity_clause_and_background(self):
        for clue in ("Alice is beside Bob.", "Alice might be in the first house.",
                     "Nobody is Alice.", "Alice is in the fourth house."):
            with self.subTest(clue=clue), self.assertRaises(ValidationError):
                parse_puzzle(fixture([clue])["puzzle"])
        row = fixture(["Alice is Bob."])
        with self.assertRaises(ValidationError):
            parse_puzzle(row["puzzle"].replace("Each person has a favorite color", "Unknown category"))

    def test_override_applicability_includes_negated_ordinals(self):
        row = fixture(["Alice is not in the first house.", "Bob is not next to Eric."])
        parsed = parse_puzzle(row["puzzle"])
        self.assertEqual(applicable_operators(parsed), ["next_to", "ordinal_house_reference"])
        self.assertTrue(all(clause.negated for clause in parsed.clauses))


@unittest.skipUnless(HAS_Z3, "z3-solver generation dependency is not installed")
class ZebraSolverTests(unittest.TestCase):
    def test_blind_unique_proof_then_gold_comparison(self):
        clues = ["Alice is in the first house.", "Bob is in the second house.",
                 "The person who loves red is Alice.", "The person who loves green is Bob."]
        row = fixture(clues)
        proof = verify_record(row)
        self.assertEqual(proof["status"], "verified_unique")
        self.assertEqual((proof["first_check"], proof["second_check"]), ("sat", "unsat"))
        self.assertFalse(proof["gold_asserted"])
        self.assertTrue(proof["integer_clauses_satisfied"])
        row["solution"]["rows"][0][1] = "Bob"
        self.assertEqual(verify_record(row)["status"], "gold_mismatch")

    def test_no_solution_and_multiple_solutions_are_rejected(self):
        self.assertEqual(verify_record(fixture(["Alice is Bob."]))["status"], "unsatisfiable")
        self.assertEqual(verify_record(fixture(["Alice is in the first house."]))["status"],
                         "multiple_solutions")
        pending = fixture(["Alice is in the first house."])
        pending["solution"] = None
        self.assertEqual(verify_record(pending)["status"], "parse_rejected")

    def test_all_operators_and_negations_match_exhaustive_small_grids(self):
        operators = ("equal", "ordinal_house_reference", "directly_left_of", "directly_right_of",
                     "somewhere_left_of", "somewhere_right_of", "next_to", "one_house_between",
                     "two_houses_between")
        for houses in (2, 3):
            names = ("Alice", "Bob", "Eric")[:houses]
            colors = ("red", "green", "blue")[:houses]
            domains = (("Name", names), ("Color", colors))
            # Color positions are fixed; names are independently enumerated.
            anchors = [Clause(i, "fixture", "ordinal_house_reference", ("Color", color), i)
                       for i, color in enumerate(colors, 1)]
            for op in operators:
                for negated in (False, True):
                    for override in (None, op) if op != "equal" else (None,):
                        right = 1 if op == "ordinal_house_reference" else ("Name", "Bob")
                        clause = Clause(houses + 1, "fixture", op, ("Name", "Alice"), right, negated)
                        parsed = ParsedPuzzle(houses, domains, tuple([*anchors, clause]))
                        solutions = []
                        for perm in permutations(range(1, houses + 1)):
                            positions = {("Name", value): p for value, p in zip(names, perm)}
                            # Ordinal overrides apply to ALL ordinal clauses, including anchors.
                            positions.update({("Color", color): (i % houses + 1 if override == "ordinal_house_reference" else i)
                                              for i, color in enumerate(colors, 1)})
                            if all(evaluate_clause(c, positions, houses, override) for c in parsed.clauses):
                                solutions.append(positions)
                        proof = solve_unique(parsed, override=override)
                        expected = "unsatisfiable" if not solutions else "verified_unique" if len(solutions) == 1 else "multiple_solutions"
                        with self.subTest(houses=houses, operator=op, negated=negated, override=override):
                            self.assertEqual(proof["status"], expected)
                            if len(solutions) == 1:
                                expected_rows = [[str(i), next(n for n in names if solutions[0]["Name", n] == i),
                                                  next(c for c in colors if solutions[0]["Color", c] == i)]
                                                 for i in range(1, houses + 1)]
                                self.assertEqual(proof["solution"]["rows"], expected_rows)

    def test_formal_counterfactual_rejects_tampered_rule_context(self):
        path = ROOT / "data/zebra/rule_counterfactual_eval.jsonl"
        if not path.is_file():
            self.skipTest("formal release not yet built")
        row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(verify_record(row)["status"], "verified_unique")
        row["rule_context"] += "\nActually use the original rules."
        self.assertEqual(verify_record(row)["status"], "parse_rejected")

    def test_formal_codebook_rejects_tampered_public_rule_context(self):
        path = ROOT / "data/zebra/rule_codebook_eval.jsonl"
        row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(verify_record(row)["status"], "verified_unique")
        row["rule_context"] = row["rule_context"].replace(row["rule_mapping"][0]["meaning"],
                                                       "somewhere to the opposite side")
        result = verify_record(row)
        self.assertEqual(result["status"], "parse_rejected")
        self.assertIn("public context", result["reason"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_zebra_paired import check_pairing, paired_records, verify_groups

HAS_Z3 = importlib.util.find_spec("z3") is not None


def fixtures():
    cf = [json.loads(line) for line in (ROOT / "data/zebra/rule_counterfactual_eval.jsonl")
          .read_text(encoding="utf-8").splitlines()][:2]
    direct = [json.loads(line) for line in (ROOT / "data/zebra/eval.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    return cf, direct


class ZebraPairedStructureTests(unittest.TestCase):
    def test_all_three_groups_keep_source_order_and_gold_contracts(self):
        cf, previous = fixtures()
        before = deepcopy(cf)
        groups = paired_records(cf, previous, expected_count=2)
        self.assertEqual(check_pairing(groups, expected_count=2), [row["source_id"] for row in cf])
        self.assertEqual(cf, before)
        self.assertEqual(groups["direct"][0]["solution"], cf[0]["original_solution"])
        self.assertEqual(groups["codebook"][0]["solution"], cf[0]["original_solution"])
        self.assertNotEqual(groups["counterfactual"][0]["solution"], cf[0]["original_solution"])

    def test_rejects_wrong_count_and_repeated_sources(self):
        cf, previous = fixtures()
        with self.assertRaisesRegex(ValueError, "expected 100"):
            paired_records(cf, previous)
        cf[1]["source_id"] = cf[0]["source_id"]
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            paired_records(cf, previous, expected_count=2)

    def test_rejects_parent_disagreement_and_variant_reordering(self):
        cf, previous = fixtures()
        groups = paired_records(cf, previous, expected_count=2)
        groups["codebook"].reverse()
        with self.assertRaisesRegex(ValueError, "source set/order"):
            check_pairing(groups, expected_count=2)
        cf[0]["puzzle"] += "\n"
        with self.assertRaisesRegex(ValueError, "parent disagrees"):
            paired_records(cf, previous, expected_count=2)

    def test_rejects_extra_clues_and_unchanged_counterfactual_answer(self):
        cf, previous = fixtures()
        groups = paired_records(cf, previous, expected_count=2)
        groups["codebook"][0]["puzzle"] += "\n3. Eric is in the first house."
        with self.assertRaisesRegex(ValueError, "source clues"):
            check_pairing(groups, expected_count=2)
        groups = paired_records(cf, previous, expected_count=2)
        groups["counterfactual"][0]["solution"] = deepcopy(groups["direct"][0]["solution"])
        with self.assertRaisesRegex(ValueError, "answer must change"):
            check_pairing(groups, expected_count=2)


@unittest.skipUnless(HAS_Z3, "z3-solver generation dependency is not installed")
class ZebraPairedSolverTests(unittest.TestCase):
    def test_fresh_proofs_cover_every_group_and_ignore_old_certificates(self):
        cf, previous = fixtures()
        cf[0]["validation_evidence"] = {"status": "untrusted-old-certificate"}
        groups = paired_records(cf, previous, expected_count=2)
        proofs = verify_groups(groups)
        self.assertEqual(len(proofs), 6)
        self.assertTrue(all(p["first_check"] == "sat" and p["second_check"] == "unsat"
                            and p["gold_matches"] and not p["gold_asserted"] for p in proofs))

    def test_rejects_wrong_gold_despite_old_verified_status(self):
        cf, previous = fixtures()
        groups = paired_records(cf, previous, expected_count=2)
        groups["direct"][0]["solution"]["rows"][0][1] = "not-a-domain-value"
        with self.assertRaisesRegex(ValueError, "fresh verification failed.*gold_mismatch"):
            verify_groups(groups)

    def test_rejects_unsatisfiable_and_underdetermined_puzzles(self):
        cf, previous = fixtures()
        for clues, failure in (("1. Eric is in the first house.\n2. Eric is in the second house.",
                               "unsatisfiable"),
                              ("1. Eric is in the first house.", "multiple_solutions")):
            groups = paired_records(cf, previous, expected_count=2)
            background = groups["direct"][0]["puzzle"].split("\n\n## Clues:\n")[0]
            groups["direct"][0]["puzzle"] = background + "\n\n## Clues:\n" + clues
            with self.subTest(failure=failure), self.assertRaisesRegex(ValueError, failure):
                verify_groups(groups)


if __name__ == "__main__":
    unittest.main()

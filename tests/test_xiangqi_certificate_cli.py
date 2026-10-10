"""Fresh-output certification must recheck witnesses, not trust cached flags."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "certificate_cli", ROOT / "scripts/certify_xiangqi_c2.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class CertificateCliTests(unittest.TestCase):
    def test_matching_witness_is_recomputed_not_reference_pv(self):
        record = {"id": "fixture", "c2_analysis": {"principal_variation_uci": ["a0a1"]}}
        witness = ["b0b1", "d9d8", "b1b8"]
        with patch.object(cli, "verify_winning_line", return_value={"valid": True}) as verify:
            key, result = cli.check(record, candidate_line=witness)
        verify.assert_called_once_with(record, witness)
        self.assertEqual(key, "fixture")
        self.assertEqual(result["moves_uci"], witness)

    def test_invalid_candidate_falls_back_to_reference_without_trusting_it(self):
        record = {"id": "fixture", "c2_analysis": {"principal_variation_uci": ["a0a1"]}}
        with patch.object(cli, "verify_winning_line", side_effect=[{"valid": False}, {"valid": False}]) as verify:
            _, result = cli.check(record, candidate_line=["b0b1"])
        self.assertEqual(verify.call_count, 2)
        self.assertFalse(result["valid"])

    def test_wrong_policy_or_dataset_cannot_seed_witnesses(self):
        expected = {"version": "v1", "policy": {"depth": 3}, "policy_fingerprint": "p", "dataset_sha256": "d"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "seed.json"
            for changed in ("version", "policy", "policy_fingerprint", "dataset_sha256"):
                prior = {**expected, changed: "wrong", "tasks": {"fixture": {"valid": True, "moves_uci": ["a0a1"]}}}
                path.write_text(json.dumps(prior), encoding="utf-8")
                with self.subTest(field=changed):
                    self.assertEqual(cli.candidate_lines((path,), expected), {})

    def test_existing_complete_output_is_revalidated(self):
        record = {"id": "fixture", "c2_analysis": {"principal_variation_uci": ["a0a1"]}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "certificate.json"
            prior = {"version": cli.CERTIFICATE_VERSION, "policy": cli.POLICY,
                     "policy_fingerprint": "policy", "dataset_sha256": "data", "complete": True,
                     "tasks": {"fixture": {"valid": True, "moves_uci": ["b0b1"]}}}
            output.write_text(json.dumps(prior), encoding="utf-8")
            with patch.object(sys, "argv", ["certify", "--output", str(output), "--seed-certificates", str(root / "absent.json")]), \
                 patch.object(cli, "load_records", return_value=[record]), \
                 patch.object(cli, "sha256_file", return_value="data"), \
                 patch.object(cli, "policy_fingerprint", return_value="policy"), \
                 patch.object(cli, "ProcessPoolExecutor", ThreadPoolExecutor), \
                 patch.object(cli, "verify_winning_line", return_value={"valid": False}) as verify, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(), 1)
            self.assertEqual(verify.call_count, 2)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(result["complete"])
            self.assertFalse(result["tasks"]["fixture"]["valid"])


if __name__ == "__main__":
    unittest.main()

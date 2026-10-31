"""Offline checks for complete-run validation and safe collection."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("zebra_server_collection", ROOT / "scripts/collect_zebra_server_run.py")
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Minimal fixtures keep one final call per job while preserving all 600 work keys.
        expected_calls = patch.object(collector, "EXPECTED_CALLS", 600)
        expected_calls.start()
        self.addCleanup(expected_calls.stop)

    def make_complete_run(self):
        batch = self.root / collector.NAME
        batch.mkdir()
        write_json(batch / "batch-status.json", {"status": "completed"})
        write_json(batch / "parallel-status.json", {"status": "completed", "completed_total": 600, "failures": [], "active_jobs": []})
        tasks = [{"id": f"task-{i}", "clue_turns": []} for i in range(100)]
        (batch / "history_eval.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
        write_json(batch / "batch-manifest.json", {"data_sha256": collector.digest(batch / "history_eval.jsonl")})
        for treatment in collector.TREATMENTS:
            rows = []
            for task in tasks:
                for mode in collector.MODES:
                    rows.append({"task_id": task["id"], "memory_mode": mode, "raw_output": "{}", "history_trace": [{"raw_output": "{}"}]})
                    folder = batch / "raw_data" / treatment / f"{task['id']}-{mode}"
                    write_json(folder / "metadata.json", {"task_id": task["id"], "mode": mode, "treatment": treatment, "planned_calls": 1})
                    calls = folder / "calls"
                    request = {"model": "qwen3.8-flash", "enable_thinking": treatment == "thinking_on"}
                    response = {"choices": [{"message": {"content": "{}", "reasoning_content": "retained raw native reasoning"}}]}
                    for name, value in [("request", request), ("response", response)]:
                        write_json(calls / f"0000.{name}.json", value)
                        (calls / f"0000.{name}.raw").write_bytes(json.dumps(value).encode())
                    record = {"status": "completed", "endpoint": collector.ENDPOINT,
                              "request_sha256": collector.fingerprint({"endpoint": collector.ENDPOINT, "payload": request}),
                              "response_raw_sha256": hashlib.sha256(json.dumps(response).encode()).hexdigest(),
                              "response": {"content": "{}", "reasoning": "retained raw native reasoning"},
                              "completion_trace": {"content": "{}", "reasoning_content": "retained raw native reasoning"}}
                    record["record_sha256"] = collector.fingerprint(record)
                    write_json(calls / "0000.record.json", record)
            target = batch / treatment / "predictions.jsonl"
            target.parent.mkdir()
            target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return batch

    def test_requires_complete_600_with_every_raw_call(self):
        batch = self.make_complete_run()
        audit = collector.validate_experiment(batch)
        self.assertEqual((audit["instances"], audit["raw_calls"]), (600, 600))
        missing = next((batch / "raw_data").rglob("*.response.raw"))
        missing.unlink()
        with self.assertRaises(FileNotFoundError):
            collector.validate_experiment(batch)

    def test_rejects_duplicate_results_and_unfinished_status(self):
        batch = self.make_complete_run()
        predictions = batch / "thinking_off/predictions.jsonl"
        with predictions.open("a", encoding="utf-8") as stream:
            stream.write(predictions.read_text(encoding="utf-8").splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            collector.validate_experiment(batch)
        write_json(batch / "parallel-status.json", {"status": "running", "completed_total": 600})
        with self.assertRaisesRegex(ValueError, "does not confirm"):
            collector.validate_experiment(batch)

    def test_archive_verified_and_conflicting_local_file_preserved(self):
        batch = self.make_complete_run()
        (batch / "__pycache__").mkdir()
        (batch / "__pycache__/skip.pyc").write_bytes(b"cache")
        receipt = collector.create_archive(batch)
        self.assertEqual(receipt, collector.create_archive(batch))
        archive = Path(receipt["archive_path"])
        manifest = collector.verify_archive(archive, receipt["archive_sha256"], receipt["manifest_sha256"])
        self.assertFalse(any("__pycache__" in row["path"] for row in manifest["files"]))
        destination = self.root / "download"
        collector.extract_checked(archive, destination, manifest)
        collector.extract_checked(archive, destination, manifest)
        self.assertEqual(collector.validate_experiment(destination), receipt["audit"])
        target = destination / "history_eval.jsonl"
        target.write_text("local work", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            collector.extract_checked(archive, destination, manifest)
        self.assertEqual(target.read_text(encoding="utf-8"), "local work")

    def test_rejects_credentials_and_archive_path_attacks(self):
        with self.assertRaises(ValueError):
            collector.reject_credentials("launcher/.env")
        with self.assertRaises(ValueError):
            collector.reject_secret_fields({"nested": {"api_key": "not-a-real-key"}}, "example.json")
        for name in (".", "../outside", "/absolute", "D:/outside", "a\\outside", "NUL", "x/COM1.txt", "x./a"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                collector.safe_name(name)
        archive = self.root / "malicious.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            link = tarfile.TarInfo("link")
            link.type, link.linkname = tarfile.SYMTYPE, "../outside"
            bundle.addfile(link)
        with self.assertRaisesRegex(ValueError, "non-regular"):
            collector.verify_archive(archive)

    def test_preserves_remote_validation_error_without_real_ssh(self):
        result = SimpleNamespace(returncode=1, stdout=b'{"error":"ValueError: missing raw data"}', stderr=b"")
        with patch.object(collector.subprocess, "run", return_value=result), self.assertRaisesRegex(RuntimeError, "missing raw data"):
            collector.remote_action("ubuntu@example.test", "unused-key-path", "archive", "/unused")

    def test_rejects_tampered_record_even_with_valid_json_and_preserves_text(self):
        batch = self.make_complete_run()
        path = next((batch / "raw_data").rglob("*.record.json"))
        record = collector.read_json(path)
        record["response"]["content"] = "changed after capture"
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, "record checksum"):
            collector.validate_experiment(batch)
        self.assertEqual(collector.message_text([{"text": " first "}, "middle", {"content": " last "}]), " first middle last ")


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.core.checkpoint import (
    RunLock,
    RunLockedError,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    fingerprint_payload,
    read_json_object,
    read_jsonl_objects,
)


class CheckpointPrimitiveTests(unittest.TestCase):
    def test_fingerprint_is_canonical(self):
        first = {"b": [2, 1], "a": {"value": "x"}}
        reordered = {"a": {"value": "x"}, "b": [2, 1]}

        self.assertEqual(fingerprint_payload(first), fingerprint_payload(reordered))
        self.assertNotEqual(
            fingerprint_payload(first),
            fingerprint_payload({"a": {"value": "x"}, "b": [1, 2]}),
        )

    def test_atomic_writers_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            atomic_write_text(root / "message.txt", "hello\n")
            atomic_write_json(root / "state.json", {"status": "running", "count": 2})
            atomic_write_jsonl(root / "rows.jsonl", ({"index": i} for i in range(3)))

            self.assertEqual((root / "message.txt").read_text(encoding="utf-8"), "hello\n")
            self.assertEqual(
                read_json_object(root / "state.json"),
                {"status": "running", "count": 2},
            )
            self.assertEqual(
                read_jsonl_objects(root / "rows.jsonl"),
                [{"index": 0}, {"index": 1}, {"index": 2}],
            )

    def test_failed_replace_preserves_previous_file_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "state.json"
            destination.write_text('{"old":true}\n', encoding="utf-8")

            with patch(
                "minibench.core.checkpoint.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    atomic_write_json(destination, {"new": True})

            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"old": True},
            )
            self.assertEqual(
                [path for path in destination.parent.iterdir() if path.suffix == ".tmp"],
                [],
            )

    def test_second_lock_owner_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with RunLock(tmpdir):
                with self.assertRaises(RunLockedError):
                    with RunLock(tmpdir):
                        self.fail("second lock unexpectedly acquired")

    def test_custom_sidecar_name_rejects_second_owner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with RunLock(tmpdir, lock_name=".zebra-lock-unit"):
                with self.assertRaises(RunLockedError):
                    with RunLock(tmpdir, lock_name=".zebra-lock-unit"):
                        self.fail("second sidecar lock unexpectedly acquired")

    def test_jsonl_reader_reports_corruption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            path.write_text('{"ok":true}\nnot-json\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid JSONL"):
                read_jsonl_objects(path)


if __name__ == "__main__":
    unittest.main()

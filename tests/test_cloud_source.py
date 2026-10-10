"""Offline release-bundle and user-facing documentation regressions."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cloud_source_export", ROOT / "scripts/export_cloud_source.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class CloudSourceTests(unittest.TestCase):
    def test_entrypoint_and_changelog_are_included(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("run.sh", "CHANGELOG.md", "README.md", ".env.example", ".env"):
                (root / name).write_text("fixture", encoding="utf-8")
            with patch.object(exporter, "ROOT", root):
                names = {p.relative_to(root).as_posix() for p in exporter.source_files()}
            self.assertEqual(names, {"run.sh", "CHANGELOG.md", "README.md", ".env.example"})

    def test_nested_artifacts_and_private_configs_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keep = {"src/example.py", "docs/cloud-run.md", "config/cloud/pilot.yaml"}
            discard = {
                "src/__pycache__/example.pyc", "src/example.log", "src/example.tmp",
                "src/pkg.egg-info/PKG-INFO", "docs/.env", "docs/.env.private",
                "config/cloud/local-account.yaml", "data/runs/result.json",
                "data/output/result.json", "scripts/tmp/debug.py",
                "run/result.json", "runs/result.json", "output/result.json",
            }
            for name in keep | discard:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("fixture", encoding="utf-8")
            with patch.object(exporter, "ROOT", root):
                names = {p.relative_to(root).as_posix() for p in exporter.source_files()}
            self.assertEqual(names, keep)

    def test_symlinks_are_rejected_including_root_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "original"
            target.write_text("fixture", encoding="utf-8")
            try:
                (root / "README.md").symlink_to(target)
            except OSError as exc:
                self.skipTest(f"Symlink creation unavailable: {type(exc).__name__}")
            with patch.object(exporter, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    list(exporter.source_files())

    def test_output_cannot_pollute_source_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "data/bundle.tar.gz"
            with patch.object(exporter, "ROOT", root), patch.object(
                sys, "argv", ["export", "--output", str(target)]
            ), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    exporter.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(target.exists())

    def test_real_bundle_contains_only_current_source_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "release.tar.gz"
            with patch.object(sys, "argv", ["export", "--output", str(target)]), redirect_stdout(io.StringIO()):
                exporter.main()
            with tarfile.open(target, "r:gz") as archive:
                members = archive.getmembers()
            names = {m.name for m in members}
            expected = {
                "MINIBENCH-FINAL/" + p.relative_to(ROOT).as_posix()
                for p in exporter.source_files()
            }
            self.assertEqual(names, expected)
            self.assertEqual(len(members), len(expected))
            self.assertIn("MINIBENCH-FINAL/run.sh", names)
            self.assertIn("MINIBENCH-FINAL/CHANGELOG.md", names)
            self.assertTrue(all(m.isfile() for m in members))
            self.assertFalse(any(set(Path(n).parts) & exporter.EXCLUDED_PARTS for n in names))

    def test_published_relative_documentation_links_resolve(self):
        for document in (ROOT / "README.md", ROOT / "docs/cloud-run.md", ROOT / "docs/task-authoring.md"):
            text = document.read_text(encoding="utf-8")
            for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                url = urlsplit(link.strip("<>"))
                if url.scheme or url.netloc or not url.path:
                    continue
                with self.subTest(document=document.name, target=link):
                    self.assertTrue((document.parent / unquote(url.path)).exists(), link)


if __name__ == "__main__":
    unittest.main()

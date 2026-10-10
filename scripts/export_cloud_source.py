"""Export a clean Linux source bundle without Git, credentials or run artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = (".github", "config", "constraints", "data", "docs", "licenses", "src", "scripts", "tests")
FILES = (".env.example", ".gitattributes", ".gitignore", "README.md", "CHANGELOG.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml", "run.sh", "LICENSE")
EXCLUDED_PARTS = {
    ".git", ".venv", ".venv-wsl", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "run", "runs", "output", "outputs",
    "artifacts", "tmp", "temp", "vendor",
}


def validate_source(path):
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"Unexpected external/symlink source: {path.relative_to(ROOT)}")


def source_files():
    for name in FILES:
        path = ROOT / name
        if path.is_file():
            validate_source(path)
            yield path
    for name in DIRECTORIES:
        for path in sorted((ROOT / name).rglob("*")):
            if not path.is_file():
                continue
            parts = path.relative_to(ROOT).parts
            if any(p in EXCLUDED_PARTS or p.endswith(".egg-info") for p in parts):
                continue
            if path.suffix in {".pyc", ".pyo", ".log", ".tmp"} or path.name == ".env" or path.name.startswith(".env."):
                continue
            if parts[:2] == ("config", "cloud") and path.name.startswith("local"):
                continue
            validate_source(path)
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        parser.error("Output already exists; refusing overwrite")
    if any(target.is_relative_to(ROOT / name) for name in DIRECTORIES):
        parser.error("Output must not be inside a submitted source directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    paths = list(source_files())
    with tarfile.open(target, "x:gz") as archive:
        for path in paths:
            archive.add(path, arcname="MINIBENCH-FINAL/" + path.relative_to(ROOT).as_posix(), recursive=False)
    print(f"Exported {len(paths)} source files to {target}; no credentials, runs or local environment included.")


if __name__ == "__main__":
    main()

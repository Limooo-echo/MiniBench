"""Crash-safe primitives used by resumable experiment runners.

The helpers in this module deliberately stay independent of any dataset.  A
runner owns the schema of its manifest and state files; this module only
provides canonical hashing, atomic replacement, and a process-held advisory
lock.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


class RunLockedError(RuntimeError):
    """Raised when another process currently owns an experiment run."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize *value* deterministically for hashing."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def fingerprint_payload(value: Any) -> str:
    """Return the SHA-256 digest of a JSON-compatible value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically replace a UTF-8 text file from a same-directory temporary."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Atomically write a consistently formatted JSON document."""

    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write a JSON Lines snapshot."""

    lines = [json.dumps(dict(row), ensure_ascii=False, sort_keys=True) for row in rows]
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))


def read_json_object(path: str | Path) -> dict[str, Any]:
    """Read a JSON object and reject other top-level shapes."""

    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_jsonl_objects(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and reject malformed or non-object records."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


class RunLock:
    """Non-blocking cross-platform advisory lock held for a runner lifetime."""

    def __init__(
        self,
        directory: str | Path,
        *,
        lock_name: str = ".run.lock",
    ) -> None:
        if (
            not lock_name
            or lock_name in {".", ".."}
            or Path(lock_name).is_absolute()
            or Path(lock_name).name != lock_name
            or "/" in lock_name
            or "\\" in lock_name
        ):
            raise ValueError("lock_name must be a single file name")
        self.path = Path(directory) / lock_name
        self._handle: Any | None = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise RunLockedError(f"Run is already locked: {self.path.parent}") from exc
        self._handle = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _fsync_directory(path: Path) -> None:
    """Best-effort parent fsync on platforms that support directory handles."""

    if os.name == "nt":
        return
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)

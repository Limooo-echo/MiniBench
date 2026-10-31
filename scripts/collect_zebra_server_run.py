"""Collect a completed Zebra server experiment, with checked raw data and hashes.

Uses only the standard library. SSH credentials remain with the SSH client;
neither the identity file nor server environment is read by this program.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile

NAME = "1028_zebra_proexp_direct"
TREATMENTS = ("thinking_off", "thinking_on")
MODES = ("deferred_reasoning", "incremental_state", "incremental_state_pro")
MANIFEST = "collection-manifest.json"
EXPECTED_CALLS = 6468
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
SKIP_PARTS = {"__pycache__", ".venv", ".venv-wsl", "venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SECRET_KEYS = {"authorization", "api_key", "apikey", "access_token", "secret_key", "password"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(value) -> str:
    """Match the frozen minibench.core.checkpoint.fingerprint_payload."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def message_text(value) -> str:
    """Match frozen provider text extraction; do not strip whitespace."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part if isinstance(part, str) else (
            part.get("text") if isinstance(part.get("text"), str) else
            part.get("content") if isinstance(part.get("content"), str) else ""
        ) if isinstance(part, dict) else "" for part in value)
    return ""


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or not path.parts or "\\" in name or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive path: {name!r}")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if (any(character in name for character in ':<>"|?*') or path.as_posix() != name
            or any(part.rstrip(" .") != part or part.split(".")[0].upper() in reserved for part in path.parts)):
        raise ValueError(f"non-portable archive path: {name!r}")
    return name


def reject_credentials(path: str) -> None:
    for part in PurePosixPath(path).parts:
        lower = part.lower()
        if (lower == ".ssh" or lower.startswith(".env") or lower in {"credentials", "credentials.json", "credentials.yaml", "id_rsa", "id_ed25519"}
                or lower.endswith((".pem", ".key")) or lower.startswith(("api_key", "apikey", "secret_key", "access_token"))):
            raise ValueError(f"credential-like file is inside the experiment: {path}")


def reject_secret_fields(value, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in SECRET_KEYS:
                raise ValueError(f"credential field is inside JSON artifact: {path}")
            reject_secret_fields(child, path)
    elif isinstance(value, list):
        for child in value:
            reject_secret_fields(child, path)


def server_status(batch: Path) -> dict:
    result = {"remote_dir": str(batch), "exists": batch.is_dir()}
    for name in ("batch-status.json", "parallel-status.json"):
        path = batch / name
        if path.exists():
            state = read_json(path)
            result[name] = {key: state[key] for key in (
                "status", "updated_at", "completed_total", "planned_total", "completed_by_treatment",
                "workers", "active_jobs", "failures", "error",
            ) if key in state}
    return result


def validate_experiment(batch: Path) -> dict:
    for filename in ("batch-status.json", "parallel-status.json"):
        state = read_json(batch / filename)
        if state.get("status") != "completed" or state.get("failures") or state.get("error"):
            raise ValueError(f"{filename} does not confirm successful completion")
    status = read_json(batch / "parallel-status.json")
    if status.get("completed_total") != 600 or status.get("active_jobs"):
        raise ValueError("expected 600 completed instances and no active jobs")
    frozen = read_json(batch / "batch-manifest.json")
    if frozen.get("data_sha256") != digest(batch / "history_eval.jsonl"):
        raise ValueError("input data differs from the frozen batch manifest")
    tasks = {}
    with (batch / "history_eval.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in tasks:
                raise ValueError("duplicate task in input data")
            tasks[row["id"]] = len(row["clue_turns"]) + 1
    if len(tasks) != 100:
        raise ValueError("expected exactly 100 input puzzles")
    expected = {(task_id, mode) for task_id in tasks for mode in MODES}
    counts, raw_calls = {}, 0
    for treatment in TREATMENTS:
        rows, keys = {}, set()
        with (batch / treatment / "predictions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = row["task_id"], row["memory_mode"]
                if key in keys:
                    raise ValueError(f"duplicate prediction in {treatment}")
                keys.add(key)
                rows[key] = row
        if len(rows) != 300 or keys != expected:
            raise ValueError(f"{treatment}: incomplete or unexpected prediction roster")
        by_mode = Counter(row["memory_mode"] for row in rows.values())
        if any(by_mode[mode] != 100 for mode in MODES):
            raise ValueError(f"{treatment}: mode counts do not equal 100")
        found = set()
        raw_root = batch / "raw_data" / treatment
        for folder in sorted(raw_root.iterdir()):
            if not folder.is_dir():
                continue
            metadata = read_json(folder / "metadata.json")
            key = metadata["task_id"], metadata["mode"]
            if metadata.get("treatment") != treatment or key not in expected or key in found:
                raise ValueError(f"{treatment}: raw job metadata is conflicting or unexpected")
            found.add(key)
            planned = tasks[key[0]]
            if metadata.get("planned_calls") != planned:
                raise ValueError(f"{treatment}: raw job call count does not match puzzle")
            history = rows[key].get("history_trace")
            if not isinstance(history, list) or len(history) != planned:
                raise ValueError(f"{treatment}: prediction history does not cover all raw calls")
            calls = folder / "calls"
            for index in range(planned):
                stem = f"{index:04d}"
                request = read_json(calls / f"{stem}.request.json")
                response = read_json(calls / f"{stem}.response.json")
                record = read_json(calls / f"{stem}.record.json")
                request_bytes = (calls / f"{stem}.request.raw").read_bytes()
                response_bytes = (calls / f"{stem}.response.raw").read_bytes()
                if request != json.loads(request_bytes) or response != json.loads(response_bytes):
                    raise ValueError(f"raw bytes differ from parsed JSON: {treatment}/{folder.name}/{stem}")
                if record.get("status") != "completed":
                    raise ValueError(f"raw call is not completed: {treatment}/{folder.name}/{stem}")
                unsigned = {name: value for name, value in record.items() if name != "record_sha256"}
                if record.get("record_sha256") != fingerprint(unsigned):
                    raise ValueError(f"raw call record checksum mismatch: {treatment}/{folder.name}/{stem}")
                if record.get("endpoint") != ENDPOINT or record.get("request_sha256") != fingerprint({"endpoint": record["endpoint"], "payload": request}):
                    raise ValueError(f"raw request fingerprint/endpoint mismatch: {treatment}/{folder.name}/{stem}")
                if record.get("response_raw_sha256") != hashlib.sha256(response_bytes).hexdigest():
                    raise ValueError(f"raw response fingerprint mismatch: {treatment}/{folder.name}/{stem}")
                if request.get("model") != "qwen3.8-flash" or request.get("enable_thinking") is not (treatment == "thinking_on"):
                    raise ValueError(f"raw request has wrong model/treatment: {treatment}/{folder.name}/{stem}")
                choice = response["choices"][0]
                message = choice["message"]
                content = message_text(message.get("content"))
                reasoning = next((text for name in ("reasoning_content", "reasoning", "reasoning_details")
                                  if (text := message_text(message.get(name))).strip()), None)
                normalized = record["response"]
                if normalized.get("content") != content or normalized.get("reasoning") != reasoning:
                    raise ValueError(f"record text differs from raw response: {treatment}/{folder.name}/{stem}")
                if history[index].get("raw_output") != content:
                    raise ValueError(f"history text differs from raw response: {treatment}/{folder.name}/{stem}")
                if index + 1 == planned and rows[key].get("raw_output") != content:
                    raise ValueError(f"final prediction differs from raw response: {treatment}/{folder.name}/{stem}")
                trace = record.get("completion_trace")
                if not isinstance(trace, dict) or trace.get("content") != message.get("content"):
                    raise ValueError(f"completion trace content mismatch: {treatment}/{folder.name}/{stem}")
                for name in ("reasoning_content", "reasoning", "reasoning_details"):
                    if trace.get(name) != message.get(name):
                        raise ValueError(f"completion trace reasoning mismatch: {treatment}/{folder.name}/{stem}")
                raw_calls += 1
            if len(list(calls.glob("*.record.json"))) != planned:
                raise ValueError(f"unexpected extra raw call records: {treatment}/{folder.name}")
        if found != expected:
            raise ValueError(f"{treatment}: missing raw data for completed instances")
        counts[treatment] = {"instances": len(rows), "by_mode": dict(by_mode)}
    expected_calls = 2 * len(MODES) * sum(tasks.values())
    if raw_calls != expected_calls or raw_calls != EXPECTED_CALLS:
        raise ValueError("raw call count differs from frozen input plan")
    return {"instances": 600, "puzzles": 100, "raw_calls": raw_calls, "by_treatment": counts}


def snapshot_files(batch: Path) -> list[dict]:
    files = []
    for path in sorted(batch.rglob("*")):
        relative = path.relative_to(batch).as_posix()
        if any(part in SKIP_PARTS for part in path.relative_to(batch).parts) or path.suffix in {".pyc", ".pyo"}:
            continue
        reject_credentials(relative)
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in an experiment bundle: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"non-regular experiment file: {relative}")
        if relative == MANIFEST:
            raise ValueError(f"reserved collector filename already exists in source: {MANIFEST}")
        safe_name(relative)
        if path.suffix == ".json":
            reject_secret_fields(read_json(path), relative)
        elif path.suffix in {".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8")
            if re.search(r"(?mi)^\s*(?:api_key|apikey|authorization|access_token|secret_key|password)\s*:", text):
                raise ValueError(f"credential field is inside config artifact: {relative}")
        stat = path.stat()
        files.append({"path": relative, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest(path)})
    return files


def create_archive(batch: Path) -> dict:
    if not batch.is_absolute() or not batch.is_dir():
        raise ValueError("remote batch must be an existing absolute directory")
    audit = validate_experiment(batch)
    files = snapshot_files(batch)
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    artifact_dir = batch.parent / ".minibench-artifacts"
    artifact_dir.mkdir(mode=0o700, exist_ok=True)
    archive = artifact_dir / f"{batch.name}.tar.gz"
    receipt_path = artifact_dir / f"{batch.name}.archive.json"
    if archive.exists() and receipt_path.exists():
        previous = read_json(receipt_path)
        if previous.get("remote_dir") != str(batch):
            raise ValueError("archive path belongs to a different experiment")
        if previous.get("source_fingerprint") == fingerprint and previous.get("archive_sha256") == digest(archive):
            return previous
    elif archive.exists() or receipt_path.exists():
        raise ValueError("unrecognized existing remote archive/receipt; refusing to overwrite")
    manifest = {"schema_version": 1, "remote_dir": str(batch), "name": batch.name,
                "collected_at": datetime.now(timezone.utc).isoformat(), "audit": audit,
                "files": [{key: value for key, value in row.items() if key != "mtime_ns"} for row in files]}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    partial = archive.with_name(archive.name + ".partial")
    with partial.open("wb") as output, gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as bundle:
            for row in files:
                bundle.add(batch / row["path"], arcname=row["path"], recursive=False)
            info = tarfile.TarInfo(MANIFEST)
            info.size, info.mode = len(manifest_bytes), 0o644
            bundle.addfile(info, io.BytesIO(manifest_bytes))
    # Detect a final log append or unexpected writer instead of publishing a mixed snapshot.
    if snapshot_files(batch) != files:
        raise ValueError("experiment files changed while archiving; retry after the runner exits")
    os.replace(partial, archive)
    receipt = {"schema_version": 1, "remote_dir": str(batch), "archive_path": str(archive),
               "archive_size": archive.stat().st_size, "archive_sha256": digest(archive),
               "source_fingerprint": fingerprint, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "file_count": len(files), "audit": audit}
    temporary = receipt_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, receipt_path)
    return receipt


def ssh_base(host: str, identity: str) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+", host):
        raise ValueError("invalid SSH destination")
    return ["ssh", "-i", identity, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host]


def remote_action(host: str, identity: str, action: str, remote_dir: str) -> dict:
    command = shlex.join(["python3", "-", "_remote", action, remote_dir])
    result = subprocess.run(ssh_base(host, identity) + [command], input=Path(__file__).read_bytes(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        try:
            detail = json.loads(result.stdout).get("error", "")
        except (ValueError, AttributeError):
            detail = ""
        detail = detail or result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"SSH {action} failed (exit {result.returncode}): {detail}")
    value = json.loads(result.stdout)
    if value.get("error"):
        raise RuntimeError(value["error"])
    return value


def verify_archive(path: Path, expected_sha256: str | None = None, expected_manifest_sha256: str | None = None) -> dict:
    if expected_sha256 and digest(path) != expected_sha256:
        raise ValueError("downloaded archive SHA-256 does not match server")
    with tarfile.open(path, "r:gz") as bundle:
        members = {}
        portable_names = set()
        for member in bundle.getmembers():
            safe_name(member.name)
            reject_credentials(member.name)
            if not member.isfile() or member.name.casefold() in portable_names:
                raise ValueError(f"non-regular or duplicate archive member: {member.name}")
            members[member.name] = member
            portable_names.add(member.name.casefold())
        if MANIFEST not in members:
            raise ValueError("archive has no collection manifest")
        manifest_bytes = bundle.extractfile(members[MANIFEST]).read()
        if expected_manifest_sha256 and hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
            raise ValueError("collection manifest SHA-256 does not match server receipt")
        manifest = json.loads(manifest_bytes)
        files = {safe_name(row["path"]): row for row in manifest["files"]}
        if len(files) != len(manifest["files"]) or set(members) != set(files) | {MANIFEST}:
            raise ValueError("archive contents do not match manifest file list")
        for name, row in files.items():
            member = members[name]
            if member.size != row["size"]:
                raise ValueError(f"archive size mismatch: {name}")
            value = hashlib.sha256()
            stream = bundle.extractfile(member)
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
            if value.hexdigest() != row["sha256"]:
                raise ValueError(f"archive file hash mismatch: {name}")
    return manifest


def extract_checked(archive: Path, destination: Path, manifest: dict) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("local destination may not be a symlink")
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            safe_name(member.name)
            if not member.isfile():
                raise ValueError(f"non-regular archive member: {member.name}")
            target = root.joinpath(*PurePosixPath(member.name).parts)
            for parent in (target, *target.parents):
                if parent == root.parent:
                    break
                if parent.is_symlink():
                    raise ValueError(f"local extraction path contains a symlink: {member.name}")
            if not target.resolve().is_relative_to(root):
                raise ValueError("local extraction escaped the target directory")
            source = bundle.extractfile(member)
            if target.exists():
                if not target.is_file() or hashlib.sha256(source.read()).hexdigest() != digest(target):
                    raise ValueError(f"local file conflicts with download: {member.name}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also avoids silently replacing a concurrent local edit.
            with target.open("xb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
    for row in manifest["files"]:
        path = root.joinpath(*PurePosixPath(row["path"]).parts)
        if path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
            raise ValueError(f"extracted file failed verification: {row['path']}")


def download(args) -> dict:
    receipt = remote_action(args.host, args.identity, "archive", args.remote_dir)
    local_dir = args.local_dir.resolve()
    artifact_dir = local_dir.parent / f"{local_dir.name}_download"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    archive = artifact_dir / f"{local_dir.name}.tar.gz"
    partial = archive.with_name(archive.name + ".partial")
    if not archive.exists() or digest(archive) != receipt["archive_sha256"]:
        if partial.is_symlink() or partial.is_dir():
            raise ValueError("download partial path is not a regular file")
        with partial.open("wb") as output:
            result = subprocess.run(ssh_base(args.host, args.identity) + [shlex.join(["cat", receipt["archive_path"]])],
                                    stdout=output, stderr=subprocess.PIPE, check=False)
        if result.returncode:
            raise RuntimeError(f"SSH download failed (exit {result.returncode}): {result.stderr.decode(errors='replace').strip()}")
        if partial.stat().st_size != receipt["archive_size"] or digest(partial) != receipt["archive_sha256"]:
            raise ValueError("partial download size/hash does not match server")
        os.replace(partial, archive)
    manifest = verify_archive(archive, receipt["archive_sha256"], receipt["manifest_sha256"])
    if manifest["audit"] != receipt["audit"] or len(manifest["files"]) != receipt["file_count"]:
        raise ValueError("archive manifest differs from server transfer receipt")
    extract_checked(archive, local_dir, manifest)
    local_audit = validate_experiment(local_dir)
    if local_audit != receipt["audit"]:
        raise ValueError("local experiment audit differs from server")
    result = {**receipt, "local_dir": str(local_dir), "local_archive": str(archive),
              "downloaded_at": datetime.now(timezone.utc).isoformat(), "status": "verified"}
    target = artifact_dir / "download-receipt.json"
    temporary = target.with_suffix(".json.partial")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--status", action="store_true")
    action.add_argument("--download", action="store_true")
    parser.add_argument("--host", default="ubuntu@115.159.156.154")
    parser.add_argument("--identity", default="C:/Users/Limo/.ssh/minibench_tencent_ed25519")
    parser.add_argument("--remote-dir", default=f"/home/ubuntu/{NAME}")
    parser.add_argument("--local-dir", type=Path, default=Path(__file__).resolve().parents[1] / "runs" / NAME)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        result = remote_action(args.host, args.identity, "status", args.remote_dir) if args.status else download(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, tarfile.TarError) as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "_remote":
        try:
            batch = Path(sys.argv[3])
            result = server_status(batch) if sys.argv[2] == "status" else create_archive(batch)
            print(json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
            raise SystemExit(1)
    else:
        raise SystemExit(main())

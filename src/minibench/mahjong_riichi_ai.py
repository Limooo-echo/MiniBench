from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shlex
import subprocess
import threading
import time
from typing import Any, Sequence


class MahjongAIError(RuntimeError):
    pass


@dataclass(frozen=True)
class MahjongAIResponse:
    raw_output: str
    action: dict[str, object]


class ExternalMahjongAI:
    """JSON bridge for a stronger Riichi Mahjong opponent process.

    The process receives one JSON request and must return one JSON action using
    the same action schema as the benchmark agent.
    """

    def __init__(
        self,
        command: str | Sequence[str],
        *,
        mode: str = "stdio",
        timeout: float = 30.0,
        cwd: str | Path | None = None,
    ):
        if mode not in {"stdio", "oneshot"}:
            raise ValueError("mode must be stdio or oneshot")
        self.command = _normalize_command(command)
        self.mode = mode
        self.timeout = timeout
        self.cwd = str(cwd) if cwd is not None else None
        self._process: subprocess.Popen[str] | None = None
        self._lines: Queue[str] = Queue()
        self._reader: threading.Thread | None = None

    def choose(self, request: dict[str, object]) -> MahjongAIResponse:
        if self.mode == "oneshot":
            return self._choose_oneshot(request)
        return self._choose_stdio(request)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        stdin = process.stdin
        stdout = process.stdout
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        for stream in (stdin, stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self._process = None

    def _choose_oneshot(self, request: dict[str, object]) -> MahjongAIResponse:
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(request, ensure_ascii=False) + "\n",
                text=True,
                capture_output=True,
                cwd=self.cwd,
                timeout=self.timeout,
                check=False,
            )
        except OSError as exc:
            raise MahjongAIError(f"failed to start external Mahjong AI: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise MahjongAIError("external Mahjong AI timed out") from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            detail = f": {stderr}" if stderr else ""
            raise MahjongAIError(
                f"external Mahjong AI exited with code {completed.returncode}{detail}"
            )
        return _parse_response(completed.stdout)

    def _choose_stdio(self, request: dict[str, object]) -> MahjongAIResponse:
        process = self._ensure_running()
        if process.stdin is None:
            raise MahjongAIError("external Mahjong AI stdin is unavailable")
        try:
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
        except OSError as exc:
            raise MahjongAIError(f"failed to write to external Mahjong AI: {exc}") from exc

        collected: list[str] = []
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = "\n".join(collected[-3:])
                suffix = f": {detail}" if detail else ""
                raise MahjongAIError(f"external Mahjong AI timed out{suffix}")
            try:
                line = self._lines.get(timeout=remaining)
            except Empty as exc:
                detail = "\n".join(collected[-3:])
                suffix = f": {detail}" if detail else ""
                raise MahjongAIError(f"external Mahjong AI timed out{suffix}") from exc
            collected.append(line)
            try:
                return _parse_response("\n".join(collected))
            except MahjongAIError:
                continue

    def _ensure_running(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        try:
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.cwd,
                bufsize=1,
            )
        except OSError as exc:
            raise MahjongAIError(f"failed to start external Mahjong AI: {exc}") from exc
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        return self._process

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            if line.strip():
                self._lines.put(line.rstrip("\n"))


def make_external_mahjong_ai(
    command: str | Sequence[str] | None,
    *,
    mode: str = "stdio",
    timeout: float = 30.0,
) -> ExternalMahjongAI:
    resolved = command or os.environ.get("MAHJONG_AI_COMMAND")
    if not resolved:
        raise ValueError(
            "external Riichi opponent requires --mahjong-ai-command or MAHJONG_AI_COMMAND"
        )
    return ExternalMahjongAI(resolved, mode=mode, timeout=timeout)


def _normalize_command(command: str | Sequence[str]) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command, posix=os.name != "nt")
    return [str(part) for part in command]


def _parse_response(raw_output: str) -> MahjongAIResponse:
    raw = raw_output.strip()
    if not raw:
        raise MahjongAIError("external Mahjong AI returned empty output")
    payload: Any | None = None
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            raw = line
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(payload, dict):
        raise MahjongAIError(f"external Mahjong AI returned non-JSON output: {raw_output}")
    return MahjongAIResponse(raw_output=raw, action=payload)

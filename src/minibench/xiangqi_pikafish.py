from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Sequence


FILES = "abcdefghi"

PIECE_ID_TO_FEN = {
    1: "K",
    2: "A",
    3: "A",
    4: "B",
    5: "B",
    6: "N",
    7: "N",
    8: "R",
    9: "R",
    10: "C",
    11: "C",
    12: "P",
    13: "P",
    14: "P",
    15: "P",
    16: "P",
}


class PikafishError(RuntimeError):
    pass


@dataclass(frozen=True)
class PikafishChoice:
    fen: str
    uci_move: str
    action: int
    info_lines: tuple[str, ...]


def board_to_pikafish_fen(
    board: Sequence[Sequence[int]],
    *,
    side_to_move: str = "ally",
) -> str:
    if side_to_move not in {"ally", "enemy"}:
        raise ValueError("side_to_move must be ally or enemy")

    if len(board) != 10 or any(len(row) != 9 for row in board):
        raise ValueError("xiangqi board must be 10x9")

    fen_rows: list[str] = []
    for row in board:
        empty_count = 0
        parts: list[str] = []

        for value in row:
            piece_id = int(value)
            if piece_id == 0:
                empty_count += 1
                continue

            if empty_count:
                parts.append(str(empty_count))
                empty_count = 0

            symbol = PIECE_ID_TO_FEN.get(abs(piece_id))
            if symbol is None:
                raise ValueError(f"invalid piece id: {piece_id}")

            parts.append(symbol if piece_id > 0 else symbol.lower())

        if empty_count:
            parts.append(str(empty_count))

        fen_rows.append("".join(parts))

    active_color = "w" if side_to_move == "ally" else "b"
    return f"{'/'.join(fen_rows)} {active_color} - - 0 1"


def square_to_uci(row: int, col: int) -> str:
    if row < 0 or row > 9 or col < 0 or col > 8:
        raise ValueError(f"invalid xiangqi square: {(row, col)}")

    return f"{FILES[col]}{9 - row}"


def uci_to_square(square: str) -> tuple[int, int]:
    if len(square) != 2:
        raise ValueError(f"invalid UCI square: {square}")

    file_name = square[0]
    rank_name = square[1]

    if file_name not in FILES or not rank_name.isdigit():
        raise ValueError(f"invalid UCI square: {square}")

    rank = int(rank_name)
    if rank < 0 or rank > 9:
        raise ValueError(f"invalid UCI square: {square}")

    return 9 - rank, FILES.index(file_name)


def action_to_uci(action: int) -> str:
    from gym_xiangqi.utils import action_space_to_move

    _piece_id, start, end = action_space_to_move(action)
    start_row, start_col = int(start[0]), int(start[1])
    end_row, end_col = int(end[0]), int(end[1])
    return f"{square_to_uci(start_row, start_col)}{square_to_uci(end_row, end_col)}"


def uci_to_action(env, uci_move: str) -> int:
    from gym_xiangqi.utils import action_space_to_move

    from minibench.xiangqi_env import legal_actions

    if len(uci_move) < 4:
        raise ValueError(f"invalid UCI move: {uci_move}")

    start = uci_to_square(uci_move[:2])
    end = uci_to_square(uci_move[2:4])

    for action in legal_actions(env):
        _piece_id, action_start, action_end = action_space_to_move(action)
        candidate_start = (int(action_start[0]), int(action_start[1]))
        candidate_end = (int(action_end[0]), int(action_end[1]))

        if candidate_start == start and candidate_end == end:
            return int(action)

    raise ValueError(f"Pikafish move {uci_move} is not legal in the gym-xiangqi state")


def find_pikafish_executable(start_dir: str | Path | None = None) -> Path | None:
    env_path = os.environ.get("PIKAFISH_PATH")
    if env_path:
        return Path(env_path)

    path_hit = shutil.which("pikafish") or shutil.which("pikafish.exe")
    if path_hit:
        return Path(path_hit)

    root = Path(start_dir).resolve() if start_dir else Path.cwd().resolve()
    names = ["pikafish.exe", "pikafish"] if os.name == "nt" else ["pikafish", "pikafish.exe"]
    candidate_dirs = [
        root,
        root / "src",
        root / "pikafish",
        root / "pikafish" / "src",
        root / "pikafish" / "Pikafish-master" / "src",
        root.parent / "pikafish",
        root.parent / "pikafish" / "src",
        root.parent / "pikafish" / "Pikafish-master" / "src",
    ]
    candidates = [
        candidate_dir / name
        for candidate_dir in candidate_dirs
        for name in names
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def resolve_pikafish_executable(
    executable: str | Path | None = None,
    *,
    start_dir: str | Path | None = None,
) -> Path:
    if executable is not None:
        return Path(executable)

    found = find_pikafish_executable(start_dir)
    if found is None:
        raise ValueError(
            "Pikafish executable was not found. Pass --pikafish-path or set "
            "PIKAFISH_PATH to the compiled engine binary."
        )

    return found


class PikafishEngine:
    def __init__(
        self,
        executable: str | Path,
        *,
        timeout: float = 30.0,
        cwd: str | Path | None = None,
        eval_file: str | Path | None = None,
    ):
        executable_path = Path(executable)
        self.executable = str(executable_path)
        self.timeout = timeout
        self.cwd = str(cwd) if cwd is not None else str(executable_path.parent)
        self._executable_path = executable_path
        self._eval_file = self._resolve_eval_file(executable_path, eval_file)
        self._process: subprocess.Popen[str] | None = None
        self._lines: Queue[str] = Queue()
        self._reader: threading.Thread | None = None

    def __enter__(self) -> PikafishEngine:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        try:
            self._process = subprocess.Popen(
                [self.executable],
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise PikafishError(f"failed to start Pikafish: {exc}") from exc

        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

        self._send("uci")
        self._read_until(lambda line: line.strip() == "uciok", "uciok")
        if self._eval_file is not None:
            eval_file = self._path_for_engine(self._eval_file)
            self._send(f"setoption name EvalFile value {eval_file}")
        self.ready()

    def ready(self) -> None:
        self._send("isready")
        self._read_until(lambda line: line.strip() == "readyok", "readyok")

    def choose(
        self,
        env,
        *,
        side_to_move: str,
        depth: int | None = 8,
        movetime_ms: int | None = None,
    ) -> PikafishChoice:
        fen = board_to_pikafish_fen(env.state, side_to_move=side_to_move)
        uci_move, info_lines = self.bestmove_for_fen(
            fen,
            depth=depth,
            movetime_ms=movetime_ms,
        )
        action = uci_to_action(env, uci_move)

        return PikafishChoice(
            fen=fen,
            uci_move=uci_move,
            action=action,
            info_lines=info_lines,
        )

    def bestmove_for_fen(
        self,
        fen: str,
        *,
        depth: int | None = 8,
        movetime_ms: int | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        if movetime_ms is not None and movetime_ms <= 0:
            raise ValueError("movetime_ms must be positive")
        if depth is not None and depth <= 0:
            raise ValueError("depth must be positive")

        self._send(f"position fen {fen}")
        if movetime_ms is not None:
            self._send(f"go movetime {movetime_ms}")
        else:
            self._send(f"go depth {depth or 8}")

        lines = self._read_until(
            lambda line: line.strip().startswith("bestmove "),
            "bestmove",
            timeout=self.timeout,
        )
        bestmove_line = next(
            line.strip() for line in reversed(lines) if line.strip().startswith("bestmove ")
        )
        parts = bestmove_line.split()
        if len(parts) < 2 or parts[1] in {"0000", "(none)"}:
            raise PikafishError(f"Pikafish returned no move: {bestmove_line}")

        return parts[1], tuple(line for line in lines if line.startswith("info "))

    def close(self) -> None:
        process = self._process
        if process is None:
            return

        if process.poll() is None:
            try:
                self._send("quit")
                process.wait(timeout=2)
            except (OSError, PikafishError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()

        self._process = None

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return

        for line in process.stdout:
            self._lines.put(line.rstrip("\n"))

    def _send(self, command: str) -> None:
        process = self._ensure_running()
        if process.stdin is None:
            raise PikafishError("Pikafish stdin is not available")

        process.stdin.write(f"{command}\n")
        process.stdin.flush()

    def _ensure_running(self) -> subprocess.Popen[str]:
        process = self._process
        if process is None:
            raise PikafishError("Pikafish has not been started")
        if process.poll() is not None:
            raise PikafishError(f"Pikafish exited with code {process.returncode}")
        return process

    def _read_until(
        self,
        predicate,
        description: str,
        *,
        timeout: float | None = None,
    ) -> list[str]:
        deadline = time.monotonic() + (timeout or self.timeout)
        lines: list[str] = []

        while time.monotonic() < deadline:
            process = self._process
            if process is not None and process.poll() is not None and self._lines.empty():
                tail = "\n".join(lines[-12:])
                detail = f":\n{tail}" if tail else ""
                raise PikafishError(
                    f"Pikafish exited while waiting for {description}{detail}"
                )

            try:
                line = self._lines.get(timeout=0.1)
            except Empty:
                continue

            lines.append(line)
            if predicate(line):
                return lines

        raise PikafishError(f"timed out waiting for {description}")

    @staticmethod
    def _resolve_eval_file(
        executable: Path,
        eval_file: str | Path | None,
    ) -> Path | None:
        if eval_file is not None:
            return Path(eval_file)

        env_eval_file = os.environ.get("PIKAFISH_EVAL_FILE")
        if env_eval_file:
            return Path(env_eval_file)

        default_eval_file = executable.parent / "pikafish.nnue"
        if default_eval_file.exists():
            return default_eval_file

        return None

    def _path_for_engine(self, path: Path) -> str:
        try:
            if path.resolve().parent == self._executable_path.resolve().parent:
                return path.name
        except OSError:
            pass

        raw_path = str(path)

        # When WSL launches a Windows .exe, UCI file options must be Windows
        # paths even though Python sees the same file through /mnt/<drive>/...
        if self._executable_path.suffix.lower() == ".exe":
            normalized = raw_path.replace("\\", "/")
            match = re.match(r"^/mnt/([a-zA-Z])/(.+)$", normalized)
            if match:
                drive = match.group(1).upper()
                tail = match.group(2).replace("/", "\\")
                return f"{drive}:\\{tail}"

        return raw_path

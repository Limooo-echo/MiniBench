from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from cchess import BLACK, ChessBoard
from cchess.common import fench_to_text


TRANSLATION = str.maketrans(
    {
        "車": "车",
        "馬": "马",
        "進": "进",
        "帥": "帅",
        "將": "将",
        "後": "后",
        "炮": "炮",
    }
)
RED_FILES = ("九", "八", "七", "六", "五", "四", "三", "二", "一")
BLACK_FILES = ("１", "２", "３", "４", "５", "６", "７", "８", "９")
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def source_kind(path: Path) -> str:
    text = path.as_posix()
    if "殺局" in text:
        return "kill-tactic"
    if "全盤戰術" in text:
        return "fullgame-tactics"
    if "殘局" in text:
        return "endgame"
    return "unknown"


def source_style_text(board: ChessBoard, start: tuple[int, int], move_text: str) -> str:
    piece = fench_to_text(board.get_fench(start))
    files = BLACK_FILES if board.move_player == BLACK else RED_FILES
    return piece + files[start[0]] + move_text[-2:]


def play_text(board: ChessBoard, token: str):
    move = board.move_text(token)
    if move is not None:
        return move
    matches = []
    for start, end in board.create_moves():
        trial = board.copy()
        candidate = trial.move(start, end)
        if candidate is None:
            continue
        text = candidate.to_text()
        if text == token or source_style_text(board, start, text) == token:
            matches.append((start, end))
    if len(matches) != 1:
        return None
    return board.move(*matches[0])


def decode_pgn(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("big5", "utf-8"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise UnicodeDecodeError("ccpd", raw, 0, 1, "unsupported text encoding")


def replay(path: Path) -> tuple[list[dict[str, object]], str]:
    text = decode_pgn(path)
    fen_match = re.search(r'\[FEN\s+"([^"]+)"\]', text)
    result_match = re.search(r'\[Result\s+"([^"]+)"\]', text)
    if fen_match is None:
        raise ValueError("missing FEN")
    result = result_match.group(1) if result_match else "*"
    board = ChessBoard(fen_match.group(1))
    positions: list[dict[str, object]] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*\d+\.\s*", "", line.strip())
        if not line or line.startswith("["):
            continue
        for raw_token in line.split():
            if raw_token in RESULTS:
                continue
            token = raw_token.translate(TRANSLATION)
            positions.append(
                {
                    "fen": board.to_full_fen(),
                    "ply_index": len(positions),
                    "side_to_move": "black" if board.move_player == BLACK else "red",
                    "played_move_text": token,
                }
            )
            move = play_text(board, token)
            if move is None:
                raise ValueError(
                    f"unparsed move at ply {len(positions)}: {token}; "
                    f"fen={board.to_full_fen()}"
                )
            positions[-1]["played_move_uci"] = move.to_iccs()
            board.next_turn()
    return positions, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-plies", type=int, default=30)
    parser.add_argument("--max-games", type=int)
    args = parser.parse_args()

    files = sorted(args.source.rglob("*.pgn"))
    if args.max_games is not None:
        files = files[: args.max_games]
    failed: list[dict[str, str]] = []
    emitted = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, path in enumerate(files, start=1):
            try:
                positions, result = replay(path)
            except Exception as exc:
                failed.append({"source_file": path.relative_to(args.source).as_posix(), "error": str(exc)})
                continue
            winner = "red" if result == "1-0" else "black" if result == "0-1" else None
            for item in positions[-args.tail_plies :]:
                if winner is not None and item["side_to_move"] != winner:
                    continue
                payload = {
                    **item,
                    "source_file": path.relative_to(args.source).as_posix(),
                    "source_kind": source_kind(path),
                    "game_result": result,
                    "game_plies": len(positions),
                    "plies_from_end": len(positions) - int(item["ply_index"]),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                emitted += 1
            if index % 100 == 0:
                print(
                    f"games={index}/{len(files)} emitted={emitted} failed={len(failed)}",
                    file=sys.stderr,
                    flush=True,
                )
    failures_path = args.output.with_suffix(".failures.json")
    failures_path.write_text(json.dumps(failed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"games": len(files), "emitted": emitted, "failed": len(failed)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""验证接口: 对 jsonl 题集逐题复核 (mate 步数/唯一性/推演将杀).

mode: exact=精确mate; win=可赢即可(推演将杀)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from minibench.datasets.xiangqi.engines.pikafish import (
    PikafishEngine,
    board_to_pikafish_fen,
    resolve_pikafish_executable,
)
from minibench.datasets.xiangqi.variants.board import Move, VariantBoard


def _mate_analysis(engine, board, depth=14):
    fen = board_to_pikafish_fen(board, side_to_move="ally")
    try:
        engine._send("ucinewgame")
        engine._send("setoption name MultiPV value 2")
        uci, info = engine.bestmove_for_fen(fen, depth=depth)
    except Exception:
        return None, None, None
    mates = {}
    for line in info:
        if "multipv" in line and "score mate" in line:
            parts = line.split()
            try:
                mates[int(parts[parts.index("multipv") + 1])] = \
                    int(parts[parts.index("mate") + 1])
            except ValueError:
                pass
    return mates.get(1), uci, mates.get(2)


def _get_best(engine, board, side, depth=14):
    side_str = "ally" if side == 1 else "enemy"
    fen = board_to_pikafish_fen(board, side_to_move=side_str)
    for _ in range(3):
        try:
            engine._send("ucinewgame")
            engine._send("setoption name MultiPV value 1")
            uci, _ = engine.bestmove_for_fen(fen, depth=depth)
            if uci:
                return uci
        except Exception:
            pass
    return None


def _checkmate_line(engine, board, max_red=20):
    vb = VariantBoard(board, [])
    red = 0
    for ply in range(30):
        side = 1 if ply % 2 == 0 else -1
        legal = vb.legal_moves(side)
        if not legal:
            if side == -1:
                return red, vb._is_in_check(-1)
            return red, False
        u = _get_best(engine, vb.board, side)
        if u is None:
            return red, False
        sc = ord(u[0]) - ord("a"); sr = 9 - int(u[1])
        tc = ord(u[2]) - ord("a"); tr = 9 - int(u[3])
        vb.apply(Move(sr, sc, tr, tc))
        if side == 1:
            red += 1
        if red > max_red:
            return red, False
    return red, False


def verify_tasks(path: str | Path, mode: str = "win") -> tuple[int, int, list]:
    """验证题集. 返回 (通过数, 总数, 失败列表)."""
    with open(path, encoding="utf-8") as f:
        tasks = [json.loads(l) for l in f if l.strip()]

    engine = PikafishEngine(
        resolve_pikafish_executable("/home/zyh/Pikafish/src/pikafish"),
        timeout=30.0,
    )
    engine.start()

    ok, fails = 0, []
    for i, t in enumerate(tasks):
        board = t["board"]
        exp = t.get("mate_steps", 1)
        if mode == "exact":
            om, ou, sec = _mate_analysis(engine, board)
        else:
            om, ou, sec = None, None, None
        red, won = _checkmate_line(engine, board)
        if mode == "exact":
            good = (om == exp) and (sec is None or sec > exp) and won
        else:
            good = won
        if good:
            ok += 1
        else:
            fails.append((t.get("id"), exp, om, sec, won))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tasks)} ... 通过 {ok}", flush=True)

    engine.close()
    return ok, len(tasks), fails


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="jsonl 题集路径")
    ap.add_argument("mode", nargs="?", default="win", choices=["exact", "win"])
    args = ap.parse_args()
    ok, total, fails = verify_tasks(args.path, args.mode)
    print(f"\n通过: {ok}/{total}")
    if fails:
        print("失败 (前10):")
        for f in fails[:10]:
            print(f"  {f}")
    else:
        print("全部通过!")

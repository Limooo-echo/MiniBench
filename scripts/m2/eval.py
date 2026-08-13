"""多模态评测入口: 文字 / 汉字图 / 符号图 三组.

每步给模型当前棋局:
  text   — 紧凑 ASCII 棋盘字符串 (board_to_compact)
  img_cn — 汉字棋子渲染图 (每步 render_board)
  img_ab — 字母代号渲染图 (每步 render_board)
指标: 0.3合法 + 0.4每步最优 + 0.3success; 困毙算胜; depth-N 对手.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scripts.common.loader import load_tasks
from scripts.m2.render import PIECE_NAME, board_to_compact, render_board
from minibench.datasets.xiangqi.variants.board import Move, VariantBoard
from minibench.datasets.xiangqi.variants.search import score_moves

API_KEY = os.environ.get("M2_API_KEY", "")
URL = os.environ.get("M2_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
MODEL = os.environ.get("M2_MODEL", "qwen3.8-max")
THINKING = os.environ.get("M2_THINKING", "disabled")
OPP_DEPTH = int(os.environ.get("M2_OPP_DEPTH", "4"))
OPT_DEPTH = 3
MAX_STEPS = 20
STEP_DIR = Path(__file__).resolve().parent.parent.parent / "vis_outputs" / f"m2_steps_d{OPP_DEPTH}"


def _call(messages, max_tokens=60):
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
    if THINKING:
        payload["thinking"] = {"type": THINKING}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={
        "Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"__ERR__{str(e)[:80]}"


def _extract_idx(raw: str) -> int | None:
    if not raw or raw.startswith("__"):
        return None
    try:
        obj = json.loads(raw)
        for v in obj.values():
            if isinstance(v, int):
                return v
            if isinstance(v, str) and v.isdigit():
                return int(v)
    except (json.JSONDecodeError, AttributeError):
        pass
    if raw.strip().isdigit():
        return int(raw.strip())
    nums = re.findall(r"(?<!\d)(\d{1,2})(?!\d)", raw)
    return int(nums[-1]) if nums else None


def _fmt_move(mv: Move, board: VariantBoard) -> str:
    pid = board.board[mv.fr][mv.fc]
    return f"{PIECE_NAME.get(abs(pid), '?')} {mv.to_uci()}"


def _build_prompt(task, legal_moves, board, mode, history_text):
    action_lines = "\n".join(f"{i+1}: {_fmt_move(mv, board)}" for i, mv in enumerate(legal_moves))
    goal = ("红方先走。请在下列合法着法中选择最佳一步（目标是击败黑方）。"
            "在当前状况下，请选择一步可以将军的棋，并尽可能实现必杀，或者有助于后续实现必杀；而不是只让它将军。")
    board_part = f"当前棋盘 (.空, 大写红/小写黑):\n{board_to_compact(board.board)}" if mode == "text" else "(见当前局面图片)"
    return f"""你是中国象棋残局玩家。
{goal}
{board_part}

历史走法:
{history_text or '(开局)'}

当前合法着法列表:
{action_lines}

【严格要求】只输出一个阿拉伯数字（你选择的编号），禁止输出任何其他文字、标点、解释或格式。例如：3"""


def _run_game(task, agent_fn, mode):
    board = VariantBoard(task["board"], [])
    steps, reasons, history, success = [], [], [], False
    step_dir = STEP_DIR / task["id"]
    step_dir.mkdir(parents=True, exist_ok=True)
    for step_idx in range(MAX_STEPS):
        side = 1 if step_idx % 2 == 0 else -1
        legal = board.legal_moves(side)
        if not legal:
            if side == -1:
                success = True
                reasons.append("agent_checkmated_opponent" if board._is_in_check(-1) else "agent_stalemated_opponent")
            else:
                reasons.append("agent_has_no_moves")
            break
        if side == 1:
            prompt = _build_prompt(task, legal, board, mode, "\n".join(history))
            if mode == "text":
                raw = agent_fn(prompt, None, None)
            else:
                b64 = render_board(board.board, mode)
                (step_dir / f"{mode}_step{step_idx:02d}.png").write_bytes(base64.b64decode(b64))
                raw = agent_fn(prompt, b64, None)
            idx = _extract_idx(raw)
            if idx is None or not (1 <= idx <= len(legal)):
                if mode != "text":
                    idx = _extract_idx(agent_fn(prompt, b64, None))
            mv = legal[idx - 1] if (idx is not None and 1 <= idx <= len(legal)) else None
            scored = score_moves(board, 1, OPT_DEPTH, 1)
            best_uci = scored[0][0].to_uci() if scored else None
            is_opt = bool(mv and best_uci and mv.to_uci() == best_uci)
            if mv is None:
                steps.append({"step": step_idx, "actor": "agent", "uci": "PARSE_FAIL",
                              "raw": (raw or "")[:80], "is_legal": False, "is_opt": False})
                reasons.append("illegal_or_parse_fail")
                break
            steps.append({"step": step_idx, "actor": "agent", "uci": mv.to_uci(),
                          "raw": (raw or "")[:80], "is_legal": True, "is_opt": is_opt,
                          "best_uci": best_uci or ""})
            history.append(f"step {step_idx//2+1}: You {mv.to_uci()}")
            board.apply(mv)
            if board.find_general(-1) is None:
                success = True
                reasons.append("agent_captured_general")
                break
        else:
            scored = score_moves(board, -1, OPP_DEPTH, 1)
            if not scored:
                success = True
                reasons.append("agent_checkmated_opponent" if board._is_in_check(-1) else "agent_stalemated_opponent")
                break
            mv = scored[0][0]
            steps.append({"step": step_idx, "actor": "opp", "uci": mv.to_uci(), "is_legal": True, "is_opt": False})
            history.append(f"step {step_idx//2+1}: Opp {mv.to_uci()}")
            board.apply(mv)
            if board.find_general(1) is None:
                reasons.append("agent_lost_general")
                break
    if not reasons:
        reasons.append("max_steps_reached")
    return steps, success, reasons


def run(
    agent_name: str = "openai-compatible",
    *,
    sample: int | None = None,
    modes: list[str] | None = None,
    **_opts,
) -> list:
    """跑多模态评测 (文字/汉字图/符号图).

    agent_name: 仅 openai-compatible (multimodal 调用)
    sample: None=全量250; 42=sample_42.jsonl
    modes: ["text","img_cn","img_ab"] 子集
    """
    if not API_KEY:
        raise RuntimeError("M2_API_KEY 未设置")
    modes = modes or ["text", "img_cn", "img_ab"]
    tasks = load_tasks("m2", sample=sample)
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[M2] model={MODEL} tasks={len(tasks)} modes={modes} opp_depth={OPP_DEPTH}")

    def agent_fn(prompt, img_b64, _name):
        if img_b64:
            return _call([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ]}])
        return _call([{"role": "user", "content": prompt}])

    results = []
    for t in tasks:
        for mode in modes:
            steps, success, reasons = _run_game(t, agent_fn, mode)
            agent_steps = [s for s in steps if s["actor"] == "agent"]
            n = len(agent_steps)
            legal_rate = sum(1 for s in agent_steps if s["is_legal"]) / n if n else 0.0
            opt_rate = sum(1 for s in agent_steps if s["is_opt"]) / n if n else 0.0
            score = 0.3 * legal_rate + 0.4 * opt_rate + 0.3 * (1.0 if success else 0.0)
            print(f"  [{mode:6s}] {t['id']:20s} success={success} score={score:.2f}", flush=True)
            results.append({"task_id": t["id"], "mode": mode, "success": success,
                            "legality_rate": round(legal_rate, 3), "opt_rate": round(opt_rate, 3),
                            "score": round(score, 2), "reasons": reasons, "steps": steps})
    # 保存结果到 runs/
    from pathlib import Path as _P
    out = _P(__file__).resolve().parent.parent.parent / "runs" / f"m2_{agent_name}_s{sample or 'full'}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 汇总
    from collections import defaultdict
    by_mode = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print(f"[M2] 结果保存: {out}")
    for m, rs in by_mode.items():
        n = len(rs)
        print(f"  {m}: success={sum(r['success'] for r in rs)}/{n} "
              f"({sum(r['success'] for r in rs)/n:.0%}) legal={sum(r['legality_rate'] for r in rs)/n:.0%} "
              f"score={sum(r['score'] for r in rs)/n:.2f}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="多模态评测 (文字/汉字图/符号图)")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--modes", default="text,img_cn,img_ab")
    args = ap.parse_args()
    run(sample=args.sample, modes=[m.strip() for m in args.modes.split(",")])

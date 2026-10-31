"""Display saved Zebra runner status without modifying the experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time

BEIJING = timezone(timedelta(hours=8))
MODES = {
    "deferred_reasoning": "延后推理",
    "incremental_state": "结构化增量",
    "incremental_state_pro": "结构化增量 Pro",
}


def render(batch: Path) -> tuple[str, str]:
    status = json.loads((batch / "parallel-status.json").read_text(encoding="utf-8"))
    manifest = json.loads((batch / "parallel-resume-manifest.json").read_text(encoding="utf-8"))
    now = datetime.now(BEIJING)
    updated = datetime.fromisoformat(status["updated_at"]).astimezone(BEIJING)
    age = max(0, (now - updated).total_seconds())
    counts = status["completed_by_treatment"]
    initial = manifest["initial_completed"]
    active = status.get("active_jobs", [])
    failures = status.get("failures", [])
    lines = [
        "Zebra 实时进度（只读，每 3 秒刷新；关闭此窗口不会停止实验）",
        f"北京时间：{now:%Y-%m-%d %H:%M:%S}    状态：{status['status']}",
        f"Thinking ON ：{counts['thinking_on']}/300    本次续跑新增：{counts['thinking_on'] - initial['thinking_on']}",
        f"Thinking OFF：{counts['thinking_off']}/300",
        f"当前并发：{len(active)} / {status.get('workers', 4)}",
        f"运行状态更新：{updated:%H:%M:%S}（{age:.0f} 秒前）",
        "",
        "正在执行的实例：",
    ]
    for index, job in enumerate(active, 1):
        lines.append(f"  {index}. {job['task_id']}  |  {MODES.get(job['mode'], job['mode'])}  |  {job['treatment']}")
    if not active:
        lines.append("  暂无")
    lines.extend(["", f"已记录错误：{len(failures)}"])
    lines.extend("  " + json.dumps(item, ensure_ascii=False) for item in failures)
    if age > 30 and status["status"] == "running":
        lines.append("提示：状态超过 30 秒未刷新，请检查实验运行终端。")
    lines.extend([
        "",
        "完成数按“题目 × 模式”统计，整题所有线索和最终作答结束后才增加。",
        "当前运行器不保存线索级实时进度；心跳刷新只表示协调器仍在更新状态。",
    ])
    return "\n".join(lines), status["status"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        while True:
            try:
                output, state = render(args.batch_dir)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                output = f"暂时无法读取进度：{exc}\n将在 3 秒后重试；实验进程不受影响。"
                state = "read_error"
            if sys.stdout.isatty() and not args.once:
                print("\033[2J\033[H", end="")
            print(output, flush=True)
            if args.once or state in {"completed", "interrupted"}:
                return 1 if state == "read_error" else 0
            time.sleep(3)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic local reports for offline scoring."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Any

from minibench.scoring import SCORING_VERSION
from minibench.scoring.manifest import load_suite, load_weights, object_hash


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("[redacted]" if str(k).lower() in {"api_key", "access_token", "authorization", "password", "secret"}
                    else _safe(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _number(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _costs(rows: list[dict], *, unverified: bool = False) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["profile"], row["architecture"]].append(row)
    result = []
    for (profile, arch), items in sorted(groups.items()):
        out = {"profile": profile, "architecture": arch, "planned_records": len(items),
               "provenance": "unverified" if unverified else "verified"}
        def measured(row: dict) -> dict:
            if (row.get("status") == "unverified") != unverified:
                return {}
            metrics = dict(row.get("metrics") or {})
            if (metrics.get("usage_available") is False and metrics.get("llm_calls") == 0
                    and metrics.get("model_elapsed_seconds") == 0
                    and metrics.get("recording_complete") is not True):
                metrics.pop("llm_calls", None)
                metrics.pop("model_elapsed_seconds", None)
            return metrics
        for metric in ("task_elapsed_seconds", "model_elapsed_seconds", "llm_calls"):
            values = [measured(r).get(metric) for r in items]
            observed = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0]
            out[metric] = {"observed_total": sum(observed) if observed else None,
                           "observed_records": len(observed), "coverage": len(observed) / len(items)}
        usable, tokens, known_calls, missing_calls = 0, 0, 0, 0
        for row in items:
            metrics = measured(row)
            value = (metrics.get("token_usage") or {}).get("total_tokens")
            # Older writers emit zero-filled usage even when it is unavailable.
            if metrics.get("usage_available") is True and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                tokens += value
                usable += 1
            calls, missing = metrics.get("llm_calls"), metrics.get("usage_missing_calls")
            if isinstance(calls, int) and isinstance(missing, int) and 0 <= missing <= calls:
                known_calls += calls
                missing_calls += missing
        out["total_tokens"] = {"observed_total": tokens if usable else None,
                                "observed_records": usable, "coverage": usable / len(items),
                                "calls_with_usage": known_calls - missing_calls,
                                "calls_with_usage_status": known_calls,
                                "call_coverage": (known_calls - missing_calls) / known_calls if known_calls else None}
        result.append(out)
    return result


def _diagnostics(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["profile"], row["architecture"], row["task"], row["dimension"], row["mode"]].append(row)
    result = []
    for key, items in sorted(groups.items()):
        reasons = Counter(reason for row in items for reason in row.get("reasons", []))
        per_item = defaultdict(lambda: defaultdict(list))
        for row in items:
            for name, value in (row.get("diagnostics") or {}).items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    per_item[row["item_id"]][name].append(float(value))
        metrics = defaultdict(list)
        for values in per_item.values():
            for name, item_values in values.items():
                metrics[name].append(sum(item_values) / len(item_values))
        result.append({"profile": key[0], "architecture": key[1], "task": key[2], "dimension": key[3], "mode": key[4],
                       "planned_records": len(items), "statuses": dict(sorted(Counter(r["status"] for r in items).items())),
                       "reasons": dict(sorted(reasons.items())),
                       "metrics": {name: {"mean_per_item": sum(v) / len(v), "observed_items": len(v)} for name, v in sorted(metrics.items())}})
    return result


def _control_lines(exclusions: list[dict]) -> list[str]:
    groups = defaultdict(list)
    for entry in exclusions:
        if "diagnostic_result" in entry:
            groups[entry['profile'], entry['architecture'], entry['experiment'], str(entry['mode']), entry['provenance_verified']].append(entry)
    lines = ["", "## 未纳入主分的记录", "", "这里只展示已保存回答的结果，含对照条件和旧题目；不会加入四维主分，也不会把不同题目的分数差当作能力损失。", "",
             "| 模型组 / 架构 | 实验 | 记录模式 | 可判定原题成功率 | 有结果的题目 / 记录题目 | 来源已核实 |",
             "|---|---|---|---:|---:|---|"]
    for key, entries in sorted(groups.items()):
        values = defaultdict(list)
        ids = {e['task_id'] for e in entries}
        for entry in entries:
            y = entry['diagnostic_result'].get('y')
            if y is not None:
                values[entry['task_id']].append(y)
        means = [sum(v)/len(v) for v in values.values()]
        average = 100*sum(means)/len(means) if means else None
        lines.append(f"| {_cell(key[0])} / {_cell(key[1])} | {_cell(key[2])} | {_cell(key[3])} | {_number(average)}% | {len(means)}/{len(ids)} | {'是' if key[4] else '否'} |")
    if not groups:
        lines += ["", "没有可单独展示的排除记录。"]
    lines += [""]
    return lines


def _sensitivity_lines(result: dict) -> list[str]:
    lines = ["", "| 模型组 / 架构 | 维度 | 所有可计算方案的最低分 | 最高分 | 可计算方案数 |",
             "|---|---|---:|---:|---:|"]
    for profile, data in result["profiles"].items():
        for arch, dimensions in data["sensitivity"]["ranges"].items():
            for dim, values in dimensions.items():
                lines.append(f"| {_cell(profile)} / {_cell(arch)} | {dim} | {_number(values['min'])} | {_number(values['max'])} | {values['available_scenarios']}/{values['total_scenarios']} |")
    lines += ["", "| 模型组 | 架构比较（左减右） | 维度 | 主分差 | 配对 95% 区间 | 权重变化后顺序反转 |", "|---|---|---|---:|---|---|"]
    count = 0
    for profile, data in result["profiles"].items():
        flips = {(p['left'], p['right'], p['dimension']): p for p in data['sensitivity']['pairwise']}
        for comparison in data.get('paired_comparisons', []):
            count += 1
            key = (comparison['left'], comparison['right'], comparison['dimension'])
            ci = comparison.get('ci95')
            interval = '—' if ci is None else f"[{_number(ci[0])}, {_number(ci[1])}]"
            flip = flips[key].get('rank_reversal')
            label = '无法判断' if flip is None else ('是' if flip else '否')
            lines.append(f"| {_cell(profile)} | {_cell(key[0])} − {_cell(key[1])} | {key[2]} | {_number(comparison.get('difference'))} | {interval} | {label} |")
    if not count:
        lines += ["", "当前每个模型组不足两个架构，不作架构顺序比较。"]
    lines += ["", "详细变更方案和顺序变化（含并列变化）见 profiles.<模型组>.sensitivity；区间描述题目抽样变化，不代表运行稳定性。", ""]
    return lines


def _markdown(result: dict) -> str:
    lines = ["# MiniBench 四维离线评分", "", f"实验清单：`{result['suite_id']}`；评分版本：`{SCORING_VERSION}`。", "",
             "本报告只读取已有题目和结果，没有调用模型。维度依次为直接解题 D、规则变化 R、历史信息 H、图像输入 V。", "",
             r"单题公式：$s=Y+(1-Y)\kappa P$；先逐题计算，再按重复运行、题目、难度、题型、模式和任务逐层汇总。", "",
             f"部分分上限 κ = {result['scoring_config']['partial_credit_cap']}；严格得分仅使用 Y。", "",
             "覆盖不足时分数为 —，不补零或重新分配权重。历史和图像维度不代表独立测得了记忆或识图能力。", "",
             "## 四维得分", "", "| 模型组 | 架构 | D | R | H | V |", "|---|---|---:|---:|---:|---:|"]
    for profile, data in result["profiles"].items():
        for arch, scores in data["architectures"].items():
            lines.append("| " + " | ".join([_cell(profile), _cell(arch), *[_number(x) for x in scores["vector"]]]) + " |")
    lines += ["", "## 严格得分与覆盖", "", "| 模型组 / 架构 | 维度 | 严格得分 | 主分覆盖 | 严格覆盖 | 主分 95% 区间 |", "|---|---|---:|---:|---:|---|"]
    for profile, data in result["profiles"].items():
        for arch, scores in data["architectures"].items():
            for dim in "DRHV":
                s = scores["dimensions"][dim]
                ci = s.get("ci95")
                interval = "—" if ci is None else f"[{_number(ci[0])}, {_number(ci[1])}]"
                lines.append(f"| {_cell(profile)} / {_cell(arch)} | {dim} | {_number(s.get('strict_score'))} | {_number(100*s['coverage'])}% | {_number(100*s['strict_coverage'])}% | {interval} |")
    lines += ["", "区间按同源原题一起抽样；估计不足时保留空值。具体原因见 scores.json 的 uncertainty/dimensions。", "",
              "## 任务得分", "", "| 模型组 / 架构 | 任务 | 维度 | 得分（0–1） | 覆盖 |", "|---|---|---|---:|---:|"]
    for profile, data in result["profiles"].items():
        for arch, scores in data["architectures"].items():
            for s in scores.get("task_scores", []):
                if s.get("weight", 0) == 0:
                    continue
                lines.append(f"| {_cell(profile)} / {_cell(arch)} | {_cell(s.get('task'))} | {s.get('dimension')} | {_number(s.get('score'), 4)} | {_number(100*s.get('coverage', 0))}% |")
    lines += ["", "## 缺失与诊断", "", "诊断量不额外加入主分。unverified_y / unverified_p 仅说明旧记录可读取，来源未核实，不能作为正式比较。", ""]
    for diag in result["diagnostics"]:
        if diag["reasons"] or diag["metrics"]:
            lines.append(f"- {_cell(diag['profile'])} / {_cell(diag['architecture'])} / {diag['task']} / {diag['dimension']} / {diag['mode']}: "
                         + _cell(json.dumps({"statuses": diag["statuses"], "reasons": diag["reasons"], "metrics": diag["metrics"]}, ensure_ascii=False, sort_keys=True)))
    lines += ["", *[f"- {_cell(w)}" for w in result["warnings"]], "",
              f"另有 {len(result['exclusions'])} 项不属于固定题目或模式（含标准规则对照），逐项原因保存在 scores.json。", "",
              "## 权重变化检查", "", "主权重、参与任务等权、每个正权重分别上下调整 20% 后归一化，与部分分上限 0、0.25、0.50 组合计算。", "",
              "分数范围与架构顺序变化保存在 scores.json 的 profiles.<模型组>.sensitivity。不跨模型组排名，也不使用雷达图面积排名。", "",
              "## 已记录成本", "", "| 模型组 / 架构 | 已记录 token | 调用用量覆盖 | 已记录调用次数 | 已记录任务时间（秒） |", "|---|---:|---:|---:|---:|"]
    control_position = lines.index("## 权重变化检查")
    lines[control_position:control_position] = _control_lines(result["exclusions"])
    cost_position = lines.index("## 已记录成本")
    lines[cost_position:cost_position] = _sensitivity_lines(result)
    for cost in result["costs"]:
        lines.append(f"| {_cell(cost['profile'])} / {_cell(cost['architecture'])} | {_number(cost['total_tokens']['observed_total'], 0)} | {_number(None if cost['total_tokens']['call_coverage'] is None else 100*cost['total_tokens']['call_coverage'])}% | {_number(cost['llm_calls']['observed_total'], 0)} | {_number(cost['task_elapsed_seconds']['observed_total'])} |")
    lines += ["", "这里只累计来源已核实的记录；未核实成本单独保存在 unverified_costs。调用用量覆盖按实际记录的调用次数计算，未知用量不补零，多分支记录不完整时不能声称成本完整。", "",
              "## 来源与复算", "", f"- 清单 SHA-256：`{result['manifest_sha256']}`", f"- 评分输入指纹：`{result['input_fingerprint']}`",
              "- 原题及结果路径、SHA-256、来源核对信息保存在 scores.json / item_scores.jsonl。",
              "- task_scores.csv 保存逐题权重和贡献，item_scores.jsonl 另含计算证据和诊断。",
              "- verified 表示清单维护者依据旧配置或记录确认来源，并非根据当前配置推断历史配置。", ""]
    if not result.get("radars"):
        lines += ["没有四维均完整的架构，本次不绘制完整雷达多边形。", ""]
    else:
        lines += [f"![{_cell(r['profile'])} 四维雷达图]({r['png']})" for r in result["radars"]]
    return "\n".join(lines)


def _radars(result: dict, output: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["svg.hashsalt"] = "minibench-scoring-v1"
    from matplotlib import font_manager
    fonts = sorted((Path(__file__).resolve().parents[1] / "assets" / "fonts").glob("*.otf"))
    if fonts:
        font_manager.fontManager.addfont(str(fonts[0]))
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(fonts[0])).get_name()
    artifacts = []
    for index, (profile, data) in enumerate(result["profiles"].items()):
        complete = {a: s for a, s in data["architectures"].items() if all(v is not None for v in s["vector"])}
        if not complete:
            continue
        theta = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        theta = np.append(theta, theta[0])
        legend_rows = math.ceil(len(complete) / 2)
        fig, ax = plt.subplots(figsize=(7, 7 + .25 * legend_rows), subplot_kw={"projection": "polar"})
        fig.subplots_adjust(left=.15, right=.85, top=.82, bottom=.16 + .022 * legend_rows)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(theta[:-1], ["D: Direct", "R: Rules", "H: History", "V: Visual"])
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        for arch, s in complete.items():
            values = [*s["vector"], s["vector"][0]]
            ax.plot(theta, values, marker="o", label=arch)
            ax.fill(theta, values, alpha=.06)
        ax.tick_params(axis="x", pad=12)
        ax.set_title(textwrap.fill(profile, width=48), pad=35, fontsize=11)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2, fontsize=9, frameon=False)
        png, svg = f"radar-{index+1}.png", f"radar-{index+1}.svg"
        fig.savefig(output / png, dpi=160, metadata={"Software": "MiniBench offline scoring"})
        fig.savefig(output / svg, metadata={"Date": None, "Creator": "MiniBench offline scoring"})
        plt.close(fig)
        artifacts.append({"profile": profile, "png": png, "svg": svg})
    return artifacts


def score_suite(manifest_path: Path, output: Path, weights_path: Path | None = None) -> dict:
    from minibench.scoring.aggregate import aggregate_suite

    suite = load_suite(manifest_path)
    config = load_weights(weights_path)
    output = output.resolve()
    protected = [*suite["protected_paths"], *([weights_path.resolve()] if weights_path else [])]
    # Input files remain read-only; output must not be an ancestor or inside any input directory.
    for source in protected:
        if output == source or output == source.parent or source.parent in output.parents or output in source.parents:
            raise ValueError(f"Output must be separate from input directories: {source}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; existing artifacts are never overwritten")
    result = aggregate_suite(suite["rows"], config["weights"], partial_credit_cap=config["partial_credit_cap"],
                             bootstrap_samples=config["bootstrap_samples"], seed=config["seed"])
    manifest = suite["manifest"]
    result.update(suite_id=manifest["suite_id"], scoring_version=SCORING_VERSION, scoring_config=config,
                  manifest_sha256=suite["manifest_sha256"], sources=suite["sources"], exclusions=suite["exclusions"],
                  warnings=suite["warnings"], experiment_manifest=_safe(manifest),
                  costs=_costs(suite["rows"]), unverified_costs=_costs(suite["rows"], unverified=True),
                  diagnostics=_diagnostics(suite["rows"]))
    result["input_fingerprint"] = object_hash({"manifest_sha256": suite["manifest_sha256"], "config": config,
                                                "sources": suite["sources"], "version": SCORING_VERSION})
    result = _safe(result)
    json.dumps(result, allow_nan=False)
    output.mkdir(parents=True, exist_ok=True)
    result["radars"] = _radars(result, output)
    item_scores = result.pop("item_scores", [])
    (output / "item_scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n" for r in item_scores), encoding="utf-8")
    fields = ["profile", "architecture", "task", "dimension", "mode", "group", "difficulty", "item_id", "source_id", "repeat_id",
              "y", "p", "score", "status", "within_task_weight", "axis_weight", "contribution", "reasons"]
    with (output / "task_scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in item_scores:
            writer.writerow(dict(row, reasons=";".join(row.get("reasons", []))))
    (output / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(result), encoding="utf-8")
    return result

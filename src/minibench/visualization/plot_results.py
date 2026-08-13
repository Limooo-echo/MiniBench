# src/minibench/visualization/plot_results.py
import os
import json
import matplotlib.pyplot as plt

# 设置字体，防止中文和负号乱码
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 🎯 核心配置：全量 Agent 名称映射字典
# 包含了你所有的 Agent 策略，将自动把英文名转换为漂亮的图表标签
# ==========================================
AGENT_DISPLAY_NAMES = {
    # 核心策略
    "direct": "Direct (直接回答)",
    "cot": "CoT (思维链)",
    "self-consistency": "Self-Consistency (自洽性)",
    "self_consistency": "Self-Consistency (自洽性)",
    "tot": "ToT (思维树)",
    "tree_of_thought": "ToT (思维树)",
    "plan-then-solve": "Plan-then-Solve (计划求解)",
    "plan_then_solve": "Plan-then-Solve (计划求解)",
    "critic-refine": "Critic-Refine (反思修正)",
    "critic_refine": "Critic-Refine (反思修正)",
    "simple": "Simple (基础策略)",
    
    # 评测基线
    "openai-compatible": "OpenAI-Compatible (基线)",
    "openai_compatible": "OpenAI-Compatible (基线)",
    "oracle": "Oracle (黄金答案)",
    "noisy": "Noisy (噪声提取)"
}

# 兼容早期手动测试的文件夹别名
LEGACY_FOLDER_MAP = {
    "siliconflow-glm52-mcq-test2": "Direct (直接回答)",
    # "siliconflow-glm52-mcq-test5": "ToT (思维树)" # 如果有其他旧文件夹，写在这里
}

def plot_accuracy_comparison(runs_dir, save_path):
    """
    自动遍历 runs_dir 下的所有结果，并生成带有所有 Agent 名称的准确率对比图
    """
    if not os.path.exists(runs_dir):
        print(f"❌ 找不到结果目录: {runs_dir}")
        return

    experiment_names = []
    success_rates = []

    print(f"🔍 正在扫描目录: {runs_dir}")
    
    # 遍历 runs 目录下的所有子文件夹
    for folder_name in sorted(os.listdir(runs_dir)):
        folder_path = os.path.join(runs_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue

        result_path = os.path.join(folder_path, "results.json")
        
        if os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                rate = data.get("success_rate", 0.0)
                
                # --- 智能解析 Agent 显示名称 ---
                display_name = folder_name # 默认使用原文件夹名
                
                # 1. 优先检查是否是旧版本手动测试的文件夹
                if folder_name in LEGACY_FOLDER_MAP:
                    display_name = LEGACY_FOLDER_MAP[folder_name]
                else:
                    # 2. 自动匹配规范命名的文件夹 (如: glm52-mcq-critic-refine)
                    # 倒序遍历匹配，防止包含连字符的 agent 被截断
                    for agent_key, pretty_name in AGENT_DISPLAY_NAMES.items():
                        if folder_name.endswith(f"-{agent_key}") or folder_name.endswith(f"_{agent_key}"):
                            display_name = pretty_name
                            break
                
                experiment_names.append(display_name)
                success_rates.append(rate)
                print(f"  - 找到数据: [{display_name}] 成功率 = {rate}")

    if not experiment_names:
        print("❌ 在提供的目录下没有找到任何包含 results.json 的有效测试数据！")
        return

    # ==========================================
    # 开始动态绘图
    # ==========================================
    # 根据 agent 的数量动态调整画布宽度，防止柱子挤在一起
    fig_width = max(10, len(experiment_names) * 1.5)
    plt.figure(figsize=(fig_width, 6))
    
    # 定义一套高级调色盘，超出数量自动循环
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#8C8C8C', '#E377C2', '#17BECF']
    bar_colors = [colors[i % len(colors)] for i in range(len(experiment_names))]
    
    bars = plt.bar(experiment_names, success_rates, color=bar_colors, width=0.55)
    
    plt.title('不同 Agent 策略下的测试成功率对比', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Agent 推理策略', fontsize=13, labelpad=10)
    plt.ylabel('成功率 (Success Rate)', fontsize=13, labelpad=10)
    
    plt.ylim(0, 1.05) 
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 在柱子上标注具体数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 自动倾斜 X 轴标签防止重叠
    plt.xticks(rotation=20 if len(experiment_names) > 4 else 0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 结果对比图已成功生成并保存至: {save_path}")

def plot_task_results(runs_dir, save_path):
    """4 任务 (D3/C2/H2/M2) 结果对比图: 扫描 runs 下结果目录, 解析 summary.txt / m2 jsonl.

    指标: success 率 (将杀/获胜) + 平均 score.
    """
    import glob
    import re

    if not os.path.exists(runs_dir):
        print(f"找不到结果目录: {runs_dir}")
        return

    task_prefix = {"d3": "d3_", "c2": "c2_", "h2": "h2_", "m2": "m2_"}
    task_success = {k: [] for k in task_prefix}
    task_score = {k: [] for k in task_prefix}

    # summary.txt (d3/c2/h2)
    for summary in glob.glob(os.path.join(runs_dir, "**", "summary.txt"), recursive=True):
        d = os.path.basename(os.path.dirname(summary))
        task = next((t for t, p in task_prefix.items() if d.startswith(p)), None)
        if task is None:
            continue
        text = open(summary, encoding="utf-8").read()
        m_score = re.search(r"Overall score:\s*([0-9.]+)", text) or re.search(r"score=\s*([0-9.]+)", text)
        m_succ = re.search(r"Success rate:\s*([0-9.]+)%", text) or re.search(r"success=\s*([0-9.]+)%", text)
        if m_score:
            task_score[task].append(float(m_score.group(1)))
        if m_succ:
            task_success[task].append(float(m_succ.group(1)) / 100.0)

    # m2 jsonl (success + score)
    for j in glob.glob(os.path.join(runs_dir, "**", "m2_*.jsonl"), recursive=True):
        succ, scores = 0.0, []
        n = 0
        for line in open(j, encoding="utf-8"):
            r = json.loads(line)
            n += 1
            succ += 1.0 if r.get("success") else 0.0
            scores.append(r.get("score", 0))
        if n:
            task_success["m2"].append(succ / n)
            task_score["m2"].append(sum(scores) / n)

    names = [t.upper() for t in task_prefix]
    succ_avg = [sum(v) / len(v) if v else 0 for v in task_success.values()]
    score_avg = [sum(v) / len(v) if v else 0 for v in task_score.values()]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = range(len(names))
    bars = ax1.bar(x, score_avg, 0.5, label="平均 Score", color="#4C72B0")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 100)
    ax2 = ax1.twinx()
    ax2.plot(list(x), [s * 100 for s in succ_avg], "o-", color="#DD8452",
             label="Success 率 (%)", linewidth=2)
    ax2.set_ylabel("Success 率 (%)")
    ax2.set_ylim(0, 100)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names)
    for bar, v in zip(bars, score_avg):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}",
                 ha="center", fontweight="bold")
    fig.legend(loc="upper right")
    plt.title("四任务评测结果对比 (Score + Success)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"四任务结果对比图已生成: {save_path}")


# --- 独立运行入口 ---
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    runs_dir = os.path.join(project_root, "runs")
    save_path = os.path.join(project_root, "outputs", "visualizations", "accuracy_comparison.png")
    
    plot_accuracy_comparison(runs_dir, save_path)
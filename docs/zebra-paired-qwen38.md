# Zebra 配对复测（2026-10-27）

本次固定 100 个源题，分别评测 direct、temporary_codebook、counterfactual_semantics，合计 300 个实例。三组均使用 **Direct agent、Qwen3.8-Flash、prompt v2、温度 0、7168 思考 + 1024 最终答案、4 workers**。没有自动格式修复或网络重试。

## 修复边界

历史云端两阶段收尾指令包含象棋的“棋盘／合法走法”。新的 `src/minibench/datasets/zebra/qwen_client.py` 只供本次 Zebra 入口使用，收尾阶段要求完整逻辑网格、准确的 `House N` 和属性键、遵守题内语义覆盖。原任务 prompt、通用 DirectAgent、其他任务及历史快照均不修改。

第一阶段仍使用现有 Zebra prompt 与 Direct v2 包装；第二阶段使用 Zebra 专用指令。仍然是 Direct 架构和原生 thinking，不是 CoT agent。

## 题目与可解性

使用原有已认证的 100 个反事实源题及其顺序，重建原规则 direct 和可逆 codebook。相比旧 direct 集保留 69 题，补入 31 题。三组均为 34 easy / 33 medium / 33 hard；题面线索在展开 codebook 后一致，direct/codebook 的答案相同，counterfactual 的答案在新语义下改变。

构建和每次启动都对全部 300 个实例重新调用 Z3，不使用答案作为求解约束：第一次必须 SAT；排除完整解后第二次必须 UNSAT；独立整数约束检查通过；求解结果必须匹配保存的 gold。无解、多解、超时、解析失败或答案不符均阻止启动。证明、题目哈希和源题顺序保存在 `data/zebra/paired-20261027/`。

同源题配对消除了换题因素，但反事实改变约束，仍不保证求解难度相同。

## WSL 运行

在 Ubuntu WSL 的项目目录运行：

```bash
cd /mnt/d/AAALimoWork/CS/Seminar/MiniBench
bash scripts/run_zebra_paired_wsl.sh
```

终端会私密提示输入 DashScope API key。输入不显示、不进入命令历史、不写入配置或结果，只用于该进程。不要把 key 发到聊天中。

只做离线验证、不调用 API：

```bash
.venv-wsl/bin/python scripts/run_zebra_paired.py --preflight
```

冻结集已存在时不要重建或覆盖；重新核验：

```bash
.venv-wsl/bin/python scripts/build_zebra_paired.py --verify-only
```

如需在新环境生成数据（目标目录必须不存在）：

```bash
.venv-wsl/bin/python scripts/build_zebra_paired.py
```

## 结果与恢复

配置：`config/experiments/zebra_paired_qwen38.yaml`。

输出：`runs/zebra-paired-qwen38-20261027/`。

- `preflight.json`：本次启动的独立可解性核验。
- `manifest.json`：配置、数据、代码和任务列表指纹。
- `status.json`、`summary.json`：进度、各条件成绩和配对比较。
- `per-task-results.jsonl`：逐题分数、原始输出、结构诊断和 trace。
- `jobs/<key>/calls/`：每个阶段的请求正文、响应、思考、用量、结束原因及中断证据；不含请求头或密钥。

三种条件按源题交错运行，并限制总请求频率为每分钟 30 次。正常完成需要 600 次物理调用。基础设施错误停止派发新题，已在运行的题完成后退出；用相同启动命令恢复，已完成阶段从检查点读取，避免重复付费。未完成的网络调用可能已有服务端用量，恢复时可能重发，因此保留其中断证据。

数据、配置或代码变更后拒绝复用原运行目录。模型输出无效 JSON 仍记录为模型失败，不能当作网络故障重试。主指标沿用原评分；额外报告完整 schema、可解析但零格正确，方便区分格式问题与求解问题。

运行期间请保持电脑和 WSL 开启；退出终端会中断前台进程。原实验数据与结果保留在原位置。

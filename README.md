# MiniBench 0.2.0

MiniBench 是一个用统一 YAML、统一 agent/provider 接口和统一结果格式评测推理模型的小型基准。当前包含 Zebra 逻辑题、象棋、一笔画与麻将。

本 README 以 **WSL 2 + Ubuntu 22.04 + Python 3.10** 为标准环境。进入 Ubuntu 后，下面所有安装、配置、运行和排错命令都在 WSL 终端执行。

> 先记住两条：文本任务默认使用 DeepSeek V4；带图片的任务必须使用支持视觉输入的模型，仓库默认使用 Qwen。不要用 DeepSeek 跑 `image`、`chinese-piece-image` 或 `latin-piece-image`。

## FINAL 云服务器入口

本版融合了 `mahjong-checkpoint-20261010` 的非象棋任务与本地新版象棋；原评分系统保留。
云端支持不同题目 8/16 路并发、单局顺序交互、账号独立限额、逐调用保存、自动跳过已完成题与柱状对比图。
先安装环境/固定 Pikafish、临时设置 key，再运行固定 direct 小样本：

```bash
bash scripts/cloud_run.sh config/cloud/xiangqi-pilot.yaml runs/cloud-pilot-8 --workers 8
```

完整安装、暂停续跑、全 Agent 配置、用量边界与文件职责见 [云端运行说明](docs/cloud-run.md)。
四个象棋数据库均为固定分层抽出的 100 题；所有 Agent 使用同一名单。C2 的 100 题已逐题认证能在运行上限内击败登记的固定 depth-3 对手，这不等于对所有防守的最短强制杀证明。顶级 LICENSE 尚待团队确定。

## 离线四维评分

已有实验结果可以离线汇总为直接解题、规则变化、历史信息、图像输入四个分数，不会重新调用模型。

```bash
minibench score-suite --manifest config/scoring/manifest.example.yaml --weights config/scoring/weights.yaml --output outputs/my-scoring-report
```

示例清单固定了原题文件及 SHA-256，`runs` 留空，运行后会如实报告缺失覆盖。填入已保存的逐题结果及其原始配置来源后再发布正式比较；不要用当前配置推断旧实验。输出目录须为空或尚不存在。

评分版本 2.0.0 使用 `s = Y + (1 - Y) × P`，过程分不放缩，每题每条件运行一次，维度内任务等权。连续麻将按给定牌墙逐步计算最早和牌动作集，过程分为命中步数 / 总动作步数；麻将图像题与文字题使用相同目标和过程分标准。旧缩放或人工权重配置需要更新。

完整公式、等权说明、逐题评分与清单字段见 [评分设计说明](docs/scoring-design.md)，实施和验收记录见 [评分实施记录](docs/scoring-plan.md)。

## 1. 在 WSL Ubuntu 中安装

如果 WSL 尚未安装，先在 Windows 管理员 PowerShell 中执行一次 `wsl --install -d Ubuntu-22.04` 并重启。之后打开 Ubuntu，确认仓库的 Windows 路径映射正确：

```bash
cd /mnt/d/AAALimoWork/CS/Seminar/MiniBench
pwd
```

安装 Python 和编译工具：

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip build-essential git curl

python3.10 -m venv ~/.venvs/minibench
source ~/.venvs/minibench/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -c constraints/ci-py310.txt -e ".[xiangqi-generation]"
```

完整测试需要 `xiangqi-generation` 中固定版本的独立象棋裁判 `cchess`；只安装基础包不足以运行全套生成器与数据验证测试。

验证安装：

```bash
minibench --help
python -m unittest discover -s tests
```

以后每次重新打开 WSL，只需：

```bash
cd /mnt/d/AAALimoWork/CS/Seminar/MiniBench
source ~/.venvs/minibench/bin/activate
```

`constraints/ci-py310.txt` 固定了 CI 使用的图片与数据依赖。用它安装最容易复现 GitHub Actions；项目本身仍在 `pyproject.toml` 中保留较宽的用户依赖范围。

## 2. 配置外部 API

MiniBench 通过 OpenAI-compatible `POST /chat/completions` 接口调用模型。API key 只从环境变量读取，不要写入 YAML、README 或 Git。

### 2.1 DeepSeek V4 Flash：文本任务

DeepSeek 官方公共 API 的模型 ID 是 `deepseek-v4-flash`，Base URL 是 `https://api.deepseek.com`。先在当前 WSL 会话中设置 key：

```bash
read -rsp "DeepSeek API key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
```

用一个很小的请求检查 key、余额和网络：

```bash
curl -sS https://api.deepseek.com/chat/completions \
  -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Return OK."}],
    "thinking": {"type": "disabled"},
    "max_tokens": 16,
    "stream": false
  }'
```

DeepSeek YAML 的 provider 写法：

```yaml
provider:
  name: deepseek
  model: deepseek-v4-flash
  api_key_env: DEEPSEEK_API_KEY
  json_mode: true
  max_tokens: 1024
  timeout: 120
  extra_body:
    thinking:
      type: disabled
```

`thinking.type: disabled` 表示只使用 MiniBench 选择的 agent 推理架构，避免再叠加 provider 原生思考。若要专门测试 DeepSeek 原生 thinking，可改成 `enabled`，同时应重新评估超时、token 上限和实验可比性。

DeepSeek 官方资料：[V4 发布说明](https://api-docs.deepseek.com/news/news260424/)、[Chat Completions 参数](https://api-docs.deepseek.com/api/create-chat-completion)。

### 2.2 Qwen/DashScope：多模态任务

获取 Model Studio API key 后，在 WSL 中设置：

```bash
read -rsp "DashScope API key: " DASHSCOPE_API_KEY
echo
export DASHSCOPE_API_KEY
```

API key 与 endpoint 必须属于同一区域。仓库提供三个别名：

| YAML `provider.name` | 区域 | 项目默认模型 | Base URL |
| --- | --- | --- | --- |
| `qwen` | 中国北京 | `qwen3.8-max` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `qwen-intl` | 新加坡 | `qwen3.8-max` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| `qwen-us` | 美国弗吉尼亚 | `qwen3.8-max` | `https://dashscope-us.aliyuncs.com/compatible-mode/v1` |

先用北京 endpoint 做文本预检；其他区域替换 URL 即可：

```bash
curl -sS https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-max",
    "messages": [{"role": "user", "content": "Return OK."}],
    "enable_thinking": false,
    "max_tokens": 16,
    "stream": false
  }'
```

MiniBench 的正式视觉 YAML 统一使用 `qwen3.8-max`。这是项目锁定的视觉基准模型 ID；运行前必须在 Model Studio 控制台确认该 ID 已向你的账号和区域开放。若尚未开放，请把本地副本的 `provider.model` 改成控制台显示的视觉模型精确 ID，不要凭简称猜模型名。

Qwen 视觉 YAML 的标准写法：

```yaml
provider:
  name: qwen
  model: qwen3.8-max
  api_key_env: DASHSCOPE_API_KEY
  json_mode: true
  max_tokens: 1024
  timeout: 120
  extra_body:
    enable_thinking: false
```

MiniBench 会把本地图片编码成 OpenAI-compatible `image_url` 数据 URL；不需要把题图上传到公网。相关官方资料：[获取 API key](https://www.alibabacloud.com/help/en/model-studio/get-api-key)、[区域 Base URL](https://www.alibabacloud.com/help/en/model-studio/base-url)、[视觉模型列表](https://www.alibabacloud.com/help/en/model-studio/vision-model)。

### 2.3 其他 OpenAI-compatible API

先设置自定义 key：

```bash
read -rsp "External API key: " MY_MODEL_API_KEY
echo
export MY_MODEL_API_KEY
```

复制一份配置到 Git 已忽略的 `tmp/`，不要直接污染正式实验配置：

```bash
mkdir -p tmp/configs
cp config/experiments/zebra.yaml tmp/configs/zebra-external.yaml
nano tmp/configs/zebra-external.yaml
```

把 provider 改成：

```yaml
provider:
  name: generic
  model: exact-model-id-from-provider
  base_url: https://example.com/v1
  api_key_env: MY_MODEL_API_KEY
  json_mode: true
  max_tokens: 4096
  timeout: 120
```

然后运行：

```bash
minibench run-config tmp/configs/zebra-external.yaml
```

`generic` 必须同时提供 `model`、`base_url` 和 `api_key_env`。如果 `base_url` 不以 `/chat/completions` 结尾，MiniBench 会自动补上。程序不会自动读取 `.env`；必须在当前 WSL shell 中 `export`。

## 3. YAML 如何工作

每份正式配置都包含五部分：

```yaml
task:        # family、数据路径、limit/task_ids 或象棋 sampling
agent:       # agent 架构及其推理参数
provider:    # API、模型、key 环境变量、超时、JSON 模式
evaluation:  # 任务特有的模式和评测参数
run:         # 输出目录与 run_name
```

运行任何 YAML 的统一命令是：

```bash
minibench run-config config/experiments/<name>.yaml
```

也可以使用薄封装：

```bash
./run.sh config/experiments/<name>.yaml
```

`run-config` 完全以 YAML 为准，不接受额外覆盖参数。要改 provider、agent 或只跑一题，请复制到 `tmp/configs/` 后修改。显式 CLI 覆盖只适用于象棋的 `run-task`/`run-suite`。

第一次连接收费 API 时，建议把本地副本中的 `task.limit` 改为 `1`。象棋则使用 `--sample-count 1`，无需改文件。

## 4. 每份任务 YAML 的调用方式

### 4.1 Zebra

| 配置 | 内容 |
| --- | --- |
| `zebra.yaml` | 正式逻辑网格推理，默认 CoT + DeepSeek V4 |
| `zebra_rule_codebook.yaml` | 临时规则/代码本变体 |
| `zebra_history.yaml` | 多轮历史记忆协议 |

```bash
minibench run-config config/experiments/zebra.yaml
minibench run-config config/experiments/zebra_rule_codebook.yaml
minibench run-config config/experiments/zebra_history.yaml
```

`zebra_history.yaml` 默认使用 `agent.name: passthrough`；所有实现 phase-aware 消息接口的推理 Agent 也可运行该多轮评测。

### 4.2 象棋 schema v2

当前提交题库为 `xiangqi-submitted-stratified100-v1`，schema 保持 v2。
四类各 100 条；seed=42，D3/M2 的 easy/medium/hard 为 34/33/33，H2 的 short/medium/long 为 34/33/33，C2 为 50 个标准/变体配对场景，覆盖全部三种变体。
题号、父题池及当前文件哈希见 `data/xiangqi/submission_manifest.json`；
记录约束见 `src/minibench/datasets/xiangqi/schema.py`，配置与方法边界见
[云服务器运行说明](docs/cloud-run.md)。

象棋现在只维护这四类任务及其出题、验证、可视化链路；只接收 schema v2 的
FEN 数据。旧吃将题、残局生存、通用对局、Gym action-index 接口和迁移工具已移除。
旧题库、中间候选池和历史实验结果不包含在 FINAL 源码包中；父题池另行备份，不作为当前实验输入。

| 配置 | 公开 family | 内容 |
| --- | --- | --- |
| `xiangqi_mate_in_one.yaml` | `xiangqi-mate-in-one` | 严格一步将死，接受全部合法答案 |
| `xiangqi_rule_variants.yaml` | `xiangqi-rule-variants` | 标准/焦点变体配对，多回合将杀，固定规则感知 depth-3 对手；CPE loss |
| `xiangqi_history.yaml` | `xiangqi-history` | 按参考将杀距离分层，配对 `full-state` / `move-history-only`，固定 Pikafish 应战 |
| `xiangqi_multimodal.yaml` | `xiangqi-multimodal` | 同一 D3 局面的文字/中文棋子图比较；一步将杀、合法率、独立棋盘识别准确率 |

D3/M2 用精确棋规判定将死，Pikafish 提供独立诊断；H2 使用固定的 Pikafish 对手。
C2 使用规则感知的 depth-3 对手，其 CPE loss 不能与 H2 的 CP loss 混为同一单位。
参考引擎与 NNUE 指纹在 `data/xiangqi/provenance.json`。云端入口采用
`engine_binding: runtime`，冻结服务器实际编译的引擎与 NNUE 指纹；续跑前重新核验，
不会把 Linux 编译结果冒充历史 Windows 引擎。新服务器按以下流程安装与准备：

```bash
bash scripts/cloud_bootstrap.sh
bash scripts/cloud_pikafish.sh
export PIKAFISH_PATH="$PWD/vendor/Pikafish/src/pikafish"
source .venv/bin/activate
mkdir -p output/xiangqi-checks
python scripts/validate_xiangqi_data.py --workers 6 --output output/xiangqi-checks/independent-validation.json
python scripts/audit_xiangqi_release.py --technical --validation-report output/xiangqi-checks/independent-validation.json --output output/xiangqi-checks/release-audit.json
python -B -m minibench.cloud prepare --config config/cloud/xiangqi-pilot.yaml --output runs/cloud-pilot-8
```

以上命令不调用语言模型。`--strict` 还检查外部来源声明，原仓库未声明顶级许可证的状态
仍会被报告。数据数量、规则、配对、哈希或来源有分歧时，技术门禁会失败。
源码变动后须重新验证；原 `data/xiangqi/independent_validation.json` 是历史证据，
不要把它的旧源码指纹当成当前代码已通过验收。

云端 pilot 冻结 D3/H2/M2 各 6 条、C2 三个配对场景（6 条）；仅 direct，
H2/M2 各两组，共 36 个条件。全 Agent 云端配置使用提交题库的全部 100 条，
九 Agent 共 5,400 个条件，各 Agent 和各组使用同一名单，不再重新抽题。
`evaluation_samples/formal` 中的 30/20 条名单仅保留作辅助探针，不是全 Agent 云端输入。
8 路 pilot 后再用相同小样本的新目录测试 16 路；同时调整 workers 和账号 max_in_flight。
完整运行、限流、原生思考及断点恢复见 [云服务器运行说明](docs/cloud-run.md)。

以下原生 YAML 是关闭思考的 direct 基线（保留参考引擎指纹），不等于云端全 Agent 思考配置。
如运行它们，须先提供匹配的参考引擎和对应服务商的临时凭据：

```bash
minibench run-config config/experiments/xiangqi_mate_in_one.yaml
minibench run-config config/experiments/xiangqi_rule_variants.yaml
minibench run-config config/experiments/xiangqi_history.yaml
minibench run-config config/experiments/xiangqi_multimodal.yaml
```

`run-task`/`run-suite` 可以覆盖 agent、provider、model 等参数。冻结配置的
`task.selection` 不能同时设置抽样、limit 或 task_ids；探索性重选题时，应复制 YAML
并明确移除 selection，记录自己的名单，避免混入正式对照。
同名结果禁止覆盖，下一次实验应使用新的 run_name 或输出目录。

当前配置只填写实际生效的参数：C2 为固定 `search_depth: 3` 的多回合规则感知对手；H2 为历史模式和固定
Pikafish；D3/M2 的引擎参数用于诊断；M2 默认仅文字棋盘与中文棋子图片两种输入。
拉丁图片仍是可选接口，但不在本轮配置或默认全 Agent 实验中。
回合上限来自题目本身，不能用 `evaluation.max_plies` 或 CLI 覆盖。
旧 `oracle_depth`、`opponent_depth`、`optimal_depth`、旧图片模式名等会直接报错。
正式 YAML 运行固定提交题库的全部 100 条；smoke YAML 引用冻结的小样本名单。

代码职责也收拢到一条链路：`schema.py` / `dataset.py` 加载当前数据；
`mate_in_one.py`、`rule_variants.py`、`history.py`、`multimodal.py` 各管一类评测；
`variants/` 管现有棋规与搜索，`engines/pikafish.py` 管 UCI 引擎；
`reference.py` / `validation.py` 管独立复验，`presentation.py` 管查看与离线图库。
`visualization/xiangqi_vis.py` 的单题与轨迹 PNG/GIF 复用 M2 绘图，图表使用当前
难度名，并将缺失诊断标为 missing，不当成零分。
出题入口保留在 `scripts/annotate_xiangqi_d3.py`、`build_xiangqi_c2.py`、
`build_xiangqi_h2.py` 和 `build_xiangqi_m2_from_d3.py`；各入口的 `--help` 提供参数说明。

网页版流程复用已准备的配置和名单：

```bash
python scripts/xiangqi_web_test.py --suite-dir runs/xiangqi-smoke-<时间戳> --prepare-only
```

人工检查数据也不需要调用模型：

```bash
minibench inspect-xiangqi --task xiangqi-history --id xiangqi-history-r1-revised-0001-6544758334 --format terminal
minibench inspect-xiangqi --task xiangqi-multimodal --id xiangqi-multimodal-m2-0001 --format png --output output/xiangqi-example.png
minibench build-xiangqi-gallery --output output/xiangqi-gallery.html
```

字段、效用函数与题库来源分别见 `schema.py`、`variants/`、
`data/xiangqi/provenance.json`；认证范围、评测单位和方法边界见
[云服务器运行说明](docs/cloud-run.md)。

### 4.3 一笔画

正式数据为 `data/one_stroke/direct.jsonl`、`history.jsonl` 和 `multimodal.jsonl`，
任务 ID 分别使用 `direct-`、`history-`、`multimodal-` 前缀。每组保留 easy/medium/hard 各 10 题；
多模态每题只有一张清晰图片，主实验 30 项，text/image 配对消融 60 项。

| 配置 | 内容 |
| --- | --- |
| `one_stroke_direct.yaml` | 直接求解，无欧拉定理提示 |
| `one_stroke.yaml` | 与正式 direct 相同的兼容入口；通常不必与 direct 重复运行 |
| `one_stroke_history.yaml` | 历史记忆：增量状态与仅历史对照 |
| `one_stroke_multimodal.yaml` | 多模态图片，默认 Qwen |
| `one_stroke_multimodal_ablation.yaml` | 多模态 text/image 配对消融 |
| `one_stroke_euler_theorem.yaml` | 与正式 direct 同题的欧拉定理提示消融 |
| `one_stroke_generated.yaml` | 生成数据，baseline prompt |
| `one_stroke_generated_euler_theorem.yaml` | 生成数据，Euler prompt |

```bash
minibench run-config config/experiments/one_stroke_direct.yaml
minibench run-config config/experiments/one_stroke.yaml
minibench run-config config/experiments/one_stroke_history.yaml
minibench run-config config/experiments/one_stroke_multimodal.yaml
minibench run-config config/experiments/one_stroke_multimodal_ablation.yaml
minibench run-config config/experiments/one_stroke_euler_theorem.yaml
minibench run-config config/experiments/one_stroke_generated.yaml
minibench run-config config/experiments/one_stroke_generated_euler_theorem.yaml
```

规则限定推理子任务已移除。旧编号路径、旧评分字段和旧图片模式不再兼容；已有 `runs/` 结果保留，
新配置默认新建运行目录，不跨版本续跑。评分字段采用 `direct_score`、`history_*` 和 `multimodal_*`。

一笔画 history 会区分 `intermediate` 与 `final` 阶段。`passthrough` 可直接运行；
CoT/ToT 等 phase-aware 包装器会在中间轮使用轻量消息调用，只在最终轮执行其完整推理流程。

### 4.4 麻将

合并兼容说明：当前麻将 CoT 配置显式使用 `prompt_version: v1`，沿用本地已验证的内部推理提示词及 history-only 的 current-turn 包装；推理阶段不要求只输出任务 JSON。`samples` 已移除（新配置校验不接受 CoT 中无效的采样参数），原有温度和 token 上限保持不变。麻将评测使用原有三次动作尝试，关闭 runtime 的额外格式修复，避免首次回答指标被隐式修复改变；内部阶段输出按轮保存。非空截断推理仍进入 final，并保留 `finish_reason`。

main 的新 ToT 是 beam/BFS 搜索，使用 `prompt_version: v2` 及 `max_depth`、`branching_factor`、`beam_width`，不再是旧版三个候选再选择；旧 ToT 成绩不能当成新版 ToT 成绩混合汇总。要切换为新 ToT，请同时修改 `name` 和 `prompt_version`，并使用新的实验目录。

| 配置 | 内容 |
| --- | --- |
| `mahjong.yaml` | 静态牌型文本推理 |
| `mahjong_solo.yaml` | 标准规则下的 full-hand 单人摸弃决策 |
| `mahjong_rule_variants.yaml` | 全手牌规则适应，默认扩展全部规则通道 |
| `mahjong_history_only.yaml` | 标准规则下的在线多轮 history-only 决策 |
| `mahjong_multimodal.yaml` | 静态 600 题的一一对应图片版本，仅运行 image，默认 Qwen |

当前静态麻将六个子题组各 100 道（共 600 道），多模态逐题镜像。Solo、history-only 和改规则共用 100 道摸打源题，每个改规则通道各 100 道。原有静态 90 道、摸打 15 道的题目和 ID 均保留；新增题使用现有生成器，随机种子为 `20261011`。摸打仍使用 50 轮上限，初始向听不超过 2、初始进张至少 12、参考策略 18 轮内胡牌，并要求 10 次并列弃牌随机模拟的胡牌率至少 0.8。题目筛选、提示词和评分逻辑不变。

```bash
minibench run-config config/experiments/mahjong.yaml
minibench run-config config/experiments/mahjong_solo.yaml
minibench run-config config/experiments/mahjong_rule_variants.yaml
minibench run-config config/experiments/mahjong_history_only.yaml
minibench run-config config/experiments/mahjong_multimodal.yaml
```

麻将 history-only 只在第一轮提供初始手牌，后续只提供新摸牌和环境已接受的历史弃牌。由于每轮弃牌都是正式计分决策，每轮都以 `final` 阶段调用 agent；兼容仅实现旧 `generate_messages()` 接口的自定义 agent。

麻将视觉图片来自仓库内牌面素材与确定性 Pillow 渲染器。视觉数据逐题复用静态麻将的 ID、手牌、公开牌、目标和标准答案，仅把牌码输入转换成图片。重新渲染使用：

```bash
minibench render-mahjong-visual \
  --tasks data/mahjong/tasks.jsonl \
  --output data/mahjong/visual_tasks.jsonl \
  --render-dir data/mahjong/visual_tasks_visual \
  --overwrite
```

## 5. 如何切换 agent 架构

### 5.1 通信客户端与推理策略是两层

`provider`、通信客户端和 Agent 策略各自负责不同事情：

```text
provider YAML
      ↓
OpenAICompatibleClient        请求、HTTP 重试、provider 兼容、usage、响应解析
      ↓
Passthrough / Direct / CoT / SC / Best-of-N / ToT / ...
                              Prompt 编排与推理算法
```

`OpenAICompatibleClient` 不是一种推理架构。它只负责把文本、历史消息和图片发送到 OpenAI-compatible `chat/completions` 服务，并解析可见内容、独立 reasoning、finish reason 与 usage。Provider 返回的 `reasoning_content` 不会再被当作最终答案；可见内容为空或被截断会产生明确错误。

`passthrough` 与 `direct` 都通常只调用模型一次，但语义不同：

- `passthrough` 保留原始调用语义，不附加策略 Prompt，也不强制 JSON 校验或格式修复，适合作为 raw baseline 和连接测试。
- `direct` 会附加 Direct v2 策略契约，要求恰好一个最终 JSON 对象，并可使用统一运行追踪、预算和一次低温格式修复。

公开的 `agent.name` 为：

```text
passthrough
direct
cot
self-consistency
best-of-n
tot
plan-then-solve
critic-refine
least-to-most
```

未显式填写名称时 factory 默认使用 `direct`。旧 YAML 名称 `openai-compatible` 已从公开列表移除，但仍会在发出 `FutureWarning` 后解析为 `passthrough`，完整保留 raw baseline 语义；它绝不会静默映射为 `direct`。旧代码直接导入的 `OpenAICompatibleAgent` 也只作为兼容包装保留。

### 5.2 各架构的算法与调用数

下表是单次静态题的正常调用数，不包含响应非法后可能发生的格式修复调用：

| `agent.name` | 正常模型调用数 | 算法 |
| --- | ---: | --- |
| `passthrough` | 1 | 原样调用；支持真实历史 messages |
| `direct` | 1 | 一次调用直接生成最终 JSON |
| `cot` | 2 | reasoning → finalizer |
| `self-consistency` | `N` | `N=samples` 条独立 CoT 路径，程序化 JSON 投票 |
| `best-of-n` | `N + 1` | `N` 条 CoT 候选 + 一次只选 ID 的 LLM judge |
| `tot` | 随搜索树变化 | beam/BFS 展开 → 每层 value 评分 → finalizer |
| `plan-then-solve` | 3 | plan → solve → finalizer |
| `critic-refine` | 3 | draft → critique → refine |
| `least-to-most` | `K + 2` | decompose → 顺序解决 `K` 个子问题 → finalizer |

Self-Consistency 不再调用 LLM judge。它从每个样本提取最后一个完整 JSON 对象，用排序键和紧凑序列化 canonicalize 后计票；非法候选被忽略，多数票获胜，最高票平局时稳定选择最早生成的候选。若所有样本都非法，且允许格式修复，则只对一个候选执行一次低温修复。`samples` 默认 3、最小 2，`reasoning_temperature` 必须非零。

Best-of-N 保留 LLM judge，但 judge 只能返回 `selected_id` 和每个候选的评分，不能改写任务答案。候选使用稳定 ID，展示顺序按 `selection_seed` 确定性打乱，默认 seed 为 42；程序最终返回被选候选中的 JSON。

当前 `tot` 是任务无关的真正 beam/BFS Tree of Thoughts，而不是多候选别名。它为 beam 中的非终止节点生成子 thought，每层批量调用 value evaluator 得到 `[0,1]` 分数并裁剪到 `beam_width`；终止节点不再展开。同分按稳定 node ID 排序。搜索在 beam 全终止、深度/节点/调用预算耗尽时停止，优先选择最高分终止节点，否则选最高分叶节点，再由统一 finalizer 生成任务 JSON。它不读取数据集标准答案或任务评分器。

Least-to-Most 先结构化分解，随后按依赖顺序解决子问题；每一步只能引用原题和此前结果。合法但为空的分解会把原题作为唯一子问题，最终再合成严格 JSON。默认最多 4 个子问题。

动态棋局或牌局会在每个 Agent 行动回合重复上述过程。多模态推理的每个需要观察原题的阶段都会传递图片，因此调用量和图片 token 成本会明显增加。

### 5.3 严格 YAML 字段

Agent 配置现在严格校验。字段放在不支持它的架构下会直接报错，不再静默忽略。字段组如下：

- 基础字段：`name`、`predictions`、`max_tokens`。
- 运行字段：`prompt_version`、`trace`、`max_llm_calls`、`max_total_tokens`、`max_format_repairs`。
- 推理字段：`reasoning_temperature`、`final_temperature`、`max_reasoning_tokens`。

各架构允许的额外字段为：

| Agent | 允许字段 |
| --- | --- |
| `passthrough` | 仅基础字段；不接受运行字段、reasoning 字段或 `samples` |
| `direct` | 基础字段 + 运行字段 + `final_temperature` |
| `cot` / `plan-then-solve` / `critic-refine` | 基础字段 + 运行字段 + 推理字段 |
| `self-consistency` | 上述推理 Agent 字段 + `samples` |
| `best-of-n` | 上述推理 Agent 字段 + `samples` + `selection_seed` |
| `tot` | 上述推理 Agent 字段 + `max_depth`、`branching_factor`、`beam_width`、`max_search_nodes` |
| `least-to-most` | 上述推理 Agent 字段 + `max_subproblems` |

`prompt_version` 默认 `v2`。`v1` 完整保留旧 Direct、CoT、Plan、Critic 等模板，供旧实验显式复现；新架构 `best-of-n`、真正的 `tot` 和 `least-to-most` 只支持 `v2`。v2 每个阶段都使用固定的 `<OBJECTIVE>`、`<INPUT>`、`<PRIOR_STATE>`、`<CONSTRAINTS>`、`<OUTPUT_CONTRACT>`、`<STOP_POLICY>` 六段契约，候选和状态以 JSON 安全编码。运行轨迹会记录 Prompt 版本和 SHA-256 hash。

`trace` 默认 `summary`：

- `off`：不保留逐阶段轨迹。
- `summary`：保留阶段名、Prompt hash、耗时、usage、解析状态和算法元数据，不保存原文。
- `full`：额外保存完整 Prompt、可见响应、独立 reasoning 和解析后 JSON。它可能包含原题或模型敏感内容，应谨慎启用。

预算与修复字段：

- `max_llm_calls`：一次 Agent run 的逻辑 LLM 调用上限；内部阶段和格式修复都计数。
- `max_total_tokens`：依据 provider `usage.total_tokens` 累积的软上限。单次调用可能越过上限；达到上限后会在下一次调用前停止。Provider 不返回 usage 时无法精确执行该限制。
- `max_format_repairs`：只能是 0 或 1，默认 1；控制需要严格 JSON 的阶段是否允许一次低温格式修复。`passthrough` 始终不修复。
- `max_search_nodes`：ToT 的可选硬节点上限；留空时由深度、分支数和 beam 宽度推导。

最终字符串接口保持不变；对使用统一运行内核的策略 Agent（即除 `passthrough` 外的公开策略），可通过 `agent.last_run` 查看最近一次运行的阶段、预算、状态和停止原因。

### 5.4 配置示例

复制正式 YAML 到 Git 已忽略的 `tmp/` 后再修改：

```bash
mkdir -p tmp/configs
cp config/experiments/mahjong.yaml tmp/configs/mahjong-agent.yaml
nano tmp/configs/mahjong-agent.yaml
```

Raw passthrough 只写基础字段：

```yaml
agent:
  name: passthrough
  max_tokens: 512
```

CoT 不接受 `samples`：

```yaml
agent:
  name: cot
  prompt_version: v2
  trace: summary
  reasoning_temperature: 0.0
  final_temperature: 0.0
  max_reasoning_tokens: 1024
  max_tokens: 512
  max_llm_calls: 3
  max_total_tokens: 8192
  max_format_repairs: 1
```

真正的 Self-Consistency：

```yaml
agent:
  name: self-consistency
  samples: 5
  reasoning_temperature: 0.7
  final_temperature: 0.0
  max_reasoning_tokens: 1024
  max_tokens: 512
  max_llm_calls: 6
```

Best-of-N：

```yaml
agent:
  name: best-of-n
  samples: 5
  selection_seed: 42
  reasoning_temperature: 0.7
  final_temperature: 0.0
  max_reasoning_tokens: 1024
  max_tokens: 512
  max_llm_calls: 8
```

默认规模的 beam/BFS ToT：

```yaml
agent:
  name: tot
  prompt_version: v2
  max_depth: 3
  branching_factor: 3
  beam_width: 2
  max_search_nodes: 16
  reasoning_temperature: 0.7
  final_temperature: 0.0
  max_reasoning_tokens: 1024
  max_tokens: 512
  max_llm_calls: 24
```

Least-to-Most：

```yaml
agent:
  name: least-to-most
  prompt_version: v2
  max_subproblems: 4
  reasoning_temperature: 0.0
  final_temperature: 0.0
  max_reasoning_tokens: 1024
  max_tokens: 512
  max_llm_calls: 12
```

运行修改后的配置：

```bash
minibench run-config tmp/configs/mahjong-agent.yaml
```

选择建议：

- 先用 `passthrough` 做一题连接和 provider 响应 smoke test。
- 再用 `direct` 或 `cot` 建立可比较基线。
- 只有在预算允许时再用 `self-consistency`、`best-of-n`、`tot`、`least-to-most`、`plan-then-solve` 或 `critic-refine`。
- Zebra history 和一笔画 history 支持 `passthrough` 以及实现 phase-aware 消息接口的推理 Agent。
- 象棋 history 可以切换 Agent，但会在多步对局中产生很多模型调用。

## 6. 多模态：DeepSeek 文本 + Qwen 图片的正确跑法

### 6.1 推荐的主实验

四份正式视觉配置已经使用 Qwen：

```bash
export DASHSCOPE_API_KEY

minibench run-config config/experiments/xiangqi_multimodal.yaml
minibench run-config config/experiments/one_stroke_multimodal.yaml
minibench run-config config/experiments/one_stroke_multimodal_ablation.yaml
minibench run-config config/experiments/mahjong_multimodal.yaml
```

其中：

- 象棋运行 `text`、`chinese-piece-image`、`latin-piece-image`。
- 一笔画 multimodal 正式运行 `image`，消融运行 `text`、`image`。
- 麻将正式运行 `image`，配对文字输入使用 `mahjong.yaml` 的同源静态题。

配对消融应让同一个 Qwen 模型同时跑文本和图片，才能把差异主要归因于输入模态，而不是模型能力差异。

### 6.2 另跑 DeepSeek 文本基线

如果还要记录 DeepSeek 文本成绩，不要让 DeepSeek 接收图片。以象棋为例：

```bash
mkdir -p tmp/configs
cp config/experiments/xiangqi_multimodal.yaml \
  tmp/configs/xiangqi-multimodal-text-deepseek.yaml
nano tmp/configs/xiangqi-multimodal-text-deepseek.yaml
```

把 provider 和输入模式改为：

```yaml
provider:
  name: deepseek
  model: deepseek-v4-flash
  api_key_env: DEEPSEEK_API_KEY
  json_mode: true
  timeout: 120
  extra_body:
    thinking:
      type: disabled

evaluation:
  input_modes:
    - text
```

然后运行：

```bash
export DEEPSEEK_API_KEY
minibench run-config tmp/configs/xiangqi-multimodal-text-deepseek.yaml
```

这份 DeepSeek 结果是“额外的跨模型文本基线”，不能替代 Qwen 自身 text/image 的配对视觉差值。

### 6.3 给多模态任务换 agent

所有当前推理架构都实现了图片转发。例如把象棋多模态改成 Qwen + CoT：

```bash
cp config/experiments/xiangqi_multimodal.yaml \
  tmp/configs/xiangqi-multimodal-qwen-cot.yaml
nano tmp/configs/xiangqi-multimodal-qwen-cot.yaml
```

只修改 agent 段：

```yaml
agent:
  name: cot
  reasoning_temperature: 0.0
  final_temperature: 0.0
  max_reasoning_tokens: 1024
  max_tokens: 512
```

按正式冻结题单运行 CoT 图片条件（不是一题）：

```bash
minibench run-task xiangqi-multimodal \
  --agent cot \
  --provider qwen \
  --model qwen3.8-max \
  --input-modes chinese-piece-image \
  --run-name xiangqi-m2-qwen-cot-image-new
```

该 CLI 命令继承正式 YAML 中的 `DASHSCOPE_API_KEY` 和其他评测设置；完整自定义仍建议运行刚复制的 YAML。
若只想试一题，在复制的 YAML 中删除 `task.selection`，显式设置
`task.sampling: {enabled: true, count: 1, seed: 42, strategy: stratified}`，
并换一个 `run.run_name`。不要改写冻结题单。

## 7. 结果、复现与离线运行

每次评测在 `runs/` 下创建独立目录，核心文件为：

- `predictions.jsonl`：原始输出与逐题结果。
- `results.json`：汇总指标。
- `summary.txt`：人类可读摘要。
- 象棋 v2 额外保存 `selected_tasks.jsonl`、`resolved_config.yaml` 与 `run_metadata.json`，记录精确抽样、数据与配置哈希、schema、renderer、运行环境和 Pikafish 指纹。

在 WSL 中查看最近产生的文件：

```bash
find runs -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM  %p\n' \
  | sort -r \
  | head -30
```

如果已有离线预测文件，可在本地 YAML 的 agent 段设置：

```yaml
agent:
  name: passthrough
  predictions: path/to/predictions.jsonl
```

存在 `predictions` 时会使用确定性的 prediction-file agent，不访问 API。

象棋不再提供旧格式迁移或旧配置兼容。历史数据与结果原样保留；新实验只使用当前
四份正式 YAML 或 `config/experiments/xiangqi_smoke/` 的四份小样本 YAML。

## 8. 常见错误

### `Missing API key`

key 没有导出到当前 WSL shell：

```bash
export DEEPSEEK_API_KEY
export DASHSCOPE_API_KEY
env | grep -E 'DEEPSEEK_API_KEY|DASHSCOPE_API_KEY' | sed 's/=.*/=<set>/'
```

### Qwen 返回 HTTP 401

通常是 API key 与 endpoint 区域不一致。北京、新加坡、美国的 key 不能混用；核对 `provider.name`/`base_url` 和创建 key 的区域。

### 图片任务返回“不支持 image”或 HTTP 400

检查三点：provider 是否为 Qwen、模型是否支持视觉输入、`evaluation.input_modes` 是否意外把图片发给了 DeepSeek。

### 模型输出为空、超时或 JSON 被截断

先使用 `passthrough`、关闭 provider 原生 thinking，并提高 `provider.timeout` 与 `agent.max_tokens`/`provider.max_tokens`。不要一开始就运行多样本 agent 或完整多模态消融。

### `Pikafish executable was not found`

```bash
export PIKAFISH_PATH="$HOME/opt/Pikafish/src/pikafish"
ls -l "$PIKAFISH_PATH"
```

### WSL 中找不到仓库

Windows 的 `D:\AAALimoWork\CS\Seminar\MiniBench` 在 WSL 中是：

```text
/mnt/d/AAALimoWork/CS/Seminar/MiniBench
```

## 9. 项目结构

```text
config/experiments/       # 可执行 YAML，是非敏感实验配置的唯一来源
data/                     # JSONL 任务数据
src/minibench/agents/     # agent 推理架构
src/minibench/factory/    # config、agent、provider、experiment 装配
src/minibench/datasets/   # 各任务加载、prompt、评测、渲染与引擎
tests/                    # 单元、数据、CLI 与图片回归测试
runs/                     # 运行输出，Git 忽略
```

版本破坏性变更见 [`CHANGELOG.md`](CHANGELOG.md)，任务编写规范见 [`docs/task-authoring.md`](docs/task-authoring.md)。

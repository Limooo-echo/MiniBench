# MiniBench 0.2.0

MiniBench 是一个用统一 YAML、统一 agent/provider 接口和统一结果格式评测推理模型的小型基准。当前包含 Zebra 逻辑题、象棋、一笔画、麻将与四人立直麻将。

本 README 以 **WSL 2 + Ubuntu 22.04 + Python 3.10** 为标准环境。进入 Ubuntu 后，下面所有安装、配置、运行和排错命令都在 WSL 终端执行。

> 先记住两条：文本任务默认使用 DeepSeek V4；带图片的任务必须使用支持视觉输入的模型，仓库默认使用 Qwen。不要用 DeepSeek 跑 `image`、`challenge_image`、`chinese-piece-image` 或 `latin-piece-image`。

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
python -m pip install -c constraints/ci-py310.txt -e .
```

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

`zebra_history.yaml` 必须保持 `agent.name: openai-compatible`，因为该评测调用真实的多轮 `generate_messages()` 接口。

### 4.2 象棋 schema v2

| 配置 | 公开 family | 内容 |
| --- | --- | --- |
| `xiangqi_mate_in_one.yaml` | `xiangqi-mate-in-one` | 一步杀 |
| `xiangqi_rule_variants.yaml` | `xiangqi-rule-variants` | 标准规则与三个规则变体 |
| `xiangqi_history.yaml` | `xiangqi-history` | `full-state` / `move-history-only` |
| `xiangqi_multimodal.yaml` | `xiangqi-multimodal` | 文本、中文棋子图、拉丁棋子图；默认 Qwen |

直接运行 YAML：

```bash
minibench run-config config/experiments/xiangqi_mate_in_one.yaml
minibench run-config config/experiments/xiangqi_rule_variants.yaml
minibench run-config config/experiments/xiangqi_history.yaml
minibench run-config config/experiments/xiangqi_multimodal.yaml
```

象棋还提供可覆盖 YAML 的便捷入口：

```bash
minibench run-task xiangqi-mate-in-one \
  --agent cot \
  --provider deepseek \
  --model deepseek-v4-flash \
  --sample-seed 42 \
  --sample-count 10

minibench run-task xiangqi-history \
  --history-mode full-state \
  --sample-count 10 \
  --pikafish-depth 8

minibench run-suite \
  --tasks xiangqi-mate-in-one,xiangqi-rule-variants \
  --sample-count 10
```

`run-task`/`run-suite` 可覆盖 agent、provider、model、key 变量、抽样和常用评测参数，但不提供 `--base-url`。自定义 endpoint 请复制 YAML，设置 `provider.base_url`，再用 `run-config`。

一步杀和历史任务需要 Pikafish。全部操作仍在 WSL 中：

```bash
mkdir -p ~/opt
git clone https://github.com/official-pikafish/Pikafish.git ~/opt/Pikafish
cd ~/opt/Pikafish/src
make -j"$(nproc)" profile-build

export PIKAFISH_PATH="$HOME/opt/Pikafish/src/pikafish"
test -x "$PIKAFISH_PATH"

cd /mnt/d/AAALimoWork/CS/Seminar/MiniBench
```

Pikafish 官方编译说明也建议在 `src` 下执行 `make -j profile-build`：[官方 README](https://github.com/official-pikafish/Pikafish#compiling-pikafish)。

人工检查数据不需要调用模型：

```bash
minibench inspect-xiangqi \
  --task xiangqi-history \
  --id xiangqi-history-0001 \
  --format terminal

minibench inspect-xiangqi \
  --task xiangqi-multimodal \
  --id xiangqi-multimodal-0001 \
  --format png \
  --output output/xiangqi-example.png

minibench build-xiangqi-gallery --output output/xiangqi-gallery.html
```

字段、FEN、坐标、UCI 和评分定义见 [`docs/xiangqi-data-card.md`](docs/xiangqi-data-card.md)。

### 4.3 一笔画

| 配置 | 内容 |
| --- | --- |
| `one_stroke_a1.yaml` | A1 直接求解，无欧拉定理提示 |
| `one_stroke.yaml` | 与正式 A1 相同的兼容入口；通常不必与 A1 重复运行 |
| `one_stroke_a2.yaml` | A2 临时规则条件 |
| `one_stroke_a2_ablation.yaml` | A2 full/standard/drop/conflicting 消融 |
| `one_stroke_a3_history.yaml` | A3 增量状态与仅历史对照 |
| `one_stroke_a4.yaml` | A4 挑战图片，默认 Qwen |
| `one_stroke_a4_ablation.yaml` | A4 text/clear image/challenge image 配对消融 |
| `one_stroke_euler_theorem.yaml` | 旧 smoke 数据的欧拉定理提示消融 |
| `one_stroke_generated.yaml` | 生成数据，baseline prompt |
| `one_stroke_generated_euler_theorem.yaml` | 生成数据，Euler prompt |

```bash
minibench run-config config/experiments/one_stroke_a1.yaml
minibench run-config config/experiments/one_stroke.yaml
minibench run-config config/experiments/one_stroke_a2.yaml
minibench run-config config/experiments/one_stroke_a2_ablation.yaml
minibench run-config config/experiments/one_stroke_a3_history.yaml
minibench run-config config/experiments/one_stroke_a4.yaml
minibench run-config config/experiments/one_stroke_a4_ablation.yaml
minibench run-config config/experiments/one_stroke_euler_theorem.yaml
minibench run-config config/experiments/one_stroke_generated.yaml
minibench run-config config/experiments/one_stroke_generated_euler_theorem.yaml
```

`one_stroke_a3_history.yaml` 与 Zebra history 一样要求 `openai-compatible`，不能直接换成当前的 CoT/ToT 包装器。

### 4.4 麻将

| 配置 | 内容 |
| --- | --- |
| `mahjong.yaml` | 静态牌型文本推理 |
| `mahjong_rule_variants.yaml` | 全手牌规则适应，默认扩展全部规则通道 |
| `mahjong_riichi.yaml` | 本地四人立直麻将；默认其余座位使用 shanten bot |
| `mahjong_multimodal.yaml` | 牌面图片输入，默认 Qwen |
| `mahjong_multimodal_ablation.yaml` | 同题 text/image 配对消融，默认 Qwen |

```bash
minibench run-config config/experiments/mahjong.yaml
minibench run-config config/experiments/mahjong_rule_variants.yaml
minibench run-config config/experiments/mahjong_riichi.yaml
minibench run-config config/experiments/mahjong_multimodal.yaml
minibench run-config config/experiments/mahjong_multimodal_ablation.yaml
```

麻将视觉图片来自仓库内牌面素材与确定性 Pillow 渲染器。更换 Qwen provider/agent 不会改变牌局数据、答案或评分逻辑。

## 5. 如何切换 agent 架构

agent 架构和 provider 是两个独立维度：`agent.name` 决定一次题目如何组织模型调用，`provider` 决定这些调用发给哪个模型服务。

| `agent.name` | 单次静态题的大致模型调用数 | 用途 |
| --- | ---: | --- |
| `openai-compatible` | 1 | 最小基线；支持真实多轮 messages |
| `direct` | 1 | 强制直接输出最终 JSON |
| `cot` | 2 | 先推理，再整理最终 JSON |
| `self-consistency` | `samples + 1` | 多条推理路径后评选 |
| `tot` | `samples + 1` | 多候选 thought branches 后评选 |
| `plan-then-solve` | 3 | 计划、求解、最终格式化 |
| `critic-refine` | 3 | 草稿、批评、修订 |

动态棋局/牌局会在每个 agent 行动回合重复上述过程；多模态推理包装器还会在各阶段重复发送图片，因此调用量和图片 token 成本会明显增加。

复制一份 YAML 后修改 agent：

```bash
mkdir -p tmp/configs
cp config/experiments/mahjong.yaml tmp/configs/mahjong-cot.yaml
nano tmp/configs/mahjong-cot.yaml
```

例如 CoT：

```yaml
agent:
  name: cot
  samples: 1
  reasoning_temperature: 0.0
  final_temperature: 0.0
  max_reasoning_tokens: 1024
  max_tokens: 512
```

例如 self-consistency：

```yaml
agent:
  name: self-consistency
  samples: 5
  reasoning_temperature: 0.7
  final_temperature: 0.0
  max_reasoning_tokens: 1024
  max_tokens: 512
```

运行修改后的配置：

```bash
minibench run-config tmp/configs/mahjong-cot.yaml
```

选择建议：

- 先用 `openai-compatible` 做一题连通性 smoke test。
- 再用 `direct` 或 `cot` 建基线。
- 只有在预算允许时再用 `self-consistency`、`tot`、`plan-then-solve`、`critic-refine`。
- `zebra_history.yaml` 和 `one_stroke_a3_history.yaml` 当前必须使用 `openai-compatible`。
- 象棋 history 可以切换 agent，但会在多步对局中产生很多模型调用。

## 6. 多模态：DeepSeek 文本 + Qwen 图片的正确跑法

### 6.1 推荐的主实验

五份正式视觉配置已经使用 Qwen：

```bash
export DASHSCOPE_API_KEY

minibench run-config config/experiments/xiangqi_multimodal.yaml
minibench run-config config/experiments/one_stroke_a4.yaml
minibench run-config config/experiments/one_stroke_a4_ablation.yaml
minibench run-config config/experiments/mahjong_multimodal.yaml
minibench run-config config/experiments/mahjong_multimodal_ablation.yaml
```

其中：

- 象棋运行 `text`、`chinese-piece-image`、`latin-piece-image`。
- 一笔画 A4 正式运行 `challenge_image`，消融运行 `text`、`clear_image`、`challenge_image`。
- 麻将正式运行 `image`，消融运行 `text`、`image`。

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
  opponent_depth: 4
  optimal_depth: 3
  max_plies: 20
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
  samples: 1
  reasoning_temperature: 0.0
  final_temperature: 0.0
  max_reasoning_tokens: 1024
  max_tokens: 512
```

先用一题验证：

```bash
minibench run-task xiangqi-multimodal \
  --agent cot \
  --provider qwen \
  --model qwen3.8-max \
  --sample-count 1 \
  --input-modes chinese-piece-image
```

该 CLI 命令继承正式 YAML 中的 `DASHSCOPE_API_KEY` 和其他评测设置；完整自定义仍建议运行刚复制的 YAML。

## 7. 结果、复现与离线运行

每次评测在 `runs/` 下创建独立目录，核心文件为：

- `predictions.jsonl`：原始输出与逐题结果。
- `results.json`：汇总指标。
- `summary.txt`：人类可读摘要。
- 象棋 v2 额外保存 `resolved_config.yaml` 与 `run_metadata.json`，记录数据哈希、schema、renderer 和依赖版本。

在 WSL 中查看最近产生的文件：

```bash
find runs -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM  %p\n' \
  | sort -r \
  | head -30
```

如果已有离线预测文件，可在本地 YAML 的 agent 段设置：

```yaml
agent:
  name: openai-compatible
  predictions: path/to/predictions.jsonl
```

存在 `predictions` 时会使用确定性的 prediction-file agent，不访问 API。

旧象棋 0.1.x 数据或 run 目录必须显式迁移：

```bash
minibench migrate-xiangqi-v2 \
  --input old-run \
  --output migrated-run \
  --dry-run

minibench migrate-xiangqi-v2 \
  --input old-run \
  --output migrated-run
```

迁移映射位于 `data/xiangqi/migration_v1_to_v2.json`；工具不会改写 `raw_output` 或模型自由文本。

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

先使用 `openai-compatible`、关闭 provider 原生 thinking，并提高 `provider.timeout` 与 `agent.max_tokens`/`provider.max_tokens`。不要一开始就运行多样本 agent 或完整多模态消融。

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

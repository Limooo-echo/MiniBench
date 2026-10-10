# 云服务器运行与断点恢复

本说明针对 `MINIBENCH-FINAL` / `cloud-20261010` 分支。它合并 `mahjong-checkpoint-20261010` 的非象棋任务及评分系统，与本地新版象棋代码和四套固定分层抽出的 100 条提交题库；不包含旧实验结果。原本地仓库不受影响。**新增云端入口不会自动启动，也不会保存 API key。** 当前题号/抽样依据见 [`submission_manifest.json`](../data/xiangqi/submission_manifest.json)，记录约束见 [`schema.py`](../src/minibench/datasets/xiangqi/schema.py)。

## 1. 先确定服务器用途

当前方案调用远程模型 API，服务器承担请求调度、图片生成、棋规搜索和 Pikafish 裁判。可先使用普通 Linux CPU 主机，不必为远程 API 租 GPU。它能提升批量吞吐和持续运行能力，不能保证单次模型思考变快。

推荐先检查 CPU、内存和出口网络，再用固定 pilot 实测；不在这里承诺配置或时长。8 路时每个 H2 任务有独立引擎（Threads=1、Hash=128MB），另有 NNUE、Python 进程和 C2 搜索开销。16 路会增加这些资源需求。未来部署本地模型是另外一套工作，需按模型参数量、量化和上下文长度评估显存，不是换一个 URL 就已完成部署。

## 2. 首次安装（Ubuntu / Debian）

将 FINAL 的干净源码包复制到服务器，或确认 `cloud-20261010` 分支已上传后克隆指定分支；不要从旧 checkpoint 分支取得云端版本。不要复制 `.venv`、`vendor`、`run`、`runs`、`output` 或私有配置；Python 环境在服务器重新建。源码包不带 `.git`，正常 Git 克隆则可保留它用于版本追踪。

```bash
git clone --branch cloud-20261010 --single-branch https://github.com/Limooo-echo/MiniBench.git MINIBENCH-FINAL
```

本机生成干净源码包（输出不得已存在）：

```bash
python3 scripts/export_cloud_source.py --output /tmp/MINIBENCH-FINAL-source.tar.gz
```

将这个包上传服务器后，用 `tar -xzf MINIBENCH-FINAL-source.tar.gz` 解压，再进入目录。

```bash
cd /path/to/MINIBENCH-FINAL
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential git curl
bash scripts/cloud_bootstrap.sh
bash scripts/cloud_pikafish.sh
export PIKAFISH_PATH="$PWD/vendor/Pikafish/src/pikafish"
source .venv/bin/activate
```

Python 需要 >=3.10。安装使用 `constraints/ci-py310.txt` 和独立裁判 `cchess==1.25.5`。Pikafish 下载/编译固定于官方 `Pikafish-2026-09-06`、提交 `4c17cee11f888ae1d48a9494f2e2239f019f0a1f`；默认保守 x86-64。ARM 主机设置 `PIKAFISH_ARCH=armv8`。引擎与网络文件只存忽略的 vendor，不作为 MiniBench 自有代码重新分发。

`engine_binding: runtime` 在准备时登记服务器上实际二进制和 NNUE 的 SHA-256，并在运行和续跑前核验。Linux 编译结果不等于历史 Windows 二进制；结果必须标明这个新的引擎条件。`reference` 则要求原正式 YAML 登记的历史指纹，不允许悄悄替换。

可离线检查：

```bash
python -B -m unittest discover -s tests
python -B scripts/validate_xiangqi_data.py --workers 4 --output output/checks/validation.json
python -B scripts/audit_xiangqi_release.py --technical --validation-report output/checks/validation.json --output output/checks/release.json
python -B scripts/certify_xiangqi_c2.py --workers 4 --output output/checks/c2-certificates.json
```

技术门禁通过不等于整个科学发布就绪；见第 8 节。
证书复验会重放已发布的、与数据及对手策略指纹匹配的 witness，并重新计算每一步；
不会因为已有 `valid: true` 就跳过验证。题内参考 PV 不一定是实际 depth-3 对手的应对，
不能直接替代对手策略证书。新题或不匹配的证书需重新搜索；搜索预算耗尽只代表未知。

## 3. 配置账号、原生思考与 Agent

默认 `config/cloud/xiangqi-pilot.yaml` 使用学校 endpoint、模型名 `qwen`，开启原生思考、仅 direct、8 路。只设置当前终端的临时变量：

```bash
read -rsp 'API key: ' MINIBENCH_SJTU_API_KEY
echo
export MINIBENCH_SJTU_API_KEY
```

不要把 key 粘入命令参数、YAML、日志、Git 或 `.env` 文件。`.env.example` 只是空变量说明，程序不自动读取它。请求认证头不落盘；原始回答、推理文本和题目会保存，结果目录也应设为私有。

`config/cloud/accounts.example.yaml` 提供独立 Qwen / DeepSeek 账号模板，可复制账号块到私有 `config/cloud/local*.yaml`。每个 experiment 的 `account` 指向对应账号，不同账号分别限制请求数。同一真实账号的所有模型应共用同一账号块、环境变量名和限额；**不要靠换环境变量名绕过限制。** 同时运行其他程序或第二个 suite 的调用不在本调度器计数中，建议每个账号只运行一个调度器。

限额设置说明：

| 字段 | 作用 |
| --- | --- |
| `workers` | 同时执行的完整题目/组别数，初始 8 |
| `max_in_flight` | 同账号最多同时在途的 API 请求数 |
| `requests_per_minute` | 本调度器 60 秒滚动请求数上限；null 不限制 |
| `tokens_per_minute` | 按预留量与已知 usage 估算的滚动 token 防护；null 不限制 |
| `reserve_tokens` | 每次请求开始预留的 token 估计值，不是输出截断长度 |
| `total_token_budget` | 本 suite 该账号的估算总预算，包含恢复前已保存调用 |
| `omit_max_tokens` | true 时不发送 max_tokens / max_completion_tokens |

RPM 和并发是当前调度器的硬限制；不限制生成长度时，token 预算只能是**估算防护**，单次响应可能超出预留量。缺失 usage 不当作 0，按预留量保守记账；本地统计不是服务商余额。按自己服务商/学校账号的实际限额填写，不能套用其他平台的数值。

`extra_body` 的思考开关和 Agent 架构是两层设置：direct/passthrough 通常单次决策请求，CoT/ToT/投票/批评等可能多次调用。原生思考与这些流程同时开启会叠加耗时和 token。默认保留每个 Agent 原来的内部流程，不将它们强行压成 direct。无 max_tokens 仅表示客户端不设截断值；服务端仍有自己的上下文、输出和时限限制。实际 `qwen` 别名背后的模型、思考开关是否生效、是否支持图片，需要在你的接口上检查 pilot 响应，离线测试不能证明这些服务端事实。

流式响应持续保存推理和答案，不是逐块重复提交请求。usage 按一条请求最终累计值统计，不累加各块。断开的流可能收不到 usage，参见 [OpenAI 流式协议](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events)。如果 endpoint 不返回标准 SSE、正常 finish_reason 或最终答案，程序停下保存证据，不静默改成另一种调用再扣一次额度。

## 4. 固定 pilot 与一键运行

```bash
python -B -m minibench.cloud prepare \
  --config config/cloud/xiangqi-pilot.yaml --output runs/cloud-pilot-8
python -B -m minibench.cloud status --output runs/cloud-pilot-8
```

以上**不调用模型**。从已提交题库冻结固定小样本：seed=42；D3/H2/M2 各 6 条，C2 保持 3 个场景的标准/焦点规则配对，共 6 条。H2 两组、M2 文字/中文图片两组，共 36 个 direct 条件；不测拉丁图。`samples` 保存题号和完整可读投影，`inputs` 保存规范原始数据、C2 证书及图片，`plan` 记录各 Agent 的同一名单、模式、配置、代码/数据指纹。pilot 是耗时探针，不应用 6 题给模型排最终名次。

运行整轮：

```bash
bash scripts/cloud_run.sh config/cloud/xiangqi-pilot.yaml runs/cloud-pilot-8 --workers 8
```

**该命令包含付费请求。** 当前 C2 100 题具备完整运行时证书，无需绕过认证。程序在任何付费调用前核对冻结的数据/代码/策略指纹，并重新计算所选 C2 获胜路线的实际对手响应；不匹配即停止。不要给正式运行加 `--allow-uncertified-c2`。

想保持 SSH 断线后运行，可在 `tmux` 会话中执行上述命令；关闭会话或服务器会影响执行，云端不等于永不掉线。不要同时启动两次相同结果目录，运行锁会阻止冲突。

每个 job 是一个完整题目在一个 Agent/模式下的交互，**H2/C2 同一局内永远顺序执行**；并发仅发生于不同 job。不同组是独立对局，不共享聊天记忆。对手走法、每次 prompt/图像请求、模型输出和原生评测记录均保存。

8 路结束后查看 `summary.json` 的均值、median、p95 和 usage，以及整轮 wall time。先确认完整输出、图片能力、超时/限流情况，再复制配置把 workers 和该账号 max_in_flight 调成 16，使用**新的目录**跑同一名单；不要直接改已冻结 plan。CLI `--workers 16` 只改调度并发，不会自动提高账号 max_in_flight。

`config/cloud/xiangqi-all-agents.yaml` 运行九 Agent 的完整提交题库：D3/H2/C2/M2 各 100 条；H2 两组、M2 两组，总计 5,400 个条件（900 + 1,800 + 900 + 1,800）。规范 YAML 的 limit=100 表示运行已固定的全部 100 条，不是从旧 250 条中取前 100。每个 Agent 的题号和组间题号完全相同，不重新随机抽题。`evaluation_samples/formal` 的 30/20 条小名单仅为兼容的辅助探针，不是这个全 Agent 计划的输入。

## 5. 保存、暂停与恢复

```bash
# 查询（无模型调用）；运行锁占用时可直接读取 state.json / summary.json
python -B -m minibench.cloud status --output runs/cloud-pilot-8

# 正常暂停：运行终端按 Ctrl+C。不要用 kill -9。

# 重新设置临时 key 后，用原命令和原目录续跑
bash scripts/cloud_run.sh config/cloud/xiangqi-pilot.yaml runs/cloud-pilot-8 --workers 8
```

恢复以 `jobs/<key>/result.json` 的原子提交和指纹为准，不以日志的 DONE 字样或 predictions 行数猜测完成。已完成条件直接跳过；未完成对局会用已保存成功模型回答和已保存 H2 对手分析重放前缀，只对缺失调用发起新请求。即使程序在通知父进程之前退出，已提交结果仍能识别。更改源码、冻结数据、引擎或计划会被拒绝，不能混入旧结果。

逐题结果与成功调用立即保存；派生的 predictions/汇总表约每 30 秒批量重建，并在正常结束、暂停或故障退出时重建。进度 state 持续更新。这样避免大型实验每完成一题就重读/重写整轮文件；运行中看完成数以 state 和 job 断点为准，汇总表可能短暂滞后。

这不是服务端事务：网络中断时，服务器可能已经生成或扣费但客户端未收全。未完成回答不评分、不当作成功缓存，人工恢复可能再次扣费；**不保证这一边界绝对零重扣**。不自动重试 429/HTTP/余额/引擎错误。错误保留在 job/state、failed_observation、calls/attempts，整体停下等待处理，不把基础设施故障计作模型答错。

SIGINT/SIGTERM 会取消新请求、尽量关闭在途流并保存已有内容；强制断电时只能保留最后成功落盘的内容。暂停后，断开的服务端请求是否仍在运行未知，恢复前应查看供应商后台，避免其外部在途量与本地新调度叠加。

## 6. 输出和柱状对比图

| 位置 | 内容 |
| --- | --- |
| `plan.json` / `inputs` / `samples` | 冻结配置、代码/数据指纹、题号、原始题目与图片 |
| `state.json` / `summary.json` | 运行进度、完成数、耗时分布、已知及未知用量 |
| `jobs/<key>/calls` | 成功调用缓存、请求正文、流式块、失败尝试，不含认证头 |
| `jobs/<key>/engine` | H2 对手分析缓存，续跑不改变已有应对 |
| `jobs/<key>/result.json` | 每个完成 job 的唯一权威结果 |
| `results/<任务>/<Agent>` | 原生 predictions、results、resolved_config |
| `charts` | 按任务/组别/指标的 PNG、SVG、comparison.json/csv |

完整运行结束自动导出象棋柱状图。需要中途画图，暂停后运行：

```bash
python -B -m minibench.cloud plot --output runs/cloud-pilot-8
```

未完成条件显示 N/A/覆盖数，不补零。主对比图仅展示 D3 的将杀率/合法率；H2 的限长内将杀率/题均合法率/CP loss；C2 的限长内将杀率/题均合法率/CPE loss；M2 的一步将杀率/合法率/棋盘识别准确率。参考步数内是否将杀仍保存在多步逐题诊断，不增加主图指标。非法着计入合法率和将杀失败，但不被赋予臆造的 CP/CPE loss；基础设施错误为缺失，不当作模型答错。

M2 在同一次回答中返回 move 和初始棋盘的 pieces 坐标表，识别准确率为正确的棋子类型/阵营/坐标数除以真实与预测非空格的并集大小。漏子、错色、错类型和多报棋子均扣分；缺失或畸形识别表为 0。文字组是转录对照，图像组才是视觉识别。这个新输出协议不能与过去只回答走法的成绩直接混合。

H2 的 Pikafish CP 与 C2 规则感知 CPE 分开图，不作统一单位排名。H2 的 mate 分值使用 sentinel 换算，可出现远大于普通局面分的 loss，不应解读为同尺度真实棋子价值差。

云端统计图采用可移植英文字体标签，避免棋子专用的中文子集字体缺字；中文棋盘图片不变。指标 key 和本节中文解释一一对应。

原有 `minibench score-suite` 仍使用同学的新评分系统；云端不重新定义它。为云端正式汇总填写真实结果及其原始配置、数据/引擎指纹，不能直接把历史 Windows manifest 当作服务器条件。

## 7. 新代码职责

- `src/minibench/cloud.py`：选题冻结、进程调度、原生评测调用、校验断点、结果汇总。
- `src/minibench/core/rate_limit.py`：共享账号并发/RPM与估算 token 预算，包括续跑前用量恢复。
- `src/minibench/core/streaming.py`：SSE 保存、成功调用重放、错误脱敏、H2 引擎重放。
- `src/minibench/visualization/xiangqi_suite.py`：象棋分任务/分组对比图，不合并 CP/CPE。
- `scripts/cloud_bootstrap.sh` / `cloud_pikafish.sh` / `cloud_run.sh`：环境安装、固定引擎构建、运行与自动导图。
- `config/cloud`：可提交的空密钥模板；`local*.yaml` 为忽略的私有配置。
- `tests/test_cloud*.py`：仅模拟接口和离线测试，不需真实 key、不扣用户额度。

其他任务继续通过 `factory.experiments` 的原生加载器、评测器和汇总器执行。通用云端入口可添加已登记家族的 experiment YAML；默认 pilot 仅象棋，不等于已经验证所有服务商能跑所有其他任务。多模态需要视觉支持，DeepSeek 文本账号不能分配图片任务。

## 8. 必须保留的学术与发布边界

1. **提交的 C2 100 题均通过实际运行时认证。** 证书逐步重算 depth=3、UCI 升序同分择着的对手应对，并由独立棋规核验每步合法性、完整状态及严格将死。它证明在登记上限内存在获胜续着，不证明所有防守下的强制杀或最短杀。认证路线不发送给实验 Agent。
2. H2 的深度 16/20 参考将死线不是“所有防守下最短强制杀”的数学证明。难度是参考杀距离分层，不应宣称严格最短步数。提交 H2 短/中/长为 34/33/33；C2 96/100 条参考距离 <=5 次 Agent 落子，全部排除了活动规则下的一步杀。
3. D3/M2 的 100 个局面完全配对，不能当成 200 个独立题。C2 是 50 个标准/焦点变体配对场景，50 条标准、17 条车变体、17 条马变体、16 条兵变体；其中 20 个来源也在提交 H2 中，另外 30 个来源仅保留为来源证据。父题池 125 个 C2 场景中有 12 个未获得完整配对证书，按预算未知排除，因此本轮 C2 存在认证筛选偏差。等量难度抽样也是人为设计分布，不代表父池总体权重。公开数据存在训练污染可能，需披露。
4. 顶级 MiniBench 许可证尚未确定，不擅自填 MIT/Apache。CCPD 的 CC-BY-4.0、字体 OFL、ZeroEval Apache 及外部 Pikafish GPL 声明分别保留。用户已确认许可证待定；上传代码不等于外部使用者已取得所有复用许可。
5. 离线通过只能说明当前覆盖范围内未发现错误，不承诺永无 bug，也不证明服务商端的模型身份/思考/图片支持。实际云端性能与 API 兼容性需由付费 pilot 测得。

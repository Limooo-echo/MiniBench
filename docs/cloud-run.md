# 云服务器运行与断点恢复

本说明针对 `MINIBENCH-FINAL` / `cloud-20261010` 分支。它合并 `mahjong-checkpoint-20261010` 的非象棋任务及评分系统，与本地新版象棋代码和四套固定分层抽出的 100 条提交题库；不包含旧实验结果。原本地仓库不受影响。**新增云端入口不会自动启动，也不会保存 API key。** 当前题号/抽样依据见 [`submission_manifest.json`](../data/xiangqi/submission_manifest.json)，记录约束见 [`schema.py`](../src/minibench/datasets/xiangqi/schema.py)。

本地后续修改尚未提交时，GitHub分支不代表本机最新内容。以当前工作区导出的源码包为准；
不要把本地保留且已被忽略的runs/output诊断证据打包或提交。

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
```

技术门禁通过不等于整个科学发布就绪；见第 8 节。
当前H2/C2使用 `evaluation.solution_tree_dir` 指定的完整最短解法树，逐题匹配FEN、规则、
树文件哈希和认证的最短步数，不把题内参考PV或旧depth-3对手证书冒充最短解集。
`certify_xiangqi_c2.py` 是旧depth-3策略条件的认证工具，不是新版最短树的认证入口。
仅作为候选诊断保留，必须显式选择工作目录输出，禁止写入正式`data/xiangqi`。
新版验证与解法树边界见 [象棋协议](xiangqi-shortest-protocol.md)。验证器搜索预算耗尽是未知，
不能把它当作题目无解；正式题库不接受未完成的树。

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

### 3.1 每次最终走法的15分钟硬期限

四任务的正式与冒烟YAML均显式设置 `evaluation.final_move_timeout_seconds: 900`。
独立`run-config`入口也开启思考，`provider.omit_max_tokens: true`使Agent各内部阶段
均不发送输出token上限（其他任务默认行为不变）；云端保留同名总开关。
计时范围是一次完整的 `Agent.generate*()` 决策：CoT规划、ToT分支、投票、批评、修复及重试
共享900秒；思考流、中间答案和心跳不重置计时。收到完整最终回答后才结束计时，返回的非法
或畸形走法仍按原规则判错。H2/C2每次实际下一步重新计时，不是整局只给15分钟。
批次排队、棋盘渲染和裁判搜索不在该决策计时内；Agent内部的API排队属于决策耗时。

超时保存 `termination_reason=final_move_timeout`、`status=invalid`、将杀失败和部分流。
不虚构走法或CP/CPE loss；合法率将未交付走法记为未成功提交合法着，原因与棋规非法区分。
这是已声明的实验判负条件，不是接口故障缺失；该job正常落盘，其他题继续，续跑也不重测它。
适用于Linux/WSL的主线程工作进程；不能保证关闭客户端连接后服务商立即停止生成或扣费。
这是时间上限，不是token上限。新规则不倒推重标旧实验；源码更新后应准备新的suite。

## 4. 固定 pilot 与一键运行

```bash
python -B -m minibench.cloud prepare \
  --config config/cloud/xiangqi-pilot.yaml --output runs/cloud-pilot-8
python -B -m minibench.cloud status --output runs/cloud-pilot-8
```

以上**不调用模型**。从已提交题库冻结固定小样本：seed=42；D3/H2/M2 各 6 条，C2 保持 3 个场景的标准/焦点规则配对，共 6 条。H2 两组、M2 仅中文图片，共 30 个 direct 条件；M2与D3同题，不额外测文字组或拉丁图。`samples` 保存题号和完整可读投影，`inputs` 保存规范原始数据与解法树，`plan` 记录各 Agent 的同一名单、模式、配置、代码/数据指纹及决策期限。pilot 是耗时探针，不应用 6 题给模型排最终名次。

运行整轮：

```bash
bash scripts/cloud_run.sh config/cloud/xiangqi-pilot.yaml runs/cloud-pilot-8 --workers 8
```

**该命令包含付费请求。** H2/C2均在任何任务开始付费调用前预检完整树、冻结数据、代码、哈希与认证步数，不匹配即停止。C2树内防守按认证策略，树外使用规则感知depth-3兜底与CPE裁判。旧depth-3诊断兼容开关不能跳过新版树预检，正式入口不再展示该开关。

想保持 SSH 断线后运行，可在 `tmux` 会话中执行上述命令；关闭会话或服务器会影响执行，云端不等于永不掉线。不要同时启动两次相同结果目录，运行锁会阻止冲突。

每个 job 是一个完整题目在一个 Agent/模式下的交互，**H2/C2 同一局内永远顺序执行**；并发仅发生于不同 job。不同组是独立对局，不共享聊天记忆。对手走法、每次 prompt/图像请求、模型输出和原生评测记录均保存。

8 路结束后查看 `summary.json` 的均值、median、p95 和 usage，以及整轮 wall time。先确认完整输出、图片能力、超时/限流情况，再复制配置把 workers 和该账号 max_in_flight 调成 16，使用**新的目录**跑同一名单；不要直接改已冻结 plan。CLI `--workers 16` 只改调度并发，不会自动提高账号 max_in_flight。

`config/cloud/xiangqi-all-agents.yaml` 运行九 Agent 的完整提交题库：D3/H2/C2/M2 各100条；H2两组、M2仅中文图片，总计4,500个条件（900 + 1,800 + 900 + 900）。C2为50个场景的标准/变体配对，共100条；不是每个场景重复四次。规范YAML的limit=100表示运行已固定的全部100条，不是从旧250条中取前100。各Agent使用同一名单，D3/M2同局面对照，不重新随机抽题。`evaluation_samples/formal`保存四库的相同100条完整名单；smoke名单才是小样本探针。

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

这不是服务端事务：网络中断时，服务器可能已经生成或扣费但客户端未收全。基础设施故障的不完整回答不评分、不当作成功缓存，人工恢复可能再次扣费；**不保证这一边界绝对零重扣**。不自动重试429/HTTP/余额/引擎错误；故障保留并停下处理，不计作模型答错。明确的900秒最终走法超时不同：它正常保存为判负并继续其他job，不当作基础设施故障。

SIGINT/SIGTERM 会取消新请求、尽量关闭在途流并保存已有内容；强制断电时只能保留最后成功落盘的内容。暂停后，断开的服务端请求是否仍在运行未知，恢复前应查看供应商后台，避免其外部在途量与本地新调度叠加。

故障收尾时，空闲工作进程通过进程池正常退出，不在共享队列读锁内直接结束进程。在途进程先保存部分响应；若 15 秒内仍未退出，收尾看门狗强制结束本轮自己的工作进程组（包括所属引擎），并在 `state.json.forced_worker_pids` 留下记录。强制结束可能收不到最终 usage，不能据此断言服务端已停止扣费。

`summary.json.usage` 按实际开始的调用尝试去重，成功缓存重放不增加调用次数。`known_success_tokens` 是成功回答的已知用量，`known_failed_tokens` 包含空答案等失败响应已返回的用量，`known_total_tokens` 是二者之和；`attempts_without_usage` 单列缺失用量的尝试，绝不当作零。对应的 `calls/attempts/*/accounting.json` 与流式证据在人工恢复时保留；恢复后的新尝试另记一次，未开始发送的排队请求不计用量。这些数字不是账户余额或服务商账单。

这些已知字段只接受通过一致性校验的服务端用量；例如reasoning大于completion的响应会被隔离。
有未知调用时完整 `total_tokens` 为null，已知小计另列；全部未知时小计也为null，不以0替代。
原始usage不改写，缺少官方单价时费用未知。

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

未完成条件显示N/A/覆盖数，不补零。D3展示将杀率/合法率；H2展示最短步数内和最短+2内将杀率、合法率、CP loss；C2对应展示两个将杀窗口、合法率、CPE loss；M2展示一步将杀率/合法率/识图准确率。900秒超时与非法回答计入失败，原因分别保留；两者都不赋予臆造的CP/CPE loss。基础设施错误仍是缺失。

M2在同一次回答中返回move和初始棋盘的pieces坐标表，识图准确率按正确类型/阵营/坐标数除以真实与预测非空格的并集计算。漏子、错色、错类型和多报棋子均扣分；缺失或畸形识别表为0。D3也输出同样的坐标表但不计识图成绩，作为同题文字对照；M2仅运行图片，不再额外增加文字组。这个协议不能与过去只答走法的成绩直接混合。

H2 的 Pikafish CP 与 C2 规则感知 CPE 分开图，不作统一单位排名。H2 的 mate 分值使用 sentinel 换算，可出现远大于普通局面分的 loss，不应解读为同尺度真实棋子价值差。

云端统计图采用可移植英文字体标签，避免棋子专用的中文子集字体缺字；中文棋盘图片不变。指标 key 和本节中文解释一一对应。

原有 `minibench score-suite` 仍使用同学的新评分系统；云端不重新定义它。为云端正式汇总填写真实结果及其原始配置、数据/引擎指纹，不能直接把历史 Windows manifest 当作服务器条件。

### 6.1 全轮结束后清理 M2 生成的 PNG（可选）

默认保留图片证据。希望自动清理时，在一键命令末尾添加开关：

```bash
bash scripts/cloud_run.sh config/cloud/xiangqi-all-agents.yaml runs/cloud-all-8 \
  --workers 8 --cleanup-m2-images
```

只有评测和导图都成功返回后才进入清理。工具还会独立核验：同一冻结计划包含 D3/H2/C2/M2，
全部题目、Agent 和组别都有有效完成断点，整轮状态为 completed、无在途请求，且没有运行锁冲突。
错误、暂停、缺题、断点损坏或只跑单个任务时都拒绝清理；未将杀或非法着但正常完成的题仍属于已完成。

已结束的一轮可先预览，再实际清理；以下命令不调用模型：

```bash
python -B scripts/maintenance/cleanup_xiangqi_m2_images.py --suite-dir runs/cloud-all-8
python -B scripts/maintenance/cleanup_xiangqi_m2_images.py --suite-dir runs/cloud-all-8 --apply
```

只删除与 M2 完成记录哈希一致的 `jobs/<key>/images/<task-id>/*_step00.png`，
不递归删除目录，不删除题库、输入资产、对比图、回答、评分或请求日志。符号链接、越界路径、
图片哈希不匹配或未知 PNG 会触发拒绝。清理情况保存到该轮 `m2-image-cleanup.json`，重复执行安全。
删除 PNG 不等于清除全部图片副本：请求日志里的 base64 图片编码继续保留，以便审计。
删除的 PNG 不进入回收站；图片哈希和渲染清单仍保留，通常可从请求中的图片数据还原。

该工具放在 `scripts/maintenance/`，属于实验结束后的维护工具，不改变当前评测源码指纹，
因此新增它不会单独阻止已冻结实验续跑。它核验保存的计划和结果，不重新调用评测器或 API。

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

1. H2/C2新版题库使用有界全宽AND/OR验证的最短步数与完整最短策略树，不把有限深度Pikafish或depth-3参考PV当作数学证明。正式库只接受已完成的树；预算耗尽是未知。认证范围是当前棋规实现与有界合法对弈，未实现官方长将长捉的完整历史裁决，不能无条件外推。
2. H2/C2各100条，分别按最短2–3步和4–5步各50条组织；标准/变体分别认证。同局面变规则后可能跨档。树外击败有限深度兜底对手，只能证明这局赢了，不能据此说树外走法在所有防守下必胜。最短与最短+2窗口分开报告。
3. D3/M2完全同题，不能当成200个独立局面。C2为50个配对场景，标准50条、车17条、马17条、兵16条。不同题库可能共享父局面或由中间局面派生，需按来源分组分析；认证筛选和人为等量难度设计不代表父池总体权重。公开数据存在训练污染可能，应披露。
4. 顶级 MiniBench 许可证尚未确定，不擅自填 MIT/Apache。CCPD 的 CC-BY-4.0、字体 OFL、ZeroEval Apache 及外部 Pikafish GPL 声明分别保留。用户已确认许可证待定；上传代码不等于外部使用者已取得所有复用许可。
5. 离线通过只能说明当前覆盖范围内未发现错误，不承诺永无 bug，也不证明服务商端的模型身份/思考/图片支持。实际云端性能与 API 兼容性需由付费 pilot 测得。

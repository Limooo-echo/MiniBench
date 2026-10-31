# Zebra 历史任务：结构化增量与 Pro

本版把 `incremental_state` 改为结构化候选域，并新增 `incremental_state_pro`。同一题在三种模式中使用相同背景、线索顺序和最终评分，研究逐轮校验能否减少错误状态对后续求解的影响。

## 三组协议

| 模式 | 中间轮次 | 下一轮可见内容 |
| --- | --- | --- |
| `deferred_reasoning` | 确认收到线索，最后求解 | 全部已见线索和确认历史 |
| `incremental_state` | 生成完整结构化候选域；格式合法即提交 | 全部已见线索与最近一次提交的状态 |
| `incremental_state_pro` | 生成相同候选域；格式合法且被已见线索支持才提交 | 全部已见线索与最近一次通过校验的状态 |

两个增量组使用相同提示、状态格式和上下文组装策略。它们的主要区别是语义门禁。每条线索只生成一次候选；拒绝后保留旧状态，等待下一条线索，不在同一轮修复或重试。被拒绝的候选、校验反例和校验器推导结果不会进入后续模型上下文。接受状态也不保留历次旧版本，避免把历史清理差异混进两组比较。

最终答案仍由模型根据全部线索和当前状态生成，使用原有完整网格评分。Pro 不让求解器代填最终答案。

## 状态含义与校验标准

状态是完整的 `domains` 映射：第一层为属性，第二层为该属性的每个值，列表为仍保留的房屋位置。例如，一个两房两属性的状态：

```json
{"domains":{"Name":{"Peter":[1,2],"Eric":[1,2]},"Drink":{"tea":[1,2],"milk":[1,2]}}}
```

真实请求需要覆盖题目中全部属性和值。初始状态为所有房屋位置；列表不得为空或含未知房号。两个增量组执行相同格式检查，格式错误均保留旧状态。关系线索保留在原始线索上下文中，第一版状态不另设关系断言语言。

Pro 的逻辑检查只使用背景和当前已收到的线索，要求它们蕴含候选域限制。若当前仍可能出现 `Peter=2`，候选却只保留 `[1]`，就拒绝；即使 `[1]` 与某个可行解一致，也不够。实现等价于检查是否存在满足已见线索、却落在候选域之外的解。

- `entailed`：没有这样的反例，可以提交。
- `not_entailed`：仍存在这样的反例，保留旧状态。这表示更新缺乏当前证据支持，不代表它一定与最终标准答案冲突。
- `solver_unknown`：校验器未能确定，例如超时；保留旧状态，单独记录，不能算成模型逻辑错误。

候选域不必已经完成全部可行的剪枝。全未知状态安全但可能无用，因此最终分析还应报告信息量与停滞情况。状态允许重新扩大候选域、撤回旧结论，普通增量组也有机会自行纠错；协议不强制单调删减。

标准答案和未来线索不参与 Pro 的在线校验。两个增量组均记录每步排除的房屋位置、是否包含标准答案及累计排除情况；该审计不进入模型上下文，也不决定普通组是否接受更新。Pro 的拒绝仍由前缀线索支持检查决定；“不被当前线索支持”和“排除了最终正确答案”是两个独立诊断。Pro 开始调用模型前会检查依赖和题目线索前缀的可满足性；缺少 Z3 或题目校验失败应先解决，不能当成模型失败继续评测。

最终评分保留整题正确率和格子正确率，不另定义过程分。合法率单独统计同一属性的行列匹配是否出现多个确定元素；候选域包含多个位置只是尚未确定，不按此定义判为不合法。

## 安装与离线检查

以下为仓库标准 WSL 环境：

```bash
cd /mnt/d/AAALimoWork/CS/Seminar/MiniBench
source ~/.venvs/minibench/bin/activate
python -m pip install -c constraints/ci-py310.txt -e '.[zebra-generation]'
python -c 'import z3; print(z3.get_version_string())'
python -m unittest discover -s tests -p 'test_zebra*.py'
```

这些测试不调用真实模型。Z3 固定依赖为 `4.16.0.0`。只有依赖安装需要访问包源；已安装后离线运行检查即可。

## 思考开／关对照

正式题集固定为 `data/zebra/history_eval.jsonl` 的 100 题，SHA-256 为 `609a08568e321f356163153c112421fb28522c97b29a8a18d33771e377442f1f`。不要替换为另一轮 direct/codebook/counterfactual 使用的 `paired-20261027` 题集。

| 参数 | 两份对照配置 |
| --- | --- |
| 模型与区域 | `qwen3.8-flash`，DashScope 北京 |
| Agent | `direct`，prompt v2，格式修复 0；每条线索一次调用，最后一次求解 |
| 请求温度 | `0.0` |
| 状态／ACK／最终答案上限 | `2048`／`32`／`4096` token |
| Pro 校验超时参数 | `state_validation_timeout_ms: 10000` |
| HTTP 超时／自动重试 | `600` 秒／`0` |
| 历史思考复用 | `preserve_thinking: false` |
| 传输 | `stream: false`，不设置 `stream_options` |
| 思考开启组 | `enable_thinking: true`，`thinking_budget: 7168` |
| 思考关闭组 | `enable_thinking: false`，不传 `thinking_budget` |

这两份配置使用 Direct 的 phase-aware 接口：中间轮次转发状态／确认请求，最终轮次使用 Direct v2 的提示包装；格式修复关闭，每轮只调用一次模型。`trace: full` 保存 Agent 包装后的请求与响应，便于与 `history_trace` 的协议输入对照。它们不使用静态配对实验中两次物理请求的 `ZebraQwenClient`。根据阿里云官方文档，商业思考模型支持非流式响应；现有通用客户端按完整 JSON 解析即可。[思考模式文档](https://www.alibabacloud.com/help/en/model-studio/deep-thinking)

`thinking_budget` 与 `max_tokens` 分别请求思考与可见回答预算；本配置不同时传 `reasoning_effort`。原生思考文本不写回下一轮上下文。服务端模型行为和实际用量应以保存的响应为准；温度 0 也不能保证逐次输出完全一致。[请求参数说明](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)

配置凭据后由用户运行：

也可以在 WSL 中运行 `bash scripts/run_zebra_history_wsl.sh`：启动器先做离线检查，再提示隐藏输入一次 API key，依次运行 Direct 的思考关闭与开启组。结果放在新建的 `runs/zebra-history-direct-qwen38flash-<时间>/` 下，包含 `thinking_off/`、`thinking_on/`、配置与源码快照、`console.log`、`batch-status.json`。密钥仅保留在运行进程环境中，不写入这些文件。`--preflight` 只检查、不调用 API。

```bash
read -rsp 'DashScope API key: ' DASHSCOPE_API_KEY
echo
export DASHSCOPE_API_KEY

minibench run-config config/experiments/zebra_history_thinking_off.yaml
minibench run-config config/experiments/zebra_history_thinking_on.yaml
```

两份配置使用不同的新运行目录，`run.on_existing: error` 防止覆盖。再次实验应复制配置并设置新的 `run_name`。需要恢复中断运行时，仅在配置和协议未变的前提下使用已有的 resume 机制；完成实例可复用，尚未完成实例可能需要从该题该模式重新调用，不能把这当作逐轮缓存恢复。

已有上述批次可用 4 路进程续跑：

```bash
bash scripts/resume_zebra_history_parallel_wsl.sh \
  /mnt/d/AAALimoWork/CS/Seminar/MiniBench/runs/zebra-history-direct-qwen38flash-<时间> \
  --workers 4
```

此入口使用批次保存的源码、配置和题集，校验指纹后只提交尚未完成的“题目 × 模式”。旧串行进程仍运行时，保留它的 1 路并新增 3 路；取得旧组的锁后升至 4 路。旧启动器随后可能因开启思考组已被接管而退出，这是预期交接。迁移期间以 `parallel-status.json` 为准；日志另存 `parallel-console.log`，逐项结果先写入 `parallel-journal/`，再更新原组目录的检查点。新进程需在终端重新隐藏输入密钥。并行调度时间记录在 `parallel-resume-manifest.json`，分析耗时应区分切换前后的并发条件。

`zebra_history.yaml` 仍保留 DeepSeek provider，提供同样的三模式入口。它不是上述 Qwen 思考开／关配对配置的一部分。

## 小样本与正式实验

先用 12 题检查状态格式、拒绝流程、截断率和调用预算，再冻结协议运行全部 100 题。冻结小样本按每个难度内 `sha256('20261027:' + task_id)` 升序取前 4 题，不参考旧成绩；12 题共 119 条线索。把同一组 ID 写入两份配置副本的 `task.task_ids`，并分别使用新的 pilot 运行目录。

每个思考设置都运行三模式，共六个条件。单次重复的预期物理请求数为：

- 12 题 pilot：`6 × (119 + 12) = 786`。
- 100 题正式：`6 × (978 + 100) = 6468`。

以上不含失败后人工恢复导致的重复请求，Pro 的本地校验不增加模型调用。它们是请求数，不是 token 用量或费用估计。状态上限已从旧协议的 512 提升至 2048；若 pilot 仍有截断，需要在正式实验前统一修改两个增量组预算并重新冻结协议，不能只为表现较差的一组补预算。

## 日志与分析边界

每个完成实例的结果包含 `history_trace`，用于保存中间候选与校验过程；其中 `request_messages` 和 `conversation` 是协议发送给 Agent 的输入。Direct 最终轮会追加提示包装，实际转换后的消息保存在完整 Runtime trace 的 `metrics.trace.stages[*].prompt` 中。被拒绝的候选只出现在审计轨迹中。主要轨迹字段包括轮次／阶段、请求消息、更新前状态、原始候选、格式检查、逻辑校验结果、提交决定、更新后状态，以及可获得的调用指标和完成信息。

`results.json` 的 `state_updates` 汇总各增量组的提交、拒绝、格式无效、未经线索支持与校验器未知计数；`incremental_pro_paired` 保存普通结构化组与 Pro 的配对四格表及 Pro 减普通组的成功率差。截断和空内容属于格式／答案失败，不自动重试；网络故障仍会中断并保存已经完成的实例。

普通增量组的在线语义校验状态为 `not_checked`；它不会因为语义问题阻止更新。要比较两组的中间错误，后续需使用同一校验器离线审计普通组，不能把 `not_checked` 当作正确。原始候选可能含模型文本，分享日志前应按实际数据需要处理。

主要终局比较是同思考设置下 `incremental_state_pro` 与 `incremental_state` 的逐题成功差；其次比较这个差在思考开／关之间的变化。建议同时报告格子准确率、可解析率、token 与耗时，并从轨迹计算首次错误、错误持续轮数、自行恢复、拒绝与停滞情况。这些诊断统计应以题目为配对单位，不能把同题的线索轮次当独立样本。

应区分三类情况：当前线索尚未支持、候选实际排除了最终真值、校验器无法确定。Pro 提升准确率只能直接支持“校验与回滚有益”；还需错误持续和恢复轨迹才能说明累计机制。100 题的线索顺序本轮固定，其他顺序和重复种子可作为后续稳健性实验，不混入本轮。

本协议把旧自由文本状态改为候选域，压缩了增量历史并提高状态输出上限。因此它是新的控制实验，不是 1013 或 1015 实验的精确复现；旧 57%／27%、76%／92% 只作背景，不能将新旧差值全部归因于校验器或思考开关。

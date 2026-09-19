# 象棋推理评测修复与验收记录

数据版本：`xiangqi-2026-09-19-r1`。协议版本：`xiangqi-reasoning-v2`。
修复基线为队友最新版 `eab452b069b40a1cfaec39fe2b3895d7503484f2`，在隔离分支 `codex/xiangqi-reasoning-fixes` 实施；按用户要求同步回原本地仓库，未 push，未运行付费语言模型评测。

## 最终数据

| 任务 | 最终数量 | 处理结果 |
| --- | ---: | --- |
| D3 | 250 | 棋盘与题号保留，重新穷举并独立核对全部一步将死答案 |
| C2 | 250 | 60 场景 × 4 规则 + 10 标准对照；旧候选池全部退役 |
| H2 | 250 | 短／中／长 80／80／90；每题深度 16、20 的两条完整路线均实际将死 |
| M2 | 250 | 从合格 D3 同源派生，棋盘、答案、来源与难度一一对应 |

C2 旧数据有 **232 条初始局面不合法**。新候选全部来自固定 CCPD 修订的合法棋谱重放，种子 `20260831`。三个焦点规则各 20 场景；四条件均满足唯一最优差值至少 0.5，且焦点变体确实改变最优走法。48 份实际使用的 PGN 原始字节随仓库归档，来源审核核对了 2,690 个前缀局面与 2,642 步走法。

H2 旧题中 200 条双深度证书有效。其中 14 条因参考距离变化导致步数上限改变，使用新 ID 并记录 `replaces_id`；另有 1 条与 C2 完全同局面，已替换。因此最终保留 185 个旧 ID，另有 14 个目标修订题和 51 个新局面。250 个来源文件内容各异。任务上限仍为 `2 × reference_mate_moves + 1` 个半回合。

首次完整门禁确实拦下了跨任务重复题 `xiangqi-history-0229`。修复后 H2 生成器默认排除 D3/C2 的全部 FEN，并记录、复查这些数据的哈希。旧失败报告保留在 [首次审核证据](../output/reviews/2026-09-19-repair/independent-validation-first.json)，最终合格报告不会掩盖该过程。

逐题保留／退役／新增及原因见 [repair_manifest.json](../data/xiangqi/repair_manifest.json)。旧文件按原始字节归档于 [legacy](../data/xiangqi/legacy/xiangqi-corpus-rebuild-2026-08-31/README.md)；旧 ID 映射只指向该归档，不会把旧答案接到新棋盘上。

## 代码与生成器

- 修复未过河退兵和退回己方河界后的规则，限制将帅所属九宫，校验士象位置、数量、照面和行棋方状态。
- 区分将死与困毙：搜索按象棋终局胜负处理，D3/H2/M2 的目标只接受严格将死。
- C2 固定三层搜索正确处理双方视角，接受全部等效最优着法，容差 `1e-6`。
- 标准规则通过 `cchess==1.25.5` 手工边界校准；变体使用独立慢速规则和搜索，不调用待测棋盘裁判。候选不足、规则分歧或非预期重复都会阻止最终发布。
- H2 固定 16/20 验证深度，拒绝困毙、截断、非法路线和引擎异常；缓存绑定引擎与规则代码指纹，输入池在开始／结束核对哈希。
- 统一 `ok / invalid / error` 与结构化错误阶段。解析失败、非法或答错记有效失败；API、裁判或必要对手故障记缺失，成功字段为 null。诊断引擎失败不抹去精确棋规已确认的成功。
- 图文使用相同回答机会；评测器不叠加模型重试。保留架构默认内部行为并记录成本。H2 仅历史模式必须支持完整消息历史。
- 逐题记录完整回答、动作和终止原因；运行前保存配置与选题，逐题 checkpoint，同名结果禁止覆盖。
- Pikafish 固定 Threads=1、Hash=128MB，每次独立分析清缓存；超时清理进程及旧输出。正式对手深度 16，D3/M2 诊断深度 8；配置核对实际二进制和 NNUE 指纹。

## 评分与实验清单

象棋仍为 `s=Y，P=0`，旧过程综合分只用于诊断。D3 各难度组等权；C2 三个非标准规则组等权；H2 仅历史模式各参考距离组等权；M2 两个图片模式各占一半。标准 C2、H2 完整棋盘、M2 文字模式作为对照。

象棋在 D／R／H／V 的权重保持 **0.20／0.35／0.20／0.35**。已修复已知调用缺少用量状态时的覆盖率分母。缺失不补零、不重新分配原有权重；不完整维度保留缺失及覆盖率。

[新版清单](../config/scoring/manifest.example.yaml) 与 [旧版归档清单](../config/scoring/manifest.legacy-v1.example.yaml) 均可载入，失效数据引用和哈希已修复。空 runs 不生成真实模型排名。

[冻结选题](../data/xiangqi/evaluation_samples/manifest.json)：小样本 D3/H2/M2 各 6 个局面、C2 3 场景；正式各 30 个局面、C2 10 场景。分别产生 48／220 个条件观测。D3/M2 同源配对；同一基础模型的各架构共用名单，各自默认架构配置应登记后复用。

## 验收结果

- 完整独立门禁：**1,000 条全部通过**，四个任务家族均 valid，最终 `full=true / release_ready=true`，无非预期跨家族重复。
- 全套回归：**508 项全部通过**，包含原有回归与本次反例。
- 本地真实引擎：四类任务共 **28 条结果、30 个断言全部通过**；涵盖正确、合法但错误、无效格式、API 超时、诊断故障、必要对手故障和状态隔离。
- 实际 smoke/web `--prepare-only`：**27 项检查通过**，该步骤 0 模型调用、0 引擎进程启动。
- 技术发布审计通过，冻结配置、数据、来源、验证源码的哈希一致；全流程没有语言模型调用。

证据入口：[验收汇总](../output/reviews/2026-09-19-repair/acceptance-summary.json)、[独立门禁](../data/xiangqi/independent_validation.json)、[H2 双路线](../data/xiangqi/history/validation_report.json)、[真实引擎验收](../output/acceptance/xiangqi-protocol-final2/protocol_acceptance.json)、[回归日志](../output/reviews/2026-09-19-repair/unittest-final-2.log)。

这些验收证明的是列出的规则与协议性质。C2 是固定三层效用表现；H2 两条引擎参考线不证明最短将杀或覆盖全部防守；图文／历史差异不是单独的识别／记忆准确率。原仓库顶级许可证仍未声明，外部来源审计保留该状态，技术门禁已单独通过。

## 在本地复验

当前已配置的 WSL 虚拟环境可以使用以下命令（仓库根目录，Bash）。固定引擎及 NNUE 同步在被 git 忽略的 `tmp/resources/pikafish/` 中：

```bash
export PYTHONPATH=src
export PIKAFISH_PATH="$PWD/tmp/resources/pikafish/Pikafish-Windows-x86-64-universal.exe"
.venv-wsl/bin/python -m unittest discover -s tests
.venv-wsl/bin/python scripts/validate_xiangqi_data.py --workers 6 --output data/xiangqi/independent_validation.json
.venv-wsl/bin/python scripts/audit_xiangqi_release.py --technical --output data/xiangqi/release_audit.json
.venv-wsl/bin/python scripts/run_xiangqi_smoke.py --prepare-only --pikafish-path "$PIKAFISH_PATH"
.venv-wsl/bin/python scripts/accept_xiangqi_protocol.py --output-dir output/acceptance/my-new-check --pikafish-path "$PIKAFISH_PATH"
```

每次验收用新的输出目录。新环境可安装 `pip install -e '.[xiangqi-generation]'`。完整的数据重建命令、固定 CCPD 修订、效用函数与引擎指纹见 [数据卡](xiangqi-data-card.md) 和 [provenance.json](../data/xiangqi/provenance.json)。运行真实模型前需要配置自己的 provider 凭据；本次未发起该步骤。

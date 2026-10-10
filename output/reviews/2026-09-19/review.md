**MiniBench 最新象棋代码与四维评分审核 — 2026-09-19**

结论：当前版本还不适合用于正式架构排名。四维评分公式和分层聚合的主要设计成立，但象棋规则数据存在错误标准答案和非法局面；此外，异常分类、图文重试策略及评分清单存在可复现的问题。需要先修复题目与评测，再重算分数。仅调整四维权重不能解决这些问题。规则变体可以改变标准象棋规则；本报告核对的是题面所声明的新规则、程序实际执行的规则及标准对照是否一致，不要求变体一律遵守标准棋规。

审核版本为 GitHub `Limooo-echo/MiniBench` 主分支 `eab452b069b40a1cfaec39fe2b3895d7503484f2`。本地原工作区仍在 `daeadc0`（Scoring design），审核使用隔离副本。对比范围为 `daeadc0..eab452b`，其中最新提交回退了前一个 web-test feedback 改动；报告按最终生效代码判断。

[审核提交](https://github.com/Limooo-echo/MiniBench/commit/eab452b069b40a1cfaec39fe2b3895d7503484f2)

1. **[P1] 规则题把未过河兵后退判成满分，题面与裁判冲突。**

   规则卡明确写“过河后可后退”，但 `_gen_soldier` 只判断 `free`，不判断 `crossed`。新生成器把红兵放在 d2，当前数据有 20 条 soldier-free-retreat 标准答案是 d2d1 或 f2f1，均是未过河兵后退。

   实际复现 `xiangqi-rule-variants-c2-0164`：提交 `{"move":"d2d1"}` 得 `legal=1, success=True, score=1`。只在内存将条件改为 `free and crossed`，相同答案立即变为 `legal=0, success=False, score=0`。这不是权重偏好，而是成功标签错误。底层棋规缺陷早已存在，此次新数据将它系统性写进 oracle。

   修复应同时包含棋规、生成器、全部受影响数据与 oracle 的重新验证。只改代码而沿用旧标准答案不够。

   位置：[board.py:192](https://github.com/Limooo-echo/MiniBench/blob/eab452b069b40a1cfaec39fe2b3895d7503484f2/src/minibench/datasets/xiangqi/variants/board.py#L192-L196)，[规则卡](https://github.com/Limooo-echo/MiniBench/blob/eab452b069b40a1cfaec39fe2b3895d7503484f2/src/minibench/datasets/xiangqi/variants/rules.py#L77-L79)，[数据第164行](https://github.com/Limooo-echo/MiniBench/blob/eab452b069b40a1cfaec39fe2b3895d7503484f2/data/xiangqi/rule_variants/tasks.jsonl#L164)。

2. **[P1] 规则变体数据中 165/250 条黑将在九宫外。**

   新 horse/soldier 模板把黑将设置为 `b[0][6]`，即 g9；镜像后是 c9，均在九宫 d/e/f 文件之外。该版本四张规则卡都没有允许将离宫。数据检查确认 165 条记录、45 个独立 FEN 存在此问题，包括 45 条 standard 对照和每种非标准规则各40条。

   例如第一题 FEN 以 `6k2` 开头，黑将就在 g9。现有 schema 验证了将的数量、FEN 可解析等，却未阻止这种基础位置非法。搜索器能对该数组计算走法，不等于题目满足其声明的棋规。

   应修模板并补足基础棋子位置约束，再重新生成同局面配对数据。本次只量化了将出九宫这一确定问题，未把“165条”解读为其他棋子全部合法。

   位置：[生成器第20行](https://github.com/Limooo-echo/MiniBench/blob/eab452b069b40a1cfaec39fe2b3895d7503484f2/scripts/generate_xiangqi_c2_candidates.py#L18-L21)，[第35行](https://github.com/Limooo-echo/MiniBench/blob/eab452b069b40a1cfaec39fe2b3895d7503484f2/scripts/generate_xiangqi_c2_candidates.py#L33-L36)。

3. **[P1] 服务或引擎故障被离线评分当成模型答错。**

   D3 捕获 `agent.generate` 异常后仅打印日志，将答案写成 EMPTY，并保存 `goal_achieved=False`，没有保留结构化异常。H2 会保存 `llm_error:...` 或 `pikafish_error`，但适配器的异常前缀清单没有这两类。

   本地固定替身抛出 TimeoutError 后，真实 D3/H2 evaluator 产生的记录都被 `score_record` 计算为 `status='ok', y=0, p=0`。H2 的 `pikafish_error` 同样如此。与之对照，已有 `error:timeout` 能正确得到 `status='missing', y=None`。

   影响是基础设施故障既降低能力得分，又被计为已覆盖观测，违反“故障缺失、不补零”的评分规范。应在任务输出中统一保存异常状态，并由适配器一致识别；真实无效模型回答仍应计失败。对于已丢失异常字段的旧 D3 记录，需要原运行日志才能区分超时和真正空答。

   位置：[mate_in_one.py:269](https://github.com/Limooo-echo/MiniBench/blob/eab452b069b40a1cfaec39fe2b3895d7503484f2/src/minibench/datasets/xiangqi/mate_in_one.py#L269-L273)，[adapters.py:78](https://github.com/Limooo-echo/MiniBench/blob/eab452b069b40a1cfaec39fe2b3895d7503484f2/src/minibench/scoring/adapters.py#L73-L82)。

4. **[P2] H2 的三条参考轨迹结束于困毙，与严格将死目标不一致。**

   回放250条提交内保存的 `h2_analysis.principal_variation_uci`，247条终局是将死，3条是“未被将军、合法回复数为0”：`xiangqi-history-0025`、`0172`、`0184`。0025 的参考序列为 `e9e8 f2e1 e2e1 f1f0 e8e7`。

   生成器检查了两次引擎 mate 分数和 PV 合法性，却未验证最终是否满足运行时的 checkmate 判据。固定双方按0025的存档 PV 行动时，evaluator 得 `goal_achieved=False, reasons=['stalemate']`；旧过程分仍可达到100，说明为何不能用旧综合分替代 Y。

   应明确任务究竟要求“赢棋”还是“严格将死”。如果保留严格将死目标，应筛选/重新标注这些参考轨迹，并验证目标可达；如果接受困毙获胜，应同步修改题面和成功判据。这里证明的是现有 oracle 验证不足，不声称这三道题所有其他路线都无解。

   位置：[build_xiangqi_h2.py:140](https://github.com/Limooo-echo/MiniBench/blob/eab452b069b40a1cfaec39fe2b3895d7503484f2/scripts/build_xiangqi_h2.py#L130-L148)，[history.py:380](https://github.com/Limooo-echo/MiniBench/blob/eab452b069b40a1cfaec39fe2b3895d7503484f2/src/minibench/datasets/xiangqi/history.py#L380-L391)。

5. **[P2] 图像题解析失败可多答一次，文字对照只有一次。**

   M2 的 `generate_multimodal` 输出无法提取 UCI 时会再调用一次；text 分支无相同逻辑。使用“第一次返回 invalid、第二次给正确将杀”的同一个固定代理：文字模式调用1次、失败；中文棋子图像模式调用2次、成功。

   因此图文差值混入了额外采样机会，不能完整归因于输入模态。建议固定两边相同的回答/解析重试预算，并记录每次尝试。若将额外重试保留为系统特性，应明确其不同协议，不能直接作为纯图文对照。

   位置：[multimodal.py:382](https://github.com/Limooo-echo/MiniBench/blob/eab452b069b40a1cfaec39fe2b3895d7503484f2/src/minibench/datasets/xiangqi/multimodal.py#L382-L392)。

6. **[P2] 评分示例清单已经无法对应当前仓库数据与象棋协议。**

   实际 `load_suite(config/scoring/manifest.example.yaml)` 首先报 `zebra-direct: dataset sha256 mismatch`。完整核对共有9个条目哈希过期：三个 Zebra、直接/历史一笔画、四个象棋。Git blob 与审核副本逐字节一致，排除了本次 Windows 换行造成的误差；四个象棋条目的哈希明确对应升级前的 daeadc0 数据。

   象棋清单还保留旧 D3 max_plies=4、H2 full-state/depth8/固定20步、M2 旧多步参数；新协议实际是 D3 一步、H2 配对且depth16/每题步数、M2 一步。不能只批量刷新哈希后继续把旧协议当作真实运行来源。

   应保留旧清单用于旧数据，另外建立新版清单，登记真实题目范围、prompt/协议版本及选题清单，加入“仓库示例清单可载入”的集成检查。禁止绕过数据哈希校验。现有哈希保护能拦住错误混用，本身是正确行为。

   位置：[manifest.example.yaml:96](https://github.com/Limooo-echo/MiniBench/blob/eab452b069b40a1cfaec39fe2b3895d7503484f2/config/scoring/manifest.example.yaml#L96-L150)。

7. **[P2] 成本报告可能把只有1%用量记录的运行显示为100%覆盖。**

   `_costs` 只有在 `llm_calls` 和 `usage_missing_calls` 同时有效时才把调用加入覆盖率分母。两条记录分别为“1次调用且有token记录”和“99次调用、usage_available=False、未保存usage_missing_calls”，输出总调用100，却输出 `call_coverage=1.0`。

   缺用量状态的已知调用被从分母删除。应把它们计为未知用量，或明确整个覆盖率不可完整计算；本例应为1%而非100%。这不改变四维能力主分，但会误导成本比较和记录完整性判断。

   位置：[report.py:64](https://github.com/Limooo-echo/MiniBench/blob/eab452b069b40a1cfaec39fe2b3895d7503484f2/src/minibench/scoring/report.py#L64-L72)。

**评分体系的判断**

`Y + (1-Y) × 0.25 × P` 的逐题计算、先重复运行再题目/分层平均、固定任务权重、缺失不重新归一化、严格分独立覆盖、配对抽样和敏感性分析，未发现主要计算错误。象棋只以目标达成 Y 进入主分、合法率和引擎质量作为诊断，是合理的；D3新版也确实接受全部合法的一步将杀，而不只接受单个引擎首选。

`κ=0.25` 和任务权重属于评价取向，不是由数据唯一推出的客观常数。当前规范已经承认这一点，并保留严格分和敏感性报告，没必要为了形式引入AHP或熵权。报告应持续将四维解释为相应条件下的任务表现，避免称为独立测得的记忆、规则理解或视觉识别能力。

还需注明一个方法限制：经验百分位 bootstrap 在极小样本、全成功或全失败时可能给出[100,100]或[0,0]，不能解读为总体表现没有不确定性。仅设“每组至少两道独立题”不足以保证可靠区间；应对边界/低样本区间标注不足。它不是主分公式算错。

**验证与交付范围**

在隔离副本运行 `PYTHONPATH=src python -m unittest discover -s tests`，得到 **410 tests，OK，53.587s**。这些测试通过不能覆盖上面的问题：与自家规则引擎一致，不代表与题面棋规一致；当前测试也没有验证最新示例清单、真实错误前缀和图文同等重试预算。

额外运行了固定代理/引擎替身的异常及重试复现、250条C2位置检查、全部250条H2存档PV终局回放、数据哈希和成本分母检查。未调用付费模型，也未重新运行Pikafish深度16/20为所有数据重新出标签。本次只做审核，未修改业务代码、题目或评分权重，未向GitHub发布评论。

同目录 `reproduce.py` 保存主要复现，`checks.json` 保存机器可读输出，`unittest.log` 保存回归日志。推荐修复顺序：C2棋规与数据 → H2目标/验证统一 → 异常与重试协议 → 新评分清单 → 成本报告。修复后需要重新生成受影响数据并补相应回归，之后再接正式模型实验和四维雷达图。


# CI #47 失败修复

根因：主 test 作业只安装了基础包，没有安装 `xiangqi-generation` extra 中的 `cchess==1.25.5`。本地之前已手动安装该依赖，因此原本地验收未发现 CI 安装步骤遗漏。

GitHub 日志：508 项测试，3 失败、12 跳过。3 个失败均为 `No package metadata was found for cchess` 导致；不存在第二个已发现根因。

修改：主测试安装 `.[xiangqi-generation]`；CI 约束固定 cchess 版本；运行测试前执行 `calibrate_cchess()`，缺包、版本错误或独立规则校准失败会明确终止。图片可移植性作业继续使用基础依赖。README 安装说明已同步。

验证：新建未继承原环境的 Python 3.10.12 / WSL 虚拟环境，先按旧安装命令成功复现相同3个失败；随后直接读取并执行修复后的 workflow 安装和预检命令。独立校准通过，508 项测试全部通过、无跳过，pip check 通过。GitHub 当前 Python 版本为3.10.20，远程新配置仍需提交后由 Actions 验证。

未修改棋规、题库、答案或评分逻辑，也没有通过放宽测试绕过异常。修改留在本地，未 commit/push；重跑旧提交 cbcc338 不会应用此修复。

详细日志和哈希见 diagnosis.json。

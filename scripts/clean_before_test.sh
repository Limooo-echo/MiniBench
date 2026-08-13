#!/bin/bash
# 每次测试前清理: 删上次测试的日志/图片/结果/抽样题目 (保留 runs/test1 旧历史)
ROOT="/home/zyh/MiniBench"
WIN="/mnt/c/Users/LENOVO/Desktop/MiniBench"

echo "===清理上次测试结果==="
for base in "$ROOT" "$WIN"; do
    cd "$base" 2>/dev/null || continue
    # 删 baseline 时间文件夹 (上次测试结果)
    rm -rf runs/baseline-* 2>/dev/null
    # 删顶层结果目录 (d3_/c2_/h2_/m2_)
    rm -rf runs/d3_* runs/c2_* runs/h2_* runs/m2_* 2>/dev/null
    # 删日志
    rm -f runs/*.log 2>/dev/null
    # 删 m2 步图 (vis_outputs/m2_steps*)
    rm -rf vis_outputs/m2_steps* 2>/dev/null
    # 删抽样文件 (上次测试抽的题, 4 任务全删, 下次测试重新抽)
    rm -f data/d3/sample_*.jsonl data/c2/sample_*.jsonl \
          data/h2/sample_*.jsonl data/m2/sample_*.jsonl 2>/dev/null
done
echo "清理完成. runs 剩: $(ls "$WIN/runs" 2>/dev/null || echo 无)"
echo "vis_outputs 剩: $(ls "$WIN/vis_outputs" 2>/dev/null || echo 无)"
echo "抽样文件剩: $(ls "$WIN"/data/*/sample_*.jsonl 2>/dev/null | wc -l)"

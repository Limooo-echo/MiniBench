"""MiniBench 可视化模块.

按职责分组:
  象棋评测图表:
    d3_vis        — D3 一步杀评测图表 (逐任务得分/难度对比)
    h2_vis        — H2 多步杀评测图表 (short-mate vs long-mate, full vs agent_only)
    xiangqi_vis   — 象棋棋盘绘制 + 走子轨迹渲染
  其他题目可视化:
    mahjong_vis   — 麻将牌面绘制
    onestroke_vis — 一笔画图形绘制
  通用:
    renderer      — 通用渲染调度 (按 task family 分发到上述模块)
    plot_results  — 跨实验结果对比图 (多 agent / 多任务)
    run_vis       — 单次实验结果可视化入口
"""

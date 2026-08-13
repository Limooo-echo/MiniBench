import json
import os
import networkx as nx
import matplotlib.pyplot as plt

# 导入我们写的模块 (兼容作为脚本直接运行或作为包导入两种方式)
try:
    from onestroke_vis import draw_one_stroke_task
    from xiangqi_vis import draw_xiangqi_board, render_xiangqi_trajectory
except ImportError:
    from minibench.visualization.onestroke_vis import draw_one_stroke_task
    from minibench.visualization.xiangqi_vis import draw_xiangqi_board, render_xiangqi_trajectory

def fen_to_board(fen_str):
    """将真实的 FEN 字符串解析为脚本需要的二维数组

    映射与 minibench.datasets.xiangqi.engines.pikafish.PIECE_ID_TO_FEN 互逆：
      K=帅(1)  A=仕(2)  B/E=相(4)  N/H=马(6)  R=车(8)  C=炮(10)  P=兵(12)
    小写为黑方，取负值。
    """
    piece_map_upper = {'K': 1, 'A': 2, 'B': 4, 'E': 4, 'N': 6, 'H': 6, 'R': 8, 'C': 10, 'P': 12}
    piece_map_lower = {'k': -1, 'a': -2, 'b': -4, 'e': -4, 'n': -6, 'h': -6, 'r': -8, 'c': -10, 'p': -12}

    board = []
    # 只取 FEN 串第一部分（棋盘分布），按斜杠分割每一行
    rows = fen_str.split(' ')[0].split('/')
    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend([0] * int(char)) # 数字代表几个空位
            elif char.isupper():
                board_row.append(piece_map_upper.get(char, 0)) # 红方棋子
            else:  # lowercase
                board_row.append(piece_map_lower.get(char, 0)) # 黑方棋子
        board.append(board_row)
    return board

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']

def parse_agent_output(raw_output_str):
    """安全解析 Agent 返回的 JSON 字符串"""
    # 【新增这一行】：如果是空字符串，直接静默返回空字典
    if not raw_output_str:
        return {}
        
    try:
        return json.loads(raw_output_str)
    except json.JSONDecodeError:
        print(f"警告：无法解析的输出格式: {raw_output_str}")
        return {}

def load_task_data(task_family, task_id, base_dir="data"):
    """根据 task_id 去原始数据集中找对应的初始状态"""
    folder_map = {
        "os": "one_stroke",
        "xq": "xiangqi",
        "mj": "mahjong"
    }
    prefix = task_id.split("-")[0]
    folder = folder_map.get(prefix, "")
    
    if not folder:
        return None
        
    # 处理象棋的 hard_tasks
    if task_id.startswith("xq-hard"):
        task_file = os.path.join(base_dir, folder, "hard_tasks.jsonl")
    else:
        task_file = os.path.join(base_dir, folder, "tasks.jsonl")
        
    if not os.path.exists(task_file):
        return None
        
    with open(task_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data.get("id") == task_id:
                return data
    return None

def render_one_stroke_trajectory(task_data, path_list, save_path):
    """
    绘制一笔画 Agent 的解答轨迹（带箭头）
    """
    vertices = task_data.get("vertices", [])
    edges = task_data.get("edges", [])
    
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42) 
    
    # 画底层静态图（灰色）
    nx.draw(G, pos, with_labels=True, node_color='#EEEEEE', 
            node_size=1000, font_size=12, edge_color='#CCCCCC', width=2)
    
    # 画 Agent 走过的轨迹（红色箭头）
    if path_list and len(path_list) > 1:
        trajectory_edges = [(path_list[i], path_list[i+1]) for i in range(len(path_list)-1)]
        
        # 1. 高亮走过的节点 (先画，防止盖住起点/终点)
        nx.draw_networkx_nodes(G, pos, nodelist=path_list, node_color='#FFB6C1', node_size=1000)
        
        # 2. 标出起点和终点 (后画，确保颜色突出)
        nx.draw_networkx_nodes(G, pos, nodelist=[path_list[-1]], node_color='#F08080', node_size=1200) # 红色终点
        nx.draw_networkx_nodes(G, pos, nodelist=[path_list[0]], node_color='#90EE90', node_size=1200) # 绿色起点
        
        # 3. 专门使用有向图 (DiGraph) 来绘制带箭头的边
        traj_G = nx.DiGraph()
        traj_G.add_edges_from(trajectory_edges)
        nx.draw_networkx_edges(traj_G, pos, edge_color='red', width=3, 
                               arrows=True, arrowstyle='-|>', arrowsize=30, 
                               node_size=1200)

    plt.title(f"Trajectory for {task_data.get('id')}\nPath: {' -> '.join(path_list)}")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"一笔画轨迹图已生成: {save_path}")

def batch_render_trajectories(predictions_path, output_dir="vis_outputs"):
    """
    读取 predictions.jsonl，批量生成所有任务的轨迹图
    包含：一笔画、一步杀静态图、多步对战GIF
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"开始解析结果文件: {predictions_path}")
    
    count = 0  # 增加计数器
    max_tasks = 3  # 控制只画前 3 题，测试完可以改大
    
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= max_tasks:
                print(f"已完成 {max_tasks} 组画图，测试结束！")
                break
                
            record = json.loads(line)
            task_id = record.get("task_id", "")
            raw_output = record.get("raw_output", "")
            
            # 1. 忽略选择题
            if task_id.startswith("mb-choice"):
                continue
                
            # 2. 获取题目初始状态数据
            task_data = load_task_data("auto", task_id)
            if not task_data:
                print(f"提示：找不到 {task_id} 的初始数据，但这不影响我们画动图，继续！")
                task_data = {}  # 塞一个空字典给它，让它继续往下走
                
            parsed_output = parse_agent_output(raw_output)
            
            # ==========================================
            # 3. 处理一笔画任务 (One-Stroke)
            # ==========================================
            if task_id.startswith("os-"):
                agent_path = parsed_output.get("path", [])
                if agent_path:
                    save_name = os.path.join(output_dir, f"{task_id}_traj.png")
                    render_one_stroke_trajectory(task_data, agent_path, save_name)
                    count += 1  # 成功画了一个，计数器加1
            
            # ==========================================
            # 4. 处理象棋任务 (Xiangqi) - 适配真实 JSONL
            # ==========================================
            elif task_id.startswith("xq-") or "xiangqi" in record.get("tags", []):
                # 修复1：适配真实的 'raw_outputs' 列表
                raw_outputs = record.get("raw_outputs", [])
                raw_output_str = raw_outputs[0] if raw_outputs else ""
                parsed_output = parse_agent_output(raw_output_str)

                # 修复2：针对“一步杀(mate-in-one)”，直接绘制初始棋盘（静态图）
                if "mate-in-one" in record.get("tags", []):
                    initial_board = task_data.get("board", [])
                    if initial_board:
                        save_name = os.path.join(output_dir, f"{task_id}_initial.png")
                        # 借用单步画图函数，标注为 Mate-in-one
                        draw_xiangqi_board(
                            {"id": f"{task_id} (Mate-in-one)", "board": initial_board}, 
                            save_path=save_name
                        )
                        count += 1  # 成功画了一张静态图，计数器加1
                    continue  # 一步杀画完就直接进入下一行，不跑后面的动图逻辑

                # 修复3：动态多步残局处理 (人机对战动图)
                fen_history = record.get("fen_history", [])
                
                # 如果没有 fen_history，看看是不是直接存了 board_history
                if not fen_history:
                    board_history = record.get("board_history", [])
                else:
                    board_history = [fen_to_board(fen) for fen in fen_history]
                
                if board_history:
                    print(f"\n[{task_id}] 发现轨迹数据，共 {len(board_history)} 步，开始生成动图...")
                    render_xiangqi_trajectory(task_id, board_history, output_dir)
                    count += 1  # 成功画出一道动图题，计数器加 1
                else:
                    print(f"[{task_id}] 没找到动态轨迹数据，跳过。")

# --- 本地独立运行测试 ---
if __name__ == "__main__":
    # ⚠️ 请确保这里是你刚跑出来的、包含一笔画或象棋最新结果的 jsonl 绝对路径！
    sample_file = "/home/zyh/MiniBench/runs/siliconflow-glm52-mcq-test2/predictions.jsonl" 
    
    if os.path.exists(sample_file):
        batch_render_trajectories(sample_file)
    else:
        print(f"找不到结果文件：{sample_file}")
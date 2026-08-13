# src/minibench/visualization/onestroke_vis.py
import networkx as nx
import matplotlib.pyplot as plt
import json
import os

def draw_one_stroke_task(task_data, save_path):
    vertices = task_data.get("vertices", [])
    edges = task_data.get("edges", [])
    start_node = task_data.get("start")
    end_node = task_data.get("end")

    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42) 

    node_colors = []
    for node in G.nodes():
        if node == start_node and node == end_node:
            node_colors.append('#FFD700')
        elif node == start_node:
            node_colors.append('#90EE90')
        elif node == end_node:
            node_colors.append('#F08080')
        else:
            node_colors.append('#ADD8E6')

    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1200, font_size=14, font_weight='bold', edge_color='gray', width=3)
    plt.title(f"One-Stroke Task: {task_data.get('id')}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def visualize_onestroke_by_id(task_id, data_file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(data_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get("id") == task_id:
                output_path = os.path.join(output_dir, f"{task_id}.png")
                draw_one_stroke_task(item, output_path)
                print(f"✅ 成功生成一笔画: {output_path}")
                return True
    print(f"❌ 未在 {data_file_path} 中找到 ID: {task_id}")
    return False
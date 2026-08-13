# src/minibench/visualization/mahjong_vis.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']

def draw_mahjong_hand(task_data, save_path):
    hand = task_data.get("hand", [])
    if not hand: return

    fig, ax = plt.subplots(figsize=(len(hand) * 0.7, 1.5))
    ax.set_xlim(-0.5, len(hand))
    ax.set_ylim(0, 1.5)

    honor_map = {'E': '東', 'S': '南', 'W': '西', 'N': '北', 'P': '白', 'F': '發', 'C': '中'}

    for i, tile in enumerate(hand):
        rect = patches.Rectangle((i - 0.45, 0.1), 0.9, 1.2, linewidth=1.5, edgecolor='#555555', facecolor='#FFFFF0', zorder=1)
        ax.add_patch(rect)
        color, display_text = 'black', tile
        
        # 为了兼容纯英文，把 万/筒/索 替换回 m/p/s，如果字体好了，换成中文即可
        if 'm' in tile:
            color, display_text = '#D32F2F', tile.replace('m', '万')
        elif 'p' in tile:
            color, display_text = '#1976D2', tile.replace('p', '筒')
        elif 's' in tile:
            color, display_text = '#388E3C', tile.replace('s', '索')
        elif tile in honor_map:
            color, display_text = '#512DA8', honor_map[tile]

        ax.text(i, 0.7, display_text, ha='center', va='center', fontsize=18, fontweight='bold', color=color, zorder=2)

    ax.axis('off')
    plt.title(f"Mahjong Hand: {task_data.get('id')}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def visualize_mahjong_by_id(task_id, data_file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(data_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get("id") == task_id:
                output_path = os.path.join(output_dir, f"{task_id}.png")
                draw_mahjong_hand(item, output_path)
                print(f"✅ 成功生成麻将: {output_path}")
                return True
    print(f"❌ 未在 {data_file_path} 中找到 ID: {task_id}")
    return False
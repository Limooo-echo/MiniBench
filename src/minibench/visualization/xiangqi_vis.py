# src/minibench/visualization/xiangqi_vis.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os

from minibench.assets.fonts import matplotlib_font
from minibench.datasets.xiangqi.schema import fen_to_board

# 棋子 ID 映射，与 gym_xiangqi 的 piece ID 编码保持一致：
# 1=帅/将  2-3=仕/士  4-5=相/象  6-7=马  8-9=车  10-11=炮  12-16=兵/卒
# （负值为黑方）
PIECE_MAP = {
    1: "帅", -1: "将",
    2: "仕", -2: "士", 3: "仕", -3: "士",
    4: "相", -4: "象", 5: "相", -5: "象",
    6: "马", -6: "马", 7: "马", -7: "马",
    8: "车", -8: "车", 9: "车", -9: "车",
    10: "炮", -10: "炮", 11: "炮", -11: "炮",
    12: "兵", -12: "卒", 13: "兵", -13: "卒",
    14: "兵", -14: "卒", 15: "兵", -15: "卒",
    16: "兵", -16: "卒",
}

def draw_xiangqi_board(task_data, save_path):
    board_matrix = task_data.get("board", [])
    if not board_matrix and task_data.get("fen"):
        board_matrix, _ = fen_to_board(task_data["fen"])
    if not board_matrix: return
    regular_font = matplotlib_font()
    bold_font = matplotlib_font(bold=True)

    fig, ax = plt.subplots(figsize=(5.5, 6.5))
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 9.5)
    ax.invert_yaxis()

    for i in range(10): ax.plot([0, 8], [i, i], color='black', linewidth=1, zorder=1)
    for j in range(9):
        ax.plot([j, j], [0, 4], color='black', linewidth=1, zorder=1)
        ax.plot([j, j], [5, 9], color='black', linewidth=1, zorder=1)
    ax.plot([3, 5], [0, 2], color='black', zorder=1)
    ax.plot([5, 3], [0, 2], color='black', zorder=1)
    ax.plot([3, 5], [7, 9], color='black', zorder=1)
    ax.plot([5, 3], [7, 9], color='black', zorder=1)

    for row_idx, row in enumerate(board_matrix):
        for col_idx, piece_val in enumerate(row):
            if piece_val != 0:
                is_red = piece_val > 0
                color = '#D32F2F' if is_red else '#212121'
                text = PIECE_MAP.get(piece_val, str(piece_val))
                circle = patches.Circle((col_idx, row_idx), 0.42, facecolor='#FFF8E7', edgecolor=color, linewidth=2, zorder=3)
                ax.add_patch(circle)
                ax.text(col_idx, row_idx, text, color=color, ha='center', va='center',
                        fontsize=15, zorder=4, fontproperties=bold_font)

    ax.axis('off')
    plt.title(f"Xiangqi Task: {task_data.get('id')}", fontproperties=regular_font)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def visualize_xiangqi_by_id(task_id, data_file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(data_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get("id") == task_id:
                output_path = os.path.join(output_dir, f"{task_id}.png")
                draw_xiangqi_board(item, output_path)
                print(f"✅ 成功生成象棋: {output_path}")
                return True
    print(f"❌ 未在 {data_file_path} 中找到 ID: {task_id}")
    return False

from PIL import Image


def render_xiangqi_trajectory(task_id, board_states_list, base_output_dir="vis_outputs"):
    """
    绘制象棋残局的多步轨迹，按题目建文件夹，并生成 GIF 动图。
    """
    if not board_states_list:
        print(f"[{task_id}] 没有轨迹数据，跳过绘图。")
        return

    task_dir = os.path.join(base_output_dir, task_id)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    image_paths = []
    print(f"正在生成 {task_id} 的轨迹图，共 {len(board_states_list)} 步...")

    for step_idx, board_matrix in enumerate(board_states_list):
        temp_task_data = {"id": f"{task_id} - Step {step_idx}", "board": board_matrix}
        img_filename = f"step_{step_idx:02d}.png"
        img_path = os.path.join(task_dir, img_filename)
        
        draw_xiangqi_board(temp_task_data, save_path=img_path)
        image_paths.append(img_path)

    if image_paths:
        gif_path = os.path.join(task_dir, f"{task_id}_replay.gif")
        frames = [Image.open(img) for img in image_paths]
        frames[0].save(
            gif_path,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=1000, 
            loop=0
        )
        print(f"✅ {task_id} 动态轨迹已生成: {gif_path}")

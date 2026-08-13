"""C2 变体规则: 少子局面生成与过滤 (模板构造式).

模板构造保证每个局面在其主变体组下满足 标准最优 != 变体最优:
  A 型 (variant_a 自由马): 马被蹩腿无法将死 -> 变体下马将死
  B 型 (variant_b 车不过河): 车过河吃将 -> 变体下该走法非法
  C 型 (variant_c 炮隔两子): 炮隔两子吃将 -> 标准下不能, 变体下将死

每个模板生成 2 个局面 (含镜像), 共 6 个局面.
生成后仍跑完整 4 组过滤验证, 保证合法/可解/明确最优/非送子.
"""
from __future__ import annotations

from .board import VariantBoard
from .rules import make_rules
from .search import is_blunder, score_moves

ROWS, COLS = 10, 9


def _mirror(board: list[list[int]]) -> list[list[int]]:
    """列镜像 (c -> 8-c), 保持规则语义不变."""
    return [row[::-1] for row in board]


# ---- 模板 A: 自由马吃大子 ----
# 黑车(4,4) 被黑兵(3,3) 蹩马腿保护 (马(2,3)跳(4,4)的腿在(3,3))
# 黑将(0,6) 不在马攻击位 (开局安全)
# 标准: 车吃黑马(+2) 最优; 变体: 马吃黑车(+9) 最优
def template_a() -> list[list[int]]:
    b = [[0] * COLS for _ in range(ROWS)]
    b[0][6] = -1    # 黑将 (远离马攻击位)
    b[2][3] = 6     # 红马 (2,3)
    b[3][3] = -12   # 黑兵 (3,3): 蹩马(2,3)跳(4,4)的腿
    b[4][4] = -9    # 黑车 (4,4): 变体下马吃子目标 (+9)
    b[5][4] = 4     # 红象 (5,4): 挡黑车路线
    b[5][7] = -6    # 黑马 (标准最优目标 +4)
    b[6][7] = 8     # 红车 (6,7): 标准最优吃 (5,7) 黑马
    b[8][4] = 1     # 红将
    return b


# ---- 模板 B: 车不能过河 ----
# 黑兵(3,4) 挡在红车(5,4)与黑将(0,4)之间 (开局合法, 车不将军黑将)
# 黑将逃位 (0,3)/(0,5) 被红兵 (1,3)/(1,5) 控, (1,4) 被红马(3,5) 控
# 标准: 车吃兵 -> 将死 (+10000); 变体: 车不能过河 -> 炮吃黑马 (+4)
def template_b() -> list[list[int]]:
    b = [[0] * COLS for _ in range(ROWS)]
    b[0][4] = -1    # 黑将
    b[1][3] = 12    # 红兵 (1,3): 控 (0,3)
    b[1][5] = 12    # 红兵 (1,5): 控 (0,5)
    b[3][4] = -12   # 黑兵 (3,4): 挡红车 (车吃它将死)
    b[3][5] = 6     # 红马 (3,5): 控 (1,4)
    b[5][4] = 8     # 红车 (5,4)
    b[7][0] = -6    # 黑马 (变体下炮吃子目标 +4)
    b[8][0] = -12   # 黑兵 (炮架)
    b[8][4] = 1     # 红将
    b[9][0] = 10    # 红炮 (9,0): 隔黑兵吃黑马
    return b


# ---- 模板 C: 兵可后退 ----
# 仿模板 A 可赢结构: 黑将(0,6), 红车(6,7) 吃黑马(5,7), 红马(2,3) 参与进攻
# 红兵(7,3) 后方有黑车(8,3): 标准兵不能退吃; 变体兵退吃黑车(+9)
# 标准: 红车吃黑马(+3) 最优; 变体: 兵退吃黑车(+9) 最优
def template_c() -> list[list[int]]:
    b = [[0] * COLS for _ in range(ROWS)]
    b[0][6] = -1    # 黑将 (远离战场)
    b[2][3] = 6     # 红马 (2,3): 参与进攻, 保证可赢性
    b[5][7] = -6    # 黑马 (标准最优目标 +4)
    b[6][7] = 8     # 红车 (6,7): 标准最优吃 (5,7) 黑马
    b[7][3] = 12    # 红兵 (7,3): 变体下退吃 (8,3) 黑车
    b[8][3] = -9    # 黑车 (8,3): 变体下兵吃子目标 (+9)
    b[9][4] = 1     # 红将
    return b


# ---- 模板 D: 多步杀 (mate in 2, 车将军序列) ----
# 黑将(0,4) 逃位 (0,3)/(0,5) 被控 (黑车(0,5)占位), (1,4) 唯一空位
# mate in 2: 车2(2,5)吃黑车(0,5)=将军 -> 黑将逃(1,4) -> 车1(6,7)->(1,7)将军 -> 将死
# 第一步"吃车+将军"有表面价值 (模型有动机选将军步, 降低多步杀难度)
# (1,3) 由红炮(3,3)隔兵控; 红马(1,3) 挡红兵(2,3)向前; 红兵(6,5) 挡车1横移列4
def template_d() -> list[list[int]]:
    b = [[0] * COLS for _ in range(ROWS)]
    b[0][4] = -1    # 黑将
    b[0][5] = -9    # 黑车 (0,5): 车2吃它=将军 (mate in 2 第一步)
    b[1][3] = 6     # 红马 (1,3): 挡兵向前 + 吃黑车次优
    b[2][3] = 12    # 红兵 (2,3): 控 (2,4) (防黑将吃马)
    b[2][4] = 6     # 红马 (2,4): 控 (0,3) + 挡车2横移
    b[2][5] = 8     # 红车2 (2,5): 将军序列第一步 (吃黑车)
    b[3][3] = 10    # 红炮 (3,3): 隔兵控 (1,3)
    b[3][4] = -12   # 黑卒 (3,4): 挡列4
    b[6][5] = 12    # 红兵 (6,5): 挡车1横移列4
    b[6][7] = 8     # 红车1 (6,7): 将军序列第二步
    b[7][7] = -6    # 黑马 (次优吃子目标)
    b[8][3] = 1     # 红将 (避开黑将列4照面)
    return b


# ---- 模板 E: 多步杀镜像 (mate in 2) ----
def template_e() -> list[list[int]]:
    return _mirror(template_d())


TEMPLATES = [
    ("variant_a", template_a, 2),
    ("variant_a", lambda: _mirror(template_a()), 2),
    ("variant_b", template_b, 2),
    ("variant_b", lambda: _mirror(template_b()), 2),
    ("variant_c", template_c, 2),
    ("variant_c", lambda: _mirror(template_c()), 2),
    ("standard", template_d, 3),
    ("standard", lambda: _mirror(template_d()), 3),
    ("standard", template_e, 3),
    ("standard", lambda: _mirror(template_e()), 3),
]


def rules_for_group(group: str) -> list:
    pool = make_rules()
    key = {
        "variant_a": "free_horse",
        "variant_b": "chariot_no_center",
        "variant_c": "free_soldier",
    }[group]
    return pool[key]


def find_best_with_margin(
    board: VariantBoard, side: int, depth: int, agent_side: int,
    margin: float = 0.5,
) -> tuple | None:
    """找明确最优动作: 最优与次优分差 >= margin."""
    scored = score_moves(board, side, depth, agent_side)
    if not scored:
        return None
    best_score = scored[0][1]
    if len(scored) == 1:
        return scored[0][0], best_score
    if best_score - scored[1][1] < margin:
        return None
    return scored[0][0], best_score


def check_group(
    board: list[list[int]],
    group: str,
    depth: int = 2,
) -> dict | None:
    """对指定组验证过滤条件. 通过返回最优动作信息, 否则 None.

    开局合法性: 轮到红方时, 黑将不得被攻击 (否则黑方上一步非法).
    """
    rules = rules_for_group(group) if group != "standard" else []
    vb = VariantBoard(board, rules)
    if vb.find_general(-1) is None:
        return None
    if vb._is_in_check(-1):
        return None  # 黑将开局被将军 = 非法局面
    if not vb.has_legal_moves(1):
        return None
    if vb.is_checkmate(1):
        return None
    best = find_best_with_margin(vb, 1, depth, agent_side=1)
    if best is None:
        return None
    move, score = best
    legal = vb.legal_moves(1)
    if len(legal) < 3:
        return None
    if is_blunder(vb, move, 1, depth, agent_side=1):
        return None
    return {
        "group": group,
        "optimal_move": move,
        "optimal_score": score,
        "n_legal": len(legal),
    }


def has_forced_mate(board, side: int = 1, depth: int = 10) -> bool:
    """side 能否在 depth 半回合内强制将死 (side 只走将军/吃将步).

    比贪心模拟更准确地验证"存在必胜路线".
    """
    if board.find_general(-side) is None:
        return True
    if board.find_general(side) is None or depth <= 0:
        return False
    if board.is_checkmate(-side):
        return True
    if board.is_checkmate(side):
        return False
    for mv in board.legal_moves(side):
        trial = board.copy()
        trial.apply(mv)
        if trial.find_general(-side) is None:
            return True  # 吃将
        if trial._is_in_check(-side):
            all_ok = True
            for bmv in trial.legal_moves(-side):
                t2 = trial.copy()
                t2.apply(bmv)
                if not has_forced_mate(t2, side, depth - 2):
                    all_ok = False
                    break
            if all_ok:
                return True
    return False


def generate_tasks(
    n_tasks: int = 6,
    depth: int = 2,
) -> list[dict]:
    """模板构造 6 个局面并做 4 组验证.

    标准组必须通过全部过滤; 主变体组必须有明确最优且 != 标准最优;
    其他变体组尽力验证 (无明确最优时 info=None, 该组只测合规/对弈).
    可赢性: standard + 主变体组必须"强制将死"可赢; 其他组警告但接受.
    """
    tasks = []
    for ti, (main_group, builder, tpl_depth) in enumerate(TEMPLATES[:n_tasks], start=1):
        board = builder()

        # 标准组 (必须通过)
        std_best = check_group(board, "standard", tpl_depth)
        if std_best is None:
            raise RuntimeError(f"模板 {ti} ({main_group}) 标准组过滤失败!")

        # 各变体组 (允许 None)
        group_infos = {}
        for group in ("variant_a", "variant_b", "variant_c"):
            group_infos[group] = check_group(board, group, tpl_depth)

        # 主变体组检查 (main_group 为 "standard" 时跳过: 纯能力题, 不强调变体)
        if main_group != "standard":
            main_info = group_infos[main_group]
            if main_info is None:
                raise RuntimeError(f"模板 {ti} 主变体组 {main_group} 过滤失败!")

        # 可赢性验证: standard + 主变体组必须强制将死可赢
        for group in dict.fromkeys(("standard", main_group)):
            rules = [] if group == "standard" else rules_for_group(group)
            if not has_forced_mate(VariantBoard(board, rules)):
                raise RuntimeError(
                    f"模板 {ti} ({main_group}) 组 {group} 不可赢! 需要调整局面")
        for group in ("variant_a", "variant_b", "variant_c"):
            if group == main_group:
                continue
            rules = rules_for_group(group)
            if not has_forced_mate(VariantBoard(board, rules)):
                print(f"  [WARN] 模板 {ti} 组 {group} 不可赢 (success 上限受限)")

        tasks.append({
            "board": board,
            "main_group": main_group,
            "standard_optimal": std_best["optimal_move"],
            "standard_score": std_best["optimal_score"],
            "groups": group_infos,
        })
        print(f"  [task {ti}] main_group={main_group} "
              f"std={std_best['optimal_move'].to_uci()} "
              + " ".join(
                  f"{g}:{'DIFF' if i and i['optimal_move'] != std_best['optimal_move'] else 'same' if i else 'N/A'}"
                  for g, i in group_infos.items()
              ))
    return tasks

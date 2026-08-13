"""C2 临时变体规则: 规则定义模块.

四类临时规则 (每道题抽取 1-2 条):
  1. MOVE_MOD    走法变化    : {piece, mod}
       mod: "no_leg_restriction" 马不蹩腿
            "two_screens"        炮吃子必须隔两枚棋子
  2. ZONE_LIMIT  区域限制    : {piece, zone}
       zone: "no_cross_river"    指定棋子不能过河
             "not_center_cols"   指定棋子不能进入中央三列
  3. IDENTITY    身份替换    : {piece, moves_as, keep_value}
       标记棋子按另一兵种走法移动, 保留原有子力价值
  4. LOCAL_WIN   局部胜负条件: {kind, ...}
       kind: "occupy_square"      {square: (r,c), within: int} 指定步数内占领指定格
             "protect_capture"    {protect: (r,c), capture: (r,c)} 保护指定子并吃掉目标子
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---- 棋子编码 (与 gym_xiangqi 一致) ----
# 1=帅/将 2/3=仕 4/5=象 6/7=马 8/9=车 10/11=炮 12-16=兵
PIECE_TYPES = {
    "general": 1, "advisor": 2, "elephant": 4, "horse": 6,
    "chariot": 8, "cannon": 10, "soldier": 12,
}
TYPE_TO_NAME = {v: k for k, v in PIECE_TYPES.items()}

# 子力价值 (标准评估)
PIECE_VALUE = {
    1: 10000,   # 将
    2: 2, 3: 2,
    4: 2, 5: 2,
    6: 4, 7: 4,
    8: 9, 9: 9,
    10: 4.5, 11: 4.5,
    12: 1, 13: 1, 14: 1, 15: 1, 16: 1,
}

# 标准规则下各兵种的走法类型 (供身份替换/走法生成使用)
MOVE_KINDS = {
    "general": "general", "advisor": "advisor", "elephant": "elephant",
    "horse": "horse", "chariot": "chariot", "cannon": "cannon",
    "soldier": "soldier",
}


@dataclass
class Rule:
    """一条临时规则."""
    kind: str                       # move_mod / zone_limit / identity / local_win
    piece: str | None = None        # 目标兵种 (general/advisor/elephant/horse/chariot/cannon/soldier, "*"=任意)
    params: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        """生成给 LLM 的规则描述 (英文, 与 prompt 一致)."""
        if self.kind == "move_mod":
            mod = self.params.get("mod")
            if mod == "no_leg_restriction":
                return ("Horse is NOT blocked by the 'horse leg' rule: it may "
                        "jump to any knight-square even if an adjacent blocking "
                        "piece is present.")
            if mod == "two_screens":
                return ("Cannon captures require EXACTLY TWO screens (pieces) "
                        "between it and the target. Normal moves still require "
                        "one screen or none (standard).")
            if mod == "free_retreat":
                return ("Soldier may also move backward (one step) after "
                        "crossing the river, in addition to forward/sideways.")
        if self.kind == "zone_limit":
            zone = self.params.get("zone")
            if zone == "no_cross_river":
                return f"{self.piece.title()} may NOT cross the river (it must stay on its own half of the board)."
            if zone == "not_center_cols":
                return f"{self.piece.title()} may NOT move to the central three columns (files d/e/f)."
        if self.kind == "identity":
            return (f"The {self.piece} piece(s) move like a {self.params.get('moves_as')} in this game, "
                    f"but keep their original piece value.")
        if self.kind == "local_win":
            kind = self.params.get("kind")
            if kind == "occupy_square":
                r, c = self.params["square"]
                return (f"Local goal: occupy square (row {r}, col {c}) within "
                        f"{self.params.get('within', 3)} of your moves.")
            if kind == "protect_capture":
                pr, pc = self.params["protect"]
                cr, cc = self.params["capture"]
                return (f"Local goal: keep your piece at (row {pr}, col {pc}) "
                        f"alive AND capture the enemy piece at (row {cr}, col {cc}).")
        return str(self.params)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "piece": self.piece, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        return cls(kind=d["kind"], piece=d.get("piece"), params=d.get("params", {}))


# ---- 常用规则工厂 ----

def make_rules() -> dict[str, list[Rule]]:
    """预定义的候选规则池 (每条规则一个 key)."""
    pool = {
        # 1. 走法变化
        "free_horse": [Rule("move_mod", "horse", {"mod": "no_leg_restriction"})],
        "cannon_two_screens": [Rule("move_mod", "cannon", {"mod": "two_screens"})],
        "free_soldier": [Rule("move_mod", "soldier", {"mod": "free_retreat"})],
        # 2. 区域限制
        "chariot_no_river": [Rule("zone_limit", "chariot", {"zone": "no_cross_river"})],
        "chariot_no_center": [Rule("zone_limit", "chariot", {"zone": "not_center_cols"})],
        "cannon_no_center": [Rule("zone_limit", "cannon", {"zone": "not_center_cols"})],
        # 3. 身份替换
        "soldier_as_chariot": [Rule("identity", "soldier", {"moves_as": "chariot", "keep_value": True})],
        "advisor_as_horse": [Rule("identity", "advisor", {"moves_as": "horse", "keep_value": True})],
        # 4. 局部胜负条件
        "occupy_square_3": [Rule("local_win", None, {"kind": "occupy_square", "within": 3})],
        "protect_and_capture": [Rule("local_win", None, {"kind": "protect_capture"})],
    }
    return pool


# ---- 规则应用辅助函数 ----

def piece_of_id(pid: int) -> str:
    """棋子 ID -> 兵种名 (取组基)."""
    aid = abs(pid)
    if aid in (1,): return "general"
    if aid in (2, 3): return "advisor"
    if aid in (4, 5): return "elephant"
    if aid in (6, 7): return "horse"
    if aid in (8, 9): return "chariot"
    if aid in (10, 11): return "cannon"
    if aid in (12, 13, 14, 15, 16): return "soldier"
    return "unknown"


def piece_matches(piece_type: str | None, abs_id: int) -> bool:
    """规则 piece 是否匹配该棋子."""
    if piece_type is None or piece_type == "*":
        return True
    return piece_of_id(abs_id) == piece_type

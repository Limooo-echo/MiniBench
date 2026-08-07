from __future__ import annotations

from collections import Counter

try:
    from mahjong.hand_calculating.hand import HandCalculator
    from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
    from mahjong.shanten import Shanten
except ImportError as exc:  # pragma: no cover - exercised only without dependency.
    HandCalculator = None  # type: ignore[assignment]
    HandConfig = None  # type: ignore[assignment]
    OptionalRules = None  # type: ignore[assignment]
    Shanten = None  # type: ignore[assignment]
    _MAHJONG_IMPORT_ERROR = exc
else:
    _MAHJONG_IMPORT_ERROR = None


HONOR_TO_INDEX = {
    "E": 27,
    "S": 28,
    "W": 29,
    "N": 30,
    "P": 31,
    "F": 32,
    "C": 33,
    "1Z": 27,
    "2Z": 28,
    "3Z": 29,
    "4Z": 30,
    "5Z": 31,
    "6Z": 32,
    "7Z": 33,
}

INDEX_TO_TILE = {
    **{index: f"{index + 1}m" for index in range(0, 9)},
    **{index: f"{index - 8}p" for index in range(9, 18)},
    **{index: f"{index - 17}s" for index in range(18, 27)},
    27: "E",
    28: "S",
    29: "W",
    30: "N",
    31: "P",
    32: "F",
    33: "C",
}

CHINESE_NUMBER_TO_DIGIT = {
    "\u4e00": "1",
    "\u4e8c": "2",
    "\u4e09": "3",
    "\u56db": "4",
    "\u4e94": "5",
    "\u516d": "6",
    "\u4e03": "7",
    "\u516b": "8",
    "\u4e5d": "9",
}

CHINESE_SUIT_TO_CODE = {
    "\u842c": "m",
    "\u4e07": "m",
    "\u7b52": "p",
    "\u9905": "p",
    "\u997c": "p",
    "\u689d": "s",
    "\u6761": "s",
    "\u7d22": "s",
}

CHINESE_HONOR_TO_CODE = {
    "\u6771": "E",
    "\u4e1c": "E",
    "\u5357": "S",
    "\u897f": "W",
    "\u5317": "N",
    "\u767d": "P",
    "\u767d\u677f": "P",
    "\u767c": "F",
    "\u53d1": "F",
    "\u4e2d": "C",
    "\u7d05\u4e2d": "C",
    "\u7ea2\u4e2d": "C",
}


def normalize_tile(tile: str) -> str:
    value = tile.strip()
    upper = value.upper()
    if upper in HONOR_TO_INDEX:
        return INDEX_TO_TILE[HONOR_TO_INDEX[upper]]
    if value in CHINESE_HONOR_TO_CODE:
        return CHINESE_HONOR_TO_CODE[value]

    if len(value) == 2:
        chinese_number = CHINESE_NUMBER_TO_DIGIT.get(value[0])
        chinese_suit = CHINESE_SUIT_TO_CODE.get(value[1])
        if chinese_number is not None and chinese_suit is not None:
            return f"{chinese_number}{chinese_suit}"

    if len(value) != 2:
        raise ValueError(f"invalid tile notation: {tile!r}")

    number, suit = value[0], value[1].lower()
    if number not in "123456789" or suit not in {"m", "p", "s"}:
        raise ValueError(f"invalid tile notation: {tile!r}")
    return f"{number}{suit}"


def tile_to_index(tile: str) -> int:
    normalized = normalize_tile(tile)
    if normalized in HONOR_TO_INDEX:
        return HONOR_TO_INDEX[normalized]

    number = int(normalized[0])
    suit = normalized[1]
    if suit == "m":
        return number - 1
    if suit == "p":
        return 9 + number - 1
    if suit == "s":
        return 18 + number - 1
    raise ValueError(f"invalid tile notation: {tile!r}")


def index_to_tile(index: int) -> str:
    if index not in INDEX_TO_TILE:
        raise ValueError(f"invalid tile index: {index}")
    return INDEX_TO_TILE[index]


def full_tile_wall() -> list[str]:
    return [
        index_to_tile(index)
        for index in range(34)
        for _copy in range(4)
    ]


def normalize_tiles(tiles: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(normalize_tile(tile) for tile in tiles)
    counts = Counter(normalized)
    overfull = sorted(tile for tile, count in counts.items() if count > 4)
    if overfull:
        raise ValueError(f"too many copies of tile(s): {', '.join(overfull)}")
    return normalized


def tiles_to_34_array(tiles: list[str] | tuple[str, ...]) -> list[int]:
    normalized = normalize_tiles(tiles)
    counts = [0] * 34
    for tile in normalized:
        counts[tile_to_index(tile)] += 1
    return counts


def tiles_to_136_array(tiles: list[str] | tuple[str, ...]) -> list[int]:
    normalized = normalize_tiles(tiles)
    used_copies = [0] * 34
    result: list[int] = []
    for tile in normalized:
        index = tile_to_index(tile)
        copy = used_copies[index]
        if copy >= 4:
            raise ValueError(f"too many copies of tile: {tile}")
        result.append(index * 4 + copy)
        used_copies[index] += 1
    return result


def calculate_shanten(tiles: list[str] | tuple[str, ...]) -> int:
    _require_mahjong()
    return Shanten().calculate_shanten(tiles_to_34_array(tiles))  # type: ignore[union-attr]


def is_winning_hand(tiles: list[str] | tuple[str, ...]) -> bool:
    if len(tiles) % 3 != 2:
        return False
    return calculate_shanten(tiles) == -1


def winning_tiles(tiles: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized = normalize_tiles(tiles)
    if len(normalized) % 3 != 1:
        raise ValueError("winning tile tasks require a 3n+1 hand, usually 13 tiles")

    counts = tiles_to_34_array(normalized)
    waits: list[str] = []
    for index in range(34):
        if counts[index] >= 4:
            continue
        candidate = index_to_tile(index)
        if is_winning_hand([*normalized, candidate]):
            waits.append(candidate)
    return tuple(waits)


def waits_by_discard(
    tiles: list[str] | tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Return every tenpai discard and its distinct winning tile types."""
    normalized = normalize_tiles(tiles)
    if len(normalized) % 3 != 2:
        raise ValueError("discard tasks require a 3n+2 hand, usually 14 tiles")

    result: dict[str, tuple[str, ...]] = {}
    for tile in sorted(set(normalized), key=tile_to_index):
        remaining = list(normalized)
        remaining.remove(tile)
        waits = winning_tiles(remaining)
        if waits:
            result[tile] = waits
    return result


def max_wait_discards(tiles: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return all discards tied for the most distinct winning tile types."""
    discard_waits = waits_by_discard(tiles)
    if not discard_waits:
        return ()
    maximum = max(len(waits) for waits in discard_waits.values())
    return tuple(
        discard
        for discard, waits in discard_waits.items()
        if len(waits) == maximum
    )


def live_wait_counts(
    tiles: list[str] | tuple[str, ...],
    visible_tiles: list[str] | tuple[str, ...] = (),
    *,
    additional_visible: list[str] | tuple[str, ...] = (),
) -> dict[str, int]:
    """Return structurally winning tiles and their remaining live copies."""
    normalized = normalize_tiles(tiles)
    visible = tuple(normalize_tile(tile) for tile in visible_tiles)
    additional = tuple(normalize_tile(tile) for tile in additional_visible)
    known_counts = Counter((*normalized, *visible, *additional))
    overfull = sorted(tile for tile, count in known_counts.items() if count > 4)
    if overfull:
        raise ValueError(
            "hand and visible tiles contain too many copies of: "
            + ", ".join(overfull)
        )
    return {
        tile: 4 - known_counts[tile]
        for tile in winning_tiles(normalized)
        if known_counts[tile] < 4
    }


def live_waits_by_discard(
    tiles: list[str] | tuple[str, ...],
    visible_tiles: list[str] | tuple[str, ...] = (),
) -> dict[str, dict[str, int]]:
    """Return every tenpai discard and live winning-copy counts after it."""
    normalized = normalize_tiles(tiles)
    if len(normalized) % 3 != 2:
        raise ValueError("discard tasks require a 3n+2 hand, usually 14 tiles")

    result: dict[str, dict[str, int]] = {}
    for discard in sorted(set(normalized), key=tile_to_index):
        remaining = list(normalized)
        remaining.remove(discard)
        waits = live_wait_counts(
            remaining,
            visible_tiles,
            additional_visible=(discard,),
        )
        if waits:
            result[discard] = waits
    return result


def max_ukeire_discards(
    tiles: list[str] | tuple[str, ...],
    visible_tiles: list[str] | tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return discards tied for the most live winning tile copies."""
    discard_waits = live_waits_by_discard(tiles, visible_tiles)
    if not discard_waits:
        return ()
    maximum = max(sum(waits.values()) for waits in discard_waits.values())
    return tuple(
        discard
        for discard, waits in discard_waits.items()
        if sum(waits.values()) == maximum
    )


def score_closed_hand(
    tiles: list[str] | tuple[str, ...],
    *,
    win_tile: str,
    is_tsumo: bool,
    is_riichi: bool = False,
    player_wind: int = 27,
    round_wind: int = 27,
    is_haitei: bool = False,
    is_houtei: bool = False,
    is_rinshan: bool = False,
) -> dict[str, object] | None:
    _require_mahjong()
    normalized = list(normalize_tiles(tiles))
    normalized_win_tile = normalize_tile(win_tile)
    if normalized_win_tile not in normalized:
        raise ValueError("win_tile must be included in tiles")

    tiles_136 = tiles_to_136_array(normalized)
    win_tile_136 = next(
        tile_id
        for tile, tile_id in zip(normalized, tiles_136)
        if tile == normalized_win_tile
    )
    config = HandConfig(  # type: ignore[operator]
        is_tsumo=is_tsumo,
        is_riichi=is_riichi,
        is_haitei=is_haitei,
        is_houtei=is_houtei,
        is_rinshan=is_rinshan,
        player_wind=player_wind,
        round_wind=round_wind,
        options=OptionalRules(has_open_tanyao=True),  # type: ignore[operator]
    )
    result = HandCalculator().estimate_hand_value(  # type: ignore[operator]
        tiles_136,
        win_tile_136,
        config=config,
    )
    if result.error:
        return None
    if not result.han:
        return None
    return {
        "han": result.han,
        "fu": result.fu,
        "cost": result.cost,
        "yaku": [str(yaku) for yaku in result.yaku],
    }

def _require_mahjong() -> None:
    if _MAHJONG_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing dependency 'mahjong'. Install project dependencies with "
            "`python -m pip install -e .` or install it directly with "
            "`python -m pip install mahjong`."
        ) from _MAHJONG_IMPORT_ERROR

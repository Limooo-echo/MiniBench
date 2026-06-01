from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import random

try:
    from mahjong.hand_calculating.hand import HandCalculator
    from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
    from mahjong.meld import Meld
    from mahjong.shanten import Shanten
except ImportError as exc:  # pragma: no cover - exercised only without dependency.
    HandCalculator = None  # type: ignore[assignment]
    HandConfig = None  # type: ignore[assignment]
    Meld = None  # type: ignore[assignment]
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


def normalize_tile(tile: str) -> str:
    value = tile.strip()
    upper = value.upper()
    if upper in HONOR_TO_INDEX:
        return INDEX_TO_TILE[HONOR_TO_INDEX[upper]]

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


def deal_table(seed: int) -> tuple[list[list[str]], list[str]]:
    wall = full_tile_wall()
    random.Random(seed).shuffle(wall)
    hands = [[] for _seat in range(4)]
    for _round in range(13):
        for seat in range(4):
            hands[seat].append(wall.pop(0))
    return hands, wall


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


def tenpai_discards(tiles: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized = normalize_tiles(tiles)
    if len(normalized) % 3 != 2:
        raise ValueError("discard tasks require a 3n+2 hand, usually 14 tiles")

    discards: list[str] = []
    for tile in sorted(set(normalized), key=tile_to_index):
        remaining = list(normalized)
        remaining.remove(tile)
        if calculate_shanten(remaining) == 0:
            discards.append(tile)
    return tuple(discards)


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
    melds: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object] | None:
    _require_mahjong()
    normalized = list(normalize_tiles(tiles))
    normalized_win_tile = normalize_tile(win_tile)
    if normalized_win_tile not in normalized:
        raise ValueError("win_tile must be included in tiles")

    tiles_136 = tiles_to_136_array(normalized)
    meld_objects, used_in_melds = _build_meld_objects(melds or [], normalized, tiles_136)
    win_tile_136 = _choose_win_tile_id(
        normalized,
        tiles_136,
        normalized_win_tile,
        used_in_melds,
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
        melds=meld_objects,
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


def _build_meld_objects(
    melds: Sequence[Mapping[str, object]],
    normalized_tiles: list[str],
    tiles_136: list[int],
) -> tuple[list[object], set[int]]:
    pools: dict[str, list[int]] = defaultdict(list)
    for tile, tile_id in zip(normalized_tiles, tiles_136):
        pools[tile].append(tile_id)

    meld_objects: list[object] = []
    used: set[int] = set()
    for meld in melds:
        meld_type = meld.get("type") or meld.get("action")
        if not isinstance(meld_type, str):
            raise ValueError("meld type must be a string")
        meld_type = meld_type.lower()
        if meld_type not in {"chi", "pon", "kan"}:
            raise ValueError(f"unsupported meld type: {meld_type}")

        raw_tiles = meld.get("tiles")
        if not isinstance(raw_tiles, Sequence) or isinstance(raw_tiles, (str, bytes)):
            raise ValueError("meld tiles must be a tile list")
        meld_tiles = [normalize_tile(str(tile)) for tile in raw_tiles]

        raw_called_tile = meld.get("called_tile")
        called_tile = normalize_tile(str(raw_called_tile)) if raw_called_tile else meld_tiles[0]

        meld_tile_ids: list[int] = []
        for tile in meld_tiles:
            try:
                tile_id = pools[tile].pop(0)
            except IndexError as exc:
                raise ValueError(f"meld tile {tile} is missing from hand tiles") from exc
            meld_tile_ids.append(tile_id)
            used.add(tile_id)

        called_tile_id = next(
            (tile_id for tile, tile_id in zip(meld_tiles, meld_tile_ids) if tile == called_tile),
            meld_tile_ids[0],
        )
        meld_objects.append(
            Meld(  # type: ignore[operator]
                meld_type=meld_type,
                tiles=meld_tile_ids,
                opened=bool(meld.get("opened", True)),
                called_tile=called_tile_id,
                who=_optional_int(meld.get("who")),
                from_who=_optional_int(meld.get("from_who")),
            )
        )
    return meld_objects, used


def _choose_win_tile_id(
    normalized_tiles: list[str],
    tiles_136: list[int],
    win_tile: str,
    used_in_melds: set[int],
) -> int:
    for tile, tile_id in zip(normalized_tiles, tiles_136):
        if tile == win_tile and tile_id not in used_in_melds:
            return tile_id
    return tiles_136[normalized_tiles.index(win_tile)]


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _require_mahjong() -> None:
    if _MAHJONG_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing dependency 'mahjong'. Install project dependencies with "
            "`python -m pip install -e .` or install it directly with "
            "`python -m pip install mahjong`."
        ) from _MAHJONG_IMPORT_ERROR

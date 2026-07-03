from __future__ import annotations

from minibench.datasets.mahjong.api import calculate_shanten, tile_to_index
from minibench.datasets.mahjong_riichi.dataset import MahjongRiichiTask


MAHJONG_RIICHI_SYSTEM_PROMPT = (
    "You are playing seat 0 in a local four-player Riichi Mahjong hand. "
    "Return exactly one JSON object and no markdown."
)


def build_mahjong_riichi_prompt(
    task: MahjongRiichiTask,
    *,
    draw_number: int,
    drawn_tile: str | None,
    hand: list[str],
    discards: list[list[str]],
    melds: list[list[dict[str, object]]],
    riichi_declared: list[bool],
    scores: list[int],
    remaining_tiles: int,
    can_riichi_discards: list[str],
    legal_closed_kan_tiles: list[str],
) -> str:
    draw_line = (
        f"You just drew: {drawn_tile}"
        if drawn_tile is not None
        else "You just called a meld and must discard from your concealed hand."
    )
    lines = [
        "Play one action in this local four-player Riichi Mahjong hand.",
        "",
        "Implemented rules in this benchmark:",
        "- You control only seat 0.",
        "- Seats 1, 2, and 3 are benchmark opponents; in external mode they may be strong mjai AIs.",
        "- Tsumo and ron are checked with Riichi Mahjong yaku/fu/point logic.",
        "- Open chi, open pon, open kan, and closed kan are supported.",
        "- You may declare riichi if the listed riichi discards are available.",
        "- No dora, ura-dora, ippatsu, abortive draws, multi-ron, or added-kan/chankan yet.",
        "- If you already declared riichi, the engine will auto-discard drawn tiles.",
        "- You cannot see future wall tiles or opponents' hidden hands.",
        "",
        "Benchmark objective:",
        "- Maximize seat 0's seat score: win first, otherwise avoid dealing in and survive.",
        "- Exhaustive draw or another player's small tsumo is much better than dealing into ron.",
        "- Worst result: discarding into another seat's ron. Fold when your hand is not close.",
        "- Treat riichi, three or more open melds, or a late two-meld hand as serious danger.",
        "",
        "Strategy reminders:",
        "- Use a push/fold gate before every discard: hand speed/value versus opponent danger.",
        "- With no serious threat, prefer discards that keep the lowest shanten.",
        "- If you are 2+ shanten and any opponent is riichi or heavily open, prefer the safest discard.",
        "- If you are tenpai or 1-shanten with a clear yaku, push only with discards that are not clearly dangerous.",
        "- When any opponent has riichi, the safety table overrides shanten hints unless you are clearly pushing tenpai.",
        "- Do not declare riichi automatically; riichi only when the hand is worth pushing.",
        "- Keep complete sequences, triplets, useful pairs, and value honor pairs.",
        "- Early: isolated terminals/honors are often expendable. Late: visible safe tiles matter more.",
        "- Against riichi, prefer genbutsu: tiles already discarded by that riichi player.",
        "- Against open hands, avoid feeding their obvious suit/honor pattern when you have safer tiles.",
        "",
        "Return one of these JSON shapes:",
        '- {"action":"discard","tile":"5m"}',
        '- {"action":"riichi","discard":"5m"}',
        '- {"action":"kan","tile":"5m"}',
        "",
        "Tile notation: 1m-9m, 1p-9p, 1s-9s, E/S/W/N, P/F/C.",
        f"Task seed: {task.seed}",
        f"Draw number: {draw_number}",
        draw_line,
        f"Your current concealed hand ({len(hand)} tiles): {' '.join(hand)}",
        f"Legal discard tiles: {' '.join(_unique_sorted_tiles(hand))}",
        f"Your riichi status: {riichi_declared[task.agent_seat]}",
        f"Riichi legal discards: {' '.join(can_riichi_discards) if can_riichi_discards else '(none)'}",
        f"Closed kan legal tiles: {' '.join(legal_closed_kan_tiles) if legal_closed_kan_tiles else '(none)'}",
        f"Scores: {' '.join(str(score) for score in scores)}",
        f"Remaining wall tiles: {remaining_tiles}",
        "",
        "Discard shanten hints (concealed hand only; lower is better):",
    ]
    lines.extend(_discard_shanten_lines(hand))
    lines.extend(
        [
            "",
            "Discard safety table:",
        ]
    )
    lines.extend(_discard_safety_lines(hand, discards, melds, riichi_declared, task.agent_seat, remaining_tiles))
    lines.extend(
        [
            "",
            "Threat assessment:",
        ]
    )
    lines.extend(_threat_lines(discards, melds, riichi_declared, task.agent_seat, remaining_tiles))
    lines.extend(
        [
            "",
            "Defense hints:",
        ]
    )
    lines.extend(_defense_lines(hand, discards, melds, riichi_declared, task.agent_seat))
    lines.extend(
        [
            "",
            "Open/declared melds:",
        ]
    )
    lines.extend(_meld_lines(melds, task.agent_seat))
    lines.extend(
        [
            "",
            "Discards:",
        ]
    )
    for seat, pond in enumerate(discards):
        label = "you" if seat == task.agent_seat else f"bot {seat}"
        riichi = " riichi" if riichi_declared[seat] else ""
        lines.append(
            f"- Seat {seat} ({label}{riichi}): {' '.join(pond) if pond else '(none)'}"
        )
    lines.extend(
        [
            "",
            "Choose a legal action.",
        ]
    )
    return "\n".join(lines)


def build_mahjong_riichi_call_prompt(
    task: MahjongRiichiTask,
    *,
    discarded_tile: str,
    discarder_seat: int,
    hand: list[str],
    discards: list[list[str]],
    melds: list[list[dict[str, object]]],
    riichi_declared: list[bool],
    scores: list[int],
    remaining_tiles: int,
    legal_call_options: list[dict[str, object]],
) -> str:
    lines = [
        "Choose whether to call the latest discard in this local four-player Riichi Mahjong hand.",
        "",
        "Return exactly one JSON object:",
        '- {"action":"pass"}',
        '- {"action":"chi","tiles":["3m","4m","5m"]}',
        '- {"action":"pon","tile":"5m"}',
        '- {"action":"kan","tile":"5m"}',
        "",
        "Tile notation: 1m-9m, 1p-9p, 1s-9s, E/S/W/N, P/F/C.",
        f"Task seed: {task.seed}",
        f"Latest discard: seat {discarder_seat} discarded {discarded_tile}",
        f"Your current concealed hand ({len(hand)} tiles): {' '.join(hand)}",
        f"Your riichi status: {riichi_declared[task.agent_seat]}",
        f"Scores: {' '.join(str(score) for score in scores)}",
        f"Remaining wall tiles: {remaining_tiles}",
        "",
        "Call decision priorities:",
        "- Pass is the default; calling removes closed riichi value and can reduce defense.",
        "- Call only if it clearly improves shanten and leaves a confirmed yaku or makes tenpai.",
        "- Avoid opening a hand that has no scoring path or only a slow one-han path.",
        "- Against riichi or a two-plus-open-meld threat, pass unless the call makes tenpai with a safe discard plan.",
        "- Do not chi or pon a marginal hand just to speed up; avoiding ron is worth more than a weak open hand.",
        "",
        "Legal call options:",
    ]
    for option in legal_call_options:
        if option["action"] == "chi":
            lines.append(f"- chi {' '.join(str(tile) for tile in option['tiles'])}")
        else:
            lines.append(f"- {option['action']} {option['called_tile']}")
    lines.extend(
        [
            "",
            "Call shanten hints (concealed hand only; lower is better):",
        ]
    )
    lines.extend(_call_shanten_lines(hand, legal_call_options))
    lines.extend(
        [
            "",
            "Threat assessment:",
        ]
    )
    lines.extend(_threat_lines(discards, melds, riichi_declared, task.agent_seat, remaining_tiles))
    lines.extend(
        [
            "",
            "Defense hints:",
        ]
    )
    lines.extend(_defense_lines(hand, discards, melds, riichi_declared, task.agent_seat))
    lines.extend(
        [
            "",
            "Open/declared melds:",
        ]
    )
    lines.extend(_meld_lines(melds, task.agent_seat))
    lines.extend(
        [
            "",
            "Discards:",
        ]
    )
    for seat, pond in enumerate(discards):
        label = "you" if seat == task.agent_seat else f"bot {seat}"
        riichi = " riichi" if riichi_declared[seat] else ""
        lines.append(
            f"- Seat {seat} ({label}{riichi}): {' '.join(pond) if pond else '(none)'}"
        )
    lines.append("")
    lines.append(
        "Pass unless the call clearly improves your hand, creates a scoring yaku, or makes you tenpai."
    )
    return "\n".join(lines)


def _unique_sorted_tiles(tiles: list[str]) -> list[str]:
    return sorted(set(tiles), key=tile_to_index)


def _discard_shanten_lines(hand: list[str]) -> list[str]:
    candidates: list[tuple[int, int, str]] = []
    for tile in _unique_sorted_tiles(hand):
        remaining = list(hand)
        remaining.remove(tile)
        try:
            shanten = calculate_shanten(remaining)
        except ValueError:
            continue
        candidates.append((shanten, tile_to_index(tile), tile))

    if not candidates:
        return ["- Not available for this hand shape."]

    best = min(shanten for shanten, _index, _tile in candidates)
    lines: list[str] = []
    for shanten, _index, tile in sorted(candidates):
        marker = " best" if shanten == best else ""
        lines.append(f"- discard {tile}: shanten {shanten}{marker}")
    return lines


def _call_shanten_lines(
    hand: list[str],
    legal_call_options: list[dict[str, object]],
) -> list[str]:
    if not legal_call_options:
        return ["- No legal calls."]

    lines: list[str] = []
    try:
        before = calculate_shanten(hand)
    except ValueError:
        before = None
    before_text = str(before) if before is not None else "unknown"
    lines.append(f"- before calling: shanten {before_text}")

    for option in legal_call_options:
        after_hand = _concealed_after_call(hand, option)
        if after_hand is None:
            lines.append(f"- {option.get('action')}: unavailable")
            continue
        try:
            after = calculate_shanten(after_hand)
        except ValueError:
            after = None
        action = str(option.get("action"))
        if action == "chi":
            detail = " ".join(str(tile) for tile in option.get("tiles", []))
        else:
            detail = str(option.get("called_tile", option.get("tile", "")))
        after_text = str(after) if after is not None else "unknown"
        marker = ""
        if before is not None and after is not None:
            marker = " improves" if after < before else " does not improve"
        lines.append(f"- {action} {detail}: shanten {after_text}{marker}")
    return lines


def _concealed_after_call(
    hand: list[str],
    option: dict[str, object],
) -> list[str] | None:
    action = option.get("action")
    called_tile = option.get("called_tile", option.get("tile"))
    if action == "chi":
        raw_tiles = option.get("tiles", [])
        if not isinstance(raw_tiles, list):
            return None
        consumed = [str(tile) for tile in raw_tiles]
    else:
        if not isinstance(called_tile, str):
            return None
        copies = 3 if action == "kan" else 2
        consumed = [called_tile] * copies

    if isinstance(called_tile, str) and called_tile in consumed:
        consumed.remove(called_tile)
    after = list(hand)
    for tile in consumed:
        if tile not in after:
            return None
        after.remove(tile)
    return after


def _defense_lines(
    hand: list[str],
    discards: list[list[str]],
    melds: list[list[dict[str, object]]],
    riichi_declared: list[bool],
    agent_seat: int,
) -> list[str]:
    hand_tiles = set(hand)
    riichi_safe_sets = [
        set(discards[seat])
        for seat in range(4)
        if seat != agent_seat and riichi_declared[seat]
    ]
    if riichi_safe_sets:
        common_safe = set.intersection(*riichi_safe_sets) & hand_tiles
        lines = [
            f"- Common genbutsu against all riichi opponents in your hand: {_format_tiles(common_safe)}",
        ]
    else:
        visible_safe = set().union(
            *(set(discards[seat]) for seat in range(4) if seat != agent_seat)
        )
        lines = [
            f"- Tiles in your hand already visible in opponents' ponds: {_format_tiles(visible_safe & hand_tiles)}",
        ]

    for seat in range(4):
        if seat == agent_seat:
            continue
        threat_parts = []
        if riichi_declared[seat]:
            threat_parts.append("riichi")
        open_melds = sum(1 for meld in melds[seat] if meld.get("opened", True))
        if open_melds:
            threat_parts.append(f"{open_melds} open melds")
        threat = ", ".join(threat_parts) if threat_parts else "no declared threat"
        safe = hand_tiles & set(discards[seat])
        lines.append(f"- Seat {seat}: {threat}; genbutsu in your hand: {_format_tiles(safe)}")

    return lines


def _discard_safety_lines(
    hand: list[str],
    discards: list[list[str]],
    melds: list[list[dict[str, object]]],
    riichi_declared: list[bool],
    agent_seat: int,
    remaining_tiles: int,
) -> list[str]:
    tiles = _unique_sorted_tiles(hand)
    riichi_seats = [
        seat for seat in range(4) if seat != agent_seat and riichi_declared[seat]
    ]
    if riichi_seats:
        lines = [
            "- Against riichi, genbutsu means the tile already appears in that player's discard pond.",
            "- Prefer RIICHI-SAFE over PARTIAL RIICHI-SAFE over RIICHI-DANGER unless you are pushing clear tenpai.",
        ]
        safe_all: list[str] = []
        partial: list[tuple[int, str]] = []
        for tile in tiles:
            safe_vs = [seat for seat in riichi_seats if tile in discards[seat]]
            unsafe_vs = [seat for seat in riichi_seats if tile not in discards[seat]]
            if not unsafe_vs:
                status = "RIICHI-SAFE"
                safe_all.append(tile)
            elif safe_vs:
                status = "PARTIAL RIICHI-SAFE"
                partial.append((len(safe_vs), tile))
            else:
                status = "RIICHI-DANGER"
            safe_text = " ".join(str(seat) for seat in safe_vs) if safe_vs else "(none)"
            unsafe_text = " ".join(str(seat) for seat in unsafe_vs) if unsafe_vs else "(none)"
            lines.append(
                f"- discard {tile}: {status}; genbutsu vs riichi seats {safe_text}; unsafe vs {unsafe_text}"
            )

        if safe_all:
            lines.append(f"- Best defensive discards: {_format_tiles(set(safe_all))}")
        elif partial:
            best_count = max(count for count, _tile in partial)
            best_partial = {tile for count, tile in partial if count == best_count}
            lines.append(
                f"- No tile is genbutsu to every riichi player; best partial defenses: {_format_tiles(best_partial)}"
            )
        else:
            lines.append(
                "- No genbutsu in hand against riichi; every discard is a push. Avoid riichi/kan and discard the least valuable tile."
            )
        return lines

    threat_seats = [
        seat
        for seat in range(4)
        if seat != agent_seat
        and _threat_level(melds[seat], riichi_declared[seat], remaining_tiles) in {"medium", "high"}
    ]
    if not threat_seats:
        return ["- No riichi or major open-hand threat. Shanten and hand efficiency may lead."]

    lines = [
        "- No opponent has riichi, but one or more open/late threats exist.",
        "- Tiles already in a threat player's pond are safer than unseen middle tiles.",
    ]
    for tile in tiles:
        visible_vs = [seat for seat in threat_seats if tile in discards[seat]]
        status = "VISIBLE-SAFER" if visible_vs else "UNSEEN"
        visible_text = " ".join(str(seat) for seat in visible_vs) if visible_vs else "(none)"
        lines.append(f"- discard {tile}: {status}; already visible vs threat seats {visible_text}")
    return lines


def _threat_lines(
    discards: list[list[str]],
    melds: list[list[dict[str, object]]],
    riichi_declared: list[bool],
    agent_seat: int,
    remaining_tiles: int,
) -> list[str]:
    lines: list[str] = []
    for seat in range(4):
        if seat == agent_seat:
            continue

        reasons: list[str] = []
        open_melds = sum(1 for meld in melds[seat] if meld.get("opened", True))
        if riichi_declared[seat]:
            reasons.append("riichi")
        if open_melds:
            reasons.append(f"{open_melds} open melds")
        if remaining_tiles <= 20:
            reasons.append("late hand")

        level = _threat_level(melds[seat], riichi_declared[seat], remaining_tiles)

        detail = ", ".join(reasons) if reasons else "closed/no riichi"
        pond_size = len(discards[seat])
        lines.append(f"- Seat {seat}: {level} threat ({detail}); pond size {pond_size}.")

    return lines


def _threat_level(
    seat_melds: list[dict[str, object]],
    riichi_declared: bool,
    remaining_tiles: int,
) -> str:
    open_melds = sum(1 for meld in seat_melds if meld.get("opened", True))
    if riichi_declared or open_melds >= 3:
        return "high"
    if open_melds >= 2 or (remaining_tiles <= 20 and open_melds >= 1):
        return "medium"
    if remaining_tiles <= 20:
        return "medium"
    return "low"


def _format_tiles(tiles: set[str]) -> str:
    if not tiles:
        return "(none)"
    return " ".join(sorted(tiles, key=tile_to_index))


def _meld_lines(melds: list[list[dict[str, object]]], agent_seat: int) -> list[str]:
    lines: list[str] = []
    for seat, seat_melds in enumerate(melds):
        label = "you" if seat == agent_seat else f"bot {seat}"
        if not seat_melds:
            lines.append(f"- Seat {seat} ({label}): (none)")
            continue
        formatted = []
        for meld in seat_melds:
            visibility = "open" if meld.get("opened", True) else "closed"
            tiles = meld.get("tiles", [])
            formatted.append(
                f"{visibility} {meld.get('type')}({' '.join(str(tile) for tile in tiles)})"
            )
        lines.append(f"- Seat {seat} ({label}): {'; '.join(formatted)}")
    return lines

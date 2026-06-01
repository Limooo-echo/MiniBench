from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from time import strftime
from typing import Any, Sequence

from minibench.agents import Agent
from minibench.mahjong_api import (
    calculate_shanten,
    deal_table,
    normalize_tile,
    score_closed_hand,
    tile_to_index,
)
from minibench.mahjong_riichi_dataset import MahjongRiichiTask
from minibench.mahjong_riichi_ai import (
    ExternalMahjongAI,
    MahjongAIError,
    make_external_mahjong_ai,
)
from minibench.mahjong_riichi_prompting import (
    build_mahjong_riichi_call_prompt,
    build_mahjong_riichi_prompt,
)


SEAT_WINDS = (27, 28, 29, 30)
ROUND_WIND = 27


@dataclass(frozen=True)
class CalledMeld:
    type: str
    tiles: tuple[str, ...]
    called_tile: str
    opened: bool
    who: int
    from_who: int

    def to_score_meld(self) -> dict[str, object]:
        return {
            "type": self.type,
            "tiles": list(self.tiles),
            "called_tile": self.called_tile,
            "opened": self.opened,
            "who": self.who,
            "from_who": self.from_who,
        }

    def to_output(self) -> dict[str, object]:
        return self.to_score_meld()


@dataclass(frozen=True)
class CallOption:
    action: str
    seat: int
    from_seat: int
    tiles: tuple[str, ...]
    called_tile: str

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "seat": self.seat,
            "from_seat": self.from_seat,
            "tiles": list(self.tiles),
            "called_tile": self.called_tile,
        }


@dataclass(frozen=True)
class PendingTurn:
    seat: int
    drawn_tile: str | None
    from_call: bool
    is_rinshan: bool = False


@dataclass(frozen=True)
class TurnResult:
    discard: str | None = None
    reason: str | None = None
    winner_seat: int | None = None
    win_type: str | None = None
    win_score: dict[str, object] | None = None


@dataclass(frozen=True)
class MahjongRiichiInstanceResult:
    task_id: str
    success: bool
    score: float
    seat_scores: list[float]
    winner_seat: int | None
    loser_seat: int | None
    win_type: str | None
    win_score: dict[str, object] | None
    final_scores: list[int]
    riichi_declared: list[bool]
    melds: list[list[dict[str, object]]]
    raw_outputs: list[str]
    agent_actions: list[dict[str, object]]
    draws: list[str]
    discards: list[list[str]]
    final_agent_hand: list[str]
    reasons: list[str]
    tags: tuple[str, ...]


def extract_riichi_action(output: str) -> dict[str, object] | None:
    payload = _parse_json_object(output)
    if payload is None:
        return None

    action = payload.get("action", "discard")
    if not isinstance(action, str):
        return None
    action = action.lower()
    if action not in {"discard", "riichi", "pass", "chi", "pon", "kan"}:
        return None

    if action == "pass":
        return {"action": "pass"}

    if action == "chi":
        raw_tiles = payload.get("tiles")
        if raw_tiles is None:
            return {"action": "chi"}
        if not isinstance(raw_tiles, list) or len(raw_tiles) != 3:
            return None
        try:
            tiles = [normalize_tile(str(tile)) for tile in raw_tiles]
        except ValueError:
            return None
        return {"action": "chi", "tiles": tiles}

    if action in {"pon", "kan"}:
        tile = payload.get("tile")
        if tile is None:
            tile = payload.get("called_tile")
        if tile is None:
            return {"action": action}
        if not isinstance(tile, str):
            return None
        try:
            return {"action": action, "tile": normalize_tile(tile)}
        except ValueError:
            return None

    tile = payload.get("tile")
    if tile is None:
        tile = payload.get("discard")
    if not isinstance(tile, str):
        return None

    try:
        discard = normalize_tile(tile)
    except ValueError:
        return None
    return {"action": action, "discard": discard}


def riichi_discards(hand: list[str]) -> list[str]:
    if len(hand) % 3 != 2:
        return []

    result: list[str] = []
    for tile in sorted(set(hand), key=tile_to_index):
        remaining = list(hand)
        remaining.remove(tile)
        if calculate_shanten(remaining) == 0:
            result.append(tile)
    return result


def legal_closed_kan_tiles(hand: list[str]) -> list[str]:
    counts = Counter(hand)
    return sorted(
        [tile for tile, count in counts.items() if count == 4],
        key=tile_to_index,
    )


def legal_call_options(
    hand: list[str],
    *,
    caller_seat: int,
    discarder_seat: int,
    discard: str,
    caller_in_riichi: bool = False,
) -> list[CallOption]:
    if caller_in_riichi:
        return []

    normalized_discard = normalize_tile(discard)
    counts = Counter(hand)
    options: list[CallOption] = []

    if counts[normalized_discard] >= 3:
        options.append(
            CallOption(
                action="kan",
                seat=caller_seat,
                from_seat=discarder_seat,
                tiles=(normalized_discard,) * 4,
                called_tile=normalized_discard,
            )
        )
    if counts[normalized_discard] >= 2:
        options.append(
            CallOption(
                action="pon",
                seat=caller_seat,
                from_seat=discarder_seat,
                tiles=(normalized_discard,) * 3,
                called_tile=normalized_discard,
            )
        )

    if caller_seat == (discarder_seat + 1) % 4:
        options.extend(_legal_chi_options(hand, caller_seat, discarder_seat, normalized_discard))

    return options


def choose_bot_action(hand: list[str], *, can_riichi: bool) -> dict[str, object]:
    legal_riichi_discards = riichi_discards(hand) if can_riichi else []
    if legal_riichi_discards:
        return {"action": "riichi", "discard": legal_riichi_discards[0]}

    candidates: list[tuple[int, int, str]] = []
    for tile in sorted(set(hand), key=tile_to_index):
        remaining = list(hand)
        remaining.remove(tile)
        try:
            shanten = calculate_shanten(remaining)
        except ValueError:
            shanten = 99
        candidates.append((shanten, tile_to_index(tile), tile))
    return {"action": "discard", "discard": min(candidates)[2]}


def choose_bot_call(
    hand: list[str],
    options: list[CallOption],
    *,
    seat: int,
    seat_melds: list[CalledMeld],
) -> CallOption | None:
    if len(seat_melds) >= 4:
        return None

    valued_tiles = {31, 32, 33, ROUND_WIND, SEAT_WINDS[seat]}
    for option in options:
        index = tile_to_index(option.called_tile)
        if option.action in {"pon", "kan"} and index in valued_tiles:
            return option

    before = _safe_shanten(hand)
    for option in options:
        if option.action == "kan":
            continue
        after_hand = _concealed_after_call(hand, option)
        if after_hand is not None and _safe_shanten(after_hand) <= before:
            return option

    return None


def _make_opponent_ais(
    task: MahjongRiichiTask,
    *,
    opponent: str,
    mahjong_ai_command: str | Sequence[str] | None,
    mahjong_ai_mode: str,
    mahjong_ai_timeout: float,
) -> dict[int, ExternalMahjongAI]:
    if opponent == "shanten":
        return {}
    if opponent != "external":
        raise ValueError("Riichi opponent must be shanten or external")

    return {
        seat: make_external_mahjong_ai(
            mahjong_ai_command,
            mode=mahjong_ai_mode,
            timeout=mahjong_ai_timeout,
        )
        for seat in range(4)
        if seat != task.agent_seat
    }


def evaluate_mahjong_riichi_tasks(
    tasks: list[MahjongRiichiTask],
    agent: Agent,
    *,
    opponent: str = "shanten",
    mahjong_ai_command: str | Sequence[str] | None = None,
    mahjong_ai_mode: str = "stdio",
    mahjong_ai_timeout: float = 30.0,
) -> list[MahjongRiichiInstanceResult]:
    return [
        evaluate_mahjong_riichi_task(
            task,
            agent,
            opponent=opponent,
            mahjong_ai_command=mahjong_ai_command,
            mahjong_ai_mode=mahjong_ai_mode,
            mahjong_ai_timeout=mahjong_ai_timeout,
        )
        for task in tasks
    ]


def evaluate_mahjong_riichi_task(
    task: MahjongRiichiTask,
    agent: Agent,
    *,
    opponent: str = "shanten",
    mahjong_ai_command: str | Sequence[str] | None = None,
    mahjong_ai_mode: str = "stdio",
    mahjong_ai_timeout: float = 30.0,
) -> MahjongRiichiInstanceResult:
    opponent_ais = _make_opponent_ais(
        task,
        opponent=opponent,
        mahjong_ai_command=mahjong_ai_command,
        mahjong_ai_mode=mahjong_ai_mode,
        mahjong_ai_timeout=mahjong_ai_timeout,
    )
    hands, wall = deal_table(task.seed)
    discards: list[list[str]] = [[] for _seat in range(4)]
    melds: list[list[CalledMeld]] = [[] for _seat in range(4)]
    riichi_declared = [False, False, False, False]
    scores = list(task.starting_scores)
    mjai_events = _initial_mjai_events(task, hands, scores)
    last_drawn_tiles: list[str | None] = [None, None, None, None]
    raw_outputs: list[str] = []
    agent_actions: list[dict[str, object]] = []
    draws: list[str] = []
    reasons: list[str] = []
    winner_seat: int | None = None
    loser_seat: int | None = None
    win_type: str | None = None
    win_score: dict[str, object] | None = None

    active_seat = 0
    draw_limit = min(task.max_draws, len(wall))
    draw_events = 0
    pending_turn: PendingTurn | None = None

    while draw_events < draw_limit or pending_turn is not None:
        if pending_turn is None:
            if not wall:
                reasons.append("wall_empty")
                break

            drawn_tile = wall.pop(0)
            hands[active_seat].append(drawn_tile)
            draws.append(f"seat{active_seat}:{drawn_tile}")
            _mjai_record_tsumo(mjai_events, active_seat, drawn_tile)
            last_drawn_tiles[active_seat] = drawn_tile
            draw_events += 1
            turn = PendingTurn(
                seat=active_seat,
                drawn_tile=drawn_tile,
                from_call=False,
            )
        else:
            turn = pending_turn
            pending_turn = None

        result = _play_discard_turn(
            task,
            agent,
            hands,
            melds,
            discards,
            riichi_declared,
            scores,
            wall,
            raw_outputs,
            agent_actions,
            draws,
            mjai_events,
            last_drawn_tiles,
            seat=turn.seat,
            drawn_tile=turn.drawn_tile,
            from_call=turn.from_call,
            is_rinshan=turn.is_rinshan,
            opponent_ais=opponent_ais,
        )
        if result.winner_seat is not None:
            winner_seat = result.winner_seat
            win_type = result.win_type
            win_score = result.win_score
            reasons.append(f"seat{winner_seat}_{win_type}:{turn.drawn_tile}")
            break
        if result.reason is not None:
            reasons.append(result.reason)
            break
        if result.discard is None:
            reasons.append("no_discard_made")
            break

        ron = _find_ron(
            hands,
            melds,
            active_seat=turn.seat,
            discard=result.discard,
            riichi_declared=riichi_declared,
            is_houtei=not wall,
        )
        if ron is not None:
            winner_seat, win_score = ron
            loser_seat = turn.seat
            win_type = "ron"
            _apply_ron_scores(scores, winner_seat, turn.seat, win_score)
            reasons.append(f"seat{winner_seat}_ron:{result.discard}:from_seat{turn.seat}")
            break

        call, call_error = _choose_call(
            task,
            agent,
            hands,
            melds,
            discards,
            riichi_declared,
            scores,
            wall,
            raw_outputs,
            agent_actions,
            mjai_events,
            discarder_seat=turn.seat,
            discard=result.discard,
            opponent_ais=opponent_ais,
        )
        if call_error is not None:
            reasons.append(call_error)
            break
        if call is not None:
            pending_turn, call_error = _execute_call(
                call,
                hands,
                melds,
                discards,
                wall,
                draws,
                mjai_events,
                last_drawn_tiles,
            )
            if call_error is not None:
                reasons.append(call_error)
                break
            active_seat = call.seat
        else:
            active_seat = (turn.seat + 1) % 4

    if winner_seat is None and not reasons:
        reasons.append("max_draws_reached")
    seat_scores = score_riichi_seats(
        winner_seat=winner_seat,
        win_type=win_type,
        loser_seat=loser_seat,
        final_scores=scores,
        reasons=reasons,
        agent_seat=task.agent_seat,
    )

    try:
        return MahjongRiichiInstanceResult(
            task_id=task.id,
            success=winner_seat == task.agent_seat,
            score=seat_scores[task.agent_seat],
            seat_scores=seat_scores,
            winner_seat=winner_seat,
            loser_seat=loser_seat,
            win_type=win_type,
            win_score=win_score,
            final_scores=scores,
            riichi_declared=riichi_declared,
            melds=_melds_for_output(melds),
            raw_outputs=raw_outputs,
            agent_actions=agent_actions,
            draws=draws,
            discards=discards,
            final_agent_hand=list(hands[task.agent_seat]),
            reasons=reasons,
            tags=task.tags,
        )
    finally:
        for ai in opponent_ais.values():
            ai.close()


def summarize_mahjong_riichi(
    results: list[MahjongRiichiInstanceResult],
) -> dict[str, Any]:
    total = len(results)
    success_count = sum(1 for result in results if result.success)
    seat_total_scores = [0.0, 0.0, 0.0, 0.0]
    seat_win_counts = [0, 0, 0, 0]
    for result in results:
        for seat, seat_score in enumerate(result.seat_scores):
            seat_total_scores[seat] += seat_score
        if result.winner_seat is not None:
            seat_win_counts[result.winner_seat] += 1
    by_tag: dict[str, dict[str, int | float]] = {}
    for result in results:
        for tag in result.tags:
            item = by_tag.setdefault(tag, {"total": 0, "success": 0, "success_rate": 0.0})
            item["total"] = int(item["total"]) + 1
            item["success"] = int(item["success"]) + int(result.success)
    for item in by_tag.values():
        item["success_rate"] = int(item["success"]) / int(item["total"])
    return {
        "total": total,
        "success": success_count,
        "success_rate": success_count / total if total else 0.0,
        "seat_total_scores": {
            f"seat{seat}": round(score, 3)
            for seat, score in enumerate(seat_total_scores)
        },
        "seat_average_scores": {
            f"seat{seat}": round(score / total, 3) if total else 0.0
            for seat, score in enumerate(seat_total_scores)
        },
        "seat_win_counts": {
            f"seat{seat}": count
            for seat, count in enumerate(seat_win_counts)
        },
        "agent_total_score": round(seat_total_scores[0], 3),
        "agent_average_score": round(seat_total_scores[0] / total, 3) if total else 0.0,
        "by_tag": by_tag,
    }


def write_mahjong_riichi_run(
    results: list[MahjongRiichiInstanceResult],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
) -> Path:
    root = Path(output_dir)
    name = run_name or f"mahjong-riichi-{strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    summary = summarize_mahjong_riichi(results)
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.txt").write_text(
        f"total={summary['total']} success={summary['success']} "
        f"success_rate={summary['success_rate']:.3f}\n"
        f"seat_total_scores={json.dumps(summary['seat_total_scores'], ensure_ascii=False)}\n"
        f"seat_average_scores={json.dumps(summary['seat_average_scores'], ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    return run_dir


def score_riichi_seats(
    *,
    winner_seat: int | None,
    win_type: str | None,
    loser_seat: int | None,
    final_scores: list[int],
    reasons: list[str],
    agent_seat: int = 0,
) -> list[float]:
    if winner_seat is None:
        if _is_normal_draw(reasons):
            return [0.5, 0.5, 0.5, 0.5]
        scores = [0.5, 0.5, 0.5, 0.5]
        scores[agent_seat] = 0.0
        return scores

    if win_type == "tsumo":
        scores = [0.25, 0.25, 0.25, 0.25]
        scores[winner_seat] = 1.0
        return scores

    if win_type == "ron":
        scores = _rank_survival_scores(final_scores)
        scores[winner_seat] = 1.0
        if loser_seat is not None:
            scores[loser_seat] = 0.0
        return scores

    return [0.0, 0.0, 0.0, 0.0]


def _is_normal_draw(reasons: list[str]) -> bool:
    return any(reason in {"max_draws_reached", "wall_empty"} for reason in reasons)


def _rank_survival_scores(final_scores: list[int]) -> list[float]:
    place_values = (0.75, 0.5, 0.25, 0.0)
    result = [0.0, 0.0, 0.0, 0.0]
    ordered_scores = sorted(set(final_scores), reverse=True)
    used_places = 0
    for score in ordered_scores:
        seats = [seat for seat, seat_score in enumerate(final_scores) if seat_score == score]
        values = place_values[used_places : used_places + len(seats)]
        value = sum(values) / len(values)
        for seat in seats:
            result[seat] = value
        used_places += len(seats)
    return result


def _play_discard_turn(
    task: MahjongRiichiTask,
    agent: Agent,
    hands: list[list[str]],
    melds: list[list[CalledMeld]],
    discards: list[list[str]],
    riichi_declared: list[bool],
    scores: list[int],
    wall: list[str],
    raw_outputs: list[str],
    agent_actions: list[dict[str, object]],
    draws: list[str],
    mjai_events: list[dict[str, object]],
    last_drawn_tiles: list[str | None],
    *,
    seat: int,
    drawn_tile: str | None,
    from_call: bool,
    is_rinshan: bool,
    opponent_ais: dict[int, ExternalMahjongAI] | None = None,
) -> TurnResult:
    current_drawn_tile = drawn_tile
    current_is_rinshan = is_rinshan

    while True:
        hand = hands[seat]
        if current_drawn_tile is not None:
            tsumo_score = _score_player_hand(
                hand,
                melds[seat],
                win_tile=current_drawn_tile,
                is_tsumo=True,
                is_riichi=riichi_declared[seat],
                player_wind=SEAT_WINDS[seat],
                round_wind=ROUND_WIND,
                is_haitei=not wall,
                is_rinshan=current_is_rinshan,
            )
            if tsumo_score is not None:
                _apply_tsumo_scores(scores, seat, tsumo_score)
                return TurnResult(
                    winner_seat=seat,
                    win_type="tsumo",
                    win_score=tsumo_score,
                )

        can_riichi = (
            current_drawn_tile is not None
            and not riichi_declared[seat]
            and _can_declare_riichi(melds[seat])
        )
        can_riichi_discards = riichi_discards(hand) if can_riichi else []
        closed_kan_tiles = (
            legal_closed_kan_tiles(hand)
            if current_drawn_tile is not None and not riichi_declared[seat]
            else []
        )

        if riichi_declared[seat]:
            if current_drawn_tile is None:
                return TurnResult(reason=f"riichi_seat_without_draw:{seat}")
            action: dict[str, object] = {"action": "discard", "discard": current_drawn_tile}
        elif seat == task.agent_seat:
            prompt = build_mahjong_riichi_prompt(
                task,
                draw_number=len(draws),
                drawn_tile=current_drawn_tile,
                hand=hand,
                discards=discards,
                melds=_melds_for_output(melds),
                riichi_declared=riichi_declared,
                scores=scores,
                remaining_tiles=len(wall),
                can_riichi_discards=can_riichi_discards,
                legal_closed_kan_tiles=closed_kan_tiles,
            )
            raw = agent.generate(prompt, task)
            raw_outputs.append(raw)
            action = extract_riichi_action(raw) or {}
            if not action:
                return TurnResult(reason="no_action_extracted")
            agent_actions.append(action)
        elif opponent_ais is not None and seat in opponent_ais:
            request = _build_external_turn_request(
                task,
                seat=seat,
                hand=hand,
                discards=discards,
                melds=melds,
                riichi_declared=riichi_declared,
                scores=scores,
                wall=wall,
                draws=draws,
                mjai_events=mjai_events,
                drawn_tile=current_drawn_tile,
                from_call=from_call,
                is_rinshan=current_is_rinshan,
                can_riichi_discards=can_riichi_discards,
                legal_closed_kan_tiles=closed_kan_tiles,
            )
            try:
                response = opponent_ais[seat].choose(request)
            except MahjongAIError as exc:
                return TurnResult(reason=f"external_ai_error:seat{seat}:{exc}")
            action = extract_riichi_action(response.raw_output) or {}
            if not action:
                return TurnResult(reason=f"external_ai_no_action:seat{seat}")
        else:
            action = choose_bot_action(hand, can_riichi=can_riichi)

        action_name = action.get("action")
        if action_name == "kan":
            tile = action.get("tile")
            if tile is None and len(closed_kan_tiles) == 1:
                tile = closed_kan_tiles[0]
            if not isinstance(tile, str) or tile not in closed_kan_tiles:
                return TurnResult(reason=f"illegal_closed_kan:{tile}")
            for _copy in range(4):
                hand.remove(tile)
            melds[seat].append(
                CalledMeld(
                    type="kan",
                    tiles=(tile,) * 4,
                    called_tile=tile,
                    opened=False,
                    who=seat,
                    from_who=seat,
                )
            )
            mjai_events.append(
                {
                    "type": "ankan",
                    "actor": seat,
                    "pais": [tile, tile, tile, tile],
                    "consumed": [tile, tile, tile, tile],
                }
            )
            if not wall:
                return TurnResult(reason="wall_empty_after_closed_kan")
            replacement_tile = wall.pop(0)
            hand.append(replacement_tile)
            draws.append(f"seat{seat}:{replacement_tile}:rinshan")
            _mjai_record_tsumo(mjai_events, seat, replacement_tile)
            last_drawn_tiles[seat] = replacement_tile
            current_drawn_tile = replacement_tile
            current_is_rinshan = True
            from_call = False
            continue

        if action_name not in {"discard", "riichi"}:
            return TurnResult(reason=f"illegal_turn_action:{action_name}")

        discard = action.get("discard")
        if not isinstance(discard, str):
            return TurnResult(reason="missing_discard")
        if discard not in hand:
            return TurnResult(reason=f"illegal_discard:{discard}")
        if action_name == "riichi":
            if discard not in can_riichi_discards:
                return TurnResult(reason=f"illegal_riichi_discard:{discard}")
            mjai_events.append({"type": "reach", "actor": seat})
            riichi_declared[seat] = True
            scores[seat] -= 1000

        hand.remove(discard)
        discards[seat].append(discard)
        mjai_events.append(
            {
                "type": "dahai",
                "actor": seat,
                "pai": discard,
                "tsumogiri": last_drawn_tiles[seat] == discard,
            }
        )
        last_drawn_tiles[seat] = None
        if action_name == "riichi":
            mjai_events.append({"type": "reach_accepted", "actor": seat})
        return TurnResult(discard=discard)


def _choose_call(
    task: MahjongRiichiTask,
    agent: Agent,
    hands: list[list[str]],
    melds: list[list[CalledMeld]],
    discards: list[list[str]],
    riichi_declared: list[bool],
    scores: list[int],
    wall: list[str],
    raw_outputs: list[str],
    agent_actions: list[dict[str, object]],
    mjai_events: list[dict[str, object]],
    *,
    discarder_seat: int,
    discard: str,
    opponent_ais: dict[int, ExternalMahjongAI] | None = None,
) -> tuple[CallOption | None, str | None]:
    options: list[CallOption] = []
    for offset in range(1, 4):
        seat = (discarder_seat + offset) % 4
        options.extend(
            legal_call_options(
                hands[seat],
                caller_seat=seat,
                discarder_seat=discarder_seat,
                discard=discard,
                caller_in_riichi=riichi_declared[seat],
            )
        )
    if not options:
        return None, None

    high_priority = [option for option in options if option.action in {"kan", "pon"}]
    candidate_options = high_priority or [option for option in options if option.action == "chi"]

    for offset in range(1, 4):
        seat = (discarder_seat + offset) % 4
        seat_options = [option for option in candidate_options if option.seat == seat]
        if not seat_options:
            continue

        if seat == task.agent_seat:
            prompt = build_mahjong_riichi_call_prompt(
                task,
                discarded_tile=discard,
                discarder_seat=discarder_seat,
                hand=hands[seat],
                discards=discards,
                melds=_melds_for_output(melds),
                riichi_declared=riichi_declared,
                scores=scores,
                remaining_tiles=len(wall),
                legal_call_options=[option.to_prompt_dict() for option in seat_options],
            )
            raw = agent.generate(prompt, task)
            raw_outputs.append(raw)
            action = extract_riichi_action(raw)
            if action is None:
                return None, "no_call_action_extracted"
            agent_actions.append(action)
            if action["action"] == "pass":
                continue
            chosen = _match_call_action(action, seat_options)
            if chosen is None:
                return None, f"illegal_call:{action['action']}"
            return chosen, None

        if opponent_ais is not None and seat in opponent_ais:
            request = _build_external_call_request(
                task,
                seat=seat,
                discarder_seat=discarder_seat,
                discarded_tile=discard,
                hand=hands[seat],
                discards=discards,
                melds=melds,
                riichi_declared=riichi_declared,
                scores=scores,
                wall=wall,
                mjai_events=mjai_events,
                legal_call_options=seat_options,
            )
            try:
                response = opponent_ais[seat].choose(request)
            except MahjongAIError as exc:
                return None, f"external_ai_error:seat{seat}:{exc}"
            action = extract_riichi_action(response.raw_output)
            if action is None:
                return None, f"external_ai_no_call_action:seat{seat}"
            if action["action"] == "pass":
                continue
            chosen = _match_call_action(action, seat_options)
            if chosen is None:
                return None, f"external_ai_illegal_call:seat{seat}:{action['action']}"
            return chosen, None

        chosen = choose_bot_call(
            hands[seat],
            seat_options,
            seat=seat,
            seat_melds=melds[seat],
        )
        if chosen is not None:
            return chosen, None

    return None, None


def _build_external_turn_request(
    task: MahjongRiichiTask,
    *,
    seat: int,
    hand: list[str],
    discards: list[list[str]],
    melds: list[list[CalledMeld]],
    riichi_declared: list[bool],
    scores: list[int],
    wall: list[str],
    draws: list[str],
    mjai_events: list[dict[str, object]],
    drawn_tile: str | None,
    from_call: bool,
    is_rinshan: bool,
    can_riichi_discards: list[str],
    legal_closed_kan_tiles: list[str],
) -> dict[str, object]:
    return {
        "protocol": "minibench-riichi-v1",
        "decision": "turn",
        "task_id": task.id,
        "seed": task.seed,
        "seat": seat,
        "agent_seat": task.agent_seat,
        "draw_number": len(draws),
        "drawn_tile": drawn_tile,
        "from_call": from_call,
        "is_rinshan": is_rinshan,
        "hand": list(hand),
        "discards": [list(seat_discards) for seat_discards in discards],
        "melds": _melds_for_output(melds),
        "riichi_declared": list(riichi_declared),
        "scores": list(scores),
        "remaining_tiles": len(wall),
        "mjai_events": _masked_mjai_events(mjai_events, seat),
        "can_riichi_discards": list(can_riichi_discards),
        "legal_closed_kan_tiles": list(legal_closed_kan_tiles),
        "legal_actions": _legal_turn_actions(
            hand,
            can_riichi_discards=can_riichi_discards,
            legal_closed_kan_tiles=legal_closed_kan_tiles,
        ),
    }


def _build_external_call_request(
    task: MahjongRiichiTask,
    *,
    seat: int,
    discarder_seat: int,
    discarded_tile: str,
    hand: list[str],
    discards: list[list[str]],
    melds: list[list[CalledMeld]],
    riichi_declared: list[bool],
    scores: list[int],
    wall: list[str],
    mjai_events: list[dict[str, object]],
    legal_call_options: list[CallOption],
) -> dict[str, object]:
    return {
        "protocol": "minibench-riichi-v1",
        "decision": "call",
        "task_id": task.id,
        "seed": task.seed,
        "seat": seat,
        "agent_seat": task.agent_seat,
        "discarder_seat": discarder_seat,
        "discarded_tile": discarded_tile,
        "hand": list(hand),
        "discards": [list(seat_discards) for seat_discards in discards],
        "melds": _melds_for_output(melds),
        "riichi_declared": list(riichi_declared),
        "scores": list(scores),
        "remaining_tiles": len(wall),
        "mjai_events": _masked_mjai_events(mjai_events, seat),
        "legal_call_options": [option.to_prompt_dict() for option in legal_call_options],
        "legal_actions": _legal_call_actions(legal_call_options),
    }


def _legal_turn_actions(
    hand: list[str],
    *,
    can_riichi_discards: list[str],
    legal_closed_kan_tiles: list[str],
) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = [
        {"action": "discard", "tile": tile, "discard": tile}
        for tile in sorted(set(hand), key=tile_to_index)
    ]
    actions.extend(
        {"action": "riichi", "tile": tile, "discard": tile}
        for tile in can_riichi_discards
    )
    actions.extend({"action": "kan", "tile": tile} for tile in legal_closed_kan_tiles)
    return actions


def _legal_call_actions(options: list[CallOption]) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = [{"action": "pass"}]
    for option in options:
        if option.action == "chi":
            actions.append(
                {
                    "action": "chi",
                    "tiles": list(option.tiles),
                    "called_tile": option.called_tile,
                }
            )
        elif option.action in {"pon", "kan"}:
            actions.append(
                {
                    "action": option.action,
                    "tile": option.called_tile,
                    "called_tile": option.called_tile,
                    "tiles": list(option.tiles),
                }
            )
    return actions


def _initial_mjai_events(
    task: MahjongRiichiTask,
    hands: list[list[str]],
    scores: list[int],
) -> list[dict[str, object]]:
    return [
        {"type": "start_game"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "5m",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": list(scores),
            "tehais": [list(hand) for hand in hands],
            "minibench_task_id": task.id,
            "minibench_seed": task.seed,
        },
    ]


def _mjai_record_tsumo(
    mjai_events: list[dict[str, object]],
    seat: int,
    tile: str,
) -> None:
    mjai_events.append({"type": "tsumo", "actor": seat, "pai": tile})


def _masked_mjai_events(
    mjai_events: list[dict[str, object]],
    perspective_seat: int,
) -> list[dict[str, object]]:
    masked: list[dict[str, object]] = []
    for event in mjai_events:
        copied = dict(event)
        event_type = copied.get("type")
        if event_type == "start_kyoku":
            tehais = copied.get("tehais")
            if isinstance(tehais, list):
                copied["tehais"] = [
                    list(tehai) if seat == perspective_seat else ["?"] * len(tehai)
                    for seat, tehai in enumerate(tehais)
                    if isinstance(tehai, list)
                ]
        elif event_type == "tsumo" and copied.get("actor") != perspective_seat:
            copied["pai"] = "?"
        masked.append(copied)
    return masked


def _execute_call(
    option: CallOption,
    hands: list[list[str]],
    melds: list[list[CalledMeld]],
    discards: list[list[str]],
    wall: list[str],
    draws: list[str],
    mjai_events: list[dict[str, object]],
    last_drawn_tiles: list[str | None],
) -> tuple[PendingTurn | None, str | None]:
    if not discards[option.from_seat] or discards[option.from_seat][-1] != option.called_tile:
        return None, "called_discard_not_available"
    discards[option.from_seat].pop()

    hand = hands[option.seat]
    consumed = list(option.tiles)
    consumed.remove(option.called_tile)
    missing = [tile for tile in consumed if tile not in hand]
    if missing:
        return None, f"call_tiles_missing:{' '.join(missing)}"
    for tile in consumed:
        hand.remove(tile)

    if option.action == "kan":
        mjai_type = "daiminkan"
    else:
        mjai_type = option.action
    mjai_events.append(
        {
            "type": mjai_type,
            "actor": option.seat,
            "target": option.from_seat,
            "pai": option.called_tile,
            "consumed": consumed,
        }
    )

    melds[option.seat].append(
        CalledMeld(
            type=option.action,
            tiles=option.tiles,
            called_tile=option.called_tile,
            opened=True,
            who=option.seat,
            from_who=option.from_seat,
        )
    )

    if option.action == "kan":
        if not wall:
            return None, "wall_empty_after_open_kan"
        replacement_tile = wall.pop(0)
        hand.append(replacement_tile)
        draws.append(f"seat{option.seat}:{replacement_tile}:rinshan")
        _mjai_record_tsumo(mjai_events, option.seat, replacement_tile)
        last_drawn_tiles[option.seat] = replacement_tile
        return PendingTurn(option.seat, replacement_tile, from_call=True, is_rinshan=True), None

    last_drawn_tiles[option.seat] = None
    return PendingTurn(option.seat, None, from_call=True), None


def _match_call_action(
    action: dict[str, object],
    options: list[CallOption],
) -> CallOption | None:
    action_name = action.get("action")
    matches = [option for option in options if option.action == action_name]
    if not matches:
        return None

    if action_name == "chi":
        tiles = action.get("tiles")
        if tiles is None and len(matches) == 1:
            return matches[0]
        if not isinstance(tiles, list):
            return None
        tile_tuple = tuple(str(tile) for tile in tiles)
        return next((option for option in matches if option.tiles == tile_tuple), None)

    tile = action.get("tile")
    if tile is None and len(matches) == 1:
        return matches[0]
    if not isinstance(tile, str):
        return None
    return next((option for option in matches if option.called_tile == tile), None)


def _concealed_after_call(hand: list[str], option: CallOption) -> list[str] | None:
    after = list(hand)
    consumed = list(option.tiles)
    consumed.remove(option.called_tile)
    for tile in consumed:
        if tile not in after:
            return None
        after.remove(tile)
    return after


def _safe_shanten(hand: list[str]) -> int:
    try:
        return calculate_shanten(hand)
    except ValueError:
        return 99


def _find_ron(
    hands: list[list[str]],
    melds: list[list[CalledMeld]],
    *,
    active_seat: int,
    discard: str,
    riichi_declared: list[bool],
    is_houtei: bool,
) -> tuple[int, dict[str, object]] | None:
    for offset in range(1, 4):
        seat = (active_seat + offset) % 4
        score = _score_player_hand(
            [*hands[seat], discard],
            melds[seat],
            win_tile=discard,
            is_tsumo=False,
            is_riichi=riichi_declared[seat],
            player_wind=SEAT_WINDS[seat],
            round_wind=ROUND_WIND,
            is_houtei=is_houtei,
        )
        if score is not None:
            return seat, score
    return None


def _score_player_hand(
    concealed_tiles: list[str],
    seat_melds: list[CalledMeld],
    *,
    win_tile: str,
    is_tsumo: bool,
    is_riichi: bool,
    player_wind: int,
    round_wind: int,
    is_haitei: bool = False,
    is_houtei: bool = False,
    is_rinshan: bool = False,
) -> dict[str, object] | None:
    all_tiles = list(concealed_tiles)
    for meld in seat_melds:
        all_tiles.extend(meld.tiles)
    if win_tile not in all_tiles:
        all_tiles.append(win_tile)

    return score_closed_hand(
        all_tiles,
        win_tile=win_tile,
        is_tsumo=is_tsumo,
        is_riichi=is_riichi,
        player_wind=player_wind,
        round_wind=round_wind,
        is_haitei=is_haitei,
        is_houtei=is_houtei,
        is_rinshan=is_rinshan,
        melds=[meld.to_score_meld() for meld in seat_melds],
    )


def _legal_chi_options(
    hand: list[str],
    caller_seat: int,
    discarder_seat: int,
    discard: str,
) -> list[CallOption]:
    discard_index = tile_to_index(discard)
    if discard_index >= 27:
        return []

    suit_start = (discard_index // 9) * 9
    rank = discard_index - suit_start
    counts = Counter(hand)
    options: list[CallOption] = []
    for start_rank in (rank - 2, rank - 1, rank):
        if start_rank < 0 or start_rank > 6:
            continue
        sequence_indices = [suit_start + start_rank + offset for offset in range(3)]
        if discard_index not in sequence_indices:
            continue
        sequence = tuple(_index_to_tile(index) for index in sequence_indices)
        needed = list(sequence)
        needed.remove(discard)
        if all(counts[tile] >= needed.count(tile) for tile in set(needed)):
            options.append(
                CallOption(
                    action="chi",
                    seat=caller_seat,
                    from_seat=discarder_seat,
                    tiles=sequence,
                    called_tile=discard,
                )
            )
    return options


def _can_declare_riichi(seat_melds: list[CalledMeld]) -> bool:
    return not any(meld.opened for meld in seat_melds)


def _melds_for_output(melds: list[list[CalledMeld]]) -> list[list[dict[str, object]]]:
    return [[meld.to_output() for meld in seat_melds] for seat_melds in melds]


def _index_to_tile(index: int) -> str:
    if index < 9:
        return f"{index + 1}m"
    if index < 18:
        return f"{index - 8}p"
    if index < 27:
        return f"{index - 17}s"
    honors = {27: "E", 28: "S", 29: "W", 30: "N", 31: "P", 32: "F", 33: "C"}
    return honors[index]


def _apply_ron_scores(
    scores: list[int],
    winner: int,
    discarder: int,
    win_score: dict[str, object],
) -> None:
    cost = win_score["cost"]
    if not isinstance(cost, dict):
        return
    payment = int(cost.get("main", 0))
    scores[winner] += payment
    scores[discarder] -= payment


def _apply_tsumo_scores(
    scores: list[int],
    winner: int,
    win_score: dict[str, object],
) -> None:
    cost = win_score["cost"]
    if not isinstance(cost, dict):
        return
    dealer_payment = int(cost.get("main", 0))
    child_payment = int(cost.get("additional", dealer_payment))
    for seat in range(4):
        if seat == winner:
            continue
        payment = dealer_payment if seat == 0 or winner == 0 else child_payment
        scores[seat] -= payment
        scores[winner] += payment


def _parse_json_object(output: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", output, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None

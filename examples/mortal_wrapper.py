"""MiniBench external-opponent wrapper for Mortal.

MiniBench sends one JSON request per decision. The request contains masked
``mjai_events`` from the target seat's perspective plus MiniBench legal actions.
This wrapper runs Mortal on those mjai events and converts Mortal's last legal
mjai action back into MiniBench's action schema.

Configuration:
    MORTAL_MODEL_DIR=/path/to/model/dir
    MORTAL_IMAGE=mortal:latest

Optional:
    MORTAL_COMMAND='docker run -i --rm -v {model_dir}:/mnt mortal:latest {seat}'
    MORTAL_TIMEOUT=30
"""

from __future__ import annotations

from collections import Counter
import json
import os
import shlex
import subprocess
import sys
from typing import Any


def main() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        action = choose_with_mortal(request)
        print(json.dumps(action, ensure_ascii=False), flush=True)
    return 0


def choose_with_mortal(request: dict[str, Any]) -> dict[str, object]:
    seat = int(request["seat"])
    events = request.get("mjai_events")
    if not isinstance(events, list) or not events:
        raise RuntimeError("request is missing mjai_events")

    completed = subprocess.run(
        _mortal_command(seat),
        input=_events_to_jsonl(events),
        text=True,
        capture_output=True,
        timeout=float(os.environ.get("MORTAL_TIMEOUT", "30")),
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Mortal exited with code {completed.returncode}: {detail}")

    mortal_events = _parse_json_lines(completed.stdout)
    if not mortal_events:
        raise RuntimeError("Mortal returned no JSON actions")

    action = _select_legal_action(mortal_events, request)
    if action is None:
        raise RuntimeError(f"Mortal returned no action matching legal_actions: {completed.stdout}")
    return action


def _mortal_command(seat: int) -> list[str]:
    model_dir = os.environ.get("MORTAL_MODEL_DIR")
    command_template = os.environ.get("MORTAL_COMMAND")
    if command_template:
        if "{seat}" in command_template or "{model_dir}" in command_template:
            command = command_template.format(seat=seat, model_dir=model_dir or "")
        else:
            command = f"{command_template} {seat}"
        return shlex.split(command, posix=os.name != "nt")

    if not model_dir:
        raise RuntimeError("set MORTAL_MODEL_DIR or MORTAL_COMMAND")

    image = os.environ.get("MORTAL_IMAGE", "mortal:latest")
    return [
        "docker",
        "run",
        "-i",
        "--rm",
        "-v",
        f"{model_dir}:/mnt",
        image,
        str(seat),
    ]


def _events_to_jsonl(events: list[Any]) -> str:
    return "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n"


def _parse_json_lines(output: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _select_legal_action(
    mortal_events: list[dict[str, Any]],
    request: dict[str, Any],
) -> dict[str, object] | None:
    legal_actions = request.get("legal_actions")
    if not isinstance(legal_actions, list):
        return None
    seat = int(request["seat"])
    declared_reach = any(
        event.get("type") == "reach" and event.get("actor") == seat
        for event in mortal_events
    )

    for event in reversed(mortal_events):
        action = _mortal_event_to_action(event, declared_reach=declared_reach)
        if action is None:
            continue
        matched = _match_legal_action(action, legal_actions)
        if matched is not None:
            return matched
    return None


def _mortal_event_to_action(
    event: dict[str, Any],
    *,
    declared_reach: bool,
) -> dict[str, object] | None:
    event_type = event.get("type")
    if event_type == "none":
        return {"action": "pass"}
    if event_type == "dahai":
        tile = _to_minibench_tile(event.get("pai"))
        if tile is None:
            return None
        action = "riichi" if declared_reach else "discard"
        return {"action": action, "tile": tile, "discard": tile}
    if event_type == "chi":
        called = _to_minibench_tile(event.get("pai"))
        consumed = [_to_minibench_tile(tile) for tile in event.get("consumed", [])]
        if called is None or any(tile is None for tile in consumed):
            return None
        tiles = sorted([called, *[str(tile) for tile in consumed]], key=_tile_sort_key)
        return {"action": "chi", "tiles": tiles, "called_tile": called}
    if event_type == "pon":
        tile = _to_minibench_tile(event.get("pai"))
        return {"action": "pon", "tile": tile} if tile else None
    if event_type in {"daiminkan", "ankan", "kakan"}:
        tile = _to_minibench_tile(event.get("pai"))
        if tile is None:
            pais = event.get("pais") or event.get("consumed") or []
            if pais:
                tile = _to_minibench_tile(pais[0])
        return {"action": "kan", "tile": tile} if tile else None
    return None


def _match_legal_action(
    action: dict[str, object],
    legal_actions: list[Any],
) -> dict[str, object] | None:
    action_name = action.get("action")
    for legal in legal_actions:
        if not isinstance(legal, dict) or legal.get("action") != action_name:
            continue
        if action_name in {"discard", "riichi"}:
            if _action_tile(legal) == _action_tile(action):
                return dict(legal)
        elif action_name in {"pon", "kan"}:
            if _action_tile(legal) == _action_tile(action):
                return dict(legal)
        elif action_name == "chi":
            if _tile_counter(legal.get("tiles")) == _tile_counter(action.get("tiles")):
                return dict(legal)
        elif action_name == "pass":
            return dict(legal)
    return None


def _action_tile(action: dict[str, object]) -> str | None:
    tile = action.get("tile") or action.get("discard") or action.get("called_tile")
    return _to_minibench_tile(tile)


def _tile_counter(value: object) -> Counter[str]:
    if not isinstance(value, list):
        return Counter()
    tiles = [_to_minibench_tile(tile) for tile in value]
    return Counter(str(tile) for tile in tiles if tile is not None)


def _to_minibench_tile(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if len(value) == 3 and value.endswith("r"):
        return value[:2]
    return value


def _tile_sort_key(tile: str) -> int:
    honors = {"E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33}
    if tile in honors:
        return honors[tile]
    rank = int(tile[0]) - 1
    suit = {"m": 0, "p": 9, "s": 18}[tile[1]]
    return suit + rank


if __name__ == "__main__":
    raise SystemExit(main())

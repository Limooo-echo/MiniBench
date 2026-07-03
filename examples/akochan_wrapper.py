"""MiniBench external-opponent wrapper for akochan.

akochan is a C++ Riichi Mahjong AI that ships its strategy parameters in the
repository's params/ directory. For solo scorer requests, this wrapper keeps one
akochan mjai client alive per MiniBench task and feeds only the newly appended
mjai events on later decisions. Other request types use the conservative
one-client-per-decision path.

Configuration:
    AKOCHAN_HOME=/path/to/akochan

Optional:
    AKOCHAN_COMMAND='{home}/system mjai_client {port} setup_mjai.json'
    AKOCHAN_TIMEOUT=30
    AKOCHAN_CONDA_PREFIX=/path/to/conda/env
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from typing import Any


def main() -> int:
    session: AkochanSession | None = None
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            request = json.loads(line)
            if _uses_persistent_session(request):
                key = _session_key(request)
                if session is None or session.key != key or not session.is_alive():
                    if session is not None:
                        session.close()
                    session = AkochanSession(request)
                try:
                    action = session.choose(request)
                except Exception:
                    session.close()
                    session = AkochanSession(request)
                    action = session.choose(request)
            else:
                if session is not None:
                    session.close()
                    session = None
                action = choose_with_akochan(request)
            print(json.dumps(action, ensure_ascii=False), flush=True)
    finally:
        if session is not None:
            session.close()
    return 0


class AkochanSession:
    def __init__(self, request: dict[str, Any]):
        self.key = _session_key(request)
        self.timeout = float(os.environ.get("AKOCHAN_TIMEOUT", "30"))
        self.events_sent = 0
        self.conn: socket.socket | None = None
        self.process: subprocess.Popen[str] | None = None
        self._start()

    def _start(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            server.settimeout(self.timeout)
            port = server.getsockname()[1]

            self.process = subprocess.Popen(
                _akochan_command(port),
                cwd=str(_akochan_home()),
                env=_akochan_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.conn, _addr = server.accept()
            self.conn.settimeout(self.timeout)
            _send_event(self.conn, {"type": "hello"})
            _recv_response(self.conn)
        except Exception:
            self.close()
            raise
        finally:
            server.close()

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None and self.conn is not None

    def choose(self, request: dict[str, Any]) -> dict[str, object]:
        legal_actions = request.get("legal_actions")
        if not isinstance(legal_actions, list) or not legal_actions:
            raise RuntimeError("request is missing legal_actions")
        if self.conn is None:
            raise RuntimeError("akochan session socket is closed")

        events = _prepare_events(request)
        if len(events) < self.events_sent:
            raise RuntimeError("akochan session event history went backwards")
        if len(events) == self.events_sent:
            raise RuntimeError("akochan session received no new mjai events")

        responses: list[dict[str, Any]] = []
        for index in range(self.events_sent, len(events)):
            outgoing = dict(events[index])
            if index == len(events) - 1 and request.get("decision") == "call":
                outgoing["possible_actions"] = _possible_actions(request)

            _send_event(self.conn, outgoing)
            response = _recv_response(self.conn)
            self.events_sent = index + 1
            if index != len(events) - 1:
                continue

            responses.append(response)
            if response.get("type") == "reach":
                _send_event(self.conn, response)
                responses.append(_recv_response(self.conn))
            break

        action = _select_legal_action(responses, request)
        if action is not None:
            return action
        return _fallback_legal_action(legal_actions)

    def close(self) -> None:
        conn = self.conn
        self.conn = None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        process = self.process
        self.process = None
        if process is not None:
            _stop_process(process)


def choose_with_akochan(request: dict[str, Any]) -> dict[str, object]:
    legal_actions = request.get("legal_actions")
    if not isinstance(legal_actions, list) or not legal_actions:
        raise RuntimeError("request is missing legal_actions")

    events = _prepare_events(request)
    timeout = float(os.environ.get("AKOCHAN_TIMEOUT", "30"))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(timeout)
        port = server.getsockname()[1]

        process = subprocess.Popen(
            _akochan_command(port),
            cwd=str(_akochan_home()),
            env=_akochan_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            conn, _addr = server.accept()
            with conn:
                conn.settimeout(timeout)
                try:
                    responses = _drive_mjai_session(conn, request, events)
                except Exception as exc:
                    detail = _process_detail(process)
                    raise RuntimeError(f"{exc}{detail}") from exc
        except socket.timeout as exc:
            detail = _process_detail(process)
            raise RuntimeError(f"akochan timed out{detail}") from exc
        finally:
            _stop_process(process)

    action = _select_legal_action(responses, request)
    if action is not None:
        return action

    # Keep the benchmark running even when akochan proposes a move outside the
    # locally simplified legal-action surface, e.g. hora in a state where
    # MiniBench did not expose a win action.
    return _fallback_legal_action(legal_actions)


def _uses_persistent_session(request: dict[str, Any]) -> bool:
    return request.get("protocol") == "minibench-mahjong-solo-v1"


def _session_key(request: dict[str, Any]) -> tuple[str, int]:
    task_id = str(request.get("task_id") or request.get("protocol") or "unknown")
    seat = int(request.get("seat", 0))
    return task_id, seat


def _drive_mjai_session(
    conn: socket.socket,
    request: dict[str, Any],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []

    _send_event(conn, {"type": "hello"})
    _recv_response(conn)

    for index, event in enumerate(events):
        outgoing = dict(event)
        if index == len(events) - 1 and request.get("decision") == "call":
            outgoing["possible_actions"] = _possible_actions(request)

        _send_event(conn, outgoing)
        response = _recv_response(conn)
        if index != len(events) - 1:
            continue

        responses.append(response)
        if response.get("type") == "reach":
            _send_event(conn, response)
            responses.append(_recv_response(conn))
        break

    return responses


def _prepare_events(request: dict[str, Any]) -> list[dict[str, Any]]:
    seat = int(request["seat"])
    events = request.get("mjai_events")
    if not isinstance(events, list) or not events:
        raise RuntimeError("request is missing mjai_events")

    prepared: list[dict[str, Any]] = []
    for raw in events:
        if not isinstance(raw, dict):
            continue
        event = dict(raw)
        if event.get("type") == "start_game":
            event["id"] = seat
            event.setdefault("names", ["MiniBench", "Akochan", "Akochan", "Akochan"])
        elif (
            event.get("type") in {"chi", "pon", "daiminkan"}
            and event.get("actor") == seat
            and prepared
            and prepared[-1].get("type") == "dahai"
        ):
            prepared[-1]["possible_actions"] = [event]
        prepared.append(event)
    return prepared


def _akochan_home() -> Path:
    configured = os.environ.get("AKOCHAN_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "akochan"


def _akochan_command(port: int) -> list[str]:
    command_template = os.environ.get("AKOCHAN_COMMAND")
    home = _akochan_home()
    if command_template:
        return shlex.split(
            command_template.format(port=port, home=str(home)),
            posix=os.name != "nt",
        )

    executable = home / "system"
    if not executable.exists():
        executable = home / "system.exe"
    if not executable.exists():
        raise RuntimeError(f"akochan executable not found: {executable}")
    return [str(executable), "mjai_client", str(port), "setup_mjai.json"]


def _akochan_env() -> dict[str, str]:
    env = os.environ.copy()
    home = str(_akochan_home())
    candidates = [home]

    conda_prefix = os.environ.get("AKOCHAN_CONDA_PREFIX") or os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(str(Path(conda_prefix).expanduser() / "lib"))

    existing = env.get("LD_LIBRARY_PATH")
    if existing:
        candidates.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(candidates)
    return env


def _send_event(conn: socket.socket, event: dict[str, Any]) -> None:
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
    conn.sendall(payload.encode("utf-8"))


def _recv_response(conn: socket.socket) -> dict[str, Any]:
    line = _recv_line(conn)
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"akochan returned invalid JSON: {line}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"akochan returned non-object JSON: {line}")
    return parsed


def _recv_line(conn: socket.socket) -> str:
    chunks: list[bytes] = []
    while True:
        chunk = conn.recv(1)
        if not chunk:
            raise RuntimeError("akochan closed the socket")
        if chunk == b"\n":
            return b"".join(chunks).decode("utf-8", errors="replace").strip()
        chunks.append(chunk)


def _possible_actions(request: dict[str, Any]) -> list[dict[str, Any]]:
    legal_actions = request.get("legal_actions")
    if not isinstance(legal_actions, list):
        return []
    possible: list[dict[str, Any]] = []
    for action in legal_actions:
        if not isinstance(action, dict):
            continue
        converted = _minibench_to_mjai_action(action, request)
        if converted is not None:
            possible.append(converted)
    return possible


def _minibench_to_mjai_action(
    action: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any] | None:
    seat = int(request["seat"])
    action_name = action.get("action")
    if action_name == "pass":
        return {"type": "none"}
    if action_name == "discard":
        tile = _action_tile(action)
        return {"type": "dahai", "actor": seat, "pai": tile, "tsumogiri": False} if tile else None
    if action_name == "riichi":
        return {"type": "reach", "actor": seat}
    if action_name == "chi":
        called = _action_tile(action)
        consumed = _consumed_tiles(action)
        if called is None or len(consumed) != 2:
            return None
        return {
            "type": "chi",
            "actor": seat,
            "target": int(request.get("discarder_seat", -1)),
            "pai": called,
            "consumed": consumed,
        }
    if action_name == "pon":
        tile = _action_tile(action)
        if tile is None:
            return None
        return {
            "type": "pon",
            "actor": seat,
            "target": int(request.get("discarder_seat", -1)),
            "pai": tile,
            "consumed": [tile, tile],
        }
    if action_name == "kan":
        tile = _action_tile(action)
        if tile is None:
            return None
        return {
            "type": "daiminkan",
            "actor": seat,
            "target": int(request.get("discarder_seat", -1)),
            "pai": tile,
            "consumed": [tile, tile, tile],
        }
    return None


def _select_legal_action(
    akochan_events: list[dict[str, Any]],
    request: dict[str, Any],
) -> dict[str, object] | None:
    legal_actions = request.get("legal_actions")
    if not isinstance(legal_actions, list):
        return None
    declared_reach = any(event.get("type") == "reach" for event in akochan_events)

    for event in reversed(akochan_events):
        action = _mjai_event_to_action(event, declared_reach=declared_reach)
        if action is None:
            continue
        matched = _match_legal_action(action, legal_actions)
        if matched is not None:
            return matched
        if action.get("action") == "riichi":
            discard_action = {
                "action": "discard",
                "tile": action.get("tile"),
                "discard": action.get("discard") or action.get("tile"),
            }
            matched = _match_legal_action(discard_action, legal_actions)
            if matched is not None:
                return matched
    return None


def _mjai_event_to_action(
    event: dict[str, Any],
    *,
    declared_reach: bool,
) -> dict[str, object] | None:
    event_type = event.get("type")
    if event_type == "hora":
        return {"action": "tsumo"}
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
        return {"action": "pon", "tile": tile, "called_tile": tile} if tile else None
    if event_type in {"daiminkan", "ankan", "kakan"}:
        tile = _to_minibench_tile(event.get("pai"))
        if tile is None:
            pais = event.get("pais") or event.get("consumed") or []
            if pais:
                tile = _to_minibench_tile(pais[0])
        return {"action": "kan", "tile": tile, "called_tile": tile} if tile else None
    return None


def _match_legal_action(
    action: dict[str, object],
    legal_actions: list[Any],
) -> dict[str, object] | None:
    action_name = action.get("action")
    for legal in legal_actions:
        if not isinstance(legal, dict) or legal.get("action") != action_name:
            continue
        if action_name in {"discard", "riichi", "pon", "kan"}:
            if _action_tile(legal) == _action_tile(action):
                return dict(legal)
        elif action_name == "chi":
            if _tile_counter(legal.get("tiles")) == _tile_counter(action.get("tiles")):
                return dict(legal)
        elif action_name == "pass":
            return dict(legal)
    return None


def _fallback_legal_action(legal_actions: list[Any]) -> dict[str, object]:
    for action in legal_actions:
        if isinstance(action, dict) and action.get("action") == "pass":
            return dict(action)
    for action in legal_actions:
        if isinstance(action, dict):
            return dict(action)
    return {"action": "pass"}


def _action_tile(action: dict[str, object]) -> str | None:
    tile = action.get("tile") or action.get("discard") or action.get("called_tile")
    return _to_minibench_tile(tile)


def _consumed_tiles(action: dict[str, object]) -> list[str]:
    called = _action_tile(action)
    tiles = action.get("tiles")
    if called is None or not isinstance(tiles, list):
        return []
    consumed = [_to_minibench_tile(tile) for tile in tiles]
    consumed = [str(tile) for tile in consumed if tile is not None]
    if called in consumed:
        consumed.remove(called)
    return consumed


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


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()


def _process_detail(process: subprocess.Popen[str]) -> str:
    output = ""
    if process.poll() is None:
        return ""
    if process.stdout is not None:
        try:
            output = process.stdout.read().strip()
        except OSError:
            output = ""
    return f": {output}" if output else ""


if __name__ == "__main__":
    raise SystemExit(main())

"""Cross-process account request gates. No API keys enter shared state."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import threading
import time
from typing import Any

from minibench.core.token_usage import known_total_tokens


class RateGate:
    """Shared by isolated workers through a multiprocessing manager.

    RPM and in-flight limits are hard local bounds. TPM/total-token guards
    reserve an estimate, then reconcile usage. Without an output cap there
    is no finite guaranteed upper bound on a request's tokens.
    """

    def __init__(self, accounts: dict[str, dict[str, Any]], window_seconds: float = 60):
        self.window = window_seconds
        self.condition = threading.Condition()
        self.cancelled = False
        self.states = {}
        for key, config in accounts.items():
            config = dict(config)
            allowed = {"max_in_flight", "requests_per_minute", "tokens_per_minute",
                       "reserve_tokens", "total_token_budget"}
            if set(config) - allowed:
                raise ValueError("Unknown account limit fields")
            for field in allowed:
                value = config.get(field)
                if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                    raise ValueError(f"{key}.{field} must be a positive integer or null")
            config.setdefault("max_in_flight", 8)
            config.setdefault("reserve_tokens", 32768)
            if config["max_in_flight"] is None or config["reserve_tokens"] is None:
                raise ValueError("max_in_flight and reserve_tokens cannot be null")
            reserve = config["reserve_tokens"]
            for field in ("tokens_per_minute", "total_token_budget"):
                if config.get(field) is not None and config[field] < reserve:
                    raise ValueError(f"{key}.{field} is below one request reservation")
            self.states[key] = dict(config=config, starts=deque(), usage=deque(),
                                    active=0, spent=0, unknown=0, calls=0,
                                    known_spent=0, known_calls=0,
                                    overshoots=0)

    def cancel(self):
        with self.condition:
            self.cancelled = True
            self.condition.notify_all()

    def acquire(self, key: str):
        with self.condition:
            state = self.states[key]
            cfg = state["config"]
            while True:
                if self.cancelled:
                    raise RuntimeError("Cloud run stopped; no new API calls permitted")
                now = time.monotonic()
                while state["starts"] and state["starts"][0] <= now - self.window:
                    state["starts"].popleft()
                while state["usage"] and state["usage"][0][0] <= now - self.window:
                    state["usage"].popleft()
                reserve = cfg["reserve_tokens"]
                pending = state["active"] * reserve
                budget = cfg.get("total_token_budget")
                if budget is not None and state["spent"] + pending + reserve > budget:
                    # Pending requests may release unused reservations; wait for them.
                    if state["active"] == 0:
                        raise RuntimeError(f"Local token budget exhausted for account {key}")
                    self.condition.wait(.1)
                    continue
                rpm = cfg.get("requests_per_minute")
                tpm = cfg.get("tokens_per_minute")
                rolling = sum(tokens for _, tokens in state["usage"]) + pending
                if (state["active"] < cfg["max_in_flight"]
                    and (rpm is None or len(state["starts"]) < rpm)
                    and (tpm is None or rolling + reserve <= tpm)):
                    state["starts"].append(now)
                    state["active"] += 1
                    state["calls"] += 1
                    return
                self.condition.wait(.1)

    def release(self, key: str, usage: dict | None):
        with self.condition:
            state = self.states[key]
            if state["active"] < 1:
                raise RuntimeError("Unbalanced account reservation release")
            tokens = known_total_tokens(usage)
            if tokens is None:
                tokens = state["config"]["reserve_tokens"]
                state["unknown"] += 1
            else:
                tokens = int(tokens)
                state["known_spent"] += tokens
                state["known_calls"] += 1
                if tokens > state["config"]["reserve_tokens"]:
                    state["overshoots"] += 1
            state["spent"] += tokens
            state["usage"].append((time.monotonic(), tokens))
            state["active"] -= 1
            self.condition.notify_all()

    def try_acquire(self, key: str) -> bool:
        """One nonblocking reservation attempt for interruptible cloud clients.

        Never leave a blocking manager RPC that could grant a slot after the
        caller's decision has already expired. The existing acquire API remains
        unchanged for other callers.
        """
        with self.condition:
            if self.cancelled:
                raise RuntimeError("Cloud run stopped; no new API calls permitted")
            state = self.states[key]
            cfg = state["config"]
            now = time.monotonic()
            for queue in ("starts", "usage"):
                while state[queue] and (state[queue][0] if queue == "starts" else state[queue][0][0]) <= now-self.window:
                    state[queue].popleft()
            reserve = cfg["reserve_tokens"]
            pending = state["active"]*reserve
            budget = cfg.get("total_token_budget")
            if budget is not None and state["spent"]+pending+reserve > budget:
                if state["active"] == 0:
                    raise RuntimeError(f"Local token budget exhausted for account {key}")
                return False
            rpm, tpm = cfg.get("requests_per_minute"), cfg.get("tokens_per_minute")
            rolling = sum(tokens for _, tokens in state["usage"])+pending
            if (state["active"] >= cfg["max_in_flight"]
                    or (rpm is not None and len(state["starts"]) >= rpm)
                    or (tpm is not None and rolling+reserve > tpm)):
                return False
            state["starts"].append(now)
            state["active"] += 1
            state["calls"] += 1
            return True

    def snapshot(self) -> dict:
        with self.condition:
            return {key: {
                "active_calls": state["active"], "network_calls": state["calls"],
                "accounted_or_reserved_tokens": state["spent"],
                "unknown_usage_calls": state["unknown"],
                "reservation_overshoots": state["overshoots"],
                "known_total_tokens": state["known_spent"] if state["known_calls"] else None,
                "total_tokens": (state["known_spent"] if state["calls"] > 0
                                 and state["unknown"] == 0 and state["active"] == 0 else None),
                "usage_complete": state["calls"] > 0 and state["unknown"] == 0 and state["active"] == 0,
                "active_reserved_tokens": state["active"]*state["config"]["reserve_tokens"],
            } for key, state in self.states.items()}

    def _recent_time(self, at):
        if not at:
            return None
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(at)).total_seconds()
        return time.monotonic() - max(0, elapsed) if elapsed < self.window else None

    def restore_start(self, key: str, at: str):
        """Retain recent request starts across a manual restart's RPM window."""
        with self.condition:
            recent = self._recent_time(at)
            if recent is not None:
                state = self.states[key]
                state["starts"].append(recent)
                state["starts"] = deque(sorted(state["starts"]))

    def restore(self, key: str, usage: dict | None, at: str | None = None):
        """Restore quota accounting and recent completions, not active calls."""
        with self.condition:
            state = self.states[key]
            tokens = known_total_tokens(usage)
            if tokens is None:
                tokens = state["config"]["reserve_tokens"]
                state["unknown"] += 1
            else:
                state["known_spent"] += tokens
                state["known_calls"] += 1
            state["spent"] += int(tokens)
            state["calls"] += 1
            recent = self._recent_time(at)
            if recent is not None:
                state["usage"].append((recent, int(tokens)))
                state["usage"] = deque(sorted(state["usage"]))

"""Hard wall deadline for one complete Xiangqi Agent decision on Linux/WSL.

The scope is outside the Agent architecture: intermediate reasoning, streaming
chunks, voting, refinement, and provider retries cannot renew this timer.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import math
import signal
import threading
from time import monotonic

FINAL_MOVE_TIMEOUT_SECONDS = 900.0
_active_deadline: ContextVar[tuple[float, float, float] | None] = ContextVar(
    "xiangqi_final_move_deadline", default=None,
)


class FinalMoveTimeout(BaseException):
    """A scored decision failure, not a provider outage or a retryable timeout.

    BaseException is deliberate: generic ``except Exception`` retry/fallback
    handlers inside an Agent must not swallow the outer benchmark deadline.
    Only the four native Xiangqi evaluators turn this into a failed observation.
    """
    def __init__(self, seconds: float, elapsed_seconds: float):
        self.seconds = seconds
        self.elapsed_seconds = elapsed_seconds
        super().__init__(
            f"Agent returned no final move within {seconds:g} seconds "
            f"(decision elapsed {elapsed_seconds:.3f}s)"
        )


def validate_move_timeout(seconds: object) -> float:
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or not 0 < seconds <= FINAL_MOVE_TIMEOUT_SECONDS):
        raise ValueError("final_move_timeout_seconds must be positive and at most 900 seconds")
    return float(seconds)


class _DeadlineAlarm:
    def __init__(self, started, seconds):
        self.started, self.seconds = started, seconds
        self.deferred, self.pending = 0, False

    def __call__(self, signum, frame):
        if self.deferred:
            self.pending = True
            return
        raise FinalMoveTimeout(self.seconds, monotonic()-self.started)

    @contextmanager
    def defer_atomic_timeout(self):
        # Python-level deferral also works when a Pikafish reader thread caused
        # the process-directed signal to be delivered outside the main thread.
        self.deferred += 1
        try:
            yield
        except BaseException:
            self.pending = False  # Preserve an actual RPC failure or user interrupt.
            raise
        finally:
            self.deferred -= 1
            if self.pending and self.deferred == 0:
                self.pending = False
                raise FinalMoveTimeout(self.seconds, monotonic()-self.started)


@contextmanager
def final_move_deadline(seconds: float = FINAL_MOVE_TIMEOUT_SECONDS):
    """Interrupt a blocked request even while it continues sending reasoning.

    A decision starts immediately before Agent.generate*(), including its own
    request-queue waits and all internal stages. Judge/render time and time in
    the batch queue are outside the decision. Each actual next move starts a
    fresh decision; no intermediate response does. There is no token cap.
    """
    seconds = validate_move_timeout(seconds)
    active = _active_deadline.get()
    if active is not None:
        # Nested calls belong to the SAME decision, never a fresh allowance.
        yield
        if monotonic() >= active[1]:
            raise FinalMoveTimeout(active[2], monotonic()-active[0])
        return
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Xiangqi final-move deadlines require a Linux/WSL main-thread worker")
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        raise RuntimeError("An existing SIGALRM timer conflicts with the Xiangqi decision deadline")
    started = monotonic()
    token = _active_deadline.set((started, started+seconds, seconds))
    previous_handler = signal.getsignal(signal.SIGALRM)

    try:
        signal.signal(signal.SIGALRM, _DeadlineAlarm(started, seconds))
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
        # Also reject a late return if an external component delayed signals.
        if monotonic()-started >= seconds:
            raise FinalMoveTimeout(seconds, monotonic()-started)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        _active_deadline.reset(token)


def call_agent_move(callback, *args, timeout_seconds=FINAL_MOVE_TIMEOUT_SECONDS, **kwargs):
    with final_move_deadline(timeout_seconds):
        return callback(*args, **kwargs)

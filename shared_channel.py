"""
In-process, synchronous hand-off channel between the RL side (GymEnv, on the
main thread) and the DES simulator (mGym_DesEnv.runDes, on a daemon thread).

RL contract guaranteed here: a correct RL/DES loop is a strict alternation
(Gym sends action #i -> DES receives it -> DES sends result #i -> Gym
receives it -> #i+1, ...). This channel assigns a monotonically increasing
sequence number to every action/result and checks the two counters stay
locked together on every hand-off, so a break in synchronicity is reported
immediately rather than silently corrupting the (s, a, r, s') tuple.

Logging: sync breaks always log at ERROR; per-step hand-offs at DEBUG (off
by default); a periodic INFO heartbeat every ``log_every`` steps and a full
summary at episode end. Logs go to console and ``rl_des_sync.log``.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
# Logging setup (configured once, reused by every StepChannel instance)
# --------------------------------------------------------------------------- #
_LOG_FILE = "rl_des_sync.log"


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("rl_des_channel")
    if logger.handlers:  # already configured (e.g. re-import) -> don't duplicate
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | rl_des_sync | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)        # console: heartbeats + breaks
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        file_handler = logging.FileHandler(_LOG_FILE, mode="a")
        file_handler.setLevel(logging.DEBUG)   # file: full detail incl. per-step DEBUG
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        # If the file cannot be opened (read-only fs, etc.) keep console logging.
        pass

    return logger


log = _build_logger()


class SyncBreakError(RuntimeError):
    """Raised (optionally) when the action/result alternation is violated."""


class ChannelStopped(Exception):
    """Internal sentinel used to unblock a waiting DES thread on reset/shutdown."""


@dataclass
class _Result:
    observation: Any
    reward: float
    terminated: bool
    info: Dict[str, Any] = field(default_factory=dict)
    seq: int = 0


class StepChannel:
    """
    Synchronous one-action / one-result rendezvous between the Gym (main
    thread: send_action then receive_result) and the DES (worker thread:
    send_result then wait_for_action). All four ops are cheap -- the two
    send_* calls are non-blocking (set an event); the receivers block on
    the peer's event. Each side sets its own event before waiting on the
    peer's, so the rendezvous is deadlock-free.
    """

    def __init__(self, name: str = "default", log_every: int = 200,
                 raise_on_break: bool = False, debug_each_step: bool = False):
        """
        name: label used in log lines, so concurrent train/eval channels
            are distinguishable. log_every: INFO heartbeat interval.
        raise_on_break: if True, a synchronicity break raises
            SyncBreakError instead of only being logged/counted (default
            False, so one glitch doesn't abort a long training run).
        debug_each_step: log every hand-off at DEBUG (verbose).
        """
        self.name = name
        self.log_every = max(1, int(log_every))
        self.raise_on_break = raise_on_break
        self.debug_each_step = debug_each_step

        # One lock + condition guards the hand-off. Each direction is a
        # size-1 slot: a producer that finds its slot still full BLOCKS
        # until the consumer drains it -- this backpressure enforces
        # strict alternation while letting either side arrive first.
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._stopped = False

        self._action_slot: Optional[tuple] = None    # (seq, action) Gym -> DES
        self._result_slot: Optional[_Result] = None  # DES -> Gym

        # Sequence bookkeeping -- the core of the synchronicity guarantee.
        self._action_seq = 0
        self._result_seq = 0
        self._break_count = 0

        self._episode_index = 0
        self._t_last_action: Optional[float] = None
        self._last_reward: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #
    def prepare_for_episode(self, episode_index: Optional[int] = None) -> None:
        """Reset all hand-off state for a fresh episode. Called by
        GymEnv.reset() before the new DES thread starts."""
        with self._cond:
            self._stopped = False
            self._action_slot = None
            self._result_slot = None
            self._action_seq = 0
            self._result_seq = 0
            self._break_count = 0
            self._t_last_action = None
            self._last_reward = None
            if episode_index is not None:
                self._episode_index = episode_index
            self._cond.notify_all()
        log.info("[%s] channel armed for episode %s (seq reset to 0)",
                 self.name, self._episode_index)

    def request_stop(self) -> None:
        """Ask a (possibly blocked) DES thread to abort; safe to call even
        if no thread is running. Wakes every waiter to unwind via
        ChannelStopped."""
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        log.info("[%s] stop requested (episode %s, after %d in-sync steps)",
                 self.name, self._episode_index, self._result_seq)

    # ------------------------------------------------------------------ #
    # Gym side (main thread)
    # ------------------------------------------------------------------ #
    def send_action(self, action: int, timeout: float = 120.0) -> int:
        """Hand an action to the DES; returns the assigned sequence number.
        Blocks only if the DES hasn't yet consumed the previous action."""
        with self._cond:
            if not self._cond.wait_for(
                    lambda: self._action_slot is None or self._stopped, timeout):
                self._report_break(
                    "TIMEOUT: DES never consumed the previous action "
                    f"#{self._action_seq} within {timeout}s"
                )
                raise TimeoutError(
                    f"[{self.name}] DES did not consume action "
                    f"#{self._action_seq} within {timeout}s"
                )
            if self._stopped:
                raise ChannelStopped("channel stopped while Gym sent an action")
            self._action_seq += 1
            seq = self._action_seq
            self._action_slot = (seq, int(action))
            self._t_last_action = time.perf_counter()
            self._cond.notify_all()
        if self.debug_each_step:
            log.debug("[%s] -> action #%d = %s", self.name, seq, action)
        return seq

    def receive_result(self, timeout: float = 120.0) -> Dict[str, Any]:
        """Block until the DES returns the result for the action just sent.
        Returns a dict with keys observation/reward/terminated/info; raises
        TimeoutError if the DES doesn't respond in time."""
        with self._cond:
            if not self._cond.wait_for(
                    lambda: self._result_slot is not None or self._stopped, timeout):
                self._report_break(
                    f"TIMEOUT: DES produced no result for action #{self._action_seq} "
                    f"within {timeout}s"
                )
                raise TimeoutError(
                    f"[{self.name}] DES did not respond for action "
                    f"#{self._action_seq} within {timeout}s"
                )
            if self._result_slot is None:           # woken only by stop
                raise ChannelStopped("channel stopped while Gym awaited a result")
            result = self._result_slot
            self._result_slot = None
            # Synchronicity check: the i-th result must answer the i-th action.
            if result.seq != self._action_seq:
                self._report_break(
                    "Gym received a result that does not match its action "
                    f"(expected result #{self._action_seq}, got #{result.seq})"
                )
            else:
                self._last_reward = result.reward
                self._heartbeat(result)
            self._cond.notify_all()
        return {
            "observation": result.observation,
            "reward": result.reward,
            "terminated": result.terminated,
            "info": result.info,
        }

    # ------------------------------------------------------------------ #
    # DES side (worker thread)
    # ------------------------------------------------------------------ #
    def send_result(self, observation: Any, reward: float,
                    terminated: bool, info: Optional[Dict[str, Any]] = None,
                    timeout: float = 120.0) -> int:
        """Hand an observation+reward back to the Gym; returns the sequence
        number. observation/info are deep-copied so the DES thread can keep
        mutating its own structures without aliasing what the agent sees."""
        with self._cond:
            if not self._cond.wait_for(
                    lambda: self._result_slot is None or self._stopped, timeout):
                self._report_break(
                    "TIMEOUT: Gym never consumed the previous result "
                    f"#{self._result_seq} within {timeout}s"
                )
                raise TimeoutError(
                    f"[{self.name}] Gym did not consume result "
                    f"#{self._result_seq} within {timeout}s"
                )
            if self._stopped:
                # Episode is being torn down; drop this result quietly rather
                # than raising up through the Salabim generator.
                return -1
            self._result_seq += 1
            seq = self._result_seq
            self._result_slot = _Result(
                observation=copy.deepcopy(observation),
                reward=float(reward),
                terminated=bool(terminated),
                info=copy.deepcopy(info) if info is not None else {},
                seq=seq,
            )
            self._cond.notify_all()
        if self.debug_each_step:
            log.debug("[%s] <- result #%d (reward=%.4f, terminated=%s)",
                      self.name, seq, float(reward), bool(terminated))
        return seq

    def wait_for_action(self, timeout: float = 120.0) -> Optional[int]:
        """Block until the Gym sends the next action; returns it as an int,
        or raises ChannelStopped if a reset/shutdown was requested."""
        with self._cond:
            if not self._cond.wait_for(
                    lambda: self._action_slot is not None or self._stopped, timeout):
                self._report_break(
                    f"TIMEOUT: Gym sent no action for decision #{self._result_seq} "
                    f"within {timeout}s"
                )
                raise TimeoutError(
                    f"[{self.name}] Gym did not send an action for decision "
                    f"#{self._result_seq} within {timeout}s"
                )
            if self._action_slot is None:           # woken only by stop
                raise ChannelStopped("channel stopped while DES awaited an action")
            seq, action = self._action_slot
            self._action_slot = None
            # Synchronicity check: at a decision point the DES has just sent its
            # i-th result, so it must now consume the i-th action.
            if seq != self._result_seq:
                self._report_break(
                    "DES consumed an action out of step with its results "
                    f"(action #{seq}, but last result was #{self._result_seq})"
                )
            self._cond.notify_all()
        return action

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def _report_break(self, message: str) -> None:
        """Record and loudly log a synchronicity break."""
        self._break_count += 1
        log.error("[%s] *** SYNC BREAK #%d (episode %s) *** %s",
                  self.name, self._break_count, self._episode_index, message)
        if self.raise_on_break:
            raise SyncBreakError(f"[{self.name}] {message}")

    def _heartbeat(self, result: _Result) -> None:
        """Emit a periodic INFO line confirming the loop is still in sync."""
        if result.seq % self.log_every == 0:
            dt_ms = None
            if self._t_last_action is not None:
                dt_ms = (time.perf_counter() - self._t_last_action) * 1e3
            log.info(
                "[%s] in sync at step %d (episode %s) | reward=%.4f | "
                "round-trip=%s | breaks so far=%d",
                self.name, result.seq, self._episode_index, result.reward,
                f"{dt_ms:.2f}ms" if dt_ms is not None else "n/a",
                self._break_count,
            )

    def episode_summary(self) -> Dict[str, int]:
        """Return and log a short integrity summary for the episode."""
        in_sync = (self._result_seq in (self._action_seq, self._action_seq + 1)
                   and self._break_count == 0)
        summary = {
            "episode": self._episode_index,
            "actions_sent": self._action_seq,
            "results_sent": self._result_seq,
            "sync_breaks": self._break_count,
            "in_sync": in_sync,
        }
        level = logging.INFO if in_sync else logging.ERROR
        log.log(
            level,
            "[%s] EPISODE %s SUMMARY: actions=%d results=%d breaks=%d -> %s",
            self.name, self._episode_index, self._action_seq, self._result_seq,
            self._break_count, "IN SYNC" if in_sync else "OUT OF SYNC",
        )
        return summary

    # Convenience read-only accessors -------------------------------------- #
    @property
    def sync_breaks(self) -> int:
        return self._break_count

    @property
    def steps_completed(self) -> int:
        return self._result_seq

    def has_pending_result(self) -> bool:
        """Return True if a result is already sitting in the channel slot.

        Used by GymEnv.step() to tell a DES thread that exited after sending
        its terminal result (slot holds a real result -> drain it) apart
        from one that crashed before sending anything (slot empty -> error).
        Without this, is_des_alive() misfires on the former case and
        discards the real PVOL, producing zero rows in eval_episodes_raw.
        """
        with self._lock:
            return self._result_slot is not None
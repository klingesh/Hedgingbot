"""
Single-instance lock. OS-level, not a PID file.

The audit found:

    N3 / 7. No lock implementation — Documented lock configuration exists, but no
    actual locking implementation was found. Concurrent overlay instances could
    produce duplicate actions once trading is enabled.

Exactly right: `lock_path` was in the config and checked by preflight for
collisions with Tradingbot's, but nothing ever acquired it. In observe mode two
instances are merely wasteful. In Phase 2 they would each independently decide the
same breach needs hedging and place the hedge twice — and because each would then
see the other's position, the second cycle could over-hedge in the opposite
direction.

This is a direct port of the approach Tradingbot arrived at in src/live/lock.py,
including the two lessons it learned the hard way:

1. USE AN OS LOCK, NOT A PID FILE. A PID file left behind by a killed process
   blocks every future start, and a recycled PID makes it worse. fcntl.flock and
   msvcrt.locking are released by the kernel when the process dies, however it
   dies.

2. KEEP A MODULE-LEVEL REFERENCE. Tradingbot shipped a bug where
   `SingleInstance().acquire()` was called and the result discarded; the object was
   garbage-collected, its file handle closed, and the kernel dropped the lock — so
   the lock silently did nothing. `_held` prevents that recurring here.

Pure stdlib. Works on Windows (msvcrt) and POSIX (fcntl).
"""

from __future__ import annotations

import os
import sys

DEFAULT_LOCK_PATH = os.path.join("logs", "hedge.lock")


class AlreadyRunning(RuntimeError):
    """Another instance holds the lock."""


class SingleInstance:
    """An exclusive OS-level lock on a file, held for the process lifetime."""

    def __init__(self, path: str = DEFAULT_LOCK_PATH) -> None:
        self.path = path
        self._fh = None

    def acquire(self) -> "SingleInstance":
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Opened "a+" so the file is created if absent and never truncated —
        # truncating would race another instance mid-check.
        self._fh = open(self.path, "a+", encoding="utf-8")

        try:
            if sys.platform.startswith("win"):
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._close()
            raise AlreadyRunning(
                f"another Hedgingbot instance already holds {self.path}. "
                "Two overlays on one account would each decide the same breach "
                "needs hedging and place the hedge twice."
            ) from exc

        # Record who holds it. Diagnostic only — the lock itself is the OS lock,
        # so a stale PID here is harmless.
        try:
            self._fh.seek(0)
            self._fh.truncate()
            self._fh.write(f"pid={os.getpid()}\n")
            self._fh.flush()
        except OSError:
            pass

        return self

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if sys.platform.startswith("win"):
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self._close()

    def _close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
        except OSError:
            pass
        self._fh = None

    def __enter__(self) -> "SingleInstance":
        return self.acquire()

    def __exit__(self, *_exc) -> None:
        self.release()


#: Module-level reference so the lock object is never garbage-collected while the
#: process runs. See lesson 2 in the module docstring — this is not redundant.
_held: SingleInstance | None = None


def hold(path: str = DEFAULT_LOCK_PATH) -> SingleInstance:
    """Acquire the single-instance lock and keep it for the process lifetime.

    Raises AlreadyRunning if another instance has it. Idempotent: calling twice in
    one process returns the same lock rather than deadlocking against itself.
    """
    global _held
    if _held is not None:
        return _held
    _held = SingleInstance(path).acquire()
    return _held


def release() -> None:
    """Release the process-wide lock. Mainly for tests."""
    global _held
    if _held is not None:
        _held.release()
        _held = None

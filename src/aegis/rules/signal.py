"""Simulated traffic light (stand-in for a real signal classifier).

Cycles RED -> GREEN -> ORANGE, `cycle` seconds each, driven by wall-clock
time since the run started. Phase 2 proper will replace this with a
vision-measured signal timeline; the violation logic won't need to change.
"""

from __future__ import annotations

import time

ORDER = ("RED", "GREEN", "ORANGE")


class SimSignal:
    """A traffic light that just cycles on a timer."""

    def __init__(self, cycle: float = 5.0) -> None:
        if cycle <= 0:
            raise ValueError(f"cycle must be positive, got {cycle}")
        self.cycle = cycle
        self._t0 = time.monotonic()

    def state_at(self, elapsed: float) -> str:
        """State `elapsed` seconds after start (pure function — easy to test)."""
        return ORDER[int(elapsed / self.cycle) % len(ORDER)]

    def state_now(self) -> str:
        """Current state according to the wall clock."""
        return self.state_at(time.monotonic() - self._t0)

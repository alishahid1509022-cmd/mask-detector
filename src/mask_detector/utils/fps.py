"""Frame-timing utilities: measuring FPS and pacing a loop to a target FPS.

Kept separate from `utils/drawing.py` (which draws detection overlays) so
that anything needing only webcam timing - like a plain camera preview -
doesn't have to depend on detection-related code at all.
"""

from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """Rolling-average FPS counter over the last ``window`` frame timestamps.

    A simple "1 / (time since last frame)" measurement is noisy and jumps
    around a lot frame to frame. Averaging over a small rolling window gives
    a much more readable, stable number to display on screen.
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Record that a frame was just processed and return the current FPS."""
        now = time.monotonic()
        self._timestamps.append(now)

        # Need at least two samples to compute an interval.
        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0

        # (N-1) intervals span the window, not N.
        return (len(self._timestamps) - 1) / elapsed


class FrameRateLimiter:
    """Paces a loop to a target FPS by sleeping out any leftover frame budget.

    Without this, a capture/display loop runs as fast as the CPU/camera
    will allow - which wastes power, can overwhelm slower downstream
    processing (like a detector), and produces a jittery, inconsistent
    frame rate. Call :meth:`wait` once per loop iteration, right before the
    iteration ends, and it will sleep just long enough to keep the loop
    running at (at most) ``target_fps``.
    """

    def __init__(self, target_fps: float = 30.0) -> None:
        # A non-positive target disables limiting entirely (run as fast as possible).
        self.target_frame_duration = 1.0 / target_fps if target_fps > 0 else 0.0
        self._last_tick: float | None = None

    def wait(self) -> None:
        """Block just long enough to keep the loop at the target FPS."""
        if self.target_frame_duration <= 0:
            return

        now = time.monotonic()
        if self._last_tick is not None:
            elapsed = now - self._last_tick
            remaining = self.target_frame_duration - elapsed
            if remaining > 0:
                time.sleep(remaining)

        self._last_tick = time.monotonic()

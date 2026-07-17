"""A small, thread-safe "don't repeat this more than once every N seconds" timer.

Shared by anything that needs to debounce a repeated event on sustained
no-mask detections - voice alerts (``AlertManager``) and automatic
screenshots (``ScreenshotManager``) both compose this instead of each
re-implementing their own timing logic.
"""

from __future__ import annotations

import threading
import time


class Cooldown:
    """Returns True from :meth:`ready` at most once per ``seconds`` window."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._last_fired = 0.0
        self._lock = threading.Lock()

    def ready(self) -> bool:
        """Check (and, if allowed, consume) the cooldown.

        Returns True and resets the timer if enough time has passed since
        the last time this returned True; otherwise returns False without
        changing any state.
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_fired < self.seconds:
                return False
            self._last_fired = now
            return True

    def set_seconds(self, seconds: float) -> None:
        """Update the cooldown window in place (e.g. a sidebar slider changed).

        Mutating an existing ``Cooldown`` instance (rather than throwing it
        away and constructing a new one) matters when this object is behind
        ``st.cache_resource``: it lets the cached owner (``AlertManager`` /
        ``ScreenshotManager``) apply new settings without needing the slider
        value baked into the cache key - see app.py's cached loaders for why
        that would otherwise leak a new cached instance per slider tick.
        """
        with self._lock:
            self.seconds = seconds

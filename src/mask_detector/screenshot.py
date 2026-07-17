"""Automatic screenshot capture on no-mask detections.

Saves a JPEG snapshot of the annotated frame the moment a no-mask detection
occurs, so there's a persistent evidence trail without any manual action.
Debounced by a :class:`~mask_detector.utils.cooldown.Cooldown` timer so a
sustained no-mask streak (30+ frames a second) doesn't write dozens of
near-identical files per second - at most one screenshot per cooldown window.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from mask_detector import config
from mask_detector.utils.cooldown import Cooldown
from mask_detector.utils.logger import get_logger

logger = get_logger(__name__, log_level=config.LOG_LEVEL, log_dir=config.LOG_DIR)

# e.g. "no_mask_2026-07-17_18-53-02.jpg" - sortable, filesystem-safe, and
# self-describing without needing to open the file.
FILENAME_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


class ScreenshotManager:
    """Saves at most one screenshot per cooldown window on no-mask detections."""

    def __init__(
        self,
        output_dir: Path | str = config.SCREENSHOTS_DIR,
        cooldown_seconds: float = config.SCREENSHOT_COOLDOWN_SECONDS,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.cooldown_seconds = cooldown_seconds
        self._cooldown = Cooldown(cooldown_seconds)
        self.total_saved = 0
        # Set whenever a save attempt fails (permissions, disk full, a bad
        # frame, ...) so the UI can surface a one-time, friendly warning
        # instead of either staying silent or re-showing the same toast on
        # every single frame of a sustained no-mask streak. Consumed via
        # pop_last_error(), which clears it after reading.
        self.last_error: str | None = None

        self._ensure_output_dir()

    def update_cooldown(self, cooldown_seconds: float) -> None:
        """Change the cooldown window on this already-constructed instance.

        Lets a cached ``ScreenshotManager`` pick up a new sidebar slider
        value without being recreated - see ``utils.cooldown.Cooldown.set_seconds``.
        """
        self.cooldown_seconds = cooldown_seconds
        self._cooldown.set_seconds(cooldown_seconds)

    def _ensure_output_dir(self) -> bool:
        """Create the screenshots folder if it doesn't exist yet.

        Returns True if the directory exists (or was just created), False if
        creation failed (e.g. permissions) - callers treat False as "skip
        saving this time" rather than crashing the detection loop.
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.exception("Permission denied creating screenshots folder at %s", self.output_dir)
            self.last_error = (
                f"Permission denied creating '{self.output_dir}'. Screenshots won't be saved "
                "until this is fixed."
            )
            return False
        except OSError as exc:
            logger.exception("Could not create screenshots folder at %s", self.output_dir)
            self.last_error = f"Could not create '{self.output_dir}': {exc}"
            return False
        return True

    def pop_last_error(self) -> str | None:
        """Return and clear the most recent save-failure message, if any.

        "Pop" (read-then-clear) semantics mean a caller polling this once
        per frame (see app.py) shows exactly one toast per new failure,
        instead of re-showing the same message on every subsequent frame
        until the underlying problem is fixed.
        """
        error, self.last_error = self.last_error, None
        return error

    def capture(self, frame: np.ndarray, no_mask_detected: bool) -> Path | None:
        """Save ``frame`` if a no-mask detection just occurred and the cooldown allows it.

        Call this once per processed frame. Returns the path the screenshot
        was saved to, or ``None`` if nothing was saved this call (mask
        present, still within the cooldown window, or a save error occurred
        - check :meth:`pop_last_error` to tell the last case apart from the
        first two).
        """
        if not no_mask_detected:
            return None

        if not self._cooldown.ready():
            return None

        if not self._ensure_output_dir():
            return None

        timestamp = datetime.now().strftime(FILENAME_TIMESTAMP_FORMAT)
        destination = self.output_dir / f"no_mask_{timestamp}.jpg"

        try:
            # cv2.imwrite can fail via either a plain OSError (e.g. disk
            # full, permission denied) or its own cv2.error (e.g. an
            # unsupported/corrupt frame array) - both are treated as a
            # recoverable, skip-this-frame failure rather than crashing
            # the detection loop.
            saved = cv2.imwrite(str(destination), frame)
        except PermissionError:
            logger.exception("Permission denied saving screenshot to %s", destination)
            self.last_error = f"Permission denied writing to '{destination}'."
            return None
        except (OSError, cv2.error) as exc:
            logger.exception("Failed to save screenshot to %s", destination)
            self.last_error = f"Failed to save screenshot: {exc}"
            return None

        if not saved:
            logger.warning("cv2.imwrite reported failure writing %s", destination)
            self.last_error = f"Failed to save screenshot to '{destination}'."
            return None

        self.total_saved += 1
        logger.info("Saved no-mask screenshot to %s", destination)
        return destination

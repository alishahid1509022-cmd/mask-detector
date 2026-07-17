"""Debounced alerting: decides *when* to actually fire a voice alert.

Without this layer, a "no mask" detection on every single frame (30+ times
a second) would spam the TTS engine. ``AlertManager`` enforces a cooldown
and gracefully degrades to on-screen-only alerts if the voice backend fails.
"""

from __future__ import annotations

from dataclasses import dataclass

from mask_detector import config
from mask_detector.alerts.voice_alert import VoiceAlertBackend
from mask_detector.utils.cooldown import Cooldown
from mask_detector.utils.exceptions import VoiceAlertError
from mask_detector.utils.logger import get_logger

logger = get_logger(__name__, log_level=config.LOG_LEVEL, log_dir=config.LOG_DIR)

# Sourced from config (env-overridable) rather than hard-coded here, so the
# exact spoken phrase lives in one place alongside every other tunable value.
DEFAULT_MESSAGE = config.ALERT_MESSAGE


@dataclass
class AlertResult:
    triggered: bool
    voice_available: bool


class AlertManager:
    """Fires (rate-limited) voice + on-screen alerts on sustained no-mask detections."""

    def __init__(
        self,
        voice_backend: VoiceAlertBackend | None,
        cooldown_seconds: float = config.ALERT_COOLDOWN_SECONDS,
        message: str = DEFAULT_MESSAGE,
    ) -> None:
        self.voice_backend = voice_backend
        self.cooldown_seconds = cooldown_seconds
        self.message = message

        self._voice_available = voice_backend is not None
        self._cooldown = Cooldown(cooldown_seconds)
        self.total_alerts = 0

    @property
    def voice_available(self) -> bool:
        return self._voice_available

    def update_cooldown(self, cooldown_seconds: float) -> None:
        """Change the cooldown window on this already-constructed instance.

        Lets a cached ``AlertManager`` pick up a new sidebar slider value
        without being recreated - see ``utils.cooldown.Cooldown.set_seconds``.
        """
        self.cooldown_seconds = cooldown_seconds
        self._cooldown.set_seconds(cooldown_seconds)

    def notify(self, no_mask_detected: bool) -> AlertResult:
        """Call once per frame with whether any face is currently mask-less.

        Returns an :class:`AlertResult` describing whether a *new* alert was
        triggered this call (respecting the cooldown) and whether voice
        playback is currently available.
        """
        if not no_mask_detected:
            return AlertResult(triggered=False, voice_available=self._voice_available)

        if not self._cooldown.ready():
            return AlertResult(triggered=False, voice_available=self._voice_available)

        self.total_alerts += 1

        if self._voice_available and self.voice_backend is not None:
            try:
                self.voice_backend.speak(self.message)
            except VoiceAlertError:
                logger.warning(
                    "Voice backend failed; falling back to on-screen alerts only", exc_info=True
                )
                self._voice_available = False

        return AlertResult(triggered=True, voice_available=self._voice_available)

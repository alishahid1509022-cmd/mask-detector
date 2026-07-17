"""Unit tests for mask_detector.alerts.alert_manager."""

from __future__ import annotations

from mask_detector.alerts.alert_manager import AlertManager
from mask_detector.utils.exceptions import VoiceAlertError


class _StubVoiceBackend:
    """Records speak() calls; can be configured to simulate failure."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def speak(self, text: str) -> None:
        if self.fail:
            raise VoiceAlertError("simulated TTS failure")
        self.calls.append(text)


def test_no_alert_when_mask_present() -> None:
    manager = AlertManager(voice_backend=_StubVoiceBackend(), cooldown_seconds=10)
    result = manager.notify(no_mask_detected=False)
    assert result.triggered is False


def test_alert_triggers_and_speaks_on_no_mask() -> None:
    backend = _StubVoiceBackend()
    manager = AlertManager(voice_backend=backend, cooldown_seconds=10)

    result = manager.notify(no_mask_detected=True)

    assert result.triggered is True
    assert result.voice_available is True
    assert backend.calls == [manager.message]
    assert manager.total_alerts == 1


def test_cooldown_prevents_repeated_alerts() -> None:
    backend = _StubVoiceBackend()
    manager = AlertManager(voice_backend=backend, cooldown_seconds=100)

    first = manager.notify(no_mask_detected=True)
    second = manager.notify(no_mask_detected=True)

    assert first.triggered is True
    assert second.triggered is False
    assert len(backend.calls) == 1
    assert manager.total_alerts == 1


def test_voice_failure_falls_back_to_silent_mode() -> None:
    backend = _StubVoiceBackend(fail=True)
    manager = AlertManager(voice_backend=backend, cooldown_seconds=0)

    result = manager.notify(no_mask_detected=True)

    assert result.triggered is True
    assert result.voice_available is False
    assert manager.voice_available is False


def test_no_backend_configured_still_triggers_on_screen_alert() -> None:
    manager = AlertManager(voice_backend=None, cooldown_seconds=0)

    result = manager.notify(no_mask_detected=True)

    assert result.triggered is True
    assert result.voice_available is False


def test_update_cooldown_changes_behavior_without_recreating_the_manager() -> None:
    # Exercises the pattern app.py relies on to avoid caching a brand-new
    # AlertManager per sidebar slider tick (see load_alert_manager).
    backend = _StubVoiceBackend()
    manager = AlertManager(voice_backend=backend, cooldown_seconds=100)

    manager.notify(no_mask_detected=True)
    assert manager.notify(no_mask_detected=True).triggered is False  # still in the old window

    manager.update_cooldown(0)

    assert manager.notify(no_mask_detected=True).triggered is True
    assert manager.cooldown_seconds == 0

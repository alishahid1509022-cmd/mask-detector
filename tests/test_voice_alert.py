"""Unit tests for mask_detector.alerts.voice_alert's cross-platform fallback.

Uses fake backend classes (never touching real pyttsx3/gTTS/OS speech
drivers) so these tests are fast, deterministic, and don't depend on what's
installed/available on the machine running them.
"""

from __future__ import annotations

import pytest

from mask_detector.alerts import voice_alert
from mask_detector.utils.exceptions import VoiceAlertError


class _WorkingBackend(voice_alert.VoiceAlertBackend):
    def speak(self, text: str) -> None:  # pragma: no cover - not exercised here
        pass


class _BrokenBackend(voice_alert.VoiceAlertBackend):
    def __init__(self) -> None:
        raise VoiceAlertError("simulated init failure")

    def speak(self, text: str) -> None:  # pragma: no cover - never reached
        pass


def test_returns_requested_engine_when_it_initializes_fine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        voice_alert, "_BACKENDS", {"pyttsx3": _WorkingBackend, "gtts": _BrokenBackend}
    )
    backend = voice_alert.get_voice_backend("pyttsx3")
    assert isinstance(backend, _WorkingBackend)


def test_falls_back_to_other_engine_when_primary_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulates e.g. pyttsx3 failing on a Linux box with no espeak installed.
    monkeypatch.setattr(
        voice_alert, "_BACKENDS", {"pyttsx3": _BrokenBackend, "gtts": _WorkingBackend}
    )
    backend = voice_alert.get_voice_backend("pyttsx3")
    assert isinstance(backend, _WorkingBackend)


def test_raises_when_both_engines_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        voice_alert, "_BACKENDS", {"pyttsx3": _BrokenBackend, "gtts": _BrokenBackend}
    )
    with pytest.raises(VoiceAlertError):
        voice_alert.get_voice_backend("pyttsx3")


def test_fallback_disabled_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        voice_alert, "_BACKENDS", {"pyttsx3": _BrokenBackend, "gtts": _WorkingBackend}
    )
    with pytest.raises(VoiceAlertError):
        voice_alert.get_voice_backend("pyttsx3", fallback=False)


def test_unknown_engine_name_raises() -> None:
    with pytest.raises(VoiceAlertError):
        voice_alert.get_voice_backend("not-a-real-engine", fallback=False)

"""Unit tests for mask_detector.utils.cooldown.Cooldown."""

from __future__ import annotations

from mask_detector.utils.cooldown import Cooldown


def test_ready_on_first_call() -> None:
    cooldown = Cooldown(seconds=100)
    assert cooldown.ready() is True


def test_not_ready_again_within_window() -> None:
    cooldown = Cooldown(seconds=100)
    assert cooldown.ready() is True
    assert cooldown.ready() is False


def test_zero_second_cooldown_is_always_ready() -> None:
    cooldown = Cooldown(seconds=0)
    assert cooldown.ready() is True
    assert cooldown.ready() is True


def test_set_seconds_updates_the_window_in_place() -> None:
    cooldown = Cooldown(seconds=100)
    assert cooldown.ready() is True
    assert cooldown.ready() is False  # still within the old 100s window

    cooldown.set_seconds(0)

    assert cooldown.ready() is True  # new window is 0s, so immediately ready again

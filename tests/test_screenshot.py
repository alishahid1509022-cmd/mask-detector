"""Unit tests for mask_detector.screenshot.ScreenshotManager.

`cv2.imwrite` is monkeypatched everywhere so these tests never touch a real
image codec or write actual JPEGs to disk - they only verify *when* a save
is attempted and *what path* it would be saved to.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from mask_detector import screenshot as screenshot_module
from mask_detector.screenshot import ScreenshotManager

_FRAME = np.zeros((10, 10, 3), dtype=np.uint8)


class _FakeImwrite:
    """Records every (path, frame) pair passed to it; always "succeeds"."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, path: str, frame: np.ndarray) -> bool:
        self.calls.append(path)
        return True


def test_output_directory_created_on_init(tmp_path) -> None:
    output_dir = tmp_path / "Screenshots"
    ScreenshotManager(output_dir=output_dir, cooldown_seconds=0)
    assert output_dir.is_dir()


def test_no_screenshot_when_mask_present(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_imwrite = _FakeImwrite()
    monkeypatch.setattr(screenshot_module.cv2, "imwrite", fake_imwrite)
    manager = ScreenshotManager(output_dir=tmp_path / "Screenshots", cooldown_seconds=0)

    result = manager.capture(_FRAME, no_mask_detected=False)

    assert result is None
    assert fake_imwrite.calls == []
    assert manager.total_saved == 0


def test_saves_screenshot_on_no_mask_detection(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_imwrite = _FakeImwrite()
    monkeypatch.setattr(screenshot_module.cv2, "imwrite", fake_imwrite)
    output_dir = tmp_path / "Screenshots"
    manager = ScreenshotManager(output_dir=output_dir, cooldown_seconds=0)

    result = manager.capture(_FRAME, no_mask_detected=True)

    assert result is not None
    assert result.parent == output_dir
    assert fake_imwrite.calls == [str(result)]
    assert manager.total_saved == 1


def test_filename_contains_date_and_time(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(screenshot_module.cv2, "imwrite", _FakeImwrite())
    manager = ScreenshotManager(output_dir=tmp_path / "Screenshots", cooldown_seconds=0)

    result = manager.capture(_FRAME, no_mask_detected=True)

    assert result is not None
    assert re.match(r"^no_mask_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.jpg$", result.name)


def test_cooldown_prevents_duplicate_screenshots(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_imwrite = _FakeImwrite()
    monkeypatch.setattr(screenshot_module.cv2, "imwrite", fake_imwrite)
    manager = ScreenshotManager(output_dir=tmp_path / "Screenshots", cooldown_seconds=100)

    first = manager.capture(_FRAME, no_mask_detected=True)
    second = manager.capture(_FRAME, no_mask_detected=True)

    assert first is not None
    assert second is None
    assert len(fake_imwrite.calls) == 1
    assert manager.total_saved == 1


def test_failed_write_does_not_increment_counter(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(screenshot_module.cv2, "imwrite", lambda *_args, **_kwargs: False)
    manager = ScreenshotManager(output_dir=tmp_path / "Screenshots", cooldown_seconds=0)

    result = manager.capture(_FRAME, no_mask_detected=True)

    assert result is None
    assert manager.total_saved == 0


def test_permission_error_writing_screenshot_is_caught_and_recorded(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise_permission_error(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(screenshot_module.cv2, "imwrite", _raise_permission_error)
    manager = ScreenshotManager(output_dir=tmp_path / "Screenshots", cooldown_seconds=0)

    result = manager.capture(_FRAME, no_mask_detected=True)

    assert result is None
    assert manager.total_saved == 0
    assert manager.last_error is not None
    assert "Permission denied" in manager.last_error


def test_cv2_error_writing_screenshot_is_caught_and_recorded(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # cv2.imwrite can fail with cv2's own error type (e.g. an unsupported
    # frame array), not just a plain OSError - both must be recoverable.
    def _raise_cv2_error(*_args: object, **_kwargs: object) -> None:
        raise screenshot_module.cv2.error("unsupported frame")

    monkeypatch.setattr(screenshot_module.cv2, "imwrite", _raise_cv2_error)
    manager = ScreenshotManager(output_dir=tmp_path / "Screenshots", cooldown_seconds=0)

    result = manager.capture(_FRAME, no_mask_detected=True)

    assert result is None
    assert manager.total_saved == 0
    assert manager.last_error is not None


def test_pop_last_error_returns_and_clears_the_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(screenshot_module.cv2, "imwrite", lambda *_args, **_kwargs: False)
    manager = ScreenshotManager(output_dir=tmp_path / "Screenshots", cooldown_seconds=0)
    manager.capture(_FRAME, no_mask_detected=True)
    assert manager.last_error is not None

    first_pop = manager.pop_last_error()
    second_pop = manager.pop_last_error()

    assert first_pop is not None
    assert second_pop is None  # cleared after the first read - no repeat toasts
    assert manager.last_error is None


def test_permission_error_creating_output_dir_is_recorded(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Construct first (needs a real, writable directory to succeed), then
    # break mkdir *after* construction so it only affects capture()'s
    # internal re-check of the output directory, not test setup itself.
    manager = ScreenshotManager(output_dir=tmp_path / "Screenshots", cooldown_seconds=0)

    def _raise_permission_error(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(screenshot_module.Path, "mkdir", _raise_permission_error)

    result = manager.capture(_FRAME, no_mask_detected=True)

    assert result is None
    assert manager.last_error is not None
    assert "Permission denied" in manager.last_error


def test_update_cooldown_changes_behavior_without_recreating_the_manager(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercises the pattern app.py relies on to avoid caching a brand-new
    # ScreenshotManager per sidebar slider tick (see load_screenshot_manager).
    monkeypatch.setattr(screenshot_module.cv2, "imwrite", _FakeImwrite())
    manager = ScreenshotManager(output_dir=tmp_path / "Screenshots", cooldown_seconds=100)

    manager.capture(_FRAME, no_mask_detected=True)
    assert manager.capture(_FRAME, no_mask_detected=True) is None  # still in the old window

    manager.update_cooldown(0)

    assert manager.capture(_FRAME, no_mask_detected=True) is not None
    assert manager.cooldown_seconds == 0

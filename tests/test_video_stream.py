"""Unit tests for mask_detector.video_stream.VideoStream's threaded capture.

A fake cv2.VideoCapture stands in for real hardware so these tests are
fast, deterministic, and run anywhere - they exercise exactly the behavior
that changed when capture moved onto a background thread: the reader
thread starting up, read_frame() never blocking, and the thread being
stopped/joined (not leaked) on release.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from mask_detector import video_stream as video_stream_module
from mask_detector.utils.exceptions import CameraNotAvailableError
from mask_detector.video_stream import VideoStream


class _FakeCapture:
    """Replaces cv2.VideoCapture: hands back an incrementing fake frame."""

    def __init__(self, camera_index: int, *, opens: bool = True) -> None:
        self.camera_index = camera_index
        self._opened = opens
        self._read_count = 0
        self.released = False
        self.props: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - matches cv2's camelCase API
        return self._opened

    def set(self, prop_id: int, value: float) -> bool:  # noqa: N802
        self.props[prop_id] = value
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        self._read_count += 1
        frame = np.full((4, 4, 3), fill_value=self._read_count % 256, dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        self.released = True
        self._opened = False


@pytest.fixture()
def fake_cv2(monkeypatch: pytest.MonkeyPatch):
    created: list[_FakeCapture] = []

    def _factory(camera_index: int, *_args: object) -> _FakeCapture:
        capture = _FakeCapture(camera_index)
        created.append(capture)
        return capture

    monkeypatch.setattr(video_stream_module.cv2, "VideoCapture", _factory)
    return created


def test_raises_when_camera_cannot_be_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        video_stream_module.cv2,
        "VideoCapture",
        lambda index, *_args: _FakeCapture(index, opens=False),
    )
    with pytest.raises(CameraNotAvailableError):
        VideoStream(camera_index=0)


def test_raises_camera_not_available_when_opening_itself_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Some backends/platforms (e.g. a permission-denied camera on Linux) can
    # raise cv2.error during construction itself, rather than just failing
    # isOpened() - this must be normalized into the same exception type as
    # every other "the camera didn't work" case.
    def _raise(_camera_index: int, *_args: object):
        raise video_stream_module.cv2.error("simulated low-level failure")

    monkeypatch.setattr(video_stream_module.cv2, "VideoCapture", _raise)
    with pytest.raises(CameraNotAvailableError):
        VideoStream(camera_index=0)


def test_read_frame_returns_a_frame_once_reader_thread_has_run(fake_cv2) -> None:
    stream = VideoStream(camera_index=0)
    try:
        # The background thread needs a brief moment to grab its first
        # frame; poll instead of a fixed sleep to keep this test fast and
        # not flaky under CI load.
        deadline = time.monotonic() + 2.0
        frame = None
        while frame is None and time.monotonic() < deadline:
            frame = stream.read_frame()
        assert frame is not None
        assert frame.shape == (4, 4, 3)
    finally:
        stream.release()


def test_read_frame_returns_an_independent_copy(fake_cv2) -> None:
    stream = VideoStream(camera_index=0)
    deadline = time.monotonic() + 2.0
    frame = None
    while frame is None and time.monotonic() < deadline:
        frame = stream.read_frame()
    assert frame is not None

    # Freeze capture (stop the reader thread) so the stream's internal
    # frame can no longer change on its own - isolates exactly what this
    # test checks (copy semantics) from unrelated background-thread timing.
    stream._stop_event.set()
    stream._reader_thread.join(timeout=2.0)

    frame[:] = 255  # mutate the caller's copy

    with stream._frame_lock:
        frozen_internal_frame = stream._latest_frame
    # If read_frame() ever stopped copying, this mutation would have
    # corrupted the stream's own internal buffer too.
    assert not np.array_equal(frozen_internal_frame, frame)

    stream.release()


def test_release_stops_the_reader_thread_and_releases_capture(fake_cv2) -> None:
    stream = VideoStream(camera_index=0)
    reader_thread = stream._reader_thread
    assert reader_thread.is_alive()

    stream.release()

    assert not reader_thread.is_alive()
    assert fake_cv2[0].released is True


def test_release_is_safe_to_call_more_than_once(fake_cv2) -> None:
    stream = VideoStream(camera_index=0)
    stream.release()
    stream.release()  # must not raise


def test_context_manager_releases_on_exception(fake_cv2) -> None:
    with pytest.raises(RuntimeError):
        with VideoStream(camera_index=0) as stream:
            raise RuntimeError("simulated failure mid-loop")

    assert fake_cv2[0].released is True
    assert not stream._reader_thread.is_alive()


def test_only_one_reader_thread_is_ever_created(fake_cv2) -> None:
    stream = VideoStream(camera_index=0)
    try:
        active_threads = [t for t in threading.enumerate() if t.name == stream._reader_thread.name]
        assert len(active_threads) == 1
    finally:
        stream.release()


class _AlwaysFailingCapture(_FakeCapture):
    """Simulates a camera that never produces a frame (unplugged, no permission, ...)."""

    def read(self) -> tuple[bool, np.ndarray | None]:
        self._read_count += 1
        return False, None


def test_read_frame_raises_camera_not_available_after_disconnect_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_stream_module.cv2, "VideoCapture", _AlwaysFailingCapture)
    stream = VideoStream(camera_index=0, disconnect_timeout=0.05)
    try:
        deadline = time.monotonic() + 2.0
        while not stream._disconnected.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (
            stream._disconnected.is_set()
        ), "reader thread never flagged the camera as disconnected"

        with pytest.raises(CameraNotAvailableError):
            stream.read_frame()
    finally:
        stream.release()


class _ToggleableCapture(_FakeCapture):
    """A camera that can be flipped between "producing frames" and "dead" mid-test."""

    def __init__(self, camera_index: int, *, opens: bool = True) -> None:
        super().__init__(camera_index, opens=opens)
        self.failing = True

    def read(self) -> tuple[bool, np.ndarray | None]:
        self._read_count += 1
        if self.failing:
            return False, None
        frame = np.full((4, 4, 3), fill_value=self._read_count % 256, dtype=np.uint8)
        return True, frame


def test_disconnected_flag_clears_once_frames_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[_ToggleableCapture] = []

    def _factory(camera_index: int, *_args: object) -> _ToggleableCapture:
        capture = _ToggleableCapture(camera_index)
        created.append(capture)
        return capture

    monkeypatch.setattr(video_stream_module.cv2, "VideoCapture", _factory)
    stream = VideoStream(camera_index=0, disconnect_timeout=0.05)
    try:
        deadline = time.monotonic() + 2.0
        while not stream._disconnected.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert stream._disconnected.is_set()

        created[0].failing = False  # the camera "comes back"

        deadline = time.monotonic() + 2.0
        while stream._disconnected.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not stream._disconnected.is_set()
        assert stream.read_frame() is not None
    finally:
        stream.release()

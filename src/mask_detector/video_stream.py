"""Webcam capture via OpenCV, read on a dedicated background thread.

This module has exactly one job: open a camera, hand back frames, and
release it cleanly - nothing about *displaying* frames or *detecting*
anything lives here. That single responsibility is what lets a plain
webcam preview (see ``scripts/webcam_preview.py``) and the full mask
detection pipeline (``detector.py`` + ``app.py``) both reuse this same
class without depending on each other.
"""

from __future__ import annotations

import sys
import threading
import time
from types import TracebackType

import cv2
import numpy as np

from mask_detector import config
from mask_detector.utils.exceptions import CameraNotAvailableError
from mask_detector.utils.logger import get_logger

logger = get_logger(__name__, log_level=config.LOG_LEVEL, log_dir=config.LOG_DIR)


class VideoStream:
    """Context-manager wrapper around ``cv2.VideoCapture`` with threaded reads.

    ``cv2.VideoCapture.read()`` is a *blocking* call - it waits for the
    camera driver to hand back the next frame, which takes a variable
    amount of time depending on the camera's exposure settings, USB bus
    contention, etc. If the main detection loop called ``read()`` directly,
    that wait would be added to every single iteration, capping the whole
    pipeline's throughput at "however fast the camera can deliver frames"
    even on a machine whose GPU/CPU could otherwise run YOLOv8 much faster.

    To decouple the two, a single daemon background thread continuously
    calls ``cv2.VideoCapture.read()`` in a tight loop and stores only the
    *most recent* frame (never a growing queue - see :meth:`_reader_loop`).
    The main loop's :meth:`read_frame` then just returns that frame under a
    lock, which is effectively free since it never waits on camera I/O.
    This is the standard "threaded video capture" pattern used by most
    real-time OpenCV pipelines, and it is a genuinely good fit for a
    background *thread* (rather than e.g. multiprocessing) specifically
    because ``cv2.VideoCapture.read()`` is I/O-bound: the underlying C++
    call releases Python's GIL while it blocks, so a second Python thread
    can run productively during that wait instead of just queueing behind it.

    Usage::

        with VideoStream(camera_index=0) as stream:
            frame = stream.read_frame()
    """

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        target_fps: float = config.TARGET_FPS,
        disconnect_timeout: float = config.CAMERA_DISCONNECT_TIMEOUT_SECONDS,
    ) -> None:
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.disconnect_timeout = disconnect_timeout

        # cv2.VideoCapture doesn't raise on a bad index/device - it just
        # returns an object that reports isOpened() == False. We turn that
        # into our own explicit exception type so callers can catch a
        # single, well-documented error instead of guessing. On some
        # backends/platforms a genuinely broken device (or a camera the OS
        # has blocked at the permission level) can also raise a low-level
        # cv2.error during construction itself rather than just failing
        # isOpened() - caught here too so callers only ever have to handle
        # one exception type for "the camera didn't work", regardless of
        # which of these two ways it failed.
        # Prefer the platform-native capture backend when OpenCV exposes one.
        # On macOS, CAP_AVFOUNDATION is far more reliable than CAP_ANY (which
        # can try FFMPEG first and report a false "camera failed to initialize"
        # even when the built-in FaceTime camera is present and free). On other
        # OSes we leave the backend unspecified so OpenCV picks its default.
        try:
            if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
                self._capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
            else:
                self._capture = cv2.VideoCapture(camera_index)
            opened = self._capture.isOpened()
        except cv2.error as exc:
            raise CameraNotAvailableError(
                f"Could not open camera at index {camera_index}: {exc}"
            ) from exc

        if not opened:
            self._capture.release()
            raise CameraNotAvailableError(
                f"Could not open camera at index {camera_index}. This usually means "
                "it's disconnected, already in use by another application, or this app "
                "doesn't have camera permission (check your OS's privacy/camera settings)."
            )

        # Best-effort only: many webcams/drivers ignore requested FPS, or
        # only support a fixed set of values. We don't treat a failure to
        # apply this as fatal - actual frame pacing is handled separately
        # by utils.fps.FrameRateLimiter, which works regardless of whether
        # the camera honored this request.
        if target_fps > 0:
            self._capture.set(cv2.CAP_PROP_FPS, target_fps)

        # Ask the driver to keep as little internal buffering as possible,
        # so a read (ours, in the reader thread) returns the newest frame
        # rather than one that's been queued up for a while. Also
        # best-effort - not every backend exposes/honors this property.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._last_success_time = time.monotonic()
        self._disconnected = threading.Event()
        self._stop_event = threading.Event()

        # daemon=True is a safety net, not the primary cleanup mechanism -
        # release() below always stops and joins this thread explicitly.
        # It just guarantees a bug in release() can never prevent the whole
        # Python process from exiting because of a stuck background thread.
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"VideoStream-{camera_index}",
            daemon=True,
        )
        self._reader_thread.start()

        logger.info(
            "Camera %s opened successfully (requested %.1f FPS, threaded capture)",
            camera_index,
            target_fps,
        )

    def _reader_loop(self) -> None:
        """Background-thread body: keep ``_latest_frame`` as fresh as possible.

        Reads as fast as the camera will allow, bounded only by the camera
        itself rather than by however long detection takes on the main
        thread. If the main thread hasn't consumed a frame before the next
        one is grabbed, the old one is simply overwritten (never queued),
        so this thread's memory footprint stays flat no matter how far
        behind the processing loop falls - there is nothing here that can
        accumulate and leak over a long-running session.

        Without ``disconnect_timeout``, a camera that goes away mid-session
        (unplugged, OS revokes permission, driver crashes) would just cause
        every subsequent ``read()`` to fail here forever - but since we'd
        never touch ``_latest_frame`` again, ``read_frame()`` would keep
        silently returning the *last good frame* forever too, making a dead
        camera look identical to a perfectly healthy, unchanging scene. This
        loop instead tracks how long it's been since the last successful
        read and flips ``_disconnected`` once that exceeds the timeout, so
        :meth:`read_frame` can turn it into a real, user-visible error.
        """
        while not self._stop_event.is_set():
            try:
                ok, frame = self._capture.read()
            except cv2.error:
                logger.warning("cv2 raised while reading from camera %s", self.camera_index)
                ok, frame = False, None

            if not ok or frame is None:
                if (
                    not self._disconnected.is_set()
                    and time.monotonic() - self._last_success_time > self.disconnect_timeout
                ):
                    logger.warning(
                        "Camera %s produced no frame for over %.1fs - treating as disconnected",
                        self.camera_index,
                        self.disconnect_timeout,
                    )
                    self._disconnected.set()
                # Avoid a tight, CPU-spinning retry loop if the camera is
                # transiently unavailable (e.g. briefly unplugged).
                time.sleep(0.01)
                continue

            with self._frame_lock:
                self._latest_frame = frame
            self._last_success_time = time.monotonic()
            if self._disconnected.is_set():
                logger.info("Camera %s started producing frames again", self.camera_index)
                self._disconnected.clear()

    def read_frame(self) -> np.ndarray | None:
        """Return the most recently captured frame - never blocks on camera I/O.

        Returns ``None`` if no frame has been captured yet (e.g. immediately
        after opening the camera, before the reader thread's first
        successful read) rather than blocking - callers already treat
        ``None`` as "skip this iteration and try again" (see app.py's
        consecutive-error handling), which is exactly the right behavior here.

        Raises:
            CameraNotAvailableError: if the background reader hasn't
                gotten a successful frame in over ``disconnect_timeout``
                seconds - i.e. the camera was very likely unplugged, lost
                permission, or its driver crashed mid-session.
        """
        if self._disconnected.is_set():
            raise CameraNotAvailableError(
                f"Camera {self.camera_index} stopped responding (no frame for over "
                f"{self.disconnect_timeout:.0f}s). It may have been disconnected, put to sleep, "
                "or had its permission revoked. Reconnect it and click Start Camera again."
            )

        with self._frame_lock:
            if self._latest_frame is None:
                return None
            # Copy while still holding the lock: guarantees the caller gets
            # an independent buffer it can safely read/annotate even if the
            # reader thread immediately grabs and stores a new frame right
            # after this returns. The cost is one small, bounded array copy
            # per frame actually *consumed* by the processing loop - not per
            # frame the camera produces - so it doesn't scale with camera FPS.
            return self._latest_frame.copy()

    def release(self) -> None:
        """Stop the reader thread, then release the underlying camera handle.

        Always stops the thread *before* releasing the capture object, so
        the reader thread never calls ``.read()`` on an already-released
        handle. This ordering, plus always joining with a timeout, is what
        prevents a leaked background thread (and the camera handle it
        holds open) from outliving a stopped/crashed detection session.
        Safe to call more than once.
        """
        self._stop_event.set()
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
            if self._reader_thread.is_alive():
                logger.warning(
                    "Reader thread for camera %s did not stop within the timeout",
                    self.camera_index,
                )

        if self._capture.isOpened():
            self._capture.release()
            logger.info("Camera %s released", self.camera_index)

    def __enter__(self) -> VideoStream:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Runs on normal exit *and* on any exception, which is exactly why
        # this class is a context manager instead of just exposing open()/
        # release() methods that callers have to remember to pair up.
        self.release()

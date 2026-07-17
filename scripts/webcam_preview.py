#!/usr/bin/env python3
"""Standalone live webcam preview - intentionally NO mask detection.

Purpose: prove out camera access, live display, error handling, and stable
FPS pacing in isolation, before any YOLOv8 inference is wired in. This
mirrors the project roadmap's Phase 1 verification step ("verified via a
plain OpenCV window, no Streamlit yet") and reuses the exact same
`VideoStream` class the full detection app (`app.py`) uses - proof that the
webcam module is decoupled and works standalone.

Usage:
    python scripts/webcam_preview.py
    python scripts/webcam_preview.py --camera 1 --fps 24

Controls:
    q or ESC   -> quit
    Ctrl+C     -> quit (also handled gracefully)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Let this script be run directly (`python scripts/webcam_preview.py`)
# without requiring the package to be installed first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from mask_detector.utils.exceptions import CameraNotAvailableError  # noqa: E402
from mask_detector.utils.fps import FPSCounter, FrameRateLimiter  # noqa: E402
from mask_detector.utils.logger import get_logger  # noqa: E402
from mask_detector.video_stream import VideoStream  # noqa: E402

logger = get_logger(__name__)

WINDOW_TITLE = "Webcam Preview (no mask detection) - press 'q' to quit"

# A single dropped frame is normal/transient (e.g. a brief USB hiccup).
# This many *in a row* means the camera has actually gone away, so we stop
# instead of spinning forever on a dead device.
MAX_CONSECUTIVE_FAILURES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open (default: 0)")
    parser.add_argument("--fps", type=float, default=30.0, help="Target display FPS (default: 30)")
    return parser.parse_args()


def draw_fps_overlay(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw the current measured FPS in the top-left corner, in place."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return frame


def run_preview(camera_index: int, target_fps: float) -> int:
    """Open the camera and show a live preview until the user quits.

    Returns a process exit code (0 = success, 1 = camera error).
    """
    fps_counter = FPSCounter()
    rate_limiter = FrameRateLimiter(target_fps=target_fps)
    consecutive_failures = 0

    try:
        # VideoStream is a context manager specifically so the camera gets
        # released on *any* exit path below - normal quit, an unexpected
        # exception, or Ctrl+C - without needing a manual try/finally here.
        with VideoStream(camera_index=camera_index, target_fps=target_fps) as stream:
            print(f"Camera {camera_index} opened. Showing live preview - press 'q' to quit.")

            while True:
                frame = stream.read_frame()

                if frame is None:
                    consecutive_failures += 1
                    logger.warning(
                        "No frame received (%d/%d consecutive failures)",
                        consecutive_failures,
                        MAX_CONSECUTIVE_FAILURES,
                    )
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(
                            "Too many consecutive frame read failures - is the camera "
                            "still connected? Stopping.",
                            file=sys.stderr,
                        )
                        return 1
                    continue

                # Reset the failure count as soon as a frame succeeds again.
                consecutive_failures = 0

                fps = fps_counter.tick()
                draw_fps_overlay(frame, fps)
                cv2.imshow(WINDOW_TITLE, frame)

                # cv2.waitKey also drives OpenCV's GUI event loop - the
                # window won't actually repaint without it, even though we
                # only care about its return value for the quit keys here.
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or ESC
                    print("Quit key pressed. Shutting down.")
                    return 0

                # Sleep out any leftover frame budget so the loop runs at a
                # steady pace instead of as fast as the CPU/camera allow.
                rate_limiter.wait()

    except CameraNotAvailableError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    args = parse_args()
    try:
        return run_preview(camera_index=args.camera, target_fps=args.fps)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down.")
        return 0
    finally:
        # Always close any OpenCV windows, even if run_preview() raised.
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

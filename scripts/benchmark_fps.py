#!/usr/bin/env python3
"""Benchmark raw YOLOv8 inference FPS on the configured webcam, with no UI.

Useful for judging whether a given machine/model size can keep up with a
live video feed before wiring things into Streamlit.

Usage:
    python scripts/benchmark_fps.py [--frames 100]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mask_detector.detector import MaskDetector  # noqa: E402
from mask_detector.utils.exceptions import CameraNotAvailableError, ModelLoadError  # noqa: E402
from mask_detector.video_stream import VideoStream  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to benchmark")
    args = parser.parse_args()

    try:
        detector = MaskDetector()
    except ModelLoadError as exc:
        print(f"Could not load model: {exc}", file=sys.stderr)
        return 1

    try:
        with VideoStream() as stream:
            print(f"Benchmarking {args.frames} frames...")
            start = time.monotonic()
            processed = 0
            for _ in range(args.frames):
                frame = stream.read_frame()
                if frame is None:
                    continue
                detector.predict(frame)
                processed += 1
            elapsed = time.monotonic() - start
    except CameraNotAvailableError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1

    if processed == 0 or elapsed <= 0:
        print("No frames were successfully processed.")
        return 1

    print(f"Processed {processed} frames in {elapsed:.2f}s -> {processed / elapsed:.2f} FPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

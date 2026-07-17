#!/usr/bin/env python3
"""CLI wrapper to pre-fetch pretrained YOLOv8 mask-detection weights.

`MaskDetector()` already downloads weights automatically on first use if
they're missing (see `mask_detector.model_downloader`), so running this
script is optional - it's useful if you want to pre-download weights
separately from first launching the app (e.g. in a Docker build step, or
just to confirm your setup works before starting Streamlit).

Usage:
    python models/download_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Let this script run directly without having installed the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mask_detector import config  # noqa: E402
from mask_detector.model_downloader import ensure_weights  # noqa: E402
from mask_detector.utils.exceptions import ModelLoadError  # noqa: E402


def main() -> int:
    try:
        ensure_weights(config.MODEL_PATH)
    except ModelLoadError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Model weights ready at {config.MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

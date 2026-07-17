"""Drawing/overlay helpers for rendering detection results onto frames.

FPS measurement/pacing lives in ``utils/fps.py`` instead, so that code which
only needs webcam timing (no detection) doesn't have to import anything
detection-related.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from mask_detector.detector import MASK, MASK_INCORRECT, NO_MASK, Detection

_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    MASK: (0, 200, 0),  # green
    NO_MASK: (0, 0, 220),  # red
    MASK_INCORRECT: (0, 165, 255),  # orange
    "unknown": (160, 160, 160),  # gray
}

_LABEL_TEXT: dict[str, str] = {
    MASK: "Mask",
    NO_MASK: "No Mask",
    MASK_INCORRECT: "Mask Incorrect",
    "unknown": "Unknown",
}


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes + labels on a copy of ``frame`` and return it."""
    annotated = frame.copy()
    for det in detections:
        color = _COLORS_BGR.get(det.label, _COLORS_BGR["unknown"])
        text = f"{_LABEL_TEXT.get(det.label, det.label)} {det.confidence:.0%}"

        cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            annotated,
            (det.x1, max(0, det.y1 - text_h - 10)),
            (det.x1 + text_w + 6, det.y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (det.x1 + 3, max(15, det.y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    return annotated


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR ``ndarray`` frame to a PIL RGB image for Streamlit."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

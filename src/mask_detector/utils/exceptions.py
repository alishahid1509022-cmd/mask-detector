"""Custom exception hierarchy for the mask-detector application.

Using specific exception types (instead of bare ``Exception``) lets callers
(the Streamlit UI in particular) distinguish between failure modes and show
the user an appropriate, actionable message.
"""

from __future__ import annotations


class MaskDetectorError(Exception):
    """Base class for all application-specific errors."""


class CameraNotAvailableError(MaskDetectorError):
    """Raised when the requested webcam cannot be opened or read from."""


class ModelLoadError(MaskDetectorError):
    """Raised when the YOLOv8 model weights cannot be found or loaded."""


class VoiceAlertError(MaskDetectorError):
    """Raised when a text-to-speech backend fails to initialize or speak."""


class FrameProcessingError(MaskDetectorError):
    """Raised when a single frame cannot be processed (should be recoverable)."""

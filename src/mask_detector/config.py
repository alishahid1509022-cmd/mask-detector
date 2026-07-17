"""Centralized, typed configuration.

Every tunable value (camera index, thresholds, paths, cooldowns) lives here
and is sourced from environment variables (with sane defaults), so nothing
is hard-coded/scattered across the codebase as a "magic number".
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
TARGET_FPS: float = float(os.getenv("TARGET_FPS", "30"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "yolov8_mask.pt")))

# Automatic model download settings (see model_downloader.py). MODEL_URL, if
# set, always wins - it's a direct link to a .pt file. Otherwise a default
# public Hugging Face Hub repo/filename is used.
MODEL_URL: str | None = os.getenv("MODEL_URL") or None
MODEL_HF_REPO: str = os.getenv("MODEL_HF_REPO", "krishnamishra8848/Face_Mask_Detection")
MODEL_HF_FILENAME: str = os.getenv("MODEL_HF_FILENAME", "best.pt")

# "auto" resolves to the best available accelerator at load time (CUDA GPU,
# then Apple Silicon MPS, then CPU) for real-time inference speed. Override
# with "cpu", "cuda", or "mps" to force a specific device.
DEVICE: str = os.getenv("DEVICE", "auto")

# Inference resolution YOLOv8 resizes each frame to internally. Lower =
# faster but less accurate on small/far-away faces; 640 is the standard
# YOLOv8 default and a good speed/accuracy balance for real-time webcam use.
INFERENCE_IMG_SIZE: int = int(os.getenv("INFERENCE_IMG_SIZE", "640"))

ALERT_COOLDOWN_SECONDS: float = float(os.getenv("ALERT_COOLDOWN_SECONDS", "5"))
ALERT_MESSAGE: str = os.getenv("ALERT_MESSAGE", "Please wear your face mask.")
TTS_ENGINE: str = os.getenv("TTS_ENGINE", "pyttsx3").lower()
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR: Path = PROJECT_ROOT / "logs"

# Where automatic no-mask screenshots are saved, and how long to wait between
# saves during a sustained no-mask streak (mirrors ALERT_COOLDOWN_SECONDS'
# purpose but is independently tunable - you may want fewer voice interruptions
# but a denser evidence trail of screenshots, or vice versa).
SCREENSHOTS_DIR: Path = Path(os.getenv("SCREENSHOTS_DIR", str(PROJECT_ROOT / "Screenshots")))
SCREENSHOT_COOLDOWN_SECONDS: float = float(os.getenv("SCREENSHOT_COOLDOWN_SECONDS", "5"))

# Normalizes the many label spellings used across public mask-detection
# datasets/weights (e.g. the popular Kaggle "Face Mask Detection" dataset
# uses these exact three class names) into a small, stable internal enum
# consumed by the rest of the app. Extend this if you fine-tune/swap weights
# that use different class names.
CLASS_NAME_MAP: dict[str, str] = {
    "with_mask": "mask",
    "mask": "mask",
    "without_mask": "no_mask",
    "no_mask": "no_mask",
    "no-mask": "no_mask",
    "mask_weared_incorrect": "mask_incorrect",
    "mask_incorrect": "mask_incorrect",
    "incorrect_mask": "mask_incorrect",
}

MAX_CONSECUTIVE_FRAME_ERRORS: int = int(os.getenv("MAX_CONSECUTIVE_FRAME_ERRORS", "10"))

# How long the background camera-reader thread will tolerate consecutive
# failed reads (e.g. the webcam was unplugged, or the OS revoked camera
# permission mid-session) before VideoStream treats the camera as
# disconnected and raises CameraNotAvailableError, instead of silently
# freezing on the last successfully-read frame forever.
CAMERA_DISCONNECT_TIMEOUT_SECONDS: float = float(
    os.getenv("CAMERA_DISCONNECT_TIMEOUT_SECONDS", "5")
)

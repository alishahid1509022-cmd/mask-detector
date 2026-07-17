"""YOLOv8-based mask detector.

Wraps an ``ultralytics.YOLO`` model behind a small, stable interface
(:class:`MaskDetector`) so the rest of the app never touches the
Ultralytics API directly - swapping model weights, or even the underlying
detection library, only requires changes in this one file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mask_detector import config
from mask_detector.model_downloader import ensure_weights
from mask_detector.utils.exceptions import FrameProcessingError, ModelLoadError
from mask_detector.utils.logger import get_logger

logger = get_logger(__name__, log_level=config.LOG_LEVEL, log_dir=config.LOG_DIR)

MASK = "mask"
NO_MASK = "no_mask"
MASK_INCORRECT = "mask_incorrect"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Detection:
    """A single normalized face detection."""

    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float

    @property
    def is_no_mask(self) -> bool:
        return self.label in (NO_MASK, MASK_INCORRECT)


def _resolve_device(requested: str) -> str:
    """Pick the fastest available accelerator for real-time inference.

    "auto" prefers a CUDA GPU, then Apple Silicon's MPS backend, falling
    back to CPU. Explicit values ("cpu"/"cuda"/"mps") are passed through
    unchanged so users can override the choice if needed.
    """
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MaskDetector:
    """Loads a pretrained YOLOv8 mask-detection model and runs inference.

    Model weights are downloaded automatically on first use if missing
    (see :mod:`mask_detector.model_downloader`) - no manual setup step is
    required for the default configuration to work.
    """

    def __init__(
        self,
        model_path: Path | str = config.MODEL_PATH,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        device: str = config.DEVICE,
        inference_size: int = config.INFERENCE_IMG_SIZE,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.inference_size = inference_size

        # Resolved once at load time (not per-frame) since device detection
        # has a small but non-zero cost and never changes mid-session.
        self.device = _resolve_device(device)
        # Half-precision (FP16) roughly doubles throughput on a CUDA GPU.
        # It's intentionally NOT used on CPU (no benefit, sometimes slower)
        # or MPS (mixed/unreliable FP16 support across PyTorch versions).
        self.half = self.device == "cuda"

        self._model = self._load_model()
        self._warmup()
        logger.info(
            "MaskDetector ready (device=%s, half_precision=%s, inference_size=%d)",
            self.device,
            self.half,
            self.inference_size,
        )

    def _load_model(self):  # noqa: ANN202 - ultralytics has no stable type stubs
        # Fetches the weights automatically if they aren't on disk yet;
        # raises ModelLoadError with actionable guidance if that's not
        # possible (see model_downloader.ensure_weights).
        ensure_weights(self.model_path)

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ModelLoadError(
                "The 'ultralytics' package is not installed. Run "
                "`pip install -r requirements.txt`."
            ) from exc

        try:
            return YOLO(str(self.model_path))
        except Exception as exc:  # noqa: BLE001 - ultralytics raises plain Exception
            raise ModelLoadError(
                f"Failed to load YOLOv8 weights from '{self.model_path}': {exc}"
            ) from exc

    def _warmup(self) -> None:
        """Run one throwaway inference so the *first real frame* is fast.

        A model's very first forward pass is slower than every subsequent
        one - CUDA kernel selection/compilation, memory allocation, and
        (on Apple Silicon) MPS graph setup all happen lazily on first use.
        Without this, that one-time cost would land on the first live
        webcam frame, showing up as a visible stutter and an artificially
        low initial FPS reading. Paying it here instead means it happens
        once, during the already-expected "Loading YOLOv8 model..." spinner
        at startup, rather than during live detection.
        """
        dummy_frame = np.zeros((self.inference_size, self.inference_size, 3), dtype=np.uint8)
        try:
            self._model(
                dummy_frame,
                verbose=False,
                device=self.device,
                half=self.half,
                imgsz=self.inference_size,
            )
        except Exception:  # noqa: BLE001 - warmup is best-effort, never fatal
            logger.warning("Model warmup inference failed; continuing anyway", exc_info=True)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single BGR frame and return normalized detections.

        Detections below ``confidence_threshold`` are filtered out here so
        callers never have to worry about low-confidence noise. Bounding
        box coordinates are always returned in the *original* frame's
        pixel space - Ultralytics rescales internally-resized (``imgsz``)
        results back automatically, so callers never need to do their own
        coordinate scaling regardless of ``inference_size``.
        """
        try:
            results = self._model(
                frame,
                verbose=False,
                device=self.device,
                half=self.half,
                imgsz=self.inference_size,
            )
        except Exception as exc:  # noqa: BLE001 - ultralytics raises plain Exception
            raise FrameProcessingError(f"YOLOv8 inference failed on this frame: {exc}") from exc

        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names  # dict[int, str] class-id -> raw label

        try:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue

                class_id = int(box.cls[0])
                raw_label = names.get(class_id, UNKNOWN)
                label = config.CLASS_NAME_MAP.get(raw_label, UNKNOWN)
                if label == UNKNOWN:
                    logger.warning(
                        "Unrecognized model class label '%s' - check CLASS_NAME_MAP", raw_label
                    )

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(Detection(x1, y1, x2, y2, label, confidence))
        except (IndexError, ValueError, AttributeError) as exc:
            raise FrameProcessingError(f"Malformed detection result from model: {exc}") from exc

        return detections

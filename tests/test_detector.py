"""Unit tests for mask_detector.detector.

These tests avoid touching real camera hardware or a real YOLOv8 model file:
MaskDetector instances are built with ``__new__`` and a stubbed ``_model``
callable so that ``predict()``'s parsing/filtering/label-mapping logic can be
exercised in isolation.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from mask_detector.detector import (
    MASK,
    MASK_INCORRECT,
    NO_MASK,
    Detection,
    MaskDetector,
    _resolve_device,
)
from mask_detector.utils.exceptions import FrameProcessingError, ModelLoadError


def test_detection_is_no_mask_property() -> None:
    assert Detection(0, 0, 10, 10, MASK, 0.9).is_no_mask is False
    assert Detection(0, 0, 10, 10, NO_MASK, 0.9).is_no_mask is True
    assert Detection(0, 0, 10, 10, MASK_INCORRECT, 0.9).is_no_mask is True


def test_explicit_device_choice_passes_through_unchanged() -> None:
    # Only "auto" triggers hardware detection (which needs torch); explicit
    # choices should be a simple, torch-free pass-through.
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda") == "cuda"
    assert _resolve_device("mps") == "mps"


def test_missing_model_file_raises_model_load_error(tmp_path, monkeypatch) -> None:
    """When weights are missing AND automatic download fails, raise ModelLoadError.

    Patches ``ensure_weights`` instead of hitting the network, so this test
    stays fast/offline/deterministic regardless of real connectivity.
    """
    missing_path = tmp_path / "does_not_exist.pt"

    def _fail_to_download(model_path: object) -> None:
        raise ModelLoadError("simulated: no weights available locally or remotely")

    monkeypatch.setattr("mask_detector.detector.ensure_weights", _fail_to_download)

    with pytest.raises(ModelLoadError):
        MaskDetector(model_path=missing_path)


class _FakeBox:
    """Mimics the subset of ultralytics' Boxes API that predict() uses."""

    def __init__(self, xyxy: list[float], conf: float, cls: int) -> None:
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.conf = [conf]
        self.cls = [cls]


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox], names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


def _make_detector(confidence_threshold: float = 0.5) -> MaskDetector:
    detector = MaskDetector.__new__(MaskDetector)
    detector.model_path = "unused-in-this-test"  # type: ignore[assignment]
    detector.confidence_threshold = confidence_threshold
    detector.device = "cpu"
    detector.half = False
    detector.inference_size = 640
    return detector


def test_predict_filters_by_confidence_and_maps_labels() -> None:
    detector = _make_detector(confidence_threshold=0.6)
    fake_result = _FakeResult(
        boxes=[
            _FakeBox([0, 0, 10, 10], conf=0.9, cls=0),  # with_mask -> mask
            _FakeBox([5, 5, 15, 15], conf=0.7, cls=1),  # without_mask -> no_mask
            _FakeBox([1, 1, 2, 2], conf=0.3, cls=1),  # below threshold -> filtered out
        ],
        names={0: "with_mask", 1: "without_mask"},
    )
    detector._model = lambda frame, **kwargs: [fake_result]  # type: ignore[attr-defined]

    detections = detector.predict(np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(detections) == 2
    assert detections[0].label == MASK
    assert detections[1].label == NO_MASK
    assert detections[1].is_no_mask is True


def test_predict_maps_unknown_labels_to_unknown() -> None:
    detector = _make_detector(confidence_threshold=0.1)
    fake_result = _FakeResult(
        boxes=[_FakeBox([0, 0, 5, 5], conf=0.8, cls=0)],
        names={0: "some_unmapped_class"},
    )
    detector._model = lambda frame, **kwargs: [fake_result]  # type: ignore[attr-defined]

    detections = detector.predict(np.zeros((5, 5, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].label == "unknown"
    assert detections[0].is_no_mask is False


def test_predict_wraps_model_errors_in_frame_processing_error() -> None:
    detector = _make_detector()

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("inference crashed")

    detector._model = _raise  # type: ignore[attr-defined]

    with pytest.raises(FrameProcessingError):
        detector.predict(np.zeros((10, 10, 3), dtype=np.uint8))


class _FakeYOLO:
    """Stands in for ultralytics.YOLO: records every call it receives."""

    def __init__(self, path: str, *, fail: bool = False) -> None:
        self.path = path
        self.fail = fail
        self.call_count = 0

    def __call__(self, frame: np.ndarray, **kwargs: object) -> list:
        self.call_count += 1
        if self.fail:
            raise RuntimeError("simulated inference failure")
        return []


def _inject_fake_ultralytics(monkeypatch: pytest.MonkeyPatch, *, fail: bool = False) -> None:
    """Makes detector.py's local `from ultralytics import YOLO` resolve to _FakeYOLO.

    `_load_model` imports ultralytics lazily *inside* the method rather than
    at module scope, so there's no `mask_detector.detector.YOLO` attribute to
    monkeypatch directly - injecting a fake module into sys.modules is the
    standard way to intercept that kind of local import in tests.
    """
    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = lambda path: _FakeYOLO(path, fail=fail)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)


def test_warmup_runs_one_dummy_inference_at_construction(tmp_path, monkeypatch) -> None:
    # A real file just needs to exist for ensure_weights() to no-op - its
    # contents are never read since _FakeYOLO doesn't parse anything.
    model_path = tmp_path / "fake.pt"
    model_path.write_bytes(b"not a real model")
    _inject_fake_ultralytics(monkeypatch)

    detector = MaskDetector(model_path=model_path, device="cpu", inference_size=32)

    assert detector._model.call_count == 1


def test_warmup_failure_does_not_prevent_construction(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "fake.pt"
    model_path.write_bytes(b"not a real model")
    _inject_fake_ultralytics(monkeypatch, fail=True)

    # Should not raise even though every call to the (fake) model fails -
    # warmup is best-effort and must never block the app from starting.
    detector = MaskDetector(model_path=model_path, device="cpu", inference_size=32)

    assert detector._model.call_count == 1

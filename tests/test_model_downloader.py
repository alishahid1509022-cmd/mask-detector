"""Unit tests for mask_detector.model_downloader.

Network calls (urllib, huggingface_hub) and filesystem operations that could
fail for OS-specific reasons (mkdir, copyfile) are always monkeypatched, so
these tests are fast, deterministic, and never touch the network or depend
on real filesystem permissions - they only verify the permission/OS-error
handling and fallback-ordering logic added for robustness.
"""

from __future__ import annotations

import sys
import types

import pytest

from mask_detector import model_downloader
from mask_detector.model_downloader import ensure_weights
from mask_detector.utils.exceptions import ModelLoadError


def test_noop_when_weights_already_exist(tmp_path) -> None:
    model_path = tmp_path / "weights.pt"
    model_path.write_bytes(b"already here")

    ensure_weights(model_path)  # must not raise, download, or touch the file

    assert model_path.read_bytes() == b"already here"


def test_permission_denied_creating_model_directory_raises_model_load_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "models" / "weights.pt"

    def _raise_permission_error(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(model_downloader.Path, "mkdir", _raise_permission_error)

    with pytest.raises(ModelLoadError, match="Permission denied"):
        ensure_weights(model_path)


def test_download_from_url_permission_error_is_wrapped(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "weights.pt"
    monkeypatch.setattr(model_downloader.config, "MODEL_URL", "https://example.com/weights.pt")

    def _raise_permission_error(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(model_downloader.urllib.request, "urlretrieve", _raise_permission_error)

    with pytest.raises(ModelLoadError, match="Permission denied"):
        ensure_weights(model_path)


def test_download_from_url_rejects_non_http_scheme(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "weights.pt"
    monkeypatch.setattr(model_downloader.config, "MODEL_URL", "file:///etc/passwd")

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("urlretrieve must not be called for a rejected scheme")

    monkeypatch.setattr(model_downloader.urllib.request, "urlretrieve", _fail_if_called)

    with pytest.raises(ModelLoadError, match="http\\(s\\)"):
        ensure_weights(model_path)


def test_download_from_url_generic_failure_is_wrapped(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "weights.pt"
    monkeypatch.setattr(model_downloader.config, "MODEL_URL", "https://example.com/weights.pt")

    def _raise_connection_error(*_args: object, **_kwargs: object) -> None:
        raise OSError("network unreachable")

    monkeypatch.setattr(model_downloader.urllib.request, "urlretrieve", _raise_connection_error)

    with pytest.raises(ModelLoadError, match="Failed to download"):
        ensure_weights(model_path)


def _inject_fake_huggingface_hub(monkeypatch: pytest.MonkeyPatch, downloaded_path) -> None:
    """Makes model_downloader.py's local `from huggingface_hub import ...` resolve to a fake.

    `_try_download_from_huggingface` imports huggingface_hub lazily *inside*
    the function, so injecting a fake module into sys.modules (rather than
    monkeypatching a module attribute) is the standard way to intercept it.
    """
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda repo_id, filename: str(downloaded_path)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)


def test_huggingface_copy_permission_error_falls_back_to_final_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "weights.pt"
    monkeypatch.setattr(model_downloader.config, "MODEL_URL", None)

    downloaded = tmp_path / "downloaded.pt"
    downloaded.write_bytes(b"weights")
    _inject_fake_huggingface_hub(monkeypatch, downloaded)

    def _raise_permission_error(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(model_downloader.shutil, "copyfile", _raise_permission_error)

    # The permission error during copy is caught and treated as "this
    # option failed", not raised directly - ensure_weights() then falls
    # through to its own final, actionable ModelLoadError.
    with pytest.raises(ModelLoadError, match="automatic download failed"):
        ensure_weights(model_path)


def test_huggingface_download_success_copies_weights(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "weights.pt"
    monkeypatch.setattr(model_downloader.config, "MODEL_URL", None)

    downloaded = tmp_path / "downloaded.pt"
    downloaded.write_bytes(b"weights-content")
    _inject_fake_huggingface_hub(monkeypatch, downloaded)

    ensure_weights(model_path)

    assert model_path.read_bytes() == b"weights-content"


def test_every_option_failing_raises_actionable_model_load_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "weights.pt"
    monkeypatch.setattr(model_downloader.config, "MODEL_URL", None)

    fake_hub = types.ModuleType("huggingface_hub")

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("no internet")

    fake_hub.hf_hub_download = _raise  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(ModelLoadError, match="MODEL_URL"):
        ensure_weights(model_path)

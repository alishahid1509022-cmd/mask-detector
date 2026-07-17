"""Automatic download of pretrained YOLOv8 mask-detection weights.

Resolution order, first one that works wins:

1. Weights already present at the target path -> nothing to do.
2. ``MODEL_URL`` (env var) set -> direct HTTP(S) download of a .pt file.
3. Default: a public Hugging Face Hub repo containing YOLOv8 weights
   fine-tuned specifically for mask detection (``config.MODEL_HF_REPO`` /
   ``MODEL_HF_FILENAME``) - this is what makes ``MaskDetector()`` work with
   zero manual setup.

If every option fails (no internet, dependency missing, repo unavailable),
a :class:`ModelLoadError` is raised with concrete next steps rather than a
raw stack trace, so the Streamlit UI can show the user something actionable.

Note: the generic ``yolov8n.pt`` (COCO-pretrained) weights shipped by
Ultralytics do NOT detect masks - they only know COCO's 80 everyday object
classes. This module always fetches weights trained on with_mask /
without_mask / mask_weared_incorrect classes instead.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from mask_detector import config
from mask_detector.utils.exceptions import ModelLoadError
from mask_detector.utils.logger import get_logger

logger = get_logger(__name__, log_level=config.LOG_LEVEL, log_dir=config.LOG_DIR)

_ALLOWED_URL_SCHEMES = ("http", "https")


def ensure_weights(model_path: Path) -> None:
    """Guarantee YOLOv8 mask-detection weights exist at ``model_path``.

    Downloads them automatically if missing. Raises ``ModelLoadError`` if
    they're missing and every automatic download option fails.
    """
    if model_path.exists():
        logger.debug("Model weights already present at %s", model_path)
        return

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise ModelLoadError(
            f"Permission denied creating the model directory '{model_path.parent}'. "
            "Fix its permissions, or set MODEL_PATH in .env to a writable location."
        ) from exc
    except OSError as exc:
        raise ModelLoadError(
            f"Could not create the model directory '{model_path.parent}': {exc}"
        ) from exc

    if config.MODEL_URL:
        _download_from_url(config.MODEL_URL, model_path)
        return

    if _try_download_from_huggingface(config.MODEL_HF_REPO, config.MODEL_HF_FILENAME, model_path):
        return

    raise ModelLoadError(
        f"No model weights found at '{model_path}' and automatic download failed "
        f"(tried Hugging Face repo '{config.MODEL_HF_REPO}'). Fixes:\n"
        "  1. Check your internet connection and try again, or\n"
        "  2. Set MODEL_URL in .env to a direct .pt download link "
        "(e.g. a Roboflow Universe export), or\n"
        "  3. Manually place a pretrained YOLOv8 mask-detection .pt file at that path.\n"
        "See README.md for details."
    )


def _download_from_url(url: str, destination: Path) -> None:
    """Download a direct .pt link (user-configured override).

    Rejects anything other than http(s) before ever calling
    ``urlretrieve`` - ``MODEL_URL`` is operator-configured (via ``.env``),
    not attacker-controlled input, but ``urlretrieve`` also happily
    follows ``file://`` and ``ftp://`` URLs, which would otherwise let a
    typo'd or copy-pasted ``MODEL_URL`` silently read a local file instead
    of downloading anything. Failing loudly with a clear message is
    strictly better than that surprising, hard-to-debug outcome.
    """
    scheme = urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        raise ModelLoadError(
            f"MODEL_URL must be an http(s) link, got '{url}' (scheme: '{scheme or 'none'}'). "
            "Set it to a direct .pt download link, e.g. a Roboflow Universe export."
        )

    logger.info("Downloading model weights from %s", url)
    try:
        urllib.request.urlretrieve(url, destination)  # noqa: S310 - scheme validated above
    except PermissionError as exc:
        raise ModelLoadError(
            f"Permission denied writing model weights to '{destination}'. "
            "Check that the models/ directory is writable, or set MODEL_PATH in .env "
            "to a location you have write access to."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - urllib raises several distinct error types
        raise ModelLoadError(f"Failed to download model weights from {url}: {exc}") from exc
    logger.info("Saved model weights to %s", destination)


def _try_download_from_huggingface(repo_id: str, filename: str, destination: Path) -> bool:
    """Best-effort automatic download from the default Hugging Face repo.

    Returns True on success, False on any failure (missing dependency, no
    internet, repo/file not found) so the caller can fall back to raising a
    clear error instead of propagating a confusing low-level exception.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning(
            "huggingface_hub is not installed; skipping automatic download. "
            "Install it (`pip install huggingface_hub`, already in requirements.txt) "
            "or set MODEL_URL instead."
        )
        return False

    try:
        logger.info(
            "Auto-downloading model weights from Hugging Face Hub: %s/%s", repo_id, filename
        )
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as exc:  # noqa: BLE001 - huggingface_hub raises many distinct error types
        logger.warning("Automatic Hugging Face download failed: %s", exc)
        return False

    try:
        shutil.copyfile(downloaded_path, destination)
    except PermissionError:
        logger.warning(
            "Permission denied copying downloaded weights to %s; falling back to the next option",
            destination,
        )
        return False
    except OSError as exc:
        logger.warning("Could not copy downloaded weights to %s: %s", destination, exc)
        return False

    logger.info("Saved model weights to %s", destination)
    return True

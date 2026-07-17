# syntax=docker/dockerfile:1
#
# Container image for the AI Face Mask Detection System.
#
# IMPORTANT: this app opens a webcam on the machine the Streamlit process
# runs on (cv2.VideoCapture) - see docs/deployment.md for exactly what that
# means (and doesn't mean) once this image is running on a cloud host
# without physical camera hardware attached.
#
# Build:  docker build -t mask-detector .
# Run:    docker run --rm -p 8501:8501 --env-file .env mask-detector

FROM python:3.11-slim

# System libraries required at *import time* by opencv-python even though
# this container never opens a GUI window (cv2's compiled extension still
# links against libGL/libSM/libXext), plus espeak for pyttsx3 (the default,
# offline voice-alert backend) and curl for the HEALTHCHECK below. Combined
# into one apt-get layer and cleaned up in the same layer to keep the image
# small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        espeak \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies are installed from requirements.txt alone (not
# requirements-dev.txt) - this is a runtime image, it doesn't need pytest/
# black/mypy/etc. Copied and installed *before* the rest of the source so
# `docker build` can reuse this layer's cache on every rebuild that only
# changes application code, not dependencies - keeps iteration fast.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

# Best-effort pre-download of pretrained mask-detection weights at build
# time, so the image is ready to detect immediately on first container
# start instead of paying that download cost live in front of a viewer.
# Allowed to fail here (e.g. no network available during `docker build`) -
# MaskDetector() already retries the same download automatically at
# runtime via model_downloader.ensure_weights(), so a build-time failure
# just means the first container start does the download instead.
RUN python models/download_model.py || true

# Runs as a non-root user - standard container hardening, and free here
# since the app never needs elevated privileges. Screenshots/ and logs/
# are created at runtime under /app, owned by this user.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Streamlit exposes a built-in health endpoint; used here and can be
# pointed to by the same path on Render/any other platform's own health
# check setting (see docs/deployment.md).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Shell form (not exec form) so ${PORT:-8501} is actually expanded - lets
# the same image bind to Streamlit's default port 8501 for a plain
# `docker run`, or to whatever port a platform like Render injects via the
# PORT environment variable at container start, with no separate
# entrypoint script needed.
CMD streamlit run src/mask_detector/app.py \
    --server.address=0.0.0.0 \
    --server.port=${PORT:-8501} \
    --server.headless=true

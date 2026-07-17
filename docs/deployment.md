# Deployment Guide

Step-by-step instructions for running this app in Docker, on Streamlit Community Cloud, and on Render - plus the environment-variable/secrets setup for each.

## Table of Contents

- [Read this first: the webcam caveat](#read-this-first-the-webcam-caveat)
- [1. Docker](#1-docker)
- [2. Streamlit Community Cloud](#2-streamlit-community-cloud)
- [3. Render](#3-render)
- [Environment variables & secrets, per platform](#environment-variables--secrets-per-platform)
- [Troubleshooting](#troubleshooting)
- [Redeploying after changes](#redeploying-after-changes)

## Read this first: the webcam caveat

This app captures video with `cv2.VideoCapture(...)`, which opens a **physical camera device attached to the machine the Streamlit process is running on** (see [docs/architecture.md](architecture.md#why-local-only-webcam-access)). That has a direct consequence for every platform below:

| Platform | Has a physical webcam? | What happens when you click "Start Camera" |
|---|---|---|
| Your own laptop/desktop (venv or Docker on Linux with `--device`) | ✅ Yes | Live detection works exactly as in local development |
| Docker Desktop on macOS/Windows | ❌ No route to host camera | `CameraNotAvailableError` - friendly error, no crash |
| Streamlit Community Cloud | ❌ No | `CameraNotAvailableError` - friendly error, no crash |
| Render | ❌ No | `CameraNotAvailableError` - friendly error, no crash |

This isn't a bug and it isn't something a Dockerfile or a platform setting can fix - it's a hardware-access limitation shared by *every* cloud host, because none of them have a camera plugged in. The app's error-handling work (see [README > Error Handling](../README.md#-error-handling)) means this fails **gracefully**: the UI, sidebar, dashboard placeholders, and status chips all render correctly, and clicking Start Camera shows a clear, friendly message instead of a crash.

Given that, the deployments below are genuinely useful for:

- Showing off the UI, architecture, and code to reviewers/recruiters without asking them to install anything
- Verifying the app builds and boots correctly in a clean, reproducible environment
- A base to build on if you implement the browser-webcam roadmap item below

For an actual **live** webcam demo, either run it locally (see the main [README](../README.md#-installation)), or use Docker on a native Linux host with `--device=/dev/video0` passthrough (see [Docker](#1-docker) below).

> **Want cloud hosts to use a *visitor's own* webcam?** That requires streaming video from the browser over WebRTC (e.g. via [`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc)) instead of `cv2.VideoCapture`, which is tracked as a roadmap item in the main README and is a larger change than this deployment guide covers.

## 1. Docker

The most portable option: one image runs identically on your laptop, a Linux server, or as the deploy artifact for Render (see [Render](#3-render)).

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- This repo cloned locally, with a `.env` file present (`cp .env.example .env`)

### Step 1 - Build the image

```bash
docker build -t mask-detector .
```

What the [`Dockerfile`](../Dockerfile) does, in order:
1. Starts from `python:3.11-slim`.
2. Installs the small set of system libraries `opencv-python` needs at import time (`libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender1`) plus `espeak` (offline voice alerts) and `curl` (health check).
3. Installs Python dependencies from `requirements.txt` in their own layer, so later rebuilds that only touch application code reuse this layer from cache instead of reinstalling everything.
4. Copies the app source in and best-effort pre-downloads the mask-detection model weights at build time via `models/download_model.py`, so the container is ready to detect immediately on first start (falls back to downloading at runtime if the build has no network access).
5. Creates and switches to a non-root `appuser`.
6. Declares a `HEALTHCHECK` against Streamlit's built-in `/_stcore/health` endpoint.
7. Starts Streamlit bound to `0.0.0.0` on `$PORT` (defaulting to `8501`) so the same image works with a fixed port locally *and* a platform-injected dynamic port on Render.

### Step 2 - Run the container

Basic run, default port 8501:

```bash
docker run --rm -p 8501:8501 --env-file .env mask-detector
```

Open `http://localhost:8501`.

With persistent screenshots/logs on the host, and (Linux only) webcam passthrough:

```bash
docker run --rm -p 8501:8501 --env-file .env \
  -v "$(pwd)/Screenshots:/app/Screenshots" \
  -v "$(pwd)/logs:/app/logs" \
  --device=/dev/video0:/dev/video0 \
  mask-detector
```

- `--device=/dev/video0:/dev/video0` passes the host's first camera into the container. **This only has any effect on Linux hosts** where the Docker daemon runs natively - Docker Desktop on macOS/Windows runs containers inside an internal VM with no path to your USB/camera hardware, so omit this flag there (the app still runs fine, it just can't see a camera - see the [caveat above](#read-this-first-the-webcam-caveat)).
- `--env-file .env` passes every variable from your local `.env` into the container; see [.env.example](../.env.example) for the full list.

### Step 3 - Or use docker-compose

[`docker-compose.yml`](../docker-compose.yml) wraps the same build/run/volumes/device steps above into one command:

```bash
docker compose up --build
```

```bash
docker compose down
```

Comment out the `devices:` block in `docker-compose.yml` if you're on macOS/Windows (see above).

### Rebuilding after code changes

```bash
docker build -t mask-detector .   # or: docker compose up --build
```

The dependency-install layer is cached and skipped unless `requirements.txt` changed, so rebuilds after an app-code-only change are fast.

## 2. Streamlit Community Cloud

Free hosting straight from a GitHub repo - no Dockerfile needed, Streamlit Cloud builds your `requirements.txt` (and optionally `packages.txt`) itself.

### Step 1 - Push to GitHub

```bash
git push origin main
```

(Streamlit Cloud deploys directly from a GitHub branch, so the repo needs to be on GitHub first.)

### Step 2 - System packages (`packages.txt`)

This repo intentionally does **not** ship a `packages.txt`. Streamlit Cloud's
host image mixes Debian releases, and common OpenCV apt deps
(`libglib2.0-0`, `libsm6`, ...) often fail there with "held broken packages" /
`libffi7` conflicts. We rely on `opencv-python-headless` +
`ultralytics-opencv-headless` in `requirements.txt` instead, which don't need
those system libraries. (Docker still installs them via the `Dockerfile` for
local/Render container builds.)

### Step 3 - Create the app

1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Click **New app**.
3. Pick your repository and branch.
4. Set **Main file path** to:

   ```
   src/mask_detector/app.py
   ```

5. Under **Advanced settings**, pick Python **3.11** (matches this project's supported versions).

### Step 4 - Add secrets (your `.env` values)

Streamlit Cloud doesn't read a `.env` file - it uses its own **Secrets** panel (Settings → Secrets) written in TOML, which the app then sees as environment variables automatically (Streamlit maps `st.secrets` entries into `os.environ`-compatible access, and this project's `config.py` reads via `python-dotenv`/`os.environ`, so plain `KEY = "value"` lines work). Paste something like:

```toml
TARGET_FPS = "15"
CONFIDENCE_THRESHOLD = "0.5"
DEVICE = "cpu"
TTS_ENGINE = "gtts"
LOG_LEVEL = "INFO"
```

Only set the variables you want to override - every one has a sensible default (see [Configuration Reference](../README.md#-configuration-reference)). `DEVICE=cpu` is worth setting explicitly since Streamlit Cloud has no GPU; auto-detection would land on `cpu` anyway, but being explicit skips the detection step.

### Step 5 - Deploy

Click **Deploy**. Streamlit Cloud installs `requirements.txt`, then starts the app. First builds typically take a few minutes (PyTorch is the largest dependency).

### Step 6 - What to expect

The UI, sidebar, and dashboard all render normally. Clicking **▶️ Start Camera** shows a friendly "camera unavailable" message - expected, per the [caveat above](#read-this-first-the-webcam-caveat), since Streamlit Cloud's containers have no webcam attached.

### A note on resource limits

Streamlit Community Cloud's free tier has modest CPU/RAM. If builds are slow or the app gets OOM-killed:
- Lower `INFERENCE_IMG_SIZE` (e.g. `480`) and `TARGET_FPS` via secrets.
- Consider swapping to a smaller YOLOv8 variant weight file.
- Remember `requirements.txt` installs the full `torch` wheel (via `ultralytics`); this is unavoidable for detection to work at all, but it's the main driver of build time/image size on any platform, not just Streamlit Cloud.

## 3. Render

Render can deploy this project two ways: from the included `Dockerfile` (recommended, most reliable), or as a "native" Python web service. Both are covered below.

### Option A - Docker-based Web Service (recommended)

This is the most reliable path because the Dockerfile already installs every system dependency `opencv-python`/`pyttsx3` need - no guessing at Render's native build environment.

**Using the included Blueprint (fastest):**

1. Push this repo (including [`render.yaml`](../render.yaml)) to GitHub.
2. In the Render dashboard, click **New +** → **Blueprint**.
3. Select the repo. Render reads `render.yaml` and proposes the `mask-detector` web service pre-filled (Docker runtime, health check path, starter env vars).
4. Review and click **Apply**.

**Manual click-through (equivalent, without the Blueprint file):**

1. Push to GitHub.
2. Render dashboard → **New +** → **Web Service** → connect the repo.
3. Render auto-detects the `Dockerfile` and offers **Docker** as the environment - select it.
4. Pick an instance type. The free tier is too memory-constrained for `torch`/`ultralytics` to run comfortably - use at least the **Starter** plan.
5. Under **Health Check Path**, set:

   ```
   /_stcore/health
   ```

6. Add environment variables under the **Environment** tab (mirror your `.env` - see the [reference table](#environment-variables--secrets-per-platform) below). Render automatically injects `PORT`; the Dockerfile's `CMD` already reads `${PORT:-8501}`, so you don't need to set it yourself.
7. Click **Create Web Service**. Render builds the Docker image and deploys it, exposing the app at `https://<your-service-name>.onrender.com`.

### Option B - Native Python Web Service (no Dockerfile)

1. Render dashboard → **New +** → **Web Service** → connect the repo, environment **Python 3**.
2. **Build Command:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Start Command:**

   ```bash
   streamlit run src/mask_detector/app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
   ```

4. Add the same environment variables as Option A.

**Caveat:** Render's native Python build environment may be missing `libGL`/`libSM` (the same libraries the Dockerfile installs via `apt-get`), which can surface as `ImportError: libGL.so.1: cannot open shared object file` when the app tries `import cv2`. If you hit that, either switch to **Option A** (Docker handles this for you), or add a build step that installs those system packages before `pip install` (exact mechanism depends on Render's current native-runtime feature set - Docker is the more predictable option and is what this repo's `Dockerfile`/`render.yaml` are set up for).

### What to expect

Same as Streamlit Cloud: the UI loads and works fully, and **Start Camera** shows the friendly "camera unavailable" message, because Render's servers have no physical webcam either (see the [caveat above](#read-this-first-the-webcam-caveat)).

## Environment variables & secrets, per platform

| Platform | Where to set them | Format |
|---|---|---|
| Local / `docker run` | `.env` file, loaded via `--env-file .env` or `python-dotenv` directly | `KEY=value` |
| `docker-compose` | `.env` file, referenced by `env_file:` in `docker-compose.yml` | `KEY=value` |
| Streamlit Community Cloud | App → **Settings → Secrets** | TOML: `KEY = "value"` |
| Render | Service → **Environment** tab, or `envVars:` in `render.yaml` | `KEY=value` pairs in the UI |

The full list of variables, defaults, and what each one does lives in [`.env.example`](../.env.example) and the [Configuration Reference](../README.md#-configuration-reference) table in the main README - it's identical across every platform, only the *mechanism* for setting them differs.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: libGL.so.1: cannot open shared object file` | `opencv-python`'s GUI build needs display libs missing on minimal hosts | Use this repo's `opencv-python-headless` / `ultralytics-opencv-headless` pins (already in `requirements.txt`); Docker installs `libgl1` etc. via the `Dockerfile` |
| "Could not open camera at index 0" / friendly camera-unavailable message on a cloud host | Expected - the host has no physical webcam attached | See the [webcam caveat](#read-this-first-the-webcam-caveat); run locally, or via Docker with `--device` on a native Linux host, for a live demo |
| Voice alerts silent on a cloud host | `pyttsx3` needs `espeak` *and* a working audio device, which cloud hosts don't have; `gTTS` needs local audio playback too | Expected degradation - the app falls back automatically and never crashes; the on-screen alert banner remains fully functional regardless |
| Slow or failing builds / out-of-memory | `torch` (pulled in by `ultralytics`) is a large dependency, and default free/starter tiers have limited CPU/RAM/build time | Lower `INFERENCE_IMG_SIZE`/`TARGET_FPS`, use a smaller model, or upgrade the plan |
| Render health check failing | Health check path doesn't match Streamlit's endpoint | Confirm **Health Check Path** is exactly `/_stcore/health` |
| Port binding errors on Render | App bound to a fixed port instead of Render's injected `$PORT` | Use this repo's `Dockerfile` (`${PORT:-8501}`) or Option B's start command (`--server.port=$PORT`) as-is |

## Redeploying after changes

| Platform | How |
|---|---|
| Docker (local) | `docker build -t mask-detector .` again (cached layers speed this up), then re-run |
| Streamlit Community Cloud | Automatic - redeploys on every push to the connected branch |
| Render | Automatic if auto-deploy is enabled for the service (default), or trigger a manual deploy from the dashboard |

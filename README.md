# 😷 AI Face Mask Detection System

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-00FFFF?logo=data%3Aimage%2Fpng%3Bbase64%2C&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/CV-OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-FCC21B.svg)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **Before pushing to GitHub:** replace `OWNER/REPO` in the CI badge URL above (and anywhere else in this repo) with your actual `github-username/repo-name`, so the badge and links resolve correctly.

Real-time face mask detection from a webcam feed, built with **YOLOv8**, **OpenCV**, and **Streamlit** - complete with voice alerts, automatic screenshot evidence, and a live analytics dashboard. Designed and documented as a portfolio-quality, production-style project rather than a quick notebook demo.

## Table of Contents

- [Screenshots & Demo](#-screenshots--demo)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration Reference](#-configuration-reference)
- [Performance](#-performance)
- [Testing & Code Quality](#-testing--code-quality)
- [Error Handling](#-error-handling)
- [Deployment](#-deployment)
- [Roadmap / Future Improvements](#-roadmap--future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 📸 Screenshots & Demo

![Demo placeholder](docs/demo.gif)

> A live webcam feed can't be embedded in a static README, so add your own media once the app is running locally:
> 1. Record a short screen capture (QuickTime on macOS, or any screen recorder), trim it, convert it to a GIF, and save it as `docs/demo.gif`.
> 2. Take a few PNG screenshots of the running app and drop them into a new `docs/screenshots/` folder, then reference them below, for example:
>
> ```markdown
> | Live detection | Sidebar controls | Dashboard |
> |---|---|---|
> | ![Detection](docs/screenshots/detection.png) | ![Sidebar](docs/screenshots/sidebar.png) | ![Dashboard](docs/screenshots/dashboard.png) |
> ```

## ✨ Features

- Real-time webcam video feed rendered in a Streamlit UI
- Per-frame YOLOv8 inference classifying faces as **Mask** / **No Mask** (and **Mask Worn Incorrectly**, if your weights support it)
- Pretrained model weights **downloaded automatically** on first run - no manual setup step required
- Bounding boxes, labels, and confidence percentage drawn on the live frame
- Auto-selects the fastest available device (CUDA > Apple Silicon MPS > CPU) for real-time speed
- Configurable confidence threshold and camera source from the sidebar
- Voice alert ("Please wear your face mask.") on sustained no-mask detection, with a cooldown timer so it never fires every frame - only once per `ALERT_COOLDOWN_SECONDS`
- On-screen status banner as a fallback when audio is unavailable
- Automatic screenshot saved to `Screenshots/` on a no-mask detection (debounced by `SCREENSHOT_COOLDOWN_SECONDS`, filename stamped with the date/time), with an in-app toast notification confirming each save
- Live dashboard updating in real time every frame: total detections, mask detections, no-mask detections, average confidence, current FPS, and session duration
- Portfolio-ready UI: modern color system, card-based layout, a branded sidebar with grouped settings, live system-health status chips, and a footer
- Robust error handling: missing/unavailable cameras (including mid-session disconnects), missing model weights, permission errors, and unexpected exceptions all degrade to a friendly message instead of a crash

## 🧰 Tech Stack

| Category | Choice | Why |
|---|---|---|
| Language | **Python** 3.10 / 3.11 | Best-supported versions for the CV/ML stack below |
| Detection model | **Ultralytics YOLOv8** | Fast, actively maintained, one-line pretrained-weight loading |
| Computer vision | **OpenCV** (`opencv-python`) | Webcam capture, frame drawing, image I/O |
| UI framework | **Streamlit** | Fast to build a real-time, interactive Python UI with no separate frontend |
| Voice alerts | **pyttsx3** (offline, default) / **gTTS** (online) | Cross-platform TTS with automatic fallback between the two |
| Model distribution | **huggingface_hub** | Automatic first-run download of pretrained mask-detection weights |
| Image/array handling | **Pillow**, **NumPy** | Standard array/image plumbing between OpenCV, YOLOv8, and Streamlit |
| Config | **python-dotenv** | `.env`-driven configuration, no hard-coded values |
| Testing | **pytest**, **pytest-mock** | Unit tests for every module, with hardware (camera/model) mocked out |
| Code quality | **black**, **isort**, **ruff**, **mypy**, **pre-commit** | Consistent formatting, linting, import order, and type checking on every commit |
| CI | **GitHub Actions** | Lint + test on every push/PR across Python 3.10 and 3.11 |
| Containerization | **Docker** | Reproducible image for local/Render deployment - see [Deployment](#-deployment) |

## 🏗️ Project Structure

```
face-mask-detector/
├── .github/workflows/ci.yml   # lint + test on every push/PR
├── Dockerfile, .dockerignore  # container image, see docs/deployment.md
├── docker-compose.yml         # local convenience wrapper around Docker
├── packages.txt, render.yaml  # Streamlit Cloud apt deps / Render blueprint
├── docs/                      # architecture, performance, deployment notes, demo gif, screenshots
├── models/                    # model weights (git-ignored) + download script
├── src/mask_detector/         # application package
│   ├── config.py              # env-driven configuration
│   ├── detector.py            # YOLOv8 wrapper
│   ├── video_stream.py        # OpenCV webcam capture
│   ├── model_downloader.py    # automatic pretrained-weight download
│   ├── alerts/                # voice_alert.py, alert_manager.py
│   ├── screenshot.py          # auto-saves a screenshot on no-mask detections
│   ├── utils/                 # drawing.py, logger.py, exceptions.py, cooldown.py, fps.py
│   └── app.py                 # Streamlit entrypoint
├── Screenshots/                # auto-saved no-mask screenshots (git-ignored)
├── tests/                     # pytest unit tests
└── scripts/                   # run_app.sh, webcam_preview.py, benchmark_fps.py
```

See [docs/architecture.md](docs/architecture.md) for a data-flow diagram and a description of every module's responsibility.

## 🚀 Installation

> **Note on this repo's current state:** the source code, tests, and configuration were scaffolded programmatically. The one-time environment setup steps below (virtual environment, git init, installing dependencies) need to be run **by you, in your own terminal**.

### Prerequisites

- Python **3.10 or 3.11** (recommended - `opencv-python` and `ultralytics`/`torch` can lag behind brand-new Python releases)
- A webcam accessible to your OS (built-in or USB)
- ~2 GB free disk space for dependencies (PyTorch + Ultralytics) and the downloaded model weights
- (Linux only, for offline voice alerts) `espeak`/`espeak-ng` - see [Voice alerts](#voice-alerts-cross-platform)

### 1. Clone the repository and create a virtual environment

```bash
git clone https://github.com/OWNER/REPO.git
cd REPO

python3.11 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements-dev.txt   # includes requirements.txt + dev tools
```

Just want to run the app, not develop it? `pip install -r requirements.txt` alone is enough.

### 3. Configure environment variables

```bash
cp .env.example .env
# edit .env if you need a non-default camera index, thresholds, etc.
```

Every variable is documented inline in `.env.example` and summarized in the [Configuration Reference](#-configuration-reference) below. Sensible defaults mean this step is optional for a first run.

### 4. Model weights (downloaded automatically)

The generic `yolov8n.pt` (COCO) weights do **not** detect masks. `MaskDetector()` handles this for you automatically the first time it runs:

1. If `models/yolov8_mask.pt` already exists, it's used as-is.
2. Else if `MODEL_URL` is set in `.env`, it downloads that direct link.
3. Else it automatically fetches a pretrained mask-detection model from the default Hugging Face Hub repo (`MODEL_HF_REPO`, see `.env.example`).

You don't need to run anything manually - just run the app (see [Usage](#-usage)) and the first launch will download weights once. If you want to pre-fetch them separately (e.g. before a demo, or in a Docker build step):

```bash
python models/download_model.py
```

If every automatic option fails (no internet, the default repo moved, etc.), you'll get a clear error with next steps - the quickest fallback options are:
- Set `MODEL_URL` to a direct `.pt` link (e.g. a [Roboflow Universe](https://universe.roboflow.com) "face mask detection" export), or
- Train your own on the Kaggle ["Face Mask Detection"](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) dataset:
  ```bash
  yolo detect train data=mask_dataset.yaml model=yolov8n.pt epochs=50
  ```
  then place the resulting `.pt` file at `models/yolov8_mask.pt`.

### 5. (Optional) Enable pre-commit hooks and initialize git

```bash
git init   # skip if you already cloned a git repo above
pre-commit install
```

See [Contributing](#-contributing) for the full branching strategy this project follows.

## ▶️ Usage

### Running the app

```bash
streamlit run src/mask_detector/app.py
# or: ./scripts/run_app.sh
```

Streamlit prints a local URL (typically `http://localhost:8501`) - open it in your browser.

### Using the interface

1. Click **▶️ Start Camera** in the main panel. The first click also triggers the one-time "Loading YOLOv8 model..." spinner and, if needed, the automatic weight download.
2. The live feed appears with bounding boxes, labels, and confidence percentages drawn on every detected face.
3. A color-coded status banner above the feed shows the current frame's result: ✅ **SAFE**, 🚨 **ALERT** (no mask), or 👀 **No face detected**.
4. Use the sidebar to tune camera index, confidence threshold, voice engine/cooldown, and screenshot cooldown - changes apply live without restarting the camera.
5. Click **⏹️ Stop Camera** to end the session; the dashboard keeps showing the final session's stats until you start again.

### Voice alerts (cross-platform)

When a face is detected without a mask, the app speaks **"Please wear your face mask."** - debounced by a cooldown timer (`ALERT_COOLDOWN_SECONDS`, default 5s) so it fires once per cooldown window instead of every single frame.

Two interchangeable engines are supported behind one interface:

- **`pyttsx3`** (default) - fully offline, uses your OS's native speech engine:
  - **Windows**: SAPI5, built in, works out of the box.
  - **macOS**: NSSpeechSynthesizer, built in, works out of the box.
  - **Linux**: espeak/espeak-ng - install it first: `sudo apt install espeak` (Debian/Ubuntu) or your distro's equivalent.
- **`gtts`** - Google's online TTS, needs internet access + `playsound` for local playback, but requires no OS-level driver. `playsound` is **not** in the base `requirements.txt` (its last release doesn't install on Python 3.12+); install it separately with `pip install playsound==1.3.0` on Python ≤3.11 if you want `gtts`'s local playback - `pyttsx3` (the default) doesn't need it at all.

If your configured engine (`TTS_ENGINE` in `.env`) fails to initialize - most commonly `pyttsx3` on a Linux box without `espeak` installed - the app **automatically falls back to the other engine** rather than failing. Voice alerts are only fully disabled (falling back to the on-screen alert banner) if both engines fail.

### Automatic screenshots

Every time a no-mask detection occurs, the app saves the annotated frame (boxes + labels included) to the `Screenshots/` folder at the project root, creating it automatically if it doesn't exist yet. Filenames are timestamped, e.g. `no_mask_2026-07-17_18-53-02.jpg`, so files sort chronologically and never collide.

Just like voice alerts, this is debounced by a cooldown (`SCREENSHOT_COOLDOWN_SECONDS`, default 5s, adjustable from the sidebar) so a sustained no-mask streak saves at most one screenshot per cooldown window instead of dozens of near-identical files per second. Each successful save shows a toast notification ("📸 Screenshot saved: ...") in the UI, and the running total is shown in the "Screenshots" stat card. A failed save (e.g. permissions) shows a one-time friendly warning toast instead of failing silently or repeating every frame.

### Live dashboard

The stats column next to the video feed is a real-time dashboard, recomputed on every processed frame:

| Metric | Meaning |
| --- | --- |
| Total Detections | Cumulative count of every face detected this session (any label) |
| Mask Detections | Cumulative count of faces classified **Mask** this session |
| No-Mask Detections | Cumulative count of faces classified **No Mask** this session |
| Average Confidence | Running average confidence across every detection this session (intentionally session-wide, not per-frame, since a single frame's confidence jitters a lot) |
| Current FPS | Instantaneous, rolling-average frames-per-second |
| Session Duration | Time elapsed (`H:MM:SS`) since the camera was started |

A smaller "This Frame" section below it shows the current frame's face count and the running screenshots-saved total, for at-a-glance context alongside the cumulative dashboard above.

### UI design

The interface is built around a small, reusable design system (defined once in `inject_custom_css()` in `app.py`) rather than default Streamlit styling:

- **Colors** - an indigo primary with semantic success/danger/warning tones, kept in sync between the custom CSS and `.streamlit/config.toml` so native widgets (buttons, sliders) match the custom-styled elements.
- **Cards** - the video feed and dashboard each sit in their own bordered `st.container`, and every metric renders as a hoverable card with a soft shadow.
- **Sidebar** - a branded header followed by clearly labeled groups (Camera & Detection, Voice Alerts, Screenshots, About) instead of one long, flat settings list.
- **Status indicators** - a row of chips (🧠 Model, 🎥 Camera, 🔊 Voice, 📸 Screenshots) shows system health at a glance, separate from the big color-coded banner that reflects the current detection result; a pulsing dot marks the LIVE/IDLE badge in the header.
- **Responsive layout** - the two-column layout stacks on narrow viewports automatically, with extra CSS breakpoints to scale down typography and spacing on mobile.
- **Footer** - a quiet credits/tech-stack strip with a link back to the source, appropriate for a portfolio project.

## ⚙️ Configuration Reference

Every setting lives in `.env` (copy from `.env.example`) and is read once at startup via `src/mask_detector/config.py`. All are optional - sensible defaults are used if unset.

| Variable | Default | Meaning |
|---|---|---|
| `CAMERA_INDEX` | `0` | Index of the webcam to open (`0` is usually the built-in camera) |
| `TARGET_FPS` | `30` | Caps the detection loop's frame rate to reduce CPU usage on fast hardware |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence (0.0-1.0) required to show/act on a detection |
| `MODEL_PATH` | `models/yolov8_mask.pt` | Where mask-detection weights live (auto-downloaded if missing) |
| `MODEL_URL` | *(unset)* | Direct `.pt` download link, tried before the Hugging Face fallback |
| `MODEL_HF_REPO` | `krishnamishra8848/Face_Mask_Detection` | Default Hugging Face Hub repo for automatic download |
| `MODEL_HF_FILENAME` | `best.pt` | Filename to fetch from that repo |
| `DEVICE` | `auto` | `auto` (CUDA > MPS > CPU), or force `cpu`/`cuda`/`mps` |
| `INFERENCE_IMG_SIZE` | `640` | Resolution YOLOv8 resizes frames to internally; lower = faster, less accurate |
| `ALERT_COOLDOWN_SECONDS` | `5` | Minimum seconds between repeated voice alerts |
| `ALERT_MESSAGE` | `Please wear your face mask.` | Exact phrase spoken/shown on a no-mask alert |
| `TTS_ENGINE` | `pyttsx3` | `pyttsx3` (offline) or `gtts` (online); auto-falls back to the other on failure |
| `SCREENSHOTS_DIR` | `Screenshots` | Folder where automatic no-mask screenshots are saved |
| `SCREENSHOT_COOLDOWN_SECONDS` | `5` | Minimum seconds between automatic screenshots |
| `CAMERA_DISCONNECT_TIMEOUT_SECONDS` | `5` | Seconds without a successful frame before a mid-session camera is treated as disconnected |
| `MAX_CONSECUTIVE_FRAME_ERRORS` | `10` | Consecutive frame/detection failures tolerated before the session stops itself |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## ⚡ Performance

`MaskDetector` auto-selects the fastest available device at load time - CUDA GPU, then Apple Silicon (MPS), then CPU - and uses FP16 half-precision on CUDA for roughly 2x throughput. Override with `DEVICE=cpu|cuda|mps` in `.env` if needed, and tune `INFERENCE_IMG_SIZE` (default `640`) to trade accuracy for speed on slower hardware.

Camera frames are read on a dedicated background thread (decoupling camera I/O latency from detection speed), the loop is rate-limited to `TARGET_FPS` to avoid burning CPU beyond what's useful, and the model runs one warm-up inference at load time so the first live frame isn't the slow one. See [docs/performance.md](docs/performance.md) for a full explanation of every optimization applied (FPS, CPU usage, memory-leak prevention, and where threading is/isn't used) and why.

## 🧪 Testing & Code Quality

```bash
pytest                        # unit tests (camera/model hardware is mocked out)
ruff check src tests          # linting
black --check src tests       # formatting
isort --check-only src tests  # import order
mypy src                      # static type checking
```

All four checks (plus `pytest`) run automatically on every push/PR via [`.github/workflows/ci.yml`](.github/workflows/ci.yml) against Python 3.10 and 3.11, and locally on every commit if you've run `pre-commit install` (step 5 of [Installation](#-installation)).

## 🛡️ Error Handling

| Failure | Behavior |
|---|---|
| Camera unavailable at startup | `CameraNotAvailableError` → friendly Streamlit error (mentions permissions as a likely cause), no crash |
| Camera disconnected mid-session | Background reader detects no frames for `CAMERA_DISCONNECT_TIMEOUT_SECONDS`; `read_frame()` raises `CameraNotAvailableError` instead of silently freezing on the last frame forever |
| Model weights missing | Downloaded automatically; `ModelLoadError` with clear next steps only if every automatic option fails |
| Permission denied writing model weights/screenshots/logs | Caught explicitly and surfaced as a specific "permission denied, check X" message rather than a generic failure; logging falls back to console-only if the log directory itself isn't writable |
| Screenshot save failure (permissions, disk full, bad frame) | Logged and skipped (never crashes the detection loop); a one-time friendly toast is shown via `ScreenshotManager.pop_last_error()` so it's not repeated every frame |
| No face in frame | Not an error - neutral "No face detected" status |
| Low-confidence detections | Filtered out via the configurable `CONFIDENCE_THRESHOLD` |
| Voice engine failure | Automatic cross-platform fallback (pyttsx3 ↔ gTTS); on-screen-only alerts if both fail |
| Per-frame processing error | `FrameProcessingError` caught, frame skipped, loop continues |
| Unexpected error during detection | Caught inside the detection loop itself: the camera is released, `camera_running` resets to idle, and a friendly message is shown so a single bad frame doesn't require reloading the page |
| Unexpected error at startup (model/voice loading) | Caught around `load_detector()`/`load_alert_manager()` with a friendly message; voice failures degrade to on-screen-only alerts rather than blocking the app |
| Any other unexpected error | Top-level boundary in `app.py` logs the full traceback and shows a friendly message |

*(Rendered as a table here for readability; see [docs/architecture.md](docs/architecture.md) for the full narrative version.)*

## 🌐 Deployment

The app can be deployed to Docker, Streamlit Community Cloud, and Render - full step-by-step instructions for all three (including a Dockerfile, docker-compose.yml, packages.txt, and a Render render.yaml blueprint, all included in this repo) live in [docs/deployment.md](docs/deployment.md).

**Read this before deploying:** this app captures video via `cv2.VideoCapture`, which opens a physical webcam on the machine the Streamlit process runs on (see [docs/architecture.md](docs/architecture.md#why-local-only-webcam-access) for why). None of the three platforms above have a physical camera attached to their servers, so on all of them the UI, dashboard, and sidebar deploy and run correctly, but clicking **Start Camera** shows a friendly camera-unavailable message by design rather than live detection. That makes these deployments genuinely useful for showcasing the UI/code to reviewers, but not for a live-webcam demo - for that, run it locally (see [Installation](#-installation)) or via Docker on a native Linux host with `--device` passthrough (see [docs/deployment.md](docs/deployment.md#1-docker)).

A true browser-based cloud deployment (streaming a visitor's own webcam over WebRTC, which would remove this limitation) is tracked as a future improvement below.

Quick start with Docker:

```bash
docker build -t mask-detector .
docker run --rm -p 8501:8501 --env-file .env mask-detector
# or: docker compose up --build
```

## 🗺️ Roadmap / Future Improvements

These are intentionally out of scope for v1 but are natural next steps:

- **Cloud/browser deployment** - stream video from the visitor's own browser via `streamlit-webrtc` (or similar) instead of `cv2.VideoCapture(0)`, so a *visitor's* webcam works out of the box on Streamlit Community Cloud/Render, not just on a locally-run instance (see the caveat in [Deployment](#-deployment))
- **Multi-camera / multi-face tracking** - stable per-person IDs across frames (e.g. via a lightweight tracker) instead of only per-frame detections, to reduce duplicate alerts/screenshots for the same person
- **Mask-type classification** - distinguish surgical / N95 / cloth masks, not just mask vs. no-mask
- **Exportable analytics** - download session stats as CSV, or persist them to a small database for historical reporting across sessions
- **Email/SMS/webhook alerts** - notify a channel (e.g. Slack, email) on sustained no-mask detections, not just voice/on-screen
- **REST API mode** - expose the detector over a lightweight API (e.g. FastAPI) for integration into other systems, decoupled from the Streamlit UI
- **Dark mode** - a second theme variant alongside the current light design system
- **Internationalization** - translate alert messages and UI copy beyond English
- **Model fine-tuning guide** - a documented, reproducible pipeline for fine-tuning YOLOv8 on a custom mask dataset, beyond the quick CLI command already in [Installation](#4-model-weights-downloaded-automatically)
- **Test coverage badge** - publish `pytest --cov` results to the README via a coverage badge

Have another idea? Open an issue or see [Contributing](#-contributing) below.

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for:

- The git branching strategy (`main` / `develop` / `feature/*` / `fix/*` / `release/*`)
- Coding standards and the pre-commit checks your PR is expected to pass
- Commit message conventions and the pull request process
- How to report bugs or propose features

Short version: fork the repo, branch off `develop`, make your change with tests, run the checks in [Testing & Code Quality](#-testing--code-quality), and open a PR against `develop`.

## 📄 License

MIT - see [LICENSE](LICENSE). You're free to use, modify, and distribute this project, including commercially, with attribution.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection model and training tooling
- [krishnamishra8848/Face_Mask_Detection](https://huggingface.co/krishnamishra8848/Face_Mask_Detection) on Hugging Face Hub for the default pretrained weights
- [Kaggle "Face Mask Detection" dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) as a reference dataset for fine-tuning your own weights
- [Streamlit](https://streamlit.io/), [OpenCV](https://opencv.org/), and the wider Python open-source ecosystem this project is built on

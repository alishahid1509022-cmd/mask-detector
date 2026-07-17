"""Streamlit UI: real-time webcam mask detection with voice alerts.

The UI is split into small, focused render functions (CSS injection,
header, status indicators, sidebar settings, start/stop controls,
dashboard layout, footer, the detection loop itself) so each concern -
styling, controls, and the actual video/inference loop - can be read,
tested, and changed independently.

Run with:
    streamlit run src/mask_detector/app.py
"""

from __future__ import annotations

import sys
import time
from datetime import timedelta
from pathlib import Path

# When launched via `streamlit run src/mask_detector/app.py`, Python's path
# only includes this file's directory - not `src/` - so bare
# `import mask_detector` fails unless the package was installed editable.
# Inserting `src/` here makes the documented one-liner work either way.
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from mask_detector import config
from mask_detector.alerts.alert_manager import AlertManager
from mask_detector.alerts.voice_alert import get_voice_backend
from mask_detector.detector import MaskDetector
from mask_detector.screenshot import ScreenshotManager
from mask_detector.utils.drawing import draw_detections
from mask_detector.utils.exceptions import (
    CameraNotAvailableError,
    FrameProcessingError,
    ModelLoadError,
    VoiceAlertError,
)
from mask_detector.utils.fps import FPSCounter, FrameRateLimiter
from mask_detector.utils.logger import get_logger
from mask_detector.video_stream import VideoStream

logger = get_logger(__name__, log_level=config.LOG_LEVEL, log_dir=config.LOG_DIR)

st.set_page_config(page_title="AI Face Mask Detection", page_icon="😷", layout="wide")

APP_VERSION = "1.0.0"
GITHUB_URL = "https://github.com/"

# (background, text, icon, message) per detection state, used by the
# color-coded status banner above the video feed. Hex values are chosen to
# match the CSS palette defined in inject_custom_css()'s :root variables -
# duplicated here (rather than parsed out of CSS) since this dict is
# consumed directly by Python, not the browser.
_STATUS_STYLES: dict[str, tuple[str, str, str, str]] = {
    "idle": ("#F1F5F9", "#475569", "⏸️", "Camera stopped — click Start Camera to begin"),
    "safe": ("#ECFDF5", "#047857", "✅", "SAFE — all detected faces are wearing masks"),
    "alert": ("#FEF2F2", "#B91C1C", "🚨", "ALERT — no mask detected!"),
    "no_face": ("#EEF2FF", "#4338CA", "👀", "No face detected"),
}


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------


def inject_custom_css() -> None:
    """Inject the app's design system: colors, typography, cards, chips, footer.

    Everything here is scoped to Streamlit's own component test-ids/classes
    or to custom classes we control (``.app-*``, ``.status-*``), so this can
    be removed at any time with zero effect on app *behavior* - it's a pure
    presentation layer on top of the functional UI built by the rest of
    this module.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --color-primary: #4F46E5;
            --color-primary-dark: #4338CA;
            --color-success: #10B981;
            --color-danger: #EF4444;
            --color-warning: #F59E0B;
            --color-border: #E2E8F0;
            --color-text: #0F172A;
            --color-muted: #64748B;
            --radius-lg: 16px;
            --radius-md: 12px;
            --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.06);
            --shadow-md: 0 6px 20px rgba(15, 23, 42, 0.10);
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--color-text); }

        .main .block-container { padding-top: 1.75rem; padding-bottom: 3rem; max-width: 1200px; }

        /* ---------- Hero header ---------- */
        .app-hero { display: flex; align-items: center; gap: 0.85rem; }
        .app-hero__icon { font-size: 2.5rem; line-height: 1; }
        .app-title { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; color: var(--color-text); }
        .app-subtitle { color: var(--color-muted); font-size: 0.95rem; margin: 0.15rem 0 0; }

        /* ---------- Live/idle badge with pulsing dot ---------- */
        .live-badge {
            display: inline-flex; align-items: center; gap: 0.45rem;
            padding: 0.4rem 0.9rem; border-radius: 999px;
            font-weight: 700; font-size: 0.85rem; letter-spacing: 0.02em;
        }
        .live-badge--on { background: #ECFDF5; color: #047857; }
        .live-badge--off { background: #F1F5F9; color: #475569; }
        .pulse-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
        .live-badge--on .pulse-dot { animation: pulse 1.6s ease-in-out infinite; }
        @keyframes pulse {
            0%   { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.55); }
            70%  { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        /* ---------- Status indicator chips ---------- */
        .status-chip-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 1rem 0 1.25rem; }
        .status-chip {
            display: inline-flex; align-items: center; gap: 0.45rem;
            padding: 0.45rem 0.85rem; border-radius: 999px;
            font-size: 0.82rem; font-weight: 600; color: var(--color-text);
            background: #FFFFFF; border: 1px solid var(--color-border); box-shadow: var(--shadow-sm);
        }
        .status-chip .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
        .dot--ok { background: var(--color-success); }
        .dot--warn { background: var(--color-warning); }
        .dot--off { background: #94A3B8; }

        /* ---------- Section labels ---------- */
        .section-label {
            font-size: 0.95rem; font-weight: 700; color: var(--color-text);
            margin: 0 0 0.75rem;
        }
        .sidebar-brand {
            display: flex; align-items: center; gap: 0.5rem;
            font-weight: 800; font-size: 1.2rem; color: var(--color-text); margin-bottom: 0;
        }
        .sidebar-tagline { color: var(--color-muted); font-size: 0.8rem; margin: 0.1rem 0 0.5rem; }
        .sidebar-section-label {
            text-transform: uppercase; letter-spacing: 0.05em;
            font-size: 0.72rem; font-weight: 700; color: #94A3B8;
            margin: 1.25rem 0 0.4rem;
        }

        /* ---------- Status banner (above the video feed) ---------- */
        .status-banner {
            border-radius: var(--radius-md);
            padding: 1rem 1.25rem;
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 0.9rem;
        }

        /* ---------- Metric "cards" ---------- */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid var(--color-border);
            border-radius: var(--radius-md);
            padding: 1rem 1rem 0.75rem 1rem;
            box-shadow: var(--shadow-sm);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        [data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
        [data-testid="stMetricLabel"] { font-weight: 600; color: var(--color-muted); }
        [data-testid="stMetricValue"] { font-weight: 700; color: var(--color-text); }

        /* ---------- Bordered containers (st.container(border=True)) as cards ---------- */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: var(--radius-lg) !important;
            box-shadow: var(--shadow-sm);
        }

        /* ---------- Live video frame ---------- */
        [data-testid="stImage"] img {
            border-radius: 14px;
            box-shadow: var(--shadow-md);
        }

        /* ---------- Buttons ---------- */
        div.stButton > button {
            border-radius: 10px;
            font-weight: 600;
            padding: 0.6rem 1rem;
            transition: transform 0.1s ease, box-shadow 0.1s ease;
        }
        div.stButton > button:hover { transform: translateY(-1px); box-shadow: var(--shadow-sm); }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
            border-right: 1px solid var(--color-border);
        }

        /* ---------- Footer ---------- */
        .app-footer {
            margin-top: 3rem;
            padding-top: 1.25rem;
            border-top: 1px solid var(--color-border);
            text-align: center;
            color: #94A3B8;
            font-size: 0.85rem;
        }
        .app-footer a { color: var(--color-primary); text-decoration: none; font-weight: 600; }
        .app-footer a:hover { color: var(--color-primary-dark); text-decoration: underline; }
        .footer-badges { margin-top: 0.5rem; display: flex; justify-content: center; gap: 0.4rem; flex-wrap: wrap; }
        .footer-badge {
            display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
            background: #F1F5F9; color: var(--color-muted); font-size: 0.72rem; font-weight: 600;
        }

        /* ---------- Responsive tweaks for narrow (mobile/tablet) viewports ---------- */
        @media (max-width: 768px) {
            .main .block-container { padding-left: 1rem; padding-right: 1rem; }
            .app-title { font-size: 1.5rem; }
            .app-hero__icon { font-size: 1.9rem; }
            .status-chip-row { gap: 0.4rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status_banner(placeholder: DeltaGenerator, status_key: str) -> None:
    """Render a color-coded status banner into a placeholder (replacing its prior content)."""
    bg, fg, icon, message = _STATUS_STYLES[status_key]
    placeholder.markdown(
        f'<div class="status-banner" style="background-color:{bg}; color:{fg};">'
        f"{icon}&nbsp;&nbsp;{message}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached resources (created once per distinct set of arguments, not per rerun)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_detector() -> MaskDetector:
    return MaskDetector()


@st.cache_resource(show_spinner=False)
def load_alert_manager(engine_name: str) -> AlertManager:
    # NOTE: cooldown_seconds is intentionally *not* a parameter here (and
    # therefore not part of the cache key). st.cache_resource never evicts
    # entries by default, so if a slider value were part of the key, every
    # tick the user drags "Alert cooldown" to would permanently retain a
    # brand-new AlertManager (plus its voice backend) for the lifetime of
    # the process - a slow, silent memory leak. Instead we cache one
    # instance per engine (at most 2: pyttsx3/gtts) and mutate its cooldown
    # in place via update_cooldown() - see main(), matching the same
    # pattern already used for detector.confidence_threshold.
    try:
        backend = get_voice_backend(engine_name)
    except VoiceAlertError as exc:
        st.warning(f"Voice alerts unavailable ({exc}). Continuing with on-screen alerts only.")
        backend = None
    return AlertManager(voice_backend=backend)


@st.cache_resource(show_spinner=False)
def load_screenshot_manager() -> ScreenshotManager:
    # Same reasoning as load_alert_manager above: no changing arguments, so
    # there is exactly one cached instance for the app's lifetime; its
    # cooldown is updated in place via update_cooldown() instead of being
    # baked into the cache key.
    return ScreenshotManager()


# ---------------------------------------------------------------------------
# Layout pieces
# ---------------------------------------------------------------------------


def render_header() -> None:
    """Hero title/subtitle on the left, a pulsing LIVE/IDLE badge on the right."""
    title_col, badge_col = st.columns([4, 1])
    with title_col:
        st.markdown(
            '<div class="app-hero">'
            '<span class="app-hero__icon">😷</span>'
            "<div>"
            '<p class="app-title">AI Face Mask Detection System</p>'
            '<p class="app-subtitle">Real-time webcam mask detection powered by YOLOv8, OpenCV &amp; Streamlit</p>'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with badge_col:
        is_live = bool(st.session_state.get("camera_running"))
        modifier = "on" if is_live else "off"
        label = "LIVE" if is_live else "IDLE"
        st.markdown(
            '<div style="display:flex; justify-content:flex-end; align-items:center; height:100%;">'
            f'<span class="live-badge live-badge--{modifier}"><span class="pulse-dot"></span>{label}</span>'
            "</div>",
            unsafe_allow_html=True,
        )


def render_status_indicators(detector_ready: bool, alert_manager: AlertManager) -> None:
    """A row of at-a-glance system-health chips: model, camera, voice, screenshots.

    Distinct from the big color-coded status banner (which reflects the
    *current detection result*, e.g. "no mask detected") - these chips
    reflect the *system's* readiness, which is what a portfolio reviewer
    skimming the page would want to confirm at a glance.
    """
    camera_live = bool(st.session_state.get("camera_running"))

    chips = [
        (
            "🧠",
            "dot--ok" if detector_ready else "dot--warn",
            "Model Ready" if detector_ready else "Model Error",
        ),
        (
            "🎥",
            "dot--ok" if camera_live else "dot--off",
            "Camera Live" if camera_live else "Camera Idle",
        ),
        (
            "🔊",
            "dot--ok" if alert_manager.voice_available else "dot--warn",
            "Voice Alerts On" if alert_manager.voice_available else "Voice Alerts Off",
        ),
        ("📸", "dot--ok", "Auto-Screenshots On"),
    ]
    chip_html = "".join(
        f'<span class="status-chip"><span class="dot {dot_class}"></span>{icon}&nbsp;{text}</span>'
        for icon, dot_class, text in chips
    )
    st.markdown(f'<div class="status-chip-row">{chip_html}</div>', unsafe_allow_html=True)


def render_settings_sidebar() -> dict:
    """Branded, grouped sidebar: camera/detection, voice alerts, screenshots, about."""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">😷 Mask Detector</div>'
            '<p class="sidebar-tagline">Control Panel</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="sidebar-section-label">📷 Camera &amp; Detection</p>', unsafe_allow_html=True
        )
        camera_index = st.number_input(
            "Camera index", min_value=0, max_value=10, value=config.CAMERA_INDEX, step=1
        )
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.1,
            max_value=0.95,
            value=config.CONFIDENCE_THRESHOLD,
            step=0.05,
        )

        st.markdown('<p class="sidebar-section-label">🔊 Voice Alerts</p>', unsafe_allow_html=True)
        tts_engine = st.selectbox(
            "Voice alert engine",
            options=["pyttsx3", "gtts"],
            index=0 if config.TTS_ENGINE == "pyttsx3" else 1,
            help="pyttsx3 works fully offline. gTTS needs internet but sounds more natural.",
        )
        cooldown_seconds = st.slider(
            "Alert cooldown (seconds)",
            min_value=1,
            max_value=30,
            value=int(config.ALERT_COOLDOWN_SECONDS),
        )

        st.markdown('<p class="sidebar-section-label">📸 Screenshots</p>', unsafe_allow_html=True)
        screenshot_cooldown_seconds = st.slider(
            "Screenshot cooldown (seconds)",
            min_value=1,
            max_value=60,
            value=int(config.SCREENSHOT_COOLDOWN_SECONDS),
            help="Minimum time between automatic no-mask screenshots.",
        )

        st.markdown('<p class="sidebar-section-label">ℹ️ About</p>', unsafe_allow_html=True)
        with st.expander("Tech stack & links"):
            st.markdown(
                "- **YOLOv8** (Ultralytics) — detection model\n"
                "- **OpenCV** — threaded webcam capture\n"
                "- **Streamlit** — this UI\n"
                "- **pyttsx3 / gTTS** — voice alerts\n\n"
                f"[View source on GitHub]({GITHUB_URL}) · v{APP_VERSION}"
            )

    return {
        "camera_index": int(camera_index),
        "confidence_threshold": confidence_threshold,
        "tts_engine": tts_engine,
        "cooldown_seconds": float(cooldown_seconds),
        "screenshot_cooldown_seconds": float(screenshot_cooldown_seconds),
    }


def render_controls() -> None:
    """Render the Start/Stop buttons and manage `camera_running` session state.

    Both buttons are always visible; whichever action doesn't currently
    apply is disabled (Start while already running, Stop while idle)
    rather than hidden, so the control bar's layout never jumps around.
    """
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    start_col, stop_col, _spacer = st.columns([1, 1, 4])
    with start_col:
        if st.button(
            "▶️ Start Camera",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.camera_running,
        ):
            st.session_state.camera_running = True
            st.rerun()
    with stop_col:
        if st.button(
            "⏹️ Stop Camera",
            use_container_width=True,
            disabled=not st.session_state.camera_running,
        ):
            st.session_state.camera_running = False
            st.rerun()


def render_dashboard_layout() -> dict[str, DeltaGenerator]:
    """Build the responsive two-column, card-based layout and return its placeholders.

    Streamlit columns stack vertically on narrow viewports automatically,
    and each column's content lives inside a bordered ``st.container`` -
    a native Streamlit "card" - so both the video feed and the dashboard
    read as distinct, professionally-framed panels rather than loose
    widgets floating on the page.
    """
    video_col, stats_col = st.columns([2, 1])

    with video_col, st.container(border=True):
        st.markdown('<p class="section-label">🎥 Live Detection Feed</p>', unsafe_allow_html=True)
        status_placeholder = st.empty()
        frame_placeholder = st.empty()

    with stats_col, st.container(border=True):
        st.markdown('<p class="section-label">📊 Dashboard</p>', unsafe_allow_html=True)
        total_placeholder = st.empty()
        mask_placeholder = st.empty()
        no_mask_placeholder = st.empty()
        confidence_placeholder = st.empty()
        fps_placeholder = st.empty()
        duration_placeholder = st.empty()

        st.markdown(
            '<p class="section-label" style="margin-top:0.5rem;">📌 This Frame</p>',
            unsafe_allow_html=True,
        )
        faces_placeholder = st.empty()
        screenshots_placeholder = st.empty()

    return {
        "status": status_placeholder,
        "frame": frame_placeholder,
        "total_detections": total_placeholder,
        "mask": mask_placeholder,
        "no_mask": no_mask_placeholder,
        "confidence": confidence_placeholder,
        "fps": fps_placeholder,
        "duration": duration_placeholder,
        "faces": faces_placeholder,
        "screenshots": screenshots_placeholder,
    }


def render_idle_stats(placeholders: dict) -> None:
    """Show placeholder ("—") stats while the camera isn't running."""
    render_status_banner(placeholders["status"], "idle")
    placeholders["total_detections"].metric("Total Detections", 0)
    placeholders["mask"].metric("Mask Detections", 0)
    placeholders["no_mask"].metric("No-Mask Detections", 0)
    placeholders["confidence"].metric("Average Confidence", "—")
    placeholders["fps"].metric("Current FPS", "—")
    placeholders["duration"].metric("Session Duration", "0:00:00")
    placeholders["faces"].metric("Faces in Frame", "—")
    placeholders["screenshots"].metric("Screenshots Saved", 0)


def render_footer() -> None:
    """A quiet, portfolio-style footer: credits, tech badges, and a source link."""
    tech = ["YOLOv8", "OpenCV", "Streamlit", "pyttsx3", "gTTS"]
    badges_html = "".join(f'<span class="footer-badge">{name}</span>' for name in tech)
    st.markdown(
        '<div class="app-footer">'
        f"Built as an AI/ML portfolio project · v{APP_VERSION} · "
        f'<a href="{GITHUB_URL}" target="_blank">View on GitHub</a>'
        f'<div class="footer-badges">{badges_html}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Detection loop
# ---------------------------------------------------------------------------


def run_detection_loop(
    settings: dict,
    detector: MaskDetector,
    alert_manager: AlertManager,
    screenshot_manager: ScreenshotManager,
    placeholders: dict,
) -> None:
    detector.confidence_threshold = settings["confidence_threshold"]

    fps_counter = FPSCounter()
    # Caps the loop at config.TARGET_FPS instead of letting it spin as fast
    # as the CPU/GPU allow. On hardware where detection is faster than the
    # target rate, this directly cuts CPU usage (fewer wasted iterations,
    # inference calls, and Streamlit re-renders per second) without
    # lowering the FPS anyone would actually perceive on screen.
    rate_limiter = FrameRateLimiter(target_fps=config.TARGET_FPS)
    session_start = time.monotonic()
    total_detections = 0
    mask_count = 0
    no_mask_count = 0
    confidence_sum = 0.0
    consecutive_errors = 0

    try:
        with VideoStream(
            camera_index=settings["camera_index"], target_fps=config.TARGET_FPS
        ) as stream:
            while True:
                frame = stream.read_frame()
                if frame is None:
                    # read_frame() no longer blocks (the camera is now read on
                    # a background thread - see video_stream.py), so without
                    # this short sleep a persistently-empty frame (most
                    # commonly: the reader thread just hasn't grabbed its
                    # first frame yet, moments after Start Camera) would spin
                    # this loop as fast as Python allows, pegging a CPU core
                    # and burning through MAX_CONSECUTIVE_FRAME_ERRORS in a
                    # few milliseconds - false-failing before the camera ever
                    # gets a real chance to warm up.
                    time.sleep(0.1)
                    consecutive_errors += 1
                    if consecutive_errors >= config.MAX_CONSECUTIVE_FRAME_ERRORS:
                        st.error("Too many consecutive failures reading from the camera. Stopping.")
                        st.session_state.camera_running = False
                        break
                    continue

                try:
                    detections = detector.predict(frame)
                except FrameProcessingError:
                    logger.exception("Detection failed on a frame; skipping it")
                    consecutive_errors += 1
                    if consecutive_errors >= config.MAX_CONSECUTIVE_FRAME_ERRORS:
                        st.error("Too many consecutive detection failures. Stopping.")
                        st.session_state.camera_running = False
                        break
                    continue

                consecutive_errors = 0

                # --- Video feed -------------------------------------------------
                # channels="BGR" lets Streamlit handle the BGR->RGB swap itself,
                # skipping our own cv2.cvtColor + PIL Image.fromarray round-trip
                # (one less full-frame color-conversion + allocation per frame,
                # every frame - a direct, easy CPU win in the hottest part of the loop).
                annotated = draw_detections(frame, detections)
                placeholders["frame"].image(annotated, channels="BGR", use_container_width=True)

                # --- Status banner -----------------------------------------------
                no_mask_detected = any(d.is_no_mask for d in detections)
                if no_mask_detected:
                    status_key = "alert"
                elif detections:
                    status_key = "safe"
                else:
                    status_key = "no_face"
                render_status_banner(placeholders["status"], status_key)

                alert_result = alert_manager.notify(no_mask_detected)
                if no_mask_detected and not alert_result.voice_available:
                    st.toast("Voice alerts unavailable — showing on-screen alerts only.", icon="🔇")

                # --- Screenshot ----------------------------------------------------
                # Saves the annotated frame (with boxes/labels) so the screenshot is
                # self-contained evidence, not just a plain, unlabeled camera frame.
                screenshot_path = screenshot_manager.capture(annotated, no_mask_detected)
                if screenshot_path is not None:
                    st.toast(f"📸 Screenshot saved: {screenshot_path.name}", icon="📸")
                screenshot_error = screenshot_manager.pop_last_error()
                if screenshot_error is not None:
                    # pop_last_error() clears itself once read, so this fires once per
                    # new failure rather than spamming a toast on every frame of a
                    # sustained no-mask streak (see ScreenshotManager.pop_last_error).
                    st.toast(f"⚠️ {screenshot_error}", icon="⚠️")

                # --- Dashboard (session-wide, cumulative) ---------------------------
                # A running average confidence is far more readable on a dashboard
                # than a single frame's snapshot, which jitters detection to
                # detection - unlike FPS/duration, which are meant to read "now".
                total_detections += len(detections)
                mask_count += sum(1 for d in detections if d.label == "mask")
                no_mask_count += sum(1 for d in detections if d.is_no_mask)
                confidence_sum += sum(d.confidence for d in detections)
                avg_confidence = confidence_sum / total_detections if total_detections else None
                session_duration = str(timedelta(seconds=int(time.monotonic() - session_start)))

                placeholders["total_detections"].metric("Total Detections", total_detections)
                placeholders["mask"].metric("Mask Detections", mask_count)
                placeholders["no_mask"].metric("No-Mask Detections", no_mask_count)
                placeholders["confidence"].metric(
                    "Average Confidence",
                    f"{avg_confidence:.0%}" if avg_confidence is not None else "—",
                )
                placeholders["fps"].metric("Current FPS", f"{fps_counter.tick():.1f}")
                placeholders["duration"].metric("Session Duration", session_duration)

                # --- This frame -------------------------------------------------------
                placeholders["faces"].metric("Faces in Frame", len(detections))
                placeholders["screenshots"].metric(
                    "Screenshots Saved", screenshot_manager.total_saved
                )

                # Sleeps out any leftover frame budget - short enough (well
                # under a second at any reasonable target FPS) that it never
                # noticeably delays reacting to a Stop Camera click, since
                # every st.* call above already gives Streamlit a chance to
                # interrupt this script if a rerun (e.g. Stop) was requested.
                rate_limiter.wait()

    except CameraNotAvailableError as exc:
        st.error(
            f"📷 {exc}\n\nTry a different camera index, or check that no other app is using it."
        )
        st.session_state.camera_running = False

    except Exception as exc:  # noqa: BLE001 - last-resort boundary for this session
        # Anything unexpected here (a corrupt frame, an OpenCV/YOLO internal
        # error, a transient OS error, ...) stops *this* detection session
        # cleanly rather than crashing the whole Streamlit script: the
        # camera is released by VideoStream's context manager either way,
        # camera_running is reset so the UI returns to a normal "IDLE" state
        # instead of a stuck "LIVE" badge, and the message stays friendly
        # and actionable instead of a raw traceback.
        logger.exception("Unexpected error during the detection loop")
        st.error(
            f"⚠️ An unexpected error stopped detection: {exc}\n\n"
            "The camera has been released. Check the logs for details, then click Start Camera to try again."
        )
        st.session_state.camera_running = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    inject_custom_css()

    try:
        detector = load_detector()
    except ModelLoadError as exc:
        # The one failure mode ensure_weights()/MaskDetector already turn
        # into specific, actionable guidance (missing file, permission
        # denied, no internet, ...) - shown to the user verbatim rather than
        # wrapped in extra text, since the message itself already tells
        # them exactly what to do next.
        st.error(f"🧠 Model failed to load\n\n{exc}")
        st.stop()
        return
    except Exception as exc:  # noqa: BLE001 - anything not already a ModelLoadError
        # Catches truly unexpected startup failures (e.g. a corrupted
        # ultralytics install, an out-of-memory error) that ensure_weights()
        # and MaskDetector don't already wrap in a friendlier exception, so
        # the user never sees a raw traceback even for cases we didn't
        # anticipate by name.
        logger.exception("Unexpected error while loading the detection model")
        st.error(
            f"⚠️ An unexpected error occurred while loading the detection model: {exc}\n\n"
            "Check the logs for details, or try restarting the app."
        )
        st.stop()
        return

    render_header()
    settings = render_settings_sidebar()

    try:
        alert_manager = load_alert_manager(settings["tts_engine"])
    except Exception as exc:  # noqa: BLE001 - voice alerts are a nice-to-have, never fatal
        # get_voice_backend() already handles the expected failure modes
        # (missing package, broken OS speech driver, ...) via automatic
        # fallback between engines - this only catches something neither
        # backend anticipated, and degrades to on-screen-only alerts rather
        # than blocking the whole app over a non-essential feature.
        logger.exception("Unexpected error initializing voice alerts")
        st.warning(
            f"Voice alerts unavailable due to an unexpected error ({exc}). Continuing with on-screen alerts only."
        )
        alert_manager = AlertManager(voice_backend=None)
    alert_manager.update_cooldown(settings["cooldown_seconds"])

    screenshot_manager = load_screenshot_manager()
    screenshot_manager.update_cooldown(settings["screenshot_cooldown_seconds"])

    render_status_indicators(detector_ready=True, alert_manager=alert_manager)
    render_controls()
    placeholders = render_dashboard_layout()

    # Rendered now (once, in the right DOM position) rather than after the
    # branch below - run_detection_loop() only updates the st.empty()
    # placeholders already created above, it never appends new content, so
    # the footer stays pinned at the bottom of the page for the entire
    # session, including while detection is actively running.
    render_footer()

    if st.session_state.camera_running:
        run_detection_loop(settings, detector, alert_manager, screenshot_manager, placeholders)
    else:
        render_idle_stats(placeholders)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - top-level UI error boundary
        logger.exception("Unexpected error in the Streamlit app")
        st.error("An unexpected error occurred. Check the logs for details.")

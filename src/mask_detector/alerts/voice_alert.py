"""Text-to-speech backends for voice alerts.

Two interchangeable backends implement the same small interface:

- ``PyttsxVoiceAlert`` (default): fully offline, low latency, runs on a
  background thread so it never blocks the video/detection loop. Uses the
  OS's native speech engine under the hood (SAPI5 on Windows, NSSS on
  macOS, espeak/espeak-ng on Linux), so availability varies by platform.
- ``GTTSVoiceAlert``: uses Google's online TTS for more natural voices, at
  the cost of requiring internet access and higher latency per phrase.

Because pyttsx3's dependency on OS-level speech drivers means it can fail
to initialize on some machines (most commonly Linux without espeak
installed), get_voice_backend() automatically falls back to the other
engine if the configured one fails, so voice alerts keep working across
platforms without per-OS configuration.
"""

from __future__ import annotations

import tempfile
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from mask_detector import config
from mask_detector.utils.exceptions import VoiceAlertError
from mask_detector.utils.logger import get_logger

logger = get_logger(__name__, log_level=config.LOG_LEVEL, log_dir=config.LOG_DIR)


class VoiceAlertBackend(ABC):
    """Common interface every TTS backend implements."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Speak ``text`` asynchronously (must not block the caller for long)."""


class PyttsxVoiceAlert(VoiceAlertBackend):
    """Offline TTS backend using ``pyttsx3``."""

    def __init__(self, rate: int = 170, volume: float = 1.0) -> None:
        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)
            self._engine.setProperty("volume", volume)
        except Exception as exc:  # noqa: BLE001 - pyttsx3 raises plain Exception/OSError
            raise VoiceAlertError(
                f"Could not initialize pyttsx3 (offline TTS): {exc}. "
                "On Linux you may need to install 'espeak'; on some systems "
                "try switching TTS_ENGINE to 'gtts' instead."
            ) from exc

        self._lock = threading.Lock()
        self._busy = False

    def speak(self, text: str) -> None:
        if self._busy:
            logger.debug("Skipping voice alert; previous phrase still playing")
            return
        threading.Thread(target=self._speak_worker, args=(text,), daemon=True).start()

    def _speak_worker(self, text: str) -> None:
        with self._lock:
            self._busy = True
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:  # noqa: BLE001
                logger.exception("pyttsx3 failed to speak alert")
            finally:
                self._busy = False


class GTTSVoiceAlert(VoiceAlertBackend):
    """Online TTS backend using Google's ``gTTS`` + local audio playback."""

    def __init__(self, language: str = "en") -> None:
        # Fail fast if the package itself is missing, mirroring
        # PyttsxVoiceAlert's init-time failure - this is what lets
        # get_voice_backend()'s automatic fallback actually detect
        # "this engine won't work here" without waiting for a real speak()
        # call (network failures, on the other hand, can only be detected
        # when actually attempting a call - those surface at speak() time).
        try:
            import gtts  # noqa: F401
        except ImportError as exc:
            raise VoiceAlertError(
                "The 'gTTS' package is not installed. Run `pip install -r requirements.txt`."
            ) from exc
        self.language = language

    def speak(self, text: str) -> None:
        try:
            from gtts import gTTS
        except ImportError as exc:
            raise VoiceAlertError("The 'gTTS' package is not installed.") from exc

        try:
            tts = gTTS(text=text, lang=self.language)
            tmp_path = Path(tempfile.gettempdir()) / "mask_detector_alert.mp3"
            tts.save(str(tmp_path))
        except Exception as exc:  # noqa: BLE001 - network/API errors
            raise VoiceAlertError(f"gTTS failed to synthesize speech: {exc}") from exc

        threading.Thread(target=self._play, args=(tmp_path,), daemon=True).start()

    @staticmethod
    def _play(path: Path) -> None:
        try:
            from playsound import playsound
        except ImportError:
            logger.warning(
                "Synthesized speech saved to %s but 'playsound' is not installed "
                "(it's optional and excluded from requirements.txt by default - "
                "see README > Voice alerts). Falling back to the on-screen alert only.",
                path,
            )
            return
        try:
            playsound(str(path))
        except Exception:  # noqa: BLE001
            logger.exception("Failed to play synthesized audio at %s", path)


_BACKENDS: dict[str, type[VoiceAlertBackend]] = {
    "pyttsx3": PyttsxVoiceAlert,
    "gtts": GTTSVoiceAlert,
}


def get_voice_backend(
    engine_name: str = config.TTS_ENGINE, fallback: bool = True
) -> VoiceAlertBackend:
    """Factory: instantiate the configured TTS backend by name.

    If ``fallback`` is True (the default) and the requested engine fails
    to initialize, the other backend is tried automatically before giving
    up - e.g. on a Linux box with no ``espeak`` installed, this silently
    switches from ``pyttsx3`` to ``gtts`` instead of leaving voice alerts
    broken. Set ``fallback=False`` to require the exact requested engine.
    """
    engine_name = engine_name.lower()
    backend_cls = _BACKENDS.get(engine_name)
    if backend_cls is None:
        raise VoiceAlertError(
            f"Unknown TTS_ENGINE '{engine_name}'. Valid options: {list(_BACKENDS)}"
        )

    try:
        return backend_cls()
    except VoiceAlertError as primary_error:
        if not fallback or len(_BACKENDS) < 2:
            raise

        other_name = next(name for name in _BACKENDS if name != engine_name)
        logger.warning(
            "Voice backend '%s' failed to initialize (%s); trying fallback '%s'",
            engine_name,
            primary_error,
            other_name,
        )
        try:
            return _BACKENDS[other_name]()
        except VoiceAlertError as fallback_error:
            raise VoiceAlertError(
                f"Both voice backends failed to initialize: "
                f"'{engine_name}' ({primary_error}); '{other_name}' ({fallback_error})"
            ) from fallback_error

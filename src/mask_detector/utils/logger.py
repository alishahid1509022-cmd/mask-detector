"""Centralized logging configuration.

All modules should obtain their logger via :func:`get_logger` rather than
using ``print`` or creating ad-hoc handlers, so log level/format/output stays
consistent across the whole application.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def _configure_root(log_level: str = "INFO", log_dir: Path | None = None) -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger("mask_detector")
    root.setLevel(log_level.upper())

    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_dir / "app.log", maxBytes=1_000_000, backupCount=3
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:
            # Non-fatal: fall back to console-only logging if the log
            # directory cannot be created (e.g. read-only filesystem).
            console_handler.emit(
                logging.LogRecord(
                    name="mask_detector",
                    level=logging.WARNING,
                    pathname=__file__,
                    lineno=0,
                    msg="Could not create log directory %s; file logging disabled",
                    args=(log_dir,),
                    exc_info=None,
                )
            )

    _configured = True


def get_logger(name: str, log_level: str = "INFO", log_dir: Path | None = None) -> logging.Logger:
    """Return a configured, namespaced logger.

    Args:
        name: Usually ``__name__`` of the calling module.
        log_level: Root log level (only applied the first time this is called).
        log_dir: Optional directory for rotating file logs.
    """
    _configure_root(log_level=log_level, log_dir=log_dir)
    return logging.getLogger(f"mask_detector.{name}")

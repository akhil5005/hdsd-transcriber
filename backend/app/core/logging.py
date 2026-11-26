"""Logging setup.

Privacy rule for this codebase: transcript text and user-supplied reference
text are patient data and must never be written to logs. Log identifiers,
durations and metrics instead. :func:`redact` exists for the rare case where a
caller wants to prove in a log line that text was present without revealing it.
"""

from __future__ import annotations

import logging
import sys

from app.core.config import settings

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format=_LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    # Uvicorn's access log duplicates our own request logging at DEBUG.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def redact(text: str | None) -> str:
    """Describe text without disclosing it."""
    if text is None:
        return "<none>"
    return f"<redacted {len(text)} chars>"

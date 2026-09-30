"""Structured logging for the hybrid support service.

One JSON object per line, so a log pipeline can parse records without a regex and
without a schema file. Anything passed through ``extra=`` on a log call becomes a
top-level field in the emitted object.

Two rules this module exists to enforce:

1. Raw customer text is never logged above DEBUG. Requests are identified by a
   session fingerprint, not by the session id the client sent.
2. Slots are logged through ``ExtractedEntities.masked_summary()`` so a PIN cannot
   reach a log sink in clear text.
"""

import hashlib
import json
import logging
import logging.handlers
import os
from pathlib import Path

# Standard LogRecord attributes, so the formatter can tell "extra" fields apart
# from the ones logging sets itself.
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def is_configured() -> bool:
    """True when a JsonFormatter is already attached to the root logger."""
    return any(
        isinstance(handler.formatter, JsonFormatter)
        for handler in logging.getLogger().handlers
    )


def configure_logging(level: str | None = None, log_file: str | None = None) -> None:
    """Attach JSON handlers to the root logger. Idempotent.

    Console always, plus a rotating file when LOG_FILE is set or log_file is passed.
    """
    if is_configured():
        return

    resolved_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    # Third-party libraries log at INFO constantly (httpx, huggingface_hub,
    # sentence_transformers). With the root at INFO every model load buries the
    # application's own records under dozens of JSON lines, so the root stays quiet
    # and this package owns the configured level.
    root.setLevel(logging.WARNING)
    logging.getLogger("src").setLevel(resolved_level)

    console = logging.StreamHandler()
    console.setFormatter(JsonFormatter())
    root.addHandler(console)

    target = log_file or os.environ.get("LOG_FILE")
    if target:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            target, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        rotating.setFormatter(JsonFormatter())
        root.addHandler(rotating)


def session_fingerprint(session_id: str) -> str:
    """Stable short hash of a session id, safe to log."""
    return hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:12]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

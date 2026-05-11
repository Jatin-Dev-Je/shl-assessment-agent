from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"
LOG_LEVEL = logging.INFO

# ── JSON Formatter ────────────────────────────────────────────────────────────
class JSONFormatter(logging.Formatter):
	"""Formats every log record as a single JSON line."""

	def format(self, record: logging.LogRecord) -> str:
		log_entry = {
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"level": record.levelname,
			"module": record.module,
			"function": record.funcName,
			"line": record.lineno,
			"message": record.getMessage(),
		}

		# Attach exception info if present
		if record.exc_info:
			log_entry["exception"] = self.formatException(record.exc_info)

		# Attach any extra fields passed via extra={}
		for key, value in record.__dict__.items():
			if key not in {
				"args", "asctime", "created", "exc_info", "exc_text",
				"filename", "funcName", "id", "levelname", "levelno",
				"lineno", "module", "msecs", "message", "msg", "name",
				"pathname", "process", "processName", "relativeCreated",
				"stack_info", "thread", "threadName",
			}:
				log_entry[key] = value

		return json.dumps(log_entry, default=str)


# ── Setup ─────────────────────────────────────────────────────────────────────
def _setup_logging() -> None:
	"""Configure root logger once at import time."""
	LOG_DIR.mkdir(parents=True, exist_ok=True)

	formatter = JSONFormatter()

	# Console handler
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(formatter)
	console_handler.setLevel(LOG_LEVEL)

	# File handler
	file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
	file_handler.setFormatter(formatter)
	file_handler.setLevel(LOG_LEVEL)

	root_logger = logging.getLogger()
	root_logger.setLevel(LOG_LEVEL)

	# Avoid duplicate handlers on reload
	if not root_logger.handlers:
		root_logger.addHandler(console_handler)
		root_logger.addHandler(file_handler)


_setup_logging()


# ── Public API ────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
	"""
	Get a named logger.

	Usage:
		from logging_config import get_logger
		logger = get_logger(__name__)
		logger.info("message", extra={"intent": "RECOMMEND"})
	"""
	return logging.getLogger(name)

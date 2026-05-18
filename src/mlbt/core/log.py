"""Single project-wide logger. Avoids leaking handler config to library users."""
from __future__ import annotations

import logging
import os
import sys

_INITIALISED = False


def _init() -> None:
    global _INITIALISED
    if _INITIALISED:
        return
    level = os.environ.get("MLBT_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger("mlbt")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _INITIALISED = True


def get_logger(name: str) -> logging.Logger:
    _init()
    return logging.getLogger(f"mlbt.{name}")

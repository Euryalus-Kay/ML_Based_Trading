"""Optional API-key loader. Reads .env if present and exposes a typed getter.

Every key is OPTIONAL — sources guard their own usage. The default universe
ships with zero-key sources so the system runs without any signup.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _load_dotenv_once() -> None:
    if getattr(_load_dotenv_once, "_done", False):
        return
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    _load_dotenv_once._done = True  # type: ignore[attr-defined]


def get_key(name: str, default: Optional[str] = None) -> Optional[str]:
    _load_dotenv_once()
    val = os.environ.get(name, default)
    if val in ("", None):
        return None
    return val


def has_key(name: str) -> bool:
    return get_key(name) is not None

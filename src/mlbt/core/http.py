"""Single shared HTTP layer: caching, retries, rate limiting, polite UA.

Every source uses this so caching/back-off is uniform and disk-backed.
"""
from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import requests_cache
    _HAS_CACHE = True
except ImportError:  # pragma: no cover
    _HAS_CACHE = False

from mlbt.core.log import get_logger

log = get_logger("http")

USER_AGENT = (
    "mlbt/0.1 (+https://github.com/euryalus-kay/ml_based_trading; "
    "research data collector)"
)


def _cache_path() -> Path:
    import os
    base = Path(os.environ.get("MLBT_DATA_DIR", "./data"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "cache" / "http"


_session_lock = threading.Lock()
_session: Optional[requests.Session] = None


def http_session() -> requests.Session:
    """Returns a process-wide cached, retrying Session."""
    global _session
    with _session_lock:
        if _session is not None:
            return _session
        if _HAS_CACHE:
            cache_dir = _cache_path()
            cache_dir.parent.mkdir(parents=True, exist_ok=True)
            sess = requests_cache.CachedSession(
                cache_name=str(cache_dir),
                backend="sqlite",
                expire_after=60 * 60 * 6,  # 6h default
                allowable_methods=("GET",),
                stale_if_error=True,
            )
        else:
            sess = requests.Session()
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "HEAD"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        sess.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
        _session = sess
        return sess


_last_call: dict[str, float] = {}
_rate_lock = threading.Lock()


def _throttle(host: str, min_interval: float) -> None:
    if min_interval <= 0:
        return
    with _rate_lock:
        now = time.monotonic()
        last = _last_call.get(host, 0.0)
        wait = (last + min_interval) - now
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()


def http_get(
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = 30.0,
    min_interval: float = 0.0,
    raise_on_error: bool = True,
) -> requests.Response:
    """GET with caching, retry, per-host rate limit. min_interval is seconds."""
    sess = http_session()
    host = url.split("/")[2] if "://" in url else url
    _throttle(host, min_interval)
    try:
        resp = sess.get(url, params=params, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        log.warning("http_get failed: %s :: %s", url, e)
        if raise_on_error:
            raise
        # Build an empty fake response
        r = requests.Response()
        r.status_code = 599
        r._content = b""
        r.url = url
        return r
    if resp.status_code >= 400:
        log.warning("http_get %s -> %s", url, resp.status_code)
        if raise_on_error and resp.status_code != 404:
            resp.raise_for_status()
    return resp

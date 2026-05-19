"""FRED economic series via the anonymous fredgraph.csv endpoint.

No key required — `https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES`
returns daily CSV. Each series becomes one column on the canonical grid.

We treat all FRED series as having a 1-business-day publish lag (release
schedules vary; this is conservative and prevents accidental peeking).
"""
from __future__ import annotations

import io
from typing import Any, List

import pandas as pd

from mlbt.core.base import DataSource
from mlbt.core.registry import register
from mlbt.core.http import http_get
from mlbt.core.log import get_logger

log = get_logger("fred")

_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _fetch_series(series_id: str) -> pd.DataFrame:
    resp = http_get(_URL, params={"id": series_id}, min_interval=0.3, raise_on_error=False)
    if resp.status_code != 200 or not resp.text:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty:
        return pd.DataFrame()
    # FRED columns: 'observation_date' (lowercase) + series_id (original case)
    date_col = next((c for c in df.columns if c.lower() in ("observation_date", "date")), None)
    if not date_col:
        return pd.DataFrame()
    val_col = next((c for c in df.columns if c != date_col), None)
    if val_col is None:
        return pd.DataFrame()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.set_index(date_col).tz_localize("UTC")
    df = df.rename(columns={val_col: series_id})
    return df[[series_id]]


@register("fred")
class FredAnonymous(DataSource):
    name: str = "fred"
    frequency: str = "1d"
    schema = {}  # populated dynamically
    publish_lag = pd.Timedelta(days=1)

    def fetch(self, start, end, *, series: List[str] | None = None, **kw: Any) -> pd.DataFrame:
        series = series or []
        frames = []
        for s in series:
            try:
                df = _fetch_series(s)
                if not df.empty:
                    frames.append(df)
            except Exception as e:  # noqa: BLE001
                log.warning("fred %s failed: %s", s, e)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=1).sort_index()
        # Shift to next-business-day 14:00 UTC (close-of-business release proxy)
        out.index = out.index.normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=14)
        from mlbt.core.base import to_utc
        return out.loc[(out.index >= to_utc(start)) & (out.index <= to_utc(end))]

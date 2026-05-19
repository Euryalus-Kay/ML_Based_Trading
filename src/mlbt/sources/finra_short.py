"""FINRA short sale volume — daily reg-SHO file.

FINRA publishes daily short volume CSVs for every exchange. Aggregated short
volume / total volume is a real-time short-selling pressure signal.
No key.
"""
from __future__ import annotations

import io
from typing import Any, List

import pandas as pd

from mlbt.core.base import DataSource
from mlbt.core.registry import register
from mlbt.core.http import http_get
from mlbt.core.log import get_logger

log = get_logger("finra_short")

_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"


def _fetch_day(day: pd.Timestamp) -> pd.DataFrame:
    url = _URL.format(ymd=day.strftime("%Y%m%d"))
    resp = http_get(url, min_interval=0.2, raise_on_error=False)
    if resp.status_code != 200 or not resp.text:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(resp.text), sep="|")
    if df.empty or "Symbol" not in df.columns:
        return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={
        "shortvolume": "short_volume",
        "shortexemptvolume": "short_exempt_volume",
        "totalvolume": "total_volume",
    })
    df["short_pct"] = df["short_volume"] / df["total_volume"].replace(0, pd.NA)
    df["ts"] = day.normalize() + pd.Timedelta(hours=22)
    return df.set_index("ts")[["symbol", "short_volume", "short_exempt_volume",
                                "total_volume", "short_pct"]]


@register("finra_short")
class FinraShort(DataSource):
    name: str = "finra_short"
    frequency: str = "1d"
    schema = {
        "symbol": "object", "short_volume": "float64",
        "short_exempt_volume": "float64", "total_volume": "float64",
        "short_pct": "float64",
    }
    publish_lag = pd.Timedelta(hours=18)

    def fetch(self, start, end, *, symbols: List[str] | None = None, **kw: Any) -> pd.DataFrame:
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        # Only weekdays
        days = pd.bdate_range(start_ts, end_ts)
        frames = []
        targets = {s.upper() for s in (symbols or [])}
        for d in days:
            try:
                df = _fetch_day(d)
                if df.empty:
                    continue
                if targets:
                    df = df[df["symbol"].str.upper().isin(targets)]
                if not df.empty:
                    frames.append(df)
            except Exception as e:  # noqa: BLE001
                log.warning("finra %s failed: %s", d, e)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()

"""Yahoo Finance bars via yfinance.

Covers EVERYTHING that has a Yahoo symbol: equities, ETFs, indices (^GSPC,
^VIX...), futures (ES=F, CL=F...), FX (EURUSD=X), crypto (BTC-USD). Free, no
key, ~30 days of 1m bars and decades of daily.

We split into two registrations:
  - yf_bars_intraday: 1m/5m/15m for recent window
  - yf_bars_daily: daily, deep history
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from mlbt.core.base import DataSource
from mlbt.core.registry import register
from mlbt.core.log import get_logger

log = get_logger("yf_bars")


def _yf():
    import yfinance as yf
    return yf


def _fetch_yf_chunk(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
) -> pd.DataFrame:
    yf = _yf()
    raw = yf.download(
        tickers=" ".join(symbols),
        start=start.tz_convert(None) if start.tz else start,
        end=end.tz_convert(None) if end.tz else end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        for sym in symbols:
            if sym not in raw.columns.get_level_values(0):
                continue
            sub = raw[sym].copy()
            sub.columns = [str(c).lower().replace(" ", "_") for c in sub.columns]
            sub = sub.dropna(how="all")
            if sub.empty:
                continue
            sub["symbol"] = sym
            frames.append(sub)
    else:
        sub = raw.copy()
        sub.columns = [str(c).lower().replace(" ", "_") for c in sub.columns]
        sub = sub.dropna(how="all")
        if not sub.empty:
            sub["symbol"] = symbols[0]
            frames.append(sub)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    else:
        df = df.tz_convert("UTC")
    return df


def _fetch_yf(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    chunk_size: int = 20,
) -> pd.DataFrame:
    """Multi-symbol yfinance pull, chunked to dodge silent batch failures.

    Returns long-form df with a 'symbol' column.
    """
    if not symbols:
        return pd.DataFrame()
    frames = []
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        try:
            df = _fetch_yf_chunk(chunk, start, end, interval)
            if not df.empty:
                frames.append(df)
        except Exception as e:  # noqa: BLE001
            log.warning("yf chunk %s failed: %s", chunk, e)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


@register("yf_intraday")
class YfIntraday(DataSource):
    name: str = "yf_intraday"
    frequency: str = "5min"
    schema = {"open": "float64", "high": "float64", "low": "float64",
              "close": "float64", "volume": "float64", "symbol": "object"}
    publish_lag = pd.Timedelta(seconds=15)  # near-realtime; small lag

    def fetch(self, start, end, *, symbols: list[str] | None = None,
              interval: str = "5m", **kw: Any) -> pd.DataFrame:
        symbols = symbols or []
        return _fetch_yf(symbols, start, end, interval=interval)


@register("yf_daily")
class YfDaily(DataSource):
    name: str = "yf_daily"
    frequency: str = "1d"
    schema = {"open": "float64", "high": "float64", "low": "float64",
              "close": "float64", "adj_close": "float64",
              "volume": "float64", "symbol": "object"}
    publish_lag = pd.Timedelta(0)

    def fetch(self, start, end, *, symbols: list[str] | None = None, **kw: Any) -> pd.DataFrame:
        symbols = symbols or []
        df = _fetch_yf(symbols, start, end, interval="1d")
        # Normalise daily bars to 21:00 UTC (≈ NYSE close 16:00 ET in winter)
        # so they line up with the bar grid.
        if not df.empty:
            df.index = df.index.normalize() + pd.Timedelta(hours=21)
        return df

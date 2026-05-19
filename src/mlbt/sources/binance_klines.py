"""Crypto klines from Binance public spot API. No key required.

Binance returns up to 1000 candles per call; we paginate. Interval mirrors
Binance's notation: 1m, 5m, 15m, 1h, 4h, 1d.

Why crypto in an equity stack? Crypto trades 24x7 — overnight BTC/ETH moves
contain regime/risk-appetite signal observable BEFORE the US equity open.
"""
from __future__ import annotations

from typing import Any, List

import pandas as pd

from mlbt.core.base import DataSource
from mlbt.core.registry import register
from mlbt.core.http import http_get
from mlbt.core.log import get_logger

log = get_logger("binance")

_API = "https://api.binance.com/api/v3/klines"
_INTERVAL_MAP = {"1min": "1m", "5min": "5m", "15min": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}


def _fetch_pair(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows: list[list] = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = http_get(_API, params=params, min_interval=0.1, raise_on_error=False)
        if resp.status_code != 200:
            break
        chunk = resp.json()
        if not chunk:
            break
        rows.extend(chunk)
        last_close_time = chunk[-1][6]
        next_cursor = last_close_time + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(chunk) < 1000:
            break
    if not rows:
        return pd.DataFrame()
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("ts")
    keep = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]
    df = df[keep].astype(float)
    df["symbol"] = symbol
    return df


@register("binance_klines")
class BinanceKlines(DataSource):
    name: str = "binance_klines"
    frequency: str = "5min"
    schema = {"open": "float64", "high": "float64", "low": "float64",
              "close": "float64", "volume": "float64", "quote_volume": "float64",
              "trades": "float64", "taker_buy_base": "float64", "symbol": "object"}
    publish_lag = pd.Timedelta(seconds=5)

    def fetch(self, start, end, *, symbols: List[str] | None = None,
              interval: str = "5min", **kw: Any) -> pd.DataFrame:
        symbols = symbols or []
        b_interval = _INTERVAL_MAP.get(interval, "5m")
        out = []
        for s in symbols:
            try:
                df = _fetch_pair(s, b_interval, start, end)
                if not df.empty:
                    out.append(df)
            except Exception as e:  # noqa: BLE001
                log.warning("binance %s failed: %s", s, e)
        if not out:
            return pd.DataFrame()
        return pd.concat(out)

"""1-hour bars from Yahoo Finance.

Yahoo allows ~2 years of history at 1h granularity (vs 60 days at 5m).
This gives ~3300 bars per symbol — enough for short-horizon ML with
real signal-to-noise (5m bars are dominated by microstructure noise).

Separately registered from yf_intraday so both can coexist in storage.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from mlbt.core.base import DataSource
from mlbt.core.registry import register
from mlbt.sources.yf_bars import _fetch_yf


@register("yf_1h")
class Yf1h(DataSource):
    name: str = "yf_1h"
    frequency: str = "1h"
    schema = {"open": "float64", "high": "float64", "low": "float64",
              "close": "float64", "volume": "float64", "symbol": "object"}
    publish_lag = pd.Timedelta(seconds=15)

    def fetch(self, start, end, *, symbols: list[str] | None = None, **kw: Any) -> pd.DataFrame:
        symbols = symbols or []
        return _fetch_yf(symbols, start, end, interval="1h")

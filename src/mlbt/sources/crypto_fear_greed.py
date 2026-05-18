"""Crypto Fear & Greed Index from alternative.me.

Daily 0-100 index of crypto sentiment. Correlates with risk-on/risk-off in
broader markets. No key required.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from mlbt.core.base import DataSource
from mlbt.core.registry import register
from mlbt.core.http import http_get


@register("crypto_fear_greed")
class CryptoFearGreed(DataSource):
    name: str = "crypto_fear_greed"
    frequency: str = "1d"
    schema = {"fear_greed": "float64", "fear_greed_class": "object"}
    publish_lag = pd.Timedelta(hours=1)

    def fetch(self, start, end, **kw: Any) -> pd.DataFrame:
        url = "https://api.alternative.me/fng/"
        resp = http_get(url, params={"limit": 0, "format": "json"}, min_interval=1.0,
                        raise_on_error=False)
        if resp.status_code != 200:
            return pd.DataFrame()
        payload = resp.json().get("data", [])
        if not payload:
            return pd.DataFrame()
        rows = [{
            "ts": pd.to_datetime(int(r["timestamp"]), unit="s", utc=True),
            "fear_greed": float(r["value"]),
            "fear_greed_class": r["value_classification"],
        } for r in payload]
        df = pd.DataFrame(rows).set_index("ts").sort_index()
        return df.loc[(df.index >= pd.Timestamp(start, tz="UTC")) &
                      (df.index <= pd.Timestamp(end, tz="UTC"))]

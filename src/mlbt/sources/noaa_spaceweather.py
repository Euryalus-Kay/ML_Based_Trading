"""NOAA Space Weather: Kp planetary geomagnetic index.

Yes, really. Some quant studies find correlation between geomagnetic
storms and short-term equity returns (mood/attention proxy via biology).
Including it as a "seemingly unrelated" feature for the model to weigh.

Free, no key, public JSON.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from mlbt.core.base import DataSource
from mlbt.core.registry import register
from mlbt.core.http import http_get


@register("noaa_spaceweather")
class NoaaSpaceWeather(DataSource):
    name: str = "noaa_spaceweather"
    frequency: str = "3h"
    schema = {"kp_index": "float64"}
    publish_lag = pd.Timedelta(hours=1)

    def fetch(self, start, end, **kw: Any) -> pd.DataFrame:
        url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
        resp = http_get(url, min_interval=1.0, raise_on_error=False)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "time_tag" not in df.columns:
            return pd.DataFrame()
        df["ts"] = pd.to_datetime(df["time_tag"], utc=True)
        df = df.set_index("ts")
        kp_col = next((c for c in df.columns if "kp" in c.lower()), None)
        if not kp_col:
            return pd.DataFrame()
        df["kp_index"] = pd.to_numeric(df[kp_col], errors="coerce")
        out = df[["kp_index"]].dropna()
        from mlbt.core.base import to_utc
        return out.loc[(out.index >= to_utc(start)) & (out.index <= to_utc(end))]

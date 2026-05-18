"""US Treasury daily par yield curve.

Source: https://home.treasury.gov/resource-center/data-chart-center/interest-rates/
We use the XML view which doesn't require a key. Daily granularity, publish
lag ~end of business day.

Output columns: y_1mo, y_3mo, y_6mo, y_1y, y_2y, y_3y, y_5y, y_7y, y_10y,
y_20y, y_30y (all annualised %, e.g. 4.25).
"""
from __future__ import annotations

import io
from typing import Any

import pandas as pd

from mlbt.core.base import DataSource
from mlbt.core.registry import register
from mlbt.core.http import http_get
from mlbt.core.log import get_logger

log = get_logger("treasury")

_BASE = ("https://home.treasury.gov/resource-center/data-chart-center/"
         "interest-rates/daily-treasury-rates.csv/")

_COL_MAP = {
    "1 mo": "y_1mo", "2 mo": "y_2mo", "3 mo": "y_3mo",
    "4 mo": "y_4mo", "6 mo": "y_6mo", "1 yr": "y_1y",
    "2 yr": "y_2y", "3 yr": "y_3y", "5 yr": "y_5y",
    "7 yr": "y_7y", "10 yr": "y_10y", "20 yr": "y_20y", "30 yr": "y_30y",
}


def _year_url(year: int) -> str:
    return (f"https://home.treasury.gov/resource-center/data-chart-center/"
            f"interest-rates/daily-treasury-rates.csv/{year}/"
            f"all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv")


@register("treasury_yields")
class TreasuryYields(DataSource):
    name: str = "treasury_yields"
    frequency: str = "1d"
    schema = {v: "float64" for v in _COL_MAP.values()}
    publish_lag = pd.Timedelta(hours=18)  # published end-of-day ET, available next morning

    def fetch(self, start, end, **kw: Any) -> pd.DataFrame:
        years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
        frames = []
        for y in years:
            try:
                resp = http_get(_year_url(y), min_interval=0.5)
                if resp.status_code != 200 or not resp.text:
                    continue
                df = pd.read_csv(io.StringIO(resp.text))
                if df.empty:
                    continue
                df.columns = [c.strip().lower() for c in df.columns]
                if "date" not in df.columns:
                    continue
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
                df = df.set_index("date").tz_localize("UTC")
                # Normalise -> publish_at = date + 1 calendar day @ 06:00 UTC
                df.index = df.index.normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=6)
                rename = {c: _COL_MAP[c] for c in df.columns if c in _COL_MAP}
                df = df.rename(columns=rename)
                df = df[[c for c in df.columns if c.startswith("y_")]].apply(pd.to_numeric, errors="coerce")
                frames.append(df)
            except Exception as e:  # noqa: BLE001
                log.warning("treasury %s failed: %s", y, e)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames).sort_index()
        from mlbt.core.base import to_utc
        return out.loc[(out.index >= to_utc(start)) & (out.index <= to_utc(end))]

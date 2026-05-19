"""Time-of-day features for intraday bars.

Intraday equity returns have very strong diurnal patterns:
  - Opening 30 min: high vol, gap-fill, news digestion
  - Lunch (~12-13 ET): low vol, low information
  - Last hour: closing flows, MOC imbalance, rebalancing
  - Friday afternoons / Monday opens different from mid-week

Encoded as cyclical (sin/cos) plus several binary zone flags so trees can
split on them directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_of_day_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx_utc = df.index
    if idx_utc.tz is None:
        idx_utc = idx_utc.tz_localize("UTC")
    et = idx_utc.tz_convert("America/New_York")

    out = df.copy()
    minutes_since_open = ((et.hour - 9) * 60 + (et.minute - 30)).astype(float)
    minutes_since_open = np.where((et.hour < 9) | (et.hour >= 16), np.nan, minutes_since_open)

    out["tod_minutes_since_open"] = minutes_since_open
    out["tod_minutes_to_close"] = 390.0 - minutes_since_open
    out["tod_hour"] = et.hour.astype(float)
    out["tod_minute"] = et.minute.astype(float)
    out["tod_dow"] = et.dayofweek.astype(float)

    # Cyclical encodings for the model to pick up smoothly
    day_radians = 2 * np.pi * minutes_since_open / 390.0
    out["tod_sin"] = np.sin(day_radians)
    out["tod_cos"] = np.cos(day_radians)
    week_radians = 2 * np.pi * et.dayofweek.values / 5.0
    out["tod_week_sin"] = np.sin(week_radians)
    out["tod_week_cos"] = np.cos(week_radians)

    # Hard zones
    out["tod_is_open_30"] = ((minutes_since_open >= 0) & (minutes_since_open <= 30)).astype(float)
    out["tod_is_open_60"] = ((minutes_since_open > 30) & (minutes_since_open <= 90)).astype(float)
    out["tod_is_lunch"] = ((minutes_since_open >= 120) & (minutes_since_open < 210)).astype(float)
    out["tod_is_last_60"] = (minutes_since_open >= 330).astype(float)
    out["tod_is_last_15"] = (minutes_since_open >= 375).astype(float)
    out["tod_is_monday"] = (et.dayofweek == 0).astype(float)
    out["tod_is_friday"] = (et.dayofweek == 4).astype(float)
    return out

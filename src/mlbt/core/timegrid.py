"""Time grid + as-of merge utilities.

The collector pulls heterogeneous sources at heterogeneous frequencies:
  - 1m equities bars
  - daily macro prints
  - irregular news / filing events
  - weekly sentiment surveys

The aligner builds a canonical bar grid (e.g. 5min RTH on NYSE) and
joins each series via `merge_asof`, shifting every series forward by its
`publish_lag` so model features are strictly observable at the bar
timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

try:
    import exchange_calendars as xcals
    _HAS_XCALS = True
except ImportError:  # pragma: no cover
    _HAS_XCALS = False


SESSION_PRESETS = {
    "rth": ("XNYS", "09:31", "16:00"),     # NYSE regular trading hours
    "full": ("XNYS", "04:00", "20:00"),    # NYSE extended (premkt + after)
    "24x7": (None, None, None),
}


@dataclass
class TimeGrid:
    """Generates the canonical bar index for a date range."""
    bar: str = "5min"
    session: str = "rth"
    tz: str = "UTC"

    def index(self, start, end) -> pd.DatetimeIndex:
        start = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tz is None else pd.Timestamp(start).tz_convert("UTC")
        end = pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tz is None else pd.Timestamp(end).tz_convert("UTC")

        if self.session == "24x7" or not _HAS_XCALS:
            idx = pd.date_range(start, end, freq=self.bar, tz="UTC")
            return idx

        cal_name, open_t, close_t = SESSION_PRESETS.get(self.session, SESSION_PRESETS["rth"])
        cal = xcals.get_calendar(cal_name)
        # session_minutes returns minute-by-minute index in the calendar's tz
        try:
            schedule = cal.schedule.loc[start.date():end.date()]
        except KeyError:
            return pd.DatetimeIndex([])
        if schedule.empty:
            return pd.DatetimeIndex([])

        parts: list[pd.DatetimeIndex] = []
        for _, row in schedule.iterrows():
            o, c = row["open"], row["close"]
            if pd.isna(o) or pd.isna(c):
                continue
            o = pd.Timestamp(o).tz_convert("UTC")
            c = pd.Timestamp(c).tz_convert("UTC")
            parts.append(pd.date_range(o, c, freq=self.bar, tz="UTC"))
        if not parts:
            return pd.DatetimeIndex([])
        return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]


def asof_merge(
    grid: pd.DatetimeIndex,
    df: pd.DataFrame,
    publish_lag: pd.Timedelta = pd.Timedelta(0),
    tolerance: Optional[pd.Timedelta] = None,
    suffix: str = "",
) -> pd.DataFrame:
    """Backward as-of merge `df` onto `grid`, shifting df forward by publish_lag.

    Result has index = grid; columns = df.columns (optionally + suffix). Any bar
    that has no preceding observation within `tolerance` is NaN.
    """
    if df is None or df.empty or len(grid) == 0:
        return pd.DataFrame(index=grid)
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    else:
        df = df.tz_convert("UTC")
    df = df.sort_index()
    shifted = df.copy()
    shifted.index = shifted.index + publish_lag

    # merge_asof requires matching datetime resolution on both sides.
    target_unit = "ns"
    grid_norm = pd.DatetimeIndex(grid).astype(f"datetime64[{target_unit}, UTC]")
    shifted.index = shifted.index.astype(f"datetime64[{target_unit}, UTC]")

    grid_df = pd.DataFrame(index=grid_norm).reset_index().rename(columns={"index": "ts"})
    shifted_df = shifted.reset_index().rename(columns={shifted.index.name or "index": "ts"})

    merged = pd.merge_asof(
        grid_df.sort_values("ts"),
        shifted_df.sort_values("ts"),
        on="ts",
        direction="backward",
        tolerance=tolerance,
    )
    merged = merged.set_index("ts")
    if suffix:
        merged = merged.rename(columns={c: f"{c}{suffix}" for c in merged.columns})
    merged.index.name = None
    return merged


def resample_to_bar(df: pd.DataFrame, bar: str, ohlc: bool = False) -> pd.DataFrame:
    """Resample a higher-freq series to a lower-freq bar.

    If `ohlc` and the df has OHLCV columns, use proper aggregations; otherwise
    forward-fill last observation.
    """
    if df.empty:
        return df
    if ohlc and {"open", "high", "low", "close"}.issubset(set(c.lower() for c in df.columns)):
        # case-insensitive column resolution
        cols = {c.lower(): c for c in df.columns}
        agg = {cols["open"]: "first", cols["high"]: "max",
               cols["low"]: "min", cols["close"]: "last"}
        if "volume" in cols:
            agg[cols["volume"]] = "sum"
        return df.resample(bar).agg(agg).dropna(how="all")
    return df.resample(bar).last().ffill()

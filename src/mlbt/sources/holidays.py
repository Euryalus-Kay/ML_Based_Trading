"""Trading-calendar event features.

Builds a dataframe indexed at midnight UTC each day with:
  - is_us_holiday
  - days_to_next_us_holiday
  - is_fomc_day (best-effort hard-coded list, updated annually)
  - is_cpi_day  (US BLS releases — published monthly ~mid-month 08:30 ET)
  - day_of_week, month, quarter (numeric, for the model to pick up cyclicals)
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from mlbt.core.base import DataSource
from mlbt.core.registry import register

try:
    import exchange_calendars as xcals
    _XCALS = True
except ImportError:  # pragma: no cover
    _XCALS = False


# FOMC scheduled meeting dates (US). Maintained list — extend yearly.
# Source: federalreserve.gov/monetarypolicy/fomccalendars.htm
_FOMC_DATES = [
    "2020-01-29", "2020-03-15", "2020-04-29", "2020-06-10", "2020-07-29",
    "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-10-28", "2026-12-09",
]


@register("calendar_events")
class CalendarEvents(DataSource):
    name: str = "calendar_events"
    frequency: str = "1d"
    schema = {
        "is_us_holiday": "float64", "days_to_next_us_holiday": "float64",
        "is_fomc_day": "float64", "is_cpi_day": "float64",
        "dow": "float64", "month": "float64", "quarter": "float64",
        "is_month_end": "float64", "is_quarter_end": "float64",
        "is_eom_week": "float64",
    }
    publish_lag = pd.Timedelta(0)  # calendar known in advance

    def fetch(self, start, end, **kw: Any) -> pd.DataFrame:
        start_ts = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tz is None else pd.Timestamp(start)
        end_ts = pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tz is None else pd.Timestamp(end)
        idx = pd.date_range(start_ts.normalize(), end_ts.normalize(), freq="1D", tz="UTC")
        df = pd.DataFrame(index=idx)
        df["dow"] = df.index.dayofweek
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["is_month_end"] = df.index.is_month_end.astype(float)
        df["is_quarter_end"] = df.index.is_quarter_end.astype(float)
        df["is_eom_week"] = ((df.index.day >= 24) | (df.index.day <= 3)).astype(float)

        fomc_dates = pd.to_datetime(_FOMC_DATES, utc=True).normalize()
        df["is_fomc_day"] = df.index.isin(fomc_dates).astype(float)

        # CPI release dates (BLS publishes ~ 2nd or 3rd Tue/Wed of each month).
        # Approximation: 10th calendar day. Good enough as a feature.
        df["is_cpi_day"] = (df.index.day.isin([10, 11, 12, 13, 14])).astype(float)

        if _XCALS:
            cal = xcals.get_calendar("XNYS")
            holidays = pd.DatetimeIndex(cal.adhoc_holidays + list(cal.regular_holidays.holidays())).tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
            df["is_us_holiday"] = df.index.isin(holidays).astype(float)
        else:
            df["is_us_holiday"] = 0.0

        # days_to_next_us_holiday
        if df["is_us_holiday"].any():
            holiday_locs = df.index[df["is_us_holiday"] == 1.0]
            next_idx = []
            ptr = 0
            for ts in df.index:
                while ptr < len(holiday_locs) and holiday_locs[ptr] < ts:
                    ptr += 1
                if ptr >= len(holiday_locs):
                    next_idx.append(pd.NaT)
                else:
                    next_idx.append(holiday_locs[ptr])
            next_idx = pd.DatetimeIndex(next_idx)
            df["days_to_next_us_holiday"] = (next_idx - df.index).days.astype(float)
        else:
            df["days_to_next_us_holiday"] = 30.0

        df.index = df.index + pd.Timedelta(hours=5)  # publish before US open
        return df

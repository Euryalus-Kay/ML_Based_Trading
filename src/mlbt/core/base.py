"""Base class every data source implements.

A DataSource turns one upstream API/feed into a tidy time-indexed DataFrame.
The aligner takes any number of these and joins them to a common time grid
using as-of merges, respecting each source's `publish_lag` so a backtest
never peeks at data that wasn't yet observable.

Subclasses declare class-level attributes (name/frequency/schema/publish_lag)
and override `fetch()`. We deliberately don't use @dataclass on the base
because dataclass fields shadow class attributes the subclass sets.
"""
from __future__ import annotations

import abc
import datetime as dt
from typing import Any, Dict, List, Optional

import pandas as pd


class DataSource(abc.ABC):
    """Abstract base for every data source.

    Subclasses set:
      - name: str
      - frequency: str (pandas offset alias)
      - schema: dict (col -> dtype)
      - publish_lag: pd.Timedelta
      - requires_key: optional env-var name
    and implement `fetch(start, end, **kwargs)`.
    """
    name: str = ""
    frequency: str = "1d"
    schema: Dict[str, str] = {}
    publish_lag: pd.Timedelta = pd.Timedelta(0)
    requires_key: Optional[str] = None
    enabled: bool = True
    universe_keys: List[str] = []

    def __init__(self) -> None:
        # Make sure subclass class attrs are visible as instance attrs.
        # Important for dynamic config (e.g. tests that mutate publish_lag).
        pass

    @abc.abstractmethod
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp, **kwargs: Any) -> pd.DataFrame:
        ...

    # --- standard helpers -----------------------------------------------------
    def is_available(self) -> bool:
        """False if a required key is missing — caller should skip silently."""
        if not self.enabled:
            return False
        if self.requires_key is None:
            return True
        from mlbt.core.secrets import has_key
        return has_key(self.requires_key)

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure tz-aware UTC DatetimeIndex and sorted.

        Dedup *only* when the df has no symbol/id column — otherwise long-form
        data with repeated timestamps across symbols collapses to a single row.
        """
        if df is None or df.empty:
            return pd.DataFrame()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"{type(df.index).__name__} index; expected DatetimeIndex")
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        else:
            df = df.tz_convert("UTC")
        df = df.sort_index()
        id_cols = [c for c in ("symbol", "id", "series_id", "ticker") if c in df.columns]
        if id_cols:
            idx_name = df.index.name or "_ts"
            tmp = df.reset_index().rename(columns={df.index.name or "index": idx_name})
            tmp = tmp.drop_duplicates(subset=[idx_name] + id_cols, keep="last")
            df = tmp.set_index(idx_name)
            df.index.name = None
        else:
            df = df[~df.index.duplicated(keep="last")]
        return df

    def fetch_safe(self, start, end, **kw) -> pd.DataFrame:
        """Catches exceptions and returns empty df so a single bad source can't
        nuke the whole collection run. Logs once."""
        from mlbt.core.log import get_logger
        log = get_logger(f"src.{self.name}")
        if not self.is_available():
            log.info("skipping %s (key %s not set)", self.name, self.requires_key)
            return pd.DataFrame()
        try:
            df = self.fetch(pd.Timestamp(start), pd.Timestamp(end), **kw)
            df = self._normalize(df)
            log.info("%s -> %d rows, %d cols", self.name, len(df), df.shape[1] if len(df) else 0)
            return df
        except Exception as e:  # noqa: BLE001
            log.warning("%s fetch failed: %s", self.name, e)
            return pd.DataFrame()


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.now(tz=dt.timezone.utc))

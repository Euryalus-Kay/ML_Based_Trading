"""Align all collected raw frames onto a single canonical bar grid.

Output: one wide dataframe indexed by the bar timestamp, columns prefixed
by their source/symbol, e.g.
   close_AAPL, volume_AAPL, close_^GSPC, vix_close, y_10y, fear_greed, ...

Each series is as-of joined with its publish_lag respected.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from mlbt.core.log import get_logger
from mlbt.core.registry import all_sources, get_source
from mlbt.core.storage import Storage
from mlbt.core.timegrid import TimeGrid, asof_merge

log = get_logger("align")


def build_aligned_frame(start, end, *,
                         bar: str = "5min",
                         session: str = "rth",
                         storage: Optional[Storage] = None) -> pd.DataFrame:
    storage = storage or Storage()
    grid = TimeGrid(bar=bar, session=session).index(start, end)
    if len(grid) == 0:
        log.warning("empty time grid for %s..%s", start, end)
        return pd.DataFrame()

    wide = pd.DataFrame(index=grid)
    wide.index.name = "ts"
    sources = all_sources()

    for src in sources:
        keys = list(storage.list_keys(src.name))
        if not keys:
            continue
        for key in keys:
            df = storage.read(src.name, key)
            if df.empty:
                continue
            df = df.select_dtypes(include=["number"])
            if df.empty:
                continue
            suffix = f"_{key}" if key != "_default" else ""
            joined = asof_merge(grid, df, publish_lag=src.publish_lag, suffix=suffix)
            wide = wide.join(joined, how="left")

    return wide

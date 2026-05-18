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
                         storage: Optional[Storage] = None,
                         only_sources: Optional[list[str]] = None) -> pd.DataFrame:
    """Build the aligned wide frame.

    If `only_sources` is provided, restrict to those source names — useful
    when building separate datasets at different bar frequencies (e.g.
    daily dataset reads only yf_daily, not yf_intraday).
    """
    storage = storage or Storage()
    grid = TimeGrid(bar=bar, session=session).index(start, end)
    if len(grid) == 0:
        log.warning("empty time grid for %s..%s", start, end)
        return pd.DataFrame()

    wide = pd.DataFrame(index=grid)
    wide.index.name = "ts"
    sources = all_sources()
    if only_sources:
        whitelist = set(only_sources)
        sources = [s for s in sources if s.name in whitelist]

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
            # Defensive: if the joined frame has columns that collide with the
            # accumulator, rename them with a source-name prefix so we never
            # raise from pandas.join.
            collisions = [c for c in joined.columns if c in wide.columns]
            if collisions:
                joined = joined.rename(
                    columns={c: f"{c}__{src.name}" for c in collisions})
            wide = wide.join(joined, how="left")

    return wide

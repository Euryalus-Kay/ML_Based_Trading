"""Build per-symbol feature+target dataset(s) ready for ML.

For each target symbol:
  - load the aligned wide frame
  - extract the symbol's OHLCV columns
  - run technical + microstructure features on it
  - run cross-asset features against the rest of the wide frame
  - generate forward-return targets
  - drop NaN rows (after warmup)

The result is saved per-symbol and concatenated into a single
long-form dataset with a 'symbol' column for multi-task training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from mlbt.core.log import get_logger
from mlbt.core.storage import Storage
from mlbt.features import (
    add_technical_features,
    add_cross_asset_features,
    add_microstructure_features,
    add_targets,
    add_time_of_day_features,
)
from mlbt.features.targets import add_residualised_targets  # noqa: F401  (defined below)
from mlbt.pipeline.align import build_aligned_frame
from mlbt.pipeline.collect import load_universe

log = get_logger("dataset")


def _symbol_ohlcv(wide: pd.DataFrame, symbol: str) -> pd.DataFrame:
    cols = ["open", "high", "low", "close", "volume"]
    have = {}
    for c in cols:
        col_name = f"{c}_{symbol}"
        if col_name in wide.columns:
            have[c] = col_name
    if "close" not in have:
        return pd.DataFrame()
    out = wide[list(have.values())].copy()
    out.columns = list(have.keys())
    return out


def build_dataset(start, end, *,
                   bar: str = "5min", session: str = "rth",
                   universe_path: Optional[str] = None,
                   horizons=(1, 3, 6, 12),
                   out_path: Optional[str] = None,
                   storage: Optional[Storage] = None,
                   only_sources: Optional[list[str]] = None) -> pd.DataFrame:
    storage = storage or Storage()
    universe = load_universe(universe_path)
    target_symbols: list[str] = []
    for group in universe.get("equities", {}).values():
        target_symbols.extend(group)
    target_symbols = sorted(set(target_symbols))

    wide = build_aligned_frame(start, end, bar=bar, session=session,
                                 storage=storage, only_sources=only_sources)
    if wide.empty:
        log.warning("aligned frame is empty; collect data first")
        return pd.DataFrame()

    market_df = wide  # keep full frame for cross-asset features
    frames = []
    for sym in target_symbols:
        bars = _symbol_ohlcv(wide, sym)
        if bars.empty:
            continue
        log.info("featurising %s (%d bars)", sym, len(bars))
        bars = add_technical_features(bars)
        bars = add_microstructure_features(bars)
        bars = add_time_of_day_features(bars)
        bars = add_cross_asset_features(bars, market_df, sym)
        bars = add_targets(bars, horizons=horizons)
        bars = add_residualised_targets(bars, market_df, horizons=horizons)
        bars["symbol"] = sym
        # join scalar/index features (FRED, calendar, sentiment) that aren't symbol-specific
        scalar_cols = [c for c in wide.columns if not any(
            c.endswith(f"_{s}") for s in target_symbols)]
        # Avoid duplicate columns
        scalar_cols = [c for c in scalar_cols if c not in bars.columns]
        if scalar_cols:
            bars = bars.join(wide[scalar_cols], how="left")
        bars = bars.dropna(how="all")
        frames.append(bars)

    if not frames:
        log.warning("no per-symbol frames built")
        return pd.DataFrame()

    dataset = pd.concat(frames)
    dataset = dataset.sort_index()
    # Drop rows where the target horizon is unobservable (NaN of largest horizon)
    largest_h = max(horizons)
    target_col = f"y_logret_{largest_h}"
    if target_col in dataset.columns:
        dataset = dataset.dropna(subset=[target_col])

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(out_path, engine="pyarrow")
        log.info("wrote %s (%d rows, %d cols)", out_path, len(dataset), dataset.shape[1])
    return dataset

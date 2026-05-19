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
from mlbt.features.targets import (
    add_residualised_targets, add_xsec_rank_targets, add_vol_scaled_targets,
)
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

    # Enrich the wide frame with derived/regime features BEFORE per-symbol
    # featurisation, so scalar regime features (dispersion, vol regime,
    # risk-on factor) propagate into every symbol's training rows.
    from mlbt.features.regime import enrich_with_derived
    wide = enrich_with_derived(wide, target_symbols)

    market_df = wide  # keep full frame for cross-asset features
    # Pre-compute the truly scalar columns (those NOT ending in _{symbol} for ANY symbol).
    # Without this fix every per-symbol frame inherited 565 * len(windows) xsmom cols
    # because xsmom_AAPL_5 doesn't end in "_AAPL" (it ends in "_5"). We now filter
    # so each per-symbol frame keeps only ITS OWN xsmom_{sym}_w columns plus the
    # universe scalars (FRED, regime, dispersion, etc).
    sym_suffixes = set(target_symbols)
    truly_scalar = []
    per_symbol_owned: dict[str, list[str]] = {s: [] for s in target_symbols}
    for c in wide.columns:
        # Treat xsmom_{sym}_w specifically — keep only the relevant sym's column
        if c.startswith("xsmom_"):
            try:
                _, sym_part, _ = c.split("_", 2)
            except ValueError:
                truly_scalar.append(c)
                continue
            if sym_part in sym_suffixes:
                per_symbol_owned.setdefault(sym_part, []).append(c)
            else:
                truly_scalar.append(c)
            continue
        # Generic suffix _{symbol}: belongs to that symbol (close_AAPL, volume_AAPL ...)
        matched = False
        for s in sym_suffixes:
            if c.endswith(f"_{s}"):
                matched = True
                break
        if not matched:
            truly_scalar.append(c)

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
        bars = add_vol_scaled_targets(bars, horizons=horizons, threshold_mult=0.5)
        bars["symbol"] = sym

        # Truly scalar features (FRED, regime, dispersion) — same for every symbol
        scalar_cols = [c for c in truly_scalar if c not in bars.columns]
        if scalar_cols:
            bars = bars.join(wide[scalar_cols], how="left")
        # This symbol's own xsmom columns (xsmom_{sym}_5, xsmom_{sym}_20, ...)
        own_xsmom = [c for c in per_symbol_owned.get(sym, []) if c not in bars.columns]
        if own_xsmom:
            bars = bars.join(wide[own_xsmom], how="left")
            # Rename to drop the sym so the column name is the same across symbols
            bars = bars.rename(columns={c: c.replace(f"_{sym}_", "_self_")
                                          for c in own_xsmom})
        bars = bars.dropna(how="all")
        frames.append(bars)

    if not frames:
        log.warning("no per-symbol frames built")
        return pd.DataFrame()

    dataset = pd.concat(frames)
    dataset = dataset.sort_index()
    # Add cross-sectional rank targets (must be done after concat so we can
    # rank ACROSS symbols at each timestamp). Uses y_resid_logret_h as source.
    dataset = add_xsec_rank_targets(dataset, horizons=horizons)
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

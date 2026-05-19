"""Derived "non-obvious" features.

Public OHLCV is priced in. What's NOT priced in (because it requires combining
multiple sources or computing cross-sectional / regime-conditional transforms):

  - Cross-sectional dispersion: when the universe moves in lockstep (low
    dispersion) vs spreads out (high dispersion). High dispersion = stock-
    picker's market; low dispersion = factor-driven.
  - Realised-vol regime: 20-bar realised vol relative to its 1-year median.
    Strategies that work in low-vol regimes break in high-vol.
  - Cross-sectional momentum: each symbol's rolling return minus the
    universe-wide average rolling return. Pure relative momentum.
  - VIX percentile rank: where is current VIX in its trailing 1y distribution?
  - Yield-curve regime: 10y-3m spread relative to its history (recession proxy).
  - Risk-on/off composite: PCA of {VIX, HYG/LQD, gold, USD} into a single
    factor.
  - Lead-lag flag: when does the symbol's return correlate with its own lag
    or its market's lag? Captures local mean-reversion vs momentum regime.
  - Time-since-event: minutes since last FOMC / CPI / OPEX day, decayed.

These are computed at the WIDE-FRAME level (across all symbols at once) and
broadcast back into per-symbol features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlbt.core.log import get_logger

log = get_logger("regime")


def add_dispersion_features(wide: pd.DataFrame, target_symbols: list[str]) -> pd.DataFrame:
    """Compute cross-sectional dispersion of returns across the universe.

    Adds two scalar columns to `wide`:
      xs_dispersion_5    : stdev of 5-bar returns across symbols
      xs_dispersion_20   : stdev of 20-bar returns
      xs_dispersion_5_z  : rolling z-score of xs_dispersion_5 over 60 bars
    """
    out = wide.copy()
    closes = []
    for s in target_symbols:
        c = f"close_{s}"
        if c in out.columns:
            closes.append(out[c].rename(s))
    if not closes:
        return out
    px = pd.concat(closes, axis=1)
    for w in (5, 20):
        rets = np.log(px / px.shift(w))
        out[f"xs_dispersion_{w}"] = rets.std(axis=1)
    z_input = out["xs_dispersion_5"]
    out["xs_dispersion_5_z"] = (
        (z_input - z_input.rolling(60, min_periods=10).mean()) /
        z_input.rolling(60, min_periods=10).std().replace(0, np.nan)
    )
    return out


def add_xsec_momentum(wide: pd.DataFrame, target_symbols: list[str],
                       windows=(5, 20, 60)) -> pd.DataFrame:
    """Per-symbol momentum, demeaned by universe-wide momentum at the same bar.

    Adds xsmom_{sym}_{w} = symbol_w_return - universe_w_return
    """
    out = wide.copy()
    closes = []
    for s in target_symbols:
        c = f"close_{s}"
        if c in out.columns:
            closes.append(out[c].rename(s))
    if not closes:
        return out
    px = pd.concat(closes, axis=1)
    for w in windows:
        rets = np.log(px / px.shift(w))
        univ_mean = rets.mean(axis=1)
        for s in rets.columns:
            out[f"xsmom_{s}_{w}"] = rets[s] - univ_mean
    return out


def add_vol_regime_features(wide: pd.DataFrame) -> pd.DataFrame:
    """VIX percentile rank in trailing 1y window, plus VIX term-structure.

    Looks for any VIX column ('vix' or 'close_^VIX' / 'close__VIX').
    """
    out = wide.copy()
    vix_col = None
    for c in ("close_^VIX", "close__VIX", "vix"):
        if c in out.columns:
            vix_col = c
            break
    if vix_col is None:
        return out
    vix = out[vix_col].astype(float)
    win = 252  # ~1y of business days; with intraday bars this is a shorter rolling
    out["vix_pct_rank_1y"] = vix.rolling(win, min_periods=20).rank(pct=True)
    out["vix_z_1y"] = (vix - vix.rolling(win, min_periods=20).mean()) / vix.rolling(win, min_periods=20).std().replace(0, np.nan)
    out["vix_change_5"] = vix.pct_change(5)
    out["vix_above_20"] = (vix > 20).astype(float)
    out["vix_above_30"] = (vix > 30).astype(float)
    return out


def add_yield_curve_regime(wide: pd.DataFrame) -> pd.DataFrame:
    """Yield-curve inversion + slope features. Looks for the FRED columns."""
    out = wide.copy()
    if "DGS10" in out.columns and "DGS3MO" in out.columns:
        spread = out["DGS10"] - out["DGS3MO"]
        out["yc_10y_3m_spread"] = spread
        out["yc_inverted"] = (spread < 0).astype(float)
        win = 252
        out["yc_spread_z_1y"] = (
            (spread - spread.rolling(win, min_periods=20).mean()) /
            spread.rolling(win, min_periods=20).std().replace(0, np.nan)
        )
    if "DGS10" in out.columns and "DGS2" in out.columns:
        out["yc_10y_2y_spread"] = out["DGS10"] - out["DGS2"]
    return out


def add_risk_on_off_composite(wide: pd.DataFrame) -> pd.DataFrame:
    """Build a single 'risk_on' factor by combining 5-bar changes in:
       - SPY (or ^GSPC) up
       - VIX down
       - HYG/LQD ratio up
       - Gold (GLD) down
       - Dollar (UUP) down (for risk-on)
    Each component standardised by its rolling vol and averaged.
    """
    out = wide.copy()
    components = []
    def _add(col_candidates, sign):
        for c in col_candidates:
            if c in out.columns:
                r = np.log(out[c] / out[c].shift(5))
                z = (r - r.rolling(60, min_periods=20).mean()) / r.rolling(60, min_periods=20).std().replace(0, np.nan)
                components.append(sign * z)
                return
    _add(["close_SPY", "close__GSPC", "close_^GSPC"], +1)
    _add(["close_^VIX", "close__VIX"], -1)
    _add(["close_HYG"], +1)
    _add(["close_LQD"], -1)
    _add(["close_GLD"], -1)
    _add(["close_UUP"], -1)
    if components:
        out["risk_on_5b"] = pd.concat(components, axis=1).mean(axis=1)
    return out


def enrich_with_derived(wide: pd.DataFrame, target_symbols: list[str]) -> pd.DataFrame:
    """One-call entrypoint that runs all the regime / derived feature blocks."""
    out = wide
    out = add_dispersion_features(out, target_symbols)
    out = add_xsec_momentum(out, target_symbols)
    out = add_vol_regime_features(out)
    out = add_yield_curve_regime(out)
    out = add_risk_on_off_composite(out)
    return out

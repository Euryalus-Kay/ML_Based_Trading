"""Cross-asset features.

Given the full aligned wide frame (multi-symbol), compute features that link
a single target symbol to the rest of the universe:
  - beta to ^GSPC, ^NDX rolling windows
  - correlation to sector ETF
  - VIX level + delta
  - rates: T10Y2Y, ^TNX delta
  - dollar (UUP) delta
  - HYG / LQD ratio (credit risk-on)
  - crypto (BTC) overnight return (intraday signal pre-open)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlbt.core.log import get_logger

log = get_logger("cross_asset")


def _ret(series: pd.Series, w: int = 1) -> pd.Series:
    return series.pct_change(w)


def add_cross_asset_features(symbol_df: pd.DataFrame, market_df: pd.DataFrame,
                              symbol: str) -> pd.DataFrame:
    """symbol_df: bar frame for one ticker with column 'close'.
    market_df: wide frame on the same time grid with columns like:
       close_^GSPC, close_^VIX, close_HYG, close_LQD, close_UUP, close_BTC-USD,
       y_10y, y_2y, etc.
    """
    out = symbol_df.copy()
    if market_df.empty:
        return out

    # Align by index intersection
    common = out.index.intersection(market_df.index)
    if len(common) == 0:
        return out

    s_close = out.loc[common, "close"].astype(float)
    s_ret = s_close.pct_change()

    def _col(col_name: str) -> pd.Series:
        if col_name in market_df.columns:
            return market_df.loc[common, col_name].astype(float)
        return pd.Series(np.nan, index=common)

    # SP500 beta + corr
    spx = _col("close_^GSPC")
    if spx.notna().any():
        spx_ret = spx.pct_change()
        for w in (20, 60, 240):
            cov = s_ret.rolling(w).cov(spx_ret)
            var = spx_ret.rolling(w).var()
            out.loc[common, f"beta_spx_{w}"] = (cov / var.replace(0, np.nan)).values
            out.loc[common, f"corr_spx_{w}"] = s_ret.rolling(w).corr(spx_ret).values

    # NDX beta
    ndx = _col("close_^NDX")
    if ndx.notna().any():
        ndx_ret = ndx.pct_change()
        for w in (20, 60):
            cov = s_ret.rolling(w).cov(ndx_ret)
            var = ndx_ret.rolling(w).var()
            out.loc[common, f"beta_ndx_{w}"] = (cov / var.replace(0, np.nan)).values

    # VIX level / change
    vix = _col("close_^VIX")
    if vix.notna().any():
        out.loc[common, "vix"] = vix.values
        out.loc[common, "vix_d1"] = vix.diff().values
        out.loc[common, "vix_d5"] = vix.diff(5).values
        out.loc[common, "vix_z_20"] = ((vix - vix.rolling(20).mean()) /
                                        vix.rolling(20).std().replace(0, np.nan)).values

    # Credit spreads via ETFs
    hyg = _col("close_HYG")
    lqd = _col("close_LQD")
    if hyg.notna().any() and lqd.notna().any():
        ratio = hyg / lqd
        out.loc[common, "hyg_lqd_ratio"] = ratio.values
        out.loc[common, "hyg_lqd_ratio_d5"] = ratio.diff(5).values

    # Dollar
    uup = _col("close_UUP")
    if uup.notna().any():
        out.loc[common, "uup_ret_5"] = uup.pct_change(5).values

    # Rates
    y10 = _col("y_10y")
    y2 = _col("y_2y")
    if y10.notna().any() and y2.notna().any():
        out.loc[common, "y10_y2"] = (y10 - y2).values
    if y10.notna().any():
        out.loc[common, "y10_d1"] = y10.diff().values
        out.loc[common, "y10_d5"] = y10.diff(5).values

    # Crypto overnight (use BTC close)
    btc = _col("close_BTCUSDT") if "close_BTCUSDT" in market_df.columns else _col("close_BTC-USD")
    if btc.notna().any():
        btc_ret = btc.pct_change()
        out.loc[common, "btc_ret"] = btc_ret.values
        out.loc[common, "btc_ret_60"] = btc.pct_change(60).values

    return out

"""Label generators for short-horizon ML.

We expose:
  - regression target:  y_logret_h  = log(close_{t+h} / close_t)
  - binary target:      y_up_h      = 1 if y_logret_h > threshold
  - triple-barrier:     y_tb_h      ∈ {-1,0,+1} (López de Prado)

The trick is shifting AHEAD: y_t is computed from data AFTER t, so during
training we drop rows where the future is unobservable. The aligner already
ensures features at t use only data observed by t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_vol_scaled_targets(df: pd.DataFrame, horizons=(1, 3, 6, 12),
                            vol_window: int = 60,
                            threshold_mult: float = 0.5) -> pd.DataFrame:
    """Vol-scaled labels: only label up/down if the move exceeds k * sigma.

    Reduces label noise: a 1bp move on a stock with 50bp vol gets discarded
    rather than being labelled as "up". Discarded rows train models on
    informative samples only. Returns extra cols y_vol_up_h ∈ {0,1,NaN}.
    """
    out = df.copy()
    if "close" not in out:
        return out
    c = out["close"].astype(float)
    log_ret = np.log(c / c.shift(1))
    realised_vol = log_ret.rolling(vol_window, min_periods=max(5, vol_window // 2)).std()
    for h in horizons:
        fwd = np.log(c.shift(-h) / c)
        threshold = realised_vol * threshold_mult * np.sqrt(h)
        label = pd.Series(np.nan, index=c.index, dtype="float64")
        label[fwd > threshold] = 1.0
        label[fwd < -threshold] = 0.0
        out[f"y_vol_up_{h}"] = label
    return out


def add_residualised_targets(df: pd.DataFrame, market_df: pd.DataFrame,
                              horizons=(1, 3, 6, 12),
                              market_col_candidates=("close_^GSPC", "close_SPY", "close_^NDX")) -> pd.DataFrame:
    """Add residualised targets: y_resid_logret_h = symbol_logret - mkt_logret.

    Removes the dominant common market factor. A model trained on this
    learns relative alpha rather than re-discovering "stocks usually go up".
    """
    out = df.copy()
    if market_df is None or market_df.empty or "close" not in out:
        return out
    mkt_col = next((c for c in market_col_candidates if c in market_df.columns), None)
    if mkt_col is None:
        return out
    mkt = market_df[mkt_col].astype(float).reindex(out.index, method="ffill")
    c = out["close"].astype(float)
    for h in horizons:
        sym_fwd = np.log(c.shift(-h) / c)
        mkt_fwd = np.log(mkt.shift(-h) / mkt)
        resid = sym_fwd - mkt_fwd
        out[f"y_resid_logret_{h}"] = resid
        out[f"y_resid_up_{h}"] = (resid > 0).astype("float64").mask(resid.isna())
    return out


def add_targets(df: pd.DataFrame, horizons=(1, 3, 6, 12),
                up_threshold: float = 0.0,
                vol_window: int = 60,
                tb_mult: float = 1.0) -> pd.DataFrame:
    """horizons in bars (not minutes). For 5-min bars, horizon=12 => 60 min."""
    out = df.copy()
    if "close" not in out:
        return out
    c = out["close"].astype(float)
    log_ret = np.log(c / c.shift(1))
    realised_vol = log_ret.rolling(vol_window, min_periods=max(5, vol_window // 2)).std()

    for h in horizons:
        fwd = np.log(c.shift(-h) / c)
        out[f"y_logret_{h}"] = fwd
        out[f"y_up_{h}"] = (fwd > up_threshold).astype("float64")
        out[f"y_up_{h}"] = out[f"y_up_{h}"].mask(fwd.isna())

        # Triple-barrier with vol-scaled barriers
        barrier = realised_vol * tb_mult * np.sqrt(h)
        upper = barrier
        lower = -barrier
        # Look ahead h bars; first crossing decides label.
        labels = pd.Series(np.nan, index=c.index)
        for i in range(len(c) - h):
            window = log_ret.iloc[i + 1:i + 1 + h].cumsum()
            if window.empty or np.isnan(barrier.iloc[i]):
                continue
            up_hit = (window >= upper.iloc[i]).idxmax() if (window >= upper.iloc[i]).any() else None
            dn_hit = (window <= lower.iloc[i]).idxmax() if (window <= lower.iloc[i]).any() else None
            if up_hit is None and dn_hit is None:
                labels.iloc[i] = 0.0
            elif up_hit is not None and (dn_hit is None or up_hit <= dn_hit):
                labels.iloc[i] = 1.0
            else:
                labels.iloc[i] = -1.0
        out[f"y_tb_{h}"] = labels.values

    return out

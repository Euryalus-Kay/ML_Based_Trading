"""Microstructure features computable from OHLCV only.

Real Level-2 microstructure needs an order book; without it the best
proxies are:
  - Amihud illiquidity: |r| / dollar_volume
  - Roll spread estimator: 2*sqrt(-cov(dp_t, dp_{t-1}))
  - Garman-Klass volatility (uses OHLC, more efficient than close-only)
  - High-low range volatility
  - Volume autocorrelation
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _gk_vol(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, win: int) -> pd.Series:
    rs = 0.5 * (np.log(h / l)) ** 2 - (2 * np.log(2) - 1) * (np.log(c / o)) ** 2
    return np.sqrt(rs.rolling(win, min_periods=max(2, win // 2)).mean())


def _roll_spread(c: pd.Series, win: int = 60) -> pd.Series:
    dp = c.diff()
    cov = dp.rolling(win, min_periods=max(5, win // 2)).cov(dp.shift(1))
    # Roll: spread = 2*sqrt(-cov) if cov < 0 else 0
    out = 2 * np.sqrt(-cov.clip(upper=0).abs())
    return out


def _amihud(ret: pd.Series, dollar_volume: pd.Series, win: int) -> pd.Series:
    raw = ret.abs() / dollar_volume.replace(0, np.nan)
    return raw.rolling(win, min_periods=max(2, win // 2)).mean()


def add_microstructure_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    out = df.copy()
    if not {"open", "high", "low", "close"}.issubset(out.columns):
        return out
    o, h, l, c = out["open"].astype(float), out["high"].astype(float), out["low"].astype(float), out["close"].astype(float)
    v = out["volume"].astype(float) if "volume" in out else pd.Series(0, index=out.index)

    out[f"{prefix}gk_vol_20"] = _gk_vol(o, h, l, c, 20)
    out[f"{prefix}gk_vol_60"] = _gk_vol(o, h, l, c, 60)
    out[f"{prefix}roll_spread_60"] = _roll_spread(c, 60)
    dv = c * v
    ret = c.pct_change()
    out[f"{prefix}amihud_20"] = _amihud(ret, dv, 20)
    out[f"{prefix}amihud_60"] = _amihud(ret, dv, 60)
    out[f"{prefix}vol_autocorr_60"] = v.rolling(60, min_periods=20).corr(v.shift(1))
    out[f"{prefix}price_vol_corr_60"] = c.pct_change().rolling(60, min_periods=20).corr(v.pct_change())
    return out

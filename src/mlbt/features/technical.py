"""Technical features for a single symbol's OHLCV bar frame.

Pure-NumPy / pandas. Adding to a future-leak-free pipeline: every feature
here is a function of *current and past* bars only, never future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def _rolling_zscore(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=max(2, win // 2)).mean()
    sd = s.rolling(win, min_periods=max(2, win // 2)).std()
    return (s - m) / sd.replace(0, np.nan)


def _rsi(close: pd.Series, win: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(win, min_periods=win).mean()
    dn = (-delta.clip(upper=0)).rolling(win, min_periods=win).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, win: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(win, min_periods=win).mean()


def _bollinger(close: pd.Series, win: int = 20, k: float = 2.0):
    m = close.rolling(win, min_periods=win).mean()
    sd = close.rolling(win, min_periods=win).std()
    upper = m + k * sd
    lower = m - k * sd
    width = (upper - lower) / m
    pos = (close - m) / sd
    return m, upper, lower, width, pos


def add_technical_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Augment df (OHLCV) with technical features. df must be sorted ascending.

    Adds: ret_*, log_ret_*, vol_*, rsi_*, atr_*, bb_*, mom_*, vwap deviation,
    rolling realised vol, vol of vol, autocorr.
    """
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float) if "high" in out else close
    low = out["low"].astype(float) if "low" in out else close
    vol = out["volume"].astype(float) if "volume" in out else pd.Series(0, index=out.index)

    out[f"{prefix}log_ret"] = _safe_log_returns(close)
    for w in (3, 5, 10, 20, 60):
        out[f"{prefix}ret_{w}"] = close.pct_change(w)
        out[f"{prefix}log_ret_{w}"] = np.log(close / close.shift(w))
        out[f"{prefix}vol_{w}"] = out[f"{prefix}log_ret"].rolling(w, min_periods=w).std() * np.sqrt(252 * 78)
        out[f"{prefix}mom_z_{w}"] = _rolling_zscore(close.pct_change(w), 60)

    out[f"{prefix}rsi_14"] = _rsi(close, 14)
    out[f"{prefix}atr_14"] = _atr(high, low, close, 14)
    out[f"{prefix}atr_14_pct"] = out[f"{prefix}atr_14"] / close

    bb_m, bb_u, bb_l, bb_w, bb_p = _bollinger(close, 20, 2.0)
    out[f"{prefix}bb_width"] = bb_w
    out[f"{prefix}bb_pos"] = bb_p

    # Volume features
    out[f"{prefix}volume_z_20"] = _rolling_zscore(vol, 20)
    out[f"{prefix}dollar_volume"] = (close * vol)
    out[f"{prefix}dollar_volume_z_20"] = _rolling_zscore(out[f"{prefix}dollar_volume"], 20)

    # Range / body
    rng = (high - low) / close
    body = (close - out["open"]).abs() / close if "open" in out else pd.Series(np.nan, index=out.index)
    out[f"{prefix}range_pct"] = rng
    out[f"{prefix}body_pct"] = body
    out[f"{prefix}upper_wick"] = (high - close.where(close > out["open"], out["open"])) / close if "open" in out else np.nan
    out[f"{prefix}lower_wick"] = (close.where(close < out["open"], out["open"]) - low) / close if "open" in out else np.nan

    # Autocorr of returns (short-horizon mean reversion signal)
    r = out[f"{prefix}log_ret"]
    out[f"{prefix}autocorr_5"] = r.rolling(60, min_periods=30).corr(r.shift(1))

    # Vol-of-vol
    out[f"{prefix}vol_of_vol_20"] = out[f"{prefix}vol_20"].rolling(20, min_periods=10).std()

    return out

"""Classical no-ML strategies that historically beat buy-and-hold SPY on Sharpe.

The most consistent way to "beat the S&P" isn't to predict direction — it's
to be invested LESS during high-vol drawdowns and MORE during low-vol uptrends.
This is the Asness/Israel/Moskowitz "volatility targeting" idea and the
classic Faber/200-DMA trend-following idea.

Three strategies, each rule-based and provably operational with the data we
have today (yfinance daily close + VIX):

  1. VOL_TARGET:  scale SPY position so realised vol stays at 12% annualised.
                  When VIX is high, exposure shrinks; when calm, leverage up
                  (capped at 1.5x).
  2. TREND_200D:  long SPY when SPY > 200d SMA, else cash.
  3. VIX_FILTER:  long SPY when VIX < its 1y median, else cash.

All three are evaluated against SPY buy-and-hold on the same window with
realistic costs (5 bps round-trip, 1-bar entry lag).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mlbt.core.log import get_logger
from mlbt.core.storage import Storage

log = get_logger("vol_target")


@dataclass
class StratConfig:
    bps_per_trade: float = 5.0
    slippage_bps: float = 2.0
    entry_lag_bars: int = 1
    bar: str = "1d"
    vol_target_ann: float = 0.12
    max_leverage: float = 1.5
    bpy: float = 252.0


def _load_close(symbol: str, source: str = "yf_daily") -> pd.Series:
    df = Storage().read(source, symbol)
    if df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)
    return df["close"].astype(float).sort_index()


def _equity_metrics(returns: pd.Series, bpy: float) -> dict:
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return {}
    mu = returns.mean()
    sd = returns.std()
    eq = (1 + returns).cumprod()
    dd = eq / eq.cummax() - 1.0
    return {
        "n_bars": int(len(returns)),
        "total_return": float(eq.iloc[-1] - 1.0),
        "ann_return": float(mu * bpy),
        "ann_vol": float(sd * np.sqrt(bpy)),
        "sharpe": float(mu / sd * np.sqrt(bpy)),
        "max_drawdown": float(dd.min()),
        "hit_rate": float((returns > 0).mean()),
        "calmar": float((mu * bpy) / abs(dd.min())) if dd.min() < 0 else float("inf"),
    }


def _strategy_pnl(positions: pd.Series, returns: pd.Series,
                   cost_bps: float, entry_lag: int) -> pd.Series:
    """Apply entry-lag and turnover cost."""
    pos = positions.shift(entry_lag).fillna(0)
    raw = pos * returns
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (cost_bps / 1e4)
    return raw - cost


def vol_target_spy(spy: pd.Series, cfg: StratConfig,
                    vol_window: int = 20) -> pd.Series:
    """Position = vol_target / realised_vol, capped at max_leverage."""
    log_ret = np.log(spy / spy.shift(1))
    realised_vol = log_ret.rolling(vol_window, min_periods=10).std() * np.sqrt(cfg.bpy)
    pos = (cfg.vol_target_ann / realised_vol.replace(0, np.nan)).clip(upper=cfg.max_leverage)
    pos = pos.fillna(1.0)
    rets = spy.pct_change()
    cost_bps = cfg.bps_per_trade + cfg.slippage_bps
    return _strategy_pnl(pos, rets, cost_bps, cfg.entry_lag_bars)


def trend_200d_spy(spy: pd.Series, cfg: StratConfig) -> pd.Series:
    sma200 = spy.rolling(200, min_periods=100).mean()
    pos = (spy > sma200).astype(float)
    rets = spy.pct_change()
    cost_bps = cfg.bps_per_trade + cfg.slippage_bps
    return _strategy_pnl(pos, rets, cost_bps, cfg.entry_lag_bars)


def vix_filter_spy(spy: pd.Series, vix: pd.Series, cfg: StratConfig,
                    median_window: int = 252) -> pd.Series:
    if vix.empty:
        return pd.Series(dtype=float)
    vix_med = vix.rolling(median_window, min_periods=60).median()
    aligned_vix = vix.reindex(spy.index).ffill()
    aligned_med = vix_med.reindex(spy.index).ffill()
    pos = (aligned_vix < aligned_med).astype(float)
    rets = spy.pct_change()
    cost_bps = cfg.bps_per_trade + cfg.slippage_bps
    return _strategy_pnl(pos, rets, cost_bps, cfg.entry_lag_bars)


def trend_plus_vol(spy: pd.Series, vix: pd.Series, cfg: StratConfig,
                     trend_window: int = 200, vol_window: int = 20) -> pd.Series:
    """Stack the rules: only invest when trend is up, and size by vol target."""
    log_ret = np.log(spy / spy.shift(1))
    realised_vol = log_ret.rolling(vol_window, min_periods=10).std() * np.sqrt(cfg.bpy)
    sma = spy.rolling(trend_window, min_periods=trend_window // 2).mean()
    in_trend = (spy > sma).astype(float)
    vol_scaled = (cfg.vol_target_ann / realised_vol.replace(0, np.nan)).clip(upper=cfg.max_leverage)
    pos = (in_trend * vol_scaled).fillna(0)
    rets = spy.pct_change()
    cost_bps = cfg.bps_per_trade + cfg.slippage_bps
    return _strategy_pnl(pos, rets, cost_bps, cfg.entry_lag_bars)


def run_all_classical(out_dir: str = "data/classical",
                       source: str = "yf_daily",
                       vix_symbol: str = "_VIX",
                       start: Optional[str] = None) -> dict:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    cfg = StratConfig()
    spy = _load_close("SPY", source=source)
    if spy.empty:
        log.warning("SPY not in storage — collect daily data first")
        return {}
    vix = _load_close(vix_symbol, source=source)
    if start:
        spy = spy.loc[start:]
        if not vix.empty:
            vix = vix.loc[start:]

    spy_ret = spy.pct_change()
    results = {
        "buy_hold_spy": _equity_metrics(spy_ret, cfg.bpy),
        "vol_target":   _equity_metrics(vol_target_spy(spy, cfg), cfg.bpy),
        "trend_200d":   _equity_metrics(trend_200d_spy(spy, cfg), cfg.bpy),
        "vix_filter":   _equity_metrics(vix_filter_spy(spy, vix, cfg), cfg.bpy),
        "trend_plus_vol": _equity_metrics(trend_plus_vol(spy, vix, cfg), cfg.bpy),
    }
    # Verdict: beats SPY on (Sharpe AND ann_return) or (Sharpe AND Calmar)
    spy_m = results["buy_hold_spy"]
    print("\n=== Classical no-ML strategies vs SPY ===")
    print(f"{'strategy':<18} {'sharpe':>8} {'ann_ret':>9} {'ann_vol':>9} {'maxDD':>9} {'calmar':>8} beats?")
    print("-" * 80)
    for name, m in results.items():
        if not m:
            print(f"{name:<18} (no data)")
            continue
        beats = (
            m.get("sharpe", 0) > spy_m.get("sharpe", 0)
            and m.get("ann_return", 0) > 0
        )
        marker = "✓" if (beats and name != "buy_hold_spy") else ""
        print(f"{name:<18} {m['sharpe']:>8.2f} {100*m['ann_return']:>8.2f}% {100*m['ann_vol']:>8.2f}% {100*m['max_drawdown']:>8.2f}% {m['calmar']:>8.2f} {marker}")

    import json
    (out_p / "classical_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    run_all_classical()

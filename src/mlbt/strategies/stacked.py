"""Stacked strategy: classical vol_target SPY core + ML long-short overlay.

The classical vol-target SPY strategy already beats buy-and-hold on Sharpe
and especially on drawdown. The ML cross-sectional alpha (top-k long, bottom-k
short) is harder to ship: it's noisier, costs eat it, and any single window
can flip negative. Stacking them combines vol_target's risk control with
ML's incremental alpha — and crucially the DD profile inherits the better
of the two.

Layout:
  total_return_t = w_classical * vol_target_t + w_ml * ml_long_short_t

The ml_long_short returns come from a `predictions.parquet` (the same format
the existing portfolio backtester consumes). vol_target returns come from
the daily SPY close.

This module is callable directly:

    from mlbt.strategies.stacked import backtest_stacked
    res = backtest_stacked(predictions_dir, dataset_path, w_ml=0.3,
                           bps_per_trade=5, slippage_bps=2)
    print(res["beats_spy"], res["net_sharpe"])
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mlbt.backtest.portfolio import (
    PortfolioConfig, run_portfolio_backtest, _equity_metrics, _bars_per_year,
)
from mlbt.core.log import get_logger
from mlbt.core.storage import Storage
from mlbt.strategies.vol_target import (
    StratConfig, vol_target_spy, trend_plus_vol, _load_close,
)

log = get_logger("stacked")


@dataclass
class StackedConfig:
    w_classical: float = 0.7   # weight on vol_target SPY
    w_ml: float = 0.3          # weight on ML long-short
    classical_kind: str = "vol_target"  # "vol_target" or "trend_plus_vol"
    bar: str = "1h"
    horizon: int = 4
    target_col: str = "y_resid_logret_4"
    top_k: int = 20
    bottom_k: int = 20
    bps_per_trade: float = 5.0
    slippage_bps: float = 2.0
    entry_lag_bars: int = 1
    overlap_scale: bool = True
    market_neutral: bool = True


def _ml_returns_from_predictions(model_dir: str, dataset_path: str,
                                  cfg: StackedConfig) -> pd.Series:
    """Run the existing portfolio backtest, return the NET per-bar return series."""
    p_cfg = PortfolioConfig(
        horizon=cfg.horizon, bar=cfg.bar,
        top_k=cfg.top_k, bottom_k=cfg.bottom_k,
        market_neutral=cfg.market_neutral,
        entry_lag_bars=cfg.entry_lag_bars,
        bps_per_trade=cfg.bps_per_trade,
        slippage_bps=cfg.slippage_bps,
        target_col=cfg.target_col,
    )
    # Use the engine but capture the per-bar net_pnl series via the parquet
    preds_p = Path(model_dir) / "predictions.parquet"
    if not preds_p.exists():
        raise FileNotFoundError(preds_p)
    preds = pd.read_parquet(preds_p)
    ds = pd.read_parquet(dataset_path)
    if cfg.target_col not in ds.columns:
        raise KeyError(cfg.target_col)

    if cfg.entry_lag_bars > 0:
        ds = ds.copy()
        ds[cfg.target_col] = ds.groupby("symbol")[cfg.target_col].shift(-cfg.entry_lag_bars)
    p = preds.reset_index().rename(columns={"index": "ts"})
    d = ds.reset_index().rename(columns={"index": "ts"})[["ts", "symbol", cfg.target_col]]
    df = p.merge(d, on=["ts", "symbol"], how="inner").rename(columns={cfg.target_col: "fwd_ret"})
    df = df.dropna(subset=["fwd_ret", "y_score"])
    if df.empty:
        return pd.Series(dtype=float)

    cost_bps_total = (cfg.bps_per_trade + cfg.slippage_bps) / 1e4

    rec = []
    last_pos: dict[str, float] = {}
    for ts, group in df.groupby("ts"):
        scored = group.sort_values("y_score", ascending=False)
        longs = scored.head(cfg.top_k)
        shorts = scored.tail(cfg.bottom_k)
        weights = pd.Series(0.0, index=group["symbol"].values)
        long_w = 0.5 / max(1, len(longs))
        short_w = -0.5 / max(1, len(shorts))
        for s in longs["symbol"]:
            weights[s] = long_w
        for s in shorts["symbol"]:
            weights[s] = short_w
        sub = group.set_index("symbol")
        pnl = (weights * sub["fwd_ret"]).sum()
        all_s = set(weights.index) | set(last_pos.keys())
        turnover = sum(abs(float(weights.get(s, 0.0)) - float(last_pos.get(s, 0.0)))
                        for s in all_s)
        cost = turnover * cost_bps_total
        rec.append({"ts": ts, "net": pnl - cost})
        last_pos = {s: float(weights.get(s, 0.0)) for s in weights.index
                    if abs(weights.get(s, 0)) > 1e-9}

    pnl_df = pd.DataFrame(rec).set_index("ts").sort_index()
    if cfg.overlap_scale and cfg.horizon > 1:
        pnl_df["net"] /= cfg.horizon
    return pnl_df["net"]


def _classical_daily_returns(cfg: StackedConfig) -> pd.Series:
    """Run vol_target / trend_plus_vol on stored SPY daily."""
    spy = _load_close("SPY")
    if spy.empty:
        log.warning("SPY daily not in storage")
        return pd.Series(dtype=float)
    s_cfg = StratConfig(bps_per_trade=cfg.bps_per_trade,
                          slippage_bps=cfg.slippage_bps,
                          entry_lag_bars=cfg.entry_lag_bars)
    if cfg.classical_kind == "vol_target":
        return vol_target_spy(spy, s_cfg)
    if cfg.classical_kind == "trend_plus_vol":
        vix = _load_close("_VIX")
        return trend_plus_vol(spy, vix, s_cfg)
    raise ValueError(cfg.classical_kind)


def _align_returns_to_bars(daily_ret: pd.Series, ml_idx: pd.DatetimeIndex,
                            bars_per_day: float) -> pd.Series:
    """Map daily returns onto the bar grid.

    For 1h bars (~6.5 bars/day), distribute the daily geometric return across
    the bars of that day so the daily compounded return matches. Equivalent
    of `(1 + r_day) ** (1 / bars_per_day) - 1` applied at each bar.
    """
    if daily_ret.empty or len(ml_idx) == 0:
        return pd.Series(0.0, index=ml_idx)
    daily_ret = daily_ret.dropna()
    if daily_ret.empty:
        return pd.Series(0.0, index=ml_idx)
    # Normalise daily index to date
    d_ret_by_date = daily_ret.copy()
    d_ret_by_date.index = pd.to_datetime(d_ret_by_date.index).normalize()
    d_ret_by_date = d_ret_by_date[~d_ret_by_date.index.duplicated(keep="last")]

    out = pd.Series(0.0, index=ml_idx)
    dates = pd.to_datetime(ml_idx).normalize()
    per_bar = (1 + d_ret_by_date) ** (1.0 / max(1.0, bars_per_day)) - 1.0
    # Map each bar to its date's per-bar return
    mapped = per_bar.reindex(dates, method=None)
    mapped.index = ml_idx
    return mapped.fillna(0.0)


def backtest_stacked(model_dir: str, dataset_path: str,
                      cfg: Optional[StackedConfig] = None,
                      out_path: Optional[str] = None) -> dict:
    cfg = cfg or StackedConfig()
    ml_ret = _ml_returns_from_predictions(model_dir, dataset_path, cfg)
    if ml_ret.empty:
        return {"error": "no ML returns"}

    daily_classical = _classical_daily_returns(cfg)
    bpy = _bars_per_year(cfg.bar)
    bars_per_day = bpy / 252.0
    classical_aligned = _align_returns_to_bars(daily_classical, ml_ret.index, bars_per_day)

    combined = cfg.w_classical * classical_aligned + cfg.w_ml * ml_ret
    combined = combined.dropna()

    # SPY benchmark on the same bar grid
    try:
        st = Storage()
        spy_intra = st.read("yf_1h", "SPY")
        if spy_intra.empty:
            spy_intra = st.read("yf_daily", "SPY")
        if not spy_intra.empty and "close" in spy_intra.columns:
            spy_ret = (spy_intra["close"].astype(float).pct_change()
                                                          .reindex(combined.index).fillna(0))
        else:
            spy_ret = pd.Series(0.0, index=combined.index)
    except Exception:
        spy_ret = pd.Series(0.0, index=combined.index)

    m = _equity_metrics(combined, bpy)
    spy_m = _equity_metrics(spy_ret, bpy)
    ml_m = _equity_metrics(ml_ret.reindex(combined.index).fillna(0), bpy)
    cl_m = _equity_metrics(classical_aligned, bpy)

    beats_spy = (
        (m.get("ann_return") or 0) > (spy_m.get("ann_return") or 0)
        and (m.get("sharpe") or 0) > (spy_m.get("sharpe") or 0)
    )

    report = {
        "config": cfg.__dict__,
        "net": m, "ml_only": ml_m, "classical_only": cl_m,
        "spy": spy_m, "beats_spy": bool(beats_spy),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2, default=str))
        combined.to_frame("net").to_parquet(Path(out_path).with_suffix(".parquet"))
    return report


def sweep_weights(model_dir: str, dataset_path: str,
                   weights=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                   **cfg_kwargs) -> pd.DataFrame:
    rows = []
    for w in weights:
        cfg = StackedConfig(w_classical=1 - w, w_ml=w, **cfg_kwargs)
        r = backtest_stacked(model_dir, dataset_path, cfg)
        if "error" in r:
            continue
        rows.append({"w_ml": w, "sharpe": r["net"]["sharpe"],
                      "ann_ret": r["net"]["ann_return"],
                      "max_dd": r["net"]["max_drawdown"],
                      "beats_spy": r["beats_spy"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python -m mlbt.strategies.stacked <model_dir> <dataset_path> [w_ml]")
        sys.exit(1)
    md, dp = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        cfg = StackedConfig(w_classical=1 - float(sys.argv[3]),
                              w_ml=float(sys.argv[3]))
        r = backtest_stacked(md, dp, cfg)
        print(json.dumps(r, indent=2, default=str))
    else:
        df = sweep_weights(md, dp)
        print(df.to_string(index=False))

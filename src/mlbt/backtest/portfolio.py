"""Long-short cross-sectional portfolio simulator.

Built for the cross-sectional rank-target models. At each rebalance bar:
  - rank symbols by model score
  - long top K (or top quantile), short bottom K
  - position-size to target NET exposure = 0 (market-neutral) by default,
    or to constant GROSS exposure
  - hold for `horizon` bars, then rebalance

Realistic accounting:
  - entry_lag_bars: signal at t executed at t + lag
  - bps_per_trade + slippage_bps: per-trade cost on |Δposition|
  - max_position_pct: cap any single name at this share of book
  - daily ann factor based on bar frequency

Outputs:
  equity curve, net Sharpe, ann return, max DD, hit rate, trade count,
  beats_spy boolean.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mlbt.core.log import get_logger

log = get_logger("portfolio")


@dataclass
class PortfolioConfig:
    horizon: int = 1                  # bars to hold
    bar: str = "1h"
    top_k: int = 10                   # number of longs
    bottom_k: int = 10                # number of shorts
    market_neutral: bool = True       # net exposure = 0
    gross_leverage: float = 1.0       # total gross exposure as fraction of capital
    max_position_pct: float = 0.15    # cap per name
    rebalance_every: int = 1          # bars between rebalances
    entry_lag_bars: int = 1
    bps_per_trade: float = 5.0        # realistic retail round-trip estimate
    slippage_bps: float = 2.0
    min_universe_size: int = 6        # need at least this many symbols ranked
    target_col: str = "y_resid_logret_1"  # which forward log-return to harvest


def _bars_per_year(bar: str) -> float:
    bar = bar.lower()
    return {"1min": 252*390, "5min": 252*78, "15min": 252*26,
            "1h": 252*6.5, "1d": 252, "1d_full": 252}.get(bar, 252)


def _equity_metrics(returns: pd.Series, bpy: float) -> dict:
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return {"n_bars": int(len(returns)), "total_return": 0.0,
                "ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "hit_rate": 0.0}
    mu = returns.mean()
    sd = returns.std()
    eq = (1 + returns).cumprod()
    dd = (eq / eq.cummax() - 1.0)
    return {
        "n_bars": int(len(returns)),
        "total_return": float(eq.iloc[-1] - 1.0),
        "ann_return": float(mu * bpy),
        "ann_vol": float(sd * np.sqrt(bpy)),
        "sharpe": float(mu / sd * np.sqrt(bpy)),
        "max_drawdown": float(dd.min()),
        "hit_rate": float((returns > 0).mean()),
    }


def run_portfolio_backtest(dataset_path: str, model_dir: str,
                            cfg: Optional[PortfolioConfig] = None,
                            out_path: Optional[str] = None) -> dict:
    cfg = cfg or PortfolioConfig()
    model_p = Path(model_dir)
    preds_p = model_p / "predictions.parquet"
    if not preds_p.exists():
        raise FileNotFoundError(f"predictions.parquet missing in {model_dir}")
    preds = pd.read_parquet(preds_p)
    if "symbol" not in preds.columns:
        raise ValueError("predictions must contain a 'symbol' column "
                          "(retrain with symbols preserved)")
    ds = pd.read_parquet(dataset_path)
    if cfg.target_col not in ds.columns:
        raise KeyError(f"{cfg.target_col} missing from dataset")

    # Shift forward return by entry_lag (we cannot earn the bar that
    # contained the score)
    if cfg.entry_lag_bars > 0:
        ds = ds.copy()
        ds[cfg.target_col] = ds.groupby("symbol")[cfg.target_col].shift(-cfg.entry_lag_bars)

    # Join preds with forward returns by (timestamp, symbol)
    p = preds.reset_index().rename(columns={"index": "ts"})
    d = ds.reset_index().rename(columns={"index": "ts"})[["ts", "symbol", cfg.target_col]]
    df = p.merge(d, on=["ts", "symbol"], how="inner")
    df = df.rename(columns={cfg.target_col: "fwd_ret"})
    df = df.dropna(subset=["fwd_ret", "y_score"])
    if df.empty:
        return {"error": "no overlapping rows"}

    # Per-bar long/short selection
    df = df.sort_values(["ts", "y_score"])
    rebalance_returns = []
    cap_per_name = cfg.max_position_pct
    last_positions: dict[str, float] = {}

    def _alloc(group: pd.DataFrame) -> pd.Series:
        """Given one bar's symbols+scores, return position weights summing to gross/2 for long and -gross/2 for short."""
        if len(group) < cfg.min_universe_size:
            return pd.Series(0.0, index=group["symbol"].values)
        scored = group.sort_values("y_score", ascending=False)
        longs = scored.head(cfg.top_k)
        shorts = scored.tail(cfg.bottom_k)
        weights = pd.Series(0.0, index=group["symbol"].values)
        if cfg.market_neutral:
            long_w = cfg.gross_leverage / 2 / max(1, len(longs))
            short_w = -cfg.gross_leverage / 2 / max(1, len(shorts))
        else:
            long_w = cfg.gross_leverage / max(1, len(longs))
            short_w = 0.0
        for s in longs["symbol"]:
            weights[s] = min(long_w, cap_per_name)
        for s in shorts["symbol"]:
            weights[s] = max(short_w, -cap_per_name)
        return weights

    # Group by timestamp and iterate
    cost_bps = (cfg.bps_per_trade + cfg.slippage_bps) / 1e4
    pnl_records = []
    for ts, group in df.groupby("ts"):
        weights = _alloc(group)
        # PnL = sum(weight * fwd_ret) over symbols
        sub = group.set_index("symbol")
        pnl = (weights * sub["fwd_ret"]).sum()
        # Turnover cost on changes from last positions
        all_symbols = set(weights.index) | set(last_positions.keys())
        turnover = 0.0
        for s in all_symbols:
            new_w = float(weights.get(s, 0.0))
            old_w = float(last_positions.get(s, 0.0))
            turnover += abs(new_w - old_w)
        cost = turnover * cost_bps
        pnl_records.append({"ts": ts, "gross_pnl": pnl, "turnover": turnover,
                              "cost": cost, "net_pnl": pnl - cost})
        last_positions = {s: float(weights.get(s, 0.0)) for s in weights.index if abs(weights.get(s, 0)) > 0}

    if not pnl_records:
        return {"error": "no PnL records"}
    pnl_df = pd.DataFrame(pnl_records).set_index("ts").sort_index()

    # If horizon > 1 we have implicit leverage; divide by horizon to size to
    # a constant single-period book.
    if cfg.horizon > 1:
        pnl_df["net_pnl"] /= cfg.horizon
        pnl_df["gross_pnl"] /= cfg.horizon

    bpy = _bars_per_year(cfg.bar)
    gross = _equity_metrics(pnl_df["gross_pnl"], bpy)
    net = _equity_metrics(pnl_df["net_pnl"], bpy)

    # SPY benchmark over the same window
    try:
        from mlbt.core.storage import Storage as _Storage
        spy = _Storage().read("yf_1h", "SPY")
        if spy.empty:
            spy = _Storage().read("yf_daily", "SPY")
        if not spy.empty and "close" in spy.columns:
            spy_ret = (spy["close"].astype(float).pct_change()
                                                  .reindex(pnl_df.index).fillna(0))
            spy_m = _equity_metrics(spy_ret, bpy)
        else:
            spy_m = {}
    except Exception:
        spy_m = {}

    beats_spy = (
        net.get("ann_return", 0) > spy_m.get("ann_return", 0)
        and net.get("sharpe", 0) > spy_m.get("sharpe", 0)
    )

    report = {
        "config": cfg.__dict__,
        "gross": gross,
        "net": net,
        "benchmark_spy": spy_m,
        "beats_spy": bool(beats_spy),
        "n_rebalances": int(len(pnl_df)),
        "avg_turnover": float(pnl_df["turnover"].mean()),
        "total_cost_drag": float(pnl_df["cost"].sum()),
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2, default=str))
        # Also persist the equity curve
        pnl_df.to_parquet(Path(out_path).with_suffix(".parquet"))
    return report

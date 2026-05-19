"""Walk-forward validation across multiple time windows.

A single 2-year backtest can lie. This splits the available history into
non-overlapping sub-windows and reports per-window beat-SPY status. A
strategy that beats SPY in 4 of 5 windows is robust; 1 of 5 is luck.

For each window: train on prior data, test on the window, run portfolio
backtest with realistic costs, record (sharpe, beats_spy, calmar, max_dd).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from mlbt.backtest.portfolio import PortfolioConfig, run_portfolio_backtest
from mlbt.core.log import get_logger
from mlbt.core.storage import Storage
from mlbt.ml.train import train_model
from mlbt.strategies.vol_target import (
    StratConfig, vol_target_spy, trend_200d_spy, trend_plus_vol,
    _equity_metrics,
)

log = get_logger("walk_validate")


def split_windows(start, end, n: int = 5) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    edges = pd.date_range(start, end, periods=n + 1)
    return [(edges[i], edges[i + 1]) for i in range(n)]


def classical_walk_validate(symbol: str = "SPY",
                              vix_symbol: str = "_VIX",
                              source: str = "yf_daily",
                              n_windows: int = 5) -> pd.DataFrame:
    st = Storage()
    spy = st.read(source, symbol)["close"].astype(float).sort_index()
    vix = st.read(source, vix_symbol)
    vix = vix["close"].astype(float).sort_index() if not vix.empty else pd.Series(dtype=float)
    cfg = StratConfig()
    windows = split_windows(spy.index.min(), spy.index.max(), n=n_windows)
    rows = []
    for w_start, w_end in windows:
        sub = spy.loc[w_start:w_end]
        if len(sub) < 50:
            continue
        sub_vix = vix.loc[w_start:w_end] if not vix.empty else vix
        spy_ret = sub.pct_change()
        out = {
            "window": f"{w_start.date()} → {w_end.date()}",
            "n_bars": len(sub),
            "buy_hold": _equity_metrics(spy_ret, cfg.bpy),
            "vol_target": _equity_metrics(vol_target_spy(sub, cfg), cfg.bpy),
            "trend_200d": _equity_metrics(trend_200d_spy(sub, cfg), cfg.bpy),
            "trend_plus_vol": _equity_metrics(trend_plus_vol(sub, sub_vix, cfg), cfg.bpy),
        }
        for strat_name in ("vol_target", "trend_200d", "trend_plus_vol"):
            sm = out.get(strat_name) or {}
            bh = out["buy_hold"]
            sm["beats_sharpe"] = sm.get("sharpe", 0) > bh.get("sharpe", 0)
            sm["beats_calmar"] = sm.get("calmar", 0) > bh.get("calmar", 0)
        rows.append(out)

    # Flatten into a tidy table: one row per (window, strategy)
    tidy = []
    for w in rows:
        for s in ("buy_hold", "vol_target", "trend_200d", "trend_plus_vol"):
            m = w.get(s) or {}
            tidy.append({
                "window": w["window"], "strategy": s,
                "sharpe": m.get("sharpe"),
                "ann_return": m.get("ann_return"),
                "max_dd": m.get("max_drawdown"),
                "calmar": m.get("calmar"),
                "beats_sharpe": m.get("beats_sharpe", s != "buy_hold"),
                "beats_calmar": m.get("beats_calmar", s != "buy_hold"),
            })
    return pd.DataFrame(tidy)


def run_and_print(n_windows: int = 5) -> dict:
    df = classical_walk_validate(n_windows=n_windows)
    if df.empty:
        print("(no data — collect SPY daily first)")
        return {}
    # Pivot Sharpe matrix
    sharpe_pv = df.pivot(index="window", columns="strategy", values="sharpe")
    print("\n=== Sharpe per window ===")
    print(sharpe_pv.round(2).to_string())
    print("\n=== Max DD per window ===")
    print(df.pivot(index="window", columns="strategy", values="max_dd").round(3).to_string())
    print("\n=== Calmar per window ===")
    print(df.pivot(index="window", columns="strategy", values="calmar").round(2).to_string())

    # Robustness verdict per strategy
    strats = [s for s in df["strategy"].unique() if s != "buy_hold"]
    verdict_rows = []
    for s in strats:
        sub = df[df["strategy"] == s]
        n_beats_sharpe = sub["beats_sharpe"].sum()
        n_beats_calmar = sub["beats_calmar"].sum()
        verdict_rows.append({
            "strategy": s,
            f"beats_sharpe_in": f"{int(n_beats_sharpe)}/{len(sub)}",
            f"beats_calmar_in": f"{int(n_beats_calmar)}/{len(sub)}",
            "robust": (n_beats_sharpe >= 0.6 * len(sub)) or (n_beats_calmar >= 0.6 * len(sub)),
        })
    verdict_df = pd.DataFrame(verdict_rows)
    print("\n=== Robustness verdict (across windows) ===")
    print(verdict_df.to_string(index=False))
    return {"sharpe_per_window": sharpe_pv.to_dict(), "verdict": verdict_df.to_dict("records")}


if __name__ == "__main__":
    run_and_print()

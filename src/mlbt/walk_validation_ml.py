"""ML / stacked-strategy walk-forward validation across multiple time windows.

Takes a trained ML model (predictions.parquet) and a dataset, slices into N
non-overlapping sub-windows, runs the portfolio backtest (or the stacked
strategy with vol_target overlay) on each window, and reports per-window
beats_spy / Sharpe / Calmar / maxDD.

Verdict: robust if Sharpe OR Calmar beats SPY in >= N*0.8 of windows AND
max_dd is strictly less than SPY's in EVERY window.

Usage:
    from mlbt.walk_validation_ml import ml_walk_validate
    df = ml_walk_validate(model_dir, dataset_path, n_windows=5)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mlbt.backtest.portfolio import (
    PortfolioConfig, _equity_metrics, _bars_per_year,
)
from mlbt.core.log import get_logger
from mlbt.core.storage import Storage
from mlbt.strategies.stacked import (
    StackedConfig, _ml_returns_from_predictions,
    _classical_daily_returns, _align_returns_to_bars,
)

log = get_logger("walk_ml")


def _split_windows(idx_start, idx_end, n: int = 5):
    edges = pd.date_range(idx_start, idx_end, periods=n + 1)
    return [(edges[i], edges[i + 1]) for i in range(n)]


def _spy_returns_on_grid(grid: pd.DatetimeIndex,
                          bar: str = "1h") -> pd.Series:
    st = Storage()
    spy = st.read("yf_1h", "SPY") if bar == "1h" else pd.DataFrame()
    if spy.empty:
        spy = st.read("yf_daily", "SPY")
    if spy.empty or "close" not in spy.columns:
        return pd.Series(0.0, index=grid)
    return spy["close"].astype(float).pct_change().reindex(grid).fillna(0)


def ml_walk_validate(model_dir: str, dataset_path: str,
                      n_windows: int = 5,
                      cfg: Optional[StackedConfig] = None,
                      mode: str = "stacked",
                      out_csv: Optional[str] = None) -> pd.DataFrame:
    """mode: 'ml_only' or 'classical_only' or 'stacked'."""
    cfg = cfg or StackedConfig()
    bpy = _bars_per_year(cfg.bar)

    ml_ret = _ml_returns_from_predictions(model_dir, dataset_path, cfg)
    if ml_ret.empty:
        log.warning("no ML returns; aborting walk validation")
        return pd.DataFrame()

    bars_per_day = bpy / 252.0
    classical_daily = _classical_daily_returns(cfg)
    classical_aligned = _align_returns_to_bars(classical_daily, ml_ret.index,
                                                 bars_per_day)
    if mode == "ml_only":
        strat = ml_ret
    elif mode == "classical_only":
        strat = classical_aligned
    else:
        strat = cfg.w_classical * classical_aligned + cfg.w_ml * ml_ret

    spy = _spy_returns_on_grid(strat.index, bar=cfg.bar)

    windows = _split_windows(strat.index.min(), strat.index.max(), n=n_windows)
    rows = []
    for ws, we in windows:
        sub_strat = strat.loc[ws:we]
        sub_spy = spy.loc[ws:we]
        if len(sub_strat) < 50:
            continue
        m_strat = _equity_metrics(sub_strat, bpy)
        m_spy = _equity_metrics(sub_spy, bpy)
        m_strat["calmar"] = (m_strat.get("ann_return", 0) /
                              abs(m_strat.get("max_drawdown", -1e-9))) if m_strat.get("max_drawdown", 0) < 0 else float("inf")
        m_spy["calmar"] = (m_spy.get("ann_return", 0) /
                              abs(m_spy.get("max_drawdown", -1e-9))) if m_spy.get("max_drawdown", 0) < 0 else float("inf")
        rows.append({
            "window": f"{ws.date()} → {we.date()}",
            "n_bars": int(len(sub_strat)),
            "strat_sharpe": m_strat["sharpe"],
            "strat_ann_return": m_strat["ann_return"],
            "strat_max_dd": m_strat["max_drawdown"],
            "strat_calmar": m_strat["calmar"],
            "spy_sharpe": m_spy["sharpe"],
            "spy_max_dd": m_spy["max_drawdown"],
            "spy_calmar": m_spy["calmar"],
            "beats_sharpe": m_strat["sharpe"] > m_spy["sharpe"],
            "beats_calmar": m_strat["calmar"] > m_spy["calmar"],
            "dd_better": m_strat["max_drawdown"] > m_spy["max_drawdown"],
        })

    df = pd.DataFrame(rows)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    # Robustness summary
    n = len(df)
    n_beats_sharpe = int(df["beats_sharpe"].sum())
    n_beats_calmar = int(df["beats_calmar"].sum())
    n_dd_better = int(df["dd_better"].sum())
    print(f"\n=== ML walk-forward ({mode}) ===")
    print(df.to_string(index=False))
    print(f"\nbeats_sharpe: {n_beats_sharpe}/{n}, beats_calmar: {n_beats_calmar}/{n}, "
          f"dd_better: {n_dd_better}/{n}")
    return df


if __name__ == "__main__":
    import sys
    md = sys.argv[1] if len(sys.argv) > 1 else "data/mac_tournament/gbm_xl_h4"
    dp = sys.argv[2] if len(sys.argv) > 2 else "data/dataset_1h_sp500.parquet"
    mode = sys.argv[3] if len(sys.argv) > 3 else "stacked"
    ml_walk_validate(md, dp, mode=mode)

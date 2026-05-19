"""Sweep classical strategy parameters across walk-forward windows.

The classical walk_validation module exposes the canonical strategies (vol_target,
trend_200d, trend_plus_vol). Here we sweep their parameters (vol target, trend
window, leverage cap) on the same 5 sub-windows so we can pick a config that
beats SPY on Sharpe OR Calmar in 4 of 5 windows AND has DD strictly less than SPY
in every window.

Usage:
    python -m mlbt.classical_walk_sweep
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlbt.core.log import get_logger
from mlbt.core.storage import Storage
from mlbt.strategies.vol_target import (
    StratConfig, vol_target_spy, trend_200d_spy, trend_plus_vol,
    _equity_metrics, _load_close,
)

log = get_logger("classical_walk_sweep")


def _split_windows(start, end, n=5):
    edges = pd.date_range(start, end, periods=n + 1)
    return [(edges[i], edges[i + 1]) for i in range(n)]


def evaluate_strategy(returns: pd.Series, label: str, windows) -> pd.DataFrame:
    rows = []
    for ws, we in windows:
        sub = returns.loc[ws:we].dropna()
        if len(sub) < 50:
            rows.append({"window": f"{ws.date()} → {we.date()}",
                          "strategy": label, "sharpe": np.nan,
                          "calmar": np.nan, "max_dd": np.nan, "n_bars": 0})
            continue
        m = _equity_metrics(sub, 252.0)
        rows.append({"window": f"{ws.date()} → {we.date()}",
                      "strategy": label, "sharpe": m.get("sharpe"),
                      "ann_return": m.get("ann_return"),
                      "max_dd": m.get("max_drawdown"),
                      "calmar": m.get("calmar", float("nan")),
                      "n_bars": len(sub)})
    return pd.DataFrame(rows)


def sweep_classical(n_windows: int = 5) -> pd.DataFrame:
    spy = _load_close("SPY")
    vix = _load_close("_VIX")
    if spy.empty:
        print("(no SPY data)")
        return pd.DataFrame()
    windows = _split_windows(spy.index.min(), spy.index.max(), n=n_windows)
    bh_ret = spy.pct_change()
    bh_df = evaluate_strategy(bh_ret, "buy_hold_SPY", windows)

    configs = []
    # Sweep vol_target parameters
    for tgt_vol in (0.08, 0.10, 0.12, 0.15):
        for max_lev in (1.0, 1.25, 1.5):
            cfg = StratConfig(vol_target_ann=tgt_vol, max_leverage=max_lev,
                                bps_per_trade=5.0, slippage_bps=2.0)
            ret = vol_target_spy(spy, cfg)
            label = f"vt_v{int(tgt_vol*100)}_lev{int(max_lev*100)}"
            configs.append((label, ret))

    # Trend
    for trend_win in (100, 150, 200, 250):
        cfg = StratConfig(bps_per_trade=5.0, slippage_bps=2.0)
        sma = spy.rolling(trend_win, min_periods=trend_win // 2).mean()
        pos = (spy > sma).astype(float)
        rets = spy.pct_change()
        pos = pos.shift(cfg.entry_lag_bars).fillna(0)
        turnover = pos.diff().abs().fillna(pos.abs())
        cost = turnover * ((cfg.bps_per_trade + cfg.slippage_bps) / 1e4)
        configs.append((f"trend_{trend_win}", pos * rets - cost))

    # Trend + vol_target combo
    for trend_win in (100, 200):
        for tgt_vol in (0.10, 0.12):
            cfg = StratConfig(vol_target_ann=tgt_vol, bps_per_trade=5.0, slippage_bps=2.0)
            ret = trend_plus_vol(spy, vix, cfg, trend_window=trend_win)
            configs.append((f"tpv_w{trend_win}_v{int(tgt_vol*100)}", ret))

    all_rows = [bh_df]
    for label, ret in configs:
        all_rows.append(evaluate_strategy(ret, label, windows))
    df = pd.concat(all_rows, ignore_index=True)

    # Pivot to compare per window
    bh_window = df[df.strategy == "buy_hold_SPY"].set_index("window")
    summary = []
    for label in [c[0] for c in configs]:
        sub = df[df.strategy == label].set_index("window")
        n_beats_sharpe = 0
        n_beats_calmar = 0
        n_dd_better = 0
        n_evaluated = 0
        details = {}
        for w in sub.index:
            if w not in bh_window.index:
                continue
            n_evaluated += 1
            ml_sh = sub.loc[w, "sharpe"]
            ml_ca = sub.loc[w, "calmar"]
            ml_dd = sub.loc[w, "max_dd"]
            bh_sh = bh_window.loc[w, "sharpe"]
            bh_ca = bh_window.loc[w, "calmar"]
            bh_dd = bh_window.loc[w, "max_dd"]
            if pd.notna(ml_sh) and ml_sh > bh_sh:
                n_beats_sharpe += 1
            if pd.notna(ml_ca) and ml_ca > bh_ca:
                n_beats_calmar += 1
            if pd.notna(ml_dd) and ml_dd > bh_dd:  # higher (less negative) DD
                n_dd_better += 1
            details[w] = {"sh": ml_sh, "ca": ml_ca, "dd": ml_dd}
        summary.append({
            "strategy": label, "n_evaluated": n_evaluated,
            "beats_sharpe": f"{n_beats_sharpe}/{n_evaluated}",
            "beats_calmar": f"{n_beats_calmar}/{n_evaluated}",
            "dd_better": f"{n_dd_better}/{n_evaluated}",
            "either_4of5": (max(n_beats_sharpe, n_beats_calmar) >= max(1, int(0.8 * n_evaluated))),
            "passes_dd": n_dd_better == n_evaluated,
        })
    summ_df = pd.DataFrame(summary).sort_values(
        ["either_4of5", "passes_dd", "beats_calmar"], ascending=[False, False, False])
    print("\n=== Classical sweep walk-forward summary (5 windows 2005-2026) ===")
    print(summ_df.to_string(index=False))
    return summ_df, df


if __name__ == "__main__":
    sweep_classical()

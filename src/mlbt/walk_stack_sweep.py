"""Quick sweep of w_ml across already-trained walk-forward models.

Re-uses model.pkl and predictions from data/wt_daily_xsN/winK to evaluate the
stacked strategy at several w_ml weights without re-training. Fast — just
runs the backtest + alignment loop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from mlbt.backtest.portfolio import (
    PortfolioConfig, run_portfolio_backtest, _equity_metrics, _bars_per_year,
)
from mlbt.core.log import get_logger
from mlbt.strategies.vol_target import (
    StratConfig, vol_target_spy, _load_close,
)

log = get_logger("walk_stack_sweep")


def sweep_walk_stack(walk_dir: str, target: str = "y_xsec_top_10",
                      weights=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                      top_k: int = 10, bottom_k: int = 0,
                      market_neutral: bool = False,
                      bps_per_trade: float = 5.0, slippage_bps: float = 2.0,
                      bar: str = "1d") -> pd.DataFrame:
    walk_p = Path(walk_dir)
    h = int(target.rsplit("_", 1)[-1])
    target_col = f"y_resid_logret_{h}"
    rows = []

    # Find all winK directories that have model.pkl + test_subset
    win_dirs = sorted([d for d in walk_p.iterdir() if d.is_dir() and d.name.startswith("win")])
    cfg = PortfolioConfig(
        horizon=h, bar=bar, target_col=target_col, entry_lag_bars=1,
        top_k=top_k, bottom_k=bottom_k, market_neutral=market_neutral,
        bps_per_trade=bps_per_trade, slippage_bps=slippage_bps,
    )
    for win_dir in win_dirs:
        model_p = win_dir / "model.pkl"
        test_p = win_dir / "test_subset.parquet"
        if not model_p.exists() or not test_p.exists():
            continue
        # Run ML backtest once (writes ml_bt.parquet)
        try:
            r = run_portfolio_backtest(dataset_path=str(test_p),
                                         model_dir=str(win_dir), cfg=cfg,
                                         out_path=str(win_dir / "ml_bt.json"))
        except Exception as e:
            log.warning("backtest failed %s: %s", win_dir.name, e)
            continue
        ml_pnl = pd.read_parquet(win_dir / "ml_bt.parquet")
        ml_ret = ml_pnl["net_pnl"]
        spy_m = r.get("benchmark_spy") or {}
        spy_sh = spy_m.get("sharpe", 0) or 0
        spy_dd = spy_m.get("max_drawdown", 0) or -1
        spy_calmar = (spy_m.get("ann_return", 0) / abs(spy_dd)) if spy_dd < 0 else float("inf")

        # Vol-target on this window
        test_df = pd.read_parquet(test_p)
        ws = test_df.index.min()
        we = test_df.index.max()
        vt_cfg = StratConfig(bps_per_trade=bps_per_trade, slippage_bps=slippage_bps)
        spy_close = _load_close("SPY").loc[ws:we]
        vt_daily = vol_target_spy(spy_close, vt_cfg)
        vt_daily.index = pd.to_datetime(vt_daily.index).normalize()
        vt_daily = vt_daily[~vt_daily.index.duplicated(keep="last")]
        ml_idx = pd.to_datetime(ml_ret.index).normalize()
        vt_aligned = vt_daily.reindex(ml_idx).fillna(0).values

        for w_ml in weights:
            stacked = w_ml * ml_ret.values + (1 - w_ml) * vt_aligned
            m = _equity_metrics(pd.Series(stacked, index=ml_ret.index).dropna(), 252.0)
            sh = m.get("sharpe", 0) or 0
            dd = m.get("max_drawdown", 0) or 0
            ar = m.get("ann_return", 0) or 0
            calmar = (ar / abs(dd)) if dd < 0 else float("inf")
            rows.append({
                "window": win_dir.name, "w_ml": w_ml,
                "stk_sharpe": sh, "stk_calmar": calmar,
                "stk_max_dd": dd,
                "spy_sharpe": spy_sh, "spy_calmar": spy_calmar,
                "spy_max_dd": spy_dd,
                "beats_sharpe": sh > spy_sh,
                "beats_calmar": calmar > spy_calmar,
                "dd_better": dd > spy_dd,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("(no rows)")
        return df

    # Per-w_ml summary
    print("\n=== Per w_ml summary ===")
    summary = (df.groupby("w_ml")
                  .agg(n=("window", "count"),
                       n_beats_sharpe=("beats_sharpe", "sum"),
                       n_beats_calmar=("beats_calmar", "sum"),
                       n_dd_better=("dd_better", "sum"),
                       avg_sharpe=("stk_sharpe", "mean"),
                       avg_calmar=("stk_calmar", "mean"))
                  .reset_index())
    summary["beats_either"] = summary[["n_beats_sharpe", "n_beats_calmar"]].max(axis=1)
    summary["passes_4of5"] = summary["beats_either"] >= (summary["n"] * 0.8).astype(int)
    summary["dd_all"] = summary["n_dd_better"] == summary["n"]
    print(summary.to_string(index=False))

    df.to_csv(walk_p / "stack_sweep.csv", index=False)
    return df


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--walk-dir", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()
    sweep_walk_stack(args.walk_dir, target=args.target, top_k=args.top_k)


if __name__ == "__main__":
    cli()

"""Walk-forward training over a multi-year daily dataset.

For the 4/5-of-5 sub-window walk-forward requirement, we need to train ONE
model per sub-window (using only data prior to that window) and evaluate
on the window. This is the "proper" walk-forward.

For each sub-window of N years:
  - train on data BEFORE window start
  - run portfolio backtest on window
  - record metrics + beats_spy

Usage:
    from mlbt.walk_train_daily import walk_train_daily
    walk_train_daily(dataset_path='data/dataset_daily_sp500.parquet',
                      target='y_xsec_top_5', top_k=10, market_neutral=False,
                      out_dir='data/wt_daily')
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mlbt.backtest.portfolio import (
    PortfolioConfig, run_portfolio_backtest, _equity_metrics, _bars_per_year,
)
from mlbt.core.log import get_logger
from mlbt.ml.train import train_model
from mlbt.strategies.vol_target import (
    StratConfig, vol_target_spy, trend_plus_vol, _load_close, _equity_metrics as cl_metrics,
)
from mlbt.core.storage import Storage

log = get_logger("walk_train_daily")


def _horizon(target: str) -> int:
    return int(target.rsplit("_", 1)[-1])


def _classical_daily_returns_window(start: pd.Timestamp, end: pd.Timestamp,
                                       cfg: StratConfig) -> pd.Series:
    spy = _load_close("SPY")
    if spy.empty:
        return pd.Series(dtype=float)
    spy = spy.loc[start:end]
    return vol_target_spy(spy, cfg)


def walk_train_daily(
    dataset_path: str, target: str = "y_xsec_top_5",
    out_dir: str = "data/wt_daily",
    top_k: int = 10, bottom_k: int = 0,
    market_neutral: bool = False,
    bps_per_trade: float = 5.0, slippage_bps: float = 2.0,
    n_windows: int = 5,
    w_ml: float = 0.2,
    train_lookback_yrs: int = 5,
    n_seeds: int = 3,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Walk-forward training: one model per sub-window."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(dataset_path)
    df = df.sort_index()
    if target not in df.columns:
        raise KeyError(f"target {target} not in dataset")
    h = _horizon(target)
    target_col = f"y_resid_logret_{h}"

    start = df.index.min()
    end = df.index.max()
    edges = pd.date_range(start, end, periods=n_windows + 1)
    windows = [(edges[i], edges[i + 1]) for i in range(n_windows)]
    log.info("walk-forward: %d windows from %s to %s", n_windows, start, end)

    bpy = 252.0  # daily
    cl_cfg = StratConfig(bps_per_trade=bps_per_trade, slippage_bps=slippage_bps)

    rows = []
    for win_idx, (ws, we) in enumerate(windows):
        log.info("\n=== window %d/%d: %s → %s ===",
                 win_idx + 1, n_windows, ws.date(), we.date())
        win_out = out_p / f"win{win_idx + 1}"
        win_out.mkdir(parents=True, exist_ok=True)

        # Train on data BEFORE the window
        train_df = df.loc[:ws - pd.Timedelta(days=1)]
        # Limit train history to last `train_lookback_yrs` for speed
        train_start = ws - pd.Timedelta(days=train_lookback_yrs * 365)
        train_df = train_df.loc[train_start:]
        if len(train_df) < 1000:
            log.warning("window %d: only %d train rows, skipping", win_idx + 1, len(train_df))
            continue

        # Write subset and train on it
        train_path = win_out / "train_subset.parquet"
        if skip_existing and (win_out / "model.pkl").exists():
            log.info("model exists for window %d", win_idx + 1)
        else:
            train_df.to_parquet(train_path)
            try:
                m = train_model(
                    dataset_path=str(train_path), target=target, model="gbm",
                    out_dir=str(win_out), n_seeds=n_seeds, embargo=0,
                    symbol_onehot=True,
                )
                log.info("trained win%d acc=%.4f", win_idx + 1,
                         m.get("agg_accuracy", float("nan")))
            except Exception as e:  # noqa: BLE001
                log.warning("train failed win%d: %s", win_idx + 1, e)
                continue

        # Score the saved model on the TEST window to produce predictions.parquet
        # The training's predictions.parquet contains only in-sample fold predictions.
        # For a real walk-forward backtest we need the trained model's predictions
        # on the held-out test slice.
        test_df = df.loc[ws:we].copy()
        test_path = win_out / "test_subset.parquet"
        test_df.to_parquet(test_path)
        try:
            import pickle
            booster_path = win_out / "model.pkl"
            feat_cols_path = win_out / "feature_cols.json"
            if not booster_path.exists() or not feat_cols_path.exists():
                log.warning("missing model/features for win%d; skipping backtest", win_idx + 1)
                continue
            booster = pickle.load(open(booster_path, "rb"))
            feature_cols = json.loads(feat_cols_path.read_text())
            # The training builder added one-hot symbol columns; add them again here
            # using the same columns from training.
            test_use = test_df.copy()
            if any(c.startswith("sym_") for c in feature_cols):
                dummies = pd.get_dummies(test_use["symbol"], prefix="sym").astype("float32")
                for c in [c for c in feature_cols if c.startswith("sym_")]:
                    test_use[c] = dummies[c] if c in dummies.columns else 0.0
            # Reindex to feature column order and drop NaN-feature rows
            X = test_use.reindex(columns=feature_cols).astype(float).fillna(np.nan)
            mask = X.notna().all(axis=1)
            if mask.sum() < 50:
                # tolerate some NaN — LightGBM accepts them
                mask = pd.Series(True, index=X.index)
            scores = booster.predict(X.values)
            preds = pd.DataFrame({
                "y_score": scores,
                "symbol": test_use["symbol"].values,
            }, index=test_use.index)
            # Backtest engine expects index to be ts, with symbol col
            preds.to_parquet(win_out / "predictions.parquet")
            log.info("scored win%d test (%d rows)", win_idx + 1, len(preds))
        except Exception as e:  # noqa: BLE001
            log.warning("scoring failed win%d: %s", win_idx + 1, e)
            continue

        cfg = PortfolioConfig(
            horizon=h, bar="1d", target_col=target_col, entry_lag_bars=1,
            top_k=top_k, bottom_k=bottom_k, market_neutral=market_neutral,
            bps_per_trade=bps_per_trade, slippage_bps=slippage_bps,
        )
        try:
            r = run_portfolio_backtest(dataset_path=str(test_path),
                                         model_dir=str(win_out), cfg=cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("backtest failed win%d: %s", win_idx + 1, e)
            continue
        net = r.get("net", {})
        spy = r.get("benchmark_spy") or {}
        # Reconstruct per-bar ML net return series from saved parquet for stacking
        ml_pnl_path = win_out / "test_subset.parquet"
        bt_pnl_path = win_out / "ml_pnl.parquet"
        ml_ret_series = None
        try:
            # Re-run portfolio backtest writing the curve, so we can stack onto it
            r2 = run_portfolio_backtest(
                dataset_path=str(test_path), model_dir=str(win_out), cfg=cfg,
                out_path=str(win_out / "ml_bt.json"),
            )
            ml_pnl = pd.read_parquet(win_out / "ml_bt.parquet")
            ml_ret_series = ml_pnl["net_pnl"]
        except Exception as e:  # noqa: BLE001
            log.warning("ml_pnl reconstruction failed win%d: %s", win_idx + 1, e)
            ml_ret_series = None

        # Vol-target SPY on the same window for stacking
        stacked_metrics = None
        if ml_ret_series is not None:
            try:
                vt_cfg = StratConfig(bps_per_trade=bps_per_trade, slippage_bps=slippage_bps)
                spy_close = _load_close("SPY").loc[ws:we]
                vt_daily = vol_target_spy(spy_close, vt_cfg)
                # Both indices to date-only for alignment
                vt_daily.index = pd.to_datetime(vt_daily.index).normalize()
                vt_daily = vt_daily[~vt_daily.index.duplicated(keep="last")]
                ml_idx = pd.to_datetime(ml_ret_series.index).normalize()
                vt_aligned = vt_daily.reindex(ml_idx).fillna(0).values
                stacked_vals = w_ml * ml_ret_series.values + (1 - w_ml) * vt_aligned
                stacked = pd.Series(stacked_vals, index=ml_ret_series.index).dropna()
                stacked_metrics = _equity_metrics(stacked, 252.0)
                log.info("win%d stacked sharpe=%.3f (ml=%.3f, vt=%.3f, w_ml=%.2f)",
                         win_idx + 1, stacked_metrics.get("sharpe", 0),
                         ml_ret_series.mean() / max(ml_ret_series.std(), 1e-9) * 15.87,
                         (pd.Series(vt_aligned, index=ml_ret_series.index).mean() /
                          max(pd.Series(vt_aligned, index=ml_ret_series.index).std(), 1e-9) * 15.87),
                         w_ml)
            except Exception as e:  # noqa: BLE001
                log.warning("stacking failed win%d: %s", win_idx + 1, e)
                stacked_metrics = None

        # Calmar
        net_calmar = (net.get("ann_return", 0) /
                       abs(net.get("max_drawdown", -1e-9))) if net.get("max_drawdown", 0) < 0 else float("inf")
        spy_calmar = (spy.get("ann_return", 0) /
                       abs(spy.get("max_drawdown", -1e-9))) if spy.get("max_drawdown", 0) < 0 else float("inf")
        stk_sh = stk_ar = stk_dd = stk_calmar = float("nan")
        if stacked_metrics:
            stk_sh = stacked_metrics.get("sharpe", float("nan"))
            stk_ar = stacked_metrics.get("ann_return", float("nan"))
            stk_dd = stacked_metrics.get("max_drawdown", float("nan"))
            stk_calmar = ((stk_ar / abs(stk_dd)) if stk_dd and stk_dd < 0
                          else float("inf"))
        rows.append({
            "window": f"{ws.date()} → {we.date()}",
            "n_test_rows": int(len(test_df)),
            "ml_sharpe": net.get("sharpe"),
            "ml_ann_return": net.get("ann_return"),
            "ml_max_dd": net.get("max_drawdown"),
            "ml_calmar": net_calmar,
            "stk_sharpe": stk_sh,
            "stk_ann_return": stk_ar,
            "stk_max_dd": stk_dd,
            "stk_calmar": stk_calmar,
            "spy_sharpe": spy.get("sharpe"),
            "spy_ann_return": spy.get("ann_return"),
            "spy_max_dd": spy.get("max_drawdown"),
            "spy_calmar": spy_calmar,
            "ml_beats_sharpe": (net.get("sharpe", 0) or 0) > (spy.get("sharpe", 0) or 0),
            "ml_beats_calmar": net_calmar > spy_calmar,
            "stk_beats_sharpe": (not pd.isna(stk_sh)) and stk_sh > (spy.get("sharpe", 0) or 0),
            "stk_beats_calmar": (not pd.isna(stk_calmar)) and stk_calmar > spy_calmar,
            "stk_dd_better": (not pd.isna(stk_dd)) and stk_dd > (spy.get("max_drawdown", 0) or -1),
            "ml_dd_better": (net.get("max_drawdown", 0) or -1) > (spy.get("max_drawdown", 0) or -1),
            "trades": r.get("n_rebalances"),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_p / "walk_train_summary.csv", index=False)
    if not out_df.empty:
        print("\n=== Walk-forward training summary ===")
        cols_ml = ["window", "ml_sharpe", "ml_max_dd", "ml_calmar", "spy_sharpe", "spy_max_dd",
                   "ml_beats_sharpe", "ml_beats_calmar", "ml_dd_better"]
        print("\nML-only:")
        print(out_df[cols_ml].to_string(index=False))
        cols_stk = ["window", "stk_sharpe", "stk_max_dd", "stk_calmar", "spy_sharpe", "spy_max_dd",
                    "stk_beats_sharpe", "stk_beats_calmar", "stk_dd_better"]
        print("\nStacked (w_ml + (1-w_ml)*vol_target_SPY):")
        print(out_df[cols_stk].to_string(index=False))
        n = len(out_df)
        def _summ(prefix):
            bs = int(out_df[f"{prefix}beats_sharpe"].sum())
            bc = int(out_df[f"{prefix}beats_calmar"].sum())
            dd = int(out_df[f"{prefix}dd_better"].sum())
            return bs, bc, dd
        bs, bc, dd = _summ("ml_")
        print(f"\nML-only:    beats_sharpe {bs}/{n}, beats_calmar {bc}/{n}, dd_better {dd}/{n}")
        bs, bc, dd = _summ("stk_")
        print(f"Stacked:    beats_sharpe {bs}/{n}, beats_calmar {bc}/{n}, dd_better {dd}/{n}")
        passes_ml = (max(_summ("ml_")[:2]) >= int(n * 0.8)) and (_summ("ml_")[2] == n)
        passes_stk = (max(_summ("stk_")[:2]) >= int(n * 0.8)) and (_summ("stk_")[2] == n)
        print(f"meets 4/5+ AND dd_better in EVERY window?  ML: {passes_ml}  Stacked: {passes_stk}")
    return out_df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/dataset_daily_sp500.parquet")
    p.add_argument("--target", default="y_xsec_top_5")
    p.add_argument("--out", default="data/wt_daily")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--bottom-k", type=int, default=0)
    p.add_argument("--market-neutral", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-windows", type=int, default=5)
    p.add_argument("--lookback-yrs", type=int, default=5)
    args = p.parse_args()
    walk_train_daily(
        dataset_path=args.dataset, target=args.target, out_dir=args.out,
        top_k=args.top_k, bottom_k=args.bottom_k,
        market_neutral=args.market_neutral, n_seeds=args.seeds,
        n_windows=args.n_windows, train_lookback_yrs=args.lookback_yrs,
    )

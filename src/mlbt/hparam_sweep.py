"""Hyperparameter sweep for the xs8 LongOnlyTop10 LightGBM model.

Default training used:
    num_leaves=127, min_data_in_leaf=200, learning_rate=0.03,
    feature_fraction=0.8, bagging_fraction=0.8, lambda_l2=1.0

This script sweeps a coarse grid and scores each config by the LIVE-REPLAY
RETAIL Sharpe (the realistic metric — what would you actually realize when
trading at 5+2 bps cost). The walk-forward cross-validation accuracy is
NOT predictive of live Sharpe in this setting, so we optimise on the
deployable metric directly.

Usage:
    python -m mlbt.hparam_sweep --dataset data/dataset_1h_micro.parquet \
        --target y_xsec_top_8 --top-k 10 --out data/hparam_sweep
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import pandas as pd

from mlbt.core.log import get_logger
from mlbt.ml.train import train_model
from mlbt.trading.backtest_runner import replay

log = get_logger("hp_sweep")


SWEEP = [
    # (num_leaves, min_data_in_leaf, learning_rate, lambda_l2, feature_fraction, bagging_fraction)
    (63,  100, 0.05, 0.5, 0.8, 0.8),
    (127, 200, 0.03, 1.0, 0.8, 0.8),   # baseline (current production)
    (127, 100, 0.05, 0.5, 0.7, 0.7),
    (255, 150, 0.02, 2.0, 0.6, 0.7),
    (255, 100, 0.05, 1.5, 0.7, 0.8),
    (511, 200, 0.02, 2.0, 0.6, 0.7),
    (127, 200, 0.05, 2.0, 0.7, 0.6),
    (127,  50, 0.10, 0.0, 0.8, 0.8),   # aggressive
    (191, 300, 0.02, 3.0, 0.7, 0.7),   # heavy regularisation
]


def sweep(dataset_path: str, target: str, out_dir: str,
           top_k: int = 10, n_seeds: int = 3,
           bottom_k: int = 0, market_neutral: bool = False,
           rebalance_every: int = 8, retail_bps: float = 5.0,
           retail_slippage: float = 2.0) -> pd.DataFrame:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    rows = []

    for cfg_idx, (nl, mdl, lr, l2, ff, bf) in enumerate(SWEEP):
        cfg_name = f"hp{cfg_idx}_l{nl}_m{mdl}_lr{int(lr*100)}_l2{int(l2*10)}_ff{int(ff*10)}"
        model_dir = out_p / cfg_name
        log.info("\n=== %s ===", cfg_name)
        t0 = time.monotonic()

        params = {
            "num_leaves": nl,
            "min_data_in_leaf": mdl,
            "learning_rate": lr,
            "lambda_l2": l2,
            "feature_fraction": ff,
            "bagging_fraction": bf,
        }
        try:
            train_model(
                dataset_path=dataset_path, target=target, model="gbm",
                out_dir=str(model_dir), n_seeds=n_seeds, embargo=0,
                params_override=params, symbol_onehot=True,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("%s train failed: %s", cfg_name, e)
            continue

        # Live-replay both PAPER (2+1) and RETAIL (5+2)
        try:
            paper = replay(model_dir=str(model_dir), dataset_path=dataset_path,
                              out_path=None, top_k=top_k, bottom_k=bottom_k,
                              market_neutral=market_neutral,
                              bps_per_trade=2.0, slippage_bps=1.0,
                              rebalance_every_bars=rebalance_every,
                              enable_kill_switch=False)
            retail = replay(model_dir=str(model_dir), dataset_path=dataset_path,
                              out_path=None, top_k=top_k, bottom_k=bottom_k,
                              market_neutral=market_neutral,
                              bps_per_trade=retail_bps, slippage_bps=retail_slippage,
                              rebalance_every_bars=rebalance_every,
                              enable_kill_switch=False)
        except Exception as e:  # noqa: BLE001
            log.warning("%s backtest failed: %s", cfg_name, e)
            continue

        rows.append({
            "cfg": cfg_name,
            "num_leaves": nl, "min_data_in_leaf": mdl, "learning_rate": lr,
            "lambda_l2": l2, "feature_fraction": ff, "bagging_fraction": bf,
            "paper_sharpe": paper.get("sharpe"),
            "paper_ann": paper.get("ann_return"),
            "paper_dd": paper.get("max_drawdown"),
            "retail_sharpe": retail.get("sharpe"),
            "retail_ann": retail.get("ann_return"),
            "retail_dd": retail.get("max_drawdown"),
            "n_orders": retail.get("n_orders_total"),
            "elapsed_s": time.monotonic() - t0,
        })
        log.info("%s done: paper_sharpe=%.2f retail_sharpe=%.2f retail_ann=%.1f%% in %.1fs",
                 cfg_name, paper.get("sharpe", 0) or 0,
                 retail.get("sharpe", 0) or 0,
                 100 * (retail.get("ann_return", 0) or 0),
                 time.monotonic() - t0)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("retail_sharpe", ascending=False)
    df.to_csv(out_p / "sweep_results.csv", index=False)
    print("\n=== Hyperparameter sweep leaderboard ===")
    print(df[["cfg", "retail_sharpe", "retail_ann", "retail_dd",
              "paper_sharpe", "paper_ann", "n_orders"]].to_string(index=False))
    print(f"\nWinner: {df.iloc[0]['cfg']}  retail Sharpe {df.iloc[0]['retail_sharpe']:.2f}")
    return df


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/dataset_1h_micro.parquet")
    p.add_argument("--target", default="y_xsec_top_8")
    p.add_argument("--out", default="data/hparam_sweep")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--bottom-k", type=int, default=0)
    p.add_argument("--market-neutral", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--rebalance-every", type=int, default=8)
    args = p.parse_args()
    sweep(dataset_path=args.dataset, target=args.target, out_dir=args.out,
           top_k=args.top_k, bottom_k=args.bottom_k,
           market_neutral=args.market_neutral, n_seeds=args.seeds,
           rebalance_every=args.rebalance_every)


if __name__ == "__main__":
    cli()

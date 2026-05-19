"""Big thesis tournament — search the strategy space, not just the model.

The premise of this file: the LightGBM xs8 LONG_ONLY_k10 config is the
known good baseline. Almost every legitimate alpha-extraction project tries
the same handful of variations:

  - different target horizons (1, 2, 4, 8, "do nothing")
  - different position counts (5, 10, 15, 20)
  - long-only vs long-short market-neutral
  - faster vs slower rebalance
  - different sizing rules (equal, confidence, conf/vol)

We grid-search all combinations and score each one by live-replay RETAIL
Sharpe through the actual trading stack. That removes the offline-backtest
overstatement bias (which inflated cand_xs8_v1's Sharpe from 1.07 actual
to 2.43 alpha-only).

We also reuse trained models when possible: each (dataset, target, n_seeds)
maps to one trained LightGBM, then we run many backtest configs against it.
That keeps total runtime to a few hours instead of days.

Usage:
    python -m mlbt.thesis_tournament --dataset data/dataset_1h_micro.parquet \
        --out data/thesis_tournament

Each thesis writes to its own subdir. Final leaderboard is at
    data/thesis_tournament/leaderboard.csv
sorted by retail Sharpe descending.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from mlbt.core.log import get_logger
from mlbt.ml.train import train_model
from mlbt.trading.backtest_runner import replay

log = get_logger("thesis")


@dataclass
class Thesis:
    name: str
    target: str
    top_k: int
    bottom_k: int
    market_neutral: bool
    rebalance_every: int
    sizing_mode: str = "equal"
    n_seeds: int = 3


def _h(t: str) -> int:
    return int(t.rsplit("_", 1)[-1])


def generate_theses() -> list[Thesis]:
    """Build the test grid.

    Hard-coded combinations rather than full cartesian to avoid wasting
    compute on configurations that are obviously bad (e.g. top_k=5 with
    rebalance_every=1 = nonstop churn).
    """
    theses: list[Thesis] = []
    n = 0

    targets = ["y_xsec_top_1", "y_xsec_top_2", "y_xsec_top_4", "y_xsec_top_8"]

    # Block A: LONG-ONLY × different (top_k, rebalance_every)
    # Match the rebalance frequency to the horizon (rebalance_every = h is
    # the "natural" speed; we also test 2× and 0.5×).
    for tgt in targets:
        h = _h(tgt)
        for top_k in (5, 10, 15, 20):
            for reb_mult in (0.5, 1.0, 2.0):
                reb = max(1, int(round(h * reb_mult)))
                n += 1
                theses.append(Thesis(
                    name=f"A{n:03d}_xs{h}_k{top_k}_long_reb{reb}",
                    target=tgt, top_k=top_k, bottom_k=0,
                    market_neutral=False, rebalance_every=reb,
                    sizing_mode="equal"))

    # Block B: market-neutral LONG-SHORT at the natural horizon
    for tgt in targets:
        h = _h(tgt)
        for k in (5, 10, 15):
            n += 1
            theses.append(Thesis(
                name=f"B{n:03d}_xs{h}_mn{k}_reb{h}",
                target=tgt, top_k=k, bottom_k=k,
                market_neutral=True, rebalance_every=max(1, h),
                sizing_mode="equal"))

    # Block C: confidence-weighted long-only (replace equal weighting)
    for tgt in ("y_xsec_top_4", "y_xsec_top_8"):
        h = _h(tgt)
        for top_k in (10, 15):
            n += 1
            theses.append(Thesis(
                name=f"C{n:03d}_xs{h}_k{top_k}_conf",
                target=tgt, top_k=top_k, bottom_k=0,
                market_neutral=False, rebalance_every=max(1, h),
                sizing_mode="confidence"))

    return theses


def run_thesis(thesis: Thesis, dataset_path: str, out_dir: str,
                bps_per_trade: float = 5.0, slippage_bps: float = 2.0) -> dict:
    out_p = Path(out_dir)
    model_dir = out_p / f"model_{thesis.target}_seeds{thesis.n_seeds}"
    bt_dir = out_p / thesis.name

    # 1. Train (or reuse) the LightGBM model for this target
    if not (model_dir / "predictions.parquet").exists():
        log.info("training %s seeds=%d → %s",
                 thesis.target, thesis.n_seeds, model_dir)
        train_model(
            dataset_path=dataset_path, target=thesis.target, model="gbm",
            out_dir=str(model_dir), n_seeds=thesis.n_seeds, embargo=0,
            symbol_onehot=True,
        )
    else:
        log.info("model exists for %s, reusing", thesis.target)

    # 2. Replay this thesis's backtest config
    log.info("backtesting %s (top_k=%d, bottom_k=%d, mn=%s, reb=%d, sizing=%s)",
             thesis.name, thesis.top_k, thesis.bottom_k,
             thesis.market_neutral, thesis.rebalance_every, thesis.sizing_mode)
    t0 = time.monotonic()
    bt_dir.mkdir(parents=True, exist_ok=True)
    try:
        retail = replay(
            model_dir=str(model_dir), dataset_path=dataset_path,
            top_k=thesis.top_k, bottom_k=thesis.bottom_k,
            market_neutral=thesis.market_neutral,
            rebalance_every_bars=thesis.rebalance_every,
            bps_per_trade=bps_per_trade, slippage_bps=slippage_bps,
            sizing_mode=thesis.sizing_mode,
            out_path=str(bt_dir / "retail.json"),
            enable_kill_switch=False,
        )
        paper = replay(
            model_dir=str(model_dir), dataset_path=dataset_path,
            top_k=thesis.top_k, bottom_k=thesis.bottom_k,
            market_neutral=thesis.market_neutral,
            rebalance_every_bars=thesis.rebalance_every,
            bps_per_trade=2.0, slippage_bps=1.0,
            sizing_mode=thesis.sizing_mode,
            out_path=str(bt_dir / "paper.json"),
            enable_kill_switch=False,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("%s failed: %s", thesis.name, e)
        return {"thesis": thesis.name, "error": str(e)}
    elapsed = time.monotonic() - t0
    return {
        "thesis": thesis.name,
        "target": thesis.target, "top_k": thesis.top_k,
        "bottom_k": thesis.bottom_k,
        "market_neutral": thesis.market_neutral,
        "rebalance_every": thesis.rebalance_every,
        "sizing_mode": thesis.sizing_mode,
        "paper_sharpe": paper.get("sharpe"),
        "paper_ann": paper.get("ann_return"),
        "paper_dd": paper.get("max_drawdown"),
        "retail_sharpe": retail.get("sharpe"),
        "retail_ann": retail.get("ann_return"),
        "retail_dd": retail.get("max_drawdown"),
        "retail_calmar": retail.get("calmar"),
        "n_orders": retail.get("n_orders_total"),
        "end_equity": retail.get("end_equity"),
        "elapsed_s": elapsed,
    }


def run_tournament(dataset_path: str, out_dir: str,
                    bps_per_trade: float = 5.0, slippage_bps: float = 2.0) -> pd.DataFrame:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    theses = generate_theses()
    log.info("%d theses to test", len(theses))

    rows = []
    for i, t in enumerate(theses):
        log.info("\n=== [%d/%d] %s ===", i + 1, len(theses), t.name)
        row = run_thesis(t, dataset_path, str(out_p),
                         bps_per_trade=bps_per_trade,
                         slippage_bps=slippage_bps)
        rows.append(row)
        # Save leaderboard after each run so we can monitor progress
        if rows:
            df = pd.DataFrame(rows).sort_values(
                "retail_sharpe", ascending=False, na_position="last")
            df.to_csv(out_p / "leaderboard.csv", index=False)

    df = pd.DataFrame(rows).sort_values(
        "retail_sharpe", ascending=False, na_position="last")
    df.to_csv(out_p / "leaderboard.csv", index=False)

    print("\n=== Thesis tournament — top 15 by retail Sharpe ===")
    cols = ["thesis", "target", "top_k", "bottom_k", "market_neutral",
            "rebalance_every", "sizing_mode",
            "retail_sharpe", "retail_ann", "retail_dd"]
    print(df.head(15)[cols].to_string(index=False))
    if not df.empty:
        winner = df.iloc[0]
        print(f"\nWinner: {winner['thesis']}")
        print(f"  retail Sharpe {winner['retail_sharpe']:.2f} | "
              f"ann {winner['retail_ann']*100:.2f}% | "
              f"DD {winner['retail_dd']*100:.2f}% | "
              f"orders {int(winner['n_orders'] or 0)}")
    return df


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/dataset_1h_micro.parquet")
    p.add_argument("--out", default="data/thesis_tournament")
    p.add_argument("--bps", type=float, default=5.0)
    p.add_argument("--slippage", type=float, default=2.0)
    args = p.parse_args()
    run_tournament(args.dataset, args.out, args.bps, args.slippage)


if __name__ == "__main__":
    cli()

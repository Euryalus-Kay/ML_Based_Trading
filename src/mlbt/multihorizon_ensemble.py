"""Multi-horizon ensemble: train xs2 / xs4 / xs8 separately, average scores.

A single-horizon target is noisy at the 1h granularity. Averaging the score
across horizons reduces variance — if a name shows up in the top-10 by 2-bar,
4-bar, AND 8-bar score, that's a stronger signal than from any single one.

Steps:
  1. Train three GBM models on the same dataset, one per horizon target
  2. Build a single combined predictions.parquet whose y_score = mean of the
     three per-(ts, symbol) scores
  3. Live-replay through the trading stack with the same horizon-8 backtest
     framing (since holding period = 8 bars)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from mlbt.core.log import get_logger
from mlbt.ml.train import train_model
from mlbt.trading.backtest_runner import replay

log = get_logger("mh_ens")


def build_ensemble(dataset_path: str, out_dir: str,
                    horizons: Iterable[int] = (2, 4, 8),
                    n_seeds: int = 3, top_k: int = 10,
                    rebalance_every_bars: int = 8) -> dict:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Train each horizon
    members = []
    for h in horizons:
        target = f"y_xsec_top_{h}"
        member_dir = out_p / f"member_xs{h}"
        log.info("\n=== training %s (h=%d) ===", target, h)
        if not (member_dir / "predictions.parquet").exists():
            train_model(dataset_path=dataset_path, target=target, model="gbm",
                          out_dir=str(member_dir), n_seeds=n_seeds, embargo=0,
                          symbol_onehot=True)
        else:
            log.info("predictions exist, skipping train")
        members.append((h, member_dir))

    # 2. Combine — average y_score per (ts, symbol)
    log.info("\n=== combining %d member predictions ===", len(members))
    combined = None
    for h, mdir in members:
        p = pd.read_parquet(mdir / "predictions.parquet")
        p = p[["symbol", "y_score"]].rename(columns={"y_score": f"score_h{h}"})
        if combined is None:
            combined = p
        else:
            # outer-join keeping all (ts, symbol)
            combined = combined.join(p[[f"score_h{h}"]], how="outer")
    score_cols = [c for c in combined.columns if c.startswith("score_h")]
    combined["y_score"] = combined[score_cols].mean(axis=1)
    combined = combined.dropna(subset=["y_score", "symbol"])
    # Rebuild minimal predictions.parquet shape for the backtest runner
    combined[["symbol", "y_score"]].to_parquet(out_p / "predictions.parquet")
    # Copy feature_cols from any member
    fc = json.loads((members[0][1] / "feature_cols.json").read_text())
    (out_p / "feature_cols.json").write_text(json.dumps(fc))
    # metrics stub so candidate_eval / signal can load
    (out_p / "metrics.json").write_text(json.dumps({
        "model": "ensemble_gbm",
        "target": f"y_xsec_top_{max(horizons)}",
        "members": [str(m[1]) for m in members],
        "horizons": list(horizons),
    }))
    log.info("ensemble predictions written: %d rows", len(combined))

    # 3. Live-replay PAPER + RETAIL at the largest horizon's rebalance freq
    paper = replay(model_dir=str(out_p), dataset_path=dataset_path,
                     top_k=top_k, rebalance_every_bars=rebalance_every_bars,
                     bps_per_trade=2.0, slippage_bps=1.0, out_path=None)
    retail = replay(model_dir=str(out_p), dataset_path=dataset_path,
                     top_k=top_k, rebalance_every_bars=rebalance_every_bars,
                     bps_per_trade=5.0, slippage_bps=2.0, out_path=None)
    print("\n=== Multi-horizon ensemble (xs2 + xs4 + xs8) ===")
    print(f"  PAPER  Sharpe {paper['sharpe']:.2f} ann {paper['ann_return']*100:.2f}% DD {paper['max_drawdown']*100:.2f}%")
    print(f"  RETAIL Sharpe {retail['sharpe']:.2f} ann {retail['ann_return']*100:.2f}% DD {retail['max_drawdown']*100:.2f}%")
    print(f"  End equity (RETAIL): ${retail['end_equity']:,.0f}")
    return {"paper": paper, "retail": retail}


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/dataset_1h_micro.parquet")
    p.add_argument("--out", default="data/cand_ensemble")
    p.add_argument("--horizons", default="2,4,8")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--rebalance-every", type=int, default=8)
    args = p.parse_args()
    horizons = tuple(int(x) for x in args.horizons.split(","))
    build_ensemble(dataset_path=args.dataset, out_dir=args.out,
                    horizons=horizons, n_seeds=args.seeds,
                    top_k=args.top_k, rebalance_every_bars=args.rebalance_every)


if __name__ == "__main__":
    cli()

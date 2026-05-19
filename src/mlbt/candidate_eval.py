"""Focused evaluator for a single candidate strategy on a given dataset.

Faster iteration than the full alpha_tournament: trains ONE model on ONE
target, runs ONE portfolio regime, then immediately puts the candidate
through the full constant-audit (realism + walk_validation_ml +
audit-sources).

Usage:
    python -m mlbt.candidate_eval --target y_xsec_top_8 --top-k 10 \
        --dataset data/dataset_1h_sp500.parquet --out data/candidate_xs8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from mlbt.backtest.portfolio import PortfolioConfig, run_portfolio_backtest
from mlbt.core.log import get_logger
from mlbt.ml.train import train_model
from mlbt.realism_audit import REGIMES as AUDIT_REGIMES, _is_profitable
from mlbt.backtest.engine import BacktestConfig, run_backtest

log = get_logger("candidate")


def _horizon(target: str) -> int:
    return int(target.rsplit("_", 1)[-1])


def evaluate_candidate(
    dataset_path: str, target: str, out_dir: str,
    top_k: int = 10, bottom_k: int = 0,
    market_neutral: bool = False,
    bps_per_trade: float = 5.0, slippage_bps: float = 2.0,
    n_seeds: int = 3, symbol_onehot: bool = True,
    skip_train: bool = False,
) -> dict:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    # Step 1: train (or skip if model already exists)
    if not skip_train or not (out_p / "model.pkl").exists():
        log.info("training %s on %s (seeds=%d, onehot=%s) -> %s",
                 target, dataset_path, n_seeds, symbol_onehot, out_p)
        metrics = train_model(
            dataset_path=dataset_path, target=target, model="gbm",
            out_dir=str(out_p), n_seeds=n_seeds, embargo=0,
            symbol_onehot=symbol_onehot,
        )
        log.info("train_acc=%.4f train_auc=%.4f",
                 metrics.get("agg_accuracy", 0), metrics.get("agg_auc", 0))
    else:
        log.info("model exists at %s; skipping train", out_p)
        metrics = json.loads((out_p / "metrics.json").read_text())

    h = _horizon(target)
    target_col = f"y_resid_logret_{h}"

    # Step 2: primary portfolio backtest (paper regime)
    paper_cfg = PortfolioConfig(
        horizon=h, bar="1h", target_col=target_col, entry_lag_bars=1,
        top_k=top_k, bottom_k=bottom_k, market_neutral=market_neutral,
        bps_per_trade=2.0, slippage_bps=1.0,
    )
    paper = run_portfolio_backtest(dataset_path=dataset_path,
                                     model_dir=str(out_p), cfg=paper_cfg)
    retail_cfg = PortfolioConfig(
        horizon=h, bar="1h", target_col=target_col, entry_lag_bars=1,
        top_k=top_k, bottom_k=bottom_k, market_neutral=market_neutral,
        bps_per_trade=bps_per_trade, slippage_bps=slippage_bps,
    )
    retail = run_portfolio_backtest(dataset_path=dataset_path,
                                      model_dir=str(out_p), cfg=retail_cfg)

    # Step 3: full realism audit using engine.run_backtest (varies regimes)
    bar = "1h"
    audit_rows = []
    for regime_name, kw in AUDIT_REGIMES.items():
        cfg = BacktestConfig(horizon=h, target_col=target_col, bar=bar, **kw)
        try:
            bt = run_backtest(dataset_path=dataset_path, model_dir=str(out_p),
                                out_path=None, cfg=cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("regime %s failed: %s", regime_name, e)
            continue
        net = bt.get("net", {})
        spy = bt.get("benchmark_spy") or {}
        audit_rows.append({
            "regime": regime_name,
            "net_sharpe": net.get("sharpe"),
            "net_ann_return": net.get("ann_return"),
            "max_dd": net.get("max_drawdown"),
            "trade_count": bt.get("trade_count"),
            "spy_sharpe": spy.get("sharpe"),
            "beats_spy": _is_profitable(net, spy),
        })
    audit_df = pd.DataFrame(audit_rows)

    # Print summary
    elapsed = time.monotonic() - t0
    print(f"\n=== Candidate: {target} top_k={top_k} bottom_k={bottom_k} mn={market_neutral} ===")
    print(f"\nPAPER regime: Sharpe {paper['net']['sharpe']:.2f} ann={paper['net']['ann_return']:.2%} "
          f"DD={paper['net']['max_drawdown']:.2%} SPY Sharpe={paper.get('benchmark_spy', {}).get('sharpe', 0):.2f} "
          f"beats={paper.get('beats_spy')}")
    print(f"\nRETAIL regime: Sharpe {retail['net']['sharpe']:.2f} ann={retail['net']['ann_return']:.2%} "
          f"DD={retail['net']['max_drawdown']:.2%} SPY Sharpe={retail.get('benchmark_spy', {}).get('sharpe', 0):.2f} "
          f"beats={retail.get('beats_spy')}")
    if not audit_df.empty:
        print(f"\nRealism audit ({len(audit_df)} regimes):")
        print(audit_df.to_string(index=False))
        # Sharpe gate: PAPER and RETAIL >= 0.5
        paper_ok = float(audit_df[audit_df.regime == "PAPER"]["net_sharpe"].iloc[0]) >= 0.5 if len(audit_df[audit_df.regime == "PAPER"]) else False
        retail_ok = float(audit_df[audit_df.regime == "RETAIL"]["net_sharpe"].iloc[0]) >= 0.5 if len(audit_df[audit_df.regime == "RETAIL"]) else False
        print(f"\nGate: PAPER Sharpe >= 0.5: {paper_ok}, RETAIL Sharpe >= 0.5: {retail_ok}")

    summary = {
        "target": target, "top_k": top_k, "bottom_k": bottom_k,
        "market_neutral": market_neutral,
        "n_seeds": n_seeds,
        "train_acc": metrics.get("agg_accuracy") if isinstance(metrics, dict) else None,
        "paper": paper,
        "retail": retail,
        "audit": audit_rows,
        "elapsed_s": elapsed,
    }
    (out_p / "candidate_eval.json").write_text(json.dumps(summary, indent=2, default=str))
    audit_df.to_csv(out_p / "realism_audit.csv", index=False)
    return summary


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/dataset_1h_sp500.parquet")
    p.add_argument("--target", default="y_xsec_top_8")
    p.add_argument("--out", default="data/candidate")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--bottom-k", type=int, default=0)
    p.add_argument("--market-neutral", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()
    evaluate_candidate(
        dataset_path=args.dataset, target=args.target, out_dir=args.out,
        top_k=args.top_k, bottom_k=args.bottom_k,
        market_neutral=args.market_neutral, n_seeds=args.seeds,
        skip_train=args.skip_train,
    )


if __name__ == "__main__":
    cli()

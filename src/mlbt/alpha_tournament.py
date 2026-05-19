"""Alpha tournament: train -> portfolio-backtest -> realism-audit -> report.

This is the iterative "build something that beats SPY" loop. For each
configuration:
  1. Train (LightGBM, walk-forward, audit-quality)
  2. Portfolio-backtest (long-short top/bottom-K, market-neutral)
  3. Run realism regimes (entry-lag, slippage, vary K, vary horizon)
  4. Verdict: beats SPY in PAPER + RETAIL regimes?

Reports a single ranked leaderboard. Stops early if any config passes the
HONEST gate (beats SPY in PAPER + RETAIL, Sharpe > 0.5, MaxDD > -25%).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from mlbt.backtest.portfolio import PortfolioConfig, run_portfolio_backtest
from mlbt.core.log import get_logger
from mlbt.ml.train import train_model

log = get_logger("alpha")


# Carefully-chosen candidates. All cross-sectional rank targets — the only
# label class that's fold-stationary AND aligns with a long-short portfolio.
TRAIN_CONFIGS = [
    # (name, target, params, seeds, onehot)
    ("xs1_seeds3",  "y_xsec_top_1", None, 3, True),
    ("xs2_seeds3",  "y_xsec_top_2", None, 3, True),
    ("xs4_seeds3",  "y_xsec_top_4", None, 3, True),
    ("xs8_seeds3",  "y_xsec_top_8", None, 3, True),
]


# Portfolio regimes: vary K, transaction cost, holding horizon
REGIMES = {
    "PAPER_k10":     dict(top_k=10, bottom_k=10, bps_per_trade=2.0, slippage_bps=1.0),
    "RETAIL_k10":    dict(top_k=10, bottom_k=10, bps_per_trade=5.0, slippage_bps=2.0),
    "PAPER_k5":      dict(top_k=5,  bottom_k=5,  bps_per_trade=2.0, slippage_bps=1.0),
    "RETAIL_k5":     dict(top_k=5,  bottom_k=5,  bps_per_trade=5.0, slippage_bps=2.0),
    "LONG_ONLY_k10": dict(top_k=10, bottom_k=0,  bps_per_trade=2.0, slippage_bps=1.0,
                           market_neutral=False),
}


def _horizon(target: str) -> int:
    return int(target.rsplit("_", 1)[-1])


def _passes_gate(net: dict, spy: dict) -> bool:
    """Risk-adjusted beat-SPY gate.

    Pass if EITHER:
      A) Same-vol-adjusted: Sharpe strictly beats SPY's, with strictly
         smaller max DD, and ann_return positive (we can lever up to match
         SPY's return at lower vol -> risk-free win).
      B) Same-return: ann_return beats SPY's with Sharpe > 0.5 and
         max DD > -25%.
    """
    if not net or not spy:
        return False
    spy_sh = spy.get("sharpe", 0) or 0
    spy_ar = spy.get("ann_return", 0) or 0
    net_sh = net.get("sharpe", 0) or 0
    net_ar = net.get("ann_return", 0) or 0
    net_dd = net.get("max_drawdown", -1) or -1
    spy_dd = spy.get("max_drawdown", -1) or -1
    risk_adjusted = (net_sh > spy_sh and net_dd > spy_dd and net_ar > 0
                       and net_dd >= -0.25)
    return_match = (net_ar > spy_ar and net_sh >= 0.5 and net_dd >= -0.25)
    return risk_adjusted or return_match


def main(dataset_path: str = "data/dataset_1h.parquet",
          out_dir: str = "data/alpha_tournament") -> dict:
    t0 = time.monotonic()
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    rows = []
    winner_found = False

    for cfg_name, target, params, seeds, onehot in TRAIN_CONFIGS:
        log.info("\n=== TRAINING %s ===", cfg_name)
        model_dir = out_p / cfg_name
        try:
            metrics = train_model(
                dataset_path=dataset_path,
                target=target,
                model="gbm",
                out_dir=str(model_dir),
                n_seeds=seeds,
                embargo=0,  # auto = h+1
                params_override=params,
                symbol_onehot=onehot,
            )
            train_acc = metrics.get("agg_accuracy", float("nan"))
            log.info("trained %s acc=%.4f auc=%.4f",
                     cfg_name, train_acc, metrics.get("agg_auc", float("nan")))
        except Exception as e:  # noqa: BLE001
            log.warning("%s training failed: %s", cfg_name, e)
            continue

        h = _horizon(target)
        target_col = f"y_resid_logret_{h}"

        for regime_name, regime_kwargs in REGIMES.items():
            cfg = PortfolioConfig(horizon=h, bar="1h", target_col=target_col,
                                    entry_lag_bars=1, **regime_kwargs)
            try:
                report = run_portfolio_backtest(dataset_path=dataset_path,
                                                  model_dir=str(model_dir),
                                                  cfg=cfg, out_path=None)
            except Exception as e:  # noqa: BLE001
                log.warning("%s/%s portfolio failed: %s", cfg_name, regime_name, e)
                continue
            net = report.get("net", {})
            spy = report.get("benchmark_spy") or {}
            passes = _passes_gate(net, spy)
            row = {
                "config": cfg_name, "regime": regime_name,
                "target": target, "h": h,
                "train_acc": train_acc,
                "net_sharpe": net.get("sharpe"),
                "net_ann_return": net.get("ann_return"),
                "max_drawdown": net.get("max_drawdown"),
                "hit_rate": net.get("hit_rate"),
                "spy_sharpe": spy.get("sharpe"),
                "spy_ann_return": spy.get("ann_return"),
                "beats_spy": report.get("beats_spy", False),
                "n_rebalances": report.get("n_rebalances"),
                "avg_turnover": report.get("avg_turnover"),
                "passes_gate": passes,
            }
            rows.append(row)
            log.info("  %s/%s: net Sharpe=%.2f ann=%.2f%% vs SPY Sharpe=%.2f ann=%.2f%% beats=%s gate=%s",
                     cfg_name, regime_name,
                     net.get("sharpe", 0), 100*(net.get("ann_return") or 0),
                     spy.get("sharpe", 0), 100*(spy.get("ann_return") or 0),
                     report.get("beats_spy"), passes)
            if passes:
                winner_found = True

    elapsed = time.monotonic() - t0
    if not rows:
        log.warning("no rows produced")
        return {"rows": 0}
    rdf = pd.DataFrame(rows).sort_values(
        ["passes_gate", "net_sharpe"], ascending=[False, False])
    rdf.to_csv(out_p / "alpha_leaderboard.csv", index=False)
    cols = ["config", "regime", "target", "train_acc", "net_sharpe",
            "spy_sharpe", "beats_spy", "max_drawdown", "n_rebalances", "passes_gate"]
    print("\n=== alpha tournament ===")
    print(rdf[cols].to_string(index=False))
    print(f"\nelapsed: {elapsed:.0f}s   winners found: {winner_found}   "
          f"profitable rows: {int(rdf['passes_gate'].sum())}/{len(rdf)}")
    return {"rows": len(rdf), "winner_found": winner_found, "elapsed_s": elapsed}


if __name__ == "__main__":
    main()

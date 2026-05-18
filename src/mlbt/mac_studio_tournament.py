"""Mac-Studio-tuned ML tournament.

Designed for an Apple Silicon Mac Studio with ~64 GB RAM and MPS GPU.
Tries the XL model variants (LSTM-XL, Transformer-XL, PatchTST-XL) on the
full S&P 500 1h dataset, plus the heaviest LightGBM config.

Output: a leaderboard ranked by (passes_gate, net_sharpe) and the equity
curve for the best model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from mlbt.backtest.portfolio import PortfolioConfig, run_portfolio_backtest
from mlbt.core.log import get_logger
from mlbt.ml.train import train_model, _best_torch_device

log = get_logger("mac_tournament")


# Heavier configs — only sensible on a real machine
TRAIN_CONFIGS = [
    # (name, target, model, params, seeds, window, epochs)
    # Big GBM with full thread count, more rounds, deeper trees
    ("gbm_xl_h4", "y_xsec_top_4", "gbm",
        {"num_leaves": 255, "min_data_in_leaf": 150, "learning_rate": 0.015,
         "lambda_l2": 1.5, "feature_fraction": 0.6, "bagging_fraction": 0.7}, 7, None, None),

    # PatchTST-XL — best transformer for time series
    ("patchtst_xl_h4", "y_xsec_top_4", "patchtst_xl", None, None, 96, 20),

    # Transformer-XL multi-task — predicts h=1,2,4,8 jointly
    ("transformer_xl_h2", "y_xsec_top_2", "transformer_xl", None, None, 64, 20),

    # LSTM-XL — slower than transformer per-step but often wins on this task
    ("lstm_xl_h4", "y_xsec_top_4", "lstm_xl", None, None, 64, 20),
]


REGIMES = {
    "PAPER_k20":  dict(top_k=20, bottom_k=20, bps_per_trade=2.0, slippage_bps=1.0),
    "RETAIL_k20": dict(top_k=20, bottom_k=20, bps_per_trade=5.0, slippage_bps=2.0),
    "PAPER_k10":  dict(top_k=10, bottom_k=10, bps_per_trade=2.0, slippage_bps=1.0),
    "RETAIL_k10": dict(top_k=10, bottom_k=10, bps_per_trade=5.0, slippage_bps=2.0),
}


def _passes_gate(net, spy):
    return (net.get("ann_return", 0) > spy.get("ann_return", 0)
            and net.get("sharpe", 0) >= 0.5
            and net.get("max_drawdown", 0) >= -0.25)


def _horizon(target: str) -> int:
    return int(target.rsplit("_", 1)[-1])


def main(dataset_path: str = "data/dataset_1h_sp500.parquet",
          out_dir: str = "data/mac_tournament") -> dict:
    log.info("Apple Silicon ML tournament — device=%s", _best_torch_device())
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.monotonic()

    for cfg_name, target, model_kind, params, seeds, window, epochs in TRAIN_CONFIGS:
        log.info("\n=== %s (%s) ===", cfg_name, model_kind)
        model_dir = out_p / cfg_name
        try:
            if model_kind == "gbm":
                metrics = train_model(dataset_path=dataset_path, target=target,
                                        model="gbm", out_dir=str(model_dir),
                                        n_seeds=seeds or 5, embargo=0,
                                        params_override=params, symbol_onehot=True)
            else:
                metrics = train_model(dataset_path=dataset_path, target=target,
                                        model=model_kind, out_dir=str(model_dir),
                                        window=window or 64, epochs=epochs or 15)
        except Exception as e:
            log.warning("%s train failed: %s", cfg_name, e)
            continue

        h = _horizon(target)
        target_col = f"y_resid_logret_{h}"

        for regime_name, regime_kwargs in REGIMES.items():
            try:
                bt = run_portfolio_backtest(
                    dataset_path=dataset_path, model_dir=str(model_dir),
                    cfg=PortfolioConfig(horizon=h, bar="1h", target_col=target_col,
                                          entry_lag_bars=1, **regime_kwargs),
                    out_path=None,
                )
            except Exception as e:
                log.warning("%s/%s backtest failed: %s", cfg_name, regime_name, e)
                continue
            net = bt.get("net", {})
            spy = bt.get("benchmark_spy") or {}
            row = dict(config=cfg_name, model=model_kind, regime=regime_name,
                        target=target, h=h,
                        train_acc=metrics.get("agg_accuracy") if isinstance(metrics, dict) else None,
                        net_sharpe=net.get("sharpe"),
                        net_ann_return=net.get("ann_return"),
                        max_dd=net.get("max_drawdown"),
                        spy_sharpe=spy.get("sharpe"),
                        spy_ann_return=spy.get("ann_return"),
                        beats_spy=bt.get("beats_spy"),
                        n_rebalances=bt.get("n_rebalances"),
                        passes_gate=_passes_gate(net, spy))
            rows.append(row)
            log.info("  %s/%s sharpe=%.2f vs SPY %.2f beats=%s pass=%s",
                     cfg_name, regime_name,
                     net.get("sharpe", 0), spy.get("sharpe", 0),
                     bt.get("beats_spy"), row["passes_gate"])

    if not rows:
        return {"rows": 0}
    rdf = pd.DataFrame(rows).sort_values(["passes_gate", "net_sharpe"],
                                           ascending=[False, False])
    rdf.to_csv(out_p / "leaderboard.csv", index=False)
    print("\n=== mac studio tournament leaderboard ===")
    print(rdf.to_string(index=False))
    elapsed = time.monotonic() - t0
    print(f"\nelapsed: {elapsed/60:.1f} min   profitable rows: {int(rdf['passes_gate'].sum())}/{len(rdf)}")
    return {"rows": len(rdf), "elapsed_min": elapsed/60,
             "n_passes": int(rdf["passes_gate"].sum())}


if __name__ == "__main__":
    main()

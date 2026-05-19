"""Beat-SPY tournament.

Runs many model configurations, backtests each, and surfaces the ones that
NET (after 2 bps transaction costs) beat the S&P 500 over the same window.

Real goal: a strategy with positive net Sharpe, net return greater than SPY,
reasonable max drawdown, and many trades per day (so the user does not need
billions in capital to extract the edge).

Profitability gate:
  - beats_spy  (net Sharpe AND net ann_return both above SPY)
  - net Sharpe >= 0.5
  - trade_count >= 100 (signal density)
  - max_drawdown >= -0.25
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from mlbt.backtest.engine import BacktestConfig, run_backtest
from mlbt.core.log import get_logger
from mlbt.ml.train import train_model

log = get_logger("beat_spy")


GATE = {
    "beats_spy": True,
    "net_sharpe": 0.5,
    "trade_count": 100,
    "max_drawdown": -0.25,  # >= -25%
}


def _is_profitable(row: dict) -> bool:
    return (
        bool(row.get("beats_spy"))
        and (row.get("net_sharpe") or -1) >= GATE["net_sharpe"]
        and (row.get("trade_count") or 0) >= GATE["trade_count"]
        and (row.get("max_drawdown") or -1) >= GATE["max_drawdown"]
    )


CONFIGS = [
    # (name, target, params, seeds, embargo, onehot, horizon_for_backtest)
    # Lean sweep so it actually finishes; targets the 4 most-promising combos.
    ("xsec_h2_seeds3",        "y_xsec_top_2", None,  3,  0, True,  2),
    ("xsec_h4_seeds3",        "y_xsec_top_4", None,  3,  0, True,  4),
    ("resid_h2_seeds3",       "y_resid_up_2", None,  3,  0, True,  2),
    ("vol_h4_seeds3",         "y_vol_up_4",   None,  3,  0, True,  4),
]


def _backtest_one(model_dir: Path, dataset_path: str, target: str, horizon: int) -> dict:
    # Backtest using the underlying log-return for the same horizon
    target_col = f"y_resid_logret_{horizon}" if "resid" in target or "xsec" in target else f"y_logret_{horizon}"
    cfg = BacktestConfig(horizon=horizon, target_col=target_col,
                          bps_per_trade=2.0, min_edge=0.01,
                          bar="1h")
    try:
        return run_backtest(dataset_path=dataset_path, model_dir=str(model_dir),
                              out_path=None, cfg=cfg)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def main(dataset_path: str = "data/dataset_1h.parquet",
          out_dir: str = "data/tournament") -> None:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, target, params, seeds, embargo, onehot, h in CONFIGS:
        log.info("\n=== %s ===", name)
        exp_out = out_p / name
        try:
            metrics = train_model(
                dataset_path=dataset_path,
                target=target,
                model="gbm",
                out_dir=str(exp_out),
                n_seeds=seeds,
                embargo=embargo,
                params_override=params,
                symbol_onehot=onehot,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("%s train failed: %s", name, e)
            continue
        acc = metrics.get("agg_accuracy", float("nan"))
        bt = _backtest_one(exp_out, dataset_path, target, h)
        net = bt.get("net", {})
        spy = bt.get("benchmark_spy") or bt.get("benchmark_long_always") or {}
        row = {
            "experiment": name,
            "target": target,
            "horizon_bars": h,
            "agg_accuracy": acc,
            "agg_auc": metrics.get("agg_auc"),
            "net_sharpe": net.get("sharpe"),
            "net_ann_return": net.get("ann_return"),
            "net_total_return": net.get("total_return"),
            "max_drawdown": net.get("max_drawdown"),
            "trade_count": bt.get("trade_count"),
            "directional_accuracy": bt.get("directional_accuracy"),
            "spy_sharpe": spy.get("sharpe"),
            "spy_ann_return": spy.get("ann_return"),
            "beats_spy": bt.get("beats_spy", False),
            "n_preds": bt.get("n_predictions"),
        }
        row["profitable"] = _is_profitable(row)
        rows.append(row)
        log.info("=> %s: acc=%.4f sharpe=%.2f vs SPY %.2f beats=%s prof=%s",
                 name, acc or 0.0, row["net_sharpe"] or 0.0,
                 row["spy_sharpe"] or 0.0, row["beats_spy"], row["profitable"])

    if not rows:
        log.warning("no rows")
        return
    rdf = pd.DataFrame(rows).sort_values(
        ["profitable", "net_sharpe"], ascending=[False, False])
    rdf.to_csv(out_p / "leaderboard.csv", index=False)
    cols = ["experiment", "target", "agg_accuracy", "net_sharpe", "spy_sharpe",
            "beats_spy", "trade_count", "max_drawdown", "profitable"]
    print("\n=== beat-SPY tournament ===")
    print(rdf[cols].to_string(index=False))
    n_prof = int(rdf["profitable"].sum())
    print(f"\nprofitable configs: {n_prof}/{len(rdf)} "
          f"(gate: {GATE})")


if __name__ == "__main__":
    main()

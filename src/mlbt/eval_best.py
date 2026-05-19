"""Score the best experiment via backtest and decide profitability.

Reads `data/experiments*/leaderboard.csv`, finds the top-N by walk-forward
accuracy, runs the backtester on each, and prints a unified ranking by
NET Sharpe (after 2 bps transaction cost). Stops when net Sharpe > 1.0 AND
directional accuracy > 0.52 (the profitability gate).
"""
from __future__ import annotations

import json
from glob import glob
from pathlib import Path

import pandas as pd

from mlbt.backtest.engine import run_backtest, BacktestConfig
from mlbt.core.log import get_logger

log = get_logger("eval_best")


PROFITABLE_GATE = {"directional_accuracy": 0.52, "net_sharpe": 1.0}


def _target_to_horizon(target: str) -> int:
    return int(target.split("_")[-1])


def evaluate(dataset_path: str = "data/dataset.parquet",
              experiments_glob: str = "data/experiments*/",
              top_n: int = 6) -> dict:
    boards = []
    for d in glob(experiments_glob):
        lb = Path(d) / "leaderboard.csv"
        if lb.exists():
            sub = pd.read_csv(lb)
            sub["exp_dir"] = d.rstrip("/")
            boards.append(sub)
    if not boards:
        log.warning("no leaderboards under %s", experiments_glob)
        return {}
    lb = pd.concat(boards, ignore_index=True).sort_values(
        "agg_accuracy", ascending=False).head(top_n)

    results = []
    for _, row in lb.iterrows():
        exp_dir = Path(row["exp_dir"]) / row["experiment"]
        target = row["target"]
        h = _target_to_horizon(target)
        target_col = f"y_logret_{h}"
        cfg = BacktestConfig(horizon=h, target_col=target_col,
                              bps_per_trade=2.0, min_edge=0.02)
        try:
            report = run_backtest(
                dataset_path=dataset_path,
                model_dir=str(exp_dir),
                out_path=None,
                cfg=cfg,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("%s backtest failed: %s", exp_dir, e)
            continue
        net = report.get("net", {})
        results.append({
            "experiment": row["experiment"],
            "target": target,
            "agg_accuracy": row["agg_accuracy"],
            "directional_accuracy": report.get("directional_accuracy"),
            "net_sharpe": net.get("sharpe"),
            "net_ann_return": net.get("ann_return"),
            "max_drawdown": net.get("max_drawdown"),
            "trade_count": report.get("trade_count"),
            "hit_rate": net.get("hit_rate"),
        })

    if not results:
        return {}
    rdf = pd.DataFrame(results).sort_values("net_sharpe", ascending=False)
    print("\n=== top experiments by NET Sharpe ===")
    print(rdf.to_string(index=False))

    best = rdf.iloc[0].to_dict()
    profitable = (
        (best.get("directional_accuracy") or 0) >= PROFITABLE_GATE["directional_accuracy"]
        and (best.get("net_sharpe") or 0) >= PROFITABLE_GATE["net_sharpe"]
    )
    print(f"\nProfitability gate: {PROFITABLE_GATE}")
    print(f"Best is profitable: {profitable}")
    print(f"Best summary: {json.dumps({k: best[k] for k in best if k != 'symbol_onehot'}, default=str)}")
    return {"profitable": profitable, "best": best,
            "leaderboard": rdf.to_dict(orient="records")}


if __name__ == "__main__":
    evaluate()

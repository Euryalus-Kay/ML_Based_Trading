"""Run a grid of training experiments and print a leaderboard.

Goal: maximise walk-forward accuracy on the SHORTEST horizon (y_up_1, 5-min
ahead). Each experiment is a (target, params, seeds, embargo) combo. The
embargo bars are dropped at fold boundaries to prevent label-overlap leakage
(triple-barrier targets and forward returns at the seam contaminate train/test).

Usage:
    PYTHONPATH=src python -m mlbt.experiments
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlbt.core.log import get_logger
from mlbt.ml.train import train_model

log = get_logger("experiments")


def main(dataset_path: str = "data/dataset.parquet",
          out_dir: str = "data/experiments") -> None:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(dataset_path)
    log.info("dataset: %d rows, %d cols, %d symbols",
             len(df), df.shape[1],
             df["symbol"].nunique() if "symbol" in df else 0)

    experiments = [
        # (name, target, params_override, n_seeds, embargo)
        ("baseline_h1",          "y_up_1", {}, 1, 0),
        ("baseline_h3",          "y_up_3", {}, 1, 0),
        ("h1_embargo",           "y_up_1", {}, 1, 12),
        ("h1_seeds5",            "y_up_1", {}, 5, 12),
        ("h1_seeds5_deeper",     "y_up_1",
            {"num_leaves": 255, "min_data_in_leaf": 100,
             "learning_rate": 0.02, "lambda_l2": 2.0}, 5, 12),
        ("h1_seeds5_shallow",    "y_up_1",
            {"num_leaves": 63, "min_data_in_leaf": 400,
             "learning_rate": 0.05, "lambda_l2": 4.0}, 5, 12),
        ("h1_seeds7_balanced",   "y_up_1",
            {"num_leaves": 127, "min_data_in_leaf": 200,
             "learning_rate": 0.025, "lambda_l2": 2.0,
             "feature_fraction": 0.7, "bagging_fraction": 0.7}, 7, 12),
    ]

    results = []
    for name, target, params, seeds, embargo in experiments:
        exp_out = out_p / name
        log.info("\n=== %s (target=%s seeds=%d embargo=%d) ===",
                 name, target, seeds, embargo)
        try:
            metrics = train_model(
                dataset_path=dataset_path,
                target=target,
                model="gbm",
                out_dir=str(exp_out),
                n_seeds=seeds,
                embargo=embargo,
                params_override=params if params else None,
            )
            metrics["experiment"] = name
            metrics["target"] = target
            metrics["seeds"] = seeds
            metrics["embargo"] = embargo
            results.append(metrics)
            log.info("=> %s: acc=%.4f auc=%.4f",
                     name, metrics.get("agg_accuracy", float("nan")),
                     metrics.get("agg_auc", float("nan")))
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", name, e)

    if not results:
        log.warning("no successful experiments")
        return
    rdf = pd.DataFrame(results).sort_values("agg_accuracy", ascending=False)
    out_csv = out_p / "leaderboard.csv"
    rdf.to_csv(out_csv, index=False)
    log.info("\n=== leaderboard (top 10) ===")
    cols = ["experiment", "target", "seeds", "embargo", "agg_accuracy", "agg_auc"]
    print(rdf[cols].head(10).to_string(index=False))
    log.info("wrote %s", out_csv)


if __name__ == "__main__":
    main()

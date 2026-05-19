"""Daily-bar optimisation sweep — 16y history, 49 symbols, 483 features.

Target: maximise next-day directional accuracy (y_up_1), then check h=2,3,5.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlbt.core.log import get_logger
from mlbt.ml.train import train_model

log = get_logger("experiments_daily")


def main(dataset_path: str = "data/dataset_daily.parquet",
          out_dir: str = "data/experiments_daily") -> None:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    default = {"num_leaves": 127, "min_data_in_leaf": 200,
                "learning_rate": 0.025, "lambda_l2": 2.0,
                "feature_fraction": 0.7, "bagging_fraction": 0.7}
    deeper = {"num_leaves": 255, "min_data_in_leaf": 100,
               "learning_rate": 0.02, "lambda_l2": 2.0,
               "feature_fraction": 0.6, "bagging_fraction": 0.7}
    shallow = {"num_leaves": 31, "min_data_in_leaf": 500,
                "learning_rate": 0.04, "lambda_l2": 4.0,
                "feature_fraction": 0.7, "bagging_fraction": 0.7}

    experiments = [
        # name, target, params, seeds, embargo, symbol_onehot
        ("d_h1_baseline",         "y_up_1", default,  1, 3, False),
        ("d_h1_seeds5",           "y_up_1", default,  5, 3, False),
        ("d_h1_seeds5_onehot",    "y_up_1", default,  5, 3, True),
        ("d_h1_seeds5_deeper",    "y_up_1", deeper,   5, 3, True),
        ("d_h1_seeds5_shallow",   "y_up_1", shallow,  5, 3, True),
        ("d_h2_seeds5_onehot",    "y_up_2", default,  5, 3, True),
        ("d_h3_seeds5_onehot",    "y_up_3", default,  5, 3, True),
        ("d_h5_seeds5_onehot",    "y_up_5", default,  5, 3, True),
        ("d_h10_seeds5_onehot",   "y_up_10", default, 5, 5, True),
        ("d_h1_seeds10_deeper",   "y_up_1", deeper,   10, 3, True),
    ]

    results = []
    for name, target, params, seeds, embargo, onehot in experiments:
        exp_out = out_p / name
        log.info("\n=== %s (target=%s seeds=%d onehot=%s) ===",
                 name, target, seeds, onehot)
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
            metrics.update({"experiment": name, "target": target,
                             "seeds": seeds, "embargo": embargo,
                             "symbol_onehot": onehot})
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
    rdf.to_csv(out_p / "leaderboard.csv", index=False)
    cols = ["experiment", "target", "seeds", "symbol_onehot",
            "agg_accuracy", "agg_auc"]
    print("\n=== daily leaderboard ===")
    print(rdf[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()

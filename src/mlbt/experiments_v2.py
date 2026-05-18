"""Round 2 optimisation: symbol one-hot + horizon sweep + bigger ensembles.

After v1 established the embargo + seed-ensemble baseline, v2 tries to push
accuracy higher by:
  - Adding symbol one-hot (lets a single model learn per-symbol intercepts)
  - Sweeping longer horizons (h=6, h=12) — often easier to predict than h=1
  - Larger ensembles (10 seeds) on the best v1 config
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlbt.core.log import get_logger
from mlbt.ml.train import train_model

log = get_logger("experiments_v2")


def main(dataset_path: str = "data/dataset.parquet",
          out_dir: str = "data/experiments_v2") -> None:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    base_params = {"num_leaves": 127, "min_data_in_leaf": 200,
                    "learning_rate": 0.025, "lambda_l2": 2.0,
                    "feature_fraction": 0.7, "bagging_fraction": 0.7}

    experiments = [
        # (name, target, params_override, n_seeds, embargo, symbol_onehot)
        ("h3_seeds5_onehot",        "y_up_3",  base_params, 5,  12, True),
        ("h6_seeds5_onehot",        "y_up_6",  base_params, 5,  12, True),
        ("h12_seeds5_onehot",       "y_up_12", base_params, 5,  12, True),
        ("h6_seeds5_nohot",         "y_up_6",  base_params, 5,  12, False),
        ("h6_seeds10_onehot_deep",  "y_up_6",
            {"num_leaves": 255, "min_data_in_leaf": 100,
             "learning_rate": 0.02, "lambda_l2": 2.0,
             "feature_fraction": 0.7, "bagging_fraction": 0.7}, 10, 12, True),
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
    print(rdf[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

"""1-hour bar optimisation sweep — target >=55% directional accuracy.

Tries the most promising combinations on the 1h dataset:
  - Plain directional targets (y_up_*) vs residualised (y_resid_up_*)
  - Multiple horizons: h=1 (1h), h=2 (2h), h=4 (4h)
  - Symbol one-hot, seed ensembles, embargo
Prints accuracy after EACH experiment so we can stop early when one hits 55%.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlbt.core.log import get_logger
from mlbt.ml.train import train_model

log = get_logger("experiments_1h")


def main(dataset_path: str = "data/dataset_1h.parquet",
          out_dir: str = "data/experiments_1h",
          stop_at: float = 0.55) -> dict:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    default = {"num_leaves": 127, "min_data_in_leaf": 200,
                "learning_rate": 0.025, "lambda_l2": 2.0,
                "feature_fraction": 0.7, "bagging_fraction": 0.7}
    deeper = {"num_leaves": 255, "min_data_in_leaf": 100,
               "learning_rate": 0.02, "lambda_l2": 1.5,
               "feature_fraction": 0.6, "bagging_fraction": 0.7}

    experiments = [
        # Highest-leverage tries first: residualised target + symbol one-hot
        ("1h_resid_h1_seeds5",     "y_resid_up_1", default,  5, 4, True),
        ("1h_resid_h2_seeds5",     "y_resid_up_2", default,  5, 4, True),
        ("1h_resid_h4_seeds5",     "y_resid_up_4", default,  5, 4, True),
        ("1h_h1_seeds5_onehot",    "y_up_1",       default,  5, 4, True),
        ("1h_h2_seeds5_onehot",    "y_up_2",       default,  5, 4, True),
        ("1h_h4_seeds5_onehot",    "y_up_4",       default,  5, 4, True),
        ("1h_h8_seeds5_onehot",    "y_up_8",       default,  5, 4, True),
        # Deeper trees
        ("1h_resid_h1_deep",       "y_resid_up_1", deeper,   5, 4, True),
        ("1h_h4_deep",             "y_up_4",       deeper,   5, 4, True),
        # Bigger ensemble on best target
        ("1h_resid_h2_seeds10",    "y_resid_up_2", deeper,   10, 4, True),
    ]

    results = []
    best_acc = 0.0
    for name, target, params, seeds, embargo, onehot in experiments:
        exp_out = out_p / name
        log.info("\n=== %s (target=%s seeds=%d) ===", name, target, seeds)
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
            acc = metrics.get("agg_accuracy", 0.0)
            log.info("=> %s: acc=%.4f auc=%.4f",
                     name, acc, metrics.get("agg_auc", float("nan")))
            if acc > best_acc:
                best_acc = acc
            if acc >= stop_at:
                log.info(">>> HIT TARGET %.4f >= %.4f — stopping early", acc, stop_at)
                break
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", name, e)

    if results:
        rdf = pd.DataFrame(results).sort_values("agg_accuracy", ascending=False)
        rdf.to_csv(out_p / "leaderboard.csv", index=False)
        cols = ["experiment", "target", "seeds", "symbol_onehot",
                "agg_accuracy", "agg_auc"]
        print("\n=== 1h leaderboard ===")
        print(rdf[cols].head(15).to_string(index=False))
    print(f"\nbest accuracy reached: {best_acc:.4f}")
    return {"best_acc": best_acc, "n_results": len(results)}


if __name__ == "__main__":
    main()

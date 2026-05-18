"""Cross-sectional rank-target sweep with audit fixes applied.

Uses the rebuilt 1h dataset which now contains:
  - y_xsec_top_h        (rank target — fold-stationary, perfectly balanced)
  - y_resid_up_h        (residualised directional)
  - y_vol_up_h          (vol-scaled directional, drops near-zero moves)
  - tod_* features (time of day)
Plus:
  - Auto-scaled embargo (h+1 bars)
  - Overlapping-label sample weights = 1/h
Goal: >=55% accuracy on a sub-daily horizon.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlbt.core.log import get_logger
from mlbt.ml.train import train_model

log = get_logger("experiments_xsec")


def main(dataset_path: str = "data/dataset_1h.parquet",
          out_dir: str = "data/experiments_xsec",
          stop_at: float = 0.55) -> None:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    default = {"num_leaves": 127, "min_data_in_leaf": 200,
                "learning_rate": 0.025, "lambda_l2": 2.0,
                "feature_fraction": 0.7, "bagging_fraction": 0.7}
    deeper = {"num_leaves": 255, "min_data_in_leaf": 100,
               "learning_rate": 0.02, "lambda_l2": 1.5,
               "feature_fraction": 0.6, "bagging_fraction": 0.7}

    experiments = [
        # (name, target, params, seeds, onehot)
        ("xsec_h1_seeds5",            "y_xsec_top_1", default, 5, True),
        ("xsec_h2_seeds5",            "y_xsec_top_2", default, 5, True),
        ("xsec_h4_seeds5",            "y_xsec_top_4", default, 5, True),
        ("vol_h1_seeds5",             "y_vol_up_1",   default, 5, True),
        ("vol_h2_seeds5",             "y_vol_up_2",   default, 5, True),
        ("vol_h4_seeds5",             "y_vol_up_4",   default, 5, True),
        ("xsec_h2_seeds10_deep",      "y_xsec_top_2", deeper,  10, True),
        ("vol_h2_seeds10_deep",       "y_vol_up_2",   deeper,  10, True),
        ("xsec_h4_seeds10_deep",      "y_xsec_top_4", deeper,  10, True),
    ]

    results = []
    best = 0.0
    for name, target, params, seeds, onehot in experiments:
        exp_out = out_p / name
        log.info("\n=== %s (target=%s seeds=%d) ===", name, target, seeds)
        try:
            m = train_model(
                dataset_path=dataset_path,
                target=target,
                model="gbm",
                out_dir=str(exp_out),
                n_seeds=seeds,
                embargo=0,  # auto = h+1
                params_override=params,
                symbol_onehot=onehot,
            )
            m.update({"experiment": name, "target": target, "seeds": seeds})
            results.append(m)
            acc = m.get("agg_accuracy", 0.0)
            log.info("=> %s: acc=%.4f auc=%.4f", name, acc, m.get("agg_auc", float("nan")))
            if acc > best:
                best = acc
            if acc >= stop_at:
                log.info(">>> HIT TARGET %.4f >= %.4f", acc, stop_at)
                break
        except Exception as e:
            log.warning("%s failed: %s", name, e)

    if results:
        rdf = pd.DataFrame(results).sort_values("agg_accuracy", ascending=False)
        rdf.to_csv(out_p / "leaderboard.csv", index=False)
        cols = ["experiment", "target", "seeds", "agg_accuracy", "agg_auc"]
        print("\n=== xsec/vol leaderboard ===")
        print(rdf[cols].head(15).to_string(index=False))
    print(f"\nbest accuracy reached: {best:.4f}")


if __name__ == "__main__":
    main()

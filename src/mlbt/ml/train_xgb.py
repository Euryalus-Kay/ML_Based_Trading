"""XGBoost trainer mirroring mlbt.ml.train._train_gbm interface.

LightGBM is the production model; XGBoost may pick up different patterns
because its histogram-quantile split heuristic and L1 regularisation
differ. Worth a head-to-head test on the same data + target.

Same outputs as train.py:
  model.pkl, feature_cols.json, metrics.json, predictions.parquet
so candidate_eval and backtest_runner work without changes.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from mlbt.core.log import get_logger
from mlbt.ml.train import (
    _binary_metrics, _horizon_from_target, _select_features,
    _add_symbol_onehot, _walk_forward_indices,
)

log = get_logger("train_xgb")


_DEFAULT_XGB = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "tree_method": "hist",     # histogram = fast on CPU, lots of features OK
    "nthread": -1,             # all cores
    "verbosity": 0,
}


def train_xgb(dataset_path: str, target: str = "y_xsec_top_8",
                out_dir: str = "data/cand_xgb",
                n_seeds: int = 3, embargo: int = 0,
                params_override: dict | None = None,
                symbol_onehot: bool = True,
                use_overlap_weights: bool = True) -> dict:
    import xgboost as xgb

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_path).sort_index()
    if target not in df.columns:
        raise KeyError(f"target {target} missing")
    feature_cols = _select_features(df, target_col=target)
    if symbol_onehot:
        df, sym_cols = _add_symbol_onehot(df)
        feature_cols = feature_cols + sym_cols
    df = df.dropna(subset=[target])
    nan_share = df[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if nan_share.get(c, 1.0) < 0.95]
    if df.empty or not feature_cols:
        return {"error": "no usable rows/features"}

    X = df[feature_cols].values.astype(np.float32)
    y = df[target].astype(int).values
    symbols = df["symbol"].values if "symbol" in df.columns else None

    h = _horizon_from_target(target)
    if embargo <= 0:
        embargo = h + 1
    if use_overlap_weights and h > 1:
        sample_w = np.full(len(df), 1.0 / h, dtype=np.float32)
    else:
        sample_w = np.ones(len(df), dtype=np.float32)
    log.info("XGBoost: %d rows x %d features; h=%d embargo=%d",
             len(df), len(feature_cols), h, embargo)

    base = dict(_DEFAULT_XGB)
    if params_override:
        base.update(params_override)

    fold_metrics, test_preds_all = [], []
    fold_id = 0
    final_booster = None
    for tr_idx, va_idx, te_idx in _walk_forward_indices(len(df), n_folds=5, embargo=embargo):
        fold_id += 1
        Xtr, ytr, wtr = X[tr_idx], y[tr_idx], sample_w[tr_idx]
        Xva, yva, wva = X[va_idx], y[va_idx], sample_w[va_idx]
        Xte, yte = X[te_idx], y[te_idx]

        scores = np.zeros(len(Xte), dtype=np.float64)
        for seed in range(n_seeds):
            params = dict(base, seed=seed)
            dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
            dva = xgb.DMatrix(Xva, label=yva, weight=wva)
            dte = xgb.DMatrix(Xte)
            booster = xgb.train(
                params, dtr, num_boost_round=2000,
                evals=[(dva, "val")],
                early_stopping_rounds=50, verbose_eval=False,
            )
            scores += booster.predict(dte)
        scores /= n_seeds

        m = _binary_metrics(yte, scores)
        m["fold"] = fold_id
        fold_metrics.append(m)
        log.info("fold %d acc=%.3f auc=%.3f n=%d", fold_id,
                 m["accuracy"], m["auc"], m["n"])

        rec = {"fold": fold_id, "y_true": yte, "y_score": scores}
        if symbols is not None:
            rec["symbol"] = symbols[te_idx]
        test_preds_all.append(pd.DataFrame(rec, index=df.index[te_idx]))
        final_booster = booster

    (out_p / "feature_cols.json").write_text(json.dumps(feature_cols))
    (out_p / "metrics.json").write_text(json.dumps({
        "model": "xgb",
        "target": target,
        "folds": fold_metrics,
        "agg_accuracy": float(np.mean([f["accuracy"] for f in fold_metrics])),
        "agg_auc": float(np.nanmean([f["auc"] for f in fold_metrics])),
    }, indent=2))
    with open(out_p / "model.pkl", "wb") as f:
        pickle.dump(final_booster, f)
    if test_preds_all:
        preds = pd.concat(test_preds_all)
        preds.to_parquet(out_p / "predictions.parquet")
    return {
        "agg_accuracy": float(np.mean([f["accuracy"] for f in fold_metrics])),
        "agg_auc": float(np.nanmean([f["auc"] for f in fold_metrics])),
        "n_folds": len(fold_metrics),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/dataset_1h_micro.parquet")
    p.add_argument("--target", default="y_xsec_top_8")
    p.add_argument("--out", default="data/cand_xgb_xs8")
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()
    print(json.dumps(train_xgb(args.dataset, args.target, args.out,
                                  n_seeds=args.seeds), indent=2))

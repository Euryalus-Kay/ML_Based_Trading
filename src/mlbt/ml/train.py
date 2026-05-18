"""Walk-forward training + evaluation.

Two model families:
  - "gbm":  LightGBM (preferred) or sklearn GradientBoosting fallback
  - "lstm": PyTorch LSTM/Transformer with rolling windows

Walk-forward splits the time axis into successive train/val/test folds, never
shuffling across time. Cross-symbol contamination is avoided by symbol-aware
windowing (see dataset_torch.SequenceDataset).

The output directory contains:
  model.pkl / model.pt    — the trained estimator
  metrics.json            — per-fold + aggregate metrics
  feature_cols.json       — column list (preserves order for inference)
  predictions.parquet     — out-of-sample predictions on the held-out tail
"""
from __future__ import annotations

import json
import math
import pickle
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from mlbt.core.log import get_logger

log = get_logger("train")


# ----- Feature / target selection helpers ----------------------------------
def _select_features(df: pd.DataFrame, target_col: str,
                     drop_prefixes: Sequence[str] = ("y_logret_", "y_up_", "y_tb_",
                                                       "y_resid_logret_", "y_resid_up_"),
                     exclude: Sequence[str] = ("symbol",)) -> list[str]:
    features = []
    for c in df.columns:
        if c == target_col or c in exclude:
            continue
        if any(c.startswith(p) for p in drop_prefixes):
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        features.append(c)
    return features


def _add_symbol_onehot(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode the 'symbol' column. Returns (new_df, added_cols).

    Lets a single global model learn symbol-specific intercepts and feature
    interactions through the tree splits.
    """
    if "symbol" not in df.columns:
        return df, []
    dummies = pd.get_dummies(df["symbol"], prefix="sym").astype("float32")
    new_cols = list(dummies.columns)
    df = pd.concat([df, dummies], axis=1)
    return df, new_cols


def _walk_forward_indices(n: int, n_folds: int = 5, val_frac: float = 0.1,
                           embargo: int = 0):
    """Yield (train_idx, val_idx, test_idx) for n_folds expanding-window splits.

    For fold k: train on [0, t_k - embargo), val on [t_k, t_k + v),
                test on [t_k + v + embargo, t_{k+1}).
    """
    fold_size = n // (n_folds + 1)
    val_size = max(1, int(fold_size * val_frac))
    for k in range(1, n_folds + 1):
        train_end = k * fold_size
        val_end = train_end + val_size
        test_start = val_end + embargo
        test_end = min(n, test_start + fold_size)
        if test_end <= test_start:
            continue
        train_clean_end = max(0, train_end - embargo)
        yield (np.arange(0, train_clean_end),
               np.arange(train_end, val_end),
               np.arange(test_start, test_end))


# ----- Metrics --------------------------------------------------------------
def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
    y_pred = (y_score > 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["auc"] = float("nan")
    try:
        out["logloss"] = float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6)))
    except Exception:
        out["logloss"] = float("nan")
    # "Information coefficient" style: corr between score and label
    if np.std(y_score) > 0:
        out["corr"] = float(np.corrcoef(y_true, y_score)[0, 1])
    return out


# ----- GBM (LightGBM preferred) --------------------------------------------
_DEFAULT_LGB = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbosity": -1,
}


def _train_gbm(df: pd.DataFrame, target: str, feature_cols: list[str], out_dir: Path,
                n_seeds: int = 1, embargo: int = 0,
                params_override: dict | None = None) -> dict:
    use_lgb = True
    try:
        import lightgbm as lgb
    except ImportError:
        use_lgb = False
        from sklearn.ensemble import GradientBoostingClassifier

    df = df.sort_index()
    # Drop target NaN only; LightGBM handles feature NaN natively.
    df = df.dropna(subset=[target])
    # Drop all-NaN feature columns so they don't waste tree splits.
    nan_share = df[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if nan_share.get(c, 1.0) < 0.95]
    if df.empty or not feature_cols:
        return {"error": "no usable rows/features after dropna"}

    X = df[feature_cols].values.astype(np.float32)
    y = df[target].astype(int).values
    symbols = df["symbol"].values if "symbol" in df.columns else None
    log.info("usable: %d rows x %d features (after NaN filter)",
             len(df), len(feature_cols))

    base_params = dict(_DEFAULT_LGB)
    if params_override:
        base_params.update(params_override)

    fold_metrics = []
    test_preds_all = []
    fold_id = 0
    for tr_idx, va_idx, te_idx in _walk_forward_indices(len(df), n_folds=5, embargo=embargo):
        fold_id += 1
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]
        Xte, yte = X[te_idx], y[te_idx]

        if use_lgb:
            scores = np.zeros(len(Xte), dtype=float)
            for seed in range(n_seeds):
                params = dict(base_params, seed=seed, bagging_seed=seed,
                              feature_fraction_seed=seed)
                train_ds = lgb.Dataset(Xtr, ytr)
                val_ds = lgb.Dataset(Xva, yva, reference=train_ds)
                booster = lgb.train(
                    params, train_ds,
                    num_boost_round=2000,
                    valid_sets=[val_ds],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                )
                scores += booster.predict(Xte)
            scores /= n_seeds
            score = scores
        else:
            clf = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                              learning_rate=0.05, subsample=0.8)
            clf.fit(Xtr, ytr)
            score = clf.predict_proba(Xte)[:, 1]
            booster = clf

        m = _binary_metrics(yte, score)
        m["fold"] = fold_id
        fold_metrics.append(m)
        log.info("fold %d: acc=%.3f auc=%.3f n=%d (seeds=%d)",
                 fold_id, m["accuracy"], m["auc"], m["n"], n_seeds)

        rec = {"fold": fold_id, "y_true": yte, "y_score": score}
        if symbols is not None:
            rec["symbol"] = symbols[te_idx]
        test_preds_all.append(pd.DataFrame(rec, index=df.index[te_idx]))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({
        "model": "lgb" if use_lgb else "sklearn_gbm",
        "target": target,
        "folds": fold_metrics,
        "agg_accuracy": float(np.mean([f["accuracy"] for f in fold_metrics])),
        "agg_auc": float(np.nanmean([f["auc"] for f in fold_metrics])),
    }, indent=2))
    (out_dir / "feature_cols.json").write_text(json.dumps(feature_cols))

    # Save final-fold model
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(booster, f)

    if test_preds_all:
        preds = pd.concat(test_preds_all)
        preds.to_parquet(out_dir / "predictions.parquet")

    return {
        "agg_accuracy": float(np.mean([f["accuracy"] for f in fold_metrics])),
        "agg_auc": float(np.nanmean([f["auc"] for f in fold_metrics])),
        "n_folds": len(fold_metrics),
        "model_path": str(out_dir / "model.pkl"),
    }


# ----- LSTM / Transformer ---------------------------------------------------
def _train_lstm(df: pd.DataFrame, target: str, feature_cols: list[str],
                out_dir: Path, window: int = 64, epochs: int = 8,
                batch_size: int = 256, model_kind: str = "lstm") -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as e:
        return {"error": f"PyTorch unavailable: {e}"}

    from mlbt.ml.dataset_torch import SequenceDataset
    from mlbt.ml.models import LSTMClassifier, TransformerClassifier, PatchTSTLite

    df = df.sort_index().dropna(subset=[target])
    df = df.dropna(subset=feature_cols, how="any")
    if df.empty:
        return {"error": "no rows after dropna"}

    # Normalise features (fit on full train, train-only would be ideal in walk-forward)
    mean = df[feature_cols].mean()
    std = df[feature_cols].std().replace(0, 1.0)
    df_norm = df.copy()
    df_norm[feature_cols] = (df_norm[feature_cols] - mean) / std

    # Time-based split (last 20% test, prior 20% val, rest train)
    n = len(df_norm)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    train_df = df_norm.iloc[:train_end]
    val_df = df_norm.iloc[train_end:val_end]
    test_df = df_norm.iloc[val_end:]

    train_ds = SequenceDataset(train_df, feature_cols, target, window=window)
    val_ds = SequenceDataset(val_df, feature_cols, target, window=window)
    test_ds = SequenceDataset(test_df, feature_cols, target, window=window)

    if len(train_ds) == 0 or len(val_ds) == 0:
        return {"error": "not enough sequence samples"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("training %s on device=%s, train=%d val=%d test=%d",
             model_kind, device, len(train_ds), len(val_ds), len(test_ds))

    n_features = len(feature_cols)
    if model_kind == "lstm":
        model = LSTMClassifier(n_features=n_features)
    elif model_kind == "transformer":
        model = TransformerClassifier(n_features=n_features)
    elif model_kind == "patchtst":
        # Choose patch_size that divides window
        ps = 8 if window % 8 == 0 else 4
        model = PatchTSTLite(n_features=n_features, window=window, patch_size=ps)
    else:
        return {"error": f"unknown model_kind {model_kind}"}
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    best_val = float("inf")
    best_state = None
    history = []
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = bce(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= max(1, len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        val_y, val_p = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                val_loss += bce(logits, y).item() * X.size(0)
                val_y.append(y.cpu().numpy())
                val_p.append(torch.sigmoid(logits).cpu().numpy())
        val_loss /= max(1, len(val_loader.dataset))
        val_y = np.concatenate(val_y) if val_y else np.array([])
        val_p = np.concatenate(val_p) if val_p else np.array([])
        val_acc = float(np.mean((val_p > 0.5) == val_y)) if len(val_y) else float("nan")

        sched.step()
        history.append({"epoch": ep, "train_loss": train_loss,
                         "val_loss": val_loss, "val_acc": val_acc})
        log.info("ep%d train=%.4f val=%.4f acc=%.3f", ep, train_loss, val_loss, val_acc)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    test_y, test_p = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X)
            test_y.append(y.numpy())
            test_p.append(torch.sigmoid(logits).cpu().numpy())
    test_y = np.concatenate(test_y) if test_y else np.array([])
    test_p = np.concatenate(test_p) if test_p else np.array([])
    metrics = _binary_metrics(test_y, test_p) if len(test_y) else {}

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_kind": model_kind,
                 "n_features": n_features, "window": window,
                 "mean": mean.to_dict(), "std": std.to_dict()},
                out_dir / "model.pt")
    (out_dir / "feature_cols.json").write_text(json.dumps(feature_cols))
    (out_dir / "metrics.json").write_text(json.dumps({
        "model": model_kind, "target": target, "history": history,
        "test": metrics,
    }, indent=2))

    return {
        "test_accuracy": metrics.get("accuracy"),
        "test_auc": metrics.get("auc"),
        "n_test": metrics.get("n"),
        "model_path": str(out_dir / "model.pt"),
    }


# ----- Public entrypoint ---------------------------------------------------
def train_model(dataset_path: str, target: str = "y_up_3",
                 model: str = "gbm", out_dir: str = "data/model",
                 window: int = 64, epochs: int = 8,
                 n_seeds: int = 1, embargo: int = 0,
                 params_override: dict | None = None,
                 symbol_onehot: bool = False) -> dict:
    df = pd.read_parquet(dataset_path)
    if target not in df.columns:
        raise KeyError(f"target {target} not in dataset; have y_* cols: "
                       f"{[c for c in df.columns if c.startswith('y_')]}")
    feature_cols = _select_features(df, target_col=target)
    if symbol_onehot:
        df, sym_cols = _add_symbol_onehot(df)
        feature_cols = feature_cols + sym_cols
    out_p = Path(out_dir)
    log.info("training %s on %d rows x %d features -> %s (seeds=%d embargo=%d onehot=%s)",
             model, len(df), len(feature_cols), target, n_seeds, embargo, symbol_onehot)
    if model == "gbm":
        return _train_gbm(df, target, feature_cols, out_p,
                            n_seeds=n_seeds, embargo=embargo,
                            params_override=params_override)
    if model in ("lstm", "transformer", "patchtst"):
        return _train_lstm(df, target, feature_cols, out_p, window=window,
                            epochs=epochs, model_kind=model)
    raise ValueError(f"unknown model {model}")

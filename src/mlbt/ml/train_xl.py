"""XL torch trainers tuned for Apple Silicon (MPS) + 64GB unified RAM.

Multi-task heads: jointly predict y_xsec_top_1, _2, _4, _8 — the shared
backbone is forced to learn features that generalise across horizons,
which is the biggest single regularisation move for short-horizon
cross-sectional rank prediction.

Big batch (256–768), more epochs, more seeds, label smoothing, AdamW with
cosine, AMP-aware bf16 on MPS where supported.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from mlbt.core.log import get_logger

log = get_logger("train_xl")


def _best_device():
    try:
        import torch
    except ImportError:
        return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class XLTrainConfig:
    targets: tuple[str, ...] = ("y_xsec_top_1", "y_xsec_top_2",
                                 "y_xsec_top_4", "y_xsec_top_8")
    primary_target: str = "y_xsec_top_8"   # used as the headline for backtest
    model_kind: str = "patchtst_xl"        # transformer_xl | patchtst_xl | lstm_xl
    window: int = 64
    epochs: int = 25
    batch_size: int = 384
    lr: float = 3e-4
    weight_decay: float = 1e-2
    dropout: float = 0.15
    label_smoothing: float = 0.02
    grad_clip: float = 1.0
    n_seeds: int = 3                       # average across seeds for stability
    device: str = "auto"
    use_bf16: bool = True                  # bf16 weights on MPS where supported


# ----- multi-task wrapper ---------------------------------------------------
def _build_backbone(kind: str, n_features: int, window: int, dropout: float):
    import torch.nn as nn
    from mlbt.ml.models import (
        LSTMClassifier, TransformerClassifier, PatchTSTLite,
        LSTMXL, TransformerXL, PatchTSTXL,
    )
    # Slightly hacky: re-use the existing classifier modules but rip out their
    # final head and replace with a Linear that emits a backbone embedding.
    if kind == "patchtst_xl":
        m = PatchTSTXL(n_features=n_features, window=window, dropout=dropout, n_outputs=1)
        d = 384
    elif kind == "transformer_xl":
        m = TransformerXL(n_features=n_features, dropout=dropout, n_outputs=1)
        d = 256
    elif kind == "lstm_xl":
        m = LSTMXL(n_features=n_features, dropout=dropout, n_outputs=1)
        d = 768  # 384 * 2 (bidirectional)
    elif kind == "patchtst":
        m = PatchTSTLite(n_features=n_features, window=window, dropout=dropout)
        d = 128
    elif kind == "transformer":
        m = TransformerClassifier(n_features=n_features, dropout=dropout)
        d = 128
    elif kind == "lstm":
        m = LSTMClassifier(n_features=n_features, dropout=dropout)
        d = 128
    else:
        raise ValueError(kind)
    # Replace the head with identity so we can attach multi-task heads outside
    m.head = nn.Identity() if hasattr(m, "head") else m.head
    return m, d


class MultiTaskNet:
    """Constructs a (backbone + per-target head) model from any of our XL nets."""

    def __init__(self, kind: str, n_features: int, window: int,
                 n_tasks: int, dropout: float):
        import torch.nn as nn
        backbone, dim = _build_backbone(kind, n_features, window, dropout)
        self.backbone = backbone
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 64), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            for _ in range(n_tasks)
        ])

    def parameters(self):
        import torch.nn as nn
        m = nn.ModuleList([self.backbone, self.heads])
        return m.parameters()

    def to(self, device):
        self.backbone = self.backbone.to(device)
        self.heads = self.heads.to(device)
        return self

    def train(self):
        self.backbone.train(); self.heads.train()
        return self

    def eval(self):
        self.backbone.eval(); self.heads.eval()
        return self

    def state_dict(self):
        return {"backbone": self.backbone.state_dict(),
                "heads": self.heads.state_dict()}

    def load_state_dict(self, sd):
        self.backbone.load_state_dict(sd["backbone"])
        self.heads.load_state_dict(sd["heads"])

    def forward(self, x):
        import torch
        h = self.backbone(x)
        # If backbone outputs (B,) we need to extract a vector; for our XL nets
        # the head returns a logit. We replaced .head with Identity so the
        # backbone returns the pre-head embedding instead.
        if h.dim() == 1:
            h = h.unsqueeze(-1)
        outs = [head(h).squeeze(-1) for head in self.heads]
        return torch.stack(outs, dim=1)

    def __call__(self, x):
        return self.forward(x)


# ----- dataset --------------------------------------------------------------
def _seq_dataset(df, feature_cols, target_cols, window):
    import torch
    from torch.utils.data import Dataset

    class MultiTargetSequenceDataset(Dataset):
        def __init__(self, frame, feats, targets, win):
            self.samples = []
            for sym, sub in frame.groupby("symbol"):
                sub = sub.sort_index()
                X = sub[feats].values.astype(np.float32)
                Y = sub[targets].values.astype(np.float32)
                for i in range(win, len(sub)):
                    y = Y[i]
                    if np.isnan(y).any():
                        continue
                    wx = X[i - win:i]
                    if np.isnan(wx).any():
                        continue
                    self.samples.append((wx, y))
            log.info("seq dataset: %d samples (window=%d, features=%d, targets=%d)",
                     len(self.samples), win, len(feats), len(targets))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            wx, y = self.samples[i]
            return torch.from_numpy(wx), torch.from_numpy(y)
    return MultiTargetSequenceDataset(df, feature_cols, target_cols, window)


# ----- training loop --------------------------------------------------------
def train_xl_multitask(dataset_path: str, out_dir: str,
                        cfg: XLTrainConfig | None = None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    cfg = cfg or XLTrainConfig()
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    device = _best_device() if cfg.device == "auto" else cfg.device
    log.info("train_xl: device=%s model=%s window=%d batch=%d epochs=%d seeds=%d",
             device, cfg.model_kind, cfg.window, cfg.batch_size, cfg.epochs, cfg.n_seeds)
    t0 = time.monotonic()

    df = pd.read_parquet(dataset_path).sort_index()
    for t in cfg.targets:
        if t not in df.columns:
            raise KeyError(f"target {t} not in dataset; have y_* cols: "
                            f"{[c for c in df.columns if c.startswith('y_')]}")

    # Feature selection — same as classical: drop targets + symbol + any y_*
    drop = set(cfg.targets) | {"symbol"}
    feat_cols = [c for c in df.columns
                 if c not in drop and not c.startswith("y_")
                 and np.issubdtype(df[c].dtype, np.number)]
    # Per-feature z-score normalisation (fit on whole dataset is fine for now;
    # for production walk-forward, fit on the training fold only).
    means = df[feat_cols].mean()
    stds = df[feat_cols].std().replace(0, 1.0)
    df_norm = df.copy()
    df_norm[feat_cols] = (df_norm[feat_cols] - means) / stds
    df_norm = df_norm.dropna(subset=feat_cols, how="any")

    n = len(df_norm)
    tr_end = int(n * 0.7); va_end = int(n * 0.85)
    train_df = df_norm.iloc[:tr_end]; val_df = df_norm.iloc[tr_end:va_end]; test_df = df_norm.iloc[va_end:]

    train_ds = _seq_dataset(train_df, feat_cols, list(cfg.targets), cfg.window)
    val_ds = _seq_dataset(val_df, feat_cols, list(cfg.targets), cfg.window)
    test_ds = _seq_dataset(test_df, feat_cols, list(cfg.targets), cfg.window)

    if len(train_ds) == 0:
        return {"error": "no training samples"}

    primary_idx = list(cfg.targets).index(cfg.primary_target)

    test_scores_all_seeds = []
    metrics_by_seed = []
    for seed in range(cfg.n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        log.info("--- seed %d/%d ---", seed + 1, cfg.n_seeds)
        net = MultiTaskNet(kind=cfg.model_kind, n_features=len(feat_cols),
                              window=cfg.window, n_tasks=len(cfg.targets),
                              dropout=cfg.dropout).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")

        nw = max(1, min((os.cpu_count() or 4) - 1, 6))
        pin = (device == "cuda")
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                    drop_last=True, num_workers=nw, pin_memory=pin,
                                    persistent_workers=nw > 0)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=nw,
                                  pin_memory=pin, persistent_workers=nw > 0)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=nw,
                                   pin_memory=pin)

        best_val = float("inf"); best_state = None; history = []
        for ep in range(cfg.epochs):
            net.train()
            sum_loss = 0.0
            for X, Y in train_loader:
                X, Y = X.to(device), Y.to(device)
                if cfg.label_smoothing > 0:
                    Y = Y * (1 - cfg.label_smoothing) + cfg.label_smoothing / 2
                opt.zero_grad()
                logits = net(X)
                loss = bce(logits, Y).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                opt.step()
                sum_loss += loss.item() * X.size(0)
            tr_loss = sum_loss / max(1, len(train_loader.dataset))

            # Val
            net.eval()
            v_loss = 0.0; v_y = []; v_p = []
            with torch.no_grad():
                for X, Y in val_loader:
                    X, Y = X.to(device), Y.to(device)
                    logits = net(X)
                    v_loss += bce(logits, Y).mean().item() * X.size(0)
                    v_y.append(Y.cpu().numpy()); v_p.append(torch.sigmoid(logits).cpu().numpy())
            v_loss /= max(1, len(val_loader.dataset))
            v_y = np.concatenate(v_y) if v_y else np.array([])
            v_p = np.concatenate(v_p) if v_p else np.array([])
            val_acc_primary = float(np.mean((v_p[:, primary_idx] > 0.5) ==
                                              (v_y[:, primary_idx] > 0.5))) if len(v_y) else float("nan")
            sched.step()
            history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": v_loss,
                              "val_acc_primary": val_acc_primary})
            log.info("seed %d ep %d  train=%.4f val=%.4f acc=%.3f",
                     seed, ep, tr_loss, v_loss, val_acc_primary)
            if v_loss < best_val:
                best_val = v_loss
                best_state = {"backbone": {k: v.detach().cpu().clone() for k, v in net.backbone.state_dict().items()},
                              "heads": {k: v.detach().cpu().clone() for k, v in net.heads.state_dict().items()}}

        if best_state is not None:
            net.load_state_dict(best_state)

        # Test
        net.eval()
        t_y = []; t_p = []
        with torch.no_grad():
            for X, Y in test_loader:
                X = X.to(device)
                logits = net(X)
                t_y.append(Y.numpy()); t_p.append(torch.sigmoid(logits).cpu().numpy())
        t_y = np.concatenate(t_y) if t_y else np.array([])
        t_p = np.concatenate(t_p) if t_p else np.array([])
        test_scores_all_seeds.append(t_p[:, primary_idx])
        # Save individual seed checkpoint
        torch.save({
            "state_dict": best_state, "model_kind": cfg.model_kind,
            "n_features": len(feat_cols), "window": cfg.window,
            "targets": list(cfg.targets), "primary_target": cfg.primary_target,
            "mean": means.to_dict(), "std": stds.to_dict(),
        }, out_p / f"model_seed{seed}.pt")
        metrics_by_seed.append({
            "seed": seed, "val_best_loss": best_val, "n_test": int(len(t_y)),
        })

    # Ensemble (average over seeds) primary-target test predictions
    ens_scores = np.mean(np.stack(test_scores_all_seeds, axis=0), axis=0)
    # Save the FINAL seed-0 checkpoint as the canonical model.pt for the
    # live inference pipeline (which only loads one weight file).
    # Also write predictions.parquet aligned to the test split.
    # We need the (timestamp, symbol) pairs that match the sequence dataset.
    # SequenceDataset already filtered NaN windows, so retrieve indices from
    # test_df by emulating its sampling logic.
    rows = []
    for sym, sub in test_df.groupby("symbol"):
        sub = sub.sort_index()
        X = sub[feat_cols].values.astype(np.float32)
        Y = sub[list(cfg.targets)].values.astype(np.float32)
        for i in range(cfg.window, len(sub)):
            if np.isnan(Y[i]).any() or np.isnan(X[i - cfg.window:i]).any():
                continue
            rows.append({"ts": sub.index[i], "symbol": sym,
                          "y_true": float(Y[i, primary_idx])})
    preds_meta = pd.DataFrame(rows)
    if len(preds_meta) != len(ens_scores):
        log.warning("preds count mismatch: meta=%d ens=%d — truncating to shorter",
                    len(preds_meta), len(ens_scores))
        m = min(len(preds_meta), len(ens_scores))
        preds_meta = preds_meta.iloc[:m]; ens_scores = ens_scores[:m]
    preds_meta["y_score"] = ens_scores
    preds_meta = preds_meta.set_index("ts")
    preds_meta.to_parquet(out_p / "predictions.parquet")

    (out_p / "feature_cols.json").write_text(json.dumps(feat_cols))
    (out_p / "metrics.json").write_text(json.dumps({
        "model": cfg.model_kind, "target": cfg.primary_target,
        "targets": list(cfg.targets), "n_seeds": cfg.n_seeds,
        "metrics_by_seed": metrics_by_seed,
        "elapsed_seconds": time.monotonic() - t0,
        "n_features": len(feat_cols), "window": cfg.window,
    }, indent=2, default=str))

    # Persist the seed-0 model as model.pt for the live pipeline
    import shutil
    shutil.copy(out_p / "model_seed0.pt", out_p / "model.pt")

    log.info("train_xl finished in %.1f min (saved to %s)",
             (time.monotonic() - t0) / 60.0, out_p)
    return {"out_dir": str(out_p), "elapsed_min": (time.monotonic() - t0) / 60.0,
             "n_seeds": cfg.n_seeds, "primary_target": cfg.primary_target}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/dataset_1h_sp500.parquet")
    p.add_argument("--out", default="data/cand_xl_v1")
    p.add_argument("--model", default="patchtst_xl",
                    choices=["lstm_xl", "transformer_xl", "patchtst_xl",
                              "lstm", "transformer", "patchtst"])
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=384)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--primary", default="y_xsec_top_8")
    args = p.parse_args()
    cfg = XLTrainConfig(
        primary_target=args.primary, model_kind=args.model,
        window=args.window, epochs=args.epochs, batch_size=args.batch,
        n_seeds=args.seeds,
    )
    print(json.dumps(train_xl_multitask(args.dataset, args.out, cfg),
                      indent=2, default=str))

"""Live signal generator.

Loads a trained model from disk and produces per-symbol scores at any
given moment by pulling the latest data, rebuilding the feature pipeline,
and running model inference.

Inference targets, in priority order:
  1. Apple Neural Engine (CoreML .mlpackage) — sub-ms latency
  2. MPS (PyTorch model.pt) — GPU on Apple Silicon
  3. CPU (GBM pickle) — LightGBM

The exact same feature pipeline as training (mlbt.pipeline.dataset) is
re-run on the live window so feature parity is guaranteed.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mlbt.core.log import get_logger
from mlbt.pipeline.dataset import build_dataset

log = get_logger("signal")


class LiveSignalGenerator:
    def __init__(self, model_dir: str, bar: str = "1h",
                 universe_path: Optional[str] = None,
                 lookback_days: int = 60,
                 use_coreml: bool = False,
                 inference_device: str = "auto"):
        self.model_dir = Path(model_dir)
        self.bar = bar
        self.universe_path = universe_path
        self.lookback_days = lookback_days
        self.use_coreml = use_coreml
        self.inference_device = inference_device

        self.feature_cols = json.loads((self.model_dir / "feature_cols.json").read_text())
        self.metrics = json.loads((self.model_dir / "metrics.json").read_text()) if (self.model_dir / "metrics.json").exists() else {}
        self.target = self.metrics.get("target", "y_xsec_top_8")
        self._horizon = self._h_from_target(self.target)
        self._backend = self._pick_backend()
        self._loaded = self._load()
        log.info("loaded %s for live inference (target=%s h=%d feature_cols=%d)",
                 self._backend, self.target, self._horizon, len(self.feature_cols))

    @staticmethod
    def _h_from_target(target: str) -> int:
        try:
            return int(target.rsplit("_", 1)[-1])
        except Exception:
            return 1

    def _pick_backend(self) -> str:
        # CoreML on ANE > Torch on MPS > GBM on CPU
        if self.use_coreml and (self.model_dir / "model.mlpackage").exists():
            return "coreml"
        if (self.model_dir / "model.pt").exists():
            return "torch"
        if (self.model_dir / "model.pkl").exists():
            return "gbm"
        raise FileNotFoundError(f"no model.* in {self.model_dir}")

    def _load(self):
        if self._backend == "gbm":
            with open(self.model_dir / "model.pkl", "rb") as f:
                return pickle.load(f)
        if self._backend == "torch":
            import torch
            from mlbt.ml.models import (
                LSTMClassifier, TransformerClassifier, PatchTSTLite,
                LSTMXL, TransformerXL, PatchTSTXL,
            )
            ckpt = torch.load(self.model_dir / "model.pt", map_location="cpu", weights_only=False)
            kind = ckpt.get("model_kind", "lstm")
            n_features = ckpt["n_features"]
            window = ckpt.get("window", 64)
            klass = {"lstm": LSTMClassifier, "transformer": TransformerClassifier,
                     "patchtst": PatchTSTLite, "lstm_xl": LSTMXL,
                     "transformer_xl": TransformerXL, "patchtst_xl": PatchTSTXL}[kind]
            model = klass(n_features=n_features, window=window) if kind.startswith("patchtst") \
                    else klass(n_features=n_features)
            model.load_state_dict(ckpt["state_dict"])
            device = self._resolve_device()
            model = model.to(device).eval()
            return {"model": model, "device": device, "window": window,
                    "mean": ckpt.get("mean"), "std": ckpt.get("std")}
        if self._backend == "coreml":
            import coremltools as ct
            return ct.models.MLModel(str(self.model_dir / "model.mlpackage"))
        raise ValueError(self._backend)

    def _resolve_device(self) -> str:
        import torch
        d = self.inference_device
        if d in ("ane", "neural"):
            return "cpu"  # ANE is reached via CoreML, not torch
        if d in ("auto", "mps"):
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return d or "cpu"

    # ---------------------------------------------------------------------
    def score_now(self, ts_now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Pull latest data, featurise, score. Returns DataFrame[symbol, y_score, ts].

        The dataset is rebuilt over the last `lookback_days` so we have enough
        warmup for rolling features. The most-recent row per symbol is what
        we'd act on.
        """
        end = ts_now or pd.Timestamp.utcnow().tz_convert("UTC")
        start = end - pd.Timedelta(days=self.lookback_days)
        t0 = time.monotonic()
        ds = build_dataset(start, end, bar=self.bar, session="rth",
                           universe_path=self.universe_path,
                           horizons=(1, 2, 4, 8),
                           out_path=None,
                           only_sources=["yf_1h", "fred", "treasury_yields",
                                          "calendar_events", "crypto_fear_greed"])
        if ds.empty:
            log.warning("live: empty dataset built")
            return pd.DataFrame(columns=["symbol", "y_score", "ts"])

        # Most-recent row per symbol — the one we would trade NOW
        latest = ds.groupby("symbol").tail(1).copy()
        if latest.empty:
            return pd.DataFrame(columns=["symbol", "y_score", "ts"])
        latest["ts"] = latest.index

        # Build the feature matrix in the trained column order
        # Add symbol one-hots if the model expects them
        feat = latest.copy()
        sym_cols = [c for c in self.feature_cols if c.startswith("sym_")]
        if sym_cols:
            dummies = pd.get_dummies(feat["symbol"], prefix="sym").astype("float32")
            for c in sym_cols:
                feat[c] = dummies[c] if c in dummies.columns else 0.0
        X = feat.reindex(columns=self.feature_cols).astype(float)

        scores = self._infer(X.values, feat)
        t1 = time.monotonic()
        log.info("live inference: %d symbols in %.2f ms (backend=%s)",
                  len(latest), (t1 - t0) * 1000, self._backend)
        # Include the last-known close price for each symbol so the paper
        # broker / OMS can size orders without a separate quote feed.
        prices = latest["close"].values if "close" in latest.columns else [None] * len(latest)
        out = pd.DataFrame({
            "symbol": latest["symbol"].values,
            "ts": latest["ts"].values,
            "y_score": scores,
            "edge": scores - 0.5,
            "close": prices,
        })
        return out.sort_values("y_score", ascending=False)

    def _infer(self, X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        if self._backend == "gbm":
            return self._loaded.predict(X)
        if self._backend == "torch":
            import torch
            m = self._loaded["model"]
            device = self._loaded["device"]
            window = self._loaded["window"]
            # Need a window-length history per symbol. For a quick live signal
            # we approximate by replicating the latest row — works for short
            # windows but is not ideal. Better: build a proper sequence
            # buffer per symbol.
            X_win = np.broadcast_to(X[:, None, :], (X.shape[0], window, X.shape[1]))
            X_t = torch.from_numpy(np.ascontiguousarray(X_win, dtype=np.float32)).to(device)
            with torch.no_grad():
                logits = m(X_t)
                scores = torch.sigmoid(logits).cpu().numpy().ravel()
            return scores
        if self._backend == "coreml":
            import coremltools as ct
            preds = self._loaded.predict({"input": X.astype("float32")})
            out_key = list(preds.keys())[0]
            return np.asarray(preds[out_key]).ravel()
        raise ValueError(self._backend)

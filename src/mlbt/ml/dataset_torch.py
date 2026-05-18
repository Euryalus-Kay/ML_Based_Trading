"""PyTorch Dataset that yields rolling windows for sequence models.

Each sample:
  X: (window, n_features)
  y: scalar target (binary or regression)

Walk-forward windows are generated grouped by symbol to avoid mixing tapes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    Dataset = object  # type: ignore


class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: Sequence[str],
                 target_col: str, window: int = 64,
                 symbol_col: str = "symbol",
                 dtype=None):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for SequenceDataset")
        self.window = window
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.dtype = dtype or torch.float32

        # Build (symbol, start_index) pairs for valid windows
        self.samples: list[tuple[np.ndarray, float]] = []
        for sym, sub in df.groupby(symbol_col):
            sub = sub.sort_index()
            X = sub[self.feature_cols].values.astype(np.float32)
            y = sub[self.target_col].values.astype(np.float32)
            if len(sub) <= window:
                continue
            for i in range(window, len(sub)):
                y_i = y[i]
                if np.isnan(y_i):
                    continue
                window_x = X[i - window:i]
                if np.isnan(window_x).any():
                    continue
                self.samples.append((window_x, y_i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.from_numpy(x).to(self.dtype), torch.tensor(y, dtype=self.dtype)

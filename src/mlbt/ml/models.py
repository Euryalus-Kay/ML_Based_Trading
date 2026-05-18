"""Sequence models for short-horizon return prediction.

Three flavours:
  - LSTMClassifier:        simple, fast baseline
  - TransformerClassifier: TCN/transformer hybrid for richer temporal patterns
  - PatchTSTLite:          patch-based transformer (PatchTST flavour, simplified)

All operate on input shape (batch, window, n_features) and output a single
logit (binary up/down). Swap final layer to predict log-return for regression.
"""
from __future__ import annotations

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:

    class LSTMClassifier(nn.Module):
        def __init__(self, n_features: int, hidden: int = 128,
                     num_layers: int = 2, dropout: float = 0.2,
                     bidirectional: bool = False, n_outputs: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features, hidden_size=hidden, num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True, bidirectional=bidirectional,
            )
            mul = 2 if bidirectional else 1
            self.norm = nn.LayerNorm(hidden * mul)
            self.head = nn.Sequential(
                nn.Linear(hidden * mul, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_outputs),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            h = self.norm(out[:, -1, :])
            return self.head(h).squeeze(-1)


    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 1024):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                 -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]


    class TransformerClassifier(nn.Module):
        def __init__(self, n_features: int, d_model: int = 128, n_heads: int = 8,
                     n_layers: int = 4, dropout: float = 0.1, n_outputs: int = 1):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos = PositionalEncoding(d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_outputs),
            )

        def forward(self, x):
            h = self.input_proj(x)
            h = self.pos(h)
            h = self.encoder(h)
            h = h.mean(dim=1)  # pooled across time
            return self.head(h).squeeze(-1)


    class PatchTSTLite(nn.Module):
        """PatchTST-flavoured model: split window into patches, project,
        transform. Much faster than per-step attention for long windows."""
        def __init__(self, n_features: int, window: int, patch_size: int = 8,
                     d_model: int = 128, n_heads: int = 8, n_layers: int = 4,
                     dropout: float = 0.1, n_outputs: int = 1):
            super().__init__()
            assert window % patch_size == 0, "window must be divisible by patch_size"
            self.patch_size = patch_size
            self.n_patches = window // patch_size
            self.input_proj = nn.Linear(n_features * patch_size, d_model)
            self.pos = PositionalEncoding(d_model, max_len=self.n_patches + 1)
            self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(64, n_outputs),
            )

        def forward(self, x):
            b, w, f = x.shape
            x = x.view(b, self.n_patches, self.patch_size * f)
            h = self.input_proj(x)
            cls = self.cls.expand(b, -1, -1)
            h = torch.cat([cls, h], dim=1)
            h = self.pos(h)
            h = self.encoder(h)
            return self.head(h[:, 0]).squeeze(-1)


    # ---------------- XL variants for 64+ GB Macs / Apple Silicon ------------
    class LSTMXL(nn.Module):
        """Bigger LSTM: 4 layers, hidden 384, bidirectional, layer-norm head."""
        def __init__(self, n_features: int, hidden: int = 384, num_layers: int = 4,
                     dropout: float = 0.3, n_outputs: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers,
                                  dropout=dropout, batch_first=True, bidirectional=True)
            mul = 2
            self.norm = nn.LayerNorm(hidden * mul)
            self.head = nn.Sequential(
                nn.Linear(hidden * mul, 256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(64, n_outputs),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            h = self.norm(out[:, -1, :])
            return self.head(h).squeeze(-1)


    class TransformerXL(nn.Module):
        """Bigger Transformer: 8 layers, 8 heads, d_model=256, FFN 1024."""
        def __init__(self, n_features: int, d_model: int = 256, n_heads: int = 8,
                     n_layers: int = 8, dropout: float = 0.15, n_outputs: int = 1):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos = PositionalEncoding(d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(128, n_outputs),
            )

        def forward(self, x):
            h = self.input_proj(x)
            h = self.pos(h)
            h = self.encoder(h)
            h = h.mean(dim=1)
            return self.head(h).squeeze(-1)


    class PatchTSTXL(nn.Module):
        """Bigger PatchTST: d_model=384, 8 heads, 8 layers, larger MLP head."""
        def __init__(self, n_features: int, window: int, patch_size: int = 8,
                     d_model: int = 384, n_heads: int = 8, n_layers: int = 8,
                     dropout: float = 0.15, n_outputs: int = 1):
            super().__init__()
            assert window % patch_size == 0
            self.patch_size = patch_size
            self.n_patches = window // patch_size
            self.input_proj = nn.Linear(n_features * patch_size, d_model)
            self.pos = PositionalEncoding(d_model, max_len=self.n_patches + 1)
            self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(64, n_outputs),
            )

        def forward(self, x):
            b, w, f = x.shape
            x = x.view(b, self.n_patches, self.patch_size * f)
            h = self.input_proj(x)
            cls = self.cls.expand(b, -1, -1)
            h = torch.cat([cls, h], dim=1)
            h = self.pos(h)
            h = self.encoder(h)
            return self.head(h[:, 0]).squeeze(-1)


    class MultiTaskHead(nn.Module):
        """Wraps a backbone and predicts multiple horizons jointly.

        Multi-task heads regularise each other on a shared encoder. Strong for
        short-horizon equity prediction.
        """
        def __init__(self, backbone: nn.Module, n_horizons: int = 4,
                     backbone_out_dim: int = 256):
            super().__init__()
            self.backbone = backbone
            self.heads = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(backbone_out_dim),
                              nn.Linear(backbone_out_dim, 64), nn.GELU(),
                              nn.Linear(64, 1))
                for _ in range(n_horizons)
            ])

        def forward(self, x):
            h = self.backbone(x)
            return torch.stack([head(h).squeeze(-1) for head in self.heads], dim=1)

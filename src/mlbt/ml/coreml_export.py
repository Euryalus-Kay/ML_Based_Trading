"""Export trained PyTorch models to CoreML for Apple Neural Engine inference.

The Neural Engine (ANE) handles inference dramatically faster than the GPU
on Apple Silicon — useful for the live signal-generation path where you
want to score every symbol every bar with negligible latency.

Training stays on MPS (no full ANE support for arbitrary nets); only
inference is converted.

Usage:
    python -m mlbt.ml.coreml_export data/model_patchtst_xl model_input.mlpackage
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def export_to_coreml(model_dir: str, output_path: str,
                      window: int = 64) -> None:
    """Convert a saved PyTorch model.pt to a CoreML .mlpackage."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for export")
    try:
        import coremltools as ct
    except ImportError:
        raise RuntimeError(
            "coremltools not installed. On the Mac: pip install coremltools"
        )

    model_p = Path(model_dir)
    state = torch.load(model_p / "model.pt", map_location="cpu")
    feature_cols = json.loads((model_p / "feature_cols.json").read_text())
    n_features = state.get("n_features", len(feature_cols))
    model_kind = state.get("model_kind", "patchtst_xl")
    window = state.get("window", window)

    from mlbt.ml.models import (
        LSTMClassifier, TransformerClassifier, PatchTSTLite,
        LSTMXL, TransformerXL, PatchTSTXL,
    )
    cls_map = {
        "lstm": LSTMClassifier, "transformer": TransformerClassifier,
        "patchtst": PatchTSTLite,
        "lstm_xl": LSTMXL, "transformer_xl": TransformerXL,
        "patchtst_xl": PatchTSTXL,
    }
    Cls = cls_map[model_kind]
    if model_kind in ("patchtst", "patchtst_xl"):
        ps = 8 if window % 8 == 0 else 4
        model = Cls(n_features=n_features, window=window, patch_size=ps)
    else:
        model = Cls(n_features=n_features)
    model.load_state_dict(state["state_dict"])
    model.eval()

    example = torch.randn(1, window, n_features)
    traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="features", shape=example.shape)],
        compute_units=ct.ComputeUnit.ALL,  # picks ANE when possible
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS13,
    )
    mlmodel.save(output_path)
    print(f"saved CoreML model -> {output_path}")
    print(f"input: features ({example.shape}) float32")
    print(f"compute units: ALL (Neural Engine when available)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python -m mlbt.ml.coreml_export <model_dir> <out.mlpackage>")
        sys.exit(1)
    export_to_coreml(sys.argv[1], sys.argv[2])

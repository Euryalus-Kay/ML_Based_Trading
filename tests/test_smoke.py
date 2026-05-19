"""Network-free smoke tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_imports():
    import mlbt  # noqa: F401
    from mlbt.core.base import DataSource  # noqa: F401
    from mlbt.core.registry import all_sources
    from mlbt.core.storage import Storage  # noqa: F401
    from mlbt.core.timegrid import TimeGrid, asof_merge  # noqa: F401
    from mlbt.pipeline.collect import load_universe  # noqa: F401
    from mlbt.features import (
        add_technical_features, add_cross_asset_features,
        add_microstructure_features, add_targets,
    )  # noqa: F401
    srcs = all_sources()
    names = {s.name for s in srcs}
    # Spot-check a few we expect
    assert "yf_intraday" in names
    assert "treasury_yields" in names
    assert "fred" in names
    assert "binance_klines" in names


def test_universe_loads():
    from mlbt.pipeline.collect import load_universe
    u = load_universe()
    assert "equities" in u
    assert "indices" in u
    assert "fred_series" in u


def test_timegrid_24x7():
    from mlbt.core.timegrid import TimeGrid
    tg = TimeGrid(bar="1h", session="24x7")
    idx = tg.index("2024-01-01", "2024-01-02")
    assert len(idx) >= 23
    assert idx.tz is not None


def test_asof_merge_basic():
    from mlbt.core.timegrid import asof_merge
    grid = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(
            ["2024-01-01 00:30", "2024-01-01 02:30", "2024-01-01 04:30"],
            tz="UTC",
        ),
    )
    out = asof_merge(grid, df, publish_lag=pd.Timedelta(0))
    assert list(out["x"]) == [pytest.approx(np.nan, nan_ok=True), 1.0, 1.0, 2.0, 2.0] or \
        list(out["x"][1:]) == [1.0, 1.0, 2.0, 2.0]


def test_technical_features_smoke():
    from mlbt.features.technical import add_technical_features
    idx = pd.date_range("2024-01-01", periods=200, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 + rng.standard_normal(200).cumsum()
    df = pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": rng.integers(1000, 5000, 200),
    }, index=idx)
    out = add_technical_features(df)
    assert "rsi_14" in out.columns
    assert "bb_pos" in out.columns
    assert "log_ret" in out.columns
    assert len(out) == 200


def test_targets_smoke():
    from mlbt.features.targets import add_targets
    idx = pd.date_range("2024-01-01", periods=300, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 + rng.standard_normal(300).cumsum()
    df = pd.DataFrame({"close": close}, index=idx)
    out = add_targets(df, horizons=(1, 3))
    assert "y_logret_1" in out
    assert "y_up_3" in out
    # Forward returns NaN at the tail
    assert out["y_logret_3"].iloc[-3:].isna().all()


def test_storage_roundtrip(tmp_path, monkeypatch):
    from mlbt.core.storage import Storage
    monkeypatch.setenv("MLBT_DATA_DIR", str(tmp_path))
    st = Storage(root=str(tmp_path))
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    df = pd.DataFrame({"a": np.arange(10, dtype=float)}, index=idx)
    st.write("source_x", "key_y", df)
    out = st.read("source_x", "key_y")
    assert len(out) == 10
    assert "a" in out.columns

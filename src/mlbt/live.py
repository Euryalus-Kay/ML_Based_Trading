"""Real-time inference path.

The trained models live on disk (data/model_*/model.pkl + feature_cols.json).
At inference time we need to materialise THE SAME feature row(s) as during
training, on the most recent observable data, and call model.predict.

This module:
  - audit_sources(): probes every registered DataSource for freshness
    (last_observation_age, expected_latency). Use to validate the live path
    BEFORE wiring to capital.
  - live_features(): pulls the latest data, aligns, featurises a single
    (latest) row per symbol in the universe.
  - live_predict(): loads the latest GBM, computes live features, returns
    a per-symbol P(up) score sorted descending. This is the deployable
    signal generator.
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
from mlbt.core.registry import all_sources
from mlbt.core.storage import Storage
from mlbt.pipeline.collect import load_universe
from mlbt.pipeline.dataset import build_dataset

log = get_logger("live")


def audit_sources(lookback_hours: int = 48) -> pd.DataFrame:
    """Probe every registered source: how fresh is its most recent observation?

    Pulls a small recent slice from each, measures (now - last_ts) and
    publish_lag, returns a tidy table.
    """
    now = pd.Timestamp.utcnow().tz_convert("UTC")
    start = now - pd.Timedelta(hours=lookback_hours)
    rows = []
    universe = load_universe()
    target_symbols: list[str] = []
    for group in universe.get("equities", {}).values():
        target_symbols.extend(group)
    target_symbols = sorted(set(target_symbols))

    for src in all_sources():
        if not src.is_available():
            rows.append({"source": src.name, "available": False,
                          "last_observation_age": None, "rows": 0, "note": "key missing"})
            continue
        kwargs = {}
        if src.name in ("yf_intraday", "yf_1h"):
            kwargs["symbols"] = target_symbols[:5]
            kwargs["interval"] = "1h" if src.name == "yf_1h" else "5m"
        elif src.name == "yf_daily":
            kwargs["symbols"] = target_symbols[:5]
        elif src.name == "yf_options":
            kwargs["symbols"] = target_symbols[:3]
        elif src.name == "binance_klines":
            kwargs["symbols"] = universe.get("crypto_binance", ["BTCUSDT"])[:1]
            kwargs["interval"] = "1h"
        elif src.name == "fred":
            kwargs["series"] = universe.get("fred_series", ["DGS10"])[:3]
        elif src.name == "wiki_pageviews":
            kwargs["pages"] = universe.get("wikipedia_pages", ["Apple_Inc."])[:1]
        elif src.name in ("sec_edgar", "finra_short"):
            kwargs["symbols"] = target_symbols[:5]
        t0 = time.monotonic()
        df = src.fetch_safe(start, now, **kwargs)
        latency = time.monotonic() - t0
        if df.empty:
            rows.append({"source": src.name, "available": True,
                          "last_observation_age": None, "rows": 0,
                          "fetch_seconds": round(latency, 2), "publish_lag": str(src.publish_lag),
                          "note": "no rows returned"})
            continue
        last_ts = df.index.max()
        age = (now - last_ts).total_seconds() / 60.0  # minutes
        rows.append({"source": src.name, "available": True,
                      "last_observation_age_min": round(age, 1),
                      "rows": len(df), "fetch_seconds": round(latency, 2),
                      "publish_lag": str(src.publish_lag),
                      "latest_ts": str(last_ts)})

    out = pd.DataFrame(rows)
    log.info("audit_sources rows=%d", len(out))
    return out


def _select_features_for_target(df: pd.DataFrame, target: str,
                                  feature_cols: list[str]) -> pd.DataFrame:
    """Subset df to known feature_cols; fill any newly-missing cols with NaN."""
    have = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.warning("live features missing %d cols (model expects them): %s ...",
                    len(missing), missing[:5])
    sub = df.reindex(columns=feature_cols)
    return sub


def live_predict(model_dir: str = "data/model_gbm",
                  bar: str = "1h",
                  lookback_days: int = 60,
                  universe_path: Optional[str] = None) -> pd.DataFrame:
    """Pull latest data → build features → score with the trained model.

    Returns: DataFrame columns [symbol, ts, score, ...] sorted by score desc.
    """
    model_p = Path(model_dir)
    feature_cols = json.loads((model_p / "feature_cols.json").read_text())
    with open(model_p / "model.pkl", "rb") as f:
        booster = pickle.load(f)

    end = pd.Timestamp.utcnow().tz_convert("UTC")
    start = end - pd.Timedelta(days=lookback_days)

    # The simplest correct path: re-run the pipeline on the latest data window
    # so identical feature engineering is guaranteed. For production, cache
    # historical bars and only fetch the last few.
    ds = build_dataset(start, end, bar=bar, session="rth",
                       universe_path=universe_path,
                       horizons=(1, 2, 4, 8),
                       out_path=None,
                       only_sources=["yf_1h", "fred", "treasury_yields",
                                      "calendar_events", "crypto_fear_greed"])
    if ds.empty:
        log.warning("live: empty dataset built")
        return pd.DataFrame()

    # Take the most recent row per symbol (the one we'd trade NOW)
    latest = ds.groupby("symbol").tail(1).copy()
    latest["ts"] = latest.index
    X = _select_features_for_target(latest, target="(live)", feature_cols=feature_cols)
    score = booster.predict(X.values) if hasattr(booster, "predict") else booster.predict_proba(X.values)[:, 1]
    out = pd.DataFrame({
        "symbol": latest["symbol"].values,
        "ts": latest["ts"].values,
        "score": score,
        "edge": score - 0.5,
        "signal": np.sign(score - 0.5),
    }).sort_values("score", ascending=False)
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "audit":
        print(audit_sources().to_string(index=False))
    else:
        print(live_predict().to_string(index=False))

"""Honest realism audit for trained models.

A claimed Sharpe is only as good as the assumptions behind it. This script
runs each model under several increasingly conservative regimes and prints
how the metrics degrade. If the edge collapses under realistic conditions,
the model is not actually deployable.

Regimes audited:
  - HEROIC       : 0 bps cost, no slippage, instant execution (no entry lag), no overlap scaling
                   (this is what naive backtests usually report)
  - PAPER        : 2 bps cost, 1 bp slippage, 1-bar entry lag, overlap-scaled (current default)
  - RETAIL       : 5 bps cost + 2 bps slippage + 1-bar lag, overlap-scaled
                   (realistic for a retail account on liquid US large-caps)
  - SMALL_CAP    : 10 bps cost + 5 bps slippage (realistic if some of the
                   universe is mid/small cap)
  - DEFENSIVE    : RETAIL + larger min_edge=0.05 (only trade strong signals)

For each regime: net Sharpe, ann return, max DD, hit rate, trade count,
beats_spy boolean. Reports a single "robustness verdict" per model — what
fraction of regimes keep it profitable.
"""
from __future__ import annotations

import json
from glob import glob
from pathlib import Path
from typing import Sequence

import pandas as pd

from mlbt.backtest.engine import BacktestConfig, run_backtest
from mlbt.core.log import get_logger

log = get_logger("realism_audit")


REGIMES = {
    "HEROIC":    dict(bps_per_trade=0.0, slippage_bps=0.0, entry_lag_bars=0,
                       overlap_scale=False, min_edge=0.0),
    "PAPER":     dict(bps_per_trade=2.0, slippage_bps=1.0, entry_lag_bars=1,
                       overlap_scale=True, min_edge=0.01),
    "RETAIL":    dict(bps_per_trade=5.0, slippage_bps=2.0, entry_lag_bars=1,
                       overlap_scale=True, min_edge=0.01),
    "SMALL_CAP": dict(bps_per_trade=10.0, slippage_bps=5.0, entry_lag_bars=1,
                       overlap_scale=True, min_edge=0.01),
    "DEFENSIVE": dict(bps_per_trade=5.0, slippage_bps=2.0, entry_lag_bars=1,
                       overlap_scale=True, min_edge=0.05),
}


def _horizon_from_target(target: str) -> int:
    try:
        return int(target.rsplit("_", 1)[-1])
    except Exception:
        return 1


def _is_profitable(net: dict, spy: dict) -> bool:
    return (
        (net.get("ann_return") or 0) > (spy.get("ann_return") or 0)
        and (net.get("sharpe") or 0) > 0.3  # liberal floor for "real edge"
        and (net.get("max_drawdown") or -1) > -0.30
    )


def audit_models(experiments_glob: str = "data/experiments*",
                  dataset_paths: dict | None = None,
                  out_dir: str = "data/realism") -> pd.DataFrame:
    """Sweep every trained model through each regime."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    dataset_paths = dataset_paths or {}

    rows = []
    for exp_root in sorted(glob(experiments_glob)):
        exp_root_p = Path(exp_root)
        # default dataset path = sibling of exp_root
        default_ds = "data/dataset_1h.parquet" if "1h" in exp_root or "xsec" in exp_root or "tournament" in exp_root else (
            "data/dataset_daily.parquet" if "daily" in exp_root else "data/dataset.parquet")
        ds_path = dataset_paths.get(exp_root_p.name, default_ds)
        if not Path(ds_path).exists():
            continue
        # Each subdirectory is a model
        for model_dir in sorted(exp_root_p.iterdir()):
            if not model_dir.is_dir():
                continue
            metrics_p = model_dir / "metrics.json"
            preds_p = model_dir / "predictions.parquet"
            if not metrics_p.exists() or not preds_p.exists():
                continue
            metrics = json.loads(metrics_p.read_text())
            target = metrics.get("target", "y_up_1")
            h = _horizon_from_target(target)
            target_col = (f"y_resid_logret_{h}"
                          if ("resid" in target or "xsec" in target) else f"y_logret_{h}")
            # If the column doesn't exist in the dataset, fall back
            try:
                cols = pd.read_parquet(ds_path, columns=None).columns
            except Exception:
                continue
            if target_col not in cols:
                target_col = f"y_logret_{h}" if f"y_logret_{h}" in cols else cols[0]

            bar = "1h" if "1h" in ds_path else ("1d" if "daily" in ds_path else "5min")

            row_base = {
                "model": f"{exp_root_p.name}/{model_dir.name}",
                "target": target, "h": h, "bar": bar,
                "train_acc": metrics.get("agg_accuracy"),
            }
            for regime_name, kw in REGIMES.items():
                cfg = BacktestConfig(horizon=h, target_col=target_col, bar=bar, **kw)
                try:
                    bt = run_backtest(dataset_path=ds_path,
                                       model_dir=str(model_dir),
                                       out_path=None, cfg=cfg)
                except Exception as e:  # noqa: BLE001
                    log.warning("%s @ %s failed: %s", model_dir.name, regime_name, e)
                    continue
                net = bt.get("net", {})
                spy = bt.get("benchmark_spy") or bt.get("benchmark_long_always") or {}
                rows.append({**row_base,
                              "regime": regime_name,
                              "net_sharpe": net.get("sharpe"),
                              "net_ann_return": net.get("ann_return"),
                              "max_dd": net.get("max_drawdown"),
                              "trade_count": bt.get("trade_count"),
                              "hit_rate": net.get("hit_rate"),
                              "directional_acc": bt.get("directional_accuracy"),
                              "spy_sharpe": spy.get("sharpe"),
                              "spy_ann_return": spy.get("ann_return"),
                              "beats_spy": _is_profitable(net, spy)})
    if not rows:
        log.warning("no models found")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv(out_p / "realism_audit.csv", index=False)

    # Pivot: model x regime -> net_sharpe and beats_spy
    sharpe_pv = df.pivot_table(index="model", columns="regime", values="net_sharpe")
    beats_pv = df.pivot_table(index="model", columns="regime", values="beats_spy")
    # Robustness verdict
    robust = beats_pv.fillna(False).astype(bool).sum(axis=1)
    print("\n=== Realism audit — net Sharpe per regime ===")
    print(sharpe_pv.round(2).to_string())
    print("\n=== Beats SPY? (True=yes) ===")
    print(beats_pv.astype(str).to_string())
    print("\n=== Robustness verdict (number of regimes beats SPY, out of",
          len(REGIMES), ") ===")
    print(robust.sort_values(ascending=False).to_string())
    return df


if __name__ == "__main__":
    audit_models()

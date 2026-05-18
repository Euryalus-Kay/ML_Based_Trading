"""Event-driven-ish vectorised backtester for short-horizon equity signals.

Design:
  - Each (timestamp, symbol) has a model score s_t ∈ [0,1] (prob of up move).
  - Convert score to a target position: position = 2*(s - 0.5) clipped to
    [-max_leverage, +max_leverage]. Optionally apply a confidence threshold
    so we trade only when |s - 0.5| > min_edge.
  - Position is held for `horizon` bars (matches the label horizon).
  - PnL per bar = position_t * forward_logret_{t+1}.
  - Transaction cost: bps applied to |Δposition|.
  - Compute strategy return series, equity curve, drawdown, Sharpe, hit rate.

The engine consumes the predictions.parquet written by training (cols:
y_true, y_score, indexed by bar timestamp) and the source dataset.parquet
to recover forward returns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mlbt.core.log import get_logger

log = get_logger("backtest")


@dataclass
class BacktestConfig:
    horizon: int = 3                 # bars to hold (must match training target)
    bar: str = "5min"
    bps_per_trade: float = 2.0       # 2 bps each side; round-trip = 4 bps
    max_leverage: float = 1.0
    min_edge: float = 0.02           # only trade if |score - 0.5| > min_edge
    long_only: bool = False
    target_col: str = "y_logret_3"   # forward log return matching horizon


def _equity_metrics(returns: pd.Series, bars_per_year: float) -> dict:
    if returns.empty:
        return {}
    mu = returns.mean()
    sd = returns.std()
    sharpe = (mu / sd) * np.sqrt(bars_per_year) if sd > 0 else 0.0
    equity = (1 + returns).cumprod()
    max_eq = equity.cummax()
    drawdown = (equity / max_eq - 1.0)
    return {
        "n_bars": int(len(returns)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "ann_return": float(mu * bars_per_year),
        "ann_vol": float(sd * np.sqrt(bars_per_year)),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((returns > 0).mean()),
        "turnover_avg": None,  # filled by caller
    }


def _bars_per_year(bar: str) -> float:
    # rough — 252 trading days x bars per day
    bar = bar.lower()
    if bar == "1min":
        return 252 * 390
    if bar == "5min":
        return 252 * 78
    if bar == "15min":
        return 252 * 26
    if bar == "1h":
        return 252 * 6.5
    if bar == "1d":
        return 252
    return 252


def run_backtest(dataset_path: str, model_dir: str,
                  out_path: Optional[str] = None,
                  cfg: Optional[BacktestConfig] = None) -> dict:
    cfg = cfg or BacktestConfig()
    model_dir_p = Path(model_dir)
    preds_path = model_dir_p / "predictions.parquet"
    if not preds_path.exists():
        # If no predictions on disk (e.g. trained an LSTM), score the dataset now
        raise FileNotFoundError(f"predictions.parquet not found in {model_dir} — "
                                f"run training first (gbm writes it automatically)")

    preds = pd.read_parquet(preds_path)
    ds = pd.read_parquet(dataset_path)
    if cfg.target_col not in ds.columns:
        raise KeyError(f"target column {cfg.target_col} not in dataset; "
                       f"have: {[c for c in ds.columns if c.startswith('y_')][:8]}")

    # Join predictions to forward returns. If both carry 'symbol', join on
    # (timestamp, symbol) to keep per-symbol PnL; otherwise fall back to a
    # cross-sectional mean.
    if "symbol" in preds.columns and "symbol" in ds.columns:
        p = preds.reset_index().rename(columns={"index": "ts"})
        d = ds.reset_index().rename(columns={"index": "ts"})[["ts", "symbol", cfg.target_col]]
        df = p.merge(d, on=["ts", "symbol"], how="inner")
        df = df.rename(columns={cfg.target_col: "fwd_ret"}).set_index("ts")
    else:
        fwd_ret = ds[cfg.target_col].groupby(level=0).mean()
        df = preds.join(fwd_ret.rename("fwd_ret"), how="inner")
    df = df.dropna(subset=["fwd_ret", "y_score"])
    if df.empty:
        return {"error": "no overlapping rows between predictions and dataset"}

    s = df["y_score"].astype(float)
    edge = (s - 0.5)
    raw_pos = 2.0 * edge
    raw_pos = raw_pos.clip(-cfg.max_leverage, cfg.max_leverage)
    if cfg.long_only:
        raw_pos = raw_pos.clip(lower=0)
    raw_pos = raw_pos.where(edge.abs() > cfg.min_edge, 0.0)
    df = df.assign(raw_pos=raw_pos)

    # Per-symbol position diff for turnover; per-bar mean PnL for aggregate.
    if "symbol" in df.columns:
        df = df.sort_values("symbol").sort_index(kind="stable")
        df["dpos"] = (
            df.groupby("symbol")["raw_pos"]
              .transform(lambda x: x.diff().abs().fillna(x.abs()))
        )
        df["pnl_per_row"] = df["raw_pos"] * df["fwd_ret"]
        # Aggregate across symbols at each timestamp (equal-weight book)
        per_bar = df.groupby(level=0).agg(
            strat_ret=("pnl_per_row", "mean"),
            dpos=("dpos", "mean"),
            fwd_ret=("fwd_ret", "mean"),
        )
        strat_ret = per_bar["strat_ret"]
        dpos = per_bar["dpos"]
        bench = per_bar["fwd_ret"]
    else:
        dpos = raw_pos.diff().abs().fillna(raw_pos.abs())
        strat_ret = raw_pos * df["fwd_ret"]
        bench = df["fwd_ret"]
    cost = dpos * (cfg.bps_per_trade / 1e4)
    net_ret = strat_ret - cost

    bpy = _bars_per_year(cfg.bar)
    gross_m = _equity_metrics(strat_ret, bpy)
    net_m = _equity_metrics(net_ret, bpy)
    net_m["turnover_avg"] = float(dpos.mean())
    gross_m["turnover_avg"] = float(dpos.mean())

    # Buy-and-hold benchmark (already computed above)
    bench_m = _equity_metrics(bench, bpy)

    report = {
        "config": cfg.__dict__,
        "gross": gross_m,
        "net": net_m,
        "benchmark_long_always": bench_m,
        "directional_accuracy": float((
            (df["raw_pos"] != 0) &
            (np.sign(df["raw_pos"]) == np.sign(df["fwd_ret"]))
        ).sum() / max(1, (df["raw_pos"] != 0).sum())),
        "n_predictions": int(len(df)),
        "trade_count": int((dpos > 0).sum()),
    }

    # Persist
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render_html_report(df, raw_pos, net_ret, report))
        json_path = out.with_suffix(".json")
        json_path.write_text(json.dumps(report, indent=2, default=str))
        log.info("wrote backtest report %s and %s", out, json_path)

    return report


def _render_html_report(df: pd.DataFrame, pos: pd.Series, net_ret: pd.Series, report: dict) -> str:
    eq = (1 + net_ret).cumprod()
    # Minimal HTML — keeps deps light
    rows = [
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in report["net"].items()
    ]
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>mlbt backtest</title>
<style>
body{{font-family:system-ui,sans-serif;padding:24px;max-width:900px}}
table{{border-collapse:collapse;margin-bottom:24px}}
td,th{{border:1px solid #ddd;padding:6px 10px}}
h2{{margin-top:32px}}
</style></head><body>
<h1>mlbt backtest</h1>
<h2>Net metrics (after costs)</h2>
<table>{''.join(rows)}</table>
<h2>Directional accuracy</h2>
<p><b>{report['directional_accuracy']:.4f}</b> on {report['n_predictions']} predictions
(trades made: {report['trade_count']})</p>
<h2>Config</h2>
<pre>{json.dumps(report['config'], indent=2)}</pre>
<h2>Benchmark (long always)</h2>
<pre>{json.dumps(report['benchmark_long_always'], indent=2)}</pre>
</body></html>"""

"""CLI entry point."""
from __future__ import annotations

import sys

import click

from mlbt.core.log import get_logger
from mlbt.core.registry import all_sources
from mlbt.core.storage import Storage
from mlbt.pipeline.collect import collect_all
from mlbt.pipeline.align import build_aligned_frame
from mlbt.pipeline.dataset import build_dataset

log = get_logger("cli")


@click.group()
def main():
    """mlbt — multi-source, time-aligned market data collector."""


@main.command("list-sources")
def list_sources():
    """List every registered data source and its key requirement."""
    rows = []
    for s in all_sources():
        rows.append((s.name, s.frequency, str(s.publish_lag),
                     s.requires_key or "-", "yes" if s.is_available() else "skip"))
    print(f"{'name':<22} {'freq':<8} {'lag':<14} {'key':<22} {'enabled'}")
    print("-" * 76)
    for r in rows:
        print(f"{r[0]:<22} {r[1]:<8} {r[2]:<14} {r[3]:<22} {r[4]}")


@main.command("collect")
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--universe", default=None)
@click.option("--only", multiple=True, help="restrict to named sources")
def cmd_collect(start, end, universe, only):
    """Pull every enabled source for the configured universe into raw store."""
    summary = collect_all(start, end, universe_path=universe,
                          only=list(only) if only else None)
    for k, v in summary.items():
        print(f"{k:<24} {v:>10} rows")


@main.command("align")
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--bar", default="5min")
@click.option("--session", default="rth", type=click.Choice(["rth", "full", "24x7"]))
@click.option("--out", default=None)
def cmd_align(start, end, bar, session, out):
    storage = Storage()
    df = build_aligned_frame(start, end, bar=bar, session=session, storage=storage)
    print(f"aligned frame: {df.shape[0]} rows x {df.shape[1]} cols")
    if out:
        df.to_parquet(out, engine="pyarrow")
        print(f"wrote {out}")


@main.command("build-dataset")
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--bar", default="5min")
@click.option("--session", default="rth", type=click.Choice(["rth", "full", "24x7"]))
@click.option("--universe", default=None)
@click.option("--out", default="data/dataset.parquet")
@click.option("--horizons", default="1,3,6,12")
@click.option("--only-source", multiple=True,
                help="Restrict aligned frame to given sources (repeatable)")
def cmd_build_dataset(start, end, bar, session, universe, out, horizons, only_source):
    """Build per-symbol feature+target dataset for ML training."""
    h = tuple(int(x) for x in horizons.split(","))
    df = build_dataset(start, end, bar=bar, session=session,
                       universe_path=universe, horizons=h, out_path=out,
                       only_sources=list(only_source) if only_source else None)
    print(f"dataset: {df.shape[0]} rows x {df.shape[1]} cols -> {out}")


@main.command("backfill")
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--bar", default="5min")
@click.option("--universe", default=None)
def cmd_backfill(start, end, bar, universe):
    """Convenience: collect + align + build-dataset in one shot."""
    collect_all(start, end, universe_path=universe)
    df = build_dataset(start, end, bar=bar, universe_path=universe,
                       out_path="data/dataset.parquet")
    print(f"backfill complete: dataset has {df.shape[0]} rows, {df.shape[1]} cols")


@main.command("train")
@click.option("--dataset", default="data/dataset.parquet")
@click.option("--target", default="y_up_3")
@click.option("--model", type=click.Choice(["gbm", "lstm"]), default="gbm")
@click.option("--out", default="data/model")
def cmd_train(dataset, target, model, out):
    """Train an ML model on the built dataset."""
    from mlbt.ml.train import train_model
    metrics = train_model(dataset_path=dataset, target=target, model=model, out_dir=out)
    print("training metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


@main.command("backtest")
@click.option("--dataset", default="data/dataset.parquet")
@click.option("--model-dir", default="data/model")
@click.option("--out", default="data/backtest_report.html")
def cmd_backtest(dataset, model_dir, out):
    from mlbt.backtest.engine import run_backtest
    report = run_backtest(dataset_path=dataset, model_dir=model_dir, out_path=out)
    print("backtest:")
    for k, v in report.items():
        print(f"  {k}: {v}")


@main.command("audit-sources")
@click.option("--hours", default=48, help="Lookback window for each source")
def cmd_audit_sources(hours):
    """Probe each data source: is it real-time accessible? How fresh?"""
    from mlbt.live import audit_sources
    df = audit_sources(lookback_hours=hours)
    print(df.to_string(index=False))


@main.command("live-predict")
@click.option("--model-dir", default="data/model_gbm")
@click.option("--bar", default="1h")
@click.option("--lookback-days", default=60)
def cmd_live_predict(model_dir, bar, lookback_days):
    """Pull latest data, featurise, score with the trained model. Output ranked signals."""
    from mlbt.live import live_predict
    df = live_predict(model_dir=model_dir, bar=bar, lookback_days=lookback_days)
    if df.empty:
        print("(no live predictions — check audit-sources)")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

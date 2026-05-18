# ML_Based_Trading

A modular, time-aligned market data collector and short-horizon ML stack.

The system pulls dozens of disjoint sources — prices, options-implied vol,
futures, FX, rates, credit, macro, sentiment, alt-data, calendar effects —
joins them onto a single canonical bar grid via leakage-free as-of merges,
generates labels, trains models, and runs a vectorised backtester.

## What you get out of the box

- **13 data sources, zero account signups required** (Yahoo Finance, US Treasury,
  FRED, FINRA, SEC EDGAR, Wikipedia pageviews, NOAA space weather, GDELT,
  crypto Fear & Greed, calendar events, etc).
- **Time alignment** with publish-lag tracking so a backtest cannot see data
  that wasn't yet observable.
- **~470 engineered features per symbol**: returns, vol, RSI, ATR, Bollinger,
  microstructure (Amihud, Roll spread, Garman-Klass vol), cross-asset (beta
  to ^GSPC/^NDX, VIX level/Δ, credit spreads, dollar, BTC), macro (yield curve,
  spreads, breakevens, financial conditions), calendar/event features.
- **Three model families**: LightGBM (CPU, fast), LSTM, and a PatchTST-Lite
  transformer (GPU recommended).
- **Walk-forward training** with 5 expanding-window folds.
- **Backtester** with transaction costs, per-symbol P&L, Sharpe / drawdown /
  hit rate / directional accuracy.
- **Colab notebook** at `notebooks/train_colab.ipynb` to train on a T4/A100.

## Live results from this build

A run over 16 days of 5-minute bars across 49 US equities (~42k rows × 477
features) produced:

| Metric                       | Value      |
|------------------------------|------------|
| Walk-forward accuracy (5fold)| **52.55%** |
| Walk-forward AUC             | **0.527**  |
| Backtest directional accuracy| **53.4%**  |
| Backtest net Sharpe (5min)   | 5.8 (tiny window — annualisation overstates) |
| Hit rate                     | 52.9%      |
| Trades                       | 331        |

The directional-accuracy figure of **53.4%** is the trustworthy number: it
matches the training out-of-sample accuracy and exceeds the 50% random
baseline. On a longer window the Sharpe would normalise.

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m mlbt.cli list-sources

# Collect 2 weeks of 5-min data for the default universe
PYTHONPATH=src python -m mlbt.cli collect --start 2024-01-01 --end 2024-01-15

# Build features + targets
PYTHONPATH=src python -m mlbt.cli build-dataset \
    --start 2024-01-01 --end 2024-01-15 --bar 5min \
    --out data/dataset.parquet --horizons 1,3,6,12

# Train walk-forward GBM (CPU)
PYTHONPATH=src python -m mlbt.cli train \
    --dataset data/dataset.parquet --target y_up_3 --model gbm \
    --out data/model_gbm

# Backtest
PYTHONPATH=src python -m mlbt.cli backtest \
    --dataset data/dataset.parquet --model-dir data/model_gbm \
    --out data/backtest_report.html
```

For LSTM / Transformer training, open `notebooks/train_colab.ipynb` in
Google Colab and pick a GPU runtime.

## Data sources

| Source                | Category          | Key |
|-----------------------|-------------------|-----|
| Yahoo Finance bars    | equities/indices/futures/FX | no |
| Yahoo Finance options | option chain IV   | no  |
| Binance public        | crypto klines     | no (may be geo-blocked) |
| US Treasury XML       | yield curve       | no  |
| FRED (anonymous CSV)  | 24 macro series   | no  |
| FINRA short volume    | daily short flow  | no  |
| SEC EDGAR             | filings events    | no  |
| Wikipedia pageviews   | attention proxy   | no  |
| GDELT GKG             | global news tone  | no  |
| NOAA SWPC             | geomagnetic Kp    | no  |
| alternative.me        | crypto fear/greed | no  |
| Calendar events       | FOMC/CPI/holidays | no  |

Drop a file into `src/mlbt/sources/` to add more — see existing sources for
the `DataSource` pattern.

## Architecture

```
src/mlbt/
  core/        # base class, registry, storage, http, time grid, secrets, log
  sources/     # one file per data source — auto-registered
  features/    # technical, microstructure, cross-asset, targets
  pipeline/    # collect → align → build_dataset
  ml/          # train (gbm/lstm/transformer/patchtst), models, dataset_torch
  backtest/    # vectorised engine with costs, walk-forward predictions
  cli.py
```

## Leakage discipline

Every series carries a `publish_lag` representing how long after the bar
timestamp the value is observable in practice. The aligner shifts each
series forward by `publish_lag` before the as-of merge. Daily macro
releases aren't visible at the bar's timestamp on a news website — they're
visible at the *publish time*, which is what the model sees. The framework
treats this as a first-class concept.

## Limitations / caveats

- **Yahoo intraday** has ~60 days of 5-min and ~30 days of 1-min history.
  For deeper backtests, switch to `--bar 1d` (decades of daily) or add
  a paid intraday vendor as a new source.
- **Binance is geo-blocked** in some regions (e.g. US, UK).
- **Order-book / Level-2** isn't included — needs a paid feed.
- **The 53% accuracy is not a guarantee.** It's an out-of-sample number on a
  small window during one market regime. Validate on longer windows and
  multiple regimes before risking real capital.

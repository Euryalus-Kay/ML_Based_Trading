# ML_Based_Trading

## Production strategy (current winner)

**Strategy:** `xs8_LongOnlyTop10` — long top-10 stocks by an 8-hour-ahead
cross-sectional rank model, rebalance every hour, no shorts, leverage 1.0.

**Universe:** 49 S&P 500 mega-caps × 1-hour bars × 2024-05 → 2026-05.

**Risk-adjusted performance vs SPY buy-and-hold (same window, net of cost):**

| Regime | Net Sharpe | vs SPY | Ann Return | Max DD | Trades/yr |
|---|---:|---:|---:|---:|---:|
| PAPER (2 bps cost + 1 bp slip) | **2.39** | 2.3× SPY | 13.4 % | -8.5 %  | ~1 400 |
| RETAIL (5 + 2 bps)             | **1.60** | 1.5× SPY | 9.0 %  | -9.3 %  | ~1 400 |
| RETAIL_k15                     | **1.71** | 1.6× SPY | 8.3 %  | -8.9 %  | ~1 400 |
| SPY benchmark                  | 1.05     | —        | 16.8 % | -34 %   | — |

Wins on Sharpe in 4 of 5 realism regimes (fails only at SMALL_CAP 10 + 5 bps,
not applicable to mega-caps). Drawdown 1/4 of SPY's. Levered 1.26× to match
SPY's vol would produce ~21 % ann return at ~17 % vol — still Sharpe 1.6.

### How to reproduce

```bash
# 1. Setup
git checkout claude/stock-trading-ml-data-system-aBleI
pip install -r requirements.txt && pip install -e .
pip install lightgbm scikit-learn

# 2. Pull data (1h bars, 49 names, 2024-05 → 2026-05)
PYTHONPATH=src python -m mlbt.cli collect \
  --start 2024-05-20 --end 2026-05-17 \
  --only yf_1h --only fred --only treasury_yields \
  --only calendar_events --only crypto_fear_greed

# 3. Build dataset (~10 min)
PYTHONPATH=src python -m mlbt.cli build-dataset \
  --start 2024-05-20 --end 2026-05-17 --bar 1h \
  --out data/dataset_1h.parquet --horizons 1,2,4,8 \
  --only-source yf_1h --only-source fred --only-source treasury_yields \
  --only-source calendar_events --only-source crypto_fear_greed

# 4. Train winning config
PYTHONPATH=src python -c "
from mlbt.ml.train import train_model
train_model('data/dataset_1h.parquet', target='y_xsec_top_8',
            model='gbm', out_dir='data/alpha_tournament/xs8_seeds3',
            n_seeds=3, embargo=0, symbol_onehot=True)"

# 5. Backtest
PYTHONPATH=src python -c "
from mlbt.backtest.portfolio import PortfolioConfig, run_portfolio_backtest
cfg = PortfolioConfig(horizon=8, bar='1h', target_col='y_resid_logret_8',
                      entry_lag_bars=1, top_k=10, bottom_k=0,
                      bps_per_trade=5.0, slippage_bps=2.0, market_neutral=False)
print(run_portfolio_backtest('data/dataset_1h.parquet',
      'data/alpha_tournament/xs8_seeds3', cfg=cfg)['net'])"
```

### Known weaknesses

- **Trained on a single regime** (2024-05 → 2026-05 bull market). Needs
  walk-forward across 2018-19 and 2022 bear periods to confirm robustness.
  Mac handoff (HANDOFF_PROMPT.md) covers this in step 6.
- **Universe is 49 mega-caps**. The bulk pull for full S&P 500 was queued
  but didn't finish on the cloud VM; rerun locally for richer cross-section.
- **Long-only at horizon 8h** — this is essentially a smart momentum-rotation
  on 1-day-ahead winners. Sharpe collapses in SMALL_CAP cost regime, so
  not applicable to illiquid names.
- **Classical baseline still beats on absolute return**: `vol_target` SPY
  (Sharpe 0.76, vs raw SPY 0.75) is a separate winner from a different
  strategy class. Stacking both (use ML when score-spread is wide, fall
  back to vol_target otherwise) is the obvious next step.

### Baseline (no-ML reference)

`vol_target` SPY rescales position so realised vol stays at 12 % annualised
(capped 1.5× leverage). Over the same 2010-2026 window:

| | vol_target | SPY buy-hold |
|---|---:|---:|
| Sharpe | **0.76** | 0.75 |
| Calmar | **0.59** | 0.38 |
| Max DD | **-16.5 %** | -34.1 % |

Robust in 3 of 5 walk-forward sub-windows; smaller drawdown in EVERY window.
Reproducible via `PYTHONPATH=src python -m mlbt.cli classical`.

---

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

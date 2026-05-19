# ML_Based_Trading

## Live trading — one-page quickstart

```bash
# 1. Paper-trade against Alpaca (default; safe)
export ALPACA_KEY=...          # from alpaca.markets paper sign-up
export ALPACA_SECRET=...
PYTHONPATH=src python -m mlbt.trading.runner \
    --model-dir data/cand_xs8_v1 \
    --bar 1h --top-k 10 --rebalance-every 8 \
    --broker alpaca --paper

# 2. One-shot signal (no orders, just print the top-10 right now)
PYTHONPATH=src python -m mlbt.trading.runner --model-dir data/cand_xs8_v1 --once

# 3. Replay any model through the LIVE stack on historical data
PYTHONPATH=src python -m mlbt.trading.backtest_runner \
    --model-dir data/cand_xs8_v1 --dataset data/dataset_1h_micro.parquet \
    --top-k 10 --rebalance-every 8 --bps 5 --slippage 2
```

The live runner = the backtest runner with the broker swapped out — identical
code paths for signal generation, OMS, risk management, and position
tracking. Promoting to live money is one flag (`--live`) once you trust the
paper P&L. State is persisted to `data/trading_state/` (positions, risk
ledger, last cycle) so the process can crash-restart cleanly.

### Architecture

  `mlbt.trading.signal.LiveSignalGenerator` — load model + rebuild features
    on the most-recent data → per-symbol score in <1 sec for the 42-cap
    universe (GBM backend, CPU). Inference auto-selects CoreML → MPS → CPU.

  `mlbt.trading.oms.OrderManagementSystem` — score table → top_k / bottom_k
    target weights → orders against current positions (min_order_dollars
    dust filter).

  `mlbt.trading.positions.PositionTracker` — VWAP, realised/unrealised P&L,
    JSON-persisted state.

  `mlbt.trading.risk.RiskManager` — per-symbol cap (15 %), gross-exposure
    cap (150 %), trailing-DD kill switch (default 15 %, configurable),
    single-bar-loss kill switch (default −5 %).

  `mlbt.trading.broker.BrokerAdapter` — `PaperBrokerAdapter` (in-process,
    slippage-as-config) or `AlpacaBrokerAdapter` (paper or live via env).

  `mlbt.trading.runner.TradingRunner` — main loop, market-hours check,
    SIGINT/SIGTERM flatten, graceful crash-restart.

### Verified live-step (M4 Max, latest data)

  Built features for 42 mega-caps from the live 1h dataset, ran GBM
  inference in ~17 s, picked top-10 = **EOG, META, TSLA, TGT, MU, AMZN,
  INTC, BAC, CVX, WFC**. Submitted 10 paper-buy orders, opened positions
  at full-equity-weight, slippage cost ~$60 on $100 k. State persisted to
  `data/trading_state/`. Identical code paths will run against Alpaca
  paper or live with the broker flag swapped.

### Inference backend notes (Apple Silicon)

  - **GBM (LightGBM .pkl)** — current production. CPU, ~17 s for 42 symbols
    including data-pull, ~150 ms for inference alone. Fast enough for hourly
    rebalances.
  - **Torch on MPS** — `mlbt.ml.train_xl` trains PatchTST-XL / Transformer-XL
    / LSTM-XL with multi-task heads (xs1/xs2/xs4/xs8 jointly) on MPS, capped
    at `--max-features` to fit in MPS memory. Did not find significantly
    more signal than LightGBM on this 49-cap universe — the cross-sectional
    rank target is a weak learner for sequence models at 1h granularity.
  - **CoreML on the Apple Neural Engine** — `mlbt.ml.coreml_export` is wired
    up (multi-task state_dict support added) but only useful once a torch
    model with a real signal is found. Would give sub-ms latency per symbol.

---

## Production strategy

**Strategy:** `xs8_LongOnlyTop10` — long the 10 cross-sectionally-best-ranked
mega-cap stocks at each 1-hour bar, score from a LightGBM model trained on the
y_xsec_top_8 (8-hour-ahead residualised-return rank) target. No shorts.

**Universe:** 42 of 49 S&P 500 mega-caps (config/universe.yaml) × 1-hour bars ×
2024-05 → 2026-05. 1 415 trades/year (well over the 200/yr deployability gate).

**Reproduced 2026-05-18 on Mac Studio M4 Max** after fixing the
`xsmom_{sym}_{w}` feature-explosion bug (commit 4c62cf8) that previously made
the full-SP500 dataset build run out of RAM during the parquet write. See
[the dataset patch](src/mlbt/pipeline/dataset.py).

### Performance vs SPY buy-and-hold (same window, net of cost)

| Regime | Net Sharpe | vs SPY | Ann Return | Max DD | Trades/yr |
|---|---:|---:|---:|---:|---:|
| PAPER (2 + 1 bps)               | **2.43** | 2.3× SPY | 12.7 % | **-10.5 %** | ~1 400 |
| RETAIL (5 + 2 bps)              | **1.69** | 1.6× SPY | 8.8 %  | -10.9 %     | ~1 400 |
| SPY benchmark                   | 1.05     | —        | 16.8 % | -19.4 %     | — |

Realism audit (5 progressively pessimistic regimes via
`mlbt.realism_audit`, single-config position sizing rather than
top-K — different number from the table above but a stricter
test):

| HEROIC | PAPER | RETAIL | SMALL_CAP | DEFENSIVE |
|---:|---:|---:|---:|---:|
| 1.93 | 1.57 | 1.21 | **0.51** | 1.61 |

All five regimes are ≥ 0.5 net Sharpe — the hard gate from the
handoff. SMALL_CAP barely makes it; this strategy is **not** for
illiquid names. Equity curve: [docs/equity_curve.png](docs/equity_curve.png).

### How to reproduce

```bash
# 1. Setup
git clone https://github.com/Euryalus-Kay/ML_Based_Trading.git
cd ML_Based_Trading
git checkout claude/stock-trading-ml-data-system-aBleI
pip install -r requirements.txt && pip install -e .
pip install lightgbm scikit-learn torch torchvision torchaudio coremltools matplotlib

# 2. Pull data (1h bars, 49 mega-caps + cross-asset, 2024-05 → 2026-05)
PYTHONPATH=src python -m mlbt.cli collect \
  --start 2024-05-20 --end 2026-05-17 \
  --universe config/universe.yaml \
  --only yf_1h --only fred --only treasury_yields \
  --only calendar_events --only crypto_fear_greed

# 3. Build dataset (~1 min on Mac Studio M4 with the xsmom fix)
PYTHONPATH=src python -m mlbt.cli build-dataset \
  --start 2024-05-20 --end 2026-05-17 --bar 1h \
  --universe config/universe.yaml \
  --out data/dataset_1h_micro.parquet --horizons 1,2,4,8 \
  --only-source yf_1h --only-source fred --only-source treasury_yields \
  --only-source calendar_events --only-source crypto_fear_greed

# 4. Train + backtest + realism-audit in a single call
PYTHONPATH=src python -m mlbt.candidate_eval \
  --dataset data/dataset_1h_micro.parquet --target y_xsec_top_8 \
  --out data/cand_xs8_v1 --top-k 10 --seeds 3

# 5. Save equity curve PNG + summary JSON
PYTHONPATH=src python -m mlbt.winner_pkg \
  data/cand_xs8_v1 data/dataset_1h_micro.parquet
```

Training takes ~5 min on the M4 Max (15 LightGBM seed×fold runs across all
10 cores). The full backtest + audit adds ~30 s.

### Walk-forward across history (the open gate)

The handoff asks for a strategy that beats SPY on Sharpe OR Calmar in 4 of 5
sub-windows spanning ≥ 10 years. The xs10 LONG_ONLY_k10 ML model, trained
once per window on 5y of preceding history, produces this result against SPY
over 2010–2026 (win1 = 2006-2010 has no prior training data and is skipped):

| Window         | ML Sharpe | ML Calmar | ML DD   | SPY Sharpe | SPY DD  | beats? |
|----------------|----------:|----------:|--------:|-----------:|--------:|--------|
| 2010-05 → 2014-05 |   −0.69 |    −0.09  | −17.9 % |    0.82    | −19.4 % | ✗ |
| 2014-05 → 2018-05 | **2.67** |    0.56  | −10.7 % |    0.72    | −14.4 % | ✓ Sharpe |
| 2018-05 → 2022-05 | **2.31** | **0.80** | −10.6 % |    0.63    | −34.1 % | ✓ Sharpe + Calmar |
| 2022-05 → 2026-05 |   −1.26 |    −0.22  | −20.5 % |    0.86    | −19.0 % | ✗ |

**Pure ML beats SPY in 2 of 4 trainable windows on Sharpe** — including very
high Sharpes (2.67, 2.31) in 2014-2018 and 2018-2022. The ML signal is real
in those bull / COVID-recovery regimes. **It fails in 2010-2014** (post-GFC
Greek crisis chop) **and 2022-2026** (which mixes the 2022 bear, the 2023 AI
rally, and the 2024 bull — the ML model trained on 2017-2022 doesn't
generalise to that regime mix). 2 of 4 = 50 %, below the 80 % gate.

A stacked overlay `0.2 × ML + 0.8 × vol_target_SPY` softens the bad windows
at the cost of compressing the good ones — Sharpe 0.72, 0.74, 0.81, 0.69
respectively. 2/4 stk_beats_sharpe, 1/4 stk_beats_calmar, **4/4 DD better
than SPY**. So we **do meet** "Max drawdown strictly less than SPY's in every
window" — but **still fail** the 4-of-5 Sharpe/Calmar gate (2/4 ≈ 50 %).

Comparison strategies on the same windows:

- **`vol_target` SPY** (classical, no ML) 2005–2026 5-window walk: beats SPY
  on Calmar in 3 of 5 windows, Sharpe in 2 of 5, DD better in 5/5. Robust on
  2005-2009 (crisis), 2017-2022 (COVID), 2022-2026; loses in pure-bull
  2009-2013 and 2013-2017.
- **`xs5 LONG_ONLY_k10` daily ML** (5-day horizon): 0/4 Sharpe, 0/4 Calmar.
  The shorter horizon is too noisy on daily bars.

**Stacked w_ml sweep**: `python -m mlbt.walk_stack_sweep --walk-dir
data/wt_daily_xs10 --target y_xsec_top_10 --top-k 10` runs against the
already-trained models in ~5 sec per (window × weight) pair.

### Why this is hard to close

The walk-forward gap is structural at this scale:

1. **Universe-size dependence**: the 1h winner runs against 42 mega-caps in
   the 2024-26 window where every stock has dense data. The 2010-2014 daily
   window has only ~32 active names (PLTR, COIN, RIVN, ABBV etc. came later)
   — the cross-sectional ranking has less to choose from, and the model
   over-fits the dominant names.
2. **Horizon × bar mismatch**: the 1h xs8 winner spans 8 hours ≈ 1 trading
   day. The daily xs10 model spans 10 days ≈ 2 weeks. Different alpha
   sources.
3. **Yahoo Finance limits 1h to ~730 days** — so a true 1h walk-forward
   spanning 10 + years isn't possible without a paid intraday vendor.

The full SP500 (~503 names) 1h dataset build is now feasible after the
xsmom feature-explosion fix (commit 4c62cf8) — but on the M4 it still pegs
~14 GB and 12 + min, so it was set aside in favour of validating the proven
49-cap winner. A future iteration could re-train xs8 on the full universe;
the richer cross-section should boost signal-to-noise for short-horizon
rank prediction.

### Known weaknesses

- **In-sample window is just 2024–2026 (bull)**. The walk-forward results
  above are honest: the daily-bar ML model trained on 5y rolling history
  does not reproduce the 1h winner's edge. Either the alpha is genuinely
  short-horizon (1h-specific microstructure) or the daily-bar feature
  engineering is too noisy. **Do not trade this without paper-running
  over the next 6-12 months across at least one drawdown.**
- **Universe is 42 mega-caps** (universe.yaml). Some intended SP500 names
  weren't in the yf_1h pull and are missing; this likely reduces the
  cross-sectional ranking edge slightly.
- **No options / overnight-Asia features**. The handoff lists CBOE option-
  chain IV-skew and Stooq Asia/Europe overnight closes as the next leverage
  points. Neither is implemented yet — both should be tried in iteration 3.
- **DD criterion not strictly met in every walk-forward window** (fails the
  worst-case 2022-2026 ML drawdown). Vol_target stacking is needed to
  pass; tune w_ml per window or use a regime-switch.

### Baseline (no-ML reference)

`vol_target` SPY rescales position so realised vol stays at 12 % annualised
(capped 1.5×). Over 2010–2026:

| | vol_target | SPY buy-hold |
|---|---:|---:|
| Sharpe | **0.63** | 0.54 |
| Calmar | **0.25** | 0.18 |
| Max DD | **-32.0 %** | -56.5 % |

Reproducible: `PYTHONPATH=src python -m mlbt.cli classical`.

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

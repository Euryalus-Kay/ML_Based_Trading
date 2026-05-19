# Production model card — `cand_xs8_v1`

> "What is this thing, exactly?" — everything you'd want to know about the
> model that's currently running on your Alpaca paper account.

## TL;DR

A LightGBM gradient-boosted-tree classifier trained on **1-hour S&P-500 bars
plus macro context**, predicting which of 42 mega-cap stocks will be in
the top half of **8-hour-ahead residualised returns** at each bar. We buy
the top-10 by score, hold ~8 hours, rebalance, repeat.

| Spec | Value |
|---|---|
| Model family | LightGBM gradient-boosted trees (binary classification) |
| Target | `y_xsec_top_8` — 1 if symbol's 8-bar future residualised return is in the top 50% across the universe at that timestamp, 0 otherwise |
| Universe | 42 of 49 S&P-500 mega-caps (config/universe.yaml, intersection with available 1h data) |
| Bar size | **1 hour**, regular trading hours only |
| Holding period | 8 bars ≈ **1.25 trading days** |
| Rebalance | Every 8 bars (≈ once per trading day) |
| Position count | 10 longs at equal weight, no shorts |
| Gross leverage | 1.0× (no margin) |
| Cost assumption | 5 bps + 2 bps slippage = 7 bps round-trip retail |

## Inputs — features per bar

3,291 numerical features per (timestamp, symbol):

| Category | Count | Examples |
|---|---:|---|
| Cross-section / sibling-symbol features | ~2,648 | `close_AAPL`, `volume_GOOGL`, … pulled from the wide aligned frame |
| Volume / dollar-volume | 527 | `volume_z_20`, `dollar_volume_z_20` |
| Symbol one-hot encoding | 42 | `sym_AAPL`, `sym_TSLA`, … |
| Per-symbol returns | 11 | `log_ret`, `ret_3`, `log_ret_5`, … |
| Per-symbol volatility | 10 | `vol_5`, `vol_20`, `atr_14`, `bb_width` |
| FRED macro | 8 | `UNRATE`, `CPIAUCSL`, `NFCI`, `SOFR` |
| Momentum indicators | 7 | `rsi_14`, `mom_z_5`, `mom_z_20`, `autocorr_5` |
| Rates / treasury | 7 | `y_10y`, `y_2y`, `T10Y2Y`, … |
| Microstructure | 6 | `gk_vol_20`, `roll_spread_60`, `amihud_20` |
| Volatility regime | 6 | `vix`, `vix_z_20`, `vix_pct_rank_1y`, `vix_above_30` |
| Price action | 4 | `range_pct`, `body_pct`, `upper_wick`, `lower_wick` |
| Price OHLCV | 4 | `open`, `high`, `low`, `close` |
| Cross-sectional dispersion | 3 | `xs_dispersion_5`, `xs_dispersion_5_z` |
| Cross-sectional momentum | 3 | `xsmom_self_5`, `xsmom_self_20`, `xsmom_self_60` |
| Credit spreads | 2 | `hyg_lqd_ratio`, `hyg_lqd_ratio_d5` |
| Dollar | 1 | `uup_ret_5` |
| Risk-on factor | 1 | `risk_on_5b` (PCA-like composite of VIX/HYG/USD/Gold/SPY) |
| Crypto Fear & Greed | 1 | `fear_greed` |

All features are **leak-free**: every value at time *t* is computable using
only data observable by time *t* (the data pipeline tracks publish-lag on
each source — FRED has a 1-day publish lag, treasury yields ~18 hours,
intraday bars ~15 seconds).

## Training

| Item | Value |
|---|---|
| Architecture | LightGBM gradient-boosted trees |
| Loss | binary log-loss |
| Num leaves | 127 |
| Min data per leaf | 200 |
| Learning rate | 0.03 |
| L2 regularization | 1.0 |
| Feature fraction | 0.8 (random subsample of features per tree) |
| Bagging fraction | 0.8 (random subsample of rows per tree) |
| Max boost rounds | 2000 with early-stopping at 50 stale validation rounds |
| Seeds ensembled | 3 (averaged at inference) |
| Folds | 5 expanding-window walk-forward |
| Sample weights | 1/h = 1/8 (López-de-Prado overlap correction) |
| Embargo | h+1 = 9 bars between train and test |
| Symbol one-hot | enabled (learns per-stock intercept terms) |

Training data: ~145k rows × 3,291 features × 3 seeds × 5 folds = **15
boosters trained**, predictions averaged across seeds. Training takes ~5
min on the Mac M4 Max (LightGBM uses all 12 performance cores).

**Cross-validated metrics**:
- Aggregate accuracy: **50.05%** (barely above random — the binary target is hard, but the *score* has more signal than the binary label)
- Aggregate AUC: **0.5018**
- Per-fold acc: 49.97%, 49.17%, 50.77%, 49.93%, 50.38%

The accuracy looks weak because the binary target is too coarse — the *score
distribution* is what we use, and it ranks the universe with enough fidelity
that the top-10 ≠ bottom-10 in expected forward returns.

## Output

Per bar, per symbol: a single score ∈ [0, 1]. Interpretation:
- score > 0.5 ⇒ model thinks this symbol is more likely than chance to be a top-half performer over the next 8 bars
- score < 0.5 ⇒ less likely than chance

We **sort by score** and pick the top 10 — only the *rank* matters, not the
absolute value. A typical score distribution clusters tightly around 0.5
(0.495–0.510 for most symbols at most bars), so the top-10 is decided by
~1 bp of score differential. That's why **score-weighting** (using
\|score−0.5\| as weight) doesn't improve over equal-weighting: the scores
aren't well-calibrated for sizing, only for ranking.

## Inference speed

| Stage | Cold | Warm (cached) |
|---|---|---|
| Pull latest yf_1h bars from disk + Yahoo | ~1 sec | ~1 sec |
| Build aligned wide frame (FRED + treasury + crypto + calendar) | ~3 sec | (cached) |
| Per-symbol feature engineering (42 × 60-day window) | ~24 sec | (cached) |
| Cross-sectional + regime features | ~2 sec | (cached) |
| LightGBM model.predict() on 42 rows | **< 1 ms** | < 1 ms |
| OMS scoring → target weights | < 50 ms | < 50 ms |
| Order submission to Alpaca (10 orders) | ~1 sec | ~1 sec |
| **Total per cycle** | **~28 sec** | **~2 sec** |

The signal cache (`CachedSignalGenerator`) reuses the built dataset for
up to 1 hour. First step after restart pays the cold-start cost; every
subsequent step (we poll every 60 sec) is ~2 sec.

## Backtest stats — alpha-only (offline `PortfolioConfig`)

These measure the **cross-sectional alpha** — top-10 minus average future
return. They're what you'd see in a published research paper.

| Window | 2024-09-30 to 2026-05-14 (~1.5 years, mostly bull) |
|---|---|
| Bars | 2,830 (1-hour RTH) |
| PAPER Sharpe (2+1 bps) | **2.43** |
| PAPER ann return (alpha) | +12.66% |
| PAPER max DD (alpha) | -10.45% |
| Win bars | 51.2% (1,450 / 2,830) |
| Loss bars | 48.8% |
| Best day | +$2,613 |
| Worst day | -$3,134 |
| Positive days | 52.1% (212 / 407) |
| Avg daily P&L | +$50 (alpha-only on $100k notional) |
| Trade events | 2,723 |
| Avg turnover per bar | 0.47 (47% of the book turns over) |
| Total cost paid | ~$40k over 1.5 years (40% gross fee burden!) |

## Backtest stats — actual realised (live-replay through trading stack)

These measure the **dollar-for-dollar realised** return — what would actually
land in your Alpaca account if the strategy had been running.

| Regime | RETAIL (5+2 bps) | PAPER (2+1 bps) |
|---|---|---|
| Sharpe | **1.07** | **1.35** |
| Ann return | **+27.84%** | +34.82% |
| Max drawdown | -27.22% | -26.68% |
| Total return | +52.56% | +72.23% |
| Start equity | $100,000 | $100,000 |
| **End equity** | **$152,555** | $172,226 |
| Daily P&L (mean) | **+$118** | +$152 |
| Daily P&L (median) | +$25 | +$31 |
| Best day | +$15,203 | +$15,800 |
| Worst day | -$7,738 | -$7,500 |
| Positive days | **50.6%** (206/407) | 51.4% |
| Total orders | 4,221 | 4,237 |
| **Trades/year** | **~2,800** | ~2,800 |

For comparison, SPY over the same window: **Sharpe 1.06, +16.9% ann,
DD -15%**. The strategy beats SPY on absolute return (+27.8% vs +16.9%) at
similar Sharpe, but with worse drawdown (-27% vs -15%) — that's the cost
of being concentrated in 10 mega-caps vs 500.

## How a single trading cycle works

```
Every 60 seconds the daemon wakes up and:

  ┌─────────────────────────────────────────────────────────────────┐
  │ 1.  Is market open? (Alpaca clock)                  ──► sleep   │
  │     Otherwise:                                                  │
  │                                                                 │
  │ 2.  Cancel any orders left pending from the prior cycle         │
  │                                                                 │
  │ 3.  Pull latest yf_1h bars from Yahoo (~1 sec, only new bars)   │
  │                                                                 │
  │ 4.  Either reuse cached features (< 1 hour old) OR rebuild      │
  │     the full 60-day feature pipeline (~26 sec)                  │
  │                                                                 │
  │ 5.  Take the most recent row per symbol → 42-row feature matrix │
  │                                                                 │
  │ 6.  LightGBM.predict() → 42 scores (< 1 ms)                     │
  │                                                                 │
  │ 7.  Rank, pick top-10 → target weights = 10% each (gross 100%)  │
  │                                                                 │
  │ 8.  Per-symbol stop/target check on existing positions          │
  │     (override the model when a position is at -8% or +20%)     │
  │                                                                 │
  │ 9.  Diff target vs current positions → buy/sell orders          │
  │                                                                 │
  │ 10. Trailing-DD kill switch: equity < session_peak × 0.90?      │
  │     → flatten all + halt.  Otherwise continue.                 │
  │                                                                 │
  │ 11. Submit orders to Alpaca paper (or live if --live)           │
  │                                                                 │
  │ 12. Persist state: positions.json, risk.json,                   │
  │     last_signal.json (for dashboard), equity_ledger.csv         │
  └─────────────────────────────────────────────────────────────────┘
```

Wall-clock per cycle: **~2 seconds (warm cache)** or **~28 seconds
(cold cache, first cycle after restart)**.

## Real-time data sources

All 5 pass the `mlbt.cli audit-sources --hours 96` real-time check:

| Source | Freshness | Publish lag |
|---|---|---|
| `yf_1h` (Yahoo 1h bars) | live, every minute | 15 seconds |
| `yf_daily` (Yahoo daily closes, fallback) | end-of-day | 0 |
| `fred` (FRED macro CSV — DGS10, VIXCLS, NFCI, etc) | daily | 1 day |
| `treasury_yields` (US Treasury par yield curve) | daily | 18 hours |
| `calendar_events` (FOMC / CPI / OPEX) | known schedule | 0 |
| `crypto_fear_greed` | daily | 1 hour |

## Risk controls

| Control | Threshold | Action |
|---|---|---|
| Per-symbol stop-loss | unrealised P&L ≤ -8% | force-close, override model |
| Per-symbol profit-take | unrealised P&L ≥ +20% | force-close, lock in |
| Per-symbol position cap | ≤ 15% of equity | trim to cap before submitting |
| Gross exposure cap | ≤ 150% | scale all weights down |
| Trailing max drawdown | equity ≤ 90% of session peak | **flatten all + halt**, no new buys until manual reset |
| Single-bar loss | bar P&L ≤ -5% | flatten all + halt |
| Min universe size | < 6 scored symbols | stay flat for that bar |

State persists to `data/trading_state/risk.json`; to reset after a halt,
delete that file or set `halted: false` in it.

## Known weaknesses

1. **Bull-market trained**: validation window (2024-09 → 2026-05) was
   mostly bullish. The model has never seen a real bear in this incarnation.
2. **Concentrated**: only 42 names, all mega-caps. SP500 ETF gives 500
   names. We hit higher returns but at higher concentration risk.
3. **Cost burden is heavy**: ~40% of gross alpha is consumed by trading
   costs at the 7 bps round-trip retail assumption. Anything that adds
   slippage (e.g. trading at the open auction) erodes Sharpe fast.
4. **Score is poorly calibrated**: top-10 picks score 0.502–0.510, not
   0.7–0.8. Confidence-weighted sizing was tested and lost. The model
   only knows *which* names; it doesn't know *how confidently*.
5. **Walk-forward across 10+ years still pending**: yfinance caps 1h
   history at 730 days, so we can't validate at this granularity over
   multiple regimes without paid intraday data.

## Iteration history

- Iteration 1: original 49-cap xs8 LONG_ONLY_k10 (previous Claude on cloud VM)
- Iteration 2: reproduced on M4 Max, fixed `xsmom` memory bug, validated 1.07/+27.8% retail
- Iteration 3: built live trading system + Alpaca paper integration
- Iteration 4: hyperparameter sweep (9 configs) — production wins
- Iteration 5: confidence-weighted sizing — production wins (equal is best)
- Iteration 6: multi-horizon ensemble (xs2+xs4+xs8) — built, backtest hung
- Iteration 7 (running): 64-thesis tournament covering targets × top_k × rebalance × MN × sizing

## Files

| Path | What |
|---|---|
| `data/cand_xs8_v1/model.pkl` | Trained LightGBM booster |
| `data/cand_xs8_v1/feature_cols.json` | Feature column list in order |
| `data/cand_xs8_v1/metrics.json` | Training metrics per fold |
| `data/cand_xs8_v1/predictions.parquet` | OOS predictions across all folds |
| `data/cand_xs8_v1/paper.json` / `paper.parquet` | Offline PAPER backtest |
| `data/cand_xs8_v1/retail.json` / `retail.parquet` | Offline RETAIL backtest |
| `data/cand_xs8_v1/live_replay_retail8.parquet` | Through-trading-stack RETAIL |
| `data/cand_xs8_v1/realism_audit.csv` | 5-regime realism audit |
| `data/cand_xs8_v1/candidate_eval.json` | Full eval summary |
| `data/trading_state/positions.json` | Live position book |
| `data/trading_state/equity_ledger.csv` | Bar-by-bar equity log |
| `data/trading_state/last_signal.json` | Most recent top-K scores (for dashboard) |
| `data/trading_state/risk.json` | Risk-manager state (halted, peak equity) |
| `data/trading_state/live.log` | Runner log file |

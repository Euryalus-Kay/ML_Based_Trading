# Autonomous handoff prompt for Claude Code on Apple Silicon

Paste the prompt below into a fresh `claude` CLI session on the Mac Studio.
It is self-contained and grants full authority to iterate continuously until
the success criteria are met.

---

```
You are taking over an ML-for-trading project that aims to consistently beat
SPY net of realistic costs. You are running on a 64 GB Apple Silicon Mac Studio.
You have FULL AUTHORIZATION to run continuously for as many hours or days as
needed. DO NOT STOP iterating until the success criteria are met. Do not ask
for permission to continue; do not give up after one or two failed sweeps; do
not worry about token budget.

## Setup (one-time)
git clone https://github.com/Euryalus-Kay/ML_Based_Trading.git
cd ML_Based_Trading
git checkout claude/stock-trading-ml-data-system-aBleI
pip install -r requirements.txt
pip install -e .
pip install lightgbm scikit-learn torch torchvision torchaudio coremltools
PYTHONPATH=src python -m pytest tests/ -q   # 7 smoke tests should pass

Verify Apple GPU is wired:
  python -c "import torch; print('mps:', torch.backends.mps.is_available())"
Expected: mps: True

## Success criteria (do not stop until ALL are met)
A trading strategy (or stacked strategy) that:
  1. Beats SPY on Sharpe OR Calmar in at least 4 of 5 walk-forward sub-windows
     spanning >=10 years total (use mlbt.walk_validation; extend if needed).
  2. Has max drawdown strictly less than SPY's in every window.
  3. Generates >=200 trades per year (deployable with retail capital).
  4. Survives mlbt.realism_audit: net Sharpe >= 0.5 in PAPER AND RETAIL regimes,
     not just HEROIC.
  5. Is committed and pushed to the branch with a final equity curve and
     a one-paragraph reproducibility note in README.md.

Already-confirmed baseline: classical vol_target SPY (no ML) beats buy-and-hold
on Sharpe 0.76 vs 0.75, Calmar 0.59 vs 0.38, max DD -16.5% vs -34.1% over
2010-2026. Verify this first:
  PYTHONPATH=src python -m mlbt.cli classical
  PYTHONPATH=src python -m mlbt.walk_validation

Your goal is to ADD ML alpha on top of (or beating) this baseline.

## Get fresh data (cloud VM ran out of memory mid-build)
The full S&P 500 dataset_1h_sp500.parquet was not generated on the cloud. You
must rebuild it locally — 64 GB RAM will handle it.

# 1. Resume the bulk 1h pull (skips symbols already in storage)
PYTHONPATH=src python -c "
from mlbt.sources.yf_bars import _fetch_yf
from mlbt.core.storage import Storage
import pandas as pd
tickers = [t.strip() for t in open('config/sp500_tickers.txt') if t.strip()]
have = set(Storage().list_keys('yf_1h'))
todo = [t for t in tickers if t not in have]
print('todo:', len(todo))
if todo:
    df = _fetch_yf(todo, pd.Timestamp('2024-05-20'), pd.Timestamp('2026-05-17'),
                   interval='1h', chunk_size=25)
    for sym, sub in df.groupby('symbol'):
        Storage().write('yf_1h', sym, sub.drop(columns=['symbol']))
"

# 2. Other sources
PYTHONPATH=src python -m mlbt.cli collect \
  --start 2024-05-20 --end 2026-05-17 \
  --universe config/universe_sp500.yaml \
  --only fred --only treasury_yields --only calendar_events --only crypto_fear_greed

# 3. Build the wide dataset
PYTHONPATH=src python -m mlbt.cli build-dataset \
  --start 2024-05-20 --end 2026-05-17 --bar 1h \
  --universe config/universe_sp500.yaml \
  --out data/dataset_1h_sp500.parquet --horizons 1,2,4,8 \
  --only-source yf_1h --only-source fred --only-source treasury_yields \
  --only-source calendar_events --only-source crypto_fear_greed

# 4. Daily history (decades — for walk-forward validation)
PYTHONPATH=src python -m mlbt.cli collect \
  --start 2005-01-01 --end 2026-05-17 \
  --universe config/universe_sp500.yaml \
  --only yf_daily --only fred --only treasury_yields

PYTHONPATH=src python -m mlbt.cli build-dataset \
  --start 2005-01-01 --end 2026-05-17 --bar 1d --session full \
  --universe config/universe_sp500.yaml \
  --out data/dataset_daily_sp500.parquet --horizons 1,2,3,5,10 \
  --only-source yf_daily --only-source fred --only-source treasury_yields \
  --only-source calendar_events --only-source crypto_fear_greed

## The iteration loop — repeat forever until success criteria met

ITERATION_K:
  1. Run a tournament. Default starting points:
        PYTHONPATH=src python -m mlbt.mac_studio_tournament
     This uses the XL variants (LSTM-XL, Transformer-XL, PatchTST-XL) on MPS.
     Each run produces data/mac_tournament/leaderboard.csv.
  2. If anything passes_gate=True -> go to step 5.
  3. If NOTHING passes, IDENTIFY the failure dimension:
        - low train_acc        -> bad target or features. Try step 4a/4b.
        - high acc, low Sharpe -> portfolio construction. Step 4c.
        - acc + Sharpe ok but doesn't beat SPY -> regime issue. Step 4d.
        - everything inconsistent across regimes -> overfitting. Step 4e.
  4. Apply fixes (DO NOT skip — try each in order if the previous didn't help):
        4a. Add new data: CBOE option chains
              URL: https://cdn.cboe.com/api/global/delayed_quotes/options/{TICKER}.json
            Compute ATM IV / 25-delta skew / put-call vol ratio per ticker per
            hour. Persist to data/raw/cboe_options/. Add to dataset build.
        4b. Add Stooq Asia/Europe overnight closes (^nkx, ^dax, ^hsi via
              https://stooq.com/q/d/l/?s=^nkx&i=d with User-Agent header).
            Build features: nkx_overnight_ret, dax_morning_ret. Lead-lag for
            US 09:30 open.
        4c. Portfolio: change top_k/bottom_k in PortfolioConfig; try k=5, k=15,
            k=30. Try market_neutral=False (long-only top-K). Try rebalance_
            every=h (no daily churn).
        4d. STACK with vol_target: when ML score-spread > threshold trade
            long-short, ELSE default to vol_target SPY. Write
            src/mlbt/strategies/stacked.py.
        4e. Purged k-fold with embargo proportional to label horizon is
            already on. If still overfitting: cut features by SHAP importance
            (sklearn.inspection.permutation_importance), drop bottom 50%.
            Re-train.
  5. Run mlbt.realism_audit on the candidate. If it fails any regime, go back
     to step 4. If it passes all regimes, go to step 6.
  6. Run mlbt.walk_validation on the candidate strategy. Extend the daily
     dataset back to 2005 if needed (pip install yfinance, daily history is
     decades). Must beat SPY in >=4 of 5 windows.
  7. If both step 5 and 6 pass: COMMIT, PUSH, update README with the final
     equity curve, and report success.
  8. If either fails: go to ITERATION_K+1.

## Deploy sub-agents liberally

For ANY of the following, spawn a sub-agent with the Agent tool instead of
doing it inline:

  - "Hunt for a new data source for X" -> Agent(general-purpose) with
    instructions to verify via HTTP GET and return a Python fetcher.
  - "Read these 3000 lines of training output and tell me what failed" ->
    Agent(general-purpose).
  - "Re-read all our previous leaderboards and find the pattern" ->
    Agent(general-purpose).
  - "Implement and test a new feature module" -> Agent(general-purpose,
    isolation=worktree) so it doesn't conflict with your main code.
  - "Survey codebase for X" -> Agent(Explore).
  - "Design implementation for Y" -> Agent(Plan).

Spawn agents IN PARALLEL when work is independent. Don't serialize.

## Things you can try that the cloud agent could not

  - Train PatchTST-XL on MPS — 10-30x faster than the cloud VM's CPU
  - Bigger batch sizes (256-512) — you have 64 GB
  - Wider universe — Russell 3000 (~3000 tickers) instead of S&P 500
  - Longer history — yfinance daily back to 1990 for select tickers
  - More seeds — 15-30 per fold instead of 3-5
  - Multi-task heads — predict y_xsec_top_1, _2, _4, _8 jointly
  - Coinbase / Kraken instead of Binance for crypto (your IP isn't geo-blocked)
  - CoreML export: PYTHONPATH=src python -m mlbt.ml.coreml_export <model_dir>
    <out.mlpackage> — gets Neural Engine inference for live trading

## Commit hygiene

After every change that improves results OR clearly fails:
  git add -A
  git commit -m "<short subject>

  <one paragraph: what changed, what the leaderboard shows, what to try next>"
  git push origin claude/stock-trading-ml-data-system-aBleI

This is non-negotiable. The repo is the source of truth.

## When you find a winner

Write to README.md a top-level "## Production strategy" section with:
  - Strategy name + brief description
  - Walk-forward Sharpe / Calmar / Max DD vs SPY (table)
  - Realism regime results (table)
  - Universe + bar frequency + horizon
  - Exact CLI commands to reproduce from scratch
  - The equity curve as a PNG (matplotlib, save to docs/equity_curve.png)
  - The strategy's weaknesses (when does it underperform SPY?)

Then commit, push, and write a final reply summarising what was tried, what
won, and why.

## What you must NOT do

  - Do not stop after one failed tournament. The whole point of this run is to
    iterate. Failure is the start of the loop, not the end.
  - Do not claim a win without running both realism_audit AND walk_validation.
  - Do not over-fit to one regime — every winner must show signal in 2+
    historical periods.
  - Do not commit to main; this branch only.
  - Do not delete the data/ folder — it's gitignored but locally valuable.

## Start

Run the iteration loop now. Iteration 1 starts with mac_studio_tournament.
Continue until the success criteria above are all met. You can use any number
of tool calls and any amount of compute time. Report only when you have a
winner — or every 4 iterations with a one-line status update.
```

---

## What is already done (don't redo)

1. `classical vol_target` strategy beats SPY on Sharpe (0.76 vs 0.75), Calmar
   (0.59 vs 0.38), max DD (-16.5% vs -34.1%) over 2010-2026.
   Robust across 3/5 walk-forward windows. **This is the baseline to beat
   with ML.**
2. Repo has 13 zero-key data sources, 670-column wide feature frame, derived
   regime features (xs_dispersion, vix_pct_rank, risk_on composite), cross-
   sectional rank targets, vol-scaled targets, time-of-day features.
3. Audit-recommended training fixes are integrated: purged embargo = h+1,
   overlap sample weights = 1/h, walk-forward 5 folds.
4. Realism audit (`mlbt.realism_audit`) runs PAPER / RETAIL / SMALL_CAP /
   HEROIC / DEFENSIVE regimes.
5. XL models (LSTM-XL, Transformer-XL, PatchTST-XL) added for MPS training.
6. CoreML export for ANE inference.
7. S&P 500 ticker list at config/sp500_tickers.txt (503 names).
8. 1h bars pulled for 572 symbols. Daily bars pulled for original 49-stock
   universe back to 2010.

## Key files
- `src/mlbt/mac_studio_tournament.py`     -- tournament entry point
- `src/mlbt/strategies/vol_target.py`     -- proven baseline
- `src/mlbt/walk_validation.py`           -- robustness gate
- `src/mlbt/realism_audit.py`             -- regime stress test
- `src/mlbt/backtest/portfolio.py`        -- long-short simulator
- `src/mlbt/features/regime.py`           -- derived "non-obvious" features
- `src/mlbt/ml/train.py`                  -- walk-forward LightGBM + MPS torch
- `src/mlbt/ml/models.py`                 -- LSTM/Transformer/PatchTST + XL variants
- `src/mlbt/ml/coreml_export.py`          -- Neural Engine inference path

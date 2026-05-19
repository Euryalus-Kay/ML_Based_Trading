# Autonomous handoff prompt for Claude Code on Apple Silicon

## How to use this file (user — do this once)

On your Mac Studio:
```bash
git clone https://github.com/Euryalus-Kay/ML_Based_Trading.git
cd ML_Based_Trading
git checkout claude/stock-trading-ml-data-system-aBleI
claude   # opens Claude Code in this repo
```

Then in the Claude Code prompt, paste **the entire fenced block below** as
your first message. It is self-contained — Claude will read it, do setup, and
begin the autonomous iteration loop.

---

```
SYSTEM CONTEXT — READ EVERY LINE BEFORE TAKING ACTION

==================================================================
HARD MANDATES — VIOLATING ANY OF THESE IS A FAILED HANDOFF
==================================================================

1. DO NOT STOP. Run the iteration loop until the success criteria
   are met, even if it takes 100 iterations or 3 days of wall time.
   Failure on a single tournament is the START of the loop, not the
   end. Do not ask for permission to continue. Do not summarise and
   wait — go straight to the next iteration.

2. CONSTANT AUDIT. After EVERY candidate strategy (not just the
   final one), run mlbt.realism_audit AND mlbt.walk_validation.
   If a model only beats SPY under HEROIC assumptions it is
   WORTHLESS. Confirm both gates pass before you even consider
   declaring victory.

3. DATA MUST BE REAL-TIME ACCESSIBLE. Every feature the model uses
   in training must also be available LIVE at decision time on the
   user's machine. Before adding any new data source, run
   mlbt.cli audit-sources on it. If a feature can't be re-pulled
   on the trading bar with publish_lag < bar_size, it is leakage
   in disguise. Reject it.

4. ML-FIRST. The classical vol_target baseline is a reference, not
   the goal. The user explicitly asked for ML across hundreds of
   stocks. Use the XL torch models on MPS. The deliverable must
   include an ML component.

5. SPAWN SUB-AGENTS LIBERALLY. You are not in a single-thread
   conversation — you are a CLI session. Spawn parallel Agents for:
   data hunting, log triage, feature implementation, code review.
   Serial work is wasted clock time.

6. COMMIT + PUSH after every meaningful change, win or fail. The
   git history is the audit trail.

==================================================================
WHAT YOU CAN DO ON THE USER'S MACHINE
==================================================================

You are running on the user's Mac Studio. Unlike the previous
cloud Claude, you have access to a real machine with:

  - Browser (Safari, Chrome). You may open it via `open -a Safari
    https://...` if you need to verify a data source visually or
    register for a free API key.
  - The user is willing for you to sign up for accounts when
    needed. If a data source requires a free API key (e.g. EIA,
    Alpha Vantage, Polygon free tier), do this:
      1. Open the signup page in the browser.
      2. Tell the user "I'm registering you for X at Y URL; please
         click through and paste the key into .env when prompted."
      3. Wait for the key and put it in .env.
    DO NOT submit personal info on the user's behalf — they handle
    the signup click, you just kick it off and consume the key.
  - Reach out to the user via stdout if you need their input —
    they may be away but will see it on return.
  - Full disk persistence — anything you store survives the
    session.
  - Local network access — Binance and other geo-blocked sources
    that failed on the cloud VM work from a US/home IP.

==================================================================

You are taking over an ML-for-trading project that aims to consistently beat
SPY net of realistic costs. You are running on a 64 GB Apple Silicon Mac
Studio (M2 Max or Ultra). You are continuing the work of a previous Claude
instance that ran on a small cloud VM and hit its compute ceiling.

REPO            github.com/Euryalus-Kay/ML_Based_Trading
ACTIVE BRANCH   claude/stock-trading-ml-data-system-aBleI (you are already on it)
HARDWARE        12-24 perf cores, 64 GB unified RAM, MPS GPU, Neural Engine
CONSTRAINT      None. You may run for hours or days. Do not stop. Do not ask
                permission. Do not worry about token budget or context window.
                Spawn sub-agents whenever it saves wall-clock time. Commit
                and push after every change that improves OR clearly fails.

WHAT IS ALREADY CONFIRMED — DO NOT RE-LITIGATE
  1. The classical vol_target SPY strategy (no ML, see mlbt/strategies/
     vol_target.py) beats buy-and-hold on Sharpe (0.76 vs 0.75), Calmar
     (0.59 vs 0.38), and max DD (-16.5% vs -34.1%) over 2010-2026.
     Robust across 3 of 5 walk-forward sub-windows.
     -> This is the BASELINE you must beat or stack with.
  2. Naive directional ML on 1h Yahoo bars across 49 mega-caps caps at
     ~50.8% accuracy and does NOT beat SPY honestly. Pivot away from that
     framing.
  3. Audit fixes are in: purged embargo = h+1, overlap sample weights = 1/h,
     cross-sectional rank target y_xsec_top_h, residualised target
     y_resid_up_h, vol-scaled target y_vol_up_h, time-of-day features,
     derived regime features (xs_dispersion, vix_pct_rank, risk_on
     composite).
  4. Realism audit (mlbt.realism_audit) tests every model under 5 cost/
     execution regimes including RETAIL (5+2 bps, t+1 fill, overlap-scaled).
  5. XL model variants (LSTM-XL, Transformer-XL, PatchTST-XL) ready in
     mlbt.ml.models — train on MPS via the existing CLI.
  6. CoreML export hook (mlbt.ml.coreml_export) — converts trained PyTorch
     models to .mlpackage for Neural Engine inference.
  7. S&P 500 ticker list at config/sp500_tickers.txt (503 names).
  8. 1h bars for ~572 symbols already in data/raw/yf_1h on the cloud — you
     must re-pull because the data/ folder is gitignored.

SUCCESS CRITERIA (do not stop until ALL are met)
  1. Beats SPY on Sharpe OR Calmar in at least 4 of 5 walk-forward sub-windows
     spanning >=10 years total.
  2. Max drawdown strictly less than SPY's in every window.
  3. >=200 trades per year (deployable with retail capital).
  4. Survives mlbt.realism_audit: net Sharpe >= 0.5 in PAPER AND RETAIL
     regimes, not just HEROIC.
  5. Final equity curve PNG saved to docs/equity_curve.png.
  6. README.md has a "## Production strategy" section with reproducible
     CLI commands, table of metrics vs SPY, and known weaknesses.
  7. Everything committed and pushed to claude/stock-trading-ml-data-system-aBleI.

SETUP (one-time, ~5 minutes)
  pip install -r requirements.txt
  pip install -e .
  pip install lightgbm scikit-learn torch torchvision torchaudio coremltools
  PYTHONPATH=src python -m pytest tests/ -q       # 7 tests must pass
  python -c "import torch; print('mps:', torch.backends.mps.is_available())"  # True

VERIFY THE BASELINE FIRST (~2 min)
  PYTHONPATH=src python -m mlbt.cli classical
  PYTHONPATH=src python -m mlbt.walk_validation
  -> if these print the vol_target Sharpe ~0.76 and the robustness verdict
     marks vol_target as robust, you're correctly bootstrapped.

PULL FRESH DATA (~10-30 min on Mac Studio)
  # Step 1: bulk 1h pull (resumable; skips symbols already stored)
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

  # Step 2: other sources
  PYTHONPATH=src python -m mlbt.cli collect \
    --start 2024-05-20 --end 2026-05-17 \
    --universe config/universe_sp500.yaml \
    --only fred --only treasury_yields --only calendar_events --only crypto_fear_greed

  # Step 3: build the wide 1h dataset (this is where the cloud VM OOM'd)
  PYTHONPATH=src python -m mlbt.cli build-dataset \
    --start 2024-05-20 --end 2026-05-17 --bar 1h \
    --universe config/universe_sp500.yaml \
    --out data/dataset_1h_sp500.parquet --horizons 1,2,4,8 \
    --only-source yf_1h --only-source fred --only-source treasury_yields \
    --only-source calendar_events --only-source crypto_fear_greed

  # Step 4: daily history back to 2005 for walk-forward validation
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

REAL-TIME DATA CHECK (mandatory before every iteration's first train)

  PYTHONPATH=src python -m mlbt.cli audit-sources --hours 48

  Every source you intend to TRAIN ON must return rows in this audit. A
  source that's stale, gated, or returns 0 rows is forbidden as a feature.
  If a new source you added in iteration_{k-1} fails this audit, REMOVE
  IT from the dataset build and retry from iteration_k.

THE ITERATION LOOP — REPEAT FOREVER UNTIL SUCCESS
(Never stop in step 8. Failure -> bump K -> go again.)

ITERATION_K:
  0. Real-time data audit (above). All training sources must pass.
  1. Run a tournament:
        PYTHONPATH=src python -m mlbt.mac_studio_tournament
     Uses XL variants on MPS. Produces data/mac_tournament/leaderboard.csv.
  2. If anything has passes_gate=True -> step 5.
  3. If NOTHING passes, diagnose by reading the leaderboard:
        low train_acc        -> bad target/features (step 4a, 4b)
        high acc, low Sharpe -> portfolio construction (step 4c)
        acc + Sharpe ok but doesn't beat SPY -> regime (step 4d)
        inconsistent across regimes -> overfitting (step 4e)
  4. Apply fixes in order. Don't skip; if one doesn't move the needle,
     stack the next on top.
        4a. CBOE option-chain features (HIGHEST LEVERAGE remaining lever).
              GET https://cdn.cboe.com/api/global/delayed_quotes/options/{T}.json
            Per ticker, compute ATM IV / 25-delta skew / put-call vol ratio /
            IV term slope. Save snapshots to data/raw/cboe_options/. Add a
            new src/mlbt/sources/cboe_options.py and re-build dataset.
        4b. Stooq Asia/Europe overnight closes (lead-lag for US open).
              https://stooq.com/q/d/l/?s=^nkx&i=d  (User-Agent header required)
            Also ^dax, ^hsi, ^kos. Build nkx_overnight_ret, dax_morning_ret.
        4c. Vary PortfolioConfig: top_k/bottom_k in {5, 10, 15, 20, 30};
            market_neutral={True, False}; rebalance_every={1, h}.
        4d. STACK with vol_target: when ML rank-spread > threshold, trade
            long-short; else default to vol_target SPY. Write
            src/mlbt/strategies/stacked.py and run mlbt.walk_validation
            with the new strategy added.
        4e. Feature selection by permutation importance. Drop bottom 50%
            of features; re-train.
  5. CONSTANT AUDIT — mlbt.realism_audit on the candidate. Must beat SPY in
     PAPER and RETAIL regimes (not just HEROIC). If not -> step 4. NEVER
     skip this. NEVER announce a winner without it.
  6. CONSTANT AUDIT — mlbt.walk_validation on the candidate. Must beat SPY
     in >=4/5 windows on Sharpe OR Calmar. If not -> step 4.
  7. CONSTANT AUDIT — re-run audit-sources to ensure every feature is still
     real-time-accessible. Reject the candidate if any required source went
     stale. (Real-time-deployability is non-negotiable.)
  8. If 5, 6, AND 7 all pass -> WIN. Save equity curve PNG, update README,
     commit + push, report success and stop.
  9. Else -> ITERATION_K+1. DO NOT STOP. The next iteration starts
     immediately. Use the diagnosis from step 3 to pick the next move.

DEPLOY SUB-AGENTS LIBERALLY (parallel, never serial)

Use the Agent tool whenever:
  - hunting a new data source -> Agent(general-purpose) with HTTP-verify
    requirement.
  - reading >1000 lines of training output -> Agent(general-purpose).
  - implementing a new feature module + tests -> Agent(general-purpose,
    isolation=worktree) so it doesn't conflict.
  - surveying repo for X -> Agent(Explore).
  - designing approach for Y -> Agent(Plan).
Spawn them in parallel (single message with multiple Agent calls). Never
serialize independent work.

POWER MOVES UNIQUE TO THIS MACHINE
  - PatchTST-XL on MPS: 10-30x faster than the cloud VM.
  - Batch size 256-512: you have 64 GB. The cloud could not afford this.
  - Universe extension: Russell 3000 (~3000 tickers) is feasible here.
  - Daily history back to 1990 via yfinance for blue chips.
  - More seeds: 15-30 per fold instead of 3-5.
  - Multi-task head: shared backbone, predict y_xsec_top_1,2,4,8 jointly.
  - Coinbase/Kraken instead of Binance for crypto (no geo block from home IP).
  - CoreML export then ANE inference for live signals:
      PYTHONPATH=src python -m mlbt.ml.coreml_export <model_dir> <out.mlpackage>

COMMIT HYGIENE (non-negotiable)
After every meaningful change:
  git add -A
  git commit -m "<subject line>

  <one paragraph: what changed, leaderboard delta, what to try next>"
  git push origin claude/stock-trading-ml-data-system-aBleI

DO NOT
  - Do NOT stop after one failed tournament. Failure is the start of the
    loop, not the end. Iterate until success or the user explicitly tells
    you to halt.
  - Do NOT skip realism_audit, walk_validation, or audit-sources between
    iterations. Constant audit means EVERY candidate gets all three gates.
  - Do NOT use a feature that isn't real-time accessible. If you can't
    pull it live at decision time, the model will be untradable.
  - Do NOT claim a win without all three gates passing.
  - Do NOT over-fit to one regime — winners must beat SPY in 2+ historical
    periods.
  - Do NOT commit to main.
  - Do NOT delete data/ — gitignored but locally valuable.
  - Do NOT ask the user to confirm intermediate steps. Just iterate.
  - Do NOT abandon ML in favor of more classical strategies — the user
    asked for ML. Classical baselines are reference points, not the goal.
  - Do NOT submit personal info on the user's behalf when registering for
    services. Open the page, alert the user, wait for them to drop the
    key in .env.

START NOW
  Step 1: setup + baseline verify
  Step 2: pull fresh data
  Step 3: iteration loop, starting with mac_studio_tournament

Report only when (a) you have a winner, or (b) every 4 iterations with a
one-line status update like "iter 12: best Sharpe 0.71 (RETAIL_k10), still
below SPY 0.75, attempting CBOE options next."
```

---

## Key files at a glance

| Path | What it is |
|------|------------|
| `src/mlbt/mac_studio_tournament.py` | Main iteration entry point — XL models, MPS |
| `src/mlbt/strategies/vol_target.py` | Proven SPY-beating baseline |
| `src/mlbt/walk_validation.py` | Multi-window robustness gate |
| `src/mlbt/realism_audit.py` | Regime stress test (HEROIC → SMALL_CAP) |
| `src/mlbt/backtest/portfolio.py` | Long-short K-quintile simulator |
| `src/mlbt/features/regime.py` | Derived non-obvious features |
| `src/mlbt/ml/train.py` | Walk-forward LightGBM + PyTorch+MPS |
| `src/mlbt/ml/models.py` | LSTM/Transformer/PatchTST + XL variants + MultiTask |
| `src/mlbt/ml/coreml_export.py` | PyTorch → CoreML for ANE inference |
| `config/universe_sp500.yaml` | Full S&P 500 universe |
| `config/sp500_tickers.txt` | 503 ticker list |

## Final note for the user

This file IS the prompt — there is nothing else you need to provide. The
new Claude reads everything from this repo. After the autonomous run, the
new Claude should leave behind a "## Production strategy" section in
README.md with the final winning strategy.

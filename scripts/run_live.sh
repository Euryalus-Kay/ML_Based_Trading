#!/bin/bash
# Continuous paper-trading loop. Restart-safe: positions, risk ledger, and
# last-cycle reports are persisted to data/trading_state/ so a crash or
# laptop sleep doesn't break the state.
#
# Run in a foreground terminal:
#     ./scripts/run_live.sh
#
# Or detached:
#     nohup ./scripts/run_live.sh > data/trading_state/live.log 2>&1 &
#
# Logs everything to data/trading_state/live.log

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/trading_state
exec env PYTHONUNBUFFERED=1 PYTHONPATH=src python3.11 -m mlbt.trading.runner \
    --model-dir data/cand_xs8_v1 \
    --bar 1h \
    --top-k 10 \
    --broker alpaca --paper \
    --poll-seconds 60 \
    2>&1 | tee -a data/trading_state/live.log

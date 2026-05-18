#!/bin/bash
# Iteration runner for ML beat-SPY hunt.
# Each iteration:
#   1. realtime data audit
#   2. train (alpha_tournament or mac_studio_tournament)
#   3. read leaderboard
#   4. for each candidate that passes the in-tournament gate, run
#      mlbt.realism_audit AND mlbt.walk_validation_ml AND audit-sources
#   5. if all 3 pass -> WIN, save equity curve, update README, commit & push
#   6. else log diagnosis and move on

set -e
cd "/Users/zainzaidi/Desktop/ML MODEL/ML_Based_Trading"
DATASET=${1:-data/dataset_1h_sp500.parquet}
OUT_DIR=${2:-data/alpha_tournament}
HOURS=${3:-96}

echo "=== ITERATION RUNNER ==="
echo "dataset=$DATASET out_dir=$OUT_DIR hours=$HOURS"
echo

echo "--- step 1: realtime audit ---"
PYTHONPATH=src python3.11 -m mlbt.cli audit-sources --hours "$HOURS" | tail -30

echo
echo "--- step 2: alpha tournament ---"
PYTHONPATH=src python3.11 -c "
from mlbt.alpha_tournament import main
main(dataset_path='$DATASET', out_dir='$OUT_DIR')
" 2>&1 | tail -40

echo
echo "--- step 3: leaderboard ---"
if [ -f "$OUT_DIR/alpha_leaderboard.csv" ]; then
  head -1 "$OUT_DIR/alpha_leaderboard.csv"
  awk -F, 'NR>1 && $NF=="True"' "$OUT_DIR/alpha_leaderboard.csv" | head -10
fi

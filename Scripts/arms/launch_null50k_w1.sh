#!/usr/bin/env bash
# Launch the 50k-streamline NULL re-validation for all 10 tumours on W1 ONLY.
# Reuses 10k null training median + registrations; re-runs validation x65 at --select 50000.
set -euo pipefail
cd /home/Archive/BRIAN/All_30
ENVBIN=/home/aalbe03/.conda/envs/simulation_cluster/bin
LOG=/home/Archive/BRIAN/All_30/null_postfit50k_w1.log
PYTHONNOUSERSITE=1 setsid "$ENVBIN/python" -u run_postfit_null50k.py \
    --workers 6 --cpus 4 \
    --env-bin "$ENVBIN" \
    > "$LOG" 2>&1 &
echo "launched null50k pid=$! log=$LOG"

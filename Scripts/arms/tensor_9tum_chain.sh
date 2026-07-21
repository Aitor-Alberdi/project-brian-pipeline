#!/usr/bin/env bash
# Resume tensor-FOD arm across the 9 remaining subset tumours (00034 already done),
# then validate. Training driver is idempotent (skips existing best.txt markers).
set -u
cd /home/Archive/BRIAN/All_30
BIN=/home/aalbe03/.conda/envs/simulation_cluster/bin
export PATH=$BIN:$PATH
export PYTHONNOUSERSITE=1
TUM="UPENN-GBM-00001_11,UPENN-GBM-00031_11,UPENN-GBM-00024_11,UPENN-GBM-00019_11,UPENN-GBM-00023_11,UPENN-GBM-00020_11,UPENN-GBM-00025_11,UPENN-GBM-00032_11,UPENN-GBM-00042_11"
LOG=/home/Archive/BRIAN/All_30/tensor_9tum_chain.log
echo "==== TRAIN start $(date) ====" > "$LOG"
"$BIN/python" run_stage_tensor.py --tumours "$TUM" >> "$LOG" 2>&1
echo "==== TRAIN done $(date) ====" >> "$LOG"
echo "==== POSTFIT start $(date) ====" >> "$LOG"
"$BIN/python" run_postfit_tensor.py --tumours "$TUM" >> "$LOG" 2>&1
echo "==== POSTFIT done $(date) ====" >> "$LOG"
echo "==== ALL DONE $(date) ====" >> "$LOG"

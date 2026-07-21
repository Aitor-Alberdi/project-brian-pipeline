#!/bin/bash
# W2 (Workstation-2, cbajada): NULL-distribution pipeline = training then postfit.
# 4 tumours, 4 workers x 4 cpus = 16 threads (8 reserved). Postfit runs only if training exits 0.
cd /home/Archive/BRIAN/All_30 || exit 1
export PYTHONNOUSERSITE=1
EB=/home/cbajada/.conda/envs/simulation_cluster/bin
PY=$EB/python
TUM=UPENN-GBM-00020_11,UPENN-GBM-00025_11,UPENN-GBM-00032_11,UPENN-GBM-00042_11
echo "[$(date)] W2 NULL training start: $TUM"
$PY -u run_stage_null.py --tumours "$TUM" --workers 4 --cpus 4 --env-bin "$EB" > null_train_w2.log 2>&1
rc=$?
echo "[$(date)] W2 training rc=$rc"
if [ $rc -eq 0 ]; then
  echo "[$(date)] W2 NULL postfit start"
  $PY -u run_postfit_null.py --tumours "$TUM" --workers 4 --cpus 4 --env-bin "$EB" > null_postfit_w2.log 2>&1
  echo "[$(date)] W2 postfit rc=$?"
else
  echo "[$(date)] W2 training failed; postfit skipped"
fi

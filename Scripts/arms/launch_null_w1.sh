#!/bin/bash
# W1 (yoda): NULL-distribution pipeline = training (stage_null) then postfit (*_null).
# 6 tumours, 6 workers x 4 cpus = 24 threads. Postfit runs only if training exits 0.
cd /home/Archive/BRIAN/All_30 || exit 1
export PYTHONNOUSERSITE=1
EB=/home/aalbe03/.conda/envs/simulation_cluster/bin
PY=$EB/python
TUM=UPENN-GBM-00001_11,UPENN-GBM-00031_11,UPENN-GBM-00024_11,UPENN-GBM-00034_11,UPENN-GBM-00019_11,UPENN-GBM-00023_11
echo "[$(date)] W1 NULL training start: $TUM"
$PY -u run_stage_null.py --tumours "$TUM" --workers 6 --cpus 4 --env-bin "$EB" > null_train_w1.log 2>&1
rc=$?
echo "[$(date)] W1 training rc=$rc"
if [ $rc -eq 0 ]; then
  echo "[$(date)] W1 NULL postfit start"
  $PY -u run_postfit_null.py --tumours "$TUM" --workers 6 --cpus 4 --env-bin "$EB" > null_postfit_w1.log 2>&1
  echo "[$(date)] W1 postfit rc=$?"
else
  echo "[$(date)] W1 training failed; postfit skipped"
fi

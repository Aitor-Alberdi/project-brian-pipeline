#!/usr/bin/env bash
# Wait for: (1) stage_dti to finish (FINAL in stage_dti_w1.log) and (2) all 85 b=1000
# DWIs extracted, then auto-launch the 50k DTI post-fit. Fully unattended.
set -u
cd /home/Archive/BRIAN/All_30
export PATH=/home/aalbe03/.conda/envs/simulation_cluster/bin:$PATH
PY=/home/aalbe03/.conda/envs/simulation_cluster/bin/python
log(){ echo "[chain $(date +%H:%M:%S)] $*"; }

log "waiting for stage_dti FINAL ..."
until grep -q "^FINAL:" stage_dti_w1.log 2>/dev/null; do sleep 120; done
log "stage_dti done: $(grep "^FINAL:" stage_dti_w1.log)"

log "waiting for all 85 b1000 DWIs ..."
until [ "$(ls dti_inputs/*/dwi_b1000_ras.mif 2>/dev/null | wc -l)" -ge 85 ] || grep -q "^FINAL:" extract_b1000_w1.log 2>/dev/null; do sleep 60; done
log "extraction: $(ls dti_inputs/*/dwi_b1000_ras.mif 2>/dev/null | wc -l)/85 mifs; $(grep "^FINAL:" extract_b1000_w1.log 2>/dev/null)"

log "launching postfit_dti (50k) ..."
PYTHONNOUSERSITE=1 $PY -u run_postfit_dti.py --workers 6 --cpus 4 > postfit_dti_w1.log 2>&1
log "postfit_dti done: $(grep "^FINAL:" postfit_dti_w1.log)"

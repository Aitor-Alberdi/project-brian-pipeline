#!/usr/bin/env bash
set -euo pipefail

HCP_ROOT="${HCP_ROOT:-/opt/nfs/neuroinformatics/HCP/HCP_1200_qc_data}"
ODF_TRAIN_ROOT="${ODF_TRAIN_ROOT:-/opt/nfs/neuroinformatics/BRIAN/dODFs/odf_generation_results_20}"
ODF_VAL_ROOT="${ODF_VAL_ROOT:-/opt/nfs/neuroinformatics/BRIAN/dODFs/odf_generation_results_66}"
UPENN_ROOT="${UPENN_ROOT:-/opt/nfs/neuroinformatics/BRIAN/UPENN-15}"
MNI_BRAIN="${MNI_BRAIN:-/opt/nfs/neuroinformatics/BRIAN/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz}"
BASE_ROOT="${BASE_ROOT:-/opt/users/aalbe03/Parallel_runs}"
HCP_TO_MNI_CACHE="${HCP_TO_MNI_CACHE:-${BASE_ROOT}/cache/hcp_to_mni}"

TRAIN_IDS="100408 101309 101915 103111 103414 103818 105014 105115 108828 110411 113619 113922 114419 115320 116524 117122 118730 118932 120111 122317"
mapfile -t VAL_IDS < <(python3 - "$HCP_ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
ids = sorted([p.name for p in root.iterdir() if p.is_dir()])
for sid in ids[21:87]:
    if sid != '756055':
        print(sid)
PY
)

missing=0
for p in "${HCP_ROOT}" "${ODF_TRAIN_ROOT}" "${ODF_VAL_ROOT}" "${UPENN_ROOT}" "${MNI_BRAIN}"; do
  [[ -e "${p}" ]] || { echo "[MISSING ROOT] ${p}"; missing=1; }
done

echo "[INFO] checking training HCP + ODFs"
for id in ${TRAIN_IDS}; do
  [[ -f "${HCP_ROOT}/${id}/T1w/T1w_acpc_dc_restore_brain.nii.gz" ]] || { echo "[MISSING] train HCP T1 ${id}"; missing=1; }
  [[ -f "${ODF_TRAIN_ROOT}/${id}/odf_sh_order12_mrtrix.mif" ]] || { echo "[MISSING] train ODF ${id}"; missing=1; }
done

echo "[INFO] checking validation HCP + ODFs"
for id in "${VAL_IDS[@]}"; do
  [[ -f "${HCP_ROOT}/${id}/T1w/T1w_acpc_dc_restore_brain.nii.gz" ]] || { echo "[MISSING] val HCP T1 ${id}"; missing=1; }
  [[ -f "${ODF_VAL_ROOT}/${id}/odf_sh_order12_mrtrix.mif" ]] || { echo "[MISSING] val ODF ${id}"; missing=1; }
done

echo "[INFO] validation IDs count=${#VAL_IDS[@]}"
for tumor_dir in "${UPENN_ROOT}"/*; do
  [[ -d "${tumor_dir}" ]] || continue
  tid=$(basename "${tumor_dir}")
  [[ -f "${tumor_dir}/${tid}_T1.nii.gz" ]] || { echo "[MISSING] UPENN T1 ${tid}"; missing=1; }
  [[ -f "${tumor_dir}/${tid}_automated_approx_segm.nii.gz" ]] || { echo "[MISSING] UPENN seg ${tid}"; missing=1; }
done

# Cache HCP->MNI: crear si no existe y reportar estado
if [[ ! -d "${HCP_TO_MNI_CACHE}" ]]; then
  echo "[INFO] cache HCP->MNI no existe; lo creo: ${HCP_TO_MNI_CACHE}"
  mkdir -p "${HCP_TO_MNI_CACHE}" || { echo "[MISSING] no puedo crear ${HCP_TO_MNI_CACHE}"; missing=1; }
fi

if [[ -d "${HCP_TO_MNI_CACHE}" ]]; then
  cache_hits=0
  cache_misses=0
  for id in ${TRAIN_IDS} "${VAL_IDS[@]}"; do
    if [[ -f "${HCP_TO_MNI_CACHE}/${id}/.cache_done" ]]; then
      cache_hits=$((cache_hits+1))
    else
      cache_misses=$((cache_misses+1))
    fi
  done
  echo "[INFO] HCP->MNI cache status: ${cache_hits} hit / ${cache_misses} miss (total $((cache_hits+cache_misses)))"
  if [[ ${cache_misses} -gt 0 ]]; then
    echo "[INFO] Para pre-rellenar el cache:"
    echo "[INFO]   bash Scripts/build_hcp_ids_list.sh > ${BASE_ROOT}/hcp_ids_all.txt"
    echo "[INFO]   sbatch Scripts/submit_prewarm_hcp_to_mni.slurm ${BASE_ROOT}/hcp_ids_all.txt"
  fi
fi

if [[ ${missing} -eq 0 ]]; then
  echo "[OK] all required inputs found"
else
  echo "[FAIL] missing inputs detected"
  exit 1
fi

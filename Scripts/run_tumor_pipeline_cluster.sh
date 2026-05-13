#!/usr/bin/env bash
set -euo pipefail

TUMOR_ID="${1:?Usage: bash run_tumor_pipeline_cluster.sh <TUMOR_ID> [extra args for run_full_tumor_pipeline.py]}"
shift || true

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
BASE_ROOT="${BASE_ROOT:-/opt/users/aalbe03/Parallel_runs}"
PIPELINE_ROOT="${PIPELINE_ROOT:-${BASE_ROOT}/runs/${TUMOR_ID}}"
UPENN_ROOT="${UPENN_ROOT:-/opt/nfs/neuroinformatics/BRIAN/UPENN-15}"
HCP_ROOT="${HCP_ROOT:-/opt/nfs/neuroinformatics/HCP/HCP_1200_qc_data}"
ODF_TRAIN_ROOT="${ODF_TRAIN_ROOT:-/opt/nfs/neuroinformatics/BRIAN/dODFs/odf_generation_results_20}"
ODF_VAL_ROOT="${ODF_VAL_ROOT:-/opt/nfs/neuroinformatics/BRIAN/dODFs/odf_generation_results_66}"
MNI_BRAIN="${MNI_BRAIN:-/opt/nfs/neuroinformatics/BRIAN/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz}"
HCP_TO_MNI_CACHE="${HCP_TO_MNI_CACHE:-${BASE_ROOT}/cache/hcp_to_mni}"
SUMMARY="${SUMMARY:-median}"
THRESHOLD="${THRESHOLD:-0.02}"
LABELS="${LABELS:-1,4}"
UPENN_TO_MNI_SUBJECT="${UPENN_TO_MNI_SUBJECT:-100408}"
TRAIN_IDS_FILE="${TRAIN_IDS_FILE:-}"
VAL_IDS_FILE="${VAL_IDS_FILE:-}"

T1_UPENN="${UPENN_ROOT}/${TUMOR_ID}/${TUMOR_ID}_T1.nii.gz"
TUMOR_MASK="${UPENN_ROOT}/${TUMOR_ID}/${TUMOR_ID}_automated_approx_segm.nii.gz"

# Crea el pipeline root y el cache compartido si no existen
mkdir -p "${PIPELINE_ROOT}" "${HCP_TO_MNI_CACHE}"

cmd=(
  python3 -u "${SCRIPT_DIR}/run_full_tumor_pipeline.py"
  --pipeline-root "${PIPELINE_ROOT}"
  --hcp-root "${HCP_ROOT}"
  --tumor-mask-upenn "${TUMOR_MASK}"
  --t1-upenn "${T1_UPENN}"
  --mni-brain "${MNI_BRAIN}"
  --hcp-to-mni-cache "${HCP_TO_MNI_CACHE}"
  --optimizer-script "${CODE_ROOT}/rand_seed_v2.py"
  --odf-train-root "${ODF_TRAIN_ROOT}"
  --odf-val-root "${ODF_VAL_ROOT}"
  --summary "${SUMMARY}"
  --threshold "${THRESHOLD}"
  --labels "${LABELS}"
  --upenn-to-mni-subject "${UPENN_TO_MNI_SUBJECT}"
  --cleanup-training-intermediates
)

if [[ -n "${TRAIN_IDS_FILE}" ]]; then
  cmd+=(--train-ids-file "${TRAIN_IDS_FILE}")
fi
if [[ -n "${VAL_IDS_FILE}" ]]; then
  cmd+=(--val-ids-file "${VAL_IDS_FILE}")
fi

cmd+=("$@")

printf '[CMD]'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

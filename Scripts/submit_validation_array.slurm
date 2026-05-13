#!/bin/bash
#SBATCH --partition=research_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=15G
#SBATCH --time=12:00:00
#SBATCH --account=brian
#SBATCH --nodelist=ori
#SBATCH --job-name=brian_validation
#SBATCH --array=0-8%5
#SBATCH --output=/opt/users/aalbe03/Parallel_runs/logs/%x_%A_%a.out
#SBATCH --error=/opt/users/aalbe03/Parallel_runs/logs/%x_%A_%a.err
#SBATCH --mail-user=aitor.alberdi@um.edu.mt
#SBATCH --mail-type=END,FAIL

set -euo pipefail

TUMOR_LIST_FILE="${1:?Usage: sbatch submit_validation_array.slurm /path/to/tumor_ids_done.txt}"
TUMOR_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TUMOR_LIST_FILE}")
[[ -n "${TUMOR_ID}" ]] || { echo "No tumor ID for array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }

BASE_ROOT="/opt/users/aalbe03/Parallel_runs"
mkdir -p "${BASE_ROOT}/logs" "${BASE_ROOT}/runs"

source /opt/conda/etc/profile.d/conda.sh
conda activate simulation

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK}"
export MRTRIX_NTHREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Phase 1: validation + averages for the 9 fully-trained tumours.
# Skip registration/mrconvert/optimizer/collect — all of those artefacts already
# exist on disk for these 9 tumours (training and collect ran during the
# original campaign). Orchestrator jumps straight to step 5 (validation per val
# subject) + step 6 (UPENN->MNI + averages + reference compare).
bash "${BASE_ROOT}/code/pipeline_changes/Scripts/run_tumor_pipeline_cluster.sh" \
  "${TUMOR_ID}" \
  --skip-registration \
  --skip-mrconvert \
  --skip-optimizer \
  --skip-collect

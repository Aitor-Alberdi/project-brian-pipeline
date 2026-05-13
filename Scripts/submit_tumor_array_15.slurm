#!/bin/bash
#SBATCH --partition=research_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=15G
#SBATCH --time=14-00:00:00
#SBATCH --account=brian
#SBATCH --nodelist=ori
#SBATCH --job-name=brian_tumor
#SBATCH --array=0-14%15
#SBATCH --output=/opt/users/aalbe03/Parallel_runs/logs/%x_%A_%a.out
#SBATCH --error=/opt/users/aalbe03/Parallel_runs/logs/%x_%A_%a.err
#SBATCH --mail-user=aitor.alberdi@um.edu.mt
#SBATCH --mail-type=END,FAIL

set -euo pipefail

TUMOR_LIST_FILE="${1:?Usage: sbatch submit_tumor_array_15.slurm /path/to/tumor_ids_15.txt}"
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

# Entrenamiento + collect (se detiene antes de validation/averages porque
# odf_generation_results_66 aun no es legible para este usuario; el lunes,
# cuando el PI cambie los permisos, relanzamos con --skip-registration
# --skip-optimizer --skip-collect para completar validation + averages).
bash "${BASE_ROOT}/code/pipeline_changes/Scripts/run_tumor_pipeline_cluster.sh" \
  "${TUMOR_ID}" \
  --skip-validation \
  --skip-averages

#!/bin/bash
#SBATCH --partition=research_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=15G
#SBATCH --time=14-00:00:00
#SBATCH --account=brian
#SBATCH --nodelist=ori
#SBATCH --job-name=brian_tumor_b2
#SBATCH --array=0-14%15
#SBATCH --output=/opt/users/aalbe03/Parallel_runs/logs/%x_%A_%a.out
#SBATCH --error=/opt/users/aalbe03/Parallel_runs/logs/%x_%A_%a.err
#SBATCH --mail-user=aitor.alberdi@um.edu.mt
#SBATCH --mail-type=END,FAIL

set -euo pipefail

TUMOR_LIST_FILE="${1:?Usage: sbatch submit_tumor_array_batch2.slurm /path/to/tumor_ids_batch2.txt}"
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

# Batch 2: full end-to-end pipeline (training + collect + validation + averages).
# odf_generation_results_66 perms now fixed, so no skip flags needed.
# Override UPENN_ROOT to the user-owned batch2 input dir; HCP cache + dODFs
# stay on the shared NFS paths via run_tumor_pipeline_cluster.sh defaults.
export UPENN_ROOT="${BASE_ROOT}/UPENN-15-batch2"

bash "${BASE_ROOT}/code/pipeline_changes/Scripts/run_tumor_pipeline_cluster.sh" \
  "${TUMOR_ID}"

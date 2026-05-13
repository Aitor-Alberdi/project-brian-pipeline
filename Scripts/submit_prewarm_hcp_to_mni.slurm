#!/bin/bash
#SBATCH --partition=research_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=15G
#SBATCH --time=08:00:00
#SBATCH --account=brian
#SBATCH --nodelist=ori
#SBATCH --job-name=brian_prewarm
#SBATCH --array=0-84%16
#SBATCH --output=/opt/users/aalbe03/Parallel_runs/logs/%x_%A_%a.out
#SBATCH --error=/opt/users/aalbe03/Parallel_runs/logs/%x_%A_%a.err
#SBATCH --mail-user=aitor.alberdi@um.edu.mt
#SBATCH --mail-type=END,FAIL

# Pre-rellena el cache compartido HCP->MNI para los 85 sujetos (20 train + 65 val).
# Lanzalo UNA vez antes del array de tumores; al terminar, todos los jobs de
# tumor encontraran cache-hit en cada sujeto HCP que visiten.
#
# Uso:
#   sbatch submit_prewarm_hcp_to_mni.slurm /opt/users/aalbe03/Parallel_runs/hcp_ids_all.txt
#
# El fichero debe contener UN ID por linea (20 train + 65 val = 85 lineas).
# Si tu lista tiene mas/menos IDs, ajusta #SBATCH --array=0-N-1 donde N = numero de lineas.

set -euo pipefail

IDS_FILE="${1:?Usage: sbatch submit_prewarm_hcp_to_mni.slurm /path/to/hcp_ids_all.txt}"
ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${IDS_FILE}")
[[ -n "${ID}" ]] || { echo "No ID for array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }

BASE_ROOT="${BASE_ROOT:-/opt/users/aalbe03/Parallel_runs}"
CACHE_ROOT="${CACHE_ROOT:-${BASE_ROOT}/cache/hcp_to_mni}"
HCP_ROOT="${HCP_ROOT:-/opt/nfs/neuroinformatics/HCP/HCP_1200_qc_data}"
MNI_BRAIN="${MNI_BRAIN:-/opt/nfs/neuroinformatics/BRIAN/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz}"

# Crea directorios si no existen
mkdir -p "${BASE_ROOT}/logs" "${CACHE_ROOT}"

source /opt/conda/etc/profile.d/conda.sh
conda activate simulation

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK}"
export MRTRIX_NTHREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SCRIPT_DIR="${BASE_ROOT}/code/pipeline_changes/Scripts"

bash "${SCRIPT_DIR}/prewarm_hcp_to_mni.sh" \
  --id "${ID}" \
  --hcp-root "${HCP_ROOT}" \
  --mni-brain "${MNI_BRAIN}" \
  --cache-root "${CACHE_ROOT}"

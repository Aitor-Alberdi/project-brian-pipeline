#!/usr/bin/env bash
set -euo pipefail

# Rellena el cache compartido HCP->MNI para una lista de sujetos HCP
# (o uno solo). Pensado para usarse como trabajo independiente antes
# de lanzar el array de tumores, o como payload de un array Slurm.
#
# Uso (un sujeto, ej. dentro de un array Slurm):
#   bash prewarm_hcp_to_mni.sh \
#     --id 100408 \
#     --hcp-root /opt/nfs/neuroinformatics/HCP/HCP_1200_qc_data \
#     --mni-brain /opt/nfs/neuroinformatics/BRIAN/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz \
#     --cache-root /opt/users/aalbe03/Parallel_runs/cache/hcp_to_mni
#
# Uso (serial sobre una lista):
#   bash prewarm_hcp_to_mni.sh \
#     --ids-file /opt/users/aalbe03/Parallel_runs/hcp_ids_all.txt \
#     --hcp-root ... --mni-brain ... --cache-root ...

err(){ echo "[ERROR] $*" >&2; exit 1; }
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

IDS_FILE=""
ONE_ID=""
HCP_ROOT=""
MNI_BRAIN=""
CACHE_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ids-file)   IDS_FILE="$2"; shift 2 ;;
    --id)         ONE_ID="$2"; shift 2 ;;
    --hcp-root)   HCP_ROOT="$2"; shift 2 ;;
    --mni-brain)  MNI_BRAIN="$2"; shift 2 ;;
    --cache-root) CACHE_ROOT="$2"; shift 2 ;;
    *) err "Argumento desconocido: $1" ;;
  esac
done
[[ -n "$HCP_ROOT" && -n "$MNI_BRAIN" && -n "$CACHE_ROOT" ]] \
  || err "Requeridos: --hcp-root --mni-brain --cache-root"

mkdir -p "$CACHE_ROOT"

process_one(){
  local id="$1"
  local t1="$HCP_ROOT/$id/T1w/T1w_acpc_dc_restore_brain.nii.gz"
  if [[ ! -f "$t1" ]]; then
    echo "[WARN] $id: no existe $t1, salto" >&2
    return 0
  fi
  bash "$SCRIPT_DIR/cache_hcp_to_mni.sh" \
    --subject-id "$id" \
    --t1-hcp "$t1" \
    --mni-brain "$MNI_BRAIN" \
    --cache-root "$CACHE_ROOT"
}

if [[ -n "$ONE_ID" ]]; then
  process_one "$ONE_ID"
  echo "[DONE] prewarm de 1 sujeto completado ($CACHE_ROOT)"
  exit 0
fi

[[ -n "$IDS_FILE" ]] || err "Pasa --ids-file o --id"
[[ -f "$IDS_FILE" ]] || err "No existe IDs file: $IDS_FILE"

n=0
while IFS= read -r raw; do
  id=$(echo "$raw" | tr -d ' \t\r')
  [[ -z "$id" ]] && continue
  process_one "$id"
  n=$((n+1))
done < "$IDS_FILE"

echo "[DONE] prewarm de $n sujetos completado ($CACHE_ROOT)"

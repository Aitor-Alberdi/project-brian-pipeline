#!/usr/bin/env bash
set -euo pipefail

# Genera hcp_ids_all.txt = 20 IDs train + 65 IDs val (85 en total),
# con el mismo criterio que el pipeline original:
#   - Train: lista hardcoded en TRAIN_IDS (coincide con check_cluster_inputs.sh
#     y con TRAIN_IDS_DEFAULT de run_full_tumor_pipeline.py)
#   - Val: IDs 22..87 (1-indexed) de HCP_1200_qc_data ordenados alfabeticamente,
#     excluyendo 756055 (mismo criterio que check_cluster_inputs.sh)
#
# Uso:
#   bash build_hcp_ids_list.sh [HCP_ROOT] > /opt/users/aalbe03/Parallel_runs/hcp_ids_all.txt

HCP_ROOT="${1:-/opt/nfs/neuroinformatics/HCP/HCP_1200_qc_data}"

TRAIN_IDS=(
  100408 101309 101915 103111 103414 103818 105014 105115
  108828 110411 113619 113922 114419 115320 116524 117122
  118730 118932 120111 122317
)

for id in "${TRAIN_IDS[@]}"; do
  echo "$id"
done

python3 - "$HCP_ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
ids = sorted([p.name for p in root.iterdir() if p.is_dir()])
for sid in ids[21:87]:
    if sid != '756055':
        print(sid)
PY

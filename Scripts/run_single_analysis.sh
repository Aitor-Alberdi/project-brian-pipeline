#!/usr/bin/env bash

set -euo pipefail

# Training IDs (equivalente a:
#   ls -l /home/Archive/HCP_1200_qc_data | head -n 22 | tail -n 20 | awk '{print $NF}'
# )
IDS="100408 101309 101915 103111 103414 103818 105014 105115 108828 110411 113619 113922 114419 115320 116524 117122 118730 118932 120111 122317"

BASE="/home/Archive/BRIAN/Pipeline"
REST="/home/Archive/BRIAN/Analysis_HCP_UPENN/"

# ODFs (se deja como en tu setup; cambia aquí si procede)
ODF_ROOT="${REST}/odf_generation_results_20"

# Registration root (nuevo naming del pipeline)
REG_ROOT="${BASE}/registration_20"

# Output root (nuevo)
OUT_ROOT="${BASE}/training_results"
mkdir -p "${OUT_ROOT}"
cd "${OUT_ROOT}"

for id in ${IDS};do
	mkdir ${id}
	cd ${id}
	python3 -u ${BASE}/rand_seed_v2.py \
	  --data "${ODF_ROOT}/${id}/odf_sh_order12_mrtrix.mif" \
	  --output "analysis_participant_${id}" \
	  -ang 20,90,2 \
	  -maxl 20,90,2 \
	  --radius 3 \
	  -st 0.3,1,0.05 \
	  --image "${REG_ROOT}/${id}/tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz" \
	  --template "${REG_ROOT}/${id}/T1_hcp_RAS.mif" \
	  -co 0.04,0.1,0.005 \
	  --threshold 0.02 \
	  --pos_window_mm 5 \
	  --init_points 10 \
	  --init_percentiles 0,25,50,75,100 \
	  --patience_passes 10 \
	  --improve_tol 0.005 \
	  --target_obj 0.1 | tee -a output.txt
	cd ../
done

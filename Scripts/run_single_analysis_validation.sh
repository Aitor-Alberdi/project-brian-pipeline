#!/usr/bin/env bash

set -euo pipefail

# Validation IDs (equivalente a:
#   ls -l /home/Archive/HCP_1200_qc_data | head -n 88 | tail -n 66 | awk '{print $NF}' | grep -v 756055
# )
IDS=$(ls -l /home/Archive/HCP_1200_qc_data | head -n 88 | tail -n 66 | awk '{print $NF}' | grep -v 756055)

BASE="/home/Archive/BRIAN/Pipeline"
REST="/home/Archive/BRIAN/Analysis_HCP_UPENN/"

ODF_ROOT="${REST}/odf_generation_results_66"
REG_ROOT="${BASE}/validation_65"

# Outputs
OUT_ROOT="${BASE}/validation_results"
mkdir -p "${OUT_ROOT}"
cd "${OUT_ROOT}"

# Qué resumen usar (median|mean). Default: median
SUMMARY="${SUMMARY:-median}"
PARAMS_CSV="${BASE}/best_mni_outputs/best_params_all.${SUMMARY}.csv"

# CSV con coords nativas por sujeto (salida típica de warp_mni_to_subject_coords.py)
COORDS_CSV="${BASE}/mni_to_native_coords.csv"

i=2

for id in ${IDS};do
	mkdir ${id}
	cd ${id}
	python3 -u ${BASE}/validate_single_run_currC.py \
	  --data "${ODF_ROOT}/${id}/odf_sh_order12_mrtrix.mif" \
	  --output "analysis_participant_${id}" \
	  --angle  $(cat "${PARAMS_CSV}" | grep "angle"  | cut -d"," -f2) \
	  --maxlen $(cat "${PARAMS_CSV}" | grep "maxlen" | cut -d"," -f2) \
	  --radius 3 \
	  --step 0.5 \
	  --image "${REG_ROOT}/${id}/tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz" \
	  --template "${REG_ROOT}/${id}/T1_hcp_RAS.mif" \
	  --cutoff $(cat "${PARAMS_CSV}" | grep "cutoff" | head -n 1 | cut -d"," -f2) \
	  --threshold 0.02 \
	  --x $(cat "${COORDS_CSV}" | awk "NR==${i}" | cut -d"," -f5) \
	  --y $(cat "${COORDS_CSV}" | awk "NR==${i}" | cut -d"," -f6) \
	  --z $(cat "${COORDS_CSV}" | awk "NR==${i}" | cut -d"," -f7) | tee -a output.txt
	cd ../
	let i++
done

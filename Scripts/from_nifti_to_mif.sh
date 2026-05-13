#!/usr/bin/env bash

set -euo pipefail

# Script genérico para convertir T1_hcp_RAS.nii.gz -> T1_hcp_RAS.mif en un reg-root.
# Uso típico:
#   ./from_nifti_to_mif.sh /home/Archive/BRIAN/Pipeline/registration_20 "100408 101309 ..."
#   ./from_nifti_to_mif.sh /home/Archive/BRIAN/Pipeline/validation_65 "$(...)"

REG_ROOT="${1:-/home/Archive/BRIAN/Pipeline/validation_65}"
IDS="${2:-}"

if [[ -z "${IDS}" ]]; then
  # fallback: IDs del set de validación
  IDS=$(ls -l /home/Archive/HCP_1200_qc_data | head -n 88 | tail -n 66 | awk '{print $NF}' | grep -v 756055)
fi

for id in ${IDS}; do
  in_nii="${REG_ROOT}/${id}/T1_hcp_RAS.nii.gz"
  out_mif="${REG_ROOT}/${id}/T1_hcp_RAS.mif"
  echo -e "\n[+] mrconvert ${in_nii} -> ${out_mif}\n"
  mrconvert "${in_nii}" "${out_mif}"
done

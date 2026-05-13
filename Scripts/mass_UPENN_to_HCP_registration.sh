#!/usr/bin/env bash

HOME_DIR="/home/Archive/BRIAN/Analysis_HCP_UPENN"
UPENN_DIR="/home/Archive/BRIAN/Pipeline"
HCP_DATA_PATH="/home/Archive/HCP_1200_qc_data"
#IDS=$(ls -l /home/Archive/HCP_1200_qc_data | head -n 22 | tail -n 20 | awk '{print $NF}')
IDS=$(ls -l /home/Archive/HCP_1200_qc_data | head -n 88 | tail -n 66 | awk '{print $NF}' | grep -v 756055)
#IDS="756055"

mkdir ${UPENN_DIR}/registration_results_65 && cd ${UPENN_DIR}/registration_results_65
#cd registration_results_66

for id in $IDS;do
	echo -e "\n[+] Warping participant ${id} to HCP space"
	mkdir ${id}
	cd ${id}
	echo -e "\n[+] Folder ${id} generated"
	echo -e "\n[+] Starting warping process...\n"
	bash ${HOME_DIR}/pipeline_original_T1_MNI.sh --t1-upenn ${UPENN_DIR}/UPENN-GBM/NIfTI-files/images_structural/UPENN-GBM-00174_11/UPENN-GBM-00174_11_T1.nii.gz --t1-hcp ${HCP_DATA_PATH}/${id}/T1w/T1w_acpc_dc_restore_brain.nii.gz --tumor-seg-upenn ${UPENN_DIR}/UPENN-GBM/NIfTI-files/automated_segm/UPENN-GBM-00174_11_automated_approx_segm.nii.gz --mni-brain ${HOME_DIR}/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz --outdir ./
	echo -e "\n[+] Warping finished for participant ${id}"
	#cd ${UPENN_DIR}/registration_results_20
	cd ${UPENN_DIR}/registration_results_65
done

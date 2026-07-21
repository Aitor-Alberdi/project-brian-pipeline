#!/usr/bin/env bash
# Build tensor diffusion-ODFs for every host (20 train + 65 val) for the iFOD2-on-DTI arm.
# Identical SF->SH->reorient path as the SHORE dODF; reorient template = that host's SHORE FOD
# (guarantees byte-identical grid). 4 hosts in parallel, idempotent, cleans intermediates.
set -u
export PATH=/home/aalbe03/.conda/envs/simulation_cluster/bin:$PATH
export PYTHONPATH=/tmp/pystub            # h5py/tqdm stubs so user-site dipy imports
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONNOUSERSITE                    # dipy lives in user-site

SCRIPT=/home/Archive/BRIAN/All_30/compute_odfs_tensor.py
TRAIN_SRC=/home/Archive/BRIAN/Analysis_HCP_UPENN/odf_generation_results_20
VAL_SRC=/home/Archive/BRIAN/Analysis_HCP_UPENN/odf_generation_results_66
TROOT=/home/Archive/BRIAN/All_30/tensor_odf_arm/odfs
EXCLUDE="756055"
MAXJOBS=2
mkdir -p $TROOT

build_host() {
  local host=$1 src=$2
  local out=$TROOT/$host
  local final=$out/odf_sh_order12_tensor_mrtrix.mif
  if [ -f "$final" ]; then echo "skip  $host (exists)"; return; fi
  mkdir -p $out
  : > $out/build.log
  # 1) brain mask = nonzero support of the SHORE SF (streaming max over directions)
  mrmath $src/odf_sf_ras_shore.nii.gz max -axis 3 - -force 2>>$out/build.log \
    | mrcalc - 0.0 -gt $out/brainmask.nii.gz -datatype bit -force >>$out/build.log 2>&1
  # 2) tensor FOD via identical amp2sh/reorient path
  /usr/bin/python3 $SCRIPT \
    --in_nii $src/data_ras.nii.gz --in_bvals $src/bvals_ras --in_bvecs $src/bvecs_ras \
    --mask_nii $out/brainmask.nii.gz --sphere_txt $src/repulsion724.txt \
    --template_mif $src/odf_sh_order12_mrtrix.mif \
    --out_dir $out --bmax 1100 --nthreads 4 --cleanup >>$out/build.log 2>&1
  if [ -f "$final" ]; then echo "OK    $host"; else echo "FAIL  $host (see $out/build.log)"; fi
}
export -f build_host
export TROOT SCRIPT

# host -> source-dir map
declare -A SRC
for h in $(ls $TRAIN_SRC); do [ -d "$TRAIN_SRC/$h" ] && SRC[$h]=$TRAIN_SRC/$h; done
for h in $(ls $VAL_SRC);   do [ -d "$VAL_SRC/$h" ]   && SRC[$h]=$VAL_SRC/$h;   done
unset 'SRC[756055]'

echo "hosts to build: ${#SRC[@]}  (max $MAXJOBS parallel)"
n=0
for h in "${!SRC[@]}"; do
  build_host "$h" "${SRC[$h]}" &
  n=$((n+1))
  while [ "$(jobs -r | wc -l)" -ge $MAXJOBS ]; do sleep 5; done
done
wait
echo "==== ALL TENSOR ODFS DONE ===="
ls $TROOT/*/odf_sh_order12_tensor_mrtrix.mif 2>/dev/null | wc -l

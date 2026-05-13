#!/usr/bin/env bash
set -euo pipefail

# Registro UPENN T1 -> MNI (Rigid + Affine + SyN), excluyendo el tumor de la
# metrica del lado UPENN. Depende SOLO del tumor UPENN y del template MNI,
# asi que se ejecuta UNA vez por tumor.
#
# Input: un --workdir preparado por prep_upenn_for_mni.sh que contenga:
#   T1_upenn_RAS.nii.gz
#   T1_upenn_mask_excl_RAS.nii.gz
#   MNI_brain_RAS.nii.gz
#
# Output (mismo --workdir):
#   upT1_2mni_0GenericAffine.mat
#   upT1_2mni_1Warp.nii.gz
#   upT1_2mni_1InverseWarp.nii.gz
#   upT1_in_MNI.nii.gz
#
# Idempotente: si ya existen los tres transforms, no re-registra.

err(){ echo "[ERROR] $*" >&2; exit 1; }
need(){ command -v "$1" >/dev/null 2>&1 || err "No encuentro '$1' en PATH"; }
need antsRegistration; need python3

WORKDIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir) WORKDIR="$2"; shift 2 ;;
    *) err "Argumento desconocido: $1" ;;
  esac
done
[[ -n "$WORKDIR" ]] || err "Requerido: --workdir"

abs(){ python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1"; }
WORKDIR=$(abs "$WORKDIR")
cd "$WORKDIR"

for f in T1_upenn_RAS.nii.gz T1_upenn_mask_excl_RAS.nii.gz MNI_brain_RAS.nii.gz; do
  [[ -f "$f" ]] || err "Falta $WORKDIR/$f (ejecuta prep_upenn_for_mni.sh antes)"
done

if [[ -f upT1_2mni_0GenericAffine.mat && -f upT1_2mni_1Warp.nii.gz && -f upT1_2mni_1InverseWarp.nii.gz ]]; then
  echo "[SKIP] UPENN->MNI ya registrado en $WORKDIR"
  exit 0
fi

antsRegistration -d 3 --float 1 --use-histogram-matching 1 -v 1 \
  -r [MNI_brain_RAS.nii.gz,T1_upenn_RAS.nii.gz,1] \
  -x [NULL,T1_upenn_mask_excl_RAS.nii.gz] \
  \
  -m MI[MNI_brain_RAS.nii.gz,T1_upenn_RAS.nii.gz,1,32] \
  -t Rigid[0.1]  -c [2000x1000x500,1e-6,10] -f 8x4x2 -s 3x2x1vox \
  \
  -m MI[MNI_brain_RAS.nii.gz,T1_upenn_RAS.nii.gz,1,32] \
  -t Affine[0.1] -c [2000x1000x500,1e-6,10] -f 8x4x2 -s 3x2x1vox \
  \
  -m CC[MNI_brain_RAS.nii.gz,T1_upenn_RAS.nii.gz,1,4] \
  -t SyN[0.1,3,0] -c [500x250x100,1e-6,10] -f 8x4x2 -s 3x2x1vox \
  \
  -o [upT1_2mni_, upT1_in_MNI.nii.gz]

echo "[OK] UPENN->MNI registrado en $WORKDIR"

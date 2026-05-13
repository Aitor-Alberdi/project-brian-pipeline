#!/usr/bin/env bash
set -euo pipefail

# Registro de UN sujeto HCP en el contexto de UN tumor UPENN.
# Sustituye al antiguo pipeline_original_T1_MNI.sh en el camino caliente.
#
# Que hace:
#   1) Asegura (o rellena) el cache HCP->MNI del sujeto (cache_hcp_to_mni.sh).
#   2) Symlinks en el --outdir (registro_<set>/<SID>/) a:
#        - cache:     T1_hcp_*, hcpT1_2mni_*, MNI_brain_RAS, etc.
#        - upenn_wd:  upT1_2mni_*, tumor14_T1up_brain_RAS, etc.
#   3) Compose (HCP->MNI)^-1 o (UPENN->MNI) y warpea el tumor al grid HCP T1.
#   4) Aplica mascara cerebro HCP al tumor warpeado.
#
# Idempotente: si el warp final ya existe, saltamos.

err(){ echo "[ERROR] $*" >&2; exit 1; }
need(){ command -v "$1" >/dev/null 2>&1 || err "No encuentro '$1' en PATH"; }
need mrcalc; need antsApplyTransforms

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SUBJECT_ID=""
T1_HCP=""
MNI_BRAIN=""
UPENN_WORKDIR=""
CACHE_ROOT=""
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject-id)      SUBJECT_ID="$2"; shift 2 ;;
    --t1-hcp)          T1_HCP="$2"; shift 2 ;;
    --mni-brain)       MNI_BRAIN="$2"; shift 2 ;;
    --upenn-workdir)   UPENN_WORKDIR="$2"; shift 2 ;;
    --cache-root)      CACHE_ROOT="$2"; shift 2 ;;
    --outdir)          OUTDIR="$2"; shift 2 ;;
    *) err "Argumento desconocido: $1" ;;
  esac
done
[[ -n "$SUBJECT_ID" && -n "$T1_HCP" && -n "$MNI_BRAIN" && -n "$UPENN_WORKDIR" && -n "$CACHE_ROOT" && -n "$OUTDIR" ]] \
  || err "Requeridos: --subject-id --t1-hcp --mni-brain --upenn-workdir --cache-root --outdir"

abs(){ python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1"; }
T1_HCP=$(abs "$T1_HCP")
MNI_BRAIN=$(abs "$MNI_BRAIN")
UPENN_WORKDIR=$(abs "$UPENN_WORKDIR")
CACHE_ROOT=$(abs "$CACHE_ROOT")
OUTDIR=$(abs "$OUTDIR")

mkdir -p "$CACHE_ROOT" "$OUTDIR"

# 1) asegurar cache
bash "$SCRIPT_DIR/cache_hcp_to_mni.sh" \
  --subject-id "$SUBJECT_ID" \
  --t1-hcp "$T1_HCP" \
  --mni-brain "$MNI_BRAIN" \
  --cache-root "$CACHE_ROOT"

CACHE_DIR="$CACHE_ROOT/$SUBJECT_ID"
cd "$OUTDIR"

# 2) symlinks de cache (lado HCP + MNI)
for f in T1_hcp_RAS.nii.gz T1_hcp_RAS.mif T1_hcp_brain_RAS.nii.gz \
         hcpT1_2mni_0GenericAffine.mat hcpT1_2mni_1Warp.nii.gz hcpT1_2mni_1InverseWarp.nii.gz \
         hcpT1_in_MNI.nii.gz MNI_brain_RAS.nii.gz; do
  ln -sfn "$CACHE_DIR/$f" "$OUTDIR/$f"
done

# 3) symlinks del upenn_workdir (lado UPENN, compartido entre sujetos del mismo tumor)
for f in upT1_2mni_0GenericAffine.mat upT1_2mni_1Warp.nii.gz upT1_2mni_1InverseWarp.nii.gz \
         upT1_in_MNI.nii.gz tumor14_T1up_brain_RAS.nii.gz T1_upenn_RAS.nii.gz; do
  [[ -f "$UPENN_WORKDIR/$f" ]] || err "Falta $UPENN_WORKDIR/$f (ejecuta prep_upenn_for_mni + register_upenn_to_mni antes)"
  ln -sfn "$UPENN_WORKDIR/$f" "$OUTDIR/$f"
done

# 4) warp tumor UPENN -> grid HCP via MNI
# ANTs aplica la ULTIMA primero: queremos (HCP->MNI)^-1 o (UPENN->MNI)
# => primero inversas de HCP->MNI, luego forward de UPENN->MNI.
if [[ -f tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz ]]; then
  echo "[SKIP] $SUBJECT_ID: tumor ya warpeado en $OUTDIR"
  exit 0
fi

antsApplyTransforms -d 3 \
  -i tumor14_T1up_brain_RAS.nii.gz \
  -r T1_hcp_RAS.nii.gz \
  -o tumor14_in_HCP_T1grid_viaMNI.nii.gz \
  -n NearestNeighbor \
  -t [hcpT1_2mni_0GenericAffine.mat,1] \
  -t hcpT1_2mni_1InverseWarp.nii.gz \
  -t upT1_2mni_1Warp.nii.gz \
  -t upT1_2mni_0GenericAffine.mat

# 5) mascara cerebro HCP
mrcalc T1_hcp_RAS.nii.gz 0 -gt T1_hcp_brainmask_fromT1_RAS.nii.gz -force
mrcalc tumor14_in_HCP_T1grid_viaMNI.nii.gz \
       T1_hcp_brainmask_fromT1_RAS.nii.gz \
       -mult tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz -force

rm -f T1_hcp_brainmask_fromT1_RAS.nii.gz

echo "[OK] $SUBJECT_ID registrado en $OUTDIR"

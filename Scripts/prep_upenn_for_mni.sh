#!/usr/bin/env bash
set -euo pipefail

# Prepara inputs UPENN para el registro a MNI (parte 1 de 2 del antiguo
# pipeline_original_T1_MNI.sh). Hace SOLO la parte que depende del tumor,
# NO del sujeto HCP — por eso puede ejecutarse UNA sola vez por tumor y
# reutilizarse para los 85 sujetos HCP.
#
# Salidas (dentro de --outdir):
#   T1_upenn_RAS.nii.gz
#   T1_upenn_mask_excl_RAS.nii.gz       (cerebro UPENN sin tumor, para mask ANTs)
#   tumor14_T1up_brain_RAS.nii.gz       (tumor binario labels seleccionados, en grid UPENN)
#   MNI_brain_RAS.nii.gz
#   .prep_done                          (marker de finalización)
#
# Idempotente: si existe .prep_done, no vuelve a calcular.

err(){ echo "[ERROR] $*" >&2; exit 1; }
need(){ command -v "$1" >/dev/null 2>&1 || err "No encuentro '$1' en PATH"; }
need mrconvert; need mrcalc; need maskfilter; need mrtransform; need python3

T1_UPENN=""
TUMOR_SEG_UPENN=""
MNI_BRAIN=""
LABELS="1,4"
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --t1-upenn)          T1_UPENN="$2"; shift 2 ;;
    --tumor-seg-upenn)   TUMOR_SEG_UPENN="$2"; shift 2 ;;
    --mni-brain)         MNI_BRAIN="$2"; shift 2 ;;
    --labels)            LABELS="$2"; shift 2 ;;
    --outdir)            OUTDIR="$2"; shift 2 ;;
    *) err "Argumento desconocido: $1" ;;
  esac
done

[[ -n "$T1_UPENN" && -n "$TUMOR_SEG_UPENN" && -n "$MNI_BRAIN" && -n "$OUTDIR" ]] \
  || err "Requeridos: --t1-upenn, --tumor-seg-upenn, --mni-brain, --outdir"

abs(){ python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1"; }
T1_UPENN=$(abs "$T1_UPENN")
TUMOR_SEG_UPENN=$(abs "$TUMOR_SEG_UPENN")
MNI_BRAIN=$(abs "$MNI_BRAIN")
OUTDIR=$(abs "$OUTDIR")

mkdir -p "$OUTDIR"
cd "$OUTDIR"

if [[ -f .prep_done ]]; then
  echo "[SKIP] UPENN prep ya hecho en $OUTDIR"
  exit 0
fi

cat > .reorient_to_RAS.py <<'PY'
import sys, nibabel as nib
inp, out = sys.argv[1:3]
img = nib.load(inp)
ras = nib.as_closest_canonical(img)
ras.set_qform(ras.affine, code=1); ras.set_sform(ras.affine, code=1)
nib.save(ras, out)
PY

to_nii(){
  local in="$1" out="$2"
  case "$in" in
    *.nii|*.nii.gz) cp -f "$in" "$out" ;;
    *.mif|*.mif.gz) mrconvert -force "$in" "$out" ;;
    *) err "Formato no soportado: $in" ;;
  esac
}

# 0) copiar / convertir
to_nii "$T1_UPENN"         T1_upenn_raw.nii.gz
to_nii "$TUMOR_SEG_UPENN"  tumor_seg_upenn_raw.nii.gz
to_nii "$MNI_BRAIN"        MNI_brain_raw.nii.gz

# 1) reorientar a RAS
python3 .reorient_to_RAS.py T1_upenn_raw.nii.gz        T1_upenn_RAS.nii.gz
python3 .reorient_to_RAS.py tumor_seg_upenn_raw.nii.gz tumor_seg_upenn_RAS.nii.gz
python3 .reorient_to_RAS.py MNI_brain_raw.nii.gz       MNI_brain_RAS.nii.gz

# 2.1) tumor a rejilla T1_UPENN
mrtransform tumor_seg_upenn_RAS.nii.gz -template T1_upenn_RAS.nii.gz -interp nearest -force tumor_seg_on_T1upenn_RAS.nii.gz

# 2.2) cerebro UPENN + exclusion del tumor (mascara para ANTs)
mrcalc T1_upenn_RAS.nii.gz 0 -gt T1_upenn_brain_raw_RAS.nii.gz -force
maskfilter T1_upenn_brain_raw_RAS.nii.gz connect T1_upenn_brain_conn_RAS.nii.gz -force
maskfilter T1_upenn_brain_conn_RAS.nii.gz dilate T1_upenn_brain_RAS.nii.gz -force
mrcalc tumor_seg_on_T1upenn_RAS.nii.gz 0 -eq no_tumor_T1upenn_RAS.nii.gz -force
mrcalc T1_upenn_brain_RAS.nii.gz no_tumor_T1upenn_RAS.nii.gz -mult T1_upenn_mask_excl_RAS.nii.gz -force

# 5) extraer labels seleccionados y mascarar con cerebro UPENN
IFS=',' read -r -a LABARR <<< "$LABELS"
[[ ${#LABARR[@]} -ge 1 ]] || err "--labels vacio"
mrcalc tumor_seg_on_T1upenn_RAS.nii.gz "${LABARR[0]}" -eq tumor14_T1up_RAS.nii.gz -force
for ((i=1; i<${#LABARR[@]}; i++)); do
  mrcalc tumor_seg_on_T1upenn_RAS.nii.gz "${LABARR[$i]}" -eq _tumor_tmp.nii.gz -force
  mrcalc tumor14_T1up_RAS.nii.gz _tumor_tmp.nii.gz -max _tumor14_tmp.nii.gz -force
  mv -f _tumor14_tmp.nii.gz tumor14_T1up_RAS.nii.gz
done
rm -f _tumor_tmp.nii.gz
mrcalc tumor14_T1up_RAS.nii.gz T1_upenn_brain_RAS.nii.gz -mult tumor14_T1up_brain_RAS.nii.gz -force

# cleanup intermedios
rm -f T1_upenn_brain_raw_RAS.nii.gz T1_upenn_brain_conn_RAS.nii.gz
rm -f T1_upenn_brain_RAS.nii.gz
rm -f no_tumor_T1upenn_RAS.nii.gz
rm -f tumor14_T1up_RAS.nii.gz
rm -f tumor_seg_upenn_raw.nii.gz tumor_seg_upenn_RAS.nii.gz tumor_seg_on_T1upenn_RAS.nii.gz
rm -f T1_upenn_raw.nii.gz MNI_brain_raw.nii.gz
rm -f .reorient_to_RAS.py

touch .prep_done
echo "[OK] UPENN prep en $OUTDIR"

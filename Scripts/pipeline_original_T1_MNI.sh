#!/usr/bin/env bash
set -xeuo pipefail

# ============================
# USO (ejemplo):
# ./pipeline_t1_via_mni_RAS.sh \
#   --t1-upenn /abs/UPENN_T1.nii.gz \
#   --t1-hcp /abs/HCP/T1w_acpc_dc_restore_brain.nii.gz \
#   --tumor-seg-upenn /abs/UPENN_tumor_seg.nii.gz \
#   --mni-brain /abs/MNI152_T1_1mm_brain.nii.gz \
#   --outdir /abs/salida
# ============================

err(){ echo "[ERROR] $*" >&2; exit 1; }
need(){ command -v "$1" >/dev/null 2>&1 || err "No encuentro '$1' en PATH"; }

# Dependencias
need mrconvert; need mrcalc; need maskfilter; need mrtransform
need antsRegistration; need antsApplyTransforms; need python3

# Parse args
T1_UPENN=""; T1_HCP=""; TUMOR_SEG_UPENN=""; MNI_BRAIN=""
OUTDIR="out"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --t1-upenn) T1_UPENN="$2"; shift 2 ;;
    --t1-hcp) T1_HCP="$2"; shift 2 ;;
    --tumor-seg-upenn) TUMOR_SEG_UPENN="$2"; shift 2 ;;
    --mni-brain) MNI_BRAIN="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    *) err "Argumento desconocido: $1" ;;
  esac
done

[[ -n "$T1_UPENN" && -n "$T1_HCP" && -n "$TUMOR_SEG_UPENN" && -n "$MNI_BRAIN" ]] \
  || err "Faltan argumentos. Revisa el bloque USO."

# Rutas absolutas (para no romperse tras cd)
abs(){ python3 - "$1" <<'PY'
import os,sys; print(os.path.abspath(sys.argv[1]))
PY
}
T1_UPENN=$(abs "$T1_UPENN")
T1_HCP=$(abs "$T1_HCP")
TUMOR_SEG_UPENN=$(abs "$TUMOR_SEG_UPENN")
MNI_BRAIN=$(abs "$MNI_BRAIN")
OUTDIR=$(abs "$OUTDIR")

mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo "[INFO] Args:
  T1_UPENN=$T1_UPENN
  T1_HCP=$T1_HCP
  TUMOR_SEG_UPENN=$TUMOR_SEG_UPENN
  MNI_BRAIN=$MNI_BRAIN
  OUTDIR=$(pwd)
"

# Convertir a NIfTI sin tocar strides
to_nii(){
  local in="$1" out="$2"
  case "$in" in
    *.nii|*.nii.gz) cp -f "$in" "$out" ;;
    *.mif|*.mif.gz) mrconvert -force "$in" "$out" ;;  # no -strides aquí
    *) err "Formato no soportado: $in" ;;
  esac
}

# --- Helper Python: reorientar a RAS preservando dtype ---
cat > reorient_to_RAS.py <<'PY'
import sys, nibabel as nib
inp, out = sys.argv[1:3]
img = nib.load(inp)
ras = nib.as_closest_canonical(img)  # preserva dtype, no fuerza float
ras.set_qform(ras.affine, code=1); ras.set_sform(ras.affine, code=1)
nib.save(ras, out); print(out)
PY

# ---------- 0) Copiar/convertir ----------
to_nii "$T1_UPENN"            T1_upenn_raw.nii.gz
to_nii "$T1_HCP"              T1_hcp_raw.nii.gz
to_nii "$TUMOR_SEG_UPENN"     tumor_seg_upenn_raw.nii.gz
to_nii "$MNI_BRAIN"           MNI_brain_raw.nii.gz

# ---------- 1) Reorientar todo a RAS ----------
python3 reorient_to_RAS.py T1_upenn_raw.nii.gz        T1_upenn_RAS.nii.gz
python3 reorient_to_RAS.py T1_hcp_raw.nii.gz          T1_hcp_RAS.nii.gz
python3 reorient_to_RAS.py tumor_seg_upenn_raw.nii.gz tumor_seg_upenn_RAS.nii.gz
python3 reorient_to_RAS.py MNI_brain_raw.nii.gz       MNI_brain_RAS.nii.gz

# ---------- 2) Preparar máscaras ----------
# 2.1 Tumor a rejilla del T1_UPENN_RAS
mrtransform tumor_seg_upenn_RAS.nii.gz -template T1_upenn_RAS.nii.gz -interp nearest tumor_seg_on_T1upenn_RAS.nii.gz

# 2.2 UPENN: cerebro y exclusión del tumor (para registro UPENN→MNI)
mrcalc T1_upenn_RAS.nii.gz 0 -gt T1_upenn_brain_raw_RAS.nii.gz
maskfilter T1_upenn_brain_raw_RAS.nii.gz connect T1_upenn_brain_conn_RAS.nii.gz
maskfilter T1_upenn_brain_conn_RAS.nii.gz dilate  T1_upenn_brain_RAS.nii.gz

mrcalc tumor_seg_on_T1upenn_RAS.nii.gz 0 -eq no_tumor_T1upenn_RAS.nii.gz
mrcalc T1_upenn_brain_RAS.nii.gz no_tumor_T1upenn_RAS.nii.gz -mult T1_upenn_mask_excl_RAS.nii.gz
# (T1_upenn_mask_excl_RAS = cerebro SIN tumor)

# 2.3 HCP: cerebro (T1_HCP ya suele venir skull-stripped; aún así creamos una 0/1 sólida)
mrcalc T1_hcp_RAS.nii.gz 0 -gt T1_hcp_brain_raw_RAS.nii.gz
maskfilter T1_hcp_brain_raw_RAS.nii.gz connect T1_hcp_brain_conn_RAS.nii.gz
maskfilter T1_hcp_brain_conn_RAS.nii.gz dilate  T1_hcp_brain_RAS.nii.gz

# ---------- 3) Registro UPENN_T1 → MNI (excluyendo tumor en moving) ----------
# Métrica: MI para rígido/afín (robusta inter-escáner), CC en SyN.
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

# ---------- 4) Registro HCP_T1 → MNI ----------
antsRegistration -d 3 --float 1 --use-histogram-matching 1 -v 1 \
  -r [MNI_brain_RAS.nii.gz,T1_hcp_RAS.nii.gz,1] \
  -x [NULL,T1_hcp_brain_RAS.nii.gz] \
  \
  -m MI[MNI_brain_RAS.nii.gz,T1_hcp_RAS.nii.gz,1,32] \
  -t Rigid[0.1]  -c [2000x1000x500,1e-6,10] -f 8x4x2 -s 3x2x1vox \
  \
  -m MI[MNI_brain_RAS.nii.gz,T1_hcp_RAS.nii.gz,1,32] \
  -t Affine[0.1] -c [2000x1000x500,1e-6,10] -f 8x4x2 -s 3x2x1vox \
  \
  -m CC[MNI_brain_RAS.nii.gz,T1_hcp_RAS.nii.gz,1,4] \
  -t SyN[0.1,3,0] -c [500x250x100,1e-6,10] -f 8x4x2 -s 3x2x1vox \
  \
  -o [hcpT1_2mni_, hcpT1_in_MNI.nii.gz]

# ---------- 5) Extraer labels 1 y 4 del tumor (en T1_UPENN_RAS) ----------
mrcalc tumor_seg_on_T1upenn_RAS.nii.gz 1 -eq tumor1_T1up_RAS.nii.gz
mrcalc tumor_seg_on_T1upenn_RAS.nii.gz 4 -eq tumor4_T1up_RAS.nii.gz
mrcalc tumor1_T1up_RAS.nii.gz tumor4_T1up_RAS.nii.gz -max tumor14_T1up_RAS.nii.gz
mrcalc tumor14_T1up_RAS.nii.gz T1_upenn_brain_RAS.nii.gz -mult tumor14_T1up_brain_RAS.nii.gz

# ---------- 6) Componer UPENN→HCP vía MNI y warpear máscara (rejilla T1_HCP) ----------
# ANTs aplica la ÚLTIMA primero.
# Queremos (HCP→MNI)^-1 ∘ (UPENN→MNI)
# => listamos primero las inversas de HCP→MNI, luego los forward de UPENN→MNI.
antsApplyTransforms -d 3 \
  -i tumor14_T1up_brain_RAS.nii.gz \
  -r T1_hcp_RAS.nii.gz \
  -o tumor14_in_HCP_T1grid_viaMNI.nii.gz \
  -n NearestNeighbor \
  -t [hcpT1_2mni_0GenericAffine.mat,1] \
  -t hcpT1_2mni_1InverseWarp.nii.gz \
  -t upT1_2mni_1Warp.nii.gz \
  -t upT1_2mni_0GenericAffine.mat

# ---------- 7) (Opcional) Limitar a cerebro HCP (rejilla T1_HCP) ----------
# Si tu T1_HCP ya es "brain", esto suele ser redundante; aún así, por seguridad:
mrcalc T1_hcp_RAS.nii.gz 0 -gt T1_hcp_brainmask_fromT1_RAS.nii.gz
mrcalc tumor14_in_HCP_T1grid_viaMNI.nii.gz \
       T1_hcp_brainmask_fromT1_RAS.nii.gz \
       -mult tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz

# ---------- 8) Salidas ----------
ls -1 \
  upT1_2mni_0GenericAffine.mat upT1_2mni_1Warp.nii.gz upT1_2mni_1InverseWarp.nii.gz \
  hcpT1_2mni_0GenericAffine.mat hcpT1_2mni_1Warp.nii.gz hcpT1_2mni_1InverseWarp.nii.gz \
  tumor14_in_HCP_T1grid_viaMNI.nii.gz \
  tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz \
  upT1_in_MNI.nii.gz hcpT1_in_MNI.nii.gz \
  MNI_brain_RAS.nii.gz

echo "✅ Pipeline T1 vía MNI completo."


#!/usr/bin/env bash
set -euo pipefail

# Asegura una entrada en el cache compartido HCP->MNI para UN sujeto.
#
# El registro HCP->MNI depende SOLO del sujeto HCP y del template MNI —
# NO del tumor UPENN. Por tanto puede calcularse una sola vez por sujeto
# y reutilizarse para todos los tumores del experimento.
#
# Layout del cache:
#   $CACHE_ROOT/<SUBJECT_ID>/
#     T1_hcp_RAS.nii.gz
#     T1_hcp_RAS.mif                    (mrconvert pre-hecho para el optimizer)
#     T1_hcp_brain_RAS.nii.gz
#     MNI_brain_RAS.nii.gz
#     hcpT1_2mni_0GenericAffine.mat
#     hcpT1_2mni_1Warp.nii.gz
#     hcpT1_2mni_1InverseWarp.nii.gz
#     hcpT1_in_MNI.nii.gz
#     .cache_done                       (marker final; presencia => cache completo)
#
# Idempotente y seguro entre procesos concurrentes:
# - Doble comprobacion del marker (antes y despues del flock).
# - Calculo en tmp dir; publicacion por rename atomico.

err(){ echo "[ERROR] $*" >&2; exit 1; }
need(){ command -v "$1" >/dev/null 2>&1 || err "No encuentro '$1' en PATH"; }
need mrconvert; need mrcalc; need maskfilter
need antsRegistration; need python3; need flock

SUBJECT_ID=""
T1_HCP=""
MNI_BRAIN=""
CACHE_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject-id) SUBJECT_ID="$2"; shift 2 ;;
    --t1-hcp)     T1_HCP="$2"; shift 2 ;;
    --mni-brain)  MNI_BRAIN="$2"; shift 2 ;;
    --cache-root) CACHE_ROOT="$2"; shift 2 ;;
    *) err "Argumento desconocido: $1" ;;
  esac
done
[[ -n "$SUBJECT_ID" && -n "$T1_HCP" && -n "$MNI_BRAIN" && -n "$CACHE_ROOT" ]] \
  || err "Requeridos: --subject-id --t1-hcp --mni-brain --cache-root"

abs(){ python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1"; }
T1_HCP=$(abs "$T1_HCP")
MNI_BRAIN=$(abs "$MNI_BRAIN")
CACHE_ROOT=$(abs "$CACHE_ROOT")

# Crea el cache si no existe (detectar+crear pedido por el usuario)
mkdir -p "$CACHE_ROOT" "$CACHE_ROOT/.locks"

TARGET="$CACHE_ROOT/$SUBJECT_ID"

REQUIRED_FILES=(
  T1_hcp_RAS.nii.gz
  T1_hcp_RAS.mif
  T1_hcp_brain_RAS.nii.gz
  MNI_brain_RAS.nii.gz
  hcpT1_2mni_0GenericAffine.mat
  hcpT1_2mni_1Warp.nii.gz
  hcpT1_2mni_1InverseWarp.nii.gz
  hcpT1_in_MNI.nii.gz
)

cache_complete(){
  local base="$1"
  [[ -f "$base/.cache_done" ]] || return 1
  local f
  for f in "${REQUIRED_FILES[@]}"; do
    [[ -f "$base/$f" ]] || return 1
  done
  return 0
}

if cache_complete "$TARGET"; then
  echo "[CACHE-HIT] $SUBJECT_ID ($TARGET)"
  exit 0
fi

LOCKFILE="$CACHE_ROOT/.locks/$SUBJECT_ID.lock"
: > "$LOCKFILE"

# Lock por sujeto (bloqueante). Exit cierra el fd => libera el lock.
exec 9>"$LOCKFILE"
flock -x 9

# Re-chequeo dentro del lock (otro job puede haberlo rellenado mientras esperabamos)
if cache_complete "$TARGET"; then
  echo "[CACHE-HIT] $SUBJECT_ID (tras lock) ($TARGET)"
  exit 0
fi

# Construccion en tmp dir
TMPDIR="$TARGET.tmp.$$"
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"

cat > "$TMPDIR/.reorient_to_RAS.py" <<'PY'
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

(
  cd "$TMPDIR"

  # 0) copiar/convertir
  to_nii "$T1_HCP"    T1_hcp_raw.nii.gz
  to_nii "$MNI_BRAIN" MNI_brain_raw.nii.gz

  # 1) reorientar a RAS
  python3 .reorient_to_RAS.py T1_hcp_raw.nii.gz    T1_hcp_RAS.nii.gz
  python3 .reorient_to_RAS.py MNI_brain_raw.nii.gz MNI_brain_RAS.nii.gz

  # 2.3) mascara cerebro HCP
  mrcalc T1_hcp_RAS.nii.gz 0 -gt T1_hcp_brain_raw_RAS.nii.gz -force
  maskfilter T1_hcp_brain_raw_RAS.nii.gz connect T1_hcp_brain_conn_RAS.nii.gz -force
  maskfilter T1_hcp_brain_conn_RAS.nii.gz dilate T1_hcp_brain_RAS.nii.gz -force

  # 4) registro HCP -> MNI
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

  # mrconvert -> .mif (antes era step 2 del pipeline; lo precomputamos aqui)
  mrconvert T1_hcp_RAS.nii.gz T1_hcp_RAS.mif -force

  # cleanup
  rm -f T1_hcp_raw.nii.gz MNI_brain_raw.nii.gz
  rm -f T1_hcp_brain_raw_RAS.nii.gz T1_hcp_brain_conn_RAS.nii.gz
  rm -f .reorient_to_RAS.py
)

# Marker final SOLO cuando todo esta escrito
touch "$TMPDIR/.cache_done"

# Publicacion atomica
if [[ -d "$TARGET" ]]; then
  # residuo de un run anterior que se quedo a medias (sin .cache_done)
  rm -rf "$TARGET"
fi
mv "$TMPDIR" "$TARGET"

echo "[CACHE-FILL] $SUBJECT_ID -> $TARGET"

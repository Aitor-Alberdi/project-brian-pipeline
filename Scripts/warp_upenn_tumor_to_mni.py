#!/usr/bin/env python3
"""warp_upenn_tumor_to_mni.py

Crea la máscara de tumor en MNI (UPENN -> MNI) usando los warps ya calculados
en tus carpetas de registro.

Notas:
- La máscara de tumor en UPENN space puede NO ser binaria.
- Por defecto, este script replica tu lógica típica: conservar SOLO las
  etiquetas {1,4} y binarizar (0/1) antes del warp.
- Si quieres usar otra selección, cambia --labels.

Estructura esperada (por sujeto dentro de --reg-root/<ID>/):
  - upT1_2mni_0GenericAffine.mat
  - upT1_2mni_1Warp.nii.gz
  - (opcional) MNI_brain_RAS.nii.gz / MNI_brain_raw.nii.gz / upT1_in_MNI.nii.gz
  - tumor_seg_on_T1upenn_RAS.nii.gz (preferido) o tumor_seg_upenn_RAS.nii.gz

Pipeline por sujeto:
  1) (opcional) selecciona etiquetas (por defecto 1,4) y binariza (0/1).
  2) antsApplyTransforms con interpolación NearestNeighbor:
        -t upT1_2mni_1Warp.nii.gz
        -t upT1_2mni_0GenericAffine.mat
     para obtener tumor_seg_MNI.nii.gz

Ejemplos:
  python3 warp_upenn_tumor_to_mni.py \
    --reg-root /home/Archive/BRIAN/Pipeline/registration_results_20_2 \
    --out-dir  /home/Archive/BRIAN/Pipeline/tumor_mni_training

  python3 warp_upenn_tumor_to_mni.py \
    --reg-root /home/Archive/BRIAN/Pipeline/registration_results_65 \
    --out-dir  /home/Archive/BRIAN/Pipeline/tumor_mni_validation \
    --ref-mni  /home/Archive/BRIAN/Pipeline/registration_results_65/123117/MNI_brain_RAS.nii.gz

Requisitos:
  - antsApplyTransforms (ANTs) en PATH
  - Python: nibabel, numpy, pandas
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib


TUMOR_CANDIDATES_DEFAULT = [
    "tumor_seg_on_T1upenn_RAS.nii.gz",   # preferido: seg en grid T1 UPENN
    "tumor_seg_upenn_RAS.nii.gz",
    "tumor_seg_upenn_raw.nii.gz",
    "tumor_seg_upenn_raw_RAS.nii.gz",
]

MNI_REF_CANDIDATES_DEFAULT = [
    "MNI_brain_RAS.nii.gz",
    "MNI_brain_raw.nii.gz",
    "upT1_in_MNI.nii.gz",
    "hcpT1_in_MNI.nii.gz",
]


def _run(cmd: List[str], dry_run: bool = False) -> None:
    if dry_run:
        print("[DRY-RUN]", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def _pick_first_existing(base: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = base / n
        if p.exists():
            return p
    return None


def _select_and_binarize_mask(in_path: Path, out_path: Path, labels: List[int]) -> None:
    """Selecciona labels (p.ej. [1,4]) y genera máscara binaria 0/1.

    Si labels está vacío, usa data>0.
    """
    img = nib.load(str(in_path))
    data = img.get_fdata(dtype=np.float32)
    if labels:
        bin_data = np.isin(data, np.array(labels, dtype=np.float32)).astype(np.uint8)
    else:
        bin_data = (data > 0).astype(np.uint8)

    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    try:
        hdr["scl_slope"] = 1
        hdr["scl_inter"] = 0
    except Exception:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(bin_data, img.affine, header=hdr), str(out_path))


def _warp_upenn_to_mni(
    in_mask: Path,
    ref_mni: Path,
    warp_fwd: Path,
    affine_fwd: Path,
    out_mask: Path,
    interp: str = "NearestNeighbor",
    dry_run: bool = False,
) -> None:
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "antsApplyTransforms",
        "-d", "3",
        "-i", str(in_mask),
        "-r", str(ref_mni),
        "-o", str(out_mask),
        "-n", interp,
        "-t", str(warp_fwd),
        "-t", str(affine_fwd),
    ]
    _run(cmd, dry_run=dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Warpea la máscara de tumor UPENN a MNI usando los transforms upT1_2mni por sujeto."
    )
    ap.add_argument("--reg-root", required=True, type=Path,
                    help="Raíz de registration_results_20_2 o registration_results_65")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Directorio de salida. Se crearán subcarpetas por sujeto.")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="CSV resumen (por defecto: <out-dir>/tumor_upenn_to_mni_summary.csv)")
    ap.add_argument("--ref-mni", type=Path, default=None,
                    help="Imagen MNI de referencia (grid) global. Si no se da, se autodetecta por sujeto.")
    ap.add_argument("--tumor-candidates", type=str, default=",".join(TUMOR_CANDIDATES_DEFAULT),
                    help="Lista separada por comas con nombres candidatos del tumor en UPENN.")
    ap.add_argument("--mni-ref-candidates", type=str, default=",".join(MNI_REF_CANDIDATES_DEFAULT),
                    help="Lista separada por comas con nombres candidatos de referencia MNI dentro del reg dir.")

    ap.add_argument(
        "--input-mask",
        type=Path,
        default=None,
        help=(
            "Opcional: máscara de tumor en espacio UPENN (puede ser multi-label). "
            "Si se especifica, se usa esta máscara en lugar de buscar tumor_seg_* dentro de cada sujeto."
        ),
    )

    ap.add_argument(
        "--labels",
        type=str,
        default="1,4",
        help="Etiquetas a conservar antes de binarizar (por defecto '1,4'). Usa '' para (data>0).",
    )

    ap.add_argument(
        "--single-subject",
        type=str,
        default=None,
        help="Procesa solo este sujeto (ID). Útil si solo quieres un tumor en MNI usando cualquier warp.",
    )

    ap.add_argument(
        "--binarize-input",
        action="store_true",
        default=True,
        help="Selecciona labels y binariza antes de warpear. (default: True)",
    )
    ap.add_argument("--no-binarize-input", dest="binarize_input", action="store_false")
    ap.add_argument("--interp", type=str, default="NearestNeighbor",
                    help="Interpolación para antsApplyTransforms (default: NearestNeighbor).")
    ap.add_argument("--force", action="store_true",
                    help="Sobrescribe outputs existentes.")
    ap.add_argument("--dry-run", action="store_true",
                    help="No ejecuta ANTs; solo imprime comandos.")

    args = ap.parse_args()

    reg_root: Path = args.reg_root
    out_dir: Path = args.out_dir

    if not reg_root.exists():
        raise SystemExit(f"--reg-root no existe: {reg_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    tumor_candidates = [s.strip() for s in args.tumor_candidates.split(",") if s.strip()]
    mni_ref_candidates = [s.strip() for s in args.mni_ref_candidates.split(",") if s.strip()]

    labels: List[int] = []
    if args.labels is not None and str(args.labels).strip() != "":
        labels = [int(x.strip()) for x in str(args.labels).split(",") if x.strip() != ""]

    if args.ref_mni is not None and not args.ref_mni.exists():
        raise SystemExit(f"--ref-mni no existe: {args.ref_mni}")

    rows: List[Dict[str, object]] = []
    included: List[str] = []
    excluded: List[str] = []

    subj_dirs = [p for p in reg_root.iterdir() if p.is_dir()]
    subj_dirs.sort(key=lambda p: p.name)
    if args.single_subject is not None:
        subj_dirs = [p for p in subj_dirs if p.name == str(args.single_subject)]

    for sd in subj_dirs:
        sid = sd.name
        warp = sd / "upT1_2mni_1Warp.nii.gz"
        aff = sd / "upT1_2mni_0GenericAffine.mat"

        if args.input_mask is not None:
            tumor_in = Path(args.input_mask)
        else:
            tumor_in = _pick_first_existing(sd, tumor_candidates)
        if args.ref_mni is not None:
            ref_mni = args.ref_mni
        else:
            ref_mni = _pick_first_existing(sd, mni_ref_candidates)

        out_sub = out_dir / sid
        out_sub.mkdir(parents=True, exist_ok=True)

        tumor_bin = out_sub / "tumor_seg_upenn_sel_bin.nii.gz"
        tumor_mni = out_sub / "tumor_seg_MNI.nii.gz"

        status = "OK"
        err_msg = ""

        # Skip if already exists (unless --force)
        if tumor_mni.exists() and not args.force:
            rows.append({
                "subject_id": sid,
                "status": "SKIP_EXISTS",
                "reg_dir": str(sd),
                "tumor_in": str(tumor_in) if tumor_in else None,
                "tumor_bin": str(tumor_bin),
                "ref_mni": str(ref_mni) if ref_mni else None,
                "warp": str(warp),
                "affine": str(aff),
                "tumor_mni": str(tumor_mni),
                "error": "",
            })
            included.append(sid)
            continue

        # Validate requirements
        if tumor_in is None or (not tumor_in.exists()):
            status = "FAIL"
            err_msg = "No se encontró tumor segmentation (ni --input-mask válido ni tumor candidates en el directorio)."
        elif not warp.exists():
            status = "FAIL"
            err_msg = f"No existe warp: {warp.name}"
        elif not aff.exists():
            status = "FAIL"
            err_msg = f"No existe affine: {aff.name}"
        elif ref_mni is None or not ref_mni.exists():
            status = "FAIL"
            err_msg = "No se encontró imagen de referencia MNI (ref) (ni --ref-mni ni candidatos)."

        try:
            if status != "FAIL":
                in_for_warp = tumor_in
                if args.binarize_input:
                    _select_and_binarize_mask(tumor_in, tumor_bin, labels=labels)
                    in_for_warp = tumor_bin
                else:
                    # si no binarizas, igual guardamos un symlink-like info
                    if not tumor_bin.exists():
                        # no creamos archivo, solo dejamos path vacío
                        pass

                _warp_upenn_to_mni(
                    in_mask=in_for_warp,
                    ref_mni=ref_mni,
                    warp_fwd=warp,
                    affine_fwd=aff,
                    out_mask=tumor_mni,
                    interp=args.interp,
                    dry_run=args.dry_run,
                )

        except subprocess.CalledProcessError as e:
            status = "FAIL"
            err_msg = f"ANTs error: {e}"
        except Exception as e:
            status = "FAIL"
            err_msg = str(e)

        rows.append({
            "subject_id": sid,
            "status": status,
            "reg_dir": str(sd),
            "tumor_in": str(tumor_in) if tumor_in else None,
            "tumor_bin": str(tumor_bin) if args.binarize_input else None,
            "ref_mni": str(ref_mni) if ref_mni else None,
            "warp": str(warp),
            "affine": str(aff),
            "tumor_mni": str(tumor_mni),
            "error": err_msg,
        })

        if status == "OK" or status == "SKIP_EXISTS":
            included.append(sid)
        else:
            excluded.append(sid)
            print(f"[WARN] {sid}: {err_msg}")

    df = pd.DataFrame(rows)
    out_csv = args.out_csv if args.out_csv is not None else (out_dir / "tumor_upenn_to_mni_summary.csv")
    df.to_csv(out_csv, index=False)
    (out_dir / "included.txt").write_text("\n".join(included) + ("\n" if included else ""))
    (out_dir / "excluded.txt").write_text("\n".join(excluded) + ("\n" if excluded else ""))

    print(f"[OK] Resumen CSV: {out_csv}")
    print(f"[OK] Sujetos OK/SKIP: {len(included)} | FAIL: {len(excluded)}")


if __name__ == "__main__":
    main()

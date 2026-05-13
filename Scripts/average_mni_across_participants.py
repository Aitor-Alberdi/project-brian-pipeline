#!/usr/bin/env python3
"""
average_mni_across_participants.py

Compute an average (frequency/probability) image across participants for:
  - training:  */best_bin_MNI.nii[.gz]
  - validation: */analysis_bin_MNI.nii[.gz]

Typical usage:
  python3 average_mni_across_participants.py \
    --train-root /home/Archive/BRIAN/Pipeline/best_mni_outputs \
    --val-root   /home/Archive/BRIAN/Pipeline/validation_mni_outputs \
    --out-dir    /home/Archive/BRIAN/Pipeline/avg_mni_outputs

Outputs (in --out-dir):
  training_avg.nii.gz        float32 average mask (0..1)
  training_sum.nii.gz        float32 voxelwise sum (0..N)
  training_majority.nii.gz   uint8 majority vote (avg>=0.5)
  training_included.txt      participant IDs included
  training_excluded.txt      files excluded + reason

  validation_avg.nii.gz
  validation_sum.nii.gz
  validation_majority.nii.gz
  validation_included.txt
  validation_excluded.txt

Notes:
- Assumes masks are already in MNI and (mostly) on the same grid.
- If a mask grid differs (shape/affine), it will be resampled to the reference grid.
  For masks, resampling uses nearest-neighbour (order=0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import nibabel as nib

try:
    from nibabel.processing import resample_from_to
except Exception as e:
    resample_from_to = None  # type: ignore


def _log(msg: str) -> None:
    print(msg, flush=True)


def _is_same_grid(img: nib.Nifti1Image, ref: nib.Nifti1Image, atol: float = 1e-5) -> bool:
    if img.shape != ref.shape:
        return False
    return np.allclose(img.affine, ref.affine, atol=atol)


def _load_and_resample_to_ref(
    img_path: Path,
    ref_img: nib.Nifti1Image,
    force_binary: bool,
) -> Tuple[np.ndarray, str]:
    """
    Load an image and return data on ref grid.

    Returns (data, status_string). Raises on hard failures.
    """
    img = nib.load(str(img_path))

    if _is_same_grid(img, ref_img):
        data = img.get_fdata(dtype=np.float32)
        status = "ok_same_grid"
    else:
        if resample_from_to is None:
            raise RuntimeError(
                "Grid mismatch but nibabel.processing.resample_from_to is unavailable. "
                "Install a newer nibabel/scipy or provide a matching --ref."
            )
        # nearest neighbour (mask)
        img_rs = resample_from_to(img, ref_img, order=0)
        data = img_rs.get_fdata(dtype=np.float32)
        status = "ok_resampled"

    # Enforce binary if requested
    if force_binary:
        data = (data > 0.5).astype(np.float32)

    # Sanity: replace NaNs
    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        status += "_nanfixed"

    return data, status


def _save_nifti_like(
    out_path: Path,
    data: np.ndarray,
    ref_img: nib.Nifti1Image,
    dtype,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(dtype)
    try:
        hdr["scl_slope"] = 1
        hdr["scl_inter"] = 0
    except Exception:
        pass
    img_out = nib.Nifti1Image(data.astype(dtype, copy=False), ref_img.affine, header=hdr)
    nib.save(img_out, str(out_path))


def _write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def average_dataset(
    root: Path,
    glob_pattern: str,
    out_prefix: str,
    out_dir: Path,
    ref_path: Optional[Path],
    force_binary: bool,
    majority_threshold: float,
) -> Dict[str, Path]:
    """
    Compute sum/avg/majority for one dataset.
    Returns dict of output paths.
    """
    files = sorted(root.glob(glob_pattern))
    if not files:
        raise SystemExit(f"[ERROR] No files found under {root} with pattern: {glob_pattern}")

    # pick reference
    if ref_path is not None:
        ref_img = nib.load(str(ref_path))
        _log(f"[INFO] {out_prefix}: Using explicit reference: {ref_path}")
    else:
        ref_img = nib.load(str(files[0]))
        _log(f"[INFO] {out_prefix}: Using first image as reference: {files[0]}")

    sum_data = np.zeros(ref_img.shape, dtype=np.float64)
    included_ids: List[str] = []
    excluded: List[str] = []
    statuses: Dict[str, int] = {}

    for p in files:
        # participant id is the parent folder name by convention
        pid = p.parent.name
        try:
            data, status = _load_and_resample_to_ref(p, ref_img, force_binary=force_binary)
            sum_data += data.astype(np.float64, copy=False)
            included_ids.append(pid)
            statuses[status] = statuses.get(status, 0) + 1
        except Exception as e:
            excluded.append(f"{pid}\t{p}\t{type(e).__name__}: {e}")

    n = len(included_ids)
    if n == 0:
        raise SystemExit(f"[ERROR] {out_prefix}: All inputs failed to load/resample.")

    avg_data = (sum_data / float(n)).astype(np.float32)
    maj_data = (avg_data >= float(majority_threshold)).astype(np.uint8)

    # outputs
    out_sum = out_dir / f"{out_prefix}_sum.nii.gz"
    out_avg = out_dir / f"{out_prefix}_avg.nii.gz"
    out_maj = out_dir / f"{out_prefix}_majority.nii.gz"
    out_incl = out_dir / f"{out_prefix}_included.txt"
    out_excl = out_dir / f"{out_prefix}_excluded.txt"

    _save_nifti_like(out_sum, sum_data.astype(np.float32), ref_img, np.float32)
    _save_nifti_like(out_avg, avg_data, ref_img, np.float32)
    _save_nifti_like(out_maj, maj_data, ref_img, np.uint8)

    _write_lines(out_incl, included_ids)
    _write_lines(out_excl, excluded)

    _log(f"[OK] {out_prefix}: included={n}/{len(files)} excluded={len(excluded)}")
    _log(f"[OK] {out_prefix}: wrote {out_avg.name}, {out_sum.name}, {out_maj.name}")

    # Status summary
    if statuses:
        st = ", ".join([f"{k}={v}" for k, v in sorted(statuses.items(), key=lambda kv: (-kv[1], kv[0]))])
        _log(f"[INFO] {out_prefix}: status_counts: {st}")

    return {
        "avg": out_avg,
        "sum": out_sum,
        "majority": out_maj,
        "included": out_incl,
        "excluded": out_excl,
    }


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Average MNI-space masks across participants for training and validation.")
    ap.add_argument("--train-root", type=Path, required=True, help="Root dir with per-subject folders containing best_bin_MNI.nii.gz")
    ap.add_argument("--val-root", type=Path, required=True, help="Root dir with per-subject folders containing analysis_bin_MNI.nii.gz")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for average/sum/majority images")

    ap.add_argument("--train-glob", type=str, default="*/best_bin_MNI.nii.gz", help="Glob under train-root to find MNI masks")
    ap.add_argument("--val-glob", type=str, default="*/analysis_bin_MNI.nii.gz", help="Glob under val-root to find MNI masks")

    ap.add_argument("--ref", type=Path, default=None,
                    help="Optional explicit MNI reference NIfTI. If provided, both datasets are resampled to this grid.")
    ap.add_argument("--force-binary", action="store_true", default=True,
                    help="Binarize each input as (data>0.5) before averaging (default: on).")
    ap.add_argument("--no-force-binary", dest="force_binary", action="store_false",
                    help="Do not force binarization; average raw values.")
    ap.add_argument("--majority-threshold", type=float, default=0.5,
                    help="Threshold on the average map to create majority mask (default: 0.5).")

    args = ap.parse_args(argv)

    if not args.train_root.exists():
        raise SystemExit(f"[ERROR] train-root not found: {args.train_root}")
    if not args.val_root.exists():
        raise SystemExit(f"[ERROR] val-root not found: {args.val_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref = args.ref if args.ref is not None else None
    if ref is not None and not ref.exists():
        raise SystemExit(f"[ERROR] --ref not found: {ref}")

    _log("[INFO] Averaging TRAINING dataset")
    average_dataset(
        root=args.train_root,
        glob_pattern=args.train_glob,
        out_prefix="training",
        out_dir=args.out_dir,
        ref_path=ref,
        force_binary=args.force_binary,
        majority_threshold=args.majority_threshold,
    )

    _log("[INFO] Averaging VALIDATION dataset")
    average_dataset(
        root=args.val_root,
        glob_pattern=args.val_glob,
        out_prefix="validation",
        out_dir=args.out_dir,
        ref_path=ref,
        force_binary=args.force_binary,
        majority_threshold=args.majority_threshold,
    )

    _log(f"[DONE] Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()

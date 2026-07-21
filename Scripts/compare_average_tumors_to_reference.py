#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt

try:
    from nibabel.processing import resample_from_to
except Exception:
    resample_from_to = None

# Curriculum C (para reportar objective de forma consistente con validation)
EPSILON = {
    "dice_loss": 0.45,
    "sd_loss": 0.55 * (0.9 ** 2),
    "vol_err": 0.15 * (0.9 ** 2),
}
LAMBDA_VIOL = 2.0
HD_DMAX_MM = 20.0
SURF_TOL_MM = 2.0
W_HD = 0.20
W_DICE = 0.60
W_PROJ = 0.20
USE_ASSD_INSTEAD_OF_HD95 = False


def _ensure_bool(a: np.ndarray) -> np.ndarray:
    return a if a.dtype == bool else a.astype(bool)


def _surface_mask(bin_mask: np.ndarray) -> np.ndarray:
    b = _ensure_bool(bin_mask)
    er = binary_erosion(b, iterations=1, border_value=0)
    return b ^ er


def dice_loss(pred: np.ndarray, gt: np.ndarray) -> float:
    p = _ensure_bool(pred)
    g = _ensure_bool(gt)
    inter = np.logical_and(p, g).sum(dtype=np.float64)
    a = p.sum(dtype=np.float64)
    b = g.sum(dtype=np.float64)
    return float(1.0 - (2.0 * inter) / (a + b + 1e-10))


def _surface_distances(bin_A: np.ndarray, bin_B: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    sA = _surface_mask(bin_A)
    sB = _surface_mask(bin_B)
    if (not sA.any()) and (not sB.any()):
        return np.array([0.0]), np.array([0.0])
    if not sA.any():
        d_to_A = distance_transform_edt(~_surface_mask(bin_A), sampling=spacing)
        d_BA = d_to_A[_surface_mask(bin_B)]
        return np.array([0.0]), (d_BA if d_BA.size else np.array([np.inf]))
    if not sB.any():
        d_to_B = distance_transform_edt(~_surface_mask(bin_B), sampling=spacing)
        d_AB = d_to_B[_surface_mask(bin_A)]
        return (d_AB if d_AB.size else np.array([np.inf])), np.array([0.0])
    d_to_B = distance_transform_edt(~sB, sampling=spacing)
    d_AB = d_to_B[sA] if sA.any() else np.array([np.inf])
    d_to_A = distance_transform_edt(~sA, sampling=spacing)
    d_BA = d_to_A[sB] if sB.any() else np.array([np.inf])
    if d_AB.size == 0:
        d_AB = np.array([np.inf])
    if d_BA.size == 0:
        d_BA = np.array([np.inf])
    return d_AB, d_BA


def hd95_mm(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    dAB, dBA = _surface_distances(pred, gt, spacing)
    both = np.concatenate([dAB, dBA])
    if not np.isfinite(both).any():
        return float("inf")
    return float(np.percentile(both[np.isfinite(both)], 95))


def assd_mm(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    dAB, dBA = _surface_distances(pred, gt, spacing)
    vals = []
    if np.isfinite(dAB).any():
        vals.append(float(dAB[np.isfinite(dAB)].mean()))
    if np.isfinite(dBA).any():
        vals.append(float(dBA[np.isfinite(dBA)].mean()))
    if not vals:
        return float("inf")
    return float(np.mean(vals))


def surface_dice(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0), tol_mm=SURF_TOL_MM) -> float:
    sp = _surface_mask(pred)
    sg = _surface_mask(gt)
    if (not sp.any()) and (not sg.any()):
        return 1.0
    if (sp.any() and not sg.any()) or (sg.any() and not sp.any()):
        return 0.0
    dt_gt = distance_transform_edt(~sg, sampling=spacing)
    dt_pr = distance_transform_edt(~sp, sampling=spacing)
    within_p = (dt_gt[sp] <= tol_mm) if sp.any() else np.array([])
    within_g = (dt_pr[sg] <= tol_mm) if sg.any() else np.array([])
    fp = within_p.mean() if within_p.size else 0.0
    fg = within_g.mean() if within_g.size else 0.0
    return float(0.5 * (fp + fg))


def volume_error_rel(pred: np.ndarray, gt: np.ndarray) -> float:
    vp = float(np.count_nonzero(pred))
    vg = float(np.count_nonzero(gt))
    return float(abs(vp - vg) / (vg + 1e-10))


def feasible(m: Dict[str, float], eps: Dict[str, float]) -> bool:
    return (
        (m['dice_loss'] <= eps['dice_loss'])
        and (m['sd_loss'] <= eps['sd_loss'])
        and (m['vol_err'] <= eps['vol_err'])
    )


def violation(m: Dict[str, float], eps: Dict[str, float]) -> float:
    vd = max(0.0, m['dice_loss'] - eps['dice_loss']) / (1.0 - eps['dice_loss'])
    vs = max(0.0, m['sd_loss'] - eps['sd_loss']) / (1.0 - eps['sd_loss'])
    vv = max(0.0, m['vol_err'] - eps['vol_err']) / (1.0 - eps['vol_err'])
    return float(vd + vs + vv)


def objective_from_metrics(m: Dict[str, float], eps: Dict[str, float]) -> float:
    base = (W_HD * m['hd_norm']) + (W_DICE * m['dice_loss']) + (W_PROJ * m.get('proj_loss_worst', 0.0))
    return float(base if feasible(m, eps) else base + LAMBDA_VIOL * violation(m, eps))


def _dice2d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    return float((2.0 * inter) / (a.sum(dtype=np.float64) + b.sum(dtype=np.float64) + 1e-10))


def compute_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing: Tuple[float, float, float]) -> Dict[str, float | bool | str]:
    dloss = dice_loss(pred_bin, gt_bin)
    dist_val = assd_mm(pred_bin, gt_bin, spacing) if USE_ASSD_INSTEAD_OF_HD95 else hd95_mm(pred_bin, gt_bin, spacing)
    hd_norm = 1.0 if not np.isfinite(dist_val) else min(dist_val / HD_DMAX_MM, 1.0)
    sdice = surface_dice(pred_bin, gt_bin, spacing, SURF_TOL_MM)
    sd_loss = 1.0 - sdice
    verr = float(min(volume_error_rel(pred_bin, gt_bin), 1.0))

    proj_losses = 1.0 - np.array([
        _dice2d(np.any(pred_bin, axis=2), np.any(gt_bin, axis=2)),
        _dice2d(np.any(pred_bin, axis=1), np.any(gt_bin, axis=1)),
        _dice2d(np.any(pred_bin, axis=0), np.any(gt_bin, axis=0)),
    ], dtype=float)

    metrics: Dict[str, float | bool | str] = {
        'stage': 'C',
        'w_hd': float(W_HD),
        'w_dice': float(W_DICE),
        'w_proj': float(W_PROJ),
        'eps_dice_loss': float(EPSILON['dice_loss']),
        'eps_sd_loss': float(EPSILON['sd_loss']),
        'eps_vol_err': float(EPSILON['vol_err']),
        'dice': float(1.0 - dloss),
        'dice_loss': float(dloss),
        'hd_mm': float(dist_val) if np.isfinite(dist_val) else float('inf'),
        'hd_norm': float(hd_norm),
        'surface_dice': float(sdice),
        'sd_loss': float(sd_loss),
        'vol_err': float(verr),
        'proj_dice_xy': float(1.0 - proj_losses[0]),
        'proj_dice_xz': float(1.0 - proj_losses[1]),
        'proj_dice_yz': float(1.0 - proj_losses[2]),
        'proj_dice_worst': float(1.0 - proj_losses.max()),
        'proj_loss_xy': float(proj_losses[0]),
        'proj_loss_xz': float(proj_losses[1]),
        'proj_loss_yz': float(proj_losses[2]),
        'proj_loss_worst': float(proj_losses.max()),
    }
    metrics['objective_currC'] = objective_from_metrics(metrics, EPSILON)  # type: ignore[arg-type]
    metrics['feasible_currC'] = bool(feasible(metrics, EPSILON))  # type: ignore[arg-type]
    return metrics


def _save_like(out_path: Path, data: np.ndarray, ref_img: nib.Nifti1Image, dtype=np.uint8) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(dtype)
    try:
        hdr['scl_slope'] = 1
        hdr['scl_inter'] = 0
    except Exception:
        pass
    nib.save(nib.Nifti1Image(data.astype(dtype, copy=False), ref_img.affine, header=hdr), str(out_path))


def _same_grid(img: nib.Nifti1Image, ref: nib.Nifti1Image, atol: float = 1e-5) -> bool:
    return img.shape == ref.shape and np.allclose(img.affine, ref.affine, atol=atol)


def _load_pred_on_ref(pred_path: Path, ref_img: nib.Nifti1Image) -> Tuple[np.ndarray, nib.Nifti1Image, str]:
    img = nib.load(str(pred_path))
    if _same_grid(img, ref_img):
        return img.get_fdata(dtype=np.float32), ref_img, 'same_grid'
    if resample_from_to is None:
        raise RuntimeError('Grid mismatch y nibabel.processing.resample_from_to no está disponible')
    rs = resample_from_to(img, ref_img, order=1)
    return rs.get_fdata(dtype=np.float32), ref_img, 'resampled_linear'


def _load_gt_binary(reference_path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(reference_path))
    data = img.get_fdata(dtype=np.float32)
    return (data > 0.5), img


def evaluate_one(
    dataset_name: str,
    avg_path: Path,
    ref_path: Path,
    threshold: float,
    save_thresholded_dir: Optional[Path],
) -> Dict[str, object]:
    gt_bin, ref_img = _load_gt_binary(ref_path)
    pred_data, ref_img, grid_status = _load_pred_on_ref(avg_path, ref_img)
    pred_data = np.nan_to_num(pred_data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    pred_bin = pred_data > float(threshold)

    spacing = tuple(np.array(ref_img.header.get_zooms()[:3], dtype=float)) if ref_img.header.get_zooms() else (1.0, 1.0, 1.0)
    metrics = compute_metrics(pred_bin, gt_bin, spacing)

    out_bin = None
    if save_thresholded_dir is not None:
        save_thresholded_dir.mkdir(parents=True, exist_ok=True)
        thr_tag = str(threshold).replace('.', 'p')
        out_bin = save_thresholded_dir / f"{dataset_name}_avg_thr_{thr_tag}.nii.gz"
        _save_like(out_bin, pred_bin.astype(np.uint8), ref_img, np.uint8)

    row: Dict[str, object] = {
        'dataset': dataset_name,
        'avg_path': str(avg_path),
        'reference_path': str(ref_path),
        'grid_status': grid_status,
        'threshold': float(threshold),
        'thresholded_mask_path': str(out_bin) if out_bin is not None else None,
        'pred_voxels': int(np.count_nonzero(pred_bin)),
        'gt_voxels': int(np.count_nonzero(gt_bin)),
    }
    row.update(metrics)
    return row


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Calcula métricas entre training/validation average tumors y tumor de referencia en MNI.')
    ap.add_argument('--train-avg', type=Path, required=True, help='training_avg.nii.gz')
    ap.add_argument('--val-avg', type=Path, required=True, help='validation_avg.nii.gz')
    ap.add_argument('--reference', type=Path, required=True, help='tumor_upenn_MNI.nii.gz')
    ap.add_argument('--threshold', type=float, default=0.02, help='Umbral aplicado al average tumor antes de binarizar (default: 0.02)')
    ap.add_argument('--out-csv', type=Path, required=True, help='CSV de salida con una fila para training y otra para validation')
    ap.add_argument('--save-thresholded-dir', type=Path, default=None, help='Guardar aquí las máscaras umbralizadas')
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    missing: List[Path] = [p for p in [args.train_avg, args.val_avg, args.reference] if not p.exists()]
    if missing:
        raise SystemExit(f'Faltan inputs: {", ".join(str(p) for p in missing)}')

    rows = [
        evaluate_one('training', args.train_avg, args.reference, args.threshold, args.save_thresholded_dir),
        evaluate_one('validation', args.val_avg, args.reference, args.threshold, args.save_thresholded_dir),
    ]
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, float_format='%.10f')
    print(f'[OK] CSV: {args.out_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

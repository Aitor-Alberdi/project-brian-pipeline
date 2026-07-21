#!/usr/bin/env python3
"""validate_single_run_currC.py

Ejecuta UNA sola simulación (tckgen + tckmap) con parámetros fijos y calcula
el error con exactamente la misma lógica que el optimizador `rand_seed_v2.py`,
pero fijando directamente el "curriculum C" (última etapa):
- Pesos: W_HD=0.20, W_DICE=0.60, W_PROJ=0.20
- EPSILON endurecido 2 veces (A->B y B->C):
    sd_loss = 0.55*(0.9^2)
    vol_err = 0.15*(0.9^2)
    dice_loss = 0.45

Nota: mantengo la implementación de métricas + objective lo más "bit a bit" posible
respecto a `rand_seed_v2.py`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt

pd.set_option('display.float_format', '{:.10f}'.format)

# =========================
# Config (Curriculum C)
# =========================

# Base (A)
_EPS_A = {
    "dice_loss": 0.45,
    "sd_loss": 0.55,
    "vol_err": 0.15,
}

# Tras A->B->C (dos veces *0.9)
EPSILON = {
    "dice_loss": _EPS_A["dice_loss"],
    "sd_loss": _EPS_A["sd_loss"] * (0.9 ** 2),
    "vol_err": _EPS_A["vol_err"] * (0.9 ** 2),
}

LAMBDA_VIOL = 2.0
HD_DMAX_MM = 20.0
SURF_TOL_MM = 2.0
USE_ASSD_INSTEAD_OF_HD95 = False

# Pesos de la etapa C del optimizador rand_seed_v2.py
W_HD = 0.20
W_DICE = 0.60
W_PROJ = 0.20

# Step fijo (el optimizador lo fija si no se optimiza)
FIXED_STEP = 0.5


# =========================
# Utilidades
# =========================

def load_image(img_path: str):
    img = nib.load(img_path)
    return img.get_fdata(), img.affine, img.header


def _delete_file_quietly(path: str) -> None:
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
    except Exception:
        pass


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
    return 1.0 - (2.0 * inter) / (a + b + 1e-10)


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
        return float('inf')
    return float(np.percentile(both[np.isfinite(both)], 95))


def assd_mm(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    dAB, dBA = _surface_distances(pred, gt, spacing)
    vals = []
    if np.isfinite(dAB).any():
        vals.append(dAB[np.isfinite(dAB)].mean())
    if np.isfinite(dBA).any():
        vals.append(dBA[np.isfinite(dBA)].mean())
    if not vals:
        return float('inf')
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
    return abs(vp - vg) / (vg + 1e-10)


# =========================
# Multi-objetivo (igual a rand_seed_v2.py)
# =========================

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


def objective_from_metrics(
    m: Dict[str, float],
    eps: Dict[str, float],
    lam: float = LAMBDA_VIOL,
    w_hd: float = W_HD,
    w_dice: float = W_DICE,
    w_proj: float = W_PROJ,
) -> float:
    base = (w_hd * m['hd_norm']) + (w_dice * m['dice_loss']) + (w_proj * m.get('proj_loss_worst', 0.0))
    return float(base if feasible(m, eps) else base + lam * violation(m, eps))


def build_metrics_record(metrics: Dict[str, float], objective: float) -> Dict[str, float | bool | str]:
    proj_loss_xy = float(metrics.get('proj_loss_xy', np.nan))
    proj_loss_xz = float(metrics.get('proj_loss_xz', np.nan))
    proj_loss_yz = float(metrics.get('proj_loss_yz', np.nan))
    proj_loss_worst = float(metrics.get('proj_loss_worst', np.nan))
    return {
        'objective': float(objective),
        'stage': 'C',
        'feasible': bool(feasible(metrics, EPSILON)),
        'w_hd': float(W_HD),
        'w_dice': float(W_DICE),
        'w_proj': float(W_PROJ),
        'eps_dice_loss': float(EPSILON['dice_loss']),
        'eps_sd_loss': float(EPSILON['sd_loss']),
        'eps_vol_err': float(EPSILON['vol_err']),
        'dice': float(1.0 - metrics['dice_loss']),
        'dice_loss': float(metrics['dice_loss']),
        'hd_mm': float(metrics['hd_mm']),
        'hd_norm': float(metrics['hd_norm']),
        'surface_dice': float(metrics['surface_dice']),
        'sd_loss': float(metrics['sd_loss']),
        'vol_err': float(metrics['vol_err']),
        'proj_dice_xy': float(1.0 - proj_loss_xy),
        'proj_dice_xz': float(1.0 - proj_loss_xz),
        'proj_dice_yz': float(1.0 - proj_loss_yz),
        'proj_dice_worst': float(1.0 - proj_loss_worst),
        'proj_loss_xy': proj_loss_xy,
        'proj_loss_xz': proj_loss_xz,
        'proj_loss_yz': proj_loss_yz,
        'proj_loss_worst': proj_loss_worst,
    }


# =========================
# Analyse (igual que rand_seed_v2.py)
# =========================

def create_roi_mask(ref_image: np.ndarray, x: float, y: float, z: float, radius: int, affine: np.ndarray) -> np.ndarray:
    """Copiado de rand_seed_v2.py (mantengo comportamiento)."""
    r = radius / 2.0
    center_voxel = nib.affines.apply_affine(np.linalg.inv(affine), (x, y, z))
    xr = np.arange(ref_image.shape[0])
    yr = np.arange(ref_image.shape[1])
    zr = np.arange(ref_image.shape[2])
    X, Y, Z = np.meshgrid(xr, yr, zr, indexing='ij')
    dist2 = (X - center_voxel[0])**2 + (Y - center_voxel[1])**2 + (Z - center_voxel[2])**2
    mask = (dist2 <= r**2)
    return np.subtract(ref_image, mask)


def analyse_data(sim_nii_prefix: str, original_mask: np.ndarray, x: float, y: float, z: float, rad: int,
                 affine: np.ndarray, threshold: float, spacing: Tuple[float, float, float]) -> Tuple[float, Dict[str, float]]:
    results, _, _ = load_image(f"{sim_nii_prefix}.nii.gz")

    # Binarización del resultado simulado (igual que rand_seed_v2.py)
    norm_mat = results / (results.max() + 1e-10)
    pred_bin = (norm_mat >= threshold)

    gt_bin = create_roi_mask(original_mask, x, y, z, rad, affine).astype(bool)

    # Métricas 3D
    dloss = dice_loss(pred_bin, gt_bin)
    dist_val = assd_mm(pred_bin, gt_bin, spacing) if USE_ASSD_INSTEAD_OF_HD95 else hd95_mm(pred_bin, gt_bin, spacing)
    hd_norm = 1.0 if not np.isfinite(dist_val) else min(dist_val / HD_DMAX_MM, 1.0)
    sdice = surface_dice(pred_bin, gt_bin, spacing, SURF_TOL_MM)
    sd_loss = 1.0 - sdice
    verr = float(min(volume_error_rel(pred_bin, gt_bin), 1.0))

    # Métricas multi-vista (proyecciones)
    def _dice2d(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(bool)
        b = b.astype(bool)
        inter = np.logical_and(a, b).sum(dtype=np.float64)
        return float((2.0 * inter) / (a.sum(dtype=np.float64) + b.sum(dtype=np.float64) + 1e-10))

    proj_losses = 1.0 - np.array([
        _dice2d(np.any(pred_bin, axis=2), np.any(gt_bin, axis=2)),  # XY
        _dice2d(np.any(pred_bin, axis=1), np.any(gt_bin, axis=1)),  # XZ
        _dice2d(np.any(pred_bin, axis=0), np.any(gt_bin, axis=0)),  # YZ
    ], dtype=float)

    metrics: Dict[str, float] = {
        "dice_loss": float(dloss),
        "hd_mm": float(dist_val) if np.isfinite(dist_val) else float('inf'),
        "hd_norm": float(hd_norm),
        "surface_dice": float(sdice),
        "sd_loss": float(sd_loss),
        "vol_err": float(verr),
        "proj_loss_xy": float(proj_losses[0]),
        "proj_loss_xz": float(proj_losses[1]),
        "proj_loss_yz": float(proj_losses[2]),
        "proj_loss_worst": float(proj_losses.max()),
    }

    obj = objective_from_metrics(metrics, EPSILON)
    return float(obj), metrics


# =========================
# MRtrix run
# =========================

def run_simulation_with_given_parameters(
    args,
    x: float,
    y: float,
    z: float,
    maxlength: float,
    angle: float,
    step: float,
    cutoff: float,
    output_prefix: str,
) -> None:
    if step is None:
        step = FIXED_STEP

    cmd_tckgen = [
        "tckgen",
        f"{args.data}",
        f"{output_prefix}.tck",
        "-seed_sphere",
        f"{x},{y},{z},{args.radius}",
        "-exclude",
        f"{x},{y},{z},{args.radius - 1}",
        "-maxlength",
        f"{maxlength}",
        "-angle",
        f"{angle}",
        "-select",
        f"{args.select}",
        "-algorithm",
        f"{getattr(args, 'algorithm', 'iFOD2')}",
        "-cutoff",
        f"{cutoff}",
        "-step",
        f"{step}",
    ]

    cmd_tckmap = [
        "tckmap",
        f"{output_prefix}.tck",
        "-template",
        f"{args.template}",
        f"{output_prefix}.nii.gz",
    ]

    if args.force:
        cmd_tckgen.append("-force")
        cmd_tckmap.append("-force")

    if args.force:
        subprocess.run(cmd_tckgen, check=True)
        subprocess.run(cmd_tckmap, check=True)
    else:
        subprocess.run(cmd_tckgen, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(cmd_tckmap, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if (not getattr(args, 'keep_tck', False)) and Path(f"{output_prefix}.nii.gz").exists():
        _delete_file_quietly(f"{output_prefix}.tck")


# =========================
# CLI + main
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("validate_single_run_currC.py")

    # MRtrix / data
    p.add_argument("-d", "--data", required=True, type=str, help="Path a dODF / FOD image para tckgen")
    p.add_argument("-tm", "--template", required=True, type=str, help="Template para tckmap")
    p.add_argument("-im", "--image", required=True, type=str, help="Imagen/mascara usada como GT en analyse_data")
    p.add_argument("-o", "--output", required=True, type=str, help="Prefijo de salida (sin extensión)")

    # Parámetros del run
    p.add_argument("--x", required=True, type=float, help="x (mm, mundo) en el espacio del sujeto")
    p.add_argument("--y", required=True, type=float, help="y (mm, mundo) en el espacio del sujeto")
    p.add_argument("--z", required=True, type=float, help="z (mm, mundo) en el espacio del sujeto")

    p.add_argument("--angle", required=True, type=float, help="-angle de tckgen")
    p.add_argument("--maxlen", required=True, type=float, help="-maxlength de tckgen")
    p.add_argument("--cutoff", required=True, type=float, help="-cutoff de tckgen")

    p.add_argument("-r", "--radius", required=True, type=int, help="Radio para seed_sphere")

    # Resto (compat)
    p.add_argument("-sl", "--select", type=int, default=50000)
    p.add_argument("--algorithm", type=str, default="iFOD2", choices=["iFOD2", "NullDist2", "Tensor_Det", "Tensor_Prob"],
                   help="Algoritmo de tckgen. iFOD2 = FOD real; NullDist2 = distribucion nula "
                        "(ignora orientacion de la FOD, muestreo aleatorio). Debe coincidir con el "
                        "algoritmo usado en el entrenamiento.")
    p.add_argument("-minl", "--minlength", type=int, default=0)  # no lo usa tckgen aquí, pero lo dejamos
    p.add_argument("-t", "--threshold", type=float, default=0.02, help="Threshold binarización (sobre norm_mat)")
    p.add_argument("-st", "--step", type=float, default=FIXED_STEP, help="Step (si quieres, aunque está comentado en el optimizador)")
    p.add_argument("-f", "--force", action="store_true", default=False)
    p.add_argument("--keep-tck", action="store_true", default=False, help="Conservar el .tck tras generar el .nii.gz (debug)")

    # Outputs
    p.add_argument("--out-json", type=str, default=None, help="Guardar métricas/params en JSON")
    p.add_argument("--out-csv", type=str, default=None, help="Guardar una fila en CSV")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Cargar GT / referencia
    gt_img, gt_affine, gt_header = load_image(args.image)

    # Spacing real
    try:
        spacing = tuple(np.array(gt_header.get_zooms()[:3], dtype=float))
    except Exception:
        spacing = (1.0, 1.0, 1.0)

    # Run
    out_prefix = str(args.output)

    print(
        f"[RUN] out={out_prefix} | x={args.x:.3f} y={args.y:.3f} z={args.z:.3f} | "
        f"angle={args.angle:.3f} maxlen={args.maxlen:.3f} cutoff={args.cutoff:.5f} | "
        f"radius={args.radius} select={args.select} thr={args.threshold:.4f} | "
        f"Curriculum=C (W_HD={W_HD:.2f}, W_DICE={W_DICE:.2f}, W_PROJ={W_PROJ:.2f})"
    )
    print(f"[EPS] dice={EPSILON['dice_loss']:.4f} sd_loss={EPSILON['sd_loss']:.4f} vol_err={EPSILON['vol_err']:.4f}")

    try:
        run_simulation_with_given_parameters(
            args,
            x=args.x,
            y=args.y,
            z=args.z,
            maxlength=args.maxlen,
            angle=args.angle,
            step=args.step,
            cutoff=args.cutoff,
            output_prefix=out_prefix,
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Falló MRtrix (returncode={e.returncode}).", file=sys.stderr)
        return 2

    # Evaluate
    try:
        obj, metrics = analyse_data(
            sim_nii_prefix=out_prefix,
            original_mask=gt_img,
            x=args.x,
            y=args.y,
            z=args.z,
            rad=args.radius,
            affine=gt_affine,
            threshold=float(args.threshold),
            spacing=spacing,
        )
    except Exception as e:
        print(f"[ERROR] Falló analyse_data: {e}", file=sys.stderr)
        return 2

    feasible_flag = feasible(metrics, EPSILON)
    print(
        f"[METRICS] Obj={obj:.6f} | DiceLoss={metrics['dice_loss']:.4f} | "
        f"HDmm={metrics['hd_mm']:.3f} (norm={metrics['hd_norm']:.3f}) | "
        f"SurfDice={metrics['surface_dice']:.3f} (sd_loss={metrics['sd_loss']:.3f}) | "
        f"VolErr={metrics['vol_err']:.3f} | ProjWorst={metrics['proj_loss_worst']:.3f} | "
        f"Feasible={feasible_flag}"
    )

    metrics_record = build_metrics_record(metrics, obj)

    payload = {
        "output": out_prefix,
        "params": {
            "x": float(args.x),
            "y": float(args.y),
            "z": float(args.z),
            "angle": float(args.angle),
            "maxlen": float(args.maxlen),
            "cutoff": float(args.cutoff),
            "radius": int(args.radius),
            "select": int(args.select),
            "threshold": float(args.threshold),
            "step": float(args.step),
        },
        "curriculum": {
            "stage": "C",
            "weights": {"W_HD": W_HD, "W_DICE": W_DICE, "W_PROJ": W_PROJ},
            "epsilon": EPSILON,
            "lambda_violation": LAMBDA_VIOL,
        },
        "metrics": metrics,
        "metrics_expanded": metrics_record,
        "objective": float(obj),
        "feasible": bool(feasible_flag),
    }

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"[OK] JSON: {out_json}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "output": out_prefix,
            "x": float(args.x),
            "y": float(args.y),
            "z": float(args.z),
            "angle": float(args.angle),
            "maxlen": float(args.maxlen),
            "cutoff": float(args.cutoff),
            "radius": int(args.radius),
            "select": int(args.select),
            "threshold": float(args.threshold),
            "step": float(args.step),
            "obj": float(obj),
            **metrics_record,
        }
        df = pd.DataFrame([row])
        # Si ya existe, append
        if out_csv.exists():
            df.to_csv(out_csv, mode='a', header=False, index=False, float_format='%.10f')
        else:
            df.to_csv(out_csv, index=False, float_format='%.10f')
        print(f"[OK] CSV: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

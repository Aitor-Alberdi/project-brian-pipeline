#!/usr/bin/env python3
# curriculum_gradiente_con_pwn.py
# Limpio, legible, prints en una sola línea. Incluye:
# - Seeds estratégicos dentro del tumor (PCA + percentiles) con parámetros fijos (midpoints)
# - Currículum A → B → C con early stopping por etapa
# - SPSA para (x,y,z) con line-search (overstep) hacia el mejor ±Δ
# - Métricas 3D + proyecciones multi-vista
# - Nombres de salida {output}_{iter}

import sys
import signal
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass, distance_transform_edt, binary_erosion
from sklearn.decomposition import PCA

pd.set_option('display.float_format', '{:.10f}'.format)

# =========================
# Señales
# =========================

def _def_handler(sig, frame):
    sys.exit(1)

signal.signal(signal.SIGINT, _def_handler)

# =========================
# Configuración global
# =========================
EPSILON = {
    "dice_loss": 0.45,
    "sd_loss": 0.55,
    "vol_err": 0.15,
}
LAMBDA_VIOL = 2.0
HD_DMAX_MM = 20.0
SURF_TOL_MM = 2.0
USE_ASSD_INSTEAD_OF_HD95 = False

# Pesos del currículum
W_HD = 0.55
W_DICE = 0.45
W_PROJ = 0.0

# Step fijo para tckgen (si no se optimiza)
FIXED_STEP = 0.5

# =========================
# CLI
# =========================

def parse_arguments():
    p = argparse.ArgumentParser('python3 curriculum_gradiente_con_pwn.py')
    p.add_argument('-co', '--cutoff', type=str, default="0.001", help='cutoff range: low,high,step')
    p.add_argument('-minl', '--minlength', type=int, default=0)
    p.add_argument('-sl', '--select', type=int, default=50000, help='tckgen -select para la simulacion FINAL (re-run del best al terminar).')
    p.add_argument('--optim-select', type=int, default=10000, help='tckgen -select durante el loop del optimizer. Reducir a 10k da ~4x menos tiempo por eval con minimo impacto en el gradiente.')
    p.add_argument('--algorithm', type=str, default='iFOD2', choices=['iFOD2', 'NullDist2', 'Tensor_Det', 'Tensor_Prob'],
                   help='Algoritmo de tckgen. iFOD2 = usa la FOD (data real de difusion). '
                        'NullDist2 = distribucion nula para iFOD2: ignora la orientacion de la FOD y '
                        'genera streamlines por muestreo aleatorio (Morris et al. 2008). Mismo input que iFOD2.')
    p.add_argument('-f', '--force', action='store_true', default=False)

    req = p.add_argument_group('Required')
    req.add_argument('-d', '--data', type=str, required=True)
    req.add_argument('-o', '--output', type=str, required=True)
    req.add_argument('-ang', '--angle', type=str, required=True, help='angle range: low,high,step')
    req.add_argument('-maxl', '--maxlength', type=str, required=True, help='maxlen range: low,high,step')
    req.add_argument('-r', '--radius', type=int, required=True)
    req.add_argument('-st', '--step', type=str, required=True, help='step range (no se optimiza si usas FIXED_STEP)')
    req.add_argument('-im', '--image', type=str, required=True)
    req.add_argument('-tm', '--template', type=str, required=True)
    req.add_argument('-t', '--threshold', type=str, default="0.02", help='threshold binarización')
    req.add_argument('-p', '--power', type=str, default="0.0,1.0,0.1")

    # Extras
    p.add_argument('--pos_window_mm', type=float, default=5.0, help='Ventana ±mm alrededor del COM para bounds de (x,y,z)')

    # Strategic seeds (tumor-informed)
    p.add_argument('--init_points', type=int, default=0, help='Nº de puntos estratégicos dentro del tumor para probar')
    p.add_argument('--init_percentiles', type=str, default='0,50,100', help='Percentiles por eje PCA, p.ej.: "0,25,50,75,100"')
    p.add_argument('--keep-tck', action='store_true', default=False, help='Conservar los .tck después de generar el .nii.gz (debug).')
    
    p.add_argument('--patience_passes', type=int, default=10, help='Pasadas sin mejora antes de actuar')
    p.add_argument('--improve_tol', type=float, default=1e-4, help='Mejora mínima para resetear paciencia')
    p.add_argument('--target_obj', type=float, default=None, help='Objetivo por etapa para parar/promocionar')


    return p

# =========================
# Utilidades I/O y nombres
# =========================

def load_image(img_path):
    img = nib.load(img_path)
    return img.get_fdata(), img.affine, img.header

def make_output_name(args, tag, params):
    iter_token = str(tag).split('_', 1)[0]
    return f"{args.output}_{iter_token}"

def _delete_file_quietly(path):
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
    except Exception:
        pass

# =========================
# Ejecución MRtrix
# =========================

def run_simulation_with_given_parameters(args, x, y, z, maxlength, angle, step, cutoff, power, output, select=None):
    if step is None:
        step = FIXED_STEP

    sel = int(select) if select is not None else int(args.select)

    cmd_tckgen = [
        "tckgen",
        f"{args.data}",
        f"{output}.tck",
        "-seed_sphere", f"{x},{y},{z},{args.radius}",
        "-exclude", f"{x},{y},{z},{args.radius - 1}",
        "-maxlength", f"{maxlength}",
        "-angle", f"{angle}",
        "-select", f"{sel}",
        "-algorithm", f"{getattr(args, 'algorithm', 'iFOD2')}",
        "-cutoff", f"{cutoff}",
        "-step", f"{step}",
    ]

    cmd_tckmap = [
        "tckmap",
        f"{output}.tck",
        "-template", f"{args.template}",
        f"{output}.nii.gz",
    ]

    if args.force:
        cmd_tckgen.append("-force")
        cmd_tckmap.append("-force")

    try:
        if args.force:
            subprocess.run(cmd_tckgen)
            subprocess.run(cmd_tckmap)
        else:
            subprocess.run(cmd_tckgen, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(cmd_tckmap, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not getattr(args, 'keep_tck', False) and Path(f"{output}.nii.gz").exists():
            _delete_file_quietly(f"{output}.tck")
    except Exception as e:
        _ = e


# =========================
# DataFrame y logging
# =========================

def create_dataframe():
    columns = [
        "run", "tag", "x", "y", "z", "angle", "cutoff", "maxlen", "minlen",
        "radius", "step", "power", "threshold", "error", "objective", "stage",
        "feasible", "w_hd", "w_dice", "w_proj", "eps_dice_loss", "eps_sd_loss",
        "eps_vol_err", "dice", "dice_loss", "hd_mm", "hd_norm", "surface_dice",
        "sd_loss", "vol_err", "proj_dice_xy", "proj_dice_xz", "proj_dice_yz",
        "proj_dice_worst", "proj_loss_xy", "proj_loss_xz", "proj_loss_yz", "proj_loss_worst",
    ]
    return pd.DataFrame(columns=columns)


def build_metrics_record(metrics, objective, eps_snapshot, stage_snapshot, weights_snapshot):
    proj_loss_xy = float(metrics.get('proj_loss_xy', np.nan))
    proj_loss_xz = float(metrics.get('proj_loss_xz', np.nan))
    proj_loss_yz = float(metrics.get('proj_loss_yz', np.nan))
    proj_loss_worst = float(metrics.get('proj_loss_worst', np.nan))

    return {
        "objective": float(objective),
        "stage": stage_snapshot,
        "feasible": bool(feasible(metrics, eps_snapshot)),
        "w_hd": float(weights_snapshot['w_hd']),
        "w_dice": float(weights_snapshot['w_dice']),
        "w_proj": float(weights_snapshot['w_proj']),
        "eps_dice_loss": float(eps_snapshot['dice_loss']),
        "eps_sd_loss": float(eps_snapshot['sd_loss']),
        "eps_vol_err": float(eps_snapshot['vol_err']),
        "dice": float(1.0 - metrics['dice_loss']),
        "dice_loss": float(metrics['dice_loss']),
        "hd_mm": float(metrics['hd_mm']),
        "hd_norm": float(metrics['hd_norm']),
        "surface_dice": float(metrics['surface_dice']),
        "sd_loss": float(metrics['sd_loss']),
        "vol_err": float(metrics['vol_err']),
        "proj_dice_xy": float(1.0 - proj_loss_xy),
        "proj_dice_xz": float(1.0 - proj_loss_xz),
        "proj_dice_yz": float(1.0 - proj_loss_yz),
        "proj_dice_worst": float(1.0 - proj_loss_worst),
        "proj_loss_xy": proj_loss_xy,
        "proj_loss_xz": proj_loss_xz,
        "proj_loss_yz": proj_loss_yz,
        "proj_loss_worst": proj_loss_worst,
    }


def add_to_dataframe(err, df, args, x, y, z, angle, maxlen, step, cutoff, power, threshold, run_id=None, tag=None, metrics_record=None):
    row = {
        "run": run_id,
        "tag": tag,
        "x": x,
        "y": y,
        "z": z,
        "angle": angle,
        "cutoff": cutoff,
        "maxlen": maxlen,
        "minlen": args.minlength,
        "radius": args.radius,
        "step": step,
        "power": power,
        "threshold": threshold,
        "error": err,
    }
    if metrics_record:
        row.update(metrics_record)
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


# =========================
# Métricas binarias
# =========================

def _ensure_bool(a):
    return a if a.dtype == bool else a.astype(bool)

def _surface_mask(bin_mask):
    b = _ensure_bool(bin_mask)
    er = binary_erosion(b, iterations=1, border_value=0)
    return b ^ er

def dice_loss(pred, gt):
    p = _ensure_bool(pred)
    g = _ensure_bool(gt)
    inter = np.logical_and(p, g).sum(dtype=np.float64)
    a = p.sum(dtype=np.float64)
    b = g.sum(dtype=np.float64)
    return 1.0 - (2.0 * inter) / (a + b + 1e-10)


def _surface_hd_sdice_shared(pred_bin, gt_bin, spacing, tol_mm=SURF_TOL_MM, use_assd=False, bbox_margin=2):
    """Compute (hd95|assd, surface_dice) in one shot.

    Shares surface masks and distance_transform_edt calls between both metrics
    (2 DT calls instead of 4), and restricts the DT domain to the bounding box
    of `pred | gt` plus a small margin. Correct because all foreground voxels
    live inside the bbox, so the nearest-surface distances computed on the crop
    match those of the full volume.

    Edge-case semantics (bit-compat with the previous split impl for common cases):
      - both masks empty: returns (0.0, 1.0)
      - exactly one empty: returns (inf, 0.0)
    """
    p = _ensure_bool(pred_bin)
    g = _ensure_bool(gt_bin)

    # Crop to bbox of union (huge speedup on full-grid volumes).
    union = p | g
    if not union.any():
        return 0.0, 1.0
    idx = np.argwhere(union)
    mn = np.maximum(idx.min(axis=0) - int(bbox_margin), 0)
    mx = np.minimum(idx.max(axis=0) + int(bbox_margin) + 1, np.array(p.shape))
    sl = tuple(slice(int(mn[i]), int(mx[i])) for i in range(3))
    p_c = p[sl]
    g_c = g[sl]

    sp = _surface_mask(p_c)
    sg = _surface_mask(g_c)
    if (not sp.any()) and (not sg.any()):
        return 0.0, 1.0
    if (not sp.any()) or (not sg.any()):
        return float('inf'), 0.0

    # Single pair of DTs, reused by both hd and surface_dice.
    dt_to_sg = distance_transform_edt(~sg, sampling=spacing)
    dt_to_sp = distance_transform_edt(~sp, sampling=spacing)

    d_AB = dt_to_sg[sp]
    d_BA = dt_to_sp[sg]

    if use_assd:
        d_AB_f = d_AB[np.isfinite(d_AB)]
        d_BA_f = d_BA[np.isfinite(d_BA)]
        vals = []
        if d_AB_f.size:
            vals.append(float(d_AB_f.mean()))
        if d_BA_f.size:
            vals.append(float(d_BA_f.mean()))
        dist_val = float(np.mean(vals)) if vals else float('inf')
    else:
        both = np.concatenate([d_AB, d_BA])
        both_f = both[np.isfinite(both)]
        dist_val = float(np.percentile(both_f, 95)) if both_f.size else float('inf')

    within_p = (d_AB <= tol_mm)
    within_g = (d_BA <= tol_mm)
    fp = float(within_p.mean()) if within_p.size else 0.0
    fg = float(within_g.mean()) if within_g.size else 0.0
    sdice = 0.5 * (fp + fg)
    return float(dist_val), float(sdice)


def volume_error_rel(pred, gt):
    vp = float(np.count_nonzero(pred))
    vg = float(np.count_nonzero(gt))
    return abs(vp - vg) / (vg + 1e-10)

# =========================
# Multi-objetivo
# =========================

def feasible(m, eps):
    return (m['dice_loss'] <= eps['dice_loss']) and (m['sd_loss'] <= eps['sd_loss']) and (m['vol_err'] <= eps['vol_err'])

def violation(m, eps):
    vd = max(0.0, m['dice_loss'] - eps['dice_loss']) / (1.0 - eps['dice_loss'])
    vs = max(0.0, m['sd_loss'] - eps['sd_loss']) / (1.0 - eps['sd_loss'])
    vv = max(0.0, m['vol_err'] - eps['vol_err']) / (1.0 - eps['vol_err'])
    return vd + vs + vv

def objective_from_metrics(m, eps, lam=LAMBDA_VIOL, w_hd=W_HD, w_dice=W_DICE, w_proj=W_PROJ):
    base = (w_hd * m['hd_norm']) + (w_dice * m['dice_loss']) + (w_proj * m.get('proj_loss_worst', 0.0))
    return base if feasible(m, eps) else base + lam * violation(m, eps)

# =========================
# Analyse
# =========================

def create_roi_mask(ref_image, x, y, z, radius, affine):
    r = radius / 2.0
    center_voxel = nib.affines.apply_affine(np.linalg.inv(affine), (x, y, z))
    xr = np.arange(ref_image.shape[0])
    yr = np.arange(ref_image.shape[1])
    zr = np.arange(ref_image.shape[2])
    X, Y, Z = np.meshgrid(xr, yr, zr, indexing='ij')
    dist2 = (X - center_voxel[0])**2 + (Y - center_voxel[1])**2 + (Z - center_voxel[2])**2
    mask = (dist2 <= r**2)
    return np.subtract(ref_image, mask)

def analyse_data(file_name, original_mask, x, y, z, rad, affine, threshold, spacing, dt_gt, coords_gt, cov_gt, centroid_gt, bbox_diag, dt_max, cov_max):
    results, _, _ = load_image(f"{file_name}.nii.gz")
    norm_mat = results / (results.max() + 1e-10)
    pred_bin = (norm_mat >= threshold)
    gt_bin = create_roi_mask(original_mask, x, y, z, rad, affine).astype(bool)

    # Métricas 3D (hd + surface_dice compartiendo DT y recortando al bbox)
    dloss = dice_loss(pred_bin, gt_bin)
    dist_val, sdice = _surface_hd_sdice_shared(
        pred_bin, gt_bin, spacing,
        tol_mm=SURF_TOL_MM,
        use_assd=USE_ASSD_INSTEAD_OF_HD95,
    )
    hd_norm = 1.0 if not np.isfinite(dist_val) else min(dist_val / HD_DMAX_MM, 1.0)
    sd_loss = 1.0 - sdice
    verr = float(min(volume_error_rel(pred_bin, gt_bin), 1.0))

    # Métricas multi-vista (proyecciones)
    def _dice2d(a, b):
        a = a.astype(bool)
        b = b.astype(bool)
        inter = np.logical_and(a, b).sum(dtype=np.float64)
        return (2.0 * inter) / (a.sum(dtype=np.float64) + b.sum(dtype=np.float64) + 1e-10)

    proj_losses = 1.0 - np.array([
        _dice2d(np.any(pred_bin, axis=2), np.any(gt_bin, axis=2)),  # XY
        _dice2d(np.any(pred_bin, axis=1), np.any(gt_bin, axis=1)),  # XZ
        _dice2d(np.any(pred_bin, axis=0), np.any(gt_bin, axis=0)),  # YZ
    ], dtype=float)

    metrics = {
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
    return obj, metrics

# =========================
# Adam
# =========================

class AdamOptimizer:
    def __init__(self, params, lr_dict, beta1=0.9, beta2=0.999, eps=1e-8, decay=0.0):
        self.params = params
        self.lr_dict = lr_dict
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.decay = decay
        self.iter = 0
        self.m = {p: 0.0 for p in params}
        self.v = {p: 0.0 for p in params}

    def step(self, param, grad, minv=None, maxv=None):
        self.iter += 1
        self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
        self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m[param] / (1 - (self.beta1 ** self.iter))
        v_hat = self.v[param] / (1 - (self.beta2 ** self.iter))
        lr_t = self.lr_dict[param] / (1 + self.decay * self.iter)
        new_val = self.params[param] - lr_t * m_hat / (np.sqrt(v_hat) + self.eps)
        if (minv is not None) and (maxv is not None):
            new_val = float(np.clip(new_val, minv, maxv))
        self.params[param] = new_val
        return new_val

# =========================
# Currículum
# =========================

class CurriculumManager:
    def __init__(self, eps_dict, K=5):
        self.stage = 'A'
        self.streak = 0
        self.K = K
        self.eps = eps_dict

    def on_eval(self, metrics):
        if feasible(metrics, self.eps):
            self.streak += 1
        else:
            self.streak = 0
        if (self.stage == 'A') and (self.streak >= self.K):
            self._promote_to_B()
            self.streak = 0
        if (self.stage == 'B') and (self.streak >= 8):
            self._promote_to_C()
            self.streak = 0

    def _promote_to_B(self):
        global W_HD, W_DICE, W_PROJ, EPSILON
        EPSILON['sd_loss'] *= 0.90
        EPSILON['vol_err'] *= 0.90
        W_HD, W_DICE, W_PROJ = 0.35, 0.55, 0.10
        print(f"[Curriculum] enter=B eps_sd={EPSILON['sd_loss']:.3f} eps_vol={EPSILON['vol_err']:.3f} W_HD={W_HD:.2f} W_DICE={W_DICE:.2f} W_PROJ={W_PROJ:.2f}")
        self.stage = 'B'

    def _promote_to_C(self):
        global W_HD, W_DICE, W_PROJ, EPSILON
        EPSILON['sd_loss'] *= 0.90
        EPSILON['vol_err'] *= 0.90
        W_HD, W_DICE, W_PROJ = 0.20, 0.60, 0.20
        print(f"[Curriculum] enter=C eps_sd={EPSILON['sd_loss']:.3f} eps_vol={EPSILON['vol_err']:.3f} W_HD={W_HD:.2f} W_DICE={W_DICE:.2f} W_PROJ={W_PROJ:.2f}")
        self.stage = 'C'

# =========================
# Helpers de evaluación
# =========================

def _eval_objective_with_params(args, params, image, affine, threshold, spacing, dt_gt, coords_gt, cov_gt, centroid_gt, bbox_diag, dt_max, cov_max, tag, dataframe, cur=None, select=None):
    output = make_output_name(args, tag, params)
    step_val = params.get('step', None)
    if step_val is None:
        step_val = FIXED_STEP

    stage_snapshot = cur.stage if cur else 'NA'
    weights_snapshot = {'w_hd': float(W_HD), 'w_dice': float(W_DICE), 'w_proj': float(W_PROJ)}
    eps_snapshot = {k: float(v) for k, v in EPSILON.items()}

    # Durante el optimizer usamos args.optim_select (default 10000); el re-run
    # final del best run al terminar usa args.select (default 50000).
    sel_used = int(select) if select is not None else int(getattr(args, 'optim_select', args.select))

    print(f"[ITER] run={output} tag={tag} stage={stage_snapshot} x={params['x']:.3f} y={params['y']:.3f} z={params['z']:.3f} angle={params['angle']:.3f} maxlen={params['maxlen']:.3f} cutoff={params['cutoff']:.5f} step={step_val:.3f} select={sel_used}")

    run_simulation_with_given_parameters(
        args,
        params['x'], params['y'], params['z'],
        params['maxlen'], params['angle'], step_val,
        params['cutoff'], params.get('power', 0.0),
        output,
        select=sel_used,
    )

    obj, metrics = analyse_data(
        output, image, params['x'], params['y'], params['z'], args.radius,
        affine, threshold, spacing,
        dt_gt, coords_gt, cov_gt, centroid_gt, bbox_diag, dt_max, cov_max
    )

    feasible_flag = feasible(metrics, eps_snapshot)
    print(f"[METRICS] run={output} Obj={obj:.6f} DiceLoss={metrics['dice_loss']:.4f} HDmm={metrics['hd_mm']:.3f} HDnorm={metrics['hd_norm']:.3f} SurfDice={metrics['surface_dice']:.3f} VolErr={metrics['vol_err']:.3f} ProjWorst={metrics['proj_loss_worst']:.3f} Feasible={feasible_flag}")

    metrics_record = build_metrics_record(metrics, obj, eps_snapshot, stage_snapshot, weights_snapshot)

    if cur is not None:
        cur.on_eval(metrics)

    dataframe = add_to_dataframe(
        obj, dataframe, args,
        params['x'], params['y'], params['z'],
        params['angle'], params['maxlen'], step_val,
        params['cutoff'], params.get('power', 0.0), threshold,
        run_id=output, tag=tag, metrics_record=metrics_record
    )
    return obj, metrics, dataframe

def _spsa_pos_gradient(base_params, deltas_xyz, bounds_xyz, eval_fn):
    """SPSA para (x,y,z). Devuelve gradientes + objetivos + metricas de ambos puntos
    + dataframe con AMBAS filas (antes se perdia la fila de 'minus' porque el caller
    solo guardaba df_plus)."""
    rng = np.random.default_rng()
    signs = {k: (1 if rng.integers(0, 2) == 1 else -1) for k in ['x', 'y', 'z']}
    delta = {k: deltas_xyz[k] * signs[k] for k in ['x', 'y', 'z']}

    p_plus = dict(base_params)
    for k in ['x', 'y', 'z']:
        p_plus[k] = float(np.clip(base_params[k] + delta[k], bounds_xyz[k][0], bounds_xyz[k][1]))
    obj_plus, m_plus, df_after_plus = eval_fn(p_plus, 'pos_plus', None)

    p_minus = dict(base_params)
    for k in ['x', 'y', 'z']:
        p_minus[k] = float(np.clip(base_params[k] - delta[k], bounds_xyz[k][0], bounds_xyz[k][1]))
    # Encadenamos el df: la llamada minus tiene que arrancar desde df_after_plus,
    # no desde el df original, o perdemos la fila de plus.
    obj_minus, m_minus, df_after_minus = eval_fn(p_minus, 'pos_minus', df_after_plus)

    g_hat = {}
    for k in ['x', 'y', 'z']:
        denom = 2.0 * delta[k]
        g_hat[k] = 0.0 if abs(denom) < 1e-12 else (obj_plus - obj_minus) / denom
    return g_hat, obj_plus, obj_minus, m_plus, m_minus, df_after_minus, p_plus, p_minus

# =========================
# Bucle principal
# =========================

def gradient_descent(args, x, y, z, image, affine, spacing, dt_gt, coords_gt, cov_gt, centroid_gt, bbox_diag, dt_max, cov_max):
    angle_bottom, angle_top, angle_step = args.angle.split(',')
    maxlen_bottom, maxlen_top, length_step = args.maxlength.split(',')
    step_bottom, step_top, step_step = args.step.split(',')
    cutoff_bottom, cutoff_top, cutoff_step = args.cutoff.split(',')
    power_bottom, power_top, power_step = args.power.split(',')
    threshold = float(args.threshold.split(',')[0]) if isinstance(args.threshold, str) else 0.02

    dataframe = create_dataframe()

    perturbations = {
        "angle": float(angle_step),
        "maxlen": float(length_step),
        "cutoff": float(cutoff_step),
        "x": 0.5,
        "y": 0.5,
        "z": 0.5,
        "power": float(power_step),
    }

    parameters = {
        "angle": 45,
        "maxlen": 45,
        "cutoff": 0.075,
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "power": 0.0,
    }

    com_x, com_y, com_z = x, y, z

    cur = CurriculumManager(EPSILON, K=5)

    # Contador de iteraciones para naming
    iter_counter = 0
    def next_tag(label):
        nonlocal iter_counter
        tag = f"{iter_counter:04d}_{label}"
        iter_counter += 1
        return tag

    # Evaluación inicial
    old_obj, old_metrics, dataframe = _eval_objective_with_params(
        args, parameters, image, affine, threshold, spacing,
        dt_gt, coords_gt, cov_gt, centroid_gt, bbox_diag, dt_max, cov_max,
        tag=next_tag('initial'), dataframe=dataframe, cur=cur
    )

    # --- tracking por etapa para early stopping ---
    best_by_stage = {'A': float('inf'), 'B': float('inf'), 'C': float('inf')}
    no_improve_by_stage = {'A': 0, 'B': 0, 'C': 0}
    best_by_stage[cur.stage] = float(old_obj)

    # Bounds
    w = float(getattr(args, 'pos_window_mm', 5.0))
    bounds = {
        'angle':  (float(angle_bottom),  float(angle_top)),
        'maxlen': (float(maxlen_bottom), float(maxlen_top)),
        'step':   (float(step_bottom),   float(step_top)),
        'cutoff': (float(cutoff_bottom), float(cutoff_top)),
        'power':  (float(power_bottom),  float(power_top)),
        'x': (com_x - w, com_x + w),
        'y': (com_y - w, com_y + w),
        'z': (com_z - w, com_z + w),
    }
    print(f"[BOUNDS] x=({bounds['x'][0]:.3f},{bounds['x'][1]:.3f}) y=({bounds['y'][0]:.3f},{bounds['y'][1]:.3f}) z=({bounds['z'][0]:.3f},{bounds['z'][1]:.3f})")

    # --- Strategic seeds dentro del tumor (parámetros fijos = midpoints) ---
    if getattr(args, 'init_points', 0) and args.init_points > 0:
        mid_angle  = 0.5*(float(angle_bottom)   + float(angle_top))
        mid_maxlen = 0.5*(float(maxlen_bottom)  + float(maxlen_top))
        mid_cutoff = 0.5*(float(cutoff_bottom)  + float(cutoff_top))
        fixed_params = dict(parameters)
        fixed_params['angle']  = mid_angle
        fixed_params['maxlen'] = mid_maxlen
        fixed_params['cutoff'] = mid_cutoff

        vox_idx = np.argwhere(image > 0.5)
        if vox_idx.size == 0:
            seeds_xyz = [(parameters['x'], parameters['y'], parameters['z'])]
        else:
            world_pts = nib.affines.apply_affine(affine, vox_idx)
            bx0,bx1 = bounds['x']; by0,by1 = bounds['y']; bz0,bz1 = bounds['z']
            mask_bounds = (
            (world_pts[:,0] >= bx0) & (world_pts[:,0] <= bx1) &
            (world_pts[:,1] >= by0) & (world_pts[:,1] <= by1) &
            (world_pts[:,2] >= bz0) & (world_pts[:,2] <= bz1)
            )
            if mask_bounds.any():
                world_pts = world_pts[mask_bounds]

            pca = PCA(n_components=3).fit(world_pts)
            comps = pca.components_
            proj = world_pts @ comps.T
            perc_list = [float(p.strip()) for p in str(getattr(args, 'init_percentiles', '0,50,100')).split(',') if p.strip()!='']
            cand_pts = []
            for j in range(3):
                vals = proj[:, j]
                for q in perc_list:
                    tq = np.percentile(vals, q)
                    idx = int(np.argmin(np.abs(vals - tq)))
                    cand_pts.append(tuple(world_pts[idx]))
            cand_pts.append(tuple(world_pts.mean(axis=0)))
            seen = set(); uniq = []
            for pt in cand_pts:
                key = (round(pt[0], 2), round(pt[1], 2), round(pt[2], 2))
                if key not in seen:
                    seen.add(key); uniq.append(pt)
            P = np.array(uniq, dtype=float)
            if len(P) <= args.init_points:
                seeds_xyz = [tuple(p) for p in P]
            else:
                # farthest-point greedy
                chosen = [int(np.argmax(np.linalg.norm(P - P.mean(axis=0), axis=1)))]
                dmin = np.linalg.norm(P - P[chosen[0]], axis=1)
                while len(chosen) < args.init_points:
                    j = int(np.argmax(dmin)); chosen.append(j)
                    dmin = np.minimum(dmin, np.linalg.norm(P - P[j], axis=1))
                seeds_xyz = [tuple(P[i]) for i in chosen]

        best_seed_obj = float(old_obj)
        best_seed_params = dict(parameters)
        best_seed_metrics = old_metrics

        for i, (sx, sy, sz) in enumerate(seeds_xyz):
            sp = dict(fixed_params)
            sp['x'], sp['y'], sp['z'] = float(sx), float(sy), float(sz)
            obj_s, met_s, dataframe = _eval_objective_with_params(
                args, sp, image, affine, threshold, spacing,
                dt_gt, coords_gt, cov_gt, centroid_gt, bbox_diag, dt_max, cov_max,
                tag=next_tag(f"seedPCA{i:02d}"), dataframe=dataframe, cur=None
            )
            if obj_s < best_seed_obj:
                best_seed_obj, best_seed_params, best_seed_metrics = float(obj_s), dict(sp), met_s

        parameters = dict(best_seed_params)
        old_obj, old_metrics = float(best_seed_obj), best_seed_metrics
        cur.streak = 0
        print(f"[SEED] strategy=PCA fixed angle={mid_angle:.3f} maxlen={mid_maxlen:.3f} cutoff={mid_cutoff:.5f} chosen_xyz=({parameters['x']:.3f},{parameters['y']:.3f},{parameters['z']:.3f}) Obj={old_obj:.6f}")

    # SPSA para (x,y,z)
    deltas_xyz = {'x': 1.0, 'y': 1.0, 'z': 1.0}

    # Adam (para escalares)
    lr_dict = {'angle': 10.0, 'maxlen': 10.0, 'cutoff': 0.01, 'x': 1.0, 'y': 1.0, 'z': 1.0}
    optimizer = AdamOptimizer(parameters, lr_dict=lr_dict, decay=1e-18)

    # Bucle con criterio por pasadas
    iteration = 1
    max_iterations = 1_000_000
    while iteration <= max_iterations:
        # ----- Bloque posición: SPSA (x,y,z) con overstep -----
        # eval_fn admite un df opcional para que SPSA pueda encadenar plus->minus
        # sin perder filas (bug previo: df_minus se descartaba).
        def _eval_fn_pos(p_try, tag_suffix, df_in=None):
            df_use = dataframe if df_in is None else df_in
            return _eval_objective_with_params(
                args, p_try, image, affine, threshold, spacing,
                dt_gt, coords_gt, cov_gt, centroid_gt, bbox_diag, dt_max, cov_max,
                tag=next_tag(f"pos_{tag_suffix}"), dataframe=df_use, cur=cur
            )

        g_xyz, obj_plus, obj_minus, m_plus, m_minus, dataframe, p_plus, p_minus = _spsa_pos_gradient(
            parameters,
            deltas_xyz,
            {'x': bounds['x'], 'y': bounds['y'], 'z': bounds['z']},
            eval_fn=_eval_fn_pos
        )

        base_params = {'x': parameters['x'], 'y': parameters['y'], 'z': parameters['z']}
        base_obj = float(old_obj)
        best_params = dict(base_params)
        best_obj = base_obj
        best_metrics = old_metrics

        if obj_plus <= obj_minus:
            dir_vec = {k: p_plus[k] - base_params[k] for k in ['x','y','z']}
            dir_name = 'plus'
            best_params, best_obj, best_metrics = p_plus, obj_plus, m_plus
        else:
            dir_vec = {k: p_minus[k] - base_params[k] for k in ['x','y','z']}
            dir_name = 'minus'
            best_params, best_obj, best_metrics = p_minus, obj_minus, m_minus

        for s in (1.5, 2.0, 3.0):
            cand = {k: float(np.clip(base_params[k] + s*dir_vec[k], bounds[k][0], bounds[k][1])) for k in ['x','y','z']}
            obj_s, met_s, dataframe = _eval_objective_with_params(
                args, {**parameters, **cand}, image, affine, threshold, spacing,
                dt_gt, coords_gt, cov_gt, centroid_gt, bbox_diag, dt_max, cov_max,
                tag=next_tag(f"pos_ls_{dir_name}_x{str(s).replace('.', 'p')}"), dataframe=dataframe, cur=cur
            )
            if obj_s < best_obj:
                best_obj, best_params, best_metrics = obj_s, cand, met_s

        # Aplicar el mejor xyz encontrado. Antes se re-evaluaba aqui ("apply_pos_best")
        # gastando 1 tckgen/pasada redundante; ahora usamos best_metrics cacheado.
        parameters['x'], parameters['y'], parameters['z'] = best_params['x'], best_params['y'], best_params['z']
        old_obj, old_metrics = float(best_obj), best_metrics

        impr = (base_obj - best_obj)
        if impr >= 0.05:
            for k in deltas_xyz: deltas_xyz[k] *= 1.6
        elif impr >= 0.01:
            for k in deltas_xyz: deltas_xyz[k] *= 1.2
        else:
            for k in deltas_xyz: deltas_xyz[k] = max(deltas_xyz[k] * 0.8, 0.25)

        # ----- Escalares: central diff + aceptación -----
        for param in ['angle', 'maxlen', 'cutoff']:
            minv, maxv = bounds[param]
            v0 = float(parameters[param])
            f0 = float(old_obj)
            h = float(max(1e-6, perturbations[param]))

            v_minus = float(np.clip(v0 - h, float(minv), float(maxv)))
            parameters[param] = v_minus
            f_minus, m_minus, dataframe = _eval_objective_with_params(
                args, parameters, image, affine, threshold, spacing,
                dt_gt, coords_gt, cov_gt, centroid_gt, bbox_diag, dt_max, cov_max,
                tag=next_tag(f"{param}_minus"), dataframe=dataframe, cur=cur
            )

            parameters[param] = v0
            v_plus = float(np.clip(v0 + h, float(minv), float(maxv)))
            parameters[param] = v_plus
            f_plus, m_plus, dataframe = _eval_objective_with_params(
                args, parameters, image, affine, threshold, spacing,
                dt_gt, coords_gt, cov_gt, centroid_gt, bbox_diag, dt_max, cov_max,
                tag=next_tag(f"{param}_plus"), dataframe=dataframe, cur=cur
            )

            parameters[param] = v0
            denom = 2.0 * h
            g = ((f_plus - f_minus) / denom) if denom > 1e-12 else 0.0
            if g == 0.0:
                g = 1e-6

            parameters[param] = float(optimizer.step(param, g, float(minv), float(maxv)))
            f_prime, m_prime, dataframe = _eval_objective_with_params(
                args, parameters, image, affine, threshold, spacing,
                dt_gt, coords_gt, cov_gt, centroid_gt, bbox_diag, dt_max, cov_max,
                tag=next_tag(f"apply_{param}"), dataframe=dataframe, cur=cur
            )

            best_f = min(f0, f_minus, f_plus)
            if best_f == f0:
                best_val, best_metrics = v0, old_metrics
            elif best_f == f_minus:
                best_val, best_metrics = v_minus, m_minus
            else:
                best_val, best_metrics = v_plus, m_plus

            tol_accept = 1e-4 * max(1.0, abs(best_f))
            if f_prime <= (best_f - tol_accept):
                old_obj, old_metrics = f_prime, m_prime
                if f_prime <= (0.99 * best_f):
                    perturbations[param] = h * 1.1
            else:
                parameters[param] = float(best_val)
                old_obj, old_metrics = float(best_f), best_metrics
                perturbations[param] = h * 0.7
                optimizer.lr_dict[param] = optimizer.lr_dict.get(param, 1.0) * 0.7

        # ----- Fin de pasada: criterio de parada / promoción por ETAPA -----
        stage = cur.stage
        prev_best = best_by_stage[stage]
        curr_best = min(prev_best, float(old_obj))
        best_by_stage[stage] = curr_best

        if (getattr(args, 'target_obj', None) is not None) and (curr_best <= args.target_obj):
            if stage == 'C':
                print(f"[STOP] Target alcanzado en C: Obj={curr_best:.6f}")
                break
            elif stage == 'A':
                print(f"[Curriculum] Target alcanzado en A (Obj={curr_best:.6f}) → B")
                cur._promote_to_B(); cur.streak = 0; no_improve_by_stage['B'] = 0
            else:
                print(f"[Curriculum] Target alcanzado en B (Obj={curr_best:.6f}) → C")
                cur._promote_to_C(); cur.streak = 0; no_improve_by_stage['C'] = 0

        if curr_best <= prev_best - args.improve_tol:  # args.improve_tol aplicado por etapa
            no_improve_by_stage[stage] = 0
        else:
            no_improve_by_stage[stage] += 1
            print(f"[EarlyStop] (stage {stage}) Sin mejora en {no_improve_by_stage[stage]}/{args.patience_passes} pasadas (best={curr_best:.6f})")
            if no_improve_by_stage[stage] >= args.patience_passes:
                if stage == 'A':
                    print("[Curriculum] Convergencia en A → B")
                    cur._promote_to_B(); cur.streak = 0; no_improve_by_stage['B'] = 0
                elif stage == 'B':
                    print("[Curriculum] Convergencia en B → C")
                    cur._promote_to_C(); cur.streak = 0; no_improve_by_stage['C'] = 0
                else:
                    print("[STOP] Paciencia agotada en C")
                    break

        iteration += 1

    # ----- Resumen final -----
    try:
        idx_best = int(dataframe['error'].idxmin())
        best_row = dataframe.loc[idx_best]
        best_run = str(best_row['run'])
        best_obj = float(best_row['error'])
        best_params_out = {
            'x': float(best_row['x']),
            'y': float(best_row['y']),
            'z': float(best_row['z']),
            'angle': float(best_row['angle']),
            'maxlen': float(best_row['maxlen']),
            'cutoff': float(best_row['cutoff']),
            'power': float(best_row['power']),
            'threshold': float(best_row['threshold']),
        }
        print(f"[BEST] run={best_run} Obj={best_obj:.6f} params={best_params_out}")

        # Re-run del best a resolucion completa (--select) para que el NIfTI final
        # que downstream warpeara a MNI tenga la densidad de streamlines esperada.
        # El loop de optimizacion corre con --optim-select (mas bajo) para ahorrar tiempo.
        final_select = int(getattr(args, 'select', 50000))
        optim_select = int(getattr(args, 'optim_select', final_select))
        if final_select != optim_select:
            print(f"[FINAL] re-running best at --select={final_select} (loop usaba {optim_select}) -> {best_run}.nii.gz")
            try:
                run_simulation_with_given_parameters(
                    args,
                    best_params_out['x'], best_params_out['y'], best_params_out['z'],
                    best_params_out['maxlen'], best_params_out['angle'], None,
                    best_params_out['cutoff'], best_params_out.get('power', 0.0),
                    best_run,
                    select=final_select,
                )
                print(f"[FINAL] {best_run}.nii.gz regenerado a {final_select} streamlines")
            except Exception as e:
                print(f"[WARN] final re-run fallo, se conserva el NIfTI del loop: {e}")

        with open(f"{args.output}_best_params.json", "w") as f:
            json.dump(best_params_out, f, indent=2)
        with open(f"{args.output}_best.txt", "w") as f:
            f.write(f"Best run: {best_run}")
            f.write(f"Obj: {best_obj:.6f}")
            for k, v in best_params_out.items():
                f.write(f"{k}: {v}")
    except Exception as e:
        _ = e

    dataframe.to_pickle('pickle_file2.pickle')
    dataframe.to_csv('simulation.csv', index=False, float_format='%.10f')

# =========================
# Main
# =========================
if __name__ == '__main__':
    args = parse_arguments().parse_args()
    image, affine, header = load_image(args.image)

    # Centro inicial (COM) → mundo
    com = center_of_mass(image)
    voxel_coords = np.array([
        int(round(com[0])),
        int(round(com[1])),
        int(round(com[2]))
    ])
    world = nib.affines.apply_affine(affine, voxel_coords)
    x0, y0, z0 = int(round(world[0])), int(round(world[1])), int(round(world[2]))

    # Spacing real
    try:
        spacing = tuple(np.array(header.get_zooms()[:3], dtype=float))
    except Exception:
        spacing = (1.0, 1.0, 1.0)

    # Pre-cómputos (no críticos para el objetivo actual)
    dt_gt = distance_transform_edt(image, sampling=spacing)
    coords_gt = np.argwhere(image) * np.array(spacing)
    cov_gt = np.cov(coords_gt, rowvar=False) if coords_gt.size else np.zeros((3, 3))
    centroid_gt = coords_gt.mean(axis=0) if coords_gt.size else np.zeros(3)
    bbox_diag = float(np.linalg.norm(coords_gt.max(axis=0) - coords_gt.min(axis=0))) if coords_gt.size else 1.0
    dt_max, cov_max = 1.0, 1.0

    gradient_descent(
        args,
        x0, y0, z0,
        image,
        affine,
        spacing,
        dt_gt,
        coords_gt,
        cov_gt,
        centroid_gt,
        bbox_diag,
        dt_max,
        cov_max
    )


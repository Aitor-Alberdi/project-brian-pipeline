#!/usr/bin/env python3
"""Collect outputs from multiple participants, binarize a chosen NIfTI,
warp the binary mask to MNI, and (optionally) warp the (x,y,z) point to MNI.

Supports two dataset layouts via --mode:

1) mode=training (default)
   - training_root/<ID>/ contains:
       - analysis_participant_<ID>_best_params.json  (preferred)
       - analysis_participant_<ID>_best.txt         (preferred for best run name)
       - simulation.csv                            (fallback for best run/params)
       - best NIfTI named like <best_run>.nii.gz

2) mode=validation
   - validation_root/<ID>/ contains exactly one run (no best.txt), typically:
       - analysis_participant_<ID>.nii.gz
     We binarize that single image and warp it to MNI. No point-warp is attempted.

Registration assumptions (adaptable via CLI flags):
- reg_root/<ID>/ contains ANTs transforms (HCP->MNI) produced by your pipeline:
    - hcpT1_2mni_0GenericAffine.mat
    - hcpT1_2mni_1Warp.nii.gz
    - hcpT1_2mni_1InverseWarp.nii.gz
    - a reference MNI image (default tries MNI_brain_RAS.nii.gz)

Outputs (training mode):
- out_dir/<ID>/best_bin.nii.gz
- out_dir/<ID>/best_bin_MNI.nii.gz

Outputs (validation mode):
- out_dir/<ID>/analysis_bin.nii.gz
- out_dir/<ID>/analysis_bin_MNI.nii.gz
- out_csv with one row per participant
- optional histograms in out_dir/hists/

Notes about ANTs point transforms:
- antsApplyTransformsToPoints expects points in ITK physical space (LPS convention).
  If your coordinates come from nibabel on a NIfTI that you treat as RAS, you likely
  need to flip x,y signs when writing/reading points.

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib

# Make matplotlib optional (only imported if --make-hists)


@dataclass
class BestInfo:
    participant_id: str
    mode: str = "training"  # "training" | "validation"
    best_run: Optional[str] = None
    obj_best_txt: Optional[float] = None
    obj_sim_min: Optional[float] = None
    params: Dict[str, float] | None = None

    # derived
    best_img_path: Optional[Path] = None
    max_intensity: Optional[float] = None
    threshold: Optional[float] = None
    cutoff_float: Optional[float] = None
    cutoff_int: Optional[int] = None

    # warped point (RAS)
    x_mni: Optional[float] = None
    y_mni: Optional[float] = None
    z_mni: Optional[float] = None

    # centroid from warped binary mask (RAS)
    cx_mni: Optional[float] = None
    cy_mni: Optional[float] = None
    cz_mni: Optional[float] = None
    mx_mni: Optional[float] = None
    my_mni: Optional[float] = None
    mz_mni: Optional[float] = None

    # paths
    bin_path: Optional[Path] = None
    bin_mni_path: Optional[Path] = None

    # metricas asociadas al best run / validation run
    metrics: Dict[str, Any] | None = None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_best_txt(best_txt: Path) -> Tuple[Optional[str], Optional[float], Dict[str, float]]:
    """Parse best.txt.

    Returns (best_run_base, obj, kv_params).
    kv_params parses tokens like 'x: 1.23' and 'threshold: 0.02'.

    Robust to missing spaces between fields (e.g., '0.219849x:').
    """
    s = _read_text(best_txt)
    s = " ".join(s.split())  # normalize whitespace

    best_run = None
    obj = None

    if "Best run:" in s and "Obj:" in s:
        i = s.find("Best run:") + len("Best run:")
        j = s.find("Obj:")
        if j > i:
            best_run = s[i:j].strip()

    m = re.search(r"Obj:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", s)
    if m:
        try:
            obj = float(m.group(1))
        except Exception:
            obj = None

    # Parse key:value pairs: x: 1.0, y: -2, angle: 80, etc.
    kv: Dict[str, float] = {}
    # Allow patterns like "x: 7.5" possibly stuck to previous token
    for key in ["x", "y", "z", "angle", "maxlen", "cutoff", "power", "threshold"]:
        mm = re.search(rf"\b{re.escape(key)}\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", s)
        if mm:
            try:
                kv[key] = float(mm.group(1))
            except Exception:
                pass

    return best_run, obj, kv


def read_best_params_json(path: Path) -> Dict[str, float]:
    d = json.loads(_read_text(path))
    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            # keep only numeric
            pass
    return out


def read_simulation_csv_best(sim_csv: Path) -> Tuple[Optional[str], Optional[float], Dict[str, float]]:
    """Read simulation.csv, return best row by minimal 'error'.

    Returns (run_name, min_error, params_dict)
    """
    try:
        df = pd.read_csv(sim_csv)
    except Exception:
        # maybe with different delimiter
        df = pd.read_csv(sim_csv, sep=None, engine="python")

    if "error" not in df.columns:
        return None, None, {}

    # drop NaNs
    dff = df.dropna(subset=["error"])
    if dff.empty:
        return None, None, {}

    idx = dff["error"].astype(float).idxmin()
    row = dff.loc[idx]

    run = None
    if "run" in dff.columns:
        run = str(row["run"]) if pd.notna(row["run"]) else None

    try:
        err = float(row["error"])
    except Exception:
        err = None

    params: Dict[str, float] = {}
    for k in ["x", "y", "z", "angle", "maxlen", "cutoff", "power", "threshold"]:
        if k in dff.columns and pd.notna(row[k]):
            try:
                params[k] = float(row[k])
            except Exception:
                pass

    return run, err, params


def _row_to_python_dict(row: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if pd.isna(v):
            continue
        if hasattr(v, 'item'):
            try:
                v = v.item()
            except Exception:
                pass
        out[str(k)] = v
    return out


def read_metrics_row_csv(csv_path: Path, run_name: Optional[str] = None) -> Dict[str, Any]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, sep=None, engine='python')

    if df.empty:
        return {}

    if run_name is not None and 'run' in df.columns:
        sel = df[df['run'].astype(str) == str(run_name)]
        if not sel.empty:
            return _row_to_python_dict(sel.iloc[0])
    if run_name is not None and 'output' in df.columns:
        sel = df[df['output'].astype(str) == str(run_name)]
        if not sel.empty:
            return _row_to_python_dict(sel.iloc[0])

    if 'error' in df.columns:
        dff = df.dropna(subset=['error'])
        if not dff.empty:
            idx = dff['error'].astype(float).idxmin()
            return _row_to_python_dict(dff.loc[idx])
    if 'obj' in df.columns:
        dff = df.dropna(subset=['obj'])
        if not dff.empty:
            idx = dff['obj'].astype(float).idxmin()
            return _row_to_python_dict(dff.loc[idx])

    return _row_to_python_dict(df.iloc[0])


def find_best_image(train_dir: Path, best_run: str) -> Optional[Path]:
    """Find the NIfTI corresponding to best_run base name."""
    candidates = []
    if best_run.endswith(".nii.gz") or best_run.endswith(".nii"):
        candidates.append(train_dir / best_run)
    else:
        candidates.append(train_dir / f"{best_run}.nii.gz")
        candidates.append(train_dir / f"{best_run}.nii")

    for c in candidates:
        if c.exists():
            return c

    # fallback: maybe best_run includes path fragments or spaces; try glob contains
    stem = best_run.strip().split()[-1]
    for ext in ("*.nii.gz", "*.nii"):
        for p in train_dir.glob(ext):
            if stem in p.name:
                return p

    return None


def find_validation_image(val_dir: Path, pid: str) -> Optional[Path]:
    """Find the single validation NIfTI for a participant.

    Expected default name:
      analysis_participant_<ID>.nii.gz (or .nii)

    Fallback: if exactly one file matches analysis_participant_*.nii.gz in the folder,
    use it.
    """
    c1 = val_dir / f"analysis_participant_{pid}.nii.gz"
    if c1.exists():
        return c1
    c2 = val_dir / f"analysis_participant_{pid}.nii"
    if c2.exists():
        return c2

    matches = sorted(val_dir.glob("analysis_participant_*.nii.gz"))
    if len(matches) == 1:
        return matches[0]

    matches2 = sorted(val_dir.glob("analysis_participant_*.nii"))
    if len(matches2) == 1:
        return matches2[0]

    # Last resort: if there's exactly one NIfTI in the directory, take it.
    all_niis = sorted(list(val_dir.glob("*.nii.gz")) + list(val_dir.glob("*.nii")))
    if len(all_niis) == 1:
        return all_niis[0]

    return None


def round_half_up_positive(x: float) -> int:
    """Round half up for non-negative x."""
    if x < 0:
        return int(math.ceil(x - 0.5))
    return int(math.floor(x + 0.5))


def binarize_best_image(img_path: Path, threshold: float, out_path: Path, min_cutoff_int: int = 0) -> Tuple[float, float, int]:
    """Binarize best image using cutoff = round(max * threshold) (integer).

    Returns (max_intensity, cutoff_float, cutoff_int).
    """
    img = nib.load(str(img_path))
    data = img.get_fdata(dtype=np.float32)

    max_val = float(np.nanmax(data))
    cutoff_float = max_val * float(threshold)
    cutoff_int = round_half_up_positive(cutoff_float)
    if cutoff_int < int(min_cutoff_int):
        cutoff_int = int(min_cutoff_int)

    bin_data = (data >= cutoff_int).astype(np.uint8)

    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    # Ensure scl_slope/inter not messing around
    try:
        hdr["scl_slope"] = 1
        hdr["scl_inter"] = 0
    except Exception:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(bin_data, img.affine, header=hdr), str(out_path))

    return max_val, cutoff_float, cutoff_int


def run_cmd(cmd: List[str], dry_run: bool = False) -> None:
    if dry_run:
        print("[DRY-RUN]", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def warp_mask_to_mni(
    bin_path: Path,
    ref_mni: Path,
    warp_fwd: Path,
    affine_fwd: Path,
    out_path: Path,
    dry_run: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "antsApplyTransforms",
        "-d",
        "3",
        "-i",
        str(bin_path),
        "-r",
        str(ref_mni),
        "-o",
        str(out_path),
        "-n",
        "NearestNeighbor",
        "-t",
        str(warp_fwd),
        "-t",
        str(affine_fwd),
    ]
    run_cmd(cmd, dry_run=dry_run)


def write_points_csv(points_path: Path, xyz_lps: Tuple[float, float, float]) -> None:
    points_path.parent.mkdir(parents=True, exist_ok=True)
    with points_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "t"])
        w.writerow([xyz_lps[0], xyz_lps[1], xyz_lps[2], 0])


def read_points_csv(points_path: Path) -> Tuple[float, float, float]:
    # reads first data row
    with points_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            return float(row["x"]), float(row["y"]), float(row["z"])
    raise RuntimeError(f"No point rows found in {points_path}")


def ras_to_lps(xyz_ras: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = xyz_ras
    return (-x, -y, z)


def lps_to_ras(xyz_lps: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = xyz_lps
    return (-x, -y, z)


def warp_point_to_mni(
    xyz_ras: Tuple[float, float, float],
    inv_warp: Path,
    affine_fwd: Path,
    out_dir: Path,
    dry_run: bool = False,
    assume_input_ras: bool = True,
) -> Tuple[float, float, float]:
    """Warp a point from moving(HCP) to fixed(MNI) using antsApplyTransformsToPoints.

    IMPORTANT:
    - For points, ANTs/ITK uses LPS and expects inverse transforms relative to image warping.
    - If your xyz come from nibabel (RAS), set assume_input_ras=True so we flip to LPS.

    Uses:
      -t [Affine,1] -t InverseWarp

    Returns xyz in RAS (if assume_input_ras=True), otherwise returns xyz as read (LPS).
    """
    in_csv = out_dir / "point_in.csv"
    out_csv = out_dir / "point_out.csv"

    if assume_input_ras:
        xyz_in = ras_to_lps(xyz_ras)
    else:
        xyz_in = xyz_ras

    write_points_csv(in_csv, xyz_in)

    cmd = [
        "antsApplyTransformsToPoints",
        "-d",
        "3",
        "-i",
        str(in_csv),
        "-o",
        str(out_csv),
        "-t",
        f"[{affine_fwd},1]",
        "-t",
        str(inv_warp),
    ]
    run_cmd(cmd, dry_run=dry_run)

    xyz_out = read_points_csv(out_csv)

    if assume_input_ras:
        return lps_to_ras(xyz_out)
    return xyz_out


def centroid_from_mask(mask_path: Path) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    """Compute (mean centroid, per-axis median) in physical coords (RAS as nibabel reports)."""
    img = nib.load(str(mask_path))
    data = img.get_fdata(dtype=np.float32)
    m = data > 0.5
    if not np.any(m):
        return None, None

    ijk = np.argwhere(m)
    xyz = nib.affines.apply_affine(img.affine, ijk)
    mean = xyz.mean(axis=0)
    med = np.median(xyz, axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2])), (float(med[0]), float(med[1]), float(med[2]))


def pick_reference_mni(reg_dir: Path) -> Optional[Path]:
    for name in [
        "MNI_brain_RAS.nii.gz",
        "MNI_brain_raw.nii.gz",
        "hcpT1_in_MNI.nii.gz",
        "upT1_in_MNI.nii.gz",
    ]:
        p = reg_dir / name
        if p.exists():
            return p
    return None


def collect_participant_training(
    train_dir: Path,
    reg_dir: Path,
    out_dir: Path,
    dry_run: bool,
    min_cutoff_int: int,
    assume_input_ras: bool,
) -> BestInfo:
    pid = train_dir.name
    info = BestInfo(participant_id=pid, mode="training", params={})

    best_txt = train_dir / f"analysis_participant_{pid}_best.txt"
    best_params_json = train_dir / f"analysis_participant_{pid}_best_params.json"
    sim_csv = train_dir / "simulation.csv"

    kv_txt: Dict[str, float] = {}

    if best_txt.exists():
        best_run, obj, kv = parse_best_txt(best_txt)
        info.best_run = best_run
        info.obj_best_txt = obj
        kv_txt = kv

    if sim_csv.exists():
        sim_run, sim_obj, sim_params = read_simulation_csv_best(sim_csv)
        info.obj_sim_min = sim_obj
        # if best_run missing, take from simulation
        if info.best_run is None and sim_run:
            info.best_run = sim_run
        # if params json missing, use sim best row
        if not best_params_json.exists():
            info.params.update(sim_params)

        metric_run = info.best_run if info.best_run is not None else sim_run
        sim_metrics = read_metrics_row_csv(sim_csv, run_name=metric_run)
        if sim_metrics:
            info.metrics = sim_metrics

    if best_params_json.exists():
        info.params.update(read_best_params_json(best_params_json))

    # fallback to parsed txt params
    for k, v in kv_txt.items():
        info.params.setdefault(k, v)

    # threshold
    thr = info.params.get("threshold", kv_txt.get("threshold"))
    info.threshold = float(thr) if thr is not None else None

    # best image
    if info.best_run:
        img_path = find_best_image(train_dir, info.best_run)
        info.best_img_path = img_path

    # Binarize
    if info.best_img_path and info.threshold is not None:
        subj_out = out_dir / pid
        bin_path = subj_out / "best_bin.nii.gz"
        max_val, cutoff_float, cutoff_int = binarize_best_image(
            info.best_img_path,
            info.threshold,
            bin_path,
            min_cutoff_int=min_cutoff_int,
        )
        info.bin_path = bin_path
        info.max_intensity = max_val
        info.cutoff_float = cutoff_float
        info.cutoff_int = cutoff_int

        # Warp mask
        warp_fwd = reg_dir / "hcpT1_2mni_1Warp.nii.gz"
        aff_fwd = reg_dir / "hcpT1_2mni_0GenericAffine.mat"
        inv_warp = reg_dir / "hcpT1_2mni_1InverseWarp.nii.gz"
        ref_mni = pick_reference_mni(reg_dir)

        if warp_fwd.exists() and aff_fwd.exists() and ref_mni is not None:
            bin_mni = subj_out / "best_bin_MNI.nii.gz"
            warp_mask_to_mni(bin_path, ref_mni, warp_fwd, aff_fwd, bin_mni, dry_run=dry_run)
            info.bin_mni_path = bin_mni

            # centroid of warped mask
            if not dry_run and bin_mni.exists():
                mean_c, med_c = centroid_from_mask(bin_mni)
                if mean_c:
                    info.cx_mni, info.cy_mni, info.cz_mni = mean_c
                if med_c:
                    info.mx_mni, info.my_mni, info.mz_mni = med_c

        # Warp point (x,y,z)
        if inv_warp.exists() and (reg_dir / "hcpT1_2mni_0GenericAffine.mat").exists():
            if all(k in info.params for k in ["x", "y", "z"]):
                xyz_ras = (float(info.params["x"]), float(info.params["y"]), float(info.params["z"]))
                pt_out_dir = (out_dir / pid / "point_xfm")
                if not dry_run:
                    pt_out_dir.mkdir(parents=True, exist_ok=True)
                try:
                    x_mni, y_mni, z_mni = warp_point_to_mni(
                        xyz_ras,
                        inv_warp=inv_warp,
                        affine_fwd=(reg_dir / "hcpT1_2mni_0GenericAffine.mat"),
                        out_dir=pt_out_dir,
                        dry_run=dry_run,
                        assume_input_ras=assume_input_ras,
                    )
                    info.x_mni, info.y_mni, info.z_mni = x_mni, y_mni, z_mni
                except Exception as e:
                    print(f"[WARN] Point warp failed for {pid}: {e}")

    return info


def collect_participant_validation(
    val_dir: Path,
    reg_dir: Path,
    out_dir: Path,
    dry_run: bool,
    min_cutoff_int: int,
    threshold: float,
    compute_centroids: bool = False,
) -> BestInfo:
    """Validation pipeline: binarize a single run image and warp to MNI.

    Expected layout:
      validation_root/<ID>/analysis_participant_<ID>.nii.gz

    No best.txt / best_params.json is required.
    """
    pid = val_dir.name
    info = BestInfo(participant_id=pid, mode="validation", params={})

    info.best_run = f"analysis_participant_{pid}"
    info.threshold = float(threshold)
    info.params["threshold"] = float(threshold)

    img_path = find_validation_image(val_dir, pid)
    info.best_img_path = img_path
    if img_path is None:
        print(f"[WARN] No se encuentra NIfTI de validación para {pid} en {val_dir}")
        return info

    metrics_csv = val_dir / "metrics.csv"
    metrics_json = val_dir / "metrics.json"
    if metrics_csv.exists():
        info.metrics = read_metrics_row_csv(metrics_csv)
    elif metrics_json.exists():
        try:
            payload = json.loads(_read_text(metrics_json))
            if isinstance(payload, dict):
                info.metrics = payload.get('metrics_expanded') or payload.get('metrics') or {}
                if 'params' in payload and isinstance(payload['params'], dict):
                    for k, v in payload['params'].items():
                        try:
                            info.params[str(k)] = float(v)
                        except Exception:
                            pass
        except Exception:
            pass

    subj_out = out_dir / pid
    bin_path = subj_out / "analysis_bin.nii.gz"
    max_val, cutoff_float, cutoff_int = binarize_best_image(
        img_path,
        float(threshold),
        bin_path,
        min_cutoff_int=min_cutoff_int,
    )
    info.bin_path = bin_path
    info.max_intensity = max_val
    info.cutoff_float = cutoff_float
    info.cutoff_int = cutoff_int

    # Warp mask to MNI (same as training)
    warp_fwd = reg_dir / "hcpT1_2mni_1Warp.nii.gz"
    aff_fwd = reg_dir / "hcpT1_2mni_0GenericAffine.mat"
    ref_mni = pick_reference_mni(reg_dir)

    if warp_fwd.exists() and aff_fwd.exists() and ref_mni is not None:
        bin_mni = subj_out / "analysis_bin_MNI.nii.gz"
        warp_mask_to_mni(bin_path, ref_mni, warp_fwd, aff_fwd, bin_mni, dry_run=dry_run)
        info.bin_mni_path = bin_mni

        # Centroid of warped mask (optional; disabled by default for validation)
        if compute_centroids and (not dry_run) and bin_mni.exists():
            mean_c, med_c = centroid_from_mask(bin_mni)
            if mean_c:
                info.cx_mni, info.cy_mni, info.cz_mni = mean_c
            if med_c:
                info.mx_mni, info.my_mni, info.mz_mni = med_c
    else:
        print(f"[WARN] No hay transforms/ref MNI para {pid} en {reg_dir} -> no se warpea")

    return info


def infos_to_dataframe(infos: List[BestInfo]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for inf in infos:
        row: Dict[str, Any] = {
            "participant_id": inf.participant_id,
            "mode": inf.mode,
            "best_run": inf.best_run,
            "obj_best_txt": inf.obj_best_txt,
            "obj_sim_min": inf.obj_sim_min,
            "max_intensity": inf.max_intensity,
            "threshold": inf.threshold,
            "cutoff_float": inf.cutoff_float,
            "cutoff_int": inf.cutoff_int,
            "x_mni": inf.x_mni,
            "y_mni": inf.y_mni,
            "z_mni": inf.z_mni,
            "cx_mni": inf.cx_mni,
            "cy_mni": inf.cy_mni,
            "cz_mni": inf.cz_mni,
            "mx_mni": inf.mx_mni,
            "my_mni": inf.my_mni,
            "mz_mni": inf.mz_mni,
            "best_img_path": str(inf.best_img_path) if inf.best_img_path else None,
            "bin_path": str(inf.bin_path) if inf.bin_path else None,
            "bin_mni_path": str(inf.bin_mni_path) if inf.bin_mni_path else None,
        }
        if inf.params:
            for k, v in inf.params.items():
                if k not in row:
                    row[k] = v
        if inf.metrics:
            for k, v in inf.metrics.items():
                if k not in row:
                    row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)

    # enforce column order
    front = [
        "participant_id",
        "mode",
        "best_run",
        "obj_best_txt",
        "obj_sim_min",
        "x",
        "y",
        "z",
        "x_mni",
        "y_mni",
        "z_mni",
        "cx_mni",
        "cy_mni",
        "cz_mni",
        "mx_mni",
        "my_mni",
        "mz_mni",
        "angle",
        "maxlen",
        "cutoff",
        "power",
        "threshold",
        "max_intensity",
        "cutoff_float",
        "cutoff_int",
        "best_img_path",
        "bin_path",
        "bin_mni_path",
    ]
    cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
    return df[cols]


def _edges_from_step(vals: np.ndarray, step: float) -> np.ndarray:
    """Create histogram bin edges with a fixed step.

    Edges span [floor(min/step)*step, ceil(max/step)*step] and include the rightmost edge.
    """
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.array([0.0, step], dtype=float)

    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    if end <= start:
        end = start + step

    n = int(round((end - start) / step))
    return start + step * np.arange(n + 1, dtype=float)


def make_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Only the requested histograms, with fixed bin widths
    # Requested:
    #  - angle: 5° bins
    #  - maxlen: 5mm bins
    #  - cutoff: 0.005 bins
    #  - coords (MNI): 0.5mm bins
    bin_steps: Dict[str, float] = {
        "angle": 5.0,
        "maxlen": 5.0,
        "cutoff": 0.005,
        "x_mni": 0.5,
        "y_mni": 0.5,
        "z_mni": 0.5,
    }

    for col, step in bin_steps.items():
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        vals = df[col].dropna().astype(float).to_numpy()
        if vals.size == 0:
            continue

        edges = _edges_from_step(vals, step=float(step))
        plt.figure(figsize=(7, 4))
        plt.hist(vals, bins=edges)
        plt.title(f"Histograma: {col} (bin={step})")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{col}.png", dpi=150)
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Procesa múltiples sujetos: binariza un NIfTI (best en training, único run en validation), "
            "warpea la máscara binaria a MNI y (solo en training) warpea el punto (x,y,z) a MNI."
        )
    )
    ap.add_argument(
        "--mode",
        choices=["training", "validation"],
        default="training",
        help="Selecciona el layout/flujo: training (best.*) o validation (un solo .nii.gz por sujeto).",
    )
    ap.add_argument(
        "--train-root",
        required=True,
        type=Path,
        help=(
            "Raíz del dataset. En modo training apunta a training_results_2. "
            "En modo validation puedes apuntar aquí a validation_results si no usas --val-root."
        ),
    )
    ap.add_argument(
        "--val-root",
        type=Path,
        default=None,
        help="Opcional: raíz de validation_results (solo se usa si --mode=validation).",
    )
    ap.add_argument("--val-threshold", type=float, default=0.02, help="Threshold para binarización en modo validation")
    ap.add_argument(
        "--val-centroids",
        action="store_true",
        help="(Validation) Calcula centroides/medianas en MNI a partir de la máscara warpeada (por defecto no).",
    )
    ap.add_argument("--reg-root", required=True, type=Path, help="Raíz de registration_results_* (mismo set de IDs)")
    ap.add_argument("--out-dir", required=True, type=Path, help="Directorio de salida")
    ap.add_argument("--out-csv", required=True, type=Path, help="CSV final")
    ap.add_argument("--min-cutoff-int", type=int, default=0, help="Forzar cutoff_int mínimo (útil si round(max*thr)=0)")
    ap.add_argument("--assume-input-ras", action="store_true", default=True, help="Asume que x,y,z están en RAS y convierte a LPS para ANTs")
    ap.add_argument("--no-assume-input-ras", dest="assume_input_ras", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="No ejecuta ANTs, solo imprime comandos")
    ap.add_argument("--make-hists", action="store_true", help="Genera histogramas (PNG) para columnas numéricas")
    # Nota: bins fijos por variable (angle/maxlen/cutoff/x_mni/y_mni/z_mni). Se mantiene sin flags.

    args = ap.parse_args()

    train_root: Path = args.train_root
    reg_root: Path = args.reg_root
    out_dir: Path = args.out_dir

    if not train_root.exists():
        raise SystemExit(f"train-root no existe: {train_root}")
    if not reg_root.exists():
        raise SystemExit(f"reg-root no existe: {reg_root}")

    mode: str = str(args.mode)
    data_root: Path
    if mode == "training":
        data_root = train_root
    else:
        data_root = args.val_root if args.val_root is not None else train_root
        if not data_root.exists():
            raise SystemExit(f"val-root no existe: {data_root}")

    # participants = directories under data_root
    subj_dirs = [p for p in data_root.iterdir() if p.is_dir()]
    subj_dirs.sort(key=lambda p: p.name)

    infos: List[BestInfo] = []

    for td in subj_dirs:
        pid = td.name
        rd = reg_root / pid
        if not rd.exists():
            print(f"[WARN] No hay registro para {pid} en {rd} -> se omite")
            continue

        print(f"[INFO] Procesando {pid} (mode={mode})")
        if mode == "training":
            inf = collect_participant_training(
                train_dir=td,
                reg_dir=rd,
                out_dir=out_dir,
                dry_run=args.dry_run,
                min_cutoff_int=args.min_cutoff_int,
                assume_input_ras=args.assume_input_ras,
            )
        else:
            inf = collect_participant_validation(
                val_dir=td,
                reg_dir=rd,
                out_dir=out_dir,
                dry_run=args.dry_run,
                min_cutoff_int=args.min_cutoff_int,
                threshold=float(args.val_threshold),
                compute_centroids=bool(args.val_centroids),
            )
        infos.append(inf)

    df = infos_to_dataframe(infos)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] CSV guardado en: {args.out_csv}")

    # stats (solo los 6 parámetros que interesan)
    selected_cols = ["angle", "maxlen", "cutoff", "x_mni", "y_mni", "z_mni"]
    selected_cols = [c for c in selected_cols if (c in df.columns) and pd.api.types.is_numeric_dtype(df[c])]
    if selected_cols:
        stats = df[selected_cols].describe(percentiles=[0.5]).T
        stats_path = args.out_csv.with_suffix(".summary.csv")
        stats.to_csv(stats_path)
        print(f"[OK] Resumen (selected describe) guardado en: {stats_path}")

        # mean/median shortcuts
        mean_vals = df[selected_cols].mean(numeric_only=True)
        med_vals = df[selected_cols].median(numeric_only=True)
        mean_path = args.out_csv.with_suffix(".mean.csv")
        med_path = args.out_csv.with_suffix(".median.csv")
        mean_vals.to_csv(mean_path, header=["mean"])
        med_vals.to_csv(med_path, header=["median"])
        print(f"[OK] Medias (selected) guardadas en: {mean_path}")
        print(f"[OK] Medianas (selected) guardadas en: {med_path}")

    if args.make_hists:
        hist_dir = out_dir / "hists"
        make_histograms(df, hist_dir)
        print(f"[OK] Histogramas guardados en: {hist_dir}")


if __name__ == "__main__":
    main()

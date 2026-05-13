#!/usr/bin/env python3
"""run_full_tumor_pipeline.py

Controlador "one-command" del pipeline completo que describiste.

Pipeline (por defecto ejecuta TODO):
  1) Registro UPENN->HCP/MNI por sujeto para TRAIN (20) y VALID (65)
  2) mrconvert: T1_hcp_RAS.nii.gz -> T1_hcp_RAS.mif (en ambos sets)
  3) Optimización (SGD/optimizer) SOLO en TRAIN
  4) Collect+warp a MNI del mejor run por sujeto (TRAIN) + histogramas/mean/median
  5) VALIDACIÓN (single-run) usando mean/median + coords MNI -> native
     + binarización+warp a MNI de outputs de validación
  6) Tumor UPENN -> MNI (máscara binaria labels 1,4)
     + average optimized tumor (TRAIN) + average validation tumor (VALID)

Requisitos externos:
  - ANTs: antsApplyTransforms, antsApplyTransformsToPoints
  - MRtrix3: mrconvert, tckgen, tckmap
  - Python deps: numpy, pandas, nibabel

"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import nibabel as nib
import pandas as pd


TRAIN_IDS_DEFAULT = [
    "100408", "101309", "101915", "103111", "103414", "103818", "105014", "105115",
    "108828", "110411", "113619", "113922", "114419", "115320", "116524", "117122",
    "118730", "118932", "120111", "122317",
]


@dataclass
class Paths:
    pipeline_root: Path
    scripts_dir: Path
    analysis_home: Path
    hcp_root: Path

    reg_train: Path
    reg_val: Path
    train_results: Path
    val_results: Path
    best_mni_outputs: Path
    val_mni_outputs: Path
    avg_mni_outputs: Path
    tumor_mni_outputs: Path

    def ensure_dirs(self) -> None:
        for p in [
            self.reg_train,
            self.reg_val,
            self.train_results,
            self.val_results,
            self.best_mni_outputs,
            self.val_mni_outputs,
            self.avg_mni_outputs,
            self.tumor_mni_outputs,
        ]:
            p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Utils
# -------------------------

def _which(exe: str) -> Optional[str]:
    return shutil.which(exe)


def run_streamed(cmd: List[str], cwd: Optional[Path] = None, log_path: Optional[Path] = None, dry_run: bool = False) -> None:
    """Run a command, streaming stdout+stderr to console and optionally to a log file."""
    if dry_run:
        print("[DRY-RUN]", " ".join(cmd))
        return

    if cwd is not None:
        cwd.mkdir(parents=True, exist_ok=True)

    print("[CMD]", " ".join(cmd))

    f = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            f = log_path.open("a", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if f is not None:
                f.write(line)
                f.flush()
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)

    finally:
        if f is not None:
            f.close()


def bash_capture(cmd: str) -> List[str]:
    """Run a bash pipeline and return stdout lines (stripped)."""
    out = subprocess.check_output(["bash", "-lc", cmd], text=True)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return lines


def load_mean_or_median(summary_csv: Path, key: str) -> float:
    """Read value from best_params_all.mean.csv or best_params_all.median.csv."""
    if not summary_csv.exists():
        raise FileNotFoundError(f"No existe summary CSV: {summary_csv}")
    df = pd.read_csv(summary_csv, index_col=0)
    # column name can be 'mean' or 'median'
    if df.shape[1] != 1:
        # best effort: take first numeric column
        col = df.columns[0]
    else:
        col = df.columns[0]
    if key not in df.index:
        raise KeyError(f"No encuentro '{key}' en {summary_csv}")
    return float(df.loc[key, col])


def parse_labels_csv(s: str) -> List[int]:
    s = str(s).strip()
    if s == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def select_labels_keep(in_mask: Path, out_mask: Path, labels: List[int]) -> None:
    """Keep only selected integer labels, others set to 0.

    Preserva el valor original de la etiqueta (p.ej. 1 y 4) para mimetizar
    tu pipeline previo.

    Si labels está vacío -> conserva (data>0) y binariza a 1.
    """
    img = nib.load(str(in_mask))
    data = img.get_fdata(dtype=np.float32)

    out = np.zeros_like(data, dtype=np.uint8)
    if labels:
        keep = np.isin(data, np.array(labels, dtype=np.float32))
        out[~keep] = 0
        for lab in labels:
            out[data == float(lab)] = int(lab)
    else:
        out[(data > 0)] = 1

    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    try:
        hdr["scl_slope"] = 1
        hdr["scl_inter"] = 0
    except Exception:
        pass

    out_mask.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, img.affine, header=hdr), str(out_mask))


def select_labels_binarize(in_mask: Path, out_mask: Path, labels: List[int]) -> None:
    """Selecciona labels y binariza a 0/1."""
    img = nib.load(str(in_mask))
    data = img.get_fdata(dtype=np.float32)

    out = np.zeros_like(data, dtype=np.uint8)
    if labels:
        out[np.isin(data, np.array(labels, dtype=np.float32))] = 1
    else:
        out[data > 0] = 1

    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    try:
        hdr["scl_slope"] = 1
        hdr["scl_inter"] = 0
    except Exception:
        pass

    out_mask.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, img.affine, header=hdr), str(out_mask))


# -------------------------
# Step 0: IDs
# -------------------------

def get_validation_ids(hcp_root: Path) -> List[str]:
    cmd = (
        f"ls -l {hcp_root} | head -n 88 | tail -n 66 | "
        "awk '{print $NF}' | grep -v 756055"
    )
    ids = bash_capture(cmd)
    return ids


def parse_ids_csv(s: str) -> List[str]:
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def read_ids_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"No existe IDs file: {path}")
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def resolve_ids(ids_csv: str, ids_file: Optional[Path], default_ids: List[str]) -> List[str]:
    if ids_file is not None:
        return read_ids_file(ids_file)
    ids_inline = parse_ids_csv(ids_csv)
    if ids_inline:
        return ids_inline
    return list(default_ids)


def do_cleanup_training_intermediates(train_results_root: Path, best_csv: Path, dry_run: bool) -> None:
    if not best_csv.exists():
        print(f"[WARN] cleanup skipped: no existe {best_csv}")
        return

    try:
        df = pd.read_csv(best_csv, dtype={"participant_id": str, "best_run": str})
    except Exception as e:
        print(f"[WARN] cleanup skipped: no pude leer {best_csv}: {e}")
        return

    if "participant_id" not in df.columns or "best_run" not in df.columns:
        print(f"[WARN] cleanup skipped: {best_csv} no tiene participant_id/best_run")
        return

    deleted = 0
    kept = 0
    for _, row in df.iterrows():
        pid = str(row["participant_id"])
        best_run = str(row["best_run"])
        subj_dir = train_results_root / pid
        if not subj_dir.exists():
            continue

        keep_names = {
            f"{best_run}.nii.gz",
            f"{best_run}.nii",
            "simulation.csv",
            "pickle_file2.pickle",
            f"analysis_participant_{pid}_best.txt",
            f"analysis_participant_{pid}_best_params.json",
            "output.txt",
        }

        for p in subj_dir.glob("analysis_participant_*.nii.gz"):
            if p.name in keep_names:
                kept += 1
                continue
            print(f"[CLEAN]{' [DRY-RUN]' if dry_run else ''} delete {p}")
            if not dry_run:
                p.unlink(missing_ok=True)
            deleted += 1

        for p in subj_dir.glob("analysis_participant_*.nii"):
            if p.name in keep_names:
                kept += 1
                continue
            print(f"[CLEAN]{' [DRY-RUN]' if dry_run else ''} delete {p}")
            if not dry_run:
                p.unlink(missing_ok=True)
            deleted += 1

        for p in subj_dir.glob("analysis_participant_*.tck"):
            print(f"[CLEAN]{' [DRY-RUN]' if dry_run else ''} delete {p}")
            if not dry_run:
                p.unlink(missing_ok=True)
            deleted += 1

    print(f"[OK] cleanup training intermediates: kept={kept} deleted={deleted}")


# -------------------------
# Step 1: Registration (split)
#
# Flow refactorizado:
#   1a) prep_upenn_for_mni.sh       -> 1 vez por tumor   (reorient + masks + labels)
#   1b) register_upenn_to_mni.sh    -> 1 vez por tumor   (ANTs UPENN->MNI)
#   1c) register_one_hcp_subject.sh -> 1 vez por sujeto  (asegura cache HCP->MNI
#                                                         compartido entre tumores,
#                                                         symlinka y compone tumor)
# -------------------------

def do_prep_upenn_for_mni(
    prep_script: Path,
    t1_upenn: Path,
    tumor_seg_upenn: Path,
    mni_brain: Path,
    labels_csv: str,
    upenn_workdir: Path,
    dry_run: bool,
) -> None:
    upenn_workdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash", str(prep_script),
        "--t1-upenn", str(t1_upenn),
        "--tumor-seg-upenn", str(tumor_seg_upenn),
        "--mni-brain", str(mni_brain),
        "--labels", str(labels_csv),
        "--outdir", str(upenn_workdir),
    ]
    run_streamed(cmd, cwd=None, log_path=upenn_workdir / "prep_upenn.log", dry_run=dry_run)


def do_register_upenn_to_mni(
    register_script: Path,
    upenn_workdir: Path,
    dry_run: bool,
) -> None:
    cmd = [
        "bash", str(register_script),
        "--workdir", str(upenn_workdir),
    ]
    run_streamed(cmd, cwd=None, log_path=upenn_workdir / "register_upenn_to_mni.log", dry_run=dry_run)


def do_register_subjects_batch(
    ids: Iterable[str],
    register_script: Path,
    upenn_workdir: Path,
    cache_root: Path,
    hcp_root: Path,
    mni_brain: Path,
    out_root: Path,
    dry_run: bool,
    skip_if_done: bool = True,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    for sid in ids:
        subj_dir = out_root / sid
        subj_dir.mkdir(parents=True, exist_ok=True)

        done_marker = subj_dir / "tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz"
        if skip_if_done and done_marker.exists():
            print(f"[SKIP] {sid}: registration ya existe ({done_marker.name})")
            continue

        t1_hcp = hcp_root / sid / "T1w" / "T1w_acpc_dc_restore_brain.nii.gz"
        if not t1_hcp.exists():
            print(f"[WARN] {sid}: no existe T1 HCP: {t1_hcp}")
            continue

        cmd = [
            "bash", str(register_script),
            "--subject-id", str(sid),
            "--t1-hcp", str(t1_hcp),
            "--mni-brain", str(mni_brain),
            "--upenn-workdir", str(upenn_workdir),
            "--cache-root", str(cache_root),
            "--outdir", str(subj_dir),
        ]
        run_streamed(cmd, cwd=None, log_path=subj_dir / "registration.log", dry_run=dry_run)


# -------------------------
# Step 2: mrconvert
# -------------------------

def do_mrconvert_batch(ids: Iterable[str], reg_root: Path, dry_run: bool) -> None:
    for sid in ids:
        subj_dir = reg_root / sid
        in_nii = subj_dir / "T1_hcp_RAS.nii.gz"
        out_mif = subj_dir / "T1_hcp_RAS.mif"
        if out_mif.exists():
            continue
        if not in_nii.exists():
            print(f"[WARN] {sid}: no existe {in_nii}")
            continue
        run_streamed(["mrconvert", str(in_nii), str(out_mif)], cwd=subj_dir, log_path=subj_dir / "mrconvert.log", dry_run=dry_run)


# -------------------------
# Step 3: Optimizer (train)
# -------------------------

def do_optimizer_train(
    ids: Iterable[str],
    train_results_root: Path,
    reg_train_root: Path,
    odf_root_train: Path,
    optimizer_py: Path,
    dry_run: bool,
    threshold: float,
) -> None:
    train_results_root.mkdir(parents=True, exist_ok=True)

    for sid in ids:
        out_dir = train_results_root / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        # skip if best exists
        if (out_dir / f"analysis_participant_{sid}_best.txt").exists():
            print(f"[SKIP] {sid}: optimizer ya produjo best.txt")
            continue

        data_path = odf_root_train / sid / "odf_sh_order12_mrtrix.mif"
        img_path = reg_train_root / sid / "tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz"
        template_path = reg_train_root / sid / "T1_hcp_RAS.mif"

        missing = [p for p in [data_path, img_path, template_path, optimizer_py] if not p.exists()]
        if missing:
            print(f"[WARN] {sid}: faltan inputs para optimizer: {', '.join(str(m) for m in missing)}")
            continue

        cmd = [
            "python3",
            "-u",
            str(optimizer_py),
            "--data",
            str(data_path),
            "--output",
            f"analysis_participant_{sid}",
            "-ang",
            "20,90,2",
            "-maxl",
            "20,90,2",
            "--radius",
            "3",
            "-st",
            "0.3,1,0.05",
            "--image",
            str(img_path),
            "--template",
            str(template_path),
            "-co",
            "0.04,0.1,0.005",
            "--threshold",
            f"{threshold}",
            "--pos_window_mm",
            "5",
            "--init_points",
            "10",
            "--init_percentiles",
            "0,25,50,75,100",
            "--patience_passes",
            "10",
            "--improve_tol",
            "0.005",
            "--target_obj",
            "0.1",
        ]

        run_streamed(cmd, cwd=out_dir, log_path=out_dir / "output.txt", dry_run=dry_run)


# -------------------------
# Step 4: Collect best -> MNI + hists
# -------------------------

def do_collect_best_to_mni(
    scripts_dir: Path,
    train_results_root: Path,
    reg_train_root: Path,
    best_mni_outputs: Path,
    dry_run: bool,
) -> Path:
    collect_py = scripts_dir / "collect_best_to_mni_3.py"
    out_csv = best_mni_outputs / "best_params_all.csv"

    cmd = [
        "python3",
        str(collect_py),
        "--mode",
        "training",
        "--train-root",
        str(train_results_root),
        "--reg-root",
        str(reg_train_root),
        "--out-dir",
        str(best_mni_outputs),
        "--out-csv",
        str(out_csv),
        "--make-hists",
    ]

    run_streamed(cmd, cwd=None, log_path=best_mni_outputs / "collect_best_to_mni.log", dry_run=dry_run)
    return out_csv


# -------------------------
# Step 5: Validation (coords + single run + warp outputs)
# -------------------------

def do_validation(
    scripts_dir: Path,
    ids_val: Iterable[str],
    reg_val_root: Path,
    val_results_root: Path,
    odf_root_val: Path,
    best_summary_csv: Path,
    summary_kind: str,
    threshold: float,
    dry_run: bool,
    pipeline_root: Path,
) -> None:
    # 5.1 Read chosen params from mean/median CSV
    angle = load_mean_or_median(best_summary_csv, "angle")
    maxlen = load_mean_or_median(best_summary_csv, "maxlen")
    cutoff = load_mean_or_median(best_summary_csv, "cutoff")
    x_mni = load_mean_or_median(best_summary_csv, "x_mni")
    y_mni = load_mean_or_median(best_summary_csv, "y_mni")
    z_mni = load_mean_or_median(best_summary_csv, "z_mni")

    print(
        f"[PARAMS:{summary_kind}] angle={angle:.6f} maxlen={maxlen:.6f} cutoff={cutoff:.6f} | "
        f"x_mni={x_mni:.6f} y_mni={y_mni:.6f} z_mni={z_mni:.6f}"
    )

    # 5.2 Warp MNI -> native (validation subjects)
    warp_coords_py = scripts_dir / "warp_mni_to_subject_coords.py"
    coords_csv = pipeline_root / "mni_to_native_coords.csv"

    cmd_coords = [
        "python3",
        str(warp_coords_py),
        "--reg-root",
        str(reg_val_root),
        "--x",
        str(x_mni),
        "--y",
        str(y_mni),
        "--z",
        str(z_mni),
        "--space",
        "hcp",
        "--out-csv",
        str(coords_csv),
    ]
    run_streamed(cmd_coords, cwd=None, log_path=pipeline_root / "warp_mni_to_native.log", dry_run=dry_run)

    # Read coords
    if not coords_csv.exists() and not dry_run:
        raise RuntimeError(f"No se generó coords CSV: {coords_csv}")
    coords_df = (
        pd.read_csv(coords_csv, dtype={"subject_id": str})
        if coords_csv.exists()
        else pd.DataFrame()
    )
    coords_df = coords_df.set_index("subject_id") if ("subject_id" in coords_df.columns) else coords_df

    validate_py = scripts_dir / "validate_single_run_currC.py"

    val_results_root.mkdir(parents=True, exist_ok=True)

    for sid in ids_val:
        out_dir = val_results_root / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        out_prefix = out_dir / f"analysis_participant_{sid}"
        # skip if output exists
        if (out_prefix.with_suffix(".nii.gz")).exists():
            print(f"[SKIP] {sid}: validation output ya existe")
            continue

        data_path = odf_root_val / sid / "odf_sh_order12_mrtrix.mif"
        gt_img = reg_val_root / sid / "tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz"
        template = reg_val_root / sid / "T1_hcp_RAS.mif"

        missing = [p for p in [data_path, gt_img, template, validate_py] if not p.exists()]
        if missing:
            print(f"[WARN] {sid}: faltan inputs para validation: {', '.join(str(m) for m in missing)}")
            continue

        if coords_df is not None and (sid in coords_df.index) and ("x_native" in coords_df.columns):
            x_native = float(coords_df.loc[sid, "x_native"])
            y_native = float(coords_df.loc[sid, "y_native"])
            z_native = float(coords_df.loc[sid, "z_native"])
        else:
            print(f"[WARN] {sid}: no encuentro coords nativas en {coords_csv} -> skip")
            continue

        cmd_val = [
            "python3",
            "-u",
            str(validate_py),
            "--data",
            str(data_path),
            "--template",
            str(template),
            "--image",
            str(gt_img),
            "--output",
            str(out_prefix),
            "--x",
            str(x_native),
            "--y",
            str(y_native),
            "--z",
            str(z_native),
            "--angle",
            str(angle),
            "--maxlen",
            str(maxlen),
            "--cutoff",
            str(cutoff),
            "--radius",
            "3",
            "--step",
            "0.5",
            "--threshold",
            str(threshold),
            "--out-json",
            str(out_dir / "metrics.json"),
            "--out-csv",
            str(out_dir / "metrics.csv"),
        ]

        run_streamed(cmd_val, cwd=out_dir, log_path=out_dir / "output.txt", dry_run=dry_run)


def do_collect_validation_to_mni(
    scripts_dir: Path,
    val_results_root: Path,
    reg_val_root: Path,
    val_mni_outputs: Path,
    threshold: float,
    dry_run: bool,
) -> Path:
    collect_py = scripts_dir / "collect_best_to_mni_3.py"
    out_csv = val_mni_outputs / "validation_all.csv"

    cmd = [
        "python3",
        str(collect_py),
        "--mode",
        "validation",
        "--train-root",
        str(val_results_root),
        "--reg-root",
        str(reg_val_root),
        "--out-dir",
        str(val_mni_outputs),
        "--out-csv",
        str(out_csv),
        "--val-threshold",
        str(threshold),
    ]

    run_streamed(cmd, cwd=None, log_path=val_mni_outputs / "collect_validation_to_mni.log", dry_run=dry_run)
    return out_csv


# -------------------------
# Step 6: Tumor UPENN -> MNI + averages
# -------------------------

def _pick_reference_mni_from_reg(reg_dir: Path) -> Path:
    for name in [
        "MNI_brain_RAS.nii.gz",
        "MNI_brain_raw.nii.gz",
        "hcpT1_in_MNI.nii.gz",
        "upT1_in_MNI.nii.gz",
    ]:
        cand = reg_dir / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No encuentro referencia MNI en {reg_dir}")


def do_warp_upenn_tumor_to_mni_shared(
    registration_root: Path,
    reg_dir_any: Path,
    input_mask: Path,
    labels: List[int],
    dry_run: bool,
) -> Path:
    registration_root.mkdir(parents=True, exist_ok=True)

    tumor_bin = registration_root / ".tumor_upenn_selected_bin.nii.gz"
    tumor_mni = registration_root / "tumor_upenn_MNI.nii.gz"
    ref_mni = _pick_reference_mni_from_reg(reg_dir_any)
    warp_fwd = reg_dir_any / "upT1_2mni_1Warp.nii.gz"
    affine_fwd = reg_dir_any / "upT1_2mni_0GenericAffine.mat"

    if not warp_fwd.exists():
        raise FileNotFoundError(f"No existe warp UPENN->MNI: {warp_fwd}")
    if not affine_fwd.exists():
        raise FileNotFoundError(f"No existe affine UPENN->MNI: {affine_fwd}")

    if not dry_run:
        select_labels_binarize(input_mask, tumor_bin, labels=labels)

    cmd = [
        "antsApplyTransforms",
        "-d", "3",
        "-i", str(tumor_bin),
        "-r", str(ref_mni),
        "-o", str(tumor_mni),
        "-n", "NearestNeighbor",
        "-t", str(warp_fwd),
        "-t", str(affine_fwd),
    ]
    run_streamed(cmd, cwd=None, log_path=registration_root / "tumor_upenn_to_mni.log", dry_run=dry_run)
    return tumor_mni


def do_averages(
    scripts_dir: Path,
    best_mni_outputs: Path,
    val_mni_outputs: Path,
    avg_out_dir: Path,
    dry_run: bool,
) -> None:
    avg_py = scripts_dir / "average_mni_across_participants.py"

    cmd = [
        "python3",
        str(avg_py),
        "--train-root",
        str(best_mni_outputs),
        "--val-root",
        str(val_mni_outputs),
        "--out-dir",
        str(avg_out_dir),
    ]
    run_streamed(cmd, cwd=None, log_path=avg_out_dir / "average.log", dry_run=dry_run)

    # Friendly names
    if not dry_run:
        tr = avg_out_dir / "training_avg.nii.gz"
        va = avg_out_dir / "validation_avg.nii.gz"
        if tr.exists():
            shutil.copyfile(tr, avg_out_dir / "average_optimized_tumor.nii.gz")
        if va.exists():
            shutil.copyfile(va, avg_out_dir / "average_validation_tumor.nii.gz")


def do_average_metrics(
    scripts_dir: Path,
    avg_out_dir: Path,
    tumor_mni: Path,
    threshold: float,
    dry_run: bool,
) -> Path:
    cmp_py = scripts_dir / "compare_average_tumors_to_reference.py"
    out_csv = avg_out_dir / "average_vs_tumor_metrics.csv"

    cmd = [
        "python3",
        str(cmp_py),
        "--train-avg",
        str(avg_out_dir / "training_avg.nii.gz"),
        "--val-avg",
        str(avg_out_dir / "validation_avg.nii.gz"),
        "--reference",
        str(tumor_mni),
        "--threshold",
        str(threshold),
        "--out-csv",
        str(out_csv),
        "--save-thresholded-dir",
        str(avg_out_dir / "thresholded_avg_masks"),
    ]
    run_streamed(cmd, cwd=None, log_path=avg_out_dir / "average_metrics.log", dry_run=dry_run)
    return out_csv

# -------------------------
# Main
# -------------------------

def main() -> int:
    p = argparse.ArgumentParser("run_full_tumor_pipeline.py")

    p.add_argument(
        "--pipeline-root",
        type=Path,
        default=Path("/home/Archive/BRIAN/Pipeline"),
        help="Carpeta madre donde se crean registration/registration_20, registration/registration_65, training_results, etc.",
    )

    p.add_argument(
        "--analysis-home",
        type=Path,
        default=Path("/home/Archive/BRIAN/Analysis_HCP_UPENN"),
        help="Carpeta donde vive pipeline_original_T1_MNI.sh y el template MNI.",
    )

    p.add_argument(
        "--hcp-root",
        type=Path,
        default=Path("/home/Archive/HCP_1200_qc_data"),
        help="Root de HCP_1200_qc_data.",
    )

    p.add_argument(
        "--tumor-mask-upenn",
        type=Path,
        required=True,
        help="Máscara de tumor en UPENN space (NO binaria). Se conservarán labels 1 y 4 por defecto.",
    )

    p.add_argument(
        "--t1-upenn",
        type=Path,
        default=Path("/home/Archive/BRIAN/Pipeline/UPENN-GBM/NIfTI-files/images_structural/UPENN-GBM-00174_11/UPENN-GBM-00174_11_T1.nii.gz"),
        help="T1 UPENN (en el mismo grid que la máscara).",
    )

    p.add_argument(
        "--mni-brain",
        type=Path,
        default=Path("/home/Archive/BRIAN/Analysis_HCP_UPENN/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz"),
        help="Template MNI brain usado por el pipeline de registro.",
    )

    p.add_argument(
        "--pipeline-original-script",
        type=Path,
        default=Path("/home/Archive/BRIAN/Analysis_HCP_UPENN/pipeline_original_T1_MNI.sh"),
        help="[DEPRECATED] Se mantiene como fallback; ya no se usa en el flujo por defecto.",
    )

    p.add_argument(
        "--prep-upenn-script",
        type=Path,
        default=None,
        help="Script prep_upenn_for_mni.sh (default: junto a este archivo).",
    )

    p.add_argument(
        "--register-upenn-script",
        type=Path,
        default=None,
        help="Script register_upenn_to_mni.sh (default: junto a este archivo).",
    )

    p.add_argument(
        "--register-subject-script",
        type=Path,
        default=None,
        help="Script register_one_hcp_subject.sh (default: junto a este archivo).",
    )

    p.add_argument(
        "--hcp-to-mni-cache",
        type=Path,
        default=Path("/opt/users/aalbe03/Parallel_runs/cache/hcp_to_mni"),
        help="Cache compartido HCP->MNI (reutilizado entre tumores). Se crea si no existe.",
    )

    p.add_argument(
        "--optimizer-script",
        type=Path,
        default=Path("/home/Archive/BRIAN/Pipeline/rand_seed_v2.py"),
        help="Optimizer (rand_seed_v2.py).",
    )

    p.add_argument(
        "--odf-train-root",
        type=Path,
        default=Path("/home/Archive/BRIAN/Analysis_HCP_UPENN/odf_generation_results_20"),
        help="Root de ODFs training (contiene <ID>/odf_sh_order12_mrtrix.mif).",
    )

    p.add_argument(
        "--odf-val-root",
        type=Path,
        default=Path("/home/Archive/BRIAN/Analysis_HCP_UPENN/odf_generation_results_66"),
        help="Root de ODFs validation (contiene <ID>/odf_sh_order12_mrtrix.mif).",
    )

    p.add_argument(
        "--summary",
        choices=["median", "mean"],
        default="median",
        help="Qué usar para parametrizar la validación: median (default) o mean.",
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Threshold (tanto para optimizer/validation como para binarización en collect).",
    )

    p.add_argument(
        "--labels",
        type=str,
        default="1,4",
        help="Labels de tumor a conservar desde la máscara UPENN (default: 1,4).",
    )

    p.add_argument(
        "--upenn-to-mni-subject",
        type=str,
        default=TRAIN_IDS_DEFAULT[0],
        help="Qué sujeto usar para warpear UPENN->MNI en el paso 6 (default: primer TRAIN ID).",
    )

    p.add_argument(
        "--train-ids",
        type=str,
        default="",
        help="Lista CSV opcional de IDs de training para sobreescribir los 20 defaults.",
    )

    p.add_argument(
        "--train-ids-file",
        type=Path,
        default=None,
        help="Fichero opcional con un ID de training por línea.",
    )

    p.add_argument(
        "--val-ids",
        type=str,
        default="",
        help="Lista CSV opcional de IDs de validation.",
    )

    p.add_argument(
        "--val-ids-file",
        type=Path,
        default=None,
        help="Fichero opcional con un ID de validation por línea.",
    )

    p.add_argument(
        "--cleanup-training-intermediates",
        action="store_true",
        help="Borra NIfTIs/TCK intermedios de training y conserva solo el best run por sujeto.",
    )

    # Step toggles
    p.add_argument("--skip-registration", action="store_true")
    p.add_argument("--skip-mrconvert", action="store_true")
    p.add_argument("--skip-optimizer", action="store_true")
    p.add_argument("--skip-collect", action="store_true")
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--skip-averages", action="store_true")

    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args()

    scripts_dir = Path(__file__).resolve().parent

    paths = Paths(
        pipeline_root=args.pipeline_root,
        scripts_dir=scripts_dir,
        analysis_home=args.analysis_home,
        hcp_root=args.hcp_root,
        reg_train=args.pipeline_root / "registration" / "registration_20",
        reg_val=args.pipeline_root / "registration" / "registration_65",
        train_results=args.pipeline_root / "training_results",
        val_results=args.pipeline_root / "validation_results",
        best_mni_outputs=args.pipeline_root / "best_mni_outputs",
        val_mni_outputs=args.pipeline_root / "validation_mni_outputs",
        avg_mni_outputs=args.pipeline_root / "avg_mni_outputs",
        tumor_mni_outputs=args.pipeline_root / "registration",
    )
    paths.ensure_dirs()

    # Dependency sanity checks (non-fatal)
    deps = ["antsApplyTransforms", "antsApplyTransformsToPoints", "mrconvert", "tckgen", "tckmap"]
    missing = [d for d in deps if _which(d) is None]
    if missing:
        print(f"[WARN] No encuentro en PATH: {', '.join(missing)}")

    # IDs
    train_ids = resolve_ids(args.train_ids, args.train_ids_file, TRAIN_IDS_DEFAULT)
    val_ids = resolve_ids(args.val_ids, args.val_ids_file, get_validation_ids(args.hcp_root))

    print(f"[INFO] TRAIN IDs: {len(train_ids)}")
    print(f"[INFO] VAL IDs:   {len(val_ids)}")

    # 0) Prepare tumor mask (selección de labels) en UPENN space
    labels_list = parse_labels_csv(args.labels)
    label_tag = "_".join(str(x) for x in labels_list) if labels_list else "nonzero"
    tumor_filtered = (
        args.pipeline_root
        / "tumor_inputs"
        / (args.tumor_mask_upenn.stem.replace(".nii", "") + f"_labels{label_tag}.nii.gz")
    )
    if not tumor_filtered.exists() and not args.dry_run:
        select_labels_keep(args.tumor_mask_upenn, tumor_filtered, labels=labels_list)
        print(f"[OK] Tumor mask filtrada (labels {args.labels}): {tumor_filtered}")
    elif args.dry_run:
        print(f"[DRY-RUN] Crearía tumor mask filtrada: {tumor_filtered}")

    # 1) Registration (flujo refactorizado)
    if not args.skip_registration:
        if not args.t1_upenn.exists():
            raise SystemExit(f"No existe T1 UPENN: {args.t1_upenn}")
        if not args.mni_brain.exists():
            raise SystemExit(f"No existe MNI brain template: {args.mni_brain}")
        if not args.tumor_mask_upenn.exists():
            raise SystemExit(f"No existe tumor mask UPENN: {args.tumor_mask_upenn}")

        prep_script = args.prep_upenn_script or (scripts_dir / "prep_upenn_for_mni.sh")
        reg_upenn_script = args.register_upenn_script or (scripts_dir / "register_upenn_to_mni.sh")
        reg_subj_script = args.register_subject_script or (scripts_dir / "register_one_hcp_subject.sh")
        for s in [prep_script, reg_upenn_script, reg_subj_script]:
            if not s.exists():
                raise SystemExit(f"No existe script de registro: {s}")

        # Cache HCP->MNI compartido entre tumores: crear si no existe
        args.hcp_to_mni_cache.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] HCP->MNI cache: {args.hcp_to_mni_cache}")

        # Workdir UPENN por tumor (prep + UPENN->MNI)
        upenn_workdir = args.pipeline_root / "registration" / "upenn_workdir"
        upenn_workdir.mkdir(parents=True, exist_ok=True)

        print("\n[STEP 1a] Prep UPENN (reorient + masks + labels) -> una vez por tumor")
        do_prep_upenn_for_mni(
            prep_script=prep_script,
            t1_upenn=args.t1_upenn,
            tumor_seg_upenn=args.tumor_mask_upenn,
            mni_brain=args.mni_brain,
            labels_csv=args.labels,
            upenn_workdir=upenn_workdir,
            dry_run=args.dry_run,
        )

        print("\n[STEP 1b] Register UPENN -> MNI (una vez por tumor)")
        do_register_upenn_to_mni(
            register_script=reg_upenn_script,
            upenn_workdir=upenn_workdir,
            dry_run=args.dry_run,
        )

        print("\n[STEP 1c] Registration TRAIN (20)  [HCP->MNI via cache compartido]")
        do_register_subjects_batch(
            ids=train_ids,
            register_script=reg_subj_script,
            upenn_workdir=upenn_workdir,
            cache_root=args.hcp_to_mni_cache,
            hcp_root=args.hcp_root,
            mni_brain=args.mni_brain,
            out_root=paths.reg_train,
            dry_run=args.dry_run,
        )

        print("\n[STEP 1c] Registration VALIDATION (65)  [HCP->MNI via cache compartido]")
        do_register_subjects_batch(
            ids=val_ids,
            register_script=reg_subj_script,
            upenn_workdir=upenn_workdir,
            cache_root=args.hcp_to_mni_cache,
            hcp_root=args.hcp_root,
            mni_brain=args.mni_brain,
            out_root=paths.reg_val,
            dry_run=args.dry_run,
        )

    # 2) mrconvert
    if not args.skip_mrconvert:
        print("\n[STEP 2] mrconvert TRAIN")
        do_mrconvert_batch(train_ids, paths.reg_train, dry_run=args.dry_run)
        print("\n[STEP 2] mrconvert VALID")
        do_mrconvert_batch(val_ids, paths.reg_val, dry_run=args.dry_run)

    # 3) optimizer (train)
    if not args.skip_optimizer:
        if not args.optimizer_script.exists():
            raise SystemExit(f"No existe optimizer script: {args.optimizer_script}")
        print("\n[STEP 3] Optimizer TRAIN")
        do_optimizer_train(
            ids=train_ids,
            train_results_root=paths.train_results,
            reg_train_root=paths.reg_train,
            odf_root_train=args.odf_train_root,
            optimizer_py=args.optimizer_script,
            dry_run=args.dry_run,
            threshold=float(args.threshold),
        )

    # 4) collect best + hists
    best_csv = paths.best_mni_outputs / "best_params_all.csv"
    if not args.skip_collect:
        print("\n[STEP 4] Collect best -> MNI + hist")
        best_csv = do_collect_best_to_mni(
            scripts_dir=paths.scripts_dir,
            train_results_root=paths.train_results,
            reg_train_root=paths.reg_train,
            best_mni_outputs=paths.best_mni_outputs,
            dry_run=args.dry_run,
        )

    if args.cleanup_training_intermediates:
        print("\n[STEP 4b] Cleanup training intermediates")
        do_cleanup_training_intermediates(
            train_results_root=paths.train_results,
            best_csv=best_csv,
            dry_run=args.dry_run,
        )

    # 5) validation
    if not args.skip_validation:
        print("\n[STEP 5] Validation single-run")
        summary_csv = best_csv.with_suffix(f".{args.summary}.csv")
        do_validation(
            scripts_dir=paths.scripts_dir,
            ids_val=val_ids,
            reg_val_root=paths.reg_val,
            val_results_root=paths.val_results,
            odf_root_val=args.odf_val_root,
            best_summary_csv=summary_csv,
            summary_kind=args.summary,
            threshold=float(args.threshold),
            dry_run=args.dry_run,
            pipeline_root=paths.pipeline_root,
        )

        print("\n[STEP 5b] Collect validation outputs -> MNI")
        do_collect_validation_to_mni(
            scripts_dir=paths.scripts_dir,
            val_results_root=paths.val_results,
            reg_val_root=paths.reg_val,
            val_mni_outputs=paths.val_mni_outputs,
            threshold=float(args.threshold),
            dry_run=args.dry_run,
        )

    # 6) tumor UPENN -> MNI (single compartido) + averages + métricas finales
    if not args.skip_averages:
        print("\n[STEP 6] Tumor UPENN -> MNI compartido + averages + métricas finales")
        single_sid = str(args.upenn_to_mni_subject)
        reg_dir_any = paths.reg_train / single_sid
        if not reg_dir_any.exists():
            raise SystemExit(f"No existe el directorio de registro para {single_sid}: {reg_dir_any}")

        tumor_mni = do_warp_upenn_tumor_to_mni_shared(
            registration_root=paths.tumor_mni_outputs,
            reg_dir_any=reg_dir_any,
            input_mask=tumor_filtered if tumor_filtered.exists() else args.tumor_mask_upenn,
            labels=labels_list,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            print(f"[OK] tumor_upenn_MNI.nii.gz: {tumor_mni}")

        do_averages(
            scripts_dir=paths.scripts_dir,
            best_mni_outputs=paths.best_mni_outputs,
            val_mni_outputs=paths.val_mni_outputs,
            avg_out_dir=paths.avg_mni_outputs,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            print(f"[OK] average_optimized_tumor.nii.gz: {paths.avg_mni_outputs / 'average_optimized_tumor.nii.gz'}")
            print(f"[OK] average_validation_tumor.nii.gz: {paths.avg_mni_outputs / 'average_validation_tumor.nii.gz'}")

        avg_metrics_csv = do_average_metrics(
            scripts_dir=paths.scripts_dir,
            avg_out_dir=paths.avg_mni_outputs,
            tumor_mni=tumor_mni,
            threshold=float(args.threshold),
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            print(f"[OK] Métricas finales average vs tumor MNI: {avg_metrics_csv}")

    print("\n[DONE] Pipeline completo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

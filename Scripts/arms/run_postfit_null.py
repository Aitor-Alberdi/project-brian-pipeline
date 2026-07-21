#!/usr/bin/env python3
"""Driver: post-fit pipeline (stages 4, 5, 5b, 6) on the NULL-DISTRIBUTION training
results (stage_null) for the 10-tumour subset.

Mirror of run_postfit_c2.py but:
  - reads training from training_results/<sid>/stage_null/
  - validation single-runs use --algorithm NullDist2 --select 10000
  - all outputs go to *_null dirs (best_mni_outputs_null, validation_results_null,
    mni_to_native_coords_null.csv, validation_mni_outputs_null, avg_mni_outputs_null)

Per tumour (idempotent; per-tumour marker = avg_mni_outputs_null/average_vs_tumor_metrics.csv):
  4   collect best (stage_null) -> MNI; regen registration_65 [pool]; warp seed -> native
  5   validate single run x65 (tckgen NullDist2) [pool] -> validation_results_null/<sid>/
  5b  collect validation -> MNI
  6   averages (train+val) + compare to ref -> avg_mni_outputs_null/average_vs_tumor_metrics.csv

Usage: python -u run_postfit_null.py --tumours <csv> --workers N --cpus M [--env-bin DIR]
"""
from __future__ import annotations
import os
import csv
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 10-tumour subset: worst / average / best by validation Dice (stage_c_2 post-fit)
TEN = [
    "UPENN-GBM-00001_11", "UPENN-GBM-00031_11", "UPENN-GBM-00024_11",
    "UPENN-GBM-00034_11", "UPENN-GBM-00019_11", "UPENN-GBM-00023_11", "UPENN-GBM-00020_11",
    "UPENN-GBM-00025_11", "UPENN-GBM-00032_11", "UPENN-GBM-00042_11",
]
BASE_RUNS = Path("/home/Archive/BRIAN/All_30/runs")
SCRIPTS   = Path("/home/Archive/BRIAN/proyect-brian-pipeline/Scripts")
CACHE     = Path("/home/Archive/BRIAN/Batch_2/cache/hcp_to_mni")
MNI       = Path("/home/Archive/BRIAN/Analysis_HCP_UPENN/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz")
ODF66     = Path("/home/Archive/BRIAN/Analysis_HCP_UPENN/odf_generation_results_66")
HCP       = Path("/home/Archive/HCP_1200_qc_data")
VAL_EXCLUDE = {"756055"}

TIMEOUT_VALIDATE = 2 * 3600
TIMEOUT_LIGHT    = 1 * 3600
TIMEOUT_COLLECT  = 4 * 3600

# set in main(); inherited by forked pool workers
ENV_BIN = "/home/aalbe03/.conda/envs/simulation_cluster/bin"
PY      = ENV_BIN + "/python"


def val_ids():
    ids = sorted(p.name for p in ODF66.iterdir() if p.is_dir())
    return [i for i in ids if i not in VAL_EXCLUDE]


def _env(cpus):
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["MRTRIX_NTHREADS"] = str(cpus)
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(cpus)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    env["PATH"] = ENV_BIN + ":" + env.get("PATH", "")
    return env


def _run(cmd, cwd, log, cpus, timeout):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        return subprocess.run(cmd, cwd=cwd, env=_env(cpus),
                              stdout=f, stderr=subprocess.STDOUT, timeout=timeout)


# ----- pooled per-subject units -----

def regen_one(job):
    tid, vsid, cpus = job["tid"], job["vsid"], job["cpus"]
    r65 = BASE_RUNS / tid / "registration" / "registration_65" / vsid
    gt = r65 / "tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz"
    if gt.exists():
        return (vsid, "skip")
    t1 = HCP / vsid / "T1w" / "T1w_acpc_dc_restore_brain.nii.gz"
    if not t1.exists():
        return (vsid, "no_t1")
    cmd = [
        "bash", str(SCRIPTS / "register_one_hcp_subject.sh"),
        "--subject-id", vsid,
        "--t1-hcp", str(t1),
        "--mni-brain", str(MNI),
        "--upenn-workdir", str(BASE_RUNS / tid / "registration" / "upenn_workdir"),
        "--cache-root", str(CACHE),
        "--outdir", str(r65),
    ]
    try:
        r = _run(cmd, None, r65 / "registration.log", cpus, TIMEOUT_LIGHT)
        return (vsid, "ok" if r.returncode == 0 else f"rc{r.returncode}")
    except subprocess.TimeoutExpired:
        return (vsid, "timeout")
    except Exception as e:
        return (vsid, f"exc:{e}")


def validate_one(job):
    tid, vsid, cpus = job["tid"], job["vsid"], job["cpus"]
    p = job["params"]; nat = job["native"]
    vout = BASE_RUNS / tid / "validation_results_null" / vsid
    out_nii = vout / f"analysis_participant_{vsid}.nii.gz"
    if out_nii.exists():
        return (vsid, "skip")
    r65 = BASE_RUNS / tid / "registration" / "registration_65" / vsid
    data = ODF66 / vsid / "odf_sh_order12_mrtrix.mif"
    gt = r65 / "tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz"
    tmpl = r65 / "T1_hcp_RAS.mif"
    for pth in (data, gt, tmpl):
        if not pth.exists():
            return (vsid, f"missing:{pth.name}")
    vout.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, "-u", str(SCRIPTS / "validate_single_run_currC.py"),
        "--data", str(data), "--template", str(tmpl), "--image", str(gt),
        "--output", str(vout / f"analysis_participant_{vsid}"),
        "--x", str(nat[0]), "--y", str(nat[1]), "--z", str(nat[2]),
        "--angle", str(p["angle"]), "--maxlen", str(p["maxlen"]), "--cutoff", str(p["cutoff"]),
        "--radius", "3", "--step", "0.5", "--threshold", "0.02",
        "--select", "10000", "--algorithm", "NullDist2",
        "--out-json", str(vout / "metrics.json"), "--out-csv", str(vout / "metrics.csv"),
    ]
    try:
        r = _run(cmd, str(vout), vout / "output.txt", cpus, TIMEOUT_VALIDATE)
        return (vsid, "ok" if r.returncode == 0 else f"rc{r.returncode}")
    except subprocess.TimeoutExpired:
        return (vsid, "timeout")
    except Exception as e:
        return (vsid, f"exc:{e}")


# ----- helpers -----

def read_median(med_csv):
    d = {}
    with open(med_csv) as f:
        for row in csv.reader(f):
            if len(row) == 2 and row[0] in ("angle", "maxlen", "cutoff", "x_mni", "y_mni", "z_mni"):
                try:
                    d[row[0]] = float(row[1])
                except ValueError:
                    pass
    return d


def read_native_coords(coords_csv):
    out = {}
    with open(coords_csv) as f:
        rows = list(csv.reader(f))
    if not rows:
        return out
    hdr = rows[0]; idx = {h: i for i, h in enumerate(hdr)}
    need = ("subject_id", "x_native", "y_native", "z_native")
    if not all(k in idx for k in need):
        return out
    for r in rows[1:]:
        if len(r) < len(hdr):
            continue
        out[r[idx["subject_id"]]] = (r[idx["x_native"]], r[idx["y_native"]], r[idx["z_native"]])
    return out


# ----- per-tumour orchestration (runs in main process) -----

def process_tumour(tid, workers, cpus, vids):
    run = BASE_RUNS / tid
    avg_dir = run / "avg_mni_outputs_null"
    avg_csv = avg_dir / "average_vs_tumor_metrics.csv"
    if avg_csv.exists():
        print(f"[{tid}] DONE (skip)", flush=True)
        return "skip_done"
    if not (run / "training_results").exists():
        print(f"[{tid}] no training_results -> skip", flush=True)
        return "no_inputs"

    # --- STAGE 4: collect training (stage_null) -> MNI ---
    out4 = run / "best_mni_outputs_null"
    med_csv = out4 / "best_params_all.median.csv"
    if not med_csv.exists():
        farm = run / "_null_train_farm"
        farm.mkdir(parents=True, exist_ok=True)
        nf = 0
        for sd in sorted((run / "training_results").iterdir()):
            if not sd.is_dir():
                continue
            scn = sd / "stage_null"
            bj = scn / f"analysis_participant_{sd.name}_best_params.json"
            if scn.is_dir() and bj.exists():
                ln = farm / sd.name
                if not ln.exists() and not ln.is_symlink():
                    ln.symlink_to(scn)
                nf += 1
        print(f"[{tid}] stage4: farmed {nf} train subjects -> collect", flush=True)
        cmd = [
            PY, str(SCRIPTS / "collect_best_to_mni_3.py"), "--mode", "training",
            "--train-root", str(farm), "--reg-root", str(CACHE),
            "--out-dir", str(out4), "--out-csv", str(out4 / "best_params_all.csv"),
        ]
        r = _run(cmd, None, out4 / "collect_train.log", cpus, TIMEOUT_COLLECT)
        if r.returncode != 0 or not med_csv.exists():
            print(f"[{tid}] STAGE4 FAILED rc={r.returncode}", flush=True)
            return "stage4_fail"
    med = read_median(med_csv)
    if not all(k in med for k in ("angle", "maxlen", "cutoff", "x_mni", "y_mni", "z_mni")):
        print(f"[{tid}] median CSV incomplete: {med}", flush=True)
        return "median_bad"

    # --- regen registration_65 (pool) ---
    rjobs = [{"tid": tid, "vsid": v, "cpus": cpus} for v in vids]
    rc = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for vsid, st in ex.map(regen_one, rjobs):
            rc[st] = rc.get(st, 0) + 1
    print(f"[{tid}] regen65: {rc}", flush=True)

    # --- warp median MNI seed -> native (65 subj) ---
    r65 = run / "registration" / "registration_65"
    coords = run / "mni_to_native_coords_null.csv"
    cmd = [
        PY, str(SCRIPTS / "warp_mni_to_subject_coords.py"), "--reg-root", str(r65),
        "--x", str(med["x_mni"]), "--y", str(med["y_mni"]), "--z", str(med["z_mni"]),
        "--space", "hcp", "--out-csv", str(coords),
    ]
    r = _run(cmd, None, run / "warp_coords_null.log", cpus, TIMEOUT_LIGHT)
    if not coords.exists():
        print(f"[{tid}] WARP-COORDS FAILED rc={r.returncode}", flush=True)
        return "coords_fail"
    nat = read_native_coords(coords)

    # --- STAGE 5: validate x65 (pool) ---
    vjobs = []
    for v in vids:
        if v in nat:
            vjobs.append({
                "tid": tid, "vsid": v, "cpus": cpus,
                "params": {"angle": med["angle"], "maxlen": med["maxlen"], "cutoff": med["cutoff"]},
                "native": nat[v],
            })
    vc = {}; done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(validate_one, j): j for j in vjobs}
        for fut in as_completed(futs):
            vsid, st = fut.result()
            vc[st] = vc.get(st, 0) + 1; done += 1
            if st not in ("ok", "skip"):
                print(f"[{tid}]   {vsid}: {st}", flush=True)
            if done % 20 == 0:
                print(f"[{tid}] validate {done}/{len(vjobs)} {vc}", flush=True)
    print(f"[{tid}] validate: {vc}", flush=True)

    # --- STAGE 5b: collect validation -> MNI ---
    out5 = run / "validation_mni_outputs_null"
    cmd = [
        PY, str(SCRIPTS / "collect_best_to_mni_3.py"), "--mode", "validation",
        "--train-root", str(run / "validation_results_null"), "--reg-root", str(r65),
        "--out-dir", str(out5), "--out-csv", str(out5 / "validation_all.csv"),
        "--val-threshold", "0.02",
    ]
    r = _run(cmd, None, out5 / "collect_val.log", cpus, TIMEOUT_COLLECT)
    if r.returncode != 0:
        print(f"[{tid}] STAGE5b FAILED rc={r.returncode}", flush=True)
        return "stage5b_fail"

    # --- STAGE 6: averages + compare (reuse existing tumor_upenn_MNI) ---
    ref = run / "registration" / "tumor_upenn_MNI.nii.gz"
    cmd = [
        PY, str(SCRIPTS / "average_mni_across_participants.py"),
        "--train-root", str(out4), "--val-root", str(out5),
        "--out-dir", str(avg_dir), "--ref", str(ref),
    ]
    r = _run(cmd, None, avg_dir / "average.log", cpus, TIMEOUT_LIGHT)
    cmd = [
        PY, str(SCRIPTS / "compare_average_tumors_to_reference.py"),
        "--train-avg", str(avg_dir / "training_avg.nii.gz"),
        "--val-avg", str(avg_dir / "validation_avg.nii.gz"),
        "--reference", str(ref), "--threshold", "0.02",
        "--out-csv", str(avg_csv),
        "--save-thresholded-dir", str(avg_dir / "thresholded_avg_masks"),
    ]
    r = _run(cmd, None, avg_dir / "compare.log", cpus, TIMEOUT_LIGHT)
    ok = avg_csv.exists()
    print(f"[{tid}] DONE -> avg_csv={ok}", flush=True)
    return "ok" if ok else "finalize_fail"


def main():
    global ENV_BIN, PY
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--tumours", type=str, default=None, help="comma-separated tumour IDs")
    ap.add_argument("--env-bin", type=str, default=ENV_BIN, help="conda env bin dir for this box")
    args = ap.parse_args()
    ENV_BIN = args.env_bin
    PY = ENV_BIN + "/python"

    target = TEN if not args.tumours else [t.strip() for t in args.tumours.split(",")]
    vids = val_ids()
    print(f"env-bin={ENV_BIN}  workers={args.workers} cpus={args.cpus}", flush=True)
    print(f"tumours={len(target)}  val-subjects={len(vids)}", flush=True)

    summary = {}
    for i, tid in enumerate(target, 1):
        print(f"\n==== [{i}/{len(target)}] {tid} ====", flush=True)
        try:
            st = process_tumour(tid, args.workers, args.cpus, vids)
        except Exception as e:
            st = f"exc:{e}"
            print(f"[{tid}] EXCEPTION {e}", flush=True)
        summary[st] = summary.get(st, 0) + 1
    print(f"\nFINAL: {summary}", flush=True)


if __name__ == "__main__":
    main()

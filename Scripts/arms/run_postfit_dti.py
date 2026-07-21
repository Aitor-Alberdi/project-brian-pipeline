#!/usr/bin/env python3
"""DTI arm -- post-fit stages 4/5/5b/6 at --select 50000 (budget-matched to ODF and the
50k NULL run), 10-tumour subset, W1.

Reads the DTI stage fit from training_results/<sid>/stage_dti/ and runs:
  4   collect best (stage_dti) -> MNI training median            -> best_mni_outputs_dti/
      regen/reuse registration_65 (algorithm-independent; shared)
      warp median MNI seed -> native (65 hosts)
  5   validate single run x65  (tckgen Tensor_Prob, --select 50000, b=1000 DWI)
                                                                  -> validation_results_dti/<sid>/
  5b  collect validation -> MNI                                  -> validation_mni_outputs_dti/
  6   average (train+val) + compare to patient reference         -> avg_mni_outputs_dti/average_vs_tumor_metrics.csv

Per-tumour marker = avg_mni_outputs_dti/average_vs_tumor_metrics.csv (idempotent).
Usage: python -u run_postfit_dti.py --tumours <csv> --workers N --cpus M [--env-bin DIR]
"""
from __future__ import annotations
import os
import csv
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

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
DTI_ROOT  = Path("/home/Archive/BRIAN/All_30/dti_inputs")
VAL_EXCLUDE = {"756055"}

SELECT = "50000"

TIMEOUT_VALIDATE = 4 * 3600
TIMEOUT_LIGHT    = 1 * 3600
TIMEOUT_COLLECT  = 4 * 3600

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
        "--subject-id", vsid, "--t1-hcp", str(t1), "--mni-brain", str(MNI),
        "--upenn-workdir", str(BASE_RUNS / tid / "registration" / "upenn_workdir"),
        "--cache-root", str(CACHE), "--outdir", str(r65),
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
    vout = BASE_RUNS / tid / "validation_results_dti" / vsid
    out_nii = vout / f"analysis_participant_{vsid}.nii.gz"
    if out_nii.exists():
        return (vsid, "skip")
    r65 = BASE_RUNS / tid / "registration" / "registration_65" / vsid
    data = DTI_ROOT / vsid / "dwi_b1000_ras.mif"
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
        "--select", SELECT, "--algorithm", "Tensor_Prob",
        "--out-json", str(vout / "metrics.json"), "--out-csv", str(vout / "metrics.csv"),
    ]
    try:
        r = _run(cmd, str(vout), vout / "output.txt", cpus, TIMEOUT_VALIDATE)
        return (vsid, "ok" if r.returncode == 0 else f"rc{r.returncode}")
    except subprocess.TimeoutExpired:
        return (vsid, "timeout")
    except Exception as e:
        return (vsid, f"exc:{e}")


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


def process_tumour(tid, workers, cpus, vids):
    run = BASE_RUNS / tid
    avg_dir = run / "avg_mni_outputs_dti"
    avg_csv = avg_dir / "average_vs_tumor_metrics.csv"
    if avg_csv.exists():
        print(f"[{tid}] DONE (skip)", flush=True)
        return "skip_done"

    # --- STAGE 4: collect training (stage_dti) -> MNI training median ---
    out4 = run / "best_mni_outputs_dti"
    med_csv = out4 / "best_params_all.median.csv"
    if not med_csv.exists():
        farm = run / "_dti_train_farm"
        farm.mkdir(parents=True, exist_ok=True)
        nf = 0
        for sd in sorted((run / "training_results").iterdir()):
            if not sd.is_dir():
                continue
            scn = sd / "stage_dti"
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

    # --- registration_65 (reuse; algorithm-independent, mostly skips fast) ---
    rjobs = [{"tid": tid, "vsid": v, "cpus": cpus} for v in vids]
    rc = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for vsid, st in ex.map(regen_one, rjobs):
            rc[st] = rc.get(st, 0) + 1
    print(f"[{tid}] regen65: {rc}", flush=True)

    # --- warp median MNI seed -> native ---
    r65 = run / "registration" / "registration_65"
    coords = run / "mni_to_native_coords_dti.csv"
    cmd = [
        PY, str(SCRIPTS / "warp_mni_to_subject_coords.py"), "--reg-root", str(r65),
        "--x", str(med["x_mni"]), "--y", str(med["y_mni"]), "--z", str(med["z_mni"]),
        "--space", "hcp", "--out-csv", str(coords),
    ]
    r = _run(cmd, None, run / "warp_coords_dti.log", cpus, TIMEOUT_LIGHT)
    if not coords.exists():
        print(f"[{tid}] WARP-COORDS FAILED rc={r.returncode}", flush=True)
        return "coords_fail"
    nat = read_native_coords(coords)

    # --- STAGE 5: validate x65 at SELECT=50000 (pool) ---
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
    out5 = run / "validation_mni_outputs_dti"
    cmd = [
        PY, str(SCRIPTS / "collect_best_to_mni_3.py"), "--mode", "validation",
        "--train-root", str(run / "validation_results_dti"), "--reg-root", str(r65),
        "--out-dir", str(out5), "--out-csv", str(out5 / "validation_all.csv"),
        "--val-threshold", "0.02",
    ]
    r = _run(cmd, None, out5 / "collect_val.log", cpus, TIMEOUT_COLLECT)
    if r.returncode != 0:
        print(f"[{tid}] STAGE5b FAILED rc={r.returncode}", flush=True)
        return "stage5b_fail"

    # --- STAGE 6: averages + compare ---
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
    ap.add_argument("--env-bin", type=str, default=ENV_BIN)
    args = ap.parse_args()
    ENV_BIN = args.env_bin
    PY = ENV_BIN + "/python"

    target = TEN if not args.tumours else [t.strip() for t in args.tumours.split(",")]
    vids = val_ids()
    print(f"[postfit_dti] env-bin={ENV_BIN} workers={args.workers} cpus={args.cpus} SELECT={SELECT}", flush=True)
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

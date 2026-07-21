#!/usr/bin/env python3
"""Driver: NULL-DISTRIBUTION training run (full A->B->C from scratch) for the 10-tumour
subset x 20 train subjects.

Same optimizer (rand_seed_v2.py) and param ranges as the corrected stage_c_2 run, but
with --algorithm NullDist2 (tckgen ignores the FOD orientation -> streamlines from random
sampling only). Stage-C weights in rand_seed_v2.py are 0.20/0.60/0.20 (match latest run).

Per (tumour, subject):
  1. Invoke rand_seed_v2.py via subprocess with cwd=stage_null/  (no resume / no state.json)
  2. Parallel via ProcessPoolExecutor.
Idempotent: skip if stage_null/analysis_participant_<sid>_best.txt exists.

Resource split (disjoint tumour sets, shared NFS no locking):
  W1: --workers 6 --cpus 4 (24 threads), --env-bin /home/aalbe03/.conda/envs/simulation_cluster/bin
  W2: --workers 4 --cpus 4 (16 threads), --env-bin /home/cbajada/.conda/envs/simulation_cluster/bin
"""
from __future__ import annotations
import os
import sys
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 10-tumour subset: worst / average / best by validation Dice (stage_c_2 post-fit)
TEN = [
    # worst
    "UPENN-GBM-00001_11", "UPENN-GBM-00031_11", "UPENN-GBM-00024_11",
    # average
    "UPENN-GBM-00034_11", "UPENN-GBM-00019_11", "UPENN-GBM-00023_11", "UPENN-GBM-00020_11",
    # best
    "UPENN-GBM-00025_11", "UPENN-GBM-00032_11", "UPENN-GBM-00042_11",
]
TRAIN = ["100408", "101309", "101915", "103111", "103414", "103818", "105014", "105115",
         "108828", "110411", "113619", "113922", "114419", "115320", "116524", "117122",
         "118730", "118932", "120111", "122317"]
BASE_RUNS = Path("/home/Archive/BRIAN/All_30/runs")
ODF_ROOT = Path("/home/Archive/BRIAN/Analysis_HCP_UPENN/odf_generation_results_20")
RAND_SEED = Path("/home/Archive/BRIAN/proyect-brian-pipeline/rand_seed_v2.py")
DEFAULT_ENV_BIN = "/home/aalbe03/.conda/envs/simulation_cluster/bin"
DEFAULT_WORKERS = 6
DEFAULT_CPUS = 4
TIMEOUT_S = 8 * 3600   # 8h ceiling per subject (full A->B->C)


def run_one(job):
    tid = job["tumour"]; sid = job["subject"]; cpus = job["cpus"]; env_bin = job["env_bin"]
    py = env_bin + "/python"
    image = BASE_RUNS / tid / "registration" / "registration_20" / sid / "tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz"
    template = BASE_RUNS / tid / "registration" / "registration_20" / sid / "T1_hcp_RAS.mif"
    odf = ODF_ROOT / sid / "odf_sh_order12_mrtrix.mif"
    out_dir = BASE_RUNS / tid / "training_results" / sid / "stage_null"
    best_txt = out_dir / f"analysis_participant_{sid}_best.txt"

    if best_txt.exists():
        return {"tid": tid, "sid": sid, "status": "skip_done"}
    if not image.exists():
        return {"tid": tid, "sid": sid, "status": "no_image"}
    if not template.exists():
        return {"tid": tid, "sid": sid, "status": "no_template"}
    if not odf.exists():
        return {"tid": tid, "sid": sid, "status": "no_odf"}
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["MRTRIX_NTHREADS"] = str(cpus)
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(cpus)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PATH"] = env_bin + ":" + env.get("PATH", "")

    cmd = [
        py, str(RAND_SEED),
        "--data", str(odf),
        "--output", f"analysis_participant_{sid}",
        "-ang", "20,90,2", "-maxl", "20,90,2", "--radius", "3", "-st", "0.3,1,0.05",
        "--image", str(image),
        "--template", str(template),
        "-co", "0.04,0.1,0.005", "--threshold", "0.02", "--pos_window_mm", "5",
        "--patience_passes", "5", "--improve_tol", "0.005", "--target_obj", "0.1",
        "--optim-select", "10000", "-sl", "10000",
        "--algorithm", "NullDist2",
    ]
    try:
        with open(out_dir / "train.log", "w") as flog, open(out_dir / "train.err", "w") as ferr:
            r = subprocess.run(cmd, cwd=str(out_dir), env=env, stdout=flog, stderr=ferr, timeout=TIMEOUT_S)
        return {"tid": tid, "sid": sid,
                "status": "ok" if r.returncode == 0 else f"rc={r.returncode}"}
    except subprocess.TimeoutExpired:
        return {"tid": tid, "sid": sid, "status": "timeout"}
    except Exception as e:
        return {"tid": tid, "sid": sid, "status": f"exc:{e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    ap.add_argument("--env-bin", type=str, default=DEFAULT_ENV_BIN)
    ap.add_argument("--tumours", type=str, default=None,
                    help="comma-separated subset of tumour IDs (must be within the 10-tumour set)")
    args = ap.parse_args()

    target = TEN if not args.tumours else [t.strip() for t in args.tumours.split(",")]
    print(f"target tumours: {len(target)}  env_bin={args.env_bin}  workers={args.workers} cpus={args.cpus}")

    jobs = []
    for tid in target:
        for sid in TRAIN:
            jobs.append({"tumour": tid, "subject": sid, "cpus": args.cpus, "env_bin": args.env_bin})
    print(f"jobs to run: {len(jobs)}")

    counts = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            done += 1
            if r["status"] not in ("ok", "skip_done"):
                print(f"  {r['tid']} {r['sid']}: {r['status']}")
            if done % 20 == 0:
                print(f"  progress: {done}/{len(jobs)}  {counts}")
    print(f"FINAL counts: {counts}")


if __name__ == "__main__":
    main()

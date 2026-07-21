#!/usr/bin/env python3
"""Driver: fresh-start stage C with corrected weights for all 30 tumours x 20 train subj.
Per subject:
  1. Extract B-end state from existing simulation.csv -> stage_tensor/state.json
  2. Invoke patched rand_seed_v2_resume.py via subprocess with cwd=stage_tensor/
  3. Parallel via ProcessPoolExecutor, 8 workers x 5 cpus each.
Skip if inputs missing (batch1 registration_20 not pulled).
Idempotent: skip if stage_tensor/analysis_participant_<sid>_best.txt exists.
"""
from __future__ import annotations
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

ALL30 = [
    "UPENN-GBM-00001_11","UPENN-GBM-00002_11","UPENN-GBM-00003_11","UPENN-GBM-00005_11",
    "UPENN-GBM-00007_11","UPENN-GBM-00008_11","UPENN-GBM-00009_11","UPENN-GBM-00010_11",
    "UPENN-GBM-00011_11","UPENN-GBM-00014_11","UPENN-GBM-00016_11","UPENN-GBM-00018_11",
    "UPENN-GBM-00019_11","UPENN-GBM-00020_11","UPENN-GBM-00021_11","UPENN-GBM-00022_11",
    "UPENN-GBM-00023_11","UPENN-GBM-00024_11","UPENN-GBM-00025_11","UPENN-GBM-00028_11",
    "UPENN-GBM-00030_11","UPENN-GBM-00031_11","UPENN-GBM-00032_11","UPENN-GBM-00033_11",
    "UPENN-GBM-00034_11","UPENN-GBM-00035_11","UPENN-GBM-00036_11","UPENN-GBM-00037_11",
    "UPENN-GBM-00041_11","UPENN-GBM-00042_11",
]
TRAIN = ["100408","101309","101915","103111","103414","103818","105014","105115",
         "108828","110411","113619","113922","114419","115320","116524","117122",
         "118730","118932","120111","122317"]
BASE_RUNS = Path("/home/Archive/BRIAN/All_30/runs")
ODF_ROOT = Path("/home/Archive/BRIAN/All_30/tensor_odf_arm/odfs")
RAND_SEED = Path("/home/Archive/BRIAN/All_30/rand_seed_v2_resume.py")
PY = "/home/aalbe03/.conda/envs/simulation_cluster/bin/python"
ENV_BIN = "/home/aalbe03/.conda/envs/simulation_cluster/bin"
DEFAULT_WORKERS = 8
DEFAULT_CPUS = 5
TIMEOUT_S = 8 * 3600   # 4h ceiling per subject


def prep_state(tid, sid):
    sim = BASE_RUNS / tid / "training_results" / sid / "simulation.csv"
    if not sim.exists():
        return None, "no_sim_csv"
    try:
        df = pd.read_csv(sim)
    except Exception as e:
        return None, f"read_fail:{e}"
    if "stage" not in df.columns:
        return None, "no_stage_col"
    b = df[df["stage"] == "B"]
    if b.empty:
        return None, "no_B_rows"
    last = b.iloc[-1]
    init = df.iloc[0]
    state = {
        "parameters": {
            "x": float(last["x"]), "y": float(last["y"]), "z": float(last["z"]),
            "angle": float(last["angle"]), "maxlen": float(last["maxlen"]),
            "cutoff": float(last["cutoff"]),
            "power": float(last["power"]) if "power" in last else 0.0,
        },
        "EPSILON": {
            "dice_loss": 0.45,
            "sd_loss": float(last["eps_sd_loss"]) * 0.90,
            "vol_err": float(last["eps_vol_err"]) * 0.90,
        },
        "com": {
            "x": float(init["x"]), "y": float(init["y"]), "z": float(init["z"]),
        },
    }
    out_dir = sim.parent / "stage_tensor"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state_path.write_text(json.dumps(state, indent=2))
    return state_path, "ok"


def run_one(job):
    tid = job["tumour"]; sid = job["subject"]; cpus = job["cpus"]
    image = BASE_RUNS / tid / "registration" / "registration_20" / sid / "tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz"
    template = BASE_RUNS / tid / "registration" / "registration_20" / sid / "T1_hcp_RAS.mif"
    odf = ODF_ROOT / sid / "odf_sh_order12_tensor_mrtrix.mif"
    out_dir = BASE_RUNS / tid / "training_results" / sid / "stage_tensor"
    state_path = out_dir / "state.json"
    best_txt = out_dir / f"analysis_participant_{sid}_best.txt"

    if best_txt.exists():
        return {"tid": tid, "sid": sid, "status": "skip_done"}
    if not state_path.exists():
        return {"tid": tid, "sid": sid, "status": "no_state"}
    if not image.exists():
        return {"tid": tid, "sid": sid, "status": "no_image"}
    if not template.exists():
        return {"tid": tid, "sid": sid, "status": "no_template"}
    if not odf.exists():
        return {"tid": tid, "sid": sid, "status": "no_odf"}

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["MRTRIX_NTHREADS"] = str(cpus)
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(cpus)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PATH"] = ENV_BIN + ":" + env.get("PATH", "")

    cmd = [
        PY, str(RAND_SEED),
        "--restart-from-state", str(state_path),
        "--data", str(odf),
        "--output", f"analysis_participant_{sid}",
        "-ang", "20,90,2", "-maxl", "20,90,2", "--radius", "3", "-st", "0.3,1,0.05",
        "--image", str(image),
        "--template", str(template),
        "-co", "0.04,0.1,0.005", "--threshold", "0.02", "--pos_window_mm", "5",
        "--patience_passes", "5", "--improve_tol", "0.005", "--target_obj", "0.1",
        "--optim-select", "10000", "-sl", "50000",
    ]
    try:
        with open(out_dir / "resume.log", "w") as flog, open(out_dir / "resume.err", "w") as ferr:
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
    ap.add_argument("--tumours", type=str, default=None,
                    help="comma-separated subset of tumour IDs")
    args = ap.parse_args()

    target = ALL30 if not args.tumours else [t.strip() for t in args.tumours.split(",")]
    print(f"target tumours: {len(target)}")

    jobs = []; skipped = []
    for tid in target:
        for sid in TRAIN:
            sp, msg = prep_state(tid, sid)
            if sp is None:
                skipped.append((tid, sid, msg))
                continue
            jobs.append({"tumour": tid, "subject": sid, "cpus": args.cpus})
    print(f"jobs to run: {len(jobs)}  pre-skip: {len(skipped)} ({sum(1 for s in skipped if s[2]!='no_sim_csv')} non-trivial)")

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
    print(f"pre-skip statuses: {sorted(set(s[2] for s in skipped))}")


if __name__ == "__main__":
    main()

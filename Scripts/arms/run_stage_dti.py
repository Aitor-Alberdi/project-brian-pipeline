#!/usr/bin/env python3
"""DTI arm -- stage fit (A->B->C curriculum), 10-tumour subset, W1.

Same optimizer (rand_seed_v2.py) and curriculum/param ranges as the corrected stage_c_2
and the NULL run, but:
  --data       = the single-shell b=1000 DWI (dti_inputs/<sid>/dwi_b1000_ras.mif)
  --algorithm  = Tensor_Prob   (probabilistic tensor tractography; the right analogue of
                 the probabilistic iFOD2 for an ensemble/stochastic growth simulator.
                 Tensor_Det would collapse the ensemble.)
Stage-C weights in rand_seed_v2.py are 0.20/0.60/0.20 (match latest run).

NOTE on -cutoff: for iFOD2/NullDist2 the cutoff is an FOD-amplitude threshold; for the
Tensor_* algorithms tckgen interprets -cutoff as an FA termination threshold. The range
is therefore widened to an FA-appropriate band so streamlines are not killed inside the
low-FA tumour/oedema tissue; the optimizer searches within it.

Output tree per (tumour, train host): training_results/<sid>/stage_dti/
Marker = analysis_participant_<sid>_best.txt  (idempotent skip).
"""
from __future__ import annotations
import os
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

TEN = [
    "UPENN-GBM-00001_11", "UPENN-GBM-00031_11", "UPENN-GBM-00024_11",
    "UPENN-GBM-00034_11", "UPENN-GBM-00019_11", "UPENN-GBM-00023_11", "UPENN-GBM-00020_11",
    "UPENN-GBM-00025_11", "UPENN-GBM-00032_11", "UPENN-GBM-00042_11",
]
TRAIN = ["100408", "101309", "101915", "103111", "103414", "103818", "105014", "105115",
         "108828", "110411", "113619", "113922", "114419", "115320", "116524", "117122",
         "118730", "118932", "120111", "122317"]
BASE_RUNS = Path("/home/Archive/BRIAN/All_30/runs")
DTI_ROOT  = Path("/home/Archive/BRIAN/All_30/dti_inputs")
RAND_SEED = Path("/home/Archive/BRIAN/proyect-brian-pipeline/rand_seed_v2.py")
DEFAULT_ENV_BIN = "/home/aalbe03/.conda/envs/simulation_cluster/bin"
DEFAULT_WORKERS = 6
DEFAULT_CPUS = 4
TIMEOUT_S = 8 * 3600   # 8h ceiling per subject (full A->B->C)

# FA-appropriate cutoff band for Tensor_* tracking (cf. FOD-amp band 0.04,0.1,0.005 for iFOD2/null)
CUTOFF_RANGE = "0.05,0.2,0.01"


def run_one(job):
    tid = job["tumour"]; sid = job["subject"]; cpus = job["cpus"]; env_bin = job["env_bin"]
    py = env_bin + "/python"
    image = BASE_RUNS / tid / "registration" / "registration_20" / sid / "tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz"
    template = BASE_RUNS / tid / "registration" / "registration_20" / sid / "T1_hcp_RAS.mif"
    dwi = DTI_ROOT / sid / "dwi_b1000_ras.mif"
    out_dir = BASE_RUNS / tid / "training_results" / sid / "stage_dti"
    best_txt = out_dir / f"analysis_participant_{sid}_best.txt"

    if best_txt.exists():
        return {"tid": tid, "sid": sid, "status": "skip_done"}
    if not image.exists():
        return {"tid": tid, "sid": sid, "status": "no_image"}
    if not template.exists():
        return {"tid": tid, "sid": sid, "status": "no_template"}
    if not dwi.exists():
        return {"tid": tid, "sid": sid, "status": "no_dwi"}
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
        "--data", str(dwi),
        "--output", f"analysis_participant_{sid}",
        "-ang", "20,90,2", "-maxl", "20,90,2", "--radius", "3", "-st", "0.3,1,0.05",
        "--image", str(image),
        "--template", str(template),
        "-co", CUTOFF_RANGE, "--threshold", "0.02", "--pos_window_mm", "5",
        "--patience_passes", "5", "--improve_tol", "0.005", "--target_obj", "0.1",
        "--optim-select", "10000", "-sl", "10000",
        "--algorithm", "Tensor_Prob",
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
    ap.add_argument("--tumours", type=str, default=None, help="comma-separated tumour IDs")
    ap.add_argument("--env-bin", type=str, default=DEFAULT_ENV_BIN)
    args = ap.parse_args()

    target = TEN if not args.tumours else [t.strip() for t in args.tumours.split(",")]
    jobs = [{"tumour": t, "subject": s, "cpus": args.cpus, "env_bin": args.env_bin}
            for t in target for s in TRAIN]
    print(f"[stage_dti] tumours={len(target)} train-hosts={len(TRAIN)} jobs={len(jobs)} "
          f"workers={args.workers} cpus={args.cpus} algorithm=Tensor_Prob cutoff={CUTOFF_RANGE}", flush=True)

    summary = {}; done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            st = res["status"]; summary[st] = summary.get(st, 0) + 1; done += 1
            if st not in ("ok", "skip_done"):
                print(f"  {res['tid']} {res['sid']}: {st}", flush=True)
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)} {summary}", flush=True)
    print(f"\nFINAL: {summary}", flush=True)


if __name__ == "__main__":
    main()

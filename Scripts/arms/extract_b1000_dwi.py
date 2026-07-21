#!/usr/bin/env python3
"""Extract the single-shell b=1000 (+b=0) DWI for every host needed by the DTI arm,
embedding the gradient scheme, on the same world grid as the dODF field.

DTI tractography (tckgen -algorithm Tensor_Det/Tensor_Prob) needs the DWI as input
(not the SH dODF). Per supervisor: restrict the tensor fit to b<=1000 (the b=2000/3000
HCP shells violate the Gaussian/mono-exponential tensor assumption).

Source : /home/Archive/HCP_1200_qc_data/<sid>/T1w/Diffusion/{data.nii.gz,bvals,bvecs}
Output : DTI_ROOT/<sid>/dwi_b1000_ras.mif  (strides 1,2,3,4 to match odf_sh_order12_mrtrix.mif)

The HCP T1w/Diffusion volume is already in T1w(acpc) world space (same as the dODF, the
template T1_hcp_RAS.mif and the seed coordinates), so no spatial resampling is performed;
we only reorder strides to match the dODF and let MRtrix carry the gradient scheme in
world coordinates. Idempotent: skips a host whose output already exists.
"""
from __future__ import annotations
import os
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

HCP      = Path("/home/Archive/HCP_1200_qc_data")
DTI_ROOT = Path("/home/Archive/BRIAN/All_30/dti_inputs")
ENV_BIN  = "/home/aalbe03/.conda/envs/simulation_cluster/bin"

TRAIN = ["100408", "101309", "101915", "103111", "103414", "103818", "105014", "105115",
         "108828", "110411", "113619", "113922", "114419", "115320", "116524", "117122",
         "118730", "118932", "120111", "122317"]
ODF66 = Path("/home/Archive/BRIAN/Analysis_HCP_UPENN/odf_generation_results_66")
VAL_EXCLUDE = {"756055"}


def all_hosts():
    val = [p.name for p in ODF66.iterdir() if p.is_dir() and p.name not in VAL_EXCLUDE]
    return sorted(set(TRAIN) | set(val))


def _env(cpus):
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["MRTRIX_NTHREADS"] = str(cpus)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    env["PATH"] = ENV_BIN + ":" + env.get("PATH", "")
    return env


def extract_one(job):
    sid, cpus = job["sid"], job["cpus"]
    diff = HCP / sid / "T1w" / "Diffusion"
    data, bvals, bvecs = diff / "data.nii.gz", diff / "bvals", diff / "bvecs"
    out_dir = DTI_ROOT / sid
    out_mif = out_dir / "dwi_b1000_ras.mif"
    if out_mif.exists():
        return (sid, "skip")
    for p in (data, bvals, bvecs):
        if not p.exists():
            return (sid, f"missing:{p.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "extract.log"
    env = _env(cpus)
    try:
        with open(log, "w") as f:
            # extract b=0 and b=1000 shells, carry gradient scheme, match dODF strides
            r = subprocess.run(
                ["dwiextract", str(data), "-fslgrad", str(bvecs), str(bvals),
                 "-shells", "0,1000", "-strides", "1,2,3,4", str(out_mif), "-force"],
                env=env, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
        if r.returncode != 0 or not out_mif.exists():
            return (sid, f"rc{r.returncode}")
        # sanity: count retained volumes (should be ~108 = 18 b0 + 90 b1000)
        with open(log, "a") as f:
            subprocess.run(["mrinfo", str(out_mif), "-size"], env=env,
                           stdout=f, stderr=subprocess.STDOUT, timeout=120)
        return (sid, "ok")
    except subprocess.TimeoutExpired:
        return (sid, "timeout")
    except Exception as e:
        return (sid, f"exc:{e}")


def main():
    global ENV_BIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--hosts", type=str, default=None, help="comma-separated host IDs (default: all)")
    ap.add_argument("--env-bin", type=str, default=ENV_BIN)
    args = ap.parse_args()
    ENV_BIN = args.env_bin

    hosts = all_hosts() if not args.hosts else [h.strip() for h in args.hosts.split(",")]
    DTI_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[extract_b1000] hosts={len(hosts)} workers={args.workers} cpus={args.cpus} -> {DTI_ROOT}", flush=True)

    jobs = [{"sid": s, "cpus": args.cpus} for s in hosts]
    summary = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_one, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            sid, st = fut.result()
            summary[st] = summary.get(st, 0) + 1
            done += 1
            if st not in ("ok", "skip"):
                print(f"  {sid}: {st}", flush=True)
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)} {summary}", flush=True)
    print(f"FINAL: {summary}", flush=True)


if __name__ == "__main__":
    main()

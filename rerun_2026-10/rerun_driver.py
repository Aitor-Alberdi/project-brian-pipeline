#!/usr/bin/env python3
"""Rerun driver (15 Oct 2026). Writes ONLY inside RERUN_ROOT; never deletes or overwrites
anything outside it. Reads existing campaign files read-only.

Jobs:
  shore50k : re-run the retained stage_c_2 configuration of each (tumour, train host) at
             50 000 streamlines (the final re-run that never happened in June).
  nullC    : NullDist2 stage-C search restarted from the same SHORE B-end state as the
             SHORE (stage_c_2) and tensor runs, 10k during search, final best at 50k.

Usage: python3 rerun_driver.py {shore50k|nullC} --workers 5 --cpus 4 [--only TID:SID,...]
"""
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

B = Path("/mnt/bsc-das-01/BRIAN")
RERUN_ROOT = B / "rerun_2026-10-15"
ODF_ROOT = B / "Analysis_HCP_UPENN" / "odf_generation_results_20"
RESUME_ALGO = RERUN_ROOT / "scripts" / "rand_seed_v2_resume_algo.py"
ANTS_BIN = "/home/bobducks/anaconda3/bin"

ALL30 = ["UPENN-GBM-000%s_11" % n for n in
         "01 02 03 05 07 08 09 10 11 14 16 18 19 20 21 22 23 24 25 28 30 31 32 33 34 35 36 37 41 42".split()]
TEN = ["UPENN-GBM-000%s_11" % n for n in "01 31 24 34 19 23 20 25 32 42".split()]
TRAIN = ["100408", "101309", "101915", "103111", "103414", "103818", "105014", "105115",
         "108828", "110411", "113619", "113922", "114419", "115320", "116524", "117122",
         "118730", "118932", "120111", "122317"]


def runs_dir(tid):
    for b in ("Batch_1", "Batch_2"):
        p = B / b / "runs" / tid
        if p.is_dir():
            return p
    raise FileNotFoundError(tid)


def env_for(cpus):
    env = os.environ.copy()
    env["MRTRIX_NTHREADS"] = str(cpus)
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(cpus)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    tmp = RERUN_ROOT / "tmp"          # keep temp files off the system SSD
    env["TMPDIR"] = str(tmp)
    env["MRTRIX_TMPFILE_DIR"] = str(tmp)
    env["PATH"] = env["PATH"] + ":" + ANTS_BIN
    return env


def inputs(tid, sid):
    r = runs_dir(tid)
    reg = r / "registration" / "registration_20" / sid
    return {
        "image": reg / "tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz",
        # reg/T1_hcp_RAS.mif is a symlink to the old host path /home/Archive/...; use the real cache file
        "template": B / "Batch_2" / "cache" / "hcp_to_mni" / sid / "T1_hcp_RAS.mif",
        "odf": ODF_ROOT / sid / "odf_sh_order12_mrtrix.mif",
        "stage_c_2": r / "training_results" / sid / "stage_c_2",
    }


def job_shore50k(tid, sid, cpus):
    inp = inputs(tid, sid)
    out_dir = RERUN_ROOT / "shore50k" / tid / sid
    out_nii = out_dir / f"analysis_participant_{sid}_best50k.nii.gz"
    if out_nii.exists():
        return {"tid": tid, "sid": sid, "status": "skip_done", "secs": 0}
    bp = json.loads((inp["stage_c_2"] / f"analysis_participant_{sid}_best_params.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    tck = out_dir / f"analysis_participant_{sid}_best50k.tck"
    x, y, z = bp["x"], bp["y"], bp["z"]
    # identical to rand_seed_v2_resume.py run_simulation_with_given_parameters (radius 3)
    gen = ["tckgen", str(inp["odf"]), str(tck),
           "-seed_sphere", f"{x},{y},{z},3", "-exclude", f"{x},{y},{z},2",
           "-maxlength", f"{bp['maxlen']}", "-angle", f"{bp['angle']}",
           "-select", "50000", "-algorithm", "iFOD2", "-cutoff", f"{bp['cutoff']}"]
    mp = ["tckmap", str(tck), "-template", str(inp["template"]), str(out_nii)]
    env = env_for(cpus)
    t0 = time.time()
    with open(out_dir / "run.log", "w") as log:
        r1 = subprocess.run(gen, stdout=log, stderr=subprocess.STDOUT, env=env)
        if r1.returncode != 0:
            return {"tid": tid, "sid": sid, "status": f"tckgen_rc={r1.returncode}", "secs": time.time() - t0}
        r2 = subprocess.run(mp, stdout=log, stderr=subprocess.STDOUT, env=env)
    if r2.returncode != 0 or not out_nii.exists():
        return {"tid": tid, "sid": sid, "status": f"tckmap_rc={r2.returncode}", "secs": time.time() - t0}
    tck.unlink()  # our own temporary .tck inside RERUN_ROOT only
    (out_dir / "params_used.json").write_text(json.dumps(bp, indent=2))
    return {"tid": tid, "sid": sid, "status": "ok", "secs": time.time() - t0}


def job_nullC(tid, sid, cpus):
    inp = inputs(tid, sid)
    out_dir = RERUN_ROOT / "nullC" / tid / sid
    best_txt = out_dir / f"analysis_participant_{sid}_best.txt"
    if best_txt.exists():
        return {"tid": tid, "sid": sid, "status": "skip_done", "secs": 0}
    state = inp["stage_c_2"] / "state.json"   # SHORE B-end state, read only
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(RESUME_ALGO),
           "--restart-from-state", str(state),
           "--data", str(inp["odf"]),
           "--output", f"analysis_participant_{sid}",
           "-ang", "20,90,2", "-maxl", "20,90,2", "--radius", "3", "-st", "0.3,1,0.05",
           "--image", str(inp["image"]), "--template", str(inp["template"]),
           "-co", "0.04,0.1,0.005", "--threshold", "0.02", "--pos_window_mm", "5",
           "--patience_passes", "5", "--improve_tol", "0.005", "--target_obj", "0.1",
           "--optim-select", "10000", "-sl", "50000",
           "--algorithm", "NullDist2"]
    t0 = time.time()
    with open(out_dir / "resume.log", "w") as fl, open(out_dir / "resume.err", "w") as fe:
        r = subprocess.run(cmd, cwd=str(out_dir), env=env_for(cpus), stdout=fl, stderr=fe)
    ok = r.returncode == 0 and best_txt.exists()
    return {"tid": tid, "sid": sid, "status": "ok" if ok else f"rc={r.returncode}", "secs": time.time() - t0}


def job_nullFull(tid, sid, cpus):
    """Full NullDist2 search A->B->C from scratch, same arguments as the original SHORE search
    (run_full_tumor_pipeline.py), patience 10 in A/B and 5 in C as in stage_c_2, final
    configuration chosen among stage-C rows only, best re-run at 50k."""
    inp = inputs(tid, sid)
    out_dir = RERUN_ROOT / "nullFull" / tid / sid
    best_txt = out_dir / f"analysis_participant_{sid}_best.txt"
    if best_txt.exists():
        return {"tid": tid, "sid": sid, "status": "skip_done", "secs": 0}
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(RERUN_ROOT / "scripts" / "rand_seed_v2_nullfull.py"),
           "--data", str(inp["odf"]),
           "--output", f"analysis_participant_{sid}",
           "-ang", "20,90,2", "-maxl", "20,90,2", "--radius", "3", "-st", "0.3,1,0.05",
           "--image", str(inp["image"]), "--template", str(inp["template"]),
           "-co", "0.04,0.1,0.005", "--threshold", "0.02", "--pos_window_mm", "5",
           "--init_points", "10", "--init_percentiles", "0,25,50,75,100",
           "--patience_passes", "10", "--patience_c", "5",
           "--improve_tol", "0.005", "--target_obj", "0.1",
           "--optim-select", "10000", "-sl", "50000",
           "--algorithm", "NullDist2"]
    t0 = time.time()
    with open(out_dir / "output.txt", "w") as fl, open(out_dir / "output.err", "w") as fe:
        r = subprocess.run(cmd, cwd=str(out_dir), env=env_for(cpus), stdout=fl, stderr=fe)
    ok = r.returncode == 0 and best_txt.exists()
    return {"tid": tid, "sid": sid, "status": "ok" if ok else f"rc={r.returncode}", "secs": time.time() - t0}


def job_nullProto(tid, sid, cpus):
    """NullDist2 with exactly the protocol of the reported SHORE run:
    (1) A/B search with the campaign rand_seed_v2.py code and arguments (stopped on entering C);
    (2) B-end state built as run_stage_c_2.prep_state did for SHORE;
    (3) stage C with rand_seed_v2_resume (stage-C weights, stage-C-only table, patience 5),
        final best re-run at 50k."""
    import pandas as pd
    inp = inputs(tid, sid)
    ab_dir = RERUN_ROOT / "nullProto_AB" / tid / sid
    c_dir = RERUN_ROOT / "nullProto" / tid / sid
    best_txt = c_dir / f"analysis_participant_{sid}_best.txt"
    if best_txt.exists():
        return {"tid": tid, "sid": sid, "status": "skip_done", "secs": 0}
    t0 = time.time()
    env = env_for(cpus)
    ab_csv = ab_dir / "simulation.csv"
    if not ab_csv.exists():
        ab_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", str(RERUN_ROOT / "scripts" / "rand_seed_v2_nullAB.py"),
               "--data", str(inp["odf"]), "--output", f"analysis_participant_{sid}",
               "-ang", "20,90,2", "-maxl", "20,90,2", "--radius", "3", "-st", "0.3,1,0.05",
               "--image", str(inp["image"]), "--template", str(inp["template"]),
               "-co", "0.04,0.1,0.005", "--threshold", "0.02", "--pos_window_mm", "5",
               "--init_points", "10", "--init_percentiles", "0,25,50,75,100",
               "--patience_passes", "10", "--improve_tol", "0.005", "--target_obj", "0.1",
               "--optim-select", "10000", "-sl", "10000",
               "--algorithm", "NullDist2", "--stop_at_C"]
        with open(ab_dir / "output.txt", "w") as fl, open(ab_dir / "output.err", "w") as fe:
            r = subprocess.run(cmd, cwd=str(ab_dir), env=env, stdout=fl, stderr=fe)
        if r.returncode != 0 or not ab_csv.exists():
            return {"tid": tid, "sid": sid, "status": f"AB_rc={r.returncode}", "secs": time.time() - t0}
    # B-end state, identical to run_stage_c_2.prep_state
    df = pd.read_csv(ab_csv)
    b = df[df["stage"] == "B"]
    if b.empty:
        return {"tid": tid, "sid": sid, "status": "no_B_rows", "secs": time.time() - t0}
    last, init = b.iloc[-1], df.iloc[0]
    state = {
        "parameters": {"x": float(last["x"]), "y": float(last["y"]), "z": float(last["z"]),
                       "angle": float(last["angle"]), "maxlen": float(last["maxlen"]),
                       "cutoff": float(last["cutoff"]), "power": float(last["power"])},
        "EPSILON": {"dice_loss": 0.45, "sd_loss": float(last["eps_sd_loss"]) * 0.90,
                    "vol_err": float(last["eps_vol_err"]) * 0.90},
        "com": {"x": float(init["x"]), "y": float(init["y"]), "z": float(init["z"])},
    }
    c_dir.mkdir(parents=True, exist_ok=True)
    (c_dir / "state.json").write_text(json.dumps(state, indent=2))
    cmd = [sys.executable, str(RESUME_ALGO),
           "--restart-from-state", str(c_dir / "state.json"),
           "--data", str(inp["odf"]), "--output", f"analysis_participant_{sid}",
           "-ang", "20,90,2", "-maxl", "20,90,2", "--radius", "3", "-st", "0.3,1,0.05",
           "--image", str(inp["image"]), "--template", str(inp["template"]),
           "-co", "0.04,0.1,0.005", "--threshold", "0.02", "--pos_window_mm", "5",
           "--patience_passes", "5", "--improve_tol", "0.005", "--target_obj", "0.1",
           "--optim-select", "10000", "-sl", "50000", "--algorithm", "NullDist2"]
    with open(c_dir / "resume.log", "w") as fl, open(c_dir / "resume.err", "w") as fe:
        r = subprocess.run(cmd, cwd=str(c_dir), env=env, stdout=fl, stderr=fe)
    ok = r.returncode == 0 and best_txt.exists()
    return {"tid": tid, "sid": sid, "status": "ok" if ok else f"C_rc={r.returncode}", "secs": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["shore50k", "nullC", "nullFull", "nullProto"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--only", type=str, default=None)
    a = ap.parse_args()
    (RERUN_ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    fn = {"shore50k": job_shore50k, "nullC": job_nullC, "nullFull": job_nullFull,
          "nullProto": job_nullProto}[a.job]
    tumours = ALL30 if a.job == "shore50k" else TEN
    pairs = [(t, s) for t in tumours for s in TRAIN]
    if a.only:
        pairs = [tuple(p.split(":")) for p in a.only.split(",")]
    log_path = RERUN_ROOT / f"{a.job}_status.csv"
    new = not log_path.exists()
    with open(log_path, "a", newline="") as fh, ProcessPoolExecutor(max_workers=a.workers) as ex:
        w = csv.DictWriter(fh, fieldnames=["tid", "sid", "status", "secs", "finished"])
        if new:
            w.writeheader()
        futs = {ex.submit(fn, t, s, a.cpus): (t, s) for t, s in pairs}
        done = 0
        for f in as_completed(futs):
            t, s = futs[f]
            try:
                res = f.result()
            except Exception as e:
                res = {"tid": t, "sid": s, "status": f"exc:{e}", "secs": 0}
            res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            w.writerow(res); fh.flush()
            done += 1
            print(f"[{done}/{len(pairs)}] {res}", flush=True)


if __name__ == "__main__":
    main()

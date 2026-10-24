#!/usr/bin/env python3
"""Tensor dODF condition re-run on ori (17 Oct 2026), with the protocol of the reported SHORE run:
  (1) stages A/B with the campaign rand_seed_v2.py code (stopped on entering C),
  (2) B-end state built as run_stage_c_2.prep_state did for SHORE,
  (3) stage C with rand_seed_v2_resume (stage-C weights, stage-C-only table, patience 5),
      final best re-run at 50k,
  then validation x65 at 50k, MNI collection, averages and comparison.
Reads registrations/cache from Parallel_runs and tensor dODFs from malta_mirror (read only).
Writes ONLY inside ROOT. Never deletes anything outside ROOT.

Usage (inside a Slurm job):  python cluster_tensor.py all --workers 30 --cpus 2
                              python cluster_tensor.py train --only TID:SID --workers 1 --cpus 2
"""
import argparse, csv, json, os, re, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path("/opt/nfs/neuroinformatics/BRIAN/tensorProto_2026-10")
SCR = ROOT / "scripts"
RUNS = Path("/opt/users/aalbe03/Parallel_runs/runs")
CACHE = Path("/opt/users/aalbe03/Parallel_runs/cache/hcp_to_mni")
ODF = Path("/opt/nfs/neuroinformatics/BRIAN/malta_mirror/All_30/tensor_odf_arm/odfs")
ODF_NAME = "odf_sh_order12_tensor_mrtrix.mif"
PY = sys.executable

TEN = ["UPENN-GBM-000%s_11" % n for n in "01 31 24 34 19 23 20 25 32 42".split()]
TRAIN = ["100408", "101309", "101915", "103111", "103414", "103818", "105014", "105115",
         "108828", "110411", "113619", "113922", "114419", "115320", "116524", "117122",
         "118730", "118932", "120111", "122317"]


def env(cpus):
    e = os.environ.copy()
    e["MRTRIX_NTHREADS"] = str(cpus)
    e["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(cpus)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        e[k] = "1"
    e["TMPDIR"] = str(ROOT / "tmp")
    e["MRTRIX_TMPFILE_DIR"] = str(ROOT / "tmp")
    e["PATH"] = str(Path(PY).parent) + ":" + e["PATH"]
    return e


def run(cmd, log, cpus, cwd=None):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        return subprocess.run([str(c) for c in cmd], cwd=cwd, env=env(cpus),
                              stdout=f, stderr=subprocess.STDOUT)


def wait_for(p, timeout_h=12):
    t0 = time.time()
    while not Path(p).exists():
        if time.time() - t0 > timeout_h * 3600:
            raise TimeoutError(str(p))
        time.sleep(120)


def common_args(tid, sid):
    reg = RUNS / tid / "registration" / "registration_20" / sid
    return ["--data", ODF / sid / ODF_NAME, "--output", f"analysis_participant_{sid}",
            "-ang", "20,90,2", "-maxl", "20,90,2", "--radius", "3", "-st", "0.3,1,0.05",
            "--image", reg / "tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz",
            "--template", reg / "T1_hcp_RAS.mif",
            "-co", "0.04,0.1,0.005", "--threshold", "0.02", "--pos_window_mm", "5",
            "--improve_tol", "0.005", "--target_obj", "0.1", "--optim-select", "10000"]


def train_one(tid, sid, cpus):
    import pandas as pd
    ab = ROOT / "tensorProto_AB" / tid / sid
    c = ROOT / "tensorProto" / tid / sid
    best = c / f"analysis_participant_{sid}_best.txt"
    if best.exists():
        return tid, sid, "skip_done", 0
    t0 = time.time()
    wait_for(ODF / sid / ODF_NAME)
    if not (ab / "simulation.csv").exists():
        ab.mkdir(parents=True, exist_ok=True)
        r = run([PY, "-u", SCR / "rand_seed_v2_nullAB.py", *common_args(tid, sid),
                 "--init_points", "10", "--init_percentiles", "0,25,50,75,100",
                 "--patience_passes", "10", "-sl", "10000", "--algorithm", "iFOD2", "--stop_at_C"],
                ab / "output.txt", cpus, cwd=str(ab))
        if r.returncode != 0 or not (ab / "simulation.csv").exists():
            return tid, sid, f"AB_rc={r.returncode}", time.time() - t0
    df = pd.read_csv(ab / "simulation.csv")
    b = df[df["stage"] == "B"]
    if b.empty:
        return tid, sid, "no_B_rows", time.time() - t0
    last, init = b.iloc[-1], df.iloc[0]
    state = {"parameters": {k: float(last[k]) for k in ("x", "y", "z", "angle", "maxlen", "cutoff", "power")},
             "EPSILON": {"dice_loss": 0.45, "sd_loss": float(last["eps_sd_loss"]) * 0.90,
                         "vol_err": float(last["eps_vol_err"]) * 0.90},
             "com": {"x": float(init["x"]), "y": float(init["y"]), "z": float(init["z"])}}
    c.mkdir(parents=True, exist_ok=True)
    (c / "state.json").write_text(json.dumps(state, indent=2))
    r = run([PY, SCR / "rand_seed_v2_resume_algo.py", "--restart-from-state", c / "state.json",
             *common_args(tid, sid), "--patience_passes", "5", "-sl", "50000", "--algorithm", "iFOD2"],
            c / "resume.log", cpus, cwd=str(c))
    ok = r.returncode == 0 and best.exists()
    return tid, sid, ("ok" if ok else f"C_rc={r.returncode}"), time.time() - t0


def read_median(p):
    d = {}
    for row in csv.reader(open(p)):
        if len(row) == 2:
            try:
                d[row[0]] = float(row[1])
            except ValueError:
                pass
    return d


def validate_one(job):
    tid, v, cpus, med, nat, vroot = job
    vout = vroot / v
    out = vout / f"analysis_participant_{v}.nii.gz"
    if out.exists():
        return v, "skip"
    wait_for(ODF / v / ODF_NAME)
    r65 = RUNS / tid / "registration" / "registration_65" / v
    vout.mkdir(parents=True, exist_ok=True)
    r = run([PY, "-u", SCR / "validate_single_run_currC.py", "--data", ODF / v / ODF_NAME,
             "--template", r65 / "T1_hcp_RAS.mif", "--image", r65 / "tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz",
             "--output", vout / f"analysis_participant_{v}", "--x", nat[0], "--y", nat[1], "--z", nat[2],
             "--angle", med["angle"], "--maxlen", med["maxlen"], "--cutoff", med["cutoff"],
             "--radius", "3", "--step", "0.5", "--threshold", "0.02", "--select", "50000",
             "--algorithm", "iFOD2", "--out-json", vout / "metrics.json", "--out-csv", vout / "metrics.csv"],
            vout / "output.txt", cpus, cwd=str(vout))
    return v, ("ok" if r.returncode == 0 and out.exists() else f"rc{r.returncode}")


def post_one(tid, workers, cpus):
    o = ROOT / "tensor_post" / tid
    avg = o / "avg_mni_outputs"
    if (avg / "average_vs_tumor_metrics.csv").exists():
        return "skip_done"
    farm, out4 = o / "train_farm", o / "best_mni_outputs"
    for sid in TRAIN:
        sd = ROOT / "tensorProto" / tid / sid
        bt = sd / f"analysis_participant_{sid}_best.txt"
        br = re.search(r"Best run: (\S+?)Obj", bt.read_text()).group(1)
        fd = farm / sid
        fd.mkdir(parents=True, exist_ok=True)
        for name in (bt.name, f"analysis_participant_{sid}_best_params.json", "simulation.csv"):
            if not (fd / name).exists():
                (fd / name).symlink_to(sd / name)
        if not (fd / f"{br}.nii.gz").exists():
            (fd / f"{br}.nii.gz").symlink_to(sd / f"{br}_50k.nii.gz")
    med_csv = out4 / "best_params_all.median.csv"
    if not med_csv.exists():
        run([PY, SCR / "collect_best_to_mni_3.py", "--mode", "training", "--train-root", farm,
             "--reg-root", CACHE, "--out-dir", out4, "--out-csv", out4 / "best_params_all.csv"],
            out4 / "collect_train.log", cpus)
    med = read_median(med_csv)
    r65 = RUNS / tid / "registration" / "registration_65"
    coords = o / "mni_to_native_coords.csv"
    if not coords.exists():
        run([PY, SCR / "warp_mni_to_subject_coords.py", "--reg-root", r65, "--x", med["x_mni"],
             "--y", med["y_mni"], "--z", med["z_mni"], "--space", "hcp", "--out-csv", coords],
            o / "warp_coords.log", cpus)
    nat = {r["subject_id"]: (r["x_native"], r["y_native"], r["z_native"]) for r in csv.DictReader(open(coords))}
    vids = sorted(p.name for p in r65.iterdir() if p.is_dir())
    vroot = o / "validation_results"
    cnt = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(validate_one, (tid, v, cpus, med, nat[v], vroot)) for v in vids]):
            s = f.result()[1]
            cnt[s] = cnt.get(s, 0) + 1
    print(f"[{tid}] validate {cnt} of {len(vids)}", flush=True)
    if cnt.get("ok", 0) + cnt.get("skip", 0) != len(vids):
        return f"validate_incomplete:{cnt}"
    out5 = o / "validation_mni_outputs"
    run([PY, SCR / "collect_best_to_mni_3.py", "--mode", "validation", "--train-root", vroot,
         "--reg-root", r65, "--out-dir", out5, "--out-csv", out5 / "validation_all.csv", "--val-threshold", "0.02"],
        out5 / "collect_val.log", cpus)
    ref = RUNS / tid / "registration" / "tumor_upenn_MNI.nii.gz"
    run([PY, SCR / "average_mni_across_participants.py", "--train-root", out4, "--val-root", out5,
         "--out-dir", avg, "--ref", ref], avg / "average.log", cpus)
    run([PY, SCR / "compare_average_tumors_to_reference.py", "--train-avg", avg / "training_avg.nii.gz",
         "--val-avg", avg / "validation_avg.nii.gz", "--reference", ref, "--threshold", "0.02",
         "--out-csv", avg / "average_vs_tumor_metrics.csv", "--save-thresholded-dir", avg / "thresholded_avg_masks"],
        avg / "compare.log", cpus)
    return "ok" if (avg / "average_vs_tumor_metrics.csv").exists() else "stage6_fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["train", "post", "all"])
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--cpus", type=int, default=2)
    ap.add_argument("--only", default=None)
    ap.add_argument("--tumours", default=None)
    a = ap.parse_args()
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    tumours = a.tumours.split(",") if a.tumours else TEN
    if a.phase in ("train", "all"):
        pairs = [tuple(p.split(":")) for p in a.only.split(",")] if a.only else [(t, s) for t in tumours for s in TRAIN]
        log = ROOT / "train_status.csv"
        new = not log.exists()
        with open(log, "a", newline="") as fh, ProcessPoolExecutor(max_workers=a.workers) as ex:
            w = csv.writer(fh)
            if new:
                w.writerow(["tid", "sid", "status", "secs", "finished"])
            futs = [ex.submit(train_one, t, s, a.cpus) for t, s in pairs]
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    res = f.result()
                except Exception as e:
                    res = ("?", "?", f"exc:{e}", 0)
                w.writerow([*res[:3], int(res[3]), time.strftime("%Y-%m-%d %H:%M:%S")]); fh.flush()
                print(f"[train {i}/{len(pairs)}] {res}", flush=True)
    if a.phase in ("post", "all"):
        for tid in tumours:
            t0 = time.time()
            try:
                st = post_one(tid, a.workers, a.cpus)
            except Exception as e:
                st = f"exc:{e}"
            print(f"[post] {tid} -> {st} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

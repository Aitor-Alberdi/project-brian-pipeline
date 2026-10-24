#!/usr/bin/env python3
"""Post-fit stages for the 15 Oct 2026 reruns. Writes ONLY inside RERUN_ROOT and never
deletes anything outside it. Original campaign folders are read only.

shore : 50k training masks (shore50k/) -> MNI (stage 4), then averages with the EXISTING
        c2 validation MNI masks (validation_mni_outputs_c2, read only) and compare.
null  : nullC best configs (their *_50k masks) -> MNI + median (stage 4), median seed to
        each validation host, 65 validation runs at 50k with NullDist2, collect to MNI,
        averages and compare.

The registration folders link to the old host path /home/Archive/...; a mirror of links
with corrected targets is built under RERUN_ROOT/regmirror/.

Usage: python3 postfit_driver.py {shore|null} --workers 5 --cpus 4 [--tumours T1,T2]
"""
import argparse, csv, os, re, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

B = Path("/mnt/bsc-das-01/BRIAN")
RR = B / "rerun_2026-10-15"
SCRIPTS = B / "proyect-brian-pipeline" / "Scripts"
CACHE = B / "Batch_2" / "cache" / "hcp_to_mni"
ODF66 = B / "Analysis_HCP_UPENN" / "odf_generation_results_66"
VAL_EXCLUDE = {"756055"}
PY = sys.executable
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


def fix_target(t):
    m = re.match(r"/home/Archive/BRIAN/(?:All_30|Batch_1|Batch_2)/runs/(UPENN-GBM-\d+_11)/(.*)", t)
    if m:
        return str(runs_dir(m.group(1)) / m.group(2))
    return t.replace("/home/Archive/BRIAN", str(B))


def mirror_dir(src, dst):
    """dst/<name> -> real file for every entry of src (fixing old /home/Archive links)."""
    dst.mkdir(parents=True, exist_ok=True)
    for e in src.iterdir():
        link = dst / e.name
        if link.exists() or link.is_symlink():
            continue
        target = fix_target(os.readlink(e)) if e.is_symlink() else str(e)
        if not Path(target).exists():
            raise FileNotFoundError(f"{e} -> {target}")
        link.symlink_to(target)


def env(cpus):
    en = os.environ.copy()
    en["MRTRIX_NTHREADS"] = str(cpus)
    en["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(cpus)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        en[k] = "1"
    en["TMPDIR"] = str(RR / "tmp")
    en["MRTRIX_TMPFILE_DIR"] = str(RR / "tmp")
    en["PATH"] = en["PATH"] + ":" + ANTS_BIN
    return en


def run(cmd, log, cpus, cwd=None):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        return subprocess.run([str(c) for c in cmd], cwd=cwd, env=env(cpus), stdout=f, stderr=subprocess.STDOUT)


def make_farm(tid, src_root, farm, img_suffix):
    """farm/<sid>/: links to best.txt, best_params.json, simulation.csv of src_root/<sid>,
    plus <best_run>.nii.gz -> the 50k image, so collect_best_to_mni_3 picks the 50k mask."""
    n = 0
    for sid in TRAIN:
        sd = src_root / sid
        bt = sd / f"analysis_participant_{sid}_best.txt"
        if not bt.exists():
            continue
        best_run = re.search(r"Best run: (\S+?)Obj", bt.read_text()).group(1)
        img50 = img_suffix(tid, sid, sd, best_run)
        if not img50.exists():
            continue
        fd = farm / sid
        fd.mkdir(parents=True, exist_ok=True)
        for name in (bt.name, f"analysis_participant_{sid}_best_params.json", "simulation.csv"):
            if (sd / name).exists() and not (fd / name).exists():
                (fd / name).symlink_to(sd / name)
        if not (fd / f"{best_run}.nii.gz").exists():
            (fd / f"{best_run}.nii.gz").symlink_to(img50)
        n += 1
    return n


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
    tid, v, cpus, med, nat, r65m, vroot = job
    vout = vroot / v
    out_nii = vout / f"analysis_participant_{v}.nii.gz"
    if out_nii.exists():
        return v, "skip"
    vout.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", SCRIPTS / "validate_single_run_currC.py",
           "--data", ODF66 / v / "odf_sh_order12_mrtrix.mif",
           "--template", r65m / v / "T1_hcp_RAS.mif",
           "--image", r65m / v / "tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz",
           "--output", vout / f"analysis_participant_{v}",
           "--x", nat[0], "--y", nat[1], "--z", nat[2],
           "--angle", med["angle"], "--maxlen", med["maxlen"], "--cutoff", med["cutoff"],
           "--radius", "3", "--step", "0.5", "--threshold", "0.02",
           "--select", "50000", "--algorithm", "NullDist2",
           "--out-json", vout / "metrics.json", "--out-csv", vout / "metrics.csv"]
    r = run(cmd, vout / "output.txt", cpus, cwd=str(vout))
    return v, ("ok" if r.returncode == 0 and out_nii.exists() else f"rc{r.returncode}")


def stage4(tid, src_root, img_fn, out_root, cpus):
    farm = out_root / "train_farm"
    out4 = out_root / "best_mni_outputs"
    n = make_farm(tid, src_root, farm, img_fn)
    if n != 20:
        return None, f"farm_{n}of20"
    med_csv = out4 / "best_params_all.median.csv"
    if not med_csv.exists():
        run([PY, SCRIPTS / "collect_best_to_mni_3.py", "--mode", "training", "--train-root", farm,
             "--reg-root", CACHE, "--out-dir", out4, "--out-csv", out4 / "best_params_all.csv"],
            out4 / "collect_train.log", cpus)
    if not med_csv.exists():
        return None, "stage4_fail"
    return out4, "ok"


def stage6(tid, out4, out5, avg_dir, cpus):
    ref = runs_dir(tid) / "registration" / "tumor_upenn_MNI.nii.gz"
    run([PY, SCRIPTS / "average_mni_across_participants.py", "--train-root", out4, "--val-root", out5,
         "--out-dir", avg_dir, "--ref", ref], avg_dir / "average.log", cpus)
    run([PY, SCRIPTS / "compare_average_tumors_to_reference.py",
         "--train-avg", avg_dir / "training_avg.nii.gz", "--val-avg", avg_dir / "validation_avg.nii.gz",
         "--reference", ref, "--threshold", "0.02", "--out-csv", avg_dir / "average_vs_tumor_metrics.csv",
         "--save-thresholded-dir", avg_dir / "thresholded_avg_masks"], avg_dir / "compare.log", cpus)
    return (avg_dir / "average_vs_tumor_metrics.csv").exists()


def do_shore(tid, workers, cpus):
    out_root = RR / "shore_post" / tid
    avg_dir = out_root / "avg_mni_outputs_50k"
    if (avg_dir / "average_vs_tumor_metrics.csv").exists():
        return "skip_done"
    src = runs_dir(tid) / "training_results"
    src_c2 = RR / "shore_post" / tid / "src_c2"          # links to each host's stage_c_2 folder
    src_c2.mkdir(parents=True, exist_ok=True)
    for sid in TRAIN:
        if not (src_c2 / sid).exists():
            (src_c2 / sid).symlink_to(src / sid / "stage_c_2")
    img = lambda t, s, sd, br: RR / "shore50k" / t / s / f"analysis_participant_{s}_best50k.nii.gz"
    out4, st = stage4(tid, src_c2, img, out_root, cpus)
    if out4 is None:
        return st
    out5 = runs_dir(tid) / "validation_mni_outputs_c2"   # existing 50k validation masks (read only)
    return "ok" if stage6(tid, out4, out5, avg_dir, cpus) else "stage6_fail"


def do_null(tid, workers, cpus):
    out_root = RR / "null_post" / tid
    avg_dir = out_root / "avg_mni_outputs_nullC50k"
    if (avg_dir / "average_vs_tumor_metrics.csv").exists():
        return "skip_done"
    img = lambda t, s, sd, br: sd / f"{br}_50k.nii.gz"
    out4, st = stage4(tid, RR / "nullProto" / tid, img, out_root, cpus)
    if out4 is None:
        return st
    med = read_median(out4 / "best_params_all.median.csv")
    # registration_65 mirror with corrected links
    r65 = runs_dir(tid) / "registration" / "registration_65"
    r65m = RR / "regmirror" / tid / "registration_65"
    vids = sorted(p.name for p in ODF66.iterdir() if p.is_dir() and p.name not in VAL_EXCLUDE)
    for v in vids:
        mirror_dir(r65 / v, r65m / v)
    coords = out_root / "mni_to_native_coords.csv"
    if not coords.exists():
        run([PY, SCRIPTS / "warp_mni_to_subject_coords.py", "--reg-root", r65m,
             "--x", med["x_mni"], "--y", med["y_mni"], "--z", med["z_mni"], "--space", "hcp",
             "--out-csv", coords], out_root / "warp_coords.log", cpus)
    rows = list(csv.DictReader(open(coords)))
    nat = {r["subject_id"]: (r["x_native"], r["y_native"], r["z_native"]) for r in rows}
    vroot = out_root / "validation_results"
    jobs = [(tid, v, cpus, med, nat[v], r65m, vroot) for v in vids if v in nat]
    cnt = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(validate_one, j) for j in jobs]):
            v, s = f.result()
            cnt[s] = cnt.get(s, 0) + 1
    print(f"[{tid}] validate {cnt}", flush=True)
    if cnt.get("ok", 0) + cnt.get("skip", 0) != 65:
        return f"validate_incomplete:{cnt}"
    out5 = out_root / "validation_mni_outputs"
    run([PY, SCRIPTS / "collect_best_to_mni_3.py", "--mode", "validation", "--train-root", vroot,
         "--reg-root", r65m, "--out-dir", out5, "--out-csv", out5 / "validation_all.csv",
         "--val-threshold", "0.02"], out5 / "collect_val.log", cpus)
    return "ok" if stage6(tid, out4, out5, avg_dir, cpus) else "stage6_fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["shore", "null"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--tumours", default=None)
    a = ap.parse_args()
    (RR / "tmp").mkdir(parents=True, exist_ok=True)
    tumours = a.tumours.split(",") if a.tumours else (ALL30 if a.job == "shore" else TEN)
    for tid in tumours:
        t0 = time.time()
        try:
            st = (do_shore if a.job == "shore" else do_null)(tid, a.workers, a.cpus)
        except Exception as e:
            st = f"exc:{e}"
        print(f"[{tid}] {a.job} -> {st} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

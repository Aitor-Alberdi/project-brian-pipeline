"""Final per-host and aggregate metrics for the thesis (17 Oct 2026), from saved masks only.

Per host (validation runs): reference = transferred tumour with the seed sphere removed LOGICALLY
(gt & ~sphere; the as-run np.subtract version is also scored to check reproduction of metrics.json);
simulated mask = density >= 2 % of its maximum (the validator's rule).
Aggregates (MNI): host-frequency map > 0.02 against the complete MNI reference.

Metrics: dice, hd95 (mm) and Hausdorff agreement, surface Dice as implemented (mean of the two
directional fractions, voxel counts), STANDARD surface Dice at 2 mm (Nikolov et al. 2021, DeepMind
surface-distance: surfel-area weighted, pooled over both surfaces), volume match, worst-projection Dice.

usage: python metrics_final.py ROOT VALIDATOR OUT_PREFIX arm=dir [arm=dir ...]
  ROOT has <...>/UPENN-GBM-XXXXX_11/ folders (any depth); dir is the validation folder name
  (e.g. shore=validation_results_c2); aggregate folders are given as aggarm=avgdir.
"""
import sys, json, glob, os, importlib.util
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd
import surface_distance as SD
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(sys.argv[1]); VAL = Path(sys.argv[2]); OUT = sys.argv[3]
SPECS = dict(a.split("=", 1) for a in sys.argv[4:])
spec = importlib.util.spec_from_file_location("v", VAL); v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)


def tumour_dirs():
    return sorted({Path(p).parent for p in glob.glob(str(ROOT / "**" / "UPENN-GBM-*_11" / "registration"), recursive=True)})


def sphere(shape, affine, x, y, z, radius=3):
    r = radius / 2.0
    c = nib.affines.apply_affine(np.linalg.inv(affine), (x, y, z))
    m = np.zeros(shape, bool)
    lo = np.maximum(np.floor(c - r - 1).astype(int), 0); hi = np.minimum(np.ceil(c + r + 2).astype(int), np.array(shape))
    if np.all(hi > lo):
        X, Y, Z = np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij")
        m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = ((X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2) <= r**2
    return m


def crop(*masks):
    anyv = np.zeros_like(masks[0], bool)
    for m in masks:
        anyv |= m
    idx = np.argwhere(anyv)
    if len(idx) == 0:
        return tuple(slice(None) for _ in range(3))
    lo = np.maximum(idx.min(0) - 3, 0); hi = np.minimum(idx.max(0) + 4, np.array(anyv.shape))
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def score(pred, gt, sp):
    hd = v.hd95_mm(pred, gt, sp)
    def d2(a, b):
        return float(2 * np.logical_and(a, b).sum() / (a.sum() + b.sum() + 1e-10))
    if pred.any() and gt.any():
        sds = SD.compute_surface_distances(gt, pred, sp)
        sd_std = float(SD.compute_surface_dice_at_tolerance(sds, 2.0))
    else:
        sd_std = 0.0
    return {"dice": 1 - v.dice_loss(pred, gt), "hd95_mm": hd,
            "hd_agr": 1 - min(hd / 20.0, 1.0) if np.isfinite(hd) else 0.0,
            "sd_variant": v.surface_dice(pred, gt, sp, 2.0), "sd_standard": sd_std,
            "volmatch": 1 - min(v.volume_error_rel(pred, gt), 1.0),
            "signed_vol_err": float((pred.sum() - gt.sum()) / max(gt.sum(), 1)),
            "projworst": min(d2(np.any(pred, ax), np.any(gt, ax)) for ax in (2, 1, 0)),
            "pred_vox": int(pred.sum()), "gt_vox": int(gt.sum())}


def per_host(job):
    tdir, arm, vdir = job
    host = vdir.name
    try:
        mj = json.loads((vdir / "metrics.json").read_text()); p = mj["params"]
        gi = nib.load(str(tdir / "registration" / "registration_65" / host / "tumor14_in_HCP_T1grid_viaMNI_brain.nii.gz"))
        g = gi.get_fdata(); sp = tuple(float(s) for s in gi.header.get_zooms()[:3])
        sim = nib.load(str(vdir / f"analysis_participant_{host}.nii.gz")).get_fdata()
        sph = sphere(g.shape, gi.affine, p["x"], p["y"], p["z"], int(p.get("radius", 3)))
        gt = g.astype(bool)
        fixed, asrun = gt & ~sph, np.subtract(g, sph).astype(bool)
        pred = (sim / (sim.max() + 1e-10)) >= float(p.get("threshold", 0.02))
        sl = crop(gt, asrun, pred, sph)
        r = {"tumour": tdir.name[10:15], "arm": arm, "host": host, "stored_dice": mj["metrics_expanded"]["dice"],
             "stored_sd": mj["metrics_expanded"]["surface_dice"]}
        a = score(pred[sl], asrun[sl], sp)
        r["asrun_dice"], r["asrun_sd_variant"] = a["dice"], a["sd_variant"]
        r.update(score(pred[sl], fixed[sl], sp))
        return r
    except Exception as e:
        return {"tumour": tdir.name, "arm": arm, "host": host, "error": repr(e)}


def aggregate(job):
    tdir, arm, avgdir = job
    try:
        ri = nib.load(str(tdir / "registration" / "tumor_upenn_MNI.nii.gz")); gt = ri.get_fdata() > 0.5
        sp = tuple(float(s) for s in ri.header.get_zooms()[:3])
        out = []
        for pool in ("training", "validation"):
            f = tdir / avgdir / f"{pool}_avg.nii.gz"
            if not f.exists():
                continue
            pred = np.nan_to_num(nib.load(str(f)).get_fdata()) > 0.02
            sl = crop(gt, pred)
            r = {"tumour": tdir.name[10:15], "arm": arm, "pool": pool}
            r.update(score(pred[sl], gt[sl], sp)); out.append(r)
        return out
    except Exception as e:
        return [{"tumour": tdir.name, "arm": arm, "error": repr(e)}]


def main():
    host_jobs, agg_jobs = [], []
    for t in tumour_dirs():
        for arm, d in SPECS.items():
            if arm.startswith("agg"):
                if (t / d).is_dir():
                    agg_jobs.append((t, arm[3:], d))
            else:
                host_jobs += [(t, arm, m.parent) for m in sorted((t / d).glob("*/metrics.json"))]
    print(len(host_jobs), "host jobs,", len(agg_jobs), "aggregate jobs", flush=True)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NW", "6"))) as ex:
        rows = list(ex.map(per_host, host_jobs, chunksize=4))
        agg = [r for rs in ex.map(aggregate, agg_jobs) for r in rs]
    pd.DataFrame(rows).to_csv(f"{OUT}_perhost.csv", index=False)
    pd.DataFrame(agg).to_csv(f"{OUT}_aggregate.csv", index=False)
    df = pd.DataFrame(rows)
    if "asrun_dice" in df:
        print("reproduction max|asrun - stored|: dice %.2e  sd %.2e" % ((df.asrun_dice - df.stored_dice).abs().max(),
                                                                      (df.asrun_sd_variant - df.stored_sd).abs().max()))
    print("errors:", int(df.get("error", pd.Series(dtype=object)).notna().sum()))


if __name__ == "__main__":
    main()

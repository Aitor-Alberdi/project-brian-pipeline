"""Tumour-level contrasts (n = 10) for the thesis, as in Section mm_analysis:
one-sample t on the 10 per-tumour differences, Holm over the 5 metrics of each contrast,
Wilcoxon signed-rank (uncorrected), Cohen's d_z, unadjusted 95 % CI, n positive.
Estimator A: validation aggregate mask (host frequency > 0.02) vs MNI reference.
Estimator B: mean over the 65 hosts of the per-host differences (per-host metrics, fixed reference).
usage: python analysis/contrasts_final.py OUTDIR   (run from the repository root; reads results/*.csv)
The condition labels are shore, null and tensorNew; read the CSVs with keep_default_na=False,
or pandas turns "null" into NaN.
"""
import sys
import numpy as np, pandas as pd
from scipy import stats

OUT = sys.argv[1]
TEN = [1, 19, 20, 23, 24, 25, 31, 32, 34, 42]
MET = ["dice", "hd_agr", "sd_standard", "volmatch", "projworst"]

ph = pd.concat([pd.read_csv("results/shore_c2_perhost.csv", keep_default_na=False, na_values=[""]), pd.read_csv("results/rerun_perhost.csv", keep_default_na=False, na_values=[""]), pd.read_csv("results/tensornew_perhost.csv", keep_default_na=False, na_values=[""])])
ph["tumour"] = ph["tumour"].astype(int)
ag = pd.concat([pd.read_csv("results/shore_c2_aggregate.csv", keep_default_na=False, na_values=[""]), pd.read_csv("results/rerun_aggregate.csv", keep_default_na=False, na_values=[""]), pd.read_csv("results/tensornew_aggregate.csv", keep_default_na=False, na_values=[""])])
ag["tumour"] = ag["tumour"].astype(int)
ag = ag[(ag.pool == "validation") & ag.tumour.isin(TEN)]


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (m - k) * p[i])); adj[i] = run
    return adj


def contrast(get, a, b, label, metrics=MET):
    rows = []
    for m in metrics:
        d = np.array([get(a, t, m) - get(b, t, m) for t in TEN])
        tt = stats.ttest_1samp(d, 0)
        ci = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
        try:
            w = stats.wilcoxon(d).pvalue
        except ValueError:
            w = np.nan
        rows.append(dict(estimator=label, contrast=f"{a}-{b}", metric=m, mean_a=np.mean([get(a, t, m) for t in TEN]),
                         mean_b=np.mean([get(b, t, m) for t in TEN]), diff=d.mean(), lo=ci[0], hi=ci[1],
                         dz=d.mean() / d.std(ddof=1), p=tt.pvalue, wilcoxon=w, n_pos=int((d > 0).sum()),
                         n_neg=int((d < 0).sum()), per_tumour=" ".join(f"{x:+.3f}" for x in d)))
    for r, q in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = q
    return rows


def get_agg(arm, t, m):
    return float(ag[(ag.arm == arm) & (ag.tumour == t)][m].iloc[0])


def get_ph(arm, t, m):
    return float(ph[(ph.arm == arm) & (ph.tumour == t)][m].mean())


def get_ph_paired(a, b):
    def f(arm, t, m):
        return get_ph(arm, t, m)
    return f


rows = []
for a, b in (("shore", "null"), ("tensorNew", "null"), ("shore", "tensorNew")):
    rows += contrast(get_agg, a, b, "aggregate")
for a, b in (("shore", "null"), ("tensorNew", "null"), ("shore", "tensorNew")):
    rows += contrast(get_ph, a, b, "per-host")
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/contrasts_final.csv", index=False)
pd.set_option("display.width", 250)
print(df.drop(columns=["per_tumour"]).round(3).to_string())
print()
print(df[(df.metric == "dice")][["estimator", "contrast", "per_tumour"]].to_string())
# host counts
print(ph.groupby("arm").size())

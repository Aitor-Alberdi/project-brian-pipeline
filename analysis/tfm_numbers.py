"""All numbers and figures for the final TFM update (24 Oct 2026).
Run from the repository root. Inputs: results/*.csv. If the averaged MNI maps are available under sdata/
(not distributed), the threshold sweep is recomputed from them; otherwise results/threshold_sweep.csv is used.
Outputs: tfm_out/numbers.txt, tfm_out/*.tex rows, tfm_out/{cohort_box,threshold_sweep,arms,null_forest}.pdf
Per-host scores use the logical reference subtraction; surface Dice is the standard (Nikolov) one at 2 mm.
"""
import glob, re
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

O = Path("tfm_out"); O.mkdir(exist_ok=True)
R = lambda f: pd.read_csv(f, keep_default_na=False, na_values=[""])
TEN = [1, 19, 20, 23, 24, 25, 31, 32, 34, 42]
MET = ["dice", "hd_agr", "sd_standard", "volmatch", "projworst"]
out = open(O / "numbers.txt", "w", encoding="utf8")
def P(*a):
    print(*a); print(*a, file=out)

ph = pd.concat([R("results/shore_c2_perhost.csv"), R("results/rerun_perhost.csv"), R("results/tensornew_perhost.csv")])
ph["tumour"] = ph.tumour.astype(int)
ag = pd.concat([R("results/shore_c2_aggregate.csv"), R("results/rerun_aggregate.csv"), R("results/tensornew_aggregate.csv")])
ag["tumour"] = ag.tumour.astype(int)
ag = ag.drop_duplicates(["arm", "pool", "tumour"])

# ---------------- main campaign, per host ----------------
s = ph[ph.arm == "shore"]
P("n shore runs", len(s), "tumours", s.tumour.nunique())
for m in MET + ["hd95_mm", "sd_variant"]:
    q = s[m].quantile([.25, .5, .75]).values
    P(f"pooled {m}: median {q[1]:.3f} IQR [{q[0]:.3f}, {q[2]:.3f}]")
pt = s.groupby("tumour")[MET + ["sd_variant", "hd95_mm"]].median()
for m in MET + ["sd_variant"]:
    q = pt[m].quantile([.25, .5, .75]).values
    P(f"median of per-tumour medians {m}: {q[1]:.3f} IQR [{q[0]:.3f}, {q[2]:.3f}]")
P("tumours dice>0.70:", (pt.dice > .70).sum(), ">0.75:", (pt.dice > .75).sum())
P("max per-tumour IQR dice:", s.groupby("tumour").dice.apply(lambda x: x.quantile(.75) - x.quantile(.25)).max().round(3))
for t in [1, 5, 18, 24, 28, 31, 35, 20, 30]:
    r = pt.loc[t]; P(f"UPENN {t:05d}: " + " ".join(f"{m}={r[m]:.3f}" for m in MET + ["sd_variant", "hd95_mm"]))
P("lowest per-tumour Dice:", pt.dice.sort_values().head(6).round(3).to_dict())
P("lowest per-tumour SD std:", pt.sd_standard.sort_values().head(6).round(3).to_dict())
P("highest SD std:", pt.sd_standard.sort_values().tail(3).round(3).to_dict())

# Table E.2 rows (volumes from the current table)
vol = {int(r.tumour): f"{r.volume_cm3:.1f}" for r in R("results/tumour_volumes.csv").itertuples()}
rows = []
for t in sorted(pt.index):
    r = pt.loc[t]
    rows.append(f"        UPENN-GBM-{t:05d} & {vol[t]} & " + " & ".join(f"{r[m]:.3f}" for m in MET) + r" \\")
med = pt[MET].median()
rows.append(r"        \midrule")
rows.append(r"        \multicolumn{2}{l}{\textbf{Median of the 30 tumours}} & " + " & ".join(f"{med[m]:.3f}" for m in MET) + r" \\")
(O / "tableE2_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf8")

# ---------------- averaged masks (training at 50k, validation) ----------------
tr = ag[(ag.arm == "shore50k") & (ag.pool == "training")].set_index("tumour")
va = ag[(ag.arm == "shore") & (ag.pool == "validation")].set_index("tumour")
va50 = ag[(ag.arm == "shore50k") & (ag.pool == "validation")].set_index("tumour")
P("validation c2 vs shore50k max diff dice:", (va.dice - va50.dice.reindex(va.index)).abs().max())
rows = []
for t in sorted(va.index):
    rows.append(f"        UPENN-GBM-{t:05d} & {tr.loc[t,'dice']:.3f} & {va.loc[t,'dice']:.3f} & {tr.loc[t,'hd_agr']:.3f} & {va.loc[t,'hd_agr']:.3f} & {tr.loc[t,'volmatch']:.3f} & {va.loc[t,'volmatch']:.3f}" + r" \\")
rows.append(r"        \midrule")
rows.append(r"        \multicolumn{1}{l}{\textbf{Cohort median}} & " + f"{tr.dice.median():.3f} & {va.dice.median():.3f} & {tr.hd_agr.median():.3f} & {va.hd_agr.median():.3f} & {tr.volmatch.median():.3f} & {va.volmatch.median():.3f}" + r" \\")
(O / "tableE1_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf8")
P("avg medians train50k: dice %.3f hd %.3f vol %.3f sd %.3f | val: dice %.3f hd %.3f vol %.3f sd %.3f" % (
    tr.dice.median(), tr.hd_agr.median(), tr.volmatch.median(), tr.sd_standard.median(),
    va.dice.median(), va.hd_agr.median(), va.volmatch.median(), va.sd_standard.median()))
P("val > train dice in", int((va.dice > tr.dice.reindex(va.index)).sum()), "of 30; hd in", int((va.hd_agr > tr.hd_agr.reindex(va.index)).sum()),
  "; volmatch in", int((va.volmatch > tr.volmatch.reindex(va.index)).sum()))
d = (tr.dice - va.dice.reindex(tr.index)).sort_values()
P("train-val dice largest:", d.tail(5).round(3).to_dict())
P("train volmatch zero in", int((tr.volmatch == 0).sum()), "tumours")

# ---------------- threshold sweep ----------------
THR = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-10)
sw = []
TDIRS = sorted(glob.glob("sdata/Batch_*/runs/UPENN-GBM-*_11"))
for tdir in TDIRS:
    t = int(Path(tdir).name[10:15])
    gt = nib.load(f"{tdir}/registration/tumor_upenn_MNI.nii.gz").get_fdata() > 0.5
    for pool in ("training", "validation"):
        a = np.nan_to_num(nib.load(f"{tdir}/avg_mni_outputs_shore50k/{pool}_avg.nii.gz").get_fdata())
        for th in THR:
            sw.append(dict(tumour=t, pool=pool, thr=th, dice=dice(a > th, gt)))
sw = pd.DataFrame(sw) if TDIRS else R("results/threshold_sweep.csv")
sw.to_csv(O / "sweep.csv", index=False)
g = sw.groupby(["pool", "thr"]).dice.quantile([.25, .5, .75]).unstack()
P("sweep check at 0.02 (should equal table): train %.3f val %.3f" % (g.loc[("training", 0.02), 0.5], g.loc[("validation", 0.02), 0.5]))
P(g.round(3).to_string())
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for pool, col, lab in (("validation", "#0072B2", "Validation average (65 hosts)"), ("training", "#D55E00", "Training average (20 hosts)")):
    gg = g.loc[pool]
    ax.fill_between(THR, gg[0.25], gg[0.75], color=col, alpha=0.12, lw=0)
    ax.plot(THR, gg[0.5], "-o", color=col, ms=4.5, lw=2, label=lab)
for x, lab in ((0.02, "0.02 (tables)"), (0.5, "0.5 (display)")):
    ax.axvline(x, ls="--", color="#444", lw=1.2); ax.text(x + 0.012, 0.842, lab, color="#444", fontsize=9)
ax.set_xlim(0, 0.82); ax.set_ylim(0.5, 0.85); ax.grid(alpha=.3)
ax.set_xlabel("threshold: fraction of contributing hosts that include a voxel"); ax.set_ylabel("cohort-median Dice")
ax.spines[["top", "right"]].set_visible(False); ax.legend(loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(O / "threshold_sweep.pdf"); plt.close(fig)

# ---------------- cohort box ----------------
lab = ["Dice", "Hausdorff\nagreement", "Surface Dice\n(2 mm)", "Volume\nmatch", "Worst-projection\nDice"]
fig, ax = plt.subplots(figsize=(7.3, 3.9))
data = [s[m].values for m in MET]
bp = ax.boxplot(data, widths=0.5, patch_artist=True, whis=1.5,
                medianprops=dict(color="k", lw=2), boxprops=dict(facecolor="#d1e3f0", edgecolor="#36607f", lw=1.3),
                whiskerprops=dict(color="#36607f", lw=1.3), capprops=dict(color="#36607f", lw=1.3),
                flierprops=dict(marker="o", ms=2.2, mfc="#36607f", mec="none", alpha=.5))
for i, x in enumerate(data, 1):
    ax.text(i, 1.05, f"{np.median(x):.3f}", ha="center", fontsize=9.5)
ax.set_xticks(range(1, 6)); ax.set_xticklabels(lab, fontsize=9.5); ax.set_ylim(-0.02, 1.1)
ax.set_ylabel("score (higher is better)"); ax.grid(axis="y", alpha=.3); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(O / "cohort_box.pdf"); plt.close(fig)

# ---------------- arms ----------------
arms = {"null": ("Isotropic null", "#9e9e9e"), "shore": ("SHORE dODF", "#0072B2"), "tensorNew": ("Tensor dODF", "#E69F00")}
v = ag[ag.pool == "validation"]
A = {a: v[v.arm == a].set_index("tumour").reindex(TEN) for a in arms}
for a in arms:
    P(f"arm {a} means: " + " ".join(f"{m}={A[a][m].mean():.3f}" for m in MET + ["hd95_mm", "signed_vol_err", "sd_variant"]))
fig = plt.figure(figsize=(7.2, 4.2)); gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1])
ax = fig.add_subplot(gs[0]); x = np.arange(10); w = 0.27
for k, (a, (nm, c)) in enumerate(arms.items()):
    ax.bar(x + (k - 1) * w, A[a].dice.values, w, color=c, label=nm)
nm_ = A["null"].dice.mean()
ax.axhline(nm_, ls="--", color="#555", lw=1.2, label=f"null mean ({nm_:.3f})")
ax.set_xticks(x); ax.set_xticklabels([f"{t:05d}" for t in TEN], rotation=45, ha="right")
ax.set_ylabel("Dice, validation aggregate mask"); ax.set_ylim(0, 0.95); ax.grid(axis="y", alpha=.3)
ax.spines[["top", "right"]].set_visible(False)
fig.legend(loc="upper center", ncol=4, frameon=False, fontsize=8.5)
ax2 = fig.add_subplot(gs[1])
bp = ax2.boxplot([A[a].dice.values for a in arms], widths=0.55, patch_artist=True, whis=1.5, showfliers=False,
                 medianprops=dict(color="k", lw=2))
for patch, (a, (nm, c)) in zip(bp["boxes"], arms.items()):
    patch.set_facecolor(c); patch.set_alpha(.75)
rng = np.random.default_rng(1)
for k, a in enumerate(arms, 1):
    ax2.scatter(k + rng.uniform(-.08, .08, 10), A[a].dice.values, color="k", s=14, alpha=.7, zorder=3)
    ax2.text(k, 0.88, f"{A[a].dice.mean():.3f}", ha="center", fontsize=9.5)
ax2.text(2, 0.93, "mean", ha="center", fontsize=9, color="#555")
ax2.set_xticks([1, 2, 3]); ax2.set_xticklabels(["Null", "SHORE", "Tensor"]); ax2.set_ylim(0, 1.0); ax2.set_ylabel("Dice")
ax2.grid(axis="y", alpha=.3); ax2.spines[["top", "right"]].set_visible(False)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(O / "arms.pdf"); plt.close(fig)

# ---------------- contrasts and forest ----------------
c = R("results/contrasts_final_asrun.csv")
name = {"dice": "Dice", "hd_agr": "Hausdorff agreement", "sd_standard": "Surface Dice", "volmatch": "Volume match", "projworst": "Worst-projection Dice"}
def f(x):
    return f"{x:+.3f}".replace("+-", "-")
rows = []
for con, title in (("shore-null", r"SHORE dODF $-$ isotropic null"), ("tensorNew-null", r"Tensor dODF $-$ isotropic null"), ("shore-tensorNew", r"SHORE dODF $-$ tensor dODF")):
    rows.append(r"        \midrule")
    rows.append(r"        \multicolumn{6}{l}{\textit{" + title + "}} \\\\")
    for m in MET:
        r = c[(c.estimator == "aggregate") & (c.contrast == con) & (c.metric == m)].iloc[0]
        dz = f"{r.dz:.2f}" if r.dz >= 0 else f"${r.dz:.2f}$"
        rows.append(f"        \\quad {name[m]:<22}& ${f(r['diff'])}\\ [{f(r.lo)},\\,{f(r.hi)}]$ & {dz} & ${r.p_holm:.3f}$ & ${r.wilcoxon:.3f}$ & {int(r.n_pos)}/10 \\\\")
(O / "table_arms_contrasts.tex").write_text("\n".join(rows[1:]).replace("$+", "$+").replace("-0.", "-0.") + "\n", encoding="utf8")
rows = []
for a, nm in (("null", r"Isotropic null (\texttt{NullDist2})"), ("shore", r"SHORE dODF (\texttt{iFOD2})        "), ("tensorNew", r"Tensor dODF (\texttt{iFOD2})       ")):
    rows.append(f"        {nm} & " + " & ".join(f"{A[a][m].mean():.3f}" for m in MET) + r" \\")
(O / "table_arms_means.tex").write_text("\n".join(rows) + "\n", encoding="utf8")
rows = []
php = ph[ph.tumour.isin(TEN)]
for con in ("shore-null", "tensorNew-null", "shore-tensorNew"):
    a_, b_ = con.split("-")
    for m in MET:
        r = c[(c.estimator == "per-host") & (c.contrast == con) & (c.metric == m)].iloc[0]
        P(f"per-host {con} {m}: A {php[php.arm==a_][m].mean():.3f} B {php[php.arm==b_][m].mean():.3f} diff {r['diff']:+.3f} [{r.lo:+.3f},{r.hi:+.3f}] dz {r.dz:.2f} holm {r.p_holm:.3f} wilc {r.wilcoxon:.3f} npos {r.n_pos}")
    for m in MET:
        if con != "shore-null":
            continue
        r = c[(c.estimator == "per-host") & (c.contrast == con) & (c.metric == m)].iloc[0]
        rows.append(f"        {name[m] if m!='projworst' else 'Worst-proj.\\ Dice'} & {php[php.arm=='shore'][m].mean():.3f} & {php[php.arm=='null'][m].mean():.3f} & ${f(r['diff'])}\\ [{f(r.lo)},\\,{f(r.hi)}]$ & {r.dz:.2f} & ${r.p_holm:.3f}$; ${r.wilcoxon:.3f}$ \\\\")
(O / "table_perhost_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf8")
for _, r in c[c.estimator == "aggregate"].iterrows():
    P(f"agg {r.contrast} {r.metric}: diff {r['diff']:+.3f} [{r.lo:+.3f},{r.hi:+.3f}] dz {r.dz:.2f} holm {r.p_holm:.3f} wilc {r.wilcoxon:.3f} npos {r.n_pos} nneg {r.n_neg} | {r.per_tumour}")

# LOO and power-equivalent effect sizes
for con, (a_, b_) in (("shore-null", ("shore", "null")), ("tensorNew-null", ("tensorNew", "null")), ("shore-tensorNew", ("shore", "tensorNew"))):
    for m in ("dice", "sd_standard", "hd_agr"):
        dd = (A[a_][m] - A[b_][m]).values
        loo = [np.delete(dd, i).mean() for i in range(10)]
        P(f"LOO {con} {m}: [{min(loo):+.3f}, {max(loo):+.3f}]  sd of diffs {dd.std(ddof=1):.3f}")
small = [1, 24, 31]
for con, (a_, b_) in (("shore-null", ("shore", "null")), ("tensorNew-null", ("tensorNew", "null"))):
    dd = (A[a_].dice - A[b_].dice)
    P(f"{con} dice small {dd.loc[small].round(3).to_dict()} other-seven mean {dd.drop(small).mean():+.3f}")
P("null agg dice on small:", A["null"].dice.loc[small].round(3).to_dict())
P("hd95 means:", {a: round(A[a].hd95_mm.mean(), 2) for a in arms}, "signed vol err:", {a: round(A[a].signed_vol_err.mean(), 3) for a in arms})
P("variant SD contrasts (aggregate):")
for a_, b_ in (("shore", "null"), ("tensorNew", "null"), ("shore", "tensorNew")):
    dd = (A[a_].sd_variant - A[b_].sd_variant).values
    P(f"  {a_}-{b_}: {dd.mean():+.3f} p {stats.ttest_1samp(dd,0).pvalue:.4f} npos {(dd>0).sum()}")

# forest (per-host estimator, SHORE - null)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for i, m in enumerate(MET):
    y = 4 - i
    pts = (php[php.arm == "shore"].groupby("tumour")[m].mean() - php[php.arm == "null"].groupby("tumour")[m].mean()).reindex(TEN).values
    ax.scatter(pts, y + rng.uniform(-.15, .15, 10), color="#a6a6a6", s=18, zorder=2, label="tumour (mean of 65 hosts)" if i == 0 else None)
    r = c[(c.estimator == "per-host") & (c.contrast == "shore-null") & (c.metric == m)].iloc[0]
    ax.plot([r.lo, r.hi], [y, y], color="#0072B2", lw=2.5, zorder=3)
    ax.scatter([r["diff"]], [y], marker="D", s=70, color="#0072B2", zorder=4, label="mean and 95% CI (n = 10)" if i == 0 else None)
ax.axvline(0, ls="--", color="#D55E00", lw=1.2)
ax.set_yticks(range(5)); ax.set_yticklabels([name[m] for m in MET][::-1]); ax.grid(axis="x", alpha=.3)
ax.set_xlabel("per-tumour mean difference, SHORE dODF $-$ isotropic null\n(positive values favour the dODF prior)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(O / "null_forest.pdf"); plt.close(fig)
out.close()

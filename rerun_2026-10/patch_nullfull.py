"""Make rand_seed_v2_nullfull.py from the campaign rand_seed_v2.py (read only):
1) separate patience for stage C (--patience_c), 2) final selection among stage-C rows only,
3) final 50k re-run written to <best>_50k (never overwrites the 10k search image)."""
src = "/mnt/bsc-das-01/BRIAN/proyect-brian-pipeline/rand_seed_v2.py"
dst = "/mnt/bsc-das-01/BRIAN/rerun_2026-10-15/scripts/rand_seed_v2_nullfull.py"
s = open(src).read()
reps = [
    ("    p.add_argument('--patience_passes', type=int, default=10,",
     "    p.add_argument('--patience_c', type=int, default=None, help='Patience in stage C (default: --patience_passes)')\n"
     "    p.add_argument('--patience_passes', type=int, default=10,"),
    ("            if no_improve_by_stage[stage] >= args.patience_passes:",
     "            if no_improve_by_stage[stage] >= (args.patience_c if (stage == 'C' and args.patience_c) else args.patience_passes):"),
    ("        idx_best = int(dataframe['error'].idxmin())",
     "        _df_c = dataframe[dataframe['stage'] == 'C']\n"
     "        if len(_df_c) == 0:\n"
     "            raise RuntimeError('no stage-C rows: selection restricted to stage C is impossible')\n"
     "        idx_best = int(_df_c['error'].idxmin())"),
    ("                    best_run,\n                    select=final_select,",
     "                    best_run + '_50k',\n                    select=final_select,"),
]
for old, new in reps:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
open(dst, "w").write(s)
print("written", dst)

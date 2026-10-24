"""Make rand_seed_v2_nullAB.py from the campaign rand_seed_v2.py (read only). Identical code,
plus --stop_at_C: the search stops when stage C is reached and only writes simulation.csv
(no selection, no final re-run). This reproduces the A/B part of the original SHORE search,
including its stage-A weights, so that stage C can be re-run from the null's own B-end state
with rand_seed_v2_resume(.py), exactly as stage_c_2 was for SHORE."""
src = "/mnt/bsc-das-01/BRIAN/proyect-brian-pipeline/rand_seed_v2.py"
dst = "/mnt/bsc-das-01/BRIAN/rerun_2026-10-15/scripts/rand_seed_v2_nullAB.py"
s = open(src).read()
reps = [
    ("    p.add_argument('--patience_passes', type=int, default=10,",
     "    p.add_argument('--stop_at_C', action='store_true', default=False)\n"
     "    p.add_argument('--patience_passes', type=int, default=10,"),
    ("        iteration += 1\n\n    # ----- Resumen final -----",
     "        iteration += 1\n"
     "        if args.stop_at_C and cur.stage == 'C':\n"
     "            print('[STOP] stage C reached (stop_at_C)')\n"
     "            break\n\n"
     "    if args.stop_at_C:\n"
     "        dataframe.to_csv('simulation.csv', index=False, float_format='%.10f')\n"
     "        return\n\n"
     "    # ----- Resumen final -----"),
]
for old, new in reps:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
open(dst, "w").write(s)
print("written", dst)

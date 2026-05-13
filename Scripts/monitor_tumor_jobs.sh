#!/usr/bin/env bash
# Snapshot del progreso del array brian_tumor.
# Uso:
#   bash monitor_tumor_jobs.sh                # un snapshot
#   watch -n 60 bash monitor_tumor_jobs.sh    # refresca cada minuto
set -u

BASE="${BASE:-/opt/users/aalbe03/Parallel_runs}"
RUNS="$BASE/runs"
LOGS="$BASE/logs"
USER_CLEAN="aalbe03"

TRAIN_IDS=(100408 101309 101915 103111 103414 103818 105014 105115 108828 110411 113619 113922 114419 115320 116524 117122 118730 118932 120111 122317)

printf "\n=== %s | monitor_tumor_jobs ===\n\n" "$(date '+%Y-%m-%d %H:%M:%S')"

# ---- 1) queue ----
printf "%s\n" "-- queue (user=${USER_CLEAN}) --"
squeue -u "$USER_CLEAN" -o "%.12i %.3t %.10M %.11l %.20j %.6D %R" 2>/dev/null | head -40
printf "\n"

# ---- 2) node ori utilisation ----
printf "%s\n" "-- node ori --"
scontrol show node ori 2>/dev/null \
  | tr ' ' '\n' \
  | grep -E '^(CPUAlloc|CPUTot|CPULoad|RealMemory|AllocMem|FreeMem)=' \
  | paste -sd ' '
printf "\n"

# ---- 3) per-tumor progress ----
printf "%s\n" "-- per-tumor training progress --"
printf "%-22s %5s %5s %5s   %s\n" "tumor" "done" "live" "todo" "active_subject(evals|last_mtime)"
now_ts=$(date +%s)
for t in "$RUNS"/*/; do
  tid=$(basename "$t")
  tr="$t/training_results"
  done_count=0
  live_count=0
  active_str=""
  if [[ ! -d "$tr" ]]; then
    printf "%-22s %5s %5s %5s   (no training_results)\n" "$tid" 0 0 20
    continue
  fi
  for sid in "${TRAIN_IDS[@]}"; do
    sd="$tr/$sid"
    if [[ ! -d "$sd" ]]; then continue; fi
    if [[ -f "$sd/analysis_participant_${sid}_best.txt" ]]; then
      done_count=$((done_count+1))
    else
      nnii=$(ls "$sd"/analysis_participant_${sid}_*.nii.gz 2>/dev/null | wc -l)
      if [[ "$nnii" -gt 0 ]]; then
        live_count=$((live_count+1))
        latest=$(ls -1t "$sd"/analysis_participant_${sid}_*.nii.gz 2>/dev/null | head -1)
        if [[ -n "$latest" ]]; then
          mt=$(stat -c %Y "$latest" 2>/dev/null || echo 0)
          age=$((now_ts - mt))
          active_str="${active_str}${sid}(${nnii}|${age}s) "
        fi
      fi
    fi
  done
  todo=$((20 - done_count - live_count))
  printf "%-22s %5d %5d %5d   %s\n" "$tid" "$done_count" "$live_count" "$todo" "$active_str"
done
printf "\n"

# ---- 4) throughput per active subject (evals/min using the 10 most recent NIfTIs) ----
printf "%s\n" "-- throughput (last 10 NIfTIs per active subject) --"
printf "%-22s %-8s %10s\n" "tumor" "subject" "evals/min"
for t in "$RUNS"/*/; do
  tid=$(basename "$t")
  tr="$t/training_results"
  [[ -d "$tr" ]] || continue
  for sid in "${TRAIN_IDS[@]}"; do
    sd="$tr/$sid"
    [[ -d "$sd" ]] || continue
    [[ -f "$sd/analysis_participant_${sid}_best.txt" ]] && continue
    mapfile -t recent < <(ls -1t "$sd"/analysis_participant_${sid}_*.nii.gz 2>/dev/null | head -10)
    if [[ ${#recent[@]} -lt 2 ]]; then continue; fi
    mt_first=$(stat -c %Y "${recent[0]}" 2>/dev/null || echo 0)
    last_idx=$((${#recent[@]} - 1))
    mt_last=$(stat -c %Y "${recent[$last_idx]}" 2>/dev/null || echo 0)
    dt=$((mt_first - mt_last))
    n="$last_idx"
    if [[ "$dt" -le 0 ]]; then continue; fi
    rate=$(awk -v n="$n" -v dt="$dt" 'BEGIN{printf "%.2f", (n*60)/dt}')
    printf "%-22s %-8s %10s\n" "$tid" "$sid" "$rate"
  done
done
printf "\n"

# ---- 5) aggregate ----
printf "%s\n" "-- aggregate --"
tot_done=$(find "$RUNS"/*/training_results -name "analysis_participant_*_best.txt" 2>/dev/null | wc -l)
printf "training subjects completed across all tumors: %d / 300\n" "$tot_done"
printf "\n"

printf "%s\n" "-- quick commands --"
printf "%s\n" "  live log:   tail -F ${LOGS}/brian_tumor_<jobid>_<N>.out"
printf "%s\n" "  queue:      squeue -u ${USER_CLEAN}"
printf "%s\n" "  cancel:     scancel <jobid>    # or scancel -u ${USER_CLEAN} for all"
printf "%s\n" "  errors:     grep -l -i 'error\\|traceback' ${LOGS}/brian_tumor_*.err"

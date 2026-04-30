#!/bin/bash
# Single-Python-process 4-window cycling DA test with sweep-KSP enabled.
# This is the direct-validation experiment for the post-fix in-process path:
# does the persistent-KSP + J_reg-reuse + malloc_trim stack actually let
# all 4 windows complete inside one process, or does the residual ~547 MB/eval
# growth still drive memory pressure past 60 evals?
#
# Memory-budget math at np=4 on LS6 (64 GB/rank):
#   ~3 GB baseline + 60 evals × ~547 MB/eval ≈ ~35 GB/rank peak
#                                            = ~55% of per-rank budget
# The pre-fix LU baseline (3130835) projected to ~70 GB/rank and hung at
# 230/251 GB. If sweep-KSP halves the rate as W0 measured, we should see
# this run complete instead of plateau.
#
# This is the experiment that decides whether sweep-KSP made the
# fresh-process-per-window chain mode optional rather than necessary.
# --------------------------------------------------------
#SBATCH -J 4w_single
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -o 4w_single.%j.out
#SBATCH -e 4w_single.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

# Post-fix defaults
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1
export SWE4DVAR_EVAL_MEM_DIAG=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_4w_single_${SLURM_JOB_ID}_rank{rank}.csv"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_4w_single_${SLURM_JOB_ID}.csv"

# Wipe stale chain bg files so cycling starts fresh
rm -f "$RES_DIR"/chain_bg_after_w*_rank*.npy

# Background watchdog — visible diagnostic if memory plateaus.
( while sleep 60; do
    echo "=== $(date +%T) ==="
    free -h | awk '/Mem:/ {print "  used="$3" avail="$7}'
    ps -eo pid,rss,comm --sort=-rss | grep -E "python|swe" | head -5
  done > "$RES_DIR/watchdog_4w_single_${SLURM_JOB_ID}.log" 2>&1 ) &
WPID=$!; trap "kill $WPID 2>/dev/null || true" EXIT

# Single ibrun: ONE Python process runs all 4 cycling windows.
# This is what failed at the LU baseline (3130835) due to ~1100 MB/eval growth.
# Sweep-KSP (~547 MB/eval) should bring it within budget.
ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 4 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo
echo "=== 4-window single-process run DONE ==="
ls -la $RES_DIR/result_*_cycling_w*.json 2>/dev/null || true
echo
echo "=== Per-window result summary ==="
for jf in $RES_DIR/result_4dvar_N_A_cycling_w*.json; do
  [[ -f "$jf" ]] || continue
  python -c "
import json
d = json.load(open('$jf'))
for w in d.get('windows', []):
    print(f'  w{w[\"window\"]-1:>2}  bg={w[\"bg_rmse\"]:.4f}  '
          f'analysis={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  '
          f'evals={w[\"n_func_evals\"]}  t={w[\"opt_time_s\"]:.0f}s')
" 2>/dev/null || true
done
echo
echo "=== Memory trajectory (rank 0, every after_cleanup row) ==="
RANK0_CSV="$RES_DIR/eval_mem_4w_single_${SLURM_JOB_ID}_rank0.csv"
if [[ -f "$RANK0_CSV" ]]; then
  awk -F, 'NR==1 || $4=="after_cleanup"' "$RANK0_CSV"
  echo "---peak---"
  awk -F, '$4=="after_cleanup" {if ($5+0 > peak) peak = $5+0} END {print "rank-0 peak after_cleanup RSS = " peak " MB"}' "$RANK0_CSV"
fi
echo
echo "=== Watchdog tail ==="
tail -40 "$RES_DIR/watchdog_4w_single_${SLURM_JOB_ID}.log" 2>/dev/null || true

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

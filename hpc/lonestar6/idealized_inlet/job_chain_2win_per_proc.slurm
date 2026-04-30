#!/bin/bash
# BOUNDED FOLLOW-UP — only run after job_chain_oneAlloc.slurm has been
# validated end-to-end with sweep-KSP on. This script tests whether
# the post-fix in-process growth (~547 MB/eval real-DA) is small enough
# to safely run TWO windows per fresh process instead of one. Halves the
# process-recycling overhead while still letting the OS reclaim allocator
# pools every two windows.
#
# Memory-budget math (np=4 on LS6, 64 GB/rank budget):
#   ~3 GB baseline + 2 windows × 15 evals × ~547 MB/eval ≈ ~19 GB/rank peak.
# Comfortable. If this passes, --nwin 4 single-process is the next test
# (~35 GB/rank, tighter but still within budget).
# --------------------------------------------------------
#SBATCH -J chain_2x
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -o chain_2x.%j.out
#SBATCH -e chain_2x.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1
export SWE4DVAR_EVAL_MEM_DIAG=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"

# Background watchdog
( while sleep 60; do
    echo "=== $(date +%T) ==="
    free -h | awk '/Mem:/ {print "  used="$3" avail="$7}'
    ps -eo pid,rss,comm --sort=-rss | grep -E "python|swe" | head -5
  done > "$RES_DIR/watchdog_chain_2x_${SLURM_JOB_ID}.log" 2>&1 ) &
WPID=$!; trap "kill $WPID 2>/dev/null || true" EXIT

# Wipe stale chain bg
rm -f "$RES_DIR"/chain_bg_after_w*_rank*.npy

# --- Two ibrun invocations, each runs n_windows=2 in one process ---

# Process 1: windows 0 + 1, no initial bg (chain start)
export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_chain2x_w01_rank{rank}.csv"
ibrun -n 4 python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 2 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 --mem-limit-gb 240
echo "[chain-2x] process 1 (w0+w1) DONE"

# Process 2: windows 2 + 3, loads chain bg from end of process 1's window 1
export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_chain2x_w23_rank{rank}.csv"
ibrun -n 4 python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 2 --start-window 2 \
  --initial-bg-file 'results/idealized_inlet_da/chain_bg_after_w1_rank{rank}.npy' \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 --mem-limit-gb 240
echo "[chain-2x] process 2 (w2+w3) DONE"

echo
echo "=== chain_2x DONE ==="
ls -la $RES_DIR/result_*_cycling_w*.json 2>/dev/null || true
echo
echo "=== peak per-rank RSS at last after_cleanup ==="
for csv in $RES_DIR/eval_mem_chain2x_*_rank0.csv; do
  [[ -f "$csv" ]] || continue
  echo "  $(basename "$csv")"
  awk -F, '$4=="after_cleanup"' "$csv" | tail -1 | awk -F, '{printf "    eval=%s rss=%s MB\n", $2, $5}'
done
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

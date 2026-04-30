#!/bin/bash
# LS6 development: 4D-Var cycling DA, 1h × 4 cycles, NO model error,
# WITH per-step forward-solve diagnostics (SWE4DVAR_FORWARD_DIAG_CSV).
# Mirror of Vista vn5_c4_diag job, submitted in parallel to whichever
# Vista queue (gh-dev / gg) finds a slot first.
# --------------------------------------------------------
#SBATCH -J ls6_c4_v10
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o ls6_c4_v10.%j.out
#SBATCH -e ls6_c4_v10.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

(while sleep 120; do echo "[watchdog $(date +%T)] $(free -h | grep Mem:)"; done) &
WPID=$!; trap "kill $WPID 2>/dev/null || true" EXIT

# Diagnostic CSV — same scheme as Vista variant. eval_id auto-increments
# at every solve_init across all forward solves and all windows.
DIAG_CSV="$WORK/SWEMniCS/results/idealized_inlet_da/forward_diag_ls6_c4_v10.csv"
rm -f "$DIAG_CSV"
export SWE4DVAR_FORWARD_DIAG_CSV="$DIAG_CSV"
echo "[diag] CSV: $DIAG_CSV"

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 4 \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo
echo "=== diag CSV summary ==="
ls -la "$DIAG_CSV" || true
wc -l "$DIAG_CSV" || true
head -5 "$DIAG_CSV" || true
echo "---tail (last 20 rows)---"
tail -20 "$DIAG_CSV" || true

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

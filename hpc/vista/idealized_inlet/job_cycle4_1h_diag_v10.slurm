#!/bin/bash
# Vista gh-dev: 4D-Var cycling DA, 1h windows × 4 cycles, NO model error,
# WITH per-step forward-solve diagnostics. Re-runs the same config as run B
# (vn5_4d_005_c4_1h, job 682192) but with SWE4DVAR_FORWARD_DIAG_CSV set so
# every solve_timestep call logs (eval_id, timestep, t, newton_iters,
# converged, correction_norm, residual_norm, min_h, max_u, max_v).
#
# When window 2+ fails, the CSV will show eval_id-by-eval_id which trial
# crashed Newton at which timestep with what state — directly answering
# the optimizer-vs-basin question.
# --------------------------------------------------------
#SBATCH -J vn5_c4_v10
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:30:00
#SBATCH -o vn5_c4_v10.%j.out
#SBATCH -e vn5_c4_v10.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

# Diagnostic CSV — single file across all forward solves and all windows.
# eval_id auto-increments across the whole run (each forward solve = +1).
DIAG_CSV="$WORK/SWEMniCS/results/idealized_inlet_da/forward_diag_vn5_c4_v10.csv"
rm -f "$DIAG_CSV"
export SWE4DVAR_FORWARD_DIAG_CSV="$DIAG_CSV"
echo "[diag] CSV: $DIAG_CSV"

mpirun -n 8 --bind-to core --map-by socket --timeout 7000 \
  -x SWE4DVAR_FORWARD_DIAG_CSV \
  python -u experiments/idealized_inlet_da.py \
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

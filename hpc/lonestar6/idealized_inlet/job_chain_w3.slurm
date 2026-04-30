#!/bin/bash
# LS6 chained cycling DA: window 3 of 4 (vmax=10) — final window.
#SBATCH -J chain_w3
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 01:30:00
#SBATCH -o chain_w3.%j.out
#SBATCH -e chain_w3.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

for R in 0 1 2 3; do
  test -f "$WORK/SWEMniCS/results/idealized_inlet_da/chain_bg_after_w2_rank${R}.npy" \
    || { echo "MISSING chain_bg_after_w2_rank${R}.npy"; exit 2; }
done

DIAG_CSV="$WORK/SWEMniCS/results/idealized_inlet_da/forward_diag_chain_w3.csv"
rm -f "$DIAG_CSV"
export SWE4DVAR_FORWARD_DIAG_CSV="$DIAG_CSV"

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 1 --start-window 3 \
  --initial-bg-file 'results/idealized_inlet_da/chain_bg_after_w2_rank{rank}.npy' \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== ALL 4 WINDOWS COMPLETE ==="
ls -la $WORK/SWEMniCS/results/idealized_inlet_da/result_4dvar_N_A_cycling_w*.json
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

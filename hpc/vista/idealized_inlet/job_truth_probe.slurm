#!/bin/bash
# Vista gh-dev: ISOLATED truth-only forward probe.
# No DA, no cycling, no optimizer. Determines whether the Newton-failure
# at t ≈ 6h+10min during cycling-DA originates in the SWE forward solver
# (forward-model limitation) or in cycling-pipeline interactions.
#
# Output: forward_diag CSV with per-timestep Newton state through 11h
# of integration. If row at t ≈ 22200s (= 6h+10min) shows newton_converged=0
# / NaN residual → forward-model issue confirmed → pivot to lower-vmax.
# --------------------------------------------------------
#SBATCH -J truth_probe
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 00:30:00
#SBATCH -o truth_probe.%j.out
#SBATCH -e truth_probe.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

DIAG_CSV="$WORK/SWEMniCS/results/idealized_inlet_truth_probe/truth_probe_v20_ts000.csv"
mkdir -p "$(dirname "$DIAG_CSV")"
rm -f "$DIAG_CSV"
export SWE4DVAR_FORWARD_DIAG_CSV="$DIAG_CSV"
echo "[diag] CSV: $DIAG_CSV"

mpirun -n 8 --bind-to core --map-by socket --timeout 1800 \
  -x SWE4DVAR_FORWARD_DIAG_CSV \
  python -u experiments/idealized_inlet_truth_probe.py \
  --vmax 20 --track-shift 0 \
  --nt-ramp 24 --nt-extra 42 --dt 600

echo
echo "=== diag CSV summary ==="
ls -la "$DIAG_CSV" || true
wc -l "$DIAG_CSV" || true
echo "---first 3 + last 10 rows---"
head -4 "$DIAG_CSV" || true
echo "..."
tail -10 "$DIAG_CSV" || true

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

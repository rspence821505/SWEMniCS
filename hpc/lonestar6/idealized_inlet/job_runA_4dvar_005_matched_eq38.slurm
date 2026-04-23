#!/bin/bash
# First-win search — R-A' (new): 4D-Var at obs_fraction=0.005 with Eq 38
# inflation matching Anchor B (σ_b²_h=0.01073, σ_b²_uv=0.1731).
#
# This is the APPLES-TO-APPLES 4D-Var baseline for Anchor B (DC-WME+Eq38).
# Both use the SAME inflated B; only the cost function differs.
#
# Uses --fixed-sigma-b-sq-* to skip the 20-min TLM Gram since the σ_b²
# values are known from Anchor B (3102561). Expected wall: ~55 min.
# -----------------------------------------------------------------------------
#SBATCH -J inlet_rAp_4dvar005_infl
#SBATCH -o inlet_rAp_4dvar005_infl.%j.out
#SBATCH -e inlet_rAp_4dvar005_infl.%j.err
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 02:00:00

set -euxo pipefail
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh

echo "=== provenance ==="
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

(while sleep 300; do echo "[watchdog $(date +%T)] $(free -h | grep Mem:)"; done) &
WPID=$!; trap "kill $WPID 2>/dev/null || true" EXIT

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --fixed-sigma-b-sq-h 0.01073 \
  --fixed-sigma-b-sq-uv 0.1731 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

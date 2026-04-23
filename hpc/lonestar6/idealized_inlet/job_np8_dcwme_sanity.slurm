#!/bin/bash
# np=8 DC-WME + TLM Eq 38 sanity — tiny config, exercises the exact
# code paths R3 will hit: truth_jacobians handoff, 58-vector TLM Gram,
# per-component eigendecomposition, DC-WME cost + L_wme.
#
# Purpose: de-risk R3 (DC-WME 0.02 Eq 38 ON) at np=8 before committing
# the full 30-min run. Short enough (nt_da=4, max_funcs=1) to finish in
# ~10 min but touches all the MPI-sensitive paths.
#
# If this completes cleanly: R3 at np=8 is safe to submit.
# If this crashes: we've isolated the failure mode with a small repro.
# -----------------------------------------------------------------------------
#SBATCH -J inlet_np8_dcwme_sanity
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 00:20:00
#SBATCH -o inlet_np8_dcwme_sanity.%j.out
#SBATCH -e inlet_np8_dcwme_sanity.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh

echo "=== provenance ==="
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

(while sleep 120; do echo "[watchdog $(date +%T)] $(free -h | grep Mem:)"; done) &
WPID=$!; trap "kill $WPID 2>/dev/null || true" EXIT

ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 4 --nt-da 4 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --eq38-component-aware \
  --max-iterations 1 --max-funcs 1 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

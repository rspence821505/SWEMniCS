#!/bin/bash
# First-win search — Run 3: DC-WME at obs_fraction=0.02 with TLM Eq 38 ON
# (component-aware, same as Anchor B but 4× more obs).
# Expected wall: Gram=80min + setup+opt=~70min → ~150 min total.
# Exceeds dev-queue 2h cap → submit to NORMAL queue (2-day walltime).
# -----------------------------------------------------------------------------
#SBATCH -J inlet_r3_dcwme020_eq38on
#SBATCH -o inlet_r3_dcwme020_eq38on.%j.out
#SBATCH -e inlet_r3_dcwme020_eq38on.%j.err
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 04:00:00

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
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.02 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --eq38-component-aware \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

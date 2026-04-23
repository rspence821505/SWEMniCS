#!/bin/bash
# UV-bisector diagnostic sbatch: tiny DC-WME+TLM+cmp-aware, one adjoint trace.
# Targets the structural finding that _compute_eq38_from_tlm produces zero u/v
# content. Instrumentation in src/swe4dvar/adjoint/implicit_adjoint.py fires
# ONLY for the first (i=0) adjoint call, printing per-step ||λ_h||, ||λ_uv||
# and BC/tiny-mask DOF accounting around the two suspect zeroing sites:
#   - _solve_transpose_system:985  (tiny-mask zeroing)
#   - _compute_initial_gradient:BC zeroing
# See docs/idealized_inlet_tlm_uv_bisector.md.
# -----------------------------------------------------------------------------
#SBATCH -J inlet_uvbis
#SBATCH -o inlet_uvbis.%j.out
#SBATCH -e inlet_uvbis.%j.err
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 00:30:00

set -euxo pipefail

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export PYTHONUNBUFFERED=1

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

# Tiny config — exercises TLM Eq 38 adjoint path but finishes in ~15 min.
ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 4 --nt-da 4 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 1 --max-funcs 1 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --eq38-component-aware \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

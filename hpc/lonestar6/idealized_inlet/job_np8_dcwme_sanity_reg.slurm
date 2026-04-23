#!/bin/bash
# Phase 1 stabilization: DC-WME + TLM Gram at np=8 with SWE4DVAR_ADJOINT_REG
# set to shift J^T + εI for MUMPS stability.
#
# Previous sanity (3104195) failed at adjoint #51/58 with
# LU(reason=-11), GMRES+ILU(reason=-3), GMRES(reason=-3) — all
# strategies diverged. Hypothesis: small per-rank partition (~25K DOFs
# at np=8) creates ill-conditioned local blocks that MUMPS can't pivot.
# Adding a small εI diagonal shift regularizes the operator.
# -----------------------------------------------------------------------------
#SBATCH -J inlet_np8_dcwme_sanity_reg
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 00:25:00
#SBATCH -o inlet_np8_dcwme_sanity_reg.%j.out
#SBATCH -e inlet_np8_dcwme_sanity_reg.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Phase 1 fix: regularize J^T by ε on transpose solves
export SWE4DVAR_ADJOINT_REG=1e-10

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh

echo "=== provenance ==="
git -C $WORK/SWEMniCS rev-parse HEAD
echo "SWE4DVAR_ADJOINT_REG=$SWE4DVAR_ADJOINT_REG"
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

#!/bin/bash
# Vista: 4D-Var inflation-matched at obs=0.005, np=8.
# σ_b²_h and σ_b²_uv passed via env SIGMA_H_SQ / SIGMA_UV_SQ (from paired DC-WME run).
# Submit with:
#   sbatch --export=ALL,SIGMA_H_SQ=0.01073,SIGMA_UV_SQ=0.1731 \
#          hpc/vista/idealized_inlet/job_i_4dvar_0005_fixed_np8.slurm
# --------------------------------------------------------
#SBATCH -J vi_4dvar_0005_fx
#SBATCH -A ADCIRC
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o vi_4dvar_0005_fx.%j.out
#SBATCH -e vi_4dvar_0005_fx.%j.err

set -euxo pipefail

: "${SIGMA_H_SQ:?env SIGMA_H_SQ (sigma_b_sq_h) must be set via --export}"
: "${SIGMA_UV_SQ:?env SIGMA_UV_SQ (sigma_b_sq_uv) must be set via --export}"

source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD
echo "SIGMA_H_SQ=$SIGMA_H_SQ   SIGMA_UV_SQ=$SIGMA_UV_SQ"

mpirun -n 8 --bind-to core --map-by socket --timeout 3600 python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --fixed-sigma-b-sq-h "$SIGMA_H_SQ" \
  --fixed-sigma-b-sq-uv "$SIGMA_UV_SQ" \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

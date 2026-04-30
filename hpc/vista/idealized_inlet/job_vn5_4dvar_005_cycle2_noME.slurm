#!/bin/bash
# Vista gg: 4D-Var obs=0.05, freq=5, bg=0.02 — CYCLING DA, n_windows=2,
# WITH NO MODEL ERROR (track-shift=0 → DA wind == truth wind).
# This isolates the cycling math: any cross-window improvement should be
# real; any degradation indicates a code/numerical issue rather than
# DA-model-truth-model divergence.
# Window 1 should produce a smooth descent from window 0's analysis.
# --------------------------------------------------------
#SBATCH -J vn5_4d_005_c2_nme
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:30:00
#SBATCH -o vn5_4d_005_c2_nme.%j.out
#SBATCH -e vn5_4d_005_c2_nme.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

mpirun -n 8 --bind-to core --map-by socket --timeout 7000 python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 0 \
  --nt-ramp 24 --nt-da 12 \
  --n-windows 2 \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

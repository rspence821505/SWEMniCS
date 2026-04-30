#!/bin/bash
# Vista: DC-WME inflation-ON at obs=0.005, np=32.
# Scaling test: same config as np=8 variant but 4x more ranks.
# Grace node has 144 cores; np=32 uses 22% of one node.
# Compare wall time + memory vs vi_dcwme_0005 (np=8) to measure scaling.
# --------------------------------------------------------
#SBATCH -J vi_dcwme_0005_np32
#SBATCH -A ADCIRC
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 01:00:00
#SBATCH -o vi_dcwme_0005_np32.%j.out
#SBATCH -e vi_dcwme_0005_np32.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

mpirun -n 32 --bind-to core --map-by socket --timeout 3600 python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --eq38-component-aware \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

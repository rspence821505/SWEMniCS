#!/bin/bash
# Vista gh-dev pipeline validation — DC-WME inflation-ON at obs=0.005 np=8.
# Full Step 1..8 pipeline in ~10-15 min to confirm env + code work end-to-end
# before firing the 4 parallel gg runs. Same config as job_i_dcwme_0005_infl_np8.slurm
# but on gh-dev with shorter walltime.
# -----------------------------------------------------------------------------
#SBATCH -J vdev_dcwme_005
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 00:30:00
#SBATCH -o vdev_dcwme_005.%j.out
#SBATCH -e vdev_dcwme_005.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

mpirun -n 8 --bind-to core --map-by socket --timeout 3600 python -u experiments/idealized_inlet_da.py \
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

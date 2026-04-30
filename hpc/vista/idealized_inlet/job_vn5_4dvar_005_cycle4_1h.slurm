#!/bin/bash
# Vista gh-dev: 4D-Var cycling DA, 1h windows × 4 cycles, NO model error.
# Test of the "error-amplification × time" hypothesis. Same total DA span
# as the 2h × 2 cycling test (4h total) but sliced finer. If error
# amplification through peak-storm dynamics is the cause of the 2h failure,
# 1h windows should integrate cleanly because errors have less time to grow
# between optimization handoffs.
#
# Decision criteria:
#   - All 4 windows succeed → window length IS the control knob (matches "B-success")
#   - Window 2+ fail at same lead-time as 2h case → length isn't the issue (run forecast probe)
#   - Window 1 succeeds, window 2 fails → multi-cycle drift, not single-window length
#
# obs-frequency=5 in 6-step window → obs at steps [0, 5] = 2 snapshots/window.
# --------------------------------------------------------
#SBATCH -J vn5_4d_005_c4_1h
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:30:00
#SBATCH -o vn5_4d_005_c4_1h.%j.out
#SBATCH -e vn5_4d_005_c4_1h.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

mpirun -n 8 --bind-to core --map-by socket --timeout 7000 python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 4 \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

#!/bin/bash
# Iteration 2: same config as 3136460 (3w/1h/2obs/peak-W2 LS6) but
# with TIGHTER background prior (--background-error-std 0.01 vs 0.02).
#
# 3136460 failed at W1 with Newton divergence (min h = -46.9 m). The
# hypothesis: cycling propagation amplifies the W0 analysis perturbation
# through the active storm, producing non-physical h<0 in the W1 forward
# solve. Tightening B by 2x (variance 4x smaller) constrains analysis
# corrections more tightly, reducing the magnitude of state shifts the
# forecast has to propagate.
# --------------------------------------------------------
#SBATCH -J recJ_tightB
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 01:30:00
#SBATCH -o recJ_tightB.%j.out
#SBATCH -e recJ_tightB.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_iter2_${SLURM_JOB_ID}.csv"
rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

# CHANGE: --background-error-std 0.01 (was 0.02)
ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --track-duration-s 60000 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 3 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.01 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240
RET=$?

echo
echo "=== Per-window result JSONs ==="
for j in $RES_DIR/result_4dvar_N_A_cycling_w*.json; do
  [[ -f "$j" ]] || continue
  echo "  $(basename "$j")"
  python3 -c "import json; d=json.load(open('$j')); w=d['windows'][0]; print(f\"    bg={w['bg_rmse']:.6f}  an={w['analysis_rmse']:.6f}  imp={w['improvement_pct']:+.2f}%  evals={w['n_func_evals']}\")" 2>/dev/null || cat "$j" | head -20
done

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

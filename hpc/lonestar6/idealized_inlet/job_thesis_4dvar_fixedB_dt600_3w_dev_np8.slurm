#!/bin/bash
# 4D-Var (traditional) 3-window cycling, dt=600, FIXED-B, LS6 development.
# Mirror of Vista 733066 (DC-WME fixed-B 3w dt=600) for the 4D-Var
# comparison arm. Dev queue's 2h cap means it likely won't finish all
# 3 windows — W0 + W1 are the primary deliverables per user request.
#
# Identical config to 733066 EXCEPT:
#   - --method 4dvar (vs dcwme_static)
#   - LS6 ibrun launcher (vs Vista mpirun)
#   - dev partition (vs Vista gh production)
#
# With the constant-R fix (eeb6332) + fixed σ_b² bypass, both arms now
# see identical B and R across windows — apples-to-apples DC-WME vs
# 4D-Var comparison is finally possible.
# --------------------------------------------------------
#SBATCH -J thesis_4dvar_fixedB_dt600_3w_dev
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o thesis_4dvar_fixedB_dt600_3w_dev.%j.out
#SBATCH -e thesis_4dvar_fixedB_dt600_3w_dev.%j.err

set -uxo pipefail
shopt -s nullglob
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

export SWE4DVAR_NEWTON_RELAX=0.3
unset SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS
unset SWE4DVAR_CAPTURE_REPLAY_META
export SWE4DVAR_FORWARD_NEWTON_REUSE=0
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

export SWE4DVAR_OUTPUT_SUBDIR=thesis_4dvar_fixedB_dt600_3w_dev
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_4dvar_fixedB_dt600_3w_dev"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_4dvar_cycling_w*-*.json 2>/dev/null

# Same fixed-B + R-const config as Vista 733066, --method 4dvar instead.
bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 3 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 24 \
  --nt-da 6 \
  --method 4dvar \
  --launcher ibrun \
  --extra "--track-shift 15 --track-duration-s 68400 --dt 600 --obs-fraction 0.10 --obs-frequency 2 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --wd --wd-alpha 1.5 --fixed-sigma-b-sq-h 5.0e-02 --fixed-sigma-b-sq-uv 2.0e+03 --predictability-gamma 1.0 --eq38-component-aware --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_*cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== 4D-Var 3w dev cycling (LS6): FIXED-B + dt=600 ==="
echo "=== σ²_h=5.0e-02, σ²_uv=2.0e+03, R constant via eeb6332 ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== Comparison arm: Vista 733066 (DC-WME fixed-B 3w dt=600) ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

#!/bin/bash
# Production 6-window cycling extension of 3201419 (the +3.84% dt=600
# TLM-Gram-B result). Vista gh production queue.
#
# Same TLM-Gram-B + γ=1.0 setup as 3201419, extended to 6 windows.
# dt=600 gives the fastest per-window cycling we can run while still
# carrying the full TLM Gram path. 8h walltime is generous: at dt=600
# the single-W0 probe finished in roughly 20 min, so 6 windows should
# take ~2-4h on Vista.
# --------------------------------------------------------
#SBATCH -J thesis_dcw_tlmgram_dt600_6w_v
#SBATCH -A ADCIRC
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 08:00:00
#SBATCH -o thesis_dcwme_tlmgram_dt600_6w_vista.%j.out
#SBATCH -e thesis_dcwme_tlmgram_dt600_6w_vista.%j.err

set -euxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

export SWE4DVAR_NEWTON_RELAX=0.3
unset SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS
unset SWE4DVAR_CAPTURE_REPLAY_META
export SWE4DVAR_FORWARD_NEWTON_REUSE=0
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_tlmgram_dt600_6w_vista
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_tlmgram_dt600_6w_vista"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

# NOTE: --fixed-sigma-b-sq-h/uv OMITTED so Step 7a activates the TLM
# Gram branch. dt=600, nt-ramp=24 (4h), nt-da=6 (1h), obs-frequency=2
# (4 obs/window @ every 20 min).
bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 6 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 24 \
  --nt-da 6 \
  --method dcwme_static \
  --launcher mpirun \
  --extra "--track-shift 15 --track-duration-s 68400 --dt 600 --obs-fraction 0.10 --obs-frequency 2 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --wd --wd-alpha 1.5 --predictability-gamma 1.0 --eq38-component-aware --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME 6w cycling (Vista): TLM-Gram B + γ=1.0 + dt=600 ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== Reference: dt=600 TLM-Gram single-W0 = +3.84% (ls6 3201419) ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

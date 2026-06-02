#!/bin/bash
# Vista companion to ls6 3201386. Six-window cycling extension of
# 3201132 (the +3.36% dt=300 TLM-Gram-B single-W0 result). Vista gh
# production queue.
#
# Same config as ls6 3201386 / 3201132 except host-specific bits
# (mpirun launcher, partition, env script). Submitting here so the
# dt=300 6w result lands without waiting on LS6 normal's priority
# backlog.
# --------------------------------------------------------
#SBATCH -J thesis_dcw_tlmgram_dt300_6w_v
#SBATCH -A ADCIRC
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH -o thesis_dcwme_tlmgram_dt300_6w_vista.%j.out
#SBATCH -e thesis_dcwme_tlmgram_dt300_6w_vista.%j.err

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

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_tlmgram_dt300_6w_vista
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_tlmgram_dt300_6w_vista"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

# NOTE: --fixed-sigma-b-sq-h/uv OMITTED to activate the TLM Gram path.
# dt=300, nt-ramp=48 (4h), nt-da=12 (1h), obs-frequency=3 (every 15 min).
bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 6 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 48 \
  --nt-da 12 \
  --method dcwme_static \
  --launcher mpirun \
  --extra "--track-shift 15 --track-duration-s 68400 --dt 300 --obs-fraction 0.10 --obs-frequency 3 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --wd --wd-alpha 1.5 --predictability-gamma 1.0 --eq38-component-aware --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME 6w cycling (Vista): TLM-Gram B + γ=1.0 + dt=300 ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== Reference: dt=300 TLM-Gram single-W0 = +3.36% (ls6 3201132) ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

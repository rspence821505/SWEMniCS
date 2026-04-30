#!/bin/bash
# Run an N-window cycling-DA chain inside ONE SLURM allocation by issuing
# one fresh `srun` per window. Each window runs as a fresh MPI process so
# the OS reclaims PETSc/glibc allocator pools between windows — same memory
# semantics as the multi-job afterok chain, but with a single sbatch.
#
# Designed for LS6 (ibrun is wrapped via srun-ish path in TACC, but srun
# also works inside an allocation). On Vista this script also works since
# `srun` is supported.
#
# Usage (inside a batch job):
#   bash run_chain_in_alloc.sh \
#       --nwin 4 \
#       --np 4 \
#       --vmax 10 \
#       --extra "--obs-fraction 0.05 --obs-frequency 5 --max-iterations 15 --max-funcs 15 --mem-limit-gb 240"
#
# All windows write per-rank chain bg files to:
#   $WORK/SWEMniCS/results/idealized_inlet_da/chain_bg_after_w{N}_rank{R}.npy
# which is the same convention as the upstream afterok chain. The {rank}
# template is passed verbatim to Python (it formats it per rank).

set -euo pipefail

NWIN=4
NP=4
VMAX=10
TRACK_SHIFT=0
NT_RAMP=24
NT_DA=6
METHOD="4dvar"
EXTRA=""
RES_DIR="${RES_DIR:-$WORK/SWEMniCS/results/idealized_inlet_da}"
TEMPLATE_REL="results/idealized_inlet_da/chain_bg_after_w{prev}_rank{rank}.npy"
LAUNCHER="${LAUNCHER:-srun}"   # set to "ibrun" on LS6 if preferred

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nwin) NWIN="$2"; shift 2 ;;
    --np) NP="$2"; shift 2 ;;
    --vmax) VMAX="$2"; shift 2 ;;
    --track-shift) TRACK_SHIFT="$2"; shift 2 ;;
    --nt-ramp) NT_RAMP="$2"; shift 2 ;;
    --nt-da) NT_DA="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --extra) EXTRA="$2"; shift 2 ;;
    --launcher) LAUNCHER="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RES_DIR"

# Wipe stale chain-bg files so a half-completed prior run cannot be picked up.
rm -f "$RES_DIR"/chain_bg_after_w*_rank*.npy
echo "[chain-alloc] cleared stale chain bg files in $RES_DIR"
echo "[chain-alloc] config: nwin=$NWIN np=$NP launcher=$LAUNCHER vmax=$VMAX nt_da=$NT_DA"

# Canonical operational defaults: persistent sweep-KSP + glibc trim. These
# are the post-fix defaults that cut adjoint memory churn ~50% in real DA
# while preserving DA quality and runtime. Set explicitly here so the chain
# scripts are self-documenting and not dependent on inherited shell state.
export SWE4DVAR_ADJOINT_SWEEP_KSP="${SWE4DVAR_ADJOINT_SWEEP_KSP:-1}"
export SWE4DVAR_MALLOC_TRIM="${SWE4DVAR_MALLOC_TRIM:-1}"
echo "[chain-alloc] SWE4DVAR_ADJOINT_SWEEP_KSP=$SWE4DVAR_ADJOINT_SWEEP_KSP"
echo "[chain-alloc] SWE4DVAR_MALLOC_TRIM=$SWE4DVAR_MALLOC_TRIM"

cd "${WORK:-$(pwd)}/SWEMniCS"

for ((w=0; w<NWIN; w++)); do
  echo
  echo "############################################################"
  echo "# CHAIN-ALLOC: launching window $w / $((NWIN-1))"
  echo "############################################################"

  # Per-window forward-diag CSV so each fresh process has its own log
  export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_chain_alloc_w${w}.csv"
  rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"

  # Per-window eval-memory CSV (gated by SWE4DVAR_EVAL_MEM_DIAG)
  if [[ "${SWE4DVAR_EVAL_MEM_DIAG:-0}" == "1" ]]; then
    export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_chain_alloc_w${w}_rank{rank}.csv"
    rm -f "$RES_DIR/eval_mem_chain_alloc_w${w}_rank"*.csv
  fi

  CMD=( python -u experiments/idealized_inlet_da.py
        --method "$METHOD"
        --vmax "$VMAX" --track-shift "$TRACK_SHIFT"
        --nt-ramp "$NT_RAMP" --nt-da "$NT_DA"
        --n-windows 1 --start-window "$w"
        --chain-mode )

  # First window of the chain has no prior background file.
  # Window K>0 loads the bg saved by window K-1 (per-rank).
  if [[ "$w" -gt 0 ]]; then
    BG_FILE="${TEMPLATE_REL/\{prev\}/$((w-1))}"
    echo "[chain-alloc] window $w loads prior bg: $BG_FILE"
    CMD+=( --initial-bg-file "$BG_FILE" )
  fi

  # Append user-provided extras (parsed by python)
  if [[ -n "$EXTRA" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=( $EXTRA )
    CMD+=( "${EXTRA_ARR[@]}" )
  fi

  echo "[chain-alloc] cmd: $LAUNCHER -n $NP ${CMD[*]}"
  if ! $LAUNCHER -n "$NP" "${CMD[@]}"; then
    echo "[chain-alloc] !!! window $w FAILED (exit=$?). Halting chain." >&2
    exit 1
  fi

  # Sanity-check that the chain bg got written before next window
  EXPECTED="$RES_DIR/chain_bg_after_w${w}_rank0.npy"
  if [[ ! -f "$EXPECTED" ]]; then
    echo "[chain-alloc] !!! expected $EXPECTED not found after window $w" >&2
    exit 2
  fi
  echo "[chain-alloc] window $w OK; chain bg file present"
done

echo
echo "[chain-alloc] DONE: $NWIN windows in one allocation"
ls -la "$RES_DIR"/chain_bg_after_w*.npy
ls -la "$RES_DIR"/result_*_cycling_w*.json 2>/dev/null || true

echo
echo "=== Per-window summary (RMSE + improvement) ==="
for json_file in "$RES_DIR"/result_*_cycling_w*.json; do
  [[ -f "$json_file" ]] || continue
  python -c "
import json, sys
d = json.load(open('$json_file'))
for w in d.get('windows', []):
    print(f'  w{w[\"window\"]-1:>2}  bg={w[\"bg_rmse\"]:.4f}  '
          f'analysis={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  '
          f'evals={w[\"n_func_evals\"]}  t={w[\"opt_time_s\"]:.0f}s')
" 2>/dev/null || true
done

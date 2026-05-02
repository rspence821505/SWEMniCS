#!/bin/bash
# Shinnecock Phase 1 smoke test with recompute-J + all session bug fixes.
#
# Phase 1 = single-window 4D-Var with no model error (inverse crime).
# Validated baseline: 8.5% error reduction (0.324 -> 0.297, see CLAUDE.md
# Phase 1 entry).
#
# This run applies all the cycling-DA solver work from this session:
#   - recompute-J adjoint (saves ~1.4 GB/eval at nt_da=12)
#   - sweep-KSP (forward + adjoint, eliminates LU factor leak)
#   - Newton α=0.5 relaxation (stability under perturbed-IC TAO probes)
#   - malloc_trim per eval
#   - replay metadata capture (Bug 3 fix)
#   - jacobians=None tolerated in adjoint ctor (Bug 4 fix)
#
# Acceptance: bg_rmse and analysis_rmse match the validated 8.5% baseline
# within ~1% — confirms math-equivalence under recompute mode.
# --------------------------------------------------------
#SBATCH -J shin_smoke
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -o shin_smoke.%j.out
#SBATCH -e shin_smoke.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

# All session-level fixes
export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1
# Newton relaxation — Phase 1 is single-window so α=0.7 default should be
# fine. Setting explicitly for traceability and so future runs are
# reproducible without reading the default.
export SWE4DVAR_NEWTON_RELAX=0.7

RES_DIR=$WORK/SWEMniCS/outputs/shinnecock_study
mkdir -p $RES_DIR
export SWE4DVAR_EVAL_MEM_DIAG=1
export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_shin_smoke_${SLURM_JOB_ID}_rank{rank}.csv"

ibrun python -u experiments/shinnecock_study/run_comparison.py \
  --phase 1 \
  --output-dir "$RES_DIR" \
  --adios-file data/shinnecock_inlet
RET=$?

echo
echo "=== peak after_cleanup RSS (rank 0) ==="
csv="$RES_DIR/eval_mem_shin_smoke_${SLURM_JOB_ID}_rank0.csv"
if [[ -f "$csv" ]]; then
  awk -F, '$4=="after_cleanup" {print "  eval="$2"  rss="$5" MB  trim_freed="$8" MB"}' "$csv" | head -15
  awk -F, 'NR>1 {if ($5+0 > peak) peak=$5+0} END {print "  peak rss = "peak" MB"}' "$csv"
fi

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

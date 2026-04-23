#!/bin/bash
# np=8 MPI parity gate — runs parity_4dvar_reduced.py at np=2 (baseline) and
# np=8, then compares J_bg / grad_l2_global / grad_linf_global at rel_err ≤ 1e-10.
# Precondition for promoting the idealized-inlet DC-WME production runs from
# Phase 1 (np=2×64) to Phase 2 (np=8×16). See
# docs/ls6_first_dcwme_production_run.md §1 / §8.
# -----------------------------------------------------------------------------
#SBATCH -J parity_np8
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 00:15:00
#SBATCH -o parity_np8.%j.out
#SBATCH -e parity_np8.%j.err

set -euxo pipefail

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh

echo "=== provenance ==="
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

OUT=$SCRATCH/parity_runs/$SLURM_JOB_ID
mkdir -p "$OUT"

# Use OMP_NUM_THREADS=16 to match Phase 2 spec (16 threads × 8 ranks = 128 cores).
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

echo "--- np=2 baseline ---"
ibrun -n 2 python hpc/lonestar6/parity/parity_4dvar_reduced.py > "$OUT/4dvar_np2.json"
head -30 "$OUT/4dvar_np2.json"

echo "--- np=8 test ---"
ibrun -n 8 python hpc/lonestar6/parity/parity_4dvar_reduced.py > "$OUT/4dvar_np8.json"
head -30 "$OUT/4dvar_np8.json"

echo "--- parity diff (rel_err vs np=2) ---"
python -c "
import json
j2 = json.load(open('$OUT/4dvar_np2.json'))
j8 = json.load(open('$OUT/4dvar_np8.json'))
TOL = 1e-10
FIELDS = ['J_bg', 'grad_l2_global', 'grad_linf_global', 'final_state_l2_global']
worst = 0.0
all_ok = True
for f in FIELDS:
    if f not in j2 or f not in j8:
        print(f'  {f}: MISSING (j2? {f in j2}, j8? {f in j8})')
        all_ok = False
        continue
    v2, v8 = float(j2[f]), float(j8[f])
    denom = max(abs(v2), 1e-300)
    rel = abs(v8 - v2) / denom
    status = 'PASS' if rel <= TOL else 'FAIL'
    if rel > worst: worst = rel
    print(f'  {f}: np=2={v2:.15e}  np=8={v8:.15e}  rel_err={rel:.3e}  [{status}]')
    if rel > TOL: all_ok = False
print(f'  worst rel_err = {worst:.3e}  (tol={TOL:.0e})')
print(f'  OVERALL: {\"PASS — np=8 safe to promote\" if all_ok else \"FAIL — stay at np=2\"}')
import sys
sys.exit(0 if all_ok else 1)
"

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

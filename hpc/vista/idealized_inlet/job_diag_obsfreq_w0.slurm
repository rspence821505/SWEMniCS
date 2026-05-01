#!/bin/bash
# Diagnostic: WHY does obs_frequency=3 (3 obs/window) cause W0 Newton
# divergence on the 1h-window cycling case, when obs_frequency=5
# (2 obs/window — the validated baseline) converges smoothly?
#
# 3136438 (LS6 dev, 3w/1h/3obs) failed in W0 after 2 evals with
# 6 consecutive forward Newton failures (Min h on rank 2 = -0.30 m).
# Hypothesis: more obs slots → richer gradient signal → TAO BLMVM
# takes a larger initial step than the validated 2-obs case → IC
# perturbation makes the forward solve produce h<0 at timestep 1.
#
# Diagnostic strategy: rerun 1 window only (n_windows=1), with
# SWE4DVAR_FORWARD_DIAG_CSV enabled. The CSV records per-timestep
# (eval_id, timestep, newton_iters, converged, min_h, max_u, max_v)
# across ALL evals including the failing ones. Then diff the
# obs_freq=3 vs obs_freq=5 CSVs to see at which eval & timestep the
# 3-obs case first diverges.
#
# Two runs in one allocation, sequential ibrun calls.
# --------------------------------------------------------
#SBATCH -J obsfreq_diag
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 01:00:00
#SBATCH -o obsfreq_diag.%j.out
#SBATCH -e obsfreq_diag.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"

# Recompute mode + Bug 5 fix + Bug 6 default off
export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

CSV_3OBS="$RES_DIR/forward_diag_obsfreq3_w0_${SLURM_JOB_ID}.csv"
CSV_5OBS="$RES_DIR/forward_diag_obsfreq5_w0_${SLURM_JOB_ID}.csv"

# ============================================================
# Run A: obs_frequency=3 (THE FAILING CONFIG)
# ============================================================
echo
echo "================ RUN A: obs_frequency=3 (3 obs/window) ================"
rm -f "$CSV_3OBS"
export SWE4DVAR_FORWARD_DIAG_CSV="$CSV_3OBS"
rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
mpirun -n 4 --bind-to core --map-by socket --timeout 3500 \
  -x SWE4DVAR_FORWARD_DIAG_CSV \
  -x SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS \
  -x SWE4DVAR_FORWARD_NEWTON_REUSE \
  -x SWE4DVAR_ADJOINT_SWEEP_KSP \
  -x SWE4DVAR_MALLOC_TRIM \
  python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 1 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 3 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240 || true   # never abort; failure is the data

# ============================================================
# Run B: obs_frequency=5 (CONTROL — validated baseline density)
# ============================================================
echo
echo "================ RUN B: obs_frequency=5 (2 obs/window, control) ================"
rm -f "$CSV_5OBS"
export SWE4DVAR_FORWARD_DIAG_CSV="$CSV_5OBS"
rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
mpirun -n 4 --bind-to core --map-by socket --timeout 3500 \
  -x SWE4DVAR_FORWARD_DIAG_CSV \
  -x SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS \
  -x SWE4DVAR_FORWARD_NEWTON_REUSE \
  -x SWE4DVAR_ADJOINT_SWEEP_KSP \
  -x SWE4DVAR_MALLOC_TRIM \
  python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 1 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240 || true

# ============================================================
# Diff summary
# ============================================================
echo
echo "================ SUMMARY ================"
for tag in obsfreq3 obsfreq5; do
  csv="$RES_DIR/forward_diag_${tag}_w0_${SLURM_JOB_ID}.csv"
  [[ -f "$csv" ]] || continue
  echo
  echo "--- $tag ---"
  echo "Header:"
  head -1 "$csv"
  echo "First 3 evals (all timesteps):"
  awk -F, 'NR==1 || $2 <= 3' "$csv" | head -25
  echo "Failed timesteps (converged != 1) — top 10:"
  awk -F, 'NR==1 || $5 == "False" || $5 == "false"' "$csv" | head -11
  echo "min_h across all rows:"
  awk -F, 'NR>1 {if(\$8+0 < min || min == "") min=\$8+0} END {print "  min(min_h) = "min}' "$csv"
done

echo
echo "Diff: at which (eval_id, timestep) does the 3-obs run first diverge from 5-obs?"
diff <(awk -F, '{print $2","$3","$4","$5}' "$CSV_3OBS" 2>/dev/null) \
     <(awk -F, '{print $2","$3","$4","$5}' "$CSV_5OBS" 2>/dev/null) | head -30 || true

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

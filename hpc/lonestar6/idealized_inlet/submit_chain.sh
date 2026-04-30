#!/bin/bash
# Submit the cycling-DA chain. Only window 0 is submitted directly;
# each window's sbatch script submits the next one upon success
# (self-chaining to bypass LS6 development-queue per-user submit limit).
set -euo pipefail

SBATCH_DIR="$WORK/SWEMniCS/hpc/lonestar6/idealized_inlet"
RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"

# Clean stale chain bg files from prior attempts so a partial chain
# cannot accidentally pick up old data.
rm -f "$RES_DIR"/chain_bg_after_w*.npy
echo "[submit_chain] cleared stale chain bg files"

sbatch "$SBATCH_DIR/job_chain_w0.slurm"
echo
echo "Window 0 submitted. Subsequent windows self-submit on success."
echo "Monitor with: ssh ls6 'squeue -u tg876971'"
echo "Diagnostics:  ssh ls6 'tail -f \$WORK/SWEMniCS/chain_w*.out'"

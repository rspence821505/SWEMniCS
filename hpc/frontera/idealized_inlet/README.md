# Idealized Inlet DA Experiments on TACC Frontera

This directory contains HPC infrastructure for running data assimilation experiments
with the idealized inlet problem on TACC Frontera.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Running Experiments](#running-experiments)
- [Scaling Studies](#scaling-studies)
- [Monitoring Jobs](#monitoring-jobs)
- [Retrieving Results](#retrieving-results)
- [Troubleshooting](#troubleshooting)

## Overview

### What This Does

These scripts run 4D-Var and DC-WME data assimilation experiments for the idealized
inlet problem, which simulates tidal flow in an idealized coastal inlet. The experiments
use MPI parallelization for large-scale computations on Frontera.

### File Structure

```
hpc/frontera/idealized_inlet/
├── job_submit.sh         # SLURM batch script for job submission
├── environment_setup.sh  # Module loads and conda environment setup
├── run_experiment.py     # Main experiment wrapper with CLI
└── README.md             # This documentation
```

### Frontera System Overview

- **Nodes:** 8,368 compute nodes (Cascade Lake)
- **Cores per node:** 56 (2x Intel Xeon 8280)
- **Memory per node:** 192 GB
- **Interconnect:** Mellanox InfiniBand HDR

## Quick Start

If you're in a hurry, here's the minimum to get started:

```bash
# 1. First-time setup (run once)
cd hpc/frontera/idealized_inlet
./environment_setup.sh --install

# 2. Edit job_submit.sh to add your allocation
vi job_submit.sh
# Change: #SBATCH --account=YOUR_ALLOCATION

# 3. Submit a test job
sbatch -p development --nodes=1 job_submit.sh

# 4. Monitor the job
squeue -u $USER
```

## Environment Setup

### First-Time Installation

Run the installation script to set up Miniconda and the required Python environment:

```bash
# Make the script executable
chmod +x environment_setup.sh

# Run installation
./environment_setup.sh --install
```

This will:
1. Install Miniconda in `$WORK/miniconda3`
2. Create a conda environment named `swe4dvar`
3. Install FEniCSx, PETSc, mpi4py, and other dependencies
4. Install the swe4dvar package in development mode

### Manual Installation (Alternative)

If you prefer manual installation:

```bash
# Load modules
module purge
module load intel/19.1.1 impi/19.0.9 python3/3.9.7

# Create conda environment
conda create -n swe4dvar -c conda-forge \
    python=3.10 fenics-dolfinx petsc4py mpi4py numpy scipy matplotlib h5py

# Activate and install swe4dvar
conda activate swe4dvar
cd /path/to/SWEMniCS
pip install -e .
```

### Verifying the Environment

```bash
./environment_setup.sh --verify
```

Expected output:
```
Verifying environment...
  Python: Python 3.10.x
  MPI: Intel(R) MPI Library
  mpi4py: 3.x.x
  petsc4py: 3.x.x
  dolfinx: 0.x.x
  swe4dvar: installed
  Mesh file: found

Environment verification: PASSED
```

## Running Experiments

### SLURM Queues on Frontera

| Queue       | Max Nodes | Max Time | Use Case                    |
|-------------|-----------|----------|----------------------------|
| development | 4         | 2 hours  | Testing, debugging         |
| normal      | 512       | 48 hours | Production runs            |
| large       | 513+      | 48 hours | Very large scaling studies |

### Basic Job Submission

```bash
# Test run on development queue (1 node)
sbatch -p development job_submit.sh

# Production run on normal queue (4 nodes)
sbatch --nodes=4 job_submit.sh

# Large scaling study (8 nodes)
sbatch --nodes=8 --time=08:00:00 job_submit.sh
```

### Configuring Experiment Parameters

You can set experiment parameters via environment variables:

```bash
# Custom problem size and observation setup
export NX=200
export NY=200
export DT=600
export OBS_FRACTION=0.5
export OBS_INTERVAL=36  # 6 hours at dt=600s
export METHOD=4dvar

sbatch --nodes=4 job_submit.sh
```

Or modify the command line in `run_experiment.py`:

```bash
# Submit with custom parameters
sbatch --wrap="ibrun python run_experiment.py \
    --nx 200 --ny 200 \
    --dt 600 \
    --final-time 691200 \
    --method 4dvar \
    --obs-fraction 0.5 \
    --obs-interval 36 \
    --max-iter 100 \
    --profile"
```

### Command-Line Options

```
usage: run_experiment.py [-h] [--nx NX] [--ny NY] [--dt DT]
                         [--final-time FINAL_TIME] [--solver {CG,SUPG,DG,DGCG,DGNC}]
                         [--friction {linear,quadratic,mannings}]
                         [--method {4dvar,dcwme}] [--obs-fraction OBS_FRACTION]
                         [--obs-interval OBS_INTERVAL] [--noise-level NOISE_LEVEL]
                         [--background-error BACKGROUND_ERROR] [--max-iter MAX_ITER]
                         [--gtol GTOL] [--output-dir OUTPUT_DIR]
                         [--checkpoint-dir CHECKPOINT_DIR]
                         [--checkpoint-interval CHECKPOINT_INTERVAL]
                         [--restart] [--restart-iter RESTART_ITER]
                         [--profile] [--verbose]

Key options:
  --nx, --ny           Grid resolution (default: 100x100)
  --dt                 Time step in seconds (default: 1200)
  --final-time         Simulation length in seconds (default: 345600 = 4 days)
  --method             DA method: 4dvar or dcwme (default: 4dvar)
  --obs-fraction       Fraction of points to observe (default: 0.5)
  --obs-interval       Observe every N timesteps (default: 18 = 6 hours)
  --max-iter           Maximum optimization iterations (default: 50)
  --profile            Enable detailed timing output
  --restart            Restart from latest checkpoint
```

## Scaling Studies

### Running a Node Scaling Study

```bash
# Submit jobs for 1, 2, 4, 8 nodes
for N in 1 2 4 8; do
    export OUTPUT_DIR=$SCRATCH/swe4dvar/scaling/nodes_${N}
    sbatch --nodes=$N --job-name="inlet_n${N}" job_submit.sh
done
```

### Expected Scaling Performance

| Nodes | MPI Ranks | Expected Time* | Speedup | Efficiency |
|-------|-----------|----------------|---------|------------|
| 1     | 56        | ~60 min        | 1.0x    | 100%       |
| 2     | 112       | ~35 min        | 1.7x    | 85%        |
| 4     | 224       | ~20 min        | 3.0x    | 75%        |
| 8     | 448       | ~12 min        | 5.0x    | 62%        |

*Times are estimates for 100x100 grid, 4-day simulation, 50 iterations

### Memory Requirements

Estimated memory per rank for different problem sizes:

| Grid Size | DOFs (approx) | Memory/Rank | Total (1 node) |
|-----------|---------------|-------------|----------------|
| 50x50     | ~15,000       | ~0.5 GB     | ~30 GB         |
| 100x100   | ~60,000       | ~2 GB       | ~110 GB        |
| 200x200   | ~240,000      | ~8 GB       | ~448 GB        |

For very large problems, request multiple nodes to ensure sufficient memory.

## Monitoring Jobs

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# View job output in real-time
tail -f inlet_<job_id>_*.out
```

### Check Job Progress

```bash
# View optimization progress from output file
grep "Iteration" $SCRATCH/swe4dvar/inlet_results/<job_id>/logs/*.out

# Check memory usage
grep "Peak memory" $SCRATCH/swe4dvar/inlet_results/<job_id>/logs/*.out
```

### Cancel a Job

```bash
scancel <job_id>
```

## Retrieving Results

### Output Directory Structure

```
$SCRATCH/swe4dvar/inlet_results/<job_id>/
├── inlet_4dvar_results.json    # Main results file
├── checkpoints/
│   ├── checkpoint_0010_state.bin
│   ├── checkpoint_0010_meta.json
│   └── ...
├── logs/
│   ├── inlet_<job_id>_*.out    # SLURM stdout
│   └── inlet_<job_id>_*.err    # SLURM stderr
└── job_submit.sh               # Copy of submission script
```

### Viewing Results

```bash
# View results summary
cat $SCRATCH/swe4dvar/inlet_results/<job_id>/inlet_4dvar_results.json | python -m json.tool

# Copy results to local machine
scp frontera:$SCRATCH/swe4dvar/inlet_results/<job_id>/inlet_4dvar_results.json ./
```

### Results JSON Format

```json
{
  "method": "4dvar_mpi",
  "test_case": "idealized_inlet",
  "background_error": 0.123456,
  "analysis_error": 0.012345,
  "error_reduction": 90.0,
  "num_iterations": 42,
  "converged": true,
  "wall_time": 1234.56,
  "config": {
    "nx": 100,
    "ny": 100,
    "mpi_ranks": 224,
    "timing": {
      "total_time": 1234.56,
      "optimization_time": 1000.0,
      "peak_memory_mb": 2048.0
    }
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Module Not Found

```
ModuleNotFoundError: No module named 'swe4dvar'
```

**Solution:** Ensure the environment is set up correctly:
```bash
source environment_setup.sh
./environment_setup.sh --verify
```

#### 2. MPI Errors

```
MPI_Init: cannot allocate memory
```

**Solution:** Request more nodes or reduce problem size:
```bash
sbatch --nodes=4 job_submit.sh  # Use more nodes
# Or reduce NX, NY in job_submit.sh
```

#### 3. Mesh File Not Found

```
ERROR: Mesh file not found: .../data/Ideal_Inlet/Ideal_Inlet.xdmf
```

**Solution:** Ensure data files are present:
```bash
ls $PROJECT_ROOT/data/Ideal_Inlet/
# Should show Ideal_Inlet.xdmf and Ideal_Inlet.h5
```

#### 4. Job Killed (OOM)

```
slurmstepd: error: Detected 1 oom-kill event
```

**Solution:** Use more nodes or reduce grid size:
```bash
# Option 1: More nodes
sbatch --nodes=4 job_submit.sh

# Option 2: Smaller grid
export NX=50 NY=50
sbatch job_submit.sh
```

#### 5. Timeout

```
DUE TO TIME LIMIT
```

**Solution:** Request more time or use checkpointing:
```bash
# Request more time
sbatch --time=08:00:00 job_submit.sh

# Or restart from checkpoint
sbatch --wrap="ibrun python run_experiment.py --restart"
```

### Getting Help

1. **TACC Support:** https://portal.tacc.utexas.edu/user-support
2. **Frontera User Guide:** https://frontera-portal.tacc.utexas.edu/user-guide/
3. **SWE4DVar Issues:** https://github.com/UT-CHG/SWE4DVar/issues

### Useful Commands

```bash
# Check allocation balance
/usr/local/etc/taccinfo

# View job history
sacct -u $USER --starttime=2024-01-01

# Check node availability
sinfo -p normal

# Interactive session for debugging
idev -p development -N 1 -n 56 -t 02:00:00
```

## Advanced Topics

### Restarting from Checkpoint

If a job times out or fails, restart from the latest checkpoint:

```bash
sbatch --wrap="ibrun python run_experiment.py \
    --restart \
    --output-dir $SCRATCH/swe4dvar/inlet_results/<original_job_id>"
```

Or restart from a specific iteration:

```bash
sbatch --wrap="ibrun python run_experiment.py \
    --restart-iter 30 \
    --checkpoint-dir $SCRATCH/swe4dvar/inlet_results/<original_job_id>/checkpoints"
```

### Custom PETSc Options

For advanced performance tuning:

```bash
export PETSC_OPTIONS="-log_view -ksp_monitor -pc_type lu"
sbatch job_submit.sh
```

### Profiling with Intel Tools

```bash
# Load Intel profiling tools
module load vtune

# Profile MPI communication
vtune -collect hotspots -- ibrun python run_experiment.py --profile
```

## References

- [Frontera User Guide](https://frontera-portal.tacc.utexas.edu/user-guide/)
- [TACC Best Practices](https://portal.tacc.utexas.edu/best-practices)
- [PETSc Documentation](https://petsc.org/release/docs/)
- [FEniCSx Documentation](https://docs.fenicsproject.org/)

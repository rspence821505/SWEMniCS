# Shinnecock Inlet on TACC Frontera

This directory contains HPC infrastructure for running Shinnecock Inlet simulations and data assimilation experiments on TACC Frontera.

## Overview

Shinnecock Inlet is a real-world tidal inlet on the south shore of Long Island, NY. Unlike the idealized test cases, this simulation uses:

- **Real bathymetry** from ADCIRC input files
- **Tidal forcing** from 5 major constituents (M2, S2, N2, K1, O1)
- **Spherical coordinates** with CPP projection
- **Manning's friction** with spatially-varying roughness

## Quick Start

### 1. Setup Environment

```bash
# SSH to Frontera
ssh username@frontera.tacc.utexas.edu

# Clone/sync the project
cd $WORK
git clone https://github.com/UT-CHG/SWE4DVar.git
cd swe4dvar

# Create conda environment (first time only)
module load python3
conda env create -f environment.yml
```

### 2. Verify Data Files

Shinnecock requires pre-processed ADCIRC data files. Check that these exist:

```bash
ls -la examples/data/shinnecock_inlet*
# Should show:
#   shinnecock_inlet_mesh.bp     (mesh in ADIOS format)
#   shinnecock_inlet_depth.bp    (bathymetry)
#   shinnecock_inlet_boundary.json  (boundary conditions + tidal forcing)
```

If missing, see [Data Preparation](#data-preparation) below.

### 3. Submit a Job

```bash
cd hpc/frontera/shinnecock

# Edit job_submit.sh to set your allocation
sed -i 's/YOUR_ALLOCATION/your-actual-allocation/' job_submit.sh

# Forward simulation (1 node, ~5 days of simulation)
sbatch job_submit.sh

# 4D-Var data assimilation (4 nodes)
DA_MODE=4dvar NODES=4 sbatch --nodes=4 job_submit.sh

# DC-WME data assimilation (8 nodes)
DA_MODE=dcwme NODES=8 sbatch --nodes=8 job_submit.sh
```

## File Structure

```
hpc/frontera/shinnecock/
├── job_submit.sh         # SLURM batch script
├── environment_setup.sh  # Module loads and environment configuration
├── run_experiment.py     # HPC wrapper with memory monitoring
└── README.md            # This file
```

## Configuration Options

### SLURM Job Script (`job_submit.sh`)

Override settings via environment variables:

```bash
# Simulation parameters
DT=300 sbatch job_submit.sh              # 5-minute timestep
T_HOURS=48 sbatch job_submit.sh          # 48-hour simulation
SOLVER=supg sbatch job_submit.sh         # Use SUPG solver

# DA parameters
DA_MODE=4dvar \
OBS_FRACTION=0.3 \
OBS_FREQUENCY=12 \
sbatch --nodes=4 job_submit.sh
```

### Command-Line Arguments

The underlying `shinnecock.py` script accepts:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dt` | 600 | Time step in seconds |
| `--T` | 119 | Simulation time in hours |
| `--solver` | dg | Solver: dg, supg, dgnc |
| `--alpha` | 1.5 | Wetting/drying parameter |
| `--da-mode` | none | DA mode: none, 4dvar, dcwme |
| `--obs-fraction` | 0.5 | Fraction of points to observe |
| `--obs-frequency` | 6 | Observe every N timesteps |
| `--verbose` | - | Enable verbose output |
| `--profile` | - | Enable timing profiling |

## Resource Requirements

### Frontera Node Specifications

- 56 cores per node (Intel Xeon Platinum 8280)
- 192 GB RAM per node (~3.4 GB per core)
- HDR InfiniBand interconnect

### Memory Estimates

| Mode | Duration | Est. Memory | Recommended Nodes |
|------|----------|-------------|-------------------|
| Forward | 24h | ~2 GB | 1 |
| Forward | 120h | ~8 GB | 2 |
| 4D-Var | 24h | ~50 GB | 2 |
| 4D-Var | 48h | ~100 GB | 4 |
| DC-WME | 24h | ~50 GB | 2 |
| DC-WME | 48h | ~100 GB | 4 |

**Note:** DA experiments require significant memory for Jacobian storage. Use more nodes if you see out-of-memory errors.

### Expected Runtimes

Approximate wall times on Frontera (with 4D-Var DA):

| Nodes | 24h Simulation | 48h Simulation |
|-------|----------------|----------------|
| 1 | ~45 min | ~90 min |
| 2 | ~25 min | ~50 min |
| 4 | ~15 min | ~30 min |
| 8 | ~10 min | ~20 min |

## Data Preparation

### Obtaining Shinnecock Data

The Shinnecock Inlet mesh and forcing data come from ADCIRC. To prepare:

1. **Get ADCIRC files** (fort.14, fort.15, etc.) from the ADCIRC test suite
2. **Run the conversion script:**

```bash
cd examples
python scripts/adcirc_to_adios.py \
    --input-dir /path/to/adcirc/shinnecock \
    --output-prefix data/shinnecock_inlet
```

This creates:
- `shinnecock_inlet_mesh.bp` - Mesh in ADIOS format
- `shinnecock_inlet_depth.bp` - Bathymetry function
- `shinnecock_inlet_boundary.json` - Boundary info and tidal constituents

### Data File Locations

On Frontera, pre-converted data may be available at:

```bash
# Check if shared data exists
ls /scratch/projects/swe4dvar/data/shinnecock_inlet*

# If available, symlink to your workspace
ln -s /scratch/projects/swe4dvar/data/shinnecock_inlet* examples/data/
```

## Running Experiments

### Forward Simulation

Basic forward run to verify setup:

```bash
sbatch job_submit.sh

# Check progress
squeue -u $USER
tail -f shinnecock_*.out
```

### 4D-Var Data Assimilation

Twin experiment with synthetic observations:

```bash
# 24-hour experiment, 50% observation coverage
DA_MODE=4dvar \
T_HOURS=24 \
OBS_FRACTION=0.5 \
OBS_FREQUENCY=6 \
sbatch --nodes=2 job_submit.sh
```

### DC-WME Data Assimilation

Diffusive-Corrective Weak-constraint Maximum Entropy:

```bash
DA_MODE=dcwme \
T_HOURS=24 \
sbatch --nodes=2 job_submit.sh
```

### Scaling Study

Test strong scaling across node counts:

```bash
python run_experiment.py \
    --mode scaling \
    --min-nodes 1 \
    --max-nodes 8 \
    --T 24
```

## Output Files

Results are written to `$SCRATCH/swe4dvar_outputs/`:

```
swe4dvar_outputs/
├── data/
│   ├── DG_p1_shinnecock_frontera_h.csv      # Water height time series
│   ├── DG_p1_shinnecock_frontera_xvel.csv   # X velocity
│   ├── DG_p1_shinnecock_frontera_yvel.csv   # Y velocity
│   └── shinnecock_4dvar_results.json        # DA results (if applicable)
├── figures/
│   ├── shinnecock_frontera_height_spherical.png
│   ├── shinnecock_frontera_xvel_spherical.png
│   └── shinnecock_frontera_yvel_spherical.png
├── logs/
│   └── newton_*.log                          # Newton solver diagnostics
└── checkpoints/
    └── (checkpoint files for restart)
```

## Troubleshooting

### Common Issues

**1. "Data file not found" error**
```bash
# Verify data files exist
ls examples/data/shinnecock_inlet*

# Check file permissions
chmod 644 examples/data/shinnecock_inlet*
```

**2. Out of memory errors**
```bash
# Use more nodes or fewer tasks per node
sbatch --nodes=4 --ntasks-per-node=28 job_submit.sh
```

**3. Conda environment not found**
```bash
# Recreate environment
module load python3
conda env create -f environment.yml -n swe4dvar
```

**4. MPI errors**
```bash
# Check module conflicts
module list

# Reset and reload
module reset
source hpc/frontera/shinnecock/environment_setup.sh
```

### Getting Help

- **TACC Support:** https://portal.tacc.utexas.edu/
- **Frontera User Guide:** https://frontera-portal.tacc.utexas.edu/user-guide/
- **SWE4DVar Issues:** [GitHub Issues](https://github.com/UT-CHG/SWE4DVar/issues)

## Advanced Usage

### Custom Observation Networks

Edit the observation configuration in `shinnecock.py`:

```python
# Define specific observation locations (lon, lat)
custom_stations = np.array([
    [-72.476519, 40.840969, 0.0],  # Channel station
    [-72.480000, 40.845000, 0.0],  # Inlet mouth
    [-72.470000, 40.835000, 0.0],  # Bay interior
])
```

### Checkpoint/Restart

For long simulations, enable checkpointing:

```bash
# In job_submit.sh
PYTHON_CMD+=" --checkpoint-interval 100"

# Restart from checkpoint
PYTHON_CMD+=" --restart-from checkpoints/shinnecock_step_1000.h5"
```

### Profiling and Performance Analysis

Enable detailed profiling:

```bash
PROFILE=true sbatch job_submit.sh

# View timing breakdown in output
grep "Timer" shinnecock_*.out
```

## References

1. Shinnecock Inlet bathymetry: NOAA National Ocean Service
2. Tidal constituents: ADCIRC tidal database
3. Westerink, J.J., et al. (2008). A basin-to-channel-scale unstructured grid hurricane storm surge model. *Monthly Weather Review*, 136(3), 833-864.

## Version History

- **v1.0** (2024-01): Initial HPC scripts for Frontera
- **v1.1** (2024-02): Added DA experiment support
- **v1.2** (2024-03): Memory optimization and scaling improvements

# Twin Experiment Workflow for 4D-Var Data Assimilation

This document describes the twin experiment (also called OSSE - Observing System Simulation Experiment) workflow implemented in the generalized `TwinExperiment` framework for testing 4D-Var and DC-WME data assimilation.

## Overview

A twin experiment is a controlled test where we:
1. Generate a "truth" simulation
2. Create synthetic observations from the truth (with added noise)
3. Perturb the initial condition to simulate uncertainty
4. Run data assimilation to recover the true initial condition
5. Measure how well DA reduces the error

This allows validation of the DA system in a controlled setting where we know the true answer.

## Generalized Framework

The twin experiment framework is designed to work with **any problem** that inherits from `BaseProblem` or `TidalProblem`, including:

- `TidalProblem` - Simple rectangular tidal test case
- `ADCIRCProblem` - Real-world ADCIRC meshes (e.g., Shinnecock Inlet)
- `IdealizedInlet` - Idealized inlet from XDMF mesh
- Custom problems implementing the same interface

### Key Files

| File | Description |
|------|-------------|
| `experiments/twin_experiment.py` | Core framework with `TwinExperiment` class |
| `experiments/run_twin_experiment.py` | Command-line runner |
| `examples/shinnecock.py` | Shinnecock example using the framework |
| `experiments/serial_da/da_experiment_utils.py` | Additional utilities |

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        4D-VAR TWIN EXPERIMENT WORKFLOW                      │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   m_true     │  True initial condition
                              │   (u₀)       │
                              └──────┬───────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │   Forward    │  │   Forward    │  │   Forward    │
            │   Model      │  │   Model      │  │   Model      │
            │   Step 1     │  │   Step 2     │  │   Step N     │
            └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                   │                 │                 │
                   ▼                 ▼                 ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │     u₁       │  │     u₂       │  │     uₙ       │
            └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                   │                 │                 │
                   ▼                 ▼                 ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │   H(u₁)      │  │   H(u₂)      │  │   H(uₙ)      │  Observation
            │   + noise    │  │   + noise    │  │   + noise    │  Operator
            └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                   │                 │                 │
                   ▼                 ▼                 ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │     y₁       │  │     y₂       │  │     yₙ       │  Synthetic
            │              │  │              │  │              │  Observations
            └──────────────┘  └──────────────┘  └──────────────┘

                              ┌──────────────┐
                              │ m_background │  Perturbed initial condition
                              │ = m_true + ε │  (simulates uncertainty)
                              └──────┬───────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │   4D-Var     │
                              │   Minimize   │  J(m) = J_b + J_obs
                              │   J(m)       │
                              └──────┬───────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │  m_analysis  │  Recovered initial condition
                              └──────────────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │   Compare    │  ||m_analysis - m_true||
                              │   Errors     │  vs ||m_background - m_true||
                              └──────────────┘
```

## Quick Start

### Using the Command-Line Runner

The simplest way to run a twin experiment is using the command-line runner:

```bash
# Tidal problem with 4D-Var
python experiments/run_twin_experiment.py --problem tidal --method 4dvar

# Tidal problem with DC-WME
python experiments/run_twin_experiment.py --problem tidal --method dcwme

# Shinnecock Inlet with 4D-Var
python experiments/run_twin_experiment.py --problem shinnecock --method 4dvar \
    --adios-file data/shinnecock_inlet --dt 600 --T 24

# Quick test with small grid
python experiments/run_twin_experiment.py --problem tidal --nx 5 --ny 3 \
    --final-time 7200 --max-iter 10

# MPI parallel run
mpirun -n 4 python experiments/run_twin_experiment.py --problem tidal
```

### Using the Python API

For more control, use the `TwinExperiment` class directly:

```python
from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig
from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver

# Create problem
problem = TidalProblem(nx=10, ny=5, dt=3600, nt=24)

# Create solver
solver = get_solver("SUPG")(problem, theta=0.5, p_degree=[1, 1])

# Configure experiment
config = TwinExperimentConfig(
    method="4dvar",           # or "dcwme"
    obs_fraction=0.5,         # Observe 50% of interior nodes
    obs_frequency=1,          # Every timestep
    obs_noise_level=0.01,     # 1% noise
    background_error_std=0.1, # 10% background error
    max_iterations=50,
    verbose=True,
)

# Run experiment
experiment = TwinExperiment(problem, solver, config)
results = experiment.run()

# Access results
print(f"Background error: {results.background_error:.6f}")
print(f"Analysis error: {results.analysis_error:.6f}")
print(f"Error reduction: {results.error_reduction:.1f}%")
```

### Using with Shinnecock

The `shinnecock.py` example uses the generalized framework:

```bash
# Run 4D-Var DA experiment
python examples/shinnecock.py --da-mode 4dvar --T 12 --verbose

# Run DC-WME DA experiment
python examples/shinnecock.py --da-mode dcwme --T 12 --verbose

# Forward simulation only (no DA)
python examples/shinnecock.py --da-mode none
```

## Configuration Options

### TwinExperimentConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `"4dvar"` | DA method: `"4dvar"` or `"dcwme"` |
| `obs_fraction` | `0.5` | Fraction of nodes to observe (0.0-1.0) |
| `obs_frequency` | `1` | Observe every N timesteps |
| `obs_noise_level` | `0.01` | Noise level (fraction of signal) |
| `obs_points_file` | `None` | JSON file with pre-selected observation points |
| `interior_only` | `True` | Only observe interior nodes (recommended) |
| `background_error_std` | `0.1` | Background error std dev (fraction) |
| `max_iterations` | `50` | Maximum optimization iterations |
| `gradient_tolerance` | `1e-6` | Gradient convergence tolerance |
| `cost_tolerance` | `1e-8` | Cost function convergence tolerance |
| `use_bounds` | `True` | Use bounded optimization (h ≥ h_min) |
| `h_min` | `0.01` | Minimum water depth bound |
| `component_aware_cov` | `False` | Use different variances for h vs u,v |
| `output_dir` | `"outputs/data"` | Directory for result files |
| `verbose` | `True` | Enable verbose output |
| `obs_seed` | `42` | Random seed for observations |
| `background_seed` | `123` | Random seed for background perturbation |

### Command-Line Options

The `run_twin_experiment.py` script accepts:

```
Problem Selection:
  --problem PROBLEM     Problem type: tidal, shinnecock, adcirc

DA Method:
  --method METHOD       DA method: 4dvar, dcwme

Problem Parameters (Tidal):
  --nx NX               Elements in x direction
  --ny NY               Elements in y direction
  --dt DT               Time step in seconds
  --final-time TIME     Final time in seconds
  --T HOURS             Final time in hours (overrides --final-time)

Problem Parameters (ADCIRC/Shinnecock):
  --adios-file PATH     Path to ADCIRC ADIOS files
  --dramp DAYS          Tidal ramp-up time in days
  --alpha ALPHA         Wetting/drying alpha parameter

Solver Parameters:
  --solver TYPE         Solver type: CG, SUPG, DG, DGNC
  --theta THETA         Time-stepping scheme (0=IE, 0.5=BDF2, 1=CN)

Observation Parameters:
  --obs-fraction FRAC   Fraction of points to observe
  --obs-frequency N     Observe every N timesteps
  --obs-noise LEVEL     Observation noise level
  --obs-points-file F   JSON file with observation points
  --all-nodes           Observe all nodes (not just interior)

Background Error:
  --background-error E  Background error std deviation

Optimization:
  --max-iter N          Maximum optimization iterations
  --no-bounds           Disable bounded optimization
  --h-min H             Minimum water depth for bounds
  --component-aware-cov Use component-aware covariance

Output:
  --output-dir DIR      Output directory for results
  --verbose             Enable verbose output
  --quiet               Suppress output
```

## Step-by-Step Workflow Details

### Step 1: Generate "Truth" Trajectory

The framework runs the forward model with the default initial condition:

```python
# Inside TwinExperiment._generate_truth()
solver.time_loop(
    solver_parameters=solver_params,
    save_state=True,
    store_jacobians=True,  # Needed for adjoint
)

# Store truth
truth_trajectory = [state.copy() for state in solver.storage.saved_states]
m_true = truth_trajectory[0]  # True initial condition
```

### Step 2: Create Observation Points

The framework generates observation points from interior mesh nodes by default:

```python
# Interior-only observation points (recommended)
obs_points = experiment._generate_interior_observation_points()

# Or from file
config = TwinExperimentConfig(
    obs_points_file="my_stations.json"
)
```

**Why interior-only?** The discrete adjoint Jacobians have identity rows at boundary DOFs due to strong BC imposition, which causes incorrect gradient propagation for boundary observations.

### Step 3: Generate Synthetic Observations

Observations are created by applying the observation operator and adding noise:

```python
# y_k = H(u_k) + ε
for k in obs_times:
    y_true = obs_operator.forward(truth_trajectory[k])
    noise = np.random.normal(0, noise_level * signal_magnitude)
    y_obs = y_true + noise
```

### Step 4: Create Perturbed Background

```python
# m_background = m_true + perturbation
perturbation = np.random.normal(0, background_error_std * |m_true|)
m_background = m_true + perturbation
```

### Step 5: Setup Cost Function

The framework supports both 4D-Var and DC-WME:

**Standard 4D-Var:**
```
J(m) = ½(m - m_b)ᵀ B⁻¹ (m - m_b) + ½ Σₖ (H(uₖ) - yₖ)ᵀ R⁻¹ (H(uₖ) - yₖ)
       └────────────────────────┘   └─────────────────────────────────────┘
            Background term                    Observation term
```

**DC-WME 4D-Var:** Uses cumulative time-averaged innovation as QoI for improved stability.

### Step 6: Run Optimization

The framework uses PETSc TAO L-BFGS with optional bounds:

```python
# Bounded optimization (default, recommended)
optimizer = PETScTAOWrapper(
    cost_function,
    tao_type="blmvm",  # Bounded L-BFGS
    lower_bounds=lower,
    upper_bounds=upper,
    options=opt_options,
)
m_analysis = optimizer.solve(m_background)
```

### Step 7: Evaluate Results

```python
# Analysis error
analysis_error = ||m_analysis - m_true||

# Error reduction (should be positive)
error_reduction = (background_error - analysis_error) / background_error * 100
```

## Example Output

```
======================================================================
TidalProblem 4DVAR Twin Experiment
======================================================================
MPI ranks: 1
Time step: 3600.0 s
Number of time steps: 24
Observation fraction: 0.5
Observation frequency: every 1 timesteps
Noise level: 0.01
Background error: 0.1
======================================================================

Step 1: Generating truth trajectory...
  Truth trajectory: 25 states

Step 2: Setting up observations...
  Generated 45 interior observation points

Step 3: Generating synthetic observations...
  Observations generated with mean noise std: 0.098765

Step 4: Setting up background state...
  Background RMS error: 0.123456

Step 5: Setting up covariance matrices...
  Background covariance: diagonal, variance = 9.87e-01
  Observation covariance: diagonal, variance = 9.75e-03

Step 6: Creating forward model wrapper...

Step 7: Setting up 4DVAR cost function...
  Zeroing 48 boundary DOF gradients

Step 8: Running optimization...
  Using bounded L-BFGS (h_min=0.01)
======================================================================
PETSc TAO Optimization (type=blmvm)
======================================================================
  Iter            Cost        ||grad||         Step
----------------------------------------------------------------------
     0    1.23456789e+02   9.87654321e+01          N/A
     1    8.76543210e+01   5.43210987e+01   1.00000e+00
    ...
    25    1.23456789e+00   9.87654321e-07   1.00000e+00
----------------------------------------------------------------------
Converged: GATOL - ||gradient|| < gatol
======================================================================

  Optimization completed in 45.32 seconds
  Iterations: 25
  Converged: True

Step 9: Evaluating results...
  Analysis RMS error: 0.045678
  Error reduction: 63.0%
  Innovation mean: 0.000123
  Innovation std: 0.009876

Results saved to: outputs/data/tidalproblem_4dvar_results.json

======================================================================
SUMMARY: TidalProblem 4DVAR Experiment
======================================================================
Background error:  0.123456
Analysis error:    0.045678
Error reduction:   63.0%
Iterations:        25
Converged:         True
Total time:        52.45 s
======================================================================
```

## Output Files

Results are saved as JSON files:

```
outputs/
├── data/
│   ├── tidalproblem_4dvar_results.json
│   ├── tidalproblem_dcwme_results.json
│   ├── adcircproblem_4dvar_results.json
│   └── ...
└── figures/
    └── (verification plots)
```

### Result File Contents

```json
{
  "method": "4dvar",
  "problem_name": "TidalProblem",
  "cost_history": [123.45, 87.65, ...],
  "gradient_norm_history": [98.76, 54.32, ...],
  "background_error": 0.123456,
  "analysis_error": 0.045678,
  "error_reduction": 63.0,
  "innovation_mean": 0.000123,
  "innovation_std": 0.009876,
  "num_iterations": 25,
  "converged": true,
  "wall_time": 52.45,
  "config": {
    "method": "4dvar",
    "obs_fraction": 0.5,
    ...
  },
  "problem_config": {
    "problem_type": "TidalProblem",
    "nx": 10,
    "ny": 5,
    "dt": 3600.0,
    "nt": 24
  }
}
```

## Extending to Custom Problems

To use the framework with a custom problem:

1. **Create a problem class** inheriting from `BaseProblem` or `TidalProblem`
2. **Implement required methods**: `_create_mesh()`, `create_bathymetry()`, etc.
3. **Use the TwinExperiment class** with your problem instance

```python
from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig
from my_module import MyCustomProblem

# Create custom problem
problem = MyCustomProblem(
    mesh_file="my_mesh.xdmf",
    dt=300,
    nt=100,
)

# Create solver
solver = get_solver("SUPG")(problem, theta=0.5, p_degree=[1, 1])

# Run twin experiment
experiment = TwinExperiment(problem, solver)
results = experiment.run()
```

## Problem Sizes and Memory Requirements

| Configuration | Elements | DG DOFs | Timesteps | Jacobian Memory |
|---------------|----------|---------|-----------|-----------------|
| Quick test | ~100 | ~900 | 24 | ~50 MB |
| Tidal (default) | ~200 | ~1800 | 24 | ~100 MB |
| Shinnecock (1 hr) | ~10,000 | ~90,000 | 6 | ~200 MB |
| Shinnecock (12 hr) | ~10,000 | ~90,000 | 72 | ~2.4 GB |
| Shinnecock (24 hr) | ~10,000 | ~90,000 | 144 | ~4.8 GB |

## Tips for Successful Experiments

1. **Start small**: Use quick tests with small grids and short simulations to validate setup
2. **Use interior-only observations**: Boundary observations can cause gradient errors
3. **Enable bounded optimization**: Prevents negative water depths
4. **Check convergence**: Error reduction should be positive if DA is working
5. **Use MPI for large problems**: The framework is fully parallelized

## References

1. Spence, R., Butler, T., & Dawson, C. (2025). Data-Consistent Variational Data Assimilation. *Submitted*.

2. Kalnay, E. (2003). Atmospheric Modeling, Data Assimilation and Predictability. Cambridge University Press.

3. Le Dimet, F.-X., & Talagrand, O. (1986). Variational algorithms for analysis and assimilation of meteorological observations: theoretical aspects. *Tellus A*, 38(2), 97-110.

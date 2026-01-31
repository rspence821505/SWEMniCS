# SWE4DVar Quick Start Guide

This guide will get you up and running with SWE4DVar in minutes.

## Installation

### Option 1: Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/UT-CHG/SWE4DVar.git
cd SWE4DVar

# Create environment with FEniCSx
conda create -n swe4dvar python=3.11
conda activate swe4dvar
conda install -c conda-forge fenics-dolfinx mpich pyvista scipy matplotlib h5py

# Install SWE4DVar
pip install -e ".[all]"
```

### Option 2: Docker

```bash
# Run DOLFINx container
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm \
    ghcr.io/fenics/dolfinx/dolfinx:v0.9.0

# Inside container
pip install -e .
```

### Verify Installation

```python
import swe4dvar
print(f"SWE4DVar version: {swe4dvar.__version__}")

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
print("Installation successful!")
```

---

## Tutorial 1: Forward Simulation

Run a tidal flow simulation in a rectangular channel.

### Step 1: Define the Problem

```python
from swe4dvar.forward.problems import TidalProblem

# Create a tidal problem
problem = TidalProblem(
    nx=40,              # 40 elements in x-direction
    ny=10,              # 10 elements in y-direction
    dt=3600,            # 1 hour time step
    nt=168,             # 168 steps = 7 days
    friction_law="mannings",
    solution_var="h"    # Solve for total depth h (not elevation eta)
)

print(f"Mesh: {problem.nx} x {problem.ny} elements")
print(f"Simulation: {problem.nt} time steps of {problem.dt}s")
```

### Step 2: Create a Solver

```python
from swe4dvar.forward.solvers import get_solver

# Get SUPG solver class
SolverClass = get_solver("SUPG")

# Create solver instance
solver = SolverClass(
    problem,
    theta=1.0,         # Implicit Euler (unconditionally stable)
    p_degree=[1, 1],   # Linear elements for elevation and velocity
    verbose=True
)
```

### Step 3: Run the Simulation

```python
# Define Newton solver parameters
solver_params = {
    "rtol": 1e-5,    # Relative tolerance
    "atol": 1e-6,    # Absolute tolerance
    "max_it": 10     # Maximum Newton iterations
}

# Run time loop
solver.time_loop(
    solver_parameters=solver_params,
    save_state=True  # Store states for visualization/analysis
)

print(f"Simulation complete! Stored {len(solver.saved_states)} states")
```

### Step 4: Visualize Results (Optional)

```python
import matplotlib.pyplot as plt
import numpy as np

# Extract elevation at center of domain over time
times = np.arange(problem.nt) * problem.dt / 3600  # Convert to hours

# Get elevation from saved states
elevations = []
for state in solver.saved_states:
    # Extract elevation component
    eta = state.split()[0]  # First component is elevation
    elevations.append(eta.x.array.mean())

plt.figure(figsize=(10, 4))
plt.plot(times, elevations)
plt.xlabel("Time (hours)")
plt.ylabel("Mean Elevation (m)")
plt.title("Tidal Flow Simulation")
plt.grid(True)
plt.savefig("outputs/figures/tidal_elevation.png", dpi=150)
plt.show()
```

---

## Tutorial 2: Data Assimilation with 4D-Var

Estimate unknown parameters using observations.

### Step 1: Generate Synthetic Observations

```python
import numpy as np
from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver

# Create "truth" simulation with known friction
true_problem = TidalProblem(
    nx=40, ny=10, dt=3600, nt=48,  # 2 days
    friction_law="mannings",
    solution_var="h"
)
true_problem.TAU = 0.025  # True Manning's n

true_solver = get_solver("SUPG")(true_problem, theta=1.0, p_degree=[1, 1])
true_solver.time_loop({"rtol": 1e-5, "atol": 1e-6, "max_it": 10}, save_state=True)

# Extract observations at specific times (every 6 hours)
obs_times = [6, 12, 18, 24, 30, 36, 42, 48]
observations = []
obs_locations = np.array([[0.25, 0.5], [0.5, 0.5], [0.75, 0.5]])  # 3 stations

for t_idx in obs_times:
    state = true_solver.saved_states[t_idx - 1]
    # Add observation noise
    obs = extract_at_points(state, obs_locations)  # You'd implement this
    obs += np.random.normal(0, 0.01, obs.shape)    # 1cm noise
    observations.append(obs)

print(f"Created {len(observations)} observations at {len(obs_times)} times")
```

### Step 2: Set Up Cost Function

```python
from swe4dvar.data_assimilation import FourDVarCost, DiagonalCovariance

# Background state (initial guess with wrong friction)
background_problem = TidalProblem(
    nx=40, ny=10, dt=3600, nt=48,
    friction_law="mannings",
    solution_var="h"
)
background_problem.TAU = 0.030  # Wrong guess

background_solver = get_solver("SUPG")(background_problem, theta=1.0, p_degree=[1, 1])

# Create covariance matrices
B = DiagonalCovariance(np.array([0.005**2]))  # Background variance
R = DiagonalCovariance(np.full(3, 0.01**2))   # Observation variance (3 stations)

# Create cost function
cost_function = FourDVarCost(
    forward_model=background_solver,
    observation_operator=obs_op,  # You'd create this
    background_cov=B,
    observation_cov=R,
    m_background=np.array([0.030]),  # Background Manning's n
    observations=observations,
    obs_times=obs_times
)
```

### Step 3: Optimize

```python
from swe4dvar.optimization.lbfgs import LBFGSOptimizer

# Create optimizer
optimizer = LBFGSOptimizer(
    cost_function,
    memory_size=5,
    options={
        "max_iterations": 20,
        "gradient_tolerance": 1e-6,
        "verbose": True
    }
)

# Initial guess
m0 = np.array([0.030])  # Background friction

# Run optimization
m_optimal = optimizer.solve(m0)

print(f"True friction:      {0.025:.4f}")
print(f"Background guess:   {0.030:.4f}")
print(f"Estimated friction: {m_optimal[0]:.4f}")
```

---

## Tutorial 3: DC-WME-4DVar

Use Data-Consistent formulation for improved estimates.

### Step 1: Estimate Predictability Covariance

```python
from swe4dvar.data_assimilation import (
    DCWMEFourDVarCost,
    WeightedMeanErrorQoI,
    QoICovarianceEstimator,
)

# Assumes you already have:
# - forward_model `solver` (background run configuration)
# - background vector `m_b`
# - observations `observations` at indices `obs_times`
# - background covariance `B` and observation covariance `R`
```

### Step 2: Create DC-WME Cost Function

```python
# Create QoI map (uses I := obs_times, and K := max(I))
qoi_map = WeightedMeanErrorQoI(
    forward_model=solver,
    observation_operator=obs_op,
    observations=observations,
    observation_cov=R,
    obs_times=obs_times,
)

# Estimate predictability covariance L_wme = DQ_wme,K B DQ_wme,K^T
estimator = QoICovarianceEstimator(qoi_map=qoi_map, background_cov=B, num_samples=100)
L_wme = estimator.estimate(m_bar=m_b, time_index=max(obs_times))

# Create DC-WME cost function
dc_cost = DCWMEFourDVarCost(
    forward_model=solver,
    observation_operator=obs_op,
    background_cov=B,
    observation_cov=R,
    predicted_cov_wme=L_wme,
    m_background=m_b,
    observations=observations,
    obs_times=obs_times
)
```

### Step 3: Compare Methods

```python
# Optimize with standard 4D-Var
opt_4dvar = LBFGSOptimizer(cost_function, options={"verbose": False})
m_4dvar = opt_4dvar.solve(m0)

# Optimize with DC-WME-4DVar
opt_dcwme = LBFGSOptimizer(dc_cost, options={"verbose": False})
m_dcwme = opt_dcwme.solve(m0)

print("Results Comparison:")
print(f"  True value:     {0.025:.4f}")
print(f"  4D-Var:         {m_4dvar[0]:.4f}")
print(f"  DC-WME-4DVar:   {m_dcwme[0]:.4f}")
```

---

## Running in Parallel with MPI

SWE4DVar automatically supports MPI parallelization.

### Basic Parallel Run

```bash
# Run tidal example on 4 processes
mpirun -np 4 python examples/tidal.py --nx 100 --ny 25
```

### Parallel Python Script

```python
from mpi4py import MPI
from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver

# MPI is automatically initialized
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Create problem (mesh automatically partitioned)
problem = TidalProblem(nx=200, ny=50, dt=1800, nt=336)
solver = get_solver("SUPG")(problem, theta=1.0, p_degree=[1, 1])

# Run simulation (automatically parallel)
solver.time_loop({"rtol": 1e-5, "atol": 1e-6, "max_it": 10})

if rank == 0:
    print("Parallel simulation complete!")
```

---

## Common Tasks

### Save and Load Checkpoints

```python
from swe4dvar.utils import get_data_path
import pickle

# Save state
checkpoint_path = get_data_path("checkpoint_step100.pkl")
with open(checkpoint_path, 'wb') as f:
    pickle.dump(solver.saved_states[-1], f)

# Load state
with open(checkpoint_path, 'rb') as f:
    loaded_state = pickle.load(f)
solver.set_state(loaded_state)
```

### Monitor Performance

```python
from swe4dvar.utils import HierarchicalTimer

timer = HierarchicalTimer()

with timer.region("simulation"):
    with timer.region("forward"):
        solver.time_loop(solver_params, save_state=True)

    with timer.region("adjoint"):
        gradient = cost_function.gradient(m)

timer.report()
```

### Output Management

```python
from swe4dvar.utils import (
    ensure_output_dirs,
    get_figure_path,
    get_log_path,
    get_data_path
)

# Create output directories if needed
ensure_output_dirs()

# Get paths for outputs
fig_path = get_figure_path("convergence.png")
log_path = get_log_path("optimization.log")
data_path = get_data_path("results.h5")
```

---

## Next Steps

1. **Explore Examples**: See `examples/` directory for complete scripts
2. **Read Notebooks**: Interactive tutorials in `notebooks/`
3. **API Reference**: Detailed documentation in `docs/api_reference.md`
4. **Run Experiments**: Pre-configured experiments in `experiments/`

## Getting Help

- Check the [API Reference](api_reference.md) for detailed documentation
- Review example scripts in `examples/`
- Open an issue on GitHub for bugs or questions

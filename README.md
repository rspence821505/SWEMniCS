# SWE4DVar

**Shallow Water Equations 4D-Var Data Assimilation Framework**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FEniCSx 0.9.0](https://img.shields.io/badge/FEniCSx-0.9.0-green.svg)](https://fenicsproject.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python framework for solving the shallow water equations using FEniCSx with advanced numerical methods, adjoint-based sensitivity analysis, and 4D-Var data assimilation. Implements the Data-Consistent (DC) variational methods from Spence et al. (2025).

## Features

### Forward Model
- **Multiple Discretizations**: Continuous Galerkin (CG), Discontinuous Galerkin (DG), SUPG, and mixed element methods
- **Time Integration**: Implicit Euler, Crank-Nicolson, and BDF2 schemes
- **Physics**: Friction models (linear, quadratic, Manning's), Coriolis forcing, meteorological forcing, wetting/drying

### Data Assimilation
- **Standard 4D-Var**: Classical variational data assimilation
- **DC-4DVar**: Data-Consistent 4D-Var with predictability correction
- **DC-WME-4DVar**: Data-Consistent 4D-Var with Weighted Mean Error formulation
- **Adjoint Methods**: Implicit adjoint solver with Jacobian caching for ~50% cost savings
- **Checkpointing**: Memory-efficient strategies for long time windows

### Optimization
- **L-BFGS**: Limited-memory BFGS optimizer
- **Gauss-Newton**: Second-order optimization with Hessian-vector products
- **PETSc TAO**: Integration with PETSc's optimization toolkit

### Parallelization
- **Full MPI Support**: Scalable to thousands of cores
- **Parallel I/O**: Efficient checkpointing via ADIOS2
- **Load Balancing**: Metrics and analysis tools for parallel performance

## Installation

### Prerequisites

- Python 3.9+
- FEniCSx (DOLFINx 0.9.0)
- MPI implementation (MPICH or OpenMPI)

### From Source (Recommended)

```bash
git clone https://github.com/UT-CHG/SWE4DVar.git
cd SWE4DVar
```

#### Using Conda

```bash
# Create environment with FEniCSx
conda create -n swe4dvar python=3.11
conda activate swe4dvar
conda install -c conda-forge fenics-dolfinx mpich pyvista scipy matplotlib h5py adios4dolfinx

# Install package
pip install -e .

# With all optional dependencies
pip install -e ".[all]"
```

#### Using Docker

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm \
    ghcr.io/fenics/dolfinx/dolfinx:v0.9.0

# Inside container
pip install -e .
```

### pip Install

```bash
# Core package only
pip install --no-build-isolation -e .

# With examples dependencies
pip install --no-build-isolation -e ".[examples]"

# For development
pip install --no-build-isolation -e ".[dev]"
```

## Quick Start

### Forward Simulation

```python
from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver

# Define problem
problem = TidalProblem(
    nx=40, ny=10,
    dt=3600,           # 1 hour timestep
    nt=168,            # 7 days simulation
    friction_law="mannings",
    solution_var="h"
)

# Create solver (SUPG with implicit Euler)
solver = get_solver("SUPG")(problem, theta=1.0, p_degree=[1, 1])

# Run simulation
solver.time_loop(
    solver_parameters={"rtol": 1e-5, "atol": 1e-6, "max_it": 10},
    save_state=True
)
```

### 4D-Var Data Assimilation

```python
from swe4dvar.data_assimilation import (
    FourDVarCost, DCWMEFourDVarCost, create_cost_function
)
from swe4dvar.optimization.lbfgs import LBFGSOptimizer

# Setup cost function
cost = FourDVarCost(
    forward_model=solver,
    observation_operator=obs_op,
    background_cov=B,
    observation_cov=R,
    m_background=m_b,
    observations=y_obs,
    obs_times=obs_times
)

# Or use DC-WME formulation
dc_cost = DCWMEFourDVarCost(
    forward_model=solver,
    observation_operator=obs_op,
    background_cov=B,
    observation_cov=R,
    predicted_cov_wme=L,
    m_background=m_b,
    observations=y_obs,
    obs_times=obs_times
)

# Optimize
optimizer = LBFGSOptimizer(cost, max_iter=50, gtol=1e-6)
m_analysis = optimizer.minimize(m_b)
```

### MPI Parallel Execution

```bash
# Forward simulation
mpirun -np 4 python examples/tidal.py --nx 100 --ny 25

# Data assimilation experiment
mpirun -np 8 python experiments/parallel_da/run_experiment.py
```

## Examples

| Example | Description |
|---------|-------------|
| `examples/tidal.py` | Tidal flow in a rectangular channel |
| `examples/dam_break.py` | Classical dam break benchmark |
| `examples/idealized_inlet.py` | Coastal inlet with tidal forcing |
| `examples/shinnecock.py` | Shinnecock Inlet (real bathymetry) |
| `examples/complete_4dvar_example.py` | Full 4D-Var demonstration |

Run examples:
```bash
python examples/tidal.py --solver SUPG --nx 40 --ny 10
python examples/dam_break.py --solver DG
python examples/idealized_inlet.py
```

## Project Structure

```
SWE4DVar/
├── src/swe4dvar/
│   ├── forward/              # Forward model
│   │   ├── problems.py       # Problem definitions (Tidal, DamBreak, etc.)
│   │   ├── solvers/          # CG, DG, SUPG solver implementations
│   │   ├── newton.py         # Custom Newton solver
│   │   └── variational_forms.py
│   ├── adjoint/              # Adjoint computation
│   │   ├── implicit_adjoint.py
│   │   ├── tangent_linear.py
│   │   └── checkpointing.py
│   ├── data_assimilation/    # 4D-Var framework
│   │   ├── cost_functions.py # 4D-Var, DC-4DVar, DC-WME
│   │   ├── observation_operator.py
│   │   ├── covariance.py
│   │   └── qoi_maps.py       # QoI maps for DC methods
│   ├── optimization/         # Optimization algorithms
│   │   ├── lbfgs.py
│   │   ├── gauss_newton.py
│   │   └── petsc_tao_wrapper.py
│   ├── physics/              # Physical models
│   └── utils/                # Utilities (parallel, timing, I/O)
├── examples/                 # Example scripts
├── experiments/              # DA experiment configurations
│   ├── serial_da/            # Serial DA experiments
│   └── parallel_da/          # MPI parallel experiments
├── notebooks/                # Jupyter notebooks
├── hpc/                      # HPC deployment (Frontera)
├── outputs/                  # Generated outputs
│   ├── logs/
│   ├── figures/
│   ├── checkpoints/
│   └── data/
└── tests/                    # Test suite
```

## Solver Options

| Solver | Description | Use Case |
|--------|-------------|----------|
| `CG` | Continuous Galerkin | General purpose, smooth solutions |
| `SUPG` | Streamline-Upwind Petrov-Galerkin | Advection-dominated flows |
| `DG` | Discontinuous Galerkin | Shock-capturing, discontinuities |
| `DGCG` | DG velocity, CG elevation | Mixed formulation |

## Time Stepping

| `theta` | Scheme | Properties |
|---------|--------|------------|
| `1.0` | Implicit Euler | 1st order, unconditionally stable |
| `0.5` | Crank-Nicolson | 2nd order, energy-conserving |
| `0.0` | BDF2 | 2nd order, A-stable |

## Documentation

- **[Quick Start Guide](docs/quickstart.md)** - Get started quickly
- **[API Reference](docs/api_reference.md)** - Detailed API documentation
- **[Notebooks](notebooks/)** - Interactive tutorials and examples

## Known Issues

The following issues have been identified and are being addressed:

| Issue | Severity | Workaround |
|-------|----------|------------|
| Jacobian caching disabled by default | Critical | Pass `store_jacobians=True` to adjoint solver |
| Gauss-Newton Hessian incomplete | Critical | Use L-BFGS optimizer instead |
| R^{-1/2} fallback to identity | Critical | Ensure observation covariance is well-conditioned |

**Important:** For reliable optimization results, always enable Jacobian caching:

```python
from swe4dvar.adjoint import ImplicitAdjointSolver

adjoint = ImplicitAdjointSolver(
    forward_model=solver,
    store_jacobians=True  # Required for consistent gradients
)
```

See [`outputs/reports/bug_hunt_report.md`](outputs/reports/bug_hunt_report.md) for the complete list of known issues.

## Testing

```bash
# Run test suite
pytest tests/

# With MPI
mpirun -np 4 python -m pytest tests/

# Specific tests
pytest tests/test_variational_forms.py -v
```

## Dependencies

| Package | Purpose |
|---------|---------|
| FEniCSx (DOLFINx 0.9.0) | Finite element framework |
| PETSc / petsc4py | Linear algebra, solvers |
| mpi4py | MPI parallelization |
| NumPy / SciPy | Numerical computing |
| h5py | HDF5 I/O (optional) |
| adios4dolfinx | Parallel I/O (optional) |
| PyVista | Visualization (optional) |
| Matplotlib | Plotting (optional) |

## Citation

If you use SWE4DVar in your research, please cite:

```bibtex
@article{spence2025dc4dvar,
  title={Data-Consistent Variational Data Assimilation},
  author={Spence, Rylan and Butler, Troy and Dawson, Clint},
  journal={Submitted},
  year={2025}
}

@software{swe4dvar2025,
  title={SWE4DVar: Shallow Water Equations 4D-Var Data Assimilation Framework},
  author={Spence, Rylan and Pachev, Benjamin},
  year={2025},
  url={https://github.com/UT-CHG/SWE4DVar}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments

This project builds upon:
- [FEniCSx](https://fenicsproject.org/) - The FEniCS Project finite element framework
- [PETSc](https://petsc.org/) - Portable, Extensible Toolkit for Scientific Computation
- [ADCIRC](https://adcirc.org/) - Advanced Circulation Model (mesh formats and test cases)

Development supported by the Computational Hydraulics Group at UT Austin.

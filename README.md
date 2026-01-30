# SWE4DVar

**Shallow Water Equations with Modern Numerical Methods for Coastal Systems**

A comprehensive Python framework for solving the shallow water equations using FEniCSx with advanced numerical methods, adjoint-based sensitivity analysis, and 4D-Var data assimilation.

> [!NOTE]
> Requires FEniCSx version 0.9.0

## Features

- **Multiple Discretizations**: Continuous Galerkin (CG), Discontinuous Galerkin (DG), SUPG, and mixed element methods
- **Time Integration**: Implicit Euler, Crank-Nicolson, and BDF2 schemes
- **4D-Var Data Assimilation**: Standard 4D-Var, DC-4DVar, and DC-WME cost functions
- **Adjoint Methods**: Implicit adjoint solver with checkpointing strategies for memory efficiency
- **Optimization**: L-BFGS, Gauss-Newton, and PETSc TAO integration
- **Physics**: Friction models (linear, quadratic, Manning's), Coriolis forcing, meteorological forcing, wetting/drying
- **Parallel Computing**: Full MPI support with efficient parallel I/O via ADIOS2

## Installation

First, clone or download this repository:

```bash
git clone https://github.com/your-username/SWE4DVar.git
cd SWE4DVar
```

### Conda (Recommended)

Set up a Python environment with FEniCSx via conda/mamba:

```bash
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista scipy matplotlib h5py adios4dolfinx
```

### Docker

Alternatively, use the DOLFINx Docker container:

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm --shm-size=512m ghcr.io/fenics/dolfinx/dolfinx:stable
```

### Install Package

After setting up your environment:

```bash
python3 -m pip install --no-build-isolation -e .
```

## Quick Start

### Tidal Flow Simulation

```python
from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver

# Define problem
prob = TidalProblem(
    nx=20, ny=5,
    dt=3600,
    nt=168,  # 7 days
    friction_law="mannings",
    solution_var="h"
)

# Create solver
solver = get_solver("SUPG")(prob, theta=1.0, p_degree=[1, 1])

# Run simulation
solver.time_loop(
    solver_parameters={"rtol": 1e-5, "atol": 1e-6, "max_it": 10},
    save_state=True
)
```

### Run Examples

```bash
# Tidal flow
python examples/tidal.py --solver SUPG --nx 40 --ny 10

# Dam break
python examples/dam_break.py

# Idealized inlet
python examples/idealized_inlet.py
```

### MPI Parallel Execution

```bash
mpirun -np 4 python examples/tidal.py --nx 100 --ny 25
```

## Project Structure

```
SWE4DVar/
├── src/swe4dvar/
│   ├── forward/              # Forward model solvers
│   │   ├── problems.py       # Problem definitions (Tidal, DamBreak, etc.)
│   │   ├── solvers.py        # CG, DG, SUPG solvers
│   │   ├── newton.py         # Custom Newton solver
│   │   └── variational_forms.py
│   ├── adjoint/              # Adjoint computation
│   │   ├── implicit_adjoint.py
│   │   ├── tangent_linear.py
│   │   └── checkpointing.py
│   ├── data_assimilation/    # 4D-Var framework
│   │   ├── cost_functions.py
│   │   ├── observation_operator.py
│   │   └── covariance.py
│   ├── optimization/         # Optimization algorithms
│   │   ├── lbfgs.py
│   │   ├── gauss_newton.py
│   │   └── petsc_tao_wrapper.py
│   ├── physics/              # Physical models
│   └── utils/                # Utilities
├── examples/                 # Example scripts
└── tests/                    # Test suite
```

## Data Assimilation (4D-Var)

SWE4DVar provides a complete 4D-Var framework for parameter estimation and state reconstruction:

```python
from swe4dvar.data_assimilation.cost_functions import FourDVarCost
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

# Optimize
optimizer = LBFGSOptimizer(cost, max_iter=50, gtol=1e-6)
m_analysis = optimizer.minimize(m_b)
```

See [examples/complete_4dvar_example.py](examples/complete_4dvar_example.py) for a complete demonstration.

## Solver Options

| Solver | Description                       | Use Case                          |
| ------ | --------------------------------- | --------------------------------- |
| `CG`   | Continuous Galerkin               | General purpose, smooth solutions |
| `SUPG` | Streamline-Upwind Petrov-Galerkin | Advection-dominated flows         |
| `DG`   | Discontinuous Galerkin            | Shock-capturing, discontinuities  |
| `DGCG` | DG velocity, CG elevation         | Mixed formulation                 |

## Time Stepping

| `theta` | Scheme         | Properties                        |
| ------- | -------------- | --------------------------------- |
| `1.0`   | Implicit Euler | 1st order, unconditionally stable |
| `0.5`   | Crank-Nicolson | 2nd order, energy-conserving      |
| `0.0`   | BDF2           | 2nd order, A-stable               |

## Testing

Run the test suite:

```bash
pytest tests/

# With MPI
mpirun -np 4 python -m pytest tests/

# Specific tests
pytest tests/test_variational_forms.py -v
```

## Dependencies

| Package                 | Purpose                  |
| ----------------------- | ------------------------ |
| FEniCSx (DOLFINx 0.9.0) | Finite element framework |
| PETSc / petsc4py        | Linear algebra, solvers  |
| mpi4py                  | MPI parallelization      |
| NumPy / SciPy           | Numerical computing      |
| h5py                    | HDF5 I/O (optional)      |
| adios4dolfinx           | Parallel I/O (optional)  |
| PyVista                 | Visualization (optional) |
| Matplotlib              | Plotting (optional)      |

<!-- ## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. -->

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

<!--
## Citation

If you use SWE4DVar or Data-Consistent Variational methods in your research, please cite:

```bibtex
@software{swe4dvar2024,
  title = {SWE4DVar: Shallow Water Equations with Modern Numerical Methods for Coastal Systems},
  author = {Pachev, Benjamin},
  year = {2024},
  url = {https://github.com/your-username/SWE4DVar}
}
``` -->

## Acknowledgments

This project builds upon [FEniCSx](https://fenicsproject.org/) and [PETSc](https://petsc.org/).

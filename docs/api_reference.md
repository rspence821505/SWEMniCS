# SWE4DVar API Reference

This document provides a comprehensive reference for the SWE4DVar Python API.

## Package Overview

```
swe4dvar/
├── forward/              # Forward model solvers
│   ├── problems.py       # Problem definitions
│   ├── solvers/          # Solver implementations
│   ├── newton.py         # Newton solver
│   └── variational_forms.py
├── adjoint/              # Adjoint computation
│   ├── implicit_adjoint.py
│   ├── tangent_linear.py
│   └── checkpointing.py
├── data_assimilation/    # 4D-Var framework
│   ├── cost_functions.py
│   ├── observation_operator.py
│   ├── covariance.py
│   └── qoi_maps.py
├── optimization/         # Optimization algorithms
│   ├── lbfgs.py
│   ├── gauss_newton.py
│   └── petsc_tao_wrapper.py
├── physics/              # Physical models
└── utils/                # Utilities
```

---

## swe4dvar.forward

The forward module contains problem definitions and solver implementations for the shallow water equations.

### swe4dvar.forward.problems

Problem classes define the physical setup including mesh, boundary conditions, initial conditions, and physics parameters.

#### FrictionLaw

```python
class FrictionLaw(str, Enum):
    """Enumeration of available friction models."""
    linear = "linear"      # Linear friction: tau = Cf * u
    mannings = "mannings"  # Manning's formula: tau = g*n^2*|u|*u/h^(1/3)
    nolibf2 = "nolibf2"    # ADCIRC NOLIBF=2 formulation
    quadratic = "quadratic"  # Quadratic friction: tau = Cf * |u| * u
    none = "none"          # No friction
```

#### BaseProblem

```python
@dataclass
class BaseProblem(ABC):
    """
    Abstract base class for shallow water equation problems.

    Attributes
    ----------
    nx, ny : int
        Number of mesh elements in x and y directions.
    dt : float
        Time step size (seconds).
    nt : int
        Number of time steps.
    h_b : float
        Reference bathymetry depth.
    h_init : callable or float
        Initial water height/elevation.
    vel_init : callable or float
        Initial velocity field.
    friction_law : str or FrictionLaw
        Friction model to use.
    solution_var : str
        Solution variable: "eta" (elevation) or "h" (total depth).
    spherical : bool
        Enable spherical coordinates.
    wd : bool
        Enable wetting/drying.
    wd_alpha : float
        Wetting/drying parameter.
    """
```

#### TidalProblem

```python
@dataclass
class TidalProblem(BaseProblem):
    """
    Tidal flow problem in a rectangular channel.

    Boundary conditions:
    - Left: Tidal elevation (sinusoidal)
    - Right: Land (no-flow)
    - Top/Bottom: Slip walls

    Parameters
    ----------
    T_M2 : float
        Tidal period (default: 12.42 hours for M2 tide).
    amplitude : float
        Tidal amplitude (meters).
    """
```

#### DamBreakProblem

```python
@dataclass
class DamBreakProblem(BaseProblem):
    """
    Classical dam break problem.

    Initial condition: Step function in water height.
    Boundary conditions: Reflective walls.
    """
```

#### IdealizedInlet

```python
@dataclass
class IdealizedInlet(BaseProblem):
    """
    Idealized coastal inlet with tidal forcing.

    Features variable bathymetry, Manning's friction,
    and realistic tidal boundary conditions.
    """
```

---

### swe4dvar.forward.solvers

Solver implementations using various numerical methods.

#### get_solver

```python
def get_solver(solver_type: str) -> type:
    """
    Factory function to get solver class by name.

    Parameters
    ----------
    solver_type : str
        One of: 'CG', 'SUPG', 'DG', 'DGCG', 'DGNC'

    Returns
    -------
    type
        Solver class

    Example
    -------
    >>> solver_cls = get_solver("SUPG")
    >>> solver = solver_cls(problem, theta=1.0, p_degree=[1, 1])
    """
```

#### BaseSolver

```python
class BaseSolver:
    """
    Base class for all shallow water equation solvers.

    Parameters
    ----------
    problem : BaseProblem
        Problem definition.
    theta : float
        Time stepping parameter:
        - 1.0: Implicit Euler (1st order)
        - 0.5: Crank-Nicolson (2nd order)
        - 0.0: BDF2 (2nd order)
    p_degree : list[int]
        Polynomial degrees [elevation, velocity].
    p_type : str
        Element type: 'CG' or 'DG'.
    swe_type : str
        'full' for nonlinear, 'linear' for linearized.
    verbose : bool
        Print solver information.

    Methods
    -------
    time_loop(solver_parameters, save_state=False, save_jacobian=False)
        Run time integration loop.
    step(solver_parameters)
        Perform single time step.
    get_state()
        Get current state as PETSc vector.
    set_state(state_vec)
        Set state from PETSc vector.
    """

    def time_loop(
        self,
        solver_parameters: dict,
        save_state: bool = False,
        save_jacobian: bool = False,
    ) -> None:
        """
        Run the forward model time integration.

        Parameters
        ----------
        solver_parameters : dict
            Newton solver parameters:
            - rtol: Relative tolerance (default: 1e-5)
            - atol: Absolute tolerance (default: 1e-6)
            - max_it: Maximum iterations (default: 10)
        save_state : bool
            Store states at each time step for adjoint.
        save_jacobian : bool
            Store Jacobians for adjoint (enables caching).
        """
```

#### CGImplicit

```python
class CGImplicit(BaseSolver):
    """
    Continuous Galerkin solver with implicit time stepping.

    Uses standard CG elements for both elevation and velocity.
    Suitable for smooth solutions without sharp gradients.
    """
```

#### SUPGImplicit

```python
class SUPGImplicit(BaseSolver):
    """
    Streamline-Upwind Petrov-Galerkin solver.

    Adds stabilization for advection-dominated problems.
    Recommended for most coastal applications.
    """
```

#### DGImplicit

```python
class DGImplicit(BaseSolver):
    """
    Discontinuous Galerkin solver with implicit time stepping.

    Uses upwind numerical fluxes for inter-element communication.
    Best for problems with discontinuities (dam breaks, shocks).
    """
```

#### DGCGImplicit

```python
class DGCGImplicit(BaseSolver):
    """
    Mixed DG-CG solver.

    Uses DG for velocity and CG for elevation.
    Combines advantages of both methods.
    """
```

---

## swe4dvar.adjoint

The adjoint module provides tools for computing sensitivities via the adjoint method.

### TangentLinearModel

```python
class TangentLinearModel:
    """
    Tangent Linear Model (TLM) for the shallow water equations.

    Computes directional derivatives of the forward model:
        dM/dm * delta_m

    Parameters
    ----------
    forward_solver : BaseSolver
        Forward model solver.

    Methods
    -------
    run(delta_m, m_background)
        Run TLM from perturbation delta_m.
    """
```

### TLMValidator

```python
class TLMValidator:
    """
    Validation utilities for the Tangent Linear Model.

    Performs Taylor remainder tests to verify TLM correctness.

    Methods
    -------
    taylor_test(m, delta_m, epsilons=None)
        Run Taylor test returning convergence rates.
    """
```

### ImplicitAdjointSolver

```python
class ImplicitAdjointSolver:
    """
    Adjoint solver for implicit BDF2 time stepping.

    Solves the adjoint equations backward in time:
        J_n^T * lambda_n = forcing + time derivative terms

    Reuses cached Jacobians from forward solve for ~50% speedup.

    Parameters
    ----------
    forward_solver : BaseSolver
        Forward model solver (must have saved_jacobians).
    checkpointer : StateCheckpointer, optional
        Checkpointing strategy for memory efficiency.

    Methods
    -------
    solve(forcing_sequence)
        Run adjoint integration backward in time.
    get_gradient()
        Extract gradient at initial time.
    """
```

### Checkpointing Strategies

```python
class StateCheckpointer:
    """Store all states (maximum memory, fastest)."""

class JacobianCheckpointer:
    """Store only Jacobians, recompute states as needed."""

class BinomialCheckpointer:
    """
    Optimal checkpointing using Griewank's binomial algorithm.

    Balances memory usage and recomputation cost.

    Parameters
    ----------
    max_steps : int
        Maximum number of time steps.
    max_checkpoints : int
        Maximum checkpoints to store.
    """
```

---

## swe4dvar.data_assimilation

The data assimilation module implements 4D-Var cost functions and related utilities.

### Cost Functions

#### CostFunction (Base Class)

```python
class CostFunction(ABC):
    """
    Abstract base class for 4D-Var cost functions.

    Parameters
    ----------
    forward_model : BaseSolver
        Forward model solver.
    observation_operator : ObservationOperator
        Maps state to observation space.
    background_cov : CovarianceMatrix
        Background error covariance B.
    observation_cov : CovarianceMatrix
        Observation error covariance R.
    comm : MPI.Comm, optional
        MPI communicator.

    Methods
    -------
    value(m) -> float
        Compute cost function J(m).
    gradient(m) -> PETSc.Vec
        Compute gradient via adjoint method.
    hessian_vector_product(m, v) -> PETSc.Vec
        Compute Hessian-vector product (Gauss-Newton).
    """
```

#### FourDVarCost

```python
class FourDVarCost(CostFunction):
    """
    Standard 4D-Var cost function.

    J(m) = 1/2 * ||m - m_b||^2_B^{-1}
         + 1/2 * sum_k ||H_k(u_k) - y_k||^2_R^{-1}

    Parameters
    ----------
    forward_model : BaseSolver
        Forward model solver.
    observation_operator : ObservationOperator
        Observation operator H.
    background_cov : CovarianceMatrix
        Background covariance B.
    observation_cov : CovarianceMatrix
        Observation covariance R.
    m_background : PETSc.Vec
        Background state m_b.
    observations : list[np.ndarray]
        Observations y_k at each time.
    obs_times : list[int]
        Time indices for observations.
    """
```

#### DCFourDVarCost

```python
class DCFourDVarCost(CostFunction):
    """
    Data-Consistent 4D-Var cost function.

    J_DC(m) = J_4DVar(m)
            - 1/2 * ||Q(m) - Q(m_b)||^2_L^{-1}

    The predictability term corrects for information already
    encoded in the prior by subtracting predictable components.

    Additional Parameters
    ---------------------
    predictability_cov : CovarianceMatrix
        Predictability covariance L.
    qoi_map : QoIMap
        Quantity of Interest map Q.
    """
```

#### DCWMEFourDVarCost

```python
class DCWMEFourDVarCost(CostFunction):
    """
    Data-Consistent 4D-Var with Weighted Mean Error.

    Uses the weighted mean error QoI:
        Q_wme,K(m) = (1/sqrt(|I|)) * sum_{j in I} R_j^{-1/2}(H_j(u_j) - y_j),
        where I is the observation index set and K := max(I).

    This formulation provides better conditioning and
    natural handling of observation correlations.

    The cost function is:
    J_WME(m) = 1/2 * ||m - m_b||^2_B^{-1}
             + 1/2 * ||Q_wme(m)||^2
             - 1/2 * ||Q_wme(m) - Q_wme(m_b)||^2_L^{-1}
    """
```

#### create_cost_function

```python
def create_cost_function(
    method: str,
    forward_model,
    observation_operator,
    background_cov,
    observation_cov,
    m_background,
    observations,
    obs_times,
    **kwargs
) -> CostFunction:
    """
    Factory function to create cost function by method name.

    Parameters
    ----------
    method : str
        One of: '4dvar', 'dc', 'dc_wme' (also accepts 'wme')
    **kwargs : dict
        Additional arguments (e.g., predictability_cov for DC methods).

    Returns
    -------
    CostFunction
        Configured cost function instance.
    """
```

### Covariance Matrices

```python
class CovarianceMatrix(ABC):
    """Abstract base class for covariance matrices."""

    def apply(self, x: PETSc.Vec) -> PETSc.Vec:
        """Apply covariance: C * x"""

    def apply_inverse(self, x: PETSc.Vec) -> PETSc.Vec:
        """Apply inverse: C^{-1} * x"""

    def apply_sqrt(self, x: PETSc.Vec) -> PETSc.Vec:
        """Apply square root: C^{1/2} * x"""

    def apply_sqrt_inverse(self, x: PETSc.Vec) -> PETSc.Vec:
        """Apply inverse square root: C^{-1/2} * x"""


class DiagonalCovariance(CovarianceMatrix):
    """
    Diagonal covariance matrix.

    Parameters
    ----------
    variances : np.ndarray
        Diagonal entries (variances).
    """


class DenseCovariance(CovarianceMatrix):
    """
    Dense covariance matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Full covariance matrix.
    """


class ImplicitCovariance(CovarianceMatrix):
    """
    Implicitly-defined covariance (e.g., correlation length).

    Never forms full matrix; uses matrix-free operations.
    """


class EnsembleCovariance(CovarianceMatrix):
    """
    Covariance estimated from ensemble.

    Parameters
    ----------
    ensemble : list[PETSc.Vec]
        Ensemble members.
    localization_radius : float, optional
        Localization distance.
    """
```

### QoI Maps

```python
class QoIMap(ABC):
    """Abstract base class for Quantity of Interest maps."""

    def evaluate(self, m: PETSc.Vec, time_index: int) -> PETSc.Vec:
        """Evaluate Q_k(m) at a time index k."""


class StandardQoI(QoIMap):
    """
    Standard QoI: observed model state at time k.

    Q_k(m) = H_k(M_{k:0}(m))
    """


class WeightedMeanErrorQoI(QoIMap):
    """
    Weighted Mean Error QoI.

    Q_wme,k(m) = (1/sqrt(|I_k|)) * sum_{j in I_k} R_j^{-1/2}(H_j(u_j) - y_j),
    where I_k := { j ∈ I : j ≤ k } and I is the observation index set.
    """


class LinearizedQoI(ABC):
    """Base class for linearized QoI maps."""

    def apply(self, delta_m: PETSc.Vec) -> PETSc.Vec:
        """Apply Jacobian: DQ * delta_m"""

    def apply_adjoint(self, delta_q: PETSc.Vec) -> PETSc.Vec:
        """Apply adjoint: (DQ)^T * delta_q"""


class QoICovarianceEstimator:
    """
    Estimates predictability covariance L_k ≈ DQ_k B DQ_k^T.

    Uses Monte Carlo directions sampled from N(0, B) and pushed through
    the linearized QoI.

    Methods
    -------
    estimate(m_bar, time_index) -> CovarianceMatrix
        Estimate L_k at linearization point m_bar.
    """
```

### Observation Operators

```python
class ObservationOperator(ABC):
    """Abstract base class for observation operators."""

    def evaluate(self, state: PETSc.Vec) -> np.ndarray:
        """Apply H: state space -> observation space."""

    def apply_adjoint(self, obs_vec: np.ndarray) -> PETSc.Vec:
        """Apply H^T: observation space -> state space."""


class PointObservationOperator(ObservationOperator):
    """
    Point-wise observations at specified locations.

    Parameters
    ----------
    coordinates : np.ndarray
        Observation locations (N x 2 or N x 3).
    variables : list[str]
        Variables to observe: 'eta', 'u', 'v'.
    function_space : FunctionSpace
        DOLFINx function space.
    """


class IntegralObservationOperator(ObservationOperator):
    """
    Integral observations over subdomains.

    Computes weighted integrals of state variables.
    """


class CompositeObservationOperator(ObservationOperator):
    """
    Combines multiple observation operators.

    Useful when observing different variables at different locations.
    """
```

---

## swe4dvar.optimization

Optimization algorithms for minimizing cost functions.

**Recommendation:** Use `TAOOptimizerFactory` or `PETScTAOWrapper` for production use.
TAO provides battle-tested optimization algorithms with robust line search and convergence monitoring.

### TAOOptimizerFactory (Recommended)

```python
class TAOOptimizerFactory:
    """
    Factory for creating PETSc TAO optimizers with common configurations.

    Provides convenience methods for typical optimization scenarios.
    Recommended for production 4D-Var applications.

    Methods
    -------
    create_lbfgs(cost_function, memory_size=10, options=None) -> PETScTAOWrapper
        Create TAO L-BFGS optimizer (unconstrained).

    create_bounded_lbfgs(cost_function, lower_bounds=None, upper_bounds=None,
                         memory_size=10, options=None) -> PETScTAOWrapper
        Create TAO bounded L-BFGS optimizer (box constraints).

    create_trust_region(cost_function, options=None) -> PETScTAOWrapper
        Create TAO Newton trust region optimizer.

    create_conjugate_gradient(cost_function, cg_type='pr', options=None) -> PETScTAOWrapper
        Create TAO nonlinear conjugate gradient optimizer.

    Example
    -------
    >>> from swe4dvar.optimization import TAOOptimizerFactory
    >>> optimizer = TAOOptimizerFactory.create_lbfgs(
    ...     cost_function,
    ...     memory_size=10,
    ...     options={'verbose': True, 'max_iterations': 50}
    ... )
    >>> m_optimal = optimizer.solve(m_initial)
    """
```

### PETScTAOWrapper

```python
class PETScTAOWrapper:
    """
    Wrapper for PETSc TAO (Toolkit for Advanced Optimization).

    Bridges SWE4DVar cost function interface with TAO's callback system.
    TAO handles the optimization loop and convergence monitoring internally.

    Supported TAO types:
        - 'lmvm': Limited-memory variable metric (L-BFGS)
        - 'blmvm': Bounded L-BFGS with box constraints
        - 'nls': Newton line search
        - 'ntr': Newton trust region
        - 'cg': Nonlinear conjugate gradient
        - 'nm': Nelder-Mead (derivative-free)

    Key advantages over custom implementation:
        - Production-grade L-BFGS-B with full active set handling
        - Sophisticated line search and trust region methods
        - Automatic convergence monitoring and diagnostics
        - Battle-tested robustness on ill-conditioned problems

    Parameters
    ----------
    cost_function : CostFunction
        Cost function to minimize. If it has a `value_gradient()` method,
        TAO will use it for efficient combined objective/gradient evaluation.
    tao_type : str
        TAO algorithm type (default: 'lmvm').
    lower_bounds : PETSc.Vec, optional
        Lower bounds for box constraints (None = unbounded).
    upper_bounds : PETSc.Vec, optional
        Upper bounds for box constraints (None = unbounded).
    options : dict, optional
        Optimizer options:
        - max_iterations: Maximum iterations (default: 100)
        - gradient_tolerance: Gradient norm tolerance (default: 1e-6)
        - cost_tolerance: Function value tolerance (default: 1e-8)
        - verbose: Print iteration info (default: False)
        - tao_monitor: Use TAO's built-in monitor (default: False)
        - Additional TAO options with 'tao_' prefix

    Methods
    -------
    solve(x0) -> PETSc.Vec
        Minimize cost function starting from initial guess x0.
    get_convergence_info() -> dict
        Get convergence information after solve.

    Example
    -------
    >>> from swe4dvar.optimization import PETScTAOWrapper
    >>> optimizer = PETScTAOWrapper(
    ...     cost_function,
    ...     tao_type='lmvm',
    ...     options={'max_iterations': 100, 'verbose': True}
    ... )
    >>> m_optimal = optimizer.solve(m_initial)
    >>> info = optimizer.get_convergence_info()
    >>> print(f"Converged: {info['converged']}, Iterations: {info['iterations']}")
    """
```

### LBFGSOptimizer (Legacy)

```python
class LBFGSOptimizer:
    """
    Limited-memory BFGS optimizer.

    .. deprecated::
        For production use, prefer `TAOOptimizerFactory.create_lbfgs()`
        which provides a battle-tested L-BFGS implementation with better
        convergence properties and line search algorithms.

    Efficient quasi-Newton method using two-loop recursion.

    Parameters
    ----------
    cost_function : CostFunction
        Cost function to minimize.
    memory_size : int
        Number of correction pairs (default: 10).
    options : dict, optional
        - max_iterations: int (default: 100)
        - gradient_tolerance: float (default: 1e-6)
        - cost_tolerance: float (default: 1e-8)
        - line_search_c1: float (default: 1e-4)
        - line_search_c2: float (default: 0.9)
        - line_search_max_iter: int (default: 20)
        - verbose: bool (default: False)

    Methods
    -------
    solve(x0) -> PETSc.Vec
        Minimize cost function starting from x0.
    """
```

### GaussNewtonOptimizer

```python
class GaussNewtonOptimizer:
    """
    Gauss-Newton optimizer using Hessian-vector products.

    Uses conjugate gradient for inner linear solves.
    Better convergence near optimum than L-BFGS.

    Parameters
    ----------
    cost_function : CostFunction
        Must support hessian_vector_product.
    options : dict, optional
        - max_iterations: int
        - cg_max_iterations: int
        - cg_tolerance: float
    """
```

---

## swe4dvar.utils

Utility modules for parallel operations, profiling, and I/O.

### Output Paths

```python
from swe4dvar.utils import (
    OUTPUT_ROOT,      # Path to outputs/
    LOGS_DIR,         # Path to outputs/logs/
    FIGURES_DIR,      # Path to outputs/figures/
    CHECKPOINTS_DIR,  # Path to outputs/checkpoints/
    DATA_DIR,         # Path to outputs/data/
    ensure_output_dirs,  # Create directories if needed
    get_figure_path,     # Get path for a figure file
    get_data_path,       # Get path for a data file
    get_log_path,        # Get path for a log file
)
```

### Parallel Operations

```python
class ParallelContext:
    """
    MPI parallel execution context.

    Provides rank-aware operations and collective utilities.

    Attributes
    ----------
    comm : MPI.Comm
        MPI communicator.
    rank : int
        Process rank.
    size : int
        Total number of processes.
    is_root : bool
        True if rank == 0.
    """


class DistributedVectorOps:
    """
    Operations on distributed PETSc vectors.

    Methods
    -------
    global_norm(vec) -> float
        Compute global L2 norm.
    global_inner(v1, v2) -> float
        Compute global inner product.
    scatter_to_all(vec) -> np.ndarray
        Gather vector to all processes.
    """
```

### Profiling

```python
class HierarchicalTimer:
    """
    Hierarchical timing with nested regions.

    Example
    -------
    >>> timer = HierarchicalTimer()
    >>> with timer.region("forward"):
    ...     with timer.region("assembly"):
    ...         # Code here
    >>> timer.report()
    """


class MemoryProfiler:
    """Track memory usage throughout computation."""


class ScalabilityAnalyzer:
    """Analyze strong/weak scaling behavior."""


class ComprehensiveProfiler:
    """Combined timing, memory, and scalability profiling."""


def profile(func):
    """Decorator to profile function execution."""
```

### PETSc Logging

```python
class PETScLogger:
    """
    Interface to PETSc's built-in logging.

    Tracks FLOPS, memory, and communication.
    """


class PerformanceMonitor:
    """
    Real-time performance monitoring.

    Logs iteration times, convergence, and resource usage.
    """


def setup_default_logging(level="INFO"):
    """Configure default logging for SWE4DVar."""


@contextmanager
def petsc_log_context(event_name: str):
    """Context manager for PETSc event logging."""
```

### Non-blocking Communication

```python
class NonBlockingScatter:
    """Asynchronous scatter operations."""


class AsyncVectorOps:
    """Asynchronous vector operations with overlap."""


class OverlapComputeComm:
    """
    Overlap computation with communication.

    Enables hiding communication latency behind computation.
    """


class AsyncObservationOperator:
    """Observation operator with non-blocking evaluation."""


class BatchedCommunication:
    """Batch multiple small messages for efficiency."""
```

### Solver Parameters

```python
def get_default_solver_params() -> dict:
    """
    Get default Newton solver parameters.

    Returns
    -------
    dict
        {'rtol': 1e-5, 'atol': 1e-6, 'max_it': 10}
    """


def get_solver_params_preset(name: str) -> dict:
    """
    Get solver parameters preset.

    Parameters
    ----------
    name : str
        Preset name: 'fast', 'accurate', 'robust'
    """
```

---

## Common Usage Patterns

### Pattern 1: Basic Forward Simulation

```python
from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver

problem = TidalProblem(nx=40, ny=10, dt=3600, nt=168)
solver = get_solver("SUPG")(problem, theta=1.0, p_degree=[1, 1])
solver.time_loop({"rtol": 1e-5, "atol": 1e-6, "max_it": 10})
```

### Pattern 2: 4D-Var with TAO Optimizer (Recommended)

```python
from swe4dvar.optimization import TAOOptimizerFactory

# Run forward with Jacobian caching
solver.time_loop(solver_params, save_state=True, save_jacobian=True)

# Setup cost function
cost = FourDVarCost(
    forward_model=solver,
    observation_operator=obs_op,
    background_cov=B,
    observation_cov=R,
    m_background=m_b,
    observations=observations,
    obs_times=obs_times,
)

# Optimize with TAO L-BFGS (production-grade)
optimizer = TAOOptimizerFactory.create_lbfgs(
    cost,
    memory_size=10,
    options={"max_iterations": 50, "verbose": True}
)
m_optimal = optimizer.solve(m_b)
```

### Pattern 3: DC-WME-4DVar

```python
from swe4dvar.data_assimilation import DCWMEFourDVarCost, QoICovarianceEstimator

# Estimate predictability covariance from ensemble
estimator = QoICovarianceEstimator()
L = estimator.estimate(prior_ensemble, qoi_map)

# Create DC-WME cost function
cost = DCWMEFourDVarCost(
    forward_model=solver,
    observation_operator=obs_op,
    background_cov=B,
    observation_cov=R,
    predictability_cov=L,
    m_background=m_b,
    observations=observations,
    obs_times=obs_times,
)
```

### Pattern 4: MPI Parallel Execution

```python
from mpi4py import MPI
from swe4dvar.utils import ParallelContext

ctx = ParallelContext(MPI.COMM_WORLD)

# Problem automatically partitions mesh across processes
problem = TidalProblem(nx=200, ny=50, dt=1800, nt=336)
solver = get_solver("SUPG")(problem, theta=1.0, p_degree=[1, 1])

# All operations are automatically parallel
solver.time_loop(solver_params, save_state=True)

if ctx.is_root:
    print("Simulation complete")
```

---

## Version Information

```python
import swe4dvar
print(swe4dvar.__version__)  # e.g., "1.0.0"
```

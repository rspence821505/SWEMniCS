"""
Utility functions for serial data assimilation experiments.

Provides common functionality for twin experiments comparing
4D-Var and DC-WME-4DVar methods.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from mpi4py import MPI
from petsc4py import PETSc


@dataclass
class DAExperimentConfig:
    """Configuration for a data assimilation experiment."""

    # Problem configuration
    nx: int = 10
    ny: int = 5
    dt: float = 3600.0
    final_time: float = 24 * 3600.0  # 1 day default
    solver_type: str = "CG"

    # Observation configuration
    obs_fraction: float = 0.5  # Fraction of spatial points to observe
    obs_frequency: int = 1  # Observe every N timesteps
    obs_noise_level: float = 0.01  # 1% noise

    # Background error configuration
    background_error_std: float = 0.1  # 10% of signal magnitude

    # Optimization configuration
    max_iterations: int = 50
    gradient_tolerance: float = 1e-6
    cost_tolerance: float = 1e-8
    lbfgs_memory: int = 10

    # Output configuration
    output_dir: str = "outputs/data"
    figures_dir: str = "outputs/figures"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class DAExperimentResults:
    """Results from a data assimilation experiment."""

    # Method information
    method: str = ""
    test_case: str = ""

    # Cost function history
    cost_history: List[float] = field(default_factory=list)
    gradient_norm_history: List[float] = field(default_factory=list)

    # Error metrics
    background_error: float = 0.0
    analysis_error: float = 0.0
    error_reduction: float = 0.0

    # Innovation statistics
    innovation_mean: float = 0.0
    innovation_std: float = 0.0

    # Runtime information
    num_iterations: int = 0
    converged: bool = False
    wall_time: float = 0.0

    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def save(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "DAExperimentResults":
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class ForwardModelWrapper:
    """
    Wrapper to adapt SWE4DVar solvers to the forward model interface
    expected by the cost functions.

    The cost functions expect:
    - solve(m, store_jacobians=True) -> (trajectory, jacobians)
    - dt: time step size
    """

    def __init__(self, solver, problem, solver_params: dict):
        """
        Initialize forward model wrapper.

        Parameters
        ----------
        solver : CGImplicit or similar
            The SWE4DVar solver instance.
        problem : Problem
            The problem definition.
        solver_params : dict
            Parameters for the Newton solver.
        """
        self.solver = solver
        self.problem = problem
        self.solver_params = solver_params
        self.dt = problem.dt
        self.nt = problem.nt

        # MPI communicator
        self.comm = MPI.COMM_WORLD

    def solve(
        self,
        m: PETSc.Vec,
        store_jacobians: bool = True
    ) -> Tuple[List[PETSc.Vec], Optional[List[PETSc.Mat]]]:
        """
        Run forward model from initial condition m.

        Parameters
        ----------
        m : PETSc.Vec
            Initial condition vector.
        store_jacobians : bool
            Whether to store Jacobians for adjoint computation.

        Returns
        -------
        trajectory : List[PETSc.Vec]
            State vectors at each time step [u_0, u_1, ..., u_N].
        jacobians : List[PETSc.Mat] or None
            Jacobian matrices at each time step if store_jacobians=True.
        """
        # Reset solver state
        self.solver.storage.clear()

        # Set initial condition
        # Use copy to avoid issues with read-only vectors (e.g., from TAO)
        m_local = m.copy()
        m_array = m_local.getArray()

        # Handle MPI: m_array may have only owned DOFs while u_n.x.array includes ghosts
        u_owned_size = self.solver.V.dofmap.index_map.size_local
        if len(m_array) == len(self.solver.u_n.x.array):
            # Arrays match (serial or vector includes ghosts)
            self.solver.u_n.x.array[:] = m_array
            self.solver.u_n_old.x.array[:] = m_array
            self.solver.u.x.array[:] = m_array
        elif len(m_array) == u_owned_size:
            # m_array has only owned DOFs - copy to owned portion only
            self.solver.u_n.x.array[:u_owned_size] = m_array
            self.solver.u_n_old.x.array[:u_owned_size] = m_array
            self.solver.u.x.array[:u_owned_size] = m_array
            # Update ghosts
            self.solver.u_n.x.scatter_forward()
            self.solver.u_n_old.x.scatter_forward()
            self.solver.u.x.scatter_forward()
        else:
            raise ValueError(
                f"Initial condition size {len(m_array)} does not match "
                f"solver DOFs (owned={u_owned_size}, total={len(self.solver.u_n.x.array)})"
            )
        m_local.destroy()

        # Reset problem time
        self.problem.t = 0.0

        # Run time loop with Jacobian storage
        self.solver.time_loop(
            solver_parameters=self.solver_params,
            stations=np.array([[0.0, 0.0, 0.0]]),  # Dummy station
            plot_every=9999,  # No plotting
            save_state=True,
            store_jacobians=store_jacobians,
            enable_video=False,
        )

        # Extract trajectory as PETSc vectors
        # Note: saved_states include ghost values, need proper distributed vectors
        from dolfinx import la
        u_owned_size = self.solver.V.dofmap.index_map.size_local
        trajectory = []
        for state_array in self.solver.storage.saved_states:
            # Create properly distributed PETSc vector
            vec = la.create_petsc_vector(
                self.solver.V.dofmap.index_map,
                self.solver.V.dofmap.index_map_bs,
            )
            # Only copy owned DOFs (not ghosts)
            vec.setArray(state_array[:u_owned_size])
            vec.assemble()
            trajectory.append(vec)

        # Extract Jacobians if stored
        jacobians = None
        if store_jacobians and len(self.solver.storage.saved_jacobians) > 0:
            jacobians = self.solver.storage.saved_jacobians.copy()

        return trajectory, jacobians

    def get_state_size(self) -> int:
        """Return size of state vector."""
        return self.solver.u.x.array.shape[0]

    def get_mass_matrix(self) -> PETSc.Mat:
        """
        Assemble and return the FEM mass matrix.

        The mass matrix is essential for proper BDF2 adjoint time-coupling.
        Without it, the adjoint solver uses an identity matrix which causes
        numerical issues with large time steps.

        Returns
        -------
        PETSc.Mat
            Assembled mass matrix M where M_ij = ∫ φ_i φ_j dx
        """
        from dolfinx import fem
        from ufl import inner, dx, TrialFunction, TestFunction

        if not hasattr(self, '_mass_matrix'):
            # Get function space
            V = self.solver.V

            # Create trial and test functions
            u = TrialFunction(V)
            v = TestFunction(V)

            # Mass form: M = ∫ u · v dx
            a = inner(u, v) * dx

            # Assemble mass matrix
            M = fem.petsc.assemble_matrix(fem.form(a))
            M.assemble()

            self._mass_matrix = M

        return self._mass_matrix


def generate_observation_points(
    mesh,
    fraction: float = 0.5,
    seed: int = 42
) -> np.ndarray:
    """
    Generate random observation points from mesh nodes.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        Computational mesh.
    fraction : float
        Fraction of nodes to use as observation points.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    obs_points : np.ndarray
        Array of observation point coordinates, shape (n_obs, 3).
    """
    rng = np.random.default_rng(seed)

    # Get mesh coordinates
    coords = mesh.geometry.x
    n_points = coords.shape[0]

    # Select random subset
    n_obs = int(n_points * fraction)
    indices = rng.choice(n_points, size=n_obs, replace=False)

    # Ensure 3D coordinates for DOLFINx
    obs_points = np.zeros((n_obs, 3))
    obs_points[:, :coords.shape[1]] = coords[indices, :]

    return obs_points


def generate_observations(
    trajectory: List[PETSc.Vec],
    obs_operator,
    obs_times: List[int],
    noise_level: float = 0.01,
    seed: int = 42
) -> Tuple[List[PETSc.Vec], np.ndarray]:
    """
    Generate synthetic observations from truth trajectory.

    Parameters
    ----------
    trajectory : List[PETSc.Vec]
        True state trajectory.
    obs_operator : ObservationOperator
        Observation operator H.
    obs_times : List[int]
        Time indices at which to observe.
    noise_level : float
        Standard deviation as fraction of signal magnitude.
    seed : int
        Random seed.

    Returns
    -------
    observations : List[PETSc.Vec]
        Observation vectors with added noise.
    obs_noise_std : np.ndarray
        Standard deviation of noise at each observation.
    """
    rng = np.random.default_rng(seed)
    observations = []
    noise_stds = []

    for k in obs_times:
        if k >= len(trajectory):
            raise IndexError(f"Observation time {k} exceeds trajectory length {len(trajectory)}")

        # Apply observation operator to get true observation
        H_u = obs_operator.forward(trajectory[k])
        H_u_array = H_u.getArray()

        # Compute noise standard deviation based on signal magnitude
        signal_magnitude = np.abs(H_u_array).mean() + 1e-10
        noise_std = noise_level * signal_magnitude
        noise_stds.append(noise_std)

        # Add Gaussian noise
        noise = rng.normal(0, noise_std, size=H_u_array.shape)
        noisy_obs = H_u_array + noise

        # Create observation vector
        obs_vec = PETSc.Vec().createSeq(len(noisy_obs), comm=PETSc.COMM_SELF)
        obs_vec.setArray(noisy_obs)
        obs_vec.assemble()

        observations.append(obs_vec)

    return observations, np.array(noise_stds)


def generate_background_state(
    truth: PETSc.Vec,
    error_std: float = 0.1,
    seed: int = 123
) -> PETSc.Vec:
    """
    Generate background state with error relative to truth.

    Parameters
    ----------
    truth : PETSc.Vec
        True initial condition.
    error_std : float
        Standard deviation of error as fraction of truth magnitude.
    seed : int
        Random seed.

    Returns
    -------
    background : PETSc.Vec
        Background state.
    """
    rng = np.random.default_rng(seed)

    truth_array = truth.getArray()

    # Compute error magnitude based on truth
    truth_magnitude = np.abs(truth_array).mean() + 1e-10
    error_magnitude = error_std * truth_magnitude

    # Generate random perturbation
    perturbation = rng.normal(0, error_magnitude, size=truth_array.shape)

    # Create background state
    background = truth.duplicate()
    background.setArray(truth_array + perturbation)
    background.assemble()

    return background


def compute_rms_error(
    state1: PETSc.Vec,
    state2: PETSc.Vec,
    comm: MPI.Comm = None
) -> float:
    """
    Compute root-mean-square error between two states.

    Parameters
    ----------
    state1 : PETSc.Vec
        First state vector.
    state2 : PETSc.Vec
        Second state vector.
    comm : MPI.Comm
        MPI communicator.

    Returns
    -------
    rms : float
        RMS error.
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    diff = state1.duplicate()
    diff.waxpy(-1.0, state2, state1)

    # Compute norm
    local_norm_sq = diff.dot(diff)
    local_size = diff.getLocalSize()

    # MPI reduction
    global_norm_sq = comm.allreduce(local_norm_sq, op=MPI.SUM)
    global_size = comm.allreduce(local_size, op=MPI.SUM)

    rms = np.sqrt(global_norm_sq / global_size)

    diff.destroy()
    return rms


def compute_innovation_statistics(
    trajectory: List[PETSc.Vec],
    obs_operator,
    observations: List[PETSc.Vec],
    obs_times: List[int]
) -> Tuple[float, float]:
    """
    Compute innovation (obs - model) statistics.

    Parameters
    ----------
    trajectory : List[PETSc.Vec]
        Model trajectory.
    obs_operator : ObservationOperator
        Observation operator.
    observations : List[PETSc.Vec]
        Observation vectors.
    obs_times : List[int]
        Observation time indices.

    Returns
    -------
    mean : float
        Mean innovation.
    std : float
        Standard deviation of innovation.
    """
    all_innovations = []

    for i, k in enumerate(obs_times):
        H_u = obs_operator.forward(trajectory[k])
        H_u_array = H_u.getArray()
        y_array = observations[i].getArray()

        innovation = y_array - H_u_array
        all_innovations.extend(innovation.tolist())

    all_innovations = np.array(all_innovations)
    return float(np.mean(all_innovations)), float(np.std(all_innovations))


def save_experiment_results(
    results: DAExperimentResults,
    output_dir: str = "outputs/data"
) -> str:
    """
    Save experiment results to JSON file.

    Parameters
    ----------
    results : DAExperimentResults
        Experiment results.
    output_dir : str
        Output directory.

    Returns
    -------
    filepath : str
        Path to saved file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"{results.test_case}_{results.method}_results.json"
    filepath = output_path / filename

    results.save(str(filepath))
    return str(filepath)


def load_all_results(output_dir: str = "outputs/data") -> Dict[str, DAExperimentResults]:
    """
    Load all experiment results from output directory.

    Parameters
    ----------
    output_dir : str
        Output directory.

    Returns
    -------
    results : Dict[str, DAExperimentResults]
        Dictionary mapping experiment name to results.
    """
    output_path = Path(output_dir)
    results = {}

    for filepath in output_path.glob("*_results.json"):
        name = filepath.stem.replace("_results", "")
        results[name] = DAExperimentResults.load(str(filepath))

    return results


def create_physical_bounds(
    m_template: PETSc.Vec,
    n_vars: int = 3,
    h_min: float = 0.01,
    momentum_bound: float = 1e10,
) -> Tuple[PETSc.Vec, PETSc.Vec]:
    """
    Create physical bounds for the control variable.

    For shallow water equations with state [h, hu, hv], enforces:
    - h >= h_min (water depth must be positive)
    - |hu|, |hv| <= momentum_bound (momentum bounded)

    Parameters
    ----------
    m_template : PETSc.Vec
        Template vector with correct size and distribution.
    n_vars : int
        Number of state variables per node (default: 3 for h, hu, hv).
    h_min : float
        Minimum water depth (default: 0.01).
    momentum_bound : float
        Maximum momentum magnitude (default: 1e10, effectively unbounded).

    Returns
    -------
    lower_bounds : PETSc.Vec
        Lower bound vector.
    upper_bounds : PETSc.Vec
        Upper bound vector.
    """
    lower = m_template.duplicate()
    upper = m_template.duplicate()

    lower_array = lower.getArray()
    upper_array = upper.getArray()

    n_dofs = len(lower_array)
    n_nodes = n_dofs // n_vars

    for i in range(n_nodes):
        # Water depth h: must be >= h_min
        lower_array[i * n_vars] = h_min
        upper_array[i * n_vars] = 1e10  # No practical upper bound

        # Momentum hu, hv: bounded
        for j in range(1, n_vars):
            lower_array[i * n_vars + j] = -momentum_bound
            upper_array[i * n_vars + j] = momentum_bound

    lower.setArray(lower_array)
    upper.setArray(upper_array)

    return lower, upper


def create_tao_optimizer(
    cost_function,
    m_template: PETSc.Vec,
    options: Dict[str, Any],
    use_bounds: bool = True,
    h_min: float = 0.01,
):
    """
    Create a TAO optimizer with optional physical bounds.

    Parameters
    ----------
    cost_function : CostFunctionBase
        4D-Var cost function.
    m_template : PETSc.Vec
        Template vector for bounds creation.
    options : Dict[str, Any]
        Optimizer options (max_iterations, gradient_tolerance, etc.).
    use_bounds : bool
        Whether to use bounded optimization (default: True).
    h_min : float
        Minimum water depth for bounds (default: 0.01).

    Returns
    -------
    optimizer : PETScTAOWrapper
        Configured TAO optimizer.
    """
    from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper

    if use_bounds:
        lower, upper = create_physical_bounds(m_template, h_min=h_min)
        tao_type = "blmvm"  # Bounded L-BFGS

        # Project initial point onto feasible region if needed
        m_arr = m_template.getArray()
        lo_arr = lower.getArray()
        up_arr = upper.getArray()

        # Check for bound violations and warn
        below = m_arr < lo_arr
        above = m_arr > up_arr
        if below.any() or above.any():
            import warnings
            n_violations = below.sum() + above.sum()
            warnings.warn(
                f"Initial point violates {n_violations} bounds. "
                "TAO will project to feasible region.",
                RuntimeWarning,
                stacklevel=2
            )
    else:
        lower, upper = None, None
        tao_type = "lmvm"  # Standard L-BFGS

    optimizer = PETScTAOWrapper(
        cost_function,
        tao_type=tao_type,
        lower_bounds=lower,
        upper_bounds=upper,
        options=options,
    )

    return optimizer

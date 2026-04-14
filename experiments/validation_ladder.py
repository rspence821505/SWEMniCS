#!/usr/bin/env python3
"""
Manning's-n Experimental Validation Ladder
==========================================

Deterministic, reproducible experiments to determine whether
augmented 4D-Var can recover Manning's-n friction coefficients.

Experiments (run in order, stop on failure):

  1. GRADIENT CHECK — finite-difference vs. adjoint gradient for θ block
  2. PARAMETER-ONLY — fix state near truth, optimize only θ
  3. WEAKLY COUPLED — joint [u0; θ] with very strong state prior
  4. FULL JOINT    — balanced priors, realistic setup

Success Metrics:
  - Parameter RMSE reduction ≥ 30% from background
  - θ moves meaningfully from initialization
  - Optimizer completes ≥ 1 iteration (no LS_FAILURE)

Usage:
  python experiments/validation_ladder.py --experiment 1   # gradient check
  python experiments/validation_ladder.py --experiment 2   # parameter-only
  python experiments/validation_ladder.py --experiment 3   # weakly coupled
  python experiments/validation_ladder.py --experiment 4   # full joint
  python experiments/validation_ladder.py --experiment all # run ladder
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LadderConfig:
    """Shared configuration for the validation ladder."""
    # Solver
    adios_file: str = "data/shinnecock_inlet"
    dt: float = 600.0
    nt_ramp: int = 6          # 1-hour ramp (minimal)
    nt_da: int = 12           # 2-hour DA window
    truth_vmax: float = 0.0   # No wind — isolate friction
    wind_ramp_s: float = 43200.0  # Wind ramp timescale (seconds)

    # Manning parameterization
    basis_type: str = "gaussian"
    basis_shape: tuple = (2, 1)  # 2 basis functions (Gaussian only)
    basis_width_fraction: float = 0.35
    n_bounds: tuple = (0.01, 0.08)
    reference_n: float = 0.02
    truth_coefficients: tuple = (0.20, -0.15)  # Strong signal
    background_coefficients: tuple = (0.0, 0.0)

    # KL-specific
    kl_correlation_length: float = None  # None = auto (1/4 domain extent)
    kl_prior_std: float = 0.5
    kl_n_modes: int = None  # None = auto (99% variance)

    # Observations
    obs_frequency: int = 2    # every 20 min
    obs_fraction: float = 0.10
    obs_noise_level: float = 0.005
    obs_seed: int = 42
    obs_velocity: bool = False  # If True, observe h + u + v (3x observations)
    obs_max_depth: float = None  # If set, only place obs where depth < this value (meters)

    # Optimizer
    max_iterations: int = 15
    gradient_tolerance: float = 1e-4  # relaxed for parameter problems
    theta_bounds: tuple = ((-0.5, 0.5), (-0.5, 0.5))

    @property
    def nt_total(self) -> int:
        return self.nt_ramp + self.nt_da

    @property
    def n_params(self) -> int:
        if self.basis_type == "kl":
            return self.kl_n_modes if self.kl_n_modes is not None else 0
        return self.basis_shape[0] * self.basis_shape[1]

    @property
    def obs_times(self) -> list:
        return list(range(self.nt_ramp, self.nt_total + 1, self.obs_frequency))


# ---------------------------------------------------------------------------
# Experiment base
# ---------------------------------------------------------------------------

class ExperimentResult:
    """Container for structured experiment output."""

    def __init__(self, name: str):
        self.name = name
        self.success = False
        self.diagnosis = "NOT RUN"
        self.metrics: dict[str, Any] = {}
        self.convergence: list[dict] = []
        self.wall_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "success": self.success,
            "diagnosis": self.diagnosis,
            "metrics": self.metrics,
            "n_iterations": len(self.convergence),
            "convergence": self.convergence,
            "wall_time_s": self.wall_time_s,
        }

    def print_summary(self):
        status = "PASS" if self.success else "FAIL"
        print(f"\n{'='*70}")
        print(f"  {self.name}: {status}")
        print(f"  Diagnosis: {self.diagnosis}")
        print(f"  Wall time: {self.wall_time_s:.1f}s")
        for k, v in self.metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6e}")
            else:
                print(f"  {k}: {v}")
        if self.convergence:
            print(f"  Iterations: {len(self.convergence)}")
            last = self.convergence[-1]
            print(f"  Final cost: {last.get('cost', 'N/A')}")
            print(f"  Final grad_norm: {last.get('grad_norm', 'N/A')}")
        print(f"{'='*70}")


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def build_problem_and_solver(cfg: LadderConfig):
    """Build Shinnecock problem, solver, and ManningsBasisController."""
    from swe4dvar.forward import ADCIRCProblem, get_solver
    from swe4dvar.forward.augmented_control import ManningsBasisController
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params
    from mpi4py import MPI

    # Wind file (Vmax=0 gives zero wind but still needs the HDF5 structure)
    wind_file = _ensure_wind_file(cfg)

    forcing = GriddedForcing(str(wind_file), lat0=35)
    problem = ADCIRCProblem(
        adios_file=cfg.adios_file,
        spherical=True,
        solution_var="h",
        friction_law="mannings",
        wd=True,
        wd_alpha=1.5,
        dt=cfg.dt,
        bathy_adjustment=0,
        nt=cfg.nt_total,
        dramp=2.0,
        forcing=forcing,
    )
    solver = get_solver("DG")(problem, theta=1.0, p_degree=[1, 1])
    kl_kwargs = {}
    if cfg.basis_type == "kl":
        kl_kwargs = dict(
            basis_type="kl",
            kl_prior_std=cfg.kl_prior_std,
            kl_n_modes=cfg.kl_n_modes,
            kl_correlation_length=cfg.kl_correlation_length,
        )
    controller = ManningsBasisController(
        basis_shape=cfg.basis_shape,
        basis_width_fraction=cfg.basis_width_fraction,
        n_bounds=cfg.n_bounds,
        reference_n=cfg.reference_n,
        **kl_kwargs,
    )
    controller.bind(problem, solver)

    solver_params = get_default_solver_params(
        rtol=1e-5,
        atol=1e-6,
        max_it=20,
        relaxation_parameter=1.0,
        comm=MPI.COMM_WORLD,
        error_if_not_converged=False,
    )
    return problem, solver, controller, solver_params, forcing


def _ensure_wind_file(cfg: LadderConfig) -> Path:
    """Create a zero-wind HDF5 file if needed."""
    from mpi4py import MPI

    wind_dir = PROJECT_ROOT / "results" / "validation_ladder" / "wind"
    if MPI.COMM_WORLD.Get_rank() == 0:
        wind_dir.mkdir(parents=True, exist_ok=True)
    MPI.COMM_WORLD.Barrier()

    wind_file = wind_dir / f"vmax{cfg.truth_vmax:.0f}_ramp{cfg.wind_ramp_s:.0f}.h5"
    if wind_file.exists():
        return wind_file

    if MPI.COMM_WORLD.Get_rank() == 0:
        from experiments.shinnecock_study.wind_models import (
            DEFAULT_TRACK,
            HollandHurricaneConfig,
            WindGridConfig,
            generate_holland_wind_field,
            write_wind_hdf5,
        )

        times = np.arange(0.0, (cfg.nt_total + 1) * cfg.dt, cfg.dt)
        grid = WindGridConfig()
        wind_cfg = HollandHurricaneConfig(
            track_waypoints=DEFAULT_TRACK,
            Vmax=cfg.truth_vmax,
        )
        windx, windy, pressure = generate_holland_wind_field(
            wind_cfg, grid, times, wind_ramp_s=cfg.wind_ramp_s,
        )
        write_wind_hdf5(str(wind_file), grid, times, windx, windy, pressure)

    MPI.COMM_WORLD.Barrier()
    return wind_file


def generate_truth_trajectory(cfg, problem, solver, controller, solver_params):
    """Run forward model with truth parameters to get truth trajectory."""
    theta_truth = np.array(cfg.truth_coefficients, dtype=float)
    controller.apply(problem, solver, theta_truth)

    solver.time_loop(
        solver_parameters=solver_params,
        stations=[],
        plot_every=9999,
        save_state=True,
        store_jacobians=False,
        enable_video=False,
        monitor_progress=False,
    )

    state_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs
    truth_state = solver.u_n.x.array[:state_size].copy()

    return truth_state, theta_truth


def create_observations(cfg, solver, truth_state, controller, solver_params, problem):
    """Create synthetic observations from truth trajectory.

    If cfg.obs_velocity is True, observes h, u, and v at each point
    (3x the observation dimension) using a CompositeObservationOperator.
    """
    from dolfinx import la
    from swe4dvar.data_assimilation import (
        CompositeObservationOperator,
        PointObservationOperator,
    )
    from mpi4py import MPI

    # Select observation points
    coords = problem.mesh.geometry.x
    all_coords = MPI.COMM_WORLD.gather(coords, root=0)
    if MPI.COMM_WORLD.Get_rank() == 0:
        coords_all = np.vstack(all_coords)
        _, unique_idx = np.unique(
            np.round(coords_all[:, :2], decimals=10), axis=0, return_index=True,
        )
        unique_coords = coords_all[unique_idx]
        x_min, x_max = unique_coords[:, 0].min(), unique_coords[:, 0].max()
        y_min, y_max = unique_coords[:, 1].min(), unique_coords[:, 1].max()
        interior = unique_coords[
            (unique_coords[:, 0] > x_min + 1e-10) &
            (unique_coords[:, 0] < x_max - 1e-10) &
            (unique_coords[:, 1] > y_min + 1e-10) &
            (unique_coords[:, 1] < y_max - 1e-10)
        ]

        # Optional depth filtering: only place obs where depth < max_depth
        obs_max_depth = getattr(cfg, "obs_max_depth", None)
        if obs_max_depth is not None:
            from scipy.spatial import cKDTree as _cKDTree
            V_scalar = solver.V.sub(0).collapse()[0]
            sc = V_scalar.tabulate_dof_coordinates()[:, :2]
            h_init = solver.u_n.sub(0).collapse().x.array[:].copy()
            _tree = _cKDTree(sc)
            _, nearest_idx = _tree.query(interior[:, :2])
            depths = h_init[nearest_idx]
            depth_mask = (depths > 0.5) & (depths < obs_max_depth)
            interior = interior[depth_mask]
            print(f"  Depth filter (<{obs_max_depth}m): {int(np.sum(depth_mask))} of {len(depth_mask)} interior points retained")

        rng = np.random.default_rng(cfg.obs_seed)
        n_obs = max(1, int(len(interior) * cfg.obs_fraction))
        chosen = rng.choice(len(interior), size=min(n_obs, len(interior)), replace=False)
        obs_points = np.zeros((len(chosen), 3))
        obs_points[:, :2] = interior[chosen, :2]
    else:
        obs_points = None
    obs_points = MPI.COMM_WORLD.bcast(obs_points, root=0)

    # Build observation operator
    if getattr(cfg, "obs_velocity", False):
        # Composite: h + u + v at each point
        op_h = PointObservationOperator(
            solver.V, obs_points, component_indices=[0], comm=MPI.COMM_WORLD,
        )
        op_u = PointObservationOperator(
            solver.V, obs_points, component_indices=[1],
            sub_component=0, comm=MPI.COMM_WORLD,
        )
        op_v = PointObservationOperator(
            solver.V, obs_points, component_indices=[1],
            sub_component=1, comm=MPI.COMM_WORLD,
        )
        obs_operator = CompositeObservationOperator([op_h, op_u, op_v], comm=MPI.COMM_WORLD)
        n_obs_per_snapshot = 3 * len(obs_points)
        print(f"  Velocity observations enabled: {len(obs_points)} points × 3 components = {n_obs_per_snapshot} obs/snapshot")
    else:
        obs_operator = PointObservationOperator(solver.V, obs_points, comm=MPI.COMM_WORLD)
        n_obs_per_snapshot = len(obs_points)

    # Sample truth observations
    state_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs
    obs_times = cfg.obs_times
    truth_obs_matrix = np.zeros((len(obs_times), n_obs_per_snapshot))
    for i, t in enumerate(obs_times):
        vec = la.create_petsc_vector(
            solver.V.dofmap.index_map, solver.V.dofmap.index_map_bs,
        )
        vec.setArray(solver.storage.saved_states[t][:state_size])
        vec.assemble()
        obs_vec = obs_operator.forward(vec)
        truth_obs_matrix[i] = obs_vec.getArray().copy()
        obs_vec.destroy()
        vec.destroy()

    # Add noise
    rng = np.random.default_rng(cfg.obs_seed + 1000)
    signal = np.mean(np.abs(truth_obs_matrix), axis=1) + 1e-10
    noise_stds = cfg.obs_noise_level * signal
    noise = rng.normal(0.0, noise_stds[:, None], size=truth_obs_matrix.shape)
    noisy_obs = truth_obs_matrix + noise

    return {
        "obs_points": obs_points,
        "obs_operator": obs_operator,
        "obs_times": obs_times,
        "truth_obs": truth_obs_matrix,
        "noisy_obs": noisy_obs,
        "noise_stds": noise_stds,
    }


def build_cost_function(
    cfg: LadderConfig,
    solver,
    problem,
    controller,
    solver_params,
    truth_state: np.ndarray,
    obs_bundle: dict,
    *,
    background_error_std: float,
    regularization_weight: float = 1.0,
    method: str = "4dvar",
):
    """Build augmented cost function for a given experiment configuration."""
    from petsc4py import PETSc
    from swe4dvar.control import ControlLayout
    from swe4dvar.data_assimilation import (
        BlockDiagonalCovariance,
        DiagonalCovariance,
        create_cost_function,
    )
    from experiments.twin_experiment import ForwardModelWrapper

    theta_background = np.array(cfg.background_coefficients, dtype=float)
    theta_truth = np.array(cfg.truth_coefficients, dtype=float)

    # Perturb state
    rng = np.random.default_rng(123)
    scale = np.maximum(np.abs(truth_state), 1e-3)
    state_background = truth_state + rng.normal(0.0, background_error_std * scale)

    layout = ControlLayout(
        state_size=truth_state.size,
        theta_size=cfg.n_params,
        comm=PETSc.COMM_SELF,
    )

    forward_model = ForwardModelWrapper(
        solver, problem, solver_params,
        control_layout=layout,
        parameter_controller=controller,
        t_start=0.0,
    )

    m_background = layout.create_petsc_vec(u0=state_background, theta=theta_background)
    m_truth = layout.create_petsc_vec(u0=truth_state, theta=theta_truth)

    # Observation vectors
    obs_vecs = []
    for row in obs_bundle["noisy_obs"]:
        vec = PETSc.Vec().createSeq(row.size, comm=PETSc.COMM_SELF)
        vec.setArray(row.copy())
        vec.assemble()
        obs_vecs.append(vec)

    # Observation covariances
    obs_covs = {}
    for t, sigma in zip(obs_bundle["obs_times"], obs_bundle["noise_stds"]):
        obs_covs[t] = DiagonalCovariance(
            PETSc.COMM_SELF, size=obs_bundle["noisy_obs"].shape[1],
            variance=float(sigma**2),
        )

    # Background covariances
    state_variances = (background_error_std * np.maximum(np.abs(truth_state), 1e-3)) ** 2
    param_stds = np.full(cfg.n_params, 0.10, dtype=float)
    theta_variances = (param_stds ** 2) / max(regularization_weight, 1e-30)

    B_state = DiagonalCovariance(PETSc.COMM_SELF, size=truth_state.size, diagonal=state_variances)
    B_theta = DiagonalCovariance(PETSc.COMM_SELF, size=cfg.n_params, diagonal=theta_variances)
    background_cov = BlockDiagonalCovariance([B_state, B_theta], comm=PETSc.COMM_SELF)

    cost_fn = create_cost_function(
        "4dvar",
        forward_model=forward_model,
        observation_operator=obs_bundle["obs_operator"],
        background_cov=background_cov,
        observation_cov=obs_covs,
        m_background=m_background,
        observations=obs_vecs,
        obs_times=obs_bundle["obs_times"],
    )

    return cost_fn, layout, m_background, m_truth, forward_model


# ---------------------------------------------------------------------------
# Experiment 1: Gradient Check
# ---------------------------------------------------------------------------

def run_gradient_check(cfg: LadderConfig) -> ExperimentResult:
    """Verify adjoint gradient matches finite-difference gradient for θ block."""
    result = ExperimentResult("Exp1: Gradient Check")
    t0 = time.time()
    print("\n" + "="*70)
    print("EXPERIMENT 1: GRADIENT CHECK (adjoint vs finite-difference)")
    print("="*70)

    try:
        problem, solver, controller, solver_params, forcing = build_problem_and_solver(cfg)
        truth_state, theta_truth = generate_truth_trajectory(
            cfg, problem, solver, controller, solver_params,
        )
        obs_bundle = create_observations(cfg, solver, truth_state, controller, solver_params, problem)

        # Build cost function with moderate state prior
        cost_fn, layout, m_bg, m_truth, fwd = build_cost_function(
            cfg, solver, problem, controller, solver_params,
            truth_state, obs_bundle,
            background_error_std=0.01,
            regularization_weight=1.0,
        )

        # Evaluate at background
        print("  Computing adjoint gradient at background...")
        J0, grad_adjoint = cost_fn.value_gradient(m_bg)
        grad_arr = grad_adjoint.getArray().copy()

        # Extract theta block of gradient
        theta_grad_adjoint = grad_arr[layout.state_slice.stop:]
        print(f"  J(m_bg) = {J0:.6f}")
        print(f"  ||grad_full|| = {np.linalg.norm(grad_arr):.6e}")
        print(f"  ||grad_state|| = {np.linalg.norm(grad_arr[:layout.state_size]):.6e}")
        print(f"  ||grad_theta|| = {np.linalg.norm(theta_grad_adjoint):.6e}")

        # Finite-difference check on theta block only
        eps = 1e-4
        theta_grad_fd = np.zeros(cfg.n_params)
        bg_arr = m_bg.getArray().copy()

        for p in range(cfg.n_params):
            print(f"  FD check parameter {p}...")
            m_plus = m_bg.copy()
            m_minus = m_bg.copy()
            arr_plus = bg_arr.copy()
            arr_minus = bg_arr.copy()
            arr_plus[layout.state_size + p] += eps
            arr_minus[layout.state_size + p] -= eps
            m_plus.setArray(arr_plus)
            m_minus.setArray(arr_minus)
            m_plus.assemble()
            m_minus.assemble()

            J_plus = cost_fn.value(m_plus)
            J_minus = cost_fn.value(m_minus)
            theta_grad_fd[p] = (J_plus - J_minus) / (2 * eps)

            m_plus.destroy()
            m_minus.destroy()
            print(f"    adjoint={theta_grad_adjoint[p]:.6e}  FD={theta_grad_fd[p]:.6e}")

        # Compare
        if np.linalg.norm(theta_grad_fd) < 1e-12:
            rel_error = float("inf")
            result.diagnosis = "FD gradient is zero — no Manning sensitivity detected"
        else:
            rel_error = np.linalg.norm(theta_grad_adjoint - theta_grad_fd) / np.linalg.norm(theta_grad_fd)
            if rel_error < 0.05:
                result.success = True
                result.diagnosis = f"Gradient check PASSED: relative error = {rel_error:.4e}"
            elif rel_error < 0.20:
                result.success = True
                result.diagnosis = f"Gradient check MARGINAL: relative error = {rel_error:.4e}"
            else:
                result.diagnosis = f"Gradient check FAILED: relative error = {rel_error:.4e}"

        result.metrics = {
            "cost_at_background": J0,
            "grad_norm_full": float(np.linalg.norm(grad_arr)),
            "grad_norm_state": float(np.linalg.norm(grad_arr[:layout.state_size])),
            "grad_norm_theta": float(np.linalg.norm(theta_grad_adjoint)),
            "theta_grad_adjoint": theta_grad_adjoint.tolist(),
            "theta_grad_fd": theta_grad_fd.tolist(),
            "relative_error": rel_error,
        }

        # Cleanup
        del cost_fn, fwd, solver, problem, forcing
        gc.collect()

    except Exception as e:
        result.diagnosis = f"EXCEPTION: {e}"
        import traceback
        traceback.print_exc()

    result.wall_time_s = time.time() - t0
    result.print_summary()
    return result


# ---------------------------------------------------------------------------
# Experiment 2: Parameter-Only Recovery (state nearly fixed)
# ---------------------------------------------------------------------------

def run_parameter_only(cfg: LadderConfig) -> ExperimentResult:
    """Optimize θ ONLY — state fixed at truth (state_size=0)."""
    result = ExperimentResult("Exp2: Parameter-Only Recovery")
    t0 = time.time()
    print("\n" + "="*70)
    print("EXPERIMENT 2: PARAMETER-ONLY RECOVERY (state_size=0, state=truth)")
    print("="*70)

    try:
        from petsc4py import PETSc
        from swe4dvar.control import ControlLayout
        from swe4dvar.data_assimilation import DiagonalCovariance, create_cost_function
        from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper
        from experiments.twin_experiment import ForwardModelWrapper

        problem, solver, controller, solver_params, forcing = build_problem_and_solver(cfg)
        truth_state, theta_truth = generate_truth_trajectory(
            cfg, problem, solver, controller, solver_params,
        )
        obs_bundle = create_observations(cfg, solver, truth_state, controller, solver_params, problem)

        theta_background = np.array(cfg.background_coefficients, dtype=float)

        # Parameter-only layout: state_size=0, reference_state=truth
        layout = ControlLayout(state_size=0, theta_size=cfg.n_params, comm=PETSc.COMM_SELF)

        forward_model = ForwardModelWrapper(
            solver, problem, solver_params,
            control_layout=layout,
            parameter_controller=controller,
            reference_state=truth_state,  # Fixed at truth
            t_start=0.0,
        )

        m_bg = layout.create_petsc_vec(theta=theta_background)
        m_truth_vec = layout.create_petsc_vec(theta=theta_truth)

        # Observation vectors
        obs_vecs = []
        for row in obs_bundle["noisy_obs"]:
            vec = PETSc.Vec().createSeq(row.size, comm=PETSc.COMM_SELF)
            vec.setArray(row.copy())
            vec.assemble()
            obs_vecs.append(vec)

        obs_covs = {}
        for t, sigma in zip(obs_bundle["obs_times"], obs_bundle["noise_stds"]):
            obs_covs[t] = DiagonalCovariance(
                PETSc.COMM_SELF, size=obs_bundle["noisy_obs"].shape[1],
                variance=float(sigma**2),
            )

        # Parameter-only covariance (no state block)
        param_stds = np.full(cfg.n_params, 0.10, dtype=float)
        regularization_weight = 0.1
        theta_variances = (param_stds ** 2) / max(regularization_weight, 1e-30)
        background_cov = DiagonalCovariance(
            PETSc.COMM_SELF, size=cfg.n_params, diagonal=theta_variances,
        )

        cost_fn = create_cost_function(
            "4dvar",
            forward_model=forward_model,
            observation_operator=obs_bundle["obs_operator"],
            background_cov=background_cov,
            observation_cov=obs_covs,
            m_background=m_bg,
            observations=obs_vecs,
            obs_times=obs_bundle["obs_times"],
        )

        # Bounds — parameter-only (no state block)
        lower = layout.create_petsc_vec(
            theta=np.array([b[0] for b in cfg.theta_bounds]),
        )
        upper = layout.create_petsc_vec(
            theta=np.array([b[1] for b in cfg.theta_bounds]),
        )

        iteration_history = []

        def callback(x, iteration, cost, grad_norm):
            control = layout.unpack_vec(x)
            entry = {
                "iteration": int(iteration),
                "cost": float(cost),
                "grad_norm": float(grad_norm),
                "theta": control.theta.tolist(),
                "param_rmse": _rmse(control.theta, theta_truth),
            }
            iteration_history.append(entry)
            print(f"  iter {iteration}: J={cost:.4f}  ||g||={grad_norm:.4e}  "
                  f"θ_rmse={entry['param_rmse']:.6f}  θ={np.array2string(control.theta, precision=4)}")
            return entry

        optimizer = PETScTAOWrapper(
            cost_fn,
            tao_type="blmvm",
            lower_bounds=lower,
            upper_bounds=upper,
            options={
                "max_iterations": cfg.max_iterations,
                "gradient_tolerance": 1e-6,
                "cost_tolerance": 1e-10,
                "verbose": True,
                "line_search_type": "armijo",
                "line_search_max_funcs": 30,
                "line_search_initial_step": 0.1,
                "iteration_callback": callback,
            },
        )

        print(f"  Background θ: {np.array(cfg.background_coefficients)}")
        print(f"  Truth θ:      {np.array(cfg.truth_coefficients)}")
        print(f"  Initial θ RMSE: {_rmse(cfg.background_coefficients, cfg.truth_coefficients):.6f}")
        print(f"  Control dim: {layout.total_size} (parameter-only, state=truth)")

        m_analysis = optimizer.solve(m_bg.copy())
        analysis = layout.unpack_vec(m_analysis)

        bg_rmse = _rmse(cfg.background_coefficients, cfg.truth_coefficients)
        analysis_rmse = _rmse(analysis.theta, theta_truth)
        reduction = (bg_rmse - analysis_rmse) / bg_rmse * 100

        result.convergence = iteration_history
        result.metrics = {
            "theta_truth": list(cfg.truth_coefficients),
            "theta_background": list(cfg.background_coefficients),
            "theta_analysis": analysis.theta.tolist(),
            "bg_param_rmse": bg_rmse,
            "analysis_param_rmse": analysis_rmse,
            "rmse_reduction_pct": reduction,
            "converged": bool(optimizer.converged),
            "iterations": int(optimizer.iteration),
            "n_func_evals": int(optimizer.n_func_evals),
        }

        # Success criteria
        theta_moved = np.linalg.norm(analysis.theta - np.array(cfg.background_coefficients)) > 1e-6
        if reduction >= 30 and theta_moved:
            result.success = True
            result.diagnosis = f"Parameter recovery: {reduction:.1f}% RMSE reduction"
        elif reduction >= 10 and theta_moved:
            result.success = True
            result.diagnosis = f"Partial recovery: {reduction:.1f}% RMSE reduction (≥10%)"
        elif theta_moved and optimizer.iteration > 0:
            result.diagnosis = f"θ moved but weak recovery: {reduction:.1f}% RMSE change"
        elif not theta_moved:
            result.diagnosis = f"θ did not move from initialization (LS failure?)"
        else:
            result.diagnosis = f"Failed: {reduction:.1f}% RMSE change, {optimizer.iteration} iters"

        del cost_fn, forward_model, solver, problem, forcing, optimizer
        gc.collect()

    except Exception as e:
        result.diagnosis = f"EXCEPTION: {e}"
        import traceback
        traceback.print_exc()

    result.wall_time_s = time.time() - t0
    result.print_summary()
    return result


# ---------------------------------------------------------------------------
# Experiment 3: Weakly Coupled Joint Estimation
# ---------------------------------------------------------------------------

def run_weakly_coupled(cfg: LadderConfig) -> ExperimentResult:
    """Joint [u0; θ] estimation with moderate state prior."""
    result = ExperimentResult("Exp3: Weakly Coupled Joint")
    t0 = time.time()
    print("\n" + "="*70)
    print("EXPERIMENT 3: WEAKLY COUPLED JOINT ESTIMATION")
    print("="*70)

    try:
        from petsc4py import PETSc
        from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper

        problem, solver, controller, solver_params, forcing = build_problem_and_solver(cfg)
        truth_state, theta_truth = generate_truth_trajectory(
            cfg, problem, solver, controller, solver_params,
        )
        obs_bundle = create_observations(cfg, solver, truth_state, controller, solver_params, problem)

        cost_fn, layout, m_bg, m_truth, fwd = build_cost_function(
            cfg, solver, problem, controller, solver_params,
            truth_state, obs_bundle,
            background_error_std=0.005,   # Small state perturbation
            regularization_weight=0.1,    # Weak parameter prior
        )

        lower = layout.create_petsc_vec(
            packed=np.concatenate([
                np.full(layout.state_size, -1e20),
                np.array([b[0] for b in cfg.theta_bounds]),
            ])
        )
        upper = layout.create_petsc_vec(
            packed=np.concatenate([
                np.full(layout.state_size, 1e20),
                np.array([b[1] for b in cfg.theta_bounds]),
            ])
        )

        iteration_history = []

        def callback(x, iteration, cost, grad_norm):
            control = layout.unpack_vec(x)
            entry = {
                "iteration": int(iteration),
                "cost": float(cost),
                "grad_norm": float(grad_norm),
                "theta": control.theta.tolist(),
                "param_rmse": _rmse(control.theta, theta_truth),
                "state_rmse": _rmse(control.u0, truth_state),
            }
            iteration_history.append(entry)
            print(f"  iter {iteration}: J={cost:.4f}  ||g||={grad_norm:.4e}  "
                  f"θ_rmse={entry['param_rmse']:.6f}  u_rmse={entry['state_rmse']:.6f}")
            return entry

        optimizer = PETScTAOWrapper(
            cost_fn, tao_type="blmvm",
            lower_bounds=lower, upper_bounds=upper,
            options={
                "max_iterations": cfg.max_iterations,
                "gradient_tolerance": cfg.gradient_tolerance,
                "cost_tolerance": 1e-8,
                "verbose": True,
                "line_search_type": "armijo",
                "line_search_max_funcs": 50,
                "line_search_initial_step": 1.0,
                "iteration_callback": callback,
            },
        )

        m_analysis = optimizer.solve(m_bg.copy())
        analysis = layout.unpack_vec(m_analysis)

        bg_rmse = _rmse(cfg.background_coefficients, cfg.truth_coefficients)
        analysis_rmse = _rmse(analysis.theta, theta_truth)
        reduction = (bg_rmse - analysis_rmse) / bg_rmse * 100

        state_bg_rmse = _rmse(m_bg.getArray()[:layout.state_size], truth_state)
        state_ana_rmse = _rmse(analysis.u0, truth_state)
        state_reduction = (state_bg_rmse - state_ana_rmse) / max(state_bg_rmse, 1e-30) * 100

        result.convergence = iteration_history
        result.metrics = {
            "theta_analysis": analysis.theta.tolist(),
            "bg_param_rmse": bg_rmse,
            "analysis_param_rmse": analysis_rmse,
            "param_rmse_reduction_pct": reduction,
            "bg_state_rmse": state_bg_rmse,
            "analysis_state_rmse": state_ana_rmse,
            "state_rmse_reduction_pct": state_reduction,
            "converged": bool(optimizer.converged),
            "iterations": int(optimizer.iteration),
            "n_func_evals": int(optimizer.n_func_evals),
        }

        theta_moved = np.linalg.norm(analysis.theta - np.array(cfg.background_coefficients)) > 1e-6
        if reduction >= 20 and theta_moved:
            result.success = True
            result.diagnosis = f"Joint recovery: θ {reduction:.1f}%, u {state_reduction:.1f}% RMSE reduction"
        elif optimizer.iteration > 0 and theta_moved:
            result.success = True  # Partial but optimizer ran
            result.diagnosis = f"Partial: θ {reduction:.1f}%, u {state_reduction:.1f}% reduction, {optimizer.iteration} iters"
        else:
            result.diagnosis = f"Failed: θ {reduction:.1f}% change, u {state_reduction:.1f}% change"

        del cost_fn, fwd, solver, problem, forcing, optimizer
        gc.collect()

    except Exception as e:
        result.diagnosis = f"EXCEPTION: {e}"
        import traceback
        traceback.print_exc()

    result.wall_time_s = time.time() - t0
    result.print_summary()
    return result


# ---------------------------------------------------------------------------
# Experiment 4: Full Joint 4D-Var
# ---------------------------------------------------------------------------

def run_full_joint(cfg: LadderConfig) -> ExperimentResult:
    """Realistic augmented 4D-Var with balanced priors."""
    result = ExperimentResult("Exp4: Full Joint 4D-Var")
    t0 = time.time()
    print("\n" + "="*70)
    print("EXPERIMENT 4: FULL JOINT 4D-VAR")
    print("="*70)

    try:
        from petsc4py import PETSc
        from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper

        problem, solver, controller, solver_params, forcing = build_problem_and_solver(cfg)
        truth_state, theta_truth = generate_truth_trajectory(
            cfg, problem, solver, controller, solver_params,
        )
        obs_bundle = create_observations(cfg, solver, truth_state, controller, solver_params, problem)

        cost_fn, layout, m_bg, m_truth, fwd = build_cost_function(
            cfg, solver, problem, controller, solver_params,
            truth_state, obs_bundle,
            background_error_std=0.02,    # Realistic state error
            regularization_weight=0.5,    # Moderate parameter prior
        )

        lower = layout.create_petsc_vec(
            packed=np.concatenate([
                np.full(layout.state_size, -1e20),
                np.array([b[0] for b in cfg.theta_bounds]),
            ])
        )
        upper = layout.create_petsc_vec(
            packed=np.concatenate([
                np.full(layout.state_size, 1e20),
                np.array([b[1] for b in cfg.theta_bounds]),
            ])
        )

        iteration_history = []

        def callback(x, iteration, cost, grad_norm):
            control = layout.unpack_vec(x)
            entry = {
                "iteration": int(iteration),
                "cost": float(cost),
                "grad_norm": float(grad_norm),
                "theta": control.theta.tolist(),
                "param_rmse": _rmse(control.theta, theta_truth),
                "state_rmse": _rmse(control.u0, truth_state),
            }
            iteration_history.append(entry)
            print(f"  iter {iteration}: J={cost:.4f}  ||g||={grad_norm:.4e}  "
                  f"θ_rmse={entry['param_rmse']:.6f}  u_rmse={entry['state_rmse']:.6f}")
            return entry

        optimizer = PETScTAOWrapper(
            cost_fn, tao_type="blmvm",
            lower_bounds=lower, upper_bounds=upper,
            options={
                "max_iterations": cfg.max_iterations,
                "gradient_tolerance": cfg.gradient_tolerance,
                "cost_tolerance": 1e-8,
                "verbose": True,
                "line_search_type": "armijo",
                "line_search_max_funcs": 50,
                "line_search_initial_step": 1.0,
                "iteration_callback": callback,
            },
        )

        m_analysis = optimizer.solve(m_bg.copy())
        analysis = layout.unpack_vec(m_analysis)

        bg_rmse = _rmse(cfg.background_coefficients, cfg.truth_coefficients)
        analysis_rmse = _rmse(analysis.theta, theta_truth)
        reduction = (bg_rmse - analysis_rmse) / bg_rmse * 100

        result.convergence = iteration_history
        result.metrics = {
            "theta_analysis": analysis.theta.tolist(),
            "bg_param_rmse": bg_rmse,
            "analysis_param_rmse": analysis_rmse,
            "param_rmse_reduction_pct": reduction,
            "converged": bool(optimizer.converged),
            "iterations": int(optimizer.iteration),
            "n_func_evals": int(optimizer.n_func_evals),
        }

        theta_moved = np.linalg.norm(analysis.theta - np.array(cfg.background_coefficients)) > 1e-6
        if reduction >= 20 and theta_moved:
            result.success = True
            result.diagnosis = f"Full joint recovery: {reduction:.1f}% RMSE reduction"
        elif optimizer.iteration > 0 and theta_moved:
            result.success = True
            result.diagnosis = f"Partial: {reduction:.1f}% reduction, {optimizer.iteration} iters"
        else:
            result.diagnosis = f"Failed: {reduction:.1f}% change"

        del cost_fn, fwd, solver, problem, forcing, optimizer
        gc.collect()

    except Exception as e:
        result.diagnosis = f"EXCEPTION: {e}"
        import traceback
        traceback.print_exc()

    result.wall_time_s = time.time() - t0
    result.print_summary()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_results(results: list[ExperimentResult], output_dir: Path, cfg: LadderConfig = None):
    """Save all results to JSON."""
    if cfg is None:
        cfg = LadderConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "nt_ramp": cfg.nt_ramp,
            "nt_da": cfg.nt_da,
            "basis_shape": list(cfg.basis_shape),
            "truth_coefficients": list(cfg.truth_coefficients),
            "truth_vmax": cfg.truth_vmax,
            "wind_ramp_s": cfg.wind_ramp_s,
        },
        "experiments": [r.to_dict() for r in results],
        "go_no_go": "GO" if all(r.success for r in results) else (
            "CONDITIONAL" if any(r.success for r in results) else "NO-GO"
        ),
    }

    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir / 'validation_results.json'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Manning's-n Validation Ladder")
    parser.add_argument(
        "--experiment", default="1",
        choices=["1", "2", "3", "4", "all"],
        help="Which experiment to run (1-4 or all)",
    )
    parser.add_argument(
        "--single-param", action="store_true",
        help="Use 1-parameter basis (basis_shape=[1,1]) for cleaner identifiability test",
    )
    parser.add_argument(
        "--wind", action="store_true",
        help="Enable wind forcing (truth_vmax=30) to excite spatially varying friction sensitivity",
    )
    parser.add_argument(
        "--wind-vmax", type=float, default=30.0,
        help="Wind Vmax in m/s (default 30.0, only used with --wind)",
    )
    parser.add_argument(
        "--dense-obs", action="store_true",
        help="Dense observations (30%% mesh, every timestep) to improve identifiability",
    )
    parser.add_argument(
        "--obs-velocity", action="store_true",
        help="Observe h + u + v (velocity) instead of h only. 3x observation dimension.",
    )
    parser.add_argument(
        "--obs-max-depth", type=float, default=None,
        help="Only place observations where depth < this value (meters). Concentrates obs in shallow water.",
    )
    parser.add_argument(
        "--kl", action="store_true",
        help="Use KL expansion basis instead of Gaussian RBF",
    )
    parser.add_argument(
        "--kl-modes", type=int, default=None,
        help="Number of KL modes to retain (default: auto from 99%% variance)",
    )
    parser.add_argument(
        "--kl-length", type=float, default=None,
        help="KL correlation length (default: 1/4 domain extent)",
    )
    parser.add_argument(
        "--long-window", action="store_true",
        help="Extended DA window (8h) for friction accumulation",
    )
    parser.add_argument(
        "--nt-da", type=int, default=None,
        help="Override DA window length (timesteps)",
    )
    parser.add_argument(
        "--nt-ramp", type=int, default=None,
        help="Override ramp length (timesteps)",
    )
    parser.add_argument(
        "--dt", type=float, default=None,
        help="Override timestep size in seconds (default 600). Smaller dt reduces implicit diffusion.",
    )
    args = parser.parse_args()

    # Build config from composable flags
    overrides = {}

    # Wind
    if args.wind:
        overrides.update(truth_vmax=args.wind_vmax, wind_ramp_s=3600.0)

    # Long window
    if args.long_window:
        overrides.update(
            nt_ramp=args.nt_ramp if args.nt_ramp else 12,
            nt_da=args.nt_da if args.nt_da else 48,
            obs_frequency=3,
            obs_fraction=0.15,
            max_iterations=20,
        )
    else:
        if args.nt_ramp:
            overrides["nt_ramp"] = args.nt_ramp
        if args.nt_da:
            overrides["nt_da"] = args.nt_da

    # Dense observations
    if args.dense_obs:
        overrides.update(obs_fraction=0.30, obs_frequency=1)

    # Basis type
    if args.kl:
        n_modes = args.kl_modes if args.kl_modes else 3
        overrides.update(
            basis_type="kl",
            kl_n_modes=n_modes,
            kl_correlation_length=args.kl_length,
            kl_prior_std=0.5,
            truth_coefficients=tuple([0.30] + [-0.25] * (n_modes - 1)),
            background_coefficients=tuple([0.0] * n_modes),
            theta_bounds=tuple([(-0.8, 0.8)] * n_modes),
        )
        # For long-window + KL, override truth coefficients for n_modes
        if args.long_window:
            overrides["truth_coefficients"] = tuple([0.30] + [-0.25] * (n_modes - 1))
    elif args.single_param:
        overrides.update(
            basis_shape=(1, 1),
            truth_coefficients=(0.25,),
            background_coefficients=(0.0,),
            theta_bounds=((-0.5, 0.5),),
        )
    elif args.long_window and "truth_coefficients" not in overrides:
        overrides["truth_coefficients"] = (0.30, -0.25)

    # Observation type overrides
    if args.obs_velocity:
        overrides["obs_velocity"] = True
    if args.obs_max_depth is not None:
        overrides["obs_max_depth"] = args.obs_max_depth

    # Timestep override — adjusts nt_ramp and nt_da to preserve physical time
    if args.dt is not None:
        base_dt = 600.0  # default
        dt_ratio = base_dt / args.dt
        overrides["dt"] = args.dt
        # Scale ramp and DA timestep counts to preserve the same physical time
        current_nt_ramp = overrides.get("nt_ramp", LadderConfig.nt_ramp)
        current_nt_da = overrides.get("nt_da", LadderConfig.nt_da)
        overrides["nt_ramp"] = int(round(current_nt_ramp * dt_ratio))
        overrides["nt_da"] = int(round(current_nt_da * dt_ratio))
        # Scale obs_frequency to preserve physical observation interval
        current_obs_freq = overrides.get("obs_frequency", LadderConfig.obs_frequency)
        overrides["obs_frequency"] = max(1, int(round(current_obs_freq * dt_ratio)))

    cfg = LadderConfig(**overrides)
    output_dir = PROJECT_ROOT / "results" / "validation_ladder"
    results = []

    experiments = {
        "1": ("Gradient Check", run_gradient_check),
        "2": ("Parameter-Only", run_parameter_only),
        "3": ("Weakly Coupled", run_weakly_coupled),
        "4": ("Full Joint", run_full_joint),
    }

    if args.experiment == "all":
        order = ["1", "2", "3", "4"]
    else:
        order = [args.experiment]

    for exp_id in order:
        name, func = experiments[exp_id]
        print(f"\n{'#'*70}")
        print(f"# Starting: {name}")
        print(f"{'#'*70}")

        result = func(cfg)
        results.append(result)

        if not result.success and args.experiment == "all":
            print(f"\n*** STOPPING LADDER: {name} FAILED ***")
            print(f"*** Diagnosis: {result.diagnosis} ***")
            break

    summary = save_results(results, output_dir, cfg=cfg)

    # Print final verdict
    print("\n" + "#"*70)
    print("# FINAL VERDICT")
    print("#"*70)
    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"  {r.name}: {status} — {r.diagnosis}")
    print(f"\n  GO/NO-GO: {summary['go_no_go']}")
    print("#"*70)


if __name__ == "__main__":
    raise SystemExit(main())

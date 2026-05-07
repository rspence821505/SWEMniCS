#!/usr/bin/env python3
"""Serial prototype: state-only vs augmented (state + Manning's-n) DA on the
idealized inlet, with a forecast-after-analysis comparison.

Goal: test whether augmenting the control with a low-dimensional Manning's-n
parameterization can reduce the multi-window forecast drift that state-only
cycling DA cannot absorb.

Scope:
  - serial only (np=1) -- the augmented control stack is serial-only by design
    in src/swe4dvar/control/augmented_control.py:ControlLayout
  - one DA window (W0) of the idealized inlet, then a forward forecast step
    that mimics the start of W1
  - comparison of forecast RMSE for state-only analysis vs augmented analysis

Usage (run from repo root, single rank):
    python experiments/idealized_inlet_augmented_serial.py --mode state_only
    python experiments/idealized_inlet_augmented_serial.py --mode augmented
    python experiments/idealized_inlet_augmented_serial.py --mode both

NOT a production driver. The full MPI cycling workflow remains the place for
distributed runs; this script exists only to prove the augmented machinery is
wired correctly and to measure whether the parameter control reduces drift.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["state_only", "augmented", "both"],
        default="both",
        help="Which DA mode(s) to run. 'both' runs each in turn and reports a "
             "side-by-side comparison.",
    )
    p.add_argument("--dt", type=float, default=600.0, help="Timestep (s)")
    p.add_argument("--nt-ramp", type=int, default=24,
                   help="Ramp/spin-up timesteps before DA")
    p.add_argument("--nt-da", type=int, default=6,
                   help="DA-window timesteps (default 6 = 1h at dt=600)")
    p.add_argument("--nt-forecast", type=int, default=6,
                   help="Forecast timesteps after analysis (default 6 = 1h)")
    p.add_argument("--vmax", type=float, default=10.0, help="Storm Vmax (m/s)")
    p.add_argument("--track-shift", type=float, default=0.0,
                   help="Storm track shift (km)")
    p.add_argument("--track-duration-s", type=float, default=28800.0,
                   help="Storm track duration (s)")
    p.add_argument("--obs-fraction", type=float, default=0.1)
    p.add_argument("--obs-frequency", type=int, default=3,
                   help="Obs every N timesteps (3 -> 3 obs/window for nt_da=6)")
    p.add_argument("--obs-noise-level", type=float, default=0.01)
    p.add_argument("--background-error-std", type=float, default=0.02)
    p.add_argument("--max-iterations", type=int, default=10)
    p.add_argument("--max-funcs", type=int, default=10)
    p.add_argument("--basis-shape", type=int, nargs=2, default=(3, 2),
                   help="Manning Gaussian basis shape (nx ny)")
    p.add_argument("--basis-width-fraction", type=float, default=0.35)
    p.add_argument("--reference-n", type=float, default=0.025,
                   help="Reference Manning's-n (uniform background field)")
    p.add_argument("--n-bounds", type=float, nargs=2, default=(0.01, 0.08))
    p.add_argument("--theta-prior-std", type=float, default=0.5,
                   help="Prior std on theta (background term scaling)")
    p.add_argument("--truth-theta-coefficients", type=float, nargs="*", default=None,
                   help="Inject Manning's-n model error into the truth trajectory. "
                        "Length must equal basis-shape[0]*basis-shape[1]. When set, "
                        "truth uses n(theta_truth) while bg uses n(theta=0); state-only "
                        "DA cannot represent this mismatch, augmented can recover via theta.")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "outputs" / "idealized_inlet_augmented_serial")
    p.add_argument("--seed", type=int, default=42)
    # ----- coarse-mesh mode -----
    p.add_argument("--coarse-mesh", action="store_true",
                   help="Use a generated coarse rectangular mesh covering the same "
                        "physical domain (x∈[0,50000] m, y∈[0,30500] m) and the same "
                        "boundary/bathymetry/friction logic as the production "
                        "Ideal_Inlet mesh, but with a much smaller DOF count. "
                        "Required for serial augmented prototypes — the production "
                        "207k-DOF mesh is too expensive for serial GMRES+ILU.")
    p.add_argument("--nx", type=int, default=50,
                   help="Coarse-mesh cells in x direction (default 50). "
                        "Each cell becomes 2 triangles. With --ny the resulting state "
                        "size is roughly 3 × (nx+1) × (ny+1) for DG-mixed P1 elements.")
    p.add_argument("--ny", type=int, default=30,
                   help="Coarse-mesh cells in y direction (default 30).")
    return p.parse_args()


def _serial_guard():
    """Refuse to run under multi-rank MPI. Augmented control is serial-only."""
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() != 1:
        raise SystemExit(
            "This prototype is serial-only (augmented controls require "
            "PETSc.COMM_SELF). Run with: python <this script>"
        )


def _build_idealized_inlet_problem(args: argparse.Namespace, *, nt: int):
    """Construct an IdealizedInlet problem + DG solver. Returns (prob, solver)."""
    from swe4dvar.forward.problems import IdealizedInlet
    from swe4dvar.forward.solvers import get_solver

    if getattr(args, "coarse_mesh", False):
        # Generate a coarse rectangular mesh covering the same physical
        # domain as the production Ideal_Inlet (x∈[0,50000] m,
        # y∈[0,30500] m). Inherit bathymetry/boundaries/friction/forcing
        # from IdealizedInlet so only mesh resolution differs from
        # the production setup.
        nx = int(getattr(args, "nx", 50))
        ny = int(getattr(args, "ny", 30))

        class _CoarseIdealizedInlet(IdealizedInlet):
            def _create_mesh(self):
                from dolfinx import mesh as _dmesh
                from mpi4py import MPI as _MPI
                self.mesh = _dmesh.create_rectangle(
                    _MPI.COMM_WORLD,
                    [(0.0, 0.0), (50000.0, 30500.0)],
                    [nx, ny],
                    cell_type=_dmesh.CellType.triangle,
                )
                self.boundaries = [
                    (1, lambda x: np.isclose(x[1], 0)),
                    (
                        2,
                        lambda x: np.logical_not(np.isclose(x[1], 0))
                        | np.logical_and(np.isclose(x[1], 0), np.isclose(x[0], 0))
                        | np.logical_and(np.isclose(x[1], 0), np.isclose(x[0], 50000)),
                    ),
                ]

        prob = _CoarseIdealizedInlet(
            dt=args.dt,
            nt=nt,
            xdmf_file=str(PROJECT_ROOT / "data" / "Ideal_Inlet" / "Ideal_Inlet.xdmf"),
            friction_law="mannings",
            solution_var="h",
            dramp=args.nt_ramp * args.dt / 86400.0,
        )
    else:
        prob = IdealizedInlet(
            dt=args.dt,
            nt=nt,
            xdmf_file=str(PROJECT_ROOT / "data" / "Ideal_Inlet" / "Ideal_Inlet.xdmf"),
            friction_law="mannings",
            solution_var="h",
            dramp=args.nt_ramp * args.dt / 86400.0,
        )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])
    return prob, solver


def _generate_wind_files(args: argparse.Namespace, output_dir: Path) -> Tuple[Path, Path]:
    """Reuse the idealized-inlet vortex generator with the shipped CartesianVortexConfig."""
    from experiments.idealized_inlet_twin import (
        CartesianVortexConfig,
        generate_cartesian_vortex,
        write_cartesian_wind_hdf5,
    )
    nt_total = args.nt_ramp + args.nt_da + args.nt_forecast
    times = np.arange(0.0, (nt_total + 1) * args.dt, args.dt)
    x_grid = np.linspace(-10000, 60000, 71)
    y_grid = np.linspace(-30000, 50000, 81)

    truth_cfg = CartesianVortexConfig(
        Vmax=args.vmax,
        Rmax=15000.0,
        ramp_time_s=args.nt_ramp * args.dt,
        track_duration_s=args.track_duration_s,
    )
    wind_dir = output_dir / "wind"
    wind_dir.mkdir(parents=True, exist_ok=True)
    truth_file = wind_dir / "truth.h5"
    pert_file = wind_dir / "perturbed.h5"

    if not truth_file.exists():
        wx, wy, p = generate_cartesian_vortex(truth_cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(truth_file), x_grid, y_grid, times, wx, wy, p)

    if not pert_file.exists():
        # Identical wind for prototype (no model error in wind; test focuses on
        # whether augmented control can absorb residual drift from IC perturbation).
        # If you want forecast bias from wind track shift, override --track-shift.
        if args.track_shift != 0.0:
            from experiments.idealized_inlet_twin import create_perturbed_config
            pert_cfg = create_perturbed_config(truth_cfg, args.track_shift)
        else:
            pert_cfg = truth_cfg
        wx, wy, p = generate_cartesian_vortex(pert_cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(pert_file), x_grid, y_grid, times, wx, wy, p)

    return truth_file, pert_file


def _spin_up_and_truth(args: argparse.Namespace, truth_wind_file: Path):
    """Run the ramp + (DA + forecast) windows on the truth solver. Returns:
        ramp_end_state: state at end of ramp (start of W0)
        truth_da_states: list of length nt_da+1 (states k=0..nt_da of W0)
        truth_forecast_states: list of length nt_forecast+1 (states for forecast window)
    """
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params

    mesh_mode = (
        f"COARSE generated rectangle ({args.nx}x{args.ny} cells)"
        if args.coarse_mesh
        else "PRODUCTION Ideal_Inlet xdmf"
    )
    print(f"\n=== Truth: ramp + W0 + forecast (mesh={mesh_mode}) ===", flush=True)
    forcing = GriddedForcing(str(truth_wind_file), cartesian=True)
    prob, solver = _build_idealized_inlet_problem(args, nt=args.nt_ramp)
    prob.forcing = forcing
    forcing.set_V(solver.V_scalar)
    forcing.evaluate(prob.t)

    state_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs
    n_vertices = prob.mesh.geometry.x.shape[0]
    n_cells = prob.mesh.topology.index_map(prob.mesh.topology.dim).size_local
    print(f"  Mesh:  {n_vertices} vertices, {n_cells} cells", flush=True)
    print(f"  State: {state_size} DOFs", flush=True)

    if args.truth_theta_coefficients is not None:
        from swe4dvar.forward.augmented_control import ManningsBasisController
        expected = args.basis_shape[0] * args.basis_shape[1]
        truth_theta = np.asarray(args.truth_theta_coefficients, dtype=float)
        if truth_theta.size != expected:
            raise ValueError(
                f"--truth-theta-coefficients length {truth_theta.size} != "
                f"basis_shape product {expected}"
            )
        truth_controller = ManningsBasisController(
            basis_shape=tuple(args.basis_shape),
            basis_width_fraction=args.basis_width_fraction,
            n_bounds=tuple(args.n_bounds),
            reference_n=args.reference_n,
        )
        truth_controller.bind(prob, solver)
        truth_controller.apply(prob, solver, truth_theta)
        n_field = truth_controller._mannings_field_function.x.array
        print(
            f"  Truth Manning: theta_norm={np.linalg.norm(truth_theta):.4f} "
            f"n.min={n_field.min():.4f} n.mean={n_field.mean():.4f} n.max={n_field.max():.4f}",
            flush=True,
        )

    params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=20, relaxation_parameter=0.7,
        comm=solver.problem.mesh.comm, error_if_not_converged=False,
    )
    # Ramp
    solver.time_loop(
        solver_parameters=params, stations=np.array([[0.0, 0.0, 0.0]]),
        plot_every=9999, save_state=False, store_jacobians=False, enable_video=False,
    )
    ramp_end_state = solver.u_n.x.array[:state_size].copy()

    # DA window — record trajectory
    prob.nt = args.nt_da
    solver.storage.clear()
    solver.time_loop(
        solver_parameters=params, stations=np.array([[0.0, 0.0, 0.0]]),
        plot_every=9999, save_state=True, store_jacobians=False, enable_video=False,
    )
    truth_da_states = [s[:state_size].copy() for s in solver.storage.saved_states]
    da_end_state = solver.u_n.x.array[:state_size].copy()

    # Forecast window — record trajectory
    prob.nt = args.nt_forecast
    solver.storage.clear()
    solver.time_loop(
        solver_parameters=params, stations=np.array([[0.0, 0.0, 0.0]]),
        plot_every=9999, save_state=True, store_jacobians=False, enable_video=False,
    )
    truth_forecast_states = [s[:state_size].copy() for s in solver.storage.saved_states]
    return state_size, ramp_end_state, truth_da_states, truth_forecast_states


def _build_da_objects(
    args: argparse.Namespace,
    pert_wind_file: Path,
    *,
    mode: str,
    state_size: int,
    ramp_end_state: np.ndarray,
    truth_da_states: list,
):
    """Build DA solver, forward wrapper, controller (if any), background, and
    observation arrays for a single window starting from `ramp_end_state`."""
    from petsc4py import PETSc
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params
    from swe4dvar.control.augmented_control import ControlLayout
    from swe4dvar.forward.augmented_control import (
        ManningsBasisController,
        AugmentedForwardModelWrapper,
    )
    from swe4dvar.data_assimilation import (
        DiagonalCovariance,
        BlockDiagonalCovariance,
        PointObservationOperator,
        create_cost_function,
    )

    # DA-side problem identical to truth except IC is perturbed
    forcing_da = GriddedForcing(str(pert_wind_file), cartesian=True)
    prob_da, solver_da = _build_idealized_inlet_problem(args, nt=args.nt_da)
    prob_da.forcing = forcing_da
    forcing_da.set_V(solver_da.V_scalar)
    forcing_da.evaluate(prob_da.t)
    # Initialize from ramp-end state
    solver_da.u_n.x.array[:state_size] = ramp_end_state
    solver_da.u_n_old.x.array[:state_size] = ramp_end_state
    solver_da.u.x.array[:state_size] = ramp_end_state
    solver_da.u_n.x.scatter_forward()
    solver_da.u_n_old.x.scatter_forward()
    solver_da.u.x.scatter_forward()

    solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=50, relaxation_parameter=0.7,
        comm=solver_da.problem.mesh.comm, error_if_not_converged=False,
    )

    # Controller (only for augmented mode)
    controller = None
    theta_size = 0
    if mode == "augmented":
        controller = ManningsBasisController(
            basis_shape=tuple(args.basis_shape),
            basis_width_fraction=args.basis_width_fraction,
            n_bounds=tuple(args.n_bounds),
            reference_n=args.reference_n,
        )
        controller.bind(prob_da, solver_da)
        theta_size = controller.parameter_size()
        print(f"  Augmented theta_size = {theta_size} "
              f"(basis_shape={tuple(args.basis_shape)})", flush=True)

    # Layout
    layout = ControlLayout(
        state_size=state_size, theta_size=theta_size, comm=PETSc.COMM_SELF,
    )

    # Forward wrapper
    forward_model = AugmentedForwardModelWrapper(
        solver_da, prob_da, solver_params,
        control_layout=layout,
        parameter_controller=controller,
        t_start=prob_da.t,
    )

    # Build perturbed background IC
    rng = np.random.default_rng(args.seed)
    state_scale = np.maximum(np.abs(ramp_end_state), 1e-3)
    state_bg = ramp_end_state + rng.normal(
        0.0, args.background_error_std * state_scale, size=ramp_end_state.shape
    )
    theta_bg = np.zeros(theta_size, dtype=float) if theta_size > 0 else np.array([])
    m_background = layout.create_petsc_vec(u0=state_bg, theta=theta_bg)

    # Observations
    rng_obs = np.random.default_rng(args.seed + 1)
    coords = solver_da.problem.mesh.geometry.x
    n_pts = coords.shape[0]
    n_obs = max(1, int(n_pts * args.obs_fraction))
    obs_idx = rng_obs.choice(n_pts, size=n_obs, replace=False)
    obs_points = np.zeros((n_obs, 3))
    obs_points[:, : coords.shape[1]] = coords[obs_idx, :]
    obs_operator = PointObservationOperator(
        solver_da.V, obs_points, comm=solver_da.problem.mesh.comm,
    )
    obs_times = list(range(0, args.nt_da + 1, args.obs_frequency))
    print(f"  obs_points = {n_obs}, obs_times = {obs_times}", flush=True)

    # Synthetic observations from truth states + noise
    observations = []
    obs_cov = {}
    from swe4dvar.utils.compat import create_petsc_vector_from_map as _cvm
    for k_obs in obs_times:
        truth_state_k = truth_da_states[k_obs]
        # Apply observation operator to truth state
        truth_vec = _cvm(
            solver_da.V.dofmap.index_map, solver_da.V.dofmap.index_map_bs,
        )
        truth_vec.setArray(truth_state_k[: solver_da.V.dofmap.index_map.size_local
                                          * solver_da.V.dofmap.index_map_bs])
        truth_vec.assemble()
        h_obs_truth = obs_operator.forward(truth_vec)
        h_arr = h_obs_truth.getArray(readonly=True).copy()
        truth_vec.destroy()
        h_obs_truth.destroy()
        # Add noise
        noise_std = args.obs_noise_level * np.maximum(np.abs(h_arr), 1e-3)
        h_arr_noisy = h_arr + rng_obs.normal(0.0, noise_std)
        obs_vec = PETSc.Vec().createSeq(h_arr_noisy.size, comm=PETSc.COMM_SELF)
        obs_vec.setArray(h_arr_noisy)
        obs_vec.assemble()
        observations.append(obs_vec)
        obs_cov[k_obs] = DiagonalCovariance(
            PETSc.COMM_SELF, size=h_arr.size, variance=float(args.obs_noise_level**2),
        )

    # Background covariance
    state_var = (args.background_error_std * state_scale) ** 2
    B_state = DiagonalCovariance(
        PETSc.COMM_SELF, size=state_size, diagonal=state_var,
    )
    if theta_size > 0:
        theta_var = np.full(theta_size, args.theta_prior_std ** 2, dtype=float)
        B_theta = DiagonalCovariance(
            PETSc.COMM_SELF, size=theta_size, diagonal=theta_var,
        )
        background_cov = BlockDiagonalCovariance([B_state, B_theta], comm=PETSc.COMM_SELF)
    else:
        background_cov = B_state

    cost = create_cost_function(
        "4dvar",
        forward_model=forward_model,
        observation_operator=obs_operator,
        background_cov=background_cov,
        observation_cov=obs_cov,
        m_background=m_background,
        observations=observations,
        obs_times=obs_times,
    )
    cost_type = type(cost).__name__
    print(f"  cost function: {cost_type} "
          f"(theta_size={layout.theta_size})", flush=True)

    return {
        "forward_model": forward_model,
        "controller": controller,
        "layout": layout,
        "cost_function": cost,
        "m_background": m_background,
        "observations": observations,
        "obs_operator": obs_operator,
        "obs_times": obs_times,
        "state_bg": state_bg,
        "theta_bg": theta_bg,
        "solver_da": solver_da,
        "prob_da": prob_da,
        "solver_params": solver_params,
    }


def _solve_window(da: dict, args: argparse.Namespace) -> dict:
    """Run TAO BLMVM on the augmented (or state-only) cost; return analysis."""
    from petsc4py import PETSc
    from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper

    layout = da["layout"]
    cost = da["cost_function"]
    m_bg = da["m_background"]
    state_size = layout.state_size
    theta_size = layout.theta_size

    # Bounds: h component >= 0.01, theta unbounded for now
    lower = layout.create_petsc_vec(
        u0=np.full(state_size, -np.inf, dtype=float),
        theta=np.full(theta_size, -np.inf, dtype=float) if theta_size > 0 else np.array([]),
    )
    upper = layout.create_petsc_vec(
        u0=np.full(state_size, +np.inf, dtype=float),
        theta=np.full(theta_size, +np.inf, dtype=float) if theta_size > 0 else np.array([]),
    )
    optimizer = PETScTAOWrapper(
        cost,
        tao_type="blmvm",
        lower_bounds=lower, upper_bounds=upper,
        options={
            "max_iterations": args.max_iterations,
            "max_funcs": args.max_funcs,
            "gradient_tolerance": 1e-6,
            "cost_tolerance": 1e-8,
            "verbose": True,
            "line_search_initial_step": 1.0,
        },
    )
    t0 = time.time()
    m_an = optimizer.solve(m_bg.copy())
    elapsed = time.time() - t0
    analysis = layout.unpack_vec(m_an)
    return {
        "analysis_state": analysis.u0,
        "analysis_theta": analysis.theta,
        "elapsed_s": elapsed,
        "converged": bool(getattr(optimizer, "converged", False)),
        "iterations": int(getattr(optimizer, "iteration", 0)),
    }


def _propagate_forecast(
    args: argparse.Namespace,
    truth_wind_file: Path,
    state_size: int,
    initial_state: np.ndarray,
    *,
    controller=None,
    theta: Optional[np.ndarray] = None,
    t_start: float,
) -> list:
    """Propagate `initial_state` (and optional Manning theta) for nt_forecast
    timesteps using the truth wind. Returns list of trajectory states."""
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params

    forcing = GriddedForcing(str(truth_wind_file), cartesian=True)
    prob, solver = _build_idealized_inlet_problem(args, nt=args.nt_forecast)
    prob.forcing = forcing
    forcing.set_V(solver.V_scalar)

    # Apply Manning's-n if augmented mode
    if controller is not None and theta is not None and theta.size > 0:
        # Re-bind controller to fresh solver, apply theta
        controller.bind(prob, solver)
        controller.apply(prob, solver, theta)

    solver.u_n.x.array[:state_size] = initial_state
    solver.u_n_old.x.array[:state_size] = initial_state
    solver.u.x.array[:state_size] = initial_state
    solver.u_n.x.scatter_forward()
    solver.u_n_old.x.scatter_forward()
    solver.u.x.scatter_forward()
    prob.t = t_start
    forcing.evaluate(prob.t)

    params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=50, relaxation_parameter=0.7,
        comm=prob.mesh.comm, error_if_not_converged=False,
    )
    solver.storage.clear()
    solver.time_loop(
        solver_parameters=params, stations=np.array([[0.0, 0.0, 0.0]]),
        plot_every=9999, save_state=True, store_jacobians=False, enable_video=False,
    )
    return [s[:state_size].copy() for s in solver.storage.saved_states]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> int:
    args = parse_args()
    _serial_guard()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: wind files
    truth_wind_file, pert_wind_file = _generate_wind_files(args, args.output_dir)

    # Step 2: truth ramp + W0 + forecast (single source for everything)
    state_size, ramp_end_state, truth_da_states, truth_forecast_states = \
        _spin_up_and_truth(args, truth_wind_file)

    da_end_truth = truth_da_states[-1]
    forecast_end_truth = truth_forecast_states[-1]
    t_da_start = args.nt_ramp * args.dt
    t_forecast_start = (args.nt_ramp + args.nt_da) * args.dt

    modes = ["state_only", "augmented"] if args.mode == "both" else [args.mode]
    results = {}
    for mode in modes:
        print(f"\n========== MODE: {mode} ==========", flush=True)
        da = _build_da_objects(
            args, pert_wind_file, mode=mode, state_size=state_size,
            ramp_end_state=ramp_end_state, truth_da_states=truth_da_states,
        )
        bg_state_rmse = _rmse(da["state_bg"], ramp_end_state)
        print(f"  bg state RMSE vs truth IC: {bg_state_rmse:.5f}", flush=True)

        # Solve W0
        sol = _solve_window(da, args)
        an_state = sol["analysis_state"]
        an_theta = sol["analysis_theta"]
        an_state_rmse = _rmse(an_state, ramp_end_state)
        print(f"  analysis IC RMSE: {an_state_rmse:.5f}  (vs bg {bg_state_rmse:.5f})", flush=True)
        if an_theta.size > 0:
            print(f"  analysis theta = {an_theta}", flush=True)
            print(f"  ||theta|| = {np.linalg.norm(an_theta):.5f}", flush=True)

        # Manning field summary (if augmented)
        mannings_summary = None
        if da["controller"] is not None:
            field = da["controller"].evaluate_field(an_theta)
            mannings_summary = {
                "min": float(field.min()),
                "mean": float(field.mean()),
                "max": float(field.max()),
                "std": float(field.std()),
            }
            print(f"  Manning field: min={field.min():.4f} "
                  f"mean={field.mean():.4f} max={field.max():.4f}", flush=True)

        # Forecast: propagate analysis through nt_forecast steps
        print(f"  Propagating analysis through {args.nt_forecast} forecast steps...",
              flush=True)
        try:
            forecast_states = _propagate_forecast(
                args, truth_wind_file, state_size, an_state,
                controller=da["controller"],
                theta=an_theta if an_theta.size > 0 else None,
                t_start=t_forecast_start,
            )
            forecast_end = forecast_states[-1]
            forecast_rmse_steps = [
                _rmse(forecast_states[k], truth_forecast_states[k])
                for k in range(len(forecast_states))
            ]
            print(f"  forecast RMSE trajectory: " + ", ".join(
                f"{r:.4f}" for r in forecast_rmse_steps
            ), flush=True)
            forecast_end_rmse = forecast_rmse_steps[-1]
        except Exception as e:
            print(f"  forecast FAILED: {e}", flush=True)
            forecast_end_rmse = float("nan")
            forecast_rmse_steps = []

        results[mode] = {
            "bg_state_rmse": bg_state_rmse,
            "analysis_state_rmse": an_state_rmse,
            "analysis_theta": an_theta.tolist() if an_theta.size > 0 else [],
            "theta_norm": float(np.linalg.norm(an_theta)) if an_theta.size > 0 else 0.0,
            "mannings_field": mannings_summary,
            "forecast_rmse_per_step": forecast_rmse_steps,
            "forecast_end_rmse": forecast_end_rmse,
            "elapsed_s": sol["elapsed_s"],
            "converged": sol["converged"],
            "iterations": sol["iterations"],
        }

    # Final comparison
    print("\n========== SUMMARY ==========", flush=True)
    print(f"{'metric':<28} {'state_only':>12} {'augmented':>12}")
    if "state_only" in results and "augmented" in results:
        for key in ("bg_state_rmse", "analysis_state_rmse",
                    "forecast_end_rmse", "elapsed_s", "iterations"):
            so = results["state_only"][key]
            au = results["augmented"][key]
            print(f"{key:<28} {so:>12.4f} {au:>12.4f}")
    else:
        for mode, r in results.items():
            print(f"  {mode}: {r}")

    out_path = args.output_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(
            {"args": {k: (v if not isinstance(v, Path) else str(v))
                      for k, v in vars(args).items()},
             "results": results},
            f, indent=2,
        )
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

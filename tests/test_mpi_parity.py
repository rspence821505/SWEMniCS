#!/usr/bin/env python3
"""
Serial vs MPI parity harness for idealized inlet 4D-Var.

Runs exactly ONE objective/gradient evaluation under the same nominal
state, observations, covariances, and bounds — then compares serial
and MPI results for numerical agreement.

Usage:
  # Serial baseline:
  python tests/test_mpi_parity.py

  # 2-rank MPI:
  PYTHONUNBUFFERED=1 mpirun -np 2 python tests/test_mpi_parity.py

  # Automated comparison (run both, diff JSON):
  python tests/test_mpi_parity.py --compare

The script writes structured JSON to results/mpi_parity/ for
machine-readable comparison.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CC", "/usr/bin/clang")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ================================================================
# Callback-phase diagnostic logger
# ================================================================
class PhaseLogger:
    """Rank-aware, structured callback-phase logger."""

    def __init__(self, rank: int, size: int, enabled: bool = True):
        self.rank = rank
        self.size = size
        self.enabled = enabled
        self.phases = []

    def log(self, phase: str, **data):
        if not self.enabled:
            return
        entry = {"rank": self.rank, "phase": phase, "time": time.time(), **data}
        self.phases.append(entry)
        msg = f"  [rank {self.rank}] {phase}"
        if data:
            kvs = ", ".join(f"{k}={v}" for k, v in data.items()
                            if k not in ("rank", "phase", "time"))
            if kvs:
                msg += f"  ({kvs})"
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()

    def to_dict(self):
        return {"rank": self.rank, "size": self.size, "phases": self.phases}


# ================================================================
# Instrumented value_gradient wrapper
# ================================================================
def instrumented_value_gradient(cost_fn, m, logger, save_vectors_path=None, smoother=None):
    """Run value_gradient with structured phase logging.

    If save_vectors_path is given, saves the adjoint λ₀ vector and the
    post-smoother gradient globally ordered by spatial coordinate, so
    serial and MPI runs can be compared vector-level after the fact.
    """

    logger.log("callback_entry", m_norm=float(m.norm()), m_local_size=m.getLocalSize())

    # Phase 1: Forward solve
    logger.log("pre_forward_solve")
    fwd_ok = True
    try:
        trajectory, jacobians = cost_fn._run_forward_model(m, store_jacobians=True)
        traj_len = len(trajectory) if trajectory else 0
        jac_len = len(jacobians) if jacobians else 0
        logger.log("post_forward_solve", traj_len=traj_len, jac_len=jac_len, success=True)
    except Exception as e:
        logger.log("post_forward_solve", success=False, error=str(e))
        fwd_ok = False

    if not fwd_ok:
        zero_grad = m.duplicate()
        zero_grad.zeroEntries()
        logger.log("callback_exit", cost=float("inf"), grad_norm=0.0, success=False)
        return float("inf"), zero_grad, False

    # Phase 2: Cost assembly
    logger.log("pre_cost_assembly")
    bg_term = cost_fn._compute_background_term(m)
    obs_term = cost_fn._compute_observation_term(trajectory)
    cost = bg_term + obs_term
    has_nan = not np.isfinite(cost)
    logger.log("post_cost_assembly",
               bg_term=float(bg_term), obs_term=float(obs_term),
               cost=float(cost), has_nan=has_nan)

    # Phase 3: Gradient assembly
    logger.log("pre_gradient_assembly")
    delta_m = m.duplicate()
    delta_m.waxpy(-1.0, cost_fn.m_b, m)
    grad_bg = cost_fn.B.apply_inverse(delta_m)
    grad_bg_norm = float(grad_bg.norm())
    logger.log("post_background_gradient", grad_bg_norm=grad_bg_norm)

    # Phase 4: Adjoint solve
    logger.log("pre_adjoint_solve")
    adj_ok = True
    try:
        lambda_0 = cost_fn._solve_adjoint(trajectory, jacobians)
        lambda_norm = float(lambda_0.norm())
        logger.log("post_adjoint_solve", lambda_norm=lambda_norm, success=True)
    except Exception as e:
        logger.log("post_adjoint_solve", success=False, error=str(e))
        adj_ok = False
        logger.log("callback_exit", cost=float(cost), grad_norm=grad_bg_norm, success=False)
        return cost, grad_bg, False

    # Phase 5: Gradient combination
    logger.log("pre_gradient_combine")
    grad = grad_bg.copy()
    grad.axpy(1.0, lambda_0)

    # Save pre-smoother adjoint vector (globally coord-ordered) for
    # vector-level serial/MPI comparison.
    if save_vectors_path is not None and smoother is not None:
        _save_global_ordered_vector(grad, smoother, f"{save_vectors_path}_lambda0", logger)

    # Phase 6: Gradient smoothing (supports both legacy callable and .apply-style)
    if cost_fn.gradient_smoother is not None:
        logger.log("pre_gradient_smooth")
        if hasattr(cost_fn.gradient_smoother, 'apply'):
            cost_fn.gradient_smoother.apply(grad)
        else:
            arr = grad.getArray().copy()
            arr = cost_fn.gradient_smoother(arr)
            grad.setArray(arr)
        logger.log("post_gradient_smooth")

    grad_norm = float(grad.norm())
    logger.log("callback_exit",
               cost=float(cost), grad_norm=grad_norm,
               bg_term=float(bg_term), obs_term=float(obs_term),
               grad_bg_norm=grad_bg_norm,
               success=True)

    # Save post-smoother gradient as well
    if save_vectors_path is not None and smoother is not None:
        _save_global_ordered_vector(grad, smoother, f"{save_vectors_path}_grad_smoothed", logger)

    return cost, grad, True


def _save_global_ordered_vector(vec, smoother, path_prefix, logger):
    """Save a distributed PETSc vec as global (coord-ordered) per-component arrays.

    Uses the DistributedGradientSmoother's allgather machinery:
    - coords are gathered in [rank0_h, rank1_h, ...] order
    - values are gathered in the same order
    - Uses h/u/v owned indices (position-paired)

    The result is rank-0 only arrays sorted by (x, y) coordinate —
    giving a canonical ordering independent of partitioning.
    """
    from mpi4py import MPI
    comm = smoother.comm

    owned_arr = vec.getArray()

    # Per-component owned values
    h_local = owned_arr[smoother.h_owned]
    u_local = owned_arr[smoother.u_owned]
    v_local = owned_arr[smoother.v_owned]

    # Use smoother's _allgather_component (already implemented)
    all_sizes = np.diff(np.concatenate([smoother.local_offsets, [smoother.global_n_h]]))
    all_sizes_int = all_sizes.astype(np.int64)
    h_global = smoother._allgather_component(h_local, all_sizes_int)
    u_global = smoother._allgather_component(u_local, all_sizes_int)
    v_global = smoother._allgather_component(v_local, all_sizes_int)

    # coords are already global in smoother.global_h_coords
    coords = smoother.global_h_coords

    if comm.Get_rank() == 0:
        # Sort by (x, y) for canonical ordering
        sort_idx = np.lexsort((coords[:, 1], coords[:, 0]))
        coords_sorted = coords[sort_idx]
        h_sorted = h_global[sort_idx]
        u_sorted = u_global[sort_idx]
        v_sorted = v_global[sort_idx]

        np.savez(
            f"{path_prefix}.npz",
            coords=coords_sorted,
            h=h_sorted, u=u_sorted, v=v_sorted,
        )
        logger.log(f"saved global vector: {path_prefix}.npz "
                   f"(n={len(coords_sorted)}, ||h||={np.linalg.norm(h_sorted):.4e}, "
                   f"||u||={np.linalg.norm(u_sorted):.4e}, "
                   f"||v||={np.linalg.norm(v_sorted):.4e})")


# ================================================================
# Build the DA problem (identical to idealized_inlet_da.py)
# ================================================================
def build_da_problem(comm, rank):
    """Build the full DA problem with deterministic, coordinate-based config.

    Uses spatial-coordinate-based perturbation (not random noise) so that
    serial and MPI produce identical backgrounds regardless of DOF ordering.
    """
    from petsc4py import PETSc
    from dolfinx import la

    from swe4dvar.forward.problems import IdealizedInlet
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params
    from swe4dvar.data_assimilation import (
        DiagonalCovariance, PointObservationOperator, create_cost_function,
    )
    from experiments.twin_experiment import (
        TwinExperiment, TwinExperimentConfig, ForwardModelWrapper,
    )
    from experiments.idealized_inlet_twin import (
        CartesianVortexConfig, generate_cartesian_vortex,
        write_cartesian_wind_hdf5, create_perturbed_config,
    )

    dt = 600.0
    nt_ramp = 2
    nt_da = 2
    nt_total = nt_ramp + nt_da
    times = np.arange(0, (nt_total + 1) * dt, dt)

    # Wind files (deterministic, cached)
    vortex_cfg = CartesianVortexConfig(Vmax=20.0, Rmax=15000.0, ramp_time_s=nt_ramp * dt)
    wind_dir = PROJECT_ROOT / "results" / "mpi_parity_wind"
    if rank == 0:
        wind_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    truth_file = wind_dir / "truth.h5"
    pert_file = wind_dir / "perturbed.h5"
    x_grid = np.linspace(-10000, 60000, 71)
    y_grid = np.linspace(-30000, 50000, 81)

    if not truth_file.exists() and rank == 0:
        wx, wy, p = generate_cartesian_vortex(vortex_cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(truth_file), x_grid, y_grid, times, wx, wy, p)
    pert_cfg = create_perturbed_config(vortex_cfg, 10.0)
    if not pert_file.exists() and rank == 0:
        wx, wy, p = generate_cartesian_vortex(pert_cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(pert_file), x_grid, y_grid, times, wx, wy, p)
    comm.Barrier()

    # Force direct LU solver for BOTH serial and MPI to eliminate
    # linear-solver-dependent trajectory differences. Without this,
    # serial uses GMRES+ILU and MPI uses MUMPS LU, giving ~1e-5
    # per-step differences that compound to ~9% cost discrepancy.
    #
    # SWE4DVAR_FORCE_MUMPS=1 forces MUMPS on BOTH serial and MPI to
    # isolate backend effect from distribution effect.
    extra_params = {"ksp_type": "preonly", "pc_type": "lu"}
    if os.environ.get("SWE4DVAR_FORCE_MUMPS", "0") == "1":
        extra_params["pc_factor_mat_solver_type"] = "mumps"
    solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=30, relaxation_parameter=0.7,
        comm=comm, error_if_not_converged=False,
        **extra_params,
    )
    if rank == 0:
        print(f"  solver_params: ksp_type={solver_params.get('ksp_type')}, "
              f"pc_type={solver_params.get('pc_type')}, "
              f"solver={solver_params.get('pc_factor_mat_solver_type', 'petsc-default')}",
              flush=True)

    # Truth problem
    forcing_truth = GriddedForcing(str(truth_file), cartesian=True)
    prob_truth = IdealizedInlet(
        dt=dt, nt=nt_total,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
        dramp=nt_ramp * dt / 86400.0, forcing=forcing_truth,
    )
    solver_truth = get_solver("DG")(prob_truth, theta=1.0, p_degree=[1, 1])
    state_size = solver_truth.V.dofmap.index_map.size_local * solver_truth.V.dofmap.index_map_bs

    # Ramp
    prob_truth.nt = nt_ramp
    solver_truth.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=False, store_jacobians=False, enable_video=False,
    )
    t_da_start = prob_truth.t
    ramp_end_state = solver_truth.u_n.x.array[:state_size].copy()

    # Truth trajectory
    solver_truth.storage.clear()
    prob_truth.nt = nt_da
    solver_truth.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=True, store_jacobians=False, enable_video=False,
    )
    m_true_arr = ramp_end_state
    truth_trajectory = []
    for s in solver_truth.storage.saved_states:
        vec = la.create_petsc_vector(solver_truth.V.dofmap.index_map, solver_truth.V.dofmap.index_map_bs)
        vec.setArray(s[:state_size])
        vec.assemble()
        truth_trajectory.append(vec)

    solver_truth.storage.clear()
    del solver_truth, prob_truth, forcing_truth
    gc.collect()

    # DA problem
    forcing_da = GriddedForcing(str(pert_file), cartesian=True)
    prob_da = IdealizedInlet(
        dt=dt, nt=nt_da,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
        dramp=nt_ramp * dt / 86400.0, forcing=forcing_da,
    )
    solver_da = get_solver("DG")(prob_da, theta=1.0, p_degree=[1, 1])
    state_size_da = solver_da.V.dofmap.index_map.size_local * solver_da.V.dofmap.index_map_bs

    solver_da.u_n.x.array[:state_size_da] = ramp_end_state[:state_size_da]
    solver_da.u_n_old.x.array[:state_size_da] = ramp_end_state[:state_size_da]
    solver_da.u.x.array[:state_size_da] = ramp_end_state[:state_size_da]
    solver_da.u_n.x.scatter_forward()
    solver_da.u_n_old.x.scatter_forward()
    solver_da.u.x.scatter_forward()
    prob_da.t = t_da_start

    # Twin experiment setup
    config = TwinExperimentConfig(
        method="4dvar", obs_fraction=0.1, obs_frequency=1,
        obs_noise_level=0.01, interior_only=True,
        background_error_std=0.02, background_correlation_length=500.0,
        component_aware_cov=True, max_iterations=2, max_funcs=5,
        gradient_tolerance=1e-3, cost_tolerance=1e-4,
        use_bounds=True, h_min=0.01,
        verbose=(rank == 0), obs_seed=42, background_seed=123,
    )
    exp = TwinExperiment(
        problem=prob_da, solver=solver_da, config=config,
        solver_params=solver_params, comm=comm,
    )
    exp.truth_trajectory = truth_trajectory
    m_true = la.create_petsc_vector(solver_da.V.dofmap.index_map, solver_da.V.dofmap.index_map_bs)
    m_true.setArray(m_true_arr[:state_size_da])
    m_true.assemble()
    exp.m_true = m_true
    exp.t_da_start = t_da_start

    obs_points, obs_operator, obs_times = exp._setup_observations()
    exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)

    # Use a COORDINATE-BASED deterministic perturbation instead of
    # random noise. This produces identical backgrounds regardless of
    # DOF ordering or mesh partitioning, enabling exact serial/MPI parity.
    #
    # We first call _setup_background() to initialize m_background (needed
    # for covariance setup), then OVERWRITE it with the deterministic one.
    background_error = exp._setup_background()

    # Get h-DOF coordinates using the same logic as _build_smoothing_matrix
    V = solver_da.V
    h_sub = V.sub(0)
    h_space_collapsed, h_map = h_sub.collapse()
    all_h_coords = h_space_collapsed.tabulate_dof_coordinates()[:, :2]

    # Map from parent h-indices to collapsed indices for coordinate lookup
    parent_to_collapsed = np.full(max(h_map) + 1, -1, dtype=int)
    for ci, pi in enumerate(h_map):
        parent_to_collapsed[pi] = ci

    h_indices_temp, u_indices_temp, v_indices_temp = exp._get_component_dof_indices(owned_only=True)
    h_collapsed_idx = parent_to_collapsed[h_indices_temp]
    h_coords = all_h_coords[h_collapsed_idx]  # (n_h_local, 2)

    Lx, Ly = 50000.0, 40000.0
    bg_std = 0.02

    true_arr = exp.m_true.getArray().copy()
    bg_arr = true_arr.copy()

    # h perturbation: smooth spatial mode
    h_pert = bg_std * 5.0 * np.cos(2 * np.pi * h_coords[:, 0] / Lx) * \
             np.sin(2 * np.pi * h_coords[:, 1] / Ly)
    bg_arr[h_indices_temp] += h_pert

    # u,v perturbation: smaller amplitude, different mode
    # u and v DOFs share the same spatial locations as h DOFs
    uv_pert = bg_std * 0.5 * np.sin(4 * np.pi * h_coords[:, 0] / Lx) * \
              np.cos(2 * np.pi * h_coords[:, 1] / Ly)
    bg_arr[u_indices_temp] += uv_pert
    bg_arr[v_indices_temp] += uv_pert

    # Clip h >= h_min
    bg_arr[h_indices_temp] = np.maximum(bg_arr[h_indices_temp], 0.01)

    exp.m_background.setArray(bg_arr)
    exp.m_background.assemble()

    # Recompute background RMSE with deterministic perturbation
    diff = bg_arr - true_arr
    local_sse = float(np.sum(diff ** 2))
    local_n = len(diff)
    global_sse = comm.allreduce(local_sse)
    global_n = comm.allreduce(local_n)
    background_error = float(np.sqrt(global_sse / global_n))

    B, R, B_lwme = exp._setup_covariances(obs_operator, obs_noise_stds)

    h_indices, u_indices, v_indices = exp._get_component_dof_indices(owned_only=True)

    # Memory guard before building the distributed smoother.
    # At L=500m on ~35K h-DOFs, the matrix has ~165M nnz / ~2 GB per rank.
    import resource, platform
    import gc as _gc
    def _rss_mb():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024) if platform.system() == "Darwin" else rss / 1024

    pre_mb = _rss_mb()
    n_h = len(h_indices)
    # New smoother: (n_owned_h x n_global_h) matrix with density ~ pi*(3L)^2 / domain_area.
    # At L=200m: ~400 nnz/row * n_h rows. 12 bytes/nnz.
    L_test = 200.0
    domain_area = 50000 * 40000  # rough
    nnz_per_row = np.pi * (3 * L_test) ** 2 / domain_area * 69312  # total global DOFs
    est_mb_per_rank = 12 * n_h * nnz_per_row / 1e6
    if rank == 0:
        print(f"  [mem] before distributed smoother: {pre_mb:.0f} MB; est +{est_mb_per_rank:.0f} MB",
              flush=True)
    mem_limit_mb = float(os.environ.get("SWE4DVAR_MEM_LIMIT_GB", "6")) * 1024
    if pre_mb + est_mb_per_rank > mem_limit_mb:
        raise MemoryError(
            f"Rank {rank}: projected {pre_mb + est_mb_per_rank:.0f} MB > "
            f"limit {mem_limit_mb:.0f} MB. Reduce correlation_length or raise "
            f"SWE4DVAR_MEM_LIMIT_GB env var."
        )

    # Use the distributed ghost-aware smoother for MPI correctness.
    # L=200m keeps memory modest (~200 MB/rank) while still exercising MPI.
    from swe4dvar.utils.distributed_smoother import DistributedGradientSmoother
    gradient_smoother = DistributedGradientSmoother(
        solver_da, correlation_length=200.0, comm=comm,
    )
    _gc.collect()
    post_mb = _rss_mb()
    if rank == 0:
        print(f"  [mem] after distributed smoother: {post_mb:.0f} MB (delta {post_mb-pre_mb:+.0f})",
              flush=True)

    forward_model = ForwardModelWrapper(
        solver=solver_da, problem=prob_da,
        solver_params=solver_params, t_start=t_da_start,
    )
    cost_fn = create_cost_function(
        "4dvar",
        forward_model=forward_model,
        observation_operator=obs_operator,
        background_cov=B,
        observation_cov=R,
        m_background=exp.m_background,
        observations=exp.observations,
        obs_times=obs_times,
    )
    inner_cf = cost_fn
    while hasattr(inner_cf, 'base_cost'):
        inner_cf = inner_cf.base_cost
    inner_cf.gradient_smoother = gradient_smoother

    # Build bounds
    lower = exp.m_background.duplicate()
    upper = exp.m_background.duplicate()
    lower_arr = lower.getArray()
    upper_arr = upper.getArray()
    lower_arr[h_indices] = 0.01
    upper_arr[h_indices] = 1e10
    lower_arr[u_indices] = -1e10
    upper_arr[u_indices] = 1e10
    lower_arr[v_indices] = -1e10
    upper_arr[v_indices] = 1e10
    lower.setArray(lower_arr)
    lower.assemble()
    upper.setArray(upper_arr)
    upper.assemble()

    return {
        "cost_fn": cost_fn,
        "m_background": exp.m_background,
        "lower": lower,
        "upper": upper,
        "obs_operator": obs_operator,
        "obs_times": obs_times,
        "n_obs": obs_operator.get_num_observations(),
        "state_size_local": state_size_da,
        "h_indices": h_indices,
        "u_indices": u_indices,
        "v_indices": v_indices,
        "background_rmse": background_error,
        "gradient_smoother": gradient_smoother,
    }


# ================================================================
# Run the parity test
# ================================================================
def run_parity_test(args):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logger = PhaseLogger(rank, size)
    logger.log("harness_start", mpi_size=size)

    # Build problem
    logger.log("build_problem_start")
    da = build_da_problem(comm, rank)
    logger.log("build_problem_done",
               state_size_local=da["state_size_local"],
               n_obs=da["n_obs"],
               n_obs_times=len(da["obs_times"]),
               n_h=len(da["h_indices"]),
               n_u=len(da["u_indices"]),
               n_v=len(da["v_indices"]),
               bg_rmse=da["background_rmse"])

    # Check bounds
    lb_arr = da["lower"].getArray()
    ub_arr = da["upper"].getArray()
    m_arr = da["m_background"].getArray()
    n_active_lower = int(np.sum(m_arr <= lb_arr + 1e-12))
    n_active_upper = int(np.sum(m_arr >= ub_arr - 1e-12))
    logger.log("bounds_check",
               n_active_lower=n_active_lower,
               n_active_upper=n_active_upper,
               m_min=float(m_arr.min()),
               m_max=float(m_arr.max()))

    # Run instrumented evaluation
    comm.Barrier()
    logger.log("evaluation_start")
    t0 = time.time()
    # Tag output with run config
    run_tag = os.environ.get("SWE4DVAR_PARITY_TAG", "default")
    out_dir = PROJECT_ROOT / "results" / "mpi_parity"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    vec_prefix = str(out_dir / f"vec_{run_tag}_size{size}")

    cost, grad, success = instrumented_value_gradient(
        da["cost_fn"], da["m_background"], logger,
        save_vectors_path=vec_prefix,
        smoother=da["gradient_smoother"],
    )
    elapsed = time.time() - t0
    comm.Barrier()

    grad_norm = float(grad.norm()) if success else 0.0
    grad_local_size = grad.getLocalSize()

    # Check for NaN/Inf in gradient
    grad_arr = grad.getArray()
    grad_has_nan = bool(not np.all(np.isfinite(grad_arr)))
    grad_max = float(np.max(np.abs(grad_arr))) if len(grad_arr) > 0 else 0.0

    logger.log("evaluation_complete",
               cost=float(cost), grad_norm=grad_norm,
               grad_local_size=grad_local_size,
               grad_has_nan=grad_has_nan, grad_max=grad_max,
               elapsed_s=elapsed, success=success)

    # Build result dict
    result = {
        "mpi_size": size,
        "rank": rank,
        "state_size_local": da["state_size_local"],
        "n_obs": da["n_obs"],
        "obs_times": da["obs_times"],
        "background_rmse": da["background_rmse"],
        "cost": float(cost),
        "grad_norm": grad_norm,
        "grad_local_size": grad_local_size,
        "grad_has_nan": grad_has_nan,
        "grad_max": grad_max,
        "n_active_lower": n_active_lower,
        "n_active_upper": n_active_upper,
        "forward_success": success,
        "elapsed_s": elapsed,
        "phases": logger.to_dict(),
    }

    # Write JSON (rank 0 only for serial, all ranks for MPI)
    out_dir = PROJECT_ROOT / "results" / "mpi_parity"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    tag = f"serial" if size == 1 else f"mpi{size}_rank{rank}"
    out_file = out_dir / f"result_{tag}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.log("result_written", file=str(out_file))

    # Cleanup
    grad.destroy()
    da["lower"].destroy()
    da["upper"].destroy()

    comm.Barrier()
    logger.log("harness_done")

    return result


# ================================================================
# Comparison mode: run serial + MPI, diff results
# ================================================================
def run_comparison():
    """Run serial and 2-rank MPI, then compare results."""
    out_dir = PROJECT_ROOT / "results" / "mpi_parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).resolve())

    print("=" * 60)
    print("MPI PARITY HARNESS — AUTOMATED COMPARISON")
    print("=" * 60)

    # Serial
    print("\n--- Running serial baseline ---")
    t0 = time.time()
    ret = subprocess.run(
        ["python", script],
        capture_output=True, text=True, timeout=1800,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    serial_time = time.time() - t0
    if ret.returncode != 0:
        print(f"SERIAL FAILED (exit {ret.returncode})")
        print(ret.stderr[-2000:] if ret.stderr else "no stderr")
        return 1
    print(f"  Serial done in {serial_time:.0f}s")

    # MPI
    print("\n--- Running 2-rank MPI ---")
    t0 = time.time()
    ret = subprocess.run(
        ["mpirun", "-np", "2", "python", script],
        capture_output=True, text=True, timeout=1800,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    mpi_time = time.time() - t0
    if ret.returncode != 0:
        print(f"MPI FAILED (exit {ret.returncode})")
        print(ret.stderr[-2000:] if ret.stderr else "no stderr")
        return 1
    print(f"  MPI done in {mpi_time:.0f}s")

    # Load results
    serial_file = out_dir / "result_serial.json"
    mpi0_file = out_dir / "result_mpi2_rank0.json"
    mpi1_file = out_dir / "result_mpi2_rank1.json"

    if not serial_file.exists():
        print(f"ERROR: {serial_file} not found")
        return 1
    if not mpi0_file.exists():
        print(f"ERROR: {mpi0_file} not found")
        return 1

    with open(serial_file) as f:
        s = json.load(f)
    with open(mpi0_file) as f:
        m0 = json.load(f)
    with open(mpi1_file) as f:
        m1 = json.load(f)

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON: Serial vs MPI")
    print("=" * 60)

    checks = []

    def check(name, serial_val, mpi_val, rtol=1e-6, atol=1e-10):
        if isinstance(serial_val, float) and isinstance(mpi_val, float):
            diff = abs(serial_val - mpi_val)
            rel = diff / max(abs(serial_val), 1e-30)
            ok = rel < rtol or diff < atol
            status = "PASS" if ok else "FAIL"
            checks.append((name, status))
            print(f"  {status}: {name}")
            print(f"         serial={serial_val:.10e}, mpi={mpi_val:.10e}, "
                  f"rel_diff={rel:.2e}")
        elif isinstance(serial_val, bool) and isinstance(mpi_val, bool):
            ok = serial_val == mpi_val
            status = "PASS" if ok else "FAIL"
            checks.append((name, status))
            print(f"  {status}: {name}  serial={serial_val}, mpi={mpi_val}")
        elif isinstance(serial_val, int) and isinstance(mpi_val, int):
            ok = serial_val == mpi_val
            status = "PASS" if ok else "FAIL"
            checks.append((name, status))
            print(f"  {status}: {name}  serial={serial_val}, mpi={mpi_val}")
        else:
            print(f"  SKIP: {name}  (type mismatch)")

    check("cost", s["cost"], m0["cost"], rtol=1e-6)
    check("grad_norm", s["grad_norm"], m0["grad_norm"], rtol=1e-4)
    check("forward_success", s["forward_success"], m0["forward_success"])
    check("grad_has_nan", s["grad_has_nan"], m0["grad_has_nan"])
    check("n_obs", s["n_obs"], m0["n_obs"])

    # MPI rank consistency
    print(f"\n  MPI rank consistency:")
    check("cost (rank0 vs rank1)", m0["cost"], m1["cost"], rtol=1e-12)
    check("grad_norm (rank0 vs rank1)", m0["grad_norm"], m1["grad_norm"], rtol=1e-12)
    check("forward_success (rank0 vs rank1)", m0["forward_success"], m1["forward_success"])

    # Summary
    n_pass = sum(1 for _, st in checks if st == "PASS")
    n_fail = sum(1 for _, st in checks if st == "FAIL")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {n_pass} PASS, {n_fail} FAIL")
    if n_fail == 0:
        print("ALL PARITY CHECKS PASSED")
    else:
        print("PARITY FAILURES DETECTED:")
        for name, st in checks:
            if st == "FAIL":
                print(f"  - {name}")
    print(f"{'=' * 60}")

    # Write comparison summary
    summary = {
        "serial_cost": s["cost"],
        "mpi_cost": m0["cost"],
        "serial_grad_norm": s["grad_norm"],
        "mpi_grad_norm": m0["grad_norm"],
        "cost_rel_diff": abs(s["cost"] - m0["cost"]) / max(abs(s["cost"]), 1e-30),
        "grad_rel_diff": abs(s["grad_norm"] - m0["grad_norm"]) / max(abs(s["grad_norm"]), 1e-30),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "serial_time_s": serial_time,
        "mpi_time_s": mpi_time,
        "checks": {name: st for name, st in checks},
    }
    summary_file = out_dir / "comparison_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_file}")

    return 1 if n_fail > 0 else 0


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="MPI parity harness")
    parser.add_argument("--compare", action="store_true",
                        help="Run serial + MPI and compare results")
    args = parser.parse_args()

    if args.compare:
        return run_comparison()
    else:
        run_parity_test(args)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

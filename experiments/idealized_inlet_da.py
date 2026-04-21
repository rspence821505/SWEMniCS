#!/usr/bin/env python3
"""
Idealized Inlet Wind-Driven DA Experiment
==========================================

Full data-assimilation twin experiment on the idealized inlet:
  - Truth run with truth wind forcing
  - DA run with perturbed wind forcing (track shift = model error)
  - Synthetic observations from truth
  - 4D-Var or DC-WME optimization
  - Error reduction comparison

Architecture follows the Shinnecock wind experiment pattern
(run_comparison.py: _run_sub_experiment_wind), using TwinExperiment
for infrastructure while managing the two-problem (truth/DA) setup
manually.

Usage:
  # 4D-Var:
  python experiments/idealized_inlet_da.py --method 4dvar

  # DC-WME static:
  python experiments/idealized_inlet_da.py --method dcwme_static

  # Both:
  python experiments/idealized_inlet_da.py --method both

  # Custom config:
  python experiments/idealized_inlet_da.py --vmax 40 --track-shift 15 --obs-fraction 0.3
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CC", "/usr/bin/clang")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Memory safety (reuse from idealized_inlet_twin.py)
MEM_LIMIT_MB = 8 * 1024


def _get_process_memory_mb():
    """Current RSS in MB. Prefers psutil (current RSS); falls back to
    ru_maxrss (which is PEAK RSS on macOS and monotonic-non-decreasing,
    overly conservative for a per-iteration guard).
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource, platform
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024) if platform.system() == "Darwin" else rss / 1024
    except Exception:
        return None


def _check_memory(context="", limit_mb=MEM_LIMIT_MB):
    rss = _get_process_memory_mb()
    if rss is not None:
        print(f"  [mem] {context}: {rss:.0f} MB", flush=True)
        if rss > limit_mb:
            raise MemoryError(f"RSS {rss:.0f} MB > limit {limit_mb:.0f} MB at '{context}'")


def _estimate_and_guard(description, bytes_needed, limit_mb=MEM_LIMIT_MB):
    mb_needed = bytes_needed / 1e6
    current = _get_process_memory_mb() or 0
    projected = current + mb_needed
    print(f"  [mem] {description}: est. {mb_needed:.0f} MB "
          f"(current {current:.0f} MB, projected {projected:.0f} MB)")
    if projected > limit_mb:
        raise MemoryError(
            f"Allocation '{description}' would use {mb_needed:.0f} MB, "
            f"pushing RSS to ~{projected:.0f} MB (limit {limit_mb:.0f} MB). Aborting."
        )


def _cleanup():
    gc.collect()
    try:
        from petsc4py import PETSc; PETSc.garbage_cleanup()
    except Exception:
        pass
    gc.collect()


# Import wind generation from the validated forward experiment
from experiments.idealized_inlet_twin import (
    CartesianVortexConfig,
    generate_cartesian_vortex,
    write_cartesian_wind_hdf5,
    create_perturbed_config,
)


def run_single_method(args, method, l_wme_mode, output_dir):
    """Run a single DA method (4dvar or dcwme) on the idealized inlet."""
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import la

    from swe4dvar.forward.problems import IdealizedInlet
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params
    from swe4dvar.data_assimilation import (
        DiagonalCovariance,
        PointObservationOperator,
        create_cost_function,
    )
    from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper
    from experiments.twin_experiment import (
        TwinExperiment, TwinExperimentConfig, ForwardModelWrapper,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    dt = args.dt
    nt_ramp = args.nt_ramp
    nt_da = args.nt_da
    nt_total = nt_ramp + nt_da
    times = np.arange(0, (nt_total + 1) * dt, dt)

    print(f"\n{'='*60}")
    print(f"  Method: {method.upper()} ({l_wme_mode})")
    print(f"  Vmax={args.vmax}, track_shift={args.track_shift}km")
    print(f"  Ramp={nt_ramp}×{dt}s = {nt_ramp*dt/3600:.1f}h")
    print(f"  DA window={nt_da}×{dt}s = {nt_da*dt/3600:.1f}h")
    print(f"  Smoother L={args.smooth_length:.0f}m, bounds=BLMVM(h>=0.01)")
    print(f"{'='*60}")

    # ================================================================
    # Step 1: Generate wind files
    # ================================================================
    vortex_cfg = CartesianVortexConfig(
        Vmax=args.vmax,
        Rmax=args.rmax_km * 1000,
        ramp_time_s=nt_ramp * dt,
    )

    wind_dir = output_dir / "wind"
    if rank == 0:
        wind_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    truth_file = wind_dir / "truth.h5"
    pert_file = wind_dir / "perturbed.h5"

    x_grid = np.linspace(-10000, 60000, 71)
    y_grid = np.linspace(-30000, 50000, 81)

    if not truth_file.exists() and rank == 0:
        print("  Generating truth wind...")
        wx, wy, p = generate_cartesian_vortex(vortex_cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(truth_file), x_grid, y_grid, times, wx, wy, p)

    pert_cfg = create_perturbed_config(vortex_cfg, args.track_shift)
    if not pert_file.exists() and rank == 0:
        print(f"  Generating perturbed wind ({args.track_shift}km shift)...")
        wx, wy, p = generate_cartesian_vortex(pert_cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(pert_file), x_grid, y_grid, times, wx, wy, p)
    comm.Barrier()

    # ================================================================
    # Step 2: Build truth problem + run warm-up + truth trajectory
    # ================================================================
    print("\n--- Step 1: Truth problem ---")
    forcing_truth = GriddedForcing(str(truth_file), cartesian=True)
    prob_truth = IdealizedInlet(
        dt=dt, nt=nt_total,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
        dramp=nt_ramp * dt / 86400.0,
        forcing=forcing_truth,
    )
    solver_truth = get_solver("DG")(prob_truth, theta=1.0, p_degree=[1, 1])

    # Use iterative solver (GMRES+ILU) with increased KSP iterations.
    # The idealized inlet's 208K-DOF system needs more linear solver
    # iterations than the default, especially for optimizer trial states
    # that are further from equilibrium than the truth trajectory.
    # Under-relaxed Newton stabilizes convergence for trial states
    # that are far from equilibrium. Default 0.7 (matches serial baseline);
    # override via SWE4DVAR_NEWTON_RELAX env var if MPI background needs
    # tighter steps.
    relax = float(os.environ.get("SWE4DVAR_NEWTON_RELAX", "0.7"))
    if rank == 0:
        print(f"  Newton relaxation_parameter = {relax}")
    solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=50, relaxation_parameter=relax,
        comm=comm, error_if_not_converged=False,
        ksp_max_it=2000,
    )

    state_size = solver_truth.V.dofmap.index_map.size_local * solver_truth.V.dofmap.index_map_bs
    print(f"  State size: {state_size} DOFs")
    _check_memory("after truth problem construction")

    # Warm-up (ramp)
    print(f"\n--- Step 2: Warm-up ({nt_ramp} steps) ---")
    prob_truth.nt = nt_ramp
    solver_truth.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=False, store_jacobians=False, enable_video=False,
    )
    t_da_start = prob_truth.t
    print(f"  Warm-up done, t={t_da_start:.0f}s")

    # CRITICAL: Save ramp-end state BEFORE running DA window.
    # This is the initial condition for the DA window — both truth and DA
    # solvers start from this state. m_true for 4D-Var is this IC,
    # NOT the final state of the truth trajectory.
    ramp_end_state = solver_truth.u_n.x.array[:state_size].copy()

    # Truth trajectory (DA window). Store Jacobians ONLY for DC-WME (needed
    # for TLM Eq 38 inflation). Pure 4D-Var doesn't use them — skipping saves
    # ~1 GB on a 208K-DOF mesh (12 sparse Jacobians + duplicates).
    need_jacobians = (method == "dcwme")
    traj_bytes = (nt_da + 1) * state_size * 8
    _estimate_and_guard(f"truth trajectory ({nt_da+1} x {state_size} DOFs)", traj_bytes)
    print(f"\n--- Step 3: Truth trajectory ({nt_da} steps, "
          f"{'with' if need_jacobians else 'no'} Jacobians) ---")
    solver_truth.storage.clear()
    prob_truth.nt = nt_da
    solver_truth.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=True, store_jacobians=need_jacobians, enable_video=False,
    )

    # m_true is the ramp-end state (DA window initial condition), NOT the
    # final truth trajectory state. The optimizer adjusts the IC to minimize
    # misfit between forward model trajectory and observations.
    m_true_arr = ramp_end_state
    truth_trajectory = []
    for s in solver_truth.storage.saved_states:
        vec = la.create_petsc_vector(solver_truth.V.dofmap.index_map, solver_truth.V.dofmap.index_map_bs)
        vec.setArray(s[:state_size])
        vec.assemble()
        truth_trajectory.append(vec)

    # Deep-copy Jacobians for TLM Eq 38 (before truth solver is freed)
    truth_jacobians = None
    if need_jacobians and solver_truth.storage.saved_jacobians:
        truth_jacobians = [J.duplicate() for J in solver_truth.storage.saved_jacobians]
        print(f"  Truth: {len(truth_trajectory)} states, {len(truth_jacobians)} Jacobians")
    else:
        print(f"  Truth: {len(truth_trajectory)} states, no Jacobians")

    # Free the truth solver and its problem — we've extracted the trajectory.
    # This releases the truth mesh, sparse matrices, and storage (~1-2 GB).
    solver_truth.storage.clear()
    del solver_truth, prob_truth, forcing_truth
    _cleanup()
    _check_memory("after truth trajectory + truth solver freed")

    # ================================================================
    # Step 4: DA problem with perturbed wind
    # ================================================================
    print(f"\n--- Step 4: DA problem (perturbed wind) ---")
    forcing_da = GriddedForcing(str(pert_file), cartesian=True)
    prob_da = IdealizedInlet(
        dt=dt, nt=nt_da,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
        dramp=nt_ramp * dt / 86400.0,
        forcing=forcing_da,
    )
    solver_da = get_solver("DG")(prob_da, theta=1.0, p_degree=[1, 1])

    # Apply same depth floor to DA problem
    if args.min_depth > 5.0:
        h_b_arr = prob_da.h_b.x.array[:]
        h_b_arr[h_b_arr < args.min_depth] = args.min_depth
        prob_da.h_b.x.array[:] = h_b_arr
        prob_da.h_b.x.scatter_forward()

    # Initialize DA solver from ramp-end state (same IC as truth DA window)
    solver_da.u_n.x.array[:state_size] = ramp_end_state
    solver_da.u_n_old.x.array[:state_size] = ramp_end_state
    solver_da.u.x.array[:state_size] = ramp_end_state
    solver_da.u_n.x.scatter_forward()
    solver_da.u_n_old.x.scatter_forward()
    solver_da.u.x.scatter_forward()
    prob_da.t = t_da_start

    # ================================================================
    # Step 5: Create TwinExperiment for DA infrastructure
    # ================================================================
    print(f"\n--- Step 5: Setting up observations and background ---")

    config = TwinExperimentConfig(
        method="4dvar",  # Will be overridden for dcwme
        obs_fraction=args.obs_fraction,
        obs_frequency=args.obs_frequency,
        obs_noise_level=args.obs_noise_level,
        interior_only=True,
        background_error_std=args.background_error_std,
        background_correlation_length=500.0,
        component_aware_cov=True,
        max_iterations=args.max_iterations,
        max_funcs=30,
        gradient_tolerance=1e-3,
        cost_tolerance=1e-4,
        use_bounds=True,
        h_min=0.01,
        verbose=(rank == 0),
        obs_seed=42,
        background_seed=123,
    )

    exp = TwinExperiment(
        problem=prob_da, solver=solver_da, config=config,
        solver_params=solver_params, comm=comm,
    )

    # Inject truth (bypass _generate_truth which would run with DA wind)
    exp.truth_trajectory = truth_trajectory
    m_true = la.create_petsc_vector(solver_da.V.dofmap.index_map, solver_da.V.dofmap.index_map_bs)
    m_true.setArray(m_true_arr)
    m_true.assemble()
    exp.m_true = m_true
    exp.t_da_start = t_da_start

    # Setup observations from truth trajectory
    obs_points, obs_operator, obs_times = exp._setup_observations()
    exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
    background_error = exp._setup_background()

    # MPI-safe background override: _setup_background's per-rank random
    # noise + local-only smoother produces an MPI background that differs
    # from serial and can contain partition-boundary spots that break
    # Newton. Replace with a coord-based deterministic perturbation that
    # is identical on all ranks and calibrated to the same bg-RMSE
    # magnitude (~0.09) as the serial random bg.
    if comm.size > 1:
        h_indices_bg, u_indices_bg, v_indices_bg = exp._get_component_dof_indices(owned_only=True)
        h_sub_bg = solver_da.V.sub(0)
        h_space_bg, h_map_bg = h_sub_bg.collapse()
        h_map_arr_bg = np.asarray(h_map_bg, dtype=np.int64)
        v_to_h_bg = {int(v): i for i, v in enumerate(h_map_arr_bg)}
        all_h_coords_bg = h_space_bg.tabulate_dof_coordinates()[:, :2]

        true_arr_local = exp.m_true.getArray(readonly=True).copy()
        bg_arr = true_arr_local.copy()
        # Coordinates for owned h DOFs
        h_coords_bg = np.zeros((len(h_indices_bg), 2))
        for i, vd in enumerate(h_indices_bg):
            h_coords_bg[i] = all_h_coords_bg[v_to_h_bg[int(vd)]]

        Lx, Ly = 50000.0, 40000.0
        h_mean = abs(true_arr_local[h_indices_bg]).mean() + 1e-10
        # scale chosen to yield bg_rmse ~ 0.09 (matches serial 15-iter baseline)
        h_amp = 0.1 * h_mean  # ~1 m for h_mean ~ 10 m
        uv_amp = 0.02
        h_pert = h_amp * np.cos(3 * np.pi * h_coords_bg[:, 0] / Lx) * \
                 np.sin(5 * np.pi * h_coords_bg[:, 1] / Ly)
        uv_pert = uv_amp * np.sin(2 * np.pi * h_coords_bg[:, 0] / Lx) * \
                  np.cos(3 * np.pi * h_coords_bg[:, 1] / Ly)
        bg_arr[h_indices_bg] = np.maximum(true_arr_local[h_indices_bg] + h_pert, 0.5)
        bg_arr[u_indices_bg] = true_arr_local[u_indices_bg] + uv_pert
        bg_arr[v_indices_bg] = true_arr_local[v_indices_bg] + uv_pert
        exp.m_background.setArray(bg_arr)
        exp.m_background.assemble()

        # Recompute bg_rmse
        diff = bg_arr - true_arr_local
        local_sse = float(np.sum(diff ** 2))
        local_n = len(diff)
        global_sse = comm.allreduce(local_sse)
        global_n = comm.allreduce(local_n)
        background_error = float(np.sqrt(global_sse / max(global_n, 1)))
        if rank == 0:
            print(f"  [MPI bg override] deterministic coord-based perturbation, "
                  f"bg_rmse={background_error:.6f}")

    B, R, B_lwme = exp._setup_covariances(obs_operator, obs_noise_stds)

    n_obs = obs_operator.get_num_observations()
    print(f"  Obs points: {n_obs}")
    print(f"  Obs times: {obs_times}")
    print(f"  Background RMSE: {background_error:.6f}")
    _check_memory("after DA setup")

    # ================================================================
    # Step 6: Build gradient smoother
    # ================================================================
    smooth_L = args.smooth_length
    print(f"\n--- Step 6: Gradient smoother (L={smooth_L:.0f}m) ---")
    h_indices, u_indices, v_indices = exp._get_component_dof_indices(owned_only=True)

    # Pre-flight nnz estimate: scipy CSR uses ~12 bytes/nnz (8 data + 4 col idx).
    # nnz per row ~ DOF density * pi * (3L)^2. Estimate density from mesh extent.
    h_space, _ = solver_da.V.sub(0).collapse()
    h_coords = h_space.tabulate_dof_coordinates()[:, :2]
    n_h = len(h_coords)
    x_extent = float(np.ptp(h_coords[:, 0]))
    y_extent = float(np.ptp(h_coords[:, 1]))
    domain_area = max(x_extent * y_extent, 1.0)
    dof_density = n_h / domain_area
    radius = 3.0 * smooth_L
    nnz_per_row = min(n_h, dof_density * np.pi * radius * radius)
    est_nnz = n_h * nnz_per_row
    est_bytes = 12 * est_nnz
    _estimate_and_guard(
        f"smoothing matrix L={smooth_L:.0f}m "
        f"(n_h={n_h}, ~{nnz_per_row:.0f} nnz/row, ~{est_nnz/1e6:.0f}M nnz)",
        est_bytes,
    )
    smoothing_matrix = exp._build_smoothing_matrix(h_indices, smooth_L)
    _cleanup()
    _check_memory("after gradient smoother build")

    def gradient_smoother(grad_array):
        smoothed = grad_array.copy()
        smoothed[h_indices] = smoothing_matrix @ grad_array[h_indices]
        smoothed[u_indices] = smoothing_matrix @ grad_array[u_indices]
        smoothed[v_indices] = smoothing_matrix @ grad_array[v_indices]
        return smoothed

    # ================================================================
    # Step 7: Build cost function
    # ================================================================
    print(f"\n--- Step 7: Cost function ({method.upper()}) ---")
    forward_model = ForwardModelWrapper(
        solver=solver_da, problem=prob_da,
        solver_params=solver_params, t_start=t_da_start,
    )

    if method == "dcwme" and l_wme_mode == "static":
        from experiments.shinnecock_study.run_comparison import (
            _compute_eq38_from_tlm,
            _apply_eq38_to_B,
            _compute_static_L_wme,
        )

        # Step 7a: TLM-based Eq 38 inflation
        # Uses the truth trajectory + Jacobians to compute the Gram matrix
        # G = J_wme^T J_wme over the full DA window, then derives
        # σ_b² ≥ γ / λ_min(G) and inflates B per-component.
        # Cost: 1 adjoint solve per observation point. For 1163 obs at
        # ~25 s/adjoint = ~8 hours. Skip with --skip-tlm-eq38 for fast
        # comparisons; falls back to default σ_b² without inflation.
        eq38_result = None
        if args.skip_tlm_eq38:
            print("  Step 7a: --skip-tlm-eq38 set — skipping TLM Eq 38 (using default σ_b²)")
            obs_variance = float(obs_noise_stds.mean() ** 2)
        elif truth_jacobians is not None:
            print("  Step 7a: Computing σ_b² from Eq 38 via TLM...")
            # Need a forward model wrapper for the WME QoI constructor
            gram_fwd = ForwardModelWrapper(
                solver=solver_da, problem=prob_da,
                solver_params=solver_params, t_start=t_da_start,
            )
            obs_variance = float(obs_noise_stds.mean() ** 2)

            eq38_result = _compute_eq38_from_tlm(
                forward_model=gram_fwd,
                obs_operator=obs_operator,
                obs_cov=R,
                m_linearize=m_true,
                observations=exp.observations,
                obs_times=obs_times,
                truth_trajectory=truth_trajectory,
                truth_jacobians=truth_jacobians,
                predictability_gamma=0.1,
                comm=comm, rank=rank,
            )
            _apply_eq38_to_B(B, eq38_result, rank=rank)
            _check_memory("after TLM Eq 38")
        else:
            print("  Step 7a: No truth Jacobians — skipping TLM Eq 38")
            obs_variance = float(obs_noise_stds.mean() ** 2)

        # Step 7b: Static L_wme (skip internal inflation since TLM already inflated B)
        already_inflated = eq38_result is not None
        print(f"  Step 7b: Computing static L_wme (skip_eq38_inflation={already_inflated}, "
              f"obs_correlation_length={args.obs_correlation_length:.0f} m, "
              f"predictability_gamma={args.predictability_gamma})...")
        static_lwme, lwme_diag = _compute_static_L_wme(
            obs_operator, B, len(obs_times), obs_variance,
            m_true, predictability_gamma=args.predictability_gamma,
            adaptive_gamma=True, comm=comm, rank=rank,
            skip_eq38_inflation=already_inflated,
            obs_correlation_length=args.obs_correlation_length,
        )
        _check_memory("after static L_wme")

        cost_fn = create_cost_function(
            "dc_wme",
            forward_model=forward_model,
            observation_operator=obs_operator,
            background_cov=B,
            observation_cov=R,
            m_background=exp.m_background,
            observations=exp.observations,
            obs_times=obs_times,
            predicted_cov_wme=static_lwme,
            n_l_wme_samples=0,
            auto_inflate_B=False,
        )
    else:
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

    # Attach gradient smoother
    inner = cost_fn
    while hasattr(inner, 'base_cost'):
        inner = inner.base_cost
    inner.gradient_smoother = gradient_smoother

    # ================================================================
    # Step 8: Optimize (bounded L-BFGS with h >= h_min)
    # ================================================================
    print(f"\n--- Step 8: Bounded L-BFGS optimization ---")
    _check_memory("before optimization")

    # Build box bounds: h DOFs >= h_min, u/v unconstrained
    lower = exp.m_background.duplicate()
    upper = exp.m_background.duplicate()
    lower_arr = lower.getArray()
    upper_arr = upper.getArray()
    h_min_bound = 0.01
    lower_arr[h_indices] = h_min_bound
    upper_arr[h_indices] = 1e10
    lower_arr[u_indices] = -1e10
    upper_arr[u_indices] = 1e10
    lower_arr[v_indices] = -1e10
    upper_arr[v_indices] = 1e10
    lower.setArray(lower_arr)
    lower.assemble()
    upper.setArray(upper_arr)
    upper.assemble()
    print(f"  Bounds: h >= {h_min_bound}, u/v unconstrained")

    max_funcs = args.max_funcs if args.max_funcs is not None else 2 * args.max_iterations + 10
    print(f"  max_iterations={args.max_iterations}, max_funcs={max_funcs}")

    # Iteration history: record (cost, grad_norm, RMSE-from-truth) per iteration.
    # RMSE uses allreduce for MPI correctness.
    history = {"iter": [], "cost": [], "grad_norm": [], "rmse_from_truth": []}
    truth_arr_cached = m_true.getArray(readonly=True).copy()

    def iteration_callback(x, iteration, f, gnorm):
        x_arr = x.getArray(readonly=True)
        local_sse = float(np.sum((x_arr - truth_arr_cached) ** 2))
        local_n = len(x_arr)
        global_sse = comm.allreduce(local_sse)
        global_n = comm.allreduce(local_n)
        rmse = float(np.sqrt(global_sse / max(global_n, 1)))
        history["iter"].append(int(iteration))
        history["cost"].append(float(f))
        history["grad_norm"].append(float(gnorm))
        history["rmse_from_truth"].append(rmse)

        # Memory safety: PETSc + Python GC between iterations to release
        # transient trajectory/Jacobian objects, then check RSS against
        # a HARD limit. If exceeded, abort cleanly instead of swap-thrashing.
        _cleanup()
        rss = _get_process_memory_mb()
        hard_limit = MEM_LIMIT_MB  # from env --mem-limit-gb
        if rank == 0:
            print(f"  [iter {iteration}] cost={f:.4f}  ||grad||={gnorm:.4e}  "
                  f"RMSE_truth={rmse:.6f}  RSS={rss:.0f} MB", flush=True)
        if rss is not None and rss > hard_limit:
            msg = (f"Rank {rank}: RSS {rss:.0f} MB exceeded limit "
                   f"{hard_limit:.0f} MB at iter {iteration}. Aborting to "
                   f"protect the system.")
            if rank == 0:
                print(f"  [ABORT] {msg}", flush=True)
            # Abort MPI cleanly rather than let the OS OOM-kill or swap-thrash
            comm.Abort(1)

    optimizer = PETScTAOWrapper(
        cost_fn, tao_type="blmvm",
        lower_bounds=lower,
        upper_bounds=upper,
        options={
            "max_iterations": args.max_iterations,
            "max_funcs": max_funcs,
            "gradient_tolerance": 1e-3,
            "cost_tolerance": 1e-4,
            "verbose": True,
            "iteration_callback": iteration_callback,
            # Cap per-step backtracks. Default 30 lets TAO spend an hour+
            # silently probing after a single forward failure (observed in
            # DC-WME runs v1/v2/v3 before LS_FAILURE would be raised).
            # With 5, any genuinely bad direction fails fast and control
            # returns to our 3-consecutive-failure bailout in the wrapper.
            "line_search_max_funcs": 5,
            # Memory-safety: cap BLMVM's limited-memory Hessian history.
            # Each history pair is 2 state vectors (208K × 8 B × 2 ≈ 3.3 MB
            # × 2 ranks), and PETSc internals duplicate these. Default is
            # 5 — we set 3 explicitly for safety on this mesh.
            "tao_lmvm_hist_size": 3,
            "tao_blmvm_hist_size": 3,
        },
    )

    t_opt = time.time()
    m_analysis = optimizer.solve(exp.m_background)
    opt_time = time.time() - t_opt

    # ================================================================
    # Step 9: Evaluate
    # ================================================================
    analysis_arr = m_analysis.getArray(readonly=True)
    truth_arr = m_true.getArray(readonly=True)
    bg_arr = exp.m_background.getArray(readonly=True)

    # MPI-correct global RMSE via allreduce (each rank holds only owned DOFs)
    def _global_rmse(a, b):
        local_sse = float(np.sum((a - b) ** 2))
        local_n = len(a)
        global_sse = comm.allreduce(local_sse)
        global_n = comm.allreduce(local_n)
        return float(np.sqrt(global_sse / max(global_n, 1)))

    bg_rmse = _global_rmse(bg_arr, truth_arr)
    analysis_rmse = _global_rmse(analysis_arr, truth_arr)
    improvement = (bg_rmse - analysis_rmse) / bg_rmse * 100

    print(f"\n{'='*60}")
    print(f"RESULTS: {method.upper()}")
    print(f"{'='*60}")
    print(f"  Background RMSE:  {bg_rmse:.6f}")
    print(f"  Analysis RMSE:    {analysis_rmse:.6f}")
    print(f"  Improvement:      {improvement:.1f}%")
    print(f"  Func evals:       {optimizer.n_func_evals}")
    print(f"  Optimization:     {opt_time:.0f}s")

    # Extract convergence info from optimizer
    try:
        conv_info = optimizer.get_convergence_info()
    except Exception:
        conv_info = {"converged": bool(optimizer.converged), "iteration": int(optimizer.iteration)}

    # Strip verbose fields from L_wme diagnostics before JSON dump (eigvalue
    # lists of size d_obs=1163 bloat the file); keep summary scalars + the
    # top/bottom 20 eigenvalues for inspection.
    lwme_diagnostics_summary = None
    if method == "dcwme" and 'lwme_diag' in locals() and lwme_diag is not None:
        evr = np.asarray(lwme_diag.get('eigvals_raw', []), dtype=float)
        lwme_diagnostics_summary = {
            "obs_correlation_length": lwme_diag.get('obs_correlation_length', 0.0),
            "sigma_b2": lwme_diag.get('sigma_b2'),
            "lambda_min_G": lwme_diag.get('lambda_min_G'),
            "lambda_max_G": lwme_diag.get('lambda_max_G'),
            "lambda_min_raw": lwme_diag.get('lambda_min_raw'),
            "lambda_max_raw": lwme_diag.get('lambda_max_raw'),
            "spectrum_ratio_raw": lwme_diag.get('spectrum_ratio_raw'),
            "inflation_factor": lwme_diag.get('inflation_factor'),
            "gamma_floor": lwme_diag.get('gamma_floor'),
            "n_natural": lwme_diag.get('n_natural'),
            "n_floored": lwme_diag.get('n_floored'),
            "raw_spectrum": lwme_diag.get('raw_spectrum'),
            "regularized_spectrum": lwme_diag.get('regularized_spectrum'),
            "eigvals_top20": sorted(evr.tolist(), reverse=True)[:20] if evr.size else [],
            "eigvals_bot20": sorted(evr.tolist())[:20] if evr.size else [],
        }

    result = {
        "method": method,
        "l_wme_mode": l_wme_mode,
        "bg_rmse": bg_rmse,
        "analysis_rmse": analysis_rmse,
        "improvement_pct": improvement,
        "n_func_evals": optimizer.n_func_evals,
        "opt_time_s": opt_time,
        "mpi_size": comm.size,
        "iteration_history": history,
        "convergence": conv_info,
        "lwme_diagnostics": lwme_diagnostics_summary,
        "config": {
            "vmax": args.vmax, "track_shift_km": args.track_shift,
            "nt_ramp": nt_ramp, "nt_da": nt_da, "dt": dt,
            "obs_fraction": args.obs_fraction,
            "obs_frequency": args.obs_frequency,
            "obs_noise_level": args.obs_noise_level,
            "background_error_std": args.background_error_std,
            "obs_correlation_length": args.obs_correlation_length,
            "predictability_gamma": args.predictability_gamma,
        },
    }

    safe_mode = l_wme_mode.replace("/", "_")
    tag = f"_Lcorr{int(args.obs_correlation_length):d}" if args.obs_correlation_length > 0 else ""
    if method == "dcwme" and abs(args.predictability_gamma - 0.1) > 1e-9:
        tag += f"_g{args.predictability_gamma:.3f}".rstrip('0').rstrip('.')
    result_file = output_dir / f"result_{method}_{safe_mode}{tag}.json"
    if rank == 0:
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
    print(f"  Saved: {result_file}")

    # Cleanup
    m_analysis.destroy()
    lower.destroy()
    upper.destroy()
    del cost_fn, optimizer, forward_model
    _cleanup()
    _check_memory("after cleanup")

    return result


def main():
    parser = argparse.ArgumentParser(description="Idealized Inlet DA Experiment")
    parser.add_argument("--method", default="4dvar",
                        choices=["4dvar", "dcwme_static", "both"])
    parser.add_argument("--vmax", type=float, default=30.0)
    parser.add_argument("--rmax-km", type=float, default=15.0)
    parser.add_argument("--track-shift", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--nt-ramp", type=int, default=24)
    parser.add_argument("--nt-da", type=int, default=12)
    parser.add_argument("--obs-fraction", type=float, default=0.1)
    parser.add_argument("--obs-frequency", type=int, default=4)
    parser.add_argument("--obs-noise-level", type=float, default=0.01)
    parser.add_argument("--background-error-std", type=float, default=0.02)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--max-funcs", type=int, default=None,
                        help="Max function evaluations. Default: 2*max_iterations+10.")
    parser.add_argument("--min-depth", type=float, default=5.0,
                        help="Minimum bathymetric depth (m). Default 5.0 matches IdealizedInlet. "
                             "Increase to 8-10 to prevent shallow-cell instability under wind.")
    parser.add_argument("--smooth-length", type=float, default=500.0,
                        help="Gradient smoother correlation length (m). Default 500. "
                             "Larger L blows up the sparse smoother (nnz ~ L^2): on idealized inlet, "
                             "L=500 -> ~2 GB, L=1000 -> ~8 GB, L=2000 -> ~32 GB. Stay <=700.")
    parser.add_argument("--skip-tlm-eq38", action="store_true",
                        help="Skip TLM-based Eq 38 inflation in DC-WME setup. "
                             "Eq 38 requires 1 adjoint solve per obs point (~8h for 1163 obs). "
                             "Falls back to default obs_variance for σ_b².")
    parser.add_argument("--obs-correlation-length", type=float, default=0.0,
                        help="If > 0, build DC-WME static L_wme from an obs-space Gaussian "
                             "kernel with this correlation length (m) — mathematically "
                             "equivalent to using a Gaussian-correlated B in the point-obs "
                             "limit (error O((cell_size/L)²)). Avoids building a dense "
                             "correlated B (infeasible at 208K DOFs). Background penalty "
                             "remains diagonal for both methods. Typical L: 1000-2000 m.")
    parser.add_argument("--predictability-gamma", type=float, default=0.1,
                        help="DC-WME spectral floor parameter γ (adaptive: floor = γ·λ_max). "
                             "Lower γ → less aggressive flooring → wider per-direction weight "
                             "range [1 − 1/γ·λ_max, 1 − 1/λ_max]. Default 0.1 collapses most "
                             "weights to ≈ 1 − 1/γ·λ_max on a correlated-B spectrum; try "
                             "0.01 or 0.001 to broaden the differential descent.")
    parser.add_argument("--mem-limit-gb", type=float, default=8.0)
    args = parser.parse_args()

    global MEM_LIMIT_MB
    MEM_LIMIT_MB = args.mem_limit_gb * 1024

    output_dir = PROJECT_ROOT / "results" / "idealized_inlet_da"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("IDEALIZED INLET DA EXPERIMENT")
    print("="*60)

    METHOD_MAP = {
        "4dvar": [("4dvar", "N/A")],
        "dcwme_static": [("dcwme", "static")],
        "both": [("4dvar", "N/A"), ("dcwme", "static")],
    }

    results = {}
    for method, l_wme_mode in METHOD_MAP[args.method]:
        result = run_single_method(args, method, l_wme_mode, output_dir)
        results[f"{method}_{l_wme_mode}"] = result

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        for key, r in results.items():
            print(f"  {key}: improvement={r['improvement_pct']:.1f}%, "
                  f"evals={r['n_func_evals']}, time={r['opt_time_s']:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())

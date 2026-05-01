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


def _check_memory(context="", limit_mb=None):
    # Resolve limit at call-time — MEM_LIMIT_MB is rebound in main() after
    # CLI parsing. Using a default arg would freeze the module-load value.
    if limit_mb is None:
        limit_mb = MEM_LIMIT_MB
    rss = _get_process_memory_mb()
    if rss is not None:
        print(f"  [mem] {context}: {rss:.0f} MB", flush=True)
        if rss > limit_mb:
            raise MemoryError(f"RSS {rss:.0f} MB > limit {limit_mb:.0f} MB at '{context}'")


def _estimate_and_guard(description, bytes_needed, limit_mb=None):
    if limit_mb is None:
        limit_mb = MEM_LIMIT_MB
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


def run_single_method(args, method, l_wme_mode, output_dir,
                      *, initial_bg_arr=None, truth_offset_steps=0,
                      window_tag="", advance_steps=0):
    """Run a single DA method (4dvar or dcwme) on the idealized inlet.

    Parameters
    ----------
    args, method, l_wme_mode, output_dir : standard
    initial_bg_arr : np.ndarray or None
        If provided, overrides the perturbed-truth background. Used by
        cycling DA to chain analysis from the previous window into the
        next window's background.
    truth_offset_steps : int
        Extra timesteps to march the truth (and DA) solvers BEFORE the
        DA window starts. Used by cycling DA so each window's truth
        starts at the correct cumulative time. Window N: offset = N*nt_da.
    window_tag : str
        Appended to result filenames. For cycling, distinguishes windows.
    advance_steps : int
        After Step 9 finishes, forward-evolve `m_analysis` through this
        many timesteps using the (still-alive) DA solver. The advanced
        state is returned as `_m_analysis_advanced_arr` for use as the
        next cycling window's background. Required for textbook cycling
        DA semantics — the analysis at this window's start time gets
        propagated through the dynamics to estimate the IC at the next
        window's start time. Default 0 (no advance).
    """
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
    n_windows = max(1, int(getattr(args, 'n_windows', 1)))
    # Total simulation time covers ramp + all DA windows so wind files
    # can be generated once for the whole cycling run. truth_offset_steps
    # is the cumulative offset for the CURRENT window (0 for window 0).
    nt_total_full = nt_ramp + n_windows * nt_da
    # The TRUTH solver in this single-window invocation runs:
    #   ramp (nt_ramp) → offset (truth_offset_steps) → DA window (nt_da)
    # Window 0: truth_offset_steps = 0
    # Window N: truth_offset_steps = N * nt_da (passed by main()'s cycling loop)
    nt_truth_pre_da = nt_ramp + int(truth_offset_steps)
    nt_total = nt_truth_pre_da + nt_da
    times = np.arange(0, (nt_total_full + 1) * dt, dt)

    print(f"\n{'='*60}")
    print(f"  Method: {method.upper()} ({l_wme_mode})")
    print(f"  Vmax={args.vmax}, track_shift={args.track_shift}km")
    print(f"  Ramp={nt_ramp}×{dt}s = {nt_ramp*dt/3600:.1f}h")
    print(f"  DA window={nt_da}×{dt}s = {nt_da*dt/3600:.1f}h")
    if n_windows > 1:
        print(f"  Cycling: n_windows={n_windows} "
              f"(total DA = {n_windows*nt_da*dt/3600:.1f}h)")
    print(f"  Smoother L={args.smooth_length:.0f}m, bounds=BLMVM(h>=0.01)")
    print(f"{'='*60}")

    # ================================================================
    # Step 1: Generate wind files
    # ================================================================
    # track_duration_s is set EXPLICITLY (not left as the legacy "spans
    # times array" default) so the storm has the same absolute trajectory
    # regardless of n_windows / nt_total. Without this, longer simulations
    # silently slow the storm down → different truth wind at the same
    # physical t → bg differs and (at certain n_windows) the truth Newton
    # diverges at the first DA-window timestep.
    #
    # Default chosen to match the validated 4-window cycling case
    # (nt_ramp=24, nt_da=6, n_windows=4 → 48 timesteps × dt=600s = 28800 s).
    # Override via --track-duration-s when running studies that need a
    # different storm passage speed.
    track_duration_s = float(getattr(args, "track_duration_s", 0.0)) or 28800.0
    vortex_cfg = CartesianVortexConfig(
        Vmax=args.vmax,
        Rmax=args.rmax_km * 1000,
        ramp_time_s=nt_ramp * dt,
        track_duration_s=track_duration_s,
        track_start_x=float(args.track_start_x),
        track_start_y=float(args.track_start_y),
        track_end_x=float(args.track_end_x),
        track_end_y=float(args.track_end_y),
    )

    wind_dir = output_dir / "wind"
    if rank == 0:
        wind_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    # Wind files are keyed by the parameters that affect their content:
    # vmax, track-shift, ramp duration, total simulation time. This
    # prevents the silent reuse-with-stale-parameters bug previously seen
    # when --track-shift was changed but cached wind from a prior run was
    # reused (e.g. cycling tests run with track-shift=0 read a cached
    # track-shift=10 file, masking model-error-free behavior).
    _ts_tag = f"ts{int(round(args.track_shift * 10)):03d}"  # e.g. "ts100" for 10km, "ts000" for 0km
    _v_tag = f"v{int(round(args.vmax))}"
    _t_tag = f"r{nt_ramp}n{n_windows*nt_da}"
    # Storm-track absolute duration tag — included so cached wind files from
    # the pre-fix "track scales with simulation length" code (which had no
    # _td tag) cannot be silently reused. After the fix, wind values for the
    # SAME physical t are independent of n_windows; the file name reflects
    # that by including the track-duration explicitly.
    _td_tag = f"td{int(round(track_duration_s))}"
    # Track-endpoints tag — also part of the cache key so a cached wind file
    # from a different track geometry can't be silently reused.
    _trk_tag = (f"sx{int(round(args.track_start_x/1000))}"
                f"sy{int(round(args.track_start_y/1000))}"
                f"ex{int(round(args.track_end_x/1000))}"
                f"ey{int(round(args.track_end_y/1000))}")
    _wind_tag = f"_{_v_tag}_{_ts_tag}_{_t_tag}_{_td_tag}_{_trk_tag}"
    truth_file = wind_dir / f"truth{_wind_tag}.h5"
    pert_file = wind_dir / f"perturbed{_wind_tag}.h5"

    x_grid = np.linspace(-10000, 60000, 71)
    y_grid = np.linspace(-30000, 50000, 81)

    if not truth_file.exists() and rank == 0:
        print(f"  Generating truth wind  → {truth_file.name}")
        wx, wy, p = generate_cartesian_vortex(vortex_cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(truth_file), x_grid, y_grid, times, wx, wy, p)

    pert_cfg = create_perturbed_config(vortex_cfg, args.track_shift)
    if not pert_file.exists() and rank == 0:
        print(f"  Generating perturbed wind ({args.track_shift}km shift)  → "
              f"{pert_file.name}")
        wx, wy, p = generate_cartesian_vortex(pert_cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(pert_file), x_grid, y_grid, times, wx, wy, p)
    comm.Barrier()

    # ================================================================
    # Step 2: Build truth problem + run warm-up + truth trajectory
    # ================================================================
    print("\n--- Step 1: Truth problem ---")
    forcing_truth = GriddedForcing(str(truth_file), cartesian=True)
    prob_truth = IdealizedInlet(
        dt=dt, nt=nt_total_full,
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

    # Warm-up (ramp + cycling offset). The offset advances the truth state
    # past previous windows so that this window's DA starts at the correct
    # cumulative time. truth_offset_steps = window_idx * nt_da (set by main).
    pre_da_label = (f"ramp+offset ({nt_ramp}+{int(truth_offset_steps)})"
                    if truth_offset_steps > 0 else f"ramp ({nt_ramp})")
    print(f"\n--- Step 2: Warm-up [{pre_da_label} steps] ---")
    prob_truth.nt = nt_truth_pre_da
    solver_truth.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=False, store_jacobians=False, enable_video=False,
    )
    t_da_start = prob_truth.t
    print(f"  Warm-up done, t={t_da_start:.0f}s "
          f"(window_tag={window_tag or 'single'})")

    # CRITICAL: Save ramp-end state BEFORE running DA window.
    # This is the initial condition for the DA window — both truth and DA
    # solvers start from this state. m_true for 4D-Var is this IC,
    # NOT the final state of the truth trajectory.
    ramp_end_state = solver_truth.u_n.x.array[:state_size].copy()

    # Truth trajectory (DA window). Store Jacobians ONLY for DC-WME (needed
    # for TLM Eq 38 inflation). Pure 4D-Var doesn't use them — skipping saves
    # ~1 GB on a 208K-DOF mesh (12 sparse Jacobians + duplicates). But since
    # the refactored Step 7a can apply Eq 38 inflation to 4D-Var too (for
    # matched-σ_b² comparisons), we also need them when 4D-Var asks for it
    # via --eq38-component-aware.
    _wants_tlm_gram = (
        (method == "dcwme" or getattr(args, "eq38_component_aware", False))
        and args.fixed_sigma_b_sq_h is None
        and not args.no_eq38_inflation
        and not args.skip_tlm_eq38
    )
    need_jacobians = _wants_tlm_gram
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
    from swe4dvar.utils.compat import create_petsc_vector_from_map as _cvm_
    for s in solver_truth.storage.saved_states:
        vec = _cvm_(solver_truth.V.dofmap.index_map, solver_truth.V.dofmap.index_map_bs)
        vec.setArray(s[:state_size])
        vec.assemble()
        truth_trajectory.append(vec)

    # Reference Jacobians directly from truth storage.
    # PETSc 3.25 (Vista conda) has an MPI_ERR_COMM bug where both .copy() and
    # .duplicate(copy=True) produce matrices whose communicator becomes invalid
    # after the source solver is freed — subsequent MatCreateVecs() calls fail.
    # Workaround: keep truth solver alive until after TLM Gram is computed.
    # Memory cost: ~1-2 GB retained through TLM phase (acceptable).
    # See docs/idealized_inlet_jacobian_handoff_trace.md.
    truth_jacobians = None
    if need_jacobians and solver_truth.storage.saved_jacobians:
        truth_jacobians = list(solver_truth.storage.saved_jacobians)
        print(f"  Truth: {len(truth_trajectory)} states, {len(truth_jacobians)} Jacobians (aliased)")
        # Truth solver retained until after Step 7a (TLM Gram) to preserve
        # Jacobian MPI comms. Freed explicitly at end of Step 7a.
        _cleanup()
        _check_memory("after truth trajectory (truth solver retained for TLM)")
    else:
        print(f"  Truth: {len(truth_trajectory)} states, no Jacobians")
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
    m_true = _cvm_(solver_da.V.dofmap.index_map, solver_da.V.dofmap.index_map_bs)
    m_true.setArray(m_true_arr)
    m_true.assemble()
    exp.m_true = m_true
    exp.t_da_start = t_da_start

    # Setup observations from truth trajectory
    obs_points, obs_operator, obs_times = exp._setup_observations()
    exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
    background_error = exp._setup_background()

    # Cycling override: replace the perturbed-truth bg with an externally-
    # supplied vector (the analysis from the previous window). Computed
    # AFTER _setup_background so that exp.m_background exists with the
    # correct PETSc layout, then we just overwrite its values.
    if initial_bg_arr is not None:
        cur_arr = exp.m_background.getArray(readonly=True)
        if cur_arr.shape != initial_bg_arr.shape:
            raise ValueError(
                f"initial_bg_arr shape {initial_bg_arr.shape} does not match "
                f"exp.m_background shape {cur_arr.shape} on rank {rank}"
            )
        exp.m_background.setArray(np.asarray(initial_bg_arr, dtype=np.float64))
        exp.m_background.assemble()
        # Recompute bg_rmse against THIS window's truth
        true_arr_local2 = exp.m_true.getArray(readonly=True).copy()
        diff2 = np.asarray(initial_bg_arr) - true_arr_local2
        local_sse2 = float(np.sum(diff2 ** 2))
        local_n2 = len(diff2)
        global_sse2 = comm.allreduce(local_sse2)
        global_n2 = comm.allreduce(local_n2)
        background_error = float(np.sqrt(global_sse2 / max(global_n2, 1)))
        if rank == 0:
            print(f"  [cycling override] bg = previous-window analysis  "
                  f"bg_rmse vs this-window truth = {background_error:.6f}")

    # MPI-safe background override: _setup_background's per-rank random
    # noise + local-only smoother produces an MPI background that differs
    # from serial and can contain partition-boundary spots that break
    # Newton. Replace with a coord-based deterministic perturbation that
    # is identical on all ranks and calibrated to the same bg-RMSE
    # magnitude (~0.09) as the serial random bg.
    # Cycling override (initial_bg_arr) takes precedence — skip MPI override.
    if comm.size > 1 and initial_bg_arr is None:
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

    # Cycling diagnostics dump: when running in cycling mode, save the
    # finalized bg vector + truth trajectory to disk so an offline forecast-
    # sensitivity probe can re-run forward integration from the same IC
    # without re-doing the optimization. Per-rank .npy files preserve the
    # partitioning so the probe can reload at the same np with no gather.
    if window_tag:
        dump_dir = output_dir / f"cycling_dump_{window_tag}"
        if rank == 0:
            dump_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()
        bg_arr_dump = exp.m_background.getArray(readonly=True).copy()
        true_arr_dump = exp.m_true.getArray(readonly=True).copy()
        np.save(dump_dir / f"bg_rank{rank}.npy", bg_arr_dump)
        np.save(dump_dir / f"truth_ic_rank{rank}.npy", true_arr_dump)
        # Truth trajectory at obs times (subset of saved_states)
        truth_states_arr = np.stack(
            [v.getArray(readonly=True).copy() for v in truth_trajectory],
            axis=0,
        )
        np.save(dump_dir / f"truth_traj_rank{rank}.npy", truth_states_arr)
        # Save metadata once on rank 0 (obs_times, dt, sizes, t_da_start)
        if rank == 0:
            import json as _json
            meta = {
                "window_tag": window_tag,
                "method": method,
                "t_da_start_seconds": float(t_da_start),
                "dt": float(dt),
                "nt_da": int(nt_da),
                "nt_ramp": int(nt_ramp),
                "truth_offset_steps": int(truth_offset_steps),
                "vmax": float(args.vmax),
                "track_shift_km": float(args.track_shift),
                "obs_fraction": float(args.obs_fraction),
                "obs_frequency": int(args.obs_frequency),
                "background_error_std": float(args.background_error_std),
                "comm_size": int(comm.size),
                "state_size_local_per_rank": "see truth_ic_rank{R}.npy shape",
                "obs_times_steps": [int(t) for t in obs_times],
            }
            with open(dump_dir / "meta.json", "w") as f:
                _json.dump(meta, f, indent=2)
            print(f"  [cycling-dump] wrote bg + truth to {dump_dir} "
                  f"(per-rank .npy)", flush=True)

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

    # ----------------------------------------------------------------
    # Step 7a: Background covariance B inflation (method-INDEPENDENT).
    # The same B inflation is applied whether the cost function is
    # 4D-Var or DC-WME. This lets us do apples-to-apples comparisons
    # where both methods see the same prior. See
    # docs/idealized_inlet_dcwme_exact_run_first_win_search.md.
    # ----------------------------------------------------------------
    from experiments.shinnecock_study.run_comparison import (
        _compute_eq38_from_tlm,
        _apply_eq38_to_B,
        _compute_static_L_wme,
    )

    eq38_result = None
    obs_variance = float(obs_noise_stds.mean() ** 2)

    if args.fixed_sigma_b_sq_h is not None and args.fixed_sigma_b_sq_uv is not None:
        # Fixed σ_b² bypass (TLM Gram skipped).
        import numpy as _np
        uv_owned = _np.concatenate([u_indices, v_indices])
        uv_owned.sort()
        component_indices = {"h": h_indices, "uv": uv_owned}
        eq38_result = {
            "sigma_b_sq": float(args.fixed_sigma_b_sq_h),
            "sigma_b_sq_h": float(args.fixed_sigma_b_sq_h),
            "sigma_b_sq_uv": float(args.fixed_sigma_b_sq_uv),
        }
        print(f"  Step 7a: fixed-σ_b² bypass — σ_b²_h={args.fixed_sigma_b_sq_h:.4e}, "
              f"σ_b²_uv={args.fixed_sigma_b_sq_uv:.4e} (TLM Gram SKIPPED) "
              f"[method={method.upper()}]",
              flush=True)
        _apply_eq38_to_B(B, eq38_result, rank=rank,
                         component_indices=component_indices)
        _check_memory("after fixed-σ_b² B inflation")
    elif args.no_eq38_inflation:
        print(f"  Step 7a: --no-eq38-inflation set — B is NOT inflated "
              f"[method={method.upper()}]")
    elif args.skip_tlm_eq38:
        print(f"  Step 7a: --skip-tlm-eq38 set — skipping TLM Gram "
              f"(H·H^T fallback active for DC-WME only) [method={method.upper()}]")
    elif truth_jacobians is not None:
        print(f"  Step 7a: Computing σ_b² from Eq 38 via TLM... [method={method.upper()}]")
        gram_fwd = ForwardModelWrapper(
            solver=solver_da, problem=prob_da,
            solver_params=solver_params, t_start=t_da_start,
        )
        component_indices = None
        if getattr(args, "eq38_component_aware", False):
            import numpy as _np
            uv_owned = _np.concatenate([u_indices, v_indices])
            uv_owned.sort()
            component_indices = {"h": h_indices, "uv": uv_owned}
            print(f"  Step 7a: component-aware Eq 38 enabled "
                  f"(h DOFs={len(h_indices)}, uv DOFs={len(uv_owned)})",
                  flush=True)
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
            component_indices=component_indices,
            component_degeneracy_policy=args.eq38_degeneracy_policy,
        )
        _apply_eq38_to_B(B, eq38_result, rank=rank,
                         component_indices=component_indices)
        _check_memory("after TLM Eq 38")

        # Free truth Jacobians + solver now that TLM Gram is complete.
        # Jacobians are the heavy retained objects (~2 GB each × 12); releasing
        # them now drops per-rank memory before the TAO optimization loop.
        # IMPORTANT: with Phase B's Jacobian pool, truth_jacobians ALIAS the
        # Mats stored in solver_truth.storage._jacobian_pool. Calling
        # _J.destroy() directly zombies the pool refs, which then deadlocks
        # PETSc.garbage_cleanup() in _cleanup() below. Route through
        # release_pool() (which destroys + clears the pool atomically) when
        # available; fall back to the legacy destroy loop in pre-pool mode.
        truth_jacobians = None
        try:
            if hasattr(solver_truth.storage, "release_pool"):
                _stats = solver_truth.storage.release_pool()
                if rank == 0:
                    print(f"  [pool] truth: released "
                          f"{_stats.get('jacobian_pool_destroyed', 0)} pool Jacobians",
                          flush=True)
            else:
                solver_truth.storage.clear()
        except Exception:
            pass
        del solver_truth, prob_truth, forcing_truth
        _cleanup()
        _check_memory("after truth solver freed post-TLM")
    else:
        print(f"  Step 7a: No truth Jacobians — skipping TLM Eq 38 "
              f"[method={method.upper()}]")

    # ----------------------------------------------------------------
    # Step 7b: Cost function construction.
    # DC-WME additionally builds a static L_wme term; 4D-Var just uses
    # the (possibly-inflated) B directly.
    # ----------------------------------------------------------------
    if method == "dcwme" and l_wme_mode == "static":
        skip_7b_inflation = (eq38_result is not None) or args.no_eq38_inflation
        _reason = ("B already inflated in Step 7a" if eq38_result is not None
                   else ("--no-eq38-inflation set" if args.no_eq38_inflation
                         else "default H·H^T-based inflation active"))
        print(f"  Step 7b: Computing static L_wme (skip_eq38_inflation={skip_7b_inflation} "
              f"[{_reason}], obs_correlation_length={args.obs_correlation_length:.0f} m, "
              f"predictability_gamma={args.predictability_gamma})...")
        static_lwme, lwme_diag = _compute_static_L_wme(
            obs_operator, B, len(obs_times), obs_variance,
            m_true, predictability_gamma=args.predictability_gamma,
            adaptive_gamma=True, comm=comm, rank=rank,
            skip_eq38_inflation=skip_7b_inflation,
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
    # h_min_bound: lower bound on analysis h DOFs. Tighter than the
    # forward-solver depth floor (args.min_depth) but loose enough that
    # the optimizer has freedom. Default 0.01 was permissive enough that
    # cycling DA (window N+1's bg = forward-evolved analysis from N) hit
    # Newton failures because near-dry cells went negative during the
    # integration. Raised to 1.0 to keep analysis safely wet.
    h_min_bound = float(getattr(args, 'h_min_bound', 1.0))
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

    # Best-iterate checkpoint path. Written on every cost improvement so that
    # a hard-abort (MPI_Abort on Newton-fail hang, walltime timeout, OOM) still
    # leaves a recoverable JSON with the latest best iterate on disk. The
    # final result file at Step 9 supersedes this on clean exit.
    _safe_mode_ckpt = l_wme_mode.replace("/", "_")
    _tag_ckpt = f"_Lcorr{int(args.obs_correlation_length):d}" if args.obs_correlation_length > 0 else ""
    if method == "dcwme" and abs(args.predictability_gamma - 0.1) > 1e-9:
        _tag_ckpt += f"_g{args.predictability_gamma:.3f}".rstrip('0').rstrip('.')
    checkpoint_file = output_dir / f"result_{method}_{_safe_mode_ckpt}{_tag_ckpt}.best_checkpoint.json"
    if rank == 0:
        print(f"  Best-iterate checkpoint: {checkpoint_file}")

    # Iteration history: record (cost, grad_norm, RMSE-from-truth) per iteration.
    # RMSE uses allreduce for MPI correctness.
    history = {"iter": [], "cost": [], "grad_norm": [], "rmse_from_truth": []}
    truth_arr_cached = m_true.getArray(readonly=True).copy()

    # Best-iterate checkpoint: persist the best accepted iterate (minimum
    # cost) to a separate vector. When optimization diverges (Newton-fail
    # hang, walltime timeout, MPI_Abort from the TAO wrapper), the final
    # analysis uses this instead of TAO's last-tried-m — which may be the
    # failing iterate TAO was mid-backtracking from.
    best_state = {
        "cost": float("inf"),
        "iter": -1,
        "rmse": float("inf"),
        "gnorm": float("inf"),
    }
    best_m = exp.m_background.duplicate()
    exp.m_background.copy(best_m)

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

        # Best-iterate checkpoint: any finite cost below current best wins.
        if np.isfinite(f) and f < best_state["cost"]:
            x.copy(best_m)
            best_state["cost"] = float(f)
            best_state["iter"] = int(iteration)
            best_state["rmse"] = float(rmse)
            best_state["gnorm"] = float(gnorm)
            # Persist to disk immediately — MPI_Abort (Newton-fail hang) or
            # walltime termination after this point still leaves the latest
            # best-iterate metadata recoverable. Rank 0 writes; no collective
            # operations here (would deadlock since other ranks skip the write).
            if rank == 0:
                try:
                    with open(checkpoint_file, "w") as _cf:
                        json.dump({
                            "method": method,
                            "l_wme_mode": l_wme_mode,
                            "best_iterate": {
                                "iter": int(iteration),
                                "cost": float(f),
                                "rmse": float(rmse),
                                "gnorm": float(gnorm),
                            },
                            "iteration_history_len": len(history["iter"]),
                            "status": "in_progress_checkpoint",
                        }, _cf, indent=2)
                except Exception as _ck_exc:
                    print(f"  [checkpoint] write failed: {_ck_exc}", flush=True)

        # Memory safety: PETSc + Python GC between iterations to release
        # transient trajectory/Jacobian objects, then check RSS against
        # a HARD limit. If exceeded, abort cleanly instead of swap-thrashing.
        _cleanup()
        rss = _get_process_memory_mb()
        hard_limit = MEM_LIMIT_MB  # from env --mem-limit-gb
        if rank == 0:
            print(f"  [iter {iteration}] cost={f:.4f}  ||grad||={gnorm:.4e}  "
                  f"RMSE_truth={rmse:.6f}  RSS={rss:.0f} MB  "
                  f"[best so far: iter {best_state['iter']} cost={best_state['cost']:.4f} "
                  f"RMSE={best_state['rmse']:.6f}]", flush=True)
        if rss is not None and rss > hard_limit:
            msg = (f"Rank {rank}: RSS {rss:.0f} MB exceeded limit "
                   f"{hard_limit:.0f} MB at iter {iteration}. Aborting to "
                   f"protect the system.")
            if rank == 0:
                print(f"  [ABORT] {msg}", flush=True)
            # Abort MPI cleanly rather than let the OS OOM-kill or swap-thrash
            comm.Abort(1)

    _tao_options = {
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
    }
    # Optional TAO globalization knobs (None → not forwarded → PETSc default).
    # See --ls-step-max / --ls-init-step help text for the rationale.
    if args.ls_step_max is not None:
        _tao_options["line_search_step_max"] = float(args.ls_step_max)
    if args.ls_init_step is not None:
        _tao_options["line_search_initial_step"] = float(args.ls_init_step)

    # Repeated-eval reproducer hook (gated). When SWE4DVAR_REPEAT_EVALS=N (N>0),
    # skip TAO and call cost_fn.value_gradient(m_background) N times in the same
    # process. Used together with SWE4DVAR_EVAL_MEM_DIAG=1 to classify the leak
    # as live PETSc state vs. allocator retention. Exits early with a stub
    # result so the cycling/aggregation code paths in main() still see a dict.
    _n_repeat = int(os.environ.get("SWE4DVAR_REPEAT_EVALS", "0"))
    if _n_repeat > 0:
        if rank == 0:
            print(f"\n[repeated-eval] SWE4DVAR_REPEAT_EVALS={_n_repeat} — skipping "
                  f"TAO; calling cost_fn.value_gradient(m_background) {_n_repeat}x "
                  f"in this process (same control vector each time)", flush=True)
        t_rep = time.time()
        bg_arr = exp.m_background.getArray(readonly=True).copy()
        for _i in range(_n_repeat):
            _t0 = time.time()
            _cost, _grad = cost_fn.value_gradient(exp.m_background)
            if rank == 0:
                print(f"[repeated-eval] eval {_i+1}/{_n_repeat} "
                      f"cost={float(_cost):.6e}  dt={time.time()-_t0:.1f}s",
                      flush=True)
            try:
                _grad.destroy()
            except Exception:
                pass
            # Keep the control vector pinned to its starting value so we measure
            # the lifecycle cost of repeated identical evals (no parameter drift).
            exp.m_background.setArray(bg_arr)
            exp.m_background.assemble()
        if rank == 0:
            print(f"[repeated-eval] done — {_n_repeat} evals in "
                  f"{time.time()-t_rep:.1f}s", flush=True)
        # Return a stub result; main() expects these keys for the cycling
        # aggregator. We mark improvement_pct = 0 because no DA was performed.
        return {
            "method": method, "l_wme_mode": l_wme_mode,
            "bg_rmse": float("nan"), "analysis_rmse": float("nan"),
            "improvement_pct": 0.0,
            "n_func_evals": _n_repeat,
            "opt_time_s": float(time.time() - t_rep),
            "best_iterate": None,
            "_m_analysis_advanced_arr": None,
        }

    optimizer = PETScTAOWrapper(
        cost_fn, tao_type="blmvm",
        lower_bounds=lower,
        upper_bounds=upper,
        options=_tao_options,
    )

    t_opt = time.time()
    m_analysis_tao = optimizer.solve(exp.m_background)
    opt_time = time.time() - t_opt

    # ================================================================
    # Step 9: Evaluate
    # ================================================================
    # Prefer the best-iterate checkpoint over TAO's returned iterate. TAO's
    # final m may be the point it was mid-backtracking from when we aborted
    # on Newton-fail hang; the best-iterate vector is always a cleanly-
    # accepted iterate with finite cost. When optimization ran to a clean
    # convergence on a descent path, best == TAO's final, so this is a
    # no-op. When TAO diverged, this preserves real work.
    if best_state["iter"] >= 0 and best_state["cost"] < float("inf"):
        m_analysis = best_m
        if rank == 0:
            print(f"  [analysis] using best-iterate checkpoint "
                  f"(iter {best_state['iter']}, cost={best_state['cost']:.4f}, "
                  f"RMSE={best_state['rmse']:.6f}) in place of TAO's final m",
                  flush=True)
    else:
        m_analysis = m_analysis_tao
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
        "best_iterate": {
            "iter": best_state["iter"],
            "cost": best_state["cost"],
            "rmse": best_state["rmse"],
            "gnorm": best_state["gnorm"],
            "used_as_analysis": bool(
                best_state["iter"] >= 0
                and best_state["cost"] < float("inf")
            ),
        },
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
            "smooth_length": args.smooth_length,
            "ls_step_max": args.ls_step_max,
            "ls_init_step": args.ls_init_step,
        },
    }

    safe_mode = l_wme_mode.replace("/", "_")
    tag = f"_Lcorr{int(args.obs_correlation_length):d}" if args.obs_correlation_length > 0 else ""
    if method == "dcwme" and abs(args.predictability_gamma - 0.1) > 1e-9:
        tag += f"_g{args.predictability_gamma:.3f}".rstrip('0').rstrip('.')
    if window_tag:
        tag += f"_{window_tag}"
    result_file = output_dir / f"result_{method}_{safe_mode}{tag}.json"
    # Stash the analysis array on the result dict for cycling — caller
    # (main's cycling loop) reads this to chain into the next window.
    result["_m_analysis_arr"] = analysis_arr.copy()
    result["_window_tag"] = window_tag
    if rank == 0:
        with open(result_file, "w") as f:
            json.dump({k: v for k, v in result.items() if not k.startswith("_")},
                      f, indent=2)
    print(f"  Saved: {result_file}")

    # Textbook cycling DA: forward-evolve m_analysis through `advance_steps`
    # using the (still-alive) DA solver. The result is the dynamics-respecting
    # estimate of the IC at the NEXT window's start, suitable as that
    # window's background. Without this, cycling would feed the analysis
    # (an estimate of the state at THIS window's start) directly to the
    # next window — biasing the bg by Δt of physical evolution.
    if advance_steps > 0:
        if rank == 0:
            print(f"\n--- Cycling: forward-evolve m_analysis "
                  f"{advance_steps} steps for next window's bg ---")
        analysis_full = m_analysis.getArray(readonly=True).copy()
        # Reset DA solver state to m_analysis at t_da_start
        solver_da.u_n.x.array[:state_size] = analysis_full[:state_size]
        solver_da.u_n_old.x.array[:state_size] = analysis_full[:state_size]
        solver_da.u.x.array[:state_size] = analysis_full[:state_size]
        solver_da.u_n.x.scatter_forward()
        solver_da.u_n_old.x.scatter_forward()
        solver_da.u.x.scatter_forward()
        prob_da.t = t_da_start
        prob_da.nt = advance_steps
        # Clear any stored data from cost-eval forward solves
        try:
            solver_da.storage.clear()
        except Exception:
            pass
        solver_da.time_loop(
            solver_parameters=solver_params, stations=[], plot_every=9999,
            save_state=False, store_jacobians=False, enable_video=False,
        )
        advanced_full = analysis_full.copy()
        advanced_full[:state_size] = solver_da.u_n.x.array[:state_size]
        result["_m_analysis_advanced_arr"] = advanced_full
        if rank == 0:
            print(f"  Forward-evolve done, t={prob_da.t:.0f}s "
                  f"(was t={t_da_start:.0f}s)")
    else:
        result["_m_analysis_advanced_arr"] = None

    # Cleanup — aggressive teardown to prevent memory accumulation across
    # cycling DA windows. Each call to run_single_method creates fresh
    # truth/DA solvers, observation operator, covariances, gradient smoother,
    # smoothing matrix, etc. Without explicit destruction of PETSc-owned
    # objects, the C-level memory leaks across iterations and OOM-kills
    # the job at window 3+.

    # m_analysis points at either best_m or m_analysis_tao. Destroy the one
    # that is NOT aliased via m_analysis, then destroy m_analysis itself.
    if m_analysis is best_m:
        try:
            m_analysis_tao.destroy()
        except Exception:
            pass
    else:
        try:
            best_m.destroy()
        except Exception:
            pass
    m_analysis.destroy()
    lower.destroy()
    upper.destroy()
    del cost_fn, optimizer, forward_model

    # Free truth-side state (large: 6+ Jacobians × ~100 MB/rank + trajectory
    # vectors). With cycling, this is per-window, so freeing it here
    # prevents the 30 GB/rank accumulation we saw at window 3.
    try:
        for v in truth_trajectory:
            try:
                v.destroy()
            except Exception:
                pass
        truth_trajectory.clear()
    except Exception:
        pass
    if truth_jacobians is not None:
        for j in truth_jacobians:
            try:
                j.destroy()
            except Exception:
                pass
        truth_jacobians.clear()
    try:
        m_true.destroy()
    except Exception:
        pass

    # Free DA-side state. solver_da and prob_da are no longer needed after
    # the optional forward-evolve advance step. exp (TwinExperiment) holds
    # references to many PETSc objects (m_background, observations, etc.).
    try:
        if 'B' in locals() and B is not None:
            B.destroy() if hasattr(B, 'destroy') else None
    except Exception:
        pass
    try:
        if 'R' in locals() and R is not None:
            R.destroy() if hasattr(R, 'destroy') else None
    except Exception:
        pass
    try:
        if 'B_lwme' in locals() and B_lwme is not None:
            B_lwme.destroy() if hasattr(B_lwme, 'destroy') else None
    except Exception:
        pass
    try:
        for v in getattr(exp, 'observations', []) or []:
            try:
                v.destroy()
            except Exception:
                pass
    except Exception:
        pass
    # Drop heavy attributes on exp before del
    for _attr in ("m_true", "m_background", "truth_trajectory",
                  "observations", "obs_operator"):
        try:
            setattr(exp, _attr, None)
        except Exception:
            pass

    # Release the persistent Jacobian pool (Phase B). Without this, the
    # ~6 stored Jacobian Mats per window survive across windows in a
    # single-process cycling run and accumulate ~720 MB/rank/window.
    try:
        if hasattr(solver_da, 'storage'):
            _stats = solver_da.storage.release_pool()
            if rank == 0 and _stats.get("jacobian_pool_destroyed", 0) > 0:
                print(f"  [pool] solver_da: destroyed "
                      f"{_stats['jacobian_pool_destroyed']} pool Jacobians",
                      flush=True)
    except Exception:
        pass
    try:
        if hasattr(solver_truth, 'storage'):
            _stats = solver_truth.storage.release_pool()
            if rank == 0 and _stats.get("jacobian_pool_destroyed", 0) > 0:
                print(f"  [pool] solver_truth: destroyed "
                      f"{_stats['jacobian_pool_destroyed']} pool Jacobians",
                      flush=True)
    except Exception:
        pass

    # Drop the solvers themselves and their problems. solver_da/solver_truth
    # hold mesh + function-space + Jacobian factor matrices.
    try: del solver_da
    except (NameError, UnboundLocalError): pass
    try: del solver_truth
    except (NameError, UnboundLocalError): pass
    try: del prob_da
    except (NameError, UnboundLocalError): pass
    try: del prob_truth
    except (NameError, UnboundLocalError): pass
    try: del forcing_da
    except (NameError, UnboundLocalError): pass
    try: del forcing_truth
    except (NameError, UnboundLocalError): pass
    try: del exp
    except (NameError, UnboundLocalError): pass
    try: del obs_operator
    except (NameError, UnboundLocalError): pass
    try: del smoothing_matrix
    except (NameError, UnboundLocalError): pass
    try: del gradient_smoother
    except (NameError, UnboundLocalError): pass

    # PETSc-level garbage collection (newer PETSc only)
    try:
        from petsc4py import PETSc
        PETSc.garbage_cleanup()
    except Exception:
        pass

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
    parser.add_argument("--track-duration-s", type=float, default=0.0,
                        help="Absolute duration (s) for the vortex track from "
                             "start to end position. 0 (default) → use the "
                             "in-driver fallback of 28800s, which matches the "
                             "validated 4-window cycling case. Set explicitly "
                             "to keep the SAME physical wind across runs with "
                             "different n_windows / nt_total. Without this, "
                             "the storm crossed the domain at different speeds "
                             "for different simulation lengths, which produced "
                             "n_windows-specific truth Newton divergence in "
                             "single-process cycling runs.")
    # Track endpoints in METERS. Default to the legacy CartesianVortexConfig
    # values (40, -20) → (25, 50) km. Override to put landfall (storm
    # crossing y=0) at a chosen window, e.g. (40, -20) → (10, 20) km with
    # track-duration-s=64800 puts landfall at the inlet at t=9h (mid-W2 of
    # a 3w/2h cycling run with nt_ramp=24).
    parser.add_argument("--track-start-x", type=float, default=40000.0,
                        help="Storm track start x in METERS (default 40000 = "
                             "40 km, far east). Inlet/coast is at y=0; "
                             "domain x ∈ [0, 50] km.")
    parser.add_argument("--track-start-y", type=float, default=-20000.0,
                        help="Storm track start y in METERS (default -20000 = "
                             "20 km south of coast / out at sea).")
    parser.add_argument("--track-end-x", type=float, default=25000.0,
                        help="Storm track end x in METERS (default 25000 = "
                             "mid-domain).")
    parser.add_argument("--track-end-y", type=float, default=50000.0,
                        help="Storm track end y in METERS (default 50000 = "
                             "50 km inland from coast). For landfall-at-W2 "
                             "experiments use 20000 (storm exits over the "
                             "shelf rather than crossing the whole domain).")
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--nt-ramp", type=int, default=24)
    parser.add_argument("--nt-da", type=int, default=12)
    parser.add_argument("--n-windows", type=int, default=1,
                        help="Number of cycling DA windows (default 1 = single window). "
                             "Window N+1 uses the analysis from window N (advanced "
                             "through nt_da steps via the DA solver) as its background. "
                             "Truth solver continues marching across windows. Total "
                             "simulation time = nt_ramp + n_windows * nt_da timesteps.")
    parser.add_argument("--start-window", type=int, default=0,
                        help="Index of first window in this job (0-based). Used to "
                             "chain cycling DA across separate SLURM jobs: each job "
                             "runs n_windows windows starting at start_window. "
                             "truth_offset_steps = start_window * nt_da, so the truth "
                             "solver picks up at the correct time. Combined with "
                             "--initial-bg-file to load the previous job's analysis.")
    parser.add_argument("--initial-bg-file", type=str, default=None,
                        help="Path template (with '{rank}') to .npy files containing "
                             "the chained background for window=start_window. Each "
                             "rank loads its own slice. Generated automatically by a "
                             "previous job at chain_bg_after_w{N}_rank{R}.npy.")
    parser.add_argument("--chain-mode", action="store_true",
                        help="Force the cycling-DA code path even when n_windows=1, "
                             "so that chain_bg_after_w{N}_rank{R}.npy files are "
                             "written for the next job in a multi-job chain. "
                             "Implied by --start-window>0 or --initial-bg-file.")
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
    parser.add_argument("--h-min-bound", type=float, default=1.0,
                        help="Lower bound on analysis h DOFs in TAO box "
                             "constraints (BLMVM projection). Default 1.0 m. "
                             "Loose enough for the optimizer to move freely "
                             "but tight enough to prevent near-dry cells "
                             "that crash Newton during cycling DA forward-"
                             "evolve. Set to 0.01 for original (looser) "
                             "behavior on single-window runs.")
    parser.add_argument("--smooth-length", type=float, default=500.0,
                        help="Gradient smoother correlation length (m). Default 500. "
                             "Larger L blows up the sparse smoother (nnz ~ L^2): on idealized inlet, "
                             "L=500 -> ~2 GB, L=1000 -> ~8 GB, L=2000 -> ~32 GB. Stay <=700.")
    parser.add_argument("--ls-step-max", type=float, default=None,
                        help="Hard cap on TAO line-search step length (plumbed to "
                             "PETSc -tao_ls_stepmax). Globalizes BLMVM when the "
                             "2nd+ iteration extrapolates into a region that crashes "
                             "forward Newton (idealized-inlet 4D-Var eval-3 failure "
                             "at obs >= 0.02). Default None (no cap). Try 1.0 or 0.5 "
                             "as a conservative starting point.")
    parser.add_argument("--ls-init-step", type=float, default=None,
                        help="Initial step length for TAO line search (PETSc's "
                             "setInitialStepLength). Default None (PETSc default, "
                             "typically 1.0). Lower values (0.25, 0.5) make each "
                             "TAO iteration start cautious; Armijo backtracks from "
                             "there.")
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
    parser.add_argument("--eq38-component-aware", action="store_true",
                        help="Run TLM Eq 38 with separate h / uv Grams and apply "
                             "per-component σ_b² bounds to B (instead of the "
                             "scalar bound applied regardless of component). No "
                             "extra adjoint solves — the per-component Grams are "
                             "built from the same adjoint vectors. See "
                             "experiments/shinnecock_study/run_comparison.py:"
                             "_compute_eq38_from_tlm.")
    parser.add_argument("--eq38-degeneracy-policy",
                        choices=["strict", "skip_unobservable"],
                        default="strict",
                        help="Per-component Eq 38 policy when a component Gram "
                             "(G_h or G_uv) is rank-deficient or structurally "
                             "unobservable. 'strict' (default) raises with the "
                             "raw eigenspectrum — catches e.g. h-only obs that "
                             "cannot bound σ_b²_uv. 'skip_unobservable' leaves "
                             "the degenerate component's block of B untouched "
                             "and proceeds (inflation applied only to the "
                             "well-posed component).")
    parser.add_argument("--no-eq38-inflation", action="store_true",
                        help="Disable ALL Eq 38 inflation (Step 7a TLM Gram AND "
                             "Step 7b H·H^T static fallback). DC-WME cost "
                             "structure is preserved: static L_wme is still "
                             "computed but without prior B inflation. For "
                             "mechanism isolation studies — see docs/"
                             "idealized_inlet_dcwme_exact_run_first_win_search.md.")
    parser.add_argument("--fixed-sigma-b-sq-h", type=float, default=None,
                        help="Skip TLM Eq 38 for the h block and directly apply "
                             "this per-DOF σ_b²_h bound via _apply_eq38_to_B. "
                             "Must be paired with --fixed-sigma-b-sq-uv.")
    parser.add_argument("--fixed-sigma-b-sq-uv", type=float, default=None,
                        help="Skip TLM Eq 38 for the uv block and directly apply "
                             "this per-DOF σ_b²_uv bound via _apply_eq38_to_B. "
                             "Must be paired with --fixed-sigma-b-sq-h.")
    parser.add_argument("--mem-limit-gb", type=float, default=64.0)
    args = parser.parse_args()

    # Fixed-sigma mode: if either flag is provided, require both and skip TLM Eq 38.
    _fs_h = args.fixed_sigma_b_sq_h
    _fs_uv = args.fixed_sigma_b_sq_uv
    if (_fs_h is None) != (_fs_uv is None):
        parser.error("--fixed-sigma-b-sq-h and --fixed-sigma-b-sq-uv must be "
                     "set together, or neither.")
    if _fs_h is not None:
        args.skip_tlm_eq38 = True  # bypass the expensive Gram path

    # --no-eq38-inflation also bypasses TLM Gram in Step 7a; Step 7b skip
    # is handled below when already_inflated is computed.
    if args.no_eq38_inflation:
        args.skip_tlm_eq38 = True
        if _fs_h is not None:
            parser.error("--no-eq38-inflation is mutually exclusive with "
                         "--fixed-sigma-b-sq-h/uv.")

    global MEM_LIMIT_MB
    MEM_LIMIT_MB = args.mem_limit_gb * 1024
    print(f"[mem-limit] args.mem_limit_gb={args.mem_limit_gb}  "
          f"MEM_LIMIT_MB={MEM_LIMIT_MB}")

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

    n_windows = max(1, int(getattr(args, 'n_windows', 1)))
    start_window = int(getattr(args, 'start_window', 0))
    initial_bg_file = getattr(args, 'initial_bg_file', None)
    chain_mode = bool(getattr(args, 'chain_mode', False))
    # Any of these implies chain semantics (cycling loop + chain_bg save).
    in_chain = chain_mode or start_window > 0 or initial_bg_file is not None or n_windows > 1

    results = {}
    for method, l_wme_mode in METHOD_MAP[args.method]:
        if not in_chain:
            # Standard single-window run (preserves existing behavior).
            result = run_single_method(args, method, l_wme_mode, output_dir)
            results[f"{method}_{l_wme_mode}"] = result
            continue

        # Cycling DA: run N windows in sequence, chaining each window's
        # m_analysis as the next window's background. Supports cross-job
        # chaining via --start-window/--initial-bg-file: each SLURM job runs
        # n_windows windows starting at global index `start_window`, exits
        # cleanly (so the OS reclaims memory), and writes the chained bg
        # to a per-rank .npy file for the next job to load.
        per_window = []
        from mpi4py import MPI as _MPI
        rank = _MPI.COMM_WORLD.Get_rank()

        # Window 0 of the chain uses the perturbed bg generated internally.
        # Subsequent windows of THIS job pull bg from the prior window's
        # in-memory `_m_analysis_advanced_arr`. When this job is itself a
        # mid-chain job (start_window > 0), the first window pulls bg from
        # the on-disk file written by the previous SLURM job.
        bg_arr_for_next = None
        if initial_bg_file:
            bg_path = initial_bg_file.format(rank=rank) if "{rank}" in initial_bg_file else initial_bg_file
            bg_arr_for_next = np.load(bg_path).astype(np.float64)
            print(f"[chain] rank {rank} loaded initial bg from {bg_path} "
                  f"shape={bg_arr_for_next.shape}", flush=True)
        elif start_window > 0:
            raise SystemExit(
                f"--start-window={start_window} requires --initial-bg-file "
                f"(no chained bg available for non-zero start window)")

        for local_w in range(n_windows):
            global_w = start_window + local_w
            tag = f"w{global_w}"
            # Always advance: this might be the last window of THIS job but
            # not the last of the cycling chain, so the chained bg must be
            # produced for the next job to consume.
            advance = int(args.nt_da)
            print(f"\n{'#'*70}\n# CYCLING WINDOW {global_w+1} (global w={global_w}, "
                  f"local {local_w+1}/{n_windows}) ({method}/{l_wme_mode}) "
                  f"tag={tag}\n{'#'*70}")
            r = run_single_method(
                args, method, l_wme_mode, output_dir,
                initial_bg_arr=bg_arr_for_next,
                truth_offset_steps=global_w * args.nt_da,
                window_tag=tag,
                advance_steps=advance,
            )
            per_window.append({
                "window": global_w + 1,
                "tag": tag,
                "bg_rmse": r["bg_rmse"],
                "analysis_rmse": r["analysis_rmse"],
                "improvement_pct": r["improvement_pct"],
                "n_func_evals": r["n_func_evals"],
                "opt_time_s": r["opt_time_s"],
                "best_iterate": r.get("best_iterate"),
            })
            # Persist the chained bg to disk (always — even on the last
            # window of this job, in case a follow-up job extends the chain).
            if r.get("_m_analysis_advanced_arr") is not None:
                chain_path = output_dir / f"chain_bg_after_w{global_w}_rank{rank}.npy"
                np.save(chain_path, np.asarray(r["_m_analysis_advanced_arr"], dtype=np.float64))
                if rank == 0:
                    print(f"[chain] saved chain bg after w{global_w}: "
                          f"{chain_path.name} (per-rank, "
                          f"{_MPI.COMM_WORLD.Get_size()} files)", flush=True)
                bg_arr_for_next = r["_m_analysis_advanced_arr"].copy()
            else:
                bg_arr_for_next = None

        # Aggregate JSON for the cycling run. Tag with absolute window range
        # so multi-job chains do not clobber each other's aggregates.
        end_window = start_window + n_windows - 1
        agg_file = (output_dir /
                    f"result_{method}_{l_wme_mode.replace('/', '_')}"
                    f"_cycling_w{start_window}-{end_window}.json")
        if _MPI.COMM_WORLD.Get_rank() == 0:
            with open(agg_file, "w") as f:
                json.dump({
                    "method": method,
                    "l_wme_mode": l_wme_mode,
                    "start_window": start_window,
                    "n_windows": n_windows,
                    "windows": per_window,
                }, f, indent=2)
            print(f"\n  Saved cycling aggregate: {agg_file}")
        results[f"{method}_{l_wme_mode}_cycling"] = {
            "start_window": start_window,
            "n_windows": n_windows, "windows": per_window
        }

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        for key, r in results.items():
            if "n_windows" in r:
                final = r["windows"][-1]
                print(f"  {key}: {r['n_windows']} windows, "
                      f"final imp={final['improvement_pct']:.1f}%")
            else:
                print(f"  {key}: improvement={r['improvement_pct']:.1f}%, "
                      f"evals={r['n_func_evals']}, time={r['opt_time_s']:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())

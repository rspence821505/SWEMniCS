#!/usr/bin/env python3
"""
Idealized Inlet QPCA-EnDCF Ensemble DA Experiment
==================================================

Twin-experiment ensemble filtering on the idealized-inlet SWE PDE using the
QPCA Ensemble Data Consistency Filter (4D variant), ported from
``QPCA-EnDCF-Paper/src/filters/qpca_endcf.py`` to operate on the FEniCSx/
PETSc SWE state instead of the original Lorenz-96-style ODE state.

Pipeline (mirrors ``experiments/idealized_inlet_da.py`` where possible):
  Step 1.  Generate truth and (perturbed) DA wind HDF5 files.
  Step 2.  Build truth solver, run warm-up + truth trajectory.
  Step 3.  Reuse ``TwinExperiment`` to set up observation points,
           noisy synthetic observations, perturbed background, and
           component-aware DOF indices.
  Step 4.  Build a single DA solver used to forecast each ensemble member.
  Step 5.  Sample an initial ensemble around the background using the
           same component magnitudes used for the perturbation.
  Step 6.  For each cycling window:
              - Propagate every ensemble member forward through
                ``nt_da = window_len * obs_frequency`` steps with
                ``save_state=True``.
              - Collect the per-member state at the L post-window obs
                times to assemble ``X_path`` (list of (n, N) arrays).
              - Build stacked observations ``z_stack`` from the matching
                synthetic obs vectors.
              - Call ``QPCAEnDCF.update`` to obtain the analysis ensemble
                at the window end.
              - The analysis becomes the prior for the next window.
  Step 7.  Diagnostics: background/analysis RMSE vs truth at each
           window end, mean obs misfit, per-window timing.

Assumptions / adaptation notes:
  * **MPI: spatial parallelism within each ensemble member.** Each rank
    holds its locally-owned slice of every member's state. The forward
    solve (`solver.time_loop`) and observation operator
    (`PointObservationOperator.forward`) are both collective and
    handle the MPI bookkeeping internally; ``apply_H`` returns the
    same ``(m, N)`` array on every rank because the obs operator
    Allgathers its result. The QPCA filter then runs on this
    replicated observation-space data on every rank, and the final
    state-space update ``A_x_end @ inner`` naturally restricts to each
    rank's owned DOFs (``inner`` is a small ``(N, N)`` matrix
    identical on every rank, so the multiplication is local). Run with
    e.g. ``mpirun -n 2 python ... --micro-smoke``. Ensemble members
    are *not* distributed across ranks — increasing rank count
    accelerates each forecast but does not parallelize over the
    ensemble dimension.
  * **State vector layout.** State arrays are the locally-owned portion
    of the mixed (h, (u, v)) DG function space, exactly as used by the
    4D-Var path. Component DOF indices (h / u / v) are obtained from the
    existing ``TwinExperiment._get_component_dof_indices`` so the
    ensemble construction matches ``_setup_background``'s scaling.
  * **Wet/dry handling.** Each ensemble draw is clipped to ``h >= h_min``
    to keep the SWE forward solve in the wet regime (same convention as
    ``_setup_background``).
  * **Observation operator.** Uses the existing
    ``PointObservationOperator``; we wrap it as a callable
    ``apply_H(X_ens)`` that loops over columns. The QPCA-EnDCF filter
    itself only sees the resulting ``(m, N)`` numpy array.
  * **Covariance.** ``R`` is built as a diagonal numpy matrix from the
    obs noise std used to generate the observations (identical to the
    diagonal observation covariance used by the 4D-Var path). The QPCA
    filter constructs its own block-diag ``R_block`` internally.

Usage:
  python experiments/idealized_inlet_qpca_endcf.py \
      --vmax 30 --track-shift 10 \
      --nt-ramp 24 --nt-da 12 --obs-frequency 4 \
      --ensemble-size 16 --window-len 3 --n-windows 1

Quick smoke test:
  python experiments/idealized_inlet_qpca_endcf.py --smoke
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


# ---------------------------------------------------------------------------
# Memory helpers (lightweight versions of the idealized_inlet_da ones).
# ---------------------------------------------------------------------------


def _cleanup():
    gc.collect()
    try:
        from petsc4py import PETSc
        PETSc.garbage_cleanup()
    except Exception:
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# Observation-operator bridge: PointObservationOperator → apply_H callable.
# ---------------------------------------------------------------------------


class _PointObsBridge:
    """Adapt a PointObservationOperator to the QPCAEnDCF apply_H contract.

    The filter wants a function ``apply_H(X_ens) -> HX`` where ``X_ens`` is
    a ``(n, N)`` numpy array (state DOFs × ensemble members) and ``HX`` is
    ``(m, N)``. ``PointObservationOperator.forward`` operates on a single
    PETSc state vector and returns a sequential PETSc obs vector, so we
    loop over columns and stack.

    Parameters
    ----------
    obs_operator : PointObservationOperator
        FEniCSx observation operator built against the SWE function space.
    template_vec : PETSc.Vec
        Template state vector (e.g. ``solver.u_n``-shaped) used to
        reconstruct a PETSc.Vec from each ensemble column.
    state_size : int
        Owned DOF count (rows of the ensemble array).
    """

    def __init__(self, obs_operator, template_vec, state_size: int):
        self.obs_operator = obs_operator
        self.template_vec = template_vec
        self.state_size = int(state_size)
        self.n_obs = int(obs_operator.get_num_observations())

    def __call__(self, X_ens: np.ndarray) -> np.ndarray:
        if X_ens.ndim != 2 or X_ens.shape[0] != self.state_size:
            raise ValueError(
                f"apply_H: X_ens has shape {X_ens.shape}, expected "
                f"({self.state_size}, N)"
            )
        N = X_ens.shape[1]
        HX = np.zeros((self.n_obs, N), dtype=float)
        for j in range(N):
            # ascontiguousarray — PETSc.Vec.setArray on a non-contiguous
            # column slice of a (n, N) array can produce stride-related
            # corruption or silent crashes. Force a contiguous copy.
            col = np.ascontiguousarray(X_ens[:, j], dtype=float)
            self.template_vec.setArray(col)
            self.template_vec.assemble()
            obs_j = self.obs_operator.forward(self.template_vec)
            HX[:, j] = obs_j.getArray()
            obs_j.destroy()
        return HX


# ---------------------------------------------------------------------------
# Ensemble initialization.
# ---------------------------------------------------------------------------


def _build_initial_ensemble(
    bg_arr: np.ndarray,
    h_indices: np.ndarray,
    u_indices: np.ndarray,
    v_indices: np.ndarray,
    h_std: float,
    uv_std: float,
    N: int,
    h_min: float,
    seed: int,
    rank: int = 0,
) -> np.ndarray:
    """Sample ``N`` ensemble members around the background.

    Members 0 is the background itself (gives the filter a sensible
    starting mean even at small N). Members 1..N-1 are
    ``bg + perturbation`` with per-component white Gaussian noise scaled
    by ``h_std``/``uv_std`` (the same magnitudes used by
    ``TwinExperiment._setup_background``).

    ``rank`` is folded into the seed so that under MPI each rank draws
    independent Gaussian noise for its owned DOFs — without this all
    ranks would consume the SAME prefix of the same RNG stream, which
    correlates the perturbation patterns across partitions.
    """
    rng = np.random.default_rng(seed + 100003 * int(rank))
    n = bg_arr.size
    X0 = np.zeros((n, N), dtype=float)
    bg_clipped = bg_arr.copy()
    bg_clipped[h_indices] = np.maximum(bg_clipped[h_indices], h_min)
    X0[:, 0] = bg_clipped
    for j in range(1, N):
        eps = np.zeros(n, dtype=float)
        eps[h_indices] = h_std * rng.standard_normal(len(h_indices))
        eps[u_indices] = uv_std * rng.standard_normal(len(u_indices))
        eps[v_indices] = uv_std * rng.standard_normal(len(v_indices))
        member = bg_arr + eps
        # Wet-cell clipping — matches _setup_background's hmin enforcement
        member[h_indices] = np.maximum(member[h_indices], h_min)
        X0[:, j] = member
    return X0


# ---------------------------------------------------------------------------
# Per-window ensemble forecast.
# ---------------------------------------------------------------------------


def _forecast_ensemble(
    X_ens_start: np.ndarray,
    *,
    solver_da,
    prob_da,
    solver_params,
    state_size: int,
    t_window_start: float,
    nt_window: int,
    obs_step_indices,
    h_indices: np.ndarray,
    h_min: float,
):
    """Forecast every ensemble member through one assimilation window.

    Returns
    -------
    X_path : list[np.ndarray]
        Length-L list of ``(state_size, N)`` arrays — the ensemble at each
        of the ``obs_step_indices`` post-window timesteps.
    n_failed : int
        Number of members whose forward solve raised (replayed as the
        starting state — same fail-safe used in the 4D-Var feasibility
        ladder, scoped to a single member).
    """
    n, N = X_ens_start.shape
    L = len(obs_step_indices)
    # Each obs_step_index is a count *within* the window
    # (1..nt_window). storage.saved_states starts at the IC (step 0),
    # so saved_states[k] is the state after k timesteps.
    X_path = [np.zeros((n, N), dtype=float) for _ in range(L)]
    n_failed = 0

    for j in range(N):
        # Reset state to member j, reset problem time.
        member = np.asarray(X_ens_start[:, j], dtype=float).copy()
        member[h_indices] = np.maximum(member[h_indices], h_min)

        solver_da.u_n.x.array[:state_size] = member
        solver_da.u_n_old.x.array[:state_size] = member
        solver_da.u.x.array[:state_size] = member
        solver_da.u_n.x.scatter_forward()
        solver_da.u_n_old.x.scatter_forward()
        solver_da.u.x.scatter_forward()
        prob_da.t = t_window_start
        prob_da.nt = nt_window
        try:
            solver_da.storage.clear()
        except Exception:
            pass

        ok = True
        try:
            solver_da.time_loop(
                solver_parameters=solver_params,
                stations=[], plot_every=9999,
                save_state=True, store_jacobians=False,
                enable_video=False,
            )
        except Exception as exc:
            ok = False
            print(f"  [warn] member {j} forward solve raised "
                  f"{type(exc).__name__}: {exc}; replaying starting state",
                  flush=True)

        saved = list(solver_da.storage.saved_states) if ok else []
        if not ok or len(saved) < nt_window + 1:
            # Forward diverged or didn't save the expected number of states
            # — replay the starting member at every obs step. This keeps the
            # ensemble shape consistent for the filter; the cost is one
            # contaminated column, but the QPCA gain stays well-defined.
            n_failed += 1
            for tk_idx, _ in enumerate(obs_step_indices):
                X_path[tk_idx][:, j] = member
            try:
                solver_da.storage.clear()
            except Exception:
                pass
            continue

        for tk_idx, step in enumerate(obs_step_indices):
            arr = np.asarray(saved[step][:state_size], dtype=float)
            X_path[tk_idx][:, j] = arr

        # Free trajectory before the next member's run.
        try:
            solver_da.storage.clear()
        except Exception:
            pass

    return X_path, n_failed


# ---------------------------------------------------------------------------
# Diagnostics helpers.
# ---------------------------------------------------------------------------


def _rmse(a: np.ndarray, b: np.ndarray, comm=None) -> float:
    """Root-mean-square error between two same-shape arrays.

    When ``comm`` is provided, the local SSE and DOF count are reduced
    with ``allreduce`` so the returned scalar reflects the GLOBAL RMSE
    across all ranks. With no ``comm`` this reduces to the serial form.
    """
    diff = a - b
    local_sse = float(np.sum(diff ** 2))
    local_n = int(len(diff))
    if comm is not None:
        local_sse = comm.allreduce(local_sse)
        local_n = comm.allreduce(local_n)
    return float(np.sqrt(local_sse / max(local_n, 1)))


def _ensemble_mean(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=1)


def _spread(X: np.ndarray, comm=None) -> float:
    """RMS ensemble spread = sqrt( mean_i Var_ens(x_i) ).

    Computed consistently with ``_rmse``: per-rank sum-of-variances and
    DOF count are Allreduced so the returned scalar is the GLOBAL spread.
    With ``ddof=1`` the per-DOF variance is the unbiased ensemble-sample
    variance. Single-member ensembles return 0.0.
    """
    if X.ndim != 2 or X.shape[1] < 2:
        return 0.0
    local_var = np.var(X, axis=1, ddof=1)
    local_sv = float(np.sum(local_var))
    local_n = int(local_var.size)
    if comm is not None:
        local_sv = comm.allreduce(local_sv)
        local_n = comm.allreduce(local_n)
    return float(np.sqrt(local_sv / max(local_n, 1)))


# ---------------------------------------------------------------------------
# Main experiment.
# ---------------------------------------------------------------------------


def run_experiment(args, output_dir: Path):
    """Run a single idealized-inlet QPCA-EnDCF experiment.

    Parallel structure follows ``idealized_inlet_da.run_single_method`` but
    omits the 4D-Var-specific TAO machinery and adds an ensemble forecast +
    QPCA-EnDCF analysis per cycling window.
    """
    from mpi4py import MPI
    from petsc4py import PETSc

    from swe4dvar.forward.problems import IdealizedInlet
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params
    from swe4dvar.utils.compat import create_petsc_vector_from_map as _cvm_
    from swe4dvar.data_assimilation import QPCAEnDCF, EnKF4D
    from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig
    from experiments.idealized_inlet_twin import (
        CartesianVortexConfig,
        generate_cartesian_vortex,
        write_cartesian_wind_hdf5,
        create_perturbed_config,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    dt = args.dt
    nt_ramp = args.nt_ramp
    nt_da = args.nt_da
    n_windows = max(1, int(args.n_windows))
    window_len = max(1, int(args.window_len))
    obs_frequency = max(1, int(args.obs_frequency))

    expected_nt_da = window_len * obs_frequency
    if nt_da != expected_nt_da:
        print(f"  [config] adjusting nt_da {nt_da} → "
              f"window_len * obs_frequency = {expected_nt_da}")
        nt_da = expected_nt_da

    nt_total = nt_ramp + n_windows * nt_da
    times = np.arange(0, (nt_total + 1) * dt, dt)

    print("=" * 60)
    print("  Idealized Inlet QPCA-EnDCF Experiment")
    print("=" * 60)
    print(f"  Vmax={args.vmax}, track_shift={args.track_shift}km")
    print(f"  Ramp={nt_ramp}×{dt}s = {nt_ramp*dt/3600:.1f}h")
    print(f"  Window={nt_da}×{dt}s = {nt_da*dt/3600:.1f}h "
          f"(window_len={window_len}, obs_frequency={obs_frequency})")
    if getattr(args, "kappa_target", None) is not None:
        pca_label = (
            f"adaptive PCA κ (target={float(args.kappa_target):.2f}, "
            f"k∈[{args.k_min}, {args.k_max if args.k_max else 'N-1'}])"
        )
    else:
        pca_label = f"PCA k={args.k_modes} (fixed)"
    print(f"  n_windows={n_windows}, ensemble N={args.ensemble_size}, "
          f"{pca_label}")
    print(f"  Obs: fraction={args.obs_fraction}, noise={args.obs_noise_level}")
    print(f"  Bg perturbation std={args.background_error_std}")
    print("=" * 60)

    # ----------------------------------------------------------------------
    # Step 1 — wind files.
    # ----------------------------------------------------------------------
    track_duration_s = float(args.track_duration_s) or 28800.0
    vortex_cfg = CartesianVortexConfig(
        Vmax=args.vmax, Rmax=args.rmax_km * 1000,
        ramp_time_s=nt_ramp * dt,
        track_duration_s=track_duration_s,
    )

    wind_dir = output_dir / "wind"
    if rank == 0:
        wind_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    _ts_tag = f"ts{int(round(args.track_shift * 10)):03d}"
    _v_tag = f"v{int(round(args.vmax))}"
    _t_tag = f"r{nt_ramp}n{n_windows*nt_da}"
    _td_tag = f"td{int(round(track_duration_s))}"
    _wind_tag = f"_{_v_tag}_{_ts_tag}_{_t_tag}_{_td_tag}"
    truth_file = wind_dir / f"truth{_wind_tag}.h5"
    pert_file = wind_dir / f"perturbed{_wind_tag}.h5"

    x_grid = np.linspace(-10000, 60000, 71)
    y_grid = np.linspace(-30000, 50000, 81)

    if rank == 0 and not truth_file.exists():
        print(f"  Generating truth wind → {truth_file.name}")
        wx, wy, p = generate_cartesian_vortex(vortex_cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(truth_file), x_grid, y_grid, times, wx, wy, p)

    pert_cfg = create_perturbed_config(vortex_cfg, args.track_shift)
    if rank == 0 and not pert_file.exists():
        print(f"  Generating perturbed wind → {pert_file.name}")
        wx, wy, p = generate_cartesian_vortex(pert_cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(pert_file), x_grid, y_grid, times, wx, wy, p)
    comm.Barrier()

    # ----------------------------------------------------------------------
    # Step 2 — truth solver + warm-up + full truth trajectory.
    # ----------------------------------------------------------------------
    print("\n--- Step 2: Truth trajectory ---")
    forcing_truth = GriddedForcing(str(truth_file), cartesian=True)
    prob_truth = IdealizedInlet(
        dt=dt, nt=nt_total,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
        dramp=nt_ramp * dt / 86400.0,
        forcing=forcing_truth,
        wd=bool(args.wd), wd_alpha=float(args.wd_alpha),
    )
    solver_truth = get_solver("DG")(prob_truth, theta=1.0, p_degree=[1, 1])
    solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=50, relaxation_parameter=0.7,
        comm=comm, error_if_not_converged=False, ksp_max_it=2000,
    )
    state_size = (
        solver_truth.V.dofmap.index_map.size_local
        * solver_truth.V.dofmap.index_map_bs
    )
    print(f"  State size: {state_size} DOFs")

    # Warm-up to end of ramp — split because truth trajectory at the DA
    # window boundary is needed to build observations.
    prob_truth.nt = nt_ramp
    solver_truth.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=False, store_jacobians=False, enable_video=False,
    )
    t_da_start = prob_truth.t
    ramp_end_state = solver_truth.u_n.x.array[:state_size].copy()
    print(f"  Warm-up done, t={t_da_start:.0f}s")

    # Truth DA trajectory (state at every step across all windows).
    prob_truth.nt = n_windows * nt_da
    solver_truth.storage.clear()
    solver_truth.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=True, store_jacobians=False, enable_video=False,
    )

    truth_trajectory_arrs = [
        np.asarray(s[:state_size], dtype=float).copy()
        for s in solver_truth.storage.saved_states
    ]
    truth_trajectory = []
    for arr in truth_trajectory_arrs:
        vec = _cvm_(solver_truth.V.dofmap.index_map,
                    solver_truth.V.dofmap.index_map_bs)
        vec.setArray(arr)
        vec.assemble()
        truth_trajectory.append(vec)

    # Truth ICs / window-end truth for diagnostics.
    truth_window_starts = [
        truth_trajectory_arrs[w * nt_da].copy() for w in range(n_windows)
    ]
    truth_window_ends = [
        truth_trajectory_arrs[(w + 1) * nt_da].copy() for w in range(n_windows)
    ]
    print(f"  Truth: {len(truth_trajectory)} saved states "
          f"(IC + {n_windows*nt_da} window steps)")

    # ----------------------------------------------------------------------
    # Step 3 — DA solver (perturbed wind).
    # ----------------------------------------------------------------------
    print("\n--- Step 3: DA solver (perturbed wind) ---")
    forcing_da = GriddedForcing(str(pert_file), cartesian=True)
    prob_da = IdealizedInlet(
        dt=dt, nt=nt_da,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
        dramp=nt_ramp * dt / 86400.0,
        forcing=forcing_da,
        wd=bool(args.wd), wd_alpha=float(args.wd_alpha),
    )
    solver_da = get_solver("DG")(prob_da, theta=1.0, p_degree=[1, 1])
    solver_da.u_n.x.array[:state_size] = ramp_end_state
    solver_da.u_n_old.x.array[:state_size] = ramp_end_state
    solver_da.u.x.array[:state_size] = ramp_end_state
    solver_da.u_n.x.scatter_forward()
    solver_da.u_n_old.x.scatter_forward()
    solver_da.u.x.scatter_forward()
    prob_da.t = t_da_start

    # ----------------------------------------------------------------------
    # Step 4 — observations + background via existing TwinExperiment utils.
    # ----------------------------------------------------------------------
    print("\n--- Step 4: Observations + background ---")
    config = TwinExperimentConfig(
        method="4dvar",
        obs_fraction=args.obs_fraction,
        obs_frequency=obs_frequency,
        obs_noise_level=args.obs_noise_level,
        interior_only=True,
        background_error_std=args.background_error_std,
        background_correlation_length=0.0,
        component_aware_cov=True,
        h_min=args.h_min,
        verbose=True,
        obs_seed=42, background_seed=123,
    )
    exp = TwinExperiment(
        problem=prob_da, solver=solver_da, config=config,
        solver_params=solver_params, comm=comm,
    )
    exp.truth_trajectory = truth_trajectory
    m_true_vec = _cvm_(solver_da.V.dofmap.index_map,
                       solver_da.V.dofmap.index_map_bs)
    m_true_vec.setArray(ramp_end_state)
    m_true_vec.assemble()
    exp.m_true = m_true_vec
    exp.t_da_start = t_da_start

    # Override nt so observation times span all windows.
    saved_nt = prob_da.nt
    prob_da.nt = n_windows * nt_da
    obs_points, obs_operator, all_obs_times = exp._setup_observations()
    observations, obs_noise_stds = exp._generate_observations(
        obs_operator, all_obs_times
    )
    exp.observations = observations
    prob_da.nt = saved_nt

    bg_rmse_init = exp._setup_background()
    h_indices, u_indices, v_indices = exp._get_component_dof_indices(owned_only=True)

    # Free the truth PETSc trajectory and solver — we keep only the numpy
    # truth_trajectory_arrs needed for window diagnostics. This drops
    # ~tens of MB of PETSc-backed state which otherwise lingers across the
    # ensemble forecast loop and contributes to the resident-set pressure
    # that kills the process during the QPCA gain step on macOS.
    for v in truth_trajectory:
        try:
            v.destroy()
        except Exception:
            pass
    truth_trajectory.clear()
    exp.truth_trajectory = None
    try:
        solver_truth.storage.clear()
        if hasattr(solver_truth.storage, "release_pool"):
            solver_truth.storage.release_pool()
    except Exception:
        pass
    del solver_truth, prob_truth, forcing_truth
    _cleanup()

    n_obs = obs_operator.get_num_observations()
    print(f"  Obs points: {n_obs}")
    print(f"  Obs times (steps from t_da_start): {all_obs_times}")
    print(f"  Initial background RMSE: {bg_rmse_init:.6f}")

    # ----------------------------------------------------------------------
    # Step 5 — initial ensemble.
    # ----------------------------------------------------------------------
    print("\n--- Step 5: Initial ensemble ---")
    bg_arr = np.asarray(exp.m_background.getArray(readonly=True),
                        dtype=float).copy()
    truth_arr = np.asarray(exp.m_true.getArray(readonly=True),
                           dtype=float).copy()
    # Compute component magnitudes from GLOBAL state so that perturbation
    # std is consistent across ranks. h_mag = mean(|h|) across all owned
    # DOFs of all ranks; uv_mag = max(|u|, |v|) across all owned DOFs.
    local_h_abs_sum = float(np.sum(np.abs(truth_arr[h_indices])))
    local_h_count = int(len(h_indices))
    local_uv_max = float(max(
        float(np.abs(truth_arr[u_indices]).max()) if len(u_indices) else 0.0,
        float(np.abs(truth_arr[v_indices]).max()) if len(v_indices) else 0.0,
    ))
    global_h_abs_sum = comm.allreduce(local_h_abs_sum)
    global_h_count = comm.allreduce(local_h_count)
    global_uv_max = comm.allreduce(local_uv_max, op=MPI.MAX)
    h_mag = global_h_abs_sum / max(global_h_count, 1) + 1e-10
    uv_mag = max(global_uv_max, 0.1)
    h_std = args.background_error_std * h_mag
    uv_std = args.background_error_std * uv_mag
    print(f"  Component perturbation std: h={h_std:.4f}, uv={uv_std:.4f}")
    X_prior = _build_initial_ensemble(
        bg_arr, h_indices, u_indices, v_indices,
        h_std=h_std, uv_std=uv_std,
        N=args.ensemble_size, h_min=args.h_min,
        seed=args.ensemble_seed,
        rank=rank,
    )
    print(f"  Initial ensemble shape: {X_prior.shape}")

    # ----------------------------------------------------------------------
    # Step 6 — QPCA-EnDCF filter (one observation time at a time).
    # ----------------------------------------------------------------------
    print("\n--- Step 6: QPCA-EnDCF filter ---")
    template_vec = _cvm_(solver_da.V.dofmap.index_map,
                         solver_da.V.dofmap.index_map_bs)
    apply_H = _PointObsBridge(obs_operator, template_vec, state_size)
    R_diag = np.full(n_obs, float(obs_noise_stds.mean()) ** 2)
    R_matrix = np.diag(R_diag)
    method = str(getattr(args, "method", "qpca")).lower()
    if method == "qpca":
        kappa_target = getattr(args, "kappa_target", None)
        kappa_target = float(kappa_target) if kappa_target is not None else None
        filter_obj = QPCAEnDCF(
            apply_H=apply_H, R=R_matrix, window_len=window_len,
            k=args.k_modes,
            kappa_target=kappa_target,
            k_min=int(getattr(args, "k_min", 1)),
            k_max=(int(args.k_max) if getattr(args, "k_max", None) else None),
            stabilize=True,
        )
        if kappa_target is None:
            print(f"\n  Filter: QPCA-EnDCF (κ={args.k_modes}, fixed)")
        else:
            print(f"\n  Filter: QPCA-EnDCF (adaptive κ, "
                  f"target={kappa_target:.2f}, "
                  f"k∈[{int(getattr(args, 'k_min', 1))}, "
                  f"{(getattr(args, 'k_max', None) or 'N-1')}])")
    elif method == "enkf4d":
        # Use ensemble_seed for the perturbed-obs RNG. Same seed on all
        # MPI ranks → all ranks generate identical ε and the gain is
        # computed identically everywhere.
        filter_obj = EnKF4D(
            apply_H=apply_H, R=R_matrix, window_len=window_len,
            seed=int(args.ensemble_seed),
        )
        print(f"\n  Filter: 4D-EnKF (stochastic baseline)")
    elif method == "seq_enkf":
        # Sequential stochastic EnKF: one update per observation time
        # inside the window, with intermediate forecast segments. The
        # filter object handles a single observation time at a time;
        # the cycling loop below interleaves it with segment forecasts.
        from swe4dvar.data_assimilation import SeqEnKF
        filter_obj = SeqEnKF(
            apply_H=apply_H, R=R_matrix,
            seed=int(args.ensemble_seed),
        )
        print(f"\n  Filter: sequential stochastic EnKF (one update per obs time)")
    elif method == "letkf":
        # 4D LETKF (Hunt, Kostelich, Szunyogh 2007): deterministic
        # ensemble-square-root analysis in (N, N) ensemble space, with
        # optional per-state-DOF R-localization when ``--loc-radius``
        # > 0. The seed argument is accepted only for API symmetry
        # with the stochastic filters; LETKF itself uses no RNG.
        from swe4dvar.data_assimilation import LETKF
        filter_obj = LETKF(
            apply_H=apply_H, R=R_matrix, window_len=window_len,
            seed=int(args.ensemble_seed),
        )
        print(f"\n  Filter: 4D LETKF (deterministic square-root)")
    else:
        raise ValueError(
            f"--method must be 'qpca', 'enkf4d', 'seq_enkf', or "
            f"'letkf', got {method!r}"
        )

    # ----------------------------------------------------------------------
    # Optional: build a Gaspari-Cohn localization taper for the gain step.
    #
    # When ``--loc-radius`` > 0, we build a sparse (state_size, m·L) taper
    # where state DOFs farther than ``L_loc`` from each observation get zero
    # weight. The taper is applied entrywise to the empirical cross-cov
    # ``P_xz`` inside QPCAEnDCF.update — see the filter's docstring.
    #
    # State DOF coordinates: we use cell centroids — for the mixed (h, (u,v))
    # DG-P1 element on triangles, all 9 DOFs in a cell share the centroid as
    # their localization-distance reference. The slight inaccuracy vs vertex
    # coordinates is negligible relative to localization radii of order
    # several km.
    # ----------------------------------------------------------------------
    rho_taper = None
    rho_taper_per_time = None  # (n_local, m) per-obs-time taper for seq_enkf
    loc_radius = float(getattr(args, "loc_radius", 0.0))
    if loc_radius > 0.0:
        from scipy.sparse import hstack as _sphstack
        from swe4dvar.data_assimilation.qpca_endcf import build_spatial_taper

        print(f"\n  Building Gaspari-Cohn localization taper (L_loc = "
              f"{loc_radius:.0f} m)...")
        mesh = solver_da.V.mesh
        tdim = mesh.topology.dim
        num_cells = mesh.topology.index_map(tdim).size_local
        mesh_geom = mesh.geometry.x  # (n_vertices, 3) including ghost vertices

        cell_centroids = np.zeros((num_cells, 2), dtype=float)
        for cell_idx in range(num_cells):
            cell_verts = mesh.geometry.dofmap[cell_idx]
            cell_centroids[cell_idx] = mesh_geom[cell_verts, :2].mean(axis=0)

        dofmap = solver_da.V.dofmap
        state_coords = np.zeros((state_size, 2), dtype=float)
        seen = np.zeros(state_size, dtype=bool)
        for cell_idx in range(num_cells):
            cdofs = dofmap.cell_dofs(cell_idx)
            for dof in cdofs:
                if dof < state_size and not seen[dof]:
                    state_coords[dof] = cell_centroids[cell_idx]
                    seen[dof] = True
        n_uncovered = int((~seen).sum())
        if n_uncovered > 0:
            print(f"  [warn] {n_uncovered} owned state DOFs had no cell — "
                  f"assigning origin as fallback coord")

        obs_coords = np.asarray(obs_points[:, :2], dtype=float)
        spatial = build_spatial_taper(state_coords, obs_coords, loc_radius)
        rho_taper = _sphstack([spatial] * window_len).tocsr()
        # Per-observation-time taper for sequential EnKF (one update
        # per obs time uses the (n, m) shape directly, not the stacked
        # (n, mL) version).
        rho_taper_per_time = spatial.tocsr()
        print(f"  Taper: spatial shape {spatial.shape} (nnz {spatial.nnz}, "
              f"avg {spatial.nnz / max(state_size, 1):.1f} obs/DOF); "
              f"stacked {rho_taper.shape} (nnz {rho_taper.nnz})")

    # Indices into observations list (built by _generate_observations) that
    # fall inside each window. _generate_observations writes one obs vec
    # per element of all_obs_times = range(0, prob_da.nt+1, obs_frequency)
    # — i.e. including step 0 of the original prob_da configuration. We
    # use post-window observations only (matches QPCA convention).
    all_obs_times_arr = np.asarray(all_obs_times, dtype=int)
    window_records = []
    for w in range(n_windows):
        window_start_step = w * nt_da   # in DA-clock steps
        window_end_step = (w + 1) * nt_da
        # Post-window obs steps inside (start, end]:
        obs_step_indices = []   # within-window step counts
        obs_global_indices = []  # indices into `observations`
        for i, gt in enumerate(all_obs_times_arr):
            if window_start_step < gt <= window_end_step:
                obs_step_indices.append(int(gt - window_start_step))
                obs_global_indices.append(i)
        if len(obs_step_indices) < window_len:
            # Trim/pad — but window_len was set from obs_frequency above,
            # so this should already match. Defensive check.
            print(f"  [warn] window {w} has {len(obs_step_indices)} obs "
                  f"in (window_start, window_end] but window_len={window_len}")
        obs_step_indices = obs_step_indices[:window_len]
        obs_global_indices = obs_global_indices[:window_len]

        t_window_start = t_da_start + window_start_step * dt
        print(f"\n  Window {w+1}/{n_windows}: "
              f"t={t_window_start:.0f}s → {t_window_start + nt_da*dt:.0f}s, "
              f"obs steps within window = {obs_step_indices}")

        # Pre-analysis (prior) ensemble diagnostics.
        prior_mean = _ensemble_mean(X_prior)
        prior_rmse_truth = _rmse(prior_mean, truth_window_starts[w], comm=comm)
        prior_spread = _spread(X_prior, comm=comm)

        # Forecast each member through the window. For the 4D filters
        # (qpca, enkf4d) this is a single full-window forecast returning
        # the ensemble at every obs time; the sequential filter
        # (seq_enkf) instead interleaves the forecast with analyses and
        # is handled in the analysis block below.
        if method != "seq_enkf":
            t0_fc = time.time()
            X_path, n_failed = _forecast_ensemble(
                X_prior,
                solver_da=solver_da, prob_da=prob_da,
                solver_params=solver_params, state_size=state_size,
                t_window_start=t_window_start, nt_window=nt_da,
                obs_step_indices=obs_step_indices,
                h_indices=h_indices, h_min=args.h_min,
            )
            fc_time = time.time() - t0_fc
            print(f"  Forecast: {fc_time:.1f}s "
                  f"({args.ensemble_size} members × {nt_da} steps, "
                  f"failed={n_failed})", flush=True)
        else:
            # Defer: the sequential branch below combines forecast and
            # analysis into one interleaved pass per observation time.
            X_path = None
            fc_time = 0.0
            n_failed = 0

        # Free PETSc memory after the per-member forward loop. The Newton
        # solves leak factor matrices into PETSc's internal pool; if we do
        # not return them before the QPCA gain computation, the resident
        # set grows enough to trip the macOS jetsam killer on a 200k-DOF
        # mesh.
        try:
            solver_da.storage.clear()
            if hasattr(solver_da.storage, "release_pool"):
                solver_da.storage.release_pool()
        except Exception:
            pass
        _cleanup()

        # Stack the synthetic obs at this window's obs times.
        z_blocks = []
        for gi in obs_global_indices:
            z_blocks.append(np.asarray(observations[gi].getArray(),
                                       dtype=float).copy())
        z_stack = np.concatenate(z_blocks) if z_blocks else np.zeros(0)

        if method != "seq_enkf":
            # 4D branch (qpca, enkf4d): compute window-end HX once,
            # call the filter once.
            HX_blocks_prior = [apply_H(X_path[t]) for t in range(len(X_path))]
            prior_HX_end = HX_blocks_prior[-1]
            prior_obs_misfit = float(np.sqrt(
                np.mean((prior_HX_end.mean(axis=1) - z_blocks[-1]) ** 2)
            )) if z_blocks else 0.0

            t0_an = time.time()
            try:
                X_post = filter_obj.update(
                    X_path, z_stack,
                    HX_blocks=HX_blocks_prior,
                    rho=rho_taper,
                )
            except Exception as exc:
                print(f"  [error] {method} update raised "
                      f"{type(exc).__name__}: {exc}; keeping forecast ensemble",
                      flush=True)
                X_post = X_path[-1].copy()
            an_time = time.time() - t0_an
        else:
            # Sequential branch: interleave L forecast segments with L
            # sequential analysis updates. Each segment forecasts from
            # the *previous analysis* to the next observation time, so
            # the analyses propagate forward through the window. The
            # window-end ensemble after the L-th analysis is X_post.
            #
            # Diagnostics: we collect the prior obs misfit at the last
            # forecast segment (the L-th obs time, equivalently the
            # window-end forecast) so it is directly comparable to the
            # 4D filters' prior_obs_misfit. The X_path list is populated
            # with the *pre-analysis* ensemble at each obs time (the
            # forecast segment endpoint, before the sequential update),
            # which the spread and spectrum diagnostics below expect.
            t0_total = time.time()
            X_curr = X_prior
            t_seg_start = t_window_start
            prev_step = 0
            X_path = []
            HX_blocks_prior = []
            seq_fc_time = 0.0
            n_failed = 0
            for t_idx, step_at in enumerate(obs_step_indices):
                seg_steps = step_at - prev_step
                if seg_steps <= 0:
                    continue
                t0_seg = time.time()
                X_seg, n_failed_seg = _forecast_ensemble(
                    X_curr,
                    solver_da=solver_da, prob_da=prob_da,
                    solver_params=solver_params, state_size=state_size,
                    t_window_start=t_seg_start,
                    nt_window=seg_steps,
                    obs_step_indices=[seg_steps],
                    h_indices=h_indices, h_min=args.h_min,
                )
                seq_fc_time += time.time() - t0_seg
                n_failed += int(n_failed_seg)
                X_at_obs = X_seg[0]
                HX_t = np.asarray(apply_H(X_at_obs), dtype=float)
                X_path.append(X_at_obs)
                HX_blocks_prior.append(HX_t)
                # Sequential update at this observation time.
                z_t = z_blocks[t_idx]
                try:
                    X_curr = filter_obj.update(
                        X_at_obs, z_t,
                        HX=HX_t,
                        rho=rho_taper_per_time,
                    )
                except Exception as exc:
                    print(f"  [error] seq_enkf update at obs {t_idx} raised "
                          f"{type(exc).__name__}: {exc}; "
                          f"keeping forecast at this time",
                          flush=True)
                    X_curr = X_at_obs.copy()
                # Wet-cell clip after each sequential analysis.
                X_curr[h_indices, :] = np.maximum(
                    X_curr[h_indices, :], args.h_min)
                t_seg_start += seg_steps * dt
                prev_step = step_at
            fc_time = seq_fc_time
            # Prior obs misfit at window end: the last pre-analysis HX
            # against the last observation (analog of the 4D
            # prior_obs_misfit).
            if HX_blocks_prior:
                prior_HX_end = HX_blocks_prior[-1]
                prior_obs_misfit = float(np.sqrt(
                    np.mean((prior_HX_end.mean(axis=1) - z_blocks[-1]) ** 2)
                )) if z_blocks else 0.0
            else:
                prior_obs_misfit = 0.0
            X_post = X_curr
            an_time = time.time() - t0_total - fc_time
            print(f"  Forecast (sequential): {fc_time:.1f}s "
                  f"({args.ensemble_size} members × {nt_da} steps total, "
                  f"failed={n_failed})", flush=True)

        # Wet-cell clipping on the analysis (matches background draws).
        X_post[h_indices, :] = np.maximum(X_post[h_indices, :], args.h_min)

        # Multiplicative inflation (paper Algorithm 1 convention):
        # X_a ← x̄_a + λ_infl · (X_a − x̄_a). Preserves the analysis mean
        # while inflating ensemble anomalies by the factor λ_infl (variance
        # by λ_infl²). 1.0 → no-op. Re-clip h after inflation so we don't
        # push members below h_min via inflated negative anomalies.
        infl = float(args.inflation)
        if infl != 1.0:
            x_mean = X_post.mean(axis=1, keepdims=True)
            X_post = x_mean + infl * (X_post - x_mean)
            X_post[h_indices, :] = np.maximum(X_post[h_indices, :], args.h_min)

        # Post-analysis diagnostics.
        post_mean = _ensemble_mean(X_post)
        post_rmse_truth = _rmse(post_mean, truth_window_ends[w], comm=comm)
        post_HX = apply_H(X_post)
        post_obs_misfit = float(np.sqrt(
            np.mean((post_HX.mean(axis=1) - z_blocks[-1]) ** 2)
        )) if z_blocks else 0.0

        forecast_end_mean = _ensemble_mean(X_path[-1])
        forecast_rmse_truth = _rmse(forecast_end_mean, truth_window_ends[w], comm=comm)
        forecast_spread = _spread(X_path[-1], comm=comm)

        improvement_state = 0.0
        if forecast_rmse_truth > 0:
            improvement_state = (
                100.0 * (forecast_rmse_truth - post_rmse_truth)
                / forecast_rmse_truth
            )
        improvement_obs = 0.0
        if prior_obs_misfit > 0:
            improvement_obs = (
                100.0 * (prior_obs_misfit - post_obs_misfit) / prior_obs_misfit
            )

        # Global RMS ensemble spread of the ANALYSIS ensemble. Computed
        # with MPI Allreduce so it is comparable to the global RMSE.
        # Must be defined BEFORE the print block below, which references
        # analysis_spread and the γ̄ ratios.
        analysis_spread = _spread(X_post, comm=comm)

        # Spread/skill ratio γ̄ = spread / ensemble-mean RMSE vs truth.
        # A well-calibrated ensemble has γ̄ ≈ √((N+1)/N); under-dispersive
        # filters (typical for small N before localization+inflation) sit
        # well below 1, over-dispersive filters above. The three γ̄s
        # (prior, forecast, analysis) tell us, in order: was the carry-in
        # ensemble calibrated, did the forecast preserve calibration, did
        # the analysis update collapse spread relative to skill.
        def _ratio(a: float, b: float) -> float:
            return float(a / b) if b > 0 else 0.0
        prior_spread_skill_truth = _ratio(prior_spread, prior_rmse_truth)
        forecast_spread_skill_truth = _ratio(forecast_spread, forecast_rmse_truth)
        analysis_spread_skill_truth = _ratio(analysis_spread, post_rmse_truth)

        print(f"  Prior ensemble-mean RMSE (vs truth IC): "
              f"{prior_rmse_truth:.6f}")
        print(f"  Forecast RMSE (vs truth window-end):    "
              f"{forecast_rmse_truth:.6f}")
        print(f"  Analysis RMSE (vs truth window-end):    "
              f"{post_rmse_truth:.6f} "
              f"(state Δ = {improvement_state:+.1f}%)")
        print(f"  Obs misfit fc→an at window end:         "
              f"{prior_obs_misfit:.6f} → {post_obs_misfit:.6f} "
              f"(Δ = {improvement_obs:+.1f}%)")
        print(f"  Spread (prior / forecast / analysis):   "
              f"{prior_spread:.4f} / {forecast_spread:.4f} / "
              f"{analysis_spread:.4f}")
        print(f"  γ̄ spread-skill (prior / fc / an):       "
              f"{prior_spread_skill_truth:.3f} / "
              f"{forecast_spread_skill_truth:.3f} / "
              f"{analysis_spread_skill_truth:.3f}")
        print(f"  Analysis time: {an_time:.2f}s")

        # Spectrum diagnostics from the filter's C_E eigendecomposition.
        # We log the top-(k+5) descending eigenvalues, the spectrum trace
        # (total whitened-residual variance), and a few derived ratios:
        #   var_explained_top1   = λ_1 / trace
        #   var_explained_top_k  = (λ_1 + … + λ_k) / trace
        #   spectral_gap_k_kp1   = (λ_k − λ_{k+1}) / λ_max  (large → stable κ)
        eigs_desc = getattr(filter_obj, "last_eigenvalues_desc", None)
        trace_val = float(getattr(filter_obj, "last_eigenvalue_trace", 0.0))
        # κ actually used by the filter on this window (== args.k_modes in
        # fixed-κ mode, and the adaptively-chosen value when kappa_target
        # is set). Falls back to args.k_modes for filters that don't expose
        # last_k_used (e.g. EnKF4D).
        kappa_used = int(getattr(filter_obj, "last_k_used", args.k_modes))
        spectrum_record = {}
        if eigs_desc is not None and len(eigs_desc) > 0:
            kappa = max(1, kappa_used)
            top_n = int(min(len(eigs_desc), kappa + 5))
            top = [float(x) for x in eigs_desc[:top_n]]
            lam_max = float(eigs_desc[0])
            lam_k = float(eigs_desc[kappa - 1]) if kappa <= len(eigs_desc) else 0.0
            lam_kp1 = float(eigs_desc[kappa]) if kappa < len(eigs_desc) else 0.0
            spectrum_record = {
                "spectrum_trace": trace_val,
                "spectrum_top_descending": top,
                "k_used": int(kappa_used),
                "spectrum_top1_over_trace": (
                    float(lam_max / trace_val) if trace_val > 0 else 0.0),
                "spectrum_top_k_over_trace": (
                    float(sum(eigs_desc[:kappa]) / trace_val)
                    if trace_val > 0 else 0.0),
                "spectrum_gap_k_kp1_over_lam_max": (
                    float((lam_k - lam_kp1) / lam_max) if lam_max > 0 else 0.0),
            }
        # Echo κ_used for the slurm log when adaptive mode is on.
        if getattr(args, "kappa_target", None) is not None:
            print(f"  κ chosen by adaptive policy: {kappa_used} "
                  f"(target = {float(args.kappa_target):.2f})")

        window_records.append({
            "window": w + 1,
            "t_window_start_s": float(t_window_start),
            "t_window_end_s": float(t_window_start + nt_da * dt),
            "obs_step_indices": [int(s) for s in obs_step_indices],
            "n_obs_per_time": int(n_obs),
            "n_failed_members": int(n_failed),
            "prior_rmse_truth_ic": float(prior_rmse_truth),
            "forecast_rmse_truth_end": float(forecast_rmse_truth),
            "analysis_rmse_truth_end": float(post_rmse_truth),
            "state_improvement_pct": float(improvement_state),
            "prior_obs_misfit_end": float(prior_obs_misfit),
            "analysis_obs_misfit_end": float(post_obs_misfit),
            "obs_improvement_pct": float(improvement_obs),
            "prior_spread": float(prior_spread),
            "forecast_spread": float(forecast_spread),
            "analysis_spread": float(analysis_spread),
            "ensemble_spread": float(analysis_spread),  # legacy alias
            "prior_spread_skill_truth": prior_spread_skill_truth,
            "forecast_spread_skill_truth": forecast_spread_skill_truth,
            "analysis_spread_skill_truth": analysis_spread_skill_truth,
            "forecast_time_s": float(fc_time),
            "analysis_time_s": float(an_time),
            **spectrum_record,
        })

        X_prior = X_post  # chain into next window
        _cleanup()

    # ----------------------------------------------------------------------
    # Step 7 — save results.
    # ----------------------------------------------------------------------
    result = {
        "method": {
            "qpca": "qpca_endcf",
            "enkf4d": "enkf4d",
            "seq_enkf": "seq_enkf",
            "letkf": "letkf",
        }.get(method, "enkf4d"),
        "n_windows": n_windows,
        "ensemble_size": args.ensemble_size,
        "window_len": window_len,
        "obs_frequency": obs_frequency,
        "k_modes": args.k_modes,
        "kappa_target": (
            float(args.kappa_target)
            if getattr(args, "kappa_target", None) is not None else None
        ),
        "k_min": int(getattr(args, "k_min", 1)),
        "k_max": (
            int(args.k_max) if getattr(args, "k_max", None) else None
        ),
        "windows": window_records,
        "config": {
            "vmax": float(args.vmax),
            "track_shift_km": float(args.track_shift),
            "dt": float(dt),
            "nt_ramp": int(nt_ramp),
            "nt_da": int(nt_da),
            "obs_fraction": float(args.obs_fraction),
            "obs_noise_level": float(args.obs_noise_level),
            "background_error_std": float(args.background_error_std),
            "h_min": float(args.h_min),
            "ensemble_seed": int(args.ensemble_seed),
            "loc_radius_m": float(args.loc_radius),
            "inflation": float(args.inflation),
        },
        "state_size_owned": int(state_size),
        "n_obs": int(n_obs),
    }

    result_file = output_dir / "result_qpca_endcf.json"
    if rank == 0:
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved: {result_file}")

    # Cleanup
    try:
        template_vec.destroy()
    except Exception:
        pass
    try:
        m_true_vec.destroy()
    except Exception:
        pass
    _cleanup()

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser():
    p = argparse.ArgumentParser(
        description="Idealized Inlet QPCA-EnDCF ensemble DA experiment."
    )
    p.add_argument("--vmax", type=float, default=20.0)
    p.add_argument("--rmax-km", type=float, default=15.0)
    p.add_argument("--track-shift", type=float, default=10.0)
    p.add_argument("--track-duration-s", type=float, default=0.0)
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--nt-ramp", type=int, default=12,
                   help="Warm-up timesteps before DA (default 12 ≈ 2h at dt=600).")
    p.add_argument("--nt-da", type=int, default=6,
                   help="Timesteps per DA window. Must equal window_len*obs_frequency.")
    p.add_argument("--n-windows", type=int, default=1,
                   help="Cycling windows (default 1 = single window).")
    p.add_argument("--window-len", type=int, default=3,
                   help="Number of observation times per QPCA-EnDCF window (L).")
    p.add_argument("--obs-frequency", type=int, default=2,
                   help="Timesteps between observations within a window.")
    p.add_argument("--obs-fraction", type=float, default=0.05)
    p.add_argument("--obs-noise-level", type=float, default=0.01)
    p.add_argument("--background-error-std", type=float, default=0.02)
    p.add_argument("--ensemble-size", type=int, default=20,
                   help="N — number of ensemble members.")
    p.add_argument("--ensemble-seed", type=int, default=2026)
    p.add_argument("--k-modes", type=int, default=1,
                   help="QPCA PCA modes retained (default 1, paper baseline). "
                        "Ignored when --kappa-target is supplied.")
    p.add_argument("--kappa-target", type=float, default=None,
                   help="Adaptive κ: smallest κ per window such that "
                        "cumulative variance-explained ≥ this target. "
                        "Lies in (0, 1]. Disables fixed --k-modes when set.")
    p.add_argument("--k-min", type=int, default=1,
                   help="Lower clamp on adaptive κ (default 1).")
    p.add_argument("--k-max", type=int, default=None,
                   help="Upper clamp on adaptive κ "
                        "(default min(N-1, mL) per window).")
    p.add_argument("--h-min", type=float, default=0.5,
                   help="Wet-cell floor on h DOFs.")
    p.add_argument("--wd", action="store_true",
                   help="Enable Karna wetting/drying lift in the forward solve.")
    p.add_argument("--wd-alpha", type=float, default=1.5)
    p.add_argument("--loc-radius", type=float, default=0.0,
                   help="Gaspari-Cohn localization cutoff radius (m). When > 0, "
                        "the gain step is Schur-multiplied by a GC taper so obs "
                        "farther than this distance from a state DOF contribute "
                        "zero. Default 0 (no localization, paper-exact filter).")
    # --- Post-smoke override knobs (intentionally do NOT have CLI defaults
    # equal to the smoke defaults) so they can be left unset and still let
    # smoke pick the standard values. When set, they override the smoke
    # block's choice for that single parameter.
    p.add_argument("--override-k", type=int, default=None,
                   help="Override --k-modes / smoke κ. Use for κ sweeps.")
    p.add_argument("--override-loc-radius", type=float, default=None,
                   help="Override --loc-radius / smoke localization. "
                        "Set to 0 to disable localization while keeping all "
                        "other smoke defaults.")
    p.add_argument("--override-n-windows", type=int, default=None,
                   help="Override --n-windows / smoke n_windows. "
                        "Use to enable cycling at otherwise-smoke config.")
    p.add_argument("--override-ensemble-size", type=int, default=None,
                   help="Override --ensemble-size / smoke N. Use for N sweeps.")
    p.add_argument("--override-obs-noise-level", type=float, default=None,
                   help="Override --obs-noise-level / smoke σ_obs. Use for "
                        "observation-informativeness sweeps. Affects BOTH the "
                        "synthetic obs noise added to truth AND the R passed "
                        "to the filter — twin-experiment self-consistent.")
    p.add_argument("--inflation", type=float, default=1.0,
                   help="Multiplicative inflation factor applied to analysis "
                        "ensemble anomalies *before* the next forecast cycle: "
                        "X_a ← x̄_a + λ_infl · (X_a − x̄_a). 1.0 = no inflation "
                        "(QPCA-EnDCF paper default). Stochastic ensemble "
                        "baselines typically use 1.02-1.10 to counteract "
                        "analysis-step variance collapse. Set higher for "
                        "cycling on undersampled PDE state where the localized "
                        "gain shrinks spread too aggressively.")
    p.add_argument("--method",
                   choices=["qpca", "enkf4d", "seq_enkf", "letkf"],
                   default="qpca",
                   help="Filter to use. 'qpca' = QPCA-EnDCF (paper Alg. 3, "
                        "spectrally truncated, deterministic 4D update). "
                        "'enkf4d' = stochastic 4D-EnKF (paper Alg. 2, "
                        "R-stabilized gain, perturbed obs, single window-end "
                        "update). 'seq_enkf' = traditional sequential "
                        "stochastic EnKF (one update per observation time "
                        "with intermediate forecast segments). 'letkf' = "
                        "4D LETKF (Hunt et al. 2007, deterministic "
                        "square-root analysis in ensemble space, per-state-"
                        "DOF R-localization when --loc-radius > 0). All "
                        "filters honor --loc-radius and --inflation "
                        "identically.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny config for a smoke test (overrides most flags).")
    p.add_argument("--micro-smoke", action="store_true",
                   help="Minimum-effort smoke test: 3-member ensemble, single 2-step window.")
    p.add_argument("--output-subdir", default="idealized_inlet_qpca_endcf",
                   help="Subdirectory of results/ for output files.")
    return p


def main():
    args = _parser().parse_args()

    if args.smoke:
        # Minimal viable config — keeps every code path covered (truth
        # generation, observation setup, ensemble init, forecast loop,
        # QPCA-EnDCF update, JSON write) while keeping total cost down to
        # roughly truth-ramp + (N+1) short forecasts.
        args.vmax = 15.0
        args.track_shift = 5.0
        args.dt = 600.0
        args.nt_ramp = 6
        args.window_len = 4
        args.obs_frequency = 2
        args.nt_da = args.window_len * args.obs_frequency
        args.n_windows = 1
        args.ensemble_size = 10
        args.k_modes = 1
        # obs_fraction = 0.043 yields ~500 obs on the inlet mesh
        # (~11,650 interior nodes × 0.043 ≈ 500).
        args.obs_fraction = 0.043
        args.obs_noise_level = 0.01
        # Bumped 0.03 → 0.10: the 3% prior was too good (prior obs misfit
        # ≈ noise floor → filter has no signal to absorb). 10% gives the
        # forecast room to drift a few-σ off truth so the analysis update
        # has real work to do.
        args.background_error_std = 0.10
        # Localization radius (m). 5000 m ≈ a few mesh element diameters on
        # the inlet domain — keeps each obs influencing roughly the cells in
        # its immediate neighborhood. Override with --loc-radius if needed;
        # set to 0 to disable.
        if args.loc_radius <= 0:
            args.loc_radius = 5000.0
        print("[smoke] using minimal config for a quick verification run")

    if args.micro_smoke:
        # Absolute minimum config — 3 members, single 2-step window, very few
        # obs. Used as a fast (<5 min) syntactic check that the whole pipeline
        # still wires together end-to-end.
        args.vmax = 10.0
        args.track_shift = 5.0
        args.dt = 600.0
        args.nt_ramp = 4
        args.window_len = 1
        args.obs_frequency = 2
        args.nt_da = args.window_len * args.obs_frequency
        args.n_windows = 1
        args.ensemble_size = 3
        args.k_modes = 1
        args.obs_fraction = 0.02
        args.obs_noise_level = 0.02
        args.background_error_std = 0.03
        print("[micro-smoke] using absolute-minimum config")

    # Apply --override-* flags AFTER smoke / micro-smoke defaults so a single
    # CLI flag can flip one knob (e.g. κ or localization) while leaving the
    # rest of the smoke config untouched. Each override is None by default.
    if args.override_k is not None:
        print(f"[override] k_modes {args.k_modes} → {args.override_k}")
        args.k_modes = int(args.override_k)
    if args.override_loc_radius is not None:
        print(f"[override] loc_radius {args.loc_radius} → "
              f"{args.override_loc_radius}")
        args.loc_radius = float(args.override_loc_radius)
    if args.override_n_windows is not None:
        print(f"[override] n_windows {args.n_windows} → "
              f"{args.override_n_windows}")
        args.n_windows = int(args.override_n_windows)
    if args.override_ensemble_size is not None:
        print(f"[override] ensemble_size {args.ensemble_size} → "
              f"{args.override_ensemble_size}")
        args.ensemble_size = int(args.override_ensemble_size)
    if args.override_obs_noise_level is not None:
        print(f"[override] obs_noise_level {args.obs_noise_level} → "
              f"{args.override_obs_noise_level}")
        args.obs_noise_level = float(args.override_obs_noise_level)

    # mkdir is gated on rank 0 with a barrier; all ranks then enter
    # run_experiment together. Importing mpi4py at the top of main() so
    # the rank check happens before any heavy setup.
    from mpi4py import MPI as _MPI
    output_dir = PROJECT_ROOT / "results" / args.output_subdir
    if _MPI.COMM_WORLD.Get_rank() == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    _MPI.COMM_WORLD.Barrier()

    run_experiment(args, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())

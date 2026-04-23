#!/usr/bin/env python3
"""
Entry point for the Shinnecock Inlet DC-WME vs 4D-Var comparison study.

Usage:
    # Phase 0: Forward verification (~10 min)
    python experiments/shinnecock_study/run_comparison.py --phase 0

    # Phase 0.5: Gradient verification (~30 min)
    python experiments/shinnecock_study/run_comparison.py --phase 0.5

    # Phase 1: Single-window 4D-Var, no model error
    python experiments/shinnecock_study/run_comparison.py --phase 1

    # Phase 2: Single-window 4D-Var, with model error
    python experiments/shinnecock_study/run_comparison.py --phase 2

    # Phase 3: Single-window DC-WME
    python experiments/shinnecock_study/run_comparison.py --phase 3

    # Phase 4: Cycling comparison
    python experiments/shinnecock_study/run_comparison.py --phase 4

    # Phase 5: Parameter sweeps
    python experiments/shinnecock_study/run_comparison.py --phase 5
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Shinnecock Inlet DC-WME vs 4D-Var comparison study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--phase",
        type=str,
        default="0",
        choices=["0", "0.5", "1", "2", "3", "4", "5", "6", "7", "all"],
        help="Experiment phase to run",
    )

    parser.add_argument(
        "--sweep-dim",
        type=str,
        default=None,
        choices=["noise", "obs_density", "obs_frequency", "bg_error",
                 "predictability_gamma", "window_length", "model_error", "all"],
        help="Phase 6: which sweep dimension to run (default: all)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/shinnecock_study",
        help="Output directory for results",
    )

    parser.add_argument(
        "--adios-file",
        type=str,
        default="data/shinnecock_inlet",
        help="Path to ADCIRC ADIOS files (without extension)",
    )

    parser.add_argument(
        "--sub",
        type=str,
        default=None,
        choices=["a", "b", "c"],
        help="Phase 3 sub-experiment to run (a=4DVar, b=static DC-WME, c=dynamic DC-WME)",
    )

    parser.add_argument(
        "--mem-limit-gb",
        type=float,
        default=12.0,
        help="Phase 6: memory limit in GB; sweep aborts gracefully if exceeded (default: 12)",
    )

    parser.add_argument(
        "--phase6-suite",
        type=str,
        default="controlled",
        choices=["legacy", "controlled"],
        help="Phase 6: legacy a/b comparison or controlled ablation suite (default: controlled)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output",
    )

    return parser.parse_args()


PHASE6_DYNAMIC_VALIDATION_DIMS = {"noise", "window_length"}


def _get_sweep_value(sweep_params: Dict, key: str, default):
    """Return a sweep override if present, otherwise the provided default."""
    if not sweep_params:
        return default
    return sweep_params.get(key, default)


def _summarize_eigenvalues(eigvals) -> Dict[str, float]:
    """Return compact spectral diagnostics for a 1D eigenvalue array."""
    arr = np.asarray(eigvals, dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "lambda_min": None,
            "lambda_max": None,
            "lambda_mean": None,
            "condition_number": None,
            "spread_pct": None,
            "rank_gt_1e-10": 0,
        }

    lam_min = float(arr.min())
    lam_max = float(arr.max())
    lam_mean = float(arr.mean())
    lam_floor = max(lam_min, 1e-30)
    return {
        "count": int(arr.size),
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "lambda_mean": lam_mean,
        "condition_number": float(lam_max / lam_floor),
        "spread_pct": float(100.0 * (lam_max - lam_min) / max(abs(lam_mean), 1e-30)),
        "rank_gt_1e-10": int(np.sum(arr > 1e-10)),
    }


def _phase6_method_suite(dim_name: str, suite: str = "controlled") -> List[Dict]:
    """Return the Phase 6 method/ablation suite for a sweep dimension."""
    legacy = [
        {
            "variant_key": "4dvar_baseline",
            "sub_label": "a",
            "method": "4dvar",
            "l_wme_mode": "N/A",
            "apply_eq38_background_scaling": False,
            "description": "Classical 4D-Var baseline",
        },
        {
            "variant_key": "dcwme_static",
            "sub_label": "b",
            "method": "dcwme",
            "l_wme_mode": "static",
            "apply_eq38_background_scaling": True,
            "description": "DC-WME with static L_wme",
        },
    ]

    if suite == "legacy":
        return legacy

    controlled = list(legacy)
    controlled.append(
        {
            "variant_key": "4dvar_eq38",
            "sub_label": "c",
            "method": "4dvar",
            "l_wme_mode": "N/A",
            "apply_eq38_background_scaling": True,
            "description": "4D-Var with matched Eq. 38 background scaling",
        }
    )
    if dim_name in PHASE6_DYNAMIC_VALIDATION_DIMS:
        controlled.append(
            {
                "variant_key": "dcwme_dynamic",
                "sub_label": "d",
                "method": "dcwme",
                "l_wme_mode": "dynamic",
                "apply_eq38_background_scaling": True,
                "description": "DC-WME with dynamic L_wme",
            }
        )

    return controlled


def _make_state_rmse_iteration_callback(m_true, m_background):
    """Create a lightweight callback that records control-space RMSE per iteration."""
    def _callback(x, iteration, cost, grad_norm):
        if x is None:
            return {}

        truth_diff = x.copy()
        truth_diff.axpy(-1.0, m_true)
        truth_rmse = float(np.sqrt(truth_diff.dot(truth_diff) / truth_diff.getSize()))
        truth_diff.destroy()

        bg_diff = x.copy()
        bg_diff.axpy(-1.0, m_background)
        bg_rmse = float(np.sqrt(bg_diff.dot(bg_diff) / bg_diff.getSize()))
        bg_diff.destroy()

        return {
            "analysis_state_rmse": truth_rmse,
            "distance_from_background_rmse": bg_rmse,
        }

    return _callback


def _jsonify_metric_value(value):
    """Convert numpy scalar metrics into JSON-friendly Python scalars."""
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


# ========================================================================
# Eq 38 variance bound: σ_b² ≥ γ / λ_min(G)
# where G = J_wme^T J_wme is the TLM-based Gram matrix
# ========================================================================

def _compute_eq38_from_tlm(forward_model, obs_operator, obs_cov,
                            m_linearize, observations, obs_times,
                            truth_trajectory, truth_jacobians,
                            predictability_gamma=0.1,
                            comm=None, rank=0,
                            component_indices=None):
    """Compute σ_b² from Eq 38 using the truth trajectory's TLM (Spence et al. 2025).

    Uses an already-computed trajectory + Jacobians (typically the truth trajectory)
    to build the WME Gram matrix G[i,j] = a_i^T a_j where a_i = J_wme^T e_i.
    No additional forward solve needed — reuses existing data.

    σ_b² ≥ γ / λ_min(G)  (J_wme absorbs R^{-1/2} and 1/√N)

    Parameters
    ----------
    forward_model : ForwardModelWrapper
        Forward model (for WME QoI construction).
    obs_operator : ObservationOperator
        Point observation operator H.
    obs_cov : CovarianceMatrix
        Observation covariance R.
    m_linearize : PETSc.Vec
        State to linearize around (typically m_true or m_background).
    observations : list of PETSc.Vec
        Observation vectors.
    obs_times : list of int
        Observation time indices (0-based, relative to DA start).
    truth_trajectory : list of PETSc.Vec
        Pre-computed trajectory states.
    truth_jacobians : list
        Pre-computed Jacobians from the trajectory.
    predictability_gamma : float
        Safety factor γ for Eq 38.
    comm : MPI.Comm, optional
        MPI communicator.
    rank : int
        MPI rank for print control.
    component_indices : dict, optional
        When provided, also build per-component Grams and return per-component
        σ_b² bounds. Expected keys: ``"h"`` (local-owned DOF indices for h)
        and ``"uv"`` (local-owned DOF indices for combined u,v). Adjoint
        solves are shared with the scalar path — this is a cheap extension.

    Returns
    -------
    dict
        Keys: sigma_b_sq, lambda_min_G, lambda_max_G, gram_eigvals,
        gram_condition, predictability_gamma.
        When ``component_indices`` is given, additionally:
        sigma_b_sq_h, sigma_b_sq_uv, lambda_min_G_h, lambda_min_G_uv.
    """
    import time as _time
    import gc
    from swe4dvar.data_assimilation.qoi_maps import WeightedMeanErrorQoI

    t0 = _time.perf_counter()

    if rank == 0:
        print(f"  [Eq 38 TLM] Using pre-computed trajectory ({len(truth_trajectory)} states, "
              f"{len(truth_jacobians) if truth_jacobians else 0} Jacobians)")

    # Create WME QoI map and linearize using pre-computed trajectory
    wme_qoi = WeightedMeanErrorQoI(
        forward_model=forward_model,
        observation_operator=obs_operator,
        observations=observations,
        observation_cov=obs_cov,
        obs_times=obs_times,
    )

    linearized_wme = wme_qoi.linearize(
        m_linearize, max(obs_times),
        trajectory=truth_trajectory, jacobians=truth_jacobians,
    )

    # Observation space dimension (= number of obs points, not times)
    n_obs = obs_operator.get_num_observations()

    if rank == 0:
        print(f"  [Eq 38 TLM] Computing Gram matrix ({n_obs} adjoint solves)...")
    t1 = _time.perf_counter()

    # Compute a_i = J_wme^T e_i for each obs basis vector
    adjoint_vectors = []
    from petsc4py import PETSc as _PETSc
    e_i = _PETSc.Vec().createSeq(n_obs)
    e_i.setUp()

    # UV bisector: if component_indices is given, arm the adjoint-sweep
    # diagnostic for EXACTLY the first adjoint call (i=0). We get one
    # complete trace through the backward sweep per-step, then clear so
    # the remaining 57 solves run without log spam.
    from swe4dvar.adjoint.implicit_adjoint import (
        _bisector_set_component_indices as _bis_arm,
        _bisector_clear as _bis_clear,
    )

    for i in range(n_obs):
        e_i.zeroEntries()
        e_i.setValue(i, 1.0)
        e_i.assemblyBegin()
        e_i.assemblyEnd()
        if component_indices is not None and i == 0:
            # Arm on EVERY rank — the diagnostic uses collective MPI allreduce
            # to report global norms AND per-rank local norms. Without arming
            # on every rank, the collective inside _bisector_log deadlocks.
            if rank == 0:
                print("  [uv-bisector] arming adjoint-sweep diagnostic for i=0 only (all ranks)",
                      flush=True)
            _bis_arm(component_indices)
        a_i = linearized_wme.apply_adjoint(e_i)
        if component_indices is not None and i == 0:
            # One-shot trace of the full backward sweep completed. Dump the
            # final per-component norm of the returned adjoint vector and
            # clear the switch.
            _a = a_i.getArray()
            if rank == 0:
                hi  = component_indices["h"]
                uvi = component_indices["uv"]
                print(f"  [uv-bisector] RETURN a_0  "
                      f"||a_0_h||={np.linalg.norm(_a[hi]):.3e}  "
                      f"||a_0_uv||={np.linalg.norm(_a[uvi]):.3e}", flush=True)
            _bis_clear()
        adjoint_vectors.append(a_i)
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = _time.perf_counter() - t1
            if rank == 0:
                print(f"    Adjoint {i+1}/{n_obs} ({elapsed:.1f}s)")
    e_i.destroy()

    t_adj = _time.perf_counter() - t1

    # Form Gram matrix G[i,j] = a_i^T a_j (full state-space dot product)
    G = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(i, n_obs):
            val = adjoint_vectors[i].dot(adjoint_vectors[j])
            G[i, j] = val
            G[j, i] = val

    gram_eigvals = np.linalg.eigvalsh(G)
    lambda_min_G = max(gram_eigvals.min(), 1e-30)
    lambda_max_G = gram_eigvals.max()
    condition = lambda_max_G / lambda_min_G

    # Eq 38: σ_b² ≥ γ / λ_min(G)  (J_wme absorbs R^{-1/2} and 1/√N)
    sigma_b_sq = predictability_gamma / lambda_min_G

    # ------------------------------------------------------------------
    # Optional: per-component Grams (h-only, uv-only) from the SAME
    # adjoint vectors. No extra PDE work — just slice getArray() by
    # component DOF indices and take local dots, then MPI Allreduce.
    # ------------------------------------------------------------------
    component_result = {}
    if component_indices is not None:
        from mpi4py import MPI as _MPI
        h_idx  = np.asarray(component_indices["h"], dtype=np.int64)
        uv_idx = np.asarray(component_indices["uv"], dtype=np.int64)

        # Cache local arrays to avoid per-(i,j) reallocation
        local_h  = [v.getArray()[h_idx]  for v in adjoint_vectors]
        local_uv = [v.getArray()[uv_idx] for v in adjoint_vectors]

        G_h  = np.zeros((n_obs, n_obs))
        G_uv = np.zeros((n_obs, n_obs))
        for i in range(n_obs):
            for j in range(i, n_obs):
                vh  = float(np.dot(local_h[i],  local_h[j]))
                vuv = float(np.dot(local_uv[i], local_uv[j]))
                if comm is not None and comm.Get_size() > 1:
                    vh  = comm.allreduce(vh,  op=_MPI.SUM)
                    vuv = comm.allreduce(vuv, op=_MPI.SUM)
                G_h[i, j]  = G_h[j, i]  = vh
                G_uv[i, j] = G_uv[j, i] = vuv

        # Extract each spectrum. Per-component Grams can be near-rank-deficient
        # (esp. G_h for h-only obs). Floor λ_min with an adaptive γ × λ_max
        # lower bound — same pattern the kernel path uses for its G (line 1701).
        eigs_h  = np.linalg.eigvalsh(G_h)
        eigs_uv = np.linalg.eigvalsh(G_uv)
        lmin_h  = max(eigs_h.min(),  predictability_gamma * eigs_h.max(),  1e-30)
        lmin_uv = max(eigs_uv.min(), predictability_gamma * eigs_uv.max(), 1e-30)
        sigma_b_sq_h  = predictability_gamma / lmin_h
        sigma_b_sq_uv = predictability_gamma / lmin_uv

        component_result = {
            "sigma_b_sq_h":  float(sigma_b_sq_h),
            "sigma_b_sq_uv": float(sigma_b_sq_uv),
            "lambda_min_G_h":  float(lmin_h),
            "lambda_max_G_h":  float(eigs_h.max()),
            "lambda_min_G_uv": float(lmin_uv),
            "lambda_max_G_uv": float(eigs_uv.max()),
        }

    if rank == 0:
        print(f"  [Eq 38 TLM] Adjoint solves: {t_adj:.1f}s")
        print(f"  [Eq 38 TLM] Gram matrix G ({n_obs}×{n_obs}):")
        print(f"    λ_min(G) = {lambda_min_G:.6e}")
        print(f"    λ_max(G) = {lambda_max_G:.6e}")
        print(f"    condition = {condition:.2f}")
        print(f"    spread = {100*(lambda_max_G - lambda_min_G)/max(gram_eigvals.mean(), 1e-30):.1f}%")
        print(f"    rank (>1e-10) = {np.sum(gram_eigvals > 1e-10)}/{n_obs}")
        print(f"  [Eq 38 TLM] σ_b² = γ / λ_min(G) = {predictability_gamma} / {lambda_min_G:.6e}")
        print(f"  [Eq 38 TLM] σ_b² = {sigma_b_sq:.6e}  (σ_b = {np.sqrt(sigma_b_sq):.6e})")
        if component_result:
            print(f"  [Eq 38 TLM] Per-component Grams:")
            print(f"    h:  λ_min(G_h)  = {component_result['lambda_min_G_h']:.6e}  "
                  f"λ_max = {component_result['lambda_max_G_h']:.6e}  "
                  f"→ σ_b²_h  = {component_result['sigma_b_sq_h']:.6e}")
            print(f"    uv: λ_min(G_uv) = {component_result['lambda_min_G_uv']:.6e}  "
                  f"λ_max = {component_result['lambda_max_G_uv']:.6e}  "
                  f"→ σ_b²_uv = {component_result['sigma_b_sq_uv']:.6e}")
        print(f"  [Eq 38 TLM] Total time: {_time.perf_counter() - t0:.1f}s")

    # Cleanup adjoint vectors (don't destroy trajectory — caller owns it)
    for v in adjoint_vectors:
        v.destroy()
    del adjoint_vectors, G
    gc.collect()

    result = {
        "sigma_b_sq": sigma_b_sq,
        "lambda_min_G": float(lambda_min_G),
        "lambda_max_G": float(lambda_max_G),
        "gram_eigvals": [float(v) for v in gram_eigvals],
        "gram_condition": float(condition),
        "predictability_gamma": predictability_gamma,
        "gram_spectrum": _summarize_eigenvalues(gram_eigvals),
    }
    result.update(component_result)
    return result


def _apply_eq38_to_B(B, eq38_result, rank=0, component_indices=None):
    """Inflate B per-component so that every diagonal entry ≥ σ_b² from Eq 38.

    Unlike uniform scaling (which over-inflates components already above the
    bound), this only inflates DOFs whose variance is below the required σ_b².
    Components already satisfying the bound are left unchanged.

    Parameters
    ----------
    B : DiagonalCovariance
        Background covariance (modified in place).
    eq38_result : dict
        Output from _compute_eq38_from_tlm. Must contain 'sigma_b_sq'. When it
        also contains 'sigma_b_sq_h' and 'sigma_b_sq_uv' AND component_indices
        is provided, per-component bounds are applied block-wise instead of the
        single scalar.
    rank : int
        MPI rank for print control.
    component_indices : dict, optional
        Maps 'h' and 'uv' to local-owned DOF indices (aligned with
        B.diagonal.getArray()). Required to activate the per-component path.

    Returns
    -------
    float
        Maximum scale factor applied to any single DOF.
    """
    # Component-aware branch: honor per-component bounds when both the Eq 38
    # result carries them AND the caller passed DOF indices. Otherwise fall
    # through to the historical scalar path.
    component_mode = (
        component_indices is not None
        and "sigma_b_sq_h" in eq38_result
        and "sigma_b_sq_uv" in eq38_result
    )
    if component_mode:
        return _apply_eq38_to_B_components(
            B, eq38_result, component_indices, rank=rank
        )

    sigma_b_sq = eq38_result["sigma_b_sq"]

    diag_arr = B.diagonal.getArray()
    inv_diag_arr = B.inv_diagonal.getArray()

    n_total = diag_arr.size
    below_mask = diag_arr < sigma_b_sq
    n_below = int(np.sum(below_mask))
    n_above = n_total - n_below

    if rank == 0:
        print(f"  [Eq 38] Required σ_b² = {sigma_b_sq:.6e}")
        print(f"  [Eq 38] DOFs below bound: {n_below}/{n_total} "
              f"(above: {n_above})")
        if n_below > 0:
            print(f"  [Eq 38] Below-bound range: [{diag_arr[below_mask].min():.6e}, "
                  f"{diag_arr[below_mask].max():.6e}]")
        if n_above > 0:
            print(f"  [Eq 38] Above-bound range: [{diag_arr[~below_mask].min():.6e}, "
                  f"{diag_arr[~below_mask].max():.6e}]")

    if n_below > 0:
        # Only inflate DOFs that are below the bound
        max_scale = sigma_b_sq / diag_arr[below_mask].min()
        diag_arr[below_mask] = sigma_b_sq
        inv_diag_arr[below_mask] = 1.0 / sigma_b_sq

        B.diagonal.setArray(diag_arr)
        B.diagonal.assemble()
        B.inv_diagonal.setArray(inv_diag_arr)
        B.inv_diagonal.assemble()

        # B.min_eigenvalue() is a COLLECTIVE (comm.allreduce on line 362 of
        # covariance.py). Must be called by every rank, NOT inside the
        # rank==0 guard — that deadlocks at np>=2 (LS6 Step 7b hang,
        # run 3098388 / 3099557 / 3099697).
        min_B = B.min_eigenvalue()
        if rank == 0:
            print(f"  [Eq 38] Inflated {n_below} DOFs to σ_b²={sigma_b_sq:.6e} "
                  f"(max scale: {max_scale:.2f}x)", flush=True)
            print(f"  [Eq 38] min(B) = {min_B:.6e}", flush=True)
    else:
        max_scale = 1.0
        if rank == 0:
            print(f"  [Eq 38] All DOFs already satisfy Eq 38 (no inflation needed)")

    return max_scale


def _apply_eq38_to_B_components(B, eq38_result, component_indices, rank=0):
    """Component-wise counterpart of :func:`_apply_eq38_to_B`.

    Applies σ_b²_h to h DOFs and σ_b²_uv to u,v DOFs independently. A DOF
    within a component is inflated only when its current variance is below
    the bound for that component.

    Returns the maximum per-DOF scale factor actually applied, across all
    components (for logging consistency with the scalar path).
    """
    sig_h  = float(eq38_result["sigma_b_sq_h"])
    sig_uv = float(eq38_result["sigma_b_sq_uv"])

    h_idx  = np.asarray(component_indices["h"],  dtype=np.int64)
    uv_idx = np.asarray(component_indices["uv"], dtype=np.int64)

    diag_arr     = B.diagonal.getArray()
    inv_diag_arr = B.inv_diagonal.getArray()

    diag_h  = diag_arr[h_idx]
    diag_uv = diag_arr[uv_idx]

    below_h_local  = h_idx[diag_h < sig_h]
    below_uv_local = uv_idx[diag_uv < sig_uv]

    n_below_h  = int(below_h_local.size)
    n_below_uv = int(below_uv_local.size)

    max_scale = 1.0
    if n_below_h > 0:
        max_scale = max(max_scale, sig_h / max(diag_arr[below_h_local].min(), 1e-300))
        diag_arr[below_h_local]     = sig_h
        inv_diag_arr[below_h_local] = 1.0 / sig_h
    if n_below_uv > 0:
        max_scale = max(max_scale, sig_uv / max(diag_arr[below_uv_local].min(), 1e-300))
        diag_arr[below_uv_local]     = sig_uv
        inv_diag_arr[below_uv_local] = 1.0 / sig_uv

    if n_below_h > 0 or n_below_uv > 0:
        B.diagonal.setArray(diag_arr)
        B.diagonal.assemble()
        B.inv_diagonal.setArray(inv_diag_arr)
        B.inv_diagonal.assemble()

    # Collective for correct global reporting at np>=2
    min_B = B.min_eigenvalue()

    if rank == 0:
        print(f"  [Eq 38/components] h:  bound σ_b²_h  = {sig_h:.6e}  "
              f"inflated {n_below_h}/{h_idx.size} DOFs", flush=True)
        print(f"  [Eq 38/components] uv: bound σ_b²_uv = {sig_uv:.6e}  "
              f"inflated {n_below_uv}/{uv_idx.size} DOFs", flush=True)
        print(f"  [Eq 38/components] min(B) = {min_B:.6e}", flush=True)

    return max_scale


def run_phase_0(args):
    """Phase 0: Forward Verification.

    Run Shinnecock forward-only simulation for 12h (72 timesteps).
    Verify: mesh loads, GMRES converges, wetting-drying stable, output physical.
    Print mesh stats: node count, element count, interior/boundary node counts.
    """
    from mpi4py import MPI
    from swe4dvar.forward.adcirc_problem import ADCIRCProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)
    comm.Barrier()

    # Simulation parameters for Phase 0: 12h forward run
    dt = 600.0  # seconds
    T_hours = 12.0
    t_f = T_hours * 3600  # seconds
    nt = int(t_f / dt)  # 72 timesteps

    if rank == 0:
        print("=" * 70)
        print("PHASE 0: FORWARD VERIFICATION")
        print("=" * 70)
        print(f"  Time step: {dt} s")
        print(f"  Simulation time: {T_hours} hours ({nt} timesteps)")
        print(f"  ADIOS file: {args.adios_file}")
        print("=" * 70)

    # ----------------------------------------------------------------
    # Step 1: Create problem (loads mesh, bathymetry, boundary data)
    # ----------------------------------------------------------------
    if rank == 0:
        print("\n--- Step 1: Loading mesh and creating problem ---")

    t_start = time.time()
    prob = ADCIRCProblem(
        adios_file=args.adios_file,
        spherical=True,
        solution_var="h",
        friction_law="mannings",
        wd=True,
        wd_alpha=1.5,
        dt=dt,
        bathy_adjustment=0,
        nt=nt,
        dramp=2.0,
    )
    t_problem = time.time() - t_start

    if rank == 0:
        print(f"  Problem created in {t_problem:.2f} s")

    # ----------------------------------------------------------------
    # Step 2: Create solver
    # ----------------------------------------------------------------
    if rank == 0:
        print("\n--- Step 2: Creating DG solver ---")

    t_start = time.time()
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])
    t_solver = time.time() - t_start

    if rank == 0:
        print(f"  Solver created in {t_solver:.2f} s")

    # ----------------------------------------------------------------
    # Step 3: Print mesh statistics
    # ----------------------------------------------------------------
    mesh = prob.mesh
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # Node and element counts
    mesh.topology.create_connectivity(tdim, 0)
    mesh.topology.create_connectivity(fdim, 0)

    num_nodes_local = mesh.topology.index_map(0).size_local
    num_cells_local = mesh.topology.index_map(tdim).size_local

    # Global counts via MPI
    num_nodes_global = comm.allreduce(num_nodes_local, op=MPI.SUM)
    num_cells_global = comm.allreduce(num_cells_local, op=MPI.SUM)

    # Function space DOFs
    V = solver.V
    total_dofs_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    total_dofs_global = comm.allreduce(total_dofs_local, op=MPI.SUM)

    # Sub-space DOF counts
    V_h, _ = V.sub(0).collapse()
    h_dofs_local = V_h.dofmap.index_map.size_local * V_h.dofmap.index_map_bs
    h_dofs_global = comm.allreduce(h_dofs_local, op=MPI.SUM)

    # Boundary DOFs
    n_open_boundary = len(prob.dof_open) if hasattr(prob, "dof_open") else 0
    n_open_global = comm.allreduce(n_open_boundary, op=MPI.SUM)

    # Interior nodes (approximate via boundary detection)
    from dolfinx import mesh as dmesh
    boundary_facets = dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    n_boundary_facets_local = len(boundary_facets)
    n_boundary_facets_global = comm.allreduce(n_boundary_facets_local, op=MPI.SUM)

    # Bathymetry statistics
    depth_array = prob.depth.x.array
    depth_min_local = depth_array.min() if len(depth_array) > 0 else 1e10
    depth_max_local = depth_array.max() if len(depth_array) > 0 else -1e10
    depth_mean_local = depth_array.sum() if len(depth_array) > 0 else 0.0
    depth_count_local = len(depth_array)

    depth_min = comm.allreduce(depth_min_local, op=MPI.MIN)
    depth_max = comm.allreduce(depth_max_local, op=MPI.MAX)
    depth_sum = comm.allreduce(depth_mean_local, op=MPI.SUM)
    depth_count = comm.allreduce(depth_count_local, op=MPI.SUM)
    depth_mean = depth_sum / depth_count if depth_count > 0 else 0.0

    # Count nodes with negative depth (potential wetting-drying regions)
    n_negative_depth_local = int(np.sum(depth_array < 0))
    n_negative_depth = comm.allreduce(n_negative_depth_local, op=MPI.SUM)

    # Manning's n statistics
    has_mannings = hasattr(prob, "mannings_n")
    if has_mannings:
        mn_array = prob.mannings_n.x.array
        mn_min_local = mn_array.min() if len(mn_array) > 0 else 1e10
        mn_max_local = mn_array.max() if len(mn_array) > 0 else -1e10
        mn_min = comm.allreduce(mn_min_local, op=MPI.MIN)
        mn_max = comm.allreduce(mn_max_local, op=MPI.MAX)

    if rank == 0:
        print("\n--- Step 3: Mesh Statistics ---")
        print(f"  Mesh nodes:              {num_nodes_global}")
        print(f"  Mesh triangles:          {num_cells_global}")
        print(f"  Boundary facets:         {n_boundary_facets_global}")
        print(f"  DG degree:               p=1")
        print(f"  DOFs per field (h):      {h_dofs_global}")
        print(f"  Total DOFs (h, ux, uy):  {total_dofs_global}")
        print(f"  Open boundary DOFs:      {n_open_global}")
        print(f"  Wetting-drying:          Enabled (alpha={prob.wd_alpha})")
        print(f"  Bathymetry range:        [{depth_min:.2f}, {depth_max:.2f}] m")
        print(f"  Bathymetry mean:         {depth_mean:.2f} m")
        print(f"  Negative depth nodes:    {n_negative_depth} (WD-active regions)")
        if has_mannings:
            print(f"  Manning's n range:       [{mn_min:.6f}, {mn_max:.6f}]")
        else:
            print(f"  Manning's n:             Not loaded (using constant TAU={prob.TAU})")
        print(f"  Tidal constituents:      {len(prob.boundaries.frequency)}")
        print(f"  Spherical projection:    Enabled (lat0={prob.lat0})")
        print(f"  MPI processes:           {comm.Get_size()}")

    # ----------------------------------------------------------------
    # Step 4: Set up observation station for verification
    # ----------------------------------------------------------------
    from swe4dvar.physics.constants import R

    stations = np.array([[-72.476519, 40.840969, 0.0]])
    stations_rad = np.deg2rad(stations)
    lat0 = prob.lat0
    stations_rad[:, 0] *= R * np.cos(np.deg2rad(lat0))
    stations_rad[:, 1] *= R

    # ----------------------------------------------------------------
    # Step 5: Run forward simulation
    # ----------------------------------------------------------------
    if rank == 0:
        print("\n--- Step 4: Running forward simulation ---")

    params = get_default_solver_params(
        rtol=1e-5,
        atol=1e-6,
        max_it=10,
        relaxation_parameter=1.0,
        comm=comm,
        error_if_not_converged=True,
    )

    # Try with stations; fall back to no stations if interpolation fails
    use_stations = True
    try:
        # Test station initialization by doing a dry run
        solver.init_stations(stations_rad)
    except Exception as e:
        if rank == 0:
            print(f"  Warning: Station init failed ({e}), running without stations")
        use_stations = False

    t_start = time.time()
    u_final, vals = solver.time_loop(
        solver_parameters=params,
        stations=stations_rad if use_stations else [],
        plot_every=1,
        plot_name="phase0_shinnecock",
        save_state=False,
        adjoint_method=False,
        store_jacobians=False,
        monitor_progress=(rank == 0),
        enable_video=False,
        newton_diagnostics_config={
            "print_to_console": False,
            "store_history": True,
        },
    )
    t_simulation = time.time() - t_start

    # ----------------------------------------------------------------
    # Step 6: Analyze results
    # ----------------------------------------------------------------
    if rank == 0:
        print(f"\n  Simulation completed in {t_simulation:.2f} s")
        print(f"  Time per timestep: {t_simulation / nt:.3f} s")
        print()

    # Newton convergence diagnostics
    diagnostics = solver.solver.diagnostics if hasattr(solver.solver, "diagnostics") else None
    newton_stats = {}

    if diagnostics is not None and hasattr(diagnostics, "history") and diagnostics.history:
        iterations_per_step = [entry["iterations"] for entry in diagnostics.history]
        converged_flags = [entry.get("converged", True) for entry in diagnostics.history]
        n_converged = sum(converged_flags)
        n_total = len(converged_flags)

        newton_stats = {
            "min_iterations": int(min(iterations_per_step)),
            "max_iterations": int(max(iterations_per_step)),
            "mean_iterations": float(np.mean(iterations_per_step)),
            "total_timesteps": n_total,
            "converged_timesteps": n_converged,
            "all_converged": n_converged == n_total,
        }
    else:
        # If no diagnostics, simulation completed without error = all converged
        newton_stats = {
            "min_iterations": -1,
            "max_iterations": -1,
            "mean_iterations": -1,
            "total_timesteps": nt,
            "converged_timesteps": nt,
            "all_converged": True,
        }

    if rank == 0:
        print("--- Step 5: Verification Results ---")
        print()
        print("  Newton Solver Convergence:")
        if newton_stats["min_iterations"] >= 0:
            print(f"    Min iterations/step:    {newton_stats['min_iterations']}")
            print(f"    Max iterations/step:    {newton_stats['max_iterations']}")
            print(f"    Mean iterations/step:   {newton_stats['mean_iterations']:.2f}")
        print(f"    Converged timesteps:    {newton_stats['converged_timesteps']}/{newton_stats['total_timesteps']}")
        print(f"    All converged:          {newton_stats['all_converged']}")

    # Solution diagnostics from station data
    station_results = {}
    has_station_data = (
        rank == 0
        and vals is not None
        and len(vals) > 0
        and vals.ndim >= 2
        and vals.shape[1] > 0
    )
    if has_station_data:
        h_vals = vals[:nt + 1, 0, 0]  # h at station
        ux_vals = vals[:nt + 1, 0, 1]  # ux at station
        uy_vals = vals[:nt + 1, 0, 2]  # uy at station

        station_results = {
            "h_min": float(np.min(h_vals)),
            "h_max": float(np.max(h_vals)),
            "h_mean": float(np.mean(h_vals)),
            "h_range": float(np.max(h_vals) - np.min(h_vals)),
            "ux_min": float(np.min(ux_vals)),
            "ux_max": float(np.max(ux_vals)),
            "uy_min": float(np.min(uy_vals)),
            "uy_max": float(np.max(uy_vals)),
        }

        print()
        print("  Station Data (channel mid-point):")
        print(f"    h range:    [{station_results['h_min']:.4f}, {station_results['h_max']:.4f}] m")
        print(f"    h mean:     {station_results['h_mean']:.4f} m")
        print(f"    h amplitude:{station_results['h_range']:.4f} m")
        print(f"    ux range:   [{station_results['ux_min']:.6f}, {station_results['ux_max']:.6f}] m/s")
        print(f"    uy range:   [{station_results['uy_min']:.6f}, {station_results['uy_max']:.6f}] m/s")

        # Check for physical plausibility
        tidal_visible = station_results["h_range"] > 0.001  # At least 1mm variation
        no_blowup = abs(station_results["h_max"]) < 100  # No unreasonable values
        velocities_reasonable = abs(station_results["ux_max"]) < 10 and abs(station_results["uy_max"]) < 10

        print()
        print("  Physical Plausibility Checks:")
        print(f"    Tidal oscillation visible (h range > 1mm): {'PASS' if tidal_visible else 'FAIL'}")
        print(f"    No blowup (|h| < 100m):                    {'PASS' if no_blowup else 'FAIL'}")
        print(f"    Velocities reasonable (|u| < 10 m/s):      {'PASS' if velocities_reasonable else 'FAIL'}")
    elif rank == 0:
        print()
        print("  Station Data: Not available (station init failed, see warning above)")
        print("  Using global field diagnostics for plausibility checks instead.")

    # Global solution diagnostics
    u_array = u_final.x.array
    h_sub, h_map = V.sub(0).collapse()
    h_final = u_array[h_map]

    h_min_local = h_final.min() if len(h_final) > 0 else 1e10
    h_max_local = h_final.max() if len(h_final) > 0 else -1e10
    h_min_global = comm.allreduce(h_min_local, op=MPI.MIN)
    h_max_global = comm.allreduce(h_max_local, op=MPI.MAX)

    # Count WD-active cells at final time
    n_wd_active_local = int(np.sum(h_final < prob.wd_alpha))
    n_wd_active_global = comm.allreduce(n_wd_active_local, op=MPI.SUM)

    wd_results = {
        "h_min_global": float(h_min_global),
        "h_max_global": float(h_max_global),
        "n_wd_active_dofs": n_wd_active_global,
        "h_dofs_total": h_dofs_global,
        "wd_fraction": float(n_wd_active_global / h_dofs_global) if h_dofs_global > 0 else 0,
    }

    if rank == 0:
        print()
        print("  Wetting-Drying Stability:")
        print(f"    Global h range at t=12h:  [{wd_results['h_min_global']:.4f}, {wd_results['h_max_global']:.4f}] m")
        print(f"    WD-active DOFs (h < alpha):{wd_results['n_wd_active_dofs']}/{wd_results['h_dofs_total']} ({wd_results['wd_fraction']:.1%})")
        wd_stable = h_min_global > -10  # No catastrophic negative depths
        print(f"    WD stable (h > -10m):      {'PASS' if wd_stable else 'FAIL'}")

    # ----------------------------------------------------------------
    # Step 7: Save results
    # ----------------------------------------------------------------
    if rank == 0:
        results = {
            "phase": 0,
            "status": "success",
            "mesh_stats": {
                "num_nodes": num_nodes_global,
                "num_cells": num_cells_global,
                "num_boundary_facets": n_boundary_facets_global,
                "h_dofs": h_dofs_global,
                "total_dofs": total_dofs_global,
                "open_boundary_dofs": n_open_global,
                "negative_depth_nodes": n_negative_depth,
                "has_mannings_n": has_mannings,
                "n_tidal_constituents": len(prob.boundaries.frequency),
            },
            "bathymetry_stats": {
                "depth_min": float(depth_min),
                "depth_max": float(depth_max),
                "depth_mean": float(depth_mean),
            },
            "simulation_config": {
                "dt": dt,
                "T_hours": T_hours,
                "nt": nt,
                "solver_type": "DG",
                "theta": 1.0,
                "wd_alpha": prob.wd_alpha,
                "dramp": prob.dramp,
            },
            "timing": {
                "problem_setup_s": t_problem,
                "solver_setup_s": t_solver,
                "simulation_s": t_simulation,
                "time_per_step_s": t_simulation / nt,
                "total_s": t_problem + t_solver + t_simulation,
            },
            "newton_convergence": newton_stats,
            "station_results": station_results,
            "wd_stability": wd_results,
        }

        results_file = output_dir / "data" / "phase0_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print()
        print("=" * 70)
        print("PHASE 0: FORWARD VERIFICATION COMPLETE")
        print("=" * 70)
        # Station checks only apply when station data was collected
        if station_results:
            station_pass = (
                station_results.get("h_range", 0) > 0.001
                and abs(station_results.get("h_max", 999)) < 100
            )
        else:
            station_pass = True  # Skip station checks if no station data
        all_pass = (
            newton_stats["all_converged"]
            and station_pass
            and h_min_global > -10
        )
        print(f"  Overall result: {'PASS' if all_pass else 'FAIL'}")
        print(f"  Results saved to: {results_file}")
        print("=" * 70)

        return results

    return None


def run_phase_1(args):
    """Phase 1: Single-Window 4D-Var, No Model Error (Inverse Crime).

    1. Run 48h ramp warm-up (288 timesteps) to reach realistic tidal state
    2. Run TwinExperiment 4D-Var on 12h post-ramp window (72 timesteps)
    3. No physics perturbation (inverse crime baseline)
    4. Expected: high error reduction (>5%)
    """
    import os
    os.environ.setdefault("CC", "/usr/bin/clang")

    from mpi4py import MPI
    from swe4dvar.forward.adcirc_problem import ADCIRCProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params
    from experiments.twin_experiment import (
        TwinExperiment, TwinExperimentConfig, ForwardModelWrapper,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)
    comm.Barrier()

    # ================================================================
    # Configuration - FULL PHASE 1 (2-3 hour runtime)
    # ================================================================
    dt = 600.0          # 10-min timestep
    nt_ramp = 144       # 24h ramp (144 × 600s = 86400s) - INTERMEDIATE TEST
    nt_da = 12          # 2h DA window (12 × 600s = 7200s) - reduced for memory
    nt_total = nt_ramp + nt_da  # 180 timesteps = 30h

    obs_fraction = 0.1      # 10% obs points (~306 points) - more observation constraint
    obs_frequency = 6       # Every 6 timesteps (= every hour) per plan
    obs_noise_level = 0.01  # 1% noise
    background_error_std = 0.02   # 2% perturbation RMS (larger gap for optimizer to find)
    max_iterations = 15     # Need more iters for full convergence with large perturbation

    if rank == 0:
        print("=" * 70)
        print("PHASE 1: SINGLE-WINDOW 4D-VAR, NO MODEL ERROR (INVERSE CRIME)")
        print("=" * 70)
        print(f"  Total simulation: {nt_total} timesteps ({nt_total * dt / 3600:.0f}h)")
        print(f"  Ramp period: {nt_ramp} timesteps ({nt_ramp * dt / 3600:.0f}h)")
        print(f"  DA window: {nt_da} timesteps ({nt_da * dt / 3600:.0f}h)")
        print(f"  Obs fraction: {obs_fraction} (~{int(obs_fraction * 3070)} points)")
        print(f"  Obs frequency: every {obs_frequency} timesteps ({obs_frequency * dt / 3600:.0f}h)")
        print(f"  Background error std: {background_error_std}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Model error: NONE (inverse crime)")
        print("=" * 70)

    # ================================================================
    # Step 1: Create problem and solver
    # ================================================================
    if rank == 0:
        print("\n--- Step 1: Creating problem and solver ---")

    prob = ADCIRCProblem(
        adios_file=args.adios_file,
        spherical=True,
        solution_var="h",
        friction_law="mannings",
        wd=True,
        wd_alpha=1.5,
        dt=dt,
        bathy_adjustment=0,
        nt=nt_total,
        dramp=2.0,
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])

    # GMRES+ILU for warm-up (fast, works fine from equilibrium IC)
    warmup_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=10,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=True,
    )

    # GMRES+bjacobi for DA forward solves - iterative solver uses far less RAM
    # than direct LU (MUMPS). Starting from m_true means the state is physical,
    # so GMRES converges without the instability seen with perturbed m_background.
    # max_it=25 and reduction_it=10: if Newton stalls after 10 iters, halve relax param.
    da_solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=35,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=False,
        reduction_it=10,
    )

    if rank == 0:
        n_nodes = prob.mesh.topology.index_map(0).size_local
        n_dofs = len(solver.u.x.array)
        print(f"  Mesh: {n_nodes} nodes, {n_dofs} DOFs")

    # ================================================================
    # Step 2: Warm-up (48h ramp period)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 2: Running {nt_ramp}-step warm-up ({nt_ramp * dt / 3600:.0f}h ramp) ---")

    t_warmup_start = time.time()
    prob.nt = nt_ramp
    solver.time_loop(
        solver_parameters=warmup_params,
        stations=[],
        plot_every=9999,
        save_state=False,
        store_jacobians=False,
        enable_video=False,
        monitor_progress=(rank == 0),
    )
    t_warmup = time.time() - t_warmup_start
    t_da_start = prob.t  # Should be 172800.0 (48h)

    if rank == 0:
        print(f"  Warm-up completed in {t_warmup:.1f}s (t = {t_da_start:.0f}s = {t_da_start/3600:.1f}h)")

    # ================================================================
    # Step 3: Set up TwinExperiment for DA window
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 3: Setting up DA window ({nt_da} timesteps from t={t_da_start/3600:.0f}h) ---")

    prob.nt = nt_da  # DA window length

    config = TwinExperimentConfig(
        method="4dvar",
        obs_fraction=obs_fraction,
        obs_frequency=obs_frequency,
        obs_noise_level=obs_noise_level,
        background_error_std=background_error_std,
        background_correlation_length=500.0,  # Smoothed perturbation (Gaussian, L=500m)
        max_iterations=max_iterations,
        gradient_tolerance=1e-3,
        cost_tolerance=1e-4,
        n_windows=1,
        perturb_friction=False,
        friction_scale_factor=1.0,
        use_bounds=True,
        h_min=0.01,
        interior_only=True,
        component_aware_cov=True,  # Match B variances to component-specific perturbation magnitudes
        verbose=(rank == 0),
    )

    exp = TwinExperiment(
        problem=prob, solver=solver, config=config,
        solver_params=da_solver_params, comm=comm,
    )

    # ================================================================
    # Step 4: Generate truth for DA window (72 timesteps from warm state)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 4: Generating truth trajectory for DA window ({nt_da} steps) ---")

    t_truth_start = time.time()
    exp._generate_truth()  # Runs 36 steps from warm state, store_jacobians=False
    t_truth = time.time() - t_truth_start

    # Explicitly clear solver storage to free memory before optimization
    exp.solver.storage.clear()

    if rank == 0:
        print(f"  Truth: {len(exp.truth_trajectory)} states, generated in {t_truth:.1f}s")

    # ================================================================
    # Step 5: Setup observations, background, covariances
    # ================================================================
    if rank == 0:
        print("\n--- Step 5: Setting up observations and background ---")

    obs_points, obs_operator, obs_times = exp._setup_observations()
    exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
    background_error = exp._setup_background()
    B, R, B_lwme = exp._setup_covariances(obs_operator, obs_noise_stds)

    n_obs = obs_operator.get_num_observations()
    if rank == 0:
        print(f"  Observation points: {n_obs}")
        print(f"  Observation times: {obs_times}")
        print(f"  Background RMS error: {background_error:.6f}")

    # ================================================================
    # Step 6: Create forward model with CORRECT t_start for tidal forcing
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 6: Creating forward model (t_start={t_da_start:.0f}s) ---")

    forward_model = exp._create_forward_model(t_start=t_da_start)

    # ================================================================
    # Step 7: Setup cost function (with M^{-1} preconditioning)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 7: Setting up 4D-Var cost function ---")

    cost_function = exp._setup_cost_function(
        forward_model, obs_operator, B, R, obs_times, B_lwme=B_lwme
    )

    # ================================================================
    # Step 7b: Attach gradient smoother (filters high-freq adjoint gradient)
    # ================================================================
    # The perturbation is spatially smoothed (L=500m) but B is diagonal,
    # so the adjoint gradient has high-frequency components that B^{-1}
    # amplifies. Smoothing the gradient is equivalent to B-preconditioning
    # and keeps L-BFGS search directions within Newton's convergence basin.
    gradient_smoothing_length = 500.0  # Match perturbation correlation length for consistent preconditioning
    if config.background_correlation_length > 0:
        if rank == 0:
            print("\n--- Step 7b: Building gradient smoother ---")

        h_indices, u_indices, v_indices = exp._get_component_dof_indices(owned_only=True)
        smoothing_matrix = exp._build_smoothing_matrix(
            h_indices, gradient_smoothing_length
        )

        def gradient_smoother(grad_array):
            """Apply spatial smoothing to each component of the gradient."""
            smoothed = grad_array.copy()
            smoothed[h_indices] = smoothing_matrix @ grad_array[h_indices]
            smoothed[u_indices] = smoothing_matrix @ grad_array[u_indices]
            smoothed[v_indices] = smoothing_matrix @ grad_array[v_indices]
            return smoothed

        # Set smoother on the inner FourDVarCost (may be wrapped by ZeroBoundaryGradientCost)
        inner_cost = cost_function
        while hasattr(inner_cost, 'base_cost'):
            inner_cost = inner_cost.base_cost
        inner_cost.gradient_smoother = gradient_smoother

        if rank == 0:
            print(f"  Gradient smoother attached (L={gradient_smoothing_length}m)")

    # ================================================================
    # Step 8: Run optimization
    # ================================================================
    if rank == 0:
        print("\n--- Step 8: Running L-BFGS optimization ---")

    optimizer, opt_time = exp._run_optimization(cost_function)

    cost_history = [h["cost"] for h in optimizer.convergence_history]
    gradient_history = [h["grad_norm"] for h in optimizer.convergence_history]

    if rank == 0:
        print(f"  Iterations: {optimizer.iteration}")
        print(f"  Converged: {optimizer.converged}")
        print(f"  Optimization time: {opt_time:.1f}s")
        if cost_history:
            print(f"  Cost: {cost_history[0]:.6f} → {cost_history[-1]:.6f}")
        if gradient_history:
            print(f"  ||grad||: {gradient_history[0]:.6e} → {gradient_history[-1]:.6e}")

    # ================================================================
    # Step 9: Evaluate results
    # ================================================================
    if rank == 0:
        print("\n--- Step 9: Evaluating results ---")

    if cost_history and cost_history[-1] >= 1e19:
        if rank == 0:
            print("  WARNING: Optimization failed (cost diverged)")
        analysis_error = background_error
        error_reduction = 0.0
        innov_mean = innov_std = mean_rmse = data_misfit = 0.0
    else:
        analysis_error, error_reduction, innov_mean, innov_std, mean_rmse, data_misfit = (
            exp._evaluate_results(obs_operator, obs_times, background_error)
        )

    total_time = t_warmup + t_truth + opt_time

    if rank == 0:
        print(f"\n  Background error:  {background_error:.6f}")
        print(f"  Analysis error:    {analysis_error:.6f}")
        print(f"  Error reduction:   {error_reduction:.1f}%")
        print(f"  Mean RMSE:         {mean_rmse:.6f}")
        print(f"  Data misfit:       {data_misfit:.6f}")

    # ================================================================
    # Step 10: Save results
    # ================================================================
    if rank == 0:
        results = {
            "phase": 1,
            "status": "success",
            "method": "4dvar",
            "model_error": False,
            "config": {
                "dt": dt,
                "nt_ramp": nt_ramp,
                "nt_da": nt_da,
                "t_da_start_s": t_da_start,
                "obs_fraction": obs_fraction,
                "obs_frequency": obs_frequency,
                "obs_noise_level": obs_noise_level,
                "background_error_std": background_error_std,
                "max_iterations": max_iterations,
                "n_obs_points": n_obs,
                "obs_times": obs_times,
            },
            "results": {
                "background_error": float(background_error),
                "analysis_error": float(analysis_error),
                "error_reduction": float(error_reduction),
                "mean_rmse": float(mean_rmse),
                "data_misfit": float(data_misfit),
                "innovation_mean": float(innov_mean),
                "innovation_std": float(innov_std),
                "num_iterations": int(optimizer.iteration),
                "converged": bool(optimizer.converged),
            },
            "convergence": {
                "cost_history": [float(c) for c in cost_history],
                "gradient_norm_history": [float(g) for g in gradient_history],
                "analysis_state_rmse_history": [
                    float(h["analysis_state_rmse"])
                    for h in optimizer.convergence_history
                    if "analysis_state_rmse" in h
                ],
                "distance_from_background_rmse_history": [
                    float(h["distance_from_background_rmse"])
                    for h in optimizer.convergence_history
                    if "distance_from_background_rmse" in h
                ],
            },
            "timing": {
                "warmup_s": float(t_warmup),
                "truth_generation_s": float(t_truth),
                "optimization_s": float(opt_time),
                "total_s": float(total_time),
            },
        }

        results_file = output_dir / "data" / "phase1_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 70}")
        phase_pass = error_reduction > 5.0 and optimizer.converged
        print(f"PHASE 1: {'PASS' if phase_pass else 'NEEDS REVIEW'}")
        print(f"{'=' * 70}")
        print(f"  Error reduction: {error_reduction:.1f}% (expected >5%)")
        print(f"  Iterations: {optimizer.iteration}")
        print(f"  Converged: {optimizer.converged}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Results saved to: {results_file}")
        print(f"{'=' * 70}")

        return results

    return None


def run_phase_2(args):
    """Phase 2: Single-Window 4D-Var, With Model Error.

    Same as Phase 1 but with friction perturbation (Manning's n × 1.15).
    Truth is generated with TRUE friction, then friction is scaled by 1.15.
    DA uses the perturbed friction model, so it cannot perfectly recover truth.
    Expected: lower error reduction than Phase 1.
    """
    import os
    os.environ.setdefault("CC", "/usr/bin/clang")

    from mpi4py import MPI
    from swe4dvar.forward.adcirc_problem import ADCIRCProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params
    from experiments.twin_experiment import (
        TwinExperiment, TwinExperimentConfig, ForwardModelWrapper,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)
    comm.Barrier()

    # ================================================================
    # Configuration - same as Phase 1 + friction perturbation
    # ================================================================
    dt = 600.0
    nt_ramp = 144       # 24h ramp
    nt_da = 12          # 2h DA window
    nt_total = nt_ramp + nt_da

    obs_fraction = 0.1
    obs_frequency = 6
    obs_noise_level = 0.01
    background_error_std = 0.02
    max_iterations = 15
    friction_scale_factor = 1.15  # 15% Manning's n error

    if rank == 0:
        print("=" * 70)
        print("PHASE 2: SINGLE-WINDOW 4D-VAR, WITH MODEL ERROR")
        print("=" * 70)
        print(f"  Total simulation: {nt_total} timesteps ({nt_total * dt / 3600:.0f}h)")
        print(f"  Ramp period: {nt_ramp} timesteps ({nt_ramp * dt / 3600:.0f}h)")
        print(f"  DA window: {nt_da} timesteps ({nt_da * dt / 3600:.0f}h)")
        print(f"  Obs fraction: {obs_fraction} (~{int(obs_fraction * 3070)} points)")
        print(f"  Obs frequency: every {obs_frequency} timesteps ({obs_frequency * dt / 3600:.0f}h)")
        print(f"  Background error std: {background_error_std}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Model error: Manning's n × {friction_scale_factor}")
        print("=" * 70)

    # ================================================================
    # Step 1: Create problem and solver
    # ================================================================
    if rank == 0:
        print("\n--- Step 1: Creating problem and solver ---")

    prob = ADCIRCProblem(
        adios_file=args.adios_file,
        spherical=True,
        solution_var="h",
        friction_law="mannings",
        wd=True,
        wd_alpha=1.5,
        dt=dt,
        bathy_adjustment=0,
        nt=nt_total,
        dramp=2.0,
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])

    warmup_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=10,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=True,
    )

    da_solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=35,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=False,
        reduction_it=10,
    )

    if rank == 0:
        n_nodes = prob.mesh.topology.index_map(0).size_local
        n_dofs = len(solver.u.x.array)
        print(f"  Mesh: {n_nodes} nodes, {n_dofs} DOFs")

    # ================================================================
    # Step 2: Warm-up (ramp period with TRUE friction)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 2: Running {nt_ramp}-step warm-up ({nt_ramp * dt / 3600:.0f}h ramp) ---")

    t_warmup_start = time.time()
    prob.nt = nt_ramp
    solver.time_loop(
        solver_parameters=warmup_params,
        stations=[],
        plot_every=9999,
        save_state=False,
        store_jacobians=False,
        enable_video=False,
        monitor_progress=(rank == 0),
    )
    t_warmup = time.time() - t_warmup_start
    t_da_start = prob.t

    if rank == 0:
        print(f"  Warm-up completed in {t_warmup:.1f}s (t = {t_da_start:.0f}s = {t_da_start/3600:.1f}h)")

    # ================================================================
    # Step 3: Set up TwinExperiment for DA window (with friction perturbation)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 3: Setting up DA window ({nt_da} timesteps from t={t_da_start/3600:.0f}h) ---")

    prob.nt = nt_da

    config = TwinExperimentConfig(
        method="4dvar",
        obs_fraction=obs_fraction,
        obs_frequency=obs_frequency,
        obs_noise_level=obs_noise_level,
        background_error_std=background_error_std,
        background_correlation_length=500.0,
        max_iterations=max_iterations,
        gradient_tolerance=1e-3,
        cost_tolerance=1e-4,
        n_windows=1,
        perturb_friction=True,  # KEY DIFFERENCE: enable friction perturbation
        friction_scale_factor=friction_scale_factor,  # 15% Manning's n error
        use_bounds=True,
        h_min=0.01,
        interior_only=True,
        component_aware_cov=True,
        verbose=(rank == 0),
    )

    exp = TwinExperiment(
        problem=prob, solver=solver, config=config,
        solver_params=da_solver_params, comm=comm,
    )

    # ================================================================
    # Step 4: Generate truth for DA window (with TRUE friction)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 4: Generating truth trajectory ({nt_da} steps, TRUE friction) ---")

    t_truth_start = time.time()
    exp._generate_truth()
    t_truth = time.time() - t_truth_start

    exp.solver.storage.clear()

    if rank == 0:
        print(f"  Truth: {len(exp.truth_trajectory)} states, generated in {t_truth:.1f}s")

    # ================================================================
    # Step 4b: Apply friction perturbation (scale Manning's n by 1.15)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 4b: Applying friction perturbation (×{friction_scale_factor}) ---")

    exp._apply_physics_perturbations()

    # ================================================================
    # Step 5: Setup observations, background, covariances
    # ================================================================
    if rank == 0:
        print("\n--- Step 5: Setting up observations and background ---")

    obs_points, obs_operator, obs_times = exp._setup_observations()
    exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
    background_error = exp._setup_background()
    B, R, B_lwme = exp._setup_covariances(obs_operator, obs_noise_stds)

    n_obs = obs_operator.get_num_observations()
    if rank == 0:
        print(f"  Observation points: {n_obs}")
        print(f"  Observation times: {obs_times}")
        print(f"  Background RMS error: {background_error:.6f}")

    # ================================================================
    # Step 6: Create forward model (with PERTURBED friction)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 6: Creating forward model (t_start={t_da_start:.0f}s, perturbed friction) ---")

    forward_model = exp._create_forward_model(t_start=t_da_start)

    # ================================================================
    # Step 7: Setup cost function
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 7: Setting up 4D-Var cost function ---")

    cost_function = exp._setup_cost_function(
        forward_model, obs_operator, B, R, obs_times, B_lwme=B_lwme
    )

    # ================================================================
    # Step 7b: Attach gradient smoother (same as Phase 1)
    # ================================================================
    gradient_smoothing_length = 500.0
    if config.background_correlation_length > 0:
        if rank == 0:
            print("\n--- Step 7b: Building gradient smoother ---")

        h_indices, u_indices, v_indices = exp._get_component_dof_indices(owned_only=True)
        smoothing_matrix = exp._build_smoothing_matrix(
            h_indices, gradient_smoothing_length
        )

        def gradient_smoother(grad_array):
            """Apply spatial smoothing to each component of the gradient."""
            smoothed = grad_array.copy()
            smoothed[h_indices] = smoothing_matrix @ grad_array[h_indices]
            smoothed[u_indices] = smoothing_matrix @ grad_array[u_indices]
            smoothed[v_indices] = smoothing_matrix @ grad_array[v_indices]
            return smoothed

        inner_cost = cost_function
        while hasattr(inner_cost, 'base_cost'):
            inner_cost = inner_cost.base_cost
        inner_cost.gradient_smoother = gradient_smoother

        if rank == 0:
            print(f"  Gradient smoother attached (L={gradient_smoothing_length}m)")

    # ================================================================
    # Step 8: Run optimization
    # ================================================================
    if rank == 0:
        print("\n--- Step 8: Running L-BFGS optimization ---")

    optimizer, opt_time = exp._run_optimization(cost_function)

    cost_history = [h["cost"] for h in optimizer.convergence_history]
    gradient_history = [h["grad_norm"] for h in optimizer.convergence_history]

    if rank == 0:
        print(f"  Iterations: {optimizer.iteration}")
        print(f"  Converged: {optimizer.converged}")
        print(f"  Optimization time: {opt_time:.1f}s")
        if cost_history:
            print(f"  Cost: {cost_history[0]:.6f} → {cost_history[-1]:.6f}")
        if gradient_history:
            print(f"  ||grad||: {gradient_history[0]:.6e} → {gradient_history[-1]:.6e}")

    # ================================================================
    # Step 9: Evaluate results
    # ================================================================
    if rank == 0:
        print("\n--- Step 9: Evaluating results ---")

    if cost_history and cost_history[-1] >= 1e19:
        if rank == 0:
            print("  WARNING: Optimization failed (cost diverged)")
        analysis_error = background_error
        error_reduction = 0.0
        innov_mean = innov_std = mean_rmse = data_misfit = 0.0
    else:
        analysis_error, error_reduction, innov_mean, innov_std, mean_rmse, data_misfit = (
            exp._evaluate_results(obs_operator, obs_times, background_error)
        )

    total_time = t_warmup + t_truth + opt_time

    if rank == 0:
        print(f"\n  Background error:  {background_error:.6f}")
        print(f"  Analysis error:    {analysis_error:.6f}")
        print(f"  Error reduction:   {error_reduction:.1f}%")
        print(f"  Mean RMSE:         {mean_rmse:.6f}")
        print(f"  Data misfit:       {data_misfit:.6f}")

    # ================================================================
    # Step 10: Save results
    # ================================================================
    if rank == 0:
        results = {
            "phase": 2,
            "status": "success",
            "method": "4dvar",
            "model_error": True,
            "model_error_type": "friction",
            "friction_scale_factor": friction_scale_factor,
            "config": {
                "dt": dt,
                "nt_ramp": nt_ramp,
                "nt_da": nt_da,
                "t_da_start_s": t_da_start,
                "obs_fraction": obs_fraction,
                "obs_frequency": obs_frequency,
                "obs_noise_level": obs_noise_level,
                "background_error_std": background_error_std,
                "max_iterations": max_iterations,
                "n_obs_points": n_obs,
                "obs_times": obs_times,
            },
            "results": {
                "background_error": float(background_error),
                "analysis_error": float(analysis_error),
                "error_reduction": float(error_reduction),
                "mean_rmse": float(mean_rmse),
                "data_misfit": float(data_misfit),
                "innovation_mean": float(innov_mean),
                "innovation_std": float(innov_std),
                "num_iterations": int(optimizer.iteration),
                "converged": bool(optimizer.converged),
            },
            "convergence": {
                "cost_history": [float(c) for c in cost_history],
                "gradient_norm_history": [float(g) for g in gradient_history],
            },
            "timing": {
                "warmup_s": float(t_warmup),
                "truth_generation_s": float(t_truth),
                "optimization_s": float(opt_time),
                "total_s": float(total_time),
            },
        }

        results_file = output_dir / "data" / "phase2_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 70}")
        # Phase 2 expects lower error reduction than Phase 1 (8.5%)
        # but still some positive reduction
        phase_pass = error_reduction > 0.0
        print(f"PHASE 2: {'PASS' if phase_pass else 'NEEDS REVIEW'}")
        print(f"{'=' * 70}")
        print(f"  Error reduction: {error_reduction:.1f}% (expected >0%, lower than Phase 1's 8.5%)")
        print(f"  Iterations: {optimizer.iteration}")
        print(f"  Converged: {optimizer.converged}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Results saved to: {results_file}")
        print(f"{'=' * 70}")

        return results

    return None


def _compute_static_L_wme(obs_operator, B, n_obs_times, obs_variance,
                           m_template, predictability_gamma=0.1,
                           adaptive_gamma=True, comm=None, rank=0,
                           skip_eq38_inflation=False,
                           obs_correlation_length=0.0,
                           obs_correlation_variance=None):
    """Compute static L_wme = I + (N / σ²_obs) · H B Hᵀ (no adjoint solves).

    This matches the near-linear limit of the dynamic L_wme. When M_{k:0} ≈ I,
    J_wme → (√N/σ_obs)·H, so L_wme → I + (N/σ²_obs)·H B Hᵀ.
    The +I term is the observation noise contribution: (1/N) Σ_k R^{-1/2} Cov(ε_k) R^{-1/2} = I.

    When skip_eq38_inflation=True, B is assumed to already have the correct
    inflation (e.g., from a prior TLM-based Eq 38 computation). The internal
    H·H^T-based inflation is skipped to avoid double-inflation.

    Parameters
    ----------
    obs_operator : ObservationOperator
        Point observation operator (H).
    B : DiagonalCovariance
        Background covariance.
    n_obs_times : int
        Number of observation times (N).
    obs_variance : float
        Observation variance (σ²_obs).
    m_template : PETSc.Vec
        Template state vector for creating unit vectors.
    predictability_gamma : float
        Relaxation parameter for eigenvalue flooring (Eq 38 γ).
    adaptive_gamma : bool
        If True, floor = gamma * lambda_max.
    comm : MPI.Comm
        MPI communicator.
    rank : int
        MPI rank.
    obs_correlation_length : float, optional
        If > 0, the H B Hᵀ contribution is replaced by an obs-space
        Gaussian kernel: K[i,j] = σ_b² × exp(−‖x_i − x_j‖² / 2 L²)
        evaluated at observation coordinates. Mathematically equivalent to
        a Gaussian-correlated B in the point-observation limit; approximation
        error O((cell_size / L)²). Use this when forming a dense correlated
        B is infeasible (e.g., ≫ 10⁴ DOFs).
    obs_correlation_variance : float, optional
        σ_b² for the obs-space kernel. If None, inferred from B.diagonal min
        (matches the current diagonal B's variance level).

    Returns
    -------
    DenseCovariance
        Regularized static L_wme covariance.
    dict
        Diagnostics dictionary.
    """
    from petsc4py import PETSc
    from mpi4py import MPI
    from swe4dvar.data_assimilation.covariance import DenseCovariance
    import time as _time, sys as _sys

    def _tlog(msg):
        if rank == 0:
            print(f"  [L_wme.t={_time.time() - _t0:.1f}s] {msg}", flush=True)

    _t0 = _time.time()
    _tlog("entered _compute_static_L_wme")
    d_obs = obs_operator.get_num_observations()
    _tlog(f"got d_obs={d_obs}")
    n_state = m_template.getSize()
    _tlog(f"got n_state={n_state}")

    # Branch: obs-space Gaussian kernel path (for correlated-B experiments)
    # ---------------------------------------------------------------------
    # Mathematically equivalent to the point-observation limit of a Gaussian-
    # correlated B: for H = point interpolation at obs coordinates,
    #   (H B Hᵀ)[i, j] ≈ σ_b² × exp(−‖x_i − x_j‖² / 2 L²)
    # Approximation error O((cell_size / L)²) ≈ 10% for our mesh with L ≥ 1000 m.
    # This avoids building a 208K × 208K dense correlated B (346 GB).
    if obs_correlation_length > 0:
        if rank == 0:
            print(f"  Using obs-space Gaussian kernel for H B Hᵀ "
                  f"(L={obs_correlation_length:.0f} m)")

        sigma_b2 = obs_correlation_variance
        if sigma_b2 is None:
            # Infer from B.diagonal. Taking the min picks up zero entries at
            # boundary DOFs / dry cells, which breaks the kernel. Use the
            # MEAN of positive entries instead — this matches the typical
            # variance of a "real" DOF under a component-aware B.
            if hasattr(B, 'diagonal'):
                B_diag_arr = B.diagonal.getArray()
                mask = B_diag_arr > 0
                local_sum = float(B_diag_arr[mask].sum()) if mask.any() else 0.0
                local_count = int(mask.sum())
                if comm is not None and comm.Get_size() > 1:
                    total_sum = float(comm.allreduce(local_sum, op=MPI.SUM))
                    total_count = int(comm.allreduce(local_count, op=MPI.SUM))
                else:
                    total_sum = local_sum
                    total_count = local_count
                if total_count == 0:
                    raise ValueError(
                        "Cannot infer σ_b² from B.diagonal: no positive entries "
                        "(pass obs_correlation_variance explicitly)."
                    )
                sigma_b2 = total_sum / total_count
            else:
                raise ValueError(
                    "obs_correlation_variance must be provided when B has no .diagonal"
                )

        if rank == 0:
            print(f"  σ_b² = {sigma_b2:.6e}")

        # Obs coordinates (identical on every rank — obs_operator holds these)
        obs_coords = np.asarray(obs_operator.obs_points)[:, :2]
        # Pairwise squared distances
        from scipy.spatial.distance import cdist
        D = cdist(obs_coords, obs_coords, metric='euclidean')
        K = sigma_b2 * np.exp(-(D ** 2) / (2.0 * obs_correlation_length ** 2))

        # Eq 38 inflation needs λ_min of the Gram matrix G = (H B Hᵀ)/σ_b² =
        # exp(−D²/2L²). G is SPD but near-rank-deficient (many eigenvalues
        # ≈ 0), which breaks np.linalg.eigvalsh on large matrices.
        # Workaround: compute the eigenspectrum of the well-conditioned
        # L_unadj = I + (N/σ²_obs)·K (all eigenvalues ≥ 1) and back out
        # λ(G) = (λ(L_unadj) − 1) · σ²_obs / (N · σ_b²).
        L_unadj = np.eye(d_obs) + (n_obs_times / obs_variance) * K
        eigvals_L = np.linalg.eigvalsh(L_unadj)
        eigvals_K = np.maximum((eigvals_L - 1.0) * obs_variance / n_obs_times, 0.0)
        gram_eigvals = eigvals_K / sigma_b2

        # Floor λ_min(G) with the adaptive γ × λ_max(G). A Gaussian kernel is
        # intentionally rank-deficient, so a literal λ_min ≈ 0 would send the
        # Eq-38 "required σ_b²" to infinity and produce an absurd inflation
        # factor (10²⁷+). The adaptive floor is the consistent choice because
        # it's the SAME floor used to regularize the L_wme eigenvalues below;
        # directions below γ·λ_max are spectrally clipped anyway and should
        # not drive B inflation.
        gram_min_floor = predictability_gamma * gram_eigvals.max()
        lambda_min_G = max(gram_eigvals.min(), gram_min_floor, 1e-30)

        current_min_var = sigma_b2
        if skip_eq38_inflation:
            alpha = 1.0
            diagnostics = {
                'obs_correlation_length': float(obs_correlation_length),
                'sigma_b2': float(sigma_b2),
                'lambda_min_G': float(lambda_min_G),
                'lambda_max_G': float(gram_eigvals.max()),
                'inflation_factor': 1.0,
                'skip_reason': 'B already inflated by TLM-based Eq 38',
            }
            if rank == 0:
                print(f"  [Eq 38] Skipped — B already inflated by TLM")
                print(f"  [Kernel] λ_min(G)={gram_eigvals.min():.4e}, "
                      f"λ_max(G)={gram_eigvals.max():.4e}, "
                      f"ratio={gram_eigvals.max()/lambda_min_G:.2e}")
        else:
            required_var = (predictability_gamma * obs_variance) / (n_obs_times * lambda_min_G)
            alpha = max(1.0, required_var / current_min_var) if current_min_var > 0 else 1.0
            diagnostics = {
                'obs_correlation_length': float(obs_correlation_length),
                'sigma_b2': float(sigma_b2),
                'lambda_min_G': float(lambda_min_G),
                'lambda_max_G': float(gram_eigvals.max()),
                'required_var': float(required_var),
                'inflation_factor': float(alpha),
            }
            if rank == 0:
                print(f"  [Kernel] λ_min(G)={gram_eigvals.min():.4e}, "
                      f"λ_max(G)={gram_eigvals.max():.4e}, "
                      f"ratio={gram_eigvals.max()/lambda_min_G:.2e}")
                if alpha > 1.0:
                    print(f"  [Eq 38] Kernel inflation: α = {alpha:.4f} "
                          f"(required σ²_b = {required_var:.4e}, "
                          f"current = {current_min_var:.4e})")
                else:
                    print(f"  [Eq 38] Kernel: no B inflation needed")

        # L_static = I + (N/σ²_obs) · α σ_b² × exp(−D²/2L²)
        if alpha > 1.0:
            L_dense = np.eye(d_obs) + (n_obs_times / obs_variance) * (alpha * K)
        else:
            L_dense = L_unadj  # reuse
        del K, D, L_unadj
        import gc; gc.collect()
    else:
        # Original diagonal-B path (fast path via explicit H materialization)
        # Extract H matrix via adjoint: H^T e_i gives row i of H
        # This is O(d_obs) applications instead of O(n_state) — much faster
        # (306 vs 52020 for Shinnecock)
        _tlog(f"else-branch: extracting H matrix ({d_obs} x {n_state}) via {d_obs} adjoint applications")

        H = np.zeros((d_obs, n_state))

        # Create observation-space unit vector
        obs_vec = obs_operator.forward(m_template)  # get correctly sized obs vector
        _tlog("obs_operator.forward(m_template) done")
        obs_vec.zeroEntries()

        for i in range(d_obs):
            obs_vec.zeroEntries()
            obs_vec.setValue(i, 1.0)
            obs_vec.assemblyBegin()
            obs_vec.assemblyEnd()
            HT_ei = obs_operator.adjoint(obs_vec)  # H^T e_i = state-space vector
            # Gather full vector across MPI ranks
            local_arr = HT_ei.getArray()
            if comm is not None and comm.Get_size() > 1:
                full_arr = np.zeros(n_state)
                start, end = HT_ei.getOwnershipRange()
                full_arr[start:end] = local_arr
                comm.Allreduce(full_arr.copy(), full_arr)
                H[i, :] = full_arr
            else:
                H[i, :] = local_arr
            HT_ei.destroy()
        obs_vec.destroy()

        _tlog(f"H extracted: {np.count_nonzero(H)} nonzeros out of {d_obs * n_state}")

        # Get B diagonal
        B_diag = B.diagonal.getArray().copy()
        _tlog(f"got local B.diagonal (size={len(B_diag)})")
        # Gather full diagonal (for MPI)
        if comm is not None and comm.Get_size() > 1:
            _tlog(f"np>1: allreducing B_diag to full n_state={n_state}")
            B_diag_full = np.zeros(n_state)
            start, end = B.diagonal.getOwnershipRange()
            B_diag_full[start:end] = B_diag
            comm.Allreduce(B_diag_full.copy(), B_diag_full)
            B_diag = B_diag_full
            _tlog("allreduce B_diag done")

        current_min_var = B_diag.min()
        _tlog(f"current_min_var={current_min_var:.4e}")

        # Step 1: B inflation for Eq 38 predictability bound
        if skip_eq38_inflation:
            # B was already inflated by a prior TLM-based Eq 38 computation.
            # Skip the internal H·H^T approximation to avoid double-inflation.
            alpha = 1.0
            diagnostics = {
                'inflation_factor': 1.0,
                'skip_reason': 'B already inflated by TLM-based Eq 38',
                'current_min_var': float(current_min_var),
            }
            if rank == 0:
                print(f"  [Eq 38] Skipped — B already inflated by TLM (min(B) = {current_min_var:.6e})")
        else:
            # Fallback: approximate Gram matrix from H alone (no TLM).
            # This is valid only when M_{k:0} ≈ I (near-identity propagator).
            G_static = H @ H.T  # (d_obs × d_obs)
            gram_eigvals = np.linalg.eigvalsh(G_static)
            lambda_min_G = max(gram_eigvals.min(), 1e-30)

            # Inflation factor: required_var = γ · σ²_obs / (N · λ_min(H H^T))
            required_var = (predictability_gamma * obs_variance) / (n_obs_times * lambda_min_G)
            alpha = max(1.0, required_var / current_min_var) if current_min_var > 0 else 1.0

            diagnostics = {
                'gram_eigvals': gram_eigvals.tolist(),
                'lambda_min_G': float(lambda_min_G),
                'required_var': float(required_var),
                'inflation_factor': float(alpha),
            }

            if rank == 0:
                if alpha > 1.0:
                    print(f"  [Eq 38] Static H·H^T inflation: α = {alpha:.4f} "
                          f"(required σ²_b = {required_var:.4e}, current = {current_min_var:.4e})")
                else:
                    print(f"  [Eq 38] Static: no B inflation needed")

        B_diag_inflated = B_diag * alpha if alpha > 1.0 else B_diag
        _tlog(f"B_diag_inflated ready (alpha={alpha:.4e})")

        # Step 2: Form L_static = I + (N/σ²_obs) · H B_inflated Hᵀ
        # The +I term arises from the observation noise contribution to the
        # WME predicted covariance: (1/N) Σ_k R^{-1/2} Cov(ε_k) R^{-1/2} = I.
        # This matches the dynamic L_wme computation in cost_functions.py:1695.
        HB = H * B_diag_inflated[np.newaxis, :]
        _tlog(f"HB formed (shape {HB.shape})")
        L_dense = np.eye(d_obs) + (n_obs_times / obs_variance) * (HB @ H.T)
        _tlog(f"L_dense formed (shape {L_dense.shape})")

        # Free large intermediate arrays
        del H, HB, B_diag_inflated
        import gc; gc.collect()

    # Step 3: Eigenvalue regularization (same as dynamic Layer 2)
    _tlog(f"entering np.linalg.eigh(L_dense) shape={L_dense.shape}")
    eigvals, eigvecs = np.linalg.eigh(L_dense)
    _tlog("eigh done")

    if adaptive_gamma:
        gamma_floor = predictability_gamma * eigvals.max()
        if rank == 0:
            print(f"  Static adaptive γ: {predictability_gamma} × λ_max={eigvals.max():.4e} "
                  f"→ γ_eff={gamma_floor:.4e}")
    else:
        gamma_floor = predictability_gamma

    n_floored = int(np.sum(eigvals < gamma_floor))
    n_natural = len(eigvals) - n_floored
    eigvals_reg = np.maximum(eigvals, gamma_floor)
    L_dense = eigvecs @ np.diag(eigvals_reg) @ eigvecs.T

    lambda_min_raw = float(eigvals.min())
    lambda_max_raw = float(eigvals.max())
    ratio_raw = lambda_max_raw / max(lambda_min_raw, 1e-30)
    if rank == 0:
        print(f"  Static L_wme: {n_natural}/{d_obs} natural, {n_floored}/{d_obs} floored")
        print(f"  [L_wme spectrum] λ_max={lambda_max_raw:.4e}  λ_min={lambda_min_raw:.4e}  "
              f"ratio={ratio_raw:.2e}")
        if obs_correlation_length > 0:
            # Report how many eigenvalues meaningfully exceed 1 — these are the
            # directions where the predictability term actually does work.
            n_above_2 = int(np.sum(eigvals > 2.0))
            n_above_10 = int(np.sum(eigvals > 10.0))
            n_above_100 = int(np.sum(eigvals > 100.0))
            print(f"  [L_wme spectrum] eigenvalues > 2.0: {n_above_2}/{d_obs}, "
                  f"> 10: {n_above_10}, > 100: {n_above_100}")

    diagnostics['eigvals_raw'] = eigvals.tolist()
    diagnostics['eigvals_regularized'] = eigvals_reg.tolist()
    diagnostics['gamma_floor'] = float(gamma_floor)
    diagnostics['n_natural'] = int(n_natural)
    diagnostics['n_floored'] = int(n_floored)
    diagnostics['lambda_min_raw'] = lambda_min_raw
    diagnostics['lambda_max_raw'] = lambda_max_raw
    diagnostics['spectrum_ratio_raw'] = float(ratio_raw)
    diagnostics['raw_spectrum'] = _summarize_eigenvalues(eigvals)
    diagnostics['regularized_spectrum'] = _summarize_eigenvalues(eigvals_reg)

    # L_wme operates on sequential Q_wme vectors (obs-space is COMM_SELF).
    # Build the matrix on COMM_SELF so every rank holds an identical full copy
    # that matches Q_wme's communicator. Distributed MPI Cholesky would require
    # MUMPS; this serial-per-rank path is cheaper and MPI-safe for small d_obs.
    mat = PETSc.Mat().create(comm=PETSc.COMM_SELF)
    mat.setSizes((d_obs, d_obs))
    mat.setType(PETSc.Mat.Type.DENSE)
    mat.setUp()

    for i in range(d_obs):
        mat.setValues(i, list(range(d_obs)), L_dense[i, :])

    mat.assemblyBegin()
    mat.assemblyEnd()

    L_wme_cov = DenseCovariance(PETSc.COMM_SELF, mat, inverse_method="cholesky")
    _tlog(f"DenseCovariance built; exiting _compute_static_L_wme")

    return L_wme_cov, diagnostics


def _run_sub_experiment(args, sub_label, method, config_overrides=None,
                        static_L_wme=None, output_dir=None,
                        nt_da=12, nt_ramp=144, phase_prefix="3",
                        sweep_params=None, l_wme_mode="dynamic",
                        apply_eq38_background_scaling=False,
                        method_variant_key=None):
    """Run a single sub-experiment.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    sub_label : str
        Sub-experiment label (e.g., "a", "b", "c").
    method : str
        DA method ("4dvar" or "dcwme").
    config_overrides : dict, optional
        Override config parameters.
    static_L_wme : DenseCovariance, optional
        Precomputed static L_wme for DC-WME static.
    output_dir : Path
        Output directory.
    nt_da : int
        Number of DA timesteps (default: 12 for 2h window).
    nt_ramp : int
        Number of ramp-up timesteps (default: 144 for 24h).
    sweep_params : dict, optional
        Parameter overrides for Phase 6 sweep. Keys override local
        configuration variables (obs_fraction, obs_frequency, etc.).
    l_wme_mode : str, optional
        DC-WME covariance mode: ``static`` or ``dynamic``.
    apply_eq38_background_scaling : bool, optional
        If True, apply the Eq. 38 background scaling even for non-DC-WME
        branches so conditioning changes can be separated from methodology.
    method_variant_key : str, optional
        Human-readable label for result metadata.

    Returns
    -------
    dict
        Results dictionary.
    """
    import os
    os.environ.setdefault("CC", "/usr/bin/clang")

    from mpi4py import MPI
    from swe4dvar.forward.adcirc_problem import ADCIRCProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params
    from experiments.twin_experiment import (
        TwinExperiment, TwinExperimentConfig, ForwardModelWrapper,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ================================================================
    # Configuration
    # ================================================================
    dt = 600.0
    nt_total = nt_ramp + nt_da

    obs_fraction = 0.1
    obs_frequency = 6
    obs_noise_level = 0.01
    background_error_std = 0.02
    max_iterations = 15
    friction_scale_factor = 1.15

    # Override with sweep parameters (Phase 6)
    if sweep_params:
        obs_fraction = _get_sweep_value(sweep_params, "obs_fraction", obs_fraction)
        obs_frequency = _get_sweep_value(sweep_params, "obs_frequency", obs_frequency)
        obs_noise_level = _get_sweep_value(sweep_params, "obs_noise_level", obs_noise_level)
        background_error_std = _get_sweep_value(sweep_params, "background_error_std", background_error_std)
        friction_scale_factor = _get_sweep_value(sweep_params, "friction_scale_factor", friction_scale_factor)

    # DC-WME / Eq 38 parameters
    predictability_gamma = _get_sweep_value(sweep_params, "predictability_gamma", 0.1)
    auto_inflate_B = True
    adaptive_gamma = True

    effective_l_wme_mode = (
        "static" if (method == "dcwme" and static_L_wme is not None)
        else (l_wme_mode if method == "dcwme" else "N/A")
    )

    if rank == 0:
        print(f"\n{'=' * 70}")
        method_label = method.upper()
        if method == "dcwme":
            method_label += f" ({effective_l_wme_mode} L_wme)"
        if apply_eq38_background_scaling and method == "4dvar":
            method_label += " + Eq38(B)"
        if method_variant_key:
            method_label += f" [{method_variant_key}]"
        print(f"PHASE {phase_prefix}{sub_label.upper()}: {method_label}")
        print(f"{'=' * 70}")

    # ================================================================
    # Step 1: Create problem and solver
    # ================================================================
    if rank == 0:
        print("\n--- Step 1: Creating problem and solver ---")

    prob = ADCIRCProblem(
        adios_file=args.adios_file,
        spherical=True,
        solution_var="h",
        friction_law="mannings",
        wd=True,
        wd_alpha=1.5,
        dt=dt,
        bathy_adjustment=0,
        nt=nt_total,
        dramp=2.0,
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])

    warmup_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=10,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=True,
    )

    da_solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=35,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=False,
        reduction_it=10,
    )

    if rank == 0:
        n_nodes = prob.mesh.topology.index_map(0).size_local
        n_dofs = len(solver.u.x.array)
        print(f"  Mesh: {n_nodes} nodes, {n_dofs} DOFs")

    # ================================================================
    # Step 2: Warm-up
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 2: Running {nt_ramp}-step warm-up ---")

    t_warmup_start = time.time()
    prob.nt = nt_ramp
    solver.time_loop(
        solver_parameters=warmup_params,
        stations=[],
        plot_every=9999,
        save_state=False,
        store_jacobians=False,
        enable_video=False,
        monitor_progress=(rank == 0),
    )
    t_warmup = time.time() - t_warmup_start
    t_da_start = prob.t

    if rank == 0:
        print(f"  Warm-up completed in {t_warmup:.1f}s")

    # ================================================================
    # Step 3: Set up TwinExperiment
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 3: Setting up DA window ({nt_da} timesteps) ---")

    prob.nt = nt_da

    config_kwargs = dict(
        method=method,
        obs_fraction=obs_fraction,
        obs_frequency=obs_frequency,
        obs_noise_level=obs_noise_level,
        background_error_std=background_error_std,
        background_correlation_length=500.0,
        max_iterations=max_iterations,
        max_funcs=30,  # Cap total function evals to prevent optimizer stalling
        gradient_tolerance=1e-3,
        cost_tolerance=1e-4,
        n_windows=1,
        perturb_friction=True,
        friction_scale_factor=friction_scale_factor,
        use_bounds=True,
        h_min=0.01,
        interior_only=True,
        component_aware_cov=True,
        verbose=(rank == 0),
    )

    # DC-WME specific parameters
    if method == "dcwme":
        config_kwargs.update(
            predictability_gamma=predictability_gamma,

            auto_inflate_B=auto_inflate_B,
            adaptive_gamma=adaptive_gamma,
        )
        if static_L_wme is not None:
            # Phase 3b: skip analytical L_wme computation
            config_kwargs['l_wme_samples'] = 0

    # Handle friction_scale_factor=1.0 (no model error / inverse crime reference)
    if sweep_params and friction_scale_factor == 1.0:
        config_kwargs["perturb_friction"] = False
        config_kwargs["friction_scale_factor"] = 1.0

    if config_overrides:
        config_kwargs.update(config_overrides)

    config = TwinExperimentConfig(**config_kwargs)

    exp = TwinExperiment(
        problem=prob, solver=solver, config=config,
        solver_params=da_solver_params, comm=comm,
    )

    # ================================================================
    # Step 4: Generate truth + apply friction perturbation
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 4: Generating truth trajectory ---", flush=True)

    t_truth_start = time.time()
    exp._generate_truth()
    t_truth = time.time() - t_truth_start
    exp.solver.storage.clear()

    if rank == 0:
        print(f"  Truth: {len(exp.truth_trajectory)} states, generated in {t_truth:.1f}s")
        print(f"\n--- Step 4b: Applying friction perturbation (×{friction_scale_factor}) ---", flush=True)

    exp._apply_physics_perturbations()

    # ================================================================
    # Step 5: Setup observations, background, covariances
    # ================================================================
    if rank == 0:
        print("\n--- Step 5: Setting up observations and background ---", flush=True)

    # Temporarily set method to "4dvar" to prevent _setup_covariances from building
    # the dense B_lwme correlation matrix (52020×52020 = 21 GB, causes OOM).
    # We use diagonal B for all Phase 3 experiments (per plan: "diagonal B for Phases 0-3").
    # Static L_wme is computed directly from H·B·Hᵀ, dynamic L_wme uses diagonal B.
    original_method = exp.config.method
    exp.config.method = "4dvar"  # Skip dense B_lwme construction

    obs_points, obs_operator, obs_times = exp._setup_observations()
    exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
    background_error = exp._setup_background()
    B, R, B_lwme = exp._setup_covariances(obs_operator, obs_noise_stds)

    exp.config.method = original_method  # Restore original method

    # Eq 38: Derive σ_b² from TLM Gram matrix — DC-WME only
    n_obs = obs_operator.get_num_observations()
    eq38_result = None
    eq38_scale = 1.0
    if method == "dcwme" or apply_eq38_background_scaling:
        if rank == 0:
            print(f"\n--- Step 5a: Computing σ_b² from Eq 38 via TLM ---")
            print(f"  Running forward solve from m_true with DA model ({nt_da} steps)...")
        from experiments.twin_experiment import ForwardModelWrapper
        prob.nt = nt_da
        gram_fwd = ForwardModelWrapper(
            solver=solver, problem=prob,
            solver_params=da_solver_params, t_start=t_da_start,
        )
        gram_trajectory, gram_jacobians = gram_fwd.solve(exp.m_true, store_jacobians=True)
        eq38_result = _compute_eq38_from_tlm(
            forward_model=gram_fwd,
            obs_operator=obs_operator, obs_cov=R,
            m_linearize=exp.m_true,
            observations=exp.observations, obs_times=obs_times,
            truth_trajectory=gram_trajectory, truth_jacobians=gram_jacobians,
            predictability_gamma=predictability_gamma,
            comm=comm, rank=rank,
        )
        eq38_scale = _apply_eq38_to_B(B, eq38_result, rank=rank)
        for v in gram_trajectory:
            v.destroy()
        del gram_trajectory, gram_jacobians
        solver.storage.clear()
        import gc; gc.collect()

    if rank == 0:
        print(f"  Observation points: {n_obs}")
        print(f"  Observation times: {obs_times}")
        print(f"  Background RMS error: {background_error:.6f}", flush=True)

    # ================================================================
    # Step 5b: Compute static L_wme if needed
    # ================================================================
    static_diagnostics = None
    if method == "dcwme" and effective_l_wme_mode == "static" and static_L_wme is None:
        already_inflated = eq38_result is not None
        if rank == 0:
            print(f"\n--- Step 5b: Computing static L_wme "
                  f"(skip_eq38_inflation={already_inflated}) ---", flush=True)
        static_L_wme, static_diagnostics = _compute_static_L_wme(
            obs_operator, B, len(obs_times), obs_noise_level ** 2,
            exp.m_true, predictability_gamma=predictability_gamma,
            adaptive_gamma=adaptive_gamma,
            comm=comm, rank=rank,
            skip_eq38_inflation=already_inflated,
        )
        if rank == 0:
            print("  Static L_wme computed successfully", flush=True)
    elif static_L_wme is not None and rank == 0:
        print("\n  Using precomputed static L_wme")

    # Memory cleanup before cost function init
    import gc
    exp.solver.storage.clear()
    gc.collect()

    # ================================================================
    # Step 6: Create forward model
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 6: Creating forward model (t_start={t_da_start:.0f}s) ---", flush=True)

    forward_model = exp._create_forward_model(t_start=t_da_start)

    # ================================================================
    # Step 7: Setup cost function
    # ================================================================
    if rank == 0:
        method_label = method.upper()
        if method == "dcwme":
            method_label += f" ({effective_l_wme_mode} L_wme)"
        print(f"\n--- Step 7: Setting up {method_label} cost function ---", flush=True)

    if method == "dcwme" and effective_l_wme_mode == "static" and static_L_wme is not None:
        # Phase 3b: pass precomputed static L_wme
        from swe4dvar.data_assimilation.cost_functions import DCWMEFourDVarCost

        cost_function = DCWMEFourDVarCost(
            forward_model=forward_model,
            observation_operator=obs_operator,
            background_cov=B,
            observation_cov=R,
            m_background=exp.m_background,
            observations=exp.observations,
            obs_times=obs_times,
            predicted_cov_wme=static_L_wme,
            n_l_wme_samples=0,  # Skip analytical computation
            auto_inflate_B=False,  # Already inflated in static computation

            predictability_gamma=predictability_gamma,
            adaptive_gamma=adaptive_gamma,
            comm=comm,
        )

        # Wrap with boundary gradient zeroing if needed
        if config.interior_only:
            from experiments.twin_experiment import ZeroBoundaryGradientCost
            boundary_dofs = exp._get_boundary_dofs()
            cost_function = ZeroBoundaryGradientCost(cost_function, boundary_dofs)
            if rank == 0:
                print(f"  Zeroing {len(boundary_dofs)} boundary DOF gradients")

        # Disable M^{-1} preconditioning (same as Phase 1/2)
        if rank == 0:
            print(f"  M^{{-1}} preconditioning DISABLED (causes ill-conditioning)")
    else:
        cost_function = exp._setup_cost_function(
            forward_model, obs_operator, B, R, obs_times, B_lwme=B_lwme
        )

    # ================================================================
    # Step 7b: Attach gradient smoother
    # ================================================================
    gradient_smoothing_length = 500.0
    if config.background_correlation_length > 0:
        if rank == 0:
            print("\n--- Step 7b: Building gradient smoother ---")

        h_indices, u_indices, v_indices = exp._get_component_dof_indices(owned_only=True)
        smoothing_matrix = exp._build_smoothing_matrix(
            h_indices, gradient_smoothing_length
        )

        def gradient_smoother(grad_array):
            smoothed = grad_array.copy()
            smoothed[h_indices] = smoothing_matrix @ grad_array[h_indices]
            smoothed[u_indices] = smoothing_matrix @ grad_array[u_indices]
            smoothed[v_indices] = smoothing_matrix @ grad_array[v_indices]
            return smoothed

        inner_cost = cost_function
        while hasattr(inner_cost, 'base_cost'):
            inner_cost = inner_cost.base_cost
        inner_cost.gradient_smoother = gradient_smoother

        if rank == 0:
            print(f"  Gradient smoother attached (L={gradient_smoothing_length}m)")

    exp.optimization_iteration_callback = _make_state_rmse_iteration_callback(
        exp.m_true, exp.m_background
    )

    # ================================================================
    # Step 8: Run optimization
    # ================================================================
    if rank == 0:
        print("\n--- Step 8: Running L-BFGS optimization ---")

    optimizer, opt_time = exp._run_optimization(cost_function)

    cost_history = [h["cost"] for h in optimizer.convergence_history]
    gradient_history = [h["grad_norm"] for h in optimizer.convergence_history]
    analysis_state_rmse_history = [
        float(h["analysis_state_rmse"])
        for h in optimizer.convergence_history
        if "analysis_state_rmse" in h
    ]
    distance_from_background_history = [
        float(h["distance_from_background_rmse"])
        for h in optimizer.convergence_history
        if "distance_from_background_rmse" in h
    ]

    if rank == 0:
        print(f"  Iterations: {optimizer.iteration}")
        print(f"  Converged: {optimizer.converged}")
        print(f"  Optimization time: {opt_time:.1f}s")
        if cost_history:
            print(f"  Cost: {cost_history[0]:.6f} → {cost_history[-1]:.6f}")
        if gradient_history:
            print(f"  ||grad||: {gradient_history[0]:.6e} → {gradient_history[-1]:.6e}")

    # ================================================================
    # Step 9: Evaluate results
    # ================================================================
    if rank == 0:
        print("\n--- Step 9: Evaluating results ---")

    if cost_history and cost_history[-1] >= 1e19:
        if rank == 0:
            print("  WARNING: Optimization failed (cost diverged)")
        analysis_error = background_error
        error_reduction = 0.0
        innov_mean = innov_std = mean_rmse = data_misfit = 0.0
    else:
        analysis_error, error_reduction, innov_mean, innov_std, mean_rmse, data_misfit = (
            exp._evaluate_results(obs_operator, obs_times, background_error)
        )

    total_time = t_warmup + t_truth + opt_time

    if rank == 0:
        print(f"\n  Background error:  {background_error:.6f}")
        print(f"  Analysis error:    {analysis_error:.6f}")
        print(f"  Error reduction:   {error_reduction:.1f}%")
        print(f"  Mean RMSE:         {mean_rmse:.6f}")
        print(f"  Data misfit:       {data_misfit:.6f}")

    # ================================================================
    # Step 10: Extract DC-WME diagnostics
    # ================================================================
    dcwme_diagnostics = {}
    eq38_diagnostics = eq38_result.copy() if eq38_result is not None else {}
    if method == "dcwme":
        # Get the inner cost function (unwrap boundary gradient zeroing)
        inner = cost_function
        while hasattr(inner, 'base_cost'):
            inner = inner.base_cost

        if hasattr(inner, '_b_inflation_factor'):
            dcwme_diagnostics['b_inflation_factor'] = float(inner._b_inflation_factor)
        if hasattr(inner, '_gram_eigenvalues') and inner._gram_eigenvalues is not None:
            dcwme_diagnostics['gram_eigenvalues'] = [float(v) for v in inner._gram_eigenvalues]
        if hasattr(inner, '_L_wme') and inner._L_wme is not None:
            # Extract L_wme eigenvalues for diagnostics
            try:
                L_wme_mat = inner._L_wme
                if hasattr(L_wme_mat, 'mat'):
                    n = L_wme_mat.mat.getSize()[0]
                    L_dense_np = np.zeros((n, n))
                    rstart, rend = L_wme_mat.mat.getOwnershipRange()
                    for i in range(rstart, rend):
                        cols, vals = L_wme_mat.mat.getRow(i)
                        for c, v in zip(cols, vals):
                            L_dense_np[i, c] = v
                    l_eigvals = np.linalg.eigvalsh(L_dense_np)
                    dcwme_diagnostics['l_wme_eigenvalues'] = [float(v) for v in l_eigvals]
                    dcwme_diagnostics['l_wme_n_natural'] = int(np.sum(l_eigvals > predictability_gamma * l_eigvals.max()))
                    dcwme_diagnostics['l_wme_spectrum'] = _summarize_eigenvalues(l_eigvals)
                    if np.all(l_eigvals > 1.0):
                        weights = 1.0 - 1.0 / l_eigvals
                        dcwme_diagnostics['effective_weight_min'] = float(weights.min())
                        dcwme_diagnostics['effective_weight_max'] = float(weights.max())
            except Exception as e:
                if rank == 0:
                    print(f"  Warning: Could not extract L_wme eigenvalues: {e}")

        if static_diagnostics is not None:
            dcwme_diagnostics['static_lwme'] = static_diagnostics

    # ================================================================
    # Step 11: Save results
    # ================================================================
    results = None
    if rank == 0:
        results = {
            "phase": f"{phase_prefix}{sub_label}",
            "status": "success",
            "method": method,
            "method_variant": method_variant_key or method,
            "model_error": True,
            "model_error_type": "friction",
            "friction_scale_factor": friction_scale_factor,
            "l_wme_type": effective_l_wme_mode,
            "experiment_controls": {
                "apply_eq38_background_scaling": bool(apply_eq38_background_scaling),
                "eq38_scaling_basis": "WME-TLM" if (eq38_result is not None) else "none",
                "phase6_suite": getattr(args, "phase6_suite", None),
                "method_variant_key": method_variant_key,
            },
            "config": {
                "dt": dt,
                "nt_ramp": nt_ramp,
                "nt_da": nt_da,
                "t_da_start_s": t_da_start,
                "obs_fraction": obs_fraction,
                "obs_frequency": obs_frequency,
                "obs_noise_level": obs_noise_level,
                "background_error_std": background_error_std,
                "eq38_sigma_b_sq": eq38_result["sigma_b_sq"] if eq38_result else None,
                "eq38_scale_factor": eq38_scale,
                "eq38_lambda_min_G": eq38_result["lambda_min_G"] if eq38_result else None,
                "max_iterations": max_iterations,
                "n_obs_points": n_obs,
                "obs_times": obs_times,
                "predictability_gamma": predictability_gamma,

            },
            "results": {
                "background_error": float(background_error),
                "analysis_error": float(analysis_error),
                "error_reduction": float(error_reduction),
                "mean_rmse": float(mean_rmse),
                "data_misfit": float(data_misfit),
                "innovation_mean": float(innov_mean),
                "innovation_std": float(innov_std),
                "num_iterations": int(optimizer.iteration),
                "converged": bool(optimizer.converged),
            },
            "convergence": {
                "cost_history": [float(c) for c in cost_history],
                "gradient_norm_history": [float(g) for g in gradient_history],
                "analysis_state_rmse_history": analysis_state_rmse_history,
                "distance_from_background_rmse_history": distance_from_background_history,
                "records": [
                    {k: _jsonify_metric_value(v)
                     for k, v in record.items()}
                    for record in optimizer.convergence_history
                ],
            },
            "dcwme_diagnostics": dcwme_diagnostics,
            "eq38_diagnostics": eq38_diagnostics,
            "timing": {
                "warmup_s": float(t_warmup),
                "truth_generation_s": float(t_truth),
                "optimization_s": float(opt_time),
                "total_s": float(total_time),
            },
        }

        results_file = output_dir / "data" / f"phase{phase_prefix}{sub_label}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 70}")
        method_label = method.upper()
        if method == "dcwme":
            method_label += f" ({effective_l_wme_mode})"
        if apply_eq38_background_scaling and method == "4dvar":
            method_label += " + Eq38(B)"
        print(f"PHASE {phase_prefix}{sub_label.upper()}: {method_label}")
        print(f"  Error reduction: {error_reduction:.1f}%")
        print(f"  Iterations: {optimizer.iteration}")
        print(f"  Converged: {optimizer.converged}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Results saved to: {results_file}")
        print(f"{'=' * 70}")

    # Explicitly delete heavy objects to help GC reclaim memory
    del optimizer, cost_function, forward_model, exp, solver, prob
    del B, R, B_lwme, obs_operator
    _cleanup_petsc_objects()

    return results


def run_phase_3(args):
    """Phase 3: Single-Window DC-WME vs 4D-Var Comparison.

    Runs three sub-experiments on the same configuration as Phase 2:
    - 3a: Standard 4D-Var baseline (for seed-consistent comparison)
    - 3b: DC-WME with static L_wme = (N/σ²_obs) · H B Hᵀ
    - 3c: DC-WME with dynamic L_wme = J_wme B J_wmeᵀ (adjoint-computed)
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)
    comm.Barrier()

    if rank == 0:
        print("=" * 70)
        print("PHASE 3: SINGLE-WINDOW DC-WME vs 4D-VAR COMPARISON")
        print("=" * 70)
        print("  Sub-experiments:")
        print("    3a: Standard 4D-Var (baseline)")
        print("    3b: DC-WME with static L_wme")
        print("    3c: DC-WME with dynamic L_wme")
        print("=" * 70)

    # Determine which sub-experiments to run
    sub = getattr(args, 'sub', None)
    if sub:
        subs_to_run = [sub]
    else:
        subs_to_run = ["a", "b", "c"]

    all_results = {}

    for sub_label in subs_to_run:
        if sub_label == "a":
            result = _run_sub_experiment(
                args, sub_label="a", method="4dvar",
                output_dir=output_dir,
            )
        elif sub_label == "b":
            result = _run_sub_experiment(
                args, sub_label="b", method="dcwme",
                output_dir=output_dir,
            )
        elif sub_label == "c":
            result = _run_sub_experiment(
                args, sub_label="c", method="dcwme",
                output_dir=output_dir,
            )
        else:
            if rank == 0:
                print(f"Unknown sub-experiment: 3{sub_label}")
            continue

        all_results[f"3{sub_label}"] = result

    # Print comparison summary
    if rank == 0 and len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("PHASE 3 COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Sub-exp':<10} {'Method':<20} {'Error Red.':<12} {'Iters':<8} {'Conv.':<8} {'Time (min)':<10}")
        print(f"{'-' * 68}")
        for key, res in all_results.items():
            if res is not None:
                r = res['results']
                t = res['timing']
                lwme = res.get('l_wme_type', 'N/A')
                method_label = f"{res['method']}"
                if lwme != 'N/A':
                    method_label += f" ({lwme})"
                print(f"{key:<10} {method_label:<20} {r['error_reduction']:.1f}%{'':<7} "
                      f"{r['num_iterations']:<8} {'Yes' if r['converged'] else 'No':<8} "
                      f"{t['total_s']/60:.1f}")
        print(f"{'=' * 70}")

    return all_results


# ====================================================================
# Phase 4: Cycling Comparison
# ====================================================================

def _run_cycling_experiment(args, sub_label, method, output_dir=None):
    """Run a single Phase 4 cycling experiment (one method).

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    sub_label : str
        Sub-experiment label ("a", "b", "c").
    method : str
        DA method ("4dvar" or "dcwme").
    output_dir : Path
        Output directory.

    Returns
    -------
    dict
        Results dictionary with per-window metrics.
    """
    import os
    import gc
    os.environ.setdefault("CC", "/usr/bin/clang")

    from mpi4py import MPI
    from petsc4py import PETSc
    from swe4dvar.forward.adcirc_problem import ADCIRCProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params
    from experiments.twin_experiment import (
        TwinExperiment, TwinExperimentConfig, ForwardModelWrapper,
        ZeroBoundaryGradientCost,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ================================================================
    # Configuration
    # ================================================================
    dt = 600.0
    nt_ramp = 288          # 48h ramp
    n_windows = 6
    window_nt = 71         # ~11.8h per window
    nt_da_total = n_windows * window_nt  # 426 timesteps total DA

    obs_fraction = 0.1
    obs_frequency = 6      # every hour
    obs_noise_level = 0.01
    background_error_std = 0.02
    max_iterations = 15
    friction_scale_factor = 1.15

    # DC-WME / Eq 38 parameters
    predictability_gamma = 0.1
    auto_inflate_B = True
    adaptive_gamma = True

    l_wme_type = "N/A"
    if sub_label == "b":
        l_wme_type = "static"
    elif sub_label == "c":
        l_wme_type = "dynamic"

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"PHASE 4{sub_label.upper()}: CYCLING {method.upper()}"
              f"{' (static L_wme)' if sub_label == 'b' else ''}"
              f"{' (dynamic L_wme)' if sub_label == 'c' else ''}")
        print(f"  {n_windows} windows × {window_nt} timesteps "
              f"({window_nt * dt / 3600:.1f}h each)")
        print(f"{'=' * 70}", flush=True)

    t_total_start = time.time()

    # ================================================================
    # Step 1: Create problem and solver
    # ================================================================
    if rank == 0:
        print("\n--- Step 1: Creating problem and solver ---", flush=True)

    prob = ADCIRCProblem(
        adios_file=args.adios_file,
        spherical=True,
        solution_var="h",
        friction_law="mannings",
        wd=True,
        wd_alpha=1.5,
        dt=dt,
        bathy_adjustment=0,
        nt=nt_ramp + nt_da_total,
        dramp=2.0,
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])

    warmup_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=10,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=True,
    )

    da_solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=35,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=False,
        reduction_it=10,
    )

    if rank == 0:
        n_nodes = prob.mesh.topology.index_map(0).size_local
        n_dofs = len(solver.u.x.array)
        print(f"  Mesh: {n_nodes} nodes, {n_dofs} DOFs")

    # ================================================================
    # Step 2: Warm-up (48h ramp)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 2: Running {nt_ramp}-step warm-up ({nt_ramp * dt / 3600:.0f}h) ---",
              flush=True)

    t_warmup_start = time.time()
    prob.nt = nt_ramp
    solver.time_loop(
        solver_parameters=warmup_params,
        stations=[],
        plot_every=9999,
        save_state=False,
        store_jacobians=False,
        enable_video=False,
        monitor_progress=(rank == 0),
    )
    t_warmup = time.time() - t_warmup_start
    t_da_start = prob.t  # Time after ramp

    if rank == 0:
        print(f"  Warm-up completed in {t_warmup:.1f}s ({t_warmup/60:.1f} min)")

    # ================================================================
    # Step 3: Set up TwinExperiment for full DA period
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 3: Setting up DA ({nt_da_total} timesteps, "
              f"{nt_da_total * dt / 3600:.1f}h) ---", flush=True)

    prob.nt = nt_da_total

    config_kwargs = dict(
        method=method,
        obs_fraction=obs_fraction,
        obs_frequency=obs_frequency,
        obs_noise_level=obs_noise_level,
        background_error_std=background_error_std,
        background_correlation_length=500.0,
        max_iterations=max_iterations,
        max_funcs=30,  # Cap total function evals per window to prevent line search stalling
        gradient_tolerance=1e-3,
        cost_tolerance=1e-4,
        n_windows=1,  # We manage cycling ourselves
        perturb_friction=True,
        friction_scale_factor=friction_scale_factor,
        use_bounds=True,
        h_min=0.01,
        interior_only=True,
        component_aware_cov=True,
        verbose=(rank == 0),
    )

    if method == "dcwme":
        config_kwargs.update(
            predictability_gamma=predictability_gamma,

            auto_inflate_B=auto_inflate_B,
            adaptive_gamma=adaptive_gamma,
        )

    config = TwinExperimentConfig(**config_kwargs)

    exp = TwinExperiment(
        problem=prob, solver=solver, config=config,
        solver_params=da_solver_params, comm=comm,
    )

    # ================================================================
    # Step 4: Generate truth for full DA period
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 4: Generating truth trajectory ---", flush=True)

    t_truth_start = time.time()
    exp._generate_truth()
    t_truth = time.time() - t_truth_start
    exp.solver.storage.clear()

    if rank == 0:
        print(f"  Truth: {len(exp.truth_trajectory)} states in {t_truth:.1f}s")
        print(f"\n--- Step 4b: Applying friction perturbation (×{friction_scale_factor}) ---",
              flush=True)

    exp._apply_physics_perturbations()

    # ================================================================
    # Step 5: Setup observations, background, covariances for full period
    # ================================================================
    if rank == 0:
        print("\n--- Step 5: Setting up observations and background ---", flush=True)

    # Use "4dvar" trick to avoid dense B_lwme construction (OOM)
    original_method = exp.config.method
    exp.config.method = "4dvar"

    obs_points, obs_operator, global_obs_times = exp._setup_observations()
    exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, global_obs_times)
    background_error = exp._setup_background()
    B, R, B_lwme = exp._setup_covariances(obs_operator, obs_noise_stds)

    exp.config.method = original_method

    # Eq 38: Derive σ_b² from TLM Gram matrix — DC-WME only
    n_obs = obs_operator.get_num_observations()
    eq38_result = None
    if method == "dcwme":
        if rank == 0:
            print(f"\n--- Step 5a: Computing σ_b² from Eq 38 via TLM (DC-WME) ---")
            print(f"  Running forward solve from m_true with DA model ({nt_da_total} steps)...")
        from experiments.twin_experiment import ForwardModelWrapper
        prob.nt = nt_da_total
        gram_fwd = ForwardModelWrapper(
            solver=solver, problem=prob,
            solver_params=da_solver_params, t_start=t_da_start,
        )
        gram_trajectory, gram_jacobians = gram_fwd.solve(exp.m_true, store_jacobians=True)
        eq38_result = _compute_eq38_from_tlm(
            forward_model=gram_fwd,
            obs_operator=obs_operator, obs_cov=R,
            m_linearize=exp.m_true,
            observations=exp.observations, obs_times=global_obs_times,
            truth_trajectory=gram_trajectory, truth_jacobians=gram_jacobians,
            predictability_gamma=predictability_gamma,
            comm=comm, rank=rank,
        )
        _apply_eq38_to_B(B, eq38_result, rank=rank)
        for v in gram_trajectory:
            v.destroy()
        del gram_trajectory, gram_jacobians
        solver.storage.clear()
        import gc; gc.collect()

    if rank == 0:
        print(f"  Observation points: {n_obs}")
        print(f"  Global observation times: {global_obs_times}")
        print(f"  Background RMS error: {background_error:.6f}", flush=True)

    # Pre-compute gradient smoother (same for all windows)
    gradient_smoothing_length = 500.0
    gradient_smoother = None
    if config.background_correlation_length > 0:
        h_indices, u_indices, v_indices = exp._get_component_dof_indices(owned_only=True)
        smoothing_matrix = exp._build_smoothing_matrix(h_indices, gradient_smoothing_length)

        def gradient_smoother(grad_array):
            smoothed = grad_array.copy()
            smoothed[h_indices] = smoothing_matrix @ grad_array[h_indices]
            smoothed[u_indices] = smoothing_matrix @ grad_array[u_indices]
            smoothed[v_indices] = smoothing_matrix @ grad_array[v_indices]
            return smoothed

        if rank == 0:
            print(f"  Gradient smoother built (L={gradient_smoothing_length}m)")

    # Pre-compute boundary DOFs (same for all windows)
    boundary_dofs = exp._get_boundary_dofs() if config.interior_only else None

    # Memory cleanup before cycling
    exp.solver.storage.clear()
    gc.collect()

    # ================================================================
    # Step 6: Cycling loop
    # ================================================================
    per_window_results = []
    all_cost_history = []
    all_gradient_history = []
    all_analysis_state_rmse_history = []
    all_distance_from_background_history = []
    total_iterations = 0

    # Store original B diagonal for per-window scaling
    B_diag_original = B.diagonal.copy()
    B_inv_diag_original = B.inv_diagonal.copy()

    for w in range(n_windows):
        t_window_start = time.time()
        t_start_window = t_da_start + w * window_nt * dt
        t_end_window = t_da_start + (w + 1) * window_nt * dt

        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"  Window {w + 1}/{n_windows}: t = {t_start_window / 3600:.1f}h - "
                  f"{t_end_window / 3600:.1f}h")
            print(f"{'=' * 60}", flush=True)

        # a) Compute per-window background error for B scaling
        truth_idx = w * window_nt
        if w == 0:
            window_bg_error = background_error
        else:
            diff_bg = exp.m_background.copy()
            diff_bg.axpy(-1.0, exp.truth_trajectory[truth_idx])
            window_bg_error = np.sqrt(diff_bg.dot(diff_bg) / diff_bg.getSize())
            diff_bg.destroy()

        # Scale B proportionally to actual background error vs initial
        # This prevents overfitting when propagated background is already close to truth
        b_scale = min(window_bg_error / background_error, 1.0) if background_error > 0 else 1.0
        b_scale = max(b_scale, 0.01)  # Floor: at least 1% of original B

        # Restore original B and apply per-window scaling
        B.diagonal.copy(result=B.diagonal)  # no-op, overwrite below
        B_diag_original.copy(result=B.diagonal)
        B_inv_diag_original.copy(result=B.inv_diagonal)
        if b_scale < 1.0:
            B.diagonal.scale(b_scale)
            B.inv_diagonal.scale(1.0 / b_scale)

        if rank == 0:
            print(f"  Background error:  {window_bg_error:.6f} "
                  f"(B scale: {b_scale:.3f}x of original)", flush=True)

        # b) Subset observations for this window
        window_obs_indices = []
        window_local_times = []
        for i, gt in enumerate(global_obs_times):
            if w * window_nt <= gt <= (w + 1) * window_nt:
                # Remap to window-local time
                local_t = gt - w * window_nt
                # Include t=0 for first obs, but skip duplicates at boundaries
                if local_t == 0 and w > 0:
                    continue  # Skip boundary overlap (was end of previous window)
                window_obs_indices.append(i)
                window_local_times.append(local_t)

        window_observations = [exp.observations[i] for i in window_obs_indices]

        if rank == 0:
            print(f"  Window obs: {len(window_local_times)} at local times {window_local_times}",
                  flush=True)

        if len(window_observations) == 0:
            if rank == 0:
                print(f"  WARNING: No observations in window {w + 1}, skipping")
            # Propagate background forward
            if w < n_windows - 1:
                new_bg = _propagate_state_safe(
                    exp, window_nt, t_start_window, da_solver_params, rank
                )
                if new_bg is not None:
                    exp.m_background = new_bg
                else:
                    if rank == 0:
                        print(f"  WARNING: Propagation failed, stopping cycling")
                    break
            per_window_results.append({
                "window": w,
                "t_start_h": float(t_start_window / 3600),
                "t_end_h": float(t_end_window / 3600),
                "n_obs_times": 0,
                "skipped": True,
            })
            continue

        # c) Set problem.nt to window length for forward model
        prob.nt = window_nt

        # d) Create forward model for this window
        if rank == 0:
            print(f"  Creating forward model (t_start={t_start_window:.0f}s)", flush=True)
        forward_model = exp._create_forward_model(t_start=t_start_window)

        # e) Compute static L_wme for this window (Phase 4b only)
        static_L_wme = None
        static_diagnostics = None
        if method == "dcwme" and sub_label == "b":
            already_inflated = eq38_result is not None
            if rank == 0:
                print(f"  Computing static L_wme for window {w + 1} "
                      f"(skip_eq38_inflation={already_inflated})...", flush=True)
            t_lwme_start = time.time()
            static_L_wme, static_diagnostics = _compute_static_L_wme(
                obs_operator, B, len(window_local_times), obs_noise_level ** 2,
                exp.m_background, predictability_gamma=predictability_gamma,
                adaptive_gamma=adaptive_gamma,
                comm=comm, rank=rank,
                skip_eq38_inflation=already_inflated,
            )
            t_lwme = time.time() - t_lwme_start
            if rank == 0:
                print(f"  Static L_wme computed in {t_lwme:.1f}s", flush=True)

        # f) Create cost function
        if rank == 0:
            method_label = method.upper()
            if sub_label == "b":
                method_label += " (static L_wme)"
            elif sub_label == "c":
                method_label += " (dynamic L_wme)"
            print(f"  Setting up {method_label} cost function", flush=True)

        if method == "dcwme" and static_L_wme is not None:
            # Phase 4b: precomputed static L_wme
            from swe4dvar.data_assimilation.cost_functions import DCWMEFourDVarCost

            cost_function = DCWMEFourDVarCost(
                forward_model=forward_model,
                observation_operator=obs_operator,
                background_cov=B,
                observation_cov=R,
                m_background=exp.m_background,
                observations=window_observations,
                obs_times=window_local_times,
                predicted_cov_wme=static_L_wme,
                n_l_wme_samples=0,
                auto_inflate_B=False,
    
                predictability_gamma=predictability_gamma,
                adaptive_gamma=adaptive_gamma,
                comm=comm,
            )

            if boundary_dofs is not None:
                cost_function = ZeroBoundaryGradientCost(cost_function, boundary_dofs)
        else:
            # Phase 4a (4D-Var) or 4c (dynamic DC-WME)
            orig_observations = exp.observations
            exp.observations = window_observations
            cost_function = exp._setup_cost_function(
                forward_model, obs_operator, B, R, window_local_times, B_lwme=B_lwme
            )
            exp.observations = orig_observations

        # g) Attach gradient smoother
        if gradient_smoother is not None:
            inner_cost = cost_function
            while hasattr(inner_cost, 'base_cost'):
                inner_cost = inner_cost.base_cost
            inner_cost.gradient_smoother = gradient_smoother

        # h) Run optimization
        if rank == 0:
            print(f"  Running L-BFGS optimization (max {max_iterations} iters)...", flush=True)

        optimizer, opt_time = exp._run_optimization(cost_function)

        window_cost = [h["cost"] for h in optimizer.convergence_history]
        window_grad = [h["grad_norm"] for h in optimizer.convergence_history]
        all_cost_history.extend(window_cost)
        all_gradient_history.extend(window_grad)
        total_iterations += optimizer.iteration

        # Check if optimization failed (inf cost on first eval)
        opt_failed = window_cost and window_cost[0] >= 1e19

        if rank == 0:
            print(f"  Window {w + 1}: {optimizer.iteration} iters, "
                  f"converged={optimizer.converged}", flush=True)
            if window_cost:
                print(f"  Cost: {window_cost[0]:.1f} → {window_cost[-1]:.1f}")
            if opt_failed:
                print(f"  WARNING: Forward model failed — using background as analysis")

        # If optimization failed, use background as analysis
        if opt_failed:
            exp.m_analysis = exp.m_background.copy()

        # i) Compute per-window analysis error
        if w == 0:
            diff = exp.m_analysis.copy()
            diff.axpy(-1.0, exp.m_true)
            analysis_error = np.sqrt(diff.dot(diff) / diff.getSize())
            diff.destroy()
        else:
            diff = exp.m_analysis.copy()
            diff.axpy(-1.0, exp.truth_trajectory[truth_idx])
            analysis_error = np.sqrt(diff.dot(diff) / diff.getSize())
            diff.destroy()

        window_err_reduction = (window_bg_error - analysis_error) / window_bg_error * 100 if window_bg_error > 0 else 0.0

        if rank == 0:
            print(f"  Analysis error:    {analysis_error:.6f}")
            print(f"  Error reduction:   {window_err_reduction:.1f}%", flush=True)

        # i) Extract DC-WME diagnostics
        window_dcwme_diag = {}
        if method == "dcwme":
            inner = cost_function
            while hasattr(inner, 'base_cost'):
                inner = inner.base_cost

            if hasattr(inner, '_b_inflation_factor'):
                window_dcwme_diag['b_inflation_factor'] = float(inner._b_inflation_factor)
            if hasattr(inner, '_gram_eigenvalues') and inner._gram_eigenvalues is not None:
                window_dcwme_diag['gram_eigenvalues'] = [float(v) for v in inner._gram_eigenvalues]
            if hasattr(inner, '_L_wme') and inner._L_wme is not None:
                try:
                    L_wme_mat = inner._L_wme
                    if hasattr(L_wme_mat, 'mat'):
                        n = L_wme_mat.mat.getSize()[0]
                        L_dense_np = np.zeros((n, n))
                        rstart, rend = L_wme_mat.mat.getOwnershipRange()
                        for i in range(rstart, rend):
                            cols, vals = L_wme_mat.mat.getRow(i)
                            for c, v in zip(cols, vals):
                                L_dense_np[i, c] = v
                        l_eigvals = np.linalg.eigvalsh(L_dense_np)
                        window_dcwme_diag['l_wme_eigenvalues'] = [float(v) for v in l_eigvals]
                        window_dcwme_diag['l_wme_n_natural'] = int(
                            np.sum(l_eigvals > predictability_gamma * l_eigvals.max())
                        )
                except Exception as e:
                    if rank == 0:
                        print(f"  Warning: Could not extract L_wme eigenvalues: {e}")

            if static_diagnostics is not None:
                window_dcwme_diag['static_lwme'] = static_diagnostics

        t_window = time.time() - t_window_start
        per_window_results.append({
            "window": w,
            "t_start_h": float(t_start_window / 3600),
            "t_end_h": float(t_end_window / 3600),
            "n_obs_times": len(window_local_times),
            "iterations": int(optimizer.iteration),
            "converged": bool(optimizer.converged),
            "background_error": float(window_bg_error),
            "analysis_error": float(analysis_error),
            "error_reduction_pct": float(window_err_reduction),
            "cost_initial": float(window_cost[0]) if window_cost else None,
            "cost_final": float(window_cost[-1]) if window_cost else None,
            "cost_history": [float(c) for c in window_cost],
            "gradient_norm_history": [float(g) for g in window_grad],
            "optimization_time_s": float(opt_time),
            "window_time_s": float(t_window),
            "dcwme_diagnostics": window_dcwme_diag,
        })

        if rank == 0:
            print(f"  Window {w + 1} time: {t_window:.1f}s ({t_window / 60:.1f} min)")

        # j) Propagate analysis forward to next window
        if w < n_windows - 1:
            if rank == 0:
                print(f"  Propagating analysis to next window...", flush=True)
            new_bg = _propagate_state_safe(
                exp, window_nt, t_start_window, da_solver_params, rank
            )
            if new_bg is not None:
                exp.m_background = new_bg
            else:
                if rank == 0:
                    print(f"  WARNING: Propagation failed at window {w + 1}, stopping cycling")
                # Record remaining windows as skipped
                for w_skip in range(w + 1, n_windows):
                    t_s = t_da_start + w_skip * window_nt * dt
                    t_e = t_da_start + (w_skip + 1) * window_nt * dt
                    per_window_results.append({
                        "window": w_skip,
                        "t_start_h": float(t_s / 3600),
                        "t_end_h": float(t_e / 3600),
                        "n_obs_times": 0,
                        "skipped": True,
                        "skip_reason": "propagation_failure",
                    })
                break

        # k) Memory cleanup
        exp.solver.storage.clear()
        gc.collect()

    # Restore problem.nt
    prob.nt = nt_ramp + nt_da_total

    # ================================================================
    # Aggregate results
    # ================================================================
    total_time = time.time() - t_total_start

    # Final analysis error (from last window)
    final_analysis_error = per_window_results[-1]["analysis_error"] if per_window_results else background_error
    cumulative_reduction = (background_error - final_analysis_error) / background_error * 100

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"  CYCLING COMPLETE: {total_iterations} total iterations across {n_windows} windows")
        print(f"  Initial background error: {background_error:.6f}")
        print(f"  Final analysis error:     {final_analysis_error:.6f}")
        print(f"  Cumulative reduction:     {cumulative_reduction:.1f}%")
        print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"{'=' * 60}", flush=True)

    # Save results
    results = None
    if rank == 0:
        results = {
            "phase": f"4{sub_label}",
            "status": "success",
            "method": method,
            "l_wme_type": l_wme_type,
            "model_error": True,
            "model_error_type": "friction",
            "friction_scale_factor": friction_scale_factor,
            "config": {
                "dt": dt,
                "nt_ramp": nt_ramp,
                "n_windows": n_windows,
                "window_nt": window_nt,
                "nt_da_total": nt_da_total,
                "t_da_start_s": t_da_start,
                "obs_fraction": obs_fraction,
                "obs_frequency": obs_frequency,
                "obs_noise_level": obs_noise_level,
                "background_error_std": background_error_std,
                "eq38_sigma_b_sq": eq38_result["sigma_b_sq"] if eq38_result else None,
                "eq38_lambda_min_G": eq38_result["lambda_min_G"] if eq38_result else None,
                "max_iterations": max_iterations,
                "n_obs_points": n_obs,
                "predictability_gamma": predictability_gamma,

            },
            "per_window": per_window_results,
            "aggregate": {
                "total_iterations": total_iterations,
                "initial_background_error": float(background_error),
                "final_analysis_error": float(final_analysis_error),
                "cumulative_error_reduction_pct": float(cumulative_reduction),
                "total_time_s": float(total_time),
                "warmup_time_s": float(t_warmup),
                "truth_time_s": float(t_truth),
            },
            "convergence": {
                "cost_history": [float(c) for c in all_cost_history],
                "gradient_norm_history": [float(g) for g in all_gradient_history],
            },
        }

        results_file = output_dir / "data" / f"phase4{sub_label}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n  Results saved to: {results_file}")

    return results


def _propagate_state_safe(exp, window_nt, t_start, solver_params, rank=0):
    """Propagate analysis state forward by window_nt timesteps.

    Uses the same pattern as TwinExperiment._propagate_forward().
    Returns a PETSc Vec with the final state, or None if propagation fails.
    """
    exp.solver.storage.clear()

    # Set initial condition from analysis
    m_array = exp.m_analysis.getArray()
    u_owned_size = exp.solver.V.dofmap.index_map.size_local
    if len(m_array) == len(exp.solver.u_n.x.array):
        exp.solver.u_n.x.array[:] = m_array
        exp.solver.u_n_old.x.array[:] = m_array
        exp.solver.u.x.array[:] = m_array
    else:
        exp.solver.u_n.x.array[:u_owned_size] = m_array
        exp.solver.u_n_old.x.array[:u_owned_size] = m_array
        exp.solver.u.x.array[:u_owned_size] = m_array
        exp.solver.u_n.x.scatter_forward()
        exp.solver.u_n_old.x.scatter_forward()
        exp.solver.u.x.scatter_forward()

    # Set time and nt for propagation
    orig_nt = exp.problem.nt
    exp.problem.nt = window_nt
    exp.problem.t = t_start

    try:
        exp.solver.time_loop(
            solver_parameters=solver_params,
            stations=np.array([[0.0, 0.0, 0.0]]),
            plot_every=9999,
            save_state=True,
            enable_video=False,
        )

        if not exp.solver.storage.saved_states:
            if rank == 0:
                print("  WARNING: No states saved during propagation")
            exp.problem.nt = orig_nt
            exp.solver.storage.clear()
            return None

        # Get final state
        final_state = exp.solver.storage.saved_states[-1]
        from dolfinx import la
        vec = la.create_petsc_vector(
            exp.solver.V.dofmap.index_map,
            exp.solver.V.dofmap.index_map_bs,
        )
        vec.setArray(final_state[:u_owned_size])
        vec.assemble()

        # Sanity check: reject states with NaN or extremely large values
        arr = vec.getArray()
        if np.any(np.isnan(arr)) or np.any(np.abs(arr) > 1e10):
            if rank == 0:
                print(f"  WARNING: Propagated state has NaN or extreme values "
                      f"(max={np.nanmax(np.abs(arr)):.2e})")
            vec.destroy()
            exp.problem.nt = orig_nt
            exp.solver.storage.clear()
            return None

        exp.problem.nt = orig_nt
        exp.solver.storage.clear()
        return vec

    except Exception as e:
        if rank == 0:
            print(f"  WARNING: Propagation failed with error: {e}")
        exp.problem.nt = orig_nt
        exp.solver.storage.clear()
        return None


def run_phase_4(args):
    """Phase 4: Cycling DC-WME vs 4D-Var Comparison.

    Runs three cycling experiments:
    - 4a: Standard 4D-Var cycling
    - 4b: DC-WME cycling with static L_wme
    - 4c: DC-WME cycling with dynamic L_wme
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)
    comm.Barrier()

    if rank == 0:
        print("=" * 70)
        print("PHASE 4: CYCLING DC-WME vs 4D-VAR COMPARISON")
        print("=" * 70)
        print("  Sub-experiments:")
        print("    4a: Standard 4D-Var cycling (6 windows)")
        print("    4b: DC-WME cycling with static L_wme")
        print("    4c: DC-WME cycling with dynamic L_wme")
        print("=" * 70)

    sub = getattr(args, 'sub', None)
    if sub:
        subs_to_run = [sub]
    else:
        subs_to_run = ["a", "b", "c"]

    all_results = {}

    for sub_label in subs_to_run:
        if sub_label == "a":
            result = _run_cycling_experiment(
                args, sub_label="a", method="4dvar", output_dir=output_dir,
            )
        elif sub_label == "b":
            result = _run_cycling_experiment(
                args, sub_label="b", method="dcwme", output_dir=output_dir,
            )
        elif sub_label == "c":
            result = _run_cycling_experiment(
                args, sub_label="c", method="dcwme", output_dir=output_dir,
            )
        else:
            if rank == 0:
                print(f"Unknown sub-experiment: 4{sub_label}")
            continue

        all_results[f"4{sub_label}"] = result

    # Print comparison summary
    if rank == 0 and len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("PHASE 4 CYCLING COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Sub':<6} {'Method':<22} {'Cum. Err Red.':<14} {'Tot Iters':<10} "
              f"{'Time (min)':<10}")
        print(f"{'-' * 62}")
        for key, res in all_results.items():
            if res is not None:
                agg = res['aggregate']
                method_label = res['method']
                if res.get('l_wme_type', 'N/A') != 'N/A':
                    method_label += f" ({res['l_wme_type']})"
                print(f"{key:<6} {method_label:<22} "
                      f"{agg['cumulative_error_reduction_pct']:.1f}%{'':<9} "
                      f"{agg['total_iterations']:<10} "
                      f"{agg['total_time_s'] / 60:.1f}")

        # Per-window comparison
        print(f"\n{'=' * 70}")
        print("PER-WINDOW ERROR REDUCTION (%)")
        print(f"{'=' * 70}")
        header = f"{'Window':<8}"
        for key in all_results:
            header += f" {key:<12}"
        print(header)
        print(f"{'-' * (8 + 12 * len(all_results))}")

        first_res = next(iter(all_results.values()))
        if first_res is not None:
            n_win = len(first_res['per_window'])
            for w in range(n_win):
                row = f"W{w + 1:<7}"
                for key, res in all_results.items():
                    if res is not None and w < len(res['per_window']):
                        pw = res['per_window'][w]
                        if pw.get('skipped'):
                            row += f" {'skip':<12}"
                        else:
                            row += f" {pw['error_reduction_pct']:.1f}%{'':<8}"
                    else:
                        row += f" {'N/A':<12}"
                print(row)
        print(f"{'=' * 70}")

    return all_results


def run_phase_5(args):
    """Phase 5: Single Long-Window DC-WME vs 4D-Var Comparison.

    Same as Phase 3 but with a 71-step (~11.8h) DA window instead of 12-step (2h).
    Tests whether longer windows create meaningful eigenvalue spread in L_wme,
    which would enable DC-WME's directional predictability reweighting.

    - 5a: Standard 4D-Var baseline
    - 5b: DC-WME with static L_wme
    - 5c: DC-WME with dynamic L_wme
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)
    comm.Barrier()

    if rank == 0:
        print("=" * 70)
        print("PHASE 5: LONG-WINDOW DC-WME vs 4D-VAR COMPARISON")
        print("  71 timesteps (~11.8h), 48h ramp, single window")
        print("=" * 70)
        print("  Sub-experiments:")
        print("    5a: Standard 4D-Var (baseline)")
        print("    5b: DC-WME with static L_wme")
        print("    5c: DC-WME with dynamic L_wme")
        print("=" * 70)

    sub = getattr(args, 'sub', None)
    if sub:
        subs_to_run = [sub]
    else:
        subs_to_run = ["a", "b", "c"]

    all_results = {}

    for sub_label in subs_to_run:
        if sub_label == "a":
            result = _run_sub_experiment(
                args, sub_label="a", method="4dvar",
                output_dir=output_dir,
                nt_da=71, nt_ramp=288, phase_prefix="5",
            )
        elif sub_label == "b":
            result = _run_sub_experiment(
                args, sub_label="b", method="dcwme",
                output_dir=output_dir,
                nt_da=71, nt_ramp=288, phase_prefix="5",
            )
        elif sub_label == "c":
            result = _run_sub_experiment(
                args, sub_label="c", method="dcwme",
                output_dir=output_dir,
                nt_da=71, nt_ramp=288, phase_prefix="5",
            )
        else:
            if rank == 0:
                print(f"Unknown sub-experiment: 5{sub_label}")
            continue

        all_results[f"5{sub_label}"] = result

    # Print comparison summary
    if rank == 0 and len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("PHASE 5 COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Sub-exp':<10} {'Method':<20} {'Error Red.':<12} {'Iters':<8} {'Conv.':<8} {'Time (min)':<10}")
        print(f"{'-' * 68}")
        for key, res in all_results.items():
            if res is not None:
                r = res['results']
                t = res['timing']
                lwme = res.get('l_wme_type', 'N/A')
                method_label = f"{res['method']}"
                if lwme != 'N/A':
                    method_label += f" ({lwme})"
                print(f"{key:<10} {method_label:<20} {r['error_reduction']:.1f}%{'':<7} "
                      f"{r['num_iterations']:<8} {'Yes' if r['converged'] else 'No':<8} "
                      f"{t['total_s']/60:.1f}")
        print(f"{'=' * 70}")

    return all_results


# ========================================================================
# Phase 6: Parameter Sweep — 4D-Var vs Static DC-WME
# ========================================================================

SWEEP_BASELINE = {
    "obs_noise_level": 0.01,
    "obs_fraction": 0.1,
    "obs_frequency": 6,
    "background_error_std": 0.02,
    "nt_da": 12,
    "nt_ramp": 144,
    "friction_scale_factor": 1.15,
}

SWEEP_DIMS = {
    "noise": {
        "param": "obs_noise_level",
        "values": [0.01, 0.005, 0.05, 0.1, 0.001],
    },
    "obs_density": {
        "param": "obs_fraction",
        "values": [0.05, 0.1, 0.25, 0.5],
    },
    "obs_frequency": {
        "param": "obs_frequency",
        "values": [2, 4, 6, 12],
    },
    "bg_error": {
        "param": "background_error_std",
        "values": [0.01, 0.02, 0.05, 0.1],
    },
    "predictability_gamma": {
        "param": "predictability_gamma",
        "values": [0.01, 0.05, 0.1, 0.5, 1.0],
    },
    "window_length": {
        "param": "nt_da",
        "values": [6, 12, 36, 71],
        "nt_ramp_map": {6: 144, 12: 144, 36: 144, 71: 288},
    },
    "model_error": {
        "param": "friction_scale_factor",
        "values": [1.0, 1.15, 1.3, 1.5],
    },
}


def _print_sweep_summary(all_results):
    """Print formatted summary table of sweep results."""
    for dim_name, dim_results in all_results.items():
        print(f"\n{'=' * 70}")
        print(f"SWEEP: {dim_name}")
        print(
            f"{'Value':<15} {'4DVar':<10} {'DC-stat':<10} "
            f"{'4DVar+Eq38':<12} {'DC-dyn':<10}"
        )
        print(f"{'-' * 62}")
        for point in dim_results:
            val = point["value"]
            variants = [
                point.get("4dvar_baseline", point.get("4dvar", {})),
                point.get("dcwme_static", point.get("dcwme", {})),
                point.get("4dvar_eq38", {}),
                point.get("dcwme_dynamic", {}),
            ]

            rendered = []
            for result in variants:
                err = result.get("results", {}).get("error_reduction", None)
                if result.get("status") == "failed":
                    err = None
                rendered.append(f"{err:.1f}" if err is not None else "FAIL")

            print(f"{str(val):<15} {rendered[0]:<10} {rendered[1]:<10} {rendered[2]:<12} {rendered[3]:<10}")
        print(f"{'=' * 70}")


def _cleanup_petsc_objects():
    """Aggressively destroy PETSc objects to reclaim memory.

    PETSc/FEniCSx can leak memory across solver instantiations.
    This forces cleanup of the PETSc garbage and Python GC.
    """
    import gc
    gc.collect()

    try:
        from petsc4py import PETSc
        PETSc.garbage_cleanup()
    except Exception:
        pass

    # Second GC pass to collect anything PETSc released
    gc.collect()


def _get_process_memory_mb():
    """Return current process RSS in MB, or None if unavailable."""
    try:
        import resource
        # maxrss on macOS is in bytes, on Linux in KB
        import platform
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return rss / (1024 * 1024)
        else:
            return rss / 1024
    except Exception:
        return None


def _get_pid_rss_mb(pid):
    """Return RSS in MB for a given PID, or None if unavailable.

    Uses psutil if available, otherwise falls back to ``ps``.
    """
    try:
        import psutil
        proc = psutil.Process(pid)
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        pass

    # Fallback: parse ``ps`` output (works on macOS and Linux)
    try:
        import subprocess as _sp
        out = _sp.check_output(["ps", "-o", "rss=", "-p", str(pid)],
                               text=True, timeout=5).strip()
        if out:
            return int(out) / 1024  # ps reports KB
    except Exception:
        pass

    return None


MEM_WATCHDOG_INTERVAL_S = 5  # How often the watchdog checks child RSS


def _run_in_subprocess(run_config, mem_limit_mb, script_path):
    """Run a single sweep config as a child process with a memory watchdog.

    Parameters
    ----------
    run_config : dict
        Serializable dict with all parameters needed by ``_sweep_worker``.
    mem_limit_mb : float
        RSS limit in MB. The watchdog SIGKILLs the child if exceeded.
    script_path : str
        Absolute path to this script (used to launch the worker).

    Returns
    -------
    dict
        Result dict loaded from the child's output JSON, or a failure dict.
    """
    import subprocess as sp
    import signal
    import threading

    config_json = json.dumps(run_config)

    # Launch child: python <this_script> --_sweep_worker <json>
    cmd = [sys.executable, script_path, "--_sweep-worker", config_json]
    child = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)

    killed_by_watchdog = threading.Event()

    # --- Memory watchdog thread ---
    def watchdog():
        while child.poll() is None:
            rss = _get_pid_rss_mb(child.pid)
            if rss is not None and rss > mem_limit_mb:
                print(f"\n  WATCHDOG: Child PID {child.pid} using {rss:.0f} MB "
                      f"(limit {mem_limit_mb:.0f} MB) — sending SIGKILL",
                      flush=True)
                killed_by_watchdog.set()
                try:
                    import os
                    os.kill(child.pid, signal.SIGKILL)
                except OSError:
                    pass
                return
            # Sleep in small increments so we can exit quickly when child dies
            for _ in range(MEM_WATCHDOG_INTERVAL_S * 2):
                if child.poll() is not None:
                    return
                import time as _time
                _time.sleep(0.5)

    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()

    # Stream child stdout to parent stdout in real time
    output_lines = []
    for line in child.stdout:
        print(f"    [child] {line}", end="", flush=True)
        output_lines.append(line)

    child.wait()
    watcher.join(timeout=2)

    result_file = Path(run_config["result_file"])

    if killed_by_watchdog.is_set():
        result = {
            "phase": run_config["phase_prefix"] + run_config["sub_label"],
            "status": "failed",
            "error": f"Killed by memory watchdog (>{mem_limit_mb:.0f} MB)",
            "method": run_config["method"],
            "method_variant": run_config.get("variant_key"),
            "sweep_dimension": run_config.get("dim_name", ""),
            "sweep_param": run_config.get("param_name", ""),
            "sweep_value": run_config.get("val"),
        }
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        return result

    if child.returncode != 0:
        stderr_tail = "".join(output_lines[-20:])
        result = {
            "phase": run_config["phase_prefix"] + run_config["sub_label"],
            "status": "failed",
            "error": f"Child exited with code {child.returncode}: {stderr_tail[-500:]}",
            "method": run_config["method"],
            "method_variant": run_config.get("variant_key"),
            "sweep_dimension": run_config.get("dim_name", ""),
            "sweep_param": run_config.get("param_name", ""),
            "sweep_value": run_config.get("val"),
        }
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        return result

    # Read result JSON written by the child
    if result_file.exists():
        with open(result_file) as f:
            return json.load(f)

    return {
        "phase": run_config["phase_prefix"] + run_config["sub_label"],
        "status": "failed",
        "error": "Child completed but result file not found",
        "method": run_config["method"],
    }


def _sweep_worker_main(config_json):
    """Entry point for a subprocess that runs a single sweep config.

    Called via: ``python run_comparison.py --_sweep-worker '<json>'``

    The child process runs ``_run_sub_experiment``, writes the result JSON,
    then exits. All PETSc/FEniCSx memory is reclaimed by the OS on exit.
    """
    import os
    os.environ.setdefault("CC", "/usr/bin/clang")

    config = json.loads(config_json)

    args = argparse.Namespace(
        phase="6",
        sweep_dim=None,
        output_dir=config["output_dir"],
        adios_file=config["adios_file"],
        sub=None,
        verbose=True,
        mem_limit_gb=config.get("mem_limit_gb", 12.0),
        phase6_suite=config.get("phase6_suite", "controlled"),
    )

    sweep_params = config["sweep_params"]

    try:
        result = _run_sub_experiment(
            args,
            sub_label=config["sub_label"],
            method=config["method"],
            output_dir=Path(config["output_dir"]),
            nt_da=config["nt_da"],
            nt_ramp=config["nt_ramp"],
            phase_prefix=config["phase_prefix"],
            sweep_params=sweep_params,
            l_wme_mode=config.get("l_wme_mode", "dynamic"),
            apply_eq38_background_scaling=config.get("apply_eq38_background_scaling", False),
            method_variant_key=config.get("variant_key"),
        )
    except Exception as e:
        result_file = Path(config["result_file"])
        result = {
            "phase": config["phase_prefix"] + config["sub_label"],
            "status": "failed",
            "error": str(e),
            "method": config["method"],
            "method_variant": config.get("variant_key"),
            "sweep_dimension": config.get("dim_name", ""),
            "sweep_param": config.get("param_name", ""),
            "sweep_value": config.get("val"),
        }
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        sys.exit(1)

    sys.exit(0)


SWEEP_BATCH_SIZE = 10  # Max configs per batch before forced cleanup


def run_phase_6(args):
    """Phase 6: Parameter Sweep with controlled ablations.

    One-at-a-time sweep across 7 parameter dimensions.
    In controlled mode, each point runs:
      - classical 4D-Var baseline
      - DC-WME with static L_wme
      - 4D-Var with matched Eq. 38 background scaling
      - DC-WME with dynamic L_wme on the noise/window-length sweeps

    Memory safety: each config runs in its own subprocess.
    A watchdog thread in the parent monitors the child's RSS and SIGKILLs
    it if it exceeds --mem-limit-gb (default 12 GB). The parent process
    stays small and safe regardless of how much memory the child uses.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(exist_ok=True)
    comm.Barrier()

    mem_limit_mb = float(getattr(args, 'mem_limit_gb', 12.0)) * 1024
    script_path = str(Path(__file__).resolve())

    sweep_dim_filter = getattr(args, 'sweep_dim', None) or "all"
    dims_to_run = list(SWEEP_DIMS.keys()) if sweep_dim_filter == "all" else [sweep_dim_filter]
    suite = getattr(args, "phase6_suite", "controlled")

    all_sweep_results = {}

    for dim_name in dims_to_run:
        dim_config = SWEEP_DIMS[dim_name]
        param_name = dim_config["param"]
        values = dim_config["values"]

        if rank == 0:
            print(f"\n{'#' * 70}", flush=True)
            print(f"# SWEEP DIMENSION: {dim_name} ({param_name})", flush=True)
            print(f"#   Values: {values}", flush=True)
            print(f"# Memory limit: {mem_limit_mb:.0f} MB per child", flush=True)
            print(f"{'#' * 70}", flush=True)

        dim_results = []

        for val in values:
            sweep_params = dict(SWEEP_BASELINE)
            sweep_params[param_name] = val

            nt_da = sweep_params.pop("nt_da")
            nt_ramp = sweep_params.pop("nt_ramp")
            if dim_name == "window_length":
                nt_ramp = dim_config["nt_ramp_map"][val]
                nt_da = val

            val_str = str(val).replace(".", "p").replace(",", "_")
            point_results = {"dimension": dim_name, "param": param_name, "value": val}
            method_suite = _phase6_method_suite(dim_name, suite=suite)

            for spec in method_suite:
                method = spec["method"]
                sub_label = spec["sub_label"]
                variant_key = spec["variant_key"]
                phase_prefix = f"6_{dim_name}_{val_str}_"
                result_file = data_dir / f"phase{phase_prefix}{sub_label}_results.json"

                # Skip/resume
                if result_file.exists():
                    if rank == 0:
                        print(f"\n  Skipping {result_file.name} (already exists)", flush=True)
                    with open(result_file) as f:
                        result = json.load(f)
                    point_results[variant_key] = result
                    if sub_label == "a":
                        point_results["4dvar"] = result
                    elif sub_label == "b":
                        point_results["dcwme"] = result
                    continue

                if rank == 0:
                    print(f"\n  Running: {dim_name}={val}, variant={variant_key} "
                          f"[subprocess, limit={mem_limit_mb:.0f} MB]", flush=True)

                run_config = {
                    "output_dir": str(output_dir),
                    "adios_file": args.adios_file,
                    "sub_label": sub_label,
                    "method": method,
                    "variant_key": variant_key,
                    "l_wme_mode": spec["l_wme_mode"],
                    "apply_eq38_background_scaling": spec["apply_eq38_background_scaling"],
                    "nt_da": nt_da,
                    "nt_ramp": nt_ramp,
                    "phase_prefix": phase_prefix,
                    "sweep_params": sweep_params,
                    "result_file": str(result_file),
                    "dim_name": dim_name,
                    "param_name": param_name,
                    "val": val,
                    "mem_limit_gb": getattr(args, 'mem_limit_gb', 12.0),
                    "phase6_suite": suite,
                }

                result = _run_in_subprocess(run_config, mem_limit_mb, script_path)
                point_results[variant_key] = result
                if sub_label == "a":
                    point_results["4dvar"] = result
                elif sub_label == "b":
                    point_results["dcwme"] = result

                status = result.get("status", "unknown")
                if status == "failed":
                    if rank == 0:
                        err = result.get("error", "unknown error")
                        print(f"  FAILED: {err[:200]}", flush=True)
                else:
                    if rank == 0:
                        err_red = result.get("results", {}).get("error_reduction", "?")
                        print(f"  Done ({variant_key}): error_reduction={err_red}", flush=True)

            dim_results.append(point_results)

        all_sweep_results[dim_name] = dim_results

    # Save summary
    if rank == 0:
        summary_file = data_dir / "phase6_summary.json"
        with open(summary_file, "w") as f:
            json.dump(all_sweep_results, f, indent=2, default=str)
        print(f"\nSummary saved to {summary_file}", flush=True)

        _print_sweep_summary(all_sweep_results)
        sys.stdout.flush()

    return all_sweep_results


# ======================================================================
# Phase 7: Wind-Driven Twin Experiment
# ======================================================================

WIND_SWEEP_BASELINE = {
    "obs_noise_level": 0.01,
    "obs_fraction": 0.1,
    "obs_frequency": 6,
    "background_error_std": 0.02,
    "nt_da": 12,
    "nt_ramp": 144,
}


def _run_sub_experiment_wind(args, sub_label, method, wind_truth_file, wind_perturbed_file,
                              nt_da=12, nt_ramp=144, output_dir=None, phase_prefix="7_",
                              sweep_params=None, l_wme_mode="dynamic", n_windows=1,
                              apply_eq38_background_scaling=False,
                              method_variant_key=None,
                              skip_eq38=False):
    """Run a single wind-driven twin experiment, optionally with cycling.

    Two-problem architecture: truth problem uses truth wind, DA problem uses
    perturbed wind. Model error comes from wind field mismatch.

    Parameters
    ----------
    wind_truth_file : str
        Path to truth wind HDF5 (used for warm-up + truth trajectory).
    wind_perturbed_file : str
        Path to perturbed wind HDF5 (used for DA model).
    l_wme_mode : str
        "dynamic" (adjoint-computed, 306 adjoint solves) or "static" (H·B·Hᵀ).
    n_windows : int
        Number of cycling DA windows. nt_da is the TOTAL DA timesteps,
        divided into n_windows equal windows. Default 1 (no cycling).
    """
    import os
    os.environ.setdefault("CC", "/usr/bin/clang")

    from mpi4py import MPI
    from swe4dvar.forward.adcirc_problem import ADCIRCProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params
    from swe4dvar.physics.forcing import GriddedForcing
    from experiments.twin_experiment import (
        TwinExperiment, TwinExperimentConfig, ForwardModelWrapper,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ================================================================
    # Configuration
    # ================================================================
    dt = 600.0
    obs_fraction = 0.1
    obs_frequency = 6
    obs_noise_level = 0.01
    background_error_std = 0.02
    max_iterations = 15
    predictability_gamma = _get_sweep_value(sweep_params, "predictability_gamma", 0.1)

    if sweep_params:
        dt = _get_sweep_value(sweep_params, "dt", dt)
        obs_fraction = _get_sweep_value(sweep_params, "obs_fraction", obs_fraction)
        obs_frequency = _get_sweep_value(sweep_params, "obs_frequency", obs_frequency)
        obs_noise_level = _get_sweep_value(sweep_params, "obs_noise_level", obs_noise_level)
        background_error_std = _get_sweep_value(sweep_params, "background_error_std", background_error_std)

    nt_total = nt_ramp + nt_da

    method_label = method.upper()
    if method == "dcwme":
        method_label += f" ({l_wme_mode} L_wme)"
    elif apply_eq38_background_scaling:
        method_label += " + Eq38(B)"
    if method_variant_key:
        method_label += f" [{method_variant_key}]"

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"PHASE {phase_prefix}{sub_label.upper()}: {method_label}")
        print(f"  Wind truth: {wind_truth_file}")
        print(f"  Wind DA:    {wind_perturbed_file}")
        print(f"{'=' * 70}")

    # ================================================================
    # Step 1: Create truth problem with truth wind
    # ================================================================
    if rank == 0:
        print("\n--- Step 1: Creating truth problem ---")

    forcing_truth = GriddedForcing(wind_truth_file, lat0=35)
    prob_truth = ADCIRCProblem(
        adios_file=args.adios_file,
        spherical=True, solution_var="h", friction_law="mannings",
        wd=True, wd_alpha=1.5, dt=dt, bathy_adjustment=0,
        nt=nt_total, dramp=2.0, forcing=forcing_truth,
    )
    solver_truth = get_solver("DG")(prob_truth, theta=1.0, p_degree=[1, 1])

    warmup_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=10,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=True,
    )

    if rank == 0:
        n_dofs = len(solver_truth.u.x.array)
        print(f"  DOFs: {n_dofs}")

    # ================================================================
    # Step 2: Warm-up (truth wind, tidal ramp)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 2: Running {nt_ramp}-step warm-up ---")

    t_warmup_start = time.time()
    prob_truth.nt = nt_ramp
    solver_truth.time_loop(
        solver_parameters=warmup_params,
        stations=[], plot_every=9999,
        save_state=False, store_jacobians=False,
        enable_video=False, monitor_progress=(rank == 0),
    )
    t_warmup = time.time() - t_warmup_start
    t_da_start = prob_truth.t

    if rank == 0:
        print(f"  Warm-up: {t_warmup:.1f}s, t_da_start={t_da_start:.0f}s")

    # ================================================================
    # Step 3: Generate truth trajectory (DA window)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 3: Generating truth trajectory ({nt_da} steps) ---")

    t_truth_start = time.time()
    prob_truth.nt = nt_da
    # store_jacobians=True: needed for TLM Gram matrix in Eq 38 (DC-WME)
    solver_truth.time_loop(
        solver_parameters=warmup_params,
        stations=[], plot_every=9999,
        save_state=True, store_jacobians=True,
        enable_video=False, monitor_progress=(rank == 0),
    )
    t_truth = time.time() - t_truth_start

    # Extract truth trajectory as PETSc vectors
    from dolfinx import la
    u_owned_size = solver_truth.V.dofmap.index_map.size_local
    truth_trajectory = []
    for state_array in solver_truth.storage.saved_states:
        vec = la.create_petsc_vector(
            solver_truth.V.dofmap.index_map,
            solver_truth.V.dofmap.index_map_bs,
        )
        vec.setArray(state_array[:u_owned_size])
        vec.assemble()
        truth_trajectory.append(vec)

    # Deep-copy Jacobians before truth solver is deleted (needed for Eq 38 TLM).
    # Must use copy=True — bare PETSc.Mat.duplicate() preserves the sparsity
    # pattern but leaves values UNSET (zero). The next `del solver_truth` at
    # line ~4345 then destroys the real originals, leaving the TLM adjoint with
    # a list of zero-valued matrices. This is the same bug documented in
    # docs/idealized_inlet_jacobian_handoff_trace.md — the Shinnecock Phase
    # 3/5 "all eigenvalues identical" DC-WME degeneracy is very likely an
    # artifact of this same missing flag rather than a tidal-linearity finding.
    truth_jacobians = None
    if len(solver_truth.storage.saved_jacobians) > 0:
        truth_jacobians = [J.duplicate(copy=True) for J in solver_truth.storage.saved_jacobians]
        if rank == 0:
            print(f"  Jacobians deep-copied: {len(truth_jacobians)} matrices")

    m_true = truth_trajectory[0].copy()

    if rank == 0:
        print(f"  Truth: {len(truth_trajectory)} states in {t_truth:.1f}s")

    # Extract truth IC at DA start for background perturbation
    truth_ic_array = m_true.getArray().copy()

    # ================================================================
    # Step 4: Clean up truth, create DA problem with perturbed wind
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 4: Creating DA problem with perturbed wind ---")

    del solver_truth, prob_truth, forcing_truth
    import gc; gc.collect()

    forcing_da = GriddedForcing(wind_perturbed_file, lat0=35)
    prob_da = ADCIRCProblem(
        adios_file=args.adios_file,
        spherical=True, solution_var="h", friction_law="mannings",
        wd=True, wd_alpha=1.5, dt=dt, bathy_adjustment=0,
        nt=nt_da, dramp=2.0, forcing=forcing_da,
    )
    solver_da = get_solver("DG")(prob_da, theta=1.0, p_degree=[1, 1])

    da_solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=50,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=False,
        reduction_it=10,
    )

    # ================================================================
    # Step 5: Set up TwinExperiment with injected truth
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 5: Setting up observations and background ---")

    config_kwargs = dict(
        method=method if method != "dcwme" else "4dvar",  # Prevent dense B_lwme
        obs_fraction=obs_fraction,
        obs_frequency=obs_frequency,
        obs_noise_level=obs_noise_level,
        background_error_std=background_error_std,
        background_correlation_length=500.0,
        max_iterations=max_iterations,
        max_funcs=30,
        gradient_tolerance=1e-3,
        cost_tolerance=1e-4,
        n_windows=1,
        perturb_friction=False,  # Model error is from wind, not friction
        friction_scale_factor=1.0,
        use_bounds=True,
        h_min=0.01,
        interior_only=True,
        component_aware_cov=True,
        verbose=(rank == 0),
    )
    config = TwinExperimentConfig(**config_kwargs)

    exp = TwinExperiment(
        problem=prob_da, solver=solver_da, config=config,
        solver_params=da_solver_params, comm=comm,
    )

    # Inject truth (skip _generate_truth which would run with DA wind)
    exp.truth_trajectory = truth_trajectory
    exp.m_true = m_true
    exp.t_da_start = t_da_start

    # Set up observations, background, covariances using the injected truth
    obs_points, obs_operator, obs_times = exp._setup_observations()
    exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
    background_error = exp._setup_background()
    B, R, B_lwme = exp._setup_covariances(obs_operator, obs_noise_stds)

    n_obs = obs_operator.get_num_observations()
    N_obs_times = len(obs_times)

    # ================================================================
    # Eq 38: Derive σ_b² from TLM Gram matrix — DC-WME only
    # Runs throwaway forward solve over full DA span, computes
    # G[i,j] = a_i^T a_j where a_i = J_wme^T e_i, then:
    # σ_b² ≥ γ / λ_min(G)
    # ================================================================
    eq38_result = None
    eq38_scale = 1.0
    if skip_eq38:
        if rank == 0:
            print(f"\n--- Step 5a: SKIPPED (--skip-eq38 flag set, using B as-is) ---")
            print(f"  min(B) = {B.min_eigenvalue():.6e}")
    elif (method == "dcwme" or apply_eq38_background_scaling) and truth_jacobians is not None:
        if rank == 0:
            print(f"\n--- Step 5a: Computing σ_b² from Eq 38 via TLM ---")
            print(f"  Using truth trajectory ({len(truth_trajectory)} states, "
                  f"{len(truth_jacobians)} Jacobians) — no extra forward solve")

        # Use truth trajectory + Jacobians directly for TLM Gram matrix.
        # The truth solver ran with truth wind (stable, converges easily).
        # Need a ForwardModelWrapper for WME QoI constructor (won't solve).
        from experiments.twin_experiment import ForwardModelWrapper
        prob_da.nt = nt_da
        gram_fwd = ForwardModelWrapper(
            solver=solver_da, problem=prob_da,
            solver_params=da_solver_params, t_start=t_da_start,
        )

        eq38_result = _compute_eq38_from_tlm(
            forward_model=gram_fwd,
            obs_operator=obs_operator, obs_cov=R,
            m_linearize=m_true,
            observations=exp.observations, obs_times=obs_times,
            truth_trajectory=truth_trajectory, truth_jacobians=truth_jacobians,
            predictability_gamma=predictability_gamma,
            comm=comm, rank=rank,
        )
        eq38_scale = _apply_eq38_to_B(B, eq38_result, rank=rank)
    elif method == "dcwme" or apply_eq38_background_scaling:
        if rank == 0:
            print(f"\n  WARNING: No truth Jacobians available for Eq 38 TLM computation")

    if rank == 0:
        print(f"  Final min(B) = {B.min_eigenvalue():.6e}")
        print(f"  Observations: {n_obs} points, times={obs_times}")
        print(f"  Background error: {background_error:.6f}")

    # ================================================================
    # Step 5b: Compute L_wme if DC-WME
    # ================================================================
    static_L_wme = None
    static_diagnostics = None
    if method == "dcwme":
        if l_wme_mode == "static":
            # If TLM-based Eq 38 already inflated B (step 5a), skip the
            # internal H·H^T inflation to avoid double-inflation.
            already_inflated = eq38_result is not None
            if rank == 0:
                print(f"\n--- Step 5b: Computing STATIC L_wme "
                      f"(skip_eq38_inflation={already_inflated}) ---")
            static_L_wme, static_diagnostics = _compute_static_L_wme(
                obs_operator, B, len(obs_times), obs_noise_level ** 2,
                m_true, predictability_gamma=predictability_gamma,
                adaptive_gamma=True,
                comm=comm, rank=rank,
                skip_eq38_inflation=already_inflated,
            )
        elif l_wme_mode == "dynamic":
            if rank == 0:
                print("\n--- Step 5b: Dynamic L_wme will be computed during cost init ---")
        # dynamic: L_wme computed inside DCWMEFourDVarCost via adjoint solves

    # Memory cleanup
    exp.solver.storage.clear()
    gc.collect()

    # ================================================================
    # Steps 6-9: DA optimization (single window or cycling)
    # ================================================================
    window_nt = nt_da // n_windows if n_windows > 1 else nt_da
    if n_windows > 1 and nt_da % n_windows != 0:
        raise ValueError(f"nt_da ({nt_da}) must be divisible by n_windows ({n_windows})")

    # Pre-build gradient smoother (reused across windows)
    gradient_smoothing_length = 500.0
    h_indices = u_indices = v_indices = smoothing_matrix = None
    if config.background_correlation_length > 0:
        h_indices, u_indices, v_indices = exp._get_component_dof_indices(owned_only=True)
        smoothing_matrix = exp._build_smoothing_matrix(h_indices, gradient_smoothing_length)

    boundary_dofs = exp._get_boundary_dofs() if config.interior_only else None

    exp.optimization_iteration_callback = _make_state_rmse_iteration_callback(
        exp.m_true, exp.m_background
    )

    all_cost_history = []
    all_gradient_history = []
    all_analysis_state_rmse_history = []
    all_distance_from_background_history = []
    total_opt_time = 0.0
    total_iterations = 0
    last_converged = False
    window_results = []

    if rank == 0 and n_windows > 1:
        print(f"\n{'='*60}")
        print(f"  CYCLING: {n_windows} windows × {window_nt} steps ({window_nt*dt/3600:.1f}h each)")
        print(f"{'='*60}")

    for w in range(n_windows):
        t_start_window = t_da_start + w * window_nt * dt

        if rank == 0:
            if n_windows > 1:
                print(f"\n--- Window {w+1}/{n_windows}: t={t_start_window/3600:.1f}h-{(t_start_window + window_nt*dt)/3600:.1f}h ---")
            else:
                print(f"\n--- Step 6: Creating forward model (t_start={t_start_window:.0f}s) ---")

        # a) Temporarily set problem to window length
        prob_da.nt = window_nt

        # b) Subset observations for this window
        window_obs_indices = []
        window_local_times = []
        for i, gt in enumerate(obs_times):
            if n_windows == 1:
                window_obs_indices.append(i)
                window_local_times.append(gt)
            else:
                global_step = gt  # obs_times are relative to DA start (0-based)
                win_start = w * window_nt
                win_end = (w + 1) * window_nt
                if win_start <= global_step <= win_end:
                    window_obs_indices.append(i)
                    window_local_times.append(global_step - win_start)

        window_observations = [exp.observations[i] for i in window_obs_indices]

        if rank == 0 and n_windows > 1:
            print(f"  Window obs: {len(window_local_times)} at local times {window_local_times}")

        if len(window_observations) == 0:
            if rank == 0:
                print(f"  WARNING: No observations in window {w+1}, skipping")
            continue

        # c) Create forward model for this window
        forward_model = ForwardModelWrapper(
            solver=solver_da,
            problem=prob_da,
            solver_params=da_solver_params,
            t_start=t_start_window,
        )

        # d) Setup cost function
        if method == "dcwme" and l_wme_mode == "static":
            from swe4dvar.data_assimilation.cost_functions import DCWMEFourDVarCost
            # Recompute static L_wme for this window's obs times.
            # B was already inflated by TLM-based Eq 38 (step 5a) if available.
            already_inflated = eq38_result is not None
            win_static_L_wme, _ = _compute_static_L_wme(
                obs_operator, B, len(window_local_times), obs_noise_level ** 2,
                exp.m_background, predictability_gamma=predictability_gamma,
                adaptive_gamma=True,
                comm=comm, rank=rank,
                skip_eq38_inflation=already_inflated,
            )
            cost_function = DCWMEFourDVarCost(
                forward_model=forward_model,
                observation_operator=obs_operator,
                background_cov=B, observation_cov=R,
                m_background=exp.m_background,
                observations=window_observations, obs_times=window_local_times,
                predicted_cov_wme=win_static_L_wme,
                n_l_wme_samples=0,
                auto_inflate_B=False,
    
                predictability_gamma=predictability_gamma,
                adaptive_gamma=True, comm=comm,
            )
        elif method == "dcwme" and l_wme_mode == "dynamic":
            from swe4dvar.data_assimilation.cost_functions import DCWMEFourDVarCost
            cost_function = DCWMEFourDVarCost(
                forward_model=forward_model,
                observation_operator=obs_operator,
                background_cov=B, observation_cov=R,
                m_background=exp.m_background,
                observations=window_observations, obs_times=window_local_times,
                predicted_cov_wme=None,
                n_l_wme_samples=100,
                auto_inflate_B=True,
    
                predictability_gamma=predictability_gamma,
                adaptive_gamma=True, comm=comm,
            )
        else:
            orig_obs = exp.observations
            exp.observations = window_observations
            exp.config.method = "4dvar"
            cost_function = exp._setup_cost_function(
                forward_model, obs_operator, B, R, window_local_times, B_lwme=B_lwme
            )
            exp.observations = orig_obs

        # Wrap with boundary gradient zeroing
        if boundary_dofs is not None:
            from experiments.twin_experiment import ZeroBoundaryGradientCost
            cost_function = ZeroBoundaryGradientCost(cost_function, boundary_dofs)

        # Attach gradient smoother
        if smoothing_matrix is not None:
            def gradient_smoother(grad_array, _h=h_indices, _u=u_indices, _v=v_indices, _sm=smoothing_matrix):
                smoothed = grad_array.copy()
                smoothed[_h] = _sm @ grad_array[_h]
                smoothed[_u] = _sm @ grad_array[_u]
                smoothed[_v] = _sm @ grad_array[_v]
                return smoothed

            inner_cost = cost_function
            while hasattr(inner_cost, 'base_cost'):
                inner_cost = inner_cost.base_cost
            inner_cost.gradient_smoother = gradient_smoother

        # e) Run optimization
        if rank == 0:
            print(f"  Running L-BFGS optimization...")

        optimizer, opt_time_w = exp._run_optimization(cost_function)

        w_costs = [h["cost"] for h in optimizer.convergence_history]
        w_grads = [h["grad_norm"] for h in optimizer.convergence_history]
        w_analysis_state_rmse = [
            float(h["analysis_state_rmse"])
            for h in optimizer.convergence_history
            if "analysis_state_rmse" in h
        ]
        w_background_distance = [
            float(h["distance_from_background_rmse"])
            for h in optimizer.convergence_history
            if "distance_from_background_rmse" in h
        ]
        all_cost_history.extend(w_costs)
        all_gradient_history.extend(w_grads)
        all_analysis_state_rmse_history.extend(w_analysis_state_rmse)
        all_distance_from_background_history.extend(w_background_distance)
        total_opt_time += opt_time_w
        total_iterations += optimizer.iteration
        last_converged = optimizer.converged

        # Evaluate this window (use window-local obs times, not global)
        w_err_red = 0.0
        if w_costs and w_costs[-1] < 1e19:
            w_analysis_error, w_err_red, _, _, _, _ = exp._evaluate_results(
                obs_operator, window_local_times, background_error
            )
        else:
            w_analysis_error = background_error

        window_results.append({
            "window": w, "error_reduction": float(w_err_red),
            "analysis_error": float(w_analysis_error),
            "iterations": int(optimizer.iteration),
        })

        if rank == 0:
            print(f"  Window {w+1}: err_red={w_err_red:.1f}%, iters={optimizer.iteration}")

        # f) Propagate analysis to next window's background
        if w < n_windows - 1 and hasattr(exp, 'm_analysis') and exp.m_analysis is not None:
            if rank == 0:
                print(f"  Propagating analysis to next window...")
            next_bg = exp._propagate_forward(exp.m_analysis, window_nt, t_start_window)
            exp.m_background = next_bg

        # Cleanup this window (keep last cost function for diagnostics)
        last_cost_function = cost_function
        exp.solver.storage.clear()
        gc.collect()

    # Restore nt
    prob_da.nt = nt_da

    # Final evaluation against truth
    cost_history = all_cost_history
    gradient_history = all_gradient_history
    opt_time = total_opt_time

    if cost_history and cost_history[-1] >= 1e19:
        analysis_error = background_error
        error_reduction = 0.0
        innov_mean = innov_std = mean_rmse = data_misfit = 0.0
    else:
        analysis_error, error_reduction, innov_mean, innov_std, mean_rmse, data_misfit = (
            exp._evaluate_results(obs_operator, obs_times, background_error)
        )

    total_time = t_warmup + t_truth + opt_time

    if rank == 0:
        print(f"\n  Background error: {background_error:.6f}")
        print(f"  Analysis error:   {analysis_error:.6f}")
        print(f"  Error reduction:  {error_reduction:.1f}%")
        if n_windows > 1:
            print(f"  Windows: {n_windows}, Total iterations: {total_iterations}")

    # ================================================================
    # Step 10: Extract DC-WME diagnostics
    # ================================================================
    dcwme_diagnostics = {}
    cost_function = locals().get('last_cost_function', None)
    if method == "dcwme" and cost_function is not None:
        inner = cost_function
        while hasattr(inner, 'base_cost'):
            inner = inner.base_cost

        if hasattr(inner, '_b_inflation_factor'):
            dcwme_diagnostics['b_inflation_factor'] = float(inner._b_inflation_factor)
        if hasattr(inner, '_gram_eigenvalues') and inner._gram_eigenvalues is not None:
            dcwme_diagnostics['gram_eigenvalues'] = [float(v) for v in inner._gram_eigenvalues]
        if hasattr(inner, '_L_wme') and inner._L_wme is not None:
            try:
                L_wme_mat = inner._L_wme
                if hasattr(L_wme_mat, 'mat'):
                    n = L_wme_mat.mat.getSize()[0]
                    L_dense_np = np.zeros((n, n))
                    rstart, rend = L_wme_mat.mat.getOwnershipRange()
                    for i in range(rstart, rend):
                        cols, vals = L_wme_mat.mat.getRow(i)
                        for c, v in zip(cols, vals):
                            L_dense_np[i, c] = v
                    l_eigvals = np.linalg.eigvalsh(L_dense_np)
                    dcwme_diagnostics['l_wme_eigenvalues'] = [float(v) for v in l_eigvals]
                    dcwme_diagnostics['l_wme_spread_pct'] = float(
                        100.0 * (l_eigvals.max() - l_eigvals.min()) / max(l_eigvals.mean(), 1e-30)
                    )
                    dcwme_diagnostics['l_wme_ratio'] = float(
                        l_eigvals.max() / max(l_eigvals.min(), 1e-30)
                    )
                    dcwme_diagnostics['l_wme_spectrum'] = _summarize_eigenvalues(l_eigvals)
                    if np.all(l_eigvals > 1.0):
                        weights = 1.0 - 1.0 / l_eigvals
                        dcwme_diagnostics['effective_weight_min'] = float(weights.min())
                        dcwme_diagnostics['effective_weight_max'] = float(weights.max())
            except Exception as e:
                if rank == 0:
                    print(f"  Warning: Could not extract L_wme eigenvalues: {e}")

        if static_diagnostics is not None:
            dcwme_diagnostics['static_lwme'] = static_diagnostics

    # ================================================================
    # Step 11: Save results
    # ================================================================
    results = None
    if rank == 0:
        results = {
            "phase": f"{phase_prefix}{sub_label}",
            "status": "success",
            "method": method,
            "method_variant": method_variant_key or method,
            "model_error": True,
            "model_error_type": "wind",
            "l_wme_type": l_wme_mode if method == "dcwme" else "N/A",
            "wind_truth_file": str(wind_truth_file),
            "wind_perturbed_file": str(wind_perturbed_file),
            "experiment_controls": {
                "apply_eq38_background_scaling": bool(apply_eq38_background_scaling),
                "eq38_scaling_basis": "WME-TLM" if (eq38_result is not None) else "none",
                "method_variant_key": method_variant_key,
            },
            "config": {
                "dt": dt, "nt_ramp": nt_ramp, "nt_da": nt_da,
                "n_windows": n_windows, "window_nt": window_nt,
                "t_da_start_s": t_da_start,
                "obs_fraction": obs_fraction, "obs_frequency": obs_frequency,
                "obs_noise_level": obs_noise_level,
                "background_error_std": background_error_std,
                "eq38_sigma_b_sq": eq38_result["sigma_b_sq"] if eq38_result else None,
                "eq38_scale_factor": eq38_scale,
                "eq38_lambda_min_G": eq38_result["lambda_min_G"] if eq38_result else None,
                "max_iterations": max_iterations,
                "n_obs_points": n_obs, "obs_times": obs_times,
                "predictability_gamma": predictability_gamma,

            },
            "window_results": window_results,
            "results": {
                "background_error": float(background_error),
                "analysis_error": float(analysis_error),
                "error_reduction": float(error_reduction),
                "mean_rmse": float(mean_rmse),
                "data_misfit": float(data_misfit),
                "innovation_mean": float(innov_mean),
                "innovation_std": float(innov_std),
                "num_iterations": int(optimizer.iteration),
                "converged": bool(optimizer.converged),
            },
            "convergence": {
                "cost_history": [float(c) for c in cost_history],
                "gradient_norm_history": [float(g) for g in gradient_history],
                "analysis_state_rmse_history": all_analysis_state_rmse_history,
                "distance_from_background_rmse_history": all_distance_from_background_history,
            },
            "dcwme_diagnostics": dcwme_diagnostics,
            "eq38_diagnostics": eq38_result.copy() if eq38_result is not None else {},
            "timing": {
                "warmup_s": float(t_warmup),
                "truth_generation_s": float(t_truth),
                "optimization_s": float(opt_time),
                "total_s": float(total_time),
            },
        }

        results_file = output_dir / "data" / f"phase{phase_prefix}{sub_label}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 70}")
        print(f"  Error reduction: {error_reduction:.1f}%")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Results: {results_file}")
        print(f"{'=' * 70}")

    del optimizer, cost_function, forward_model, exp, solver_da, prob_da
    del B, R, B_lwme, obs_operator
    _cleanup_petsc_objects()

    return results


def _wind_sweep_worker_main(config_json):
    """Entry point for Phase 7 subprocess worker."""
    import os
    os.environ.setdefault("CC", "/usr/bin/clang")

    config = json.loads(config_json)

    args = argparse.Namespace(
        phase="7",
        output_dir=config["output_dir"],
        adios_file=config["adios_file"],
        mem_limit_gb=config.get("mem_limit_gb", 12.0),
        verbose=True,
    )

    try:
        result = _run_sub_experiment_wind(
            args,
            sub_label=config["sub_label"],
            method=config["method"],
            wind_truth_file=config["wind_truth_file"],
            wind_perturbed_file=config["wind_perturbed_file"],
            output_dir=Path(config["output_dir"]),
            nt_da=config["nt_da"],
            nt_ramp=config["nt_ramp"],
            phase_prefix=config["phase_prefix"],
            sweep_params=config.get("sweep_params"),
            l_wme_mode=config.get("l_wme_mode", "dynamic"),
            n_windows=config.get("n_windows", 1),
            apply_eq38_background_scaling=config.get("apply_eq38_background_scaling", False),
            method_variant_key=config.get("variant_key"),
            skip_eq38=config.get("skip_eq38", False),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        result_file = Path(config["result_file"])
        result = {
            "phase": config["phase_prefix"] + config["sub_label"],
            "status": "failed",
            "error": str(e),
            "method": config["method"],
            "method_variant": config.get("variant_key"),
        }
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        sys.exit(1)

    sys.exit(0)


def run_phase_7(args):
    """Phase 7: restored WSE wind-ramp twin experiment.

    This phase is intentionally a pure state-estimation problem:
    the unknown is the initial hydrodynamic state at the start of the DA
    window, while the wind forcing is prescribed from the truth or perturbed
    HDF5 file and is never estimated. The original three-method sweep is
    preserved: 4D-Var, DC-WME with dynamic L_wme, and DC-WME with static
    L_wme.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    wind_dir = output_dir / "wind_fields"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(exist_ok=True)
        wind_dir.mkdir(exist_ok=True)
    comm.Barrier()

    mem_limit_mb = float(getattr(args, 'mem_limit_gb', 12.0)) * 1024
    script_path = str(Path(__file__).resolve())

    dt = 600.0
    nt_ramp = WIND_SWEEP_BASELINE["nt_ramp"]
    nt_da = WIND_SWEEP_BASELINE["nt_da"]

    # ================================================================
    # Step 1: Generate wind HDF5 files
    # ================================================================
    if rank == 0:
        print("=" * 70)
        print("PHASE 7: WIND-DRIVEN TWIN EXPERIMENT")
        print("=" * 70)
        print("\n--- Generating wind fields ---")

        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from wind_models import (
            WindGridConfig, HollandHurricaneConfig,
            generate_holland_wind_field, generate_zero_wind_field,
            write_wind_hdf5, generate_perturbed_config, DEFAULT_TRACK,
        )

        grid = WindGridConfig()
        times = np.arange(0, (nt_ramp + nt_da + 1) * dt, dt)
        track_shift_km = 15.0

        for Vmax in [0, 10, 20, 30, 40]:
            truth_file = wind_dir / f"holland_V{Vmax}_truth.h5"
            pert_file = wind_dir / f"holland_V{Vmax}_pert{track_shift_km:.0f}km.h5"

            if not truth_file.exists():
                if Vmax == 0:
                    wx, wy, p = generate_zero_wind_field(grid, times)
                else:
                    config_h = HollandHurricaneConfig(
                        track_waypoints=DEFAULT_TRACK, Vmax=float(Vmax),
                    )
                    wx, wy, p = generate_holland_wind_field(
                        config_h, grid, times, wind_ramp_s=43200.0,
                    )
                write_wind_hdf5(str(truth_file), grid, times, wx, wy, p)

            if not pert_file.exists():
                if Vmax == 0:
                    # No perturbation for zero wind (tidal-only control)
                    import shutil
                    shutil.copy(truth_file, pert_file)
                    print(f"  Copied zero-wind: {pert_file.name}")
                else:
                    config_h = HollandHurricaneConfig(
                        track_waypoints=DEFAULT_TRACK, Vmax=float(Vmax),
                    )
                    config_pert = generate_perturbed_config(
                        config_h, "track_shift", track_shift_km,
                    )
                    wx_p, wy_p, p_p = generate_holland_wind_field(
                        config_pert, grid, times, wind_ramp_s=43200.0,
                    )
                    write_wind_hdf5(str(pert_file), grid, times, wx_p, wy_p, p_p)

    comm.Barrier()

    # ================================================================
    # Step 2: Run sweep
    # ================================================================
    track_shift_km = 15.0
    Vmax_values = [0, 10, 20, 30, 40]
    methods = [
        ("4dvar", "a", "N/A", False, "4dvar"),
        ("dcwme", "b", "dynamic", False, "dynamic"),
        ("dcwme", "c", "static", False, "static"),
    ]

    all_results = []

    for Vmax in Vmax_values:
        truth_file = str(wind_dir / f"holland_V{Vmax}_truth.h5")
        pert_file = str(wind_dir / f"holland_V{Vmax}_pert{track_shift_km:.0f}km.h5")

        if rank == 0:
            print(f"\n{'#' * 70}")
            print(f"# Vmax = {Vmax} m/s")
            print(f"# Memory limit: {mem_limit_mb:.0f} MB per child")
            print(f"{'#' * 70}")

        point_results = {"Vmax": Vmax}

        for method, sub_label, l_wme_mode, apply_eq38_background_scaling, variant_key in methods:
            phase_prefix = f"7_wind{Vmax}_"
            result_file = data_dir / f"phase{phase_prefix}{sub_label}_results.json"

            # Skip/resume
            if result_file.exists():
                if rank == 0:
                    print(f"\n  Skipping {result_file.name} (already exists)")
                with open(result_file) as f:
                    result = json.load(f)
                point_results[variant_key] = result
                continue

            if rank == 0:
                method_label = method.upper()
                if method == "dcwme":
                    method_label += f" ({l_wme_mode})"
                elif apply_eq38_background_scaling:
                    method_label += " + Eq38(B)"
                print(f"\n  Running: Vmax={Vmax}, {method_label} "
                      f"[subprocess, limit={mem_limit_mb:.0f} MB]")

            run_config = {
                "output_dir": str(output_dir),
                "adios_file": args.adios_file,
                "sub_label": sub_label,
                "method": method,
                "wind_truth_file": truth_file,
                "wind_perturbed_file": pert_file,
                "nt_da": nt_da,
                "nt_ramp": nt_ramp,
                "phase_prefix": phase_prefix,
                "sweep_params": dict(WIND_SWEEP_BASELINE),
                "l_wme_mode": l_wme_mode,
                "apply_eq38_background_scaling": apply_eq38_background_scaling,
                "variant_key": variant_key,
                "result_file": str(result_file),
                "mem_limit_gb": getattr(args, 'mem_limit_gb', 12.0),
                "worker_type": "wind",  # Tells dispatcher to use wind worker
            }

            result = _run_in_subprocess(run_config, mem_limit_mb, script_path)
            point_results[variant_key] = result

            status = result.get("status", "unknown")
            if status == "failed":
                if rank == 0:
                    print(f"  FAILED: {result.get('error', '?')[:200]}")
            else:
                err_red = result.get("results", {}).get("error_reduction", "?")
                if rank == 0:
                    print(f"  Done: error_reduction={err_red}")

        all_results.append(point_results)

    # ================================================================
    # Step 3: Summary
    # ================================================================
    if rank == 0:
        summary_file = data_dir / "phase7_summary.json"
        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n{'=' * 70}")
        print("PHASE 7 SUMMARY: Restored WSE Wind-Ramp Experiment")
        print(f"{'=' * 70}")
        print(f"{'Vmax':>6} {'4DVar':>8} {'DC-dyn':>8} {'DC-stat':>8} {'L_spread':>10}")
        print("-" * 52)
        for point in all_results:
            Vmax = point["Vmax"]
            fdvar = point.get("4dvar", {})
            dcdyn = point.get("dynamic", {})
            dcstat = point.get("static", {})

            def _err(r):
                if r.get("status") == "failed":
                    return "FAIL"
                return f"{r.get('results', {}).get('error_reduction', '?'):.1f}"

            spread = dcdyn.get("dcwme_diagnostics", {}).get("l_wme_spread_pct", "?")
            spread_str = f"{spread:.1f}%" if isinstance(spread, (int, float)) else str(spread)

            print(
                f"{Vmax:>6} {_err(fdvar):>8} "
                f"{_err(dcdyn):>8} {_err(dcstat):>8} {spread_str:>10}"
            )
        print(f"{'=' * 70}")

    return all_results


def main():
    """Main entry point."""
    import os
    os.environ.setdefault("CC", "/usr/bin/clang")

    args = parse_args()

    if args.phase == "0":
        run_phase_0(args)
    elif args.phase == "1":
        run_phase_1(args)
    elif args.phase == "2":
        run_phase_2(args)
    elif args.phase == "3":
        run_phase_3(args)
    elif args.phase == "4":
        run_phase_4(args)
    elif args.phase == "5":
        run_phase_5(args)
    elif args.phase == "6":
        run_phase_6(args)
    elif args.phase == "7":
        run_phase_7(args)
    else:
        print(f"Phase {args.phase} not yet implemented.")
        return 1

    return 0


if __name__ == "__main__":
    # Hidden entry point for subprocess sweep workers
    if len(sys.argv) >= 3 and sys.argv[1] == "--_sweep-worker":
        config_json = sys.argv[2]
        config = json.loads(config_json)
        if config.get("worker_type") == "wind":
            _wind_sweep_worker_main(config_json)
        else:
            _sweep_worker_main(config_json)
    else:
        sys.exit(main())

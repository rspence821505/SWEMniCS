#!/usr/bin/env python3
"""
Manning's-n Identifiability Audit
=================================

Forensic diagnostic to determine whether multi-parameter Manning's-n
estimation is structurally identifiable from surface elevation observations
on the Shinnecock Inlet mesh.

Objectives:
  A. Build whitened parameter sensitivity matrix J_θ and compute SVD
  B. Per-mode ±ε response study (sign symmetry check)
  C. State-parameter coupling audit (cross-block probes)
  D. Classify failure modes per parameter direction

Usage:
  python experiments/identifiability_audit.py --kl-modes 3
  python experiments/identifiability_audit.py --kl-modes 3 --with-state-coupling
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditConfig:
    """Configuration for identifiability audit."""
    # Solver
    adios_file: str = "data/shinnecock_inlet"
    dt: float = 600.0
    nt_ramp: int = 6
    nt_da: int = 12
    truth_vmax: float = 0.0
    wind_ramp_s: float = 43200.0

    # Manning parameterization
    basis_type: str = "kl"
    basis_shape: tuple = (2, 1)
    basis_width_fraction: float = 0.35
    n_bounds: tuple = (0.01, 0.08)
    reference_n: float = 0.02
    kl_correlation_length: float = None
    kl_prior_std: float = 0.5
    kl_n_modes: int = 3

    # Observations
    obs_frequency: int = 2
    obs_fraction: float = 0.10
    obs_noise_level: float = 0.005
    obs_seed: int = 42
    obs_velocity: bool = False
    obs_max_depth: float = None

    # FD perturbation for sensitivity columns
    fd_epsilon: float = 0.01

    @property
    def nt_total(self) -> int:
        return self.nt_ramp + self.nt_da

    @property
    def n_params(self) -> int:
        if self.basis_type == "kl":
            return self.kl_n_modes
        return self.basis_shape[0] * self.basis_shape[1]

    @property
    def obs_times(self) -> list:
        return list(range(self.nt_ramp, self.nt_total + 1, self.obs_frequency))


# ---------------------------------------------------------------------------
# Reuse validation_ladder infrastructure
# ---------------------------------------------------------------------------

from experiments.validation_ladder import (
    build_problem_and_solver as _build_from_ladder_cfg,
    create_observations as _create_obs_from_ladder,
    generate_truth_trajectory as _gen_truth_from_ladder,
    LadderConfig,
)


def _to_ladder_config(cfg: AuditConfig) -> LadderConfig:
    """Convert audit config to ladder config for shared infrastructure."""
    kw = {}
    if cfg.basis_type == "kl":
        kw.update(
            basis_type="kl",
            kl_n_modes=cfg.kl_n_modes,
            kl_correlation_length=cfg.kl_correlation_length,
            kl_prior_std=cfg.kl_prior_std,
        )
    truth_coeffs = tuple([0.30] + [-0.25] * (cfg.n_params - 1))
    bg_coeffs = tuple([0.0] * cfg.n_params)
    theta_bounds = tuple([(-0.8, 0.8)] * cfg.n_params)

    return LadderConfig(
        adios_file=cfg.adios_file,
        dt=cfg.dt,
        nt_ramp=cfg.nt_ramp,
        nt_da=cfg.nt_da,
        truth_vmax=cfg.truth_vmax,
        wind_ramp_s=cfg.wind_ramp_s,
        basis_shape=cfg.basis_shape,
        basis_width_fraction=cfg.basis_width_fraction,
        n_bounds=cfg.n_bounds,
        reference_n=cfg.reference_n,
        obs_frequency=cfg.obs_frequency,
        obs_fraction=cfg.obs_fraction,
        obs_noise_level=cfg.obs_noise_level,
        obs_seed=cfg.obs_seed,
        obs_velocity=cfg.obs_velocity,
        obs_max_depth=cfg.obs_max_depth,
        truth_coefficients=truth_coeffs,
        background_coefficients=bg_coeffs,
        theta_bounds=theta_bounds,
        **kw,
    )


# ---------------------------------------------------------------------------
# Core: forward observation map G(θ) with state fixed at truth
# ---------------------------------------------------------------------------

def evaluate_observation_map(
    theta: np.ndarray,
    *,
    solver,
    problem,
    controller,
    solver_params,
    truth_state: np.ndarray,
    obs_operator,
    obs_times: list,
    cfg: AuditConfig,
) -> np.ndarray:
    """Run forward model with given θ (state fixed at truth) and return
    the full observation vector G(θ) ∈ R^{n_obs_times × n_obs_points}.

    Returns a flat vector of all observations concatenated across times.
    """
    from dolfinx import la

    state_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs

    # Apply θ to Manning field
    controller.apply(problem, solver, theta)

    # Reset state to truth
    solver.storage.clear()
    solver.u_n.x.array[:state_size] = truth_state
    solver.u_n_old.x.array[:state_size] = truth_state
    solver.u.x.array[:state_size] = truth_state
    solver.u_n.x.scatter_forward()
    solver.u_n_old.x.scatter_forward()
    solver.u.x.scatter_forward()
    problem.t = 0.0
    if hasattr(problem, "update_boundary"):
        problem.update_boundary()

    # Run forward model
    solver.time_loop(
        solver_parameters=solver_params,
        stations=[],
        plot_every=9999,
        save_state=True,
        store_jacobians=False,
        enable_video=False,
        monitor_progress=False,
    )

    # Extract observations at each obs time
    obs_list = []
    for t_idx in obs_times:
        vec = la.create_petsc_vector(
            solver.V.dofmap.index_map, solver.V.dofmap.index_map_bs,
        )
        vec.setArray(solver.storage.saved_states[t_idx][:state_size])
        vec.assemble()
        obs_vec = obs_operator.forward(vec)
        obs_list.append(obs_vec.getArray().copy())
        obs_vec.destroy()
        vec.destroy()

    gc.collect()
    return np.concatenate(obs_list)


# ---------------------------------------------------------------------------
# Objective A: Build J_θ and compute whitened SVD
# ---------------------------------------------------------------------------

def build_parameter_sensitivity_matrix(
    *,
    solver,
    problem,
    controller,
    solver_params,
    truth_state: np.ndarray,
    obs_operator,
    obs_times: list,
    cfg: AuditConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build J_θ[i, p] = ∂G_i/∂θ_p via centered finite differences.

    Returns (J_theta, G_ref) where G_ref = G(θ=0) is the reference observation.
    """
    n_params = cfg.n_params
    eps = cfg.fd_epsilon
    theta_ref = np.zeros(n_params)

    fwd_kwargs = dict(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params, truth_state=truth_state,
        obs_operator=obs_operator, obs_times=obs_times, cfg=cfg,
    )

    print(f"  Computing reference G(θ=0)...")
    G_ref = evaluate_observation_map(theta_ref, **fwd_kwargs)
    n_obs = G_ref.size
    print(f"  Observation vector dimension: {n_obs}")

    J_theta = np.zeros((n_obs, n_params))

    for p in range(n_params):
        print(f"  Computing J_θ column {p} (±ε={eps})...")
        theta_plus = theta_ref.copy()
        theta_minus = theta_ref.copy()
        theta_plus[p] += eps
        theta_minus[p] -= eps

        # Check clipping
        field_plus = controller.evaluate_field(theta_plus)
        field_minus = controller.evaluate_field(theta_minus)
        n_clipped_plus = np.sum(
            (field_plus <= cfg.n_bounds[0] + 1e-10) |
            (field_plus >= cfg.n_bounds[1] - 1e-10)
        )
        n_clipped_minus = np.sum(
            (field_minus <= cfg.n_bounds[0] + 1e-10) |
            (field_minus >= cfg.n_bounds[1] - 1e-10)
        )
        if n_clipped_plus > 0 or n_clipped_minus > 0:
            print(f"    WARNING: clipping active — {n_clipped_plus}/{n_clipped_minus} DOFs clipped at +/-ε")

        G_plus = evaluate_observation_map(theta_plus, **fwd_kwargs)
        G_minus = evaluate_observation_map(theta_minus, **fwd_kwargs)

        J_theta[:, p] = (G_plus - G_minus) / (2 * eps)

        # Report column norm
        col_norm = np.linalg.norm(J_theta[:, p])
        print(f"    ||J_θ[:, {p}]|| = {col_norm:.6e}")

    return J_theta, G_ref


def whiten_sensitivity(
    J_theta: np.ndarray,
    noise_stds: np.ndarray,
    n_obs_points: int,
) -> np.ndarray:
    """Apply R^{-1/2} to J_θ using the actual observation noise model.

    noise_stds has shape (n_obs_times,) — one std per observation time.
    Each observation time has n_obs_points observations.
    R^{-1/2} = diag(1/σ_k) applied blockwise.
    """
    n_obs_times = len(noise_stds)
    n_total = n_obs_times * n_obs_points
    assert J_theta.shape[0] == n_total, (
        f"J_theta has {J_theta.shape[0]} rows but expected {n_total}"
    )

    # Build full R^{-1/2} diagonal
    R_inv_sqrt = np.zeros(n_total)
    for k in range(n_obs_times):
        start = k * n_obs_points
        end = (k + 1) * n_obs_points
        R_inv_sqrt[start:end] = 1.0 / max(noise_stds[k], 1e-30)

    # Apply whitening
    return R_inv_sqrt[:, np.newaxis] * J_theta


def spectral_analysis(
    J_theta_whitened: np.ndarray,
    basis_matrix: np.ndarray,
    output_dir: Path,
):
    """SVD of whitened J_θ and interpretation."""
    U, sigma, Vt = np.linalg.svd(J_theta_whitened, full_matrices=False)

    print("\n" + "="*70)
    print("SPECTRAL ANALYSIS OF WHITENED J_θ")
    print("="*70)
    print(f"  Matrix shape: {J_theta_whitened.shape}")
    print(f"  Singular values: {sigma}")
    print(f"  Condition number: {sigma[0] / max(sigma[-1], 1e-30):.2e}")

    # Cumulative energy
    energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    print(f"  Cumulative energy: {energy}")

    # Effective rank at various thresholds
    for tol in [0.01, 0.05, 0.10]:
        rank = int(np.sum(sigma > tol * sigma[0]))
        print(f"  Effective rank (tol={tol}): {rank}")

    # Right singular vectors = parameter-space directions
    # V[:, i] is the i-th right singular vector in θ-space
    V = Vt.T  # (n_params, n_params)

    results = {
        "singular_values": sigma.tolist(),
        "condition_number": float(sigma[0] / max(sigma[-1], 1e-30)),
        "cumulative_energy": energy.tolist(),
        "right_singular_vectors": V.tolist(),
    }

    # Map each right singular vector to spatial friction perturbation
    # δn(x) ≈ n_ref * B @ v_i (linearized around θ=0 where exp(B@θ)≈1)
    print("\n  Right singular vectors (parameter-space directions):")
    for i in range(len(sigma)):
        v = V[:, i]
        spatial_perturbation = basis_matrix @ v
        print(f"    σ_{i} = {sigma[i]:.6e}  |  v_{i} = {v}  |  ||δn||_∞ = {np.max(np.abs(spatial_perturbation)):.6e}")

    # Save plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Singular value spectrum
        ax = axes[0]
        ax.bar(range(len(sigma)), sigma)
        ax.set_xlabel("Mode index")
        ax.set_ylabel("Singular value")
        ax.set_title("Singular values of whitened J_θ")
        ax.set_yscale("log")
        for i, s in enumerate(sigma):
            ax.text(i, s * 1.1, f"{s:.2e}", ha="center", fontsize=8)

        # Cumulative energy
        ax = axes[1]
        ax.bar(range(len(energy)), energy)
        ax.axhline(0.99, color="r", linestyle="--", label="99%")
        ax.axhline(0.95, color="orange", linestyle="--", label="95%")
        ax.set_xlabel("Mode index (cumulative)")
        ax.set_ylabel("Fraction of total information")
        ax.set_title("Cumulative energy")
        ax.legend()

        # Right singular vectors as bar chart
        ax = axes[2]
        x = np.arange(V.shape[0])
        width = 0.25
        for i in range(V.shape[1]):
            ax.bar(x + i * width, V[:, i], width, label=f"v_{i} (σ={sigma[i]:.2e})")
        ax.set_xlabel("Parameter index")
        ax.set_ylabel("Weight")
        ax.set_title("Right singular vectors (θ-space)")
        ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(output_dir / "spectral_analysis.png", dpi=150)
        plt.close()
        print(f"\n  Saved spectral plot to {output_dir / 'spectral_analysis.png'}")
    except ImportError:
        print("  (matplotlib not available — skipping plots)")

    return results


# ---------------------------------------------------------------------------
# Objective B: Per-mode ±ε response study
# ---------------------------------------------------------------------------

def mode_response_study(
    *,
    solver,
    problem,
    controller,
    solver_params,
    truth_state: np.ndarray,
    obs_operator,
    obs_times: list,
    cfg: AuditConfig,
    G_ref: np.ndarray,
    output_dir: Path,
) -> dict:
    """For each mode, compare G(+ε·e_i) vs G(-ε·e_i) responses."""
    n_params = cfg.n_params
    eps = cfg.fd_epsilon
    n_obs_points = G_ref.size // len(obs_times)

    fwd_kwargs = dict(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params, truth_state=truth_state,
        obs_operator=obs_operator, obs_times=obs_times, cfg=cfg,
    )

    print("\n" + "="*70)
    print("PER-MODE ±ε RESPONSE STUDY")
    print("="*70)

    responses_plus = []
    responses_minus = []

    for p in range(n_params):
        theta_plus = np.zeros(n_params)
        theta_minus = np.zeros(n_params)
        theta_plus[p] = +eps
        theta_minus[p] = -eps

        print(f"  Mode {p}: computing G(+ε·e_{p}) and G(-ε·e_{p})...")
        G_plus = evaluate_observation_map(theta_plus, **fwd_kwargs)
        G_minus = evaluate_observation_map(theta_minus, **fwd_kwargs)

        responses_plus.append(G_plus)
        responses_minus.append(G_minus)

    # Analysis
    results = {"modes": []}

    # Perturbation responses relative to reference
    delta_plus = [rp - G_ref for rp in responses_plus]
    delta_minus = [rm - G_ref for rm in responses_minus]

    print("\n  --- Sign symmetry analysis ---")
    for p in range(n_params):
        dp = delta_plus[p]
        dm = delta_minus[p]

        # Sign symmetry: if δ+ ≈ -δ-, the response is antisymmetric (linear, good)
        # If δ+ ≈ δ-, the response is symmetric (even function, sign-ambiguous)
        antisymmetric_component = (dp - dm) / 2  # odd part
        symmetric_component = (dp + dm) / 2      # even part

        antisym_norm = np.linalg.norm(antisymmetric_component)
        sym_norm = np.linalg.norm(symmetric_component)
        total_norm = np.linalg.norm(dp)

        # Cosine similarity between +ε and -ε responses
        if np.linalg.norm(dp) > 1e-30 and np.linalg.norm(dm) > 1e-30:
            cosine = np.dot(dp, dm) / (np.linalg.norm(dp) * np.linalg.norm(dm))
        else:
            cosine = 0.0

        # If cosine ≈ -1: perfectly antisymmetric (linear response, good)
        # If cosine ≈ +1: perfectly symmetric (even response, sign-ambiguous)
        # If cosine ≈  0: mixed
        sign_type = "LINEAR (good)" if cosine < -0.5 else ("SIGN-AMBIGUOUS" if cosine > 0.5 else "MIXED")

        print(f"  Mode {p}: ||δ+||={np.linalg.norm(dp):.4e}  ||δ-||={np.linalg.norm(dm):.4e}  "
              f"cos(δ+,δ-)={cosine:+.6f}  antisym/sym={antisym_norm:.4e}/{sym_norm:.4e}  → {sign_type}")

        results["modes"].append({
            "mode": p,
            "delta_plus_norm": float(np.linalg.norm(dp)),
            "delta_minus_norm": float(np.linalg.norm(dm)),
            "cosine_plus_minus": float(cosine),
            "antisymmetric_norm": float(antisym_norm),
            "symmetric_norm": float(sym_norm),
            "sign_type": sign_type,
        })

    # Pairwise collinearity between mode responses
    print("\n  --- Pairwise collinearity (using antisymmetric/linear part) ---")
    linear_responses = [(delta_plus[p] - delta_minus[p]) / 2 for p in range(n_params)]
    collinearity = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            ni = np.linalg.norm(linear_responses[i])
            nj = np.linalg.norm(linear_responses[j])
            if ni > 1e-30 and nj > 1e-30:
                collinearity[i, j] = np.dot(linear_responses[i], linear_responses[j]) / (ni * nj)
            else:
                collinearity[i, j] = 0.0
    print(f"  Collinearity matrix:")
    for i in range(n_params):
        row = "    " + "  ".join(f"{collinearity[i, j]:+.4f}" for j in range(n_params))
        print(row)

    results["collinearity_matrix"] = collinearity.tolist()

    # Save plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Response norms per mode
        ax = axes[0]
        modes = list(range(n_params))
        ax.bar([m - 0.15 for m in modes],
               [np.linalg.norm(delta_plus[p]) for p in modes],
               0.3, label="+ε", color="blue", alpha=0.7)
        ax.bar([m + 0.15 for m in modes],
               [np.linalg.norm(delta_minus[p]) for p in modes],
               0.3, label="-ε", color="red", alpha=0.7)
        ax.set_xlabel("Mode")
        ax.set_ylabel("||G(±ε·e_i) - G(0)||")
        ax.set_title("Response magnitude per mode")
        ax.legend()

        # Collinearity heatmap
        ax = axes[1]
        im = ax.imshow(collinearity, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xlabel("Mode j")
        ax.set_ylabel("Mode i")
        ax.set_title("Pairwise collinearity of mode responses")
        for i in range(n_params):
            for j in range(n_params):
                ax.text(j, i, f"{collinearity[i, j]:.2f}",
                        ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(output_dir / "mode_responses.png", dpi=150)
        plt.close()
        print(f"\n  Saved mode response plot to {output_dir / 'mode_responses.png'}")
    except ImportError:
        pass

    return results


# ---------------------------------------------------------------------------
# Objective C: State-parameter coupling (lightweight probe)
# ---------------------------------------------------------------------------

def state_parameter_coupling_probe(
    *,
    J_theta: np.ndarray,
    solver,
    problem,
    controller,
    solver_params,
    truth_state: np.ndarray,
    obs_operator,
    obs_times: list,
    cfg: AuditConfig,
    n_state_probes: int = 20,
    output_dir: Path,
) -> dict:
    """Probe the cross-coupling between state and parameter sensitivities.

    Uses random state perturbation directions to estimate ||J_{u0}||
    and the alignment between state and parameter observation responses.
    """
    from dolfinx import la

    print("\n" + "="*70)
    print("STATE-PARAMETER COUPLING PROBE")
    print("="*70)

    state_size = truth_state.size
    n_obs = J_theta.shape[0]
    eps_state = 1e-4

    # Generate random state perturbation directions
    rng = np.random.default_rng(42)

    # Compute a few J_{u0} columns via random projections
    # J_{u0} @ v ≈ (G(u0 + ε*v, θ=0) - G(u0 - ε*v, θ=0)) / (2ε)
    state_response_norms = []
    state_response_vectors = []

    fwd_kwargs = dict(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params, obs_operator=obs_operator,
        obs_times=obs_times, cfg=cfg,
    )

    for probe_idx in range(n_state_probes):
        # Random direction in state space (normalized)
        direction = rng.standard_normal(state_size)
        direction /= np.linalg.norm(direction)

        state_plus = truth_state + eps_state * direction
        state_minus = truth_state - eps_state * direction

        print(f"  State probe {probe_idx+1}/{n_state_probes}...")

        G_plus = evaluate_observation_map(
            np.zeros(cfg.n_params),
            truth_state=state_plus,
            **fwd_kwargs,
        )
        G_minus = evaluate_observation_map(
            np.zeros(cfg.n_params),
            truth_state=state_minus,
            **fwd_kwargs,
        )

        J_u0_v = (G_plus - G_minus) / (2 * eps_state)
        state_response_norms.append(np.linalg.norm(J_u0_v))
        state_response_vectors.append(J_u0_v)

    # Compare magnitudes
    param_col_norms = [np.linalg.norm(J_theta[:, p]) for p in range(cfg.n_params)]
    mean_state_response = np.mean(state_response_norms)

    print(f"\n  --- Block magnitude comparison ---")
    print(f"  ||J_θ|| column norms: {param_col_norms}")
    print(f"  ||J_{'{u0}'}|| random-direction norms: mean={mean_state_response:.6e}, "
          f"std={np.std(state_response_norms):.6e}")
    print(f"  Ratio ||J_θ||/||J_{'{u0}'}||: {np.mean(param_col_norms)/max(mean_state_response, 1e-30):.4f}")

    # Alignment: project state responses onto parameter response subspace
    # This tells us whether state perturbations can mimic parameter effects
    if len(state_response_vectors) > 0 and cfg.n_params > 0:
        # SVD of J_theta to get parameter response subspace
        U_param, _, _ = np.linalg.svd(J_theta, full_matrices=False)
        # U_param columns span the parameter response subspace in obs space

        alignments = []
        for sv in state_response_vectors:
            sv_norm = np.linalg.norm(sv)
            if sv_norm < 1e-30:
                alignments.append(0.0)
                continue
            # Project onto parameter subspace
            proj = U_param @ (U_param.T @ sv)
            proj_frac = np.linalg.norm(proj) / sv_norm
            alignments.append(proj_frac)

        mean_alignment = np.mean(alignments)
        print(f"\n  --- State-parameter alignment ---")
        print(f"  Fraction of state response in parameter subspace: "
              f"mean={mean_alignment:.4f}, std={np.std(alignments):.4f}")
        print(f"  (1.0 = state responses perfectly mimic parameter responses = high confounding)")
        print(f"  (0.0 = state responses orthogonal to parameter responses = no confounding)")
    else:
        mean_alignment = 0.0
        alignments = []

    results = {
        "param_column_norms": param_col_norms,
        "state_response_norm_mean": float(mean_state_response),
        "state_response_norm_std": float(np.std(state_response_norms)),
        "param_to_state_ratio": float(np.mean(param_col_norms) / max(mean_state_response, 1e-30)),
        "state_param_alignment_mean": float(mean_alignment),
        "state_param_alignment_std": float(np.std(alignments)) if alignments else 0.0,
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Manning's-n Identifiability Audit")
    parser.add_argument("--kl-modes", type=int, default=3, help="Number of KL modes")
    parser.add_argument("--kl-length", type=float, default=None, help="KL correlation length")
    parser.add_argument("--fd-epsilon", type=float, default=0.01, help="FD perturbation size")
    parser.add_argument("--with-state-coupling", action="store_true",
                        help="Run state-parameter coupling audit (expensive: adds ~40 forward solves)")
    parser.add_argument("--n-state-probes", type=int, default=10,
                        help="Number of random state probes for coupling audit")
    parser.add_argument("--obs-velocity", action="store_true",
                        help="Observe h + u + v (velocity) instead of h only")
    parser.add_argument("--obs-max-depth", type=float, default=None,
                        help="Only place obs where depth < this value (meters)")
    args = parser.parse_args()

    cfg = AuditConfig(
        kl_n_modes=args.kl_modes,
        kl_correlation_length=args.kl_length,
        fd_epsilon=args.fd_epsilon,
        obs_velocity=args.obs_velocity,
        obs_max_depth=args.obs_max_depth,
    )

    output_dir = PROJECT_ROOT / "results" / "identifiability_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("MANNING'S-n IDENTIFIABILITY AUDIT")
    print("="*70)
    print(f"  Basis: KL with {cfg.kl_n_modes} modes")
    print(f"  Window: {cfg.nt_da * cfg.dt / 3600:.1f}h DA, {cfg.nt_ramp * cfg.dt / 3600:.1f}h ramp")
    obs_type = "h + u + v (velocity)" if cfg.obs_velocity else "h only (elevation)"
    print(f"  Obs: {cfg.obs_fraction*100:.0f}% mesh, every {cfg.obs_frequency} steps, {obs_type}")
    print(f"  FD epsilon: {cfg.fd_epsilon}")
    print(f"  Output: {output_dir}")

    # Setup
    t0 = time.time()
    ladder_cfg = _to_ladder_config(cfg)
    problem, solver, controller, solver_params, forcing = _build_from_ladder_cfg(ladder_cfg)
    truth_state, theta_truth = _gen_truth_from_ladder(
        ladder_cfg, problem, solver, controller, solver_params,
    )
    obs_bundle = _create_obs_from_ladder(
        ladder_cfg, solver, truth_state, controller, solver_params, problem,
    )
    print(f"\n  Setup time: {time.time() - t0:.1f}s")
    n_obs_per_snapshot = obs_bundle["truth_obs"].shape[1]
    print(f"  Observation points: {obs_bundle['obs_points'].shape[0]}")
    print(f"  Obs per snapshot: {n_obs_per_snapshot} ({'h+u+v' if cfg.obs_velocity else 'h only'})")
    print(f"  Observation times: {obs_bundle['obs_times']} ({len(obs_bundle['obs_times'])} snapshots)")
    print(f"  Total obs dimension: {n_obs_per_snapshot * len(obs_bundle['obs_times'])}")

    # ===== OBJECTIVE A: Parameter sensitivity matrix =====
    print("\n" + "#"*70)
    print("# OBJECTIVE A: PARAMETER SENSITIVITY MATRIX")
    print("#"*70)

    t1 = time.time()
    J_theta, G_ref = build_parameter_sensitivity_matrix(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params, truth_state=truth_state,
        obs_operator=obs_bundle["obs_operator"],
        obs_times=obs_bundle["obs_times"], cfg=cfg,
    )
    print(f"  J_θ build time: {time.time() - t1:.1f}s")

    # Whiten — n_obs_points is obs per snapshot (3x points if velocity enabled)
    n_obs_per_snapshot = obs_bundle["truth_obs"].shape[1]
    J_theta_w = whiten_sensitivity(
        J_theta, obs_bundle["noise_stds"],
        n_obs_points=n_obs_per_snapshot,
    )

    # Spectral analysis
    spectral_results = spectral_analysis(
        J_theta_w, controller._basis_matrix, output_dir,
    )

    # ===== OBJECTIVE B: Per-mode ±ε response =====
    print("\n" + "#"*70)
    print("# OBJECTIVE B: PER-MODE ±ε RESPONSE STUDY")
    print("#"*70)

    t2 = time.time()
    mode_results = mode_response_study(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params, truth_state=truth_state,
        obs_operator=obs_bundle["obs_operator"],
        obs_times=obs_bundle["obs_times"], cfg=cfg,
        G_ref=G_ref, output_dir=output_dir,
    )
    print(f"  Mode response time: {time.time() - t2:.1f}s")

    # ===== OBJECTIVE C: State-parameter coupling (optional) =====
    coupling_results = None
    if args.with_state_coupling:
        print("\n" + "#"*70)
        print("# OBJECTIVE C: STATE-PARAMETER COUPLING PROBE")
        print("#"*70)

        t3 = time.time()
        coupling_results = state_parameter_coupling_probe(
            J_theta=J_theta,
            solver=solver, problem=problem, controller=controller,
            solver_params=solver_params, truth_state=truth_state,
            obs_operator=obs_bundle["obs_operator"],
            obs_times=obs_bundle["obs_times"], cfg=cfg,
            n_state_probes=args.n_state_probes,
            output_dir=output_dir,
        )
        print(f"  Coupling probe time: {time.time() - t3:.1f}s")

    # ===== OBJECTIVE D: Classification and verdict =====
    print("\n" + "#"*70)
    print("# OBJECTIVE D: FAILURE MODE CLASSIFICATION")
    print("#"*70)

    sigma = np.array(spectral_results["singular_values"])
    classify_failure_modes(sigma, mode_results, coupling_results, cfg)

    # Save all results
    all_results = {
        "config": {
            "kl_n_modes": cfg.kl_n_modes,
            "dt": cfg.dt,
            "nt_ramp": cfg.nt_ramp,
            "nt_da": cfg.nt_da,
            "obs_fraction": cfg.obs_fraction,
            "obs_frequency": cfg.obs_frequency,
            "fd_epsilon": cfg.fd_epsilon,
        },
        "spectral": spectral_results,
        "mode_responses": mode_results,
        "coupling": coupling_results,
        "total_time_s": time.time() - t0,
    }

    results_file = output_dir / "identifiability_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")
    print(f"  Total audit time: {time.time() - t0:.1f}s")


def classify_failure_modes(sigma, mode_results, coupling_results, cfg):
    """Objective D: classify each mode's failure type."""

    print(f"\n  Singular values: {sigma}")
    cond = sigma[0] / max(sigma[-1], 1e-30)
    print(f"  Condition number: {cond:.2e}")

    # Effective rank
    rank_5pct = int(np.sum(sigma > 0.05 * sigma[0]))
    rank_10pct = int(np.sum(sigma > 0.10 * sigma[0]))
    print(f"  Effective rank (5% threshold): {rank_5pct}")
    print(f"  Effective rank (10% threshold): {rank_10pct}")

    print(f"\n  --- Per-mode classification ---")
    for mode_info in mode_results["modes"]:
        p = mode_info["mode"]
        cos = mode_info["cosine_plus_minus"]

        # Determine observation strength from collinearity with leading mode
        collinearity = np.array(mode_results["collinearity_matrix"])

        # Classification logic
        if abs(cos) > 0.5 and cos > 0:
            sign_status = "SIGN-AMBIGUOUS (even response)"
        elif abs(cos) > 0.5 and cos < 0:
            sign_status = "LINEAR (odd response, no sign ambiguity)"
        else:
            sign_status = "MIXED (partially nonlinear)"

        # Check if mode is collinear with other modes
        max_collinear = 0.0
        collinear_with = -1
        for j in range(cfg.n_params):
            if j != p and abs(collinearity[p, j]) > max_collinear:
                max_collinear = abs(collinearity[p, j])
                collinear_with = j

        if max_collinear > 0.95:
            identity_status = f"COLLINEAR with mode {collinear_with} (cos={max_collinear:.3f}, indistinguishable)"
        elif max_collinear > 0.80:
            identity_status = f"WEAKLY SEPARABLE from mode {collinear_with} (cos={max_collinear:.3f})"
        else:
            identity_status = "SEPARABLE from other modes"

        print(f"  Mode {p}: {sign_status} | {identity_status}")

    # State-parameter coupling
    if coupling_results is not None:
        alignment = coupling_results["state_param_alignment_mean"]
        if alignment > 0.5:
            print(f"\n  STATE-PARAMETER CONFOUNDING: HIGH (alignment={alignment:.3f})")
            print(f"  State perturbations can substantially mimic parameter responses.")
        elif alignment > 0.2:
            print(f"\n  STATE-PARAMETER CONFOUNDING: MODERATE (alignment={alignment:.3f})")
        else:
            print(f"\n  STATE-PARAMETER CONFOUNDING: LOW (alignment={alignment:.3f})")

    # Final verdict
    print(f"\n  {'='*60}")
    if rank_5pct <= 1:
        print(f"  VERDICT: NO-GO — Only {rank_5pct} parameter direction(s) observable.")
        print(f"  Elevation-only multi-parameter Manning's-n estimation is")
        print(f"  structurally underdetermined on this mesh.")
    elif rank_5pct < cfg.n_params:
        print(f"  VERDICT: LIMITED GO — {rank_5pct}/{cfg.n_params} directions observable.")
        print(f"  Reduce to {rank_5pct}-parameter estimation or add velocity observations.")
    else:
        print(f"  VERDICT: GO — All {cfg.n_params} directions appear observable.")
    print(f"  {'='*60}")


if __name__ == "__main__":
    raise SystemExit(main())

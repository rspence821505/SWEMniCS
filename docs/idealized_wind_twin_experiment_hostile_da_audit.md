# Idealized Wind Twin Experiment: Hostile DA Audit

## 1. Executive Verdict: PARTIAL PASS — valid after critical fix applied

One critical bug was found and fixed. After the fix, the DA pipeline is structurally sound.

---

## 2. Truth / Background / Observation Separation Audit

### 2.1 Truth separation: PASS (after fix)

**BUG FOUND AND FIXED**: `m_true_arr` was set to `solver_truth.u_n.x.array` which, after running both the ramp AND the DA window, contained the **end-of-DA** truth state instead of the **ramp-end** (DA window initial condition) state.

**Impact**: The background perturbation was applied around the final DA state, not the initial condition. The optimizer would start near the end state instead of the beginning, making the problem trivially easy.

**Fix applied**: Save `ramp_end_state = solver_truth.u_n.x.array[:state_size].copy()` BEFORE the DA window time loop, then set `m_true_arr = ramp_end_state`. This matches the Shinnecock pattern (`m_true = truth_trajectory[0].copy()` at line 3989 of `run_comparison.py`).

**After fix**: Truth trajectory is generated from the truth solver only. `m_true` is the ramp-end state. DA solver starts from the same ramp-end state. No truth leakage into the DA path.

### 2.2 Observation generation: PASS

- `_generate_observations()` (line 929 of `twin_experiment.py`) extracts from `self.truth_trajectory[k]` only
- `truth_trajectory` is set to the truth solver's saved states
- Noise is added via `rng.normal(0, noise_std, ...)` with seed=42
- No forecast/background arrays are used in observation generation
- Observation vectors are PETSc.Vec on COMM_SELF — no MPI leakage

### 2.3 Background separation: PASS (after fix)

- `_setup_background()` perturbs `self.m_true` (ramp-end state, after fix)
- Perturbation uses separate seed (background_seed=123)
- Component-aware h/u/v perturbation magnitudes
- Spatial smoothing with L=500m correlation
- Physical clipping (h ≥ h_min=0.01)
- Background is distinct from truth when perturbation is nonzero

---

## 3. Temporal Integrity Audit: PASS

- Ramp runs from t=0 to t=nt_ramp*dt (e.g., 24×600=14400s)
- `t_da_start = prob_truth.t` captures the ramp-end time
- DA window runs from t_da_start for nt_da steps
- `prob_da.t = t_da_start` correctly sets the DA problem's start time
- Truth trajectory states `[0, 1, ..., nt_da]` correspond to DA-relative times
- Observation times `range(0, nt_da+1, obs_frequency)` are DA-relative
- Forcing is evaluated at `self.t` which advances each timestep — correct physical time
- No future information leaks into observation construction

---

## 4. Observation Operator Audit: PASS

- Same `PointObservationOperator` instance (`obs_operator`) used for:
  - observation generation (`obs_operator.forward(truth_trajectory[k])`)
  - cost function evaluation (`FourDVarCost` uses `self.obs_op.forward(u_k)`)
- Observation points generated once via `_generate_interior_observation_points()`
- Interior-only selection with seed=42, reproducible
- DG-aware averaging for discontinuous spaces

---

## 5. Cost Function Audit: PASS

### 4D-Var
- `create_cost_function("4dvar", ...)` → `FourDVarCost`
- Background term: `½(m - m_b)^T B^{-1}(m - m_b)` — uses `m_background` (perturbed ramp-end)
- Observation term: `½ Σ_k ||H_k(u_k) - y_k||²_{R^{-1}}` — uses truth-derived observations
- Gradient: adjoint-based via `ImplicitAdjointSolver`
- Forward model: `ForwardModelWrapper` with DA problem (perturbed wind) — correct

### DC-WME static
- `create_cost_function("dc_wme", ..., predicted_cov_wme=static_lwme)` → `DCWMEFourDVarCost`
- Static L_wme built from `_compute_static_L_wme(obs_operator, B, ...)` with `skip_eq38_inflation=True`
- L_wme uses observation operator H and background covariance B — consistent
- Predictability correction: `½ δQ^T L_wme^{-1} δQ` subtracted from cost

---

## 6. Covariance / Noise Audit: PASS

### Observation noise
- Noise level: `obs_noise_level * signal_magnitude` per observation time
- RNG: `default_rng(obs_seed=42)` — reproducible
- Applied to truth observations only
- Zero noise when `obs_noise_level=0`

### Observation covariance R
- `DiagonalCovariance(comm, size=n_obs, variance=obs_variance)` where `obs_variance = mean(noise_stds)²`
- Used in both cost function weighting and (implicitly) in noise generation
- Consistent: noise generated with std `σ_k`, R uses `mean(σ_k)²` — slight inconsistency because time-varying noise stds are averaged. This is a known simplification, not a bug.

### Background covariance B
- Component-aware: h and u/v get different variances
- `DiagonalCovariance` with per-DOF variances
- Matches the perturbation model (both use `background_error_std × component_magnitude`)

---

## 7. Identifiability / Triviality Audit: ACCEPTABLE

- Model error comes from wind track shift (10km) — creates spatially structured mismatch
- Background error comes from state perturbation (background_error_std=0.02)
- Combined: optimizer must correct both IC perturbation AND wind-induced trajectory divergence
- With 208K DOFs and ~1,000 observations × 3-4 time points, the problem is underdetermined (more unknowns than observations) — requires regularization from B
- Not trivially solvable — genuine optimization needed

---

## 8. Numerical Integrity Audit: PASS

- All arrays copied (not referenced) when transferring between solvers
- PETSc vectors created fresh for truth trajectory (not views into solver storage)
- `scatter_forward()` called after setting DA solver state
- Memory guards prevent OOM
- Gradient smoother attached to cost function (critical for convergence)

---

## 9. Reproducibility Audit: PASS

- Wind generation: deterministic from `CartesianVortexConfig` parameters
- Observations: seeded with `obs_seed=42`
- Background: seeded with `background_seed=123`
- All parameters exposed via CLI args — no hidden constants
- Wind HDF5 files cached (skip if exists) — reproducible across runs

---

## 10. Exact Failures Found

| # | Severity | Description | Status |
|---|---|---|---|
| 1 | **CRITICAL** | `m_true` set to end-of-DA state instead of ramp-end state | **FIXED** |

---

## 11. Minimal Fixes Required

1. ~~Save `ramp_end_state` before DA window and use as `m_true_arr`~~ **DONE**

---

## 12. Confidence Level

**HIGH after fix.** The critical bug (wrong `m_true`) was the only structural issue found. All other components (observation generation, noise model, covariance construction, cost function wiring, temporal semantics) are correct.

The Shinnecock pipeline (`run_comparison.py`) handles this correctly at line 3989: `m_true = truth_trajectory[0].copy()`. The idealized inlet DA script now matches this pattern.

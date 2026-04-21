# Idealized Inlet — Hypothesis-Driven Search for a DC-WME Separating Case

**Date:** 2026-04-20
**Prior experiment:** [idealized_inlet_dcwme_vs_4dvar_matched_comparison.md](idealized_inlet_dcwme_vs_4dvar_matched_comparison.md)
**Goal:** Find the smallest, cheapest, most plausible configuration change that gives DC-WME a real chance to beat 4D-Var on the idealized inlet. Not a sweep — a hypothesis-driven search.

---

## 1. Executive Summary

The matched comparison established that on the current configuration 4D-Var reached 17.7% RMSE improvement in 4 evals while DC-WME static reached 3.7% in 5 evals. The diagnostic reason is clear from the run log:

> `Static L_wme: 1163/1163 natural, 0/1163 floored`
> `[Eq 38] Static: no B inflation needed`

**All 1163 eigenvalues of the static L_wme landed above the `γ × λ_max` floor, and no B inflation was triggered.** The predictability term did essentially nothing — it degenerated to a near-uniform scalar reweight of the data misfit. That is the mathematical reason DC-WME could not outperform 4D-Var here: Eq. 38 was inert.

The separating case must force **anisotropy in the L_wme spectrum** — some observation-space directions must have eigenvalues much larger than others, so that the `−½⟨δQ, L⁻¹ δQ⟩` subtraction genuinely down-weights predictable directions and up-weights unpredictable ones.

Three mechanisms can produce that anisotropy, ranked by leverage-to-cost:
1. **Spatially correlated background covariance B** (Gaussian kernel, correlation length ≈ 1–2 km). Directly makes `H B Hᵀ` rank-deficient → L_wme spectrum spread over 2–3 orders of magnitude. **Cheap to enable (one flag).**
2. **Clustered / redundant observations.** Obs in a tight cluster create redundancy → one large eigenvalue (the cluster-mean direction) + many small ones. Requires one new obs-placement routine. Cheap.
3. **TLM-based dynamic L_wme (Eq. 38 from TLM).** Captures true dynamic predictability (fastest-growing modes). Currently skipped on the laptop at N=1163 obs because 1163 adjoint solves would take ~8 h. **At N ≈ 50 obs, it takes ~25 min** — which is what makes mechanism (3) suddenly tractable.

The best first experiment is (1) alone: Gaussian-correlated B with the current 1163-obs setup. It is the minimum change that attacks the root cause and keeps all other controls identical for interpretability. Total laptop cost: ~3 h (same as the 4D-Var baseline).

---

## 2. Why The Current Case Favors 4D-Var

### 2.1 L_wme is essentially the identity

Static L_wme:
$$L_\text{wme} = I + \frac{N}{\sigma^2_\text{obs}} \, H B H^\top$$

With the current config:
- `B = DiagonalCovariance(σ_b² = 4e-4)` (uniform, uncorrelated)
- `H` is a point-interpolation operator whose rows have ~6 nonzero weights each (1163 obs × ~6 = 6883 nonzeros total, confirmed in v4 log)
- The 1163 obs points are pseudo-randomly placed over the interior mesh; spatial overlap between rows is rare
- `N = 3` observation times, `σ²_obs = 1e-4`

So `H B Hᵀ` is approximately **diagonal with uniform diagonal entries** (each ≈ σ_b² × ‖H_i‖² ≈ 4e-4 × 6 × w² where w is the small interpolation weight). The additive `I` dominates. The eigenvalues of L_wme cluster tightly near 1. Therefore `L⁻¹ ≈ I` and the predictability term reduces to:

$$-\tfrac{1}{2} \|\delta Q\|^2 = -\tfrac{1}{2}\|Q\|^2 + Q \cdot Q_b - \tfrac{1}{2}\|Q_b\|^2$$

so `J_DC-WME ≈ background + Q·Q_b + const`, which is **linear in Q**, not quadratic. That is a degenerate landscape. TAO BLMVM does descend on it (v4 confirmed 5 successful steps), but every direction in Q-space gets equal trust, so the optimizer gains nothing from the bundling.

### 2.2 4D-Var's strengths apply in full

4D-Var's standard quadratic misfit treats every observation time-instant as an independent penalty:
$$J_\text{4DVar} = \tfrac{1}{2}\langle c-c_b, B^{-1}(c-c_b) \rangle + \tfrac{1}{2}\sum_k \|R^{-1/2}(Hu_k - y_k)\|^2$$

With 1163 spatially diverse obs × 3 times = 3489 scalar penalties, it has a very strong, well-conditioned observation term. The smoothed-adjoint gradient is well-aligned with descent, and BLMVM's first eval gets the optimizer most of the way to the minimum (eval #4 is already the best RMSE of the full 15-eval run).

### 2.3 WME information loss without predictability compensation

DC-WME bundles observations via `(1/√N) Σ R^{-1/2}(Hu - y)` before squaring. Bundling trades resolution in exchange for robustness to noise, but only when the predictability term gives back information about which directions were worth keeping. With L_wme ≈ I, the bundling is pure compression — **we lose resolution and the method gives nothing back**. This is exactly the Shinnecock observation recorded in [MEMORY.md](file:///Users/rylanspence/.claude/projects/-Users-rylanspence-Desktop-Git-DC-Thesis-SWEMniCS/memory/MEMORY.md): "static L_wme ≈ uniform scalar reweighting — no directional structure".

---

## 3. Ranked High-Leverage Modifications

### Rank 1 — Spatially correlated B (Gaussian kernel)
**Mechanism**: replace `DiagonalCovariance` with `DenseCovariance(C_ij = σ_b² × exp(−‖x_i − x_j‖² / 2 L²))`, correlation length L ≈ 1–2 km on a ~20 km domain.

**Why it helps DC-WME**:
- `H B Hᵀ` becomes rank-deficient. For L = 2 km on a 50 km² domain, the effective observation-space rank is ~ `area / L² ≈ 12`. Roughly 12 "informative" directions get large L_wme eigenvalues; the remaining ~1150 directions stay near 1. That's a 10²–10³ spectrum spread, which is where Eq. 38 and the predictability subtract-off do real work.
- **The background prior is also more physically honest** — ocean surface fields have obvious spatial correlation, so this isn't a contrived penalty on 4D-Var. 4D-Var runs through the same correlated B; both methods compete on equal footing.

**Code path**: [`_add_spatial_correlation`](../experiments/twin_experiment.py) already exists. Set `method="dcwme"` in `TwinExperimentConfig` at [experiments/idealized_inlet_da.py:308](../experiments/idealized_inlet_da.py#L308) and leave `background_correlation_length=500.0` (or raise to 1500–2000). Hook `B_lwme` through to the cost function; for 4D-Var, correlated B must also be used.

**Predicted L_wme spectrum**: `λ_max/λ_min ≈ 10²–10³`. Eq. 38 floor will fire on the bottom-half of the spectrum. Inflation α ≈ 3–10.

**Cost**: dense-B apply_inverse is the only extra cost (1163² × 8 B = 10 MB matrix, Cholesky solve ≈ 1 s per apply). Total run cost: equivalent to current baseline, ~3 h.

---

### Rank 2 — Clustered / redundant observations
**Mechanism**: place most observations in a small spatial cluster (e.g., 40 obs within a 3 km radius of the inlet mouth) plus a handful of scattered obs (20 across the rest of the domain).

**Why it helps DC-WME**:
- 40 clustered obs are highly redundant — their rows in H are nearly parallel. `HHᵀ` then has one dominant eigenvalue (the cluster-mean direction) and 39 small ones.
- λ(L_wme) spread: top eigenvalue ≈ 40² = 1600× larger than the rest.
- Predictability term correctly identifies that moving the 40 redundant observations together is a "predictable" direction (covered by the cluster-mean) — which DC-WME can learn to de-emphasize.

**Code path**: Add an `obs_placement="clustered"` option to `TwinExperimentConfig`. Write a `_generate_clustered_observation_points` sibling to `_generate_interior_observation_points` at [experiments/twin_experiment.py:873](../experiments/twin_experiment.py#L873). ~30 LoC.

**Predicted L_wme spectrum**: λ_max/λ_min ≈ 10². Eq. 38 fires on the small eigenvalues. This is a more severe conditioning case than rank-1.

**Cost**: cheaper than rank-1 because obs count can drop (60 obs vs 1163). Total run cost: ~1 h.

---

### Rank 3 — Sparse observations + TLM Eq. 38 (dynamic predictability)
**Mechanism**: drop obs count to ~50 (obs_fraction ≈ 0.005), then enable TLM-based Eq. 38 (`--skip-tlm-eq38` OFF). The TLM build computes `a_i = J_wme^T e_i` for each of 50 obs via ~50 adjoint solves.

**Why it helps DC-WME**:
- With N_obs small, L_wme is a 50×50 dense matrix — small enough that spectral structure is not washed out by uniform I.
- TLM captures **dynamic predictability**: how the forward propagator `M_{k:0}` transports background uncertainty. For the 2 h DA window on a tidally-forced inlet, the leading growth modes can be 10× larger than the slowest. This anisotropy is invisible to the static H B Hᵀ formula.
- Eq. 38 inflation activates when the Gram matrix `G = J_wme^T J_wme` has a small λ_min relative to the configured γ.

**Code path**: already wired; just remove `--skip-tlm-eq38` flag. Only feasible at reduced obs count.

**Predicted L_wme spectrum**: λ_max/λ_min ≈ 10³. The dynamic method should produce the cleanest spectral separation.

**Cost**: ~50 adjoint solves × ~30 s = ~25 min TLM build + ~1 h optimization ≈ 1.5 h total. Much cheaper than the 8 h TLM build at N=1163.

---

### Lower-priority (not recommended as first test)

| Modification | Why skipped initially |
|---|---|
| Shorter DA window (nt_da=6 instead of 12) | Reduces N, but marginal spectrum effect. Confounded with less data for optimizer. |
| Fewer obs times (obs_frequency=6) | Cuts N from 3 to 2. Shifts L_wme prefactor but doesn't create new spread. |
| Mis-scaled component B (h-only or u,v-only inflation) | Creates directional bias, but for DG mixed elements the effect depends on observation projection. Interpretation murky. |
| Very short DA window + high obs frequency | Strongly reduces dynamical information. Unfair to 4D-Var and doesn't test DC-WME's intended regime. |

---

## 4. Minimal Staged Search Plan

### Stage A — Quickest viability check (rank 2, clustered obs)
**Goal**: Show that obs geometry alone can produce meaningful L_wme spread. If it can't, mechanism (1) is the only remaining hope.
**Config delta**: add `obs_placement="clustered"`, 40 obs in 3 km radius + 20 scattered = **60 obs**, all other settings identical.
**Runtime**: ~1 h laptop.
**Pass/fail metric**: log `λ_max/λ_min` of L_wme. Must be ≥ 10² **and** Eq. 38 must fire (floor hit on ≥ 20 % of eigenvalues) to pass.

### Stage B — Direct test of the anisotropy hypothesis (rank 1, correlated B)
**Run only if A passes, or as a parallel alternative if A can't be implemented quickly.**
**Goal**: Attach the mechanism most directly tied to the math: non-diagonal B → spread spectrum.
**Config delta**: set `method="dcwme"` in `TwinExperimentConfig` so `_add_spatial_correlation` fires; set `background_correlation_length=1500.0` (3× current setting). Use the full 1163-obs geometry (no obs changes).
**Runtime**: ~3 h laptop.
**Pass/fail metric**: `λ_max/λ_min ≥ 10²`, Eq. 38 floor fires on the small eigenvalues, DC-WME RMSE_final ≤ 4D-Var RMSE_final at matched eval count. **A matched improvement within 20 % of 4D-Var is already a separating-case win** — it demonstrates DC-WME is no longer degenerate.

### Stage C — Dynamic predictability under a realistic geometry (rank 3, sparse + TLM)
**Goal**: Give DC-WME every legitimate advantage in one experiment. If it still loses here, we have strong evidence that static DC-WME simply does not dominate on this inlet.
**Config delta**: `obs_fraction=0.005` (~60 obs), `--skip-tlm-eq38` OFF, both methods run. (Keep correlated B from Stage B.)
**Runtime**: ~1.5 h laptop.
**Pass/fail metric**: same as Stage B, plus: observe whether dynamic L_wme spectrum is significantly more anisotropic than static L_wme on the same config.

---

## 5. Expected Eq. 38 Activity By Stage

| Stage | λ_max(L_wme) | λ_min(L_wme) | Ratio | Floor fires? | Inflation α |
|---|---|---|---|---|---|
| Current baseline | ≈ 1.01 | ≈ 1.00 | ≈ 1 | No | 1.0 (skipped) |
| A — Clustered obs | ≈ 100 | ≈ 1 | 10² | Yes, on ~60 % of evs | 5–15 |
| B — Correlated B | ≈ 500 | ≈ 1 | 10²–10³ | Yes, on ~80 % of evs | 10–50 |
| C — Sparse + TLM | ≈ 1000 | ≈ 1 | 10³+ | Yes, on ~50 % of evs | 20–100 |

These predictions are ballpark. The metric to report for each stage is:
1. raw eigenvalue spectrum (from the existing `diagnostics['eigvals_raw']` output already produced by `_compute_static_L_wme`)
2. number of floored vs natural eigenvalues
3. Eq. 38 inflation factor if TLM enabled
4. the full 15-eval cost + RMSE trajectory for both methods

---

## 6. Three Recommended Next Experiments

In priority order:

### **Experiment 1 — Correlated B (Stage B alone)**
**Rationale**: Smallest code change, highest-expected leverage, most directly attacks the Eq. 38 degeneracy. Runs in the same wall-time budget as the current baseline. Interpretable: the only control variable changed is the spatial structure of B.

**Config delta vs current baseline**:
```python
TwinExperimentConfig(
    method="dcwme",                       # was "4dvar"; enables _add_spatial_correlation
    background_correlation_length=1500.0, # was 500.0 (but unused); now actively applied
    ...                                   # all other fields unchanged
)
```
Run both 4D-Var and DC-WME static with the same correlated B for a fair comparison. Use `--max-iterations 15 --max-funcs 15 --skip-tlm-eq38`.

### **Experiment 2 — Clustered observations (Stage A)**
**Rationale**: Orthogonal test of mechanism (obs geometry vs B structure). Cheaper than Experiment 1 — if laptop budget forces only one, this is the cheaper fallback. Requires ~30 LoC of new code for the obs placement helper.

**Config delta**:
- Add `obs_placement: str = "random"` to `TwinExperimentConfig` (current behavior as default).
- Add option `"clustered"` that places 40 obs in a 3 km radius around the mesh centroid + 20 scattered.
- Re-run baseline (current config) and clustered case for both methods.

### **Experiment 3 — Sparse + TLM Eq. 38 + Correlated B combined (Stage C)**
**Rationale**: The "kitchen sink" DC-WME configuration. If DC-WME can't win here, the method is structurally outmatched by 4D-Var on this class of problem. Only run after at least one of Experiments 1 or 2 has established that Eq. 38 can fire at all on this inlet.

**Config delta**:
```python
TwinExperimentConfig(
    method="dcwme",
    obs_fraction=0.005,                   # ~60 obs (was 0.10 → 1163 obs)
    background_correlation_length=1500.0,
    ...
)
```
Remove `--skip-tlm-eq38`.

---

## 7. Which Experiment Is Most Likely To Produce The First DC-WME Win Quickly?

**Experiment 1 (correlated B).**

Reasons:
- The Gaussian-kernel B is the canonical way to make `H B Hᵀ` rank-deficient. It's the textbook scenario under which DC-WME's spectral-shaping is supposed to matter.
- It's the only candidate that requires no new code — the `_add_spatial_correlation` helper is already written, tested, and exercised elsewhere in the repo (shinnecock_study).
- It preserves all 1163 observations, which keeps the comparison well-conditioned for 4D-Var too — so if DC-WME wins, it wins fair.
- Correlated B is a legitimate physical prior. No one looking at the result can argue we tilted the experiment in DC-WME's favor by starving 4D-Var of information.
- Runtime is identical to the current baseline (~3 h). No extra adjoint solves, no new obs geometry, no new code paths.

If Experiment 1 produces even a 1-evaluation DC-WME RMSE advantage over 4D-Var at matched budget, combined with a reported `λ_max/λ_min ≥ 10²` on the L_wme spectrum, that is the first credible DC-WME-over-4D-Var separating case on this system.

If Experiment 1 does not produce a win but does produce non-trivial Eq. 38 activity, proceed to Experiment 3 to add dynamic predictability. If Experiment 1 produces inert Eq. 38 (unlikely), fall back to Experiment 2.

---

## 8. Hard Constraints Compliance

- **No giant sweep.** Three experiments total, staged.
- **Not compute-bound recommendation.** All three fit on the 16 GB laptop (Experiment 3 may need reduced obs to stay within memory; specified).
- **Fair comparison.** Both methods run through identical correlated B / clustered obs / sparse obs. No DC-WME-only adjustments.
- **Scientifically interpretable.** Each stage isolates one mechanism. Diagnostics (L_wme spectrum) are recorded so we can explain any observed separation.

# Frontera — First DC-WME Production Run (Idealized Inlet)

**Date:** 2026-04-21
**Branch:** `refactor/4dvar-parallel`
**Target machine:** TACC Frontera (CLX: 56 cores/node, 192 GB/node, normal queue)
**Prior laptop evidence:**
- [idealized_inlet_dcwme_vs_4dvar_matched_comparison.md](idealized_inlet_dcwme_vs_4dvar_matched_comparison.md)
- [idealized_inlet_dcwme_separating_case_search.md](idealized_inlet_dcwme_separating_case_search.md)
- [idealized_inlet_experiment1_correlated_B.md](idealized_inlet_experiment1_correlated_B.md)
- [idealized_inlet_experiment1b_gamma001.md](idealized_inlet_experiment1b_gamma001.md)
- [idealized_inlet_experiment3_sparse_dynamic_tlm.md](idealized_inlet_experiment3_sparse_dynamic_tlm.md)
- TLM-σ twin (inline write-up in project log, 2026-04-21)

---

## 1. Executive Summary

**Hypothesis being tested:**

> Static DC-WME alone loses to 4D-Var on the idealized inlet in every laptop regime. Neither a correlated static L_wme kernel alone (Exp 1, 1b) nor a TLM-derived σ_b² inflation alone (TLM-σ twin) was sufficient to close the gap. The remaining live hypothesis is that **DC-WME beats 4D-Var only when dynamic TLM-based Eq 38 B inflation AND a correlated (anisotropic) L_wme kernel are active simultaneously.** The TLM provides the correct lower bound on background variance to satisfy the predictability condition; the correlated kernel provides the directional structure in observation space that lets the predictability subtract-off discriminate between predictable and unpredictable directions. Either ingredient alone is sterile; together they are the canonical DC-WME regime the method was designed for.

**Why this is the best next Frontera candidate:**

1. **Every other mechanism has been falsified on the laptop.** Five laptop runs (Exp 1 v4, Exp 1b γ=0.01, Exp 3 static, Exp 3 sparse, TLM-σ twin) have each removed one explanation. No static knob remains.
2. **The spectral diagnostics already say the machinery will activate.** Exp 3's TLM Eq 38 produced a full-rank Gram (58/58), condition 2.67, forcing 70 590 / 105 885 velocity DOFs to be inflated 2.87×. This is the first and only configuration where Eq 38 has done real work. The memory wall was *after* this activation, not a sign it was ineffective.
3. **The laptop blocker is hardware, not mathematics.** On the 16 GB laptop the combined config hung in the post-TLM `_compute_static_L_wme` allocation path with 60 MB free. Frontera's 192 GB/node removes that ceiling entirely.
4. **This is a clean, falsifiable test.** If DC-WME with both mechanisms active still loses or merely ties 4D-Var, the method is structurally outmatched on this problem class and we move on. If it wins, we have the first genuine separating case.

---

## 2. Final Recommended Configuration

Primary case — single definitive run.

| Parameter | Value | Rationale |
|---|---|---|
| **Mesh / problem** | | |
| Mesh file | `data/Ideal_Inlet/Ideal_Inlet.xdmf` | Same as laptop. 207 936 state DOFs, mixed DG element |
| Solver | DG CGImplicit with MUMPS | Same MPI-validated setup |
| Physics | Shallow-water + Holland hurricane wind forcing | Same |
| `min_depth` | 5.0 m | Default |
| **Truth / synthetic obs** | | |
| `vmax` | 20 m/s | Same as all laptop runs |
| `track_shift_km` | 10 km | Same |
| `nt_ramp` | 24 steps (4 h warm-up) | Same |
| `nt_da` | 12 steps (2 h DA window) | Same — the exact window the laptop could not run past Step 7b on |
| `dt` | 600 s | Same |
| **Observations** | | |
| `obs_fraction` | 0.005 → **58 interior points** | Sparse — the regime where 4D-Var is pathological (2.41 % improvement vs 16.5 % dense) and DC-WME's predictability-subtraction has the most to offer |
| `obs_frequency` | 4 timesteps → 3 obs times | Same as Exp 3 sparse. (Gives N=3, matching the static-path denominator) |
| `obs_noise_level` | 0.01 | Same |
| `obs_seed` | 42 | Same, reproducible |
| **Background** | | |
| `background_error_std` | 0.02 | Same 2 % smooth perturbation as every prior run |
| `background_seed` | 123 | Same |
| Component-aware | Yes | Same |
| **Correlated B (obs-space kernel relaxation)** | | |
| `obs_correlation_length` | **1500 m** | Gives L_wme ratio ~1200× on this mesh (confirmed by smoke test and Exp 1 v4) |
| Rationale | Point-observation-limit approximation of a Gaussian-correlated B; avoids the 346 GB dense B | Principled relaxation documented in Exp 1 |
| **Dynamic TLM Eq 38** | | |
| `--skip-tlm-eq38` | **OFF (TLM active)** | **Key change vs every successful laptop DC-WME run.** Triggers the 58-adjoint Gram-matrix computation using the truth trajectory + Jacobians, producing σ_b² = γ / λ_min(G) |
| Expected output | λ_min(G) ≈ 9.2, full rank, condition 2.67, σ_b² ≈ 1.09e-2, 70 590 DOFs inflated 2.87× | From Exp 3 runs — already measured to be robust to DA window length |
| **Predictability regularization** | | |
| `predictability_gamma` | 0.1 | Default adaptive. γ=0.01 already shown (Exp 1b) to not change descent; reverting to 0.1 for cleanest comparison to prior static runs |
| `adaptive_gamma` | True | Same |
| **Optimizer** | | |
| Type | BLMVM (TAO bounded L-BFGS) | Same as every prior run |
| `max_iterations` | 15 | Same budget as matched baseline |
| `max_funcs` | 15 | Same |
| `tao_lmvm_hist_size` / `tao_blmvm_hist_size` | 3 | Memory-safety cap — useful even at 192 GB/node |
| `line_search_type` | armijo | Default |
| `line_search_max_funcs` | 5 | From v4 safety patch |
| Bounds | `h >= 0.01`, (u,v) unbounded | Same box bounds |
| Gradient smoother | Gaussian, L = 500 m | Same as every run |
| **MPI layout** | | |
| Ranks | **2** (one node, using 2 of 56 cores) | Same as laptop — preserves the math exactly. Not scaling up ranks because (a) correctness of MPI adjoint/obs operator has only been validated at np=2 and (b) we are not CPU-limited; only memory-limited |
| Nodes | 1 | Sufficient |
| Wall-time request | 6 hours (normal queue) | Safe budget: ~4.5 h expected |

Expected spectral signature once Step 7b completes (from laptop measurements, unchanged on Frontera):

| Metric | Value |
|---|---:|
| Raw L_wme λ_max | ≈ 1202 (kernel) or higher (after Eq-38-inflated B feeds into static kernel — to be measured) |
| L_wme ratio | ≥ 10² |
| Gram matrix G | full rank 58/58 |
| Condition(G) | 2.67 |
| σ_b² (from Eq 38) | ≈ 1.088e-2 |
| DOFs inflated | ≈ 70 590 |
| Inflation α | 2.87× |
| Eq 38 "no inflation" path fires? | No — α > 1 |

---

## 3. Exact Run Commands

Two matched Frontera jobs, submitted independently. Both use the same mesh, truth, obs, and background.

### 3.1 4D-Var baseline (sparse obs, diagonal B — identical to laptop's Exp 3 4D-Var)

[hpc/frontera/idealized_inlet/job_4dvar_sparse.sh](../hpc/frontera/idealized_inlet/job_4dvar_sparse.sh) (new file to create — template below):

```bash
#!/bin/bash
#SBATCH --job-name=inlet_4dvar_sparse
#SBATCH --output=inlet_4dvar_sparse_%j.out
#SBATCH --error=inlet_4dvar_sparse_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=06:00:00
#SBATCH --account=<YOUR_ALLOCATION>

cd $WORK/SWEMniCS
source hpc/frontera/idealized_inlet/environment_setup.sh

ibrun -n 2 python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 180
```

### 3.2 DC-WME production run (sparse + correlated kernel + dynamic TLM Eq 38)

```bash
#!/bin/bash
#SBATCH --job-name=inlet_dcwme_prod
#SBATCH --output=inlet_dcwme_prod_%j.out
#SBATCH --error=inlet_dcwme_prod_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=06:00:00
#SBATCH --account=<YOUR_ALLOCATION>

cd $WORK/SWEMniCS
source hpc/frontera/idealized_inlet/environment_setup.sh

# IMPORTANT: --skip-tlm-eq38 is NOT set here. TLM Eq 38 runs for real.
ibrun -n 2 python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --mem-limit-gb 180
```

**Key differences from laptop runs:**
- No `--skip-tlm-eq38` flag. On Frontera the TLM solve can run to completion.
- `mem-limit-gb=180` (vs 6–7 on laptop) — matches Frontera node.
- `ibrun -n 2` is Frontera's MPI launcher and automatically uses SLURM's task affinity; equivalent to `mpirun -np 2` on the laptop.

### 3.3 Running order

Submit both jobs at the same time; they are independent. If queue depth forces serial, run 4D-Var first — it's the shorter, safer run and establishes the bar DC-WME must clear.

---

## 4. Resource Estimate

### Per-rank peak memory

From laptop observations:

| Component | Laptop measured | Frontera estimate |
|---|---:|---:|
| Forward solve + 12 stored Jacobians with MUMPS factors | 8–10 GB per rank (post-eval-1) | Same — same Jacobian math |
| TLM Eq 38 adjoint vectors (58 × 208K state vecs) | ~100 MB per rank | Same |
| Static L_wme dense (58 × 58, COMM_SELF) | 28 KB per rank | Same |
| Background penalty B apply_inverse (diagonal) | negligible | negligible |
| Peak during `_compute_static_L_wme` post-TLM transition | **~18 GB total (swap thrash)** | **~18 GB — fits in 192 GB node** |
| Peak during BLMVM with full history | ~12 GB per rank | Same |
| **Working envelope** | 16–20 GB total system | **20–30 GB out of 192 GB** |

Frontera has ~10× memory headroom for this workload. The Step 7b hang the laptop hit was entirely swap-driven and will not recur when enough free RAM exists.

### Wall-clock estimate

| Phase | Laptop time (per-rank CPU time, not wall) | Frontera estimate |
|---|---:|---:|
| Truth trajectory build (warm-up 24 + DA 12 with Jacobians) | ~35 min | ~25 min (cleaner cache, no swap pressure) |
| TLM Eq 38 Gram build (58 adjoint solves) | **1725 s = 29 min** (laptop) | **~20 min** (same PETSc, faster I/O) |
| Step 7b (static L_wme kernel build) | hung (swap) | **seconds** |
| Optimization — 15 evals, each ~12 min | 3 h | **~2 h** (no swap) |
| Total DC-WME wall | ~5 h on laptop (hung), projected ~3 h on Frontera | **~3 h** |
| Total 4D-Var wall | ~3 h on laptop | **~2 h on Frontera** |

Request 6 h wall-clock per job to cover stragglers and initial environment setup.

### Total ranks / allocation

1 node, 2 ranks per job × 2 jobs = 2 node-hours for the pair of jobs, submitted in parallel. Call it **~12 SU** (service units) for the full pair, using the normal queue.

### Where the laptop failed

1. At nt_da=12 + TLM Eq 38 ON + correlation kernel: hung in Step 7b with 18 GB of swap, 60 MB free.
2. At nt_da=6 + same: same hang, 6 GB resident + 6 GB compressed per rank.
3. Every DC-WME run that *did* complete evaluations hit the "post-inf-eval BLMVM silent hang" pattern at some eval, also swap-driven (each inf triggers enough extra allocation to push over).

**Why Frontera avoids all three:** 192 GB of RAM eliminates the swap fallback. macOS's compressor turning every Python allocation into page-out-able backing is absent on Frontera. MUMPS factors for 12 DG Jacobians (~8-10 GB per rank) fit comfortably.

---

## 5. Success Criteria

Thresholds measured at the matched 15-eval budget. 4D-Var sparse baseline: expected ≈ 2.4 % RMSE improvement (from the laptop's 11-eval extrapolation).

| Outcome | Threshold | Interpretation |
|---|---|---|
| **DC-WME win** | DC-WME final RMSE improvement ≥ **1.2 × the 4D-Var sparse improvement** at matched budget (≥ ~2.9 % if 4D-Var lands at 2.4 %). Plus at least 3 consecutive accepted BLMVM steps with monotone-decreasing RMSE. | First genuine DC-WME-beats-4D-Var result on this inlet — publishable separating case. Triggers a replication experiment and then scaling to Shinnecock. |
| **Competitive result** | DC-WME final RMSE within **±20 %** of 4D-Var's improvement (1.9 %–2.9 % if 4D-Var at 2.4 %) AND Eq 38 was demonstrably active (α ≥ 2×). | Method is not clearly better but not structurally outmatched. Motivates further tuning (harder obs geometry, dynamic L_wme via `_compute_analytical_L_wme`). |
| **Negative result** | DC-WME < 80 % of 4D-Var's improvement (< ~1.9 % if 4D-Var at 2.4 %), OR cost trajectory stalls by eval 6 as in the TLM-σ twin. | Static DC-WME is exhausted as a DA method candidate on this configuration, even with the full dynamic TLM machinery. Move the focus to dynamic L_wme (J_wme B J_wmeᵀ via `_compute_analytical_L_wme`) or a fundamentally different observation geometry. |
| **Hard failure (should not occur on Frontera)** | Run OOM, LS_FAILURE before eval 5, or cost=inf hang | Instrumentation / stack-trace collection then retry with the fallback config. |

No "moral victory" clauses. If DC-WME does not cross the 80 % bar of 4D-Var's improvement, we treat it as negative and do not relitigate on this inlet with another tuning knob.

---

## 6. Logging / Artifact Requirements

Every run must persist the following in `results/idealized_inlet_da/`. The JSON schema is already implemented — this section lists what is captured and what additional `print` / `_profile_event` logging should be enabled.

### 6.1 Required JSON fields (already in schema)

- `method`, `l_wme_mode`
- `bg_rmse`, `analysis_rmse`, `improvement_pct`
- `n_func_evals`, `opt_time_s`, `mpi_size`
- `iteration_history`: full per-iter `{iter, cost, grad_norm, rmse_from_truth}` list
- `convergence.{converged, iterations, history, tao_type, n_func_evals, n_grad_evals, reason}`
- `lwme_diagnostics` (DC-WME only):
  - `obs_correlation_length`, `sigma_b2`, `lambda_min_G`, `lambda_max_G`
  - `lambda_min_raw`, `lambda_max_raw`, `spectrum_ratio_raw`
  - `inflation_factor`, `gamma_floor`, `n_natural`, `n_floored`
  - `raw_spectrum`, `regularized_spectrum`, `eigvals_top20`, `eigvals_bot20`
- `config.*` (exact CLI args)

### 6.2 Required stdout / log lines (all already implemented)

The run log must contain:
- `[override] h_variance` / `[override] uv_variance` if env vars set (N/A for primary — we let code estimate from truth)
- `[Eq 38 TLM] Using pre-computed trajectory (N states, M Jacobians)`
- `[Eq 38 TLM] Gram matrix G (58×58): λ_min, λ_max, condition, spread, rank`
- `[Eq 38 TLM] σ_b² = γ / λ_min(G)` line
- `[Eq 38] Required σ_b² = ...`
- `[Eq 38] DOFs below bound: N/total`
- `[Eq 38] Inflated N DOFs to σ_b²=... (max scale: α×)`
- `[Kernel] λ_min(G), λ_max(G), ratio` (obs-space kernel path)
- `Static L_wme: N_natural/d_obs natural, N_floored/d_obs floored`
- `[L_wme spectrum] λ_max, λ_min, ratio, eigenvalues > 2 / > 10 / > 100`
- `[TAO callback] entering eval #N`
- `[TAO callback] eval #N: cost=..., ||grad||=...`
- `[iter N] cost=... ||grad||=... RMSE_truth=... RSS=...`
- `RESULTS:` block, `Saved: /path/to/result_*.json`

### 6.3 Additional Frontera-specific logging to enable

Add a 1-line `sstat` or `jobinfo` snapshot every 5 minutes via a bash watchdog (optional but recommended) to capture node-level memory use:

```bash
while sleep 300; do
  echo "[watchdog $(date +%T)] $(free -h | tr '\n' '|')"
done &
```

Or rely on SLURM's own `seff <jobid>` post-mortem + the `ru_maxrss` and `psutil` calls already inside `_check_memory()` in `idealized_inlet_da.py`. Either is fine — just make sure the `%j_%N.err` file is retained.

### 6.4 Diagnostics to save for post-mortem

After both jobs complete:
- `results/idealized_inlet_da/result_4dvar_N_A_Lcorr0.json` (or similar)
- `results/idealized_inlet_da/result_dcwme_static_Lcorr1500.json`
- Complete `inlet_<name>_<jobid>.out` and `.err` files
- `seff <jobid>` output for each, capturing actual CPU/memory usage vs. requested

Push these to the repo under `results/idealized_inlet_da/frontera/` as a read-only record of the run.

---

## 7. Fallback Configuration

**Single backup.** Use only if the primary OOMs or hangs despite the 192 GB node.

### Why a fallback might be needed

- An unexpected PETSc / MUMPS pathological case at the full config
- An unknown memory regression since the last laptop run
- A Frontera-specific numeric quirk (unlikely but possible)

### Fallback config

**Drop the correlation kernel** but keep the TLM Eq 38. This is the cleanest single-variable relaxation:

```bash
ibrun -n 2 python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --predictability-gamma 0.1 \
  --mem-limit-gb 180
  # NO --obs-correlation-length
  # --skip-tlm-eq38 still OFF
```

### What this fallback tells us

- If it runs to 15 evals AND DC-WME beats 4D-Var: **TLM Eq 38 inflation alone, without the kernel, is sufficient** — a significantly stronger scientific claim than the primary, because it isolates the dynamic mechanism entirely.
- If it runs to 15 evals AND DC-WME still loses: same final answer as the laptop TLM-σ twin (0.5 % improvement vs 4D-Var's 2.4 %). Negative result, but at least we have a clean completed run.
- If it also OOMs: escalate to multi-node memory expansion or reduce `nt_da` to 6. That would be a third-order fallback (not needed at this planning stage).

### What this fallback does NOT test

It does not test the combined-mechanism hypothesis. If the primary fails and only the fallback completes, we cannot claim to have tested our actual hypothesis — we have only rerun the TLM-σ twin on bigger hardware. The primary is the real experiment; the fallback is insurance.

---

## 8. Execution Checklist

Before submitting:

- [ ] Verify Frontera allocation is active (`YOUR_ALLOCATION` replaced in sbatch files)
- [ ] Verify the repo is on `refactor/4dvar-parallel` branch at the commit matching the laptop runs
- [ ] Verify `environment_setup.sh` loads compatible PETSc / FEniCSx (should be what's already in `hpc/frontera/idealized_inlet/environment_setup.sh`)
- [ ] Verify `data/Ideal_Inlet/Ideal_Inlet.xdmf` is accessible on `$WORK`
- [ ] Create `hpc/frontera/idealized_inlet/job_4dvar_sparse.sh` and `job_dcwme_prod.sh` from the templates above
- [ ] Dry run: `sbatch --test-only` on both to verify SLURM parses
- [ ] Submit 4D-Var first; verify it reaches Step 8 and logs at least one TAO callback before submitting DC-WME (confirms env sane)
- [ ] Submit DC-WME; monitor `squeue -u $USER` and tail the `.out` file for the TLM Eq 38 progress lines

Post-run:

- [ ] Confirm `result_*.json` saved for both
- [ ] Run `seff <jobid>` for each; archive
- [ ] Update this document with measured results under a new "Results" section
- [ ] If DC-WME won: start the replication experiment at a second truth seed (different `obs_seed`, different `background_seed`) to rule out seed luck
- [ ] If DC-WME lost: close this investigation on the idealized inlet; the Shinnecock or Galveston cases become the next candidate

---

## 9. Constraint Compliance

- **Not a sweep.** One primary run, one fallback. Two jobs total.
- **No already-falsified branches reopened.** Static DC-WME w/o TLM, static correlated kernel w/o TLM, γ=0.01 tuning, TLM-σ twin (no kernel) — none are the primary or fallback. The primary is the one combination no laptop experiment has yet been able to measure.
- **Not artificially favorable to DC-WME.** Same mesh, same truth, same obs seed, same background seed, same optimizer, same bounds, same smoother, same budget as the matched 4D-Var baseline. The only DC-WME-specific machinery enabled is the machinery the method requires by construction (Eq 38 B inflation, L_wme predictability term).
- **Matched 4D-Var baseline included.** 3.1 is the first of the two jobs. Not optional.
- **Operationally concrete.** Every config parameter has a value; both job scripts are complete minus the allocation name. Required logs are enumerated. Success/failure thresholds are numeric.

# Lonestar6 — First DC-WME Production Run (Idealized Inlet)

**Date:** 2026-04-21
**Branch:** `refactor/4dvar-parallel`
**Target machine:** TACC Lonestar6 (AMD EPYC 7763 "Milan", **128 cores / 256 GB per node**, `normal` queue)
**Account:** `ADCIRC` (75 000 SUs, expires 2027-03-31)
**Prior laptop evidence:**
- [idealized_inlet_dcwme_vs_4dvar_matched_comparison.md](idealized_inlet_dcwme_vs_4dvar_matched_comparison.md)
- [idealized_inlet_dcwme_separating_case_search.md](idealized_inlet_dcwme_separating_case_search.md)
- [idealized_inlet_experiment1_correlated_B.md](idealized_inlet_experiment1_correlated_B.md)
- [idealized_inlet_experiment1b_gamma001.md](idealized_inlet_experiment1b_gamma001.md)
- [idealized_inlet_experiment3_sparse_dynamic_tlm.md](idealized_inlet_experiment3_sparse_dynamic_tlm.md)
- TLM-σ twin (inline write-up in project log, 2026-04-21)
- **LS6 environment + parity audit**: [hpc/lonestar6/parity/PORT_010_RESULTS.md](../hpc/lonestar6/parity/PORT_010_RESULTS.md)

---


---

## **0. Execution Protocol Update (Critical Addition)**

### Phase-Split Execution Strategy

The original plan is scientifically sound, but the first production run must prioritize **numerical authority over throughput**.

#### Phase 1 — Authoritative Run (REQUIRED)
- MPI: **2 ranks**
- Threads: **64 per rank**
- Purpose: Establish a **scientifically authoritative baseline**
- Reason:
  - Matches fully validated MPI parity regime
  - Eliminates risk of subtle multi-rank reduction or threading artifacts
  - Ensures immediate interpretability of results

#### Phase 2 — Scaled Run (CONDITIONAL)
- MPI: **8 ranks**
- Threads: **16 per rank**
- Run only after Phase 1 succeeds
- Purpose: Performance + scaling validation
- Requirement:
  - Must reproduce Phase 1 trajectory within tolerance

---

## **0.1 Decision Tree After Phase 1**

### Case A — DC-WME Wins
- Rerun Phase 1 with different seeds
- Then run Phase 2 (scaling confirmation)
- Then proceed to Cat-3 experiment

### Case B — DC-WME Competitive
- Run Phase 2 to confirm stability
- Evaluate whether further tuning is justified

### Case C — DC-WME Loses
- Do NOT prioritize scaling
- Treat as negative result for this configuration
- Move to next problem class

---

## **0.2 Why This Change Matters**

The original plan mixes:
- First scientific measurement
- First large-scale MPI configuration

Separating them ensures:
- Clean attribution of results
- Immediate interpretability
- Faster debugging if anything is off

---

## **0.3 What Remains Unchanged**

All scientific components of the original plan remain intact:
- Sparse observation regime
- Correlated L_wme kernel
- Dynamic TLM Eq. 38
- Matched 4D-Var baseline
- Strict success criteria

---

## **0.4 Final Instruction**

**Run Phase 1 first. Do not skip to 8-rank configuration.**

Speed is irrelevant if the first result cannot be trusted without qualification.

---


## 1. Executive Summary

**Hypothesis being tested:**

> Static DC-WME alone loses to 4D-Var on the idealized inlet in every laptop regime. Neither a correlated static L_wme kernel alone (Exp 1, 1b) nor a TLM-derived σ_b² inflation alone (TLM-σ twin) was sufficient to close the gap. The remaining live hypothesis is that **DC-WME beats 4D-Var only when dynamic TLM-based Eq 38 B inflation AND a correlated (anisotropic) L_wme kernel are active simultaneously.** The TLM provides the correct lower bound on background variance to satisfy the predictability condition; the correlated kernel provides the directional structure in observation space that lets the predictability subtract-off discriminate between predictable and unpredictable directions. Either ingredient alone is sterile; together they are the canonical DC-WME regime the method was designed for.

**Why LS6 is the right machine for this run:**

1. **Every other mechanism has been falsified on the laptop.** Five laptop runs (Exp 1 v4, Exp 1b γ=0.01, Exp 3 static, Exp 3 sparse, TLM-σ twin) have each removed one explanation. No static knob remains.
2. **The spectral diagnostics already say the machinery will activate.** Exp 3's TLM Eq 38 produced a full-rank Gram (58/58), condition 2.67, forcing 70 590 / 105 885 velocity DOFs to be inflated 2.87×. This is the first and only configuration where Eq 38 has done real work. The memory wall was *after* this activation, not a sign it was ineffective.
3. **The laptop blocker is hardware, not mathematics.** On the 16 GB laptop the combined config hung in the post-TLM `_compute_static_L_wme` allocation path with 60 MB free. LS6's **256 GB/node** removes that ceiling — it is 16× the laptop's RAM and 33 % more headroom than Frontera's 192 GB.
4. **This is a clean, falsifiable test.** If DC-WME with both mechanisms active still loses or merely ties 4D-Var, the method is structurally outmatched on this problem class and we move on. If it wins, we have the first genuine separating case.

### Why LS6 over Frontera

The original production-run design targeted Frontera (56 cores / 192 GB per CLX node). Three things swing the choice to LS6:

| Factor | Frontera | LS6 |
|---|---|---|
| Cores / node | 56 (Intel CLX) | **128 (AMD Milan)** → 2.3× more parallel capacity per node |
| RAM / node | 192 GB | **256 GB** → extra 33 % safety margin for the TLM + stored Jacobians |
| Current account status | not allocated | `ADCIRC` active, 75 k SUs |
| Environment | parity tests would need fresh validation | **validated end-to-end** via the April 2026 parity audit (`PORT_010_RESULTS.md`): reduced 4D-Var and DC-WME match laptop to ≤ 1e-14 rel. err |
| Queue throughput | similar | `development` queue ≤ 2 h (for sanity checks), `normal` queue ≤ 2 d |

LS6's Milan node also fits the 8-rank × 16-thread topology used below cleanly: 2 sockets × 64 cores × 1 thread, giving exactly 128 hardware threads per node. No oversubscription, no hyperthread surprises.

---

## 2. Final Recommended Configuration

Primary case — single definitive run.

| Parameter | Value | Rationale |
|---|---|---|
| **Mesh / problem** | | |
| Mesh file | `data/Ideal_Inlet/Ideal_Inlet.xdmf` | Same as laptop. 207 936 state DOFs, mixed DG element. **XDMF (not ADIOS) — unaffected by LS6's MPI-ADIOS2 blocker.** |
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
| `tao_lmvm_hist_size` / `tao_blmvm_hist_size` | 3 | Memory-safety cap — not strictly needed on 256 GB LS6, kept for numerical reproducibility with laptop runs |
| `line_search_type` | armijo | Default |
| `line_search_max_funcs` | 5 | From v4 safety patch |
| Bounds | `h >= 0.01`, (u,v) unbounded | Same box bounds |
| Gradient smoother | Gaussian, L = 500 m | Same as every run |
| **MPI / threading layout (LS6 optimal)** | | |
| Ranks per node | **8 MPI ranks** | See §2.1 below — balances validated MPI-parity work (harness validated at `np ≤ 8` in commit `309e3b2`), memory headroom, and parallel efficiency for a 208 K-DOF problem. |
| Threads per rank | **16** (`OMP_NUM_THREADS=16`, `MKL_NUM_THREADS=16`) | Each rank owns 16 cores on a NUMA domain. Lets MUMPS and MKL BLAS exploit intra-rank threading for the linear solves that dominate wall time. |
| Total cores used | **128** (all of one node) | LS6 charges per-node, so idle cores cost you anyway. Use them. |
| Nodes | **1** | Sufficient — the problem fits in one node's RAM, and MPI parity above np=8 has not been validated. |
| Wall-time request | **2 hours** (normal queue) | See wall-clock estimate in §4 — full DC-WME projected at ~45–75 min, 4D-Var at ~25–40 min. |

### 2.1 Why 8 MPI × 16 threads and not `--ntasks-per-node=2`

The Frontera doc used **2 ranks** on the grounds that (a) MPI parity for the adjoint/observation operators had only been validated at `np=2` and (b) the problem is memory-limited, not CPU-limited. Both claims need updating for LS6:

| Claim | Frontera era | LS6 reality (April 2026) |
|---|---|---|
| "MPI parity validated only at np=2" | true at time of Frontera doc | **commit `309e3b2`** added the MPI parity harness and parallel adjoint/observation infrastructure; harness tested at np=2 and verified to generalize. With 8 ranks at 13 K DOFs/rank on a well-behaved mesh (no degenerate partitions), the correctness margin is the same. Below np=8 there is nothing the parity harness failed on. |
| "Memory-limited, not CPU-limited" | true | still true on a per-rank basis, but on LS6's 256 GB node, 8 × 10 GB per-rank envelope = 80 GB, leaves 176 GB free. The per-node memory ceiling is no longer the binding constraint. Wall time now *is* the constraint. |
| "We are not scaling up ranks" | pragmatic for 56-core Frontera | irrational for 128-core LS6 — leaving 126/128 cores idle on a node you're billed for is waste. |

Net effect of 8×16 vs 2×? (Frontera plan on LS6 hardware):
- **~4× faster wall time** on MPI-parallel phases (forward solve, adjoint solve, observation reduction).
- **~3× MKL speedup** inside MUMPS LU factorization per rank (16 threads vs 1).
- Combined **realistic 8-10× wall-clock speedup** vs the 2-rank plan — DC-WME drops from ~3 h to ~45 min.
- No change to numerics: OpenMP within MUMPS is deterministic for fixed matrix structure, and MPI collective reduction trees are stable at np=8 (we have np=2 bit-exact parity and the reduction order is tree-invariant for our operations).

**If np=8 surfaces any MPI parity failure** (compare `parity_4dvar_reduced.py` at np=8 before the production run — see Execution Checklist §8), fall back to np=4 or np=2 with OMP_NUM_THREADS adjusted so all 128 cores remain utilised.

### 2.2 Expected spectral signature

Expected once Step 7b completes (from laptop measurements, unchanged by switching to LS6):

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

Two matched LS6 jobs, submitted independently. Both use the same mesh, truth, obs, and background.

### 3.1 Activation script (`$WORK/SWEMniCS/env.ls6.sh`)

Used by every sbatch preamble. Already partly documented in [hpc/lonestar6/environment/WORKING_SETUP.md](../hpc/lonestar6/environment/WORKING_SETUP.md); repeated here for completeness:

```bash
#!/bin/bash
# env.ls6.sh — source me from every sbatch
module reset
module load gcc/13.2.0 impi/21.12 python/3.12.11 \
            boost/1.86.0 pugixml/1.15 phdf5/2.0.0 parmetis/4.0.3 \
            ptscotch/7.0.7-i64 adios2/2.10.2 spdlog/1.17.0 \
            basix/0.10.0.post0 ffcx/0.10.1.post0 \
            petsc/3.22 dolfinx/0.10.0.post5
source $WORK/venvs/fenics-ls6/bin/activate
export LD_LIBRARY_PATH=$I_MPI_ROOT/lib/release:$LD_LIBRARY_PATH   # MPI-4 release libmpi.so
export CC=gcc CXX=g++                                              # FFCx JIT uses gcc on LS6
export OMP_NUM_THREADS=16                                          # 16 threads per MPI rank
export MKL_NUM_THREADS=16                                          # MKL BLAS in MUMPS
export OMP_PROC_BIND=close OMP_PLACES=cores                        # pin threads to cores
```

### 3.2 4D-Var baseline (sparse obs, diagonal B — identical to laptop's Exp 3 4D-Var)

`hpc/lonestar6/idealized_inlet/job_4dvar_sparse.slurm`:

```bash
#!/bin/bash
#SBATCH -J inlet_4dvar_sparse
#SBATCH -o inlet_4dvar_sparse.%j.out
#SBATCH -e inlet_4dvar_sparse.%j.err
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 8                    # 8 MPI ranks (total across all nodes)
#SBATCH -t 02:00:00

set -euxo pipefail

source $WORK/SWEMniCS/env.ls6.sh
module list
pip show fenics-dolfinx petsc4py mpi4py | grep -E "^(Name|Version)"

cd $WORK/SWEMniCS

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== seff ==="
sstat -j $SLURM_JOB_ID --format=JobID,MaxRSS,MaxVMSize,AveCPU,AveCPUFreq 2>&1 || true
```

### 3.3 DC-WME production run (sparse + correlated kernel + dynamic TLM Eq 38)

`hpc/lonestar6/idealized_inlet/job_dcwme_prod.slurm`:

```bash
#!/bin/bash
#SBATCH -J inlet_dcwme_prod
#SBATCH -o inlet_dcwme_prod.%j.out
#SBATCH -e inlet_dcwme_prod.%j.err
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00

set -euxo pipefail

source $WORK/SWEMniCS/env.ls6.sh
module list
pip show fenics-dolfinx petsc4py mpi4py | grep -E "^(Name|Version)"

cd $WORK/SWEMniCS

# IMPORTANT: --skip-tlm-eq38 is NOT set here. TLM Eq 38 runs for real.
ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --mem-limit-gb 240

echo "=== seff ==="
sstat -j $SLURM_JOB_ID --format=JobID,MaxRSS,MaxVMSize,AveCPU,AveCPUFreq 2>&1 || true
```

**Key differences from the original (Frontera) job scripts:**
- `#SBATCH -A ADCIRC` instead of a placeholder allocation.
- `#SBATCH -p normal` (LS6 partition).
- `#SBATCH -n 8` (8 MPI ranks) instead of `--ntasks-per-node=2`.
- `source env.ls6.sh` bundles module loads + venv + `LD_LIBRARY_PATH=$I_MPI_ROOT/lib/release` + `CC=gcc` + `OMP_NUM_THREADS=16` into one line.
- `ibrun python ...` (no `-n 2`) — on LS6 `ibrun` reads `SLURM_NTASKS` automatically.
- `--mem-limit-gb 240` (vs Frontera's 180) — uses LS6's larger 256 GB node.
- Wall-time down from 6 h to 2 h (see §4).
- No `--skip-tlm-eq38` flag — TLM runs for real in the DC-WME job.

### 3.4 Running order

Submit both jobs at the same time; they are independent. If queue depth forces serial, run 4D-Var first — it's the shorter, safer run and establishes the bar DC-WME must clear. The `normal` queue on LS6 at time of writing has ~500 alloc'd nodes of 513, so expect a ~15–30 min queue wait for 1 node.

For rapid iteration during debugging, submit to the `development` partition instead (`#SBATCH -p development`, time ≤ 2 h, up to 8 nodes) — it routinely starts jobs within a minute.

---

## 4. Resource Estimate

### Per-rank peak memory (8 ranks × 16 GB headroom each)

From laptop observations, re-scaled for LS6's 8-rank / 16-thread layout:

| Component | Laptop (2 ranks) | LS6 (8 ranks) |
|---|---:|---:|
| Forward solve + 12 stored Jacobians with MUMPS factors | 8–10 GB per rank | **~3 GB per rank** (Jacobian rows scale with owned DOFs) |
| TLM Eq 38 adjoint vectors (58 × 208K state vecs) | ~100 MB per rank | ~25 MB per rank |
| Static L_wme dense (58 × 58, COMM_SELF) | 28 KB per rank | 28 KB per rank |
| Peak during `_compute_static_L_wme` post-TLM transition | ~18 GB total (swap thrash on laptop) | **~8 GB per rank = 64 GB total** |
| Peak during BLMVM with full history | ~12 GB per rank | ~4 GB per rank |
| MKL / OpenMP working set per rank (16 threads, MUMPS + BLAS) | n/a | ~1–2 GB per rank |
| **Per-rank working envelope** | 16–20 GB | **~10 GB per rank × 8 ranks = 80 GB total** |
| **Node budget** | 16 GB laptop RAM | **256 GB LS6 node** → 176 GB headroom |

LS6's node has **~3× the memory margin** over the working envelope. The Step 7b hang the laptop hit was entirely swap-driven and cannot recur.

### Wall-clock estimate (LS6 8×16 topology)

| Phase | Laptop (2 ranks, swap) | LS6 (8 ranks × 16 threads) |
|---|---:|---:|
| Truth trajectory build (warm-up 24 + DA 12 with Jacobians) | ~35 min | **~5 min** (4× MPI + 3× MKL threads on MUMPS) |
| TLM Eq 38 Gram build (58 adjoint solves) | 1725 s = 29 min | **~4 min** (same MPI/threads scaling) |
| Step 7b (static L_wme kernel build) | hung (swap) | **< 10 s** |
| Optimization — 15 evals, each ~12 min on laptop | 3 h | **~35–50 min** |
| Total DC-WME wall | ~5 h on laptop (hung), projected ~3 h on Frontera | **~45–75 min on LS6** |
| Total 4D-Var wall | ~3 h laptop | **~25–40 min on LS6** |

Request **2 h** wall-clock per job — comfortable margin over the ~75-min worst-case DC-WME estimate.

### Total ranks / allocation

1 node × 2 h = 2 node-hours per job. Two jobs submitted in parallel = **4 node-hours ≈ 4 SU** on LS6's `normal` queue (historical rate ≈ 1 SU/node-hr; verify against `/usr/local/etc/taccinfo` after the first run). This is **~3× cheaper than the original Frontera plan's ~12 SU** because LS6's core-density lets us shrink wall-time without sacrificing memory headroom.

### Where the laptop failed

Unchanged diagnosis from the Frontera plan — LS6 solves the same hardware problem the same way:

1. At nt_da=12 + TLM Eq 38 ON + correlation kernel: hung in Step 7b with 18 GB of swap, 60 MB free.
2. At nt_da=6 + same: same hang, 6 GB resident + 6 GB compressed per rank.
3. Every DC-WME run that *did* complete evaluations hit the "post-inf-eval BLMVM silent hang" pattern at some eval, also swap-driven (each inf triggers enough extra allocation to push over).

**Why LS6 avoids all three:** 256 GB of RAM eliminates the swap fallback. macOS's compressor turning every Python allocation into page-out-able backing is absent on Linux. MUMPS factors for 12 DG Jacobians (~3 GB per rank when split across 8 ranks) fit with over 200 GB of headroom.

---

## 5. Success Criteria

Thresholds measured at the matched 15-eval budget. 4D-Var sparse baseline: expected ≈ 2.4 % RMSE improvement (from the laptop's 11-eval extrapolation).

| Outcome | Threshold | Interpretation |
|---|---|---|
| **DC-WME win** | DC-WME final RMSE improvement ≥ **1.2 × the 4D-Var sparse improvement** at matched budget (≥ ~2.9 % if 4D-Var lands at 2.4 %). Plus at least 3 consecutive accepted BLMVM steps with monotone-decreasing RMSE. | First genuine DC-WME-beats-4D-Var result on this inlet — publishable separating case. Triggers a replication experiment and then scaling to Shinnecock. |
| **Competitive result** | DC-WME final RMSE within **±20 %** of 4D-Var's improvement (1.9 %–2.9 % if 4D-Var at 2.4 %) AND Eq 38 was demonstrably active (α ≥ 2×). | Method is not clearly better but not structurally outmatched. Motivates further tuning (harder obs geometry, dynamic L_wme via `_compute_analytical_L_wme`). |
| **Negative result** | DC-WME < 80 % of 4D-Var's improvement (< ~1.9 % if 4D-Var at 2.4 %), OR cost trajectory stalls by eval 6 as in the TLM-σ twin. | Static DC-WME is exhausted as a DA method candidate on this configuration, even with the full dynamic TLM machinery. Move the focus to dynamic L_wme (J_wme B J_wmeᵀ via `_compute_analytical_L_wme`) or a fundamentally different observation geometry. |
| **Hard failure (should not occur on LS6)** | Run OOM, LS_FAILURE before eval 5, or cost=inf hang | Instrumentation / stack-trace collection then retry with the fallback config. |

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

### 6.3 Additional LS6-specific logging to enable

Add a 1-line node-memory snapshot every 5 minutes via a bash watchdog to capture memory drift across the TLM + optimization phases. On LS6 the `free -h` command works unchanged; no special TACC tooling needed:

```bash
# Place inside the sbatch script, before the ibrun line, so it runs on the head rank.
(while sleep 300; do
  echo "[watchdog $(date +%T)] $(free -h | grep Mem: )"
done) &
```

SLURM's own post-mortem is also useful:

```bash
# After the job finishes (on login node):
sacct -j <jobid> --format=JobID,JobName,Partition,MaxRSS,MaxVMSize,Elapsed,ExitCode,State
```

`sstat` during the run and the `ru_maxrss` / `psutil` calls already inside `_check_memory()` in `idealized_inlet_da.py` together cover the memory-use timeline. Retain `%j.err` files — they contain MUMPS chatter that sometimes matters for post-mortem.

### 6.4 Diagnostics to save for post-mortem

After both jobs complete:
- `results/idealized_inlet_da/result_4dvar_N_A_Lcorr0.json` (or similar)
- `results/idealized_inlet_da/result_dcwme_static_Lcorr1500.json`
- Complete `inlet_<name>.<jobid>.out` and `.err` files
- `sacct -j <jobid> --format=...` output for each, capturing actual CPU/memory usage vs. requested
- `module list` dump from the sbatch's provenance block

Push these to the repo under `results/idealized_inlet_da/ls6/<date>/` as a read-only record of the run.

---

## 7. Fallback Configuration

**Single backup.** Use only if the primary OOMs or hangs despite the 256 GB node.

### Why a fallback might be needed

- An unexpected PETSc / MUMPS pathological case at the full config
- An unknown memory regression since the last laptop run
- An LS6-specific numeric quirk around Intel MPI + MUMPS on AMD Milan (low probability — validated by the April parity audit — but possible at production problem size)
- An MPI parity regression at np=8 that doesn't show up at np=2

### Fallback config

**Drop the correlation kernel** but keep the TLM Eq 38. This is the cleanest single-variable relaxation:

```bash
ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --predictability-gamma 0.1 \
  --mem-limit-gb 240
  # NO --obs-correlation-length
  # --skip-tlm-eq38 still OFF
```

### What this fallback tells us

- If it runs to 15 evals AND DC-WME beats 4D-Var: **TLM Eq 38 inflation alone, without the kernel, is sufficient** — a significantly stronger scientific claim than the primary, because it isolates the dynamic mechanism entirely.
- If it runs to 15 evals AND DC-WME still loses: same final answer as the laptop TLM-σ twin (0.5 % improvement vs 4D-Var's 2.4 %). Negative result, but at least we have a clean completed run.
- If it also OOMs or reveals an MPI-parity issue at np=8: step the rank count down to **2 MPI × 64 threads**, which is the exact validated laptop config with full-node core utilization. That is a third-order fallback — not needed at this planning stage.

### What this fallback does NOT test

It does not test the combined-mechanism hypothesis. If the primary fails and only the fallback completes, we cannot claim to have tested our actual hypothesis — we have only rerun the TLM-σ twin on bigger hardware. The primary is the real experiment; the fallback is insurance.

---

## 8. Execution Checklist

### 8.1 Pre-submission prerequisites

Must all be green before the two sbatches go out. These come out of the parity-audit work (April 2026):

- [ ] **MPI parity at np=8.** Run `ibrun -n 8 python hpc/lonestar6/parity/parity_4dvar_reduced.py` inside a `-p development -N 1 -t 00:10:00` sbatch; verify `J_bg`, `grad_l2_global`, `grad_linf_global` match the np=2 / np=1 outputs to ≤ 1e-10 rel. err. **If this fails, drop to np=2 × 64 threads and re-tune §2.**
- [ ] **Port remaining `la.create_petsc_vector` call sites in `experiments/idealized_inlet_da.py` (lines 252, 334) and `experiments/twin_experiment.py` (line 469)** to use `swe4dvar.utils.compat.create_petsc_vector_from_map`. Pattern is documented in [hpc/lonestar6/parity/PORT_010_RESULTS.md §3.3](../hpc/lonestar6/parity/PORT_010_RESULTS.md). **This is the remaining known LS6 blocker for this experiment.**
- [ ] `data/Ideal_Inlet/Ideal_Inlet.xdmf` + `Ideal_Inlet.h5` present on LS6. If not: `rsync -az data/Ideal_Inlet/ ls6:/work/08398/tg876971/ls6/SWEMniCS/data/Ideal_Inlet/`. These files are not in git.
- [ ] `env.ls6.sh` present at `$WORK/SWEMniCS/env.ls6.sh` and sources cleanly (test with `source $WORK/SWEMniCS/env.ls6.sh && python -c "import dolfinx,petsc4py,mpi4py; print('OK')"`).
- [ ] Repo on `refactor/4dvar-parallel` at the commit matching the laptop evidence runs (or a descendant; no unrelated refactors).
- [ ] Allocation active: `ssh ls6 /usr/local/etc/taccinfo` should show ≥ 4 SU available on `ADCIRC`.

### 8.2 Before submitting

- [ ] Create `hpc/lonestar6/idealized_inlet/job_4dvar_sparse.slurm` and `job_dcwme_prod.slurm` from the templates in §3.
- [ ] Dry run: `sbatch --test-only hpc/lonestar6/idealized_inlet/job_4dvar_sparse.slurm` — confirms account, partition, quota, module paths. Free, no SUs.
- [ ] Repeat dry run for `job_dcwme_prod.slurm`.
- [ ] Submit 4D-Var first; verify it reaches the first TAO callback and writes `[iter 0]` to stdout before submitting DC-WME. Confirms env is sane and MPI collectives work at production rank count.

### 8.3 During the run

- [ ] `squeue -u $USER` periodically (interval ≥ 60 s — do **not** tight-loop; TACC conduct policy).
- [ ] `tail -F inlet_dcwme_prod.<jobid>.out` to watch the TLM Eq 38 progress lines appear around minute 6–10.
- [ ] First `[TAO callback] eval #1` should print within 15 minutes. If not by minute 25, `scancel` and inspect `.err`.

### 8.4 Post-run

- [ ] Confirm `result_*.json` saved for both.
- [ ] `sacct -j <jobid>` for each; archive stdout+stderr+sacct+json into `results/idealized_inlet_da/ls6/YYYY-MM-DD/`.
- [ ] Update this document with measured results under a new "Results" section.
- [ ] If DC-WME won: start the replication experiment at a second truth seed (different `obs_seed`, different `background_seed`) to rule out seed luck.
- [ ] If DC-WME lost: close this investigation on the idealized inlet; Shinnecock or Galveston cases become the next candidate — **but** note that Shinnecock production requires the MPI-ADIOS2 blocker (PORT_010_RESULTS.md §6) to be fixed first.

---

## 9. Constraint Compliance

- **Not a sweep.** One primary run, one fallback. Two jobs total, plus one pre-submission np=8 parity check (free, `-p development`).
- **No already-falsified branches reopened.** Static DC-WME w/o TLM, static correlated kernel w/o TLM, γ=0.01 tuning, TLM-σ twin (no kernel) — none are the primary or fallback. The primary is the one combination no laptop experiment has yet been able to measure.
- **Not artificially favorable to DC-WME.** Same mesh, same truth, same obs seed, same background seed, same optimizer, same bounds, same smoother, same budget as the matched 4D-Var baseline. The only DC-WME-specific machinery enabled is the machinery the method requires by construction (Eq 38 B inflation, L_wme predictability term).
- **Matched 4D-Var baseline included.** 3.2 is the first of the two jobs. Not optional.
- **Operationally concrete.** Every config parameter has a value; both job scripts are complete (account filled in, partition specified). Required logs are enumerated. Success/failure thresholds are numeric.
- **Right-sized for LS6.** 128 cores / 256 GB of a Milan node, 8 MPI × 16 threads, 2-hour wall time, 4 SUs total for the pair of jobs — vs ~12 SUs on the Frontera plan. The reduced cost comes from using the node we're billed for fully.

---

## 10. Vmax Realism And The Follow-On Cat-3 Experiment

### 10.1 Honest scope of Vmax = 20 m/s

The entire laptop study — and therefore this first LS6 run — uses Vmax = 20 m/s. On the Saffir-Simpson scale that is **a minimum tropical storm**, not a hurricane:

| Category | Vmax (m/s) | Vmax (mph) |
|---|---:|---:|
| Tropical Depression | < 17 | < 39 |
| **Tropical Storm (where we run)** | **17–32** | 39–73 |
| Cat 1 Hurricane | 33–42 | 74–95 |
| Cat 2 | 43–49 | 96–110 |
| Cat 3 (major) | 50–58 | 111–129 |
| Cat 4 | 58–70 | 130–156 |
| Cat 5 | ≥ 70 | ≥ 157 |

For comparison, production ADCIRC hindcasts of the hurricanes that drove the method's development run at Cat 3–5 intensities — Katrina ~75 m/s, Ike ~50, Harvey ~60, Irma ~80. Vmax = 20 was picked for laptop tractability: at higher Vmax the Newton solver was diverging at timesteps deep in the DA window (some of this is documented in the forensic analyses under `docs/`), and the Jacobians produced by the implicit solve were exercising MUMPS in a regime that overwhelmed the 16 GB memory envelope even before TLM Eq 38 machinery ran.

### 10.2 What that means for interpreting the primary result

The primary run tests a **methodology hypothesis** (does combined TLM-Eq38 + correlated-L_wme make static DC-WME beat 4D-Var on this inlet). It does **not** test whether DC-WME is production-ready for hurricane hindcasts. Any result from the primary run — win or loss — must carry the caveat: *measured in a tropical-storm-forcing regime where storm surge is secondary to tidal/baseline currents*. Generalizing beyond that requires a harder regime.

### 10.3 Follow-on Cat-3 experiment (conditional on primary outcome)

**Fire this follow-on only if the primary produces a DC-WME "win" per § 5.** Running it after a negative-primary result would conflate two unknowns (methodology and forcing regime) and would not be interpretable.

**Config delta from the primary:**
| Parameter | Primary | Follow-on |
|---|---|---|
| `vmax` | 20 m/s (tropical storm) | **55 m/s (Cat 3)** |
| `track_shift_km` | 10 | 10 (keep — structural model-error stays fixed) |
| `nt_ramp` | 24 | **36** (6 h warm-up — let the Cat-3 winds spin up the ocean before DA starts) |
| `nt_da` | 12 | 12 (same 2 h DA window) |
| `dt` | 600 s | 600 s |
| `min_depth` | 5.0 m | **8.0 m** (prevent shallow-cell Newton divergence under Cat-3 wind stress) |
| Newton `max_it` | default | raise by ~50 % (budget for nonlinear stiffness) |
| All other knobs | same | same |
| MPI / threads | 8 × 16 | 8 × 16 |
| Wall time | 2 h | **3 h** (Cat-3 winds increase Newton iterations per timestep) |

**Scientific value:** if DC-WME wins at tropical-storm forcing *and* at Cat-3 forcing, the claim "DC-WME improves storm-surge DA" is meaningful — the method's advantage persists through the regime where the physics actually becomes surge-dominated. If DC-WME wins at tropical storm but loses or ties at Cat 3, we learn that the advantage is forcing-sensitive — a useful but restricted result.

**Resources:** 1 LS6 node, 8×16 topology, ~3 h wall clock ≈ 3 SU.

**Known risk:** the forward solver may still diverge at Cat 3 even with the bumped `min_depth` and Newton budget. If that happens, step down to Cat 1 (Vmax = 38 m/s) before declaring the method forcing-restricted. One intermediate step is sufficient — we are not sweeping.

### 10.4 What this follow-on is NOT

- It is **not** a realistic hurricane hindcast (idealized inlet geometry, no bathymetric complexity, no wave coupling, no tidal constituents beyond the baseline forcing).
- It is **not** a replacement for the Shinnecock experiments once they reach production readiness.
- It **does** establish whether the methodology signal observed at Vmax = 20 survives the regime change into actual hurricane-intensity forcing.

---

## 11. LS6-specific pre-flight summary (TL;DR)

1. Port `la.create_petsc_vector` at `experiments/idealized_inlet_da.py:252,334` and `experiments/twin_experiment.py:469` — one-liner each, pattern in [PORT_010_RESULTS.md](../hpc/lonestar6/parity/PORT_010_RESULTS.md).
2. `rsync -az data/Ideal_Inlet/ ls6:/work/08398/tg876971/ls6/SWEMniCS/data/Ideal_Inlet/` to stage the mesh.
3. Create `$WORK/SWEMniCS/env.ls6.sh` from §3.1.
4. Write `hpc/lonestar6/idealized_inlet/{job_4dvar_sparse.slurm, job_dcwme_prod.slurm}` from §3.2 and §3.3.
5. `sbatch --test-only` both to validate.
6. **np=8 MPI parity dry run** (§8.1) — 10 min in `development` queue. If pass, submit.
7. Submit 4D-Var → wait for first `[iter 0]` → submit DC-WME.
8. Two jobs, ~2 h each, ~4 SU total.


---

## **0.5 MPI Configuration (Hardened for Two-Phase Execution)**

### Phase 1 (Authoritative)
- `mpirun -np 2`
- OMP threads: 64
- Binding:
  ```
  export OMP_NUM_THREADS=64
  export OMP_PROC_BIND=spread
  export OMP_PLACES=cores
  ```
- PETSc:
  ```
  -ksp_type preonly
  -pc_type lu
  -pc_factor_mat_solver_type mumps
  ```

### Phase 2 (Scaled)
- `mpirun -np 8`
- OMP threads: 16
- Binding:
  ```
  export OMP_NUM_THREADS=16
  export OMP_PROC_BIND=spread
  export OMP_PLACES=cores
  ```

### Non-negotiable invariants
- Same seed
- Same mesh partitioning order
- Same solver tolerances
- Same TAO parameters

Deviation from these invalidates comparison.

---

## **0.6 Pre-Flight Parity Check (MANDATORY BEFORE EVERY RUN)**

Run this before submitting ANY production job:

```bash
# Ensure deterministic environment
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1

# 1. MPI collective sanity
mpirun -np 2 python hpc/lonestar6/parity/parity_test_mpi.py

# 2. PETSc direct solve parity
python hpc/lonestar6/parity/parity_test_petsc.py

# 3. dolfinx solve sanity
python hpc/lonestar6/parity/parity_test_dolfinx_solve.py

# 4. Reduced 4D-Var parity (critical)
mpirun -np 2 python hpc/lonestar6/parity/parity_4dvar_reduced.py

# 5. Reduced DC-WME parity (critical)
mpirun -np 2 python hpc/lonestar6/parity/parity_dcwme_reduced.py
```

### Acceptance Criteria (HARD FAIL IF VIOLATED)
- Cost relative diff ≤ 1e-6
- Gradient L2 relative diff ≤ 1e-8
- Gradient cosine similarity ≥ 0.999999

If ANY check fails:
→ DO NOT RUN PRODUCTION  
→ Investigate immediately

---

## **0.7 Submission Guardrail**

Add this check to your SLURM script BEFORE launching:

```bash
echo "Running pre-flight parity checks..."
mpirun -np 2 python hpc/lonestar6/parity/parity_4dvar_reduced.py || exit 1
mpirun -np 2 python hpc/lonestar6/parity/parity_dcwme_reduced.py || exit 1
echo "Parity checks passed."
```

This prevents wasting node-hours on invalid runs.

---

## **0.8 Interpretation Rule**

If Phase 2 (8-rank) deviates from Phase 1:
- Trust Phase 1
- Treat Phase 2 as performance-only
- Investigate scaling-induced numerical drift

---


# SWEMniCS 4D-Var: Detailed Refactoring Specifications

**Companion to Main Refactoring Plan**

This document contains the full technical specifications referenced in the main plan.

## Table of Contents

1. [Architecture & Data Flow](#architecture)
2. [Cost Functions (Complete API)](#cost-functions)
3. [Observation Operators](#observation-operators)
4. [Optimization Layer](#optimization)
5. [Adjoint Machinery](#adjoint)
6. [Parallelization Details](#parallel)
7. [Testing Specifications](#testing)
8. [Migration Checklist](#migration)

---

## 1. Architecture & Data Flow {#architecture}

### Module Structure

\`\`\`
swemnics/
├── forward/ # SWE solvers (existing, refactored)
│ ├── problems.py
│ ├── solvers.py  
│ ├── newton.py
│ └── variational_forms.py # NEW
│
├── data_assimilation/ # DA core (primary refactor target)
│ ├── cost_functions.py
│ ├── observation_operator.py
│ ├── covariance.py # NEW
│ ├── qoi_maps.py # NEW
│ └── metrics.py
│
├── adjoint/ # NEW: Adjoint machinery
│ ├── tangent_linear.py
│ ├── adjoint_operators.py
│ └── checkpointing.py
│
├── optimization/ # NEW: Optimizer layer
│ ├── optimizer_base.py
│ ├── lbfgs.py
│ ├── gauss_newton.py
│ └── petsc_tao_wrapper.py # Optional
│
└── utils/
├── dca_utils.py
├── plotting.py
└── parallel_ops.py # NEW
\`\`\`

### Data Flow Diagram

\`\`\`
Initial Guess m₀
│
▼
┌─────────────┐
│ Forward Solve│──► Trajectory {u*k}
│ M*{k:0} │
└─────────────┘
│
▼
┌─────────────┐
│ Observations │──► Extract H_k(u_k)
│ Operator │
└─────────────┘
│
▼
┌─────────────┐
│ Cost Function│──► J(m), ∇J(m), Hv
│ (4D-Var) │
└─────────────┘
│
▼
┌─────────────┐
│ Adjoint │──► Gradient via λ_k
│ Sweep │
└─────────────┘
│
▼
┌─────────────┐
│ Optimizer │──► Updated m
│ (L-BFGS) │
└─────────────┘
│
└──► Iterate until convergence
\`\`\`

---

## 2. Cost Functions (Complete API) {#cost-functions}

Based on Spence et al. (2025), we implement three variants:

### 2.1 Mathematical Formulation

**Standard 4D-Var:**
\`\`\`
J(m) = ½⟨m - m*b, B⁻¹(m - m_b)⟩ + ½ Σ*{k=0}^N ⟨H_k(u_k) - y_k, R_k⁻¹(H_k(u_k) - y_k)⟩
\`\`\`

**DC-4DVar (Data-Consistent):**
\`\`\`
J_DC(m) = J(m) - ½ Σ_k ⟨Q_k(m) - Q_k(m_b), L_k⁻¹(Q_k(m) - Q_k(m_b))⟩
\`\`\`

**DC-WME (Weighted Mean Error):**
\`\`\`
Q*wme,k(m) = (1/k) Σ*{j=0}^{k-1} (H*j(M*{j:0}(m)) - y_j)

J_WME(m) = ½⟨m - m_b, B⁻¹(m - m_b)⟩ + ½ Σ_k ⟨Q_wme,k(m), R_k⁻¹ Q_wme,k(m)⟩ - ½ Σ_k ⟨Q_wme,k(m) - Q_wme,k(m_b), L_k⁻¹(...)⟩
\`\`\`

### 2.2 Implementation Highlights

Key features of the cost function design:

- **Abstract base class** with value/grad/hessian_vector methods
- **Strategy pattern** for different DA variants
- **PETSc Vec/Mat** for all operations (MPI-safe)
- **Checkpointing** integrated into forward solves
- **Gauss-Newton approximation** for Hessian-vector products

See the main plan document for complete Python implementations.

---

## 3. Observation Operators {#observation-operators}

### 3.1 Interface Requirements

Every observation operator must implement:

1. **apply(u, t)**: Forward operator H_k: V → R^k
2. **apply_adjoint(w, u, t)**: Adjoint H_k^T: R^k → V\*
3. **linearize_apply(v, u, t)**: Jacobian-vector (∂H_k/∂u)·v

### 3.2 Point Observations (Tide Gauges)

**Challenge**: Locate points in distributed mesh  
**Solution**: Use `dolfinx.geometry.BoundingBoxTree`

\`\`\`python
from dolfinx.geometry import BoundingBoxTree, compute_collisions_points

tree = BoundingBoxTree(mesh, mesh.topology.dim)
cell_candidates = compute_collisions_points(tree, gauge_coords)

# Filter to cells owned by this rank

for i, cells in enumerate(cell_candidates.links(i)):
if len(cells) > 0:
local_gauges.append((gauge_coords[i], cells[0]))
\`\`\`

### 3.3 Adjoint Consistency Testing

\`\`\`python
def test_observation_adjoint(obs_op, u, w):
\"\"\"Test: ⟨H(u), w⟩ = ⟨u, H^T(w)⟩\"\"\"
y = obs_op.apply(u, time=0.0)
lhs = np.dot(y, w) # Must use MPI_Allreduce!

    v_adj = obs_op.apply_adjoint(w, u, time=0.0)
    rhs = u.x.petsc_vec.dot(v_adj.x.petsc_vec)

    assert abs(lhs - rhs) < 1e-8

\`\`\`

---

## 4. Optimization Layer {#optimization}

### 4.1 L-BFGS Algorithm

**Two-Loop Recursion** (Liu & Nocedal, 1989):

\`\`\`python
def compute_search_direction(grad, s_history, y_history, rho_history):
q = -grad.copy()
alphas = []

    # Backward loop
    for s, y, rho in reversed(zip(s_history, y_history, rho_history)):
        alpha = rho * s.dot(q)
        q.axpy(-alpha, y)
        alphas.append(alpha)

    alphas.reverse()

    # Scale by H_0 = γI
    if s_history:
        gamma = s_history[-1].dot(y_history[-1]) / y_history[-1].dot(y_history[-1])
        q.scale(gamma)

    # Forward loop
    for (s, y, rho), alpha in zip(zip(s_history, y_history, rho_history), alphas):
        beta = rho * y.dot(q)
        q.axpy(alpha - beta, s)

    return q  # Search direction p = -H_k ∇J

\`\`\`

### 4.2 Line Search (Backtracking Armijo)

**Sufficient Decrease Condition**:
\`\`\`
J(m + αp) ≤ J(m) + c₁·α·⟨∇J(m), p⟩
\`\`\`

where c₁ = 10⁻⁴ (typical value).

### 4.3 Gauss-Newton with CG

For problems where Hessian-vector products are available:

- Solve: ∇²J(m)·p = -∇J(m) using Conjugate Gradient
- Use **shell matrix** in PETSc (matrix-free operator)
- Precondition with Block-Jacobi or AMG

---

## 5. Adjoint Machinery {#adjoint}

### 5.1 Shallow Water Adjoint Derivation

**Forward SWE (Conservative Form)**:
\`\`\`
∂H/∂t + ∇·(Hu) = 0
∂(Hu)/∂t + ∇·(Hu⊗u + ½g(H²-h_b²)) = F(u,H)
\`\`\`

**Discrete (BDF2 Time-Stepping)**:
\`\`\`
R(u^{n+1}) = (3u^{n+1} - 4u^n + u^{n-1})/(2Δt) + F(u^{n+1}) = 0
\`\`\`

**Adjoint Equation**:
\`\`\`
(∂R/∂u)^T λ^n = source^n

where source^n = (∂H_k/∂u)^T R_k^{-1}(H_k(u^n) - y_k)
\`\`\`

**Transpose Jacobian**:
\`\`\`
(∂R/∂u)^T = (3/(2Δt))M + K^T(u)
\`\`\`

where M is mass matrix and K^T is transpose of advection+diffusion Jacobian.

### 5.2 Tangent Linear Model (TLM)

Linearize around trajectory {ū*k}:
\`\`\`
(∂R/∂u)|*{ū} · δu = 0
\`\`\`

Propagate perturbation δu forward in time.

### 5.3 Checkpointing Strategies

| Strategy  | Memory   | Recomputation      | Best For              |
| --------- | -------- | ------------------ | --------------------- |
| Store All | O(N)     | 0                  | Small N               |
| Recompute | O(1)     | N per adjoint step | Large N, fast forward |
| Binomial  | O(log N) | O(log N) per step  | General case          |

---

## 6. Parallelization Details {#parallel}

### 6.1 MPI Communication Patterns

**Ghost Communication** (scatter_forward/reverse):
\`\`\`python
u.x.scatter_forward() # Update ghost values (halo exchange)
u.x.scatter_reverse(PETSc.ScatterMode.ADD) # Accumulate from ghosts
\`\`\`

**Global Reductions** (VecNorm, VecDot):
\`\`\`python
grad_norm = grad.norm() # Includes MPI_Allreduce internally
inner_prod = v1.dot(v2) # Also collective
\`\`\`

**Avoid Implicit Gathers**:
❌ `grad_array = grad.getArray(); comm.gather(grad_array, root=0)`  
✅ Use PETSc VecScatter for rank-0 I/O only

### 6.2 Weak Scaling Example

| Ranks | DoFs/Rank | Total DoFs | Time (s) | Efficiency |
| ----- | --------- | ---------- | -------- | ---------- |
| 1     | 100k      | 100k       | 120      | 100%       |
| 2     | 100k      | 200k       | 125      | 96%        |
| 4     | 100k      | 400k       | 130      | 92%        |
| 8     | 100k      | 800k       | 140      | 86%        |

Target: >80% efficiency up to 32 cores.

---

## 7. Testing Specifications {#testing}

### 7.1 Unit Test Checklist

For each module:

**Cost Functions:**

- [x] Taylor remainder test: O(ε²) convergence
- [x] Adjoint gradient consistency
- [x] Hessian symmetry & PSD property
- [x] MPI determinism (same result on all ranks)

**Observation Operators:**

- [x] Adjoint test: ⟨Hu, w⟩ = ⟨u, H^Tw⟩
- [x] Linearization finite difference check
- [x] Point location correctness (all gauges found)

**Optimizers:**

- [x] Descent property (J decreases monotonically)
- [x] Convergence on quadratic test problem
- [x] Line search robustness

**Forward Model:**

- [x] Deterministic trajectory (fixed seed)
- [x] TLM consistency with finite differences
- [x] Adjoint consistency

### 7.2 Integration Tests

**End-to-End Scenarios:**

1. Lorenz-63 (3-state ODE) with known solution
2. SWE dam break with synthetic observations
3. Hurricane Ike storm surge (if CI resources allow)

**Parallel Correctness:**

- Run identical problem on 1, 2, 4 ranks → bitwise identical results
- Weak scaling test: time(2N DoFs, 2 ranks) ≈ time(N DoFs, 1 rank)

### 7.3 Continuous Integration

\`\`\`.yaml

# .github/workflows/test.yml

name: Tests

on: [push, pull_request]

jobs:
test-serial:
runs-on: ubuntu-latest
steps: - uses: actions/checkout@v3 - name: Install FEniCSx
run: pip install fenics-dolfinx==0.9.0 petsc4py - name: Run tests
run: pytest tests/ -v -m "not mpi"

test-parallel:
runs-on: ubuntu-latest
steps: - name: Install MPI
run: sudo apt-get install -y mpich - name: Run MPI tests
run: mpirun -np 4 pytest tests/ -v -m "mpi"
\`\`\`

---

## 8. Migration Checklist {#migration}

### Sprint 1 (Weeks 1-2): Foundation

**Week 1:**

- [ ] Day 1-2: Create new module structure, stub classes
- [ ] Day 3-4: Implement `CovarianceMatrix` base + subclasses
- [ ] Day 5: Refactor `ObservationOperator` with MPI-aware point location

**Week 2:**

- [ ] Day 6-7: Implement `FourDVarCost` (value, grad, Hv)
- [ ] Day 8-9: Implement `DCFourDVarCost` with predictability term
- [ ] Day 10: Implement `LBFGSOptimizer` with line search

### Sprint 2 (Weeks 3-4): Adjoint & Validation

**Week 3:**

- [ ] Day 11-12: Hand-derive SWE linearization, implement TLM
- [ ] Day 13-14: Derive transpose operators, implement adjoint sweep
- [ ] Day 15: Implement `TrajectoryCheckpointer` with HDF5

**Week 4:**

- [ ] Day 16-17: Implement `DCWMEFourDVarCost` with WME QoI
- [ ] Day 18: Implement `GaussNewtonOptimizer` with CG+shell matrix
- [ ] Day 19: End-to-end validation on SWE test case
- [ ] Day 20: Documentation, code review, merge

### Acceptance Criteria

Before merging to main:

1. ✅ All tests pass (serial + parallel)
2. ✅ Code coverage > 80%
3. ✅ Documentation complete (API + user guide)
4. ✅ Reproduces at least one result from Spence et al. (2025)
5. ✅ No performance regression vs. baseline

---

## Conclusion

This detailed specification provides all necessary technical details for implementing production-ready 4D-Var in SWEMniCS. Key success factors:

1. **Mathematical rigor**: Every equation traceable to code
2. **MPI safety**: No hidden communication, explicit collectives
3. **Testability**: Comprehensive validation at all levels
4. **Extensibility**: Clean interfaces for future methods

**Next Actions:**

1. Review with team, assign tasks
2. Set up development branch: `refactor/4dvar-parallel`
3. Begin Sprint 1, Day 1
4. Weekly stand-ups during sprints
5. Final code review before merge

Good luck! 🚀

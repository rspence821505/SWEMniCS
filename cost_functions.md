# Section 2: Cost Functions (Complete API) - Detailed Overview

## Mathematical Foundation

The refactoring plan implements **three 4D-Var variants** based on Spence et al. (2025):

### 1. **Standard 4D-Var**

```
J(m) = ½⟨m - m_b, B⁻¹(m - m_b)⟩
     + ½ Σ_{k=0}^N ⟨H_k(u_k) - y_k, R_k⁻¹(H_k(u_k) - y_k)⟩
```

- **Background term**: Penalizes deviation from prior state `m_b`
- **Observation term**: Penalizes mismatch with observations `y_k`
- Uses covariance matrices `B` (background) and `R_k` (observation errors)

### 2. **DC-4DVar (Data-Consistent)**

```
J_DC(m) = J(m) - ½ Σ_k ⟨Q_k(m) - Q_k(m_b), L_k⁻¹(Q_k(m) - Q_k(m_b))⟩
```

- **Adds predictability term**: Reduces weight where model is unpredictable
- `Q_k(m)`: Quantities of Interest (QoI) map at time k
- `L_k`: Predictability covariance matrix

### 3. **DC-WME (Weighted Mean Error)**

```
Q_wme,k(m) = (1/k) Σ_{j=0}^{k-1} (H_j(M_{j:0}(m)) - y_j)

J_WME(m) = ½⟨m - m_b, B⁻¹(m - m_b)⟩
         + ½ Σ_k ⟨Q_wme,k(m), R_k⁻¹ Q_wme,k(m)⟩
         - ½ Σ_k ⟨Q_wme,k(m) - Q_wme,k(m_b), L_k⁻¹(...)⟩
```

- Uses **cumulative mean error** as QoI
- Better for problems with systematic model biases

## Implementation Architecture

### Key Design Features:

1. **Abstract Base Class Pattern**

   - `CostFunctionBase` defines interface: `value()`, `gradient()`, `hessian_vector()`
   - Concrete classes: `FourDVarCost`, `DCFourDVarCost`, `DCWMEFourDVarCost`

2. **PETSc Integration**

   - All vectors/matrices use PETSc for MPI-safe operations
   - No raw NumPy arrays in parallel code
   - Explicit collective operations (`VecNorm`, `VecDot`, `MPI_Allreduce`)

3. **Checkpointing**

   - Integrated trajectory saving for adjoint computation
   - Strategies: Store-all, Recompute, or Binomial checkpointing
   - Memory vs. compute tradeoff

4. **Gauss-Newton Approximation**
   - Hessian-vector products without forming full Hessian
   - Shell matrix operators for iterative solvers
   - Critical for large-scale problems

## API Structure (From Current Codebase)

Looking at the existing `cost_functions.py`, the current implementation has:

### Current Functions:

- **`bayes_cost_function()`**: Standard 4D-Var (J_b + J_o terms)
- **`dci_cost_function()`**: DC-4DVar variant (adds prediction term J_p)
- **Helper functions**: `_background_loss()`, `_observation_loss()`, `_prediction_loss()`
- **`get_trajectory()`**: Forward model integration with state saving

### Refactoring Goals:

The plan calls for transforming these into **class-based OOP design**:

```python
class CostFunctionBase(ABC):
    @abstractmethod
    def value(self, m: PETScVec) -> float:
        """Compute J(m)"""

    @abstractmethod
    def gradient(self, m: PETScVec) -> PETScVec:
        """Compute ∇J(m) via adjoint"""

    @abstractmethod
    def hessian_vector(self, m: PETScVec, v: PETScVec) -> PETScVec:
        """Compute ∇²J(m)·v (Gauss-Newton approx)"""
```

## Implementation Highlights

### 1. **Math-to-Code Transparency**

- Every equation in the docs maps directly to code lines
- LaTeX comments above implementations
- Validation via Taylor remainder tests

### 2. **MPI Safety**

- All reductions explicit (no hidden gathers)
- Ghost communication via `scatter_forward/reverse`
- Deterministic results regardless of core count

### 3. **Testing Requirements**

- **Taylor test**: `|J(m+εv) - J(m) - ε⟨∇J(m),v⟩| = O(ε²)`
- **Adjoint consistency**: `⟨H(u), w⟩ = ⟨u, H^T(w)⟩`
- **Hessian symmetry**: `⟨v, Hw⟩ = ⟨w, Hv⟩`

### 4. **Performance Targets**

- Weak scaling >80% efficiency up to 32 cores
- Memory: O(log N) with binomial checkpointing
- Gradient computation: ~1.5× forward solve cost

## Migration Path

**Sprint 1 (Weeks 1-2)**:

- Days 6-9: Implement `FourDVarCost` and `DCFourDVarCost`
- Includes value, gradient, and Hessian-vector methods

**Validation**:

- Must reproduce results from Spence et al. (2025)
- End-to-end test on SWE dam break problem
- Parallel correctness: identical results on 1, 2, 4, 8 cores

This refactoring transforms the current function-based approach into a robust, extensible framework suitable for production-scale data assimilation while maintaining mathematical rigor and computational efficiency.

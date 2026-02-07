# Resume Prompt: Fixing DG Adjoint Gradient for Twin Experiment Optimization

## Task
Fix the friction sweep comparison study (`experiments/comparison_study/run_comparison.py --experiments friction`) so that the 4D-Var and DC-WME optimizers actually perform iterations and reduce error. Currently all 8 experiments show 0 iterations and 0% error reduction.

## Root Cause Found & Fixed (Partially)
The **identity mass matrix fallback** in the adjoint solver was the root cause. For DG elements, `ForwardModelWrapper.var_form` is `None`, causing `ImplicitAdjointSolver._get_mass_matrix()` (in `src/swe4dvar/adjoint/implicit_adjoint.py:854`) to fall back to an identity matrix. But the forward Jacobian `J = (3/(2dt))*M_DG + F'(u)` contains the real DG mass matrix `M_DG` (diagonal ~8333). This mismatch caused the adjoint to decay by ~3.3e-6 per step, making λ₀ ≈ 1e-13 (effectively zero).

### Fix 1: Provide proper mass matrix (DONE, WORKING)
In `experiments/twin_experiment.py`, `ForwardModelWrapper.get_mass_matrix()` (line ~177) was replaced with UFL-based assembly:
```python
def get_mass_matrix(self) -> PETSc.Mat:
    import os
    os.environ.setdefault("CC", "/usr/bin/clang")
    from ufl import TrialFunction, TestFunction, inner, dx
    from dolfinx import fem
    V = self.solver.V
    u, v = TrialFunction(V), TestFunction(V)
    M = fem.petsc.assemble_matrix(fem.form(inner(u, v) * dx))
    M.assemble()
    self._mass_matrix_cache = M
    return M
```
**IMPORTANT**: Requires `CC=/usr/bin/clang` because conda's clang v18.1.8 has an LTO library naming issue that breaks FFCx JIT compilation.

**Result**: λ₀ is now 3.2e+08 (non-zero!), gradient norm is 4.4e+08. Diagnostic script (`experiments/comparison_study/diagnose_gradient.py`) confirms mass matrix diagonal is ~8333 (correct for DG P1).

### Fix 2: M⁻¹ gradient preconditioning (DONE, NOT YET VERIFIED)
The gradient is now non-zero but **too large** (~4.4e+08) for the optimizer. The TAO L-BFGS line search fails every iteration — cost stays at 1022.47 and gradient at 4.4e+08 with step "N/A" for all iterations. The issue is that the gradient contains the mass matrix M (from ∂R₁/∂u₀ = -M/dt), putting it in the FE dual space, while B⁻¹(m-m_b) uses `DiagonalCovariance` with B=σ²·I (coefficient space).

A `MassMatrixPreconditionedCost` wrapper was added at line ~1372 in `twin_experiment.py`. It solves M·g = ∇J to get the Riesz (L²) gradient. It's inserted in `_setup_cost_function` (line ~1112) before the `ZeroBoundaryGradientCost` wrapper.

**THIS FIX HAS NOT BEEN VERIFIED YET.** The last test run still showed 0 iterations, which means either:
1. The preconditioner isn't reducing the gradient enough
2. There's a sign or scaling issue in the preconditioned gradient
3. The `MassMatrixPreconditionedCost` wrapper isn't being reached (check if `hasattr(forward_model, 'get_mass_matrix')` is True at that point — note `forward_model` is passed as a parameter to `_setup_cost_function`, verify it's the `ForwardModelWrapper` instance)
4. The gradient might need ONLY the adjoint part preconditioned (not the B⁻¹ part), since B⁻¹ is already in coefficient space

## Where You Are Now
1. **Investigate why the preconditioned gradient still causes 0 iterations**. Run the diagnostic or a quick test to check:
   - Is `MassMatrixPreconditionedCost` actually being applied? (check log output for "Applied M^{-1} gradient preconditioning")
   - What is the preconditioned gradient norm? (should be O(1)-O(100), not O(10^8))
   - Does the cost decrease with a manual step in the negative gradient direction?

2. **Key hypothesis**: Preconditioning the ENTIRE gradient by M⁻¹ might over-precondition the background term B⁻¹(m-m_b), since that term is already in coefficient space and doesn't need M⁻¹. Consider preconditioning only the adjoint contribution λ₀, not the full gradient. This could be done by modifying `_compute_initial_gradient()` in `implicit_adjoint.py:671` to NOT multiply by M (just use λ₁ directly instead of M·λ₁).

3. **Alternative approach**: Instead of preconditioning the gradient, modify B to be M-weighted: B = σ²·M⁻¹ so that B⁻¹ = (1/σ²)·M. Then both terms in the gradient contain M and scale consistently. This requires changing how `_create_component_aware_covariance` works.

## Key Files
- `experiments/twin_experiment.py` — ForwardModelWrapper (line 162), MassMatrixPreconditionedCost (line 1372), _setup_cost_function (line 1084)
- `src/swe4dvar/adjoint/implicit_adjoint.py` — _get_mass_matrix (line 854), _compute_initial_gradient (line 671), _assemble_adjoint_forcing (line 772)
- `src/swe4dvar/data_assimilation/cost_functions.py` — FourDVarCost.value_gradient (line 485), _solve_adjoint (line 528)
- `experiments/comparison_study/diagnose_gradient.py` — diagnostic script
- `experiments/comparison_study/run_comparison.py` — friction sweep entry point

## Key Constants (for nx=20, ny=10, dt=1800, nt=96)
- DOFs: 3600 (400 cells × 9 DOFs/cell for mixed DG P1)
- Mass matrix diagonal: ~8333 (for DG P1 on this mesh)
- Background variance: σ² = 0.01 (background_error_std=0.1, component-aware)
- BDF2 coefficients: c_next = 2/dt ≈ 0.00111, c_next_next = -1/(2dt) ≈ -0.000278
- Block size: 1 (mixed element, 9 DOFs per cell: 3 for h, 6 for [ux,uy])

## Run Commands
```bash
# Diagnostic (verifies gradient chain)
CC=/usr/bin/clang python experiments/comparison_study/diagnose_gradient.py

# Single quick test (5 iterations max)
CC=/usr/bin/clang python -c "
import sys; sys.path.insert(0, '.')
from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig
from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils import get_default_solver_params
problem = TidalProblem(nx=20, ny=10, dt=1800, nt=96)
solver = get_solver('DG')(problem, theta=0.5, p_degree=[1, 1])
config = TwinExperimentConfig(method='4dvar', obs_fraction=0.5, obs_frequency=4, obs_noise_level=0.01, background_error_std=0.1, max_iterations=5, gradient_tolerance=1e-6, verbose=True)
exp = TwinExperiment(problem=problem, solver=solver, config=config, solver_params=get_default_solver_params())
results = exp.run()
"

# Full friction sweep
CC=/usr/bin/clang python experiments/comparison_study/run_comparison.py --experiments friction --no-resume
```

## BDF2 Adjoint Amplification Note
The BDF2 adjoint time coupling has an inherent amplification factor of ~4/3 per step when F'≈0 (from the characteristic equation roots). Over 96 steps this gives (4/3)^96 ≈ 10^12 growth. With friction damping (F' contribution), the actual growth is ~10^10 (λ₀ = 3.2e+08 from λ₉₆ = 0.049). This growth is physically correct — it reflects sensitivity of observations to the initial condition accumulated over the assimilation window. The gradient magnitude issue is purely about converting between FE spaces (dual → primal), not about the adjoint being wrong.

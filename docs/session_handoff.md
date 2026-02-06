# Session Handoff: Fixing Twin Experiment DCWME

## Current Task
Fix the DCWME (Decentralized Conditional Weighted Minimum Error) data assimilation method in the twin experiment framework. The 4D-Var method is now working, but DCWME is still failing.

## Immediate Issue to Fix
**Empty array dtype bug in ZeroBoundaryGradientCost**

When `boundary_dofs` is an empty numpy array, it defaults to `float64` dtype, which causes an error when used as an index:

```python
# FAILS:
empty = np.array([])  # defaults to float64
arr[empty] = 0.0  # "arrays used as indices must be of integer (or boolean) type"

# FIX:
empty_int = np.array([], dtype=int)
arr[empty_int] = 0.0  # WORKS
```

**Files to check:**
- `experiments/twin_experiment.py` - search for `ZeroBoundaryGradientCost`
- `experiments/serial_da/da_experiment_utils.py`

The fix: Ensure `boundary_dofs` arrays use `dtype=int` even when empty.

## What's Already Been Fixed
1. **Forward solver stability** - Uses direct LU solver instead of GMRES+ILU
2. **Component-aware background perturbation** - `_get_component_dof_indices()` method handles DG DOF layout
3. **Physical bounds DOF indices** - `_create_physical_bounds()` uses correct component indices
4. **Adjoint solver stability** - `src/swe4dvar/adjoint/implicit_adjoint.py` uses LU solver
5. **TLM solver stability** - `src/swe4dvar/adjoint/tangent_linear.py` uses LU solver

## Test Command
```bash
cd /Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS
python experiments/test_inverse_crime.py
```

Current parameters: TidalProblem, nx=20, ny=10, dt=1800, nt=12 (short window for debugging)

## Key Files
- `experiments/twin_experiment.py` - Main twin experiment framework
- `experiments/test_inverse_crime.py` - Test script
- `src/swe4dvar/adjoint/implicit_adjoint.py` - Adjoint solver
- `src/swe4dvar/adjoint/tangent_linear.py` - TLM solver
- `docs/inverse_crime.md` - Documentation on inverse crime avoidance

## Context
The goal is to test data assimilation methods with physics perturbation (friction scaling) to avoid "inverse crimes" in twin experiments. 4D-Var works; DCWME returns cost=1e20 because the ZeroBoundaryGradientCost wrapper fails on empty boundary_dofs array.

## Prompt for New Session
```
I'm continuing work on fixing the DCWME data assimilation method in the twin experiment framework.

The immediate issue is in ZeroBoundaryGradientCost - when boundary_dofs is an empty numpy array, it defaults to float64 dtype which causes "arrays used as indices must be of integer type" error.

Please:
1. Read experiments/twin_experiment.py and find ZeroBoundaryGradientCost
2. Fix the empty array dtype issue by ensuring boundary_dofs uses dtype=int
3. Run python experiments/test_inverse_crime.py to verify the fix

The 4D-Var method is working. DCWME should work after this fix.
```

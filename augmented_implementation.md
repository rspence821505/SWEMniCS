# Augmented Control Implementation Record

## 1. Overview

This pass implements an augmented-control pathway for joint estimation of the initial hydrodynamic state and a reduced Manning's-\(n\) parameterization. The control is no longer restricted to the initial state alone. The implemented target is

\[
c = \begin{bmatrix} u_0 \\ \theta_n \end{bmatrix},
\]

where \(u_0\) is the initial state vector and \(\theta_n\) is a low-dimensional coefficient vector defining a bounded spatial Manning's-\(n\) field.

The change was made because the earlier system could only optimize over \(u_0\). That was inadequate for experiments in which the dominant uncertainty enters through friction rather than the initial condition. The earlier augmented-control prototype also relied on centered finite differences of the full nonlinear forward model for parameter sensitivities. This pass replaces that as the primary Manning pathway with a solver-facing residual-derivative path.

The main limitations addressed were:

- state-only control in the core 4D-Var/DC-WME stack,
- no first-class Manning's-\(n\) control in the augmented framework,
- parameter sensitivities assembled from repeated whole-model finite differences rather than from the discrete residual,
- and a separation between the newer augmented-control prototype and the registered experiment path.

## 2. Control Formulation

The control moved from a state-only vector to a packed mixed control.

- Previous control: `m = u0`
- Current augmented control: `c = [u0; theta]`
- Manning experiment target: `c = [u0; theta_n]`

This is represented by:

- `src/swe4dvar/control/augmented_control.py`
  - `ControlVector`: logical split representation with `u0` and `theta`
  - `ControlLayout`: authoritative source of block sizes, slices, packing, unpacking, and mixed-gradient assembly

`ControlLayout` prevents scattered implicit slicing. All packed PETSc vectors for augmented runs are created and unpacked through this class. It supports:

- state-only control,
- parameter-only control,
- and combined state-plus-parameter control.

For the Manning experiment, the packed control is serial-only. `ControlLayout` explicitly rejects multi-rank augmented controls with a nonzero parameter block.

## 3. Forward Model Modifications

Manning's \(n\) now enters the forward model through an explicit parameter controller rather than through ad hoc problem mutation in the optimizer loop.

The relevant implementation is in:

- `src/swe4dvar/forward/augmented_control.py`
  - `ParameterController`
  - `ParameterSensitivityProvider`
  - `ManningsBasisController`
  - `AugmentedForwardModelWrapper`

The Manning controller works as follows.

1. A smooth Gaussian basis is built over the scalar mesh coordinates.
2. A reference Manning field is converted to a logit variable.
3. The coefficient vector `theta_n` perturbs that logit field through the basis.
4. A logistic map enforces the physical interval `[n_min, n_max]`.
5. The resulting expression is assigned into the problem through `problem.TAU` and `problem.TAU_const`.

The implemented field is therefore

\[
n(x;\theta_n)
=
n_{\min} + (n_{\max}-n_{\min})
\sigma\!\left(\eta_0(x) + \sum_{j=1}^{p}\theta_j \phi_j(x)\right),
\]

where:

- \(\sigma(\cdot)\) is the logistic map,
- \(\eta_0(x)\) is the logit-transformed reference field,
- and \(\phi_j(x)\) are row-normalized Gaussian basis functions.

This differs from the previous state-only setup in two ways:

- the forward wrapper no longer assumes the control is identical to the initial state,
- and parameter injection is explicit and centralized in the controller rather than hidden in experiment-specific perturbation logic.

The underlying friction dependence is still the existing Manning source term in `src/swe4dvar/forward/problems.py`. The controller changes the coefficient entering that term; it does not rewrite the solver internals.

## 4. Gradient / Adjoint Implementation

### State gradient

The state component continues to use the existing discrete adjoint machinery.

- Forward solves store timestep Jacobians \(\partial R_k/\partial u_k\).
- `ImplicitAdjointSolver` solves the backward transpose systems.
- The state gradient is still

\[
\nabla_{u_0} J = B_{u_0}^{-1}(u_0-u_0^b) + \lambda_0.
\]

This remains the same discrete-adjoint path already used for state-only 4D-Var.

### Manning gradient

The Manning block no longer uses centered finite differences of the full nonlinear forward model as its primary gradient mechanism.

For the implemented Manning basis coefficients, the code now uses:

- exact discrete residual derivatives with respect to the parameter coefficients at each timestep,
- a linear forced tangent model for forward parameter sensitivities,
- and a timestep-adjoint accumulation for \(\nabla_{\theta_n} J\).

The core implementation is in:

- `src/swe4dvar/forward/augmented_control.py`
  - `ManningsBasisController.compute_timestep_parameter_jacobian`
- `src/swe4dvar/adjoint/tangent_linear.py`
  - `ForcedTangentLinearModel`
- `src/swe4dvar/adjoint/implicit_adjoint.py`
  - `return_history=True` support
- `src/swe4dvar/data_assimilation/augmented_cost_functions.py`
  - `_parameter_gradient_from_forcings`

The current implementation is best described as:

- exact discrete derivatives for \(\partial R_k / \partial \theta_n\) of the assembled residual form,
- combined with a discrete adjoint/TLM time propagation,
- but not yet a fully general monolithic augmented PDE adjoint framework for arbitrary parameter classes.

That distinction matters. The Manning path is substantially deeper than the earlier reduced finite-difference prototype, but the code is still specialized to the implemented Manning basis controller and serial augmented runs.

## 5. Parameter Sensitivity Pathway

The new abstraction is `ParameterSensitivityProvider` in `src/swe4dvar/forward/augmented_control.py`.

Its purpose is to make the derivative source explicit. For Manning's \(n\):

- `ManningsBasisController` subclasses `ParameterSensitivityProvider`.
- `compute_timestep_parameter_jacobian(...)` assembles one residual-derivative vector per Manning coefficient at the current timestep.

At the discrete level, the residual is written as

\[
R_k(u_k,u_{k-1},u_{k-2};\theta_n)=0.
\]

The code exposes

\[
G_k := \frac{\partial R_k}{\partial \theta_n},
\]

coefficient-by-coefficient through UFL differentiation of the assembled residual form with respect to the coefficient constants.

The timestep sensitivity propagation then uses the linearized forced solve

\[
J_k \,\delta u_k
=
T_k(\delta u_{k-1},\delta u_{k-2})
-
G_k \,\delta\theta_n,
\]

where:

- \(J_k = \partial R_k / \partial u_k\) is the stored forward Jacobian,
- and \(T_k(\cdot,\cdot)\) is the BDF2 time-coupling contribution.

This is implemented in `ForcedTangentLinearModel`, which extends the original TLM with an explicit timestep forcing term.

For the gradient, the backward sweep accumulates the parameter contribution as

\[
\nabla_{\theta_n} J
=
B_n^{-1}(\theta_n-\theta_n^b)
-
\sum_{k=1}^{N} G_k^\top \lambda_k,
\]

where \(\lambda_k\) is the discrete adjoint state at timestep \(k\). The sign matches the linearized residual convention used in the code.

This is the primary Manning sensitivity pathway. The old whole-model finite-difference sensitivity bundle remains available only as a fallback for parameter controllers that do not implement `ParameterSensitivityProvider`.

## 6. Prior / Covariance Structure

The augmented prior is block diagonal:

\[
B_c =
\begin{bmatrix}
B_{u_0} & 0 \\
0 & B_n
\end{bmatrix}.
\]

This is implemented with:

- `src/swe4dvar/data_assimilation/covariance.py`
  - `BlockDiagonalCovariance`

For the registered `mannings_n` experiment:

- `B_{u_0}` is diagonal, with variance scale set from `background_error_std` times a state magnitude proxy.
- `B_n` is diagonal, with entries given by the configured parameter standard deviations and the experiment regularization weight.

The experiment code constructing this block prior is in:

- `experiments/twin_framework/parameter_runners.py`
  - `ManningsNRunner.solve_inverse`

Regularization and admissibility for Manning's \(n\) are enforced by three layers:

- reduced coefficient space rather than raw nodal inversion,
- bounded logistic parameterization in physical space,
- and diagonal coefficient prior through `B_n`.

Bounds are configured per coefficient in the experiment registry and then applied to the packed control through TAO. The state block is effectively unbounded in the current experiment by using very large numerical bounds, while the Manning block uses the specified coefficient limits.

## 7. Optimization Changes

The optimizer itself was not rewritten into a new algorithm, but it now operates on the packed mixed control.

- Optimizer: `src/swe4dvar/optimization/petsc_tao_wrapper.py`
- Experiment use: `experiments/twin_framework/parameter_runners.py`

For `mannings_n`, the runner constructs:

- a packed background vector,
- packed lower and upper bounds,
- and a block covariance.

It then calls `PETScTAOWrapper` with `tao_type="blmvm"` so the mixed control can be optimized under box constraints.

No separate state-like and parameter-like step rules are implemented inside TAO. The distinction is handled through:

- block covariance scaling,
- parameter bounds,
- and the coefficient-space representation of Manning's \(n\).

That is sufficient for the current mixed-control experiment, but it is not a custom block-preconditioned augmented optimizer.

## 8. Experiment Integration

The main experiment integration point is:

- `experiments/twin_framework/parameter_runners.py`
  - `ManningsNRunner`

The relevant registered experiment is:

- `experiments/twin_framework/registry.py`
  - `EXPERIMENT_REGISTRY["mannings_n"]`

The key change is that `mannings_n` is no longer described as a parameter-only inversion with a separate prototype path. Its inverse solve now uses:

- `ControlLayout` for packed mixed controls,
- `ForwardModelWrapper` backed by `AugmentedForwardModelWrapper`,
- `ManningsBasisController` for friction parameter injection,
- and `create_cost_function(...)` for either standard 4D-Var or DC-WME on the packed control.

State-only and state+Manning runs are distinguished as follows:

- `wse_wind_ramp`: state-only experiment over the initial condition
- `mannings_n`: augmented state-plus-Manning experiment over `[u_0; \theta_n]`

The experiment is still launched through the main registry entrypoint:

```bash
python run_experiment.py --experiment mannings_n
```

Truth generation and synthetic observation generation for `mannings_n` still reuse the base runner's direct forward-simulation helper `_run_parameterized_forward(...)`, but that helper now builds the same `ManningsBasisController` used by the inverse solve and applies the same coefficient-to-field map before running the model. The truth path therefore no longer bypasses the augmented Manning parameterization.

## 9. Validation and Testing

The current validation added or preserved in this implementation includes:

- packed-control round-trip tests for state-only, parameter-only, and mixed controls,
- admissibility test for the bounded exponential Manning map and field evaluation,
- forced-TLM recurrence test for the timestep forcing sign and propagation logic,
- registry dry-run tests verifying that the standardized experiment entrypoints still work.

The test files are:

- `tests/test_augmented_control.py`
- `tests/test_twin_framework_registry.py`
- `tests/test_augmented_gradient_hostile.py`

Smoke validation was also run through:

- `python run_experiment.py --experiment mannings_n --dry-run --tag pass2_smoke`
- `python run_experiment.py --experiment wse_wind_ramp --dry-run --tag pass2_smoke`

What was not added in this pass:

- a full end-to-end Taylor remainder or finite-difference check for the complete augmented Manning cost gradient on a live PDE problem,
- or a completed production Shinnecock solve demonstrating method superiority.

Those omissions should be treated as real remaining validation gaps, not as implicit confirmations.

## 10. Limitations

This is not yet a full general PDE-level augmented adjoint framework.

Precisely:

- For the implemented Manning basis coefficients, the timestep residual derivatives \(\partial R_k/\partial \theta_n\) are assembled exactly from the discrete UFL residual.
- The state adjoint remains the existing discrete adjoint of the stored timestep Jacobians.
- The parameter contribution is accumulated through discrete adjoint states and timestep residual derivatives.

That is stronger than the previous centered full-model finite-difference prototype, but several limitations remain.

- Augmented controls with nonempty parameter blocks are serial-only.
- The solver is not formulated as one monolithic block Newton system over \((u,\theta_n)\).
- The Manning parameterization is reduced-order. It is not an unconstrained distributed nodal inversion.
- The parameter path is specialized to `ManningsBasisController`. Other parameter controllers can still fall back to the older finite-difference sensitivity bundle.
- Truth generation for the `mannings_n` experiment still uses a direct forward helper rather than the packed augmented wrapper, but it now uses the same `ManningsBasisController` and coefficient map as the inverse path.
- No new production experiment result has been generated in this pass proving that DC-WME outperforms classical 4D-Var.

So the correct characterization is:

- exact discrete residual derivatives for the implemented Manning coefficient map,
- integrated into a discrete augmented state-plus-parameter optimization path,
- but not yet a universal, fully general, parallel augmented PDE adjoint framework.

## 11. Files Modified

- `src/swe4dvar/control/augmented_control.py`: introduced the packed control abstractions `ControlVector` and `ControlLayout`.
- `src/swe4dvar/control/__init__.py`: exported the augmented control abstractions.
- `src/swe4dvar/forward/augmented_control.py`: added augmented forward wrapper, parameter-controller interface, Manning basis controller, and parameter-sensitivity provider machinery.
- `src/swe4dvar/forward/__init__.py`: exported the augmented-control forward classes.
- `src/swe4dvar/forward/problems.py`: allowed function-valued Manning friction coefficients to propagate through the forward physics.
- `experiments/twin_experiment.py`: made the main forward wrapper inherit the augmented-control wrapper.
- `src/swe4dvar/forward/solvers/base_solver.py`: exposed saved parameter-derivative storage alongside states and Jacobians.
- `src/swe4dvar/forward/solvers/cg_implicit.py`: stored timestep parameter-derivative vectors during time stepping.
- `src/swe4dvar/utils/solver_storage.py`: added storage and cleanup for residual-derivative vectors \(\partial R_k/\partial \theta_n\).
- `src/swe4dvar/utils/timestep_manager.py`: coordinated saving of parameter-derivative data by timestep.
- `src/swe4dvar/adjoint/tangent_linear.py`: added `ForcedTangentLinearModel` and timestep forcing support.
- `src/swe4dvar/adjoint/implicit_adjoint.py`: added optional adjoint-history return for timestepwise parameter-gradient accumulation.
- `src/swe4dvar/data_assimilation/augmented_cost_functions.py`: assembled mixed gradients for standard 4D-Var and DC-4DVar using the Manning residual-derivative path.
- `src/swe4dvar/data_assimilation/qoi_maps.py`: extended QoI linearizations so DC-WME uses the packed augmented-control sensitivities.
- `src/swe4dvar/data_assimilation/cost_functions.py`: routed augmented controls to the augmented cost-function classes.
- `src/swe4dvar/data_assimilation/covariance.py`: provided block-diagonal covariance support used for mixed controls.
- `src/swe4dvar/data_assimilation/__init__.py`: exported the augmented-control cost and covariance utilities.
- `experiments/twin_framework/parameter_runners.py`: migrated `mannings_n` inverse solves onto the augmented-control path and built block priors, bounds, and diagnostics for mixed controls.
- `experiments/twin_framework/registry.py`: redefined `mannings_n` as an augmented state-plus-Manning experiment and added the needed noise/background settings.
- `run_experiment.py`: kept the registry entrypoint aligned with the augmented Manning experiment interface.
- `tests/test_augmented_control.py`: added unit checks for control packing, Manning field admissibility, and forced-TLM recurrence behavior.
- `tests/test_twin_framework_registry.py`: updated registry expectations and dry-run coverage for the augmented Manning experiment.
- `tests/test_augmented_gradient_hostile.py`: added adversarial checks for parameterization consistency, legacy-path regressions, and timestep accumulation helpers.
- `paper/key_differences.tex`: updated manuscript-side technical description so the documented control structure matches the implemented system.

## 12. Gradient Correction Pass

### 12.1 Summary of failures found

The hostile validation pass exposed two real defects:

- the `mannings_n` truth-generation helper still constructed Manning's \(n\) with the legacy map
  \[
  n(x;\theta_n) = \operatorname{clip}\!\left(n_{\mathrm{ref}}(x)\exp(\Phi(x)\theta_n),\, n_{\min},\, n_{\max}\right),
  \]
  while the augmented inverse path used a different bounded logistic/logit map inside `ManningsBasisController`;
- the legacy exponential construction still existed inline in `experiments/twin_framework/parameter_runners.py`, so the experiment architecture was not actually unified around the augmented controller.

Those failures were structural, not tolerance artifacts. They meant the optimizer and the synthetic truth were living in different coefficient coordinates.

### 12.2 Root causes

The root cause was not the adjoint accumulation itself. The hostile unit checks on the timestep accumulation helpers did not reveal a sign or off-by-one bug in `_parameter_gradient_from_forcings(...)`.

The root cause was a parameterization split:

- `ManningsNRunner._run_parameterized_forward(...)` still generated truth with the legacy inline field construction;
- `ManningsBasisController` implemented a different coefficient-to-field map;
- the documentation had been updated around the newer controller, but the experiment helper still encoded the older geometry.

That is enough to invalidate end-to-end gradient validation even if the residual-level derivative code is locally correct.

### 12.3 Fixes applied

Two code fixes were applied.

1. `experiments/twin_framework/parameter_runners.py`

- `_run_parameterized_forward(...)` was refactored to build the standard augmented assimilation objects,
- obtain the repository's `ManningsBasisController`,
- apply the supplied Manning coefficients through that controller,
- and run the truth/observation forward simulation from the same injected parameter field used by the inverse solve.

2. `src/swe4dvar/forward/augmented_control.py`

- `ManningsBasisController` was changed from the previous bounded logistic/logit map to the bounded exponential map now used consistently throughout the `mannings_n` experiment;
- the bound handling is
  \[
  n(x;\theta_n) = \operatorname{clip}\!\left(n_{\mathrm{ref}}(x)\exp(\Phi(x)\theta_n),\, n_{\min},\, n_{\max}\right),
  \]
  implemented in the forward model with the corresponding UFL expression
  `max_value(n_min, min_value(n_max, reference_field * exp(...)))`;
- `evaluate_field(...)` was updated to return the same clipped exponential field used by the forward solve.

This removes the coordinate mismatch between:

- truth generation,
- forward-model parameter injection,
- diagnostic field evaluation,
- and the residual-derivative path differentiated by the adjoint machinery.

### 12.4 Before vs after behavior

Before the correction pass:

- the hostile suite detected that the truth path and augmented controller produced different Manning fields for the same coefficient vector;
- `parameter_runners.py` still contained the legacy inline exponential field builder;
- the end-to-end experiment could therefore not be treated as a valid test of the augmented gradient.

After the correction pass:

- the legacy inline Manning map was removed from `parameter_runners.py`;
- the hostile consistency test now checks the actual controller implementation and passes;
- the combined validation suite
  `tests/test_augmented_control.py`,
  `tests/test_twin_framework_registry.py`,
  and `tests/test_augmented_gradient_hostile.py`
  passes in this environment, aside from the pre-existing `petsc4py`-dependent skips.

### 12.5 Remaining limitations

This correction pass resolved the exposed parameterization inconsistency, but it did not remove the broader limits of the current implementation.

- The hostile suite in this environment is still import-light; the two PETSc-dependent tests remain skipped because `petsc4py` is not available in the plain local Python environment here.
- The augmented Manning path is still serial-only.
- The bound handling for Manning's \(n\) is piecewise smooth because of clipping. With the current coefficient bounds in the experiment registry, the active test configurations stay inside the admissible interval, so the clipped branches are not expected to dominate, but this is still not a globally smooth unconstrained parameterization.
- No new production Shinnecock solve was run in this correction pass, so this pass establishes internal gradient-path consistency, not empirical superiority of DC-WME.

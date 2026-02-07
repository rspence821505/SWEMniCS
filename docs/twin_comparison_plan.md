# Twin Experiment Comparison: 4D-Var vs DC-WME 4D-Var

## Overview

Create a comprehensive comparison study between standard 4D-Var and DC-WME 4D-Var using twin experiments with the tidal forward model. The study will evaluate both methods across multiple experimental dimensions.

## Problem Configuration

| Parameter | Value |
|-----------|-------|
| Forward Model | TidalProblem |
| Mesh | nx=20, ny=10 (~700 DOFs for DG) |
| Time Step | dt=1800s (30 minutes) |
| Simulation Time | 2 days (172,800s) → 96 time steps |
| Solver | DG (Discontinuous Galerkin) |
| Polynomial Degree | P1 (linear) |
| Bathymetry | h_b=10.0m (default) |
| Friction | quadratic (default) |
| Solution Variable | "h" (water depth) |

## Reproducibility

- **Random seeds**: Use fixed seeds (obs_seed=42, background_seed=123) for reproducibility
- **Fair comparison**: Both 4D-Var and DC-WME use identical observation points, noise, and background perturbation for each parameter value
- **Single run per configuration**: No replicates (sufficient for initial comparison)

## Physics Perturbation (Model Error)

The key advantage of DC-WME over standard 4D-Var is robustness to **model error**. Without model error (inverse crime), both methods perform similarly. To meaningfully compare the methods, we introduce physics perturbation.

### Primary: Friction Perturbation

Friction scaling is the primary model error axis because:
- Simple to implement (just scale `TAU_const`)
- Works reliably with TidalProblem
- Physically meaningful (Manning's n uncertainty is ±20-50% in practice)

```python
friction_scale_factors = [1.0, 1.1, 1.15, 1.2]
# 1.0 = inverse crime baseline (sanity check)
# 1.1-1.2 = 10-20% friction error (realistic)
```

### Secondary (Optional): Bathymetry Perturbation

Bathymetry perturbation can be added after friction experiments succeed:
- More complex for TidalProblem (requires initial condition adjustment for solution_var="h")
- Recommended: additive smooth noise with 0.5-1.0m amplitude
- See [inverse_crime.md](inverse_crime.md) for details

```python
# Only if friction experiments work well
bathymetry_noise_std = [0.0, 0.5, 1.0]  # meters (0.0 = no perturbation)
```

### Experimental Matrix

| Experiment | Friction Scale | Bathymetry Noise | Purpose |
|------------|---------------|------------------|---------|
| Baseline | 1.0 | 0.0 | Inverse crime (sanity check) |
| F1 | 1.1 | 0.0 | 10% friction error |
| F2 | 1.15 | 0.0 | 15% friction error |
| F3 | 1.2 | 0.0 | 20% friction error |
| B1 (optional) | 1.0 | 0.5 | Bathymetry error only |
| FB1 (optional) | 1.1 | 0.5 | Combined errors |

## Prerequisites (codebase modifications before implementing comparison study)

### 1. Integrate DAMetrics into TwinExperiment (Required)

**Issue**: `TwinExperiment` manually computes innovation statistics but doesn't use `DAMetrics` class for RMSE/data misfit.

**Solution**: Modify [twin_experiment.py](experiments/twin_experiment.py) `_evaluate_results()` method to:
```python
from swe4dvar.data_assimilation.metrics import DAMetrics

# In _evaluate_results():
obs_dict = {k: self.observations[i] for i, k in enumerate(obs_times)}
obs_op_dict = {k: self.obs_operator for k in obs_times}
metrics = DAMetrics(obs_dict, obs_op_dict)

rmse_dict = metrics.compute_rmse(self.analysis_trajectory)
results.mean_rmse = np.mean(list(rmse_dict.values()))
results.data_misfit = metrics.compute_data_misfit(self.analysis_trajectory)
```

Also add `mean_rmse` and `data_misfit` fields to `TwinExperimentResults` dataclass.

### 2. Add Checkpointer Parameter to Cost Functions (Required)

**Issue**: `FourDVarCost.__init__()` has no `checkpointer` parameter - can't force different strategies.

**Solution**: Modify [cost_functions.py](src/swe4dvar/data_assimilation/cost_functions.py):
```python
def __init__(self, ..., checkpointer: Optional[CheckpointerBase] = None):
    self.checkpointer = checkpointer
    # If checkpointer provided, use it instead of solver.storage
```

This requires:
1. Import `CheckpointerBase` from `swe4dvar.adjoint.checkpointing`
2. Add `checkpointer` parameter with `None` default (backward compatible)
3. Modify `_run_forward_model()` to use checkpointer for storing/retrieving Jacobians if provided
4. Update adjoint solver call to use checkpointer if available

### 3. Verify DG + TwinExperiment Works (Required)

**Status**: Codebase investigation shows this combination should work, but we need to verify before running the full sweep.

**Action**: Create a minimal test script that:
1. Creates TidalProblem with nx=20, ny=10, dt=1800, nt=10 (short run)
2. Creates DG solver with p_degree=[1, 1]
3. Runs a single TwinExperiment with default config
4. Verifies results are produced without errors

```python
# experiments/comparison_study/verify_setup.py
from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig

problem = TidalProblem(nx=20, ny=10, dt=1800, nt=10)
solver = get_solver("DG")(problem, theta=0.5, p_degree=[1, 1])
config = TwinExperimentConfig(method="4dvar", obs_fraction=0.5)
experiment = TwinExperiment(problem, solver, config)
results = experiment.run()
print(f"Verification passed: error_reduction={results.error_reduction:.1f}%")
```

Run this before proceeding to ensure the setup works.

## File Structure

```
experiments/
  comparison_study/
    __init__.py
    config.py              # Configuration dataclasses
    diagnostics.py         # ExperimentDiagnostics class and analysis utilities
    runner.py              # Main experiment runner with diagnostics integration
    plotting.py            # Visualization utilities (including diagnostic plots)
    run_comparison.py      # Entry point script
```

## Implementation Steps

### Step 1: Create Configuration Module (config.py)

Create dataclasses for experiment configuration:
- `ComparisonStudyConfig`: Base problem setup (nx, ny, dt, final_time, solver_type)
- `SweepConfig`: Define sweep parameters for each experiment type

Key sweep parameters:
- **Friction Scale** (PRIMARY): [1.0, 1.1, 1.15, 1.2] (model error via friction perturbation)
- **Observation Frequency**: [1, 2, 4, 8, 12] (observe every N timesteps)
- **Observation Fraction**: [0.1, 0.25, 0.5, 0.75] (fraction of mesh nodes observed)
- **Observation Noise**: [0.001, 0.01, 0.05, 0.1] (noise as fraction of signal)
- **Background Error**: [0.05, 0.1, 0.2, 0.5] (background_error_std)
- **Checkpointing**: ["full", "state_only", "binomial"]
- **Bathymetry Noise** (OPTIONAL): [0.0, 0.5, 1.0] (meters, additive noise)

Default values (when not being swept):
| Parameter | Default |
|-----------|---------|
| friction_scale_factor | 1.0 (or 1.1 for non-baseline) |
| obs_frequency | 4 |
| obs_fraction | 0.5 |
| obs_noise_level | 0.01 |
| background_error_std | 0.1 |
| bathymetry_noise_std | 0.0 (no perturbation) |

**Note**: For sweeps other than friction_scale, use `friction_scale_factor=1.1` as the default to ensure model error is present. The inverse crime baseline (friction_scale=1.0) should only be used as a sanity check, not as the basis for comparing methods.

### Step 2: Create Experiment Runner (runner.py)

**This is a thin orchestration layer that reuses the existing `TwinExperiment` framework.**

The existing [twin_experiment.py](experiments/twin_experiment.py) already provides:
- `TwinExperimentConfig`: All configuration options (method, obs_fraction, obs_frequency, obs_noise_level, background_error_std, etc.)
- `TwinExperiment`: Full experiment execution (truth generation, observations, DA optimization, evaluation)
- `TwinExperimentResults`: Structured results with cost_history, error metrics, wall_time, etc.

The `runner.py` simply:
1. Loops through parameter sweep values
2. Creates `TwinExperimentConfig` for each (method, parameter_value) combination
3. Instantiates and runs `TwinExperiment` **with try/except to handle failures**
4. Collects `TwinExperimentResults` into a dictionary for plotting
5. Logs failed experiments and continues to the next parameter

**Robust Error Handling with Diagnostics**: Physics perturbation and extreme parameter combinations can cause solver divergence or optimization failures. The runner MUST continue running even when individual experiments fail, AND capture diagnostic data to understand why:

```python
import traceback
from .diagnostics import ExperimentDiagnostics, classify_failure

for friction_scale in friction_scale_factors:
    for method in ["4dvar", "dcwme"]:
        experiment_id = f"{method}_friction_{friction_scale}"
        diagnostics = ExperimentDiagnostics()

        try:
            config = TwinExperimentConfig(
                method=method,
                perturb_friction=True,
                friction_scale_factor=friction_scale,
                ...
            )
            experiment = TwinExperiment(problem, solver, config)

            # Hook cost function to capture gradient/cost history
            diagnostics.attach_to_cost_function(experiment.cost_function)

            result = experiment.run()

            # Capture final cost breakdown
            diagnostics.capture_final_state(experiment.cost_function)

            results[experiment_id] = {
                "status": "success",
                "result": result.to_dict(),
                "diagnostics": diagnostics.to_dict(),
                "friction_scale": friction_scale,
                "method": method,
            }
        except Exception as e:
            print(f"FAILED: {experiment_id}: {e}")
            diagnostics.failure_traceback = traceback.format_exc()
            diagnostics.failure_iteration = len(diagnostics.cost_history)

            results[experiment_id] = {
                "status": "failed",
                "error": str(e),
                "failure_type": classify_failure(diagnostics.to_dict()),
                "diagnostics": diagnostics.to_dict(),
                "friction_scale": friction_scale,
                "method": method,
            }

        # Save after EACH experiment (allows resume if script crashes)
        save_results(results, output_path)
```

**Key principles**:
1. **Never stop the full sweep** - catch all exceptions and continue to the next experiment
2. **Record failure details** - store the error message for debugging
3. **Save incrementally** - write results after each experiment so progress isn't lost
4. **Handle in plotting** - plotting functions skip failed experiments or mark them distinctly

Failed experiments are recorded with `status: "failed"` and plotting functions handle missing data gracefully (skip or mark with special symbols in plots).

**Incremental Saving**: After each experiment completes, results are saved to JSON immediately. This allows resumption if the script crashes partway through. On startup, the runner checks for existing results and skips already-completed experiments.

**Parameter ordering**: Run "easier" cases first (more observations, less noise) to catch configuration issues early before long runs.

**Timeout**: Each experiment has a 30-minute timeout to prevent infinite hangs on pathological cases.

```python
from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig

class ComparisonRunner:
    def run_friction_sweep(self, friction_scales):
        """Primary experiment: vary friction to introduce model error."""
        results = {}
        for friction_scale in friction_scales:
            for method in ["4dvar", "dcwme"]:
                exp_id = f"{method}_friction_{friction_scale}"
                try:
                    config = TwinExperimentConfig(
                        method=method,
                        perturb_friction=True,
                        friction_scale_factor=friction_scale,
                        obs_frequency=4,  # default
                        obs_fraction=0.5,  # default
                    )
                    problem, solver = self._create_fresh_problem_solver()
                    experiment = TwinExperiment(problem, solver, config)
                    result = experiment.run()
                    results[exp_id] = {"status": "success", "result": result.to_dict()}
                except Exception as e:
                    results[exp_id] = {"status": "failed", "error": str(e)}
                self._save_results(results)  # Save after each experiment
        return results

    def run_obs_frequency_sweep(self, frequencies):
        """Secondary experiment: vary observation frequency WITH model error."""
        results = {}
        for freq in frequencies:
            for method in ["4dvar", "dcwme"]:
                exp_id = f"{method}_obsfreq_{freq}"
                try:
                    config = TwinExperimentConfig(
                        method=method,
                        perturb_friction=True,
                        friction_scale_factor=1.1,  # Always include model error!
                        obs_frequency=freq,
                        obs_fraction=0.5,
                    )
                    problem, solver = self._create_fresh_problem_solver()
                    experiment = TwinExperiment(problem, solver, config)
                    result = experiment.run()
                    results[exp_id] = {"status": "success", "result": result.to_dict()}
                except Exception as e:
                    results[exp_id] = {"status": "failed", "error": str(e)}
                self._save_results(results)
        return results
```

Uses existing infrastructure:
- `TidalProblem` from [problems.py](src/swe4dvar/forward/problems.py)
- `get_solver("DG")` from [solvers/__init__.py](src/swe4dvar/forward/solvers/__init__.py)
- `TwinExperiment` and `TwinExperimentConfig` from [twin_experiment.py](experiments/twin_experiment.py)

### Step 3: Implement Checkpointing Study

**Challenge**: With 96 timesteps, the automatic checkpointer factory always selects FullTrajectory (< 500 steps).

**Solution**: Force all 3 strategies explicitly by:
1. Bypass `CheckpointerFactory` and directly instantiate each strategy:
   - `FullTrajectoryCheckpointer(num_steps=96)`
   - `StateOnlyCheckpointer(num_steps=96, forward_model=fm)`
   - `BinomialCheckpointer(num_steps=96, num_checkpoints=10, forward_model=fm)`
2. Create a custom wrapper or subclass that injects the explicit checkpointer (avoid modifying core source files if possible)
3. Run 4D-Var (only) with each strategy to compare performance

**Note**: If injecting checkpointer requires modifying [cost_functions.py](src/swe4dvar/data_assimilation/cost_functions.py), we'll add an optional `checkpointer` parameter to `FourDVarCost.__init__()` with backward compatibility.

Metrics to compare:
- Wall time for full optimization (including adjoint computation)
- Estimated memory usage (computed from DOF count and strategy)
- Analysis error (should be identical for all strategies - validates correctness)

### Step 4: Create Plotting Module (plotting.py)

**Handle missing data**: Plotting functions check for `None` results (failed experiments) and either skip those data points or mark them with a distinct symbol/annotation in the plots.

Generate the following plots for each experiment:

1. **Convergence Comparison**: Cost vs iteration for 4D-Var and DC-WME (side-by-side subplots for each parameter value)

2. **Error Reduction Bar Chart**: Grouped bars showing error reduction % for both methods across parameter values

3. **RMSE Comparison**: Line plots of RMSE vs parameter value for both methods

4. **Data Misfit Comparison**: Line plots of final data misfit vs parameter value

5. **Checkpointing Comparison**: Bar chart of wall time and memory for each strategy

6. **Summary Heatmap**: DC-WME advantage (error reduction difference) across all experiments

### Step 5: Create Main Entry Point (run_comparison.py)

CLI interface with options:
```bash
# Run primary friction sweep (recommended first)
python experiments/comparison_study/run_comparison.py \
    --experiments friction \
    --output-dir outputs/comparison_study

# Run all experiments
python experiments/comparison_study/run_comparison.py \
    --experiments all \
    --output-dir outputs/comparison_study

# Run specific sweeps
python experiments/comparison_study/run_comparison.py \
    --experiments friction,obs_freq,obs_fraction \
    --output-dir outputs/comparison_study
```

Experiment options:
- `friction` - PRIMARY: Friction scale sweep [1.0, 1.1, 1.15, 1.2] (run this first!)
- `obs_freq` - Observation frequency sweep (with friction_scale=1.1)
- `obs_fraction` - Observation fraction sweep (with friction_scale=1.1)
- `noise` - Observation noise sweep (with friction_scale=1.1)
- `background` - Background error sweep (with friction_scale=1.1)
- `bathymetry` - OPTIONAL: Bathymetry noise sweep
- `checkpointing` - Checkpointing strategy comparison
- `all` - Run all experiments

**Execution**: Sequential (one experiment at a time) for simplicity and easier debugging.

**Progress Logging**: Print progress after each experiment:
```
[1/8] friction sweep: scale=1.0, method=4dvar - COMPLETED (error_reduction=82.3%, 15 iters, 32.1s)
[2/8] friction sweep: scale=1.0, method=dcwme - COMPLETED (error_reduction=81.9%, 12 iters, 45.2s)
[3/8] friction sweep: scale=1.1, method=4dvar - COMPLETED (error_reduction=45.2%, 23 iters, 45.3s)
[4/8] friction sweep: scale=1.1, method=dcwme - COMPLETED (error_reduction=52.1%, 18 iters, 51.2s)
[5/8] friction sweep: scale=1.15, method=4dvar - COMPLETED (error_reduction=28.4%, 35 iters, 67.8s)
[6/8] friction sweep: scale=1.15, method=dcwme - COMPLETED (error_reduction=41.2%, 22 iters, 55.4s)
[7/8] friction sweep: scale=1.2, method=4dvar - FAILED (Newton solver diverged after 8 iterations)
[8/8] friction sweep: scale=1.2, method=dcwme - COMPLETED (error_reduction=32.5%, 28 iters, 62.1s)
```

Note: The example above illustrates the expected pattern where DC-WME maintains better error reduction as model error increases, while 4D-Var degrades more rapidly and may fail at higher friction perturbations.

## Metrics Computed

For each experiment, compute and store:

| Metric | Source | Notes |
|--------|--------|-------|
| RMSE | `DAMetrics.compute_rmse()` | Per-timestep RMSE of analysis trajectory vs observations |
| Data Misfit | `DAMetrics.compute_data_misfit()` | Final cost function observation term |
| Background Error | `TwinExperimentResults.background_error` | Already computed by TwinExperiment |
| Analysis Error | `TwinExperimentResults.analysis_error` | Already computed by TwinExperiment |
| Error Reduction | `TwinExperimentResults.error_reduction` | Already computed by TwinExperiment |
| Cost History | `TwinExperimentResults.cost_history` | Already stored |
| Wall Time | `TwinExperimentResults.wall_time` | Already stored |
| Iterations | `TwinExperimentResults.num_iterations` | Already stored |

**Note**: RMSE and Data Misfit require running the forward model with the analysis to compute. We'll extend `TwinExperiment._evaluate_results()` to compute and store these additional metrics in `TwinExperimentResults`.

## Output Structure

```
outputs/comparison_study/
  data/
    friction_sweep.json          # PRIMARY: model error comparison
    obs_frequency_sweep.json     # With friction_scale=1.1
    obs_fraction_sweep.json      # With friction_scale=1.1
    noise_level_sweep.json       # With friction_scale=1.1
    background_error_sweep.json  # With friction_scale=1.1
    bathymetry_sweep.json        # OPTIONAL: secondary model error
    checkpointing_study.json
    summary.csv
  figures/
    # Primary: Model error analysis
    error_reduction_vs_friction.png      # Key plot: how methods degrade with model error
    convergence_friction_sweep.png

    # Secondary: Parameter sweeps (all with model error)
    error_reduction_obs_frequency.png
    error_reduction_obs_fraction.png
    error_reduction_noise.png
    error_reduction_background.png

    # Convergence plots
    convergence_obs_frequency.png
    convergence_obs_fraction.png
    convergence_noise.png
    convergence_background.png

    # Summary
    rmse_comparison.png
    data_misfit_comparison.png
    checkpointing_comparison.png
    summary_heatmap.png
    method_robustness_comparison.png    # DC-WME vs 4D-Var across all experiments
```

## Critical Files to Modify/Create

**Prerequisites (do first):**
| File | Action | Priority |
|------|--------|----------|
| [experiments/twin_experiment.py](experiments/twin_experiment.py) | Add DAMetrics integration, RMSE/data_misfit fields | Required |
| [src/swe4dvar/data_assimilation/cost_functions.py](src/swe4dvar/data_assimilation/cost_functions.py) | Add optional checkpointer parameter | Required |

**New files:**
| File | Action |
|------|--------|
| [experiments/comparison_study/__init__.py](experiments/comparison_study/__init__.py) | Create |
| [experiments/comparison_study/verify_setup.py](experiments/comparison_study/verify_setup.py) | Create (run first to verify DG+TwinExperiment) |
| [experiments/comparison_study/config.py](experiments/comparison_study/config.py) | Create |
| [experiments/comparison_study/diagnostics.py](experiments/comparison_study/diagnostics.py) | Create - ExperimentDiagnostics dataclass, failure classification, report generation |
| [experiments/comparison_study/runner.py](experiments/comparison_study/runner.py) | Create - with diagnostics integration |
| [experiments/comparison_study/plotting.py](experiments/comparison_study/plotting.py) | Create - including gradient analysis plots |
| [experiments/comparison_study/run_comparison.py](experiments/comparison_study/run_comparison.py) | Create |

## Pre-flight Checks (run before full sweep)

Before running the full experiment suite, the runner performs these checks:

1. **Forward model validation**: Run TidalProblem forward for 5 timesteps to verify solver stability with DG
2. **Output directory creation**: Create `outputs/comparison_study/data/` and `outputs/comparison_study/figures/` if they don't exist
3. **Memory estimation**: Log estimated memory for 96 timesteps with full Jacobian storage (~50-100 MB for 700 DOFs)
4. **Dependency check**: Verify matplotlib is available for plotting (warn if not)

## Key Configuration Inherited from TwinExperimentConfig

These defaults are important for experiment success:
- `interior_only=True`: **Critical for DG** - boundary observations cause incorrect gradients with discrete adjoint
- `use_bounds=True` with `h_min=0.01`: Prevents negative water depth during optimization
- `max_iterations=50`: Usually sufficient; increase if experiments don't converge
- `verbose=True`: Enables progress logging

## Diagnostics and Debugging Framework

To understand WHY solvers fail (not just THAT they fail), we implement comprehensive diagnostics that capture solver state before and during failures.

### Diagnostic Data Captured for Each Experiment

Every experiment (success or failure) captures:

```python
@dataclass
class ExperimentDiagnostics:
    """Comprehensive diagnostics for debugging solver behavior."""

    # Optimization trajectory
    cost_history: List[float]              # Cost at each iteration
    gradient_norm_history: List[float]     # ||∇J|| at each iteration
    step_size_history: List[float]         # Step size (if available from TAO)

    # Cost function breakdown (at final iteration)
    background_term: float                 # J_b = ||m - m_b||^2_B
    observation_term: float                # J_o = sum ||H(x) - y||^2_R

    # Forward model health
    min_depth_per_timestep: List[float]    # Track if depth approaches zero
    max_velocity_per_timestep: List[float] # Detect velocity blow-up
    forward_solver_iterations: List[int]   # Newton iterations per timestep

    # Condition indicators (when available)
    estimated_condition_number: Optional[float]  # From final Jacobian
    max_cfl_number: Optional[float]        # Stability indicator

    # Failure information
    failure_iteration: Optional[int]       # Which iteration failed
    failure_traceback: Optional[str]       # Full stack trace on failure
    last_valid_state: Optional[np.ndarray] # State just before failure
```

### Diagnostic Levels

Three diagnostic levels balance detail vs. overhead:

| Level | When to Use | Overhead | Data Captured |
|-------|-------------|----------|---------------|
| `minimal` | Production runs | ~0% | Final cost, error reduction, wall time only |
| `standard` | Normal experiments | ~5% | Cost/gradient history, cost breakdown, failure info |
| `verbose` | Debugging failures | ~20% | Everything including per-timestep forward model health |

Default: `standard` for all runs. Switch to `verbose` when investigating specific failures.

### Diagnosing Solver Failures

When an experiment fails, use this diagnostic workflow:

#### Step 1: Check the Gradient Norm History

```python
# In results JSON for a failed experiment:
{
    "status": "failed",
    "diagnostics": {
        "gradient_norm_history": [1.2e3, 4.5e2, 1.1e8, NaN],  # <- Explosion!
        "failure_iteration": 3,
        ...
    }
}
```

**Pattern: Gradient explosion** → Model error causing unstable adjoint
- Gradient norms should decrease monotonically
- Sudden increase indicates ill-conditioning
- NaN indicates forward model failure

**Action**: Reduce physics perturbation magnitude

#### Step 2: Check Cost Function Breakdown

```python
"diagnostics": {
    "background_term": 0.05,      # Small - initial condition close to truth
    "observation_term": 1e15,     # Huge! - model-data mismatch extreme
}
```

**Pattern: Observation term dominates** → Model error too large to fit observations
- Background term should be ~10-100x smaller than observation term
- If observation term is 1e10+, model produces states far from observations

**Action**: Reduce friction_scale or bathymetry_noise

#### Step 3: Check Forward Model Health

```python
"diagnostics": {
    "min_depth_per_timestep": [9.8, 8.2, 3.1, 0.05, -0.2],  # <- Negative!
    "max_velocity_per_timestep": [0.5, 1.2, 15.3, 1e6],      # <- Blow-up!
    "forward_solver_iterations": [3, 4, 12, 50, 50],         # <- Not converging!
}
```

**Pattern: Depth goes negative** → Physics perturbation too extreme
**Pattern: Velocity blow-up** → Friction too low, system unstable
**Pattern: Newton iterations maxing out** → Nonlinear solver struggling

**Action**: Check perturbation parameters, may need smaller timestep

### Implementing Diagnostics in runner.py

```python
class DiagnosticRunner:
    def __init__(self, diagnostic_level: str = "standard"):
        self.diagnostic_level = diagnostic_level

    def run_with_diagnostics(self, config, problem, solver):
        diagnostics = ExperimentDiagnostics()

        # Hook into cost function to capture per-iteration data
        original_value_gradient = cost_function.value_gradient
        def instrumented_vg(m, g=None):
            cost, grad = original_value_gradient(m, g)
            diagnostics.cost_history.append(cost)
            if grad is not None:
                diagnostics.gradient_norm_history.append(np.linalg.norm(grad.getArray()))
            return cost, grad
        cost_function.value_gradient = instrumented_vg

        try:
            result = experiment.run()
            # Capture final cost breakdown
            diagnostics.background_term = cost_function.get_background_term()
            diagnostics.observation_term = cost_function.get_observation_term()
            return {"status": "success", "result": result, "diagnostics": diagnostics}

        except Exception as e:
            diagnostics.failure_traceback = traceback.format_exc()
            diagnostics.failure_iteration = len(diagnostics.cost_history)
            # Capture last valid state if available
            if hasattr(cost_function, '_last_valid_m'):
                diagnostics.last_valid_state = cost_function._last_valid_m.copy()
            return {"status": "failed", "error": str(e), "diagnostics": diagnostics}
```

### Debugging Guide: Common Failure Patterns

| Symptom | Gradient Pattern | Cost Pattern | Diagnosis | Fix |
|---------|-----------------|--------------|-----------|-----|
| Immediate failure (iter 0) | No gradient computed | Cost = inf | Forward model fails on background | Reduce background_error_std |
| Early explosion (iter 1-5) | Sharp increase then NaN | Cost increases rapidly | Adjoint unstable | Reduce physics perturbation |
| Slow divergence (iter 10+) | Oscillating, increasing | Cost oscillates | Step size too large | Reduce optimizer step size |
| Stagnation | Near-zero gradient | Cost flat (not at minimum) | Local minimum or saddle | Try different initial guess |
| One method fails, other succeeds | Method-specific | Method-specific | Method less robust to this error | Valuable result - record! |

### Failure Threshold Analysis

A key output is identifying the **failure threshold** - the perturbation magnitude at which each method fails:

```python
# Example analysis output
failure_thresholds = {
    "4dvar": {
        "friction_scale": 1.25,  # Fails at 25% perturbation
        "bathymetry_noise_std": 1.2,  # Fails at 1.2m noise
    },
    "dcwme": {
        "friction_scale": 1.40,  # Handles 40% perturbation
        "bathymetry_noise_std": 1.8,  # Handles 1.8m noise
    },
    "interpretation": "DC-WME is ~60% more robust to friction error, ~50% more robust to bathymetry error"
}
```

This analysis should be generated automatically from the sweep results:

```python
def compute_failure_thresholds(results: Dict) -> Dict:
    """Identify the friction_scale at which each method first fails."""
    thresholds = {"4dvar": None, "dcwme": None}
    for method in ["4dvar", "dcwme"]:
        for scale in sorted(friction_scale_factors):
            exp_id = f"{method}_friction_{scale}"
            if results[exp_id]["status"] == "failed":
                thresholds[method] = scale
                break
    return thresholds
```

---

## Common Failure Modes and Handling

| Failure Mode | Likely Cause | Diagnostic Indicator | Handling |
|--------------|--------------|---------------------|----------|
| Forward solver divergence | Extreme background perturbation or physics perturbation | `forward_solver_iterations` maxing out, `min_depth` → 0 | Caught by try/except, logged with diagnostics |
| Optimizer non-convergence | Insufficient iterations or poor conditioning | `gradient_norm` not decreasing | Log final gradient norm, mark as "not converged" |
| Negative water depth | h_min too small or extreme noise | `min_depth_per_timestep` < 0 | Bounds enforce h >= h_min |
| Memory error | Too many Jacobians stored | N/A (system error) | Reduce obs_frequency or use state_only checkpointing |
| Newton solver failure | Large friction perturbation destabilizes dynamics | `forward_solver_iterations` = max, velocity blow-up | Caught by try/except, reduce perturbation magnitude |
| Cost function returns inf | Forward model produces NaN/negative depths | `observation_term` = inf | Cost function returns inf, optimizer rejects step |
| Adjoint solver failure | Jacobian becomes ill-conditioned with perturbation | Gradient explosion in history | Use LU solver (already default), reduce perturbation |

**Physics perturbation-specific failures**:
- **Friction too high** (e.g., scale > 1.3): Over-damped system may not converge or produce unrealistic flows. **Diagnostic**: Low velocities, slow convergence.
- **Friction too low** (e.g., scale < 0.7): Under-damped system may become unstable. **Diagnostic**: Velocity blow-up, gradient explosion.
- **Bathymetry noise too large**: Can create negative depths in shallow areas. **Diagnostic**: `min_depth` approaching zero.

**Recommended approach**: Start with mild perturbations (friction_scale=1.1) and increase gradually. If experiments fail at higher perturbation levels, that's valuable information about method robustness - the failure threshold analysis quantifies this.

## Post-Experiment Analysis

After running experiments, use these analysis steps to understand solver behavior:

### 1. Automatic Diagnostic Report Generation

The runner generates a `diagnostic_report.md` after each sweep:

```python
def generate_diagnostic_report(results: Dict, output_path: Path):
    """Generate human-readable diagnostic report."""

    report = ["# Experiment Diagnostic Report\n"]

    # Summary statistics
    n_success = sum(1 for r in results.values() if r["status"] == "success")
    n_failed = sum(1 for r in results.values() if r["status"] == "failed")
    report.append(f"## Summary\n- Successful: {n_success}\n- Failed: {n_failed}\n")

    # Failure analysis
    if n_failed > 0:
        report.append("## Failed Experiments\n")
        for exp_id, r in results.items():
            if r["status"] == "failed":
                diag = r.get("diagnostics", {})
                report.append(f"### {exp_id}\n")
                report.append(f"- Error: `{r['error']}`\n")
                report.append(f"- Failed at iteration: {diag.get('failure_iteration', 'N/A')}\n")
                report.append(f"- Last gradient norm: {diag.get('gradient_norm_history', [])[-1] if diag.get('gradient_norm_history') else 'N/A'}\n")
                report.append(f"- Cost breakdown: background={diag.get('background_term', 'N/A')}, observation={diag.get('observation_term', 'N/A')}\n")
                if diag.get('failure_traceback'):
                    report.append(f"<details><summary>Full traceback</summary>\n\n```\n{diag['failure_traceback']}\n```\n</details>\n")

    # Method comparison
    report.append("## Method Robustness Comparison\n")
    thresholds = compute_failure_thresholds(results)
    for method, threshold in thresholds.items():
        if threshold:
            report.append(f"- {method.upper()}: Failed at friction_scale={threshold}\n")
        else:
            report.append(f"- {method.upper()}: Completed all experiments\n")

    with open(output_path, "w") as f:
        f.write("\n".join(report))
```

### 2. Gradient Norm Analysis Plot

A key diagnostic plot shows gradient norm trajectory for all experiments:

```python
def plot_gradient_analysis(results: Dict, output_path: Path):
    """Plot gradient norm histories to identify instability patterns."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for method, ax in zip(["4dvar", "dcwme"], axes):
        for scale in friction_scale_factors:
            exp_id = f"{method}_friction_{scale}"
            if results[exp_id]["status"] == "success":
                grad_hist = results[exp_id]["diagnostics"]["gradient_norm_history"]
                ax.semilogy(grad_hist, label=f"scale={scale}")
            else:
                # Mark failed experiments
                grad_hist = results[exp_id]["diagnostics"].get("gradient_norm_history", [])
                if grad_hist:
                    ax.semilogy(grad_hist, '--', alpha=0.5, label=f"scale={scale} (FAILED)")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Gradient Norm (log scale)")
        ax.set_title(f"{method.upper()} Gradient Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.savefig(output_path)
```

### 3. Failure Pattern Classification

Automatically classify failure modes based on diagnostic patterns:

```python
def classify_failure(diagnostics: Dict) -> str:
    """Classify the failure mode based on diagnostic patterns."""

    grad_hist = diagnostics.get("gradient_norm_history", [])
    cost_hist = diagnostics.get("cost_history", [])
    min_depths = diagnostics.get("min_depth_per_timestep", [])

    if not grad_hist:
        return "immediate_failure"  # Failed before any optimization

    if any(np.isnan(g) or np.isinf(g) for g in grad_hist):
        return "gradient_explosion"

    if len(grad_hist) >= 2 and grad_hist[-1] > grad_hist[-2] * 10:
        return "gradient_divergence"

    if min_depths and min(min_depths) < 0:
        return "negative_depth"

    if cost_hist and any(np.isinf(c) for c in cost_hist):
        return "forward_model_failure"

    return "unknown"
```

### 4. Interactive Debugging (for specific failures)

When a specific experiment fails and you need to investigate deeper:

```bash
# Re-run single experiment with verbose diagnostics
python experiments/comparison_study/run_comparison.py \
    --single-experiment "dcwme_friction_1.2" \
    --diagnostic-level verbose \
    --save-intermediate-states

# This produces:
# outputs/debug/dcwme_friction_1.2/
#   iteration_0_state.npy    # State vector at each iteration
#   iteration_1_state.npy
#   ...
#   forward_trajectory.npy   # Full forward model output
#   jacobians/               # Jacobian matrices (if small enough)
#   diagnostic_report.json   # Full diagnostic data
```

---

## Verification Plan

1. **Pre-flight**: Run single forward solve with DG to verify setup
2. **Friction baseline**: Run friction sweep with scale=1.0 and scale=1.1, both methods
3. **Error handling test**: Run with friction_scale=1.3 (expected to fail for at least one method) to verify error handling works
4. **Integration Test**: Run full friction sweep [1.0, 1.1, 1.15, 1.2]
5. **Visual Verification**: Check that error reduction degrades with increasing friction_scale
6. **Method Comparison**: Verify DC-WME degrades more gracefully than 4D-Var as model error increases
7. **Checkpointing Verification**: Confirm all 3 strategies produce identical analysis errors (up to numerical tolerance)

Run the comparison:
```bash
cd /Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS

# Step 1: Verify friction sweep works
python experiments/comparison_study/run_comparison.py --experiments friction

# Step 2: If friction sweep succeeds, run parameter sweeps
python experiments/comparison_study/run_comparison.py --experiments obs_freq,obs_fraction,noise

# Step 3: Full run (if all above succeed)
python experiments/comparison_study/run_comparison.py --experiments all
```

Expected runtime:
- Friction sweep only: ~30-60 minutes (8 experiments)
- Full comparison: ~2-4 hours (96 timesteps × 2 methods × ~50 parameter combinations)

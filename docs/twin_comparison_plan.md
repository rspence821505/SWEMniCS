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

## Inverse Crime Settings

- **No physics perturbation**: `perturb_bathymetry=False`, `perturb_friction=False`
- Same physics model used for truth generation and data assimilation
- This provides a clean baseline comparison between 4D-Var and DC-WME methods
- Future work could add experiments with physics perturbation to test robustness

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
    runner.py              # Main experiment runner
    plotting.py            # Visualization utilities
    run_comparison.py      # Entry point script
```

## Implementation Steps

### Step 1: Create Configuration Module (config.py)

Create dataclasses for experiment configuration:
- `ComparisonStudyConfig`: Base problem setup (nx, ny, dt, final_time, solver_type)
- `SweepConfig`: Define sweep parameters for each experiment type

Key sweep parameters:
- **Observation Frequency**: [1, 2, 4, 8, 12] (observe every N timesteps)
- **Observation Fraction**: [0.1, 0.25, 0.5, 0.75] (fraction of mesh nodes observed)
- **Observation Noise**: [0.001, 0.01, 0.05, 0.1] (noise as fraction of signal)
- **Background Error**: [0.05, 0.1, 0.2, 0.5] (background_error_std)
- **Checkpointing**: ["full", "state_only", "binomial"]

Default values (when not being swept):
| Parameter | Default |
|-----------|---------|
| obs_frequency | 4 |
| obs_fraction | 0.5 |
| obs_noise_level | 0.01 |
| background_error_std | 0.1 |

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

**Error Handling**: Some parameter combinations (e.g., very high noise, very sparse observations, extreme background error) may cause solver divergence or optimization failures. The runner wraps each experiment in try/except:

```python
for freq in frequencies:
    for method in ["4dvar", "dcwme"]:
        try:
            experiment = TwinExperiment(...)
            result = experiment.run()
            results[method].append(result)
        except Exception as e:
            print(f"FAILED: {method} with obs_freq={freq}: {e}")
            results[method].append(None)  # Placeholder for failed experiment
```

Failed experiments are recorded as `None` in results, and plotting functions handle missing data gracefully (skip or mark as failed in plots).

**Incremental Saving**: After each experiment completes, results are saved to JSON immediately. This allows resumption if the script crashes partway through. On startup, the runner checks for existing results and skips already-completed experiments.

**Parameter ordering**: Run "easier" cases first (more observations, less noise) to catch configuration issues early before long runs.

**Timeout**: Each experiment has a 30-minute timeout to prevent infinite hangs on pathological cases.

```python
from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig

class ComparisonRunner:
    def run_obs_frequency_sweep(self, frequencies):
        results = {"4dvar": [], "dcwme": []}
        for freq in frequencies:
            for method in ["4dvar", "dcwme"]:
                config = TwinExperimentConfig(
                    method=method,
                    obs_frequency=freq,
                    obs_fraction=0.5,  # default
                    # ... other defaults
                )
                experiment = TwinExperiment(self.problem, self.solver, config)
                results[method].append(experiment.run())
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
python experiments/comparison_study/run_comparison.py \
    --experiments all \
    --output-dir outputs/comparison_study
```

Experiment options: `obs_freq`, `obs_fraction`, `noise`, `background`, `checkpointing`, `all`

**Execution**: Sequential (one experiment at a time) for simplicity and easier debugging.

**Progress Logging**: Print progress after each experiment:
```
[12/40] obs_freq sweep: freq=4, method=dcwme - COMPLETED (error_reduction=45.2%, 23 iters, 45.3s)
[13/40] obs_freq sweep: freq=8, method=4dvar - FAILED (Newton solver diverged)
```

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
    obs_frequency_sweep.json
    obs_fraction_sweep.json
    noise_level_sweep.json
    background_error_sweep.json
    checkpointing_study.json
    summary.csv
  figures/
    convergence_obs_frequency.png
    convergence_obs_fraction.png
    convergence_noise.png
    convergence_background.png
    error_reduction_obs_frequency.png
    error_reduction_obs_fraction.png
    error_reduction_noise.png
    error_reduction_background.png
    rmse_comparison.png
    data_misfit_comparison.png
    checkpointing_comparison.png
    summary_heatmap.png
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
| [experiments/comparison_study/runner.py](experiments/comparison_study/runner.py) | Create |
| [experiments/comparison_study/plotting.py](experiments/comparison_study/plotting.py) | Create |
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

## Common Failure Modes and Handling

| Failure Mode | Likely Cause | Handling |
|--------------|--------------|----------|
| Forward solver divergence | Extreme background perturbation | Caught by try/except, logged as failed |
| Optimizer non-convergence | Insufficient iterations or poor conditioning | Log final gradient norm, mark as "not converged" |
| Negative water depth | h_min too small or extreme noise | Bounds enforce h >= h_min |
| Memory error | Too many Jacobians stored | Reduce obs_frequency or use state_only checkpointing |

## Verification Plan

1. **Pre-flight**: Run single forward solve with DG to verify setup
2. **Unit Test**: Run single experiment with both methods, verify results are stored correctly
3. **Integration Test**: Run obs_frequency sweep with reduced parameter set (2 values instead of 5)
4. **Visual Verification**: Check generated plots for reasonable trends
5. **Checkpointing Verification**: Confirm all 3 strategies produce identical analysis errors (up to numerical tolerance)

Run the full comparison:
```bash
cd /Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS
python experiments/comparison_study/run_comparison.py --experiments all
```

Expected runtime: ~2-4 hours for all experiments (96 timesteps × 2 methods × ~40 parameter combinations)

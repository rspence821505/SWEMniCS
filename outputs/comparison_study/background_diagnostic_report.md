# Experiment Diagnostic Report

## Summary

- Total experiments: 8
- Successful: 6
- Failed: 2

## Failed Experiments

### dcwme_background_0.2

- **Error**: `error code 91
[0] KSPSolve() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/interface/itfunc.c:1094
[0] KSPSolve_Private() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/interface/itfunc.c:917
[0] KSPSolve_GMRES() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/impls/gmres/gmres.c:228
[0] KSPGMRESCycle() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/impls/gmres/gmres.c:111
[0] KSPSolve has not converged due to Nan or Inf norm`
- **Failure type**: immediate_failure
- **Failed at iteration**: 0
- **Background term**: None
- **Observation term**: None

<details><summary>Full traceback</summary>

```
Traceback (most recent call last):
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/experiments/comparison_study/runner.py", line 202, in _run_single_experiment
    result = experiment.run()
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/experiments/twin_experiment.py", line 372, in run
    cost_function = self._setup_cost_function(
        forward_model, obs_operator, B, R, obs_times
    )
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/experiments/twin_experiment.py", line 1098, in _setup_cost_function
    cost_function = DCWMEFourDVarCost(
        forward_model=forward_model,
    ...<7 lines>...
        comm=self.comm,
    )
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/src/swe4dvar/data_assimilation/cost_functions.py", line 1155, in __init__
    self._wme_cache["Q_wme_mb"] = self._compute_wme(self.m_b)
                                  ~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/src/swe4dvar/data_assimilation/cost_functions.py", line 1287, in _compute_wme
    return self.qoi_map.evaluate(
           ~~~~~~~~~~~~~~~~~~~~~^
        m, k_final, store_jacobians=store_jacobians, obs_times_only=obs_times_only
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/src/swe4dvar/data_assimilation/qoi_maps.py", line 374, in evaluate
    trajectory, _ = self._get_trajectory(
                    ~~~~~~~~~~~~~~~~~~~~^
        m,
        ^^
    ...<2 lines>...
        obs_times=self.obs_times if obs_times_only else None,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/src/swe4dvar/data_assimilation/qoi_maps.py", line 162, in _get_trajectory
    trajectory, jacobians = self.forward_model.solve(m, store_jacobians)
                            ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/experiments/twin_experiment.py", line 230, in solve
    self.solver.time_loop(
    ~~~~~~~~~~~~~~~~~~~~~^
        solver_parameters=self.solver_params,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
        enable_video=False,
        ^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/src/swe4dvar/forward/solvers/cg_implicit.py", line 663, in time_loop
    J = self.solve_timestep(
        solver,
    ...<2 lines>...
        time=(a + 1) * self.problem.dt,
    )
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/src/swe4dvar/forward/solvers/cg_implicit.py", line 277, in solve_timestep
    _, J = solver.solve(
           ~~~~~~~~~~~~^
        self.u, return_jacobian=True, timestep=timestep, time=time
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/src/swe4dvar/forward/newton.py", line 214, in solve
    solver.solve(L, dx.x.petsc_vec)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "petsc4py/PETSc/KSP.pyx", line 1782, in petsc4py.PETSc.KSP.solve
petsc4py.PETSc.Error: error code 91
[0] KSPSolve() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/interface/itfunc.c:1094
[0] KSPSolve_Private() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/interface/itfunc.c:917
[0] KSPSolve_GMRES() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/impls/gmres/gmres.c:228
[0] KSPGMRESCycle() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/impls/gmres/gmres.c:111
[0] KSPSolve has not converged due to Nan or Inf norm

```
</details>


### dcwme_background_0.5

- **Error**: `error code 91
[0] KSPSolve() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/interface/itfunc.c:1094
[0] KSPSolve_Private() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/interface/itfunc.c:917
[0] KSPSolve_GMRES() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/impls/gmres/gmres.c:228
[0] KSPGMRESCycle() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/impls/gmres/gmres.c:111
[0] KSPSolve has not converged due to Nan or Inf norm`
- **Failure type**: immediate_failure
- **Failed at iteration**: 0
- **Background term**: None
- **Observation term**: None

<details><summary>Full traceback</summary>

```
Traceback (most recent call last):
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/experiments/comparison_study/runner.py", line 202, in _run_single_experiment
    result = experiment.run()
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/experiments/twin_experiment.py", line 372, in run
    cost_function = self._setup_cost_function(
        forward_model, obs_operator, B, R, obs_times
    )
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/experiments/twin_experiment.py", line 1098, in _setup_cost_function
    cost_function = DCWMEFourDVarCost(
        forward_model=forward_model,
    ...<7 lines>...
        comm=self.comm,
    )
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/src/swe4dvar/data_assimilation/cost_functions.py", line 1155, in __init__
    self._wme_cache["Q_wme_mb"] = self._compute_wme(self.m_b)
                                  ~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/src/swe4dvar/data_assimilation/cost_functions.py", line 1287, in _compute_wme
    return self.qoi_map.evaluate(
           ~~~~~~~~~~~~~~~~~~~~~^
        m, k_final, store_jacobians=store_jacobians, obs_times_only=obs_times_only
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/src/swe4dvar/data_assimilation/qoi_maps.py", line 374, in evaluate
    trajectory, _ = self._get_trajectory(
                    ~~~~~~~~~~~~~~~~~~~~^
        m,
        ^^
    ...<2 lines>...
        obs_times=self.obs_times if obs_times_only else None,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/src/swe4dvar/data_assimilation/qoi_maps.py", line 162, in _get_trajectory
    trajectory, jacobians = self.forward_model.solve(m, store_jacobians)
                            ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/experiments/twin_experiment.py", line 230, in solve
    self.solver.time_loop(
    ~~~~~~~~~~~~~~~~~~~~~^
        solver_parameters=self.solver_params,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
        enable_video=False,
        ^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/src/swe4dvar/forward/solvers/cg_implicit.py", line 629, in time_loop
    J = self.solve_timestep(
        solver,
    ...<2 lines>...
        time=(a + 1) * self.problem.dt,
    )
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/src/swe4dvar/forward/solvers/cg_implicit.py", line 277, in solve_timestep
    _, J = solver.solve(
           ~~~~~~~~~~~~^
        self.u, return_jacobian=True, timestep=timestep, time=time
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/src/swe4dvar/forward/newton.py", line 214, in solve
    solver.solve(L, dx.x.petsc_vec)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "petsc4py/PETSc/KSP.pyx", line 1782, in petsc4py.PETSc.KSP.solve
petsc4py.PETSc.Error: error code 91
[0] KSPSolve() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/interface/itfunc.c:1094
[0] KSPSolve_Private() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/interface/itfunc.c:917
[0] KSPSolve_GMRES() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/impls/gmres/gmres.c:228
[0] KSPGMRESCycle() at /Users/runner/miniforge3/conda-bld/bld/rattler-build_petsc_1741089177/work/src/ksp/ksp/impls/gmres/gmres.c:111
[0] KSPSolve has not converged due to Nan or Inf norm

```
</details>


## Method Robustness Comparison

- **4DVAR**: Completed all experiments
- **DCWME**: Completed all experiments

**Interpretation**: Both methods completed all experiments.

## Gradient Convergence Summary

| Experiment | Method | Initial Grad | Final Grad | Ratio |
|------------|--------|-------------|------------|-------|
| 4dvar_background_0.05 | 4dvar | 2.16e-05 | 6.75e-06 | 3.12e-01 |
| 4dvar_background_0.1 | 4dvar | 2.17e-05 | 6.82e-06 | 3.14e-01 |
| dcwme_background_0.05 | dcwme | 1.69e-05 | 3.06e-08 | 1.82e-03 |
| dcwme_background_0.1 | dcwme | 1.69e-05 | 2.57e-07 | 1.52e-02 |
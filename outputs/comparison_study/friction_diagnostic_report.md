# Experiment Diagnostic Report

## Summary

- Total experiments: 8
- Successful: 8
- Failed: 0

## Method Robustness Comparison

- **4DVAR**: Completed all experiments
- **DCWME**: Completed all experiments

**Interpretation**: Both methods completed all experiments.

## Gradient Convergence Summary

| Experiment | Method | Initial Grad | Final Grad | Ratio |
|------------|--------|-------------|------------|-------|
| 4dvar_friction_1.0 | 4dvar | 1.05e-03 | 6.78e-06 | 6.46e-03 |
| 4dvar_friction_1.1 | 4dvar | 1.04e-03 | 7.16e-06 | 6.87e-03 |
| 4dvar_friction_1.15 | 4dvar | 1.04e-03 | 9.62e-06 | 9.28e-03 |
| 4dvar_friction_1.2 | 4dvar | 1.03e-03 | 5.94e-06 | 5.76e-03 |
| dcwme_friction_1.0 | dcwme | 6.13e-04 | 9.18e+02 | 1.50e+06 |
| dcwme_friction_1.1 | dcwme | 6.10e-04 | 9.70e+06 | 1.59e+10 |
| dcwme_friction_1.15 | dcwme | 6.09e-04 | 1.93e+05 | 3.17e+08 |
| dcwme_friction_1.2 | dcwme | 6.07e-04 | 1.25e+04 | 2.07e+07 |
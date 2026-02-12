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
| 4dvar_friction_1.0 | 4dvar | 1.68e-02 | 8.95e-06 | 5.32e-04 |
| 4dvar_friction_1.1 | 4dvar | 1.69e-02 | 7.54e-06 | 4.47e-04 |
| 4dvar_friction_1.15 | 4dvar | 1.69e-02 | 6.72e-06 | 3.98e-04 |
| 4dvar_friction_1.2 | 4dvar | 1.69e-02 | 8.06e-06 | 4.77e-04 |
| dcwme_friction_1.0 | dcwme | 6.99e-04 | 8.68e+04 | 1.24e+08 |
| dcwme_friction_1.1 | dcwme | 6.97e-04 | 2.03e+03 | 2.91e+06 |
| dcwme_friction_1.15 | dcwme | 6.96e-04 | 1.67e+07 | 2.40e+10 |
| dcwme_friction_1.2 | dcwme | 6.96e-04 | 1.00e+03 | 1.44e+06 |
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
| 4dvar_noise_0.001 | 4dvar | 1.53e-04 | 1.64e-06 | 1.07e-02 |
| 4dvar_noise_0.01 | 4dvar | 1.34e-05 | 2.05e-07 | 1.53e-02 |
| 4dvar_noise_0.05 | 4dvar | 2.68e-06 | 1.98e-06 | 7.39e-01 |
| 4dvar_noise_0.1 | 4dvar | 1.34e-06 | 9.90e-07 | 7.38e-01 |
| dcwme_noise_0.001 | dcwme | 1.20e-04 | 3.66e-07 | 3.05e-03 |
| dcwme_noise_0.01 | dcwme | 1.14e-05 | 1.03e-07 | 9.01e-03 |
| dcwme_noise_0.05 | dcwme | 2.28e-06 | 5.61e-07 | 2.46e-01 |
| dcwme_noise_0.1 | dcwme | 1.14e-06 | 2.79e-07 | 2.45e-01 |
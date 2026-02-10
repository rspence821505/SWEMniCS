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
| 4dvar_noise_0.001 | 4dvar | 2.17e-04 | 5.19e-06 | 2.39e-02 |
| 4dvar_noise_0.01 | 4dvar | 2.17e-05 | 6.82e-06 | 3.14e-01 |
| dcwme_noise_0.001 | dcwme | 1.69e-04 | 2.27e-02 | 1.34e+02 |
| dcwme_noise_0.01 | dcwme | 1.69e-05 | 2.57e-07 | 1.52e-02 |
| dcwme_noise_0.05 | dcwme | 3.37e-06 | 9.56e-07 | 2.83e-01 |
| dcwme_noise_0.1 | dcwme | 1.69e-06 | 4.76e-07 | 2.82e-01 |
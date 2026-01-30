# Changelog

All notable changes to the SWE4DVar project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-30

### Major Release - Package Finalization

This release marks the first stable version of SWE4DVar, a comprehensive framework for shallow water equations with 4D-Var data assimilation.

### Changed

- **Package Renamed**: Renamed from `swemnics` to `swe4dvar` to better reflect the package's focus on 4D-Var data assimilation
  - All imports updated: `from swe4dvar import ...`
  - Module structure preserved
  - All 14 notebooks updated with new import paths

- **LaTeX Documentation Consolidated**: Merged `opt.tex` and `opt2.tex` into single `dci_adjoint.tex`
  - Unified mathematical notation
  - Complete derivation of DC-WME-4DVar cost function
  - Adjoint equations for implicit time-stepping

### Added

- **Output Directory Infrastructure**: Standardized output management
  - `outputs/logs/` - Simulation and optimization logs
  - `outputs/figures/` - Generated plots and visualizations
  - `outputs/checkpoints/` - State checkpoints for restart
  - `outputs/data/` - Processed data files
  - `.gitkeep` files to preserve structure in version control

- **Serial Data Assimilation Experiments** (`experiments/serial_da/`)
  - Idealized inlet friction estimation
  - Comparison framework: 4D-Var vs DC-WME-4DVar
  - Analysis notebooks with convergence studies

- **Parallel Data Assimilation Experiments** (`experiments/parallel_da/`)
  - MPI-parallel 4D-Var implementation verification
  - Weak and strong scaling studies
  - Load balancing analysis

- **HPC Deployment Roadmaps** (`hpc/frontera/`)
  - Frontera supercomputer configuration
  - Job scripts for idealized_inlet and shinnecock cases
  - Scaling guidelines for production runs

- **Comprehensive Utility Modules** (`swe4dvar.utils`)
  - `output_paths.py` - Centralized output path management
  - `parallel_ops.py` - MPI parallel operations
  - `profiling.py` - Performance profiling tools
  - `petsc_logging.py` - PETSc logging integration
  - `nonblocking_comm.py` - Asynchronous communication
  - `load_balancing_metrics.py` - Parallel load analysis

- **Data Assimilation Enhancements** (`swe4dvar.data_assimilation`)
  - `covariance.py` - Covariance matrix implementations
    - DiagonalCovariance, DenseCovariance, ImplicitCovariance
    - EnsembleCovariance for ensemble-based methods
  - `qoi_maps.py` - QoI maps for DC methods
    - StandardQoI, WeightedMeanErrorQoI
    - Linearized variants for adjoint computation
    - QoICovarianceEstimator

- **Adjoint Module** (`swe4dvar.adjoint`)
  - TangentLinearModel with validation
  - ImplicitAdjointSolver for BDF2 schemes
  - Checkpointing strategies (State, Jacobian, Binomial)

- **Environment Files**
  - Clean `environment.yml` for conda
  - Updated `requirements.txt` with pinned versions

### Improved

- **MPI Parallelization**: Verified 100% feature coverage
  - All solvers (CG, DG, SUPG, DGCG) fully parallel
  - Observation operators support distributed evaluation
  - Cost function gradient computation scales linearly

- **Documentation**
  - Comprehensive README with installation, examples, API overview
  - API reference documentation (`docs/api_reference.md`)
  - Quick start guide (`docs/quickstart.md`)
  - Updated docstrings throughout codebase

- **Code Organization**
  - Solvers refactored into `forward/solvers/` subpackage
  - Clear separation of concerns between modules
  - Consistent naming conventions

### Fixed

- Import paths in all notebooks updated for `swe4dvar`
- Gauss-Newton optimizer filename typo corrected
- Covariance matrix symmetry checks

### Dependencies

- FEniCSx (DOLFINx) 0.9.0
- PETSc with petsc4py
- mpi4py for MPI support
- NumPy, SciPy for numerical operations
- Optional: h5py, adios4dolfinx, pyvista, matplotlib

## [0.1.0] - 2024-06-16

### Initial Development Release

- Basic forward model solvers (CG, DG, SUPG)
- Standard 4D-Var cost function
- L-BFGS optimizer
- Example problems (Tidal, DamBreak)
- Test suite foundation

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2025-01-30 | Package rename, DC-WME-4DVar, HPC support |
| 0.1.0 | 2024-06-16 | Initial release |

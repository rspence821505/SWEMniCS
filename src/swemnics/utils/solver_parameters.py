"""Utilities for generating solver parameters with MPI-aware defaults.

This module provides helper functions for creating solver parameter dictionaries
with appropriate defaults for serial and parallel execution, commonly used in
forward solves and data assimilation workflows.
"""

from typing import Optional, Dict, Any
from mpi4py import MPI


def get_default_solver_params(
    rtol: float = 1e-5,
    atol: float = 1e-6,
    max_it: int = 10,
    relaxation_parameter: float = 1.0,
    comm: Optional[MPI.Comm] = None,
    error_if_not_converged: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Generate solver parameters with MPI-aware linear solver defaults.

    This function creates a parameter dictionary for nonlinear solvers with
    appropriate linear solver settings based on MPI communicator size:

    - **Serial (1 process)**: Uses GMRES + ILU (iterative, memory-efficient)
    - **Parallel (>1 process)**: Uses direct solver (LU) with MUMPS (robust)

    Args:
        rtol: Relative tolerance for Newton convergence
        atol: Absolute tolerance for Newton convergence
        max_it: Maximum Newton iterations
        relaxation_parameter: Relaxation parameter for Newton method (1.0 = full step)
        comm: MPI communicator (defaults to COMM_WORLD)
        error_if_not_converged: Raise error if linear solver doesn't converge
        **kwargs: Additional parameters to override defaults

    Returns:
        Dictionary of solver parameters suitable for CustomNewtonProblem

    Example:
        >>> # Simple usage
        >>> params = get_default_solver_params()
        >>> solver.time_loop(solver_parameters=params, ...)

        >>> # Override specific parameters
        >>> params = get_default_solver_params(
        ...     rtol=1e-6,
        ...     max_it=15,
        ...     ksp_type="cg"  # Override linear solver
        ... )

        >>> # For data assimilation experiments with tighter tolerances
        >>> da_params = get_default_solver_params(
        ...     rtol=1e-7,
        ...     atol=1e-8,
        ...     max_it=20
        ... )
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    # Base parameters (Newton solver)
    params = {
        "rtol": rtol,
        "atol": atol,
        "max_it": max_it,
        "relaxation_parameter": relaxation_parameter,
        "ksp_error_if_not_converged": error_if_not_converged,
    }

    # MPI-aware linear solver defaults
    if comm.size == 1:
        # Serial: iterative solver with ILU preconditioner
        params.update({
            "ksp_type": "gmres",
            "pc_type": "ilu",
        })
    else:
        # Parallel: direct solver with MUMPS
        params.update({
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        })

    # Override with any user-specified parameters
    params.update(kwargs)

    return params


def get_solver_params_preset(
    preset: str = "default",
    comm: Optional[MPI.Comm] = None,
    **kwargs
) -> Dict[str, Any]:
    """Get predefined parameter presets for common use cases.

    Available presets:
    - **"default"**: Balanced accuracy and speed (rtol=1e-5, atol=1e-6, max_it=10)
    - **"accurate"**: High accuracy for verification (rtol=1e-7, atol=1e-8, max_it=20)
    - **"fast"**: Quick testing/prototyping (rtol=1e-4, atol=1e-5, max_it=8)
    - **"robust"**: Robust for difficult problems (rtol=1e-6, max_it=25, relaxation=0.8)
    - **"data_assim"**: Suitable for 4D-Var (rtol=1e-6, atol=1e-7, max_it=15)

    Args:
        preset: Name of the preset configuration
        comm: MPI communicator (defaults to COMM_WORLD)
        **kwargs: Additional parameters to override preset defaults

    Returns:
        Dictionary of solver parameters

    Example:
        >>> # For 4D-Var experiments
        >>> params = get_solver_params_preset("data_assim")
        >>> for iteration in range(num_iterations):
        ...     solver.time_loop(solver_parameters=params, ...)

        >>> # Quick testing
        >>> params = get_solver_params_preset("fast")

    Raises:
        ValueError: If preset name is not recognized
    """
    presets = {
        "default": {
            "rtol": 1e-5,
            "atol": 1e-6,
            "max_it": 10,
            "relaxation_parameter": 1.0,
        },
        "accurate": {
            "rtol": 1e-7,
            "atol": 1e-8,
            "max_it": 20,
            "relaxation_parameter": 1.0,
        },
        "fast": {
            "rtol": 1e-4,
            "atol": 1e-5,
            "max_it": 8,
            "relaxation_parameter": 1.0,
        },
        "robust": {
            "rtol": 1e-6,
            "atol": 1e-7,
            "max_it": 25,
            "relaxation_parameter": 0.8,  # More conservative steps
        },
        "data_assim": {
            "rtol": 1e-6,
            "atol": 1e-7,
            "max_it": 15,
            "relaxation_parameter": 1.0,
        },
    }

    if preset not in presets:
        available = ", ".join(f"'{p}'" for p in presets.keys())
        raise ValueError(
            f"Unknown preset '{preset}'. Available presets: {available}"
        )

    preset_params = presets[preset].copy()
    preset_params.update(kwargs)

    return get_default_solver_params(comm=comm, **preset_params)

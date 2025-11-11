"""Data assimilation module for SWEMniCS.

This module provides tools for 4D-Var data assimilation including:
- Covariance matrices (background, observation, predictability)
- Cost functions (standard 4D-Var, DC-4DVar, DC-WME)
- Observation operators
- Quality of interest (QoI) maps

Submodules
----------
covariance : Covariance matrix implementations
    - DiagonalCovariance
    - DenseCovariance
    - ImplicitCovariance

cost_functions : 4D-Var cost functionals (TODO: to be implemented)
observation_operator : Observation operators (existing, to be refactored)
qoi_maps : QoI maps for DC-4DVar (TODO: to be implemented)
"""

from .covariance import (
    CovarianceMatrix,
    DiagonalCovariance,
    DenseCovariance,
    ImplicitCovariance,
    create_observation_covariance,
    create_background_covariance_from_ensemble,
    check_covariance_symmetry,
    check_inverse_consistency,
)

__all__ = [
    "CovarianceMatrix",
    "DiagonalCovariance",
    "DenseCovariance",
    "ImplicitCovariance",
    "create_observation_covariance",
    "create_background_covariance_from_ensemble",
    "check_covariance_symmetry",
    "check_inverse_consistency",
]

__version__ = "0.1.0"

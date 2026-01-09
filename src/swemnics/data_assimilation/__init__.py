"""Data assimilation module for SWEMniCS.

This module provides tools for 4D-Var data assimilation including:
- Covariance matrices (background, observation, predictability)
- Cost functions (standard 4D-Var, DC-4DVar, DC-WME)
- Observation operators
- Quantity of Interest (QoI) maps for DC methods
- Diagnostic metrics

Submodules
----------
covariance : Covariance matrix implementations
    - CovarianceMatrix (base class)
    - DiagonalCovariance
    - DenseCovariance
    - ImplicitCovariance
    - BlockDiagonalCovariance

cost_functions : 4D-Var cost functionals
    - FourDVarCost (standard 4D-Var)
    - DCFourDVarCost (Data-Consistent 4D-Var)
    - DCWMEFourDVarCost (DC-4DVar with Weighted Mean Error)

observation_operator : Observation operators
    - PointObservationOperator
    - IntegralObservationOperator
    - CompositeObservationOperator

qoi_maps : QoI maps for DC-4DVar
    - StandardQoI
    - WeightedMeanErrorQoI
    - LinearizedStandardQoI
    - LinearizedWMEQoI
    - QoICovarianceEstimator

metrics : Diagnostic metrics
    - DAMetrics
    - CostFunctionHistory

Mathematical Background
-----------------------
The module implements three cost function variants from Spence et al. (2025):

Standard 4D-Var:
    J(m) = ½⟨m - m_b, B⁻¹(m - m_b)⟩
         + ½ Σ_k ⟨H_k(u_k) - y_k, R_k⁻¹(H_k(u_k) - y_k)⟩

DC-4DVar (Data-Consistent):
    J_DC(m) = J(m) - ½ Σ_k ⟨Q_k(m) - Q_k(m_b), L_k⁻¹(Q_k(m) - Q_k(m_b))⟩

DC-WME (Weighted Mean Error):
    Q_wme(m) = (1/√N) Σ_k R_k^{-1/2}(H_k(u_k) - y_k)
    J_WME(m) = ½⟨m - m_b, B⁻¹(m - m_b)⟩ + ½‖Q_wme‖²
             - ½⟨Q_wme(m) - Q_wme(m_b), L_wme⁻¹(...)⟩

Key Features
------------
- MPI-parallel implementation using PETSc
- Jacobian caching for efficient adjoint computation
- Taylor remainder tests for gradient verification
- Gauss-Newton Hessian-vector products
- Predictability assumption checking

References
----------
Spence, R., Butler, T., & Dawson, C. (2025). Variational Data-Consistent
Assimilation. Submitted.

Examples
--------
>>> from swemnics.data_assimilation import (
...     FourDVarCost, DCFourDVarCost, create_cost_function
... )
>>>
>>> # Create standard 4D-Var cost function
>>> cost_fn = FourDVarCost(
...     forward_model=model,
...     observation_operator=obs_op,
...     background_cov=B,
...     observation_cov=R,
...     m_background=m_b,
...     observations=y_obs,
...     obs_times=[10, 20, 30],
... )
>>>
>>> # Evaluate cost and gradient
>>> J = cost_fn.value(m)
>>> grad = cost_fn.gradient(m)
>>>
>>> # Or use factory function
>>> dc_cost = create_cost_function(
...     "dc", forward_model=model, ...
... )
"""

from __future__ import annotations

# Import covariance classes
from .covariance import (
    CovarianceMatrix,
    DiagonalCovariance,
    DenseCovariance,
    ImplicitCovariance,
    EnsembleCovariance,
    create_observation_covariance,
    create_background_covariance_from_ensemble,
    check_covariance_symmetry,
    check_inverse_consistency,
)

# Import cost functions
from .cost_functions import (
    CostFunction,
    CostFunctionResult,
    FourDVarCost,
    DCFourDVarCost,
    DCWMEFourDVarCost,
    create_cost_function,
)

# Import QoI maps
from .qoi_maps import (
    QoIMap,
    LinearizedQoI,
    StandardQoI,
    LinearizedStandardQoI,
    WeightedMeanErrorQoI,
    LinearizedWMEQoI,
    QoICovarianceEstimator,
)

# Import observation operators (when available)
try:
    from .observation_operator import (
        ObservationOperator,
        PointObservationOperator,
        IntegralObservationOperator,
        CompositeObservationOperator,
    )
except ImportError:
    ObservationOperator = None
    PointObservationOperator = None
    IntegralObservationOperator = None
    CompositeObservationOperator = None

# Import metrics (when available)
try:
    from .metrics import (
        DAMetrics,
        CostFunctionHistory,
    )
except ImportError:
    DAMetrics = None
    CostFunctionHistory = None

__all__ = [
    # Covariance utilities
    "CovarianceMatrix",
    "DiagonalCovariance",
    "DenseCovariance",
    "ImplicitCovariance",
    "create_observation_covariance",
    "create_background_covariance_from_ensemble",
    "check_covariance_symmetry",
    "check_inverse_consistency",
    # Cost functions
    "CostFunction",
    "CostFunctionResult",
    "FourDVarCost",
    "DCFourDVarCost",
    "DCWMEFourDVarCost",
    "create_cost_function",
    # QoI maps
    "QoIMap",
    "LinearizedQoI",
    "StandardQoI",
    "LinearizedStandardQoI",
    "WeightedMeanErrorQoI",
    "LinearizedWMEQoI",
    "QoICovarianceEstimator",
    "EnsembleCovariance",
    # Observation operators
    "ObservationOperator",
    "PointObservationOperator",
    "IntegralObservationOperator",
    "CompositeObservationOperator",
    # Metrics
    "DAMetrics",
    "CostFunctionHistory",
]

__version__ = "0.2.0"

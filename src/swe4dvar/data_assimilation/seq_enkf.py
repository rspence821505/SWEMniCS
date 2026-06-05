"""
Sequential stochastic Ensemble Kalman Filter — second baseline for QPCA-EnDCF.

Implements the textbook sequential (per-observation-time) stochastic
EnKF with perturbed observations, adapted from the Lorenz-96 reference
implementation ``QPCA-EnDCF-Paper/src/filters/seq_enkf.py``.

What this is and isn't
----------------------
- This is a *sequential* stochastic EnKF: the caller forecasts the
  ensemble to one observation time, calls :meth:`update` once with the
  ensemble at that time and the per-time observation, then forecasts the
  analysis ensemble forward to the next observation time before calling
  :meth:`update` again. There is no 4D stacking.
- Gain is the standard R-stabilized Kalman form
  ``K = P_xy (P_yy + R)^{-1}`` at each observation time, applied to the
  same ensemble that produced ``HX``.
- Each ensemble member sees an independent perturbed observation
  ``y + ε^(j)`` with ``ε^(j) ~ N(0, R)``.
- Optional Gaspari-Cohn localization of ``P_xy`` (Schur product on the
  cross-covariance) is supported with the same sparse ``(n, m)`` matrix
  interface as :class:`QPCAEnDCF` and :class:`EnKF4D`. The taper here is
  the *per-observation-time* taper (shape ``(n, m)``), not the stacked
  ``(n, mL)`` taper used by the 4D filters; the
  ``build_spatial_taper`` helper in ``qpca_endcf.py`` already returns
  this shape and can be reused directly.

MPI
---
The filter operates purely on numpy arrays. The same seed-coherent
sampling rule as :class:`EnKF4D` applies: every MPI rank seeds its
local RNG with the same value at construction so each rank generates
the *same* perturbation matrix ``ε``. Because the gain inversion is a
function of the global per-rank-collected ``P_yy`` (already broadcast
through ``apply_H``), every rank applies the same gain to its local
state-anomaly rows. The per-rank fold used for ensemble *generation*
(see :func:`experiments.idealized_inlet_qpca_endcf._draw_ensemble`)
operates on the background-perturbation RNG, which is separate from
the filter's perturbed-observation RNG.

Comparison to the 4D filters
----------------------------
The 4D filters in this package (:class:`QPCAEnDCF`, :class:`EnKF4D`)
take a length-``L`` list ``X_path`` of forecast ensembles, stack the
``L`` observations into a ``mL``-vector, and perform a single
analysis update at window end. The sequential EnKF instead performs
``L`` smaller updates, each based on the ensemble at one observation
time and the observation at that time. Operationally this requires the
experiment driver to interleave forecasts and analyses within each
window; see :func:`experiments.idealized_inlet_qpca_endcf` for the
corresponding intra-window cycling loop.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.linalg import cholesky, solve


class SeqEnKF:
    """
    Sequential stochastic Ensemble Kalman Filter.

    Parameters
    ----------
    apply_H : callable
        ``apply_H(X_ens)`` where ``X_ens`` is ``(n, N)`` and the return
        value is ``(m, N)`` — identical contract to :class:`QPCAEnDCF`
        and :class:`EnKF4D`, but with the per-observation-time ``m``
        rather than the stacked ``mL``.
    R : ndarray, shape ``(m, m)``
        Per-observation-time observation-error covariance.
    seed : int, optional
        RNG seed for perturbed-observation sampling. The **same seed
        must be used on every MPI rank** so that all ranks generate
        identical perturbations and compute the same gain. Default 0.

    Notes
    -----
    There is no truncation rank and no window length here — both
    concepts are 4D-filter specific. The filter knows only how to
    handle a single observation time at a time. The caller is
    responsible for the intra-window forecast/analyze interleaving.
    """

    def __init__(
        self,
        apply_H: Callable[[np.ndarray], np.ndarray],
        R: np.ndarray,
        seed: int = 0,
    ):
        R = np.asarray(R, dtype=float)
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise ValueError(f"R must be square (got shape {R.shape})")

        self.apply_H = apply_H
        self.R = R
        self.m = R.shape[0]

        # Cholesky factor of R (with mild jitter) used both to
        # generate perturbed observations and to back the gain
        # stabilization. ε = L_R · ξ with ξ ~ N(0, I).
        jitter = 1e-10
        R_reg = R + jitter * np.eye(self.m)
        self.R_chol = cholesky(R_reg)
        self.rng = np.random.default_rng(int(seed))

    def update(
        self,
        X: np.ndarray,
        z: np.ndarray,
        HX: Optional[np.ndarray] = None,
        rho=None,
    ) -> np.ndarray:
        """
        Single sequential analysis update at one observation time.

        Parameters
        ----------
        X : ndarray, shape ``(n, N)``
            Forecast ensemble at the current observation time. ``n``
            is the per-rank local state dimension.
        z : ndarray, shape ``(m,)``
            Observation vector at this time.
        HX : ndarray, shape ``(m, N)``, optional
            Pre-computed ``H @ X``. Performance shortcut for callers
            that already have the projection from an earlier
            diagnostic step; otherwise ``apply_H`` is invoked.
        rho : scipy.sparse matrix or None, optional
            Optional Gaspari-Cohn localization taper of shape
            ``(n, m)``. When supplied, the empirical cross-covariance
            is Schur-multiplied entrywise by ``rho`` before being
            applied as the gain. Only nonzero entries of ``rho`` are
            formed.

        Returns
        -------
        X_a : ndarray, shape ``(n, N)``
            Analysis ensemble at this observation time.
        """
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.size != self.m:
            raise ValueError(
                f"z has size {z.size}, expected m={self.m}"
            )
        n, N = X.shape
        if N < 2:
            raise ValueError(
                f"Need at least 2 ensemble members for anomalies (got N={N})"
            )

        if HX is None:
            HX = np.asarray(self.apply_H(X), dtype=float)
        else:
            HX = np.asarray(HX, dtype=float)
        if HX.shape != (self.m, N):
            raise ValueError(
                f"HX has shape {HX.shape}, expected ({self.m}, {N})"
            )

        # Ensemble anomalies. We never form the dense n×n state
        # covariance — the analysis is built from reassociated
        # anomaly products to avoid the n² memory cost that destroys
        # the paper's reference implementation on PDE state.
        A_x = X - X.mean(axis=1, keepdims=True)            # (n, N)
        A_y = HX - HX.mean(axis=1, keepdims=True)           # (m, N)

        P_yy = (A_y @ A_y.T) / (N - 1)
        S = P_yy + self.R
        jitter = 1e-10
        S_chol = cholesky(S + jitter * np.eye(self.m))

        # Perturbed observations (paper Alg. 1, sequential variant).
        # ε^(j) ~ N(0, R), one per member. Identical on every MPI
        # rank by construction.
        xi = self.rng.standard_normal((self.m, N))
        eps = self.R_chol @ xi                              # (m, N)
        innov = z[:, None] + eps - HX                       # (m, N)

        # Solve S · B = innov for B = S^{-1} innov.
        B = solve(S_chol.T, solve(S_chol, innov))            # (m, N)

        if rho is None:
            # Reassociated standard sequential EnKF update:
            #   ΔX = K · innov = P_xy · S^{-1} · innov
            #      = A_x · (A_y^T · B) / (N - 1).
            # Avoids forming the (n, m) dense gain matrix.
            inner = (A_y.T @ B) / (N - 1)                   # (N, N)
            return X + A_x @ inner

        # --- Localized update: ΔX = (ρ ⊙ P_xy) · B with sparse ρ ---
        import scipy.sparse as _sp
        if not _sp.issparse(rho):
            raise TypeError(
                f"rho must be a scipy.sparse matrix, got {type(rho).__name__}"
            )
        if rho.shape != (n, self.m):
            raise ValueError(
                f"rho has shape {rho.shape}, expected ({n}, {self.m})"
            )

        rho_coo = rho.tocoo()
        rows = rho_coo.row
        cols = rho_coo.col
        vals = rho_coo.data
        # P_xy[i, j] = (A_x[i, :] · A_y[j, :]) / (N-1); evaluate only
        # at the nonzero entries of rho.
        inner_products = np.sum(
            A_x[rows, :] * A_y[cols, :], axis=1
        ) / (N - 1)
        localized_data = vals * inner_products
        P_xy_loc = _sp.csr_matrix(
            (localized_data, (rows, cols)),
            shape=rho.shape,
        )
        return X + (P_xy_loc @ B)

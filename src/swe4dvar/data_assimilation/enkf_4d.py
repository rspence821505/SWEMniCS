"""
Stochastic 4D Ensemble Kalman Filter — fair baseline for QPCA-EnDCF.

Implements paper Algorithm 2 (Four-Dimensional Stochastic Ensemble Kalman
Filter) with the same observation-callable / sparse-localization
interfaces as ``QPCAEnDCF``, so the experiment driver can swap one for
the other transparently.

What this is and isn't
----------------------
- This is a *stochastic* EnKF: each ensemble member sees a different
  perturbed observation ``y + ε^(j)`` with ``ε^(j) ~ N(0, R^(L))``.
- Gain is the standard R-stabilized Kalman form ``K = P_xz (P_zz + R)^{-1}``,
  not the pseudoinverse used by QPCA-EnDCF.
- No whitening, no spectral truncation — that is the QPCA-EnDCF
  contribution. If the experiment uses ``EnKF4D``, the κ parameter is
  ignored.
- Optional Gaspari-Cohn localization of ``P_xz`` (Schur-product on the
  cross-covariance) is supported with exactly the same ``(n, mL)`` sparse
  matrix interface as ``QPCAEnDCF.update(..., rho=...)``. The localization
  helper ``build_spatial_taper`` in ``qpca_endcf.py`` produces a taper
  that is consumable by both filters.

MPI
---
The filter operates purely on numpy arrays. Under MPI the experiment
script holds ``X_path`` as a list of ``(n_local, N)`` arrays per rank;
``apply_H`` already returns the same ``(m, N)`` on every rank (the
``PointObservationOperator`` Allgathers internally). All rank-0
operations in this filter (eigh-equivalent linear solves, perturbed-obs
sampling) are computed identically on every rank because every rank
seeds the RNG with the same value at construction — so each rank
generates the same ``ε`` matrix and every rank's gain inversion sees the
same global ``P_zz``. The only state-space step,
``A_x_end @ (A_y.T @ B) / (N-1)`` or ``(rho ⊙ P_xz) @ B``, naturally
operates on each rank's local rows.

References
----------
- Paper Algorithm 2 (``QPCA-EnDCF-Paper/main.tex`` §B.2).
- ``QPCAEnDCF.update`` localized-update path — the sparse Schur product
  is identical there.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from numpy.linalg import cholesky, solve

from .qpca_endcf import _block_diag_repeat


class EnKF4D:
    """
    Stochastic 4D Ensemble Kalman Filter with optional GC localization.

    Parameters
    ----------
    apply_H : callable
        ``apply_H(X_ens)`` where ``X_ens`` is ``(n, N)`` and the return
        value is ``(m, N)`` — identical contract to ``QPCAEnDCF``.
    R : ndarray, shape ``(m, m)``
        Per-observation-time observation-error covariance.
    window_len : int
        ``L`` — number of observation times in the window.
    seed : int, optional
        RNG seed for perturbed-observation sampling. The **same seed must
        be used on every MPI rank** so that all ranks generate identical
        perturbations and compute the same gain. Default 0.

    Notes
    -----
    The κ parameter from QPCA-EnDCF is not present here — this filter
    has no spectral truncation. Localization is provided externally
    through the optional ``rho`` argument to ``update()``.
    """

    def __init__(
        self,
        apply_H: Callable[[np.ndarray], np.ndarray],
        R: np.ndarray,
        window_len: int,
        seed: int = 0,
    ):
        if window_len <= 0:
            raise ValueError(f"window_len must be positive (got {window_len})")
        R = np.asarray(R, dtype=float)
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise ValueError(f"R must be square (got shape {R.shape})")

        self.apply_H = apply_H
        self.R = R
        self.window_len = int(window_len)
        self.m = R.shape[0]

        # Block-diagonal stacked R^(L).
        self.R_block = _block_diag_repeat(R, window_len)
        jitter = 1e-10
        R_block_reg = self.R_block + jitter * np.eye(self.R_block.shape[0])
        # Cholesky for both perturbed-observation sampling
        #   (ε = L · ξ with ξ ~ N(0, I))
        # and for the (P_yy + R) inverse.
        self.R_block_chol = cholesky(R_block_reg)
        self.rng = np.random.default_rng(int(seed))

    def update(
        self,
        X_path: List[np.ndarray],
        z_stack: np.ndarray,
        HX_blocks: Optional[List[np.ndarray]] = None,
        rho=None,
    ) -> np.ndarray:
        """
        4D-EnKF analysis update with optional GC localization.

        Parameters
        ----------
        X_path : list of ndarray
            Length-L list of state ensembles at each observation time,
            each shape ``(n, N)``.
        z_stack : ndarray, shape ``(mL,)``
            Stacked observation vector.
        HX_blocks : list of ndarray, optional
            Pre-computed ``H @ X_path[t]``. Same shortcut as
            ``QPCAEnDCF.update`` — avoids re-applying ``H`` when the
            caller already has it.
        rho : scipy.sparse matrix or None, optional
            ``(n, mL)`` Gaspari-Cohn taper. When supplied, the empirical
            cross-covariance ``P_xz`` is Schur-multiplied entrywise by
            ``rho`` before being applied as the gain. Only nonzero
            entries of ``rho`` are formed — ``O(nnz(rho))`` cost, not
            ``O(n · mL)``.

        Returns
        -------
        X_a_end : ndarray, shape ``(n, N)``
            Analysis ensemble at window end.
        """
        L = self.window_len
        if len(X_path) != L:
            raise ValueError(
                f"X_path has length {len(X_path)} but window_len={L}"
            )
        z_stack = np.asarray(z_stack, dtype=float).reshape(-1)
        if z_stack.size != self.m * L:
            raise ValueError(
                f"z_stack has size {z_stack.size}, expected m*L={self.m * L}"
            )

        n, N = X_path[-1].shape
        if N < 2:
            raise ValueError(
                f"Need at least 2 ensemble members for anomalies (got N={N})"
            )

        # --- Stacked forward in obs space ---
        if HX_blocks is None:
            HX_blocks = [
                np.asarray(self.apply_H(X_path[t]), dtype=float)
                for t in range(L)
            ]
        else:
            if len(HX_blocks) != L:
                raise ValueError(
                    f"HX_blocks has length {len(HX_blocks)} but window_len={L}"
                )
            HX_blocks = [np.asarray(b, dtype=float) for b in HX_blocks]
        for t, b in enumerate(HX_blocks):
            if b.shape != (self.m, N):
                raise ValueError(
                    f"HX_blocks[{t}] has shape {b.shape}, "
                    f"expected ({self.m}, {N})"
                )
        HX_stack = np.vstack(HX_blocks)

        # --- Anomalies ---
        X_end = X_path[-1]
        A_x_end = X_end - X_end.mean(axis=1, keepdims=True)         # (n, N)
        A_y = HX_stack - HX_stack.mean(axis=1, keepdims=True)        # (mL, N)

        # --- Innovation covariance with R stabilization ---
        # S = P_yy + R^(L) = (A_y A_y^T)/(N-1) + R^(L). SPD by construction.
        P_yy = (A_y @ A_y.T) / (N - 1)
        S = P_yy + self.R_block
        jitter = 1e-10
        S_chol = cholesky(S + jitter * np.eye(S.shape[0]))

        # --- Perturbed observations (paper Alg. 2 line 9 ε_w^(j) ~ N(0, R^(L)))
        # ε = L_R · ξ, ξ ~ N(0, I). Shape (mL, N) — one perturbation per
        # ensemble member. Identical on every MPI rank by construction
        # (same RNG seed at __init__).
        xi = self.rng.standard_normal((self.R_block.shape[0], N))
        eps = self.R_block_chol @ xi                                # (mL, N)
        innov = z_stack[:, None] + eps - HX_stack                    # (mL, N)

        # --- Solve S · B = innov for B = S^{-1} innov ---
        B = solve(S_chol.T, solve(S_chol, innov))                    # (mL, N)

        # --- Apply gain ---
        if rho is None:
            # Reassociated standard 4D-EnKF update:
            #   ΔX = K · innov = P_xz · S^{-1} · innov
            #      = A_x_end · (A_y^T · B) / (N - 1).
            # Avoids forming the (n, mL) dense gain matrix.
            inner = (A_y.T @ B) / (N - 1)                            # (N, N)
            return X_end + A_x_end @ inner

        # --- Localized update: ΔX = (ρ ⊙ P_xz) · B with sparse ρ ---
        import scipy.sparse as _sp
        if not _sp.issparse(rho):
            raise TypeError(
                f"rho must be a scipy.sparse matrix, got {type(rho).__name__}"
            )
        if rho.shape != (n, A_y.shape[0]):
            raise ValueError(
                f"rho has shape {rho.shape}, expected ({n}, {A_y.shape[0]})"
            )

        rho_coo = rho.tocoo()
        rows = rho_coo.row
        cols = rho_coo.col
        vals = rho_coo.data
        # P_xz[i, j] = (A_x_end[i, :] · A_y[j, :]) / (N-1); evaluate only
        # at the nonzero entries of rho.
        inner_products = np.sum(
            A_x_end[rows, :] * A_y[cols, :], axis=1
        ) / (N - 1)
        localized_data = vals * inner_products
        P_xz_loc = _sp.csr_matrix(
            (localized_data, (rows, cols)),
            shape=rho.shape,
        )
        return X_end + (P_xz_loc @ B)

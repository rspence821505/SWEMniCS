"""
4D Local Ensemble Transform Kalman Filter (LETKF).

Implements the deterministic ensemble-square-root filter of Hunt,
Kostelich, and Szunyogh (2007), in its four-dimensional form: a single
analysis update at window end that consumes observations from all
``L`` observation times stacked into one ``mL``-dimensional vector.
The analysis is computed in ``(N, N)`` ensemble space via an
eigendecomposition, then mapped back to state space through the
forecast anomaly basis. There is no perturbed-observation step
(distinguishing LETKF from :class:`EnKF4D` and :class:`SeqEnKF`) and
no spectral truncation (distinguishing it from :class:`QPCAEnDCF`).

What this filter is and isn't
-----------------------------
- **Deterministic square-root**: each ensemble member is a linear
  combination of forecast anomalies; the transform matrix is the
  symmetric positive square root of the analysis covariance in
  ensemble space, which preserves the analysis-ensemble mean exactly.
- **R-localization** is supported through the same sparse ``(n, mL)``
  Gaspari-Cohn taper interface used by :class:`QPCAEnDCF` and
  :class:`EnKF4D`. When ``rho`` is supplied, the analysis is performed
  per state degree of freedom using only the observations whose taper
  weights against that DOF are non-zero. This is the operationally
  standard form of LETKF in the literature (state-DOF-local
  observation patches).
- **Global LETKF** (``rho=None``) applies the same ensemble-space
  transform to every state DOF — appropriate when localization is not
  needed or when the global ensemble subspace is already
  well-sampled.

MPI
---
The filter operates purely on numpy arrays. Under MPI the experiment
script holds ``X_path`` as a list of ``(n_local, N)`` arrays per rank;
``apply_H`` already returns the same ``(m, N)`` matrix on every rank
(the :class:`PointObservationOperator` Allgathers internally). The
ensemble-space eigendecomposition is therefore computed identically on
every rank from the globally-shared observation-space anomalies, and
the resulting ``(N, N)`` transform matrix is applied locally to each
rank's state anomalies.

References
----------
Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007).
*Efficient data assimilation for spatiotemporal chaos: A local
ensemble transform Kalman filter.*  Physica D 230, 112–126.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from numpy.linalg import eigh

from .qpca_endcf import _block_diag_repeat


class LETKF:
    """
    4D Local Ensemble Transform Kalman Filter.

    Parameters
    ----------
    apply_H : callable
        ``apply_H(X_ens)`` where ``X_ens`` is ``(n, N)`` and the return
        value is ``(m, N)`` — identical contract to :class:`QPCAEnDCF`
        and :class:`EnKF4D`.
    R : ndarray, shape ``(m, m)``
        Per-observation-time observation-error covariance. For the
        operating point of this codebase ``R = sigma_obs**2 * I``,
        i.e. diagonal; the local-LETKF path exploits this by only
        consulting the diagonal entries.
    window_len : int
        ``L`` — number of observation times in the window.
    seed : int, optional
        Accepted for API parity with :class:`EnKF4D`; not used because
        LETKF is fully deterministic.

    Notes
    -----
    The ``seed`` argument is unused because there is no perturbed-
    observation sampling. It is kept in the constructor signature so
    callers can swap between :class:`QPCAEnDCF`, :class:`EnKF4D`, and
    :class:`LETKF` without changing keyword arguments.
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
        # Stacked R^(L) and its diagonal inverse used by the global
        # LETKF path. The diagonal-only inverse is exact when R is
        # itself diagonal (the operating-point case) and a
        # conservative approximation otherwise.
        self.R_block = _block_diag_repeat(R, window_len)
        self._R_block_diag_inv = 1.0 / np.maximum(
            np.diag(self.R_block), 1e-30
        )
        # Unused; recorded so the API matches EnKF4D.
        self.seed = int(seed)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def update(
        self,
        X_path: List[np.ndarray],
        z_stack: np.ndarray,
        HX_blocks: Optional[List[np.ndarray]] = None,
        rho=None,
    ) -> np.ndarray:
        """
        4D LETKF analysis update.

        Parameters
        ----------
        X_path : list of ndarray
            Length-``L`` list of state ensembles at each observation
            time within the window, each shape ``(n, N)``.
        z_stack : ndarray, shape ``(mL,)``
            Stacked observation vector.
        HX_blocks : list of ndarray, optional
            Pre-computed ``H @ X_path[t]``. Same performance shortcut
            as the other 4D filters in this module; if omitted,
            ``apply_H`` is called once per observation time.
        rho : scipy.sparse matrix or None, optional
            ``(n, mL)`` Gaspari-Cohn taper. When supplied, the
            analysis is computed per state DOF using only the
            observations whose taper weight at that DOF is non-zero,
            with the observation-error inverse multiplied by the
            taper weight (standard R-localization). When ``None``,
            the global LETKF update is applied uniformly to every
            state DOF.

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
                f"z_stack has size {z_stack.size}, expected m*L="
                f"{self.m * L}"
            )

        n, N = X_path[-1].shape
        if N < 2:
            raise ValueError(
                f"Need at least 2 ensemble members for anomalies (got N={N})"
            )

        # Stacked HX in observation space.
        if HX_blocks is None:
            HX_blocks = [
                np.asarray(self.apply_H(X_path[t]), dtype=float)
                for t in range(L)
            ]
        else:
            if len(HX_blocks) != L:
                raise ValueError(
                    f"HX_blocks has length {len(HX_blocks)} but "
                    f"window_len={L}"
                )
            HX_blocks = [np.asarray(b, dtype=float) for b in HX_blocks]
        for t, b in enumerate(HX_blocks):
            if b.shape != (self.m, N):
                raise ValueError(
                    f"HX_blocks[{t}] has shape {b.shape}, "
                    f"expected ({self.m}, {N})"
                )
        HX_stack = np.vstack(HX_blocks)                              # (mL, N)

        # Anomalies + innovation vector.
        X_end = X_path[-1]
        x_b = X_end.mean(axis=1)                                     # (n,)
        y_b = HX_stack.mean(axis=1)                                  # (mL,)
        X_anom = X_end - x_b[:, None]                                # (n, N)
        Y_anom = HX_stack - y_b[:, None]                             # (mL, N)
        innov = z_stack - y_b                                        # (mL,)

        if rho is None:
            return self._global_update(x_b, X_anom, Y_anom, innov, N)
        return self._local_update(x_b, X_anom, Y_anom, innov, N, rho)

    # -----------------------------------------------------------------
    # Internal: global LETKF (no R-localization)
    # -----------------------------------------------------------------

    def _global_update(self, x_b, X_anom, Y_anom, innov, N) -> np.ndarray:
        """LETKF analysis with a single ensemble-space transform
        applied uniformly to every state DOF.
        """
        Nm1 = N - 1
        # C = Y^T R^{-1}. R^{-1} is diagonal in our setting, so the
        # contraction reduces to a column-wise rescale.
        C = Y_anom.T * self._R_block_diag_inv[None, :]               # (N, mL)
        # A = (N-1) I + C Y. Symmetric, positive-definite for any
        # non-trivial ensemble.
        A = Nm1 * np.eye(N) + C @ Y_anom                             # (N, N)
        # Symmetric eigendecomposition. The constant vector is an
        # eigenvector of A with eigenvalue (N-1) because Y_anom @ 1
        # vanishes; this is what guarantees the analysis-ensemble
        # mean is preserved by the symmetric positive square root.
        w, V = eigh(0.5 * (A + A.T))                                 # ensure sym
        w = np.maximum(w, 1e-12)

        # Mean weight vector. P_tilde @ v = V @ ((V^T v) / w).
        Cd = C @ innov                                               # (N,)
        w_bar = V @ ((V.T @ Cd) / w)                                 # (N,)
        # Symmetric square root of (N-1) P_tilde.
        sqrt_diag = np.sqrt(Nm1 / w)                                 # (N,)
        W_sqrt = V * sqrt_diag[None, :] @ V.T                        # (N, N)

        # Final transform: column j is w_bar + W_sqrt[:, j].
        W_all = w_bar[:, None] + W_sqrt                              # (N, N)
        return x_b[:, None] + X_anom @ W_all                         # (n, N)

    # -----------------------------------------------------------------
    # Internal: per-state-DOF LETKF with R-localization
    # -----------------------------------------------------------------

    def _local_update(self, x_b, X_anom, Y_anom, innov, N, rho) -> np.ndarray:
        """LETKF analysis with state-DOF-local observation patches.

        For each state DOF ``i``, the analysis uses only the
        observations whose Gaspari-Cohn taper weight against ``i`` is
        non-zero, with the observation-error inverse rescaled by the
        taper weight. This produces a different ``(N, N)`` transform
        matrix at every state DOF.
        """
        import scipy.sparse as _sp
        if not _sp.issparse(rho):
            raise TypeError(
                f"rho must be a scipy.sparse matrix, got "
                f"{type(rho).__name__}"
            )
        n = X_anom.shape[0]
        if rho.shape != (n, self.m * self.window_len):
            raise ValueError(
                f"rho has shape {rho.shape}, expected "
                f"({n}, {self.m * self.window_len})"
            )

        Nm1 = N - 1
        Imat = np.eye(N)
        rho_csr = rho.tocsr()
        indptr = rho_csr.indptr
        indices = rho_csr.indices
        data = rho_csr.data
        R_inv_diag = self._R_block_diag_inv

        X_a = np.empty_like(X_anom.shape and X_anom)
        X_a = X_anom.copy()  # buffer of correct shape and dtype
        # Reset to fill from the analysis below.
        for i in range(n):
            start, end = indptr[i], indptr[i + 1]
            if start == end:
                # No observations affect this state DOF — keep the
                # forecast (analysis = ensemble mean + anomalies).
                X_a[i, :] = x_b[i] + X_anom[i, :]
                continue
            cols = indices[start:end]
            weights = data[start:end]                                # (m_loc,)

            Y_loc = Y_anom[cols, :]                                   # (m_loc, N)
            d_loc = innov[cols]                                       # (m_loc,)
            R_inv_loc = weights * R_inv_diag[cols]                    # (m_loc,)

            # C_loc = Y_loc^T diag(R_inv_loc)
            C_loc = Y_loc.T * R_inv_loc[None, :]                      # (N, m_loc)
            A_loc = Nm1 * Imat + C_loc @ Y_loc                        # (N, N)
            w_loc, V_loc = eigh(0.5 * (A_loc + A_loc.T))
            w_loc = np.maximum(w_loc, 1e-12)

            Cd_loc = C_loc @ d_loc                                    # (N,)
            w_bar = V_loc @ ((V_loc.T @ Cd_loc) / w_loc)              # (N,)
            sqrt_diag = np.sqrt(Nm1 / w_loc)
            W_sqrt = V_loc * sqrt_diag[None, :] @ V_loc.T             # (N, N)
            W_all = w_bar[:, None] + W_sqrt                           # (N, N)
            X_a[i, :] = x_b[i] + X_anom[i, :] @ W_all
        return X_a

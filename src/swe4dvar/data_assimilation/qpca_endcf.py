"""
QPCA Ensemble Data Consistency Filter (4D variant) — SWE PDE adaptation.

Faithful port of the algorithm from
``QPCA-EnDCF-Paper/src/filters/qpca_endcf.py``. The mathematical update is
unchanged; the only adaptation is the observation interface:

  * The original took a dense observation matrix ``H`` of shape ``(m, n)``
    and computed ``HX_stack`` via matrix multiplication.
  * Here we take a callable ``apply_H(X_ens) -> HX`` so we can plug in the
    existing FEniCSx/PETSc ``PointObservationOperator`` that maps state
    vectors (mixed (h, u, v) DOFs on an unstructured SWE mesh) to point
    observations. The QPCA-EnDCF analysis remains numpy-only at the
    observation/ensemble level.

Algorithm (preserved exactly from the paper code):

  1. Stack per-time forecasts in observation space:
        HX_stack = vstack([H X_t for t in range(L)])  — shape (mL, N).
  2. Whitened residuals in stacked data space:
        E = R_block^{-1/2} (HX_stack - z_stack 1^T).
  3. Centered ensemble covariance C = Ec Ec^T / (N-1) in whitened space.
  4. Top-k eigenvectors V_k from eigh(C).
  5. QPCA map (WME sign): Q_qpca = -V_k V_k^T E.
  6. EnDCF gain from ensemble cross-/auto-covariances at window end:
        P_xy = A_x_end @ A_y^T / (N-1),
        P_yy = A_y    @ A_y^T / (N-1),
        S    = P_yy   (optionally stabilized vs R_block),
        K    = P_xy S^{-1}.
  7. De-whitened correction applied to window-end ensemble:
        X_a_end = X_path[-1] + K (R_block^{1/2} Q_qpca).
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
from numpy.linalg import cholesky, eigh, solve


# ---------------------------------------------------------------------------
# Linear-algebra helpers — copies of QPCA-EnDCF-Paper/src/utils/linalg.py.
# Kept local so this module does not depend on the external paper repo.
# ---------------------------------------------------------------------------


def _cov_and_anoms(X: np.ndarray):
    """Return (cov, centered anomalies) for ensemble X of shape (d, N)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    C = (Xc @ Xc.T) / (X.shape[1] - 1)
    return C, Xc


def _sym_posdef_inverse(A: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """Invert a symmetric positive-definite matrix via Cholesky with fallback."""
    A_sym = 0.5 * (A + A.T) + jitter * np.eye(A.shape[0])
    try:
        L = cholesky(A_sym)
        return solve(L.T, solve(L, np.eye(A.shape[0])))
    except np.linalg.LinAlgError:
        w, V = eigh(A_sym)
        scale = float(np.abs(w).max()) if w.size else 0.0
        adaptive_jitter = max(jitter, 1e-4 * scale, 1e-8)
        w_reg = np.maximum(w, adaptive_jitter)
        return V @ np.diag(1.0 / w_reg) @ V.T


def _stabilize_spd_like(S: np.ndarray, R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Ensure lambda_min(S) >= max(eps, 1% of lambda_max(R))."""
    wS, _ = eigh(S)
    wR, _ = eigh(R)
    min_eig = max(eps, float(wR.max()) * 1e-2)
    if float(wS.min()) < min_eig:
        delta = (min_eig - float(wS.min())) + eps
        S = S + delta * np.eye(S.shape[0])
    return S


def _block_diag_repeat(A: np.ndarray, k: int) -> np.ndarray:
    """Block-diagonal kron(I_k, A)."""
    return np.kron(np.eye(k), A)


def gaspari_cohn(r: np.ndarray) -> np.ndarray:
    """Gaspari-Cohn 5th-order compactly-supported correlation function.

    Evaluated at ``r = distance / c`` where ``c`` is the half-cutoff (so the
    function has support ``r ∈ [0, 2]``, i.e. distance ∈ ``[0, 2c]``).

      C(r) = 1 − 5/3·r² + 5/8·r³ + 1/2·r⁴ − 1/4·r⁵           for 0 ≤ r ≤ 1
           = 4 − 5·r + 5/3·r² + 5/8·r³ − 1/2·r⁴ + 1/12·r⁵
             − 2/(3·r)                                        for 1 ≤ r ≤ 2
           = 0                                                for r > 2

    See Gaspari & Cohn 1999. Returns 0 outside the support; clamped to [0, 1].
    """
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    near = r < 1.0
    mid = (r >= 1.0) & (r < 2.0)

    rn = r[near]
    out[near] = (1.0
                 - (5.0 / 3.0) * rn ** 2
                 + (5.0 / 8.0) * rn ** 3
                 + 0.5 * rn ** 4
                 - 0.25 * rn ** 5)

    rm = r[mid]
    out[mid] = (4.0
                - 5.0 * rm
                + (5.0 / 3.0) * rm ** 2
                + (5.0 / 8.0) * rm ** 3
                - 0.5 * rm ** 4
                + (1.0 / 12.0) * rm ** 5
                - (2.0 / 3.0) / rm)
    return np.clip(out, 0.0, 1.0)


def build_spatial_taper(
    state_coords: np.ndarray,
    obs_coords: np.ndarray,
    L_loc: float,
):
    """Build a sparse ``(n_local, m)`` Gaspari-Cohn taper.

    Parameters
    ----------
    state_coords : ndarray, shape ``(n_local, 2)``
        (x, y) location of every locally-owned state DOF (one entry per DOF).
    obs_coords : ndarray, shape ``(m, 2)``
        (x, y) location of each observation point.
    L_loc : float
        Localization cutoff distance in the same units as ``state_coords`` and
        ``obs_coords``. The GC taper has support ``[0, L_loc]``; obs farther
        than ``L_loc`` from a state DOF contribute zero.

    Returns
    -------
    scipy.sparse.csr_matrix, shape ``(n_local, m)``
        Sparse taper with values in ``[0, 1]``. Use ``scipy.sparse.hstack`` of
        ``L`` copies to obtain the full ``(n_local, m·L)`` stacked taper.
    """
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree

    state_coords = np.asarray(state_coords, dtype=float)
    obs_coords = np.asarray(obs_coords, dtype=float)
    n_local = state_coords.shape[0]
    m = obs_coords.shape[0]

    if L_loc <= 0:
        # Degenerate: no taper, return an empty CSR.
        return csr_matrix((n_local, m), dtype=float)

    c = 0.5 * float(L_loc)  # GC half-cutoff so support is [0, 2c] = [0, L_loc]
    tree = cKDTree(obs_coords)

    rows: list = []
    cols: list = []
    data: list = []
    for i in range(n_local):
        neigh = tree.query_ball_point(state_coords[i], r=L_loc)
        if not neigh:
            continue
        idx_arr = np.asarray(neigh, dtype=np.int64)
        diffs = obs_coords[idx_arr] - state_coords[i]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        weights = gaspari_cohn(dists / c)
        keep = weights > 0
        if not np.any(keep):
            continue
        kept_idx = idx_arr[keep]
        kept_w = weights[keep]
        rows.extend([i] * int(kept_w.size))
        cols.extend(kept_idx.tolist())
        data.extend(kept_w.tolist())

    return csr_matrix(
        (np.asarray(data, dtype=float),
         (np.asarray(rows, dtype=np.int64),
          np.asarray(cols, dtype=np.int64))),
        shape=(n_local, m),
    )


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class QPCAEnDCF:
    """
    QPCA-EnDCF 4D filter (PDE-friendly observation interface).

    Parameters
    ----------
    apply_H : callable
        ``apply_H(X_ens)`` where ``X_ens`` is an array of shape ``(n, N)``
        (state DOFs by ensemble members) and the return value has shape
        ``(m, N)`` where ``m`` is the number of observations per time. This
        replaces the dense ``H`` matrix in the original paper code with a
        functional interface so we can call an FEniCSx point-observation
        operator one column at a time.
    R : ndarray, shape ``(m, m)``
        Observation-error covariance for one observation time.
    window_len : int
        ``L`` — number of observation times per assimilation window.
    k : int, optional
        Fixed number of PCA modes to retain (default 1, matching the paper).
        Ignored when ``kappa_target`` is supplied — see ``kappa_target``.
    kappa_target : float, optional
        If not ``None``, switch the filter into **adaptive-κ** mode: at each
        call to :meth:`update` the kept-mode count is chosen as the smallest
        ``κ`` such that ``Σ_{i=1..κ} λ_i / Σ_i λ_i ≥ kappa_target`` (the
        variance-explained criterion), with the eigenvalues ``λ_i`` sorted in
        descending order. Must lie in ``(0, 1]``. The result is clipped to
        ``[k_min, k_max]`` (see below). Motivation: the empirical W1→W2
        spectrum collapse on the SWE PDE problem (top-1/trace going from
        ~32 % to ~90 % in two windows) makes any fixed κ a bad fit for both
        windows.
    k_min, k_max : int, optional
        Optional clamps on adaptive κ. ``k_min`` defaults to 1; ``k_max``
        defaults to ``min(N-1, mL)`` per call. Ignored when ``kappa_target``
        is ``None``.
    stabilize : bool, optional
        Whether to stabilize the innovation covariance against ``R_block``.

    Notes
    -----
    With ``kappa_target=None`` (the default) the filter is algorithmically
    identical to the paper implementation; only the H plumbing changed.
    See the module docstring for details. With ``kappa_target`` set, the
    update keeps the same code path except for how many leading eigenvectors
    enter the QPCA projector — so the parity test in
    ``tests/test_qpca_endcf.py`` continues to hold in default mode.
    """

    def __init__(
        self,
        apply_H: Callable[[np.ndarray], np.ndarray],
        R: np.ndarray,
        window_len: int,
        k: int = 1,
        kappa_target: float = None,
        k_min: int = 1,
        k_max: int = None,
        stabilize: bool = True,
    ):
        if window_len <= 0:
            raise ValueError(f"window_len must be positive (got {window_len})")
        R = np.asarray(R, dtype=float)
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise ValueError(f"R must be square (got shape {R.shape})")
        if kappa_target is not None:
            if not (0.0 < float(kappa_target) <= 1.0):
                raise ValueError(
                    f"kappa_target must lie in (0, 1] (got {kappa_target})"
                )
        if int(k_min) < 1:
            raise ValueError(f"k_min must be >= 1 (got {k_min})")

        self.apply_H = apply_H
        self.R = R
        self.window_len = int(window_len)
        self.k = int(k)
        self.kappa_target = (
            float(kappa_target) if kappa_target is not None else None
        )
        self.k_min = int(k_min)
        self.k_max = int(k_max) if k_max is not None else None
        self.stabilize = bool(stabilize)
        self.m = R.shape[0]
        # Last-update diagnostic — populated by ``update`` so the experiment
        # driver can log per-window κ in adaptive mode.
        self.last_k_used: int = int(k)

        # Block structures: jitter the block-diag R to make Cholesky robust.
        self.R_block = _block_diag_repeat(R, window_len)
        jitter = 1e-10
        R_block_reg = self.R_block + jitter * np.eye(self.R_block.shape[0])
        self.R_block_chol = cholesky(R_block_reg)  # R_block^{1/2} = L
        # R_block^{-1/2} via L^{-1} (more stable than computing R^{-1} then sqrt).
        L_inv = solve(self.R_block_chol, np.eye(self.R_block_chol.shape[0]))
        self.R_block_inv_sqrt = L_inv.T

    def update(
        self,
        X_path: List[np.ndarray],
        z_stack: np.ndarray,
        HX_blocks: List[np.ndarray] = None,
        rho=None,
    ) -> np.ndarray:
        """
        Perform 4D analysis update with whitened PCA filtering.

        Parameters
        ----------
        X_path : list of ndarray
            Length-L list of state ensembles at each observation time inside
            the window. Each element has shape ``(n, N)``.
        z_stack : ndarray, shape ``(mL,)``
            Stacked observation vector at those times.
        HX_blocks : list of ndarray, optional
            Pre-computed ``H @ X_path[t]`` for each ``t``. When supplied,
            ``apply_H`` is **not** invoked here — this is a pure performance
            shortcut for callers (e.g. an SWE PDE experiment) where applying
            H requires expensive FEM evaluations and the caller already has
            the result in hand from an earlier diagnostic step. The
            mathematical update is unchanged regardless of which path is
            taken.
        rho : scipy.sparse matrix or None, optional
            Optional Gaspari-Cohn-style localization taper of shape
            ``(n, mL)`` with values in ``[0, 1]``. When supplied, the empirical
            cross-covariance is Schur-multiplied entrywise by ``rho`` before
            being applied as the gain:

                K_loc = (rho ⊙ P_xz) · S^{-1}
                δX    = K_loc · corr_obs

            Only the ``(i, j)`` entries where ``rho[i, j] != 0`` are formed,
            so the cost is ``O(nnz(rho) · N)`` rather than ``O(n · mL · N)``.
            This is **not** part of the paper's Algorithm 3 — passing
            ``rho=None`` (the default) recovers the paper-exact update and
            preserves the bit-for-bit parity test in
            ``tests/test_qpca_endcf.py``.

        Returns
        -------
        X_a_end : ndarray, shape ``(n, N)``
            Updated ensemble at window end.
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

        # --- Whitened residuals + PCA in stacked data space ---
        E = self.R_block_inv_sqrt.T @ (HX_stack - z_stack[:, None])
        Ec = E - E.mean(axis=1, keepdims=True)
        C = (Ec @ Ec.T) / (N - 1)

        w, V = eigh(C)  # ascending eigenvalues
        eigs_desc = np.asarray(w[::-1], dtype=float)
        trace_val = float(np.sum(w))

        # Choose κ for this update. Fixed mode (kappa_target is None) keeps
        # the paper-exact behavior — Vk is the top-k eigenvectors of C and
        # the parity test in tests/test_qpca_endcf.py continues to pass.
        if self.kappa_target is None:
            k_eff = self.k
        else:
            # Adaptive: smallest κ s.t. cumulative variance ≥ kappa_target.
            # All-zero spectrum guard: fall back to k_min.
            if trace_val > 0:
                cum = np.cumsum(eigs_desc) / trace_val
                # +1 because cumsum index 0 corresponds to top-1 mode.
                k_eff = int(np.searchsorted(cum, self.kappa_target) + 1)
            else:
                k_eff = self.k_min
            k_max_eff = (
                self.k_max if self.k_max is not None else min(N - 1, eigs_desc.size)
            )
            k_eff = int(max(self.k_min, min(k_eff, k_max_eff)))
        Vk = V[:, -k_eff:]

        # Stash spectrum + κ-used diagnostics for the caller (purely
        # informational — the analysis update itself is fully determined by
        # Vk above). Eigenvalues are in descending order; trace is the total
        # variance in the whitened residual covariance.
        self.last_eigenvalues_desc = eigs_desc
        self.last_eigenvalue_trace = trace_val
        self.last_k_used = int(k_eff)

        # QPCA map (whitened space) with WME sign.
        Q_qpca = -Vk @ (Vk.T @ E)

        # --- Ensemble anomalies for gain ---
        # The paper helper ``_cov_and_anoms`` returns (cov, anoms). For the
        # state block on this SWE PDE problem, ``n`` is ~2e5 and the cov
        # matrix it would compute is ``n x n`` ≈ 350 GB — instantly killed
        # by macOS jetsam. We only need the anomalies here (the covariance
        # is unused below), so inline the cheap (n, N) computation directly
        # for both blocks.
        X_end = X_path[-1]
        A_x_end = X_end - X_end.mean(axis=1, keepdims=True)
        A_y = HX_stack - HX_stack.mean(axis=1, keepdims=True)

        P_yy = (A_y @ A_y.T) / (N - 1)

        S = P_yy
        if self.stabilize:
            S = _stabilize_spd_like(S, self.R_block)
        S_inv = _sym_posdef_inverse(S)

        # Gain and de-whitened correction.
        # Mathematically: K = P_xy @ S_inv with P_xy = A_x_end @ A_y^T / (N-1).
        # Forming K explicitly costs O(n * mL) memory, which on the SWE PDE
        # state (n ≈ 200k) and L * m ≈ a few hundred gives ~hundreds of MB
        # and can OOM the process on workstations. The full update is
        #
        #     X_a_end = X_path[-1] + K @ corr_obs
        #               = X_path[-1] + A_x_end @ (A_y^T @ S_inv @ corr_obs) / (N-1)
        #
        # The inner parenthesised factor is shape (N, N) (tiny ensemble-size
        # matrix), so we evaluate it first and only multiply by the
        # (n, N) state-anomaly block. This preserves the original algorithm
        # — same K and same corr_obs are involved — only the order of the
        # final two matrix products is swapped.
        corr_obs = self.R_block_chol @ Q_qpca

        if rho is None:
            # Paper-exact reassociated update — preserves bit-for-bit parity
            # with the upstream reference implementation.
            inner = (A_y.T @ (S_inv @ corr_obs)) / (N - 1)
            X_a_end = X_path[-1] + A_x_end @ inner
            return X_a_end

        # Localized update: form rho ⊙ P_xz only at the nnz of rho.
        # Memory cost: O(nnz(rho)) instead of O(n · mL). Time: O(nnz(rho) · N).
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
        # P_xz[i, j] = (A_x_end[i, :] · A_y[j, :]) / (N - 1). We only need it
        # at the (rows, cols) pairs where rho is nonzero.
        inner_products = np.sum(
            A_x_end[rows, :] * A_y[cols, :], axis=1
        ) / (N - 1)
        localized_data = vals * inner_products

        P_xz_loc = _sp.csr_matrix(
            (localized_data, (rows, cols)),
            shape=rho.shape,
        )
        B = S_inv @ corr_obs   # (mL, N)
        delta_X = P_xz_loc @ B  # sparse-dense product → dense (n, N)
        X_a_end = X_path[-1] + delta_X
        return X_a_end

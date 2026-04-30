"""Storage classes for solver state, Jacobians, and adjoints.

This module provides dedicated storage containers for different types of
solver data used in forward solves and data assimilation.
"""

from typing import List, Optional
import numpy as np
from petsc4py import PETSc


# ---------------------------------------------------------------------------
# R2 Jacobian-handoff one-shot diagnostic state. See
# docs/idealized_inlet_jacobian_handoff_trace.md.
# ---------------------------------------------------------------------------
_HANDOFF = {
    "last_saved": None,
    "cg_entry_fired": False,
    "storage_fired": False,
    "postzero_fired": False,
    "adjoint_fired": False,
}


def _jac_handoff_log(stage: str, mat, extras: Optional[dict] = None) -> None:
    """One-line handoff log emitted on rank 0. PETSc `.norm()` and `.getInfo()`
    are COLLECTIVE, so all ranks must enter — only the print is rank-gated.
    Guarded against exceptions; no-op when mat is None."""
    try:
        from mpi4py import MPI as _MPI
        rank = _MPI.COMM_WORLD.Get_rank()
        if mat is None:
            if rank == 0:
                print(f"[jac-handoff] stage={stage} mat=None", flush=True)
            return
        # Collective on all ranks:
        norm = mat.norm(PETSc.NormType.NORM_FROBENIUS)
        nz = int(mat.getInfo().get("nz_used", -1))
        if rank == 0:
            msg = f"[jac-handoff] stage={stage} norm={norm:.3e} nz={nz} id={id(mat)}"
            if extras:
                for k, v in extras.items():
                    msg += f" {k}={v}"
            print(msg, flush=True)
    except Exception as _e:
        try:
            from mpi4py import MPI as _MPI
            if _MPI.COMM_WORLD.Get_rank() == 0:
                print(f"[jac-handoff] log failed at {stage}: {_e}", flush=True)
        except Exception:
            print(f"[jac-handoff] log failed at {stage}: {_e}", flush=True)


class SolverStateStorage:
    """Centralized storage for solver states, Jacobians, and adjoint data.

    This class manages arrays and matrices generated during time-stepping,
    particularly for 4D-Var data assimilation where historical data must
    be retained for adjoint computations.
    """

    def __init__(self):
        """Initialize empty storage containers."""
        # Forward solve data
        self.saved_states: List[np.ndarray] = []
        """List of state vectors from forward time integration"""

        self.saved_jacobians: List[PETSc.Mat] = []
        """List of Jacobian matrices from Newton solves (distributed)"""

        self.saved_parameter_derivatives: List[List[PETSc.Vec]] = []
        """Per-timestep residual derivatives ∂R_k/∂θ_i stored as PETSc vectors"""

        # Adjoint solve data
        self.saved_adjoints: List[PETSc.Mat] = []
        """List of adjoint Jacobian matrices (distributed)"""

        # Wetting/drying data
        self.dry_nodes: List[np.ndarray] = []
        """List of dry node indices at each timestep"""

        self.saved_bathy: List[np.ndarray] = []
        """List of bathymetry values at observation points"""

        self.saved_true_bathy: List[np.ndarray] = []
        """List of true bathymetry values (before wetting/drying adjustments)"""

        # Replay metadata (parity-debug infrastructure for the recompute
        # Jacobians-in-adjoint feature). Captured per solved timestep at the
        # *exact* moment the forward solver assembles its stored J_n
        # ("Reassembling Jacobian at converged solution"). Stores full
        # ghosted ``Function.x.array`` snapshots — not owned-only Vec
        # representations — so a shadow reassembly context can restore the
        # form's state byte-for-byte and produce a J that should be
        # bit-equal (up to assembly floating-point) to the stored J_n.
        #
        # Each entry is a dict with:
        #   "u":          np.ndarray (full ghosted, copied)
        #   "u_n":        np.ndarray
        #   "u_n_old":    np.ndarray
        #   "u_bc":       np.ndarray (problem.u_bc.x.array, full ghosted)
        #   "theta1":     float
        #   "t":          float
        #   "timestep":   int (the n that produced this Jacobian)
        #
        # Captured only when SWE4DVAR_CAPTURE_REPLAY_META=1. Cheap (~MB-scale
        # numpy data per timestep, no Mat allocation). Independent of the
        # legacy ``saved_jacobians`` storage path so the parity harness can
        # run side-by-side without disturbing production paths.
        self.replay_metadata: List[dict] = []

        # Persistent Jacobian pool (Phase B of memory-leak fix).
        # Each cost-function evaluation does a forward solve that calls
        # save_jacobian() once per timestep (~6 calls for nt_da=6). The
        # naive `jacobian.copy()` allocates a fresh ~120 MB Mat per call;
        # PETSc holds the slabs in its internal allocator pool even after
        # destroy(), and that's the dominant residual leak after sweep-KSP
        # (~720 MB of the remaining ~547 MB/eval growth).
        #
        # The pool keeps a stable set of Mats across evals, refilling them
        # via SAME_NONZERO_PATTERN copy at each save_jacobian. clear() only
        # rewinds an index counter; the Mats are not destroyed until
        # release_pool() is explicitly called at end of life. For cycling
        # DA, that means at end-of-window cleanup in run_single_method.
        # Disable with SWE4DVAR_JACOBIAN_POOL=0.
        # Default OFF until validated end-to-end. The pool helps in tightly-
        # repeated identical-control reproducer cases but introduced numerical
        # blow-up in real cycling DA on 6w_pool (3132492). Investigate before
        # re-enabling by default; in the meantime sweep-KSP alone provides
        # the dominant memory savings.
        import os as _os
        self._jacobian_pool_enabled = (
            _os.environ.get("SWE4DVAR_JACOBIAN_POOL", "0").strip() == "1"
        )
        self._jacobian_pool: List[PETSc.Mat] = []
        self._jacobian_pool_idx: int = 0

    def clear(self):
        """Clear all stored data to free memory.

        Saved Jacobians use the persistent pool when enabled — the pool's
        Mats are NOT destroyed by clear(); only the index pointer is rewound
        so the next eval refills them in place. This avoids the per-eval
        ~120 MB × num_steps allocator churn that PETSc's internal pool
        retains after destroy(). Call release_pool() at end of cost-function
        life (run_single_method's teardown block) to actually free the pool.
        Disable the pool with SWE4DVAR_JACOBIAN_POOL=0 to fall back to the
        legacy destroy-per-clear behavior.
        """
        # Clear numpy arrays - no special handling needed
        self.saved_states.clear()
        self.dry_nodes.clear()
        self.saved_bathy.clear()
        self.saved_true_bathy.clear()

        # Replay metadata is per-eval and ALWAYS cleared with the rest of
        # the per-eval data — it has no pool semantics.
        self.replay_metadata.clear()

        if self._jacobian_pool_enabled:
            # Pool mode: don't destroy the Jacobian Mats — they live in
            # _jacobian_pool and get refilled in place on the next eval.
            # Just empty the public list and rewind the pool index.
            self.saved_jacobians.clear()
            self._jacobian_pool_idx = 0
        else:
            # Legacy mode: destroy each Mat (PETSc internal pool retains
            # the freed slabs anyway, but at least we release them).
            for J in self.saved_jacobians:
                try:
                    if hasattr(J, 'destroy') and callable(J.destroy):
                        J.destroy()
                except (PETSc.Error, RuntimeError, AttributeError):
                    # Ignore errors - matrix may already be destroyed or invalid
                    pass
            self.saved_jacobians.clear()

        for A in self.saved_adjoints:
            try:
                if hasattr(A, 'destroy') and callable(A.destroy):
                    A.destroy()
            except (PETSc.Error, RuntimeError, AttributeError):
                # Ignore errors - matrix may already be destroyed or invalid
                pass
        self.saved_adjoints.clear()

        for derivative_row in self.saved_parameter_derivatives:
            for vec in derivative_row:
                try:
                    if hasattr(vec, 'destroy') and callable(vec.destroy):
                        vec.destroy()
                except (PETSc.Error, RuntimeError, AttributeError):
                    pass
        self.saved_parameter_derivatives.clear()

    def save_state(self, state: np.ndarray):
        """Save a state vector.

        Args:
            state: State vector to save (will be copied)
        """
        self.saved_states.append(state.copy())

    def save_jacobian(self, jacobian: PETSc.Mat):
        """Save a Jacobian matrix.

        Pool-mode (default, when SWE4DVAR_JACOBIAN_POOL=1): refill the next
        free Mat in self._jacobian_pool via SAME_NONZERO_PATTERN copy. The
        Mat's nonzero pattern matches across timesteps because the
        discretization (DG element + mesh + forms) is fixed. First call
        seeds the pool with a fresh duplicate; subsequent calls reuse the
        slot.

        Legacy mode (SWE4DVAR_JACOBIAN_POOL=0): full ``jacobian.copy()``
        every call (the original behavior).

        Args:
            jacobian: Jacobian matrix to save.
        """
        # R2 handoff pre-copy check (one-shot, first call only)
        _first = not _HANDOFF["storage_fired"]
        if _first and hasattr(jacobian, "copy"):
            _jac_handoff_log("storage_pre_copy", jacobian)

        if hasattr(jacobian, 'copy'):
            if self._jacobian_pool_enabled:
                idx = self._jacobian_pool_idx
                if idx < len(self._jacobian_pool):
                    target = self._jacobian_pool[idx]
                    try:
                        jacobian.copy(target,
                                      structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
                    except (PETSc.Error, RuntimeError) as _e:
                        # Pattern mismatch (rare — e.g. mesh changed); replace slot.
                        try:
                            target.destroy()
                        except Exception:
                            pass
                        target = jacobian.copy()
                        self._jacobian_pool[idx] = target
                else:
                    # First time at this index: allocate and stash.
                    target = jacobian.copy()
                    self._jacobian_pool.append(target)
                self.saved_jacobians.append(target)
                self._jacobian_pool_idx += 1
            else:
                # Legacy: fresh allocation every save.
                self.saved_jacobians.append(jacobian.copy())
        else:
            # For testing purposes, allow non-PETSc objects
            self.saved_jacobians.append(jacobian)

        if _first:
            stored = self.saved_jacobians[-1] if self.saved_jacobians else None
            aliased = (id(stored) == id(jacobian)) if stored is not None else False
            _jac_handoff_log("storage_post_copy", stored,
                             extras={"aliased_input": aliased})
            _HANDOFF["last_saved"] = stored
            _HANDOFF["storage_fired"] = True

    def save_replay_metadata(
        self,
        timestep: int,
        u_array: np.ndarray,
        u_n_array: np.ndarray,
        u_n_old_array: np.ndarray,
        u_bc_array: Optional[np.ndarray],
        theta1_value: float,
        problem_t: float,
    ) -> None:
        """Capture a per-timestep replay snapshot.

        Stores the FULL ghosted ``Function.x.array`` for u/u_n/u_n_old/u_bc,
        plus the discretization scalar ``theta1`` and the time ``t``. These
        are exactly the form-visible inputs at the moment the forward
        solver finalized J_n via "Reassembling Jacobian at converged
        solution". A downstream JacobianReplayContext restores this state
        verbatim into a snapshot/restore-protected forward solver and
        reassembles a Jacobian that should match the stored J_n to
        floating-point assembly noise.

        Cheap: scalar metadata + a few numpy arrays per timestep. No PETSc
        Mat allocation.
        """
        self.replay_metadata.append({
            "timestep":   int(timestep),
            "u":          np.asarray(u_array).copy(),
            "u_n":        np.asarray(u_n_array).copy(),
            "u_n_old":    np.asarray(u_n_old_array).copy(),
            "u_bc":       (np.asarray(u_bc_array).copy()
                           if u_bc_array is not None else None),
            "theta1":     float(theta1_value),
            "t":          float(problem_t),
        })

    def release_pool(self) -> dict:
        """Destroy all Mats in the persistent Jacobian pool.

        Call this at end-of-life for the cost-function / solver / storage,
        e.g. inside the cycling-DA window-teardown block. Process exit
        also reclaims pool memory but explicit release is preferred when
        the same Python process spans multiple windows.
        """
        n_destroyed = 0
        for M in self._jacobian_pool:
            try:
                if hasattr(M, "destroy") and callable(M.destroy):
                    M.destroy()
                    n_destroyed += 1
            except (PETSc.Error, RuntimeError, AttributeError):
                pass
        self._jacobian_pool.clear()
        self._jacobian_pool_idx = 0
        # The public saved_jacobians may still hold references; drop them.
        self.saved_jacobians.clear()
        return {"jacobian_pool_destroyed": int(n_destroyed)}

    def save_adjoint(self, adjoint: PETSc.Mat):
        """Save an adjoint Jacobian matrix.

        Args:
            adjoint: Adjoint matrix to save (will be copied)
        """
        self.saved_adjoints.append(adjoint.copy())

    def save_parameter_derivatives(self, derivative_vectors: List[PETSc.Vec]):
        """Save timestep residual derivatives with respect to parameters.

        Args:
            derivative_vectors: One residual-derivative vector per parameter.
        """
        copied = []
        for vec in derivative_vectors:
            if hasattr(vec, "copy"):
                copied.append(vec.copy())
            else:
                copied.append(vec)
        self.saved_parameter_derivatives.append(copied)

    def save_dry_nodes(self, dry_indices: np.ndarray):
        """Save dry node indices.

        Args:
            dry_indices: Array of dry node indices
        """
        self.dry_nodes.append(dry_indices.copy())

    def save_bathymetry(self, bathy: np.ndarray, is_true_bathy: bool = False):
        """Save bathymetry values.

        Args:
            bathy: Bathymetry array
            is_true_bathy: If True, save to true_bathy storage
        """
        if is_true_bathy:
            self.saved_true_bathy.append(bathy.copy())
        else:
            self.saved_bathy.append(bathy.copy())

    def get_state(self, index: int) -> Optional[np.ndarray]:
        """Retrieve a saved state by index.

        Args:
            index: Index of the state to retrieve (supports negative indexing)

        Returns:
            State vector or None if index out of bounds
        """
        try:
            return self.saved_states[index]
        except IndexError:
            return None

    def get_jacobian(self, index: int) -> Optional[PETSc.Mat]:
        """Retrieve a saved Jacobian by index.

        Args:
            index: Index of the Jacobian to retrieve (supports negative indexing)

        Returns:
            Jacobian matrix or None if index out of bounds
        """
        try:
            return self.saved_jacobians[index]
        except IndexError:
            return None

    def num_states(self) -> int:
        """Get number of saved states."""
        return len(self.saved_states)

    def num_jacobians(self) -> int:
        """Get number of saved Jacobians."""
        return len(self.saved_jacobians)

    def num_adjoints(self) -> int:
        """Get number of saved adjoints."""
        return len(self.saved_adjoints)

    def num_parameter_derivative_steps(self) -> int:
        """Get number of timesteps with saved parameter derivatives."""
        return len(self.saved_parameter_derivatives)

    def estimate_memory_mb(self) -> dict:
        """Estimate memory usage in MB for each storage component.

        Returns:
            Dictionary with memory estimates for each component
        """
        bytes_to_mb = 1.0 / (1024**2)

        estimates = {}

        # State vectors
        if self.saved_states:
            state_bytes = sum(s.nbytes for s in self.saved_states)
            estimates["states"] = state_bytes * bytes_to_mb

        # Jacobians (rough estimate based on PETSc matrix info)
        if self.saved_jacobians:
            # This is approximate - actual memory depends on sparsity
            jac_info = self.saved_jacobians[0].getInfo()
            nnz = jac_info["nz_used"]
            estimates["jacobians"] = len(self.saved_jacobians) * nnz * 8 * bytes_to_mb

        # Adjoints (similar to Jacobians)
        if self.saved_adjoints:
            adj_info = self.saved_adjoints[0].getInfo()
            nnz = adj_info["nz_used"]
            estimates["adjoints"] = len(self.saved_adjoints) * nnz * 8 * bytes_to_mb

        if self.saved_parameter_derivatives:
            first_row = self.saved_parameter_derivatives[0]
            if first_row:
                vec_bytes = first_row[0].getLocalSize() * bytes_per_float
                estimates["parameter_derivatives"] = (
                    len(self.saved_parameter_derivatives)
                    * len(first_row)
                    * vec_bytes
                    * bytes_to_mb
                )

        # Wetting/drying data
        if self.dry_nodes:
            dry_bytes = sum(d.nbytes for d in self.dry_nodes)
            estimates["dry_nodes"] = dry_bytes * bytes_to_mb

        if self.saved_bathy:
            bathy_bytes = sum(b.nbytes for b in self.saved_bathy)
            estimates["bathymetry"] = bathy_bytes * bytes_to_mb

        if self.saved_true_bathy:
            true_bathy_bytes = sum(b.nbytes for b in self.saved_true_bathy)
            estimates["true_bathymetry"] = true_bathy_bytes * bytes_to_mb

        estimates["total"] = sum(estimates.values())

        return estimates

    def __repr__(self) -> str:
        return (
            f"SolverStateStorage("
            f"states={self.num_states()}, "
            f"jacobians={self.num_jacobians()}, "
            f"adjoints={self.num_adjoints()}, "
            f"parameter_derivatives={self.num_parameter_derivative_steps()}, "
            f"dry_nodes={len(self.dry_nodes)})"
        )

"""Shadow Jacobian-replay context for the recompute-Jacobians-in-adjoint feature.

Goal: reassemble J_n from saved replay metadata without corrupting the live
forward solver's mutable state. The earlier in-place mutation strategy
(directly poking ``solver.u.x.array``, ``solver.u_n.x.array``, etc.) had
two problems:

  1. It mutated state the *next* forward solve depends on, which produced
     subtle Newton failures and silent gradient drift.
  2. It used owned-only PETSc Vec slices instead of the full ghosted
     ``Function.x.array`` snapshots the form actually sees, so the
     reassembled J was not bit-equal to the stored one.

The fix is two changes used together:

  - ``solver_storage.SolverStateStorage.save_replay_metadata`` saves
    full ghosted arrays + theta1 + t + u_bc per timestep at the exact
    moment the forward stored its J.
  - ``JacobianReplayContext`` snapshots the live forward state, restores
    the saved replay record verbatim, assembles J via the *same form
    objects* the forward solver used, copies the result, and restores
    the live state — so the live solver leaves the replay unchanged.

Use this only for parity validation. Default operational adjoint stays
on the legacy stored-Jacobian path until parity is proven.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from petsc4py import PETSc


class JacobianReplayContext:
    """Snapshot/restore-protected Jacobian reassembly.

    Constructed once per parity session. Each call to ``reassemble(meta)``
    is fully self-contained: live state is snapshotted, saved replay
    metadata is restored byte-for-byte, J is assembled into a buffer
    Mat, the buffer is copied out as the result, and live state is
    restored from the snapshot.
    """

    def __init__(self, forward_solver):
        """
        Parameters
        ----------
        forward_solver
            The CGImplicit (or compatible) instance that ran the original
            forward solve. The replay context will:
              * use forward_solver.solver (CustomNewtonProblem) for the
                form ``self.jacobian`` and the assembly target ``self.A``
              * snapshot/restore forward_solver.u / u_n / u_n_old,
                forward_solver.theta1, forward_solver.problem.t, and
                forward_solver.problem.u_bc (if present).
        """
        self.solver = forward_solver
        self.problem = forward_solver.problem
        # The persistent Newton problem owns the assembly target Mat A and
        # the form. Reuse it; we do not allocate a separate Mat.
        np_problem = getattr(forward_solver, "solver", None)
        if np_problem is None:
            raise RuntimeError(
                "JacobianReplayContext requires forward_solver.solver "
                "(CustomNewtonProblem) to be present. Phase C-1 forward "
                "Newton reuse keeps this alive across cost-evals.")
        self.newton_problem = np_problem

    # ---- snapshot / restore --------------------------------------------

    def _snapshot(self) -> dict:
        """Capture the live forward solver state into a dict of arrays/scalars.

        Uses full ``Function.x.array.copy()`` to capture both owned and
        ghost rows. Float scalars are captured by value.
        """
        snap = {
            "u":         self.solver.u.x.array.copy(),
            "u_n":       self.solver.u_n.x.array.copy(),
            "u_n_old":   self.solver.u_n_old.x.array.copy(),
            "theta1":    float(self.solver.theta1.value),
            "t":         float(getattr(self.problem, "t", 0.0)),
            "u_bc":      None,
        }
        if hasattr(self.problem, "u_bc"):
            try:
                snap["u_bc"] = self.problem.u_bc.x.array.copy()
            except Exception:
                snap["u_bc"] = None
        return snap

    def _restore(self, snap: dict) -> None:
        """Write ``snap`` back into the live solver state, byte-for-byte."""
        self.solver.u.x.array[:]     = snap["u"]
        self.solver.u_n.x.array[:]   = snap["u_n"]
        self.solver.u_n_old.x.array[:] = snap["u_n_old"]
        self.solver.theta1.value     = snap["theta1"]
        try:
            self.problem.t = snap["t"]
        except Exception:
            pass
        if snap["u_bc"] is not None and hasattr(self.problem, "u_bc"):
            try:
                self.problem.u_bc.x.array[:] = snap["u_bc"]
            except Exception:
                pass
        # Re-evaluate time-dependent forcing (wind/pressure) at the
        # snapshot t so the form's source term is consistent with the
        # restored state. The forward's source UFL references
        # self.problem.forcing.{windx, windy, pressure} as Functions,
        # and those Functions are mutated in-place by evaluate(t).
        # Without this, after a replay call the live wind state stays
        # at the t of the most recent _load, contaminating any
        # downstream production solve.
        self._evaluate_forcing_at(snap["t"])

    def _load(self, meta: dict) -> None:
        """Write the replay metadata into the live solver state.

        ``meta`` is a single entry from
        ``solver.storage.replay_metadata``: contains FULL ghosted arrays.
        Uses ``[:]`` slicing to overwrite both owned and ghost rows
        directly — no scatter_forward needed because the saved arrays
        already include consistent ghost values from the original forward
        solve.
        """
        self.solver.u.x.array[:]      = meta["u"]
        self.solver.u_n.x.array[:]    = meta["u_n"]
        self.solver.u_n_old.x.array[:] = meta["u_n_old"]
        self.solver.theta1.value      = meta["theta1"]
        try:
            self.problem.t = meta["t"]
        except Exception:
            pass
        if meta["u_bc"] is not None and hasattr(self.problem, "u_bc"):
            try:
                self.problem.u_bc.x.array[:] = meta["u_bc"]
            except Exception:
                pass
        # Re-evaluate time-dependent forcing at meta["t"]. The form's
        # source term reads self.problem.forcing.{windx, windy, pressure}
        # — Functions populated by forcing.evaluate(t). The forward
        # advances these via problem.advance_time(); the replay context
        # must do the same so the assembled J reflects the wind state
        # at the saved timestep, not at the live solver's last t.
        # Without this, the J error grows linearly with backward distance
        # (bisector observed: rel_F=0 at first replay step, ~1e-04 by step 1).
        self._evaluate_forcing_at(meta["t"])

    def _evaluate_forcing_at(self, t: float) -> None:
        """Refresh self.problem.forcing's wind/pressure Functions at ``t``.

        No-op if the problem has no forcing or if evaluate fails — this
        is observational (snapshot/restore) code and must not throw.
        """
        forcing = getattr(self.problem, "forcing", None)
        if forcing is None:
            return
        evaluator = getattr(forcing, "evaluate", None)
        if evaluator is None:
            return
        try:
            evaluator(float(t))
        except Exception:
            pass

    # ---- public API ----------------------------------------------------

    def reassemble(self, meta: dict, copy: bool = True) -> "PETSc.Mat":
        """Reassemble J at the saved state captured in ``meta``.

        Sequence:
          1. snapshot live state
          2. load meta into live state (full ghosted arrays + scalars)
          3. assemble J into self.newton_problem.A using the form
             ``self.newton_problem.jacobian`` — same form, same Mat A as
             the legacy "Reassembling at converged solution" code path
             in newton.py
          4. either return ``A`` directly (caller will not modify it) or
             return a copy (default — safe for caller)
          5. restore live state

        BCs: assembly is done WITHOUT bcs, matching the unmodified-physics
        Jacobian convention used by the legacy stored-J path.
        """
        snap = self._snapshot()
        try:
            self._load(meta)
            try:
                from dolfinx.fem import petsc as _petsc
            except ImportError:
                from dolfinx import fem as _fe
                _petsc = _fe.petsc
            A = self.newton_problem.A
            A.zeroEntries()
            _petsc.assemble_matrix(A, self.newton_problem.jacobian)
            A.assemble()
            if copy:
                result = A.copy()
            else:
                result = A
            return result
        finally:
            self._restore(snap)

    # ---- introspection -------------------------------------------------

    def saved_state_summary(self, meta: dict) -> dict:
        """Quick summary of one replay record, for logging."""
        return {
            "timestep": int(meta.get("timestep", -1)),
            "t":        float(meta.get("t", float("nan"))),
            "theta1":   float(meta.get("theta1", float("nan"))),
            "u_norm":   float(np.linalg.norm(meta["u"])),
            "u_n_norm": float(np.linalg.norm(meta["u_n"])),
            "u_bc_norm": (
                float(np.linalg.norm(meta["u_bc"]))
                if meta.get("u_bc") is not None else None
            ),
        }

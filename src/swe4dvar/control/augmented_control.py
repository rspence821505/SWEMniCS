"""Packed control abstractions for augmented state/parameter estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
try:
    from petsc4py import PETSc
except ModuleNotFoundError:  # pragma: no cover - import-light path for tests/docs
    PETSc = None


@dataclass
class ControlVector:
    """Logical representation of the augmented control c = [u0; theta]."""

    u0: np.ndarray
    theta: np.ndarray

    def __post_init__(self):
        self.u0 = np.asarray(self.u0, dtype=float).reshape(-1)
        self.theta = np.asarray(self.theta, dtype=float).reshape(-1)

    def pack(self) -> np.ndarray:
        if self.u0.size == 0:
            return self.theta.copy()
        if self.theta.size == 0:
            return self.u0.copy()
        return np.concatenate([self.u0, self.theta])

    @classmethod
    def unpack(
        cls,
        packed: np.ndarray,
        *,
        state_size: int,
        theta_size: int,
    ) -> "ControlVector":
        packed = np.asarray(packed, dtype=float).reshape(-1)
        expected = int(state_size) + int(theta_size)
        if packed.size != expected:
            raise ValueError(
                f"Packed control has size {packed.size}, expected {expected} "
                f"(state={state_size}, theta={theta_size})"
            )
        u0 = packed[:state_size].copy() if state_size > 0 else np.zeros(0, dtype=float)
        theta = (
            packed[state_size: state_size + theta_size].copy()
            if theta_size > 0
            else np.zeros(0, dtype=float)
        )
        return cls(u0=u0, theta=theta)


class ControlLayout:
    """Centralized indexing/packing rules for augmented controls.

    Stage-1 implementation is intentionally serial-only for augmented controls.
    Existing distributed state-only workflows remain unchanged elsewhere.
    """

    def __init__(
        self,
        *,
        state_size: int,
        theta_size: int = 0,
        comm=None,
    ):
        self.state_size = int(state_size)
        self.theta_size = int(theta_size)
        self.total_size = self.state_size + self.theta_size
        self.comm = comm

        if self.total_size <= 0:
            raise ValueError("Control layout must have at least one degree of freedom")

        # Packed augmented controls are serial-only for now.
        if PETSc is not None and self.comm is None:
            self.comm = PETSc.COMM_SELF

        if self.comm is not None and self.theta_size > 0 and self.comm.getSize() > 1:
            raise NotImplementedError(
                "Augmented controls with theta != 0 are currently supported only in serial. "
                "State-only distributed controls remain available through the existing path."
            )

        self.state_slice = slice(0, self.state_size)
        self.theta_slice = slice(self.state_size, self.total_size)

    def empty(self) -> ControlVector:
        return ControlVector(
            u0=np.zeros(self.state_size, dtype=float),
            theta=np.zeros(self.theta_size, dtype=float),
        )

    def unpack_array(self, packed: np.ndarray) -> ControlVector:
        return ControlVector.unpack(
            packed,
            state_size=self.state_size,
            theta_size=self.theta_size,
        )

    def unpack_vec(self, packed_vec: PETSc.Vec) -> ControlVector:
        arr = packed_vec.getArray(readonly=True).copy()
        return self.unpack_array(arr)

    def pack_array(
        self,
        *,
        u0: Optional[np.ndarray] = None,
        theta: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        control = ControlVector(
            u0=np.zeros(self.state_size, dtype=float) if u0 is None else np.asarray(u0, dtype=float),
            theta=np.zeros(self.theta_size, dtype=float) if theta is None else np.asarray(theta, dtype=float),
        )
        if control.u0.size != self.state_size:
            raise ValueError(
                f"u0 size {control.u0.size} does not match state_size {self.state_size}"
            )
        if control.theta.size != self.theta_size:
            raise ValueError(
                f"theta size {control.theta.size} does not match theta_size {self.theta_size}"
            )
        return control.pack()

    def create_petsc_vec(
        self,
        *,
        u0: Optional[np.ndarray] = None,
        theta: Optional[np.ndarray] = None,
        packed: Optional[np.ndarray] = None,
    ) -> PETSc.Vec:
        if PETSc is None:
            raise ModuleNotFoundError("petsc4py is required to create PETSc vectors")
        if packed is None:
            packed = self.pack_array(u0=u0, theta=theta)
        else:
            packed = np.asarray(packed, dtype=float).reshape(-1)
        if packed.size != self.total_size:
            raise ValueError(
                f"Packed array has size {packed.size}, expected {self.total_size}"
            )
        vec = PETSc.Vec().createSeq(self.total_size, comm=PETSc.COMM_SELF)
        vec.setArray(packed.copy())
        vec.assemble()
        return vec

    def state_view(self, packed: np.ndarray) -> np.ndarray:
        return np.asarray(packed, dtype=float)[self.state_slice]

    def theta_view(self, packed: np.ndarray) -> np.ndarray:
        return np.asarray(packed, dtype=float)[self.theta_slice]

    def combine_gradient(
        self,
        *,
        background_gradient: PETSc.Vec,
        state_gradient: Optional[PETSc.Vec] = None,
        theta_gradient: Optional[np.ndarray] = None,
    ) -> PETSc.Vec:
        if PETSc is None:
            raise ModuleNotFoundError("petsc4py is required to combine PETSc gradients")
        grad = background_gradient.copy()
        grad_arr = grad.getArray()

        if state_gradient is not None and self.state_size > 0:
            state_arr = state_gradient.getArray(readonly=True)
            grad_arr[self.state_slice] += state_arr[: self.state_size]

        if theta_gradient is not None and self.theta_size > 0:
            theta_arr = np.asarray(theta_gradient, dtype=float).reshape(-1)
            if theta_arr.size != self.theta_size:
                raise ValueError(
                    f"theta gradient size {theta_arr.size} does not match theta_size {self.theta_size}"
                )
            grad_arr[self.theta_slice] += theta_arr

        grad.setArray(grad_arr)
        grad.assemble()
        return grad

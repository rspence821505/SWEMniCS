"""Augmented-control cost functions for state-plus-parameter estimation."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

from .cost_functions import DCFourDVarCost, DCWMEFourDVarCost, FourDVarCost


class _AugmentedGradientMixin:
    """Shared utilities for packed state/parameter gradients."""

    def _control_layout(self):
        layout = getattr(self.forward_model, "control_layout", None)
        if layout is None:
            raise ValueError("Augmented cost requires forward_model.control_layout")
        return layout

    def _solve_augmented_adjoint(
        self,
        trajectory: List[PETSc.Vec],
        jacobians: List[PETSc.Mat],
        forcings: Optional[List[Optional[PETSc.Vec]]] = None,
        *,
        return_history: bool = False,
    ):
        from ..adjoint.implicit_adjoint import ImplicitAdjointSolver
        from ..utils import get_boundary_dofs

        variational_form = getattr(self.forward_model, "var_form", None)
        if variational_form is None and hasattr(self.forward_model, "solver"):
            variational_form = getattr(self.forward_model.solver, "var_form", None)

        bc_dof_indices = None
        problem = getattr(self.forward_model, "problem", None)
        has_strong_bcs = (
            problem is not None
            and hasattr(problem, "dirichlet_bcs")
            and len(problem.dirichlet_bcs) > 0
        )
        if has_strong_bcs:
            if hasattr(self.forward_model, "solver") and hasattr(self.forward_model, "problem"):
                V = self.forward_model.solver.V
                mesh = self.forward_model.problem.mesh
                boundary_dofs = get_boundary_dofs(V, mesh)
                bc_dof_indices = set(boundary_dofs.tolist())
            elif hasattr(self.forward_model, "V") and hasattr(self.forward_model, "mesh"):
                V = self.forward_model.V
                mesh = self.forward_model.mesh
                boundary_dofs = get_boundary_dofs(V, mesh)
                bc_dof_indices = set(boundary_dofs.tolist())

        solver = ImplicitAdjointSolver(
            self.forward_model,
            trajectory,
            jacobians,
            self.forward_model.dt,
            variational_form=variational_form,
            bc_dof_indices=bc_dof_indices,
            flux_formulation=True,
        )
        terminal = trajectory[-1].duplicate()
        terminal.zeroEntries()
        return solver.solve(terminal, forcings, return_history=return_history)

    def _parameter_gradient_from_forcings(
        self,
        m: PETSc.Vec,
        forcings: List[Optional[PETSc.Vec]],
        trajectory: Optional[List[PETSc.Vec]] = None,
        jacobians: Optional[List[PETSc.Mat]] = None,
    ) -> np.ndarray:
        layout = self._control_layout()
        if layout.theta_size == 0:
            return np.zeros(0, dtype=float)

        timestep_derivatives = None
        if hasattr(self.forward_model, "get_timestep_parameter_derivatives"):
            timestep_derivatives = self.forward_model.get_timestep_parameter_derivatives(
                m,
                base_trajectory=trajectory,
                base_jacobians=jacobians,
            )

        if timestep_derivatives and jacobians is not None and trajectory is not None:
            _, lambda_history = self._solve_augmented_adjoint(
                trajectory,
                jacobians,
                forcings,
                return_history=True,
            )
            num_steps = len(jacobians)
            if len(timestep_derivatives) == len(trajectory):
                derivative_rows = timestep_derivatives[1:]
            elif len(timestep_derivatives) == num_steps:
                derivative_rows = timestep_derivatives
            else:
                raise ValueError(
                    "Timestep parameter derivatives are misaligned with the forward solve: "
                    f"got {len(timestep_derivatives)} derivative rows for "
                    f"{len(trajectory)} states / {num_steps} Jacobians"
                )

            if len(lambda_history) != len(trajectory):
                raise ValueError(
                    "Adjoint history length does not match forward trajectory length: "
                    f"got {len(lambda_history)} adjoints for {len(trajectory)} states"
                )

            theta_grad = np.zeros(layout.theta_size, dtype=float)
            for step_index, derivative_row in enumerate(derivative_rows, start=1):
                lambda_step = lambda_history[step_index]
                if lambda_step is None:
                    continue
                for p, residual_derivative in enumerate(derivative_row):
                    # Lagrangian: L = J + Σ_k λ_k^T F_k  →  ∂L/∂θ = Σ_k λ_k^T (∂F_k/∂θ_p)
                    # Sign is POSITIVE: the adjoint λ already encodes the correct direction.
                    theta_grad[p] += lambda_step.dot(residual_derivative)
            return theta_grad

        bundle = self.forward_model.get_parameter_sensitivity_bundle(
            m,
            base_trajectory=trajectory,
            base_jacobians=jacobians,
        )
        if bundle is None:
            return np.zeros(layout.theta_size, dtype=float)

        theta_grad = np.zeros(layout.theta_size, dtype=float)
        for p in range(layout.theta_size):
            accum = 0.0
            for k, forcing in enumerate(forcings):
                if forcing is None:
                    continue
                accum += bundle.sensitivities[p][k].dot(forcing)
            theta_grad[p] = accum
        return theta_grad

    def _zero_state_boundary_gradient(self, grad: PETSc.Vec) -> PETSc.Vec:
        from ..utils import get_boundary_dofs

        layout = self._control_layout()
        if layout.state_size == 0:
            return grad

        bc_dof_indices = None
        if hasattr(self.forward_model, 'solver') and hasattr(self.forward_model, 'problem'):
            V = self.forward_model.solver.V
            mesh = self.forward_model.problem.mesh
            boundary_dofs = get_boundary_dofs(V, mesh)
            bc_dof_indices = set(boundary_dofs.tolist())
        elif hasattr(self.forward_model, 'V') and hasattr(self.forward_model, 'mesh'):
            V = self.forward_model.V
            mesh = self.forward_model.mesh
            boundary_dofs = get_boundary_dofs(V, mesh)
            bc_dof_indices = set(boundary_dofs.tolist())

        if bc_dof_indices:
            grad_arr = grad.getArray()
            for dof in bc_dof_indices:
                if dof < layout.state_size:
                    grad_arr[dof] = 0.0
            grad.setArray(grad_arr)
        return grad


class AugmentedFourDVarCost(_AugmentedGradientMixin, FourDVarCost):
    """Standard 4D-Var over packed control c = [u0; theta]."""

    def gradient(self, m: PETSc.Vec) -> PETSc.Vec:
        trajectory, jacobians = self._run_forward_model(m, store_jacobians=True)

        delta_m = m.duplicate()
        delta_m.waxpy(-1.0, self.m_b, m)
        grad_background = self.B.apply_inverse(delta_m)

        state_grad = self._solve_adjoint(trajectory, jacobians)
        obs_forcings = self._compute_observation_forcings(trajectory)
        theta_grad = self._parameter_gradient_from_forcings(
            m,
            obs_forcings,
            trajectory=trajectory,
            jacobians=jacobians,
        )

        grad = self._control_layout().combine_gradient(
            background_gradient=grad_background,
            state_gradient=state_grad,
            theta_gradient=theta_grad,
        )
        return self._zero_state_boundary_gradient(grad)

    def value_gradient(self, m: PETSc.Vec) -> Tuple[float, PETSc.Vec]:
        try:
            trajectory, jacobians = self._run_forward_model(m, store_jacobians=True)
        except Exception:
            zero_grad = m.duplicate()
            zero_grad.zeroEntries()
            return float("inf"), zero_grad

        background_term = self._compute_background_term(m)
        observation_term = self._compute_observation_term(trajectory)
        cost = background_term + observation_term

        delta_m = m.duplicate()
        delta_m.waxpy(-1.0, self.m_b, m)
        grad_background = self.B.apply_inverse(delta_m)
        state_grad = self._solve_adjoint(trajectory, jacobians)
        obs_forcings = self._compute_observation_forcings(trajectory)
        theta_grad = self._parameter_gradient_from_forcings(
            m,
            obs_forcings,
            trajectory=trajectory,
            jacobians=jacobians,
        )

        grad = self._control_layout().combine_gradient(
            background_gradient=grad_background,
            state_gradient=state_grad,
            theta_gradient=theta_grad,
        )
        grad = self._zero_state_boundary_gradient(grad)

        # Log blockwise gradient norms for diagnostics.
        # Without these, it is impossible to diagnose state/parameter scaling issues.
        layout = self._control_layout()
        if layout.theta_size > 0:
            bg_arr = grad_background.getArray(readonly=True)
            bg_state_norm = float(np.linalg.norm(bg_arr[:layout.state_size]))
            bg_theta_norm = float(np.linalg.norm(bg_arr[layout.state_size:]))
            state_obs_norm = float(state_grad.norm()) if state_grad is not None else 0.0
            theta_obs_norm = float(np.linalg.norm(theta_grad)) if theta_grad is not None else 0.0
            self._profile_event(
                "augmented_4dvar.gradient_blocks",
                background_state_norm=bg_state_norm,
                background_theta_norm=bg_theta_norm,
                adjoint_state_norm=state_obs_norm,
                parameter_obs_norm=theta_obs_norm,
                total_grad_norm=float(grad.norm()),
            )

        return cost, grad


class AugmentedDCFourDVarCost(_AugmentedGradientMixin, DCFourDVarCost):
    """DC-4DVar over packed control c = [u0; theta]."""

    def gradient(self, m: PETSc.Vec) -> PETSc.Vec:
        grad_standard = AugmentedFourDVarCost.gradient(self, m)
        grad_predictability = self._compute_predictability_gradient(m)
        grad_standard.axpy(-1.0, grad_predictability)
        return self._zero_state_boundary_gradient(grad_standard)

    def value_gradient(self, m: PETSc.Vec) -> Tuple[float, PETSc.Vec]:
        J_standard, grad_standard = AugmentedFourDVarCost.value_gradient(self, m)
        if not np.isfinite(J_standard):
            return J_standard, grad_standard

        predictability = self._compute_predictability_term(m)
        grad_predictability = self._compute_predictability_gradient(m)
        grad_standard.axpy(-1.0, grad_predictability)
        grad_standard = self._zero_state_boundary_gradient(grad_standard)
        return J_standard - predictability, grad_standard


class AugmentedDCWMEFourDVarCost(DCWMEFourDVarCost):
    """DC-WME over packed controls.

    The heavy lifting is delegated to the augmented WME QoI linearization,
    which returns packed control adjoints and therefore makes the existing
    DC-WME machinery valid in the augmented control space.
    """

    pass

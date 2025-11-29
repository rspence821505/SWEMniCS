"""
PETSc TAO (Toolkit for Advanced Optimization) wrapper.

Provides interface between SWEMniCS cost functions and PETSc's TAO
optimization library. Supports various TAO algorithms including L-BFGS,
bounded L-BFGS, trust region methods, and more.

References:
    Munson, T., et al. (2014). TAO 2.0 Users Manual.
    Argonne National Laboratory Technical Report ANL/MCS-TM-322.
"""

from typing import Optional, Dict, Callable
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI

from .optimizer_base import Optimizer


class PETScTAOWrapper(Optimizer):
    """
    Wrapper for PETSc TAO optimization algorithms.

    Bridges SWEMniCS cost function interface with TAO's callback system.
    TAO handles the optimization loop and convergence monitoring internally.

    Supported TAO types:
        - 'lmvm': Limited-memory variable metric (L-BFGS)
        - 'blmvm': Bounded L-BFGS with box constraints
        - 'nls': Newton line search
        - 'ntr': Newton trust region
        - 'cg': Nonlinear conjugate gradient
        - 'nm': Nelder-Mead (derivative-free)

    Key advantages over custom implementation:
        - Production-grade L-BFGS-B with full active set handling
        - Sophisticated line search and trust region methods
        - Automatic convergence monitoring and diagnostics
        - Battle-tested robustness on ill-conditioned problems
    """

    def __init__(
        self,
        cost_function,
        tao_type: str = "lmvm",
        lower_bounds: Optional[PETSc.Vec] = None,
        upper_bounds: Optional[PETSc.Vec] = None,
        options: Optional[Dict] = None,
    ):
        """
        Initialize TAO optimizer.

        Args:
            cost_function: Cost function with value() and gradient() methods
            tao_type: TAO algorithm type (see class docstring)
            lower_bounds: Lower bounds for box constraints (None = unbounded)
            upper_bounds: Upper bounds for box constraints (None = unbounded)
            options: Optimizer options:
                - max_iterations: Maximum iterations (default: 100)
                - gradient_tolerance: Gradient norm tolerance (default: 1e-6)
                - cost_tolerance: Function value tolerance (default: 1e-8)
                - verbose: Print iteration info (default: False)
                - tao_monitor: Use TAO's built-in monitor (default: False)
                - Additional TAO-specific options can be passed with 'tao_' prefix
        """
        super().__init__(cost_function, options)

        self.tao_type = tao_type
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.comm = MPI.COMM_WORLD

        # TAO solver object (created in solve())
        self.tao = None

        # Convergence tracking
        self.verbose = self.options.get("verbose", False)
        self.use_tao_monitor = self.options.get("tao_monitor", False)

        # Function evaluation counters
        self.n_func_evals = 0
        self.n_grad_evals = 0

    def solve(self, x0: PETSc.Vec) -> PETSc.Vec:
        """
        Minimize cost function using TAO.

        Args:
            x0: Initial guess

        Returns:
            Optimal solution
        """
        # Create TAO solver
        self.tao = PETSc.TAO().create(comm=self.comm)
        self.tao.setType(self.tao_type)

        # Set initial guess
        x = x0.copy()
        self.tao.setInitialVector(x)

        # Set objective and gradient callbacks
        self.tao.setObjectiveGradient(self._compute_objective_gradient)

        # Set bounds if provided
        if self.lower_bounds is not None or self.upper_bounds is not None:
            self.tao.setVariableBounds(self.lower_bounds, self.upper_bounds)

        # Configure convergence tolerances
        fatol = self.options.get("cost_tolerance", 1e-8)
        frtol = self.options.get("cost_relative_tolerance", 1e-12)
        gatol = self.options.get("gradient_tolerance", 1e-6)
        grtol = self.options.get("gradient_relative_tolerance", 1e-8)
        gttol = self.options.get("gradient_ttolerance", 0.0)  # Trust region specific

        self.tao.setTolerances(
            gatol=gatol,
            grtol=grtol,
            gttol=gttol,
        )

        # Set maximum iterations
        max_iter = self.options.get("max_iterations", 100)
        self.tao.setMaximumIterations(max_iter)

        # Apply additional TAO-specific options
        self._apply_tao_options()

        # Set up monitoring
        if self.use_tao_monitor:
            self.tao.setMonitor(self._tao_monitor_callback)
        elif self.verbose:
            self.tao.setMonitor(self._custom_monitor_callback)

        # Solve the optimization problem
        if self.verbose and self.comm.rank == 0:
            print(f"\n{'='*70}")
            print(f"PETSc TAO Optimization (type={self.tao_type})")
            print(f"{'='*70}")
            if not self.use_tao_monitor:
                print(f"{'Iter':>6} {'Cost':>15} {'||grad||':>15} {'Step':>12}")
                print(f"{'-'*70}")

        self.tao.solve()

        # Extract solution
        solution = self.tao.getSolution()

        # Get convergence information
        reason = self.tao.getConvergedReason()
        iterations = self.tao.getIterationNumber()

        self.converged = reason > 0
        self.iteration = iterations

        if self.verbose and self.comm.rank == 0:
            self._print_convergence_info(reason, iterations)

        # Clean up TAO object
        self.tao.destroy()
        self.tao = None

        return solution

    def _compute_objective_gradient(
        self, tao: PETSc.TAO, x: PETSc.Vec, g: PETSc.Vec
    ) -> float:
        """
        TAO callback for objective and gradient evaluation.

        TAO requires combined function that returns objective value
        and fills gradient vector.

        Args:
            tao: TAO solver object
            x: Current iterate
            g: Gradient vector (output)

        Returns:
            Objective function value
        """
        # Compute objective
        f = self.cost_function.value(x)
        self.n_func_evals += 1

        # Compute gradient
        grad = self.cost_function.gradient(x)
        g.copy(grad)
        grad.destroy()
        self.n_grad_evals += 1

        return f

    def _tao_monitor_callback(self, tao: PETSc.TAO):
        """
        TAO's built-in monitor callback.

        Displays iteration info using TAO's default format.
        """
        # TAO handles printing internally
        pass

    def _custom_monitor_callback(self, tao: PETSc.TAO):
        """
        Custom monitor callback for consistent output format.

        Matches the output style of L-BFGS optimizer.
        """
        if self.comm.rank != 0:
            return

        iteration = tao.getIterationNumber()
        f = tao.getFunctionValue()
        grad = tao.getGradient()
        gnorm = grad.norm()

        # Get step size (if available)
        step = "N/A"
        try:
            step_vec = tao.getSolution()
            if hasattr(tao, "getStepSize"):
                step_size = tao.getStepSize()
                step = f"{step_size:.5e}"
        except:
            pass

        print(f"{iteration:6d} {f:15.8e} {gnorm:15.8e} {step:>12}")

        # Record for convergence history
        self.record_iteration(None, f, gnorm)  # x not available in callback

    def _apply_tao_options(self):
        """
        Apply additional TAO-specific options.

        Options with 'tao_' prefix are passed directly to TAO.
        """
        for key, value in self.options.items():
            if key.startswith("tao_"):
                # Remove prefix and set option
                tao_key = key[4:]  # Remove 'tao_' prefix

                # Convert to TAO option format
                option_name = f"-tao_{tao_key}"

                if isinstance(value, bool):
                    if value:
                        PETSc.Options().setValue(option_name, None)
                elif isinstance(value, (int, float, str)):
                    PETSc.Options().setValue(option_name, str(value))

        # Set from options database
        self.tao.setFromOptions()

    def _print_convergence_info(self, reason: int, iterations: int):
        """
        Print convergence information.

        Args:
            reason: TAO convergence reason code
            iterations: Number of iterations performed
        """
        print(f"\n{'='*70}")

        if reason > 0:
            print(f"✓ TAO Converged after {iterations} iterations")
            print(
                f"  Convergence reason: {self._get_convergence_reason_string(reason)}"
            )
        else:
            print(f"✗ TAO did not converge after {iterations} iterations")
            print(
                f"  Termination reason: {self._get_convergence_reason_string(reason)}"
            )

        print(f"\nFunction evaluations: {self.n_func_evals}")
        print(f"Gradient evaluations: {self.n_grad_evals}")
        print(f"{'='*70}\n")

    @staticmethod
    def _get_convergence_reason_string(reason: int) -> str:
        """
        Convert TAO convergence reason code to string.

        Args:
            reason: TAO convergence reason code

        Returns:
            Human-readable description
        """
        # Positive reasons indicate convergence
        if reason == 3:
            return "GATOL - ||gradient|| < gatol"
        elif reason == 4:
            return "GRTOL - ||gradient||/f < grtol"
        elif reason == 5:
            return "GTTOL - ||gradient|| < gttol (trust region)"
        elif reason == 6:
            return "STEPTOL - step size small"
        elif reason == 7:
            return "MINF - f < minf"
        elif reason == 8:
            return "USER - user-defined convergence"

        # Negative reasons indicate termination without convergence
        elif reason == -2:
            return "MAXITS - maximum iterations reached"
        elif reason == -4:
            return "NAN - NaN or Inf detected"
        elif reason == -5:
            return "MAXFCN - maximum function evaluations"
        elif reason == -6:
            return "LS_FAILURE - line search failure"
        elif reason == -7:
            return "TR_REDUCTION - trust region too small"
        elif reason == -8:
            return "USER_TERMINATE - user requested termination"
        else:
            return f"Unknown reason code: {reason}"

    def get_convergence_info(self) -> Dict:
        """
        Get convergence information.

        Returns:
            Dictionary with convergence data
        """
        info = super().get_convergence_info()
        info.update(
            {
                "tao_type": self.tao_type,
                "n_func_evals": self.n_func_evals,
                "n_grad_evals": self.n_grad_evals,
            }
        )
        return info


class TAOOptimizerFactory:
    """
    Factory for creating TAO optimizers with common configurations.

    Provides convenience methods for typical optimization scenarios.
    """

    @staticmethod
    def create_lbfgs(
        cost_function, memory_size: int = 10, options: Optional[Dict] = None
    ) -> PETScTAOWrapper:
        """
        Create TAO L-BFGS optimizer (unconstrained).

        Args:
            cost_function: Cost function to minimize
            memory_size: L-BFGS memory size
            options: Additional options

        Returns:
            Configured TAO optimizer
        """
        opts = options or {}
        opts["tao_lmvm_hist_size"] = memory_size

        return PETScTAOWrapper(cost_function, tao_type="lmvm", options=opts)

    @staticmethod
    def create_bounded_lbfgs(
        cost_function,
        lower_bounds: Optional[PETSc.Vec] = None,
        upper_bounds: Optional[PETSc.Vec] = None,
        memory_size: int = 10,
        options: Optional[Dict] = None,
    ) -> PETScTAOWrapper:
        """
        Create TAO bounded L-BFGS optimizer (box constraints).

        Args:
            cost_function: Cost function to minimize
            lower_bounds: Lower bounds
            upper_bounds: Upper bounds
            memory_size: L-BFGS memory size
            options: Additional options

        Returns:
            Configured TAO optimizer
        """
        opts = options or {}
        opts["tao_blmvm_hist_size"] = memory_size

        return PETScTAOWrapper(
            cost_function,
            tao_type="blmvm",
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            options=opts,
        )

    @staticmethod
    def create_trust_region(
        cost_function, options: Optional[Dict] = None
    ) -> PETScTAOWrapper:
        """
        Create TAO Newton trust region optimizer.

        Requires Hessian or uses finite differences.

        Args:
            cost_function: Cost function to minimize
            options: Additional options

        Returns:
            Configured TAO optimizer
        """
        return PETScTAOWrapper(cost_function, tao_type="ntr", options=options)

    @staticmethod
    def create_conjugate_gradient(
        cost_function,
        cg_type: str = "pr",  # Polak-Ribiere
        options: Optional[Dict] = None,
    ) -> PETScTAOWrapper:
        """
        Create TAO nonlinear conjugate gradient optimizer.

        Args:
            cost_function: Cost function to minimize
            cg_type: CG variant ('pr', 'fr', 'prp', 'dy', 'hs')
            options: Additional options

        Returns:
            Configured TAO optimizer
        """
        opts = options or {}
        opts["tao_cg_type"] = cg_type

        return PETScTAOWrapper(cost_function, tao_type="cg", options=opts)

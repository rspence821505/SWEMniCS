"""
PETSc TAO (Toolkit for Advanced Optimization) wrapper.

Provides interface between SWE4DVar cost functions and PETSc's TAO
optimization library. Supports various TAO algorithms including L-BFGS,
bounded L-BFGS, trust region methods, and more.

References:
    Munson, T., et al. (2014). TAO 2.0 Users Manual.
    Argonne National Laboratory Technical Report ANL/MCS-TM-322.
"""

from typing import Optional, Dict, Callable
import time
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI

from .optimizer_base import Optimizer


class PETScTAOWrapper(Optimizer):
    """
    Wrapper for PETSc TAO optimization algorithms.

    Bridges SWE4DVar cost function interface with TAO's callback system.
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
        self.runtime_profiler = self.options.get("runtime_profiler")
        self.runtime_profile_label = self.options.get("runtime_profile_label", "tao")

        # Function evaluation counters
        self.n_func_evals = 0
        self.n_grad_evals = 0
        self._max_funcs = self.options.get("max_funcs", None)
        self._last_grad = None

        # Failure tracking. `_failures_since_accept` is reset ONLY when the TAO
        # monitor fires (true iteration boundary), not on a backtracked line-
        # search eval that happens to converge Newton. This prevents the
        # silent-hang pattern where Newton flips between fail/pass during
        # Armijo backtracking and the counter never reaches the bailout.
        self._failures_since_accept = 0
        self._abort_after_n_failures = int(
            self.options.get("abort_after_n_failures", 6)
        )
        # Δm diagnostics: store last accepted-iterate array for norm reporting.
        self._last_accepted_x_arr = None

    def solve(self, x0: PETSc.Vec) -> PETSc.Vec:
        """
        Minimize cost function using TAO.

        Args:
            x0: Initial guess

        Returns:
            Optimal solution
        """
        self._profile("solve_start", tao_type=self.tao_type)
        # Create TAO solver
        self.tao = PETSc.TAO().create(comm=self.comm)
        self.tao.setType(self.tao_type)

        # Set initial guess
        x = x0.copy()
        # API changed: setInitialVector -> setSolution in newer PETSc
        if hasattr(self.tao, 'setInitial'):
            self.tao.setInitial(x)
        elif hasattr(self.tao, 'setSolution'):
            self.tao.setSolution(x)
        else:
            # Fallback: set via solving with x as initial
            self.tao.setSolution(x)

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

        # Set maximum iterations and function evaluations
        max_iter = self.options.get("max_iterations", 100)
        self.tao.setMaximumIterations(max_iter)
        max_funcs = self.options.get("max_funcs", None)
        if max_funcs is not None:
            self.tao.setMaximumFunctionEvaluations(max_funcs)

        # Configure line search
        # Default More-Thuente requires Wolfe conditions which can fail on
        # flat cost surfaces (e.g., 4D-Var with strong background penalty).
        # Armijo (sufficient decrease only) is more robust for these cases.
        ls_type = self.options.get("line_search_type", "armijo")
        ls_max_funcs = self.options.get("line_search_max_funcs", 30)
        ls_initial_step = self.options.get("line_search_initial_step", None)
        # Hard cap on TAO line-search step length, plumbed to PETSc option
        # `-tao_ls_stepmax`. Useful for globalizing quasi-Newton methods
        # (BLMVM) when the 2nd+ iteration extrapolates into regions that
        # crash the forward Newton solver — specifically, the eval-3 failure
        # pattern observed on idealized-inlet 4D-Var at obs ≥ 0.02. Setting
        # it conservatively lets Armijo backtrack FROM a bounded starting
        # length instead of from the unrestricted quasi-Newton proposal.
        ls_step_max = self.options.get("line_search_step_max", None)
        ls_step_min = self.options.get("line_search_step_min", None)
        try:
            ls = self.tao.getLineSearch()
            if ls is not None:
                ls.setType(ls_type)
                ls.setMaximumFunctionEvaluations(ls_max_funcs)
                if ls_initial_step is not None:
                    ls.setInitialStepLength(ls_initial_step)
                if ls_step_max is not None:
                    # Route through the PETSc options database because
                    # petsc4py's TaoLineSearch does not expose a direct
                    # setStepMax() wrapper across versions.
                    PETSc.Options().setValue("-tao_ls_stepmax", str(float(ls_step_max)))
                if ls_step_min is not None:
                    PETSc.Options().setValue("-tao_ls_stepmin", str(float(ls_step_min)))
        except Exception:
            pass  # Line search config not supported for all TAO types

        # Concise line-search activation log so run output records exactly
        # what globalization settings are active.
        if self.comm.rank == 0:
            parts = [f"type={ls_type}", f"max_funcs={ls_max_funcs}"]
            if ls_initial_step is not None:
                parts.append(f"init_step={ls_initial_step}")
            if ls_step_max is not None:
                parts.append(f"stepmax={ls_step_max}")
            if ls_step_min is not None:
                parts.append(f"stepmin={ls_step_min}")
            print(f"  [TAO line-search] {'  '.join(parts)}", flush=True)

        # Apply additional TAO-specific options (calls setFromOptions())
        self._apply_tao_options()

        # Re-apply max iterations and tolerances AFTER setFromOptions(),
        # which may override programmatic settings from the options database
        self.tao.setMaximumIterations(max_iter)
        if max_funcs is not None:
            self.tao.setMaximumFunctionEvaluations(max_funcs)
        self.tao.setTolerances(gatol=gatol, grtol=grtol, gttol=gttol)

        # Set up monitoring. Even when verbose output is disabled, we still attach
        # the custom monitor if iteration_callback is present so per-iteration
        # diagnostics are recorded in convergence_history.
        if self.iteration_callback is not None:
            self.tao.setMonitor(self._custom_monitor_callback)
        elif self.use_tao_monitor:
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
        self._profile(
            "solve_end",
            converged=bool(self.converged),
            reason=int(reason),
            iterations=int(iterations),
            n_func_evals=int(self.n_func_evals),
            n_grad_evals=int(self.n_grad_evals),
        )

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

        Uses the efficient value_gradient() method when available, which
        computes both value and gradient in a single forward/adjoint pass.
        This avoids the double forward model execution that would occur
        if value() and gradient() were called separately.

        Args:
            tao: TAO solver object
            x: Current iterate
            g: Gradient vector (output)

        Returns:
            Objective function value
        """
        eval_id = self.n_func_evals + 1
        callback_start = time.perf_counter()
        self._profile(
            "objective_gradient_start",
            eval_id=int(eval_id),
            iter=int(tao.getIterationNumber()),
        )
        try:
            # Max_funcs is enforced in the monitor callback (where
            # setConvergedReason is respected). Here we just short-circuit
            # to avoid expensive forward solves after the limit is hit.
            # TAO will still call us a few more times before checking the
            # monitor, but these calls are instant (no forward model).
            if self._max_funcs is not None and self.n_func_evals >= self._max_funcs:
                if hasattr(self, '_last_grad') and self._last_grad is not None:
                    self._last_grad.copy(g)
                else:
                    g.set(0.0)
                return self._last_cost if hasattr(self, '_last_cost') else 0.0

            # Diagnostic: log entry to obj_grad callback so we can see if TAO
            # is calling us (vs. spinning in internal BLMVM work).
            if self.verbose and self.comm.rank == 0:
                print(f"  [TAO callback] entering eval #{self.n_func_evals + 1}",
                      flush=True)

            # Use efficient combined method if available (avoids double forward solve)
            if hasattr(self.cost_function, 'value_gradient'):
                f, grad = self.cost_function.value_gradient(x)

                self.n_func_evals += 1
                self.n_grad_evals += 1

                # Δm diagnostic: norm of step from last ACCEPTED iterate (monitor).
                # Tells us whether TAO is backtracking hard (Δm shrinking) or
                # still trying aggressive steps (Δm stable).
                dm_norm_str = ""
                if self._last_accepted_x_arr is not None:
                    try:
                        dx = x.getArray(readonly=True) - self._last_accepted_x_arr
                        local_sq = float(np.dot(dx, dx))
                        global_sq = self.comm.allreduce(local_sq)
                        dm_norm_str = f"  ||Δm from last accept||={float(global_sq)**0.5:.3e}"
                    except Exception:
                        pass

                # Check for infinity (forward model failure)
                if not np.isfinite(f):
                    self._failures_since_accept += 1
                    fail_msg = getattr(
                        self.cost_function, "_last_forward_failure_msg", None
                    )
                    fail_str = f"  [fwd: {fail_msg}]" if fail_msg else ""
                    if self.verbose and self.comm.rank == 0:
                        print(f"  [TAO callback] eval #{self.n_func_evals}: cost=inf "
                              f"(forward model failure, "
                              f"since_accept={self._failures_since_accept}/"
                              f"{self._abort_after_n_failures}){dm_norm_str}{fail_str}",
                              flush=True)
                    # CRITICAL FIX: Don't set gradient to zero! TAO will think it converged.
                    # Instead, return background penalty gradient to push back toward m_b
                    if hasattr(self.cost_function, 'compute_background_gradient'):
                        grad_b = self.cost_function.compute_background_gradient(x)
                        grad_b.copy(g)
                        grad_b.destroy()
                    else:
                        # Fallback: set small non-zero gradient to prevent TAO from thinking it converged
                        g.set(1e-10)
                    # Hard-abort after N failures since last accepted iterate.
                    # Sending DIVERGED_USER first lets TAO unwind its line search,
                    # but BLMVM has been observed to ignore it mid-Armijo. After
                    # 2 more calls with no acceptance, hard-abort the MPI job so
                    # the best-iterate checkpoint saved by the driver is used.
                    if self._failures_since_accept >= self._abort_after_n_failures:
                        if self.comm.rank == 0:
                            print(f"  [TAO callback] {self._failures_since_accept} "
                                  f"forward failures since last accepted iterate — "
                                  f"hard-aborting to preserve best-iterate checkpoint",
                                  flush=True)
                        # Give caller a chance to save best iterate via exit
                        # handler; then MPI_Abort to terminate cleanly.
                        self.comm.Abort(42)
                    elif self._failures_since_accept >= max(3, self._abort_after_n_failures - 2):
                        if self.verbose and self.comm.rank == 0:
                            print(f"  [TAO callback] requesting TAO termination "
                                  f"(DIVERGED_USER)", flush=True)
                        tao.setConvergedReason(
                            PETSc.TAO.ConvergedReason.DIVERGED_USER
                        )
                    return 1e20
                # Finite cost — DO NOT reset _failures_since_accept here. That
                # happens only when the TAO monitor fires (true acceptance).

                self._last_cost = f
                self._last_grad = grad.copy()
                gnorm = grad.norm()
                self._profile(
                    "objective_gradient_end",
                    eval_id=int(eval_id),
                    iter=int(tao.getIterationNumber()),
                    cost=float(f),
                    grad_norm=float(gnorm),
                    duration_s=float(time.perf_counter() - callback_start),
                    mode="value_gradient",
                )
                if self.verbose and self.comm.rank == 0:
                    print(f"  [TAO callback] eval #{self.n_func_evals}: cost={f:.6f}, ||grad||={gnorm:.4e}",
                          flush=True)

                # Copy gradient values into output vector g
                # PETSc copy: source.copy(dest) copies FROM source TO dest
                grad.copy(g)
                grad.destroy()

                return f
            else:
                # Fallback to separate calls (less efficient, but works with any cost function)
                f = self.cost_function.value(x)
                self.n_func_evals += 1

                # Check for infinity (forward model failure)
                if not np.isfinite(f):
                    # CRITICAL FIX: Don't set gradient to zero! TAO will think it converged.
                    # Instead, return background penalty gradient to push back toward m_b
                    if hasattr(self.cost_function, 'compute_background_gradient'):
                        grad_b = self.cost_function.compute_background_gradient(x)
                        grad_b.copy(g)
                        grad_b.destroy()
                    else:
                        # Fallback: set small non-zero gradient to prevent TAO from thinking it converged
                        g.set(1e-10)
                    return 1e20

                # Compute gradient
                grad = self.cost_function.gradient(x)
                # Copy gradient values into output vector g
                grad.copy(g)
                grad.destroy()
                self.n_grad_evals += 1
                self._profile(
                    "objective_gradient_end",
                    eval_id=int(eval_id),
                    iter=int(tao.getIterationNumber()),
                    cost=float(f),
                    grad_norm=float(g.norm()),
                    duration_s=float(time.perf_counter() - callback_start),
                    mode="separate",
                )

                return f
        except Exception as e:
            # Forward model failed - return large cost and background gradient
            import warnings
            warnings.warn(
                f"TAO callback: forward model failed: {e}. "
                "Returning large cost to signal bad point.",
                RuntimeWarning,
                stacklevel=2
            )
            # CRITICAL FIX: Don't set gradient to zero! TAO will think it converged.
            if hasattr(self.cost_function, 'compute_background_gradient'):
                grad_b = self.cost_function.compute_background_gradient(x)
                grad_b.copy(g)
                grad_b.destroy()
            else:
                # Fallback: set small non-zero gradient to prevent TAO from thinking it converged
                g.set(1e-10)
            self._profile(
                "objective_gradient_error",
                eval_id=int(eval_id),
                iter=int(tao.getIterationNumber()),
                duration_s=float(time.perf_counter() - callback_start),
                error=str(e),
            )
            return 1e20

    def _tao_monitor_callback(self, tao: PETSc.TAO):
        """
        TAO's built-in monitor callback.

        Displays iteration info using TAO's default format.
        """
        f = tao.getFunctionValue()
        grad = tao.getGradient()
        if isinstance(grad, tuple):
            grad = grad[0]
        gnorm = grad.norm() if grad is not None else 0.0
        x = tao.getSolution() if hasattr(tao, "getSolution") else None
        self.record_iteration(x, f, gnorm)
        self._profile(
            "monitor",
            iter=int(tao.getIterationNumber()),
            cost=float(f),
            grad_norm=float(gnorm),
        )

    def _custom_monitor_callback(self, tao: PETSc.TAO):
        """
        Custom monitor callback for consistent output format.

        Matches the output style of L-BFGS optimizer.
        Also enforces max_funcs: TAO BLMVM ignores
        setMaximumFunctionEvaluations, so we check here and set
        the converged reason. The monitor runs after each TAO
        iteration (not each line-search eval), where
        setConvergedReason is respected.
        """
        iteration = tao.getIterationNumber()
        f = tao.getFunctionValue()
        grad = tao.getGradient()
        # getGradient may return (grad_vec, grad_vec) tuple in newer PETSc
        if isinstance(grad, tuple):
            grad = grad[0]
        gnorm = grad.norm() if grad is not None else 0.0
        x = tao.getSolution() if hasattr(tao, "getSolution") else None

        # True iteration boundary — TAO has accepted this iterate. Reset the
        # since-last-accept failure counter and snapshot x for Δm diagnostics.
        self._failures_since_accept = 0
        if x is not None:
            try:
                self._last_accepted_x_arr = x.getArray(readonly=True).copy()
            except Exception:
                pass

        # Record for convergence history before any rank-0-only printing.
        self.record_iteration(x, f, gnorm)
        self._profile(
            "monitor",
            iter=int(iteration),
            cost=float(f),
            grad_norm=float(gnorm),
        )

        # Enforce max_funcs from the monitor (reliable, unlike the callback).
        if self._max_funcs is not None and self.n_func_evals >= self._max_funcs:
            if self.comm.rank == 0:
                print(f"  [TAO monitor] max_funcs={self._max_funcs} reached at iter {iteration}, stopping",
                      flush=True)
            tao.setConvergedReason(PETSc.TAO.ConvergedReason.CONVERGED_USER)
            return

        if self.comm.rank != 0:
            return

        # Get step size (if available)
        step = "N/A"
        try:
            if hasattr(tao, "getStepSize"):
                step_size = tao.getStepSize()
                step = f"{step_size:.5e}"
        except Exception:
            pass

        if self.verbose:
            print(f"{iteration:6d} {f:15.8e} {gnorm:15.8e} {step:>12}")

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

    def _profile(self, event: str, **payload):
        if self.runtime_profiler is None:
            return
        try:
            self.runtime_profiler.log_event(
                f"{self.runtime_profile_label}.{event}",
                **payload,
            )
        except Exception:
            pass

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

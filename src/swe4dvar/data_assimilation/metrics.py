"""
Metrics for evaluating data assimilation performance.

Implements:
- RMSE (Root Mean Square Error)
- Innovation statistics
- Degrees of Freedom for Signal (DFS)
- A-posteriori error estimates
"""

import petsc4py.PETSc as PETSc
import numpy as np


class DAMetrics:
    """
    Collection of metrics for evaluating DA performance.

    Computes standard diagnostics for 4D-Var assimilation.
    """

    def __init__(self, observations, observation_operators):
        """
        Initialize metrics calculator.

        Args:
            observations: Dict[int, PETSc.Vec] of observations
            observation_operators: Dict[int, ObservationOperator]
        """
        self.observations = observations
        self.obs_operators = observation_operators

    def compute_rmse(self, trajectory):
        """
        Compute Root Mean Square Error against observations.

        Args:
            trajectory: List of state vectors from forward solve

        Returns:
            dict: RMSE at each observation time
        """
        rmse = {}

        for time_idx, obs in self.observations.items():
            if time_idx >= len(trajectory):
                continue

            # Get observation operator for this time
            H = self.obs_operators[time_idx]

            # Apply observation operator to state: H·x
            state = trajectory[time_idx]
            Hx = H.apply(state)

            # Compute difference: obs - H·x
            diff = obs.duplicate()
            diff.copy(obs)
            diff.axpy(-1.0, Hx)  # diff = obs - Hx

            # Compute RMSE: sqrt(||diff||²/n_obs)
            n_obs = obs.getSize()
            mse = diff.dot(diff) / n_obs
            rmse[time_idx] = np.sqrt(mse)

            # Clean up
            Hx.destroy()
            diff.destroy()

        return rmse

    def compute_data_misfit(self, trajectory):
        """
        Compute data misfit term of cost function.

        Args:
            trajectory: List of state vectors from forward solve

        Returns:
            float: Total data misfit
        """
        total_misfit = 0.0

        for time_idx, obs in self.observations.items():
            if time_idx >= len(trajectory):
                continue

            # Get observation operator for this time
            H = self.obs_operators[time_idx]

            # Apply observation operator to state: H·x
            state = trajectory[time_idx]
            Hx = H.apply(state)

            # Compute innovation: d = obs - H·x
            innovation = obs.duplicate()
            innovation.copy(obs)
            innovation.axpy(-1.0, Hx)  # innovation = obs - Hx

            # Compute R⁻¹·d (assuming R is stored in observation operator)
            # For now, assume identity or diagonal R
            # misfit = 1/2 * d^T · R⁻¹ · d
            misfit = 0.5 * innovation.dot(innovation)
            total_misfit += misfit

            # Clean up
            Hx.destroy()
            innovation.destroy()

        return total_misfit

    def compute_bias(self, trajectory) -> float:
        """
        Compute mean bias.

        Bias = (1/n) Σᵢ (xᵢ - xᵢ_true)

        Args:
            analysis: Analysis/forecast state
            truth: True state

        Returns:
            Mean bias
        """
        total_bias = 0.0
        total_count = 0

        for time_idx, obs in self.observations.items():
            if time_idx >= len(trajectory):
                continue

            # Get observation operator for this time
            H = self.obs_operators[time_idx]

            # Apply observation operator to state: H·x
            state = trajectory[time_idx]
            Hx = H.apply(state)

            # Compute bias: H·x - obs (model minus observations)
            bias_vec = Hx.duplicate()
            bias_vec.copy(Hx)
            bias_vec.axpy(-1.0, obs)  # bias_vec = Hx - obs

            # Sum the bias components
            bias_array = bias_vec.getArray()
            total_bias += np.sum(bias_array)
            total_count += bias_vec.getSize()

            # Clean up
            Hx.destroy()
            bias_vec.destroy()

        # Compute mean bias across all observations
        if total_count > 0:
            return total_bias / total_count
        else:
            return 0.0

    def compute_innovations(self, trajectory):
        """
        Compute innovation statistics (O - B).

        Args:
            trajectory: List of state vectors

        Returns:
            dict: Innovation mean, variance at each time
        """
        innovations = {}

        for time_idx, obs in self.observations.items():
            if time_idx >= len(trajectory):
                continue

            # Get observation operator for this time
            H = self.obs_operators[time_idx]

            # Apply observation operator to state: H·x
            state = trajectory[time_idx]
            Hx = H.apply(state)

            # Compute innovation: d = obs - H·x
            innovation = obs.duplicate()
            innovation.copy(obs)
            innovation.axpy(-1.0, Hx)  # innovation = obs - Hx

            # Get innovation array for statistics
            innov_array = innovation.getArray()

            # Compute mean and variance
            mean = np.mean(innov_array)
            variance = np.var(innov_array)
            std = np.sqrt(variance)

            innovations[time_idx] = {
                "mean": mean,
                "variance": variance,
                "std": std,
                "min": np.min(innov_array),
                "max": np.max(innov_array),
            }

            # Clean up
            Hx.destroy()
            innovation.destroy()

        return innovations

    def compute_dfs(self, hessian_eigenvalues):
        """
        Compute Degrees of Freedom for Signal.

        DFS = Tr(I - H⁻¹B⁻¹) where H is Hessian of J

        Args:
            hessian_eigenvalues: Eigenvalues of Hessian

        Returns:
            float: DFS value
        """
        # DFS can be computed from eigenvalues as:
        # DFS = Σᵢ λᵢ/(1 + λᵢ)
        # where λᵢ are the eigenvalues of B^{1/2} H^T R^{-1} H B^{1/2}

        if hessian_eigenvalues is None or len(hessian_eigenvalues) == 0:
            return 0.0

        eigenvalues = np.array(hessian_eigenvalues)

        # Remove non-positive eigenvalues
        eigenvalues = eigenvalues[eigenvalues > 0]

        # Compute DFS
        dfs = np.sum(eigenvalues / (1.0 + eigenvalues))

        return float(dfs)

    def compute_analysis_error(self, analysis, truth=None):
        """
        Compute a-posteriori analysis error estimate.

        Args:
            analysis: Optimized state
            truth: Optional true state (if available)

        Returns:
            dict: Error statistics
        """
        error_stats = {}

        if truth is not None:
            # Compute error against true state
            error = analysis.duplicate()
            error.copy(analysis)
            error.axpy(-1.0, truth)  # error = analysis - truth

            # Compute error statistics
            error_array = error.getArray()
            error_stats["l2_norm"] = error.norm()
            error_stats["mean_error"] = np.mean(error_array)
            error_stats["std_error"] = np.std(error_array)
            error_stats["max_abs_error"] = np.max(np.abs(error_array))
            error_stats["relative_error"] = error.norm() / truth.norm()

            # Clean up
            error.destroy()
        else:
            # If no truth available, just report analysis statistics
            analysis_array = analysis.getArray()
            error_stats["analysis_norm"] = analysis.norm()
            error_stats["analysis_mean"] = np.mean(analysis_array)
            error_stats["analysis_std"] = np.std(analysis_array)
            error_stats["analysis_min"] = np.min(analysis_array)
            error_stats["analysis_max"] = np.max(analysis_array)

        return error_stats


class CostFunctionHistory:
    """
    Track cost function evolution during optimization.

    Useful for convergence analysis and debugging.
    """

    def __init__(self):
        """Initialize empty history."""
        self.iterations = []
        self.cost_values = []
        self.gradient_norms = []
        self.step_sizes = []

    def record(self, iteration, cost, grad_norm, step_size=None):
        """
        Record one optimization iteration.

        Args:
            iteration: Iteration number
            cost: Cost function value
            grad_norm: Gradient norm
            step_size: Optional line search step size
        """
        self.iterations.append(iteration)
        self.cost_values.append(cost)
        self.gradient_norms.append(grad_norm)
        if step_size is not None:
            self.step_sizes.append(step_size)

    def get_reduction_rate(self):
        """
        Compute cost reduction rate per iteration.

        Returns:
            np.array: Cost reduction percentages
        """
        if len(self.cost_values) < 2:
            return np.array([])
        costs = np.array(self.cost_values)
        return (costs[:-1] - costs[1:]) / costs[:-1] * 100

    def plot(self, filename=None):
        """
        Plot optimization history.

        Args:
            filename: Optional filename to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available, cannot plot")
            return

        if len(self.iterations) == 0:
            print("No data to plot")
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot cost function values
        axes[0].semilogy(self.iterations, self.cost_values, "b-o", linewidth=2)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Cost Function Value")
        axes[0].set_title("Cost Function Evolution")
        axes[0].grid(True, alpha=0.3)

        # Plot gradient norms
        axes[1].semilogy(self.iterations, self.gradient_norms, "r-s", linewidth=2)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Gradient Norm")
        axes[1].set_title("Gradient Norm Evolution")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {filename}")
        else:
            plt.show()

        plt.close()

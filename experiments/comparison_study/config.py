"""
Configuration dataclasses for the comparison study.

This module defines the configuration for the 4D-Var vs DC-WME comparison study,
including base problem setup and sweep parameters for each experiment type.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class ComparisonStudyConfig:
    """Base configuration for the comparison study.

    These parameters define the problem setup and remain constant across
    all experiments within a study.
    """

    # Problem configuration
    nx: int = 20
    ny: int = 10
    dt: float = 1800.0  # 30 minutes
    final_time: float = 172800.0  # 2 days = 96 timesteps
    solver_type: str = "DG"
    p_degree: List[int] = field(default_factory=lambda: [1, 1])

    # Base DA configuration (defaults when not being swept)
    obs_frequency: int = 4
    obs_fraction: float = 0.5
    obs_noise_level: float = 0.001
    background_error_std: float = 0.1
    component_aware_cov: bool = True
    background_correlation_length: float = 2000.0  # Spatial correlation length (m)

    # Cycling 4D-Var
    n_windows: int = 4  # Number of assimilation windows

    # DC-WME L_wme estimation
    l_wme_samples: int = 100  # >0 = analytical L_wme, 0 = 2R fallback
    auto_inflate_B: bool = True  # Auto-inflate B based on Gram matrix bound (paper eq. 38)
    max_inflate_factor: float = 100.0  # Maximum B inflation factor
    predictability_gamma: float = 1.0  # Relaxation γ for predictability condition (paper eq. 36)
    adaptive_gamma: bool = False  # Absolute γ=1.0 is the predictability boundary

    # Physics perturbation defaults
    friction_scale_factor: float = 1.1  # Default to 10% model error
    bathymetry_noise_std: float = 0.0  # No bathymetry perturbation by default

    # Optimization configuration
    max_iterations: int = 50
    gradient_tolerance: float = 1e-10

    # Reproducibility
    obs_seed: int = 42
    background_seed: int = 123
    perturbation_seed: int = 456

    # Output
    output_dir: Path = field(default_factory=lambda: Path("outputs/comparison_study"))

    # Diagnostics
    diagnostic_level: str = "standard"  # "minimal", "standard", or "verbose"

    @property
    def nt(self) -> int:
        """Number of timesteps."""
        return int(self.final_time / self.dt)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "nx": self.nx,
            "ny": self.ny,
            "dt": self.dt,
            "final_time": self.final_time,
            "nt": self.nt,
            "solver_type": self.solver_type,
            "p_degree": self.p_degree,
            "obs_frequency": self.obs_frequency,
            "obs_fraction": self.obs_fraction,
            "obs_noise_level": self.obs_noise_level,
            "background_error_std": self.background_error_std,
            "component_aware_cov": self.component_aware_cov,
            "background_correlation_length": self.background_correlation_length,
            "n_windows": self.n_windows,
            "l_wme_samples": self.l_wme_samples,
            "auto_inflate_B": self.auto_inflate_B,
            "max_inflate_factor": self.max_inflate_factor,
            "predictability_gamma": self.predictability_gamma,
            "adaptive_gamma": self.adaptive_gamma,
            "friction_scale_factor": self.friction_scale_factor,
            "bathymetry_noise_std": self.bathymetry_noise_std,
            "max_iterations": self.max_iterations,
            "gradient_tolerance": self.gradient_tolerance,
            "obs_seed": self.obs_seed,
            "background_seed": self.background_seed,
            "perturbation_seed": self.perturbation_seed,
            "output_dir": str(self.output_dir),
            "diagnostic_level": self.diagnostic_level,
        }


@dataclass
class SweepConfig:
    """Configuration for parameter sweeps.

    Defines the sweep parameters for each experiment type.
    Only one parameter is swept at a time; others use defaults from ComparisonStudyConfig.
    """

    # Methods to compare
    methods: List[str] = field(default_factory=lambda: ["4dvar", "dcwme"])

    # PRIMARY: Friction scale sweep (model error)
    # 1.0 = inverse crime baseline, 1.1-1.2 = 10-20% friction error
    friction_scale_factors: List[float] = field(
        default_factory=lambda: [1.0, 1.1, 1.15, 1.2]
    )

    # Observation frequency sweep (observe every N timesteps)
    obs_frequencies: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 12])

    # Observation fraction sweep (fraction of mesh nodes observed)
    obs_fractions: List[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75]
    )

    # Observation noise sweep (noise as fraction of signal)
    noise_levels: List[float] = field(
        default_factory=lambda: [0.001, 0.01, 0.05, 0.1]
    )

    # Background error sweep
    background_errors: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.5]
    )

    # OPTIONAL: Bathymetry noise sweep (meters, additive noise)
    bathymetry_noises: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])

    # Checkpointing strategies (for 4D-Var only)
    checkpointing_strategies: List[str] = field(
        default_factory=lambda: ["full", "state_only", "binomial"]
    )

    def get_sweep_values(self, sweep_name: str) -> List[Any]:
        """Get sweep values for a given sweep name."""
        sweep_map = {
            "friction": self.friction_scale_factors,
            "obs_freq": self.obs_frequencies,
            "obs_fraction": self.obs_fractions,
            "noise": self.noise_levels,
            "background": self.background_errors,
            "bathymetry": self.bathymetry_noises,
            "checkpointing": self.checkpointing_strategies,
        }
        if sweep_name not in sweep_map:
            raise ValueError(
                f"Unknown sweep: {sweep_name}. Available: {list(sweep_map.keys())}"
            )
        return sweep_map[sweep_name]


# Experiment types available
AVAILABLE_EXPERIMENTS = [
    "friction",  # PRIMARY: Friction scale sweep
    "obs_freq",  # Observation frequency sweep
    "obs_fraction",  # Observation fraction sweep
    "noise",  # Observation noise sweep
    "background",  # Background error sweep
    "bathymetry",  # OPTIONAL: Bathymetry noise sweep
    "checkpointing",  # Checkpointing strategy comparison
]

# Experiments that include model error by default
# (friction_scale_factor > 1.0 when not being swept)
EXPERIMENTS_WITH_MODEL_ERROR = [
    "obs_freq",
    "obs_fraction",
    "noise",
    "background",
]


def get_experiment_description(experiment: str) -> str:
    """Get a description of what each experiment type tests."""
    descriptions = {
        "friction": "Friction scale sweep - tests robustness to model error (PRIMARY)",
        "obs_freq": "Observation frequency sweep - tests temporal coverage impact",
        "obs_fraction": "Observation fraction sweep - tests spatial coverage impact",
        "noise": "Observation noise sweep - tests noise tolerance",
        "background": "Background error sweep - tests initial guess sensitivity",
        "bathymetry": "Bathymetry noise sweep - tests robustness to bathymetry error (OPTIONAL)",
        "checkpointing": "Checkpointing strategy comparison - tests memory/speed tradeoffs",
    }
    return descriptions.get(experiment, "Unknown experiment type")

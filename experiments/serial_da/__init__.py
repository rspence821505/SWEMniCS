"""
Serial Data Assimilation Experiments

This module contains experiments comparing 4D-Var and DC-WME-4DVar
data assimilation methods for shallow water equations.

Experiments:
- tidal_4dvar.py: Tidal case with standard 4D-Var
- tidal_dcwme.py: Tidal case with DC-WME-4DVar
- dam_break_4dvar.py: Dam break case with standard 4D-Var
- dam_break_dcwme.py: Dam break case with DC-WME-4DVar

Utilities:
- da_experiment_utils.py: Common utilities for twin experiments
- analyze_results.py: Comparison analysis and plotting

Usage:
    # Run all experiments
    ./run_serial_experiments.sh

    # Run individual experiments
    python tidal_4dvar.py [--nx 10] [--verbose]
"""

from .da_experiment_utils import (
    DAExperimentConfig,
    DAExperimentResults,
    ForwardModelWrapper,
    generate_observation_points,
    generate_observations,
    generate_background_state,
    compute_rms_error,
    compute_innovation_statistics,
    save_experiment_results,
    load_all_results,
)

__all__ = [
    "DAExperimentConfig",
    "DAExperimentResults",
    "ForwardModelWrapper",
    "generate_observation_points",
    "generate_observations",
    "generate_background_state",
    "compute_rms_error",
    "compute_innovation_statistics",
    "save_experiment_results",
    "load_all_results",
]

"""
Comparison study: 4D-Var vs DC-WME twin experiments.

This module provides tools for running systematic comparisons between
standard 4D-Var and Data-Consistent Weighted Mean Error (DC-WME) 4D-Var
using twin experiments with physics perturbation.

Main components:
- config: Configuration dataclasses for experiments
- diagnostics: Diagnostic data capture and analysis
- runner: Experiment runner with robust error handling
- plotting: Visualization utilities
"""

from .config import ComparisonStudyConfig, SweepConfig
from .diagnostics import ExperimentDiagnostics, classify_failure
from .runner import ComparisonRunner

__all__ = [
    "ComparisonStudyConfig",
    "SweepConfig",
    "ExperimentDiagnostics",
    "classify_failure",
    "ComparisonRunner",
]

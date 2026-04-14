"""Registry-driven twin experiment framework."""

from .specs import TwinExperimentSpec
from .registry import EXPERIMENT_REGISTRY
from .pipeline import run_registered_experiment

__all__ = [
    "TwinExperimentSpec",
    "EXPERIMENT_REGISTRY",
    "run_registered_experiment",
]

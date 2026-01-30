"""Centralized output path management for SWE4DVar.

This module provides consistent output directories for:
- logs: Runtime logs and diagnostic output
- figures: Generated plots and visualizations
- checkpoints: Solver checkpoints and restart files
- data: CSV files, JSON results, and other data outputs
"""

from pathlib import Path
import os

# Determine output root - can be overridden via environment variable
_default_output_root = Path(__file__).parent.parent.parent.parent / "outputs"
OUTPUT_ROOT = Path(os.environ.get("SWE4DVAR_OUTPUT_DIR", _default_output_root))

LOGS_DIR = OUTPUT_ROOT / "logs"
FIGURES_DIR = OUTPUT_ROOT / "figures"
CHECKPOINTS_DIR = OUTPUT_ROOT / "checkpoints"
DATA_DIR = OUTPUT_ROOT / "data"

def ensure_output_dirs():
    """Create output directories if they don't exist."""
    for dir_path in [LOGS_DIR, FIGURES_DIR, CHECKPOINTS_DIR, DATA_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

def get_figure_path(name: str) -> Path:
    """Get path for a figure file."""
    ensure_output_dirs()
    return FIGURES_DIR / name

def get_data_path(name: str) -> Path:
    """Get path for a data file."""
    ensure_output_dirs()
    return DATA_DIR / name

def get_log_path(name: str) -> Path:
    """Get path for a log file."""
    ensure_output_dirs()
    return LOGS_DIR / name

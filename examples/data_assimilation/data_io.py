import pickle
import os
import sys


from mpi4py import MPI

from enum import IntEnum
from pathlib import Path


def save_pickle(filename, data, directory="da_output", verbose=False):
    """Save data to pickle file, overwriting if it exists."""
    filepath = Path(directory) / filename

    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and optionally notify
    if verbose and filepath.exists():
        print(f"Overwriting existing file: {filepath}")

    # Save the file (wb mode overwrites by default)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    if verbose:
        print(f"Saved to: {filepath}")


def load_pickle(filename, directory="da_output"):
    filepath = Path(directory) / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")
    with open(filepath, "rb") as f:
        return pickle.load(f)

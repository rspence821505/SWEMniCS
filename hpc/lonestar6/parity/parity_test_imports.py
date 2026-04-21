"""parity_test_imports.py — emit a JSON record of every relevant package version.

Runs serially. Designed to be identical on both environments; the only
differences should be in the reported version numbers.

Usage:
    python parity_test_imports.py > imports.json
"""
import json
import platform
import sys
from importlib import import_module

PACKAGES = [
    "numpy", "scipy", "mpi4py", "petsc4py",
    "basix", "ufl", "ffcx", "dolfinx",
    "h5py", "adios4dolfinx",
]

out = {
    "platform": platform.platform(),
    "arch": platform.machine(),
    "python": sys.version.split()[0],
    "packages": {},
    "errors": {},
}

for pkg in PACKAGES:
    try:
        mod = import_module(pkg)
        ver = getattr(mod, "__version__", "unknown")
        out["packages"][pkg] = ver
    except Exception as e:
        out["errors"][pkg] = f"{type(e).__name__}: {e}"

# PETSc C-library version (may differ from petsc4py)
try:
    from petsc4py import PETSc
    out["petsc_c_version"] = list(PETSc.Sys.getVersion())
except Exception as e:
    out["errors"]["petsc_c_version"] = f"{type(e).__name__}: {e}"

# MPI library identity
try:
    from mpi4py import MPI
    out["mpi_library"] = MPI.Get_library_version().strip().split("\n")[0]
    out["mpi_version_tuple"] = list(MPI.Get_version())
except Exception as e:
    out["errors"]["mpi_library"] = f"{type(e).__name__}: {e}"

print(json.dumps(out, indent=2, sort_keys=True))

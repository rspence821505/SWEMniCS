#!/usr/bin/env python
"""Verify that the swe4dvar environment is correctly configured."""

import sys


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    package_name = package_name or module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"[OK] {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"[MISSING] {package_name}: {e}")
        return False


def main():
    print("=" * 50)
    print("SWE4DVar Environment Verification")
    print("=" * 50)
    print(f"\nPython: {sys.version}")
    print(f"Executable: {sys.executable}\n")

    all_ok = True

    print("--- Core Dependencies ---")
    all_ok &= check_import("numpy")
    all_ok &= check_import("scipy")
    all_ok &= check_import("matplotlib")

    print("\n--- FEniCSx Ecosystem ---")
    all_ok &= check_import("dolfinx")
    all_ok &= check_import("basix")
    all_ok &= check_import("ffcx")
    all_ok &= check_import("ufl")

    print("\n--- Parallel Computing ---")
    all_ok &= check_import("mpi4py", "mpi4py")
    all_ok &= check_import("petsc4py", "petsc4py")

    print("\n--- I/O Libraries ---")
    all_ok &= check_import("h5py")
    all_ok &= check_import("adios4dolfinx")

    print("\n--- SWE4DVar Package ---")
    all_ok &= check_import("swe4dvar")

    print("\n" + "=" * 50)
    if all_ok:
        print("[OK] All dependencies verified successfully!")
        return 0
    else:
        print("[ERROR] Some dependencies are missing. See above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

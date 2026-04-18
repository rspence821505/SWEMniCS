#!/usr/bin/env python3
"""
Compare Jacobian-action and adjoint-RHS parity between serial and MPI.

Loads results/mpi_parity/jac_probe_{serial,mpi2}.npz and reports:
  - J @ x parity (input vs output)
  - J^T @ x parity
  - Observation forcing (adjoint RHS) parity
  - Per-component norms, cosine similarity, max abs diff
  - Spatial localization: is the discrepancy near partition boundaries?
"""
import sys
import numpy as np
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "mpi_parity"


def compare_component(name, a, b, coords=None, tol_partition_m=500.0):
    """Compare two globally-coord-ordered 1D arrays."""
    if a is None or b is None:
        return
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    diff = a - b
    ndiff = float(np.linalg.norm(diff))
    rel = ndiff / max(na, 1e-30)
    denom = na * nb
    cos = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    max_abs = float(np.max(np.abs(diff))) if len(diff) > 0 else 0.0

    print(f"  {name}")
    print(f"    ‖serial‖={na:.4e}  ‖mpi‖={nb:.4e}  rel_diff={rel:.4e}  "
          f"cos={cos:.6f}  max|diff|={max_abs:.4e}")

    # Spatial localization if coords given
    if coords is not None and len(coords) == len(a):
        # Compute discrepancy by region: bulk vs a simplified "boundary" region.
        # For partition-boundary diagnosis, use MPI file to identify x-coords
        # that might correspond to partition boundaries.
        # Simplest: just print top-5 largest-diff locations.
        abs_diff = np.abs(diff)
        worst = np.argsort(abs_diff)[-5:][::-1]
        if max_abs > 1e-10 * max(na, nb):
            print(f"    top 5 worst-diff locations:")
            for w in worst:
                print(f"      coord=({coords[w,0]:.1f},{coords[w,1]:.1f}) "
                      f"serial={a[w]:.4e}  mpi={b[w]:.4e}  diff={diff[w]:.4e}")


def main():
    ser_file = RESULTS / "jac_probe_serial.npz"
    mpi_file = RESULTS / "jac_probe_mpi2.npz"
    if not ser_file.exists():
        print(f"Missing: {ser_file}")
        return 1
    if not mpi_file.exists():
        print(f"Missing: {mpi_file}")
        return 1

    s = np.load(ser_file)
    m = np.load(mpi_file)

    # Coord sanity
    coord_diff = float(np.max(np.abs(s["coords"] - m["coords"])))
    print(f"Coord ordering alignment: max|serial.coords - mpi.coords| = {coord_diff:.2e}")
    if coord_diff > 1e-6:
        print("WARNING: coords differ — comparison may be misleading")
    coords = s["coords"]

    print()
    print("=" * 78)
    print("JACOBIAN APPLICATION PARITY (J @ x_pattern)")
    print("=" * 78)
    for pattern in ["ones", "sin2D", "localized"]:
        print(f"\n--- pattern = {pattern} ---")
        for target in ["input", "Jx", "JTx"]:
            for comp in ["h", "u", "v"]:
                key = f"{pattern}_{target}__{comp}"
                if key not in s or key not in m:
                    continue
                compare_component(f"{target:>5}.{comp}", s[key], m[key], coords)
            print()

    print("=" * 78)
    print("ADJOINT RHS PARITY (observation forcings)")
    print("=" * 78)
    for i in range(5):  # up to 5 obs times
        key_present = f"obs_forcing_{i}_present"
        if key_present not in s or not bool(s[key_present]):
            continue
        print(f"\n--- obs_forcing[{i}] ---")
        for comp in ["h", "u", "v"]:
            key = f"obs_forcing_{i}__{comp}"
            if key not in s or key not in m:
                continue
            compare_component(f"f[{i}].{comp}", s[key], m[key], coords)


if __name__ == "__main__":
    sys.exit(main() or 0)

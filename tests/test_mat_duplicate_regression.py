"""
Regression guard for PETSc.Mat.duplicate() value-copy pitfall.

Motivation
----------
PETSc.Mat.duplicate() defaults to copy=False, which preserves the sparsity
pattern but leaves values UNSET (zero). Two experiment drivers used the
bare form to deep-copy truth Jacobians before tearing down the truth
solver:
  - experiments/idealized_inlet_da.py   (was line 261, fix commit 3852e6f)
  - experiments/shinnecock_study/run_comparison.py  (was line 4327, fix 0c06c8c)

Both sites combined `[J.duplicate() for J in storage.saved_jacobians]`
with a subsequent `del solver_truth` / `storage.clear()` that destroyed
the originals, leaving the TLM Eq 38 adjoint with a list of zero-valued
matrices. See docs/idealized_inlet_jacobian_handoff_trace.md.

This test file does two things:
  1. Pins the PETSc behavior — so if PETSc ever flips the default,
     the test fails loudly at CI time (giving us a chance to remove
     the explicit copy=True flags).
  2. Guards the two historically-buggy call sites — any future edit
     that drops copy=True will fail here.
"""

import re
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc


# Repo root relative to this test file: tests/ → repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. PETSc behavior pin
# ---------------------------------------------------------------------------

def _make_small_aij_with_values() -> PETSc.Mat:
    """Build a 4x4 AIJ matrix with real nonzero values on rank 0 only.

    Uses COMM_SELF so this test is valid under any mpirun size without
    needing distributed assembly.
    """
    A = PETSc.Mat().createAIJ(size=(4, 4), comm=PETSc.COMM_SELF)
    A.setPreallocationNNZ(4)
    A.setUp()
    # Load some values — tridiagonal + a few off-diag to make Frobenius nontrivial.
    for i in range(4):
        A.setValue(i, i, float(i + 1))
        if i > 0:
            A.setValue(i, i - 1, 0.5)
        if i < 3:
            A.setValue(i, i + 1, -0.25)
    A.assemble()
    return A


def test_mat_duplicate_default_is_zero_values():
    """Mat.duplicate() (no flag) preserves sparsity but leaves values zero.

    This is the PETSc default behavior that caused the handoff bug. If
    PETSc ever changes this default, we want to know so we can remove
    our `copy=True` guards.
    """
    # Only rank 0 runs the serial-COMM test to avoid cross-rank confusion.
    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    A = _make_small_aij_with_values()
    fro_A = A.norm(PETSc.NormType.NORM_FROBENIUS)
    assert fro_A > 1.0, f"setup error: source matrix Frobenius norm = {fro_A}"

    A_dup = A.duplicate()  # no copy=True — the pitfall
    fro_dup = A_dup.norm(PETSc.NormType.NORM_FROBENIUS)
    assert fro_dup == 0.0, (
        f"PETSc behavior changed: Mat.duplicate() without copy=True now "
        f"returns a matrix with Frobenius norm {fro_dup}. If this is "
        f"the new default, the `copy=True` guards in idealized_inlet_da.py "
        f"and run_comparison.py are redundant and can be removed."
    )

    # Sanity: nz_used should match — sparsity is preserved even when values are not.
    assert A_dup.getInfo()["nz_used"] == A.getInfo()["nz_used"]

    A.destroy()
    A_dup.destroy()


def test_mat_duplicate_with_copy_flag_preserves_values():
    """Mat.duplicate(copy=True) produces a matrix equal to the source.

    This is the correct pattern for deep-copying a Jacobian.
    """
    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    A = _make_small_aij_with_values()
    fro_A = A.norm(PETSc.NormType.NORM_FROBENIUS)

    A_copy = A.duplicate(copy=True)
    fro_copy = A_copy.norm(PETSc.NormType.NORM_FROBENIUS)

    assert np.isclose(fro_copy, fro_A), (
        f"duplicate(copy=True) should preserve values: "
        f"src Fro = {fro_A}, copy Fro = {fro_copy}"
    )

    # Mutating the copy must not affect the source.
    A_copy.scale(2.0)
    assert np.isclose(A.norm(PETSc.NormType.NORM_FROBENIUS), fro_A), (
        "copy=True must allocate an independent value buffer; "
        "scaling the copy mutated the source."
    )

    A.destroy()
    A_copy.destroy()


# ---------------------------------------------------------------------------
# 2. Source-level guard against recurrence at the two known bug sites
# ---------------------------------------------------------------------------

KNOWN_CALL_SITES = [
    "experiments/idealized_inlet_da.py",
    "experiments/shinnecock_study/run_comparison.py",
]

# Match: [J.duplicate() for J in ANYTHING.saved_jacobians] with no copy kwarg.
# The `duplicate(` with an empty arg list or only whitespace+ `)` is the
# dangerous form. We look specifically for the "saved_jacobians" idiom
# because Vec duplicates in other contexts are intentional workspace zeros.
_BAD_PATTERN = re.compile(
    r"\[\s*\w+\.duplicate\(\s*\)\s*for\s+\w+\s+in\s+[^\]]*saved_jacobians"
)


@pytest.mark.parametrize("rel_path", KNOWN_CALL_SITES)
def test_known_jacobian_deepcopy_sites_use_copy_true(rel_path: str):
    """The two historically-buggy truth-Jacobian deep-copy sites must
    keep using ``duplicate(copy=True)``. A bare ``duplicate()`` in a
    list comprehension over ``saved_jacobians`` is the exact bug
    signature from docs/idealized_inlet_jacobian_handoff_trace.md.
    """
    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    path = REPO_ROOT / rel_path
    assert path.exists(), f"expected file not found: {path}"

    text = path.read_text(encoding="utf-8")
    match = _BAD_PATTERN.search(text)
    assert match is None, (
        f"{rel_path}: found a bare `.duplicate()` over `saved_jacobians` — "
        f"this is the bug signature from the handoff trace memo. Use "
        f"`.duplicate(copy=True)` to preserve matrix values. "
        f"Offending snippet:\n    {match.group(0) if match else ''}"
    )

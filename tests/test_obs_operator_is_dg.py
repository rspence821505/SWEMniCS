#!/usr/bin/env python3
"""
Regression test: is_discontinuous_space must return True for
SWE's mixed DG space (scalar h + vector [u, v], both DG P1).

Prior to the fix in this file, `is_discontinuous_space` returned
False for any mixed element, silently sending `PointObservationOperator`
down its CG adjoint path. Under MPI partitioning this placed the
adjoint RHS at different DG DOFs per rank, giving a cosine-similarity
of ~0.18 between serial and 2-rank adjoint vectors.

This test builds the actual solver's V and asserts is_discontinuous_space
returns True.
"""
import os, sys
os.environ.setdefault("CC", "/usr/bin/clang")
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from swe4dvar.data_assimilation.observation_operator import is_discontinuous_space
from swe4dvar.forward.problems import IdealizedInlet
from swe4dvar.forward.solvers import get_solver


def test_idealized_inlet_V_is_dg():
    prob = IdealizedInlet(
        dt=600.0, nt=1,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])
    V = solver.V
    assert is_discontinuous_space(V), (
        "is_discontinuous_space must return True for SWE's mixed DG space. "
        "If this fails, the PointObservationOperator.adjoint will silently "
        "take the CG path and produce incorrect results under MPI."
    )
    print("PASS: is_discontinuous_space(V) is True for idealized inlet mixed DG space")


if __name__ == "__main__":
    test_idealized_inlet_V_is_dg()

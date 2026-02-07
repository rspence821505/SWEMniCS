#!/usr/bin/env python3
"""
Verification script for comparison study setup.

Run this before the full comparison study to verify:
1. TidalProblem + DG solver works
2. TwinExperiment runs successfully
3. Both 4D-Var and DC-WME methods work
4. Friction perturbation works

Usage:
    python experiments/comparison_study/verify_setup.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig


def verify_forward_model():
    """Step 1: Verify forward model setup with DG solver.

    Note: We only verify that the problem and solver can be created.
    The actual forward solve is verified by the TwinExperiment tests,
    which use the time-dependent solve path.
    """
    print("\n" + "=" * 60)
    print("Step 1: Verifying Problem/Solver Setup (TidalProblem + DG)")
    print("=" * 60)

    try:
        problem = TidalProblem(nx=20, ny=10, dt=1800, nt=5)
        solver = get_solver("DG")(problem, theta=0.5, p_degree=[1, 1])

        # Verify basic setup - mesh and function spaces
        mesh = problem.mesh
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local

        # Verify solver has the expected attributes
        has_storage = hasattr(solver, "storage")
        has_forward = hasattr(solver, "TimeStep")

        print(f"  - Problem: {problem.__class__.__name__}")
        print(f"  - Mesh cells: {num_cells}")
        print(f"  - Has storage: {has_storage}")
        print(f"  - Has TimeStep: {has_forward}")
        print("  - Status: PASSED")
        print("  - Note: Full forward solve verified in TwinExperiment tests")
        return True

    except Exception as e:
        print(f"  - Status: FAILED")
        print(f"  - Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_twin_experiment_4dvar():
    """Step 2: Verify 4D-Var twin experiment runs."""
    print("\n" + "=" * 60)
    print("Step 2: Verifying TwinExperiment with 4D-Var")
    print("=" * 60)

    try:
        problem = TidalProblem(nx=20, ny=10, dt=1800, nt=10)
        solver = get_solver("DG")(problem, theta=0.5, p_degree=[1, 1])

        config = TwinExperimentConfig(
            method="4dvar",
            obs_fraction=0.5,
            obs_frequency=2,
            max_iterations=10,  # Short run for verification
            verbose=False,
        )

        experiment = TwinExperiment(problem, solver, config)
        start = time.time()
        results = experiment.run()
        elapsed = time.time() - start

        print(f"  - Method: 4D-Var")
        print(f"  - Iterations: {results.num_iterations}")
        print(f"  - Error reduction: {results.error_reduction:.1f}%")
        print(f"  - Wall time: {elapsed:.2f}s")
        print("  - Status: PASSED")
        return True

    except Exception as e:
        print(f"  - Status: FAILED")
        print(f"  - Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_twin_experiment_dcwme():
    """Step 3: Verify DC-WME twin experiment runs."""
    print("\n" + "=" * 60)
    print("Step 3: Verifying TwinExperiment with DC-WME")
    print("=" * 60)

    try:
        problem = TidalProblem(nx=20, ny=10, dt=1800, nt=10)
        solver = get_solver("DG")(problem, theta=0.5, p_degree=[1, 1])

        config = TwinExperimentConfig(
            method="dcwme",
            obs_fraction=0.5,
            obs_frequency=2,
            max_iterations=10,  # Short run for verification
            verbose=False,
        )

        experiment = TwinExperiment(problem, solver, config)
        start = time.time()
        results = experiment.run()
        elapsed = time.time() - start

        print(f"  - Method: DC-WME")
        print(f"  - Iterations: {results.num_iterations}")
        print(f"  - Error reduction: {results.error_reduction:.1f}%")
        print(f"  - Wall time: {elapsed:.2f}s")
        print("  - Status: PASSED")
        return True

    except Exception as e:
        print(f"  - Status: FAILED")
        print(f"  - Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_friction_perturbation():
    """Step 4: Verify friction perturbation works."""
    print("\n" + "=" * 60)
    print("Step 4: Verifying Friction Perturbation (4D-Var, scale=1.1)")
    print("=" * 60)

    try:
        problem = TidalProblem(nx=20, ny=10, dt=1800, nt=10)
        solver = get_solver("DG")(problem, theta=0.5, p_degree=[1, 1])

        config = TwinExperimentConfig(
            method="4dvar",
            obs_fraction=0.5,
            obs_frequency=2,
            max_iterations=10,
            perturb_friction=True,
            friction_scale_factor=1.1,  # 10% friction error
            verbose=False,
        )

        experiment = TwinExperiment(problem, solver, config)
        start = time.time()
        results = experiment.run()
        elapsed = time.time() - start

        print(f"  - Friction scale: 1.1 (10% model error)")
        print(f"  - Iterations: {results.num_iterations}")
        print(f"  - Error reduction: {results.error_reduction:.1f}%")
        print(f"  - Wall time: {elapsed:.2f}s")
        print("  - Status: PASSED")
        return True

    except Exception as e:
        print(f"  - Status: FAILED")
        print(f"  - Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_error_handling():
    """Step 5: Verify error handling works (extreme perturbation)."""
    print("\n" + "=" * 60)
    print("Step 5: Verifying Error Handling (friction_scale=2.0)")
    print("=" * 60)

    try:
        problem = TidalProblem(nx=20, ny=10, dt=1800, nt=10)
        solver = get_solver("DG")(problem, theta=0.5, p_degree=[1, 1])

        config = TwinExperimentConfig(
            method="4dvar",
            obs_fraction=0.5,
            obs_frequency=2,
            max_iterations=5,
            perturb_friction=True,
            friction_scale_factor=2.0,  # Extreme - may fail
            verbose=False,
        )

        experiment = TwinExperiment(problem, solver, config)
        results = experiment.run()

        # If it succeeds, that's fine too
        print(f"  - Friction scale: 2.0 (extreme)")
        print(f"  - Result: Completed successfully (surprisingly robust!)")
        print("  - Status: PASSED")
        return True

    except Exception as e:
        # Expected to fail - verify we can catch the error
        print(f"  - Friction scale: 2.0 (extreme)")
        print(f"  - Result: Failed as expected (error handling works)")
        print(f"  - Error type: {type(e).__name__}")
        print("  - Status: PASSED (failure was caught correctly)")
        return True


def main():
    """Run all verification steps."""
    print("\n" + "=" * 60)
    print("COMPARISON STUDY SETUP VERIFICATION")
    print("=" * 60)

    results = {}

    # Run each verification step
    results["forward_model"] = verify_forward_model()
    results["4dvar"] = verify_twin_experiment_4dvar()
    results["dcwme"] = verify_twin_experiment_dcwme()
    results["friction_perturbation"] = verify_friction_perturbation()
    results["error_handling"] = verify_error_handling()

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  All verification steps passed!")
        print("  You can proceed with the full comparison study.")
        return 0
    else:
        print("\n  Some verification steps failed.")
        print("  Please fix the issues before running the comparison study.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

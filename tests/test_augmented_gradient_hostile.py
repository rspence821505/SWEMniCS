from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.twin_framework.parameter_runners import ManningsNRunner
from experiments.twin_framework.registry import EXPERIMENT_REGISTRY
from swe4dvar.forward.augmented_control import ManningsBasisController


def _build_normalized_gaussian_basis(
    coords: np.ndarray,
    basis_shape: tuple[int, int],
    basis_width_fraction: float,
) -> np.ndarray:
    nx, ny = basis_shape
    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
    y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
    centers_x = np.linspace(x_min, x_max, nx)
    centers_y = np.linspace(y_min, y_max, ny)
    centers = np.array([(x, y) for y in centers_y for x in centers_x], dtype=float)
    width = basis_width_fraction * max(x_max - x_min, y_max - y_min)
    width = max(width, 1e-30)
    basis = np.zeros((coords.shape[0], centers.shape[0]), dtype=float)
    for j, center in enumerate(centers):
        dist_sq = np.sum((coords - center) ** 2, axis=1)
        basis[:, j] = np.exp(-0.5 * dist_sq / width**2)
    row_sums = np.maximum(np.sum(basis, axis=1, keepdims=True), 1e-30)
    return basis / row_sums


def _legacy_truth_field(theta: np.ndarray, basis: np.ndarray, n_bounds: tuple[float, float]) -> np.ndarray:
    base_n = np.full(basis.shape[0], 0.02, dtype=float)
    return np.clip(base_n * np.exp(basis @ theta), *n_bounds)


def _install_fake_parallel_runtime(monkeypatch):
    class _FakeComm:
        def getSize(self):
            return 1

        def Get_size(self):
            return 1

        def Get_rank(self):
            return 0

        def Barrier(self):
            return None

    class _FakeMPIType:
        Comm = _FakeComm
        COMM_WORLD = _FakeComm()

    class _FakeVec:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=float).copy()

        def duplicate(self):
            return _FakeVec(np.zeros_like(self.data))

        def copy(self):
            return _FakeVec(self.data.copy())

        def zeroEntries(self):
            self.data[:] = 0.0

        def waxpy(self, alpha, x, y):
            self.data = alpha * x.data + y.data

        def axpy(self, alpha, x):
            self.data = self.data + alpha * x.data

        def dot(self, other):
            return float(np.dot(self.data, other.data))

        def getArray(self, readonly=False):
            return self.data.copy() if readonly else self.data

        def setArray(self, arr):
            self.data = np.asarray(arr, dtype=float).copy()

        def assemble(self):
            return None

    fake_comm = _FakeMPIType.COMM_WORLD
    fake_pc = type(
        "FakePC",
        (),
        {"Type": types.SimpleNamespace(HYPRE="hypre", JACOBI="jacobi", CHOLESKY="chol")},
    )
    fake_ksp = type(
        "FakeKSP",
        (),
        {"Type": types.SimpleNamespace(CG="cg", PREONLY="preonly")},
    )
    fake_mat = type("FakeMat", (), {"Type": types.SimpleNamespace(AIJ="aij")})
    fake_petsc = types.SimpleNamespace(
        Vec=_FakeVec,
        Mat=fake_mat,
        PC=fake_pc,
        KSP=fake_ksp,
        COMM_SELF=fake_comm,
        COMM_WORLD=fake_comm,
        Error=RuntimeError,
        ScalarType=float,
        DECIDE=-1,
    )
    fake_mpi = _FakeMPIType

    monkeypatch.setitem(sys.modules, "petsc4py", types.SimpleNamespace(PETSc=fake_petsc))
    monkeypatch.setitem(sys.modules, "mpi4py", types.SimpleNamespace(MPI=fake_mpi))
    return fake_petsc


def _import_augmented_cost_functions(monkeypatch):
    _install_fake_parallel_runtime(monkeypatch)
    for name in list(sys.modules):
        if name == "swe4dvar.data_assimilation" or name.startswith("swe4dvar.data_assimilation."):
            sys.modules.pop(name, None)
    return importlib.import_module("swe4dvar.data_assimilation.augmented_cost_functions")


def test_mannings_truth_generation_matches_augmented_controller_parameterization():
    cfg = EXPERIMENT_REGISTRY["mannings_n"].forward_model_config
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=float,
    )
    basis = _build_normalized_gaussian_basis(
        coords,
        tuple(cfg["basis_shape"]),
        float(cfg["basis_width_fraction"]),
    )
    theta = np.asarray(cfg["truth_coefficients"], dtype=float)

    truth_field = _legacy_truth_field(theta, basis, tuple(cfg["n_clip"]))
    controller = ManningsBasisController(
        basis_shape=tuple(cfg["basis_shape"]),
        basis_width_fraction=float(cfg["basis_width_fraction"]),
        n_bounds=tuple(cfg["n_clip"]),
        reference_n=0.02,
    )
    controller._basis_matrix = basis
    controller._base_field_array = np.full(basis.shape[0], 0.02, dtype=float)
    augmented_field = controller.evaluate_field(theta)

    assert np.allclose(
        truth_field,
        augmented_field,
        rtol=1e-7,
        atol=1e-10,
    ), (
        "The mannings_n truth path and augmented controller do not use the same "
        "coefficient-to-field map. This invalidates any end-to-end gradient "
        "validation because truth is generated in different coordinates than the "
        "coordinates differentiated by the optimizer."
    )


def test_parameter_runner_still_contains_legacy_exponential_truth_map():
    source = Path(PROJECT_ROOT / "experiments/twin_framework/parameter_runners.py").read_text()
    assert "base_n * np.exp(log_multiplier)" not in source, (
        "The mannings_n experiment still contains the legacy exponential Manning "
        "map inside parameter_runners.py. That means the experiment architecture "
        "has not been fully unified around the augmented controller."
    )


def test_residual_parameter_gradient_accumulates_all_timesteps_with_correct_sign(monkeypatch):
    module = _import_augmented_cost_functions(monkeypatch)
    FakeVec = sys.modules["petsc4py"].PETSc.Vec

    class _Layout:
        theta_size = 2

    class _ForwardModel:
        def __init__(self):
            self.control_layout = _Layout()

        def get_timestep_parameter_derivatives(self, *args, **kwargs):
            return [
                [FakeVec([1.0, 0.0]), FakeVec([0.0, 2.0])],
                [FakeVec([3.0, 0.0]), FakeVec([0.0, 5.0])],
                [FakeVec([7.0, 0.0]), FakeVec([0.0, 11.0])],
            ]

        def get_parameter_sensitivity_bundle(self, *args, **kwargs):
            raise AssertionError("Residual-derivative path should have been used")

    class _Harness(module._AugmentedGradientMixin):
        def __init__(self):
            self.forward_model = _ForwardModel()

        def _solve_augmented_adjoint(self, *args, **kwargs):
            lambda_history = [
                None,
                FakeVec([13.0, 17.0]),
                FakeVec([19.0, 23.0]),
                FakeVec([29.0, 31.0]),
            ]
            return FakeVec([0.0]), lambda_history

    harness = _Harness()
    theta_grad = harness._parameter_gradient_from_forcings(
        m=FakeVec([0.0]),
        forcings=[None, FakeVec([0.0]), FakeVec([0.0]), FakeVec([0.0])],
        trajectory=[object(), object(), object(), object()],
        jacobians=[object(), object(), object()],
    )

    # Lagrangian: ∂L/∂θ = +Σ_k λ_k^T (∂F_k/∂θ_p)
    expected = np.array(
        [
            +(13.0 * 1.0 + 19.0 * 3.0 + 29.0 * 7.0),
            +(17.0 * 2.0 + 23.0 * 5.0 + 31.0 * 11.0),
        ],
        dtype=float,
    )
    assert np.allclose(theta_grad, expected)


def test_residual_parameter_gradient_ignores_spurious_timestep_zero_derivatives(monkeypatch):
    module = _import_augmented_cost_functions(monkeypatch)
    FakeVec = sys.modules["petsc4py"].PETSc.Vec

    class _Layout:
        theta_size = 2

    class _ForwardModel:
        def __init__(self):
            self.control_layout = _Layout()

        def get_timestep_parameter_derivatives(self, *args, **kwargs):
            return [
                [FakeVec([101.0, 0.0]), FakeVec([0.0, 103.0])],
                [FakeVec([1.0, 0.0]), FakeVec([0.0, 2.0])],
                [FakeVec([3.0, 0.0]), FakeVec([0.0, 5.0])],
                [FakeVec([7.0, 0.0]), FakeVec([0.0, 11.0])],
            ]

        def get_parameter_sensitivity_bundle(self, *args, **kwargs):
            raise AssertionError("Residual-derivative path should have been used")

    class _Harness(module._AugmentedGradientMixin):
        def __init__(self):
            self.forward_model = _ForwardModel()

        def _solve_augmented_adjoint(self, *args, **kwargs):
            lambda_history = [
                None,
                FakeVec([13.0, 17.0]),
                FakeVec([19.0, 23.0]),
                FakeVec([29.0, 31.0]),
            ]
            return FakeVec([0.0]), lambda_history

    harness = _Harness()
    theta_grad = harness._parameter_gradient_from_forcings(
        m=FakeVec([0.0]),
        forcings=[None, FakeVec([0.0]), FakeVec([0.0]), FakeVec([0.0])],
        trajectory=[object(), object(), object(), object()],
        jacobians=[object(), object(), object()],
    )

    # Lagrangian: ∂L/∂θ = +Σ_k λ_k^T (∂F_k/∂θ_p)
    expected = np.array(
        [
            +(13.0 * 1.0 + 19.0 * 3.0 + 29.0 * 7.0),
            +(17.0 * 2.0 + 23.0 * 5.0 + 31.0 * 11.0),
        ],
        dtype=float,
    )
    assert np.allclose(theta_grad, expected), (
        "A saved derivative row at timestep 0 must be ignored. Otherwise the "
        "Manning gradient is indexed against the wrong adjoint states and can "
        "crash on the last step."
    )


def test_fallback_parameter_gradient_uses_all_forcings_without_block_mixing(monkeypatch):
    module = _import_augmented_cost_functions(monkeypatch)
    FakeVec = sys.modules["petsc4py"].PETSc.Vec

    class _Layout:
        theta_size = 2

    bundle = types.SimpleNamespace(
        sensitivities=[
            [FakeVec([1.0, 0.0]), FakeVec([0.0, 2.0]), FakeVec([3.0, 0.0])],
            [FakeVec([0.0, 5.0]), FakeVec([7.0, 0.0]), FakeVec([0.0, 11.0])],
        ]
    )

    class _ForwardModel:
        def __init__(self):
            self.control_layout = _Layout()

        def get_timestep_parameter_derivatives(self, *args, **kwargs):
            return None

        def get_parameter_sensitivity_bundle(self, *args, **kwargs):
            return bundle

    class _Harness(module._AugmentedGradientMixin):
        def __init__(self):
            self.forward_model = _ForwardModel()

    harness = _Harness()
    forcings = [FakeVec([2.0, 3.0]), None, FakeVec([5.0, 7.0])]
    theta_grad = harness._parameter_gradient_from_forcings(
        m=FakeVec([0.0]),
        forcings=forcings,
        trajectory=None,
        jacobians=None,
    )

    expected = np.array(
        [
            np.dot([1.0, 0.0], [2.0, 3.0]) + np.dot([3.0, 0.0], [5.0, 7.0]),
            np.dot([0.0, 5.0], [2.0, 3.0]) + np.dot([0.0, 11.0], [5.0, 7.0]),
        ],
        dtype=float,
    )
    assert np.allclose(theta_grad, expected)


def test_runner_basis_builder_matches_hostile_reference_formula():
    cfg = EXPERIMENT_REGISTRY["mannings_n"].forward_model_config
    runner = object.__new__(ManningsNRunner)
    runner.spec = EXPERIMENT_REGISTRY["mannings_n"]
    coords = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )

    runner_basis = runner._build_gaussian_basis(coords)
    reference_basis = _build_normalized_gaussian_basis(
        coords,
        tuple(cfg["basis_shape"]),
        float(cfg["basis_width_fraction"]),
    )
    assert np.allclose(runner_basis, reference_basis)

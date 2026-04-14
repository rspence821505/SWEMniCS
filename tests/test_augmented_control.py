from pathlib import Path
import importlib.util
import sys
import types

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swe4dvar.control import ControlLayout, ControlVector
from swe4dvar.forward.augmented_control import ManningsBasisController

_TIMESTEP_MANAGER_PATH = SRC_ROOT / "swe4dvar" / "utils" / "timestep_manager.py"
_TIMESTEP_MANAGER_SPEC = importlib.util.spec_from_file_location(
    "test_timestep_manager_module",
    _TIMESTEP_MANAGER_PATH,
)
_TIMESTEP_MANAGER_MODULE = importlib.util.module_from_spec(_TIMESTEP_MANAGER_SPEC)
assert _TIMESTEP_MANAGER_SPEC.loader is not None
_stubbed_petsc4py = False
if "petsc4py" not in sys.modules:
    sys.modules["petsc4py"] = types.SimpleNamespace(PETSc=types.SimpleNamespace(Mat=object))
    _stubbed_petsc4py = True
_TIMESTEP_MANAGER_SPEC.loader.exec_module(_TIMESTEP_MANAGER_MODULE)
if _stubbed_petsc4py:
    sys.modules.pop("petsc4py", None)
TimeStepDataManager = _TIMESTEP_MANAGER_MODULE.TimeStepDataManager


def test_control_vector_pack_unpack_round_trip():
    layout = ControlLayout(state_size=2, theta_size=3)
    packed = layout.pack_array(u0=[1.0, -2.0], theta=[0.5, 1.5, -3.0])
    unpacked = layout.unpack_array(packed)
    assert np.allclose(unpacked.u0, [1.0, -2.0])
    assert np.allclose(unpacked.theta, [0.5, 1.5, -3.0])
    assert np.allclose(unpacked.pack(), packed)


def test_parameter_only_layout_round_trip():
    layout = ControlLayout(state_size=0, theta_size=3)
    packed = layout.pack_array(theta=[2.0, -1.0, 0.25])
    unpacked = layout.unpack_array(packed)
    assert unpacked.u0.size == 0
    assert np.allclose(unpacked.theta, [2.0, -1.0, 0.25])


def test_petsc_vector_round_trip_if_available():
    petsc4py = pytest.importorskip("petsc4py")
    _ = petsc4py
    layout = ControlLayout(state_size=1, theta_size=2)
    vec = layout.create_petsc_vec(u0=[4.0], theta=[5.0, 6.0])
    unpacked = layout.unpack_vec(vec)
    assert np.allclose(unpacked.pack(), [4.0, 5.0, 6.0])


def test_mannings_controller_field_evaluation_respects_bounds():
    controller = ManningsBasisController(
        basis_shape=(2, 1),
        basis_width_fraction=0.5,
        n_bounds=(0.01, 0.08),
        reference_n=0.02,
    )
    controller._basis_matrix = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    controller._base_field_array = np.array([0.02, 0.02, 0.02])

    field = controller.evaluate_field(np.array([2.0, -2.0]))
    assert np.all(field >= 0.01)
    assert np.all(field <= 0.08)
    assert field.shape == (3,)


def test_forced_tlm_matches_simple_linear_recurrence_if_petsc_available():
    petsc4py = pytest.importorskip("petsc4py")
    _ = petsc4py
    from petsc4py import PETSc

    from swe4dvar.adjoint.tangent_linear import ForcedTangentLinearModel

    class _FakeForwardModel:
        def __init__(self):
            self.dt = 1.0
            self._mass_matrix = PETSc.Mat().createAIJ(size=[1, 1], nnz=1, comm=PETSc.COMM_SELF)
            self._mass_matrix.setValue(0, 0, 1.0)
            self._mass_matrix.assemble()

        def get_mass_matrix(self):
            return self._mass_matrix

    def _vec(value: float):
        vec = PETSc.Vec().createSeq(1, comm=PETSc.COMM_SELF)
        vec.setValue(0, value)
        vec.assemble()
        return vec

    model = _FakeForwardModel()
    trajectory = [_vec(0.0), _vec(0.0), _vec(0.0)]
    jacobians = []
    for _ in range(2):
        J = PETSc.Mat().createAIJ(size=[1, 1], nnz=1, comm=PETSc.COMM_SELF)
        J.setValue(0, 0, 1.0)
        J.assemble()
        jacobians.append(J)

    forced_tlm = ForcedTangentLinearModel(model, trajectory, jacobians)
    forcings = [_vec(3.0), _vec(-1.0)]
    perturbations = forced_tlm.propagate_forcing(forcings, end_time=2)
    values = [float(vec.getArray(readonly=True)[0]) for vec in perturbations]

    assert values == pytest.approx([0.0, -3.0, -5.0])


def test_timestep_manager_does_not_store_parameter_derivatives_at_timestep_zero():
    class _FakeStorage:
        saved_states = []
        saved_jacobians = []
        saved_adjoints = []
        saved_parameter_derivatives = []
        dry_nodes = []

    class _FakeSolver:
        def __init__(self):
            self.problem = type("Problem", (), {"nt": 4})()
            self.storage = _FakeStorage()
            self.saved_states = self.storage.saved_states
            self.saved_jacobians = self.storage.saved_jacobians
            self.saved_adjoints = self.storage.saved_adjoints
            self.saved_parameter_derivatives = self.storage.saved_parameter_derivatives
            self.dry_nodes = self.storage.dry_nodes
            self.mpi_rank = 0
            self.parameter_derivative_save_calls = 0

        def save_parameter_derivatives(self):
            self.parameter_derivative_save_calls += 1

    solver = _FakeSolver()
    manager = TimeStepDataManager(
        solver=solver,
        store_parameter_derivatives=True,
        save_state=False,
        store_jacobians=False,
        save_adjoints=False,
    )

    manager.save_timestep(timestep=0)
    manager.save_timestep(timestep=1)

    assert solver.parameter_derivative_save_calls == 1, (
        "Parameter derivatives should be saved only for physical residual "
        "timesteps k >= 1. Saving one at timestep 0 misaligns the augmented "
        "adjoint gradient with the forward trajectory."
    )

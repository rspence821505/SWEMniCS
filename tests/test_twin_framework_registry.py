from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.twin_framework import EXPERIMENT_REGISTRY, run_registered_experiment
from experiments.twin_framework.wse_runner import WSE_METHOD_MAP


def test_registry_contains_exactly_three_experiments():
    assert set(EXPERIMENT_REGISTRY.keys()) == {
        "wse_wind_ramp",
        "wind_param",
        "mannings_n",
    }
    assert EXPERIMENT_REGISTRY["wse_wind_ramp"].inverse_problem_type == "state"
    assert EXPERIMENT_REGISTRY["wind_param"].inverse_problem_type == "low_dim_param"
    assert EXPERIMENT_REGISTRY["mannings_n"].inverse_problem_type == "distributed_param"
    assert EXPERIMENT_REGISTRY["mannings_n"].control_variables == [
        "initial_state_u",
        "mannings_n_basis_coefficients",
    ]
    assert EXPERIMENT_REGISTRY["mannings_n"].sweep_parameters == {
        "regularization_weight": [0.5, 1.0, 2.0],
        "noise_levels": [0.005, 0.01, 0.02],
        "window_lengths": [12, 24],
    }


def test_wse_method_map_restores_original_three_method_sweep():
    assert set(WSE_METHOD_MAP.keys()) == {
        "4dvar",
        "dcwme_dynamic",
        "dcwme_static",
    }


def test_dry_run_writes_standardized_output_tree(tmp_path):
    output_dir = tmp_path / "mannings_n"
    result = run_registered_experiment(
        "mannings_n",
        output_root=str(output_dir),
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    assert result["methods"] == ["4dvar", "dcwme_static", "dcwme_dynamic"]
    assert result["num_cases"] == 54
    assert (output_dir / "config.json").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "diagnostics").exists()
    assert (output_dir / "trajectories").exists()
    assert (output_dir / "diagnostics" / "case_manifest.json").exists()


def test_mannings_dry_run_can_restrict_to_single_method(tmp_path):
    output_dir = tmp_path / "mannings_n_single_method"
    result = run_registered_experiment(
        "mannings_n",
        method="4dvar",
        output_root=str(output_dir),
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    assert result["methods"] == ["4dvar"]
    assert result["num_cases"] == 18


def test_mannings_dry_run_accepts_explicit_static_and_dynamic_methods(tmp_path):
    static_out = tmp_path / "mannings_n_static"
    dynamic_out = tmp_path / "mannings_n_dynamic"

    static_result = run_registered_experiment(
        "mannings_n",
        method="dcwme_static",
        output_root=str(static_out),
        dry_run=True,
    )
    dynamic_result = run_registered_experiment(
        "mannings_n",
        method="dcwme_dynamic",
        output_root=str(dynamic_out),
        dry_run=True,
    )

    assert static_result["methods"] == ["dcwme_static"]
    assert dynamic_result["methods"] == ["dcwme_dynamic"]
    assert static_result["num_cases"] == 18
    assert dynamic_result["num_cases"] == 18


def test_mannings_dry_run_supports_single_static_case_override(tmp_path):
    output_dir = tmp_path / "mannings_n_single_static_case"
    result = run_registered_experiment(
        "mannings_n",
        method="dcwme_static",
        output_root=str(output_dir),
        dry_run=True,
        case_overrides={
            "single_case": True,
            "regularization_weight": 0.5,
            "obs_noise_level": 0.005,
            "window_length": 12,
        },
    )
    assert result["status"] == "dry_run"
    assert result["single_case"] is True
    assert result["methods"] == ["dcwme_static"]
    assert result["regularization_weights"] == [0.5]
    assert result["noise_levels"] == [0.005]
    assert result["window_lengths"] == [12]
    assert result["num_cases"] == 1
    assert result["case_manifest"][0]["case_id"] == "reg_0p5__noise_0p005__window_12__dcwme_static"


def test_mannings_dry_run_supports_cheap_static_case_mode(tmp_path):
    output_dir = tmp_path / "mannings_n_cheap_static_case"
    result = run_registered_experiment(
        "mannings_n",
        method="dcwme_static",
        output_root=str(output_dir),
        dry_run=True,
        case_overrides={
            "cheap_static_case": True,
        },
    )
    assert result["status"] == "dry_run"
    assert result["single_case"] is True
    assert result["methods"] == ["dcwme_static"]
    assert result["regularization_weights"] == [1.0]
    assert result["noise_levels"] == [0.01]
    assert result["window_lengths"] == [4]
    assert result["num_cases"] == 1

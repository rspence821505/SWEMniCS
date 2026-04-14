"""Central registry for the standardized Shinnecock twin experiments."""

from .specs import TwinExperimentSpec


EXPERIMENT_REGISTRY = {
    "wse_wind_ramp": TwinExperimentSpec(
        name="wse_wind_ramp",
        description=(
            "State-estimation twin experiment over the initial hydrodynamic state "
            "with prescribed wind-ramp forcing. Wind is known and never estimated."
        ),
        inverse_problem_type="state",
        control_variables=["initial_state_u"],
        forward_model_config={
            "problem": "shinnecock",
            "solver": "DG",
            "dt": 600.0,
            "nt_ramp": 144,
            "nt_da": 12,
            "n_windows": 1,
            "forcing_model": "holland_hurricane",
            "wind_ramp_s": 43200.0,
            "track_shift_km": 15.0,
            "vmax_values": [0, 10, 20, 30, 40],
            "methods": ["4dvar", "dcwme_dynamic", "dcwme_static"],
            "adios_file": "data/shinnecock_inlet",
        },
        observation_config={
            "obs_fraction": 0.1,
            "obs_frequency": 6,
            "interior_only": True,
            "solution_var": "h",
        },
        noise_model={
            "obs_noise_level": 0.01,
            "background_error_std": 0.02,
            "obs_seed": 42,
            "background_seed": 123,
        },
        sweep_parameters={
            "vmax": [0, 10, 20, 30, 40],
        },
        output_dir="results/wse_wind_ramp",
    ),
    "wind_param": TwinExperimentSpec(
        name="wind_param",
        description=(
            "Low-dimensional wind-parameter inversion with fixed hydrodynamic "
            "state initial condition and explicit forcing controls."
        ),
        inverse_problem_type="low_dim_param",
        control_variables=[
            "track_shift_km",
            "intensity_bias",
            "timing_offset_s",
        ],
        forward_model_config={
            "problem": "shinnecock",
            "solver": "DG",
            "dt": 600.0,
            "nt_total": 156,
            "nt_ramp": 144,
            "forcing_model": "holland_hurricane",
            "wind_ramp_s": 43200.0,
            "truth_vmax": 30.0,
            "truth_track_shift_km": 0.0,
            "truth_intensity_bias": 0.0,
            "truth_timing_offset_s": 0.0,
            "background_parameters": [15.0, -0.10, 3600.0],
            "parameter_std": [10.0, 0.10, 3600.0],
            "bounds": [[-30.0, 30.0], [-0.4, 0.4], [-21600.0, 21600.0]],
            "fd_steps": [1.0, 0.01, 900.0],
            "max_iterations": 10,
            "adios_file": "data/shinnecock_inlet",
        },
        observation_config={
            "obs_fraction": 0.1,
            "obs_frequency": 6,
            "interior_only": True,
            "solution_var": "h",
        },
        noise_model={
            "obs_noise_level": 0.01,
            "obs_seed": 42,
        },
        sweep_parameters={
            "noise_levels": [0.005, 0.01, 0.02],
            "window_lengths": [12, 24],
        },
        output_dir="results/wind_param",
    ),
    "mannings_n": TwinExperimentSpec(
        name="mannings_n",
        description=(
            "Augmented state-plus-Manning's-n inversion with a smooth spatial "
            "basis for Manning coefficients and an explicit initial-state block."
        ),
        inverse_problem_type="distributed_param",
        control_variables=["initial_state_u", "mannings_n_basis_coefficients"],
        forward_model_config={
            "problem": "shinnecock",
            "solver": "DG",
            "dt": 600.0,
            "nt_total": 156,
            "nt_ramp": 144,
            "forcing_model": "holland_hurricane",
            "wind_ramp_s": 43200.0,
            "truth_vmax": 30.0,
            "basis_shape": [3, 2],
            "basis_width_fraction": 0.35,
            "truth_coefficients": [0.10, -0.08, 0.05, -0.06, 0.07, -0.04],
            "background_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "parameter_std": [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
            "bounds": [[-0.35, 0.35], [-0.35, 0.35], [-0.35, 0.35],
                       [-0.35, 0.35], [-0.35, 0.35], [-0.35, 0.35]],
            "fd_steps": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            "regularization_weight": 1.0,
            "n_clip": [0.01, 0.08],
            "max_iterations": 12,
            "adios_file": "data/shinnecock_inlet",
        },
        observation_config={
            "obs_fraction": 0.1,
            "obs_frequency": 6,
            "interior_only": True,
            "solution_var": "h",
        },
        noise_model={
            "obs_noise_level": 0.01,
            "obs_seed": 42,
            "background_seed": 123,
            "background_error_std": 0.02,
        },
        sweep_parameters={
            "regularization_weight": [0.5, 1.0, 2.0],
            "noise_levels": [0.005, 0.01, 0.02],
            "window_lengths": [12, 24],
        },
        output_dir="results/mannings_n",
    ),
}

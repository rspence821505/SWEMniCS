from dataclasses import dataclass


@dataclass(frozen=True)
class TwinExperimentSpec:
    name: str
    description: str
    inverse_problem_type: str  # "state", "low_dim_param", "distributed_param"
    control_variables: list[str]
    forward_model_config: dict
    observation_config: dict
    noise_model: dict
    sweep_parameters: dict
    output_dir: str

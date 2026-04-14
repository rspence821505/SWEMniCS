import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.shinnecock_study.run_comparison import (
    _get_sweep_value,
    _phase6_method_suite,
    _summarize_eigenvalues,
)


def test_get_sweep_value_respects_override():
    sweep_params = {"predictability_gamma": 0.5}
    assert _get_sweep_value(sweep_params, "predictability_gamma", 0.1) == 0.5
    assert _get_sweep_value(sweep_params, "obs_noise_level", 0.01) == 0.01


def test_phase6_method_suite_controlled_adds_ablations():
    legacy = _phase6_method_suite("noise", suite="legacy")
    controlled = _phase6_method_suite("noise", suite="controlled")

    assert [spec["variant_key"] for spec in legacy] == [
        "4dvar_baseline",
        "dcwme_static",
    ]
    assert [spec["variant_key"] for spec in controlled] == [
        "4dvar_baseline",
        "dcwme_static",
        "4dvar_eq38",
        "dcwme_dynamic",
    ]


def test_phase6_method_suite_limits_dynamic_to_validation_dims():
    controlled = _phase6_method_suite("obs_density", suite="controlled")
    assert [spec["variant_key"] for spec in controlled] == [
        "4dvar_baseline",
        "dcwme_static",
        "4dvar_eq38",
    ]


def test_summarize_eigenvalues_reports_condition_and_spread():
    summary = _summarize_eigenvalues([2.0, 4.0, 8.0])

    assert summary["count"] == 3
    assert math.isclose(summary["lambda_min"], 2.0)
    assert math.isclose(summary["lambda_max"], 8.0)
    assert math.isclose(summary["lambda_mean"], 14.0 / 3.0)
    assert math.isclose(summary["condition_number"], 4.0)
    assert math.isclose(summary["spread_pct"], 100.0 * 6.0 / (14.0 / 3.0))
    assert summary["rank_gt_1e-10"] == 3

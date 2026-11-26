"""Unit tests for `scripts.eval_uncertainty_abstention` risk-coverage & calibration metrics."""

from __future__ import annotations

import numpy as np

from scripts.eval_uncertainty_abstention import compute_selective_risk_curve


def test_selective_risk_curve_monotonic_with_calibrated_sigma() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    # Errors grow strictly with index, and sigma is proportional to true error.
    errors = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5])
    y_pred = y_true + errors
    sigma = errors * 1.1

    res = compute_selective_risk_curve(y_true, y_pred, sigma, metric="mse")
    assert res["spearman_sigma_err"] > 0.99
    assert res["cov70_risk"] < res["cov80_risk"] < res["full_risk"]
    assert res["aurc"] <= res["random_risk"][0]
    assert abs(res["e_aurc"]) < 1e-12
    assert 0.8 <= res["picp_90"] <= 1.0

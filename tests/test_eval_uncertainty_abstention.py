"""Unit tests for `scripts.eval_uncertainty_abstention` risk-coverage & calibration metrics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from scripts.eval_uncertainty_abstention import compute_selective_risk_curve
from src.models.baselines import MultiTaskHead, POLAR_LOGVAR_BOUNDS, SIZE_LOGVAR_BOUNDS
from src.training.metrics import compute_multitask_loss


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


def test_polar_logvar_unclamped_below_legacy_floor() -> None:
    torch.manual_seed(42)
    head = MultiTaskHead(64, 32, 4, predict_uncertainty=True)
    graph_emb = torch.randn(16, 64)
    node_emb = torch.randn(16, 32)
    out = head(graph_emb, node_emb)
    assert (out["size_logvar"] >= SIZE_LOGVAR_BOUNDS[0] - 1e-5).all()
    assert (out["size_logvar"] <= SIZE_LOGVAR_BOUNDS[1] + 1e-5).all()
    assert (out["polar_logvar"] >= POLAR_LOGVAR_BOUNDS[0] - 1e-5).all()
    assert (out["polar_logvar"] <= POLAR_LOGVAR_BOUNDS[1] + 1e-5).all()

    # Optimize head on tiny polarization residuals (~0.015) and verify polar_logvar drops below -6.0
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(120):
        opt.zero_grad()
        o = head(graph_emb, node_emb)
        batch = SimpleNamespace(
            y_size=o["y_size_pred"].detach() + 0.35,
            y_polar=o["y_polar_pred"].detach() + 0.015,
        )
        loss = compute_multitask_loss(o, batch, lambda_uncertainty=0.5, beta_nll=0.5)
        loss.backward()
        opt.step()

    final_out = head(graph_emb, node_emb)
    assert float(final_out["polar_logvar"].mean().item()) < -6.0
    assert float(torch.exp(0.5 * final_out["polar_logvar"]).mean().item()) < 0.0498

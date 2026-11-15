"""Multi-task loss functions and evaluation metrics for SBi-FAGAT."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, r2_score
from torch import Tensor


def compute_multitask_loss(
    outputs: dict[str, Tensor],
    batch: Any,
    *,
    alpha_size: float = 1.0,
    alpha_polar: float = 1.0,
    alpha_rumor: float = 1.0,
    huber_delta: float | None = None,
    huber_delta_size: float = 1.0,
    huber_delta_polar: float = 0.15,
    class_weights: Tensor | None = None,
    classification_only: bool = False,
) -> Tensor:
    """Compute multi-task Huber + Cross-Entropy loss (or classification-only for UPFD)."""
    logits = outputs["rumor_logits"]
    ce_weight = (
        class_weights.to(device=logits.device, dtype=logits.dtype)
        if class_weights is not None and class_weights.numel() == logits.size(-1)
        else None
    )

    if classification_only:
        y_cls = getattr(batch, "y_rumor", getattr(batch, "y", None))
        if y_cls is None:
            raise ValueError("Batch is missing classification labels ('y_rumor' or 'y')")
        return F.cross_entropy(logits, y_cls.view(-1).long(), weight=ce_weight)

    d_size = float(huber_delta) if huber_delta is not None else float(huber_delta_size)
    d_polar = float(huber_delta) if huber_delta is not None else float(huber_delta_polar)

    y_size = batch.y_size.view(-1).float()
    y_polar = batch.y_polar.view(-1).float()

    loss_size = F.huber_loss(outputs["y_size_pred"].view(-1), y_size, delta=d_size)
    loss_polar = F.huber_loss(outputs["y_polar_pred"].view(-1), y_polar, delta=d_polar)
    total_loss = alpha_size * loss_size + alpha_polar * loss_polar

    y_rumor = getattr(batch, "y_rumor", None)
    if y_rumor is not None:
        y_rumor_flat = y_rumor.view(-1).long()
        valid_mask = y_rumor_flat >= 0
        if valid_mask.any():
            loss_rumor = F.cross_entropy(
                logits[valid_mask],
                y_rumor_flat[valid_mask],
                weight=ce_weight,
            )
            total_loss = total_loss + alpha_rumor * loss_rumor

    return total_loss


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size >= 2 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        val = float(spearmanr(y_true, y_pred).statistic)
        return 0.0 if np.isnan(val) else val
    return 0.0


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size >= 2 and np.std(y_true) > 1e-12:
        return float(r2_score(y_true, y_pred))
    return 0.0


def evaluate_predictions(
    y_size_true: np.ndarray,
    y_size_pred: np.ndarray,
    y_polar_true: np.ndarray,
    y_polar_pred: np.ndarray,
    y_rumor_true: np.ndarray | None = None,
    rumor_logits: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute size (MSLE/RMSE/MAE/R^2/Spearman), polarization (RMSE/MAE/R^2/Pearson/Spearman), and rumor metrics.

    Note: Since `y_size = ln(1 + N_total)`, MSE between `y_size_pred` and `y_size_true`
    is mathematically identical to MSLE on raw reply counts `N_total`.
    """
    y_size_true = np.asarray(y_size_true, dtype=np.float64).reshape(-1)
    y_size_pred = np.asarray(y_size_pred, dtype=np.float64).reshape(-1)
    y_polar_true = np.asarray(y_polar_true, dtype=np.float64).reshape(-1)
    y_polar_pred = np.asarray(y_polar_pred, dtype=np.float64).reshape(-1)

    msle = float(np.mean((y_size_pred - y_size_true) ** 2)) if y_size_true.size > 0 else float("nan")
    rmse_size = float(np.sqrt(msle)) if y_size_true.size > 0 else float("nan")
    mae = float(np.mean(np.abs(y_size_pred - y_size_true))) if y_size_true.size > 0 else float("nan")
    r2_size = _safe_r2(y_size_true, y_size_pred) if y_size_true.size > 0 else float("nan")
    spearman_size = _safe_spearman(y_size_true, y_size_pred) if y_size_true.size > 0 else float("nan")

    rmse = (
        float(np.sqrt(np.mean((y_polar_pred - y_polar_true) ** 2)))
        if y_polar_true.size > 0
        else float("nan")
    )
    mae_polar = float(np.mean(np.abs(y_polar_pred - y_polar_true))) if y_polar_true.size > 0 else float("nan")
    r2_polar = _safe_r2(y_polar_true, y_polar_pred) if y_polar_true.size > 0 else float("nan")

    if (
        y_polar_true.size >= 2
        and np.std(y_polar_true) > 1e-12
        and np.std(y_polar_pred) > 1e-12
    ):
        pearson_val = float(pearsonr(y_polar_true, y_polar_pred).statistic)
        if np.isnan(pearson_val):
            pearson_val = 0.0
    else:
        pearson_val = 0.0
    spearman_polar = _safe_spearman(y_polar_true, y_polar_pred) if y_polar_true.size > 0 else float("nan")

    macro_f1 = float("nan")
    accuracy = float("nan")
    if y_rumor_true is not None and rumor_logits is not None:
        y_rumor_arr = np.asarray(y_rumor_true, dtype=np.int64).reshape(-1)
        valid_mask = y_rumor_arr >= 0
        if np.any(valid_mask):
            preds = np.argmax(np.asarray(rumor_logits)[valid_mask], axis=-1)
            truth = y_rumor_arr[valid_mask]
            macro_f1 = float(f1_score(truth, preds, average="macro", zero_division=0))
            accuracy = float(accuracy_score(truth, preds))

    return {
        "msle": msle,
        "mae": mae,
        "rmse_size": rmse_size,
        "r2_size": r2_size,
        "spearman_size": spearman_size,
        "rmse": rmse,
        "mae_polar": mae_polar,
        "r2_polar": r2_polar,
        "pearson_r": pearson_val,
        "spearman_polar": spearman_polar,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }

"""Closed-loop self-auditing validation optimizer & regime error analyzer for SBi-FAGAT v2.

Searches a bounded hyperparameter grid strictly on `(train_loader, val_loader)` — never
peeking at `test_loader` — and audits validation residuals across three practical failure modes:
  1. One-sided early echo chambers (`|mean(early s_uv)| > 0.6`)
  2. Low-signal early chatter (`substantive_text_ratio < 0.4`)
  3. High-velocity viral bursts (`burst_acceleration >= 0.6`)

Exports `loop_diagnostics.json` and `worst_val_cases_for_review.json` so teammates and AI
reviewers can inspect validation failure cases before locking final 5-seed test runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.models.sbi_fagat import SBiFAGAT
from src.training.trainer import evaluate_loader, train_and_evaluate_single_seed

DEFAULT_V2_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"hidden_dim": 256, "lr": 1e-3, "eps": 0.5, "dropout": 0.10, "lambda_uncertainty": 0.05},
    {"hidden_dim": 256, "lr": 5e-4, "eps": 0.4, "dropout": 0.15, "lambda_uncertainty": 0.05},
    {"hidden_dim": 256, "lr": 1e-3, "eps": 0.6, "dropout": 0.20, "lambda_uncertainty": 0.08},
    {"hidden_dim": 128, "lr": 1e-3, "eps": 0.5, "dropout": 0.10, "lambda_uncertainty": 0.05},
)


def _compute_regime_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0, "size_mae": None, "polar_rmse": None, "mean_confidence": None}
    size_errs = np.asarray([abs(r["y_size_pred"] - r["y_size_true"]) for r in records], dtype=np.float64)
    polar_sq_errs = np.asarray(
        [(r["y_polar_pred"] - r["y_polar_true"]) ** 2 for r in records], dtype=np.float64
    )
    confs = np.asarray([r["confidence_score"] for r in records], dtype=np.float64)
    return {
        "count": len(records),
        "size_mae": float(np.mean(size_errs)),
        "polar_rmse": float(np.sqrt(np.mean(polar_sq_errs))),
        "mean_confidence": float(np.mean(confs)),
    }


def run_self_auditing_validation_loop(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    in_dim: int = 770,
    edge_dim: int = 4,
    num_rumor_classes: int = 4,
    use_cascade_stats: bool = True,
    predict_uncertainty: bool = True,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    epochs: int = 60,
    patience: int = 10,
    seed: int = 42,
    output_dir: Path = Path("outputs/auto_loop"),
    device: torch.device | str | None = None,
    worst_k: int = 15,
) -> dict[str, Any]:
    """Run bounded validation optimization + regime residual diagnostics without touching test."""
    target_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    grid = [dict(c) for c in (candidates or DEFAULT_V2_CANDIDATES)]
    if not grid:
        raise ValueError("candidates grid must be non-empty")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_results: list[dict[str, Any]] = []
    best_entry: dict[str, Any] | None = None
    best_model: torch.nn.Module | None = None

    for idx, cfg in enumerate(grid):
        h_dim = int(cfg.get("hidden_dim", 256))
        lr = float(cfg.get("lr", 1e-3))
        eps = float(cfg.get("eps", 0.5))
        dropout = float(cfg.get("dropout", 0.1))
        lam_unc = float(cfg.get("lambda_uncertainty", 0.05 if predict_uncertainty else 0.0))

        def _factory(
            _h: int = h_dim,
            _eps: float = eps,
            _drop: float = dropout,
        ) -> SBiFAGAT:
            return SBiFAGAT(
                in_dim=in_dim,
                hidden_dim=_h,
                edge_dim=edge_dim,
                num_rumor_classes=num_rumor_classes,
                eps=_eps,
                dropout=_drop,
                use_cascade_stats=use_cascade_stats,
                predict_uncertainty=predict_uncertainty,
            )

        # Evaluate candidate strictly on validation loader (passing val_loader as eval target)
        model, val_metrics = train_and_evaluate_single_seed(
            _factory,
            train_loader,
            val_loader,
            val_loader,
            seed=seed,
            epochs=epochs,
            lr=lr,
            patience=patience,
            lambda_uncertainty=lam_unc,
            device=target_device,
            return_predictions=False,
        )
        clean_metrics = {
            k: (None if isinstance(v, float) and np.isnan(v) else float(v))
            for k, v in val_metrics.items()
            if isinstance(v, (int, float))
        }
        entry = {
            "candidate_id": idx,
            "config": {
                "hidden_dim": h_dim,
                "lr": lr,
                "eps": eps,
                "dropout": dropout,
                "lambda_uncertainty": lam_unc,
                "edge_dim": edge_dim,
                "use_cascade_stats": use_cascade_stats,
                "predict_uncertainty": predict_uncertainty,
            },
            "val_metrics": clean_metrics,
        }
        candidate_results.append(entry)

        cur_loss = float(val_metrics["loss"])
        if best_entry is None or cur_loss < float(best_entry["val_metrics"]["loss"]):
            best_entry = entry
            best_model = model

    assert best_entry is not None and best_model is not None

    # Sample-level validation audit across hard regimes
    best_model.eval()
    val_dataset = val_loader.dataset
    sample_records: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            item = val_dataset[idx]
            batch_item = item.to(target_device) if hasattr(item, "to") else item
            diag = best_model.predict_with_diagnostics(batch_item)  # type: ignore[attr-defined]
            y_size_true = float(item.y_size.view(-1)[0].item())
            y_polar_true = float(item.y_polar.view(-1)[0].item())
            y_size_pred = float(diag["y_size_pred"].view(-1)[0].item())
            y_polar_pred = float(diag["y_polar_pred"].view(-1)[0].item())
            conf = float(diag["confidence_score"].view(-1)[0].item())
            sub_ratio = float(diag["substantive_text_ratio"].view(-1)[0].item())

            c_stats = getattr(item, "cascade_stats", None)
            if c_stats is not None and c_stats.numel() >= 8:
                s_vec = c_stats.view(-1).tolist()
                burst_accel = float(s_vec[2])
                one_sided_host = float(s_vec[7])
            else:
                burst_accel = 0.5
                one_sided_host = 0.0

            ea = getattr(item, "edge_attr", None)
            mean_early_stance = (
                float(ea[:, 0].mean().item()) if ea is not None and ea.numel() > 0 else 0.0
            )
            err_score = abs(y_size_pred - y_size_true) + 4.0 * abs(y_polar_pred - y_polar_true)
            node_texts = list(getattr(item, "node_texts", []) or [])

            sample_records.append(
                {
                    "cascade_id": str(getattr(item, "cascade_id", f"val_{idx}")),
                    "topic_id": str(getattr(item, "topic_id", "unknown")),
                    "y_size_true": y_size_true,
                    "y_size_pred": y_size_pred,
                    "y_polar_true": y_polar_true,
                    "y_polar_pred": y_polar_pred,
                    "combined_error": err_score,
                    "confidence_score": conf,
                    "substantive_text_ratio": sub_ratio,
                    "burst_acceleration": burst_accel,
                    "one_sided_hostility_index": one_sided_host,
                    "mean_early_stance": mean_early_stance,
                    "low_signal_warning": bool(diag["low_signal_warning"].view(-1)[0].item()),
                    "root_snippet": node_texts[0][:240] if node_texts else "",
                    "early_replies_sample": [t[:160] for t in node_texts[1:4]],
                }
            )

    one_sided_cases = [r for r in sample_records if abs(r["mean_early_stance"]) > 0.6]
    low_signal_cases = [r for r in sample_records if r["substantive_text_ratio"] < 0.4]
    high_burst_cases = [r for r in sample_records if r["burst_acceleration"] >= 0.6]

    regime_diagnostics = {
        "overall_val": _compute_regime_metrics(sample_records),
        "one_sided_echo_chambers": _compute_regime_metrics(one_sided_cases),
        "low_signal_chatter": _compute_regime_metrics(low_signal_cases),
        "high_velocity_bursts": _compute_regime_metrics(high_burst_cases),
    }

    worst_cases = sorted(sample_records, key=lambda r: r["combined_error"], reverse=True)[
        : max(1, int(worst_k))
    ]

    best_ckpt_path = out_dir / "best_validated_sbifagat.pt"
    torch.save(
        {
            "config": best_entry["config"],
            "model_state_dict": best_model.state_dict(),
            "in_dim": in_dim,
            "num_rumor_classes": num_rumor_classes,
        },
        best_ckpt_path,
    )

    diagnostics_payload = {
        "best_config": best_entry["config"],
        "best_val_metrics": best_entry["val_metrics"],
        "candidate_results": candidate_results,
        "regime_diagnostics": regime_diagnostics,
        "best_checkpoint_path": str(best_ckpt_path),
    }
    diag_json = out_dir / "loop_diagnostics.json"
    worst_json = out_dir / "worst_val_cases_for_review.json"
    diag_json.write_text(json.dumps(diagnostics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    worst_json.write_text(json.dumps(worst_cases, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "best_config": best_entry["config"],
        "best_val_metrics": best_entry["val_metrics"],
        "candidate_results": candidate_results,
        "regime_diagnostics": regime_diagnostics,
        "worst_val_cases": worst_cases,
        "best_checkpoint_path": str(best_ckpt_path),
        "diagnostics_json": str(diag_json),
        "worst_cases_json": str(worst_json),
    }

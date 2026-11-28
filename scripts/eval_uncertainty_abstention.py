"""5-Seed beta-NLL Heteroscedastic Uncertainty & Selective Abstention Evaluation.

Trains `SBiFAGAT(..., predict_uncertainty=True)` using Seitzer et al. (ICLR 2022)
beta-NLL (beta=0.5, detached residuals in `src.training.metrics`) across the
5 official seeds `[42, 100, 2024, 777, 999]` on VN Canonical (`K=20, N>=25`)
and PHEME Canonical (`K=10, N>=15`), then evaluates:
  - Zero-degradation check vs deterministic regression metrics
  - Uncertainty rank calibration Spearman(sigma_hat, |y - mu_hat|) & PICP@90%
  - Selective Abstention Risk-Coverage curves (Coverage 100% -> 50%), AURC, E-AURC
    comparing beta-NLL uncertainty gating vs Random and Oracle abstention.

Exports:
  - `results/results_uncertainty_abstention.json`
  - `results/tables_uncertainty_abstention.tex`
  - `results/figures/vn/fig7_uncertainty_risk_coverage.{pdf,svg}`
  - `checkpoints/best_sbifagat_vn_unc.pt` (gitignored, never overwrites `best_sbifagat_vn.pt`)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pyg_dataset import (  # noqa: E402
    DEFAULT_FORECAST_MIN_REPLIES,
    EarlyCascadeDataset,
    TOTAL_NODE_FEATURE_DIM,
    load_cascade_parquet,
)
from src.models.baselines import POLAR_LOGVAR_BOUNDS, SIZE_LOGVAR_BOUNDS  # noqa: E402
from src.models.sbi_fagat import SBiFAGAT  # noqa: E402
from src.training.metrics import evaluate_predictions  # noqa: E402
from src.training.trainer import train_and_evaluate_single_seed  # noqa: E402
from src.utils.seed import OFFICIAL_SEEDS  # noqa: E402
from src.utils.visualize import _configure_academic_rcparams, _save_vector_pair  # noqa: E402

COVERAGE_GRID: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def compute_selective_risk_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    *,
    coverages: Sequence[float] = COVERAGE_GRID,
    metric: str = "mse",
) -> dict[str, Any]:
    """Compute Risk-Coverage curve, AURC, E-AURC, PICP@90%, and error-sigma Spearman."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    sig = np.asarray(sigma, dtype=np.float64).reshape(-1)
    n = int(yt.size)
    if n == 0:
        raise ValueError("Empty prediction array passed to compute_selective_risk_curve")

    sq_err = (yp - yt) ** 2
    abs_err = np.abs(yp - yt)
    full_mse = float(np.mean(sq_err))
    full_rmse = float(np.sqrt(full_mse))

    order_unc = np.argsort(sig, kind="mergesort")
    order_oracle = np.argsort(abs_err, kind="mergesort")

    cov_list = sorted({float(c) for c in coverages}, reverse=True)
    unc_risks: list[float] = []
    oracle_risks: list[float] = []
    rand_risks: list[float] = []

    for cov in cov_list:
        k_keep = max(1, min(n, int(round(cov * n))))
        sel_u = sq_err[order_unc[:k_keep]]
        sel_o = sq_err[order_oracle[:k_keep]]
        r_u = float(np.mean(sel_u)) if metric == "mse" else float(np.sqrt(np.mean(sel_u)))
        r_o = float(np.mean(sel_o)) if metric == "mse" else float(np.sqrt(np.mean(sel_o)))
        r_rand = full_mse if metric == "mse" else full_rmse
        unc_risks.append(r_u)
        oracle_risks.append(r_o)
        rand_risks.append(r_rand)

    cov_asc = np.asarray(cov_list[::-1], dtype=np.float64)
    unc_asc = np.asarray(unc_risks[::-1], dtype=np.float64)
    ora_asc = np.asarray(oracle_risks[::-1], dtype=np.float64)
    span = max(1e-8, float(cov_asc[-1] - cov_asc[0]))
    aurc = float(np.trapezoid(unc_asc, cov_asc) / span)
    aurc_oracle = float(np.trapezoid(ora_asc, cov_asc) / span)

    sp_res = spearmanr(sig, abs_err)
    sp_val = 0.0 if (sp_res.statistic is None or np.isnan(sp_res.statistic)) else float(sp_res.statistic)
    picp_90 = float(np.mean(abs_err <= 1.6448536269514722 * np.maximum(sig, 1e-8)))

    return {
        "coverages": cov_list,
        "uncertainty_risk": unc_risks,
        "oracle_risk": oracle_risks,
        "random_risk": rand_risks,
        "aurc": aurc,
        "aurc_oracle": aurc_oracle,
        "e_aurc": float(aurc - aurc_oracle),
        "spearman_sigma_err": sp_val,
        "picp_90": picp_90,
        "full_risk": unc_risks[0],
        "cov80_risk": unc_risks[cov_list.index(0.8)] if 0.8 in cov_list else unc_risks[-1],
        "cov70_risk": unc_risks[cov_list.index(0.7)] if 0.7 in cov_list else unc_risks[-1],
    }


def _collect_uncertainty_predictions(
    model: SBiFAGAT,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    ys_t, ys_p, ys_sig = [], [], []
    yp_t, yp_p, yp_sig = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            ys_t.append(batch.y_size.view(-1).cpu().numpy())
            ys_p.append(out["y_size_pred"].view(-1).cpu().numpy())
            s_lv = out["size_logvar"].view(-1).clamp(*SIZE_LOGVAR_BOUNDS)
            ys_sig.append(torch.exp(0.5 * s_lv).cpu().numpy())

            yp_t.append(batch.y_polar.view(-1).cpu().numpy())
            yp_p.append(out["y_polar_pred"].view(-1).cpu().numpy())
            p_lv = out["polar_logvar"].view(-1).clamp(*POLAR_LOGVAR_BOUNDS)
            yp_sig.append(torch.exp(0.5 * p_lv).cpu().numpy())
    return {
        "y_size_true": np.concatenate(ys_t),
        "y_size_pred": np.concatenate(ys_p),
        "sigma_size": np.concatenate(ys_sig),
        "y_polar_true": np.concatenate(yp_t),
        "y_polar_pred": np.concatenate(yp_p),
        "sigma_polar": np.concatenate(yp_sig),
    }


def _agg_mean_std(vals: Sequence[float]) -> list[float]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [0.0, 0.0]
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    return [float(np.mean(arr)), std]


def _run_dataset_uncertainty(
    *,
    dataset_name: str,
    data_dir: Path,
    k_horizon: int,
    min_total_replies: int,
    epochs: int,
    patience: int,
    batch_size: int,
    lambda_uncertainty: float,
    seeds: Sequence[int],
    ckpt_out: Path | None,
    device: torch.device,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    nodes_tbl, edges_tbl = load_cascade_parquet(data_dir, validate=True)
    nlp_feats = torch.load(data_dir / "nlp_features.pt", map_location="cpu", weights_only=False)
    common = {
        "root_dir": data_dir,
        "k_horizon": k_horizon,
        "min_total_replies": min_total_replies,
        "polar_target_mode": "future_only",
        "label_scheme": "veracity4",
        "edge_source": "auto",
        "nodes_table": nodes_tbl,
        "edges_table": edges_tbl,
        "nlp_features": nlp_feats,
    }
    train_ds = EarlyCascadeDataset(split="train", **common)
    val_ds = EarlyCascadeDataset(split="val", **common)
    test_ds = EarlyCascadeDataset(split="test", **common)
    del nlp_feats

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    seed_records: list[dict[str, Any]] = []
    best_val_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for seed in seeds:
        def _factory() -> SBiFAGAT:
            return SBiFAGAT(
                in_dim=TOTAL_NODE_FEATURE_DIM,
                hidden_dim=64,
                edge_dim=2,
                num_rumor_classes=4,
                eps=0.5,
                dropout=0.1,
                use_cascade_stats=True,
                predict_uncertainty=True,
            )

        model, test_m = train_and_evaluate_single_seed(
            _factory,
            train_loader,
            val_loader,
            test_loader,
            seed=int(seed),
            epochs=epochs,
            lr=1e-3,
            patience=patience,
            lambda_uncertainty=lambda_uncertainty,
            device=device,
        )
        preds = _collect_uncertainty_predictions(model, test_loader, device)
        reg_m = evaluate_predictions(
            y_size_true=preds["y_size_true"],
            y_size_pred=preds["y_size_pred"],
            y_polar_true=preds["y_polar_true"],
            y_polar_pred=preds["y_polar_pred"],
        )
        size_curve = compute_selective_risk_curve(
            preds["y_size_true"], preds["y_size_pred"], preds["sigma_size"], metric="mse"
        )
        polar_curve = compute_selective_risk_curve(
            preds["y_polar_true"], preds["y_polar_pred"], preds["sigma_polar"], metric="rmse"
        )
        val_tl = float(test_m.get("val_task_loss", float("inf")))
        if ckpt_out is not None and val_tl < best_val_score:
            best_val_score = val_tl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        seed_records.append(
            {
                "seed": int(seed),
                "regression": reg_m,
                "sigma_size_mean": float(np.mean(preds["sigma_size"])),
                "sigma_polar_mean": float(np.mean(preds["sigma_polar"])),
                "size_abstention": size_curve,
                "polar_abstention": polar_curve,
            }
        )

    if ckpt_out is not None and best_state is not None:
        ckpt_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": {
                    "hidden_dim": 64,
                    "edge_dim": 2,
                    "eps": 0.5,
                    "dropout": 0.1,
                    "use_cascade_stats": True,
                    "predict_uncertainty": True,
                    "use_polar_persistence_anchor": True,
                },
                "model_state_dict": best_state,
                "in_dim": TOTAL_NODE_FEATURE_DIM,
                "num_rumor_classes": 4,
            },
            ckpt_out,
        )

    # Aggregate mean +- std over seeds
    def _agg_curve(key: str) -> dict[str, Any]:
        covs = seed_records[0][key]["coverages"]
        unc_mat = np.array([r[key]["uncertainty_risk"] for r in seed_records], dtype=np.float64)
        ora_mat = np.array([r[key]["oracle_risk"] for r in seed_records], dtype=np.float64)
        rnd_mat = np.array([r[key]["random_risk"] for r in seed_records], dtype=np.float64)
        full_r = _agg_mean_std([r[key]["full_risk"] for r in seed_records])
        c80_r = _agg_mean_std([r[key]["cov80_risk"] for r in seed_records])
        c70_r = _agg_mean_std([r[key]["cov70_risk"] for r in seed_records])
        drop80 = float((full_r[0] - c80_r[0]) / max(1e-8, full_r[0]) * 100.0)
        drop70 = float((full_r[0] - c70_r[0]) / max(1e-8, full_r[0]) * 100.0)
        return {
            "coverages": covs,
            "uncertainty_risk_mean": unc_mat.mean(axis=0).tolist(),
            "uncertainty_risk_std": unc_mat.std(axis=0, ddof=1 if len(seed_records) >= 2 else 0).tolist(),
            "oracle_risk_mean": ora_mat.mean(axis=0).tolist(),
            "random_risk_mean": rnd_mat.mean(axis=0).tolist(),
            "full_risk": full_r,
            "cov80_risk": c80_r,
            "cov70_risk": c70_r,
            "rel_drop_80_pct": drop80,
            "rel_drop_70_pct": drop70,
            "aurc": _agg_mean_std([r[key]["aurc"] for r in seed_records]),
            "e_aurc": _agg_mean_std([r[key]["e_aurc"] for r in seed_records]),
            "spearman_sigma_err": _agg_mean_std([r[key]["spearman_sigma_err"] for r in seed_records]),
            "picp_90": _agg_mean_std([r[key]["picp_90"] for r in seed_records]),
        }

    reg_summary = {
        k: _agg_mean_std([r["regression"][k] for r in seed_records])
        for k in ("msle", "r2_size", "rmse", "r2_polar", "pearson_r", "spearman_polar")
    }
    elapsed = time.perf_counter() - t0
    return {
        "dataset": dataset_name,
        "k_horizon": k_horizon,
        "min_total_replies": min_total_replies,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "lambda_uncertainty": lambda_uncertainty,
        "beta_nll": 0.5,
        "size_logvar_bounds": list(SIZE_LOGVAR_BOUNDS),
        "polar_logvar_bounds": list(POLAR_LOGVAR_BOUNDS),
        "sigma_size_mean_std": _agg_mean_std([r["sigma_size_mean"] for r in seed_records]),
        "sigma_polar_mean_std": _agg_mean_std([r["sigma_polar_mean"] for r in seed_records]),
        "elapsed_sec": round(elapsed, 2),
        "regression_summary": reg_summary,
        "size_abstention": _agg_curve("size_abstention"),
        "polar_abstention": _agg_curve("polar_abstention"),
    }


def _export_uncertainty_artifacts(
    results: Mapping[str, Any],
    *,
    output_tex: Path,
    figures_dir: Path,
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{$\beta$-NLL Heteroscedastic Uncertainty Calibration and Selective Abstention on \texttt{SBiFAGAT} ($\beta=0.5, \lambda_u=0.1$, 5 Seeds, Mean $\pm$ Std).}",
        r"\label{tab:uncertainty_abstention}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Corpus & Target & $\rho(\hat{\sigma}, |e|)$ $\uparrow$ & PICP@90\% $\uparrow$ & Risk @ 100\% $\downarrow$ & Risk @ 80\% $\downarrow$ & Risk @ 70\% $\downarrow$ & $\Delta$ @ 70\% & E-AURC $\downarrow$ \\",
        r"\midrule",
    ]
    for ds_key, ds_lbl in (("vn", "VN ($K=20$)"), ("pheme", "PHEME ($K=10$)")):
        ds = results["datasets"][ds_key]
        for t_key, t_lbl in (("size_abstention", "Size (MSLE)"), ("polar_abstention", "Polar (RMSE)")):
            c = ds[t_key]
            lines.append(
                f"{ds_lbl} & {t_lbl} & "
                f"{c['spearman_sigma_err'][0]:.3f} $\\pm$ {c['spearman_sigma_err'][1]:.3f} & "
                f"{c['picp_90'][0]:.3f} $\\pm$ {c['picp_90'][1]:.3f} & "
                f"{c['full_risk'][0]:.4f} & "
                f"{c['cov80_risk'][0]:.4f} & "
                f"{c['cov70_risk'][0]:.4f} & "
                f"{c['rel_drop_70_pct']:+.1f}\\% & "
                f"{c['e_aurc'][0]:.4f} \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _configure_academic_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    panels = (
        (axes[0, 0], "vn", "size_abstention", "(a) VN Size Risk-Coverage (MSLE)", "Cascade Size MSLE"),
        (axes[0, 1], "pheme", "size_abstention", "(b) PHEME Size Risk-Coverage (MSLE)", "Cascade Size MSLE"),
        (axes[1, 0], "vn", "polar_abstention", "(c) VN Polarization Risk-Coverage (RMSE)", "Polarization RMSE"),
        (axes[1, 1], "pheme", "polar_abstention", "(d) PHEME Polarization Risk-Coverage (RMSE)", "Polarization RMSE"),
    )
    for ax, ds_k, t_k, title, ylabel in panels:
        c = results["datasets"][ds_k][t_k]
        covs = np.asarray(c["coverages"], dtype=np.float64) * 100.0
        u_m = np.asarray(c["uncertainty_risk_mean"], dtype=np.float64)
        u_s = np.asarray(c["uncertainty_risk_std"], dtype=np.float64)
        r_m = np.asarray(c["random_risk_mean"], dtype=np.float64)
        o_m = np.asarray(c["oracle_risk_mean"], dtype=np.float64)

        ax.plot(covs, r_m, "--", color="#7F7F7F", linewidth=1.6, label="Random Abstention")
        ax.plot(covs, u_m, "-o", color="#D55E00", linewidth=2.2, markersize=5, label=r"$\beta$-NLL $\hat{\sigma}$ Abstention")
        ax.fill_between(covs, u_m - u_s, u_m + u_s, color="#D55E00", alpha=0.15)
        ax.plot(covs, o_m, ":", color="#009E73", linewidth=1.8, label="Oracle Error Bound")
        ax.invert_xaxis()
        ax.set_xlabel("Coverage (% Retained Predictions)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", frameon=False, fontsize=8.5)
        ax.grid(True, alpha=0.25, linestyle="--")

    fig.tight_layout()
    _save_vector_pair(fig, figures_dir, "fig7_uncertainty_risk_coverage")


def main() -> None:
    parser = argparse.ArgumentParser(description="5-seed beta-NLL selective abstention benchmark.")
    parser.add_argument("--vn-dir", type=Path, default=Path("data/processed/vn"))
    parser.add_argument("--pheme-dir", type=Path, default=Path("data/processed/pheme"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lambda-uncertainty", type=float, default=0.1)
    parser.add_argument("--ckpt-out", type=Path, default=Path("checkpoints/best_sbifagat_vn_unc.pt"))
    parser.add_argument("--output-json", type=Path, default=Path("results/results_uncertainty_abstention.json"))
    parser.add_argument("--output-tex", type=Path, default=Path("results/tables_uncertainty_abstention.tex"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures/vn"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vn_res = _run_dataset_uncertainty(
        dataset_name="vn",
        data_dir=args.vn_dir,
        k_horizon=20,
        min_total_replies=DEFAULT_FORECAST_MIN_REPLIES,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lambda_uncertainty=args.lambda_uncertainty,
        seeds=OFFICIAL_SEEDS,
        ckpt_out=args.ckpt_out,
        device=device,
    )
    pheme_res = _run_dataset_uncertainty(
        dataset_name="pheme",
        data_dir=args.pheme_dir,
        k_horizon=10,
        min_total_replies=15,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lambda_uncertainty=args.lambda_uncertainty,
        seeds=OFFICIAL_SEEDS,
        ckpt_out=None,
        device=device,
    )

    payload = {
        "is_real_data": True,
        "device": str(device),
        "seeds": list(OFFICIAL_SEEDS),
        "datasets": {"vn": vn_res, "pheme": pheme_res},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _export_uncertainty_artifacts(payload, output_tex=args.output_tex, figures_dir=args.figures_dir)
    print(f"[EvalUncertainty] Saved {args.output_json}, {args.output_tex}, and fig7 in {args.figures_dir}", flush=True)


if __name__ == "__main__":
    main()

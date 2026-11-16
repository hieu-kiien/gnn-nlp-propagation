"""Main 5-seed multi-model benchmark, paired t-test, and figure generation script."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.pyg_dataset import (
    EarlyCascadeDataset,
    TOTAL_NODE_FEATURE_DIM,
    leave_one_event_out_splits,
)
from src.data.schema import load_cascade_parquet
from src.models.baselines import (
    BiGCN,
    EdgeGATv2,
    FAGCNBaseline,
    StandardGATv2,
    StandardGCN,
    TemporalStanceMLP,
    TextMLP,
)
from src.models.sbi_fagat import SBiFAGAT
from src.training.reporting import (
    MAIN_METRICS,
    VN_MAIN_METRICS,
    build_markdown_table,
    export_latex_table,
    serialize_benchmark_json,
)
from src.training.trainer import (
    compute_paired_bootstrap_pvalue,
    compute_paired_ttest,
    holm_bonferroni_adjust,
    run_multi_seed_benchmark,
    select_best_comparator,
)
from src.utils.seed import OFFICIAL_SEEDS
from src.utils.visualize import plot_cascade_heatmap, plot_umap_polarization


def _resolve_output_paths(
    data_dir: Path,
    output_table: Path | None,
    output_json: Path | None,
    figures_dir: Path | None,
) -> tuple[Path, Path, Path]:
    ds_name = data_dir.name.lower() if data_dir.name.lower() in {"pheme", "vn"} else ""
    suffix = f"_{ds_name}" if ds_name else ""
    tbl = output_table or Path(f"paper/tables_main{suffix}.tex")
    jsn = output_json or Path(f"paper/results_main{suffix}.json")
    figs = figures_dir or (Path("paper/figures") / ds_name if ds_name else Path("paper/figures"))
    return tbl, jsn, figs


def _build_loaders(
    data_dir: Path,
    k_horizon: int,
    batch_size: int,
    *,
    min_total_replies: int = 0,
    polar_target_mode: str = "full",
    label_scheme: str = "veracity4",
    custom_topic_splits: dict[str, set[str]] | None = None,
    nodes_table: Any = None,
    edges_table: Any = None,
    nlp_features: Any = None,
) -> tuple[EarlyCascadeDataset, DataLoader, DataLoader, DataLoader]:
    if nodes_table is None or edges_table is None or nlp_features is None:
        required = ("nodes.parquet", "edges.parquet", "nlp_features.pt")
        missing = [f for f in required if not (data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing processed dataset files in '{data_dir}': {missing}. "
                "Run Phase 2 preprocessing first; synthetic data is prohibited in official reports."
            )

    common = {
        "root_dir": data_dir,
        "k_horizon": k_horizon,
        "min_total_replies": min_total_replies,
        "polar_target_mode": polar_target_mode,
        "label_scheme": label_scheme,
        "custom_topic_splits": custom_topic_splits,
        "nodes_table": nodes_table,
        "edges_table": edges_table,
        "nlp_features": nlp_features,
    }
    train_ds = EarlyCascadeDataset(split="train", **common)
    val_ds = EarlyCascadeDataset(split="val", **common)
    test_ds = EarlyCascadeDataset(split="test", **common)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return test_ds, train_loader, val_loader, test_loader


def _export_diagnostic_figures(
    test_ds: EarlyCascadeDataset,
    test_loader: DataLoader,
    gcn_model: torch.nn.Module,
    sbifagat_model: SBiFAGAT,
    figures_dir: Path,
    device: torch.device,
) -> None:
    gcn_model.eval()
    sbifagat_model.eval()

    gcn_embs: list[np.ndarray] = []
    sbi_embs: list[np.ndarray] = []
    node_stances: list[np.ndarray] = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            gcn_out = gcn_model(batch)
            sbi_out = sbifagat_model(batch)
            gcn_embs.append(gcn_out["node_emb"].cpu().numpy())
            sbi_embs.append(sbi_out["node_emb"].cpu().numpy())

            # Map parent->child stance onto each node (root nodes stay 0.0)
            stances = torch.zeros(batch.num_nodes, dtype=torch.float32, device=device)
            if batch.edge_index_td.numel() > 0 and hasattr(batch, "edge_stance"):
                dst = batch.edge_index_td[1]
                stances[dst] = batch.edge_stance.view(-1).to(device)
            node_stances.append(stances.cpu().numpy())

    if gcn_embs:
        plot_umap_polarization(
            np.concatenate(gcn_embs, axis=0),
            np.concatenate(sbi_embs, axis=0),
            np.concatenate(node_stances, axis=0),
            output_dir=figures_dir,
        )

    # Select the most polarized test cascade with >= 2 edges for Fig 3 heatmap
    candidates = [d for d in test_ds if d.edge_index_td.size(1) >= 2]
    if candidates:
        sample = max(candidates, key=lambda d: float(d.y_polar.item()))
        gates = sbifagat_model.extract_edge_gates(sample.to(device))
        plot_cascade_heatmap(
            sample.cpu(),
            gates["edge_importance"],
            gates["gamma_td"],
            output_dir=figures_dir,
        )


def _infer_edge_dim(data_dir: Path, nlp_features: dict[str, Any] | None = None) -> int:
    feats = nlp_features
    if feats is None:
        feat_path = data_dir / "nlp_features.pt"
        if feat_path.exists():
            feats = torch.load(feat_path, map_location="cpu", weights_only=False)
    if feats is not None:
        if "edge_dim" in feats:
            return int(feats["edge_dim"])
        raw_attr = feats.get("edge_attr")
        if isinstance(raw_attr, torch.Tensor) and raw_attr.ndim == 2 and raw_attr.size(1) in {2, 4}:
            return int(raw_attr.size(1))
    return 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5-seed benchmark for SBi-FAGAT and baselines.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--edge-dim", type=int, choices=[2, 4], default=None)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--v2", action="store_true", help="Enable v2 burst stats + uncertainty + 0.05 NLL reg.")
    parser.add_argument("--use-cascade-stats", action="store_true")
    parser.add_argument("--predict-uncertainty", action="store_true")
    parser.add_argument("--lambda-uncertainty", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-total-replies", type=int, default=21)
    parser.add_argument(
        "--polar-target-mode", choices=["full", "future_only"], default="future_only"
    )
    parser.add_argument(
        "--label-scheme", choices=["veracity4", "binary"], default="veracity4"
    )
    parser.add_argument(
        "--eval-protocol", choices=["inductive", "loo"], default="inductive"
    )
    parser.add_argument("--comparator-model", type=str, default="auto")
    parser.add_argument("--output-table", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    out_table, out_json, fig_dir = _resolve_output_paths(
        args.data_dir, args.output_table, args.output_json, args.figures_dir
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = TOTAL_NODE_FEATURE_DIM
    h_dim = args.hidden_dim
    edge_dim = args.edge_dim if args.edge_dim is not None else _infer_edge_dim(args.data_dir)
    use_stats = bool(args.use_cascade_stats or args.v2)
    use_unc = bool(args.predict_uncertainty or args.v2)
    lam_unc = (
        float(args.lambda_uncertainty)
        if args.lambda_uncertainty is not None
        else (0.05 if use_unc else 0.0)
    )
    num_cls = 2 if args.label_scheme == "binary" else 4
    model_factories = {
        "TextMLP": lambda: TextMLP(in_dim=in_dim, hidden_dim=h_dim, num_rumor_classes=num_cls, dropout=args.dropout),
        "TemporalStanceMLP": lambda: TemporalStanceMLP(
            in_dim=in_dim, hidden_dim=h_dim, edge_dim=edge_dim, num_rumor_classes=num_cls, dropout=args.dropout
        ),
        "StandardGCN": lambda: StandardGCN(in_dim=in_dim, hidden_dim=h_dim, num_rumor_classes=num_cls, dropout=args.dropout),
        "StandardGATv2": lambda: StandardGATv2(in_dim=in_dim, hidden_dim=h_dim, num_rumor_classes=num_cls, dropout=args.dropout),
        "EdgeGATv2": lambda: EdgeGATv2(in_dim=in_dim, hidden_dim=h_dim, edge_dim=edge_dim, num_rumor_classes=num_cls, dropout=args.dropout),
        "FAGCN": lambda: FAGCNBaseline(in_dim=in_dim, hidden_dim=h_dim, num_rumor_classes=num_cls, eps=args.eps, dropout=args.dropout),
        "BiGCN": lambda: BiGCN(in_dim=in_dim, hidden_dim=h_dim, num_rumor_classes=num_cls, dropout=args.dropout),
        "SBiFAGAT": lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
    }

    if args.eval_protocol == "inductive":
        test_ds, train_loader, val_loader, test_loader = _build_loaders(
            args.data_dir,
            args.k_horizon,
            args.batch_size,
            min_total_replies=args.min_total_replies,
            polar_target_mode=args.polar_target_mode,
            label_scheme=args.label_scheme,
        )
        has_rumor_labels = any(int(d.y_rumor.item()) >= 0 for d in test_ds)
        active_metrics = MAIN_METRICS if has_rumor_labels else VN_MAIN_METRICS
        result = run_multi_seed_benchmark(
            model_factories,
            train_loader,
            val_loader,
            test_loader,
            seeds=OFFICIAL_SEEDS,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            lambda_uncertainty=lam_unc,
            reference_model="SBiFAGAT",
            comparator_model=args.comparator_model,
            device=device,
        )
    else:
        nodes_tbl, edges_tbl = load_cascade_parquet(args.data_dir, validate=True)
        nlp_feats = torch.load(
            args.data_dir / "nlp_features.pt", map_location="cpu", weights_only=False
        )
        base_all = EarlyCascadeDataset(
            root_dir=args.data_dir,
            k_horizon=args.k_horizon,
            split="all",
            nodes_table=nodes_tbl,
            edges_table=edges_tbl,
            nlp_features=nlp_feats,
            min_total_replies=args.min_total_replies,
            polar_target_mode=args.polar_target_mode,
            label_scheme=args.label_scheme,
        )
        folds = leave_one_event_out_splits(base_all._all_cascades)
        has_rumor_labels = any(int(d.y_rumor.item()) >= 0 for d in base_all._all_cascades)
        active_metrics = MAIN_METRICS if has_rumor_labels else VN_MAIN_METRICS
        fold_raw: dict[str, list[dict[str, float]]] = {m: [] for m in model_factories}
        last_bundle: dict[str, Any] = {}
        for fold in folds:
            t_ds, tr_l, va_l, te_l = _build_loaders(
                args.data_dir,
                args.k_horizon,
                args.batch_size,
                min_total_replies=args.min_total_replies,
                polar_target_mode=args.polar_target_mode,
                label_scheme=args.label_scheme,
                custom_topic_splits=fold,
                nodes_table=nodes_tbl,
                edges_table=edges_tbl,
                nlp_features=nlp_feats,
            )
            fold_res = run_multi_seed_benchmark(
                model_factories,
                tr_l,
                va_l,
                te_l,
                seeds=OFFICIAL_SEEDS,
                epochs=args.epochs,
                lr=args.lr,
                patience=args.patience,
                lambda_uncertainty=lam_unc,
                reference_model="SBiFAGAT",
                comparator_model=args.comparator_model,
                device=device,
            )
            for m_name, m_stats in fold_res["summary_stats"].items():
                fold_raw[m_name].append({k: v[0] for k, v in m_stats.items()})
            test_ds, test_loader, last_bundle = t_ds, te_l, fold_res

        summary_stats: dict[str, dict[str, tuple[float, float]]] = {}
        metric_keys = tuple(fold_raw["SBiFAGAT"][0].keys())
        for m_name, rows in fold_raw.items():
            agg: dict[str, tuple[float, float]] = {}
            for k in metric_keys:
                arr = np.asarray([r[k] for r in rows], dtype=np.float64)
                arr = arr[~np.isnan(arr)]
                agg[k] = (
                    (float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0)
                    if arr.size > 0
                    else (float("nan"), float("nan"))
                )
            summary_stats[m_name] = agg
        cmp_name = (
            select_best_comparator(summary_stats, reference_model="SBiFAGAT")
            if args.comparator_model == "auto"
            else args.comparator_model
        )
        p_vals = {
            k: compute_paired_ttest(
                [r[k] for r in fold_raw["SBiFAGAT"]],
                [r[k] for r in fold_raw[cmp_name]],
            )
            for k in metric_keys
        }
        boot_p_vals = {
            k: compute_paired_bootstrap_pvalue(
                [r[k] for r in fold_raw["SBiFAGAT"]],
                [r[k] for r in fold_raw[cmp_name]],
            )
            for k in metric_keys
        }
        result = {
            "seeds": list(OFFICIAL_SEEDS),
            "raw_results": fold_raw,
            "summary_stats": summary_stats,
            "p_values": p_vals,
            "adjusted_p_values": holm_bonferroni_adjust(p_vals),
            "bootstrap_p_values": boot_p_vals,
            "adjusted_bootstrap_p_values": holm_bonferroni_adjust(boot_p_vals),
            "reference_model": "SBiFAGAT",
            "comparator_model": cmp_name,
            "models": last_bundle["models"],
        }

    print(f"\n=== Main Benchmark Results (5 Seeds, K={args.k_horizon}, {args.eval_protocol}) ===")
    print(build_markdown_table(result, active_metrics))

    cmp_caption = str(result.get("comparator_model") or "BiGCN").replace("_", "\\_")
    export_latex_table(
        result,
        out_table,
        caption=(
            f"Early Cascade Size, Polarization, and Rumor Prediction at $K={args.k_horizon}$ "
            f"({args.eval_protocol.upper()}, Mean $\\pm$ Std). Stars indicate significance "
            f"vs. {cmp_caption} ($^{{*}}p<0.05$, $^{{**}}p<0.01$)."
        ),
        label="tab:main_results",
        metrics=active_metrics,
    )
    serialize_benchmark_json(
        result,
        out_json,
        extra={
            "k_horizon": args.k_horizon,
            "eval_protocol": args.eval_protocol,
            "polar_target_mode": args.polar_target_mode,
            "min_total_replies": args.min_total_replies,
            "label_scheme": args.label_scheme,
        },
    )

    if not args.skip_figures:
        _export_diagnostic_figures(
            test_ds,
            test_loader,
            result["models"]["StandardGCN"],
            result["models"]["SBiFAGAT"],
            fig_dir,
            device,
        )


if __name__ == "__main__":
    main()

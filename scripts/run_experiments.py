"""Main 5-seed multi-model benchmark, paired t-test, and figure generation script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from scripts.export_dataset_stats import compute_early_k_persistence_metrics
from src.data.pyg_dataset import (
    DEFAULT_FORECAST_MIN_REPLIES,
    EarlyCascadeDataset,
    TOTAL_NODE_FEATURE_DIM,
    compute_embedding_cluster_splits,
    compute_temporal_topic_splits,
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
    *,
    split_mode: str = "canonical",
) -> tuple[Path, Path, Path]:
    ds_name = data_dir.name.lower() if data_dir.name.lower() in {"pheme", "vn"} else ""
    suffix = f"_{ds_name}" if ds_name else ""
    prefix = (
        "temporal"
        if split_mode == "temporal"
        else ("cluster" if split_mode == "embedding_cluster" else "main")
    )
    tbl = output_table or Path(f"paper/tables_{prefix}{suffix}.tex")
    jsn = output_json or Path(f"paper/results_{prefix}{suffix}.json")
    if split_mode != "canonical" and figures_dir is None:
        figs = Path("paper/figures") / (f"{ds_name}_{prefix}" if ds_name else prefix)
    else:
        figs = figures_dir or (Path("paper/figures") / ds_name if ds_name else Path("paper/figures"))
    return tbl, jsn, figs


def _resolve_main_table_label(
    data_dir: Path,
    eval_protocol: str,
    out_table: Path,
    *,
    split_mode: str = "canonical",
) -> str:
    ds_tag = data_dir.name.lower() if data_dir.name.lower() in {"pheme", "vn"} else ""
    if not ds_tag:
        stem = out_table.stem.lower()
        if "pheme" in stem:
            ds_tag = "pheme"
        elif "vn" in stem:
            ds_tag = "vn"
    if str(eval_protocol).lower() == "loo":
        prefix = "loo"
    elif split_mode == "temporal":
        prefix = "temporal"
    elif split_mode == "embedding_cluster":
        prefix = "cluster"
    else:
        prefix = "main"
    return f"tab:{prefix}_{ds_tag}" if ds_tag else f"tab:{prefix}_results"


def _build_loaders(
    data_dir: Path,
    k_horizon: int,
    batch_size: int,
    *,
    min_total_replies: int = 0,
    polar_target_mode: str = "full",
    label_scheme: str = "veracity4",
    custom_topic_splits: dict[str, set[str]] | None = None,
    edge_source: str = "auto",
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
    if nodes_table is None or edges_table is None:
        nodes_table, edges_table = load_cascade_parquet(data_dir, validate=True)
    if nlp_features is None:
        nlp_features = torch.load(
            data_dir / "nlp_features.pt",
            map_location="cpu",
            weights_only=False,
        )

    common = {
        "root_dir": data_dir,
        "k_horizon": k_horizon,
        "min_total_replies": min_total_replies,
        "polar_target_mode": polar_target_mode,
        "label_scheme": label_scheme,
        "custom_topic_splits": custom_topic_splits,
        "edge_source": edge_source,
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

            # Prefer causal root-aligned stance orientations on V_K for UMAP coloring
            if hasattr(batch, "early_node_root_stance") and batch.early_node_root_stance is not None:
                stances = batch.early_node_root_stance.view(-1).to(device)
            else:
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


def _build_persistence_summary_row(
    metrics: dict[str, float] | dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Convert single-split or aggregated Early-K Persistence dict into summary_stats format."""
    reg_keys = (
        "msle",
        "mae",
        "rmse_size",
        "r2_size",
        "spearman_size",
        "rmse",
        "mae_polar",
        "r2_polar",
        "pearson_r",
        "spearman_polar",
    )
    all_keys = reg_keys + ("macro_f1", "accuracy", "auc_roc", "loss", "task_loss")
    row: dict[str, tuple[float, float]] = {}
    for k in all_keys:
        if k in reg_keys and k in metrics:
            val = metrics[k]
            if isinstance(val, tuple):
                row[k] = (float(val[0]), float(val[1]))
            else:
                v = float(val)
                row[k] = (v, 0.0 if not np.isnan(v) else float("nan"))
        else:
            row[k] = (float("nan"), float("nan"))
    return row


def attach_early_k_persistence_to_result(
    result: dict[str, Any],
    test_cascades: Sequence[Any],
    *,
    train_cascades: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Attach the non-learned Early-K Persistence baseline row onto a benchmark result."""
    persist_m = compute_early_k_persistence_metrics(
        list(test_cascades),
        train_cascades=list(train_cascades) if train_cascades is not None else None,
    )
    result["early_k_persistence"] = persist_m
    result["summary_stats"] = {
        "Early-K Persistence": _build_persistence_summary_row(persist_m),
        **result["summary_stats"],
    }
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run 5-seed benchmark for SBi-FAGAT and baselines.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
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
    parser.add_argument("--min-total-replies", type=int, default=DEFAULT_FORECAST_MIN_REPLIES)
    parser.add_argument(
        "--polar-target-mode", choices=["full", "future_only"], default="future_only"
    )
    parser.add_argument(
        "--label-scheme", choices=["veracity4", "binary"], default="veracity4"
    )
    parser.add_argument(
        "--eval-protocol", choices=["inductive", "loo"], default="inductive"
    )
    parser.add_argument(
        "--split-mode",
        choices=["canonical", "temporal", "embedding_cluster"],
        default="canonical",
        help="Holdout split strategy under inductive eval; non-canonical splits force teacher edges.",
    )
    parser.add_argument(
        "--edge-source", choices=["auto", "student", "teacher"], default="auto"
    )
    parser.add_argument("--comparator-model", type=str, default="auto")
    parser.add_argument("--output-table", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--ckpt-out", type=Path, default=None)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    out_table, out_json, fig_dir = _resolve_output_paths(
        args.data_dir,
        args.output_table,
        args.output_json,
        args.figures_dir,
        split_mode=args.split_mode,
    )
    effective_edge_source = (
        "teacher" if args.split_mode != "canonical" else args.edge_source
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
        "TextMLP": lambda: TextMLP(
            in_dim=in_dim,
            hidden_dim=h_dim,
            num_rumor_classes=num_cls,
            dropout=args.dropout,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
        "TemporalStanceMLP": lambda: TemporalStanceMLP(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            dropout=args.dropout,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
        "StandardGCN": lambda: StandardGCN(
            in_dim=in_dim,
            hidden_dim=h_dim,
            num_rumor_classes=num_cls,
            dropout=args.dropout,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
        "StandardGATv2": lambda: StandardGATv2(
            in_dim=in_dim,
            hidden_dim=h_dim,
            num_rumor_classes=num_cls,
            dropout=args.dropout,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
        "EdgeGATv2": lambda: EdgeGATv2(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            dropout=args.dropout,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
        "FAGCN": lambda: FAGCNBaseline(
            in_dim=in_dim,
            hidden_dim=h_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
        "BiGCN": lambda: BiGCN(
            in_dim=in_dim,
            hidden_dim=h_dim,
            num_rumor_classes=num_cls,
            dropout=args.dropout,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
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
        custom_splits: dict[str, set[str]] | None = None
        if args.split_mode != "canonical":
            nodes_tbl, edges_tbl = load_cascade_parquet(args.data_dir, validate=True)
            nlp_feats = torch.load(
                args.data_dir / "nlp_features.pt", map_location="cpu", weights_only=False
            )
            all_ds = EarlyCascadeDataset(
                root_dir=args.data_dir,
                k_horizon=args.k_horizon,
                split="all",
                nodes_table=nodes_tbl,
                edges_table=edges_tbl,
                nlp_features=nlp_feats,
                min_total_replies=args.min_total_replies,
                polar_target_mode=args.polar_target_mode,
                label_scheme=args.label_scheme,
                edge_source=effective_edge_source,
            )
            if args.split_mode == "temporal":
                custom_splits = compute_temporal_topic_splits(all_ds._all_cascades)
            else:
                custom_splits = compute_embedding_cluster_splits(
                    all_ds._all_cascades, seed=OFFICIAL_SEEDS[0]
                )
            test_ds, train_loader, val_loader, test_loader = _build_loaders(
                args.data_dir,
                args.k_horizon,
                args.batch_size,
                min_total_replies=args.min_total_replies,
                polar_target_mode=args.polar_target_mode,
                label_scheme=args.label_scheme,
                custom_topic_splits=custom_splits,
                edge_source=effective_edge_source,
                nodes_table=nodes_tbl,
                edges_table=edges_tbl,
                nlp_features=nlp_feats,
            )
        else:
            test_ds, train_loader, val_loader, test_loader = _build_loaders(
                args.data_dir,
                args.k_horizon,
                args.batch_size,
                min_total_replies=args.min_total_replies,
                polar_target_mode=args.polar_target_mode,
                label_scheme=args.label_scheme,
                edge_source=effective_edge_source,
            )
        has_rumor_labels = any(int(d.y_rumor.item()) >= 0 for d in test_ds)
        active_metrics = MAIN_METRICS if has_rumor_labels else VN_MAIN_METRICS
        active_keys = tuple(k for k, _, _ in active_metrics)
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
            primary_hypothesis_keys=active_keys,
            device=device,
        )
        attach_early_k_persistence_to_result(
            result,
            list(test_ds),
            train_cascades=list(train_loader.dataset),
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
            edge_source="teacher",
        )
        folds = leave_one_event_out_splits(
            base_all._all_cascades, min_cascades_per_fold=15
        )
        has_rumor_labels = any(int(d.y_rumor.item()) >= 0 for d in base_all._all_cascades)
        active_metrics = MAIN_METRICS if has_rumor_labels else VN_MAIN_METRICS
        active_keys = [k for k, _, _ in active_metrics]
        fold_raw: dict[str, list[dict[str, float]]] = {m: [] for m in model_factories}
        persist_folds: list[dict[str, Any]] = []
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
                edge_source="teacher",
                nodes_table=nodes_tbl,
                edges_table=edges_tbl,
                nlp_features=nlp_feats,
            )
            persist_folds.append(
                compute_early_k_persistence_metrics(
                    list(t_ds),
                    train_cascades=list(tr_l.dataset),
                )
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
                compute_bootstrap=False,
                primary_hypothesis_keys=active_keys,
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

        persist_agg: dict[str, tuple[float, float]] = {}
        for pk in (
            "msle",
            "mae",
            "rmse_size",
            "r2_size",
            "spearman_size",
            "rmse",
            "mae_polar",
            "r2_polar",
            "pearson_r",
            "spearman_polar",
        ):
            parr = np.asarray([float(pf.get(pk, float("nan"))) for pf in persist_folds], dtype=np.float64)
            parr = parr[~np.isnan(parr)]
            persist_agg[pk] = (
                (float(np.mean(parr)), float(np.std(parr, ddof=1)) if parr.size >= 2 else 0.0)
                if parr.size > 0
                else (float("nan"), float("nan"))
            )
        summary_stats = {
            "Early-K Persistence": _build_persistence_summary_row(persist_agg),
            **summary_stats,
        }

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
        adj_p_scoped = holm_bonferroni_adjust({k: p_vals[k] for k in active_keys if k in p_vals})
        adj_boot_scoped = holm_bonferroni_adjust(
            {k: boot_p_vals[k] for k in active_keys if k in boot_p_vals}
        )
        result = {
            "seeds": list(OFFICIAL_SEEDS),
            "raw_results": fold_raw,
            "summary_stats": summary_stats,
            "early_k_persistence": {
                k: [None if np.isnan(v[0]) else v[0], None if np.isnan(v[1]) else v[1]]
                for k, v in persist_agg.items()
            },
            "p_values": p_vals,
            "adjusted_p_values": {k: adj_p_scoped.get(k, float("nan")) for k in metric_keys},
            "bootstrap_p_values": boot_p_vals,
            "adjusted_bootstrap_p_values": {
                k: adj_boot_scoped.get(k, float("nan")) for k in metric_keys
            },
            "reference_model": "SBiFAGAT",
            "comparator_model": cmp_name,
            "nonparametric_test_label": (
                "Exact Perm." if len(fold_raw["SBiFAGAT"]) < 6 else "Bootstrap"
            ),
            "models": last_bundle["models"],
        }

    print(f"\n=== Main Benchmark Results (5 Seeds, K={args.k_horizon}, {args.eval_protocol}) ===")
    print(build_markdown_table(result, active_metrics))

    cmp_caption = str(result.get("comparator_model") or "BiGCN").replace("_", "\\_")
    table_label = _resolve_main_table_label(
        args.data_dir, args.eval_protocol, out_table, split_mode=args.split_mode
    )
    proto_label = (
        args.split_mode.replace("_", "-").upper()
        if args.eval_protocol == "inductive" and args.split_mode != "canonical"
        else args.eval_protocol.upper()
    )
    task_desc = (
        "Early Cascade Size, Polarization, and Rumor Prediction"
        if has_rumor_labels
        else "Early Cascade Size and Future Polarization Forecasting"
    )
    export_latex_table(
        result,
        out_table,
        caption=(
            f"{task_desc} at $K={args.k_horizon}$ "
            f"({proto_label}, Mean $\\pm$ Std). Stars indicate significance "
            f"vs. {cmp_caption} ($^{{*}}p<0.05$, $^{{**}}p<0.01$)."
        ),
        label=table_label,
        metrics=active_metrics,
    )
    serialize_benchmark_json(
        result,
        out_json,
        extra={
            "k_horizon": args.k_horizon,
            "eval_protocol": args.eval_protocol,
            "split_mode": args.split_mode,
            "edge_source": effective_edge_source,
            "polar_target_mode": args.polar_target_mode,
            "min_total_replies": args.min_total_replies,
            "label_scheme": args.label_scheme,
            "use_cascade_stats": use_stats,
            "early_k_persistence": result.get("early_k_persistence"),
        },
    )

    if "SBiFAGAT" in result.get("models", {}) and (
        args.split_mode == "canonical" or args.ckpt_out is not None
    ):
        ds_tag = args.data_dir.name.lower() if args.data_dir.name.lower() in {"pheme", "vn"} else "main"
        proto_tag = "_loo" if args.eval_protocol == "loo" else ""
        ckpt_path = args.ckpt_out or Path(f"checkpoints/best_sbifagat_{ds_tag}{proto_tag}.pt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": {
                    "hidden_dim": h_dim,
                    "edge_dim": edge_dim,
                    "eps": args.eps,
                    "dropout": args.dropout,
                    "use_cascade_stats": use_stats,
                    "predict_uncertainty": use_unc,
                    "use_polar_persistence_anchor": bool(
                        getattr(result["models"]["SBiFAGAT"], "use_polar_persistence_anchor", False)
                    ),
                },
                "model_state_dict": result["models"]["SBiFAGAT"].state_dict(),
                "in_dim": in_dim,
                "num_rumor_classes": num_cls,
            },
            ckpt_path,
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

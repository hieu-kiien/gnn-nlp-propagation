"""Ablation study (2D edge components, conflict gate, bottom-up) and K-horizon sensitivity."""

from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.data.pyg_dataset import (
    DEFAULT_FORECAST_MIN_REPLIES,
    EarlyCascadeDataset,
    SUPPORTED_HORIZONS,
    TOTAL_NODE_FEATURE_DIM,
    load_topic_splits_json,
)
from src.data.schema import load_cascade_parquet
from src.models.baselines import BiGCN, StandardGCN
from src.models.sbi_fagat import SBiFAGAT
from src.training.reporting import (
    MAIN_METRICS,
    VN_MAIN_METRICS,
    build_markdown_table,
    export_latex_table,
    serialize_benchmark_json,
)
from src.training.trainer import run_multi_seed_benchmark
from src.utils.seed import OFFICIAL_SEEDS
from src.utils.visualize import plot_early_k_curve


def _resolve_ablation_table_label(data_dir: Path, out_table: Path) -> str:
    ds_tag = data_dir.name.lower() if data_dir.name.lower() in {"pheme", "vn"} else ""
    if not ds_tag:
        stem = out_table.stem.lower()
        if "pheme" in stem:
            ds_tag = "pheme"
        elif "vn" in stem:
            ds_tag = "vn"
    return f"tab:ablation_{ds_tag}" if ds_tag else "tab:ablation_results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SBi-FAGAT ablation study and K-horizon curve.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(SUPPORTED_HORIZONS))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--edge-dim", type=int, choices=[2, 4], default=None)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--v2", action="store_true")
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
    parser.add_argument("--comparator-model", type=str, default="auto")
    parser.add_argument("--output-table", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()

    ds_name = args.data_dir.name.lower() if args.data_dir.name.lower() in {"pheme", "vn"} else ""
    suffix = f"_{ds_name}" if ds_name else ""
    out_table = args.output_table or Path(f"paper/tables_ablation{suffix}.tex")
    out_json = args.output_json or Path(f"paper/results_ablation{suffix}.json")
    fig_dir = args.figures_dir or (Path("paper/figures") / ds_name if ds_name else Path("paper/figures"))

    for fname in ("nodes.parquet", "edges.parquet", "nlp_features.pt"):
        if not (args.data_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing '{args.data_dir / fname}'. Run Phase 2 preprocessing first."
            )

    nodes_table, edges_table = load_cascade_parquet(args.data_dir, validate=True)
    nlp_features = torch.load(
        args.data_dir / "nlp_features.pt", map_location="cpu", weights_only=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = TOTAL_NODE_FEATURE_DIM
    h_dim = args.hidden_dim
    edge_dim = args.edge_dim or int(
        nlp_features.get("edge_dim", nlp_features["edge_attr"].size(1) if nlp_features["edge_attr"].ndim == 2 else 2)
    )
    if edge_dim not in {2, 4}:
        edge_dim = 2
    use_stats = bool(args.use_cascade_stats or args.v2)
    use_unc = bool(args.predict_uncertainty or args.v2)
    lam_unc = (
        float(args.lambda_uncertainty)
        if args.lambda_uncertainty is not None
        else (0.05 if use_unc else 0.0)
    )
    num_cls = 2 if args.label_scheme == "binary" else 4
    canonical_splits_path = args.data_dir / "topic_splits.json"
    canonical_splits = (
        load_topic_splits_json(canonical_splits_path)
        if canonical_splits_path.is_file()
        else None
    )

    common_ds_kwargs = {
        "root_dir": args.data_dir,
        "nodes_table": nodes_table,
        "edges_table": edges_table,
        "nlp_features": nlp_features,
        "min_total_replies": args.min_total_replies,
        "polar_target_mode": args.polar_target_mode,
        "label_scheme": args.label_scheme,
        "custom_topic_splits": canonical_splits,
    }

    # Part 1: Component Ablation at K=20
    train_ds = EarlyCascadeDataset(k_horizon=args.k_horizon, split="train", **common_ds_kwargs)
    val_ds = EarlyCascadeDataset(k_horizon=args.k_horizon, split="val", **common_ds_kwargs)
    test_ds = EarlyCascadeDataset(k_horizon=args.k_horizon, split="test", **common_ds_kwargs)
    has_rumor_labels = any(int(d.y_rumor.item()) >= 0 for d in test_ds)
    active_metrics = MAIN_METRICS if has_rumor_labels else VN_MAIN_METRICS

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    ablation_factories = {
        "SBiFAGAT (Full)": lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
        "w/o Stance Score (s_uv=0)": lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            ablate_stance=True,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
        "w/o Hostility Score (h_uv=0)": lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            ablate_hostility=True,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
        "w/o Edge Signals (s_uv=0, h_uv=0)": lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            ablate_stance=True,
            ablate_hostility=True,
            ablate_irrationality=(edge_dim == 4),
            ablate_constructive=(edge_dim == 4),
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
        "w/o Conflict Gate (Sigmoid)": lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            gate_mode="sigmoid",
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
        "w/o Bottom-Up": lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            use_bottom_up=False,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
        "w/o Attention Readout (Mean)": lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            readout_mode="mean",
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
    }
    if edge_dim == 4:
        ablation_factories["w/o Irrationality (r_uv=0)"] = lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            ablate_irrationality=True,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        )
        ablation_factories["w/o Constructive (c_uv=0)"] = lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            ablate_constructive=True,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        )
    if use_stats:
        ablation_factories["w/o Burst Stats"] = lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            use_cascade_stats=False,
            predict_uncertainty=use_unc,
        )

    ablation_res = run_multi_seed_benchmark(
        ablation_factories,
        train_loader,
        val_loader,
        test_loader,
        seeds=OFFICIAL_SEEDS,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        lambda_uncertainty=lam_unc,
        reference_model="SBiFAGAT (Full)",
        comparator_model="w/o Stance Score (s_uv=0)",
        device=device,
    )

    print("\n=== Ablation Study Results (5 Seeds, K={}) ===".format(args.k_horizon))
    print(build_markdown_table(ablation_res, active_metrics))

    export_latex_table(
        ablation_res,
        out_table,
        caption=(
            f"Ablation Study of SBi-FAGAT Components at $K={args.k_horizon}$ Across 5 Random Seeds. "
            "Separates stance polarity $s_{uv}$, hostility $h_{uv}$, signed $\\tanh$ gate, "
            "Bottom-Up feedback, and Global Attention Readout."
        ),
        label=_resolve_ablation_table_label(args.data_dir, out_table),
        metrics=active_metrics,
    )
    del train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    gc.collect()

    # Part 2: Early Horizon Sensitivity K in {10, 20, 30, 50} on a common cascade subset
    k_horizons_sorted = sorted(args.horizons)
    common_min_replies = max(int(args.min_total_replies), max(k_horizons_sorted) + 1)
    k_common_kwargs = dict(common_ds_kwargs)
    k_common_kwargs["min_total_replies"] = common_min_replies

    probe_splits = [
        len(EarlyCascadeDataset(k_horizon=k_horizons_sorted[-1], split=sp, **k_common_kwargs))
        for sp in ("train", "val", "test")
    ]
    used_common_filter = all(cnt >= 1 for cnt in probe_splits)
    if not used_common_filter:
        warnings.warn(
            f"Common K-horizon filter min_total_replies={common_min_replies} produced empty split "
            f"{probe_splits}; falling back to min_total_replies={args.min_total_replies}.",
            stacklevel=1,
        )
        k_common_kwargs["min_total_replies"] = args.min_total_replies

    k_curve_results: dict[int, dict[str, dict[str, tuple[float, float]]]] = {}
    for k in k_horizons_sorted:
        k_train = EarlyCascadeDataset(k_horizon=k, split="train", **k_common_kwargs)
        k_val = EarlyCascadeDataset(k_horizon=k, split="val", **k_common_kwargs)
        k_test = EarlyCascadeDataset(k_horizon=k, split="test", **k_common_kwargs)

        k_res = run_multi_seed_benchmark(
            {
                "StandardGCN": lambda: StandardGCN(
                    in_dim=in_dim,
                    hidden_dim=h_dim,
                    num_rumor_classes=num_cls,
                    dropout=args.dropout,
                    use_cascade_stats=use_stats,
                ),
                "BiGCN": lambda: BiGCN(
                    in_dim=in_dim,
                    hidden_dim=h_dim,
                    num_rumor_classes=num_cls,
                    dropout=args.dropout,
                    use_cascade_stats=use_stats,
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
            },
            DataLoader(k_train, batch_size=args.batch_size, shuffle=True),
            DataLoader(k_val, batch_size=args.batch_size, shuffle=False),
            DataLoader(k_test, batch_size=args.batch_size, shuffle=False),
            seeds=OFFICIAL_SEEDS,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            lambda_uncertainty=lam_unc,
            reference_model="SBiFAGAT",
            comparator_model=args.comparator_model,
            device=device,
        )
        k_curve_results[k] = k_res["summary_stats"]
        del k_train, k_val, k_test
        gc.collect()

    plot_early_k_curve(k_curve_results, output_dir=fig_dir)

    serialize_benchmark_json(
        ablation_res,
        out_json,
        extra={
            "k_horizon": args.k_horizon,
            "k_curve_common_min_replies": k_common_kwargs["min_total_replies"],
            "k_curve": {
                str(k): {
                    model_name: {
                        m_key: [stats[0], stats[1]]
                        for m_key, stats in m_stats.items()
                        if m_key in {"msle", "mae", "r2_size", "rmse", "pearson_r", "spearman_polar", "macro_f1", "accuracy"}
                    }
                    for model_name, m_stats in models_dict.items()
                }
                for k, models_dict in k_curve_results.items()
            },
        },
    )


if __name__ == "__main__":
    main()

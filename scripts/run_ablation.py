"""Ablation study (2D edge components, conflict gate, bottom-up) and K-horizon sensitivity."""

from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pyg_dataset import (
    DEFAULT_FORECAST_MIN_REPLIES,
    EarlyCascadeDataset,
    SUPPORTED_HORIZONS,
    TOTAL_NODE_FEATURE_DIM,
    load_topic_splits_json,
)
from src.data.schema import load_cascade_parquet
from src.models.baselines import BiEdgeGATv2, BiGCN, StandardGCN
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run SBi-FAGAT ablation study and K-horizon curve.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(SUPPORTED_HORIZONS))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--edge-dim", type=int, choices=[2], default=None)
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
    parser.add_argument(
        "--edge-source", choices=["auto", "student", "teacher"], default="auto"
    )
    parser.add_argument("--comparator-model", type=str, default="BiEdgeGATv2 (Matched)")
    parser.add_argument("--output-table", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()

    ds_name = args.data_dir.name.lower() if args.data_dir.name.lower() in {"pheme", "vn"} else ""
    suffix = f"_{ds_name}" if ds_name else ""
    out_table = args.output_table or Path(f"results/tables_ablation{suffix}.tex")
    out_json = args.output_json or Path(f"results/results_ablation{suffix}.json")
    fig_dir = args.figures_dir or (Path("results/figures") / ds_name if ds_name else Path("results/figures"))

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
        "edge_source": args.edge_source,
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
        "w/o Camp Dipole & Moments": lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            use_camp_dipole=False,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
        "w/o Dir-GNN Mixing": lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            use_dir_mix=False,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
        "w/o Dual-Channel Split": lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            use_dual_channel=False,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
        "w/o Persistence Anchor": lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            eps=args.eps,
            dropout=args.dropout,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
            use_polar_persistence_anchor=False,
        ),
        "BiEdgeGATv2 (Matched)": lambda: BiEdgeGATv2(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_cls,
            dropout=args.dropout,
            use_cascade_stats=use_stats,
            predict_uncertainty=use_unc,
        ),
    }
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

    active_keys = tuple(k for k, _, _ in active_metrics)
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
        comparator_model=args.comparator_model,
        primary_hypothesis_keys=active_keys,
        device=device,
    )

    print("\n=== Ablation Study Results (5 Seeds, K={}) ===".format(args.k_horizon))
    print(build_markdown_table(ablation_res, active_metrics))

    cmp_abl = str(ablation_res.get("comparator_model") or "BiEdgeGATv2 (Matched)").replace("_", "\\_").replace("&", "\\&")
    cmp_desc = (
        f"against a matched {cmp_abl} comparator "
        if cmp_abl == "BiEdgeGATv2 (Matched)"
        else f"(significance comparator: {cmp_abl}, "
    )
    sig_suffix = (
        f"(significance stars vs. {cmp_abl}, Holm-Bonferroni adjusted)."
        if cmp_abl == "BiEdgeGATv2 (Matched)"
        else "Holm-Bonferroni adjusted)."
    )
    export_latex_table(
        ablation_res,
        out_table,
        caption=(
            f"Ablation Study of SBi-FAGAT Components at $K={args.k_horizon}$ Across 5 Random Seeds. "
            "Separates stance polarity $s_{uv}$, hostility $h_{uv}$, signed $\\tanh$ gate, "
            "Bottom-Up feedback, Dir-GNN mixing, dual-channel split, Camp Dipole \\& Moments readout, "
            "Persistence-Residual anchoring, 8-D Burst Stats, and Global Attention Readout "
            f"{cmp_desc}{sig_suffix}"
        ),
        label=_resolve_ablation_table_label(args.data_dir, out_table),
        metrics=active_metrics,
    )
    del train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    gc.collect()

    # Part 2: Early Horizon Sensitivity K in {10, 20, 30, 50} on a common cascade subset
    # with a FIXED future polarization target window (t > K_max) across every K.
    k_horizons_sorted = sorted(args.horizons)
    active_horizons = list(k_horizons_sorted)
    k_common_kwargs = dict(common_ds_kwargs)
    common_min_replies = max(int(args.min_total_replies), max(active_horizons) + 5)

    while len(active_horizons) >= 1:
        cand_min = max(int(args.min_total_replies), max(active_horizons) + 5)
        k_common_kwargs["min_total_replies"] = cand_min
        if args.polar_target_mode == "future_only":
            k_common_kwargs["polar_future_start_k"] = max(active_horizons)
        try:
            probe_splits = [
                len(EarlyCascadeDataset(k_horizon=active_horizons[-1], split=sp, **k_common_kwargs))
                for sp in ("train", "val", "test")
            ]
        except ValueError:
            probe_splits = [0, 0, 0]
        if all(cnt >= 1 for cnt in probe_splits):
            common_min_replies = cand_min
            break
        if len(active_horizons) == 1:
            warnings.warn(
                f"Common K-horizon filter min_total_replies={cand_min} produced empty split "
                f"{probe_splits}; retaining min_total_replies={args.min_total_replies} for K={active_horizons}.",
                stacklevel=1,
            )
            common_min_replies = int(args.min_total_replies)
            k_common_kwargs["min_total_replies"] = common_min_replies
            k_common_kwargs.pop("polar_future_start_k", None)
            break
        dropped_k = active_horizons.pop()
        warnings.warn(
            f"Dropping horizon K={dropped_k} from K-curve because min_total_replies={cand_min} "
            f"left empty split {probe_splits}; evaluating remaining horizons {active_horizons} "
            "on an identical fixed future target window.",
            stacklevel=1,
        )

    k_curve_results: dict[int, dict[str, dict[str, tuple[float, float]]]] = {}
    k_cmp = (
        args.comparator_model
        if args.comparator_model in {"StandardGCN", "BiGCN"}
        else "auto"
    )
    for k in active_horizons:
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
            comparator_model=k_cmp,
            primary_hypothesis_keys=active_keys,
            device=device,
        )
        k_curve_results[k] = k_res["summary_stats"]
        del k_train, k_val, k_test
        gc.collect()

    plot_early_k_curve(k_curve_results, output_dir=fig_dir, target_k=int(args.k_horizon))

    serialize_benchmark_json(
        ablation_res,
        out_json,
        extra={
            "k_horizon": args.k_horizon,
            "k_curve_active_horizons": active_horizons,
            "k_curve_common_min_replies": common_min_replies,
            "k_curve_polar_future_start_k": k_common_kwargs.get("polar_future_start_k"),
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

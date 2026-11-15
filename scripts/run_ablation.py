"""Ablation study (2D edge components, conflict gate, bottom-up) and K-horizon sensitivity."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.data.pyg_dataset import EarlyCascadeDataset, SUPPORTED_HORIZONS, TOTAL_NODE_FEATURE_DIM
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SBi-FAGAT ablation study and K-horizon curve.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(SUPPORTED_HORIZONS))
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-total-replies", type=int, default=0)
    parser.add_argument(
        "--polar-target-mode", choices=["full", "future_only"], default="full"
    )
    parser.add_argument(
        "--label-scheme", choices=["veracity4", "binary"], default="veracity4"
    )
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
    num_cls = 2 if args.label_scheme == "binary" else 4

    common_ds_kwargs = {
        "root_dir": args.data_dir,
        "nodes_table": nodes_table,
        "edges_table": edges_table,
        "nlp_features": nlp_features,
        "min_total_replies": args.min_total_replies,
        "polar_target_mode": args.polar_target_mode,
        "label_scheme": args.label_scheme,
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
            in_dim=in_dim, hidden_dim=h_dim, edge_dim=2, num_rumor_classes=num_cls
        ),
        "w/o Stance Score (s_uv=0)": lambda: SBiFAGAT(
            in_dim=in_dim, hidden_dim=h_dim, edge_dim=2, num_rumor_classes=num_cls, ablate_stance=True
        ),
        "w/o Hostility Score (h_uv=0)": lambda: SBiFAGAT(
            in_dim=in_dim, hidden_dim=h_dim, edge_dim=2, num_rumor_classes=num_cls, ablate_hostility=True
        ),
        "w/o Edge Signals (s_uv=0, h_uv=0)": lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=2,
            num_rumor_classes=num_cls,
            ablate_stance=True,
            ablate_hostility=True,
        ),
        "w/o Conflict Gate (Sigmoid)": lambda: SBiFAGAT(
            in_dim=in_dim, hidden_dim=h_dim, edge_dim=2, num_rumor_classes=num_cls, gate_mode="sigmoid"
        ),
        "w/o Bottom-Up": lambda: SBiFAGAT(
            in_dim=in_dim, hidden_dim=h_dim, edge_dim=2, num_rumor_classes=num_cls, use_bottom_up=False
        ),
        "w/o Attention Readout (Mean)": lambda: SBiFAGAT(
            in_dim=in_dim, hidden_dim=h_dim, edge_dim=2, num_rumor_classes=num_cls, readout_mode="mean"
        ),
    }

    ablation_res = run_multi_seed_benchmark(
        ablation_factories,
        train_loader,
        val_loader,
        test_loader,
        seeds=OFFICIAL_SEEDS,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
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
        label="tab:ablation_results",
        metrics=active_metrics,
    )
    del train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    gc.collect()

    # Part 2: Early Horizon Sensitivity K in {10, 20, 30, 50} reusing loaded tables/tensors
    k_curve_results: dict[int, dict[str, dict[str, tuple[float, float]]]] = {}
    for k in sorted(args.horizons):
        k_train = EarlyCascadeDataset(k_horizon=k, split="train", **common_ds_kwargs)
        k_val = EarlyCascadeDataset(k_horizon=k, split="val", **common_ds_kwargs)
        k_test = EarlyCascadeDataset(k_horizon=k, split="test", **common_ds_kwargs)

        k_res = run_multi_seed_benchmark(
            {
                "StandardGCN": lambda: StandardGCN(in_dim=in_dim, hidden_dim=h_dim, num_rumor_classes=num_cls),
                "BiGCN": lambda: BiGCN(in_dim=in_dim, hidden_dim=h_dim, num_rumor_classes=num_cls),
                "SBiFAGAT": lambda: SBiFAGAT(in_dim=in_dim, hidden_dim=h_dim, edge_dim=2, num_rumor_classes=num_cls),
            },
            DataLoader(k_train, batch_size=args.batch_size, shuffle=True),
            DataLoader(k_val, batch_size=args.batch_size, shuffle=False),
            DataLoader(k_test, batch_size=args.batch_size, shuffle=False),
            seeds=OFFICIAL_SEEDS,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            reference_model="SBiFAGAT",
            comparator_model="BiGCN",
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

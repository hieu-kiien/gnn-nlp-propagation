"""UPFD GossipCop (BERT 768-d) generalization benchmark with LatentConflictMLP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

from src.data.pyg_dataset import UPFDAdapterTransform
from src.models.baselines import BiGCN, EdgeGATv2, FAGCNBaseline, StandardGATv2, StandardGCN
from src.models.sbi_fagat import SBiFAGAT
from src.training.reporting import (
    UPFD_METRICS,
    build_markdown_table,
    export_latex_table,
    serialize_benchmark_json,
)
from src.training.trainer import run_multi_seed_benchmark
from src.utils.seed import OFFICIAL_SEEDS


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run 5-seed UPFD post-hoc text-free structural transfer benchmark.")
    parser.add_argument("--upfd-root", type=Path, default=Path("data/raw/upfd"))
    parser.add_argument("--dataset-name", type=str, default="gossipcop", choices=["gossipcop", "politifact"])
    parser.add_argument(
        "--feature",
        type=str,
        default="bert",
        choices=["bert", "spacy", "profile", "content"],
        help="UPFD PyG node feature representation.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output-table", type=Path, default=Path("paper/tables_upfd.tex"))
    parser.add_argument("--output-json", type=Path, default=Path("paper/results_upfd.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = UPFDAdapterTransform()

    train_ds = UPFD(root=str(args.upfd_root), name=args.dataset_name, feature=args.feature, split="train", transform=transform)
    val_ds = UPFD(root=str(args.upfd_root), name=args.dataset_name, feature=args.feature, split="val", transform=transform)
    test_ds = UPFD(root=str(args.upfd_root), name=args.dataset_name, feature=args.feature, split="test", transform=transform)

    in_dim = int(train_ds[0].x.size(-1))
    h_dim = args.hidden_dim

    model_factories = {
        "StandardGCN": lambda: StandardGCN(in_dim=in_dim, hidden_dim=h_dim, num_rumor_classes=2),
        "StandardGATv2": lambda: StandardGATv2(in_dim=in_dim, hidden_dim=h_dim, num_rumor_classes=2),
        "EdgeGATv2 (Latent)": lambda: EdgeGATv2(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=2,
            num_rumor_classes=2,
            use_latent_edge_mlp=True,
        ),
        "FAGCN": lambda: FAGCNBaseline(in_dim=in_dim, hidden_dim=h_dim, num_rumor_classes=2),
        "BiGCN": lambda: BiGCN(in_dim=in_dim, hidden_dim=h_dim, num_rumor_classes=2),
        "SBiFAGAT (Latent)": lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=h_dim,
            edge_dim=2,
            num_rumor_classes=2,
            use_latent_edge_mlp=True,
        ),
    }

    result = run_multi_seed_benchmark(
        model_factories,
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False),
        seeds=OFFICIAL_SEEDS,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        classification_only=True,
        reference_model="SBiFAGAT (Latent)",
        comparator_model="auto",
        primary_hypothesis_keys=("accuracy", "macro_f1"),
        device=device,
    )

    print(f"\n=== UPFD ({args.dataset_name}, {args.feature}) Structural Transfer Results (5 Seeds) ===")
    print(build_markdown_table(result, UPFD_METRICS))

    tbl_label = (
        "tab:upfd_results"
        if args.dataset_name == "gossipcop"
        else f"tab:upfd_{args.dataset_name}"
    )
    export_latex_table(
        result,
        args.output_table,
        caption=(
            f"Post-Hoc Text-Free Structural Transfer on UPFD ({args.dataset_name}, {args.feature.upper()} {in_dim}-d) "
            "Across 5 Random Seeds. Without raw reply text, SBi-FAGAT and EdgeGATv2 use a shared learned latent edge "
            "projection $\\mathbf{e}_{\\text{latent}} = "
            "\\mathrm{MLP}([\\mathbf{x}_u \\parallel \\mathbf{x}_v \\parallel \\mathbf{x}_u - \\mathbf{x}_v])$."
        ),
        label=tbl_label,
        metrics=UPFD_METRICS,
    )
    serialize_benchmark_json(
        result,
        args.output_json,
        extra={"dataset": f"UPFD-{args.dataset_name}", "feature": args.feature, "in_dim": in_dim},
    )


if __name__ == "__main__":
    main()

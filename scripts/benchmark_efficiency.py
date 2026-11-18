"""Profile parameter count, latency, and peak memory across all 8 SBi-FAGAT & baseline models.

NOTE: Uses controlled synthetic `CascadeData` tensor batches purely to benchmark
computational complexity (#Params, ms/cascade, peak RAM/VRAM) under identical input shapes.
NEVER use synthetic outputs from this profiler in paper accuracy/error tables.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from src.data.pyg_dataset import CascadeData, TOTAL_NODE_FEATURE_DIM
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
from src.utils.seed import seed_everything


def build_profiling_batch(
    *,
    num_cascades: int = 32,
    k_horizon: int = 20,
    in_dim: int = TOTAL_NODE_FEATURE_DIM,
    edge_dim: int = 2,
    num_classes: int = 4,
    seed: int = 42,
) -> Batch:
    """Create a deterministic fixed-shape PyG Batch matching early K=20 cascades."""
    seed_everything(seed)
    graphs: list[CascadeData] = []
    num_nodes = k_horizon + 1
    for idx in range(num_cascades):
        x = torch.randn((num_nodes, in_dim), dtype=torch.float32)
        src = [i // 3 for i in range(1, num_nodes)]
        dst = list(range(1, num_nodes))
        edge_index_td = torch.tensor([src, dst], dtype=torch.long)
        edge_index_bu = torch.tensor([dst, src], dtype=torch.long)
        s_uv = torch.linspace(-0.9, 0.9, steps=len(dst))
        h_uv = torch.linspace(0.1, 0.85, steps=len(dst))
        if edge_dim == 4:
            r_uv = torch.linspace(0.05, 0.75, steps=len(dst))
            c_uv = torch.linspace(0.85, 0.15, steps=len(dst))
            edge_attr = torch.stack([s_uv, h_uv, r_uv, c_uv], dim=-1)
        else:
            edge_attr = torch.stack([s_uv, h_uv], dim=-1)
        graphs.append(
            CascadeData(
                x=x,
                edge_index=edge_index_td,
                edge_index_td=edge_index_td,
                edge_index_bu=edge_index_bu,
                edge_attr=edge_attr,
                edge_stance=s_uv.unsqueeze(-1),
                cascade_stats=torch.rand((1, 8), dtype=torch.float32),
                root_index=torch.tensor([0], dtype=torch.long),
                y_size=torch.tensor([3.2], dtype=torch.float32),
                y_polar=torch.tensor([0.55], dtype=torch.float32),
                y_bimodality=torch.tensor([0.62], dtype=torch.float32),
                y_rumor=torch.tensor([idx % num_classes], dtype=torch.long),
            )
        )
    return Batch.from_data_list(graphs)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def profile_single_model(
    name: str,
    factory: Callable[[], nn.Module],
    batch: Batch,
    *,
    device: torch.device,
    warmup: int = 3,
    repeats: int = 10,
) -> dict[str, Any]:
    model = factory().to(device)
    batch_dev = batch.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    num_cascades = int(batch.num_graphs)

    # Warmup forward + backward
    for _ in range(max(1, warmup)):
        model.train()
        optimizer.zero_grad()
        out = model(batch_dev)
        loss = (
            out["y_size_pred"].mean()
            + out["y_polar_pred"].mean()
            + out["rumor_logits"].pow(2).mean()
        )
        loss.backward()
        optimizer.step()

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Forward inference latency
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    else:
        tracemalloc.start()

    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max(1, repeats)):
            _ = model(batch_dev)
    _sync()
    t1 = time.perf_counter()
    infer_ms_per_cascade = ((t1 - t0) * 1000.0) / float(max(1, repeats) * max(1, num_cascades))

    # Training step latency
    model.train()
    _sync()
    t2 = time.perf_counter()
    for _ in range(max(1, repeats)):
        optimizer.zero_grad()
        out = model(batch_dev)
        loss = (
            out["y_size_pred"].mean()
            + out["y_polar_pred"].mean()
            + out["rumor_logits"].pow(2).mean()
        )
        loss.backward()
        optimizer.step()
    _sync()
    t3 = time.perf_counter()
    train_ms_per_batch = ((t3 - t2) * 1000.0) / float(max(1, repeats))

    if device.type == "cuda":
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        mem_label = "Peak VRAM (MB)"
    else:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem_mb = float(peak_bytes) / (1024.0 * 1024.0)
        mem_label = "Peak Alloc RAM (MB)"

    params = count_trainable_params(model)
    del model, optimizer, batch_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": name,
        "params": params,
        "infer_ms_per_cascade": infer_ms_per_cascade,
        "train_ms_per_batch": train_ms_per_batch,
        "peak_mem_mb": peak_mem_mb,
        "mem_label": mem_label,
    }


def run_efficiency_benchmark(
    *,
    num_cascades: int = 32,
    k_horizon: int = 20,
    hidden_dim: int = 128,
    edge_dim: int = 2,
    use_cascade_stats: bool = False,
    predict_uncertainty: bool = False,
    num_classes: int = 4,
    warmup: int = 3,
    repeats: int = 10,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = TOTAL_NODE_FEATURE_DIM
    batch = build_profiling_batch(
        num_cascades=num_cascades,
        k_horizon=k_horizon,
        in_dim=in_dim,
        edge_dim=edge_dim,
        num_classes=num_classes,
    )
    factories: dict[str, Callable[[], nn.Module]] = {
        "TextMLP": lambda: TextMLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_rumor_classes=num_classes,
            use_cascade_stats=use_cascade_stats,
        ),
        "TemporalStanceMLP": lambda: TemporalStanceMLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_classes,
            use_cascade_stats=use_cascade_stats,
        ),
        "StandardGCN": lambda: StandardGCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_rumor_classes=num_classes,
            use_cascade_stats=use_cascade_stats,
        ),
        "StandardGATv2": lambda: StandardGATv2(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_rumor_classes=num_classes,
            use_cascade_stats=use_cascade_stats,
        ),
        "EdgeGATv2": lambda: EdgeGATv2(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_classes,
            use_cascade_stats=use_cascade_stats,
        ),
        "FAGCN": lambda: FAGCNBaseline(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_rumor_classes=num_classes,
            use_cascade_stats=use_cascade_stats,
        ),
        "BiGCN": lambda: BiGCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_rumor_classes=num_classes,
            use_cascade_stats=use_cascade_stats,
        ),
        "SBiFAGAT": lambda: SBiFAGAT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_classes,
            use_cascade_stats=use_cascade_stats,
            predict_uncertainty=predict_uncertainty,
        ),
    }

    rows: list[dict[str, Any]] = []
    for name, factory in factories.items():
        rows.append(
            profile_single_model(
                name,
                factory,
                batch,
                device=target_device,
                warmup=warmup,
                repeats=repeats,
            )
        )
    return rows


def format_efficiency_markdown(rows: list[dict[str, Any]]) -> str:
    mem_col = rows[0]["mem_label"] if rows else "Peak Mem (MB)"
    lines = [
        f"| Model | #Params | Infer (ms/cascade) | Train Step (ms/batch) | {mem_col} |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['params']:,} | {r['infer_ms_per_cascade']:.4f} | "
            f"{r['train_ms_per_batch']:.2f} | {r['peak_mem_mb']:.2f} |"
        )
    return "\n".join(lines)


def format_efficiency_latex(
    rows: list[dict[str, Any]],
    *,
    caption: str = "Computational efficiency benchmark (parameters, per-cascade latency, and peak memory).",
    label: str = "tab:efficiency",
) -> str:
    """Format computational efficiency rows into an IEEEtran `booktabs` LaTeX table."""
    mem_col = rows[0]["mem_label"] if rows else "Peak Mem (MB)"
    body_lines: list[str] = []
    for r in rows:
        escaped_model = str(r["model"]).replace("_", "\\_")
        body_lines.append(
            f"{escaped_model} & {int(r['params']):,} & {float(r['infer_ms_per_cascade']):.4f} & "
            f"{float(r['train_ms_per_batch']):.2f} & {float(r['peak_mem_mb']):.2f} \\\\"
        )
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            f"Model & \\#Params & Infer (ms/cascade)$\\downarrow$ & Train Step (ms/batch)$\\downarrow$ & {mem_col}$\\downarrow$ \\\\",
            "\\midrule",
            *body_lines,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark parameter count, inference/train latency, and memory across all 8 models."
    )
    parser.add_argument("--num-cascades", type=int, default=32)
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--edge-dim", type=int, choices=[2, 4], default=2)
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--use-cascade-stats", action="store_true")
    parser.add_argument("--predict-uncertainty", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-table", type=Path, default=None)
    args = parser.parse_args()

    use_stats = bool(args.use_cascade_stats or args.v2)
    use_unc = bool(args.predict_uncertainty or args.v2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[EFFICIENCY & COMPLEXITY PROFILING ONLY - SYNTHETIC SHAPES, NOT REAL ACCURACY DATA]")
    print(f"Device: {device} | Cascades/batch={args.num_cascades} | K={args.k_horizon} | hidden_dim={args.hidden_dim}\n")

    rows = run_efficiency_benchmark(
        num_cascades=args.num_cascades,
        k_horizon=args.k_horizon,
        hidden_dim=args.hidden_dim,
        edge_dim=args.edge_dim,
        use_cascade_stats=use_stats,
        predict_uncertainty=use_unc,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    print(format_efficiency_markdown(rows))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nSaved efficiency profile JSON to: {args.output_json}")

    if args.output_table is not None:
        args.output_table.parent.mkdir(parents=True, exist_ok=True)
        args.output_table.write_text(format_efficiency_latex(rows), encoding="utf-8")
        print(f"Saved efficiency LaTeX table to: {args.output_table}")


if __name__ == "__main__":
    main()

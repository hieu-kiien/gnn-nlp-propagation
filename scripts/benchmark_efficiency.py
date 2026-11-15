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
        edge_attr = torch.stack([s_uv, h_uv], dim=-1)
        graphs.append(
            CascadeData(
                x=x,
                edge_index=edge_index_td,
                edge_index_td=edge_index_td,
                edge_index_bu=edge_index_bu,
                edge_attr=edge_attr,
                edge_stance=s_uv.unsqueeze(-1),
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
        num_classes=num_classes,
    )
    factories: dict[str, Callable[[], nn.Module]] = {
        "TextMLP": lambda: TextMLP(in_dim=in_dim, hidden_dim=hidden_dim, num_rumor_classes=num_classes),
        "TemporalStanceMLP": lambda: TemporalStanceMLP(
            in_dim=in_dim, hidden_dim=hidden_dim, edge_dim=2, num_rumor_classes=num_classes
        ),
        "StandardGCN": lambda: StandardGCN(in_dim=in_dim, hidden_dim=hidden_dim, num_rumor_classes=num_classes),
        "StandardGATv2": lambda: StandardGATv2(
            in_dim=in_dim, hidden_dim=hidden_dim, num_rumor_classes=num_classes
        ),
        "EdgeGATv2": lambda: EdgeGATv2(
            in_dim=in_dim, hidden_dim=hidden_dim, edge_dim=2, num_rumor_classes=num_classes
        ),
        "FAGCN": lambda: FAGCNBaseline(in_dim=in_dim, hidden_dim=hidden_dim, num_rumor_classes=num_classes),
        "BiGCN": lambda: BiGCN(in_dim=in_dim, hidden_dim=hidden_dim, num_rumor_classes=num_classes),
        "SBiFAGAT": lambda: SBiFAGAT(
            in_dim=in_dim, hidden_dim=hidden_dim, edge_dim=2, num_rumor_classes=num_classes
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark parameter count, inference/train latency, and memory across all 8 models."
    )
    parser.add_argument("--num-cascades", type=int, default=32)
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[EFFICIENCY & COMPLEXITY PROFILING ONLY - SYNTHETIC SHAPES, NOT REAL ACCURACY DATA]")
    print(f"Device: {device} | Cascades/batch={args.num_cascades} | K={args.k_horizon} | hidden_dim={args.hidden_dim}\n")

    rows = run_efficiency_benchmark(
        num_cascades=args.num_cascades,
        k_horizon=args.k_horizon,
        hidden_dim=args.hidden_dim,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    print(format_efficiency_markdown(rows))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nSaved efficiency profile JSON to: {args.output_json}")


if __name__ == "__main__":
    main()

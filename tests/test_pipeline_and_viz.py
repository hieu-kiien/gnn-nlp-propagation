"""Tests for EarlyStopping, 5-seed runner, t-test reporting, LaTeX export, and vector figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.pyg_dataset import CascadeData
from src.models.baselines import BiGCN, StandardGCN
from src.models.sbi_fagat import SBiFAGAT
from src.training.reporting import (
    MAIN_METRICS,
    UPFD_METRICS,
    build_markdown_table,
    export_latex_table,
    serialize_benchmark_json,
)
from src.training.trainer import EarlyStopping, compute_paired_ttest, run_multi_seed_benchmark
from src.utils.seed import OFFICIAL_SEEDS, seed_everything
from src.utils.visualize import (
    plot_cascade_heatmap,
    plot_early_k_curve,
    plot_umap_polarization,
)


def _make_mini_loaders(in_dim: int = 32) -> tuple[DataLoader, DataLoader, DataLoader, list[CascadeData]]:
    seed_everything(77)
    graphs: list[CascadeData] = []
    for i in range(12):
        n_nodes = 8
        src = [0, 0, 1, 1, 2, 3, 4]
        dst = [1, 2, 3, 4, 5, 6, 7]
        e_td = torch.tensor([src, dst], dtype=torch.long)
        e_bu = e_td[[1, 0], :]
        stances = torch.tensor([[-0.8], [0.7], [-0.9], [0.6], [-0.5], [0.8], [-0.7]], dtype=torch.float32)
        host = (stances < 0).float() * 0.85
        e_attr = torch.cat([stances, host], dim=-1)
        graphs.append(
            CascadeData(
                x=torch.randn(n_nodes, in_dim),
                edge_index=e_td,
                edge_index_td=e_td,
                edge_index_bu=e_bu,
                edge_attr=e_attr,
                edge_stance=stances,
                root_index=torch.tensor([0], dtype=torch.long),
                y_size=torch.tensor([1.5 + 0.1 * (i % 4)], dtype=torch.float32),
                y_polar=torch.tensor([0.2 + 0.15 * (i % 4)], dtype=torch.float32),
                y_rumor=torch.tensor([i % 4], dtype=torch.long),
                cascade_id=f"viz_casc_{i}",
                topic_id=f"viz_topic_{i // 2}",
                node_ids=[f"n_{k}" for k in range(n_nodes)],
                node_texts=[f"Sample comment text number {k}" for k in range(n_nodes)],
            )
        )
    return (
        DataLoader(graphs[:6], batch_size=3, shuffle=True),
        DataLoader(graphs[6:9], batch_size=3, shuffle=False),
        DataLoader(graphs[9:], batch_size=3, shuffle=False),
        graphs,
    )


def test_early_stopping_patience_10_and_weight_restore() -> None:
    model = torch.nn.Linear(4, 1)
    torch.nn.init.constant_(model.weight, 2.0)
    stopper = EarlyStopping(patience=10, mode="min")

    assert stopper.step(0.50, model) is False
    torch.nn.init.constant_(model.weight, 99.0)

    for step_idx in range(9):
        assert stopper.step(0.60 + step_idx * 0.01, model) is False
    # 10th non-improving step triggers stop
    assert stopper.step(0.75, model) is True

    stopper.restore_best(model)
    assert torch.allclose(model.weight, torch.full_like(model.weight, 2.0))


def test_five_seed_benchmark_ttest_and_latex_export(tmp_path: Path) -> None:
    train_loader, val_loader, test_loader, _ = _make_mini_loaders(in_dim=32)

    factories = {
        "StandardGCN": lambda: StandardGCN(in_dim=32, hidden_dim=32, dropout=0.0),
        "BiGCN": lambda: BiGCN(in_dim=32, hidden_dim=32, dropout=0.0),
        "SBiFAGAT": lambda: SBiFAGAT(in_dim=32, hidden_dim=32, edge_dim=2, dropout=0.0),
    }

    res = run_multi_seed_benchmark(
        factories,
        train_loader,
        val_loader,
        test_loader,
        seeds=OFFICIAL_SEEDS,
        epochs=2,
        patience=10,
        reference_model="SBiFAGAT",
        comparator_model="BiGCN",
        device="cpu",
    )

    assert res["seeds"] == [42, 100, 2024, 777, 999]
    assert len(res["raw_results"]["SBiFAGAT"]) == 5
    assert "msle" in res["p_values"]
    assert 0.0 <= res["p_values"]["msle"] <= 1.0

    # Paired t-test helper edge checks
    p_sig = compute_paired_ttest([0.10, 0.11, 0.09, 0.10, 0.12], [0.40, 0.42, 0.39, 0.41, 0.43])
    assert p_sig < 0.001
    assert compute_paired_ttest([0.5, 0.5], [0.5, 0.5]) == 1.0

    md = build_markdown_table(res, MAIN_METRICS)
    assert "SBiFAGAT" in md and "BiGCN" in md

    tex_path = export_latex_table(
        res,
        tmp_path / "tables_main.tex",
        caption="Test Main Table",
        label="tab:test_main",
        metrics=MAIN_METRICS,
    )
    assert tex_path.exists()
    tex_content = tex_path.read_text(encoding="utf-8")
    assert "\\toprule" in tex_content and "\\bottomrule" in tex_content
    assert "SBiFAGAT" in tex_content

    upfd_tex = export_latex_table(
        res,
        tmp_path / "tables_upfd.tex",
        caption="Test UPFD Table",
        label="tab:test_upfd",
        metrics=UPFD_METRICS,
    )
    assert upfd_tex.exists()

    json_path = serialize_benchmark_json(res, tmp_path / "results.json")
    assert json_path.exists() and json_path.stat().st_size > 0


def test_vector_figure_generation_pdf_svg_and_top5_text(tmp_path: Path) -> None:
    fig_dir = tmp_path / "figures"
    rng = np.random.default_rng(42)

    # Fig 1: UMAP polarization
    gcn_emb = rng.normal(0.0, 1.0, size=(36, 32)).astype(np.float32)
    sbi_emb = np.vstack(
        [
            rng.normal(-2.5, 0.3, size=(18, 32)),
            rng.normal(2.5, 0.3, size=(18, 32)),
        ]
    ).astype(np.float32)
    stances = np.concatenate([np.full(18, -0.85), np.full(18, 0.85)]).astype(np.float32)

    f1_pdf, f1_svg = plot_umap_polarization(gcn_emb, sbi_emb, stances, output_dir=fig_dir)
    assert f1_pdf.exists() and f1_pdf.stat().st_size > 0
    assert f1_svg.exists() and f1_svg.stat().st_size > 0

    # Fig 2: Early K curve
    k_results = {
        10: {
            "StandardGCN": {"msle": (0.42, 0.04), "rmse": (0.28, 0.03)},
            "BiGCN": {"msle": (0.35, 0.03), "rmse": (0.24, 0.02)},
            "SBiFAGAT": {"msle": (0.25, 0.02), "rmse": (0.17, 0.02)},
        },
        20: {
            "StandardGCN": {"msle": (0.38, 0.03), "rmse": (0.26, 0.03)},
            "BiGCN": {"msle": (0.30, 0.03), "rmse": (0.21, 0.02)},
            "SBiFAGAT": {"msle": (0.20, 0.02), "rmse": (0.14, 0.01)},
        },
        30: {
            "StandardGCN": {"msle": (0.35, 0.03), "rmse": (0.24, 0.02)},
            "BiGCN": {"msle": (0.27, 0.02), "rmse": (0.19, 0.02)},
            "SBiFAGAT": {"msle": (0.18, 0.02), "rmse": (0.12, 0.01)},
        },
        50: {
            "StandardGCN": {"msle": (0.33, 0.02), "rmse": (0.22, 0.02)},
            "BiGCN": {"msle": (0.25, 0.02), "rmse": (0.18, 0.02)},
            "SBiFAGAT": {"msle": (0.16, 0.01), "rmse": (0.11, 0.01)},
        },
    }
    f2_pdf, f2_svg = plot_early_k_curve(k_results, output_dir=fig_dir)
    assert f2_pdf.exists() and f2_pdf.stat().st_size > 0
    assert f2_svg.exists() and f2_svg.stat().st_size > 0

    # Fig 3: Cascade heatmap + Top-5 trigger comments txt
    _, _, _, graphs = _make_mini_loaders(in_dim=32)
    sample = graphs[0]
    model = SBiFAGAT(in_dim=32, hidden_dim=32, edge_dim=2, dropout=0.0)
    gates = model.extract_edge_gates(sample)
    f3_pdf, f3_svg, f3_txt = plot_cascade_heatmap(
        sample,
        gates["edge_importance"],
        gates["gamma_td"],
        output_dir=fig_dir,
    )
    assert f3_pdf.exists() and f3_pdf.stat().st_size > 0
    assert f3_svg.exists() and f3_svg.stat().st_size > 0
    assert f3_txt.exists()
    txt = f3_txt.read_text(encoding="utf-8")
    assert "#1 |" in txt and "#5 |" in txt

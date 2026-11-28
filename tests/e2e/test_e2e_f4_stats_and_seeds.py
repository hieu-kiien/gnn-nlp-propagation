"""Opaque-box E2E tests for Feature Area F4: Statistical Testing, Multi-Task Loss & 5-Seed Contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from scripts.assemble_paper_artifacts import find_unescaped_underscores_in_latex
from src.training.metrics import compute_multitask_loss, evaluate_predictions
from src.training.reporting import (
    MAIN_METRICS,
    build_markdown_table,
    export_latex_table,
    serialize_benchmark_json,
)
from src.training.trainer import (
    _compute_sample_level_bootstrap_pvalues,
    compute_paired_bootstrap_pvalue,
    compute_paired_ttest,
    compute_paired_wilcoxon,
    holm_bonferroni_adjust,
    run_multi_seed_benchmark,
)
from src.utils.seed import OFFICIAL_SEEDS


def _make_synthetic_seed_predictions(
    *,
    num_samples: int = 40,
    num_events: int = 2,
    num_seeds: int = 5,
    ref_better: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(42)
    y_size = rng.normal(3.5, 0.6, size=num_samples)
    y_polar = np.clip(rng.normal(0.25, 0.08, size=num_samples), 0.02, 0.95)
    y_rumor = rng.integers(0, 4, size=num_samples)
    topic_ids = [f"event_{i % num_events}" for i in range(num_samples)]

    preds_ref, preds_cmp = [], []
    for s in range(num_seeds):
        noise_ref = rng.normal(0.0, 0.04 if ref_better else 0.25, size=num_samples)
        noise_cmp = rng.normal(0.0, 0.25 if ref_better else 0.04, size=num_samples)
        logits_ref = np.zeros((num_samples, 4), dtype=np.float64)
        logits_cmp = np.zeros((num_samples, 4), dtype=np.float64)
        for i in range(num_samples):
            logits_ref[i, y_rumor[i]] = 3.0 if ref_better else 0.2
            logits_cmp[i, y_rumor[i]] = 0.2 if ref_better else 3.0
        preds_ref.append(
            {
                "y_size_true": y_size,
                "y_size_pred": y_size + noise_ref,
                "y_polar_true": y_polar,
                "y_polar_pred": np.clip(y_polar + noise_ref * 0.2, 0.01, 0.99),
                "y_rumor_true": y_rumor,
                "rumor_logits": logits_ref,
                "topic_ids": topic_ids,
            }
        )
        preds_cmp.append(
            {
                "y_size_true": y_size,
                "y_size_pred": y_size + noise_cmp,
                "y_polar_true": y_polar,
                "y_polar_pred": np.clip(y_polar + noise_cmp * 0.2, 0.01, 0.99),
                "y_rumor_true": y_rumor,
                "rumor_logits": logits_cmp,
                "topic_ids": topic_ids,
            }
        )
    return preds_ref, preds_cmp


# ---------------------------------------------------------------------------
# Tier 1: Isolated Feature Coverage
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_f4_t1_official_seeds_exact_tuple() -> None:
    """Iron Law 4 contract: OFFICIAL_SEEDS must equal (42, 100, 2024, 777, 999)."""
    assert OFFICIAL_SEEDS == (42, 100, 2024, 777, 999)
    assert len(OFFICIAL_SEEDS) == 5


@pytest.mark.tier1
def test_f4_t1_no_iid_bootstrap_when_fewer_than_6_events() -> None:
    """B7-stats contract: <6 events must NEVER fall back to IID sample-level bootstrap."""
    for n_events in (2, 5):
        preds_ref, preds_cmp = _make_synthetic_seed_predictions(num_samples=48, num_events=n_events)
        pvals = _compute_sample_level_bootstrap_pvalues(
            preds_ref,
            preds_cmp,
            metric_keys=("msle", "r2_polar", "accuracy", "macro_f1"),
            seed=42,
            n_resamples=200,
            n_stat_resamples=200,
        )
        # When <6 events exist, cluster bootstrap is invalid and IID fallback is forbidden,
        # so sample-level bootstrap p-values must be NaN (not a fake IID p < 0.05).
        for k, pv in pvals.items():
            assert math.isnan(float(pv)), (
                f"Expected NaN (no IID fallback) for {k!r} with {n_events} events, got {pv}"
            )


@pytest.mark.tier1
def test_f4_t1_cluster_bootstrap_activates_at_6_plus_events() -> None:
    """With >= 6 distinct events, cluster bootstrap produces valid p-values and Holm-Bonferroni adjusts them."""
    preds_ref, preds_cmp = _make_synthetic_seed_predictions(num_samples=60, num_events=8)
    pvals = _compute_sample_level_bootstrap_pvalues(
        preds_ref,
        preds_cmp,
        metric_keys=("msle", "r2_polar", "accuracy", "macro_f1"),
        seed=42,
        n_resamples=300,
        n_stat_resamples=300,
    )
    for k, pv in pvals.items():
        assert math.isfinite(float(pv)), f"Expected finite cluster bootstrap p-value for {k!r}, got {pv}"
        assert 0.0 < float(pv) <= 1.0

    adj = holm_bonferroni_adjust(pvals)
    assert set(adj.keys()) == set(pvals.keys())
    for k in pvals:
        assert adj[k] >= pvals[k] - 1e-12
        assert 0.0 <= adj[k] <= 1.0


@pytest.mark.tier1
def test_f4_t1_reporting_no_one_sided_comparator_star_masking() -> None:
    """B7-stats contract: reporting must NOT hide statistically significant comparator wins via one-sided masking."""
    benchmark_result = {
        "reference_model": "SBiFAGAT",
        "comparator_model": "BiEdgeGATv2",
        "summary_stats": {
            "BiEdgeGATv2": {
                "msle": (0.1200, 0.0100),
                "mae": (0.2500, 0.0120),
                "r2_size": (0.6500, 0.0200),
                "rmse": (0.0800, 0.0050),
                "r2_polar": (0.4200, 0.0150),  # Comparator BEATS SBiFAGAT!
                "pearson_r": (0.6500, 0.0120),
                "spearman_polar": (0.6200, 0.0140),
                "macro_f1": (0.7100, 0.0100),
                "accuracy": (0.7400, 0.0110),
            },
            "SBiFAGAT": {
                "msle": (0.1100, 0.0100),
                "mae": (0.2400, 0.0110),
                "r2_size": (0.6700, 0.0180),
                "rmse": (0.0950, 0.0060),
                "r2_polar": (0.2100, 0.0200),  # Reference LOSES here
                "pearson_r": (0.5000, 0.0180),
                "spearman_polar": (0.4800, 0.0190),
                "macro_f1": (0.7200, 0.0110),
                "accuracy": (0.7500, 0.0100),
            },
        },
        "p_values": {"r2_polar": 0.002, "msle": 0.250},
        "adjusted_p_values": {"r2_polar": 0.004, "msle": 0.500},
        "adjusted_bootstrap_p_values": {"r2_polar": 0.004, "msle": 0.500},
    }
    md = build_markdown_table(benchmark_result, metrics=MAIN_METRICS, decimals=4)
    # Extract the SBiFAGAT row and BiEdgeGATv2 row
    rows = {line.split("|")[1].strip(): line for line in md.splitlines() if line.startswith("| ")}
    sbif_row = rows["SBiFAGAT"]
    cmp_row = rows["BiEdgeGATv2"]
    r2_idx = 1 + 1 + [m[0] for m in MAIN_METRICS].index("r2_polar")
    sbif_r2_cell = sbif_row.split("|")[r2_idx].strip()
    cmp_r2_cell = cmp_row.split("|")[r2_idx].strip()
    assert "0.2100" in sbif_r2_cell and "0.4200" in cmp_r2_cell
    assert "(-)**" in sbif_r2_cell, (
        f"Expected directional loss marker '(-)**' on SBiFAGAT r2_polar cell, got {sbif_r2_cell!r}"
    )
    assert "(+)" not in sbif_r2_cell
    # Forbid unadorned significance '*' without '-'
    marker_tail = sbif_r2_cell.split("0.0200", 1)[-1]
    assert "(-)" in marker_tail and not ("*" in marker_tail and "-" not in marker_tail)


@pytest.mark.tier1
def test_f4_t1_multitask_loss_rumor_scale_normalized() -> None:
    """M2 contract: compute_multitask_loss normalizes loss_rumor scale so 4-class CE does not dominate regression."""
    B = 4
    y_size = torch.tensor([2.0, 2.5, 3.0, 3.5], dtype=torch.float32)
    y_polar = torch.tensor([0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    outputs = {
        "y_size_pred": y_size + 0.5,
        "y_polar_logit": torch.logit(y_polar),
        "y_polar_pred": y_polar,
        "rumor_logits": torch.zeros((B, 4), dtype=torch.float32),  # raw CE == ln(4) = 1.3863
    }
    batch_reg = type("B", (), {"y_size": y_size, "y_polar": y_polar, "y_rumor": None})()
    batch_cls = type(
        "B", (), {"y_size": y_size, "y_polar": y_polar, "y_rumor": torch.tensor([0, 1, 2, 3])}
    )()

    loss_reg = float(compute_multitask_loss(outputs, batch_reg).item())
    loss_full = float(compute_multitask_loss(outputs, batch_cls).item())
    rumor_contribution = loss_full - loss_reg
    # Raw unnormalized 4-class uniform CE is ln(4) = 1.3863; normalized contribution must be strictly < 1.05
    assert rumor_contribution > 0.0
    assert rumor_contribution < 1.05, (
        f"Rumor loss contribution ({rumor_contribution:.4f}) appears unnormalized against regression ({loss_reg:.4f})"
    )


# ---------------------------------------------------------------------------
# Tier 2: Boundary, Corner & Negative Rejection Cases
# ---------------------------------------------------------------------------


@pytest.mark.tier2
def test_f4_t2_constant_predictions_paired_ttest_and_metrics_no_crash() -> None:
    """Zero-variance predictions and identical seed scores yield finite metrics and safe p-values."""
    p_ident = compute_paired_ttest([0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5])
    assert p_ident == pytest.approx(1.0, abs=1e-6) or math.isnan(p_ident)

    m = evaluate_predictions(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([2.5, 2.5, 2.5, 2.5]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0.25, 0.25, 0.25, 0.25]),
        y_rumor_true=np.array([0, 1, 0, 1]),
        rumor_logits=np.zeros((4, 2)),
    )
    for k in ("msle", "mae", "rmse_size", "rmse", "mae_polar", "pearson_r", "spearman_polar", "accuracy", "macro_f1"):
        assert math.isfinite(m[k]), f"Non-finite metric {k}: {m[k]}"


@pytest.mark.tier2
def test_f4_t2_holm_bonferroni_edge_cases() -> None:
    """holm_bonferroni_adjust handles empty, NaN, zero, and 1.0 p-values while clamping to [0, 1]."""
    assert holm_bonferroni_adjust({}) == {}

    res = holm_bonferroni_adjust({"a": 0.0, "b": 0.01, "c": 0.8, "d": float("nan"), "e": 1.0})
    assert res["a"] == 0.0
    assert 0.0 <= res["b"] <= 1.0
    assert 0.0 <= res["c"] <= 1.0
    assert math.isnan(res["d"])
    assert res["e"] == 1.0


@pytest.mark.tier2
def test_f4_t2_serialize_benchmark_json_strict_rfc8259(tmp_path: Path) -> None:
    """serialize_benchmark_json sanitizes NaN/Inf and numpy scalars into strict RFC 8259 JSON."""
    payload = {
        "seeds": [42, 100, 2024, 777, 999],
        "summary_stats": {"SBiFAGAT": {"msle": (float(np.float32(0.12)), float("nan"))}},
        "p_values": {"msle": float("inf"), "r2_polar": 0.01},
        "raw_results": [],
    }
    out_path = tmp_path / "bench.json"
    serialize_benchmark_json(payload, out_path)
    raw = out_path.read_text(encoding="utf-8")

    def _reject_nonfinite(tok: str) -> Any:
        raise ValueError(f"Non-RFC8259 token {tok!r}")

    parsed = json.loads(raw, parse_constant=_reject_nonfinite)
    assert parsed["summary_stats"]["SBiFAGAT"]["msle"][1] is None
    assert parsed["p_values"]["msle"] is None


@pytest.mark.tier2
def test_f4_t2_latex_export_no_unescaped_underscores(tmp_path: Path) -> None:
    """export_latex_table escapes underscores in model names so IEEEtran LaTeX compiles cleanly."""
    benchmark_result = {
        "reference_model": "SBiFAGAT_v2",
        "comparator_model": "BiEdgeGATv2_Matched",
        "summary_stats": {
            "BiEdgeGATv2_Matched": {"msle": (0.15, 0.02), "r2_polar": (0.20, 0.03)},
            "SBiFAGAT_v2": {"msle": (0.11, 0.01), "r2_polar": (0.35, 0.02)},
        },
        "p_values": {"msle": 0.02, "r2_polar": 0.004},
        "adjusted_p_values": {"msle": 0.02, "r2_polar": 0.008},
    }
    tex_path = export_latex_table(
        benchmark_result,
        tmp_path / "tab.tex",
        caption="Test Table",
        label="tab:e2e_test",
        metrics=MAIN_METRICS[:2],
    )
    tex = tex_path.read_text(encoding="utf-8")
    assert "\\begin{table" in tex
    assert find_unescaped_underscores_in_latex(tex) == []


@pytest.mark.tier2
def test_f4_t2_multitask_loss_handles_all_negative_rumor_labels() -> None:
    """VN-style batches with y_rumor == -1 compute pure regression loss without CrossEntropy error."""
    outputs = {
        "y_size_pred": torch.tensor([2.2, 3.1], dtype=torch.float32),
        "y_polar_logit": torch.tensor([-1.0, 0.5], dtype=torch.float32),
        "y_polar_pred": torch.sigmoid(torch.tensor([-1.0, 0.5], dtype=torch.float32)),
        "rumor_logits": torch.randn((2, 4), dtype=torch.float32),
    }
    batch = type(
        "B",
        (),
        {
            "y_size": torch.tensor([2.0, 3.0]),
            "y_polar": torch.tensor([0.2, 0.6]),
            "y_rumor": torch.tensor([-1, -1], dtype=torch.long),
        },
    )()
    loss = compute_multitask_loss(outputs, batch)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


@pytest.mark.tier1
def test_f4_t1_wilcoxon_and_n5_exact_sign_flip_permutation() -> None:
    """PROJECT #9 contract: Wilcoxon signed-rank & exact 2^n sign-flip permutation work cleanly at n=5."""
    from torch_geometric.loader import DataLoader
    from src.data.pyg_dataset import TOTAL_NODE_FEATURE_DIM
    from src.models.baselines import StandardGCN, TextMLP
    from tests.e2e.conftest import make_synthetic_cascade_data_list

    a = [0.90, 0.91, 0.92, 0.93, 0.94]
    b = [0.50, 0.51, 0.52, 0.53, 0.54]
    w_p = compute_paired_wilcoxon(a, b)
    perm_p = compute_paired_bootstrap_pvalue(a, b)
    # At n=5 with zero ties and uniform sign, two-sided exact Wilcoxon & 2^5 sign-flip both equal 2/32 = 0.0625
    assert w_p == pytest.approx(0.0625, abs=1e-6)
    assert perm_p == pytest.approx(0.0625, abs=1e-6)
    assert compute_paired_wilcoxon(a, a) == pytest.approx(1.0, abs=1e-6)
    assert compute_paired_bootstrap_pvalue(a, a) == pytest.approx(1.0, abs=1e-6)
    assert math.isnan(compute_paired_wilcoxon([0.5], [0.4]))
    assert math.isnan(compute_paired_bootstrap_pvalue([0.5], [0.4]))

    graphs = make_synthetic_cascade_data_list(num_graphs=12, in_dim=TOTAL_NODE_FEATURE_DIM, num_topics=12)
    res = run_multi_seed_benchmark(
        {
            "TextMLP": lambda: TextMLP(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=16, num_rumor_classes=4),
            "StandardGCN": lambda: StandardGCN(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=16, num_rumor_classes=4),
        },
        DataLoader(graphs[:6], batch_size=3, shuffle=False),
        DataLoader(graphs[6:9], batch_size=3, shuffle=False),
        DataLoader(graphs[9:], batch_size=3, shuffle=False),
        seeds=OFFICIAL_SEEDS,
        epochs=1,
        patience=1,
        reference_model="StandardGCN",
        comparator_model="TextMLP",
        compute_bootstrap=False,
        device=torch.device("cpu"),
    )
    assert "wilcoxon_p_values" in res and "adjusted_wilcoxon_p_values" in res
    assert "msle" in res["wilcoxon_p_values"]
    assert 0.0 <= float(res["wilcoxon_p_values"]["msle"]) <= 1.0
    assert 0.0 <= float(res["adjusted_wilcoxon_p_values"]["msle"]) <= 1.0

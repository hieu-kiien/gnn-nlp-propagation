"""CLI and benchmark suite for Tier 2 Macro Signed User-Interaction Social Network.

Supports:
  1. Sparse CPU macro network polarization analytics (Structural Balance Theory
     + Davis Weak Balance + uniform edge-sign permutations, Signed Laplacian spectral
     bipartition, Gómez Signed Modularity Q_signed + null-graph permutations,
     Garimella Random Walk Controversy RWC, and fig5).
  2. Multi-seed Zero-Leakage Chronological Signed Link Prediction GNN benchmark
     comparing UserTextMLP, PriorPairSignPersistence, StandardGCN, StandardGATv2,
     SGCN (Derr et al., ICDM 2018), FAGCN (Bo et al., AAAI 2021), BiEdgeGATv2-Link,
     SiGAT (Huang et al., ICDM 2019), SDGNN (Huang et al., AAAI 2021), and
     SBiFAGAT-Link (bidirectional StanceGuidedFAConv).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.user_network import (
    build_temporal_signed_link_dataset,
    compute_macro_signed_network_stats,
    format_macro_network_latex,
    plot_macro_signed_network,
)
from src.models.signed_gnns import (
    BiEdgeGATv2Link,
    FAGCNLink,
    LINK_PRED_MODEL_NAMES,
    PairwiseSignedHead,
    PriorPairSignPersistence,
    SBiFAGATLink,
    SDGNN,
    SGCNLink,
    SiGAT,
    StandardGATv2Link,
    StandardGCNLink,
    UserNodeStem,
    UserTextMLP,
    build_link_model,
    prepare_mp_graph,
)
from src.training.reporting import serialize_benchmark_json
from src.training.trainer import (
    compute_paired_bootstrap_pvalue,
    compute_paired_ttest,
    holm_bonferroni_adjust,
)
from src.utils.seed import OFFICIAL_SEEDS, seed_everything

LINK_PRED_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("roc_auc", "ROC-AUC $\\uparrow$", False),
    ("pr_auc", "PR-AUC $\\uparrow$", False),
    ("macro_f1", "Macro-F1 $\\uparrow$", False),
    ("trans_roc_auc", "Trans.\\ ROC-AUC $\\uparrow$", False),
    ("trans_pr_auc", "Trans.\\ PR-AUC $\\uparrow$", False),
    ("trans_macro_f1", "Trans.\\ Macro-F1 $\\uparrow$", False),
    ("unseen_pair_trans_roc_auc", "Unseen-Pair Trans.\\ ROC-AUC $\\uparrow$", False),
    ("unseen_pair_trans_pr_auc", "Unseen-Pair Trans.\\ PR-AUC $\\uparrow$", False),
    ("unseen_pair_trans_macro_f1", "Unseen-Pair Trans.\\ Macro-F1 $\\uparrow$", False),
    ("unseen_roc_auc", "Unseen ROC-AUC $\\uparrow$", False),
    ("unseen_pr_auc", "Unseen PR-AUC $\\uparrow$", False),
    ("unseen_macro_f1", "Unseen Macro-F1 $\\uparrow$", False),
)

_NON_LEARNED_BASELINES: frozenset[str] = frozenset({"PriorPairSignPersistence"})

_MODEL_TIE_PRIORITY: dict[str, int] = {
    "BiEdgeGATv2-Link": 0,
    "SDGNN": 1,
    "SiGAT": 2,
    "SGCN": 3,
    "FAGCN": 4,
    "StandardGATv2": 5,
    "StandardGCN": 6,
    "UserTextMLP": 7,
    "PriorPairSignPersistence": 8,
}


def _safe_binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> tuple[float, float, float]:
    """Compute ROC-AUC, PR-AUC, and Macro-F1 safely for finite/non-empty splits."""
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")
    roc = float(roc_auc_score(y_true, probs))
    pr = float(average_precision_score(y_true, probs))
    preds = (probs >= float(threshold)).astype(np.int64)
    mf1 = float(f1_score(y_true.astype(np.int64), preds, average="macro", zero_division=0))
    return roc, pr, mf1


def _find_best_val_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Select binary threshold strictly on Val probabilities to maximize Macro-F1."""
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    best_th = 0.5
    best_f1 = -1.0
    for th in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        score = float(
            f1_score(y_true.astype(np.int64), (probs >= th).astype(np.int64), average="macro", zero_division=0)
        )
        if score > best_f1:
            best_f1 = score
            best_th = float(th)
    return best_th


@torch.no_grad()
def evaluate_link_split(
    model: nn.Module,
    x: Tensor,
    mp_edge_index: Tensor,
    mp_edge_attr: Tensor,
    split: Mapping[str, Tensor],
    *,
    threshold: float = 0.5,
    tune_threshold: bool = False,
) -> tuple[dict[str, float], float]:
    """Evaluate all temporal edges, both_seen, unseen_pair_transductive, and unseen subsets."""
    model.eval()
    logits = model(x, mp_edge_index, mp_edge_attr, split["pairs"])
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = split["y_oppose"].detach().cpu().numpy()
    both_seen = split["both_seen"].detach().cpu().numpy().astype(bool)
    if "unseen_pair_mask" in split:
        unseen_pair_mask = split["unseen_pair_mask"].detach().cpu().numpy().astype(bool)
    else:
        unseen_pair_mask = np.ones_like(both_seen, dtype=bool)
    unseen_pair_trans = both_seen & unseen_pair_mask
    unseen = ~both_seen

    used_th = _find_best_val_threshold(y_true, probs) if tune_threshold else float(threshold)
    roc, pr, mf1 = _safe_binary_metrics(y_true, probs, used_th)
    t_roc, t_pr, t_mf1 = _safe_binary_metrics(y_true[both_seen], probs[both_seen], used_th)
    ut_roc, ut_pr, ut_mf1 = _safe_binary_metrics(y_true[unseen_pair_trans], probs[unseen_pair_trans], used_th)
    u_roc, u_pr, u_mf1 = _safe_binary_metrics(y_true[unseen], probs[unseen], used_th)

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "macro_f1": mf1,
        "trans_roc_auc": t_roc,
        "trans_pr_auc": t_pr,
        "trans_macro_f1": t_mf1,
        "unseen_pair_trans_roc_auc": ut_roc,
        "unseen_pair_trans_pr_auc": ut_pr,
        "unseen_pair_trans_macro_f1": ut_mf1,
        "unseen_roc_auc": u_roc,
        "unseen_pr_auc": u_pr,
        "unseen_macro_f1": u_mf1,
    }, used_th


def _summarize_seed_metrics(runs: Sequence[Mapping[str, float]]) -> dict[str, tuple[float, float]]:
    """Compute (mean, ddof=1 std) across seed runs."""
    keys = [k for k, _, _ in LINK_PRED_METRICS]
    summary: dict[str, tuple[float, float]] = {}
    for k in keys:
        vals = [float(r.get(k, float("nan"))) for r in runs]
        finite = [v for v in vals if math.isfinite(v)]
        if not finite:
            summary[k] = (float("nan"), float("nan"))
        elif len(finite) == 1:
            summary[k] = (finite[0], 0.0)
        else:
            arr = np.asarray(finite, dtype=np.float64)
            summary[k] = (float(arr.mean()), float(arr.std(ddof=1)))
    return summary


def _is_finite_num(val: Any) -> bool:
    if val is None or isinstance(val, bool):
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def _select_best_val_comparator(
    val_summary_stats: Mapping[str, Mapping[str, Sequence[float | None]]],
    *,
    reference_model: str = "SBiFAGAT-Link",
) -> str:
    """Pick the strongest learned competing model strictly by average Val rank (zero Test peeking)."""
    candidates = [
        m
        for m in val_summary_stats.keys()
        if m != reference_model and m not in _NON_LEARNED_BASELINES
    ]
    if not candidates:
        candidates = [m for m in val_summary_stats.keys() if m != reference_model]
    if not candidates:
        return "BiEdgeGATv2-Link"
    ranks: dict[str, list[float]] = {c: [] for c in candidates}
    for key, _, _ in LINK_PRED_METRICS:
        scored = [
            (c, float(val_summary_stats[c][key][0]))
            for c in candidates
            if key in val_summary_stats[c]
            and val_summary_stats[c][key] is not None
            and len(val_summary_stats[c][key]) >= 1
            and _is_finite_num(val_summary_stats[c][key][0])
        ]
        if len(scored) < 2:
            continue
        scored.sort(key=lambda kv: kv[1], reverse=True)
        for rank_idx, (c, _) in enumerate(scored):
            ranks[c].append(float(rank_idx))
    if not any(ranks.values()):
        return min(candidates, key=lambda c: (_MODEL_TIE_PRIORITY.get(c, 99), c))
    return min(
        candidates,
        key=lambda c: (
            float(np.mean(ranks[c])) if ranks[c] else float("inf"),
            _MODEL_TIE_PRIORITY.get(c, 99),
            c,
        ),
    )


def compute_wilcoxon_signed_rank(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
) -> float:
    """Run two-sided paired Wilcoxon signed-rank test across matched seeds."""
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    valid = np.isfinite(arr_a) & np.isfinite(arr_b)
    arr_a, arr_b = arr_a[valid], arr_b[valid]
    n_obs = int(arr_a.size)
    if n_obs < 2:
        return float("nan")
    diffs = arr_a - arr_b
    if np.allclose(diffs, 0.0):
        return 1.0
    try:
        res = wilcoxon(
            arr_a,
            arr_b,
            zero_method="wilcox",
            alternative="two-sided",
            method="exact" if n_obs <= 25 else "auto",
        )
        p_val = float(res.pvalue)
        return 1.0 if math.isnan(p_val) else p_val
    except ValueError:
        return 1.0


def _resolve_link_marker(
    benchmark_result: Mapping[str, Any],
    key: str,
    ref_model: str,
    cmp_model: str,
    lower_is_better: bool,
    *,
    latex: bool,
) -> str:
    """Return two-sided directional significance marker WITHOUT hiding comparator wins."""
    summary_stats = benchmark_result["summary_stats"]
    ref_stat = summary_stats.get(ref_model, {}).get(key)
    cmp_stat = summary_stats.get(cmp_model, {}).get(key)
    if ref_stat is None or cmp_stat is None or len(ref_stat) < 1 or len(cmp_stat) < 1:
        return ""
    if not (_is_finite_num(ref_stat[0]) and _is_finite_num(cmp_stat[0])):
        return ""
    ref_val, cmp_val = float(ref_stat[0]), float(cmp_stat[0])
    if math.isclose(ref_val, cmp_val, abs_tol=1e-12):
        return ""

    p_val: float | None = None
    for map_key in ("adjusted_p_values", "adjusted_bootstrap_p_values", "adjusted_wilcoxon_p_values", "p_values"):
        pmap = benchmark_result.get(map_key) or {}
        if key in pmap and _is_finite_num(pmap[key]):
            p_val = float(pmap[key])
            break
    if p_val is None or p_val >= 0.05:
        return ""

    stars = "**" if p_val < 0.01 else "*"
    ref_better = (ref_val < cmp_val) if lower_is_better else (ref_val > cmp_val)
    direction = "+" if ref_better else "-"
    if latex:
        return f"$^{{{stars}({direction})}}$"
    return f"{stars}({direction})"


def _format_link_mean_std(
    mean: float | None,
    std: float | None,
    *,
    decimals: int = 4,
    latex: bool = False,
    bold: bool = False,
    marker: str = "",
) -> str:
    if not _is_finite_num(mean):
        return "--"
    mean_val = float(mean)
    std_safe = float(std) if _is_finite_num(std) else 0.0
    pm = " $\\pm$ " if latex else " ± "
    core = f"{mean_val:.{decimals}f}{pm}{std_safe:.{decimals}f}"
    if bold and latex:
        core = f"\\textbf{{{core}}}"
    elif bold:
        core = f"**{core}**"
    return f"{core}{marker}"


def _best_link_model_per_metric(
    summary_stats: Mapping[str, Mapping[str, Sequence[float | None]]],
    metrics: Sequence[tuple[str, str, bool]],
) -> dict[str, str]:
    best: dict[str, str] = {}
    for key, _, lower_is_better in metrics:
        candidates = [
            (model_name, float(stats[key][0]))
            for model_name, stats in summary_stats.items()
            if key in stats
            and stats[key] is not None
            and len(stats[key]) >= 1
            and _is_finite_num(stats[key][0])
        ]
        if not candidates:
            continue
        best[key] = (
            min(candidates, key=lambda kv: kv[1])[0]
            if lower_is_better
            else max(candidates, key=lambda kv: kv[1])[0]
        )
    return best


def build_link_markdown_table(
    benchmark_result: Mapping[str, Any],
    metrics: Sequence[tuple[str, str, bool]] = LINK_PRED_METRICS,
    *,
    decimals: int = 4,
) -> str:
    """Render Tier 2 signed link benchmark with two-sided directional significance markers."""
    summary_stats = benchmark_result["summary_stats"]
    ref_model = str(benchmark_result.get("reference_model", "SBiFAGAT-Link"))
    cmp_model = str(benchmark_result.get("comparator_model", "BiEdgeGATv2-Link"))
    best_by_metric = _best_link_model_per_metric(summary_stats, metrics)

    headers = ["Model"] + [
        label.replace("$\\downarrow$", "(↓)")
        .replace("$\\uparrow$", "(↑)")
        .replace(".\\", ".")
        for _, label, _ in metrics
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] + [":---:"] * len(metrics)) + " |",
    ]
    for model_name, stats in summary_stats.items():
        cells = [model_name]
        for key, _, lower_is_better in metrics:
            pair = stats.get(key)
            mean_v, std_v = (pair[0], pair[1]) if pair is not None and len(pair) >= 2 else (None, None)
            marker = (
                _resolve_link_marker(benchmark_result, key, ref_model, cmp_model, lower_is_better, latex=False)
                if model_name == ref_model
                else ""
            )
            cells.append(
                _format_link_mean_std(
                    mean_v,
                    std_v,
                    decimals=decimals,
                    latex=False,
                    bold=(best_by_metric.get(key) == model_name),
                    marker=marker,
                )
            )
        lines.append("| " + " | ".join(cells) + " |")

    np_label = str(benchmark_result.get("nonparametric_test_label") or "Exact Perm")
    stat_rows = (
        (f"p-value ({ref_model} vs. {cmp_model})", "p_values"),
        (f"Wilcoxon p ({ref_model} vs. {cmp_model})", "wilcoxon_p_values"),
        (f"{np_label} p ({ref_model} vs. {cmp_model})", "bootstrap_p_values"),
        (f"Holm-Bonf adj p ({ref_model} vs. {cmp_model})", "adjusted_p_values"),
        (f"Holm-Bonf Wilcoxon p ({ref_model} vs. {cmp_model})", "adjusted_wilcoxon_p_values"),
        (f"Holm-Bonf {np_label} p ({ref_model} vs. {cmp_model})", "adjusted_bootstrap_p_values"),
    )
    for row_title, map_key in stat_rows:
        pmap = benchmark_result.get(map_key) or {}
        if not pmap:
            continue
        p_cells = [row_title]
        for key, _, _ in metrics:
            pv = pmap.get(key)
            p_cells.append("--" if not _is_finite_num(pv) else f"{float(pv):.4e}")
        lines.append("| " + " | ".join(p_cells) + " |")

    return "\n".join(lines)


def export_link_latex_table(
    benchmark_result: Mapping[str, Any],
    output_path: str | Path,
    *,
    caption: str,
    label: str,
    metrics: Sequence[tuple[str, str, bool]] = LINK_PRED_METRICS,
    decimals: int = 4,
) -> Path:
    """Export IEEEtran `booktabs` LaTeX table with two-sided directional significance markers."""
    summary_stats = benchmark_result["summary_stats"]
    ref_model = str(benchmark_result.get("reference_model", "SBiFAGAT-Link"))
    cmp_model = str(benchmark_result.get("comparator_model", "BiEdgeGATv2-Link"))
    ref_esc = ref_model.replace("_", "\\_").replace("&", "\\&")
    cmp_esc = cmp_model.replace("_", "\\_").replace("&", "\\&")
    best_by_metric = _best_link_model_per_metric(summary_stats, metrics)

    col_spec = "l" + "c" * len(metrics)
    header_row = " & ".join(["Model"] + [m_label for _, m_label, _ in metrics]) + r" \\"

    body_rows: list[str] = []
    for model_name, stats in summary_stats.items():
        escaped_name = model_name.replace("_", "\\_").replace("&", "\\&")
        cells = [escaped_name]
        for key, _, lower_is_better in metrics:
            pair = stats.get(key)
            mean_v, std_v = (pair[0], pair[1]) if pair is not None and len(pair) >= 2 else (None, None)
            marker = (
                _resolve_link_marker(benchmark_result, key, ref_model, cmp_model, lower_is_better, latex=True)
                if model_name == ref_model
                else ""
            )
            cells.append(
                _format_link_mean_std(
                    mean_v,
                    std_v,
                    decimals=decimals,
                    latex=True,
                    bold=(best_by_metric.get(key) == model_name),
                    marker=marker,
                )
            )
        body_rows.append(" & ".join(cells) + r" \\")

    np_label = str(benchmark_result.get("nonparametric_test_label") or "Exact Perm").replace("_", "\\_")
    stat_rows = (
        (f"$p$-value ({ref_esc} vs. {cmp_esc})", "p_values"),
        (f"Wilcoxon $p$ ({ref_esc} vs. {cmp_esc})", "wilcoxon_p_values"),
        (f"{np_label} $p$ ({ref_esc} vs. {cmp_esc})", "bootstrap_p_values"),
        (f"Holm-Bonferroni $p$ ({ref_esc} vs. {cmp_esc})", "adjusted_p_values"),
        (f"Holm-Bonf Wilcoxon $p$ ({ref_esc} vs. {cmp_esc})", "adjusted_wilcoxon_p_values"),
        (f"Holm-Bonf {np_label} $p$ ({ref_esc} vs. {cmp_esc})", "adjusted_bootstrap_p_values"),
    )
    p_lines: list[str] = []
    for row_title, map_key in stat_rows:
        pmap = benchmark_result.get(map_key) or {}
        if not pmap:
            continue
        p_cells = [row_title]
        for key, _, _ in metrics:
            pv = pmap.get(key)
            p_cells.append("--" if not _is_finite_num(pv) else f"{float(pv):.2e}")
        p_lines.append(" & ".join(p_cells) + r" \\")

    p_row_str = ("\\midrule\n" + "\n".join(p_lines) + "\n") if p_lines else ""
    is_wide = len(metrics) >= 5
    env = "table*" if is_wide else "table"
    box_width = "\\textwidth" if is_wide else "\\columnwidth"

    latex = (
        f"\\begin{{{env}}}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\resizebox{{{box_width}}}{{!}}{{\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"{header_row}\n"
        "\\midrule\n"
        + "\n".join(body_rows)
        + "\n"
        + p_row_str
        + "\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        f"\\end{{{env}}}\n"
    )
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(latex, encoding="utf-8")
    return out_file


def _train_single_seed(
    model_name: str,
    dataset: Mapping[str, Any],
    *,
    seed: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    lr: float,
    patience: int,
    batch_size: int,
    mp_graph_mode: str = "coalesced",
    aux_weight: float = 0.2,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    """Train one model for one seed with strict intra-batch unordered-pair masking."""
    seed_everything(seed)
    coalesce_edges = mp_graph_mode == "coalesced"
    x = dataset["x"].to(device)
    train_split = {k: v.to(device) for k, v in dataset["train"].items()}
    val_split = {k: v.to(device) for k, v in dataset["val"].items()}
    test_split = {k: v.to(device) for k, v in dataset["test"].items()}

    train_pairs = train_split["pairs"]
    train_attr = train_split["edge_attr"]
    train_y = train_split["y_oppose"]
    num_users = int(dataset["num_users"])
    num_train = int(train_y.size(0))

    model = build_link_model(
        model_name,
        in_dim=int(dataset["in_dim"]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        coalesce_edges=coalesce_edges,
    ).to(device)

    if isinstance(model, PriorPairSignPersistence):
        model.fit(train_pairs, train_y, num_users=num_users)
        final_val, best_threshold = evaluate_link_split(
            model, x, train_pairs, train_attr, val_split, tune_threshold=True
        )
        final_test, _ = evaluate_link_split(
            model, x, train_pairs, train_attr, test_split, threshold=best_threshold, tune_threshold=False
        )
        return final_val, final_test

    # Precompute unordered pair IDs so mini-batch supervision pairs are masked out
    # of message passing (prevents reading the target edge's own [s_uv, h_uv] attr).
    pair_lo = torch.minimum(train_pairs[0], train_pairs[1])
    pair_hi = torch.maximum(train_pairs[0], train_pairs[1])
    train_pair_keys = pair_lo * num_users + pair_hi

    n_pos = float((train_y > 0.5).sum().item())
    n_neg = float((train_y <= 0.5).sum().item())
    pos_weight_val = min(10.0, max(0.1, n_neg / max(1.0, n_pos))) if n_pos > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_score = -1.0
    best_state: dict[str, Tensor] | None = None
    best_threshold = 0.5
    epochs_no_improve = 0
    gen = torch.Generator(device="cpu").manual_seed(seed)

    eff_batch = (
        min(max(1, batch_size), max(1, num_train // 4))
        if num_train >= 8
        else max(1, batch_size)
    )
    for _epoch in range(1, max(1, epochs) + 1):
        model.train()
        perm = torch.randperm(num_train, generator=gen).to(device)
        for start in range(0, num_train, eff_batch):
            b_idx = perm[start : start + eff_batch]
            batch_keys = torch.unique(train_pair_keys[b_idx])
            keep_mask = ~torch.isin(train_pair_keys, batch_keys)
            mp_ei = train_pairs[:, keep_mask]
            mp_ea = train_attr[keep_mask]

            optimizer.zero_grad(set_to_none=True)
            if hasattr(model, "encode") and hasattr(model, "head"):
                h = model.encode(x, mp_ei, mp_ea)
                logits = model.head(h, train_pairs[:, b_idx])
                loss = criterion(logits, train_y[b_idx])
                if hasattr(model, "compute_aux_loss") and aux_weight > 0.0:
                    loss = loss + float(aux_weight) * model.compute_aux_loss(h, mp_ei, mp_ea)
            else:
                logits = model(x, mp_ei, mp_ea, train_pairs[:, b_idx])
                loss = criterion(logits, train_y[b_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_metrics, val_th = evaluate_link_split(
            model, x, train_pairs, train_attr, val_split, tune_threshold=True
        )
        score = float(val_metrics.get("roc_auc", float("nan")))
        if not math.isfinite(score):
            score = float(val_metrics.get("macro_f1", -1.0))
        if best_state is None or (math.isfinite(score) and score > best_score + 1e-4):
            best_score = score
            best_threshold = val_th
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= max(1, patience):
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    final_val, best_threshold = evaluate_link_split(
        model, x, train_pairs, train_attr, val_split, tune_threshold=True
    )
    final_test, _ = evaluate_link_split(
        model, x, train_pairs, train_attr, test_split, threshold=best_threshold, tune_threshold=False
    )
    return final_val, final_test


def _fmt_metric_inline(val: float) -> str:
    return "nan" if not math.isfinite(val) else f"{val:.4f}"


def run_signed_link_benchmark(
    data_dir: Path,
    *,
    dataset_name: str | None = None,
    models: Sequence[str] = LINK_PRED_MODEL_NAMES,
    seeds: Sequence[int] = OFFICIAL_SEEDS,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 40,
    lr: float = 1e-3,
    patience: int = 8,
    batch_size: int = 1024,
    stance_threshold: float = 0.1,
    mp_graph_mode: str = "coalesced",
    aux_weight: float = 0.2,
    prefer_teacher: bool = True,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run multi-seed chronological signed link prediction benchmark."""
    if mp_graph_mode not in {"coalesced", "multiedge"}:
        raise ValueError(f"mp_graph_mode must be 'coalesced' or 'multiedge', got {mp_graph_mode!r}")
    run_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_temporal_signed_link_dataset(
        data_dir,
        dataset_name=dataset_name,
        stance_threshold=stance_threshold,
        prefer_teacher=prefer_teacher,
    )
    raw_val: dict[str, list[dict[str, float]]] = {m: [] for m in models}
    raw_test: dict[str, list[dict[str, float]]] = {m: [] for m in models}

    for model_name in models:
        for s in seeds:
            v_m, t_m = _train_single_seed(
                model_name,
                dataset,
                seed=int(s),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                epochs=epochs,
                lr=lr,
                patience=patience,
                batch_size=batch_size,
                mp_graph_mode=mp_graph_mode,
                aux_weight=aux_weight,
                device=run_device,
            )
            raw_val[model_name].append(v_m)
            raw_test[model_name].append(t_m)
            print(
                f"  [{dataset['dataset']}] {model_name} seed={s}: "
                f"Val ROC={_fmt_metric_inline(v_m['roc_auc'])} | "
                f"Test ROC={_fmt_metric_inline(t_m['roc_auc'])} "
                f"UnseenPairTrans={_fmt_metric_inline(t_m['unseen_pair_trans_roc_auc'])} "
                f"Unseen={_fmt_metric_inline(t_m['unseen_roc_auc'])}",
                flush=True,
            )

    val_summary = {m: _summarize_seed_metrics(raw_val[m]) for m in models}
    test_summary = {m: _summarize_seed_metrics(raw_test[m]) for m in models}

    ref_model = "SBiFAGAT-Link" if "SBiFAGAT-Link" in models else str(models[-1])
    cmp_model = _select_best_val_comparator(val_summary, reference_model=ref_model)

    p_values: dict[str, float] = {}
    boot_p_values: dict[str, float] = {}
    wilcoxon_p_values: dict[str, float] = {}
    if ref_model in raw_test and cmp_model in raw_test and len(seeds) >= 2:
        for key, _, _ in LINK_PRED_METRICS:
            a_vals = [float(r[key]) for r in raw_test[ref_model]]
            b_vals = [float(r[key]) for r in raw_test[cmp_model]]
            p_values[key] = compute_paired_ttest(a_vals, b_vals)
            boot_p_values[key] = compute_paired_bootstrap_pvalue(a_vals, b_vals)
            wilcoxon_p_values[key] = compute_wilcoxon_signed_rank(a_vals, b_vals)

    test_split = dataset["test"]
    return {
        "dataset": dataset["dataset"],
        "edge_label_source": dataset["edge_label_source"],
        "edge_source": dataset.get("edge_source", dataset["edge_label_source"]),
        "mp_graph_mode": mp_graph_mode,
        "seeds": [int(s) for s in seeds],
        "reference_model": ref_model,
        "comparator_model": cmp_model,
        "summary_stats": test_summary,
        "val_summary_stats": val_summary,
        "p_values": p_values,
        "adjusted_p_values": holm_bonferroni_adjust(p_values),
        "bootstrap_p_values": boot_p_values,
        "adjusted_bootstrap_p_values": holm_bonferroni_adjust(boot_p_values),
        "wilcoxon_p_values": wilcoxon_p_values,
        "adjusted_wilcoxon_p_values": holm_bonferroni_adjust(wilcoxon_p_values),
        "nonparametric_test_label": "Exact Perm" if len(seeds) < 6 else "Bootstrap",
        "raw_results": raw_test,
        "val_raw_results": raw_val,
        "split_counts": {
            "num_users": int(dataset["num_users"]),
            "train_users": int(len(dataset["train_user_set"])),
            "train_edges": int(dataset["train"]["y_oppose"].numel()),
            "val_edges": int(dataset["val"]["y_oppose"].numel()),
            "test_edges": int(test_split["y_oppose"].numel()),
            "test_transductive_edges": int(test_split["both_seen"].sum().item()),
            "test_unseen_pair_transductive_edges": int(test_split["unseen_pair_transductive"].sum().item()),
            "test_unseen_edges": int((~test_split["both_seen"]).sum().item()),
        },
    }


def _collect_macro_stats_for_table(paper_dir: Path, latest_stats: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Combine any sibling `results_macro_network_*.json` artifacts so tables stay multi-corpus."""
    by_name: dict[str, Mapping[str, Any]] = {}
    for jf in sorted(paper_dir.glob("results_macro_network_*.json")):
        try:
            loaded = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(loaded, Mapping) and "dataset" in loaded and "structural_balance" in loaded:
            by_name[str(loaded["dataset"]).lower()] = loaded
    by_name[str(latest_stats["dataset"]).lower()] = latest_stats
    order = {"vn": 0, "pheme": 1}
    return [by_name[k] for k in sorted(by_name.keys(), key=lambda s: (order.get(s, 99), s))]


def run_smoke_test() -> None:
    """Offline <15s synthetic smoke test verifying macro stats + all 10 signed link models end-to-end."""
    from datetime import datetime, timezone

    import pyarrow as pa

    from src.data.schema import EDGES_ARROW_SCHEMA, NODES_ARROW_SCHEMA, save_cascade_parquet

    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "smoke_ds"
        d.mkdir(parents=True, exist_ok=True)
        out_paper = Path(tmp) / "paper"
        out_fig = Path(tmp) / "figs"

        nodes_rows: list[dict[str, Any]] = []
        edges_rows: list[dict[str, Any]] = []
        edge_stance_map: dict[str, float] = {}
        edge_host_map: dict[str, float] = {}
        node_idx = 0
        t_base = 1_700_000_000.0
        for c_id in range(6):
            r_user = f"u_{c_id % 10:02d}"
            t_root = datetime.fromtimestamp(t_base, tz=timezone.utc)
            root_nid = f"n_{node_idx}"
            nodes_rows.append(
                {
                    "node_id": root_nid,
                    "cascade_id": f"c_{c_id}",
                    "topic_id": f"topic_{c_id % 2}",
                    "parent_id": None,
                    "is_root": True,
                    "text": f"Root post {c_id}",
                    "created_at_utc": t_root,
                    "lang": "vi",
                    "source_dataset": "smoke",
                    "user_id_hash": r_user,
                    "rumor_label": None,
                }
            )
            node_idx += 1
            for r_i in range(14):
                t_child_sec = t_base + 60.0 * (r_i + 1)
                t_child = datetime.fromtimestamp(t_child_sec, tz=timezone.utc)
                src_u = f"u_{(c_id * 3 + r_i) % 16:02d}"
                dst_u = r_user if src_u != r_user else "u_17"
                same_camp = (int(src_u.split("_")[1]) % 2) == (int(dst_u.split("_")[1]) % 2)
                s_uv = float(0.65 if same_camp else -0.65)
                h_uv = float(0.10 if same_camp else 0.72)
                nid = f"n_{node_idx}"
                eid = f"e_{c_id}_{r_i}"
                node_idx += 1
                nodes_rows.append(
                    {
                        "node_id": nid,
                        "cascade_id": f"c_{c_id}",
                        "topic_id": f"topic_{c_id % 2}",
                        "parent_id": root_nid,
                        "is_root": False,
                        "text": f"Reply {r_i} in cascade {c_id}",
                        "created_at_utc": t_child,
                        "lang": "vi",
                        "source_dataset": "smoke",
                        "user_id_hash": src_u,
                        "rumor_label": None,
                    }
                )
                edges_rows.append(
                    {
                        "edge_id": eid,
                        "cascade_id": f"c_{c_id}",
                        "source_id": root_nid,
                        "target_id": nid,
                        "created_at_utc": t_child,
                        "time_delay_sec": float(60.0 * (r_i + 1)),
                        "edge_type": "reply",
                    }
                )
                edge_stance_map[eid] = s_uv
                edge_host_map[eid] = h_uv
            t_base += 1200.0

        nodes_tbl = pa.Table.from_pylist(nodes_rows, schema=NODES_ARROW_SCHEMA)
        edges_tbl = pa.Table.from_pylist(edges_rows, schema=EDGES_ARROW_SCHEMA)
        save_cascade_parquet(nodes_tbl, edges_tbl, d, validate=True)

        embs = torch.tensor(rng.normal(0.0, 0.3, size=(len(nodes_rows), 16)), dtype=torch.float32)
        for row_i, row in enumerate(nodes_rows):
            camp = int(row["user_id_hash"].split("_")[1]) % 2
            embs[row_i, camp * 4 : camp * 4 + 4] += 1.5
        torch.save(
            {
                "node_embeddings": embs,
                "node_id_to_idx": {r["node_id"]: i for i, r in enumerate(nodes_rows)},
                "edge_stance": edge_stance_map,
                "edge_hostility": edge_host_map,
                "teacher_edge_stance": edge_stance_map,
                "teacher_edge_hostility": edge_host_map,
            },
            d / "nlp_features.pt",
        )

        stats = compute_macro_signed_network_stats(d, num_null_perms=20, num_q_null_perms=5, seed=42)
        assert stats["num_users"] >= 10
        assert stats["structural_balance"]["num_triangles"] >= 0
        assert "weak_balance_ratio" in stats["structural_balance"]
        assert "q_signed_z_score" in stats["polarization"]
        out_paper.mkdir(parents=True, exist_ok=True)
        out_fig.mkdir(parents=True, exist_ok=True)
        (out_paper / "results_macro_network_smoke.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
        tex_macro = format_macro_network_latex([stats])
        assert "\\begin{table" in tex_macro
        pdf_p, svg_p = plot_macro_signed_network(stats, output_dir=out_fig)
        assert pdf_p.exists() and svg_p.exists()

        bench = run_signed_link_benchmark(
            d,
            models=LINK_PRED_MODEL_NAMES,
            seeds=[42, 100],
            hidden_dim=16,
            num_layers=1,
            epochs=1,
            patience=1,
            batch_size=32,
        )
        assert len(bench["summary_stats"]) == len(LINK_PRED_MODEL_NAMES)
        assert "wilcoxon_p_values" in bench and "adjusted_wilcoxon_p_values" in bench
        assert "unseen_pair_trans_roc_auc" in bench["summary_stats"]["SBiFAGAT-Link"]
        assert "unseen_roc_auc" in bench["summary_stats"]["SBiFAGAT-Link"]

        md_table = build_link_markdown_table(bench)
        assert "Wilcoxon p" in md_table
        tex_path = export_link_latex_table(
            bench,
            out_paper / "tables_signed_link_smoke.tex",
            caption="Smoke Signed Link Benchmark.",
            label="tab:smoke_link",
            metrics=LINK_PRED_METRICS,
        )
        assert tex_path.exists()
        serialize_benchmark_json(
            bench,
            out_paper / "results_signed_link_smoke.json",
            extra={
                "val_summary_stats": bench["val_summary_stats"],
                "split_counts": bench["split_counts"],
                "wilcoxon_p_values": bench["wilcoxon_p_values"],
                "adjusted_wilcoxon_p_values": bench["adjusted_wilcoxon_p_values"],
                "nonparametric_test_label": bench["nonparametric_test_label"],
                "mp_graph_mode": bench["mp_graph_mode"],
            },
        )
        print("SMOKE TEST PASSED:", json.dumps(bench["split_counts"]))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tier 2 Macro Signed User Network & Link Prediction")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/vn"), help="Processed dataset directory")
    parser.add_argument("--dataset-name", type=str, default=None, help="Override canonical dataset tag (e.g. vn or pheme)")
    parser.add_argument("--analysis-only", action="store_true", help="Run sparse CPU network polarization analytics + fig5 only")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Subset of link prediction models to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(OFFICIAL_SEEDS), help="Random seeds")
    parser.add_argument("--epochs", type=int, default=40, help="Max epochs per seed for signed link prediction")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience on Val ROC-AUC")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training mini-batch size")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GNN latent width")
    parser.add_argument("--num-layers", type=int, default=2, help="GNN layer depth")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num-null-perms", type=int, default=100, help="Uniform edge-sign permutations for SBT z-score")
    parser.add_argument("--num-q-null-perms", type=int, default=20, help="Sign permutations for Q_signed null-graph test")
    parser.add_argument(
        "--mp-graph-mode",
        type=str,
        choices=["coalesced", "multiedge"],
        default="coalesced",
        help="Message-passing edge canonicalization mode across all GNNs",
    )
    parser.add_argument("--aux-weight", type=float, default=0.2, help="Auxiliary structural loss weight (SDGNN)")
    parser.add_argument("--stance-threshold", type=float, default=0.1, help="Minimum |s_uv| for signed interactions")
    parser.add_argument("--prefer-student", action="store_true", help="Use student edge stance instead of teacher")
    parser.add_argument("--paper-dir", type=Path, default=Path("results"), help="Output directory for JSON and LaTeX tables")
    parser.add_argument("--figures-dir", type=Path, default=None, help="Override figure output dir (default results/figures/<ds>)")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast offline synthetic smoke test in tempdir")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.smoke_test:
        run_smoke_test()
        return

    data_dir = Path(args.data_dir)
    paper_dir = Path(args.paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_macro_signed_network_stats(
        data_dir,
        dataset_name=args.dataset_name,
        stance_threshold=args.stance_threshold,
        num_null_perms=args.num_null_perms,
        num_q_null_perms=args.num_q_null_perms,
        seed=42,
    )
    ds_name = str(stats["dataset"])
    macro_json_path = paper_dir / f"results_macro_network_{ds_name}.json"
    macro_json_path.write_text(json.dumps(stats, indent=2, allow_nan=False), encoding="utf-8")

    merged_stats = _collect_macro_stats_for_table(paper_dir, stats)
    macro_tex_path = paper_dir / "tables_macro_network.tex"
    macro_tex_path.write_text(format_macro_network_latex(merged_stats), encoding="utf-8")

    fig_dir = Path(args.figures_dir) if args.figures_dir is not None else (paper_dir / "figures" / ds_name)
    pdf_p, svg_p = plot_macro_signed_network(stats, output_dir=fig_dir)
    print(
        f"[{ds_name}] Macro Signed Network: active={stats['active_graph_users']:,}/{stats['num_users']:,} "
        f"interactions={stats['num_interactions']:,} balanced={stats['structural_balance']['balanced_ratio']:.4f} "
        f"(z={stats['structural_balance']['balance_z_score']:.2f}), "
        f"weak_bal={stats['structural_balance']['weak_balance_ratio']:.4f} "
        f"(z={stats['structural_balance']['weak_balance_z_score']:.2f}), "
        f"Q_signed={stats['polarization']['signed_modularity']:.4f} "
        f"(z_Q={stats['polarization']['q_signed_z_score']:.2f}), "
        f"RWC={stats['polarization']['rwc']:.4f}",
        flush=True,
    )
    print(f"Exported: {macro_json_path}, {macro_tex_path}, {pdf_p}, {svg_p}", flush=True)

    if args.analysis_only:
        return

    selected_models = tuple(args.models) if args.models else LINK_PRED_MODEL_NAMES
    bench = run_signed_link_benchmark(
        data_dir,
        dataset_name=args.dataset_name,
        models=selected_models,
        seeds=args.seeds,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        batch_size=args.batch_size,
        stance_threshold=args.stance_threshold,
        mp_graph_mode=args.mp_graph_mode,
        aux_weight=args.aux_weight,
        prefer_teacher=not args.prefer_student,
    )
    link_json = paper_dir / f"results_signed_link_{ds_name}.json"
    link_tex = paper_dir / f"tables_signed_link_{ds_name}.tex"
    link_md = paper_dir / f"summary_signed_link_{ds_name}.md"
    ds_upper = ds_name.upper().replace("_", "\\_")
    export_link_latex_table(
        bench,
        link_tex,
        caption=(
            f"Macro Temporal Signed Link Prediction on {ds_upper} (opposing vs.\\ supporting reply edges, "
            f"mean $\\pm$ std over {len(args.seeds)} seeds; train batch pairs are masked; user features are aggregated strictly from pre-cutoff Train timestamps)."
        ),
        label=f"tab:signed_link_{ds_name}",
        metrics=LINK_PRED_METRICS,
    )
    link_md.write_text(build_link_markdown_table(bench) + "\n", encoding="utf-8")
    serialize_benchmark_json(
        bench,
        link_json,
        extra={
            "val_summary_stats": bench["val_summary_stats"],
            "split_counts": bench["split_counts"],
            "wilcoxon_p_values": bench["wilcoxon_p_values"],
            "adjusted_wilcoxon_p_values": bench["adjusted_wilcoxon_p_values"],
            "nonparametric_test_label": bench["nonparametric_test_label"],
            "mp_graph_mode": bench["mp_graph_mode"],
            "edge_source": bench["edge_source"],
        },
    )
    print(f"Exported signed link benchmark: {link_json}, {link_tex}, {link_md}", flush=True)


if __name__ == "__main__":
    main()

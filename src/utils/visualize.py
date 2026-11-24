"""Publication-quality vector figure exporters (.pdf and .svg) for SBi-FAGAT.

Produces three camera-ready IEEE/Springer figures in `paper/figures/`:
  1. `fig1_umap_polarization.{pdf,svg}`: 2D UMAP node embedding comparison
     between `StandardGCN` and `SBiFAGAT`.
  2. `fig2_early_k_curve.{pdf,svg}`: Early horizon sensitivity curve across
     `K in {10, 20, 30, 50}` for MSLE (size) and RMSE (polarization).
  3. `fig3_cascade_heatmap.{pdf,svg}` + `fig3_top5_trigger_comments.txt`:
     Hierarchical propagation tree colored by `|gamma_uv|` with top-5 trigger
     comments exported to text.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D


# Okabe-Ito academic colorblind-safe palette
MODEL_PALETTE: dict[str, tuple[str, str]] = {
    "TextMLP": ("#7F7F7F", ":"),
    "StandardGCN": ("#0072B2", "--"),
    "StandardGATv2": ("#E69F00", "-."),
    "BiGCN": ("#009E73", "--"),
    "SBiFAGAT": ("#D55E00", "-"),
}


def _configure_academic_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _to_numpy(arr: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _save_vector_pair(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(svg_path, format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return pdf_path, svg_path


def _project_umap_2d(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    import umap

    n_samples = embeddings.shape[0]
    if n_samples < 4:
        # Degenerate tiny fallback (safety guard; real graphs have dozens/hundreds of nodes)
        coords = np.zeros((n_samples, 2), dtype=np.float32)
        coords[:, : min(2, embeddings.shape[1])] = embeddings[:, : min(2, embeddings.shape[1])]
        return coords

    n_neighbors = max(2, min(15, n_samples - 1))
    init_mode = "spectral" if n_samples >= 20 else "random"
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.15,
        metric="cosine",
        init=init_mode,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def plot_umap_polarization(
    gcn_embeddings: np.ndarray | torch.Tensor,
    sbifagat_embeddings: np.ndarray | torch.Tensor,
    node_stances: np.ndarray | torch.Tensor,
    *,
    output_dir: str | Path = "paper/figures",
    stem: str = "fig1_umap_polarization",
) -> tuple[Path, Path]:
    """Generate Fig 1: UMAP 2D comparison of StandardGCN vs. SBi-FAGAT node embeddings."""
    _configure_academic_rcparams()
    gcn_np = _to_numpy(gcn_embeddings).astype(np.float32)
    sbi_np = _to_numpy(sbifagat_embeddings).astype(np.float32)
    stances_np = np.clip(_to_numpy(node_stances).reshape(-1).astype(np.float32), -1.0, 1.0)

    gcn_2d = _project_umap_2d(gcn_np)
    sbi_2d = _project_umap_2d(sbi_np)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    cmap = plt.get_cmap("coolwarm_r")
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    panels = [
        (axes[0], gcn_2d, "StandardGCN (Undirected Low-Pass)"),
        (axes[1], sbi_2d, "SBi-FAGAT (Stance-Guided Bidirectional)"),
    ]
    scatter_handle = None
    for ax, coords, title in panels:
        scatter_handle = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=stances_np,
            cmap=cmap,
            norm=norm,
            s=28,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.4,
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True, linestyle=":", alpha=0.35)

    cbar = fig.colorbar(scatter_handle, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label("Root-Aligned Stance ($\\tilde{s}_v$: Oppose $-1$ $\\leftrightarrow$ Support $+1$)")

    return _save_vector_pair(fig, Path(output_dir), stem)


def plot_early_k_curve(
    k_results: Mapping[int, Mapping[str, Mapping[str, tuple[float, float]]]],
    *,
    output_dir: str | Path = "paper/figures",
    stem: str = "fig2_early_k_curve",
    target_k: int | None = 20,
) -> tuple[Path, Path]:
    """Generate Fig 2: MSLE & RMSE error curves across observation horizons K in {10..50}."""
    _configure_academic_rcparams()
    ks = sorted(k_results.keys())
    model_names = list(k_results[ks[0]].keys()) if ks else []

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    metric_specs = [
        ("msle", "Cascade Size Error (MSLE $\\downarrow$)", axes[0]),
        ("rmse", "Polarization Error (RMSE $\\downarrow$)", axes[1]),
    ]

    for metric_key, ylabel, ax in metric_specs:
        for model_name in model_names:
            means = np.array([k_results[k][model_name][metric_key][0] for k in ks], dtype=float)
            stds = np.array([k_results[k][model_name][metric_key][1] for k in ks], dtype=float)
            color, linestyle = MODEL_PALETTE.get(model_name, ("#333333", "-"))
            lw = 2.4 if model_name == "SBiFAGAT" else 1.6
            marker = "o" if model_name == "SBiFAGAT" else "s"

            ax.plot(
                ks,
                means,
                label=model_name,
                color=color,
                linestyle=linestyle,
                linewidth=lw,
                marker=marker,
                markersize=5.5,
            )
            ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.14)

        if target_k is not None and int(target_k) in ks:
            tk = int(target_k)
            ax.axvline(tk, color="#444444", linestyle="--", linewidth=1.1, alpha=0.75)
            y_top = ax.get_ylim()[1]
            ax.text(tk + 0.6, y_top * 0.96, f"$K={tk}$ (Target)", fontsize=8.5, va="top", color="#333333")

        ax.set_xticks(ks)
        ax.set_xlabel("Early Observation Horizon $K$ (Replies)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[0].set_title("Early Cascade Size Prediction")
    axes[1].set_title("Early Polarization Prediction")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=max(1, len(model_names)),
        frameon=False,
    )

    return _save_vector_pair(fig, Path(output_dir), stem)


def _compute_tree_layout(num_nodes: int, edges: list[tuple[int, int]], root: int = 0) -> dict[int, tuple[float, float]]:
    """Compute non-overlapping hierarchical 2D coordinates for a rooted tree."""
    children: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
    for u, v in edges:
        if u != v and v not in children[u]:
            children[u].append(v)

    subtree_width: dict[int, float] = {}
    visited_w: set[int] = set()

    def _measure(node: int) -> float:
        if node in visited_w:
            return 1.0
        visited_w.add(node)
        ch = [c for c in children.get(node, []) if c not in visited_w]
        if not ch:
            subtree_width[node] = 1.0
            return 1.0
        w = sum(_measure(c) for c in ch)
        subtree_width[node] = max(1.0, w)
        return subtree_width[node]

    _measure(root)

    positions: dict[int, tuple[float, float]] = {}

    def _assign(node: int, x_left: float, depth: int) -> None:
        w = subtree_width.get(node, 1.0)
        positions[node] = (x_left + 0.5 * w, -float(depth))
        cursor = x_left
        for c in children.get(node, []):
            if c not in positions:
                cw = subtree_width.get(c, 1.0)
                _assign(c, cursor, depth + 1)
                cursor += cw

    _assign(root, 0.0, 0)

    # Place any unreached nodes defensively
    fallback_x = float(subtree_width.get(root, 1.0)) + 0.5
    for idx in range(num_nodes):
        if idx not in positions:
            positions[idx] = (fallback_x, -1.0)
            fallback_x += 1.0

    return positions


def plot_cascade_heatmap(
    data: Any,
    edge_importance: np.ndarray | torch.Tensor,
    signed_gamma: np.ndarray | torch.Tensor,
    *,
    output_dir: str | Path = "paper/figures",
    stem: str = "fig3_cascade_heatmap",
    top_k_comments: int = 5,
) -> tuple[Path, Path, Path]:
    """Generate Fig 3: Propagation tree edge heatmap by `|gamma_uv|` + top-5 triggers."""
    _configure_academic_rcparams()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_index_td = getattr(data, "edge_index_td", data.edge_index).detach().cpu().numpy()
    imp = _to_numpy(edge_importance).reshape(-1)
    gamma = _to_numpy(signed_gamma).reshape(-1)
    edge_attr = (
        data.edge_attr.detach().cpu().numpy()
        if hasattr(data, "edge_attr") and data.edge_attr is not None
        else np.zeros((edge_index_td.shape[1], 2), dtype=np.float32)
    )

    num_nodes = int(data.num_nodes)
    root_idx = int(data.root_index.view(-1)[0].item()) if hasattr(data, "root_index") else 0
    edge_pairs = [(int(edge_index_td[0, i]), int(edge_index_td[1, i])) for i in range(edge_index_td.shape[1])]
    pos = _compute_tree_layout(num_nodes, edge_pairs, root=root_idx)

    node_ids = list(getattr(data, "node_ids", [f"node_{i}" for i in range(num_nodes)]))
    node_texts = list(getattr(data, "node_texts", [f"[Node {i}]" for i in range(num_nodes)]))

    ranked_edge_indices = np.argsort(-imp) if imp.size > 0 else np.array([], dtype=np.int64)
    top_indices = ranked_edge_indices[: min(top_k_comments, len(ranked_edge_indices))]
    top_target_rank = {edge_pairs[idx][1]: rank + 1 for rank, idx in enumerate(top_indices)}

    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    cmap = plt.get_cmap("inferno_r")
    norm = mcolors.Normalize(vmin=0.0, vmax=max(1e-3, float(imp.max()) if imp.size > 0 else 1.0))

    for e_i, (u, v) in enumerate(edge_pairs):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        val = float(imp[e_i])
        g_signed = float(gamma[e_i])
        color = cmap(norm(val))
        linestyle = "-" if g_signed >= 0.0 else "--"
        lw = 1.2 + 3.2 * (val / max(1e-6, float(norm.vmax)))
        ax.plot([x0, x1], [y0, y1], color=color, linestyle=linestyle, linewidth=lw, alpha=0.92, zorder=1)

    for node_idx in range(num_nodes):
        xn, yn = pos[node_idx]
        if node_idx == root_idx:
            ax.scatter([xn], [yn], s=180, color="#1F77B4", edgecolors="black", linewidths=1.2, zorder=3)
            ax.text(xn, yn + 0.14, "Root", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        elif node_idx in top_target_rank:
            rank = top_target_rank[node_idx]
            ax.scatter([xn], [yn], s=135, color="#D55E00", edgecolors="black", linewidths=1.1, zorder=3)
            ax.text(xn, yn - 0.14, f"#{rank}", ha="center", va="top", fontsize=8.5, fontweight="bold", color="#B22222")
        else:
            ax.scatter([xn], [yn], s=65, color="#E0E0E0", edgecolors="#444444", linewidths=0.7, zorder=2)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Edge Routing Weight $|\\gamma_{uv}|$")

    legend_items = [
        Line2D([0], [0], color="#333333", lw=2.0, linestyle="-", label="Positive Gate ($\\gamma_{uv} \\geq 0$)"),
        Line2D([0], [0], color="#333333", lw=2.0, linestyle="--", label="Conflict Gate ($\\gamma_{uv} < 0$)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1F77B4", markeredgecolor="k", markersize=8, label="Source Post (Root)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D55E00", markeredgecolor="k", markersize=8, label="Top-5 Trigger Replies"),
    ]
    ax.legend(handles=legend_items, loc="lower left", frameon=True, framealpha=0.9)

    cascade_label = getattr(data, "cascade_id", "cascade")
    ax.set_title(f"Cascade Propagation Gate Heatmap ($|\\gamma_{{uv}}|$) — {cascade_label}")
    ax.set_xticks([])
    ax.set_ylabel("Tree Depth Level")
    ax.margins(x=0.08, y=0.15)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(False)

    pdf_path, svg_path = _save_vector_pair(fig, out_dir, stem)

    # Export Top-5 trigger comments text report
    txt_path = out_dir / "fig3_top5_trigger_comments.txt"
    lines = [
        f"Cascade ID: {cascade_label}",
        f"Top-{len(top_indices)} Trigger Comments Ranked by |gamma_uv|",
        "=" * 72,
    ]
    for rank, e_idx in enumerate(top_indices, start=1):
        u, v = edge_pairs[e_idx]
        s_uv = float(edge_attr[e_idx, 0]) if edge_attr.shape[1] >= 1 else 0.0
        h_uv = float(edge_attr[e_idx, 1]) if edge_attr.shape[1] >= 2 else 0.0
        u_id = node_ids[u] if u < len(node_ids) else f"node_{u}"
        v_id = node_ids[v] if v < len(node_ids) else f"node_{v}"
        v_text = node_texts[v] if v < len(node_texts) else ""
        lines.append(
            f"#{rank} | node_id={v_id} (parent={u_id}) | |gamma_mean|={float(imp[e_idx]):.4f} "
            f"| gamma_td={float(gamma[e_idx]):+.4f} | s_uv={s_uv:+.4f} | h_uv={h_uv:.4f}"
        )
        lines.append(f"    Text: {v_text}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return pdf_path, svg_path, txt_path


def plot_counterfactual_interventions(
    interventions_summary: Mapping[str, Mapping[str, Any]],
    *,
    output_dir: str | Path = "paper/figures",
    stem: str = "fig4_counterfactual_interventions",
) -> tuple[Path, Path]:
    """Generate Fig 4: Counterfactual moderation impact on predicted size & polarization."""
    _configure_academic_rcparams()

    default_order = [
        ("cool_top3_hostility", "Cool Top-3\nHostility"),
        ("cool_all_hostility", "Cool All\nHostility"),
        ("neutralize_top3_conflict", "Neutralize Top-3\nConflict"),
        ("neutralize_all_oppose", "Neutralize All\nOpposition"),
    ]
    ordered = [
        (key, lbl) for key, lbl in default_order if key in interventions_summary
    ]
    for key in interventions_summary:
        if all(k != key for k, _ in ordered):
            ordered.append((key, key.replace("_", "\n")))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    if not ordered:
        for ax in axes:
            ax.text(0.5, 0.5, "No counterfactual data", ha="center", va="center")
        return _save_vector_pair(fig, Path(output_dir), stem)

    x = np.arange(len(ordered), dtype=float)
    bar_width = 0.36
    cohorts = [
        ("all_test", "All Test Cascades", "#0072B2", -0.5 * bar_width),
        ("high_polar_q75", "High-Polarization (Q75+)", "#D55E00", 0.5 * bar_width),
    ]
    panels = [
        (axes[0], "mean_delta_y_size", "ci95_delta_y_size", "Predicted Log-Size Shift ($\\Delta\\hat{y}_{\\mathrm{size}}$)", "Propagation Size Response"),
        (axes[1], "mean_delta_y_polar", "ci95_delta_y_polar", "Predicted Polarization Shift ($\\Delta\\hat{y}_{\\mathrm{polar}}$)", "Polarization Response"),
    ]

    for ax, mean_key, ci_key, ylabel, title in panels:
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.8)
        for c_key, c_label, color, offset in cohorts:
            means: list[float] = []
            err_lo: list[float] = []
            err_hi: list[float] = []
            for int_key, _ in ordered:
                entry = interventions_summary.get(int_key, {}).get(c_key, {})
                mv = entry.get(mean_key)
                m_val = 0.0 if mv is None or not math.isfinite(float(mv)) else float(mv)
                ci = entry.get(ci_key)
                if isinstance(ci, (list, tuple)) and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
                    lo_v = float(ci[0])
                    hi_v = float(ci[1])
                    if math.isfinite(lo_v) and math.isfinite(hi_v):
                        err_lo.append(max(0.0, m_val - lo_v))
                        err_hi.append(max(0.0, hi_v - m_val))
                    else:
                        err_lo.append(0.0)
                        err_hi.append(0.0)
                else:
                    err_lo.append(0.0)
                    err_hi.append(0.0)
                means.append(m_val)

            ax.bar(
                x + offset,
                means,
                width=bar_width,
                color=color,
                alpha=0.88,
                edgecolor="#222222",
                linewidth=0.6,
                yerr=[err_lo, err_hi],
                capsize=3.5,
                error_kw={"elinewidth": 1.1, "ecolor": "#222222"},
                label=c_label,
                zorder=2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in ordered], fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle=":", alpha=0.45, zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=2,
        frameon=False,
    )
    return _save_vector_pair(fig, Path(output_dir), stem)

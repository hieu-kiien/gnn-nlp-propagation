"""Mechanistic Polarization-Propagation Coupling & Counterfactual Moderation Analysis.

Performs strictly leakage-free post-hoc scientific analysis on a trained
`SBiFAGAT` checkpoint and validated `EarlyCascadeDataset` split:
  1. Part A — Empirical Coupling Audit: early discourse hostility/stance
     correlations with future growth & future polarization, stratification into
     3 early discourse regimes (Low Conflict, Civil Disagreement, Hostile Conflict),
     and SBi-FAGAT top-3 trigger edge depth profiling (depth=1 vs depth>=2).
  2. Part B — Counterfactual Moderation Stress Test: controlled non-mutating
     what-if interventions (`cool_top3_hostility`, `cool_all_hostility`,
     `neutralize_top3_conflict`, `neutralize_all_oppose`) reporting paired
     bootstrap 95% confidence intervals on `All Test` and `High-Polar (Q75+)`
     cascades.
  3. Part C — Deterministic JSON, IEEEtran booktabs LaTeX table, and vector
     figure (`fig4_counterfactual_interventions.{pdf,svg}`) export.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr

from src.data.pyg_dataset import (
    DEFAULT_FORECAST_MIN_REPLIES,
    EarlyCascadeDataset,
    TOTAL_NODE_FEATURE_DIM,
    _compute_full_tree_polarization,
    _compute_root_aligned_stances_and_orientations,
)
from src.models.sbi_fagat import SBiFAGAT
from src.training.reporting import _sanitize_json_value
from src.utils.visualize import plot_counterfactual_interventions


INTERVENTION_LABELS: dict[str, str] = {
    "cool_top3_hostility": "Cool Top-3 Trigger Hostility",
    "cool_all_hostility": "Cool All Early Hostility",
    "neutralize_top3_conflict": "Neutralize Top-3 Trigger Conflict",
    "neutralize_all_oppose": "Neutralize All Opposition",
}


def load_sbifagat_checkpoint(
    ckpt_path: Path,
    *,
    device: torch.device | None = None,
) -> tuple[SBiFAGAT, dict[str, Any]]:
    """Reconstruct and load a trained `SBiFAGAT` checkpoint with `strict=True`."""
    payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    cfg = dict(payload.get("config") or {})
    state_dict = payload["model_state_dict"]
    in_dim = int(payload.get("in_dim", TOTAL_NODE_FEATURE_DIM))
    num_cls = int(payload.get("num_rumor_classes", 4))
    has_persist = (
        bool(cfg["use_polar_persistence_anchor"])
        if "use_polar_persistence_anchor" in cfg
        else ("head.persist_weight" in state_dict)
    )
    model = SBiFAGAT(
        in_dim=in_dim,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        edge_dim=int(cfg.get("edge_dim", 2)),
        num_rumor_classes=num_cls,
        eps=float(cfg.get("eps", 0.5)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_cascade_stats=bool(cfg.get("use_cascade_stats", True)),
        predict_uncertainty=bool(cfg.get("predict_uncertainty", False)),
        use_polar_persistence_anchor=has_persist,
    )
    model.load_state_dict(state_dict, strict=True)
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(target_device)
    model.eval()
    return model, payload


def _safe_corr_pair(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    """Compute Spearman rho and Pearson r safely, returning NaN on constant/tiny arrays."""
    ax = np.asarray(x, dtype=np.float64)
    ay = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(ax) & np.isfinite(ay)
    ax = ax[mask]
    ay = ay[mask]
    if ax.size < 3 or np.ptp(ax) <= 1e-12 or np.ptp(ay) <= 1e-12:
        return {"spearman_rho": float("nan"), "pearson_r": float("nan"), "n": int(ax.size)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        rho_res = spearmanr(ax, ay)
        r_res = pearsonr(ax, ay)
    return {
        "spearman_rho": float(rho_res.statistic),
        "pearson_r": float(r_res.statistic),
        "n": int(ax.size),
    }


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, list[float]]:
    """Return `(mean, [ci_2.5, ci_97.5])` using a deterministic paired bootstrap."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), [float("nan"), float("nan")]
    mean_v = float(np.mean(arr))
    if arr.size == 1:
        return mean_v, [mean_v, mean_v]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(max(100, int(n_boot)), arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return mean_v, [float(lo), float(hi)]


def _resolve_edge_depths(data: Any) -> np.ndarray:
    """Compute destination node BFS depth relative to root for each edge in `edge_index_td`."""
    edge_index_td = getattr(data, "edge_index_td", getattr(data, "edge_index", None))
    if edge_index_td is None or edge_index_td.numel() == 0:
        return np.zeros((0,), dtype=np.int64)

    src_arr = edge_index_td[0].detach().cpu().numpy().astype(np.int64)
    dst_arr = edge_index_td[1].detach().cpu().numpy().astype(np.int64)
    num_nodes = int(getattr(data, "num_nodes", int(data.x.size(0))))
    root_t = getattr(data, "root_index", None)
    root_idx = int(root_t.view(-1)[0].item()) if root_t is not None and root_t.numel() > 0 else 0

    children: dict[int, list[int]] = {}
    for u, v in zip(src_arr.tolist(), dst_arr.tolist()):
        children.setdefault(u, []).append(v)

    node_depths: dict[int, int] = {root_idx: 0}
    queue = [root_idx]
    head = 0
    while head < len(queue) and len(node_depths) < max(1, num_nodes):
        cur = queue[head]
        head += 1
        d_cur = node_depths[cur]
        for nxt in children.get(cur, ()):
            if nxt not in node_depths:
                node_depths[nxt] = d_cur + 1
                queue.append(nxt)

    edge_depths = np.empty((dst_arr.size,), dtype=np.int64)
    for i, (u, v) in enumerate(zip(src_arr.tolist(), dst_arr.tolist())):
        if v in node_depths:
            edge_depths[i] = node_depths[v]
        elif u == root_idx:
            edge_depths[i] = 1
        else:
            edge_depths[i] = 2
    return edge_depths


def _clone_cascade_with_edge_attr(data: Any, new_edge_attr: torch.Tensor) -> Any:
    """Return a non-mutating copy of `data` with updated `edge_attr` and `cascade_stats[7]`."""
    d_new = data.clone() if callable(getattr(data, "clone", None)) else copy.copy(data)
    d_new.edge_attr = new_edge_attr
    if hasattr(d_new, "edge_stance") and getattr(d_new, "edge_stance", None) is not None:
        if new_edge_attr.size(0) > 0 and new_edge_attr.size(1) >= 1:
            d_new.edge_stance = new_edge_attr[:, 0:1].clone()
        else:
            d_new.edge_stance = new_edge_attr[:, :1].clone()

    raw_stats = getattr(data, "cascade_stats", None)
    if isinstance(raw_stats, torch.Tensor) and raw_stats.numel() >= 8:
        stats_new = raw_stats.clone()
        if new_edge_attr.size(0) > 0 and new_edge_attr.size(1) >= 2:
            mean_s = float(new_edge_attr[:, 0].mean().item())
            mean_h = float(new_edge_attr[:, 1].mean().item())
            one_sided = min(1.0, max(0.0, abs(mean_s) * mean_h))
        else:
            one_sided = 0.0
        stats_new[..., 7] = one_sided
        d_new.cascade_stats = stats_new

    raw_early = getattr(data, "early_y_polar", None)
    edge_index_td = getattr(data, "edge_index_td", getattr(data, "edge_index", None))
    if isinstance(raw_early, torch.Tensor) and edge_index_td is not None:
        num_nodes = int(
            getattr(
                data,
                "num_nodes",
                int(data.x.size(0)) if getattr(data, "x", None) is not None else 0,
            )
        )
        root_t = getattr(data, "root_index", None)
        root_idx = (
            int(root_t.view(-1)[0].item())
            if root_t is not None and root_t.numel() > 0
            else 0
        )
        children_adj: dict[str, list[str]] = {}
        edge_lookup: dict[tuple[str, str], dict[str, str]] = {}
        stance_map: dict[str, float] = {}
        n_edges = int(edge_index_td.size(1)) if edge_index_td.dim() == 2 else 0
        if n_edges > 0 and new_edge_attr.size(0) >= n_edges and new_edge_attr.size(1) >= 1:
            src_arr = edge_index_td[0].detach().cpu().numpy().astype(np.int64)
            dst_arr = edge_index_td[1].detach().cpu().numpy().astype(np.int64)
            for idx_e, (u, v) in enumerate(zip(src_arr.tolist(), dst_arr.tolist())):
                u_s, v_s = str(u), str(v)
                children_adj.setdefault(u_s, []).append(v_s)
                eid = f"{u}->{v}"
                edge_lookup[(u_s, v_s)] = {"edge_id": eid}
                stance_map[eid] = float(new_edge_attr[idx_e, 0].item())
        _, early_ori = _compute_root_aligned_stances_and_orientations(
            root_id=str(root_idx),
            children_adj=children_adj,
            edge_lookup=edge_lookup,
            stance_map=stance_map,
            root_stance_map=None,
        )
        early_stances = [
            float(early_ori.get(str(nid), 0.0))
            for nid in range(num_nodes)
            if nid != root_idx
        ]
        new_ep_val = _compute_full_tree_polarization(early_stances)
        d_new.early_y_polar = torch.full_like(raw_early, float(new_ep_val))
    return d_new


def _summarize_cohort(
    records: Sequence[Mapping[str, float]],
    *,
    seed: int,
    polar_sd: float = 0.0,
) -> dict[str, Any]:
    if not records:
        return {
            "num_cascades": 0,
            "mean_delta_y_size": float("nan"),
            "ci95_delta_y_size": [float("nan"), float("nan")],
            "mean_pct_reply_change": float("nan"),
            "ci95_pct_reply_change": [float("nan"), float("nan")],
            "mean_delta_y_polar": float("nan"),
            "ci95_delta_y_polar": [float("nan"), float("nan")],
            "mean_delta_polar_sd_pct": float("nan"),
        }
    d_sizes = [r["delta_y_size"] for r in records]
    pct_replies = [r["pct_reply_change"] for r in records]
    d_polars = [r["delta_y_polar"] for r in records]
    m_size, ci_size = _bootstrap_mean_ci(d_sizes, seed=seed)
    m_pct, ci_pct = _bootstrap_mean_ci(pct_replies, seed=seed + 1)
    m_pol, ci_pol = _bootstrap_mean_ci(d_polars, seed=seed + 2)
    sd_pct = (
        100.0 * m_pol / polar_sd
        if math.isfinite(m_pol) and math.isfinite(polar_sd) and polar_sd > 1e-8
        else float("nan")
    )
    return {
        "num_cascades": int(len(records)),
        "mean_delta_y_size": m_size,
        "ci95_delta_y_size": ci_size,
        "mean_pct_reply_change": m_pct,
        "ci95_pct_reply_change": ci_pct,
        "mean_delta_y_polar": m_pol,
        "ci95_delta_y_polar": ci_pol,
        "mean_delta_polar_sd_pct": sd_pct,
    }


@torch.no_grad()
def run_mechanism_analysis(
    model: SBiFAGAT,
    cascades: Sequence[Any],
    *,
    dataset_name: str = "dataset",
    device: torch.device | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Empirical Coupling Audit + Counterfactual Moderation Stress Tests on `cascades`."""
    if not cascades:
        raise ValueError("cascades must contain at least one cascade sample")

    target_device = device or next(model.parameters()).device
    model = model.to(target_device)
    model.eval()

    # Part A trackers
    y_sizes: list[float] = []
    n_totals: list[float] = []
    n_futures: list[float] = []
    y_polars: list[float] = []
    mean_hs: list[float] = []
    mean_abs_ss: list[float] = []
    oppose_ratios: list[float] = []
    stance_stds: list[float] = []
    one_sided_hosts: list[float] = []

    regime_bins: dict[str, list[dict[str, float]]] = {
        "low_conflict": [],
        "civil_disagreement": [],
        "hostile_conflict": [],
    }

    trig_depth1 = 0
    trig_depth2plus = 0
    trig_abs_s: list[float] = []
    non_trig_abs_s: list[float] = []
    trig_h: list[float] = []
    non_trig_h: list[float] = []
    trig_gamma: list[float] = []
    non_trig_gamma: list[float] = []

    # Part B per-cascade counterfactual deltas
    intervention_keys = tuple(INTERVENTION_LABELS.keys())
    per_cascade_cf: dict[str, list[dict[str, float]]] = {k: [] for k in intervention_keys}

    for item in cascades:
        y_size = float(item.y_size.view(-1)[0].item())
        y_polar = float(item.y_polar.view(-1)[0].item())
        n_total = max(0.0, float(math.expm1(max(0.0, y_size))))
        n_nodes = int(getattr(item, "num_nodes", int(item.x.size(0))))
        n_early = max(0, n_nodes - 1)
        n_future = max(0.0, n_total - float(n_early))

        raw_ea = getattr(item, "edge_attr", None)
        if isinstance(raw_ea, torch.Tensor) and raw_ea.numel() > 0:
            ea_cpu = raw_ea.detach().cpu().float()
            if ea_cpu.dim() == 1:
                ea_cpu = ea_cpu.unsqueeze(-1)
            s_vec = ea_cpu[:, 0].numpy()
            h_vec = ea_cpu[:, 1].numpy() if ea_cpu.size(1) >= 2 else np.zeros_like(s_vec)
            mean_h = float(np.mean(h_vec))
            mean_abs_s = float(np.mean(np.abs(s_vec)))
            opp_ratio = float(np.mean(s_vec < -0.15))
            s_std = float(np.std(s_vec, ddof=1)) if s_vec.size >= 2 else 0.0
            one_sided = min(1.0, max(0.0, abs(float(np.mean(s_vec))) * mean_h))
        else:
            ea_cpu = torch.empty((0, 2), dtype=torch.float32)
            s_vec = np.zeros((0,), dtype=np.float32)
            h_vec = np.zeros((0,), dtype=np.float32)
            mean_h = 0.0
            mean_abs_s = 0.0
            opp_ratio = 0.0
            s_std = 0.0
            one_sided = 0.0

        y_sizes.append(y_size)
        n_totals.append(n_total)
        n_futures.append(n_future)
        y_polars.append(y_polar)
        mean_hs.append(mean_h)
        mean_abs_ss.append(mean_abs_s)
        oppose_ratios.append(opp_ratio)
        stance_stds.append(s_std)
        one_sided_hosts.append(one_sided)

        if mean_h >= 0.25:
            regime_name = "hostile_conflict"
        elif opp_ratio >= 0.20:
            regime_name = "civil_disagreement"
        else:
            regime_name = "low_conflict"
        regime_bins[regime_name].append(
            {"y_size": y_size, "n_total": n_total, "y_polar": y_polar}
        )

        # Model baseline + edge gates on this cascade
        item_dev = (
            item.clone() if callable(getattr(item, "clone", None)) else copy.copy(item)
        ).to(target_device)
        base_out = model(item_dev)
        base_size = float(base_out["y_size_pred"].view(-1)[0].item())
        base_polar = float(base_out["y_polar_pred"].view(-1)[0].item())
        base_replies = max(1.0, float(math.expm1(max(0.0, base_size))))

        gates = model.extract_edge_gates(item_dev)
        edge_imp = gates["edge_importance"].detach().cpu().numpy().reshape(-1)
        num_edges = int(ea_cpu.size(0))
        if num_edges > 0 and edge_imp.size == num_edges:
            top3_idx = np.argsort(-edge_imp)[: min(3, num_edges)]
            top3_set = set(top3_idx.tolist())
            depths = _resolve_edge_depths(item)
            for idx_e in top3_idx.tolist():
                if idx_e < depths.size and depths[idx_e] <= 1:
                    trig_depth1 += 1
                else:
                    trig_depth2plus += 1
            for idx_e in range(num_edges):
                abs_s_e = float(abs(s_vec[idx_e]))
                h_e = float(h_vec[idx_e])
                g_e = float(edge_imp[idx_e])
                if idx_e in top3_set:
                    trig_abs_s.append(abs_s_e)
                    trig_h.append(h_e)
                    trig_gamma.append(g_e)
                else:
                    non_trig_abs_s.append(abs_s_e)
                    non_trig_h.append(h_e)
                    non_trig_gamma.append(g_e)
        else:
            top3_idx = np.zeros((0,), dtype=np.int64)

        # Counterfactuals on cloned edge_attr (never mutating caller tensors!)
        cf_specs: dict[str, torch.Tensor] = {}
        ea_cool_top3 = ea_cpu.clone()
        if num_edges > 0 and ea_cool_top3.size(1) >= 2 and top3_idx.size > 0:
            ea_cool_top3[torch.as_tensor(top3_idx, dtype=torch.long), 1] = 0.0
        cf_specs["cool_top3_hostility"] = ea_cool_top3

        ea_cool_all = ea_cpu.clone()
        if num_edges > 0 and ea_cool_all.size(1) >= 2:
            ea_cool_all[:, 1] = 0.0
        cf_specs["cool_all_hostility"] = ea_cool_all

        ea_neut_top3 = ea_cpu.clone()
        if num_edges > 0 and top3_idx.size > 0:
            ea_neut_top3[torch.as_tensor(top3_idx, dtype=torch.long), : min(2, ea_neut_top3.size(1))] = 0.0
        cf_specs["neutralize_top3_conflict"] = ea_neut_top3

        ea_neut_opp = ea_cpu.clone()
        if num_edges > 0 and ea_neut_opp.size(1) >= 1:
            opp_mask = ea_neut_opp[:, 0] < 0.0
            if bool(opp_mask.any()):
                ea_neut_opp[opp_mask, : min(2, ea_neut_opp.size(1))] = 0.0
        cf_specs["neutralize_all_oppose"] = ea_neut_opp

        for int_key in intervention_keys:
            d_cf = _clone_cascade_with_edge_attr(item, cf_specs[int_key])
            cf_out = model(d_cf.to(target_device))
            cf_size = float(cf_out["y_size_pred"].view(-1)[0].item())
            cf_polar = float(cf_out["y_polar_pred"].view(-1)[0].item())
            cf_replies = max(0.0, float(math.expm1(max(0.0, cf_size))))
            per_cascade_cf[int_key].append(
                {
                    "y_polar_obs": y_polar,
                    "delta_y_size": cf_size - base_size,
                    "pct_reply_change": 100.0 * (cf_replies - base_replies) / base_replies,
                    "delta_y_polar": cf_polar - base_polar,
                }
            )

    correlations = {
        "early_hostility_vs_y_size": _safe_corr_pair(mean_hs, y_sizes),
        "early_hostility_vs_y_polar": _safe_corr_pair(mean_hs, y_polars),
        "early_abs_stance_vs_y_size": _safe_corr_pair(mean_abs_ss, y_sizes),
        "early_oppose_ratio_vs_y_size": _safe_corr_pair(oppose_ratios, y_sizes),
        "early_oppose_ratio_vs_y_polar": _safe_corr_pair(oppose_ratios, y_polars),
        "early_stance_std_vs_y_polar": _safe_corr_pair(stance_stds, y_polars),
        "one_sided_hostility_vs_y_size": _safe_corr_pair(one_sided_hosts, y_sizes),
        "future_y_polar_vs_y_size": _safe_corr_pair(y_polars, y_sizes),
        "early_hostility_vs_future_replies": _safe_corr_pair(mean_hs, n_futures),
    }

    regimes: dict[str, dict[str, Any]] = {}
    for r_name, items in regime_bins.items():
        if items:
            regimes[r_name] = {
                "count": int(len(items)),
                "mean_y_size": float(np.mean([x["y_size"] for x in items])),
                "median_n_total": float(np.median([x["n_total"] for x in items])),
                "mean_y_polar": float(np.mean([x["y_polar"] for x in items])),
            }
        else:
            regimes[r_name] = {
                "count": 0,
                "mean_y_size": float("nan"),
                "median_n_total": float("nan"),
                "mean_y_polar": float("nan"),
            }

    total_trig = trig_depth1 + trig_depth2plus
    trigger_profile = {
        "num_trigger_edges": int(total_trig),
        "depth1_fraction": float(trig_depth1) / float(total_trig) if total_trig > 0 else float("nan"),
        "depth2plus_fraction": float(trig_depth2plus) / float(total_trig) if total_trig > 0 else float("nan"),
        "top3_mean_abs_s": float(np.mean(trig_abs_s)) if trig_abs_s else float("nan"),
        "non_top3_mean_abs_s": float(np.mean(non_trig_abs_s)) if non_trig_abs_s else float("nan"),
        "top3_mean_h": float(np.mean(trig_h)) if trig_h else float("nan"),
        "non_top3_mean_h": float(np.mean(non_trig_h)) if non_trig_h else float("nan"),
        "top3_mean_abs_gamma": float(np.mean(trig_gamma)) if trig_gamma else float("nan"),
        "non_top3_mean_abs_gamma": float(np.mean(non_trig_gamma)) if non_trig_gamma else float("nan"),
    }

    polar_arr = np.asarray(y_polars, dtype=np.float64)
    obs_polar_sd = float(np.std(polar_arr, ddof=1)) if polar_arr.size >= 2 else 0.0
    q75_polar = float(np.quantile(polar_arr, 0.75))
    interventions: dict[str, Any] = {}
    for offset, int_key in enumerate(intervention_keys):
        recs = per_cascade_cf[int_key]
        high_recs = [r for r in recs if r["y_polar_obs"] >= q75_polar]
        interventions[int_key] = {
            "label": INTERVENTION_LABELS[int_key],
            "all_test": _summarize_cohort(recs, seed=seed + offset * 10, polar_sd=obs_polar_sd),
            "high_polar_q75": _summarize_cohort(high_recs, seed=seed + 100 + offset * 10, polar_sd=obs_polar_sd),
        }

    report: dict[str, Any] = {
        "dataset": dataset_name,
        "analysis_scope": "observational_edge_feature_sensitivity",
        "num_test_cascades": int(len(cascades)),
        "observed_y_polar_sd": obs_polar_sd,
        "q75_observed_y_polar": q75_polar,
        "empirical_coupling": correlations,
        "discourse_regimes": regimes,
        "trigger_depth_profile": trigger_profile,
        "interventions": interventions,
    }
    return _sanitize_json_value(report)


def _fmt_val_ci(mean_v: Any, ci_v: Any, *, decimals: int = 4) -> str:
    if mean_v is None or not math.isfinite(float(mean_v)):
        return "--"
    m = float(mean_v)
    if (
        isinstance(ci_v, (list, tuple))
        and len(ci_v) == 2
        and ci_v[0] is not None
        and ci_v[1] is not None
        and math.isfinite(float(ci_v[0]))
        and math.isfinite(float(ci_v[1]))
    ):
        return f"${m:+.{decimals}f}$ $[{float(ci_v[0]):+.{decimals}f}, {float(ci_v[1]):+.{decimals}f}]$"
    return f"${m:+.{decimals}f}$"


def format_mechanisms_latex(
    report: Mapping[str, Any],
    *,
    caption: str | None = None,
    label: str = "tab:mechanisms",
) -> str:
    """Format Counterfactual Moderation Stress Test results as a valid IEEEtran booktabs table."""
    ds = str(report.get("dataset") or "dataset").upper().replace("_", "\\_")
    cap = caption or (
        f"Observational Edge-Feature Sensitivity Analysis on {ds} Test Cascades "
        "(Non-Causal Input Perturbation Mean Shift with 95\\% Paired Bootstrap CI)."
    )
    interventions = report.get("interventions", {})
    rows: list[str] = []
    for int_key, int_blob in interventions.items():
        disp = str(int_blob.get("label") or int_key).replace("_", "\\_")
        all_c = int_blob.get("all_test", {})
        hi_c = int_blob.get("high_polar_q75", {})
        cells = [
            disp,
            _fmt_val_ci(all_c.get("mean_delta_y_size"), all_c.get("ci95_delta_y_size")),
            _fmt_val_ci(all_c.get("mean_delta_y_polar"), all_c.get("ci95_delta_y_polar")),
            _fmt_val_ci(hi_c.get("mean_delta_y_size"), hi_c.get("ci95_delta_y_size")),
            _fmt_val_ci(hi_c.get("mean_delta_y_polar"), hi_c.get("ci95_delta_y_polar")),
        ]
        rows.append(" & ".join(cells) + r" \\")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{cap}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Edge-Feature Perturbation & All Test $\Delta \hat{y}_{\mathrm{size}}$ & All Test $\Delta \hat{y}_{\mathrm{polar}}$ & Q75+ $\Delta \hat{y}_{\mathrm{size}}$ & Q75+ $\Delta \hat{y}_{\mathrm{polar}}$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Run Empirical Coupling & Counterfactual Moderation Stress Tests on trained SBi-FAGAT."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--min-total-replies", type=int, default=DEFAULT_FORECAST_MIN_REPLIES)
    parser.add_argument(
        "--polar-target-mode", choices=["full", "future_only"], default="future_only"
    )
    parser.add_argument("--label-scheme", choices=["veracity4", "binary"], default="veracity4")
    parser.add_argument("--edge-source", choices=["auto", "student", "teacher"], default="auto")
    parser.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-table", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds_tag = args.data_dir.name.lower() if args.data_dir.name.lower() in {"pheme", "vn"} else ""
    suffix = f"_{ds_tag}" if ds_tag else ""
    out_json = args.output_json or Path(f"paper/results_mechanisms{suffix}.json")
    out_table = args.output_table or Path(f"paper/tables_mechanisms{suffix}.tex")
    fig_dir = args.figures_dir or (Path("paper/figures") / ds_tag if ds_tag else Path("paper/figures"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_sbifagat_checkpoint(args.ckpt, device=device)
    dataset = EarlyCascadeDataset(
        root_dir=args.data_dir,
        k_horizon=args.k_horizon,
        split=args.split,
        min_total_replies=args.min_total_replies,
        polar_target_mode=args.polar_target_mode,
        label_scheme=args.label_scheme,
        edge_source=args.edge_source,
    )
    report = run_mechanism_analysis(
        model,
        list(dataset),
        dataset_name=ds_tag or args.data_dir.name,
        device=device,
        seed=args.seed,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    tbl_label = f"tab:mechanisms{suffix}"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_table.write_text(
        format_mechanisms_latex(report, label=tbl_label),
        encoding="utf-8",
    )

    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_counterfactual_interventions(
        report["interventions"],
        output_dir=fig_dir,
        stem="fig4_counterfactual_interventions",
    )
    print(f"[AnalyzeMechanisms] Saved JSON -> {out_json}, Table -> {out_table}, Figures -> {fig_dir}")


if __name__ == "__main__":
    main()

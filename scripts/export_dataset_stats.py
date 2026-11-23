"""Dataset Exploratory Data Analysis (Table 1) & Early-K Persistence Baseline.

Computes exact leakage-free structural, temporal, topic-split, and label statistics
from validated `nodes.parquet` / `edges.parquet` tables, plus the non-learned
Early-K Persistence Baseline (comparing y_polar(G_K) against future-only y_polar
when `nlp_features.pt` is available). Exports IEEEtran booktabs LaTeX tables and
JSON summaries without fabricating missing metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.pyg_dataset import (
    DEFAULT_FORECAST_MIN_REPLIES,
    EarlyCascadeDataset,
    _compute_full_tree_polarization,
    _compute_root_aligned_stances,
    compute_canonical_topic_splits,
    load_topic_splits_json,
)
from src.data.schema import load_cascade_parquet
from src.training.metrics import evaluate_predictions
from src.training.reporting import _sanitize_json_value


def compute_early_k_persistence_metrics(
    test_cascades: Sequence[Any],
    *,
    lambda_polar: float = 0.5,
    train_cascades: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a strictly causal Early-K Persistence Baseline on test cascades.

    Predicts future polarization solely by evaluating the exact polarization metric
    on the observed early tree G_K (first K chronological interactions, using causal
    root-aligned orientations on V_K). Predicts log-size via the Train-split mean
    when `train_cascades` is provided (avoiding the constant `ln(1+K)` strawman),
    falling back to `ln(1 + |V_K| - 1)` when invoked standalone without a train split.
    """
    if not test_cascades:
        return {
            "num_cascades": 0,
            "msle": float("nan"),
            "mae": float("nan"),
            "rmse_size": float("nan"),
            "r2_size": float("nan"),
            "spearman_size": float("nan"),
            "rmse": float("nan"),
            "mae_polar": float("nan"),
            "r2_polar": float("nan"),
            "pearson_r": float("nan"),
            "spearman_polar": float("nan"),
            "mean_true_polar": float("nan"),
            "mean_early_polar": float("nan"),
        }

    train_mean_size: float | None = None
    if train_cascades:
        tr_sizes = [float(c.y_size.view(-1)[0].item()) for c in train_cascades if hasattr(c, "y_size")]
        if tr_sizes:
            train_mean_size = float(np.mean(tr_sizes))

    y_size_trues: list[float] = []
    y_size_preds: list[float] = []
    y_polar_trues: list[float] = []
    y_polar_preds: list[float] = []

    for d in test_cascades:
        y_size_trues.append(float(d.y_size.view(-1)[0].item()))
        y_polar_trues.append(float(d.y_polar.view(-1)[0].item()))

        if train_mean_size is not None:
            y_size_preds.append(train_mean_size)
        else:
            num_nodes = int(getattr(d, "num_nodes", int(d.x.size(0))))
            y_size_preds.append(math.log1p(max(0, num_nodes - 1)))

        early_yp_attr = getattr(d, "early_y_polar", None)
        if early_yp_attr is not None and early_yp_attr.numel() >= 1:
            y_polar_preds.append(float(early_yp_attr.view(-1)[0].item()))
            continue

        early_rs = getattr(d, "early_node_root_stance", None)
        if early_rs is not None and early_rs.numel() > 1:
            aligned = [float(v) for v in early_rs.view(-1)[1:].detach().cpu().tolist()]
        else:
            edge_index_td = getattr(d, "edge_index_td", getattr(d, "edge_index", None))
            raw_attr = getattr(d, "edge_attr", None)
            if raw_attr is None:
                raw_attr = getattr(d, "edge_stance", None)

            root_tensor = getattr(d, "root_index", None)
            root_id = str(int(root_tensor.view(-1)[0].item())) if root_tensor is not None else "0"

            children_adj: dict[str, list[str]] = {}
            stance_map: dict[tuple[str, str], float] = {}
            if edge_index_td is not None and edge_index_td.numel() > 0:
                num_edges = int(edge_index_td.size(1))
                for idx in range(num_edges):
                    u_str = str(int(edge_index_td[0, idx].item()))
                    v_str = str(int(edge_index_td[1, idx].item()))
                    children_adj.setdefault(u_str, []).append(v_str)
                    if raw_attr is not None and raw_attr.numel() > 0:
                        s_uv = float(raw_attr[idx, 0].item()) if raw_attr.dim() == 2 else float(raw_attr[idx].item())
                    else:
                        s_uv = 0.0
                    stance_map[(u_str, v_str)] = float(min(1.0, max(-1.0, s_uv)))

            aligned = _compute_root_aligned_stances(
                root_id=root_id,
                children_adj=children_adj,
                edge_lookup={},
                stance_map=stance_map,
            )
        y_polar_preds.append(float(_compute_full_tree_polarization(aligned, lambda_polar=lambda_polar)))

    metrics = evaluate_predictions(
        np.asarray(y_size_trues, dtype=np.float64),
        np.asarray(y_size_preds, dtype=np.float64),
        np.asarray(y_polar_trues, dtype=np.float64),
        np.asarray(y_polar_preds, dtype=np.float64),
    )
    return {
        "num_cascades": len(test_cascades),
        "msle": float(metrics["msle"]),
        "mae": float(metrics["mae"]),
        "rmse_size": float(metrics["rmse_size"]),
        "r2_size": float(metrics["r2_size"]),
        "spearman_size": float(metrics["spearman_size"]),
        "rmse": float(metrics["rmse"]),
        "mae_polar": float(metrics["mae_polar"]),
        "r2_polar": float(metrics["r2_polar"]),
        "pearson_r": float(metrics["pearson_r"]),
        "spearman_polar": float(metrics["spearman_polar"]),
        "mean_true_polar": float(np.mean(y_polar_trues)),
        "mean_early_polar": float(np.mean(y_polar_preds)),
    }


def compute_dataset_statistics(
    data_dir: str | Path,
    *,
    name: str | None = None,
    k_horizon: int = 20,
    min_total_replies: int = DEFAULT_FORECAST_MIN_REPLIES,
    polar_target_mode: str = "future_only",
    edge_dim: int | None = None,
) -> dict[str, Any]:
    """Compute exact corpus EDA statistics and optional Persistence Baseline metrics."""
    root = Path(data_dir)
    nodes_table, edges_table = load_cascade_parquet(root, validate=True)

    node_rows = nodes_table.to_pylist()
    nodes_by_cid: dict[str, list[dict[str, Any]]] = {}
    langs: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for row in node_rows:
        nodes_by_cid.setdefault(str(row["cascade_id"]), []).append(row)
        langs[str(row["lang"])] += 1
        sources[str(row["source_dataset"])] += 1

    raw_reply_counts: list[int] = []
    raw_duration_hours: list[float] = []
    eligible_reply_counts: list[int] = []
    eligible_duration_hours: list[float] = []
    topic_ids_raw: set[str] = set()
    topic_ids_eligible: set[str] = set()
    rumor_counts: Counter[str] = Counter()
    eligible_rumor_counts: Counter[str] = Counter()

    for cid, rows in nodes_by_cid.items():
        n_replies = max(0, len(rows) - 1)
        raw_reply_counts.append(n_replies)
        ts_list = [r["created_at_utc"] for r in rows]
        span_h = max(0.0, (max(ts_list) - min(ts_list)).total_seconds() / 3600.0)
        raw_duration_hours.append(span_h)

        roots = [r for r in rows if bool(r["is_root"])]
        root_row = roots[0] if roots else rows[0]
        tid = str(root_row["topic_id"])
        topic_ids_raw.add(tid)
        rlab = root_row.get("rumor_label")
        if rlab is not None:
            rumor_counts[str(rlab)] += 1
        if n_replies >= int(min_total_replies):
            topic_ids_eligible.add(tid)
            eligible_reply_counts.append(n_replies)
            eligible_duration_hours.append(span_h)
            if rlab is not None:
                eligible_rumor_counts[str(rlab)] += 1

    eligible_cids = {
        cid for cid, rows in nodes_by_cid.items() if max(0, len(rows) - 1) >= int(min_total_replies)
    }

    splits_path = root / "topic_splits.json"
    if splits_path.exists():
        topic_splits = load_topic_splits_json(splits_path)
    else:
        topic_splits = compute_canonical_topic_splits(
            nodes_table,
            min_total_replies=min_total_replies,
        )

    split_cascade_counts = {"train": 0, "val": 0, "test": 0}
    for cid in eligible_cids:
        roots = [r for r in nodes_by_cid[cid] if bool(r["is_root"])]
        tid = str((roots[0] if roots else nodes_by_cid[cid][0])["topic_id"])
        for sp in ("train", "val", "test"):
            if tid in topic_splits[sp]:
                split_cascade_counts[sp] += 1
                break

    arr_raw_replies = np.asarray(raw_reply_counts, dtype=np.float64)
    arr_raw_dur = np.asarray(raw_duration_hours, dtype=np.float64)
    active_replies = eligible_reply_counts if eligible_reply_counts else raw_reply_counts
    active_dur = eligible_duration_hours if eligible_duration_hours else raw_duration_hours
    arr_replies = np.asarray(active_replies, dtype=np.float64)
    arr_dur = np.asarray(active_dur, dtype=np.float64)

    summary: dict[str, Any] = {
        "name": name or root.name,
        "data_dir": str(root),
        "k_horizon": int(k_horizon),
        "min_total_replies": int(min_total_replies),
        "polar_target_mode": polar_target_mode,
        "total_nodes": int(nodes_table.num_rows),
        "total_edges": int(edges_table.num_rows),
        "raw_cascades": len(nodes_by_cid),
        "eligible_cascades": len(eligible_cids),
        "raw_topics": len(topic_ids_raw),
        "eligible_topics": len(topic_ids_eligible),
        "median_replies": float(np.median(arr_replies)) if arr_replies.size > 0 else 0.0,
        "mean_replies": float(np.mean(arr_replies)) if arr_replies.size > 0 else 0.0,
        "p90_replies": float(np.percentile(arr_replies, 90)) if arr_replies.size > 0 else 0.0,
        "median_duration_hours": float(np.median(arr_dur)) if arr_dur.size > 0 else 0.0,
        "mean_duration_hours": float(np.mean(arr_dur)) if arr_dur.size > 0 else 0.0,
        "raw_median_replies": float(np.median(arr_raw_replies)) if arr_raw_replies.size > 0 else 0.0,
        "raw_mean_replies": float(np.mean(arr_raw_replies)) if arr_raw_replies.size > 0 else 0.0,
        "raw_median_duration_hours": float(np.median(arr_raw_dur)) if arr_raw_dur.size > 0 else 0.0,
        "languages": dict(langs),
        "source_datasets": dict(sources),
        "rumor_label_counts": dict(eligible_rumor_counts if eligible_rumor_counts else rumor_counts),
        "raw_rumor_label_counts": dict(rumor_counts),
        "split_cascade_counts": split_cascade_counts,
        "topic_splits_source": str(splits_path) if splits_path.exists() else "computed_canonical",
        "early_k_persistence": None,
        "target_stats": None,
    }

    nlp_path = root / "nlp_features.pt"
    if nlp_path.exists():
        split_datasets = {
            sp: EarlyCascadeDataset(
                root,
                k_horizon=k_horizon,
                split=sp,
                min_total_replies=min_total_replies,
                polar_target_mode=polar_target_mode,
            )
            for sp in ("train", "val", "test")
        }
        summary["split_cascade_counts"] = {sp: len(ds) for sp, ds in split_datasets.items()}
        all_items = [item for ds in split_datasets.values() for item in ds]
        if all_items:
            y_sizes = [float(it.y_size.view(-1)[0].item()) for it in all_items]
            y_polars = [float(it.y_polar.view(-1)[0].item()) for it in all_items]
            y_bcs = [float(it.y_bimodality.view(-1)[0].item()) for it in all_items]
            y_assorts = [
                float(it.y_assortativity.view(-1)[0].item())
                for it in all_items
                if hasattr(it, "y_assortativity")
            ]
            summary["target_stats"] = {
                "num_cascades": len(all_items),
                "y_size_mean": float(np.mean(y_sizes)),
                "y_size_std": float(np.std(y_sizes)),
                "y_polar_mean": float(np.mean(y_polars)),
                "y_polar_std": float(np.std(y_polars)),
                "y_bimodality_mean": float(np.mean(y_bcs)),
                "y_assortativity_mean": float(np.mean(y_assorts)) if y_assorts else 0.0,
            }
        summary["early_k_persistence"] = compute_early_k_persistence_metrics(
            list(split_datasets["test"]),
            train_cascades=list(split_datasets["train"]),
        )

    return summary


def format_dataset_stats_latex(
    stats_list: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    caption: str = (
        "Exploratory dataset statistics ($K=20$, $N_{\\text{reply}} \\ge 25$ forecasting filter) "
        "and Early-$K$ Persistence Baseline on Test splits."
    ),
    label: str = "tab:dataset_eda",
) -> str:
    """Render dataset EDA statistics as an IEEEtran `booktabs` LaTeX table."""
    items = [stats_list] if isinstance(stats_list, Mapping) else list(stats_list)
    rows: list[str] = []
    for item in items:
        ds_name = str(item.get("name", "dataset")).replace("_", "\\_")
        raw_c = int(item.get("raw_cascades", 0))
        elig_c = int(item.get("eligible_cascades", 0))
        topics = int(item.get("eligible_topics", item.get("raw_topics", 0)))
        sp = item.get("split_cascade_counts", {})
        split_str = f"{int(sp.get('train', 0))}/{int(sp.get('val', 0))}/{int(sp.get('test', 0))}"
        med_rep = float(item.get("median_replies", 0.0))
        mean_rep = float(item.get("mean_replies", 0.0))
        med_dur = float(item.get("median_duration_hours", 0.0))
        persist = item.get("early_k_persistence")
        if (
            isinstance(persist, Mapping)
            and int(persist.get("num_cascades", 0) or 0) > 0
            and persist.get("rmse") is not None
            and persist.get("pearson_r") is not None
        ):
            rmse_val = float(persist["rmse"])
            r_val = float(persist["pearson_r"])
            p_rmse = f"{rmse_val:.4f}" if math.isfinite(rmse_val) else "--"
            p_r = f"{r_val:.4f}" if math.isfinite(r_val) else "--"
        else:
            p_rmse = "--"
            p_r = "--"
        rows.append(
            f"{ds_name} & {raw_c:,} & {elig_c:,} & {topics:,} & {split_str} & "
            f"{med_rep:.1f} ({mean_rep:.1f}) & {med_dur:.2f} & {p_rmse} & {p_r} \\\\"
        )

    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\resizebox{\\textwidth}{!}{",
            "\\begin{tabular}{lcccccccc}",
            "\\toprule",
            "Dataset & Raw & Forecast ($\\ge 25$) & Topics & Train/Val/Test & Med (Mean) Replies & Med Span (h) & Persist RMSE$\\downarrow$ & Persist $r$$\\uparrow$ \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table*}",
            "",
        ]
    )


def format_dataset_stats_markdown(stats_list: Sequence[Mapping[str, Any]]) -> str:
    """Render concise Markdown summary table for CLI inspection."""
    lines = [
        "| Dataset | Raw Cascades | Forecast (>=25) | Topics | Train/Val/Test | Med (Mean) Replies | Med Span (h) | Persist RMSE | Persist r |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for item in stats_list:
        sp = item.get("split_cascade_counts", {})
        split_str = f"{int(sp.get('train', 0))}/{int(sp.get('val', 0))}/{int(sp.get('test', 0))}"
        persist = item.get("early_k_persistence")
        if (
            isinstance(persist, Mapping)
            and int(persist.get("num_cascades", 0) or 0) > 0
            and persist.get("rmse") is not None
            and persist.get("pearson_r") is not None
        ):
            rmse_val = float(persist["rmse"])
            r_val = float(persist["pearson_r"])
            p_rmse = f"{rmse_val:.4f}" if math.isfinite(rmse_val) else "--"
            p_r = f"{r_val:.4f}" if math.isfinite(r_val) else "--"
        else:
            p_rmse = "--"
            p_r = "--"
        lines.append(
            f"| {item.get('name')} | {int(item.get('raw_cascades', 0)):,} | "
            f"{int(item.get('eligible_cascades', 0)):,} | {int(item.get('eligible_topics', 0)):,} | "
            f"{split_str} | {float(item.get('median_replies', 0.0)):.1f} ({float(item.get('mean_replies', 0.0)):.1f}) | "
            f"{float(item.get('median_duration_hours', 0.0)):.2f} | {p_rmse} | {p_r} |"
        )
    return "\n".join(lines)


def _resolve_eda_table_label(
    output_table: Path, stats_list: Sequence[Mapping[str, Any]]
) -> str:
    stem = output_table.stem.lower()
    if stem.startswith("tables_dataset_eda_") and len(stem) > len("tables_dataset_eda_"):
        suffix = stem.removeprefix("tables_dataset_eda_")
        return f"tab:dataset_eda_{suffix}"
    if len(stats_list) == 1 and stem != "tables_dataset_eda":
        ds_name = str(stats_list[0].get("name", "")).strip().lower()
        if ds_name in {"pheme", "vn"}:
            return f"tab:dataset_eda_{ds_name}"
    return "tab:dataset_eda"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Export Table 1 Dataset EDA & Early-K Persistence Baseline."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        nargs="+",
        default=[Path("data/processed")],
        help="One or more directories containing validated nodes.parquet and edges.parquet.",
    )
    parser.add_argument("--names", type=str, nargs="*", default=None)
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--min-total-replies", type=int, default=DEFAULT_FORECAST_MIN_REPLIES)
    parser.add_argument(
        "--polar-target-mode", choices=["future_only", "full"], default="future_only"
    )
    parser.add_argument("--edge-dim", type=int, choices=[2, 4], default=None)
    parser.add_argument(
        "--output-table",
        type=Path,
        default=Path("paper/tables_dataset_eda.tex"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("paper/results_dataset_eda.json"),
    )
    args = parser.parse_args()

    stats_list: list[dict[str, Any]] = []
    for idx, ddir in enumerate(args.data_dir):
        ds_name = (
            args.names[idx]
            if args.names is not None and idx < len(args.names)
            else ddir.name
        )
        stats_list.append(
            compute_dataset_statistics(
                ddir,
                name=ds_name,
                k_horizon=args.k_horizon,
                min_total_replies=args.min_total_replies,
                polar_target_mode=args.polar_target_mode,
                edge_dim=args.edge_dim,
            )
        )

    print("\n=== Dataset EDA & Early-K Persistence Baseline ===")
    print(format_dataset_stats_markdown(stats_list))

    if args.output_table is not None:
        args.output_table.parent.mkdir(parents=True, exist_ok=True)
        eda_label = _resolve_eda_table_label(args.output_table, stats_list)
        args.output_table.write_text(
            format_dataset_stats_latex(stats_list, label=eda_label), encoding="utf-8"
        )
        print(f"Saved LaTeX table -> {args.output_table}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                _sanitize_json_value({"datasets": stats_list}),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        print(f"Saved JSON stats -> {args.output_json}")


if __name__ == "__main__":
    main()

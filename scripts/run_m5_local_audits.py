"""M5 Local Real-Data Quantitative Audits & Fast 5-Seed Tabular Benchmarks.

Executes CPU/GPU-lightweight M5 benchmarks and quantitative invariants strictly on real
on-disk data (`data/processed/{vn,pheme}` and `data/interim/*`) and writes outputs into
`results/` without touching `paper/`, `reports/`, or `docs/` (R4 protection).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_dataset_stats import compute_early_k_persistence_metrics
from scripts.run_experiments import _build_loaders, _build_persistence_summary_row, attach_early_k_persistence_to_result
from src.data.arctic_shift_vn import compare_prune_vs_contract_on_threads
from src.data.pyg_dataset import (
    EarlyCascadeDataset,
    compute_embedding_cluster_splits,
    compute_temporal_topic_splits,
    leave_one_event_out_splits,
)
from src.data.schema import load_cascade_parquet
from src.models.sota_cascades import (
    CalibratedEarlyKPersistence,
    LightGBMBaseline,
    RidgeBaseline,
)
from src.training.reporting import MAIN_METRICS, VN_MAIN_METRICS, serialize_benchmark_json
from src.training.trainer import (
    compute_paired_bootstrap_pvalue,
    compute_paired_ttest,
    holm_bonferroni_adjust,
    run_multi_seed_benchmark,
)
from src.utils.seed import OFFICIAL_SEEDS


def _assert_finite_json_obj(obj: Any, path_prefix: str = "$") -> None:
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float {obj!r} at {path_prefix}")
        return
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            _assert_finite_json_obj(v, f"{path_prefix}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            _assert_finite_json_obj(item, f"{path_prefix}[{idx}]")
        return


def _to_utc_ts(val: Any) -> float:
    if isinstance(val, datetime):
        dt = val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    if hasattr(val, "to_pydatetime"):
        dt = val.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    return float(val)


def reconstruct_vn_threads_from_parquet(data_dir: Path) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Reconstruct `(post, comments)` thread tuples from `data/processed/vn/nodes.parquet`."""
    nodes_tbl, _ = load_cascade_parquet(data_dir, validate=True)
    rows = nodes_tbl.to_pylist()
    by_cascade: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cid = str(row["cascade_id"])
        by_cascade.setdefault(cid, []).append(row)

    threads: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for cid in sorted(by_cascade.keys()):
        c_rows = by_cascade[cid]
        root_row = next((r for r in c_rows if bool(r.get("is_root"))), None)
        if root_row is None:
            continue
        post_id = cid[3:] if cid.startswith("vn_") else cid
        src_ds = str(root_row.get("source_dataset") or "arctic_shift_unknown")
        subreddit = src_ds[len("arctic_shift_") :] if src_ds.startswith("arctic_shift_") else src_ds
        post = {
            "id": post_id,
            "name": f"t3_{post_id}",
            "title": str(root_row.get("text") or ""),
            "selftext": "",
            "created_utc": _to_utc_ts(root_row["created_at_utc"]),
            "author": str(root_row.get("user_id_hash") or "u_root"),
            "subreddit": subreddit,
        }
        prefix = f"{cid}_"
        comments: list[dict[str, Any]] = []
        for r in c_rows:
            if bool(r.get("is_root")):
                continue
            nid = str(r["node_id"])
            fn = nid[len(prefix) :] if nid.startswith(prefix) else nid
            if not fn.startswith("t1_"):
                fn = f"t1_{fn}"
            pid_raw = str(r.get("parent_id") or "")
            p_fn = pid_raw[len(prefix) :] if pid_raw.startswith(prefix) else pid_raw
            if not p_fn.startswith(("t3_", "t1_")):
                p_fn = f"t3_{post_id}"
            comments.append(
                {
                    "fullname": fn,
                    "link_id": f"t3_{post_id}",
                    "parent_id": p_fn,
                    "text": str(r.get("text") or ""),
                    "created_utc": _to_utc_ts(r["created_at_utc"]),
                    "author_hash": r.get("user_id_hash"),
                }
            )
        threads.append((post, comments))
    return threads


def run_vn_contraction_eval(
    *,
    data_dir: Path = Path("data/processed/vn"),
    output_json: Path = Path("results/contraction_eval_vn.json"),
) -> dict[str, Any]:
    """Run `compare_prune_vs_contract_on_threads` on reconstructed 1,500 VN cascades."""
    t0 = time.perf_counter()
    threads = reconstruct_vn_threads_from_parquet(data_dir)
    res = compare_prune_vs_contract_on_threads(
        threads,
        min_comments=25,
        min_depth=3,
        min_vi_ratio=0.65,
        k_horizon=20,
    )
    elapsed = time.perf_counter() - t0
    payload = {
        **res,
        "dataset": "vn",
        "source_parquet": str((data_dir / "nodes.parquet").as_posix()),
        "elapsed_sec": round(elapsed, 3),
        "is_real_data": True,
        "post_crawl_parquet_caveat": (
            "Raw pre-filter Arctic Shift crawl_checkpoint.jsonl is not stored on disk; "
            "this evaluation reconstructs threads from post-crawl data/processed/vn/nodes.parquet. "
            "Consequently, [deleted]/[removed]/AutoModerator/spam nodes were already pruned at crawl time, "
            "leaving short filler comments (evaluated here via 34 contracted bridge nodes / 35 relinked edges) "
            "and yielding total_stance_inversion_risk_edges=0 on the post-crawl parquet."
        ),
    }
    _assert_finite_json_obj(payload)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"[Contraction] Saved {output_json} ({elapsed:.2f}s): {json.dumps(res)}", flush=True)
    return payload


def _tabular_factories(num_cls: int) -> dict[str, Any]:
    return {
        "Calibrated Early-K Persistence": lambda: CalibratedEarlyKPersistence(num_rumor_classes=num_cls),
        "Ridge": lambda: RidgeBaseline(num_rumor_classes=num_cls),
        "LightGBM": lambda: LightGBMBaseline(num_rumor_classes=num_cls),
    }


def _run_tabular_inductive(
    *,
    data_dir: Path,
    ds_name: str,
    split_mode: str,
    k_horizon: int,
    min_total_replies: int,
    edge_source: str,
    output_json: Path,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    nodes_tbl, edges_tbl = load_cascade_parquet(data_dir, validate=True)
    nlp_feats = torch.load(data_dir / "nlp_features.pt", map_location="cpu", weights_only=False)

    custom_splits: dict[str, set[str]] | None = None
    if split_mode != "canonical":
        all_ds = EarlyCascadeDataset(
            root_dir=data_dir,
            k_horizon=k_horizon,
            split="all",
            nodes_table=nodes_tbl,
            edges_table=edges_tbl,
            nlp_features=nlp_feats,
            min_total_replies=min_total_replies,
            polar_target_mode="future_only",
            label_scheme="veracity4",
            edge_source=edge_source,
        )
        if split_mode == "temporal":
            custom_splits = compute_temporal_topic_splits(all_ds._all_cascades)
        elif split_mode == "embedding_cluster":
            custom_splits = compute_embedding_cluster_splits(all_ds._all_cascades, seed=OFFICIAL_SEEDS[0])
        else:
            raise ValueError(f"Unsupported split_mode={split_mode!r}")

    test_ds, train_loader, val_loader, test_loader = _build_loaders(
        data_dir,
        k_horizon,
        batch_size=32,
        min_total_replies=min_total_replies,
        polar_target_mode="future_only",
        label_scheme="veracity4",
        custom_topic_splits=custom_splits,
        edge_source=edge_source,
        include_tier2_prior=False,
        nodes_table=nodes_tbl,
        edges_table=edges_tbl,
        nlp_features=nlp_feats,
    )
    has_rumor = any(int(d.y_rumor.item()) >= 0 for d in test_ds)
    active_metrics = MAIN_METRICS if has_rumor else VN_MAIN_METRICS
    active_keys = tuple(k for k, _, _ in active_metrics)
    factories = _tabular_factories(num_cls=4)
    lgb_backend = LightGBMBaseline(num_rumor_classes=4).backend_name

    result = run_multi_seed_benchmark(
        factories,
        train_loader,
        val_loader,
        test_loader,
        seeds=OFFICIAL_SEEDS,
        epochs=1,
        patience=1,
        reference_model="LightGBM",
        comparator_model="Ridge",
        primary_hypothesis_keys=active_keys,
        device=torch.device("cpu"),
    )
    attach_early_k_persistence_to_result(
        result,
        list(test_ds),
        train_cascades=list(train_loader.dataset),
    )
    elapsed = time.perf_counter() - t0
    serialize_benchmark_json(
        result,
        output_json,
        extra={
            "dataset": ds_name,
            "k_horizon": k_horizon,
            "eval_protocol": "inductive",
            "split_mode": split_mode,
            "edge_source": edge_source,
            "polar_target_mode": "future_only",
            "min_total_replies": min_total_replies,
            "lightgbm_backend": lgb_backend,
            "model_subset": "tabular_and_calibrated_persistence_only",
            "is_real_data": True,
            "elapsed_sec": round(elapsed, 3),
            "counts": {
                "train": len(train_loader.dataset),
                "val": len(val_loader.dataset),
                "test": len(test_ds),
            },
            "early_k_persistence": result.get("early_k_persistence"),
        },
    )
    print(f"[Tier1-Tabular:{ds_name}/{split_mode}] Saved {output_json} ({elapsed:.2f}s)", flush=True)
    return result


def _run_tabular_pheme_loo(
    *,
    data_dir: Path,
    k_horizon: int = 10,
    min_total_replies: int = 15,
    output_json: Path = Path("results/tier1_tabular_loo_pheme.json"),
) -> dict[str, Any]:
    t0 = time.perf_counter()
    nodes_tbl, edges_tbl = load_cascade_parquet(data_dir, validate=True)
    nlp_feats = torch.load(data_dir / "nlp_features.pt", map_location="cpu", weights_only=False)
    base_all = EarlyCascadeDataset(
        root_dir=data_dir,
        k_horizon=k_horizon,
        split="all",
        nodes_table=nodes_tbl,
        edges_table=edges_tbl,
        nlp_features=nlp_feats,
        min_total_replies=min_total_replies,
        polar_target_mode="future_only",
        label_scheme="veracity4",
        edge_source="teacher",
    )
    folds = leave_one_event_out_splits(base_all._all_cascades, min_cascades_per_fold=15)
    active_keys = tuple(k for k, _, _ in MAIN_METRICS)
    factories = _tabular_factories(num_cls=4)
    lgb_backend = LightGBMBaseline(num_rumor_classes=4).backend_name

    fold_raw: dict[str, list[dict[str, float]]] = {m: [] for m in factories}
    fold_val_raw: dict[str, list[dict[str, float]]] = {m: [] for m in factories}
    persist_folds: list[dict[str, Any]] = []
    for fold in folds:
        t_ds, tr_l, va_l, te_l = _build_loaders(
            data_dir,
            k_horizon,
            batch_size=32,
            min_total_replies=min_total_replies,
            polar_target_mode="future_only",
            label_scheme="veracity4",
            custom_topic_splits=fold,
            edge_source="teacher",
            include_tier2_prior=False,
            nodes_table=nodes_tbl,
            edges_table=edges_tbl,
            nlp_features=nlp_feats,
        )
        persist_folds.append(compute_early_k_persistence_metrics(list(t_ds), train_cascades=list(tr_l.dataset)))
        fold_res = run_multi_seed_benchmark(
            factories,
            tr_l,
            va_l,
            te_l,
            seeds=OFFICIAL_SEEDS,
            epochs=1,
            patience=1,
            reference_model="LightGBM",
            comparator_model="Ridge",
            compute_bootstrap=False,
            primary_hypothesis_keys=active_keys,
            device=torch.device("cpu"),
        )
        for m_name, m_stats in fold_res["summary_stats"].items():
            fold_raw[m_name].append({k: v[0] for k, v in m_stats.items()})
        for m_name, vm_stats in (fold_res.get("val_summary_stats") or {}).items():
            if m_name in fold_val_raw:
                fold_val_raw[m_name].append({k: v[0] for k, v in vm_stats.items()})

    metric_keys = tuple(fold_raw["LightGBM"][0].keys())
    summary_stats: dict[str, dict[str, tuple[float, float]]] = {}
    for m_name, rows in fold_raw.items():
        agg: dict[str, tuple[float, float]] = {}
        for k in metric_keys:
            arr = np.asarray([r.get(k, float("nan")) for r in rows], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            agg[k] = (
                (float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0)
                if arr.size > 0
                else (float("nan"), float("nan"))
            )
        summary_stats[m_name] = agg

    persist_agg: dict[str, tuple[float, float]] = {}
    for pk in ("msle", "mae", "rmse_size", "r2_size", "spearman_size", "rmse", "mae_polar", "r2_polar", "pearson_r", "spearman_polar"):
        parr = np.asarray([float(pf.get(pk, float("nan"))) for pf in persist_folds], dtype=np.float64)
        parr = parr[np.isfinite(parr)]
        persist_agg[pk] = (
            (float(np.mean(parr)), float(np.std(parr, ddof=1)) if parr.size >= 2 else 0.0)
            if parr.size > 0
            else (float("nan"), float("nan"))
        )
    summary_stats = {"Early-K Persistence": _build_persistence_summary_row(persist_agg), **summary_stats}

    p_vals = {
        k: compute_paired_ttest([r[k] for r in fold_raw["LightGBM"]], [r[k] for r in fold_raw["Ridge"]])
        for k in metric_keys
    }
    boot_p_vals = {
        k: compute_paired_bootstrap_pvalue([r[k] for r in fold_raw["LightGBM"]], [r[k] for r in fold_raw["Ridge"]])
        for k in metric_keys
    }
    adj_p = holm_bonferroni_adjust({k: p_vals[k] for k in active_keys if k in p_vals})
    adj_boot = holm_bonferroni_adjust({k: boot_p_vals[k] for k in active_keys if k in boot_p_vals})
    result: dict[str, Any] = {
        "seeds": list(OFFICIAL_SEEDS),
        "raw_results": fold_raw,
        "summary_stats": summary_stats,
        "early_k_persistence": {
            k: [None if math.isnan(v[0]) else v[0], None if math.isnan(v[1]) else v[1]]
            for k, v in persist_agg.items()
        },
        "p_values": p_vals,
        "adjusted_p_values": {k: adj_p.get(k, float("nan")) for k in metric_keys},
        "bootstrap_p_values": boot_p_vals,
        "adjusted_bootstrap_p_values": {k: adj_boot.get(k, float("nan")) for k in metric_keys},
        "reference_model": "LightGBM",
        "comparator_model": "Ridge",
        "nonparametric_test_label": "Exact Perm." if len(fold_raw["LightGBM"]) < 6 else "Bootstrap",
    }
    elapsed = time.perf_counter() - t0
    serialize_benchmark_json(
        result,
        output_json,
        extra={
            "dataset": "pheme",
            "k_horizon": k_horizon,
            "eval_protocol": "loo",
            "split_mode": "canonical",
            "edge_source": "teacher",
            "polar_target_mode": "future_only",
            "min_total_replies": min_total_replies,
            "num_loo_folds": len(folds),
            "lightgbm_backend": lgb_backend,
            "model_subset": "tabular_and_calibrated_persistence_only",
            "is_real_data": True,
            "elapsed_sec": round(elapsed, 3),
            "early_k_persistence": result["early_k_persistence"],
        },
    )
    print(f"[Tier1-Tabular:pheme/loo] Saved {output_json} ({elapsed:.2f}s, folds={len(folds)})", flush=True)
    return result


def run_all_tier1_tabular(results_dir: Path = Path("results")) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["vn_canonical"] = _run_tabular_inductive(
        data_dir=Path("data/processed/vn"),
        ds_name="vn",
        split_mode="canonical",
        k_horizon=20,
        min_total_replies=25,
        edge_source="auto",
        output_json=results_dir / "tier1_tabular_canonical_vn.json",
    )
    out["vn_cluster"] = _run_tabular_inductive(
        data_dir=Path("data/processed/vn"),
        ds_name="vn",
        split_mode="embedding_cluster",
        k_horizon=20,
        min_total_replies=25,
        edge_source="teacher",
        output_json=results_dir / "tier1_tabular_cluster_vn.json",
    )
    out["vn_temporal"] = _run_tabular_inductive(
        data_dir=Path("data/processed/vn"),
        ds_name="vn",
        split_mode="temporal",
        k_horizon=20,
        min_total_replies=25,
        edge_source="teacher",
        output_json=results_dir / "tier1_tabular_temporal_vn.json",
    )
    out["pheme_inductive"] = _run_tabular_inductive(
        data_dir=Path("data/processed/pheme"),
        ds_name="pheme",
        split_mode="canonical",
        k_horizon=10,
        min_total_replies=15,
        edge_source="auto",
        output_json=results_dir / "tier1_tabular_inductive_pheme.json",
    )
    out["pheme_loo"] = _run_tabular_pheme_loo(
        data_dir=Path("data/processed/pheme"),
        k_horizon=10,
        min_total_replies=15,
        output_json=results_dir / "tier1_tabular_loo_pheme.json",
    )
    return out


def validate_results_dir(results_dir: Path = Path("results")) -> dict[str, Any]:
    """Verify strict RFC-8259 JSON finiteness and M5 real-data invariants across `results/*.json`."""
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {results_dir}")

    checked: list[str] = []
    for jf in json_files:
        raw_text = jf.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
        _assert_finite_json_obj(payload, path_prefix=jf.name)
        checked.append(jf.name)

    for ds in ("vn", "pheme"):
        macro_p = results_dir / f"results_macro_network_{ds}.json"
        if macro_p.is_file():
            m = json.loads(macro_p.read_text(encoding="utf-8"))
            rwc_val = float(m["polarization"]["rwc"])
            if not (-1.0 - 1e-6 <= rwc_val <= 1.0 + 1e-6):
                raise AssertionError(f"{macro_p.name}: rwc={rwc_val} outside [-1, 1]")
            if "q_signed_z_score" not in m["polarization"] or "q_signed_p_value" not in m["polarization"]:
                raise AssertionError(f"{macro_p.name}: missing Gómez Q_signed permutation fields")
            sb = m["structural_balance"]
            if "weak_balance_ratio" not in sb or "weak_balance_z_score" not in sb:
                raise AssertionError(f"{macro_p.name}: missing Davis Weak Balance fields")
            if ds == "pheme" and int(m.get("filtered_zero_embedding_posts", -1)) != 24354:
                raise AssertionError(
                    f"{macro_p.name}: expected filtered_zero_embedding_posts==24354, got {m.get('filtered_zero_embedding_posts')}"
                )

    summary = {"results_dir": str(results_dir), "validated_json_count": len(checked), "files": checked}
    print("[ValidateResults] OK:", json.dumps(summary), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="M5 local real-data audits & fast 5-seed tabular baselines.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-contraction", action="store_true")
    parser.add_argument("--run-tier1-tabular", action="store_true")
    parser.add_argument("--validate-results", action="store_true")
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.run_contraction:
        run_vn_contraction_eval(output_json=args.results_dir / "contraction_eval_vn.json")
    if args.run_tier1_tabular:
        run_all_tier1_tabular(results_dir=args.results_dir)
    if args.validate_results:
        validate_results_dir(results_dir=args.results_dir)


if __name__ == "__main__":
    main()

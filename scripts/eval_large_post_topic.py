"""Real quantitative evaluation for Zero-Shot Topic Classification & Large-Post Dynamics.

Evaluates `src.nlp.topic_post_analyzer` without synthetic data or data leakage:
  1. PHEME 9-Event Ground-Truth Zero-Shot Topic Benchmark (N=2,664 cohort + N=334 Test):
     compares `lexical_ngram_fallback`, `transformer_cosine_embedding` (RoBERTa encoder),
     and `mdeberta_zero_shot_nli` (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`).
  2. VN Topic-Stratified Polarization & Multi-Milestone Large-Post Trajectory (N=1,500;
     608 threads >= 50 replies; 164 threads >= 100 replies; Test N=225):
     evaluates inter-backend topic agreement, Kruskal-Wallis topic stratification,
     milestone polarization drift across scale tiers, and K=20 early-warning ROC-AUC/PR-AUC.

Exports:
  - `results/results_large_post_topic.json`
  - `results/tables_large_post_topic.tex`
  - `results/figures/vn/fig6_large_post_topic_trajectory.{pdf,svg}`
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from scipy.stats import kruskal
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pyg_dataset import (  # noqa: E402
    EarlyCascadeDataset,
    _compute_full_tree_polarization,
    _compute_root_aligned_stances,
    load_topic_splits_json,
)
from src.models.sbi_fagat import SBiFAGAT  # noqa: E402
from src.models.sota_cascades import RidgeBaseline  # noqa: E402
from src.nlp.dual_edge_scorer import DualHeadCrossEncoder, MDEBERTA_NLI_MODEL_NAME  # noqa: E402
from src.nlp.extract_features import compute_masked_mean_pooling  # noqa: E402
from src.nlp.topic_post_analyzer import (  # noqa: E402
    DEFAULT_TOPICS_VI,
    VN_POLAR_Q75,
    _build_topic_context_text,
    _resolve_nli_indices,
    _score_topics_lexical,
    analyze_large_thread_dynamics,
)
from src.utils.visualize import _configure_academic_rcparams, _save_vector_pair  # noqa: E402

# Neutral event-level topic descriptions for PHEME 9 real news events (zero slug leakage).
PHEME_EVENT_TOPICS: tuple[dict[str, str], ...] = (
    {
        "id": "charliehebdo",
        "label": "Paris Satirical Newspaper Attack",
        "description": "Paris satirical newspaper office terrorist shooting and free speech crisis",
    },
    {
        "id": "sydneysiege",
        "label": "Australian Cafe Hostage Standoff",
        "description": "Martin Place Lindt cafe hostage standoff and police siege in Australia",
    },
    {
        "id": "ferguson",
        "label": "Missouri Street Protests & Policing",
        "description": "Missouri street protests, grand jury verdict and civil policing unrest",
    },
    {
        "id": "ottawashooting",
        "label": "Canadian Parliament Armed Attack",
        "description": "Canadian parliament building armed attack, soldier shooting and lockdown",
    },
    {
        "id": "germanwings-crash",
        "label": "Alpine Passenger Airliner Crash",
        "description": "Commercial passenger airliner crash in the French Alps mountains",
    },
    {
        "id": "putinmissing",
        "label": "Russian Leader Absence Rumor",
        "description": "Public absence speculation and political rumors about the Russian president",
    },
    {
        "id": "prince-toronto",
        "label": "Surprise Pop Concert Rumor",
        "description": "Surprise pop music artist live concert ticket rumor in Ontario Canada",
    },
    {
        "id": "gurlitt",
        "label": "Discovered Art Hoard Inheritance",
        "description": "Munich apartment discovered Nazi-looted fine art collection inheritance",
    },
    {
        "id": "ebola-essien",
        "label": "Footballer Disease Hoax Rumor",
        "description": "Viral medical hoax claiming an international soccer midfielder contracted a deadly virus",
    },
)


def _extract_early_thread_contexts(
    nodes_path: Path,
    *,
    min_total_replies: int,
    k_prefix: int,
    splits: Mapping[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic early-prefix text contexts (root + first 5 replies) per valid cascade."""
    tbl = pq.read_table(nodes_path)
    rows = tbl.to_pylist()
    by_cid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cid[str(r["cascade_id"])].append(r)

    split_lookup: dict[str, str] = {}
    if splits:
        for sp_name in ("train", "val", "test"):
            for tid in splits.get(sp_name, set()):
                split_lookup[str(tid)] = sp_name

    contexts: list[dict[str, Any]] = []
    for cid in sorted(by_cid.keys()):
        c_rows = by_cid[cid]
        roots = [r for r in c_rows if bool(r.get("is_root"))]
        if not roots:
            continue
        root = roots[0]
        non_roots = [r for r in c_rows if not bool(r.get("is_root"))]
        if len(non_roots) < min_total_replies:
            continue

        def _ts(r: Mapping[str, Any]) -> tuple[float, str]:
            v = r.get("created_at_utc")
            ts_val = float(v.timestamp()) if hasattr(v, "timestamp") else float(v or 0.0)
            return (ts_val, str(r.get("node_id")))

        non_roots.sort(key=_ts)
        tid = str(root.get("topic_id") or "")
        root_txt = str(root.get("text") or "").strip()
        first_line = root_txt.splitlines()[0] if root_txt else ""
        reply_snips = [str(r.get("text") or "") for r in non_roots[: min(5, k_prefix)]]
        ctx = _build_topic_context_text(first_line[:140], root_txt, reply_snips, max_chars=500)
        contexts.append(
            {
                "cascade_id": cid,
                "topic_id": tid,
                "split": split_lookup.get(tid, "unknown"),
                "num_replies": len(non_roots),
                "context_text": ctx or root_txt[:400],
            }
        )
    return contexts


def _score_batch_lexical(
    contexts: Sequence[str],
    candidates: Sequence[Mapping[str, str]],
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    probs = np.zeros((len(contexts), len(candidates)), dtype=np.float64)
    for i, ctx in enumerate(contexts):
        sc, _ = _score_topics_lexical(ctx, candidates)
        probs[i] = np.asarray(sc, dtype=np.float64)
    ms_per_post = ((time.perf_counter() - t0) * 1000.0) / max(1, len(contexts))
    return probs, ms_per_post


def _score_batch_encoder_cosine(
    contexts: Sequence[str],
    candidates: Sequence[Mapping[str, str]],
    *,
    ce_dir: Path,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, float]:
    ce_model, tok = DualHeadCrossEncoder.load_checkpoint(ce_dir, map_location="cpu")
    encoder = ce_model.encoder.to(device)
    encoder.eval()
    cand_texts = [f"{c['label']}: {c['description']}" for c in candidates]
    use_amp = device.type == "cuda"

    with torch.no_grad():
        c_enc = tok(
            cand_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        c_enc = {k: v.to(device) for k, v in c_enc.items()}
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            c_out = encoder(**c_enc)
            c_hid = c_out.last_hidden_state if hasattr(c_out, "last_hidden_state") else c_out[0]
            cand_vecs = F.normalize(
                compute_masked_mean_pooling(c_hid.float(), c_enc["attention_mask"]),
                p=2,
                dim=-1,
            )

        t0 = time.perf_counter()
        all_probs: list[np.ndarray] = []
        for start in range(0, len(contexts), batch_size):
            chunk = list(contexts[start : start + batch_size])
            b_enc = tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            b_enc = {k: v.to(device) for k, v in b_enc.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                b_out = encoder(**b_enc)
                b_hid = b_out.last_hidden_state if hasattr(b_out, "last_hidden_state") else b_out[0]
                ctx_vecs = F.normalize(
                    compute_masked_mean_pooling(b_hid.float(), b_enc["attention_mask"]),
                    p=2,
                    dim=-1,
                )
            sims = torch.matmul(ctx_vecs, cand_vecs.T)
            probs = F.softmax(sims / 0.08, dim=-1).detach().cpu().numpy()
            all_probs.append(probs)
        ms_per_post = ((time.perf_counter() - t0) * 1000.0) / max(1, len(contexts))

    del encoder, ce_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(all_probs), ms_per_post


def _score_batch_mdeberta_nli(
    contexts: Sequence[str],
    candidates: Sequence[Mapping[str, str]],
    *,
    lang: str,
    device: torch.device,
    batch_size: int = 128,
    max_length: int = 160,
    allow_download: bool = True,
) -> tuple[np.ndarray, float]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    local_only = not allow_download
    tok = AutoTokenizer.from_pretrained(MDEBERTA_NLI_MODEL_NAME, local_files_only=local_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        MDEBERTA_NLI_MODEL_NAME, local_files_only=local_only
    ).to(device)
    model.eval()
    ent_idx, con_idx = _resolve_nli_indices(model.config)

    if lang == "en":
        hypotheses = [f"This text is about {c['label']}." for c in candidates]
    else:
        hypotheses = [f"Văn bản này nói về {c['label']}." for c in candidates]

    n_posts = len(contexts)
    n_cands = len(candidates)
    pair_premises: list[str] = []
    pair_hyps: list[str] = []
    for ctx in contexts:
        for hyp in hypotheses:
            pair_premises.append(ctx)
            pair_hyps.append(hyp)

    use_amp = device.type == "cuda"
    total_pairs = len(pair_premises)
    # Length-sorted dynamic batching minimizes Transformer O(L^2) padding overhead without changing logits.
    order = sorted(range(total_pairs), key=lambda i: len(pair_premises[i]) + len(pair_hyps[i]))
    sorted_premises = [pair_premises[i] for i in order]
    sorted_hyps = [pair_hyps[i] for i in order]

    diffs_sorted: list[np.ndarray] = []
    next_pct = 25
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, total_pairs, batch_size):
            enc = tok(
                sorted_premises[start : start + batch_size],
                sorted_hyps[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits.float()
            diff = (logits[:, ent_idx] - logits[:, con_idx]).detach().cpu().numpy()
            diffs_sorted.append(diff)
            done = min(total_pairs, start + batch_size)
            pct = (100 * done) // max(1, total_pairs)
            if pct >= next_pct:
                print(
                    f"[mDeBERTa-NLI:{lang}] {pct}% ({done}/{total_pairs} pairs, {time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
                while next_pct <= pct:
                    next_pct += 25
    ms_per_post = ((time.perf_counter() - t0) * 1000.0) / max(1, n_posts)

    diff_arr = np.empty(total_pairs, dtype=np.float32)
    diff_arr[np.asarray(order, dtype=np.int64)] = np.concatenate(diffs_sorted).astype(np.float32)
    diff_mat = diff_arr.reshape(n_posts, n_cands)
    probs = F.softmax(torch.from_numpy(diff_mat) / 1.5, dim=-1).numpy()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs, ms_per_post


def _classification_metrics(
    y_true_ids: Sequence[str],
    probs: np.ndarray,
    cand_ids: Sequence[str],
) -> dict[str, float]:
    id_to_idx = {cid: i for i, cid in enumerate(cand_ids)}
    y_true = np.array([id_to_idx[c] for c in y_true_ids], dtype=np.int64)
    y_pred = np.argmax(probs, axis=1)
    top2_idx = np.argsort(-probs, axis=1)[:, : min(2, probs.shape[1])]
    top2_acc = float(np.mean([yt in row for yt, row in zip(y_true, top2_idx)]))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top2_accuracy": top2_acc,
    }


def _compute_real_trajectories_and_early_targets(
    data_dir: Path,
    *,
    k_early: int,
    min_total_replies: int,
) -> dict[str, dict[str, Any]]:
    """Compute true full-tree trajectory milestones and future_only y_polar for every valid cascade."""
    nodes_tbl = pq.read_table(data_dir / "nodes.parquet")
    edges_tbl = pq.read_table(data_dir / "edges.parquet")
    nlp_pt = torch.load(data_dir / "nlp_features.pt", map_location="cpu", weights_only=False)
    stance_map = nlp_pt.get("edge_stance") or nlp_pt.get("teacher_edge_stance") or {}
    hostility_map = nlp_pt.get("edge_hostility") or nlp_pt.get("teacher_edge_hostility") or {}
    target_stance_map = nlp_pt.get("teacher_edge_stance") or stance_map
    root_stance_map = nlp_pt.get("teacher_node_root_stance") or nlp_pt.get("node_root_stance")
    del nlp_pt

    nodes_by_cid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in nodes_tbl.to_pylist():
        nodes_by_cid[str(r["cascade_id"])].append(r)
    edges_by_cid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in edges_tbl.to_pylist():
        edges_by_cid[str(r["cascade_id"])].append(r)

    out: dict[str, dict[str, Any]] = {}
    for cid in sorted(nodes_by_cid.keys()):
        c_nodes = nodes_by_cid[cid]
        roots = [n for n in c_nodes if bool(n.get("is_root"))]
        if not roots:
            continue
        root = roots[0]
        root_id = str(root["node_id"])
        non_roots = [n for n in c_nodes if str(n["node_id"]) != root_id]
        if len(non_roots) < min_total_replies:
            continue

        c_edges = edges_by_cid.get(cid, [])
        children_adj: dict[str, list[str]] = {root_id: []}
        edge_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        edge_scores: dict[str, dict[str, float]] = {}
        for e in c_edges:
            src, dst = str(e["source_id"]), str(e["target_id"])
            eid = str(e.get("edge_id") or f"{src}->{dst}")
            children_adj.setdefault(src, []).append(dst)
            edge_lookup[(src, dst)] = {"edge_id": eid}
            edge_scores[eid] = {
                "s_uv": float(stance_map.get(eid, 0.0)),
                "h_uv": float(hostility_map.get(eid, 0.0)),
            }

        depth_map: dict[str, int] = {root_id: 0}
        dq: deque[str] = deque([root_id])
        while dq:
            cur = dq.popleft()
            for ch in children_adj.get(cur, []):
                if ch not in depth_map:
                    depth_map[ch] = depth_map[cur] + 1
                    dq.append(ch)

        dyn = analyze_large_thread_dynamics(
            root_node=root,
            all_nodes=c_nodes,
            all_edges=c_edges,
            edge_scores=edge_scores,
            depth_map=depth_map,
            k_early_ref=k_early,
            lang="vi" if data_dir.name == "vn" else "en",
            scope="full_analysis",
        )

        # Also compute strictly future_only polarization (t > k_early) matching training targets
        def _ts(n: Mapping[str, Any]) -> tuple[float, int, str]:
            v = n.get("created_at_utc")
            tv = float(v.timestamp()) if hasattr(v, "timestamp") else float(v or 0.0)
            return (tv, depth_map.get(str(n["node_id"]), 999), str(n["node_id"]))

        non_roots_sorted = sorted(non_roots, key=_ts)
        kept_nodes = [root] + non_roots_sorted[: k_prefix_idx(k_early)]
        kept_id_to_idx = {str(n["node_id"]): idx for idx, n in enumerate(kept_nodes)}
        excluded_pair_set = {
            (str(n.get("parent_id")), str(n["node_id"]))
            for n in kept_nodes[1:]
            if str(n.get("parent_id")) in kept_id_to_idx
        }
        all_aligned = _compute_root_aligned_stances(
            root_id=root_id,
            children_adj=children_adj,
            edge_lookup=edge_lookup,
            stance_map=target_stance_map,
            root_stance_map=root_stance_map,
            excluded_pairs=excluded_pair_set,
        )
        y_polar_future = (
            float(_compute_full_tree_polarization(all_aligned, lambda_polar=0.5))
            if len(all_aligned) >= 2
            else 0.0
        )

        out[cid] = {
            "num_replies": len(non_roots_sorted),
            "scale_tier": dyn["scale_tier"],
            "evolution_trend": dyn["evolution_trend"],
            "post_archetype": dyn["post_archetype"],
            "early_ref_y_polar": dyn["early_ref_y_polar"],
            "full_y_polar": dyn["full_y_polar"],
            "polar_delta": dyn["polar_delta"],
            "y_polar_future": y_polar_future,
            "trajectory": dyn["polarization_trajectory"],
        }
    return out


def k_prefix_idx(k: int) -> int:
    return max(0, int(k))


def _aggregate_scale_trajectories(
    per_cascade: Mapping[str, Mapping[str, Any]],
    *,
    scale_tiers: Sequence[str],
    milestones: Sequence[int] = (5, 10, 20, 35, 50, 75, 100, 150),
) -> dict[str, Any]:
    """Aggregate milestone y_polar and escalation/de-escalation counts by thread size tier."""
    summary: dict[str, Any] = {}
    for tier in scale_tiers:
        if tier == "large_ge_50":
            items = [v for v in per_cascade.values() if int(v["num_replies"]) >= 50]
        elif tier == "mega_ge_100":
            items = [v for v in per_cascade.values() if int(v["num_replies"]) >= 100]
        else:
            items = [v for v in per_cascade.values() if v["scale_tier"] == tier]

        n = len(items)
        if n == 0:
            summary[tier] = {"count": 0}
            continue

        ms_stats: list[dict[str, Any]] = []
        for m in milestones:
            vals = [
                float(pt["y_polar"])
                for item in items
                for pt in item["trajectory"]
                if int(pt["k"]) == m
            ]
            if len(vals) >= 3:
                arr = np.asarray(vals, dtype=np.float64)
                ms_stats.append(
                    {
                        "k": int(m),
                        "n": int(len(vals)),
                        "mean_y_polar": float(np.mean(arr)),
                        "std_y_polar": float(np.std(arr, ddof=1)),
                    }
                )
        final_vals = np.asarray([float(it["full_y_polar"]) for it in items], dtype=np.float64)
        delta_vals = np.asarray([float(it["polar_delta"]) for it in items], dtype=np.float64)
        esc = sum(1 for it in items if it["evolution_trend"] == "escalating")
        deesc = sum(1 for it in items if it["evolution_trend"] == "de_escalating")
        stab = sum(1 for it in items if it["evolution_trend"] == "stable")

        summary[tier] = {
            "count": n,
            "mean_replies": float(np.mean([it["num_replies"] for it in items])),
            "mean_full_y_polar": float(np.mean(final_vals)),
            "mean_polar_delta": float(np.mean(delta_vals)),
            "escalating_rate": float(esc / n),
            "de_escalating_rate": float(deesc / n),
            "stable_rate": float(stab / n),
            "milestones": ms_stats,
        }
    return summary


def _binary_early_warning_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    val_y_true: np.ndarray | None = None,
    val_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    if np.unique(y_true).size < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "best_f1": float("nan")}
    roc = float(roc_auc_score(y_true, scores))
    pr = float(average_precision_score(y_true, scores))
    if val_y_true is not None and val_scores is not None and np.unique(val_y_true).size >= 2:
        best_val_f1 = -1.0
        selected_thr = float(np.median(val_scores))
        for q in np.linspace(0.1, 0.9, 17):
            thr = float(np.quantile(val_scores, q))
            vf1 = float(f1_score(val_y_true, (val_scores >= thr).astype(np.int64), zero_division=0))
            if vf1 > best_val_f1:
                best_val_f1 = vf1
                selected_thr = thr
        test_f1 = float(f1_score(y_true, (scores >= selected_thr).astype(np.int64), zero_division=0))
        return {
            "roc_auc": roc,
            "pr_auc": pr,
            "best_f1": test_f1,
            "val_selected_threshold": selected_thr,
            "threshold_selection": "val_split_only",
        }
    best_f1 = 0.0
    for q in np.linspace(0.1, 0.9, 17):
        thr = float(np.quantile(scores, q))
        f1 = float(f1_score(y_true, (scores >= thr).astype(np.int64), zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
    return {"roc_auc": roc, "pr_auc": pr, "best_f1": best_f1}


def _evaluate_vn_early_warning(
    vn_dir: Path,
    ckpt_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate K=20 early-warning detection of high future polarization & large/viral threads on VN Test."""
    nlp_pt = torch.load(vn_dir / "nlp_features.pt", map_location="cpu", weights_only=False)
    splits = load_topic_splits_json(vn_dir / "topic_splits.json")
    train_ds = EarlyCascadeDataset(
        vn_dir, k_horizon=20, split="train", min_total_replies=25,
        polar_target_mode="future_only", nlp_features=nlp_pt, custom_topic_splits=splits, edge_source="auto",
    )
    val_ds = EarlyCascadeDataset(
        vn_dir, k_horizon=20, split="val", min_total_replies=25,
        polar_target_mode="future_only", nlp_features=nlp_pt, custom_topic_splits=splits, edge_source="auto",
    )
    test_ds = EarlyCascadeDataset(
        vn_dir, k_horizon=20, split="test", min_total_replies=25,
        polar_target_mode="future_only", nlp_features=nlp_pt, custom_topic_splits=splits, edge_source="auto",
    )
    del nlp_pt

    val_list = list(val_ds)
    test_list = list(test_ds)
    n_replies_val = np.array([float(np.expm1(float(d.y_size.item()))) for d in val_list], dtype=np.float64)
    y_polar_val = np.array([float(d.y_polar.item()) for d in val_list], dtype=np.float64)
    y_high_polar_val = (y_polar_val >= VN_POLAR_Q75).astype(np.int64)
    y_large_50_val = (n_replies_val >= 50.0).astype(np.int64)
    y_viral_100_val = (n_replies_val >= 100.0).astype(np.int64)

    n_replies_test = np.array([float(np.expm1(float(d.y_size.item()))) for d in test_list], dtype=np.float64)
    y_polar_test = np.array([float(d.y_polar.item()) for d in test_list], dtype=np.float64)

    y_high_polar = (y_polar_test >= VN_POLAR_Q75).astype(np.int64)
    y_large_50 = (n_replies_test >= 50.0).astype(np.int64)
    y_viral_100 = (n_replies_test >= 100.0).astype(np.int64)

    # 1. Early-K Persistence
    ep_scores_val = np.array([float(d.early_y_polar.item()) for d in val_list], dtype=np.float64)
    esize_scores_val = np.array([float(math.log1p(max(0, int(d.x.size(0)) - 1))) for d in val_list], dtype=np.float64)
    ep_scores = np.array([float(d.early_y_polar.item()) for d in test_list], dtype=np.float64)
    esize_scores = np.array([float(math.log1p(max(0, int(d.x.size(0)) - 1))) for d in test_list], dtype=np.float64)

    # 2. Ridge trained strictly on VN Train/Val
    ridge = RidgeBaseline(num_rumor_classes=4).fit_dataset(list(train_ds), val_list, seed=42)
    ridge_val_out = ridge.predict_dataset(val_list)
    ridge_out = ridge.predict_dataset(test_list)

    # 3. Trained SBiFAGAT checkpoint
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    model = SBiFAGAT(
        in_dim=int(payload.get("in_dim", 770)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        edge_dim=int(cfg.get("edge_dim", 2)),
        num_rumor_classes=int(payload.get("num_rumor_classes", 4)),
        eps=float(cfg.get("eps", 0.5)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_cascade_stats=bool(cfg.get("use_cascade_stats", True)),
        predict_uncertainty=bool(cfg.get("predict_uncertainty", False)),
        use_polar_persistence_anchor=bool(cfg.get("use_polar_persistence_anchor", True)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    def _predict_sbi(ds_list: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        ld = DataLoader(ds_list, batch_size=64, shuffle=False)
        sz_p, po_p = [], []
        with torch.no_grad():
            for batch in ld:
                out = model(batch.to(device))
                sz_p.append(out["y_size_pred"].view(-1).cpu().numpy())
                po_p.append(out["y_polar_pred"].view(-1).cpu().numpy())
        return np.concatenate(sz_p), np.concatenate(po_p)

    sbi_size_val, sbi_polar_val = _predict_sbi(val_list)
    sbi_size, sbi_polar = _predict_sbi(test_list)

    return {
        "n_test": len(test_list),
        "positive_rates": {
            "high_future_polar_ge_q75": float(np.mean(y_high_polar)),
            "large_replies_ge_50": float(np.mean(y_large_50)),
            "viral_replies_ge_100": float(np.mean(y_viral_100)),
        },
        "tasks": {
            "high_future_polar_ge_q75": {
                "Early-K Persistence": _binary_early_warning_metrics(
                    y_high_polar, ep_scores, val_y_true=y_high_polar_val, val_scores=ep_scores_val,
                ),
                "Ridge": _binary_early_warning_metrics(
                    y_high_polar, ridge_out["y_polar_pred"], val_y_true=y_high_polar_val, val_scores=ridge_val_out["y_polar_pred"],
                ),
                "SBiFAGAT": _binary_early_warning_metrics(
                    y_high_polar, sbi_polar, val_y_true=y_high_polar_val, val_scores=sbi_polar_val,
                ),
            },
            "large_replies_ge_50": {
                "Early-K Persistence": _binary_early_warning_metrics(
                    y_large_50, esize_scores, val_y_true=y_large_50_val, val_scores=esize_scores_val,
                ),
                "Ridge": _binary_early_warning_metrics(
                    y_large_50, ridge_out["y_size_pred"], val_y_true=y_large_50_val, val_scores=ridge_val_out["y_size_pred"],
                ),
                "SBiFAGAT": _binary_early_warning_metrics(
                    y_large_50, sbi_size, val_y_true=y_large_50_val, val_scores=sbi_size_val,
                ),
            },
            "viral_replies_ge_100": {
                "Early-K Persistence": _binary_early_warning_metrics(
                    y_viral_100, esize_scores, val_y_true=y_viral_100_val, val_scores=esize_scores_val,
                ),
                "Ridge": _binary_early_warning_metrics(
                    y_viral_100, ridge_out["y_size_pred"], val_y_true=y_viral_100_val, val_scores=ridge_val_out["y_size_pred"],
                ),
                "SBiFAGAT": _binary_early_warning_metrics(
                    y_viral_100, sbi_size, val_y_true=y_viral_100_val, val_scores=sbi_size_val,
                ),
            },
        },
    }


def _export_latex_and_figure(
    results: Mapping[str, Any],
    *,
    output_tex: Path,
    figures_dir: Path,
) -> None:
    ph = results["pheme_zero_shot_9_events"]
    ew = results["vn_early_warning_k20"]["tasks"]
    tr_vn = results["vn_large_post_trajectory"]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Zero-Shot Event Topic Classification on PHEME (9 Real Events, $N=2{,}664$) and $K=20$ Early-Warning Detection on VN Test ($N=225$).}",
        r"\label{tab:large_post_topic}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\multicolumn{7}{l}{\textbf{Panel A: PHEME 9-Event Zero-Shot Topic Classification (Full Tier-2A Cohort, $N=2{,}664$)}} \\",
        r"\midrule",
        r"Backend & Accuracy $\uparrow$ & Macro-F1 $\uparrow$ & Weighted-F1 $\uparrow$ & Top-2 Acc $\uparrow$ & Latency (ms/post) $\downarrow$ & Notes \\",
        r"\midrule",
    ]
    for b_key, b_label, note in (
        ("lexical_ngram_fallback", "Lexical N-gram Fallback", "Offline surface keywords"),
        ("transformer_cosine_embedding", "RoBERTa Cosine (Student Enc.)", "Frozen 768-D cosine"),
        ("mdeberta_zero_shot_nli", "mDeBERTa-v3-XNLI (Zero-Shot)", "GPU FP16 NLI entailment"),
    ):
        m = ph["full_cohort"][b_key]
        lines.append(
            f"{b_label} & {m['accuracy']:.4f} & {m['macro_f1']:.4f} & {m['weighted_f1']:.4f} & "
            f"{m['top2_accuracy']:.4f} & {m['ms_per_post']:.2f} & {note} \\\\"
        )

    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{7}{l}{\textbf{Panel B: VN Test ($N=225$) $K=20$ Early-Warning Classification (High Future Polar. $\ge Q_{75}^{\text{train}}$, Large $\ge 50$, Viral $\ge 100$)}} \\",
            r"\midrule",
            r"Model (from $K=20$ prefix) & Polar ROC-AUC $\uparrow$ & Polar PR-AUC $\uparrow$ & Large ($\ge 50$) ROC $\uparrow$ & Large PR $\uparrow$ & Viral ($\ge 100$) ROC $\uparrow$ & Viral PR $\uparrow$ \\",
            r"\midrule",
        ]
    )
    for m_name in ("Early-K Persistence", "Ridge", "SBiFAGAT"):
        hp = ew["high_future_polar_ge_q75"][m_name]
        lg = ew["large_replies_ge_50"][m_name]
        vr = ew["viral_replies_ge_100"][m_name]
        lines.append(
            f"{m_name} & {hp['roc_auc']:.4f} & {hp['pr_auc']:.4f} & {lg['roc_auc']:.4f} & "
            f"{lg['pr_auc']:.4f} & {vr['roc_auc']:.4f} & {vr['pr_auc']:.4f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 2-panel vector figure
    _configure_academic_rcparams()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))

    backends = ("lexical_ngram_fallback", "transformer_cosine_embedding", "mdeberta_zero_shot_nli")
    labels = ("Lexical\nN-gram", "RoBERTa\nCosine", "mDeBERTa-v3\nZero-Shot NLI")
    accs = [ph["full_cohort"][b]["accuracy"] for b in backends]
    f1s = [ph["full_cohort"][b]["macro_f1"] for b in backends]
    top2s = [ph["full_cohort"][b]["top2_accuracy"] for b in backends]

    x = np.arange(len(backends))
    w = 0.25
    ax1.bar(x - w, accs, width=w, label="Accuracy", color="#0072B2")
    ax1.bar(x, f1s, width=w, label="Macro-F1", color="#D55E00")
    ax1.bar(x + w, top2s, width=w, label="Top-2 Acc", color="#009E73")
    for xi, a, f, t2 in zip(x, accs, f1s, top2s):
        ax1.text(xi - w, a + 0.015, f"{a:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax1.text(xi, f + 0.015, f"{f:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax1.text(xi + w, t2 + 0.015, f"{t2:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("Score (PHEME 9-Event Real Labels)")
    ax1.set_title("(a) Zero-Shot Topic Classification (N=2,664)")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(axis="y", alpha=0.25, linestyle="--")

    tier_styles = (
        ("medium", "Medium (25-49)", "#0072B2", "--"),
        ("large_ge_50", "Large (>=50)", "#E69F00", "-."),
        ("mega_ge_100", "Viral (>=100)", "#D55E00", "-"),
    )
    for t_key, t_lbl, col, ls in tier_styles:
        info = tr_vn.get(t_key, {})
        ms = info.get("milestones") or []
        if not ms:
            continue
        ks = [int(pt["k"]) for pt in ms]
        ys = [float(pt["mean_y_polar"]) for pt in ms]
        ax2.plot(ks, ys, marker="o", markersize=4.5, linewidth=2.0, color=col, linestyle=ls, label=f"{t_lbl} (n={info['count']})")

    ax2.axhline(VN_POLAR_Q75, color="#64748b", linestyle=":", linewidth=1.2, label=f"VN Train Q75 ({VN_POLAR_Q75:.3f})")
    ax2.set_xlabel("Observed Comment Prefix Milestone (K)")
    ax2.set_ylabel("Mean Cumulative Polarization (y_polar)")
    ax2.set_title("(b) VN Multi-Milestone Trajectory by Thread Scale")
    ax2.legend(loc="lower right", frameon=False, fontsize=8)
    ax2.grid(True, alpha=0.25, linestyle="--")

    fig.tight_layout()
    _save_vector_pair(fig, figures_dir, "fig6_large_post_topic_trajectory")


def run_benchmark(
    *,
    vn_dir: Path = Path("data/processed/vn"),
    pheme_dir: Path = Path("data/processed/pheme"),
    ce_vn_dir: Path = Path("checkpoints/dual_ce_vn"),
    ce_pheme_dir: Path = Path("checkpoints/dual_ce_pheme"),
    gnn_vn_ckpt: Path = Path("checkpoints/best_sbifagat_vn.pt"),
    output_json: Path = Path("results/results_large_post_topic.json"),
    output_tex: Path = Path("results/tables_large_post_topic.tex"),
    figures_dir: Path = Path("results/figures/vn"),
    allow_download: bool = True,
    batch_size: int = 128,
    max_length: int = 160,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. PHEME 9-Event Zero-Shot Topic Benchmark
    ph_splits = load_topic_splits_json(pheme_dir / "topic_splits.json")
    ph_contexts = _extract_early_thread_contexts(
        pheme_dir / "nodes.parquet",
        min_total_replies=15,
        k_prefix=10,
        splits=ph_splits,
    )
    ph_texts = [c["context_text"] for c in ph_contexts]
    ph_true = [c["topic_id"] for c in ph_contexts]
    ph_cand_ids = [t["id"] for t in PHEME_EVENT_TOPICS]
    ph_test_mask = np.array([c["split"] == "test" for c in ph_contexts], dtype=bool)

    ph_lex_probs, ph_lex_ms = _score_batch_lexical(ph_texts, PHEME_EVENT_TOPICS)
    ph_emb_probs, ph_emb_ms = _score_batch_encoder_cosine(
        ph_texts, PHEME_EVENT_TOPICS, ce_dir=ce_pheme_dir, device=device, batch_size=batch_size
    )
    ph_nli_probs, ph_nli_ms = _score_batch_mdeberta_nli(
        ph_texts, PHEME_EVENT_TOPICS, lang="en", device=device, batch_size=batch_size, max_length=max_length, allow_download=allow_download
    )

    ph_full_metrics = {}
    ph_test_metrics = {}
    for b_name, probs, ms in (
        ("lexical_ngram_fallback", ph_lex_probs, ph_lex_ms),
        ("transformer_cosine_embedding", ph_emb_probs, ph_emb_ms),
        ("mdeberta_zero_shot_nli", ph_nli_probs, ph_nli_ms),
    ):
        m_full = _classification_metrics(ph_true, probs, ph_cand_ids)
        m_full["ms_per_post"] = round(ms, 3)
        ph_full_metrics[b_name] = m_full

        test_true = [t for t, keep in zip(ph_true, ph_test_mask) if keep]
        m_test = _classification_metrics(test_true, probs[ph_test_mask], ph_cand_ids)
        m_test["ms_per_post"] = round(ms, 3)
        ph_test_metrics[b_name] = m_test

    # 2. VN Topic Classification + Cross-Backend Agreement + Kruskal-Wallis Stratification
    vn_splits = load_topic_splits_json(vn_dir / "topic_splits.json")
    vn_contexts = _extract_early_thread_contexts(
        vn_dir / "nodes.parquet",
        min_total_replies=25,
        k_prefix=20,
        splits=vn_splits,
    )
    vn_texts = [c["context_text"] for c in vn_contexts]
    vn_cand_ids = [t["id"] for t in DEFAULT_TOPICS_VI]
    vn_cand_labels = {t["id"]: t["label"] for t in DEFAULT_TOPICS_VI}

    vn_lex_probs, vn_lex_ms = _score_batch_lexical(vn_texts, DEFAULT_TOPICS_VI)
    vn_emb_probs, vn_emb_ms = _score_batch_encoder_cosine(
        vn_texts, DEFAULT_TOPICS_VI, ce_dir=ce_vn_dir, device=device, batch_size=batch_size
    )
    vn_nli_probs, vn_nli_ms = _score_batch_mdeberta_nli(
        vn_texts, DEFAULT_TOPICS_VI, lang="vi", device=device, batch_size=batch_size, max_length=max_length, allow_download=allow_download
    )

    vn_pred_lex = np.argmax(vn_lex_probs, axis=1)
    vn_pred_emb = np.argmax(vn_emb_probs, axis=1)
    vn_pred_nli = np.argmax(vn_nli_probs, axis=1)
    vn_test_mask = np.array([c["split"] == "test" for c in vn_contexts], dtype=bool)

    agreement = {
        "full_n": len(vn_contexts),
        "test_n": int(np.sum(vn_test_mask)),
        "mdeberta_vs_embedding_acc": float(accuracy_score(vn_pred_nli, vn_pred_emb)),
        "mdeberta_vs_embedding_kappa": float(cohen_kappa_score(vn_pred_nli, vn_pred_emb)),
        "mdeberta_vs_lexical_acc": float(accuracy_score(vn_pred_nli, vn_pred_lex)),
        "mdeberta_vs_lexical_kappa": float(cohen_kappa_score(vn_pred_nli, vn_pred_lex)),
        "latency_ms_per_post": {
            "lexical_ngram_fallback": round(vn_lex_ms, 3),
            "transformer_cosine_embedding": round(vn_emb_ms, 3),
            "mdeberta_zero_shot_nli": round(vn_nli_ms, 3),
        },
    }

    # 3. Full-tree multi-milestone trajectory & future_only polarization for VN & PHEME
    vn_dyn = _compute_real_trajectories_and_early_targets(vn_dir, k_early=20, min_total_replies=25)
    ph_dyn = _compute_real_trajectories_and_early_targets(pheme_dir, k_early=10, min_total_replies=15)

    vn_traj_summary = _aggregate_scale_trajectories(
        vn_dyn,
        scale_tiers=("medium", "large_ge_50", "mega_ge_100"),
        milestones=(5, 10, 20, 35, 50, 75, 100, 150),
    )
    ph_traj_summary = _aggregate_scale_trajectories(
        ph_dyn,
        scale_tiers=("small", "medium", "large_ge_50", "mega_ge_100"),
        milestones=(5, 10, 15, 20, 35, 50, 75, 100, 150),
    )

    # Topic stratification (using mDeBERTa Zero-Shot NLI assignments)
    def _stratify_by_topic(indices: Sequence[int]) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for i in indices:
            cid = vn_contexts[i]["cascade_id"]
            if cid not in vn_dyn:
                continue
            tid = vn_cand_ids[int(vn_pred_nli[i])]
            buckets[tid].append(vn_dyn[cid])

        per_topic: dict[str, Any] = {}
        kw_polar_groups: list[list[float]] = []
        kw_delta_groups: list[list[float]] = []
        for tid in vn_cand_ids:
            items = buckets.get(tid, [])
            if not items:
                continue
            polars = [float(it["y_polar_future"]) for it in items]
            deltas = [float(it["polar_delta"]) for it in items]
            reps = [int(it["num_replies"]) for it in items]
            esc = sum(1 for it in items if it["evolution_trend"] == "escalating")
            per_topic[tid] = {
                "label": vn_cand_labels[tid],
                "count": len(items),
                "mean_replies": float(np.mean(reps)),
                "mean_y_polar_future": float(np.mean(polars)),
                "mean_polar_delta": float(np.mean(deltas)),
                "escalating_rate": float(esc / len(items)),
            }
            if len(items) >= 3:
                kw_polar_groups.append(polars)
                kw_delta_groups.append(deltas)

        kw_polar_h, kw_polar_p = (
            kruskal(*kw_polar_groups) if len(kw_polar_groups) >= 2 else (float("nan"), float("nan"))
        )
        kw_delta_h, kw_delta_p = (
            kruskal(*kw_delta_groups) if len(kw_delta_groups) >= 2 else (float("nan"), float("nan"))
        )
        return {
            "topics": per_topic,
            "kruskal_y_polar_future": {
                "H": float(kw_polar_h),
                "p_value": float(kw_polar_p),
                "num_groups_ge3": len(kw_polar_groups),
            },
            "kruskal_polar_delta": {
                "H": float(kw_delta_h),
                "p_value": float(kw_delta_p),
                "num_groups_ge3": len(kw_delta_groups),
            },
        }

    vn_topic_strat = {
        "full_cohort": _stratify_by_topic(range(len(vn_contexts))),
        "test_split": _stratify_by_topic([i for i, keep in enumerate(vn_test_mask) if keep]),
    }

    # 4. VN K=20 Early-Warning Classification
    early_warning = _evaluate_vn_early_warning(vn_dir, gnn_vn_ckpt, device)

    results = {
        "is_real_data": True,
        "device": str(device),
        "provenance_note": (
            "PHEME event labels are human-verified news event ground truth (N=2,664 cohort, N=334 Test). "
            "VN polarization targets and discourse archetypes are silver-standard NLI-derived metrics "
            "anchored strictly on VN-Train quantile VN_POLAR_Q75 (zero Val/Test leakage)."
        ),
        "pheme_zero_shot_9_events": {
            "cohort_count": len(ph_contexts),
            "test_count": int(np.sum(ph_test_mask)),
            "num_events": len(PHEME_EVENT_TOPICS),
            "full_cohort": ph_full_metrics,
            "test_split": ph_test_metrics,
        },
        "vn_topic_agreement": agreement,
        "vn_topic_stratification": vn_topic_strat,
        "vn_large_post_trajectory": vn_traj_summary,
        "pheme_large_post_trajectory": ph_traj_summary,
        "vn_early_warning_k20": early_warning,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    _export_latex_and_figure(results, output_tex=output_tex, figures_dir=figures_dir)
    print(f"[EvalLargePostTopic] Saved {output_json}, {output_tex}, and fig6 in {figures_dir}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Zero-Shot Topic Classification & Large-Post Trajectory.")
    parser.add_argument("--vn-dir", type=Path, default=Path("data/processed/vn"))
    parser.add_argument("--pheme-dir", type=Path, default=Path("data/processed/pheme"))
    parser.add_argument("--ce-vn-dir", type=Path, default=Path("checkpoints/dual_ce_vn"))
    parser.add_argument("--ce-pheme-dir", type=Path, default=Path("checkpoints/dual_ce_pheme"))
    parser.add_argument("--gnn-vn-ckpt", type=Path, default=Path("checkpoints/best_sbifagat_vn.pt"))
    parser.add_argument("--output-json", type=Path, default=Path("results/results_large_post_topic.json"))
    parser.add_argument("--output-tex", type=Path, default=Path("results/tables_large_post_topic.tex"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures/vn"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--only-early-warning", action="store_true")
    args = parser.parse_args()

    if args.only_early_warning:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        existing = json.loads(args.output_json.read_text(encoding="utf-8"))
        existing["vn_early_warning_k20"] = _evaluate_vn_early_warning(args.vn_dir, args.gnn_vn_ckpt, dev)
        args.output_json.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        _export_latex_and_figure(existing, output_tex=args.output_tex, figures_dir=args.figures_dir)
        print(f"[EvalLargePostTopic] Updated vn_early_warning_k20 -> {args.output_json}", flush=True)
        return

    run_benchmark(
        vn_dir=args.vn_dir,
        pheme_dir=args.pheme_dir,
        ce_vn_dir=args.ce_vn_dir,
        ce_pheme_dir=args.ce_pheme_dir,
        gnn_vn_ckpt=args.gnn_vn_ckpt,
        output_json=args.output_json,
        output_tex=args.output_tex,
        figures_dir=args.figures_dir,
        allow_download=not args.no_download,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()

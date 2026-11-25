"""Profile parameter count, latency, and peak memory across all 9 SBi-FAGAT & baseline models,
plus optional End-to-End Live Inference profiling on raw cascade text.

NOTE: Downstream GNN complexity rows use controlled synthetic `CascadeData` tensor batches
purely to benchmark computational complexity (#Params, ms/cascade, peak RAM/VRAM) under
identical input shapes. End-to-End raw-text profiling (`--include-e2e`) times the real
graph-construction, SpanPool node encoding, edge scoring, and GNN forward stages on a
representative `(1 + K)`-node Vietnamese cascade. NEVER use synthetic outputs from this
profiler in paper accuracy/error tables.
"""

from __future__ import annotations

import argparse
from collections import deque
import ctypes
import json
import math
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arctic_shift_vn import build_vn_cascade_from_thread
from src.data.pyg_dataset import (
    CascadeData,
    STRUCTURAL_NODE_DIM,
    TEMPORAL_REF_SECONDS,
    TOTAL_NODE_FEATURE_DIM,
    _compute_full_tree_polarization,
    _compute_root_aligned_stances,
    compute_causal_structural_node_features,
    compute_early_cascade_burst_stats,
)
from src.models.baselines import (
    BiEdgeGATv2,
    BiGCN,
    EdgeGATv2,
    FAGCNBaseline,
    StandardGATv2,
    StandardGCN,
    TemporalStanceMLP,
    TextMLP,
)
from src.models.sbi_fagat import SBiFAGAT
from src.nlp.discourse_labels import _heuristic_behavior_fallback
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    TWITTER_ROBERTA_MODEL_NAME,
    VISOBERT_MODEL_NAME,
)
from src.nlp.extract_features import encode_edges_with_cross_encoder, encode_node_embeddings
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
                struct_x=torch.rand((num_nodes, STRUCTURAL_NODE_DIM), dtype=torch.float32),
                edge_index=edge_index_td,
                edge_index_td=edge_index_td,
                edge_index_bu=edge_index_bu,
                edge_attr=edge_attr,
                edge_stance=s_uv.unsqueeze(-1),
                cascade_stats=torch.rand((1, 8), dtype=torch.float32),
                root_index=torch.tensor([0], dtype=torch.long),
                y_size=torch.tensor([3.2], dtype=torch.float32),
                y_polar=torch.tensor([0.55], dtype=torch.float32),
                early_y_polar=torch.tensor([0.5], dtype=torch.float32),
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


def _get_process_rss_mb() -> float | None:
    """Best-effort query of current process Resident Set Size in MiB."""
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
        except Exception:
            return None
    statm = Path("/proc/self/statm")
    if statm.is_file():
        try:
            pages = int(statm.read_text(encoding="utf-8").split()[1])
            return float(pages * 4096) / (1024.0 * 1024.0)
        except Exception:
            return None
    return None


def _module_weights_mb(*modules: nn.Module | None) -> float:
    total_bytes = 0
    for mod in modules:
        if mod is None:
            continue
        for p in mod.parameters():
            total_bytes += int(p.numel()) * int(p.element_size())
        for b in mod.buffers():
            total_bytes += int(b.numel()) * int(b.element_size())
    return float(total_bytes) / (1024.0 * 1024.0)


def profile_e2e_raw_text_cascade(
    *,
    k_horizon: int = 20,
    hidden_dim: int = 64,
    edge_dim: int = 2,
    lang: str = "vi",
    ce_ckpt: Path | None = Path("checkpoints/dual_ce_vn"),
    gnn_ckpt: Path | None = Path("checkpoints/best_sbifagat_vn.pt"),
    node_backbone: str | None = None,
    use_cascade_stats: bool = True,
    predict_uncertainty: bool = False,
    warmup: int = 1,
    repeats: int = 3,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Measure genuine end-to-end live inference latency & memory on a (1+K)-node raw-text cascade."""
    from scripts.predict_live_thread import (
        _REQUIRED_CE_FILES,
        _deterministic_text_projection,
        _load_local_model_from_pretrained,
        _load_local_tokenizer,
        _make_demo_offline_thread,
    )

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    post, base_comments = _make_demo_offline_thread()
    comments = list(base_comments[:k_horizon])
    if len(comments) < k_horizon and base_comments:
        base_len = len(base_comments)
        last_ts = float(comments[-1]["created_utc"]) if comments else float(post["created_utc"])
        while len(comments) < k_horizon:
            idx_new = len(comments) + 1
            src_c = dict(base_comments[(idx_new - 1) % base_len])
            cid = f"c_ext_{idx_new}"
            src_c["id"] = cid
            src_c["fullname"] = f"t1_{cid}"
            src_c["parent_id"] = f"t3_{post['id']}"
            src_c["created_utc"] = last_ts + 30.0 * idx_new
            comments.append(src_c)

    ce_dir = Path(ce_ckpt) if ce_ckpt is not None else None
    ce_model: DualHeadCrossEncoder | None = None
    ce_tok: Any | None = None
    has_valid_ce = bool(
        ce_dir is not None
        and ce_dir.is_dir()
        and all((ce_dir / fname).is_file() for fname in _REQUIRED_CE_FILES)
        and (ce_dir / "dual_head_weights.pt").stat().st_size > 0
    )
    if has_valid_ce and ce_dir is not None:
        try:
            ce_model, ce_tok = DualHeadCrossEncoder.load_checkpoint(ce_dir, map_location="cpu")
            ce_model.to(target_device)
            ce_model.eval()
        except Exception:
            ce_model = None
            ce_tok = None
    cross_encoder_loaded = ce_model is not None and ce_tok is not None

    backbone_id = node_backbone or (VISOBERT_MODEL_NAME if lang == "vi" else TWITTER_ROBERTA_MODEL_NAME)
    node_enc = _load_local_model_from_pretrained(backbone_id, allow_network=False)
    node_tok: Any | None = None
    if node_enc is not None:
        node_tok = _load_local_tokenizer(
            backbone_id,
            ce_dir,
            lang=lang,
            node_backbone_overridden=node_backbone is not None,
            allow_network=False,
        )
        if node_tok is None:
            node_enc = None
    if node_enc is None and node_backbone is None and lang == "vi":
        vihsd_dir = Path("checkpoints/visobert_vihsd")
        if vihsd_dir.is_dir():
            node_enc = _load_local_model_from_pretrained(str(vihsd_dir), allow_network=False)
            if node_enc is not None:
                node_tok = _load_local_tokenizer(
                    str(vihsd_dir),
                    ce_dir,
                    lang=lang,
                    node_backbone_overridden=True,
                    allow_network=False,
                )
                if node_tok is None:
                    node_enc = None
    if node_enc is None and ce_model is not None and ce_tok is not None:
        node_enc = ce_model.encoder
        node_tok = ce_tok

    spanpool_loaded = node_enc is not None and node_tok is not None
    if spanpool_loaded and node_enc is not None:
        node_enc = node_enc.to(target_device)
        node_enc.eval()
        for p in node_enc.parameters():
            p.requires_grad_(False)
        spanpool_status = "real_local_transformer_spanpool"
    else:
        spanpool_status = "unavailable_offline_hash_fallback"

    gnn_ckpt_loaded = False
    gnn_model: SBiFAGAT
    if gnn_ckpt is not None and Path(gnn_ckpt).is_file():
        payload = torch.load(Path(gnn_ckpt), map_location="cpu", weights_only=False)
        cfg = payload.get("config", {})
        state_dict = payload["model_state_dict"]
        has_persist = (
            bool(cfg["use_polar_persistence_anchor"])
            if "use_polar_persistence_anchor" in cfg
            else ("head.persist_weight" in state_dict)
        )
        gnn_model = SBiFAGAT(
            in_dim=int(payload.get("in_dim", TOTAL_NODE_FEATURE_DIM)),
            hidden_dim=int(cfg.get("hidden_dim", hidden_dim)),
            edge_dim=int(cfg.get("edge_dim", edge_dim)),
            num_rumor_classes=int(payload.get("num_rumor_classes", 4)),
            eps=float(cfg.get("eps", 0.5)),
            dropout=float(cfg.get("dropout", 0.1)),
            use_cascade_stats=bool(cfg.get("use_cascade_stats", use_cascade_stats)),
            predict_uncertainty=bool(cfg.get("predict_uncertainty", predict_uncertainty)),
            use_polar_persistence_anchor=has_persist,
        )
        gnn_model.load_state_dict(state_dict, strict=True)
        gnn_ckpt_loaded = True
    else:
        gnn_model = SBiFAGAT(
            in_dim=TOTAL_NODE_FEATURE_DIM,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            use_cascade_stats=use_cascade_stats,
            predict_uncertainty=predict_uncertainty,
        )
    gnn_model = gnn_model.to(target_device)
    gnn_model.eval()
    target_edge_dim = int(getattr(gnn_model, "edge_dim", edge_dim))

    def _sync() -> None:
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)

    def _run_once() -> tuple[float, float, float, float, int, int]:
        # Stage 1: thread -> causal early graph & structural stats
        _sync()
        t_g0 = time.perf_counter()
        built = build_vn_cascade_from_thread(
            post,
            comments,
            min_comments=1,
            min_depth=1,
            min_vi_ratio=0.0,
            junk_mode="contract",
        )
        assert built is not None
        node_records, edge_records, _ = built
        node_by_id = {str(n["node_id"]): n for n in node_records}
        root_row = next(n for n in node_records if bool(n["is_root"]))
        root_id = str(root_row["node_id"])
        root_ts = root_row["created_at_utc"]
        root_text_str = str(root_row["text"])

        children_adj: dict[str, list[str]] = {str(n["node_id"]): [] for n in node_records}
        edge_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for e in edge_records:
            children_adj.setdefault(str(e["source_id"]), []).append(str(e["target_id"]))
            edge_lookup[(str(e["source_id"]), str(e["target_id"]))] = dict(e)
        depth_map: dict[str, int] = {root_id: 0}
        q: deque[str] = deque([root_id])
        while q:
            curr = q.popleft()
            for ch in children_adj.get(curr, []):
                if ch not in depth_map:
                    depth_map[ch] = depth_map[curr] + 1
                    q.append(ch)
        non_roots = [n for n in node_records if not bool(n["is_root"])]
        non_roots.sort(key=lambda n: (n["created_at_utc"], depth_map.get(str(n["node_id"]), 999), str(n["node_id"])))
        kept_nodes = [root_row] + non_roots[:k_horizon]
        kept_id_to_idx = {str(n["node_id"]): idx for idx, n in enumerate(kept_nodes)}
        delays_k = [max(0.0, (n["created_at_utc"] - root_ts).total_seconds()) for n in kept_nodes]
        struct_x = compute_causal_structural_node_features(
            kept_nodes=kept_nodes, delays_k=delays_k, depth_map=depth_map, k_horizon=k_horizon
        )

        kept_edge_records: list[dict[str, Any]] = []
        td_src: list[int] = []
        td_dst: list[int] = []
        kept_pairs: list[tuple[str, str]] = []
        for n in kept_nodes[1:]:
            dst_id = str(n["node_id"])
            src_id = str(n["parent_id"])
            if src_id not in kept_id_to_idx:
                continue
            td_src.append(kept_id_to_idx[src_id])
            td_dst.append(kept_id_to_idx[dst_id])
            kept_pairs.append((src_id, dst_id))
            kept_edge_records.append(edge_lookup.get((src_id, dst_id), {"edge_id": f"{src_id}->{dst_id}", "source_id": src_id, "target_id": dst_id}))
        _sync()
        graph_prep_ms = (time.perf_counter() - t_g0) * 1000.0

        # Stage 2: SpanPool node encoding
        _sync()
        t_n0 = time.perf_counter()
        denom_t = math.log1p(TEMPORAL_REF_SECONDS)
        if spanpool_loaded and node_enc is not None and node_tok is not None:
            kept_texts = [str(n["text"]) for n in kept_nodes]
            kept_parents = [
                None if (bool(n["is_root"]) or n.get("parent_id") is None) else str(node_by_id.get(str(n["parent_id"]), {}).get("text", ""))
                for n in kept_nodes
            ]
            kept_roots = [
                None if (bool(n["is_root"]) or n.get("parent_id") is None) else root_text_str
                for n in kept_nodes
            ]
            emb_mat = encode_node_embeddings(
                kept_texts,
                encoder=node_enc,
                tokenizer=node_tok,
                parent_texts=kept_parents,
                root_texts=kept_roots,
                lang=lang,
                batch_size=32,
                device=target_device,
            )
        else:
            emb_mat = torch.stack([_deterministic_text_projection(str(n["text"]), 768) for n in kept_nodes], dim=0)
        x_rows = []
        for idx_n, (n, dt_sec) in enumerate(zip(kept_nodes, delays_k)):
            nid = str(n["node_id"])
            t_norm = min(1.0, math.log1p(dt_sec) / denom_t)
            d_norm = min(1.0, float(depth_map.get(nid, 0)) / float(max(1, k_horizon)))
            x_rows.append(torch.cat([emb_mat[idx_n].detach().cpu(), torch.tensor([t_norm, d_norm], dtype=torch.float32)], dim=0))
        x_tensor = torch.stack(x_rows, dim=0)
        _sync()
        spanpool_ms = (time.perf_counter() - t_n0) * 1000.0

        # Stage 3: Edge scoring (DualHeadCrossEncoder if weights exist, else heuristic fallback)
        _sync()
        t_e0 = time.perf_counter()
        parent_texts = [str(node_by_id[src]["text"]) for src, _ in kept_pairs]
        reply_texts = [str(node_by_id[dst]["text"]) for _, dst in kept_pairs]
        edge_attr_rows: list[list[float]] = []
        early_stance_map: dict[str, float] = {}
        early_children_adj: dict[str, list[str]] = {}
        early_edge_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        if kept_pairs and cross_encoder_loaded and ce_model is not None and ce_tok is not None:
            ce_out = encode_edges_with_cross_encoder(
                parent_texts,
                reply_texts,
                cross_encoder=ce_model,
                tokenizer=ce_tok,
                root_texts=[root_text_str] * len(kept_pairs),
                include_root_context=bool(getattr(ce_model, "include_root_context", False)),
                lang=lang,
                batch_size=32,
                device=target_device,
            )
            for pos, ((src_s, dst_s), e_rec) in enumerate(zip(kept_pairs, kept_edge_records)):
                s_v = float(ce_out["s_uv"][pos])
                h_v = float(ce_out["h_uv"][pos])
                edge_attr_rows.append([s_v, h_v])
                eid_s = str(e_rec["edge_id"])
                early_children_adj.setdefault(src_s, []).append(dst_s)
                early_edge_lookup[(src_s, dst_s)] = {"edge_id": eid_s}
                early_stance_map[eid_s] = s_v
        else:
            for (src_s, dst_s), e_rec, r_txt in zip(kept_pairs, kept_edge_records, reply_texts):
                heur = _heuristic_behavior_fallback(r_txt)
                s_v = float(heur["s_uv"])
                h_v = float(heur["h_uv"])
                edge_attr_rows.append([s_v, h_v])
                eid_s = str(e_rec["edge_id"])
                early_children_adj.setdefault(src_s, []).append(dst_s)
                early_edge_lookup[(src_s, dst_s)] = {"edge_id": eid_s}
                early_stance_map[eid_s] = s_v

        cascade_stats = compute_early_cascade_burst_stats(
            kept_nodes=kept_nodes,
            delays_k=delays_k,
            depth_map=depth_map,
            edge_attr_rows=edge_attr_rows,
            k_horizon=k_horizon,
        )
        early_aligned = _compute_root_aligned_stances(
            root_id=root_id,
            children_adj=early_children_adj,
            edge_lookup=early_edge_lookup,
            stance_map=early_stance_map,
        )
        early_y_polar_val = _compute_full_tree_polarization(early_aligned, lambda_polar=0.5) if early_aligned else 0.0
        _sync()
        edge_ms = (time.perf_counter() - t_e0) * 1000.0

        # Stage 4: Downstream GNN forward pass
        _sync()
        t_m0 = time.perf_counter()
        edge_index_td = torch.tensor([td_src, td_dst], dtype=torch.long) if td_src else torch.empty((2, 0), dtype=torch.long)
        edge_index_bu = edge_index_td[[1, 0], :] if edge_index_td.numel() > 0 else edge_index_td
        edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float32) if edge_attr_rows else torch.empty((0, target_edge_dim), dtype=torch.float32)
        if edge_attr.numel() > 0:
            edge_attr = edge_attr[:, :target_edge_dim]
        data_obj = CascadeData(
            x=x_tensor,
            struct_x=struct_x,
            edge_index=edge_index_td,
            edge_index_td=edge_index_td,
            edge_index_bu=edge_index_bu,
            edge_attr=edge_attr,
            edge_stance=edge_attr[:, 0:1] if edge_attr.numel() > 0 else torch.empty((0, 1)),
            early_y_polar=torch.tensor([early_y_polar_val], dtype=torch.float32),
            cascade_stats=cascade_stats,
            root_index=torch.tensor([0], dtype=torch.long),
        ).to(target_device)
        with torch.no_grad():
            _ = gnn_model(data_obj)
        _sync()
        gnn_ms = (time.perf_counter() - t_m0) * 1000.0

        return graph_prep_ms, spanpool_ms, edge_ms, gnn_ms, len(kept_nodes), len(kept_pairs)

    for _ in range(max(1, warmup)):
        _run_once()

    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    else:
        tracemalloc.start()

    n_rep = max(1, repeats)
    sum_g = sum_n = sum_e = sum_m = 0.0
    num_nodes_obs = num_edges_obs = 0
    try:
        for _ in range(n_rep):
            g_ms, n_ms, e_ms, m_ms, num_nodes_obs, num_edges_obs = _run_once()
            sum_g += g_ms
            sum_n += n_ms
            sum_e += e_ms
            sum_m += m_ms
    finally:
        if target_device.type == "cuda":
            peak_alloc_mem_mb = float(torch.cuda.max_memory_allocated(target_device)) / (1024.0 * 1024.0)
        else:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_alloc_mem_mb = float(peak_bytes) / (1024.0 * 1024.0)

    avg_g = sum_g / float(n_rep)
    avg_n = sum_n / float(n_rep)
    avg_e = sum_e / float(n_rep)
    avg_m = sum_m / float(n_rep)
    total_ms = avg_g + avg_n + avg_e + avg_m

    unique_mods: list[nn.Module] = [gnn_model]
    if node_enc is not None:
        unique_mods.append(node_enc)
    if ce_model is not None and (node_enc is None or ce_model.encoder is not node_enc):
        unique_mods.append(ce_model)
    weights_mb = _module_weights_mb(*unique_mods)
    rss_mb = _get_process_rss_mb()

    if cross_encoder_loaded:
        edge_status = "real_dual_head_cross_encoder"
    else:
        edge_status = "missing_dual_head_weights_heuristic_fallback"

    return {
        "num_nodes": int(num_nodes_obs),
        "num_edges": int(num_edges_obs),
        "k_horizon": int(k_horizon),
        "device": str(target_device),
        "spanpool_loaded": bool(spanpool_loaded),
        "spanpool_status": spanpool_status,
        "cross_encoder_loaded": bool(cross_encoder_loaded),
        "edge_scorer_status": edge_status,
        "is_partial_without_dual_ce": not cross_encoder_loaded,
        "gnn_checkpoint_loaded": bool(gnn_ckpt_loaded),
        "gnn_hidden_dim": int(getattr(gnn_model, "hidden_dim", hidden_dim)),
        "graph_and_stats_ms": round(avg_g, 4),
        "spanpool_node_encoding_ms": round(avg_n, 4),
        "edge_scoring_ms": round(avg_e, 4),
        "cross_encoder_ms": round(avg_e, 4) if cross_encoder_loaded else None,
        "heuristic_edge_ms": None if cross_encoder_loaded else round(avg_e, 4),
        "gnn_forward_ms": round(avg_m, 4),
        "e2e_total_ms": round(total_ms, 4),
        "peak_alloc_mem_mb": round(peak_alloc_mem_mb, 3),
        "total_model_weights_mb": round(weights_mb, 2),
        "process_rss_mb": round(rss_mb, 2) if rss_mb is not None else None,
    }


def run_efficiency_benchmark(
    *,
    num_cascades: int = 32,
    k_horizon: int = 20,
    hidden_dim: int = 64,
    edge_dim: int = 2,
    use_cascade_stats: bool = False,
    predict_uncertainty: bool = False,
    num_classes: int = 4,
    warmup: int = 3,
    repeats: int = 10,
    include_e2e: bool = False,
    ce_ckpt: Path | None = Path("checkpoints/dual_ce_vn"),
    gnn_ckpt: Path | None = Path("checkpoints/best_sbifagat_vn.pt"),
    device: torch.device | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
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
            predict_uncertainty=predict_uncertainty,
        ),
        "TemporalStanceMLP": lambda: TemporalStanceMLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_classes,
            use_cascade_stats=use_cascade_stats,
            predict_uncertainty=predict_uncertainty,
        ),
        "StandardGCN": lambda: StandardGCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_rumor_classes=num_classes,
            use_cascade_stats=use_cascade_stats,
            predict_uncertainty=predict_uncertainty,
        ),
        "StandardGATv2": lambda: StandardGATv2(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_rumor_classes=num_classes,
            use_cascade_stats=use_cascade_stats,
            predict_uncertainty=predict_uncertainty,
        ),
        "EdgeGATv2": lambda: EdgeGATv2(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_classes,
            use_cascade_stats=use_cascade_stats,
            predict_uncertainty=predict_uncertainty,
        ),
        "BiEdgeGATv2": lambda: BiEdgeGATv2(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_rumor_classes=num_classes,
            use_cascade_stats=use_cascade_stats,
            predict_uncertainty=predict_uncertainty,
        ),
        "FAGCN": lambda: FAGCNBaseline(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_rumor_classes=num_classes,
            use_cascade_stats=use_cascade_stats,
            predict_uncertainty=predict_uncertainty,
        ),
        "BiGCN": lambda: BiGCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_rumor_classes=num_classes,
            use_cascade_stats=use_cascade_stats,
            predict_uncertainty=predict_uncertainty,
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
    if not include_e2e:
        return rows
    e2e_profile = profile_e2e_raw_text_cascade(
        k_horizon=k_horizon,
        hidden_dim=hidden_dim,
        edge_dim=edge_dim,
        ce_ckpt=ce_ckpt,
        gnn_ckpt=gnn_ckpt,
        use_cascade_stats=use_cascade_stats,
        predict_uncertainty=predict_uncertainty,
        warmup=max(1, min(warmup, 2)),
        repeats=max(1, min(repeats, 3)),
        device=target_device,
    )
    return {"models": rows, "e2e_raw_text_profile": e2e_profile}


def format_efficiency_markdown(rows: list[dict[str, Any]] | dict[str, Any]) -> str:
    row_list = rows.get("models", []) if isinstance(rows, dict) else rows
    mem_col = row_list[0]["mem_label"] if row_list else "Peak Mem (MB)"
    lines = [
        f"| Model | #Params | Infer (ms/cascade) | Train Step (ms/batch) | {mem_col} |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in row_list:
        lines.append(
            f"| {r['model']} | {r['params']:,} | {r['infer_ms_per_cascade']:.4f} | "
            f"{r['train_ms_per_batch']:.2f} | {r['peak_mem_mb']:.2f} |"
        )
    return "\n".join(lines)


def format_e2e_profile_markdown(e2e_profile: dict[str, Any]) -> str:
    ce_str = (
        f"{e2e_profile['cross_encoder_ms']:.2f} ms (DualHeadCrossEncoder)"
        if e2e_profile.get("cross_encoder_loaded") and e2e_profile.get("cross_encoder_ms") is not None
        else f"{e2e_profile['edge_scoring_ms']:.2f} ms (heuristic fallback; cross_encoder_loaded=False)"
    )
    rss_str = (
        f"{e2e_profile['process_rss_mb']:.2f} MB"
        if e2e_profile.get("process_rss_mb") is not None
        else "N/A"
    )
    lines = [
        "| E2E Stage (Raw Text -> Forecast) | Value | Notes |",
        "| :--- | :---: | :--- |",
        f"| Cascade Size | {e2e_profile['num_nodes']} nodes / {e2e_profile['num_edges']} edges | K={e2e_profile['k_horizon']} on {e2e_profile['device']} |",
        f"| 1. Graph & Causal Stats | {e2e_profile['graph_and_stats_ms']:.2f} ms | Path Contraction + 10D struct_x + 8D stats |",
        f"| 2. SpanPool Node Encoding | {e2e_profile['spanpool_node_encoding_ms']:.2f} ms | {e2e_profile['spanpool_status']} |",
        f"| 3. Edge Scoring | {ce_str} | {e2e_profile['edge_scorer_status']} |",
        f"| 4. Downstream GNN Forward | {e2e_profile['gnn_forward_ms']:.2f} ms | gnn_checkpoint_loaded={e2e_profile['gnn_checkpoint_loaded']} |",
        f"| **Total E2E Latency** | **{e2e_profile['e2e_total_ms']:.2f} ms** | is_partial_without_dual_ce={e2e_profile['is_partial_without_dual_ce']} |",
        f"| Model Weights / Peak Alloc / RSS | {e2e_profile['total_model_weights_mb']:.2f} / {e2e_profile['peak_alloc_mem_mb']:.3f} / {rss_str} | Weights MB / Peak Alloc MB / Process RSS |",
    ]
    return "\n".join(lines)


def format_efficiency_latex(
    rows: list[dict[str, Any]] | dict[str, Any],
    *,
    caption: str = "Computational efficiency benchmark (parameters, per-cascade latency, and peak memory).",
    label: str = "tab:efficiency",
) -> str:
    """Format computational efficiency rows into an IEEEtran `booktabs` LaTeX table."""
    row_list = rows.get("models", []) if isinstance(rows, dict) else rows
    mem_col = row_list[0]["mem_label"] if row_list else "Peak Mem (MB)"
    body_lines: list[str] = []
    for r in row_list:
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
            "\\resizebox{\\columnwidth}{!}{",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            f"Model & \\#Params & Infer (ms/cascade)$\\downarrow$ & Train Step (ms/batch)$\\downarrow$ & {mem_col}$\\downarrow$ \\\\",
            "\\midrule",
            *body_lines,
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table}",
            "",
        ]
    )


def _resolve_efficiency_table_label(output_table: Path) -> str:
    stem = output_table.stem.lower()
    if stem.startswith("tables_efficiency_") and len(stem) > len("tables_efficiency_"):
        suffix = stem.removeprefix("tables_efficiency_")
        return f"tab:efficiency_{suffix}"
    return "tab:efficiency"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Benchmark parameter count, inference/train latency, and memory across all 9 models + optional E2E raw-text profiling."
    )
    parser.add_argument("--num-cascades", type=int, default=32)
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--edge-dim", type=int, choices=[2, 4], default=2)
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--use-cascade-stats", action="store_true")
    parser.add_argument("--predict-uncertainty", action="store_true")
    parser.add_argument("--include-e2e", "--e2e", dest="include_e2e", action="store_true")
    parser.add_argument("--ce-ckpt", type=Path, default=Path("checkpoints/dual_ce_vn"))
    parser.add_argument("--gnn-ckpt", type=Path, default=Path("checkpoints/best_sbifagat_vn.pt"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-table", type=Path, default=None)
    args = parser.parse_args()

    use_stats = bool(args.use_cascade_stats or args.v2)
    use_unc = bool(args.predict_uncertainty or args.v2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[EFFICIENCY & COMPLEXITY PROFILING ONLY - SYNTHETIC SHAPES FOR GNN TABLE, REAL TEXT FOR --include-e2e]")
    print(f"Device: {device} | Cascades/batch={args.num_cascades} | K={args.k_horizon} | hidden_dim={args.hidden_dim}\n")

    bench_out = run_efficiency_benchmark(
        num_cascades=args.num_cascades,
        k_horizon=args.k_horizon,
        hidden_dim=args.hidden_dim,
        edge_dim=args.edge_dim,
        use_cascade_stats=use_stats,
        predict_uncertainty=use_unc,
        warmup=args.warmup,
        repeats=args.repeats,
        include_e2e=bool(args.include_e2e),
        ce_ckpt=args.ce_ckpt,
        gnn_ckpt=args.gnn_ckpt,
        device=device,
    )
    rows = bench_out["models"] if isinstance(bench_out, dict) else bench_out
    print("[Downstream GNN-Only Tensor Forward/Backward Latency]")
    print(format_efficiency_markdown(rows))

    if isinstance(bench_out, dict) and "e2e_raw_text_profile" in bench_out:
        print("\n[End-to-End Live Inference Latency (Raw Text -> SpanPool -> Edge Scorer -> GNN)]")
        print(format_e2e_profile_markdown(bench_out["e2e_raw_text_profile"]))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(bench_out, indent=2), encoding="utf-8")
        print(f"\nSaved efficiency profile JSON to: {args.output_json}")

    if args.output_table is not None:
        args.output_table.parent.mkdir(parents=True, exist_ok=True)
        eff_label = _resolve_efficiency_table_label(args.output_table)
        args.output_table.write_text(
            format_efficiency_latex(rows, label=eff_label), encoding="utf-8"
        )
        print(f"Saved efficiency LaTeX table to: {args.output_table}")


if __name__ == "__main__":
    main()

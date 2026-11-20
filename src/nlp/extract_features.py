"""Offline frozen NLP feature extractor exporting clean `.pt` feature banks.

Extracts:
  1. 768-dimensional semantic node embeddings `e_v` via leakage-free
     `Ancestor-Conditioned SpanPool` on the frozen backbone (`uitnlp/visobert`
     for Vietnamese, `cardiffnlp/twitter-roberta-base` for English PHEME).
  2. Leakage-free intra-K normalized temporal delay `t_norm` and tree depth
     `depth_norm`, concatenated into `node_features_770` (`[N_nodes, 770]`).
  3. 2D edge attribute vectors `e_uv = [s_uv, h_uv]` (`[N_edges, 2]`, or
     optional 4D) plus `edge_stance`, `edge_hostility`, `edge_oppose_flag`,
     and `edge_pred_label` dictionaries compatible with `EarlyCascadeDataset`.

All tensors are verified strictly finite (no NaN or Inf) before saving to
`data/processed/{pheme,vn}/`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import torch
from torch import Tensor
from transformers import AutoModel

from src.data.pyg_dataset import TEMPORAL_REF_SECONDS
from src.data.schema import load_cascade_parquet, save_cascade_parquet
from src.nlp.discourse_labels import DISCOURSE_BEHAVIOR_LABELS
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    HostilityTeacherScorer,
    NLIStanceTeacherScorer,
    STANCE_LABELS,
    TWITTER_ROBERTA_MODEL_NAME,
    VISOBERT_MODEL_NAME,
    _cuda_autocast,
    _effective_max_length,
    _encode_no_specials,
    _inspect_pair_tokenizer,
    _normalize_nfc,
    encode_cross_encoder_batch,
    load_tokenizer_safe,
    normalize_social_text,
)


def compute_masked_mean_pooling(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """Compute masked mean pooling in float32: `sum(m_i * h_i) / (sum(m_i) + 1e-8)`."""
    mask = attention_mask.unsqueeze(-1).to(dtype=torch.float32)
    hidden = last_hidden_state.to(dtype=torch.float32)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-8)
    return summed / counts


def _freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


@torch.no_grad()
def encode_node_embeddings(
    texts: Sequence[str],
    *,
    encoder: torch.nn.Module,
    tokenizer: Any,
    parent_texts: Sequence[str | None] | None = None,
    root_texts: Sequence[str | None] | None = None,
    lang: str = "vi",
    batch_size: int = 32,
    max_length: int = 256,
    min_target_tokens: int = 96,
    device: torch.device | None = None,
) -> Tensor:
    """Encode node texts into `[N, D]` float32 tensors using leakage-free Ancestor SpanPool.

    When `parent_texts` / `root_texts` are provided for a node `v`:
      - Root / standalone: `[CLS] v [SEP]` -> pool over `v`.
      - Depth 1 (`parent == root`): `[CLS] root [SEP] v [SEP]` (no duplication) -> pool strictly over `v`.
      - Depth >= 2: `[CLS] root [SEP] parent [SEP] v [SEP]` -> pool strictly over `v`.
    At least `min_target_tokens` (up to half of budget) are reserved for `v` so `v` is never truncated away.
    """
    n = len(texts)
    if parent_texts is not None and len(parent_texts) != n:
        raise ValueError("parent_texts must match texts length when provided")
    if root_texts is not None and len(root_texts) != n:
        raise ValueError("root_texts must match texts length when provided")
    hidden_dim = int(getattr(getattr(encoder, "config", None), "hidden_size", 768))
    if n == 0:
        return torch.empty((0, hidden_dim), dtype=torch.float32)

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _freeze_module(encoder).to(target_device)
    use_amp = target_device.type == "cuda"

    spec = _inspect_pair_tokenizer(tokenizer, encoder=encoder)
    pad_id = spec["pad_id"]
    cls_id = spec["cls_id"]
    sep_id = spec["sep_id"]
    num_specials = spec["num_specials"]
    eff_max = _effective_max_length(max_length, encoder=encoder, tokenizer=tokenizer)

    chunks: list[Tensor] = []
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        seqs: list[list[int]] = []
        span_masks: list[list[int]] = []

        for idx in range(start, end):
            t_norm = normalize_social_text(texts[idx], lang=lang)
            p_raw = parent_texts[idx] if parent_texts is not None else None
            r_raw = root_texts[idx] if root_texts is not None else None
            p_norm = normalize_social_text(p_raw, lang=lang) if p_raw else ""
            r_norm = normalize_social_text(r_raw, lang=lang) if r_raw else ""

            target_ids = _encode_no_specials(tokenizer, t_norm)
            if not p_norm and not r_norm:
                content_budget = max(1, eff_max - 2)
                target_keep = target_ids[:content_budget]
                seq = [cls_id] + target_keep + [sep_id]
                span = (
                    [0] + [1] * len(target_keep) + [0]
                    if target_keep
                    else [1] * len(seq)
                )
            else:
                content_budget = max(4, eff_max - num_specials)
                guaranteed_target = min(
                    len(target_ids),
                    max(1, min(int(min_target_tokens), content_budget // 2)),
                )
                ctx_budget = max(0, content_budget - guaranteed_target)

                if r_norm and p_norm and r_norm != p_norm and ctx_budget >= 3:
                    root_ids = _encode_no_specials(tokenizer, r_norm)
                    parent_ids = _encode_no_specials(tokenizer, p_norm)
                    root_cap = min(len(root_ids), max(1, ctx_budget // 3))
                    parent_cap = min(len(parent_ids), max(0, ctx_budget - 1 - root_cap))
                    root_keep = root_ids[: min(len(root_ids), max(0, ctx_budget - 1 - parent_cap))]
                    parent_keep = parent_ids[:parent_cap]
                    if root_keep and parent_keep:
                        ctx_ids = root_keep + [sep_id] + parent_keep
                    else:
                        ctx_ids = root_keep or parent_keep
                else:
                    ctx_str = p_norm or r_norm
                    ctx_ids = _encode_no_specials(tokenizer, ctx_str)[:ctx_budget]

                reply_budget = max(1, content_budget - len(ctx_ids))
                target_keep = target_ids[:reply_budget]
                mid_seps = [sep_id, sep_id] if num_specials >= 4 else [sep_id]
                seq = [cls_id] + ctx_ids + mid_seps + target_keep + [sep_id]
                span = (
                    [0] * (1 + len(ctx_ids) + len(mid_seps)) + [1] * len(target_keep) + [0]
                    if target_keep
                    else [1] * len(seq)
                )

            seqs.append(seq)
            span_masks.append(span)

        batch_max_len = max(len(s) for s in seqs)
        bsz = len(seqs)
        input_ids = torch.full((bsz, batch_max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, batch_max_len), dtype=torch.long)
        span_mask = torch.zeros((bsz, batch_max_len), dtype=torch.long)
        for row_i, (seq, sm) in enumerate(zip(seqs, span_masks)):
            cur_len = len(seq)
            input_ids[row_i, :cur_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[row_i, :cur_len] = 1
            span_mask[row_i, :cur_len] = torch.tensor(sm, dtype=torch.long)

        input_ids_dev = input_ids.to(target_device)
        attn_mask_dev = attention_mask.to(target_device)
        span_mask_dev = span_mask.to(target_device)

        with _cuda_autocast(enabled=use_amp):
            outputs = encoder(input_ids=input_ids_dev, attention_mask=attn_mask_dev)
            pooled = compute_masked_mean_pooling(outputs.last_hidden_state, span_mask_dev)
        if not torch.isfinite(pooled).all():
            with _cuda_autocast(enabled=False):
                outputs = encoder(input_ids=input_ids_dev, attention_mask=attn_mask_dev)
                pooled = compute_masked_mean_pooling(outputs.last_hidden_state, span_mask_dev)

        chunks.append(pooled.cpu().to(dtype=torch.float32))

    return torch.cat(chunks, dim=0)


@torch.no_grad()
def encode_edges_with_cross_encoder(
    parent_texts: Sequence[str],
    reply_texts: Sequence[str],
    *,
    cross_encoder: DualHeadCrossEncoder,
    tokenizer: Any,
    root_texts: Sequence[str | None] | None = None,
    include_root_context: bool = False,
    lang: str | None = None,
    batch_size: int = 32,
    max_length: int = 256,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run frozen segment-isolated `DualHeadCrossEncoder` inference on `(u, v)` edge text pairs."""
    n_edges = len(parent_texts)
    if n_edges != len(reply_texts):
        raise ValueError("parent_texts and reply_texts must have equal length")
    if root_texts is not None and len(root_texts) != n_edges:
        raise ValueError("root_texts must match parent_texts length when provided")
    ce_edge_dim = int(getattr(cross_encoder, "edge_dim", 2))
    if n_edges == 0:
        empty_res: dict[str, Any] = {
            "s_uv": np.zeros((0,), dtype=np.float32),
            "h_uv": np.zeros((0,), dtype=np.float32),
            "pred_label": [],
        }
        if ce_edge_dim == 4:
            empty_res["r_uv"] = np.zeros((0,), dtype=np.float32)
            empty_res["c_uv"] = np.zeros((0,), dtype=np.float32)
            empty_res["behavior_label"] = []
        return empty_res

    eff_lang = str(lang or getattr(cross_encoder, "lang", "vi") or "vi")
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _freeze_module(cross_encoder).to(target_device)
    use_amp = target_device.type == "cuda"

    all_s: list[Tensor] = []
    all_h: list[Tensor] = []
    all_r: list[Tensor] = []
    all_c: list[Tensor] = []
    all_probs: list[Tensor] = []
    all_b_probs: list[Tensor] = []

    for start in range(0, n_edges, batch_size):
        end = min(n_edges, start + batch_size)
        root_slice = root_texts[start:end] if root_texts is not None else None
        enc = encode_cross_encoder_batch(
            parent_texts[start:end],
            reply_texts[start:end],
            tokenizer=tokenizer,
            root_texts=root_slice,
            include_root_context=include_root_context,
            lang=eff_lang,
            max_length=max_length,
            encoder=cross_encoder.encoder,
        )
        enc_dev = {k: v.to(target_device) for k, v in enc.items()}
        with _cuda_autocast(enabled=use_amp):
            out = cross_encoder(**enc_dev)
        if not torch.isfinite(out["s_uv"]).all() or not torch.isfinite(out["h_uv"]).all():
            with _cuda_autocast(enabled=False):
                out = cross_encoder(**enc_dev)
        all_s.append(out["s_uv"].cpu().to(dtype=torch.float32))
        all_h.append(out["h_uv"].cpu().to(dtype=torch.float32))
        all_probs.append(out["stance_probs"].cpu().to(dtype=torch.float32))
        if ce_edge_dim == 4 and "r_uv" in out:
            all_r.append(out["r_uv"].cpu().to(dtype=torch.float32))
            all_c.append(out["c_uv"].cpu().to(dtype=torch.float32))
            all_b_probs.append(out["behavior_probs"].cpu().to(dtype=torch.float32))

    s_np = torch.cat(all_s, dim=0).numpy()
    h_np = torch.cat(all_h, dim=0).numpy()
    probs_np = torch.cat(all_probs, dim=0).numpy()
    labels = [STANCE_LABELS[int(i)] for i in probs_np.argmax(axis=-1)]
    res: dict[str, Any] = {"s_uv": s_np, "h_uv": h_np, "pred_label": labels}
    if ce_edge_dim == 4 and all_r:
        res["r_uv"] = torch.cat(all_r, dim=0).numpy()
        res["c_uv"] = torch.cat(all_c, dim=0).numpy()
        bp_np = torch.cat(all_b_probs, dim=0).numpy()
        res["behavior_label"] = [DISCOURSE_BEHAVIOR_LABELS[int(i)] for i in bp_np.argmax(axis=-1)]
    return res


def compute_leakage_free_node_metadata(
    nodes_table: pa.Table,
    edges_table: pa.Table,
    *,
    k_horizon: int = 20,
) -> Tensor:
    """Compute `[N_nodes, 2]` normalized `[t_norm, depth_norm]` without future-node leakage.

    For each cascade, `max_delay_k` used in temporal scaling is computed ONLY from
    the first `k_horizon` chronological replies + root (never peeking at replies
    beyond `K`), and clamped cleanly into `[0.0, 1.0]`.
    """
    node_rows = nodes_table.to_pylist()
    edge_rows = edges_table.to_pylist()
    id_to_global_idx = {str(r["node_id"]): idx for idx, r in enumerate(node_rows)}
    meta = torch.zeros((len(node_rows), 2), dtype=torch.float32)

    nodes_by_cascade: dict[str, list[dict[str, Any]]] = {}
    for row in node_rows:
        nodes_by_cascade.setdefault(str(row["cascade_id"]), []).append(row)

    edges_by_cascade: dict[str, list[dict[str, Any]]] = {}
    for row in edge_rows:
        edges_by_cascade.setdefault(str(row["cascade_id"]), []).append(row)

    for cid, c_nodes in nodes_by_cascade.items():
        c_edges = edges_by_cascade.get(cid, [])
        root_row = next(n for n in c_nodes if bool(n["is_root"]))
        root_id = str(root_row["node_id"])
        root_ts = root_row["created_at_utc"]

        adj: dict[str, list[str]] = {str(n["node_id"]): [] for n in c_nodes}
        for e in c_edges:
            adj.setdefault(str(e["source_id"]), []).append(str(e["target_id"]))

        depth_map: dict[str, int] = {root_id: 0}
        q: deque[str] = deque([root_id])
        while q:
            curr = q.popleft()
            for child in adj.get(curr, []):
                if child not in depth_map:
                    depth_map[child] = depth_map[curr] + 1
                    q.append(child)

        non_roots = [n for n in c_nodes if not bool(n["is_root"])]
        non_roots.sort(
            key=lambda n: (
                n["created_at_utc"],
                depth_map.get(str(n["node_id"]), 10**6),
                str(n["node_id"]),
            )
        )
        denom_t = math.log1p(TEMPORAL_REF_SECONDS)

        for n in c_nodes:
            nid = str(n["node_id"])
            g_idx = id_to_global_idx[nid]
            dt_sec = max(0.0, (n["created_at_utc"] - root_ts).total_seconds())
            t_norm = min(1.0, math.log1p(dt_sec) / denom_t)
            d_norm = min(1.0, float(depth_map.get(nid, 0)) / float(k_horizon))
            meta[g_idx, 0] = t_norm
            meta[g_idx, 1] = d_norm

    return meta


def validate_feature_tensors(payload: Mapping[str, Any]) -> None:
    """Strictly verify zero NaN/Inf values, tensor shapes, and value bounds."""
    node_emb = payload["node_embeddings"]
    edge_attr = payload["edge_attr"]

    if not isinstance(node_emb, Tensor) or node_emb.ndim != 2 or node_emb.size(1) != 768:
        raise ValueError(f"node_embeddings must have shape [N, 768], got {tuple(node_emb.shape)}")
    if not isinstance(edge_attr, Tensor) or edge_attr.ndim != 2 or edge_attr.size(1) not in {2, 4}:
        raise ValueError(f"edge_attr must have shape [E, 2] or [E, 4], got {tuple(edge_attr.shape)}")

    tensors_to_check: list[tuple[str, Tensor]] = [
        ("node_embeddings", node_emb),
        ("edge_attr", edge_attr),
    ]
    if "node_features_770" in payload:
        node_770 = payload["node_features_770"]
        if not isinstance(node_770, Tensor) or node_770.ndim != 2 or node_770.size(1) != 770:
            raise ValueError(f"node_features_770 must have shape [N, 770], got {tuple(node_770.shape)}")
        if node_emb.size(0) != node_770.size(0):
            raise ValueError("node_embeddings and node_features_770 must share the same row count")
        tensors_to_check.append(("node_features_770", node_770))

    for name, tensor in tensors_to_check:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Detected non-finite (NaN/Inf) entries inside '{name}'")

    if edge_attr.numel() > 0:
        s_vals = edge_attr[:, 0]
        h_vals = edge_attr[:, 1]
        if bool((s_vals < -1.0 - 1e-5).any() or (s_vals > 1.0 + 1e-5).any()):
            raise ValueError("edge_stance values out of bounds [-1, 1]")
        if bool((h_vals < -1e-5).any() or (h_vals > 1.0 + 1e-5).any()):
            raise ValueError("edge_hostility values out of bounds [0, 1]")
        if edge_attr.size(1) == 4:
            r_vals = edge_attr[:, 2]
            c_vals = edge_attr[:, 3]
            if bool((r_vals < -1e-5).any() or (r_vals > 1.0 + 1e-5).any()):
                raise ValueError("edge_irrationality values out of bounds [0, 1]")
            if bool((c_vals < -1e-5).any() or (c_vals > 1.0 + 1e-5).any()):
                raise ValueError("edge_constructive values out of bounds [0, 1]")

    for dict_name in ("edge_stance", "edge_hostility", "edge_irrationality", "edge_constructive"):
        mapping = payload.get(dict_name, {})
        for key, val in mapping.items():
            fval = float(val)
            if not math.isfinite(fval):
                raise ValueError(f"Non-finite value in '{dict_name}' for '{key}': {val!r}")

    if "node_root_stance" in payload:
        nrs = payload["node_root_stance"]
        if not isinstance(nrs, Mapping):
            raise ValueError("node_root_stance must be a mapping")
        for key, val in nrs.items():
            fval = float(val)
            if not math.isfinite(fval) or fval < -1.0 - 1e-5 or fval > 1.0 + 1e-5:
                raise ValueError(f"Invalid node_root_stance value for '{key}': {val!r}")


def extract_and_save_features(
    *,
    dataset: str,
    interim_dir: Path,
    output_dir: Path,
    cross_encoder_ckpt: Path | None = None,
    cross_encoder: DualHeadCrossEncoder | None = None,
    node_encoder: torch.nn.Module | None = None,
    tokenizer: Any | None = None,
    hostility_scorer: HostilityTeacherScorer | None = None,
    stance_scorer: NLIStanceTeacherScorer | None = None,
    scored_cache_json: Path | None = None,
    vihsd_ckpt: Path | None = None,
    edge_dim: int | None = None,
    include_root_context: bool = False,
    node_backbone: str | None = None,
    use_ancestor_span_pool: bool = True,
    batch_size: int = 32,
    max_length: int = 256,
    k_horizon: int = 20,
    device: torch.device | None = None,
    include_node_features_770: bool = False,
) -> Path:
    """Extract frozen 768-d node embeddings and 2D/4D edge attributes (`[s, h]` or `[s, h, r, c]`)."""
    if edge_dim is not None and edge_dim not in {2, 4}:
        raise ValueError(f"edge_dim must be 2 or 4, got {edge_dim}")
    lang = "en" if dataset == "pheme" else "vi"
    default_backbone = node_backbone or (
        TWITTER_ROBERTA_MODEL_NAME if lang == "en" else VISOBERT_MODEL_NAME
    )
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nodes_tbl, edges_tbl = load_cascade_parquet(interim_dir, validate=True)
    node_rows = nodes_tbl.to_pylist()
    edge_rows = edges_tbl.to_pylist()

    node_ids = [str(r["node_id"]) for r in node_rows]
    node_texts = [str(r["text"]) for r in node_rows]
    node_text_by_id = dict(zip(node_ids, node_texts))
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    cascade_root_id: dict[str, str] = {}
    for r in node_rows:
        if bool(r.get("is_root")) or int(r.get("depth", -1)) == 0:
            cascade_root_id[str(r["cascade_id"])] = str(r["node_id"])

    node_parent_texts: list[str | None] | None = None
    node_root_texts: list[str | None] | None = None
    if use_ancestor_span_pool:
        node_parent_texts = []
        node_root_texts = []
        for r in node_rows:
            if bool(r.get("is_root")) or r.get("parent_id") is None:
                node_parent_texts.append(None)
                node_root_texts.append(None)
            else:
                pid = str(r["parent_id"])
                rid = cascade_root_id.get(str(r["cascade_id"]), pid)
                node_parent_texts.append(node_text_by_id.get(pid))
                node_root_texts.append(node_text_by_id.get(rid))

    edge_ids = [str(r["edge_id"]) for r in edge_rows]
    edge_id_to_idx = {eid: idx for idx, eid in enumerate(edge_ids)}
    parent_texts = [node_text_by_id[str(r["source_id"])] for r in edge_rows]
    reply_texts = [node_text_by_id[str(r["target_id"])] for r in edge_rows]
    edge_target_ids = [str(r["target_id"]) for r in edge_rows]
    edge_root_ids = [
        cascade_root_id.get(str(r["cascade_id"]), str(r["source_id"])) for r in edge_rows
    ]
    edge_root_texts = [
        node_text_by_id.get(rid, node_text_by_id[str(r["source_id"])])
        for rid, r in zip(edge_root_ids, edge_rows)
    ]

    loaded_ckpt_model: DualHeadCrossEncoder | None = cross_encoder
    loaded_ckpt_tok: Any | None = tokenizer
    if loaded_ckpt_model is None and cross_encoder_ckpt is not None and Path(cross_encoder_ckpt).exists():
        loaded_ckpt_model, ckpt_tok = DualHeadCrossEncoder.load_checkpoint(
            Path(cross_encoder_ckpt),
            map_location="cpu",
        )
        if loaded_ckpt_tok is None:
            loaded_ckpt_tok = ckpt_tok

    # 1. Node 768-d semantic embeddings strictly via frozen independent encoder (zero cross-encoder leak)
    owns_node_encoder = False
    active_tok = tokenizer if tokenizer is not None else loaded_ckpt_tok
    if node_encoder is not None:
        active_encoder = node_encoder
    else:
        active_encoder = AutoModel.from_pretrained(default_backbone)
        owns_node_encoder = True
        if tokenizer is None:
            active_tok = load_tokenizer_safe(default_backbone)

    if active_tok is None:
        active_tok = load_tokenizer_safe(default_backbone)

    node_embeddings = encode_node_embeddings(
        node_texts,
        encoder=active_encoder,
        tokenizer=active_tok,
        parent_texts=node_parent_texts,
        root_texts=node_root_texts,
        lang=lang,
        batch_size=batch_size,
        max_length=max_length,
        device=target_device,
    )
    if owns_node_encoder:
        active_encoder.cpu()
        del active_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2. Optional 770-d compatibility concatenation
    node_features_770: Tensor | None = None
    if include_node_features_770:
        meta_2d = compute_leakage_free_node_metadata(
            nodes_tbl,
            edges_tbl,
            k_horizon=k_horizon,
        )
        node_features_770 = torch.cat([node_embeddings, meta_2d], dim=1)

    # 3. 2D or 4D Edge vectors & direct (root, reply) stance node_root_stance
    node_root_stance: dict[str, float] = {}
    r_arr: Any | None = None
    c_arr: Any | None = None
    behavior_labels: list[str] | None = None
    effective_edge_dim = int(edge_dim) if edge_dim is not None else 2

    cache_by_eid: dict[str, dict[str, Any]] = {}
    if scored_cache_json is not None and Path(scored_cache_json).exists():
        raw_cache = json.loads(Path(scored_cache_json).read_text(encoding="utf-8"))
        if not isinstance(raw_cache, list):
            raise ValueError(f"Expected JSON list in {scored_cache_json}")
        cache_by_eid = {
            str(item["edge_id"]): item
            for item in raw_cache
            if isinstance(item, dict) and item.get("edge_id") is not None
        }
    full_teacher_coverage = bool(
        edge_ids
        and cache_by_eid
        and all(
            eid in cache_by_eid and "s_uv" in cache_by_eid[eid] and "h_uv" in cache_by_eid[eid]
            for eid in edge_ids
        )
    )

    if loaded_ckpt_model is not None:
        # System-1 Student Cross-Encoder scores ALL edges consistently across Train/Val/Test
        ckpt_edge_dim = int(getattr(loaded_ckpt_model, "edge_dim", 2))
        if edge_dim is None:
            effective_edge_dim = ckpt_edge_dim
        elif int(edge_dim) == 4 and ckpt_edge_dim != 4:
            raise ValueError(
                f"Requested edge_dim=4, but Cross-Encoder checkpoint has edge_dim={ckpt_edge_dim}."
            )
        effective_root_ctx = bool(
            include_root_context or getattr(loaded_ckpt_model, "include_root_context", False)
        )
        edge_tok = loaded_ckpt_tok if loaded_ckpt_tok is not None else active_tok
        edge_pred = encode_edges_with_cross_encoder(
            parent_texts,
            reply_texts,
            cross_encoder=loaded_ckpt_model,
            tokenizer=edge_tok,
            root_texts=edge_root_texts,
            include_root_context=effective_root_ctx,
            lang=lang,
            batch_size=batch_size,
            max_length=max_length,
            device=target_device,
        )
        s_arr = edge_pred["s_uv"]
        h_arr = edge_pred["h_uv"]
        pred_labels = edge_pred["pred_label"]
        if effective_edge_dim == 4:
            r_arr = edge_pred.get("r_uv", np.zeros(len(edge_ids), dtype=np.float32))
            c_arr = edge_pred.get("c_uv", np.full(len(edge_ids), 0.5, dtype=np.float32))
            behavior_labels = edge_pred.get("behavior_label")
        deep_idx = [
            i for i, r in enumerate(edge_rows) if str(r["source_id"]) != edge_root_ids[i]
        ]
        for i in range(len(edge_ids)):
            node_root_stance[edge_target_ids[i]] = max(-1.0, min(1.0, float(s_arr[i])))
        has_full_teacher_root = bool(
            full_teacher_coverage
            and all("root_s_uv" in cache_by_eid[eid] for eid in edge_ids)
        )
        # Only use Teacher root_s_uv when Teacher cache covers 100% of edges;
        # partial caches avoid target drift between Train/Val and Test.
        if has_full_teacher_root:
            for idx, eid in enumerate(edge_ids):
                node_root_stance[edge_target_ids[idx]] = max(
                    -1.0, min(1.0, float(cache_by_eid[eid]["root_s_uv"]))
                )
        elif deep_idx:
            deep_pred = encode_edges_with_cross_encoder(
                [edge_root_texts[i] for i in deep_idx],
                [reply_texts[i] for i in deep_idx],
                cross_encoder=loaded_ckpt_model,
                tokenizer=edge_tok,
                include_root_context=False,
                lang=lang,
                batch_size=batch_size,
                max_length=max_length,
                device=target_device,
            )
            for pos, i in enumerate(deep_idx):
                node_root_stance[edge_target_ids[i]] = max(
                    -1.0, min(1.0, float(deep_pred["s_uv"][pos]))
                )
    elif scored_cache_json is not None:
        cache_file = Path(scored_cache_json)
        if not cache_file.exists():
            raise FileNotFoundError(f"Scored cache JSON not found: {cache_file}")
        missing_eids = [eid for eid in edge_ids if eid not in cache_by_eid]
        if missing_eids:
            raise ValueError(
                f"scored_cache_json {cache_file} is missing {len(missing_eids)} edges "
                f"(first: {missing_eids[0]!r}); re-run dual_edge_scorer --score-only."
            )
        has_4d_in_cache = any(
            "r_uv" in cache_by_eid[eid] or "c_uv" in cache_by_eid[eid]
            for eid in edge_ids
        )
        if edge_dim is None and has_4d_in_cache:
            effective_edge_dim = 4
        elif len(edge_ids) > 0 and effective_edge_dim == 4 and not has_4d_in_cache:
            raise ValueError(
                f"Requested edge_dim=4, but scored_cache_json {cache_file} lacks 4D r_uv/c_uv scores "
                "and no 4D Cross-Encoder checkpoint was provided."
            )

        s_list: list[float] = []
        h_list: list[float] = []
        r_list: list[float] = []
        c_list: list[float] = []
        pred_labels_list: list[str] = []
        beh_labels_list: list[str] = []
        for idx, eid in enumerate(edge_ids):
            entry = cache_by_eid[eid]
            s_val = float(entry["s_uv"])
            h_val = float(entry["h_uv"])
            s_list.append(s_val)
            h_list.append(h_val)
            if effective_edge_dim == 4:
                r_val = float(entry.get("r_uv", 0.05))
                c_val = float(entry.get("c_uv", 0.50))
                r_list.append(r_val)
                c_list.append(c_val)
                blbl = str(entry.get("behavior_label") or "")
                if blbl not in DISCOURSE_BEHAVIOR_LABELS:
                    if s_val > 0.15:
                        blbl = "support"
                    elif s_val < -0.15:
                        if h_val >= 0.65:
                            blbl = "hostile_conflict"
                        elif r_val >= 0.55:
                            blbl = "irrational_fallacy"
                        else:
                            blbl = "constructive_oppose"
                    else:
                        blbl = "neutral"
                beh_labels_list.append(blbl)
            if "pred_label" in entry and entry["pred_label"] is not None:
                pred_labels_list.append(str(entry["pred_label"]))
            else:
                lbl = "support" if s_val >= 0.1 else ("oppose" if s_val <= -0.1 else "neutral")
                pred_labels_list.append(lbl)
            root_s_raw = entry.get("root_s_uv", s_val)
            root_s_clamped = max(-1.0, min(1.0, float(root_s_raw)))
            node_root_stance[edge_target_ids[idx]] = root_s_clamped
        s_arr = s_list
        h_arr = h_list
        pred_labels = pred_labels_list
        if effective_edge_dim == 4:
            r_arr = r_list
            c_arr = c_list
            behavior_labels = beh_labels_list
    else:
        # Sequential teacher inference keeps peak VRAM minimal on 16GB GPUs
        if hostility_scorer is not None:
            h_arr = hostility_scorer.score_texts(
                reply_texts, batch_size=batch_size, max_length=max_length
            )
        else:
            if lang == "vi" and (vihsd_ckpt is None or not Path(vihsd_ckpt).exists()):
                raise FileNotFoundError(
                    f"Fine-tuned ViHSD checkpoint required for dataset='vn': {vihsd_ckpt}"
                )
            host_path = str(vihsd_ckpt) if lang == "vi" and vihsd_ckpt is not None else None
            tmp_host = HostilityTeacherScorer(lang=lang, model_name_or_path=host_path, device=target_device)
            h_arr = tmp_host.score_texts(reply_texts, batch_size=batch_size, max_length=max_length)
            del tmp_host
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        deep_idx = [
            i for i, r in enumerate(edge_rows) if str(r["source_id"]) != edge_root_ids[i]
        ]
        if stance_scorer is not None:
            nli_out = stance_scorer.score_pairs(
                parent_texts,
                reply_texts,
                batch_size=max(1, batch_size // 2),
                max_length=max_length,
                lang=lang,
            )
            root_nli_out = (
                stance_scorer.score_pairs(
                    [edge_root_texts[i] for i in deep_idx],
                    [reply_texts[i] for i in deep_idx],
                    batch_size=max(1, batch_size // 2),
                    max_length=max_length,
                    lang=lang,
                )
                if deep_idx
                else None
            )
        else:
            tmp_nli = NLIStanceTeacherScorer(lang=lang, device=target_device)
            nli_out = tmp_nli.score_pairs(
                parent_texts,
                reply_texts,
                batch_size=max(1, batch_size // 2),
                max_length=max_length,
                lang=lang,
            )
            root_nli_out = (
                tmp_nli.score_pairs(
                    [edge_root_texts[i] for i in deep_idx],
                    [reply_texts[i] for i in deep_idx],
                    batch_size=max(1, batch_size // 2),
                    max_length=max_length,
                    lang=lang,
                )
                if deep_idx
                else None
            )
            del tmp_nli
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        s_arr = nli_out["s_uv"]
        pred_labels = nli_out["pred_label"]
        for i in range(len(edge_ids)):
            node_root_stance[edge_target_ids[i]] = max(-1.0, min(1.0, float(s_arr[i])))
        if deep_idx and root_nli_out is not None:
            for pos, i in enumerate(deep_idx):
                node_root_stance[edge_target_ids[i]] = max(
                    -1.0, min(1.0, float(root_nli_out["s_uv"][pos]))
                )

    if len(edge_ids) > 0:
        s_t = torch.as_tensor(s_arr, dtype=torch.float32).clamp(-1.0, 1.0)
        h_t = torch.as_tensor(h_arr, dtype=torch.float32).clamp(0.0, 1.0)
        if effective_edge_dim == 4:
            r_t = (
                torch.as_tensor(r_arr, dtype=torch.float32).clamp(0.0, 1.0)
                if r_arr is not None
                else torch.zeros_like(h_t)
            )
            c_t = (
                torch.as_tensor(c_arr, dtype=torch.float32).clamp(0.0, 1.0)
                if c_arr is not None
                else torch.full_like(h_t, 0.5)
            )
            edge_attr = torch.stack([s_t, h_t, r_t, c_t], dim=-1)
        else:
            edge_attr = torch.stack([s_t, h_t], dim=-1)
    else:
        edge_attr = torch.empty((0, effective_edge_dim), dtype=torch.float32)

    edge_stance_map: dict[str, float] = {}
    edge_hostility_map: dict[str, float] = {}
    edge_irrational_map: dict[str, float] = {}
    edge_constructive_map: dict[str, float] = {}
    edge_oppose_flag: dict[str, bool] = {}
    edge_pred_label: dict[str, str] = {}
    edge_behavior_label: dict[str, str] = {}

    for idx, eid in enumerate(edge_ids):
        s_val = float(edge_attr[idx, 0].item())
        h_val = float(edge_attr[idx, 1].item())
        lbl = str(pred_labels[idx])
        edge_stance_map[eid] = s_val
        edge_hostility_map[eid] = h_val
        edge_oppose_flag[eid] = lbl == "oppose"
        edge_pred_label[eid] = lbl
        if effective_edge_dim == 4:
            edge_irrational_map[eid] = float(edge_attr[idx, 2].item())
            edge_constructive_map[eid] = float(edge_attr[idx, 3].item())
            if behavior_labels is not None and idx < len(behavior_labels):
                edge_behavior_label[eid] = str(behavior_labels[idx])

    payload: dict[str, Any] = {
        "dataset": dataset,
        "lang": lang,
        "edge_dim": effective_edge_dim,
        "k_horizon_metadata": int(k_horizon),
        "node_backbone": default_backbone,
        "node_pooling": "ancestor_span_pool" if use_ancestor_span_pool else "standalone_mean",
        "node_embeddings": node_embeddings,
        "node_id_to_idx": node_id_to_idx,
        "edge_attr": edge_attr,
        "edge_id_to_idx": edge_id_to_idx,
        "edge_stance": edge_stance_map,
        "edge_hostility": edge_hostility_map,
        "edge_oppose_flag": edge_oppose_flag,
        "edge_pred_label": edge_pred_label,
        "node_root_stance": node_root_stance,
    }
    if effective_edge_dim == 4:
        payload["edge_irrationality"] = edge_irrational_map
        payload["edge_constructive"] = edge_constructive_map
        if edge_behavior_label:
            payload["edge_behavior_label"] = edge_behavior_label
    if full_teacher_coverage:
        # Preserve isolated zero-shot Teacher edge scores only under 100% coverage
        t_stance: dict[str, float] = {}
        t_hostility: dict[str, float] = {}
        t_irrat: dict[str, float] = {}
        t_constr: dict[str, float] = {}
        t_oppose: dict[str, bool] = {}
        t_pred_lbl: dict[str, str] = {}
        t_beh_lbl: dict[str, str] = {}
        t_node_root_stance: dict[str, float] = {}
        for idx, eid in enumerate(edge_ids):
            entry = cache_by_eid[eid]
            sv = float(entry["s_uv"])
            hv = float(entry["h_uv"])
            t_stance[eid] = sv
            t_hostility[eid] = hv
            plbl = str(entry.get("pred_label") or ("support" if sv >= 0.1 else ("oppose" if sv <= -0.1 else "neutral")))
            t_pred_lbl[eid] = plbl
            t_oppose[eid] = plbl == "oppose"
            t_node_root_stance[edge_target_ids[idx]] = max(
                -1.0, min(1.0, float(entry.get("root_s_uv", sv)))
            )
            if "r_uv" in entry and "c_uv" in entry:
                t_irrat[eid] = float(entry["r_uv"])
                t_constr[eid] = float(entry["c_uv"])
                if entry.get("behavior_label") in DISCOURSE_BEHAVIOR_LABELS:
                    t_beh_lbl[eid] = str(entry["behavior_label"])
        if t_stance:
            payload["teacher_edge_stance"] = t_stance
            payload["teacher_edge_hostility"] = t_hostility
            payload["teacher_edge_oppose_flag"] = t_oppose
            payload["teacher_edge_pred_label"] = t_pred_lbl
            payload["teacher_node_root_stance"] = t_node_root_stance
            if len(t_irrat) == len(edge_ids) and len(t_constr) == len(edge_ids):
                payload["teacher_edge_irrationality"] = t_irrat
                payload["teacher_edge_constructive"] = t_constr
                if t_beh_lbl:
                    payload["teacher_edge_behavior_label"] = t_beh_lbl
    if node_features_770 is not None:
        payload["node_features_770"] = node_features_770

    validate_feature_tensors(payload)

    out_dir = Path(output_dir)
    save_cascade_parquet(nodes_tbl, edges_tbl, output_dir=out_dir, validate=True)
    src_splits = Path(interim_dir) / "topic_splits.json"
    if src_splits.exists():
        (out_dir / "topic_splits.json").write_text(src_splits.read_text(encoding="utf-8"), encoding="utf-8")
    pt_path = out_dir / "nlp_features.pt"
    torch.save(payload, pt_path)
    return pt_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract clean frozen NLP node embeddings & 2D/4D edge features into data/processed/."
    )
    parser.add_argument("--dataset", choices=["pheme", "vn"], required=True)
    parser.add_argument("--interim-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cross-encoder-ckpt", type=Path, default=None)
    parser.add_argument("--scored-cache-json", type=Path, default=None)
    parser.add_argument("--vihsd-dir", type=Path, default=Path("checkpoints/visobert_vihsd"))
    parser.add_argument("--edge-dim", type=int, choices=[2, 4], default=None)
    parser.add_argument("--node-backbone", type=str, default=None)
    parser.add_argument("--include-root-context", action="store_true")
    parser.add_argument("--disable-ancestor-span-pool", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--include-node-features-770", action="store_true")
    args = parser.parse_args()

    interim = args.interim_dir or Path(f"data/interim/{args.dataset}")
    output = args.output_dir or Path(f"data/processed/{args.dataset}")

    pt_path = extract_and_save_features(
        dataset=args.dataset,
        interim_dir=interim,
        output_dir=output,
        cross_encoder_ckpt=args.cross_encoder_ckpt,
        scored_cache_json=args.scored_cache_json,
        vihsd_ckpt=args.vihsd_dir,
        edge_dim=args.edge_dim,
        node_backbone=args.node_backbone,
        include_root_context=bool(args.include_root_context),
        use_ancestor_span_pool=not bool(args.disable_ancestor_span_pool),
        batch_size=args.batch_size,
        max_length=args.max_length,
        k_horizon=args.k_horizon,
        include_node_features_770=args.include_node_features_770,
    )
    print(f"Saved validated offline NLP feature bank: {pt_path}")


if __name__ == "__main__":
    main()

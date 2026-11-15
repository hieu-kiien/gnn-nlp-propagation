"""Offline frozen NLP feature extractor exporting clean `.pt` feature banks.

Extracts:
  1. 768-dimensional semantic node embeddings `e_v` via masked mean pooling on
     the frozen backbone (`uitnlp/visobert` for Vietnamese, `roberta-base` for PHEME).
  2. Leakage-free intra-K normalized temporal delay `t_norm` and tree depth
     `depth_norm`, concatenated into `node_features_770` (`[N_nodes, 770]`).
  3. 2D edge attribute vectors `e_uv = [s_uv, h_uv]` (`[N_edges, 2]`) plus
     `edge_stance`, `edge_hostility`, `edge_oppose_flag`, and `edge_pred_label`
     dictionaries compatible with `EarlyCascadeDataset`.

All tensors are verified strictly finite (no NaN or Inf) before saving to
`data/processed/{pheme,vn}/`.
"""

from __future__ import annotations

import argparse
import gc
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
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    HostilityTeacherScorer,
    NLIStanceTeacherScorer,
    ROBERTA_BASE_MODEL_NAME,
    STANCE_LABELS,
    VISOBERT_MODEL_NAME,
    _normalize_nfc,
    load_tokenizer_safe,
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
    batch_size: int = 32,
    max_length: int = 256,
    device: torch.device | None = None,
) -> Tensor:
    """Encode a list of node texts into `[N, 768]` float32 tensors using a frozen backbone."""
    if not texts:
        return torch.empty((0, 768), dtype=torch.float32)

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _freeze_module(encoder).to(target_device)
    use_amp = target_device.type == "cuda"

    chunks: list[Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = [_normalize_nfc(t) for t in texts[start : start + batch_size]]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc_dev = {k: v.to(target_device) for k, v in enc.items()}
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            outputs = encoder(**enc_dev)
            pooled = compute_masked_mean_pooling(
                outputs.last_hidden_state,
                enc_dev["attention_mask"],
            )
        chunks.append(pooled.cpu().to(dtype=torch.float32))

    return torch.cat(chunks, dim=0)


@torch.no_grad()
def encode_edges_with_cross_encoder(
    parent_texts: Sequence[str],
    reply_texts: Sequence[str],
    *,
    cross_encoder: DualHeadCrossEncoder,
    tokenizer: Any,
    batch_size: int = 32,
    max_length: int = 256,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run frozen `DualHeadCrossEncoder` inference on `(u, v)` edge text pairs."""
    n_edges = len(parent_texts)
    if n_edges != len(reply_texts):
        raise ValueError("parent_texts and reply_texts must have equal length")
    if n_edges == 0:
        return {
            "s_uv": np.zeros((0,), dtype=np.float32),
            "h_uv": np.zeros((0,), dtype=np.float32),
            "pred_label": [],
        }

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _freeze_module(cross_encoder).to(target_device)
    use_amp = target_device.type == "cuda"

    all_s: list[Tensor] = []
    all_h: list[Tensor] = []
    all_probs: list[Tensor] = []

    for start in range(0, n_edges, batch_size):
        u_batch = [_normalize_nfc(t) for t in parent_texts[start : start + batch_size]]
        v_batch = [_normalize_nfc(t) for t in reply_texts[start : start + batch_size]]
        enc = tokenizer(
            u_batch,
            v_batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc_dev = {k: v.to(target_device) for k, v in enc.items()}
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = cross_encoder(**enc_dev)
        all_s.append(out["s_uv"].cpu().to(dtype=torch.float32))
        all_h.append(out["h_uv"].cpu().to(dtype=torch.float32))
        all_probs.append(out["stance_probs"].cpu().to(dtype=torch.float32))

    s_np = torch.cat(all_s, dim=0).numpy()
    h_np = torch.cat(all_h, dim=0).numpy()
    probs_np = torch.cat(all_probs, dim=0).numpy()
    labels = [STANCE_LABELS[int(i)] for i in probs_np.argmax(axis=-1)]
    return {"s_uv": s_np, "h_uv": h_np, "pred_label": labels}


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
    if not isinstance(edge_attr, Tensor) or edge_attr.ndim != 2 or edge_attr.size(1) != 2:
        raise ValueError(f"edge_attr must have shape [E, 2], got {tuple(edge_attr.shape)}")

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

    for dict_name in ("edge_stance", "edge_hostility"):
        mapping = payload.get(dict_name, {})
        for key, val in mapping.items():
            fval = float(val)
            if not math.isfinite(fval):
                raise ValueError(f"Non-finite value in '{dict_name}' for '{key}': {val!r}")


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
    vihsd_ckpt: Path | None = None,
    batch_size: int = 32,
    max_length: int = 256,
    k_horizon: int = 20,
    device: torch.device | None = None,
    include_node_features_770: bool = False,
) -> Path:
    """Extract frozen 768-d node embeddings, optional 770-d early features, and 2D edge `[s_uv, h_uv]`."""
    lang = "en" if dataset == "pheme" else "vi"
    default_backbone = ROBERTA_BASE_MODEL_NAME if lang == "en" else VISOBERT_MODEL_NAME
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nodes_tbl, edges_tbl = load_cascade_parquet(interim_dir, validate=True)
    node_rows = nodes_tbl.to_pylist()
    edge_rows = edges_tbl.to_pylist()

    node_ids = [str(r["node_id"]) for r in node_rows]
    node_texts = [str(r["text"]) for r in node_rows]
    node_text_by_id = dict(zip(node_ids, node_texts))
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    edge_ids = [str(r["edge_id"]) for r in edge_rows]
    edge_id_to_idx = {eid: idx for idx, eid in enumerate(edge_ids)}
    parent_texts = [node_text_by_id[str(r["source_id"])] for r in edge_rows]
    reply_texts = [node_text_by_id[str(r["target_id"])] for r in edge_rows]

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

    # 3. 2D Edge vectors e_uv = [s_uv, h_uv]
    if loaded_ckpt_model is not None:
        edge_tok = loaded_ckpt_tok if loaded_ckpt_tok is not None else active_tok
        edge_pred = encode_edges_with_cross_encoder(
            parent_texts,
            reply_texts,
            cross_encoder=loaded_ckpt_model,
            tokenizer=edge_tok,
            batch_size=batch_size,
            max_length=max_length,
            device=target_device,
        )
        s_arr = edge_pred["s_uv"]
        h_arr = edge_pred["h_uv"]
        pred_labels = edge_pred["pred_label"]
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

        if stance_scorer is not None:
            nli_out = stance_scorer.score_pairs(
                parent_texts, reply_texts, batch_size=max(1, batch_size // 2), max_length=max_length
            )
        else:
            tmp_nli = NLIStanceTeacherScorer(device=target_device)
            nli_out = tmp_nli.score_pairs(
                parent_texts, reply_texts, batch_size=max(1, batch_size // 2), max_length=max_length
            )
            del tmp_nli
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        s_arr = nli_out["s_uv"]
        pred_labels = nli_out["pred_label"]

    if len(edge_ids) > 0:
        s_t = torch.as_tensor(s_arr, dtype=torch.float32).clamp(-1.0, 1.0)
        h_t = torch.as_tensor(h_arr, dtype=torch.float32).clamp(0.0, 1.0)
        edge_attr = torch.stack([s_t, h_t], dim=-1)
    else:
        edge_attr = torch.empty((0, 2), dtype=torch.float32)

    edge_stance_map: dict[str, float] = {}
    edge_hostility_map: dict[str, float] = {}
    edge_oppose_flag: dict[str, bool] = {}
    edge_pred_label: dict[str, str] = {}

    for idx, eid in enumerate(edge_ids):
        s_val = float(edge_attr[idx, 0].item())
        h_val = float(edge_attr[idx, 1].item())
        lbl = str(pred_labels[idx])
        edge_stance_map[eid] = s_val
        edge_hostility_map[eid] = h_val
        edge_oppose_flag[eid] = lbl == "oppose"
        edge_pred_label[eid] = lbl

    payload: dict[str, Any] = {
        "dataset": dataset,
        "lang": lang,
        "k_horizon_metadata": int(k_horizon),
        "node_embeddings": node_embeddings,
        "node_id_to_idx": node_id_to_idx,
        "edge_attr": edge_attr,
        "edge_id_to_idx": edge_id_to_idx,
        "edge_stance": edge_stance_map,
        "edge_hostility": edge_hostility_map,
        "edge_oppose_flag": edge_oppose_flag,
        "edge_pred_label": edge_pred_label,
    }
    if node_features_770 is not None:
        payload["node_features_770"] = node_features_770

    validate_feature_tensors(payload)

    out_dir = Path(output_dir)
    save_cascade_parquet(nodes_tbl, edges_tbl, output_dir=out_dir, validate=True)
    pt_path = out_dir / "nlp_features.pt"
    torch.save(payload, pt_path)
    return pt_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract clean frozen NLP node embeddings & 2D edge features into data/processed/."
    )
    parser.add_argument("--dataset", choices=["pheme", "vn"], required=True)
    parser.add_argument("--interim-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cross-encoder-ckpt", type=Path, default=None)
    parser.add_argument("--vihsd-dir", type=Path, default=Path("checkpoints/visobert_vihsd"))
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
        vihsd_ckpt=args.vihsd_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        k_horizon=args.k_horizon,
        include_node_features_770=args.include_node_features_770,
    )
    print(f"Saved validated offline NLP feature bank: {pt_path}")


if __name__ == "__main__":
    main()

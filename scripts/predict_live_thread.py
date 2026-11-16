"""Live Social Thread Predictor & Real-World Check Tool for SBi-FAGAT v2.

Accepts:
  - `--reddit-url https://www.reddit.com/r/vozforums/comments/<post_id>/...`
  - `--thread-id <reddit_post_id>` (via Arctic Shift REST API)
  - `--input-json path/to/thread.json`
  - `--demo-offline` (self-contained realistic 24-reply Vietnamese debate for testing)

Strictly truncates model inputs to the first `K` (default 20) chronological comments,
labels each reply across the 5 behavioral categories (`support`, `neutral`,
`constructive_oppose`, `irrational_fallacy`, `hostile_conflict`), runs `SBiFAGAT`,
and if the live thread already has `> K` replies, compares the early forecast against
the true full-thread outcome WITHOUT ever leaking post-K data into the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import requests
import torch

from src.data.arctic_shift_vn import (
    ARCTIC_SHIFT_API_BASE,
    build_vn_cascade_from_thread,
)
from src.data.pyg_dataset import (
    CASCADE_STATS_DIM,
    CascadeData,
    TEMPORAL_REF_SECONDS,
    TOTAL_NODE_FEATURE_DIM,
    _compute_full_tree_polarization,
    _compute_root_aligned_stances,
    compute_early_cascade_burst_stats,
)
from src.models.sbi_fagat import SBiFAGAT
from src.nlp.dual_edge_scorer import DualHeadCrossEncoder, STANCE_LABELS
from src.nlp.qwen_teacher import (
    BEHAVIOR_LABEL_VI,
    DISCOURSE_BEHAVIOR_LABELS,
    _heuristic_behavior_fallback,
)

_REDDIT_POST_ID_RE = re.compile(r"/comments/([A-Za-z0-9_]+)")


def _make_demo_offline_thread() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a deterministic 24-comment Vietnamese debate for zero-network verification."""
    base_ts = 1763200000.0
    post = {
        "id": "demo_vn_live01",
        "title": "Thảo luận đề xuất áp dụng phí vào nội đô đối với xe cá nhân giờ cao điểm",
        "selftext": (
            "Sở GTVT đang lấy ý kiến về phương án thu phí vào khu vực vành đai trong khung giờ "
            "cao điểm sáng chiều để giảm ùn tắc và tăng ngân sách cho xe buýt điện."
        ),
        "subreddit": "vozforums",
        "author": "op_demo",
        "created_utc": base_ts,
    }
    sample_comments = [
        ("c01", "t3_demo_vn_live01", 60, "Ủng hộ thử nghiệm ở vài tuyến trọng điểm trước, chứ sáng nào cũng kẹt cứng 45 phút không nhúc nhích nổi."),
        ("c02", "t3_demo_vn_live01", 110, "Mình chưa đồng ý lắm vì hiện tại mạng lưới metro và xe buýt kết nối ngõ hẻm còn yếu, người đi làm xa chưa có lựa chọn thay thế."),
        ("c03", "c01", 150, "Đồng ý với bác, trước mắt nên trích toàn bộ tiền thu phí này bù giá vé tháng cho tuyến xe buýt điện mới hợp lý."),
        ("c04", "t3_demo_vn_live01", 210, "Đỉnh quá cơ =)) hay thu luôn phí hít thở không khí cho ngân sách dồi dào, chắc mấy ông ra đề án đi trực thăng quen rồi."),
        ("c05", "c01", 250, "Não tàn mới đi bênh cái đề xuất móc túi dân nghèo này, đi làm lương ba cọc ba đồng mà gặp mấy thành phần lý thuyết suông phát bực!"),
        ("c06", "t3_demo_vn_live01", 300, "Cho mình hỏi xe máy điện hay sinh viên ở trọ trong vùng lõi có được miễn giảm theo biển số đăng ký không?"),
        ("c07", "c02", 340, "Chuẩn rồi bạn, nếu chưa mở rộng gấp đôi tần suất xe buýt mà thu phí ngay thì chỉ dồn áp lực chi phí lên người lao động thuê trọ."),
        ("c08", "c05", 390, "Bạn không nên công kích cá nhân như vậy, người ta chỉ bàn góc độ thử nghiệm thí điểm chứ có ai cấm bạn góp ý đâu."),
        ("c09", "c04", 430, "Cứ động tới quyền lợi cá nhân là nhảy dựng lên chửi đổng, đúng là văn hóa bàn phím rác rưởi không biết tiếp thu!"),
        ("c10", "t3_demo_vn_live01", 470, "Về mặt kỹ thuật camera nhận diện biển số ở các nút giao liệu đã đồng bộ thu phí tự động không dừng chưa mọi người?"),
        ("c11", "c07", 510, "Theo mình tham khảo mô hình bên Singapore hay London, họ phải phủ dày tàu điện trước ít nhất 5 năm rồi mới dám thu phí nội đô."),
        ("c12", "t3_demo_vn_live01", 550, "Tất cả tại mấy ông quy hoạch khu đô thị dồn hết vào trung tâm chứ xe máy có tội tình gì mà lôi ra thu phí."),
        ("c13", "c03", 590, "Mình thấy hợp lý nếu chỉ thu ô tô con dưới 9 chỗ trước, vì chiếm diện tích mặt đường lớn nhất mà chở có 1-2 người."),
        ("c14", "c09", 620, "Cút về mà đóng phí một mình đi đồ ngu, nói chuyện kiểu trịch thượng dạy đời nghe ngứa mắt thực sự!"),
        ("c15", "c13", 660, "Phương án tách riêng ô tô con giờ cao điểm như bác nói có cơ sở hơn hẳn, vừa giảm chiếm chỗ vừa ít ảnh hưởng lao động phổ thông."),
        ("c16", "c11", 700, "Đúng thế, nếu làm cuốn chiếu theo từng tuyến đường có sẵn tàu điện trên cao như Cát Linh - Hà Đông thì dân sẽ đồng thuận hơn."),
        ("c17", "c12", 740, "Hỏi thật chứ tiền thu về có minh bạch được không hay lại rơi vào vòng luẩn quẩn vẽ dự án rồi biến mất?"),
        ("c18", "t3_demo_vn_live01", 780, "Tán thành hướng điều tiết giao thông bằng kinh tế kết hợp đầu tư giao thông công cộng, nhưng phải công khai lộ trình ít nhất 2 năm."),
        ("c19", "c14", 810, "Thôi lượn đi cho nước nó trong, tranh luận toàn chửi bới văng tục thế thì giải quyết được cái gì."),
        ("c20", "c10", 850, "Hiện đã có hệ thống VETC/ePass khá ổn định cho ô tô rồi, chỉ khó phần kiểm soát xe ngoại tỉnh đi qua các đường nhánh."),
        # Future comments (> 20) used purely for post-hoc ground-truth comparison
        ("c21", "c18", 920, "Lộ trình 2 năm kèm nâng cấp xe buýt như bác đề xuất là hài hòa nhất giữa lợi ích công cộng và thu nhập người dân."),
        ("c22", "c05", 980, "Nói chung tụi ủng hộ toàn ngồi văn phòng máy lạnh, biết quái gì cảnh kẹt xe nắng nóng mà bày đặt."),
        ("c23", "c16", 1040, "Tôi vẫn giữ quan điểm nên thí điểm một chiều trên 2 trục chính trước để đo số liệu lưu lượng thực tế."),
        ("c24", "c22", 1100, "Thôi cạn lời với mấy thành phần chụp mũ, nói chuyện không nổi một câu dẫn chứng tử tế."),
    ]
    comments = [
        {
            "fullname": f"t1_{cid}",
            "id": cid,
            "link_id": "t3_demo_vn_live01",
            "parent_id": pid if pid.startswith(("t3_", "t1_")) else f"t1_{pid}",
            "body": body,
            "text": body,
            "created_utc": base_ts + dt,
            "author": f"user_{idx}",
            "author_hash": hashlib.sha256(f"user_{idx}".encode("utf-8")).hexdigest(),
        }
        for idx, (cid, pid, dt, body) in enumerate(sample_comments, start=1)
    ]
    return post, comments


def _extract_post_id_from_url(url: str) -> str:
    m = _REDDIT_POST_ID_RE.search(url)
    if m:
        return m.group(1)
    cleaned = url.rstrip("/").split("/")[-1]
    if cleaned.startswith("t3_"):
        return cleaned[3:]
    return cleaned


def _fetch_reddit_thread(post_id: str, timeout: int = 20) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fetch submission + comments via Arctic Shift API with Reddit public JSON fallback."""
    headers = {"User-Agent": "SBi-FAGAT-LiveCheck/2.0"}
    try:
        sub_resp = requests.get(
            f"{ARCTIC_SHIFT_API_BASE}/posts/search",
            params={"ids": post_id},
            headers=headers,
            timeout=timeout,
        )
        sub_resp.raise_for_status()
        sub_data = sub_resp.json().get("data") or []
        if sub_data:
            post = sub_data[0]
            com_resp = requests.get(
                f"{ARCTIC_SHIFT_API_BASE}/comments/search",
                params={"link_id": f"t3_{post_id}", "limit": 500, "sort": "asc"},
                headers=headers,
                timeout=timeout,
            )
            com_resp.raise_for_status()
            raw_comments = com_resp.json().get("data") or []
            comments = []
            for c in raw_comments:
                cid = str(c.get("id") or "").removeprefix("t1_")
                comments.append(
                    {
                        "fullname": f"t1_{cid}",
                        "id": cid,
                        "link_id": str(c.get("link_id") or f"t3_{post_id}"),
                        "parent_id": str(c.get("parent_id") or f"t3_{post_id}"),
                        "body": str(c.get("body") or c.get("text") or ""),
                        "text": str(c.get("body") or c.get("text") or ""),
                        "created_utc": c.get("created_utc"),
                        "author": c.get("author"),
                        "author_hash": hashlib.sha256(str(c.get("author") or cid).encode("utf-8")).hexdigest(),
                    }
                )
            return post, comments
    except Exception:
        pass

    # Fallback: official public Reddit thread JSON endpoint
    json_url = f"https://www.reddit.com/comments/{post_id}.json?limit=500"
    resp = requests.get(json_url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    post_raw = payload[0]["data"]["children"][0]["data"]

    flat_comments: list[dict[str, Any]] = []

    def _walk_listing(children: Sequence[Mapping[str, Any]]) -> None:
        for child in children:
            if child.get("kind") != "t1":
                continue
            d = child.get("data", {})
            cid = str(d.get("id") or "")
            flat_comments.append(
                {
                    "fullname": f"t1_{cid}",
                    "id": cid,
                    "link_id": f"t3_{post_id}",
                    "parent_id": str(d.get("parent_id") or f"t3_{post_id}"),
                    "body": str(d.get("body") or ""),
                    "text": str(d.get("body") or ""),
                    "created_utc": d.get("created_utc"),
                    "author": d.get("author"),
                    "author_hash": hashlib.sha256(str(d.get("author") or cid).encode("utf-8")).hexdigest(),
                }
            )
            replies = d.get("replies")
            if isinstance(replies, Mapping):
                _walk_listing(replies.get("data", {}).get("children", []))

    _walk_listing(payload[1]["data"]["children"])
    return post_raw, flat_comments


def _load_thread_from_json(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list) and len(raw) >= 2:
        # Raw Reddit listing JSON
        post = raw[0]["data"]["children"][0]["data"]
        flat: list[dict[str, Any]] = []
        post_id = str(post.get("id") or "local_thread")

        def _walk(children: Sequence[Mapping[str, Any]]) -> None:
            for ch in children:
                if ch.get("kind") != "t1":
                    continue
                d = ch.get("data", {})
                cid = str(d.get("id") or "")
                flat.append(
                    {
                        "fullname": f"t1_{cid}",
                        "id": cid,
                        "link_id": f"t3_{post_id}",
                        "parent_id": str(d.get("parent_id") or f"t3_{post_id}"),
                        "text": str(d.get("body") or ""),
                        "created_utc": d.get("created_utc"),
                        "author_hash": hashlib.sha256(str(d.get("author") or cid).encode("utf-8")).hexdigest(),
                    }
                )
                rep = d.get("replies")
                if isinstance(rep, Mapping):
                    _walk(rep.get("data", {}).get("children", []))

        _walk(raw[1]["data"]["children"])
        return post, flat

    if isinstance(raw, Mapping):
        post_id = str(raw.get("id") or "custom_live")
        base_ts = float(raw.get("created_utc") or 1763200000.0)
        post = {
            "id": post_id,
            "title": str(raw.get("title") or ""),
            "selftext": str(raw.get("text") or raw.get("selftext") or ""),
            "subreddit": str(raw.get("subreddit") or "custom"),
            "author": str(raw.get("author") or "op"),
            "created_utc": base_ts,
        }
        comments = []
        for idx, c in enumerate(raw.get("comments", []), start=1):
            cid = str(c.get("id") or f"c{idx}")
            pid = str(c.get("parent_id") or f"t3_{post_id}")
            if not pid.startswith(("t3_", "t1_")):
                pid = f"t3_{post_id}" if pid == post_id else f"t1_{pid}"
            comments.append(
                {
                    "fullname": f"t1_{cid.removeprefix('t1_')}",
                    "id": cid.removeprefix("t1_"),
                    "link_id": f"t3_{post_id}",
                    "parent_id": pid,
                    "text": str(c.get("text") or c.get("body") or ""),
                    "created_utc": float(c.get("created_utc") or (base_ts + idx * 45.0)),
                    "author": str(c.get("author") or f"u_{idx}"),
                    "author_hash": hashlib.sha256(str(c.get("author") or cid).encode("utf-8")).hexdigest(),
                }
            )
        return post, comments

    raise ValueError(f"Unsupported thread JSON structure in {path}")


def _deterministic_text_projection(text: str, dim: int = 768) -> torch.Tensor:
    """Fast deterministic n-gram hashed embedding when running checkpoint-free demo."""
    vec = np.zeros((dim,), dtype=np.float32)
    tokens = text.lower().split()
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return torch.from_numpy(vec)


def predict_live_cascade(
    post: Mapping[str, Any],
    comments: Sequence[Mapping[str, Any]],
    *,
    k_horizon: int = 20,
    cross_encoder_ckpt: Path | None = None,
    gnn_ckpt: Path | None = None,
    lang: str = "vi",
) -> dict[str, Any]:
    """Build cascade graph, score early behavior, run SBiFAGAT, and emit live forecast report."""
    built = build_vn_cascade_from_thread(
        post,
        comments,
        min_comments=1,
        min_depth=1,
        min_vi_ratio=0.0,
    )
    if built is None:
        raise ValueError("Thread has insufficient valid comments after spam/deleted filtering.")
    node_records, edge_records, _meta = built

    node_by_id = {str(n["node_id"]): n for n in node_records}
    root_row = next(n for n in node_records if bool(n["is_root"]))
    root_id = str(root_row["node_id"])
    root_ts = root_row["created_at_utc"]

    # Depth BFS
    children_adj: dict[str, list[str]] = {str(n["node_id"]): [] for n in node_records}
    edge_lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for e in edge_records:
        children_adj.setdefault(str(e["source_id"]), []).append(str(e["target_id"]))
        edge_lookup[(str(e["source_id"]), str(e["target_id"]))] = e

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
    kept_ids = {str(n["node_id"]) for n in kept_nodes}
    kept_id_to_idx = {str(n["node_id"]): idx for idx, n in enumerate(kept_nodes)}

    # Score edges (all edges if we want post-hoc actual polar comparison; strictly isolated to edge_map)
    ce_model: DualHeadCrossEncoder | None = None
    ce_tok: Any | None = None
    if cross_encoder_ckpt is not None and Path(cross_encoder_ckpt).exists():
        ce_model, ce_tok = DualHeadCrossEncoder.load_checkpoint(Path(cross_encoder_ckpt), map_location="cpu")
        ce_model.eval()

    edge_scores: dict[str, dict[str, Any]] = {}
    for e in edge_records:
        eid = str(e["edge_id"])
        p_txt = str(node_by_id[str(e["source_id"])]["text"])
        r_txt = str(node_by_id[str(e["target_id"])]["text"])
        if ce_model is not None and ce_tok is not None:
            enc = ce_tok([p_txt], [r_txt], padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                out = ce_model(enc["input_ids"], enc["attention_mask"])
            s_v = float(out["s_uv"][0].item())
            h_v = float(out["h_uv"][0].item())
            r_v = float(out["r_uv"][0].item()) if "r_uv" in out else 0.10
            c_v = float(out["c_uv"][0].item()) if "c_uv" in out else 0.50
            if "behavior_probs" in out:
                blbl = DISCOURSE_BEHAVIOR_LABELS[int(out["behavior_probs"][0].argmax().item())]
            else:
                blbl = "support" if s_v > 0.1 else ("hostile_conflict" if h_v > 0.6 else ("constructive_oppose" if s_v < -0.1 else "neutral"))
        else:
            heur = _heuristic_behavior_fallback(r_txt)
            s_v = float(heur["s_uv"])
            h_v = float(heur["h_uv"])
            r_v = float(heur["r_uv"])
            c_v = float(heur["c_uv"])
            blbl = str(heur["label"])
        edge_scores[eid] = {"s_uv": s_v, "h_uv": h_v, "r_uv": r_v, "c_uv": c_v, "behavior_label": blbl}

    # Build early PyG CascadeData restricted strictly to V_K
    delays_k = [max(0.0, (n["created_at_utc"] - root_ts).total_seconds()) for n in kept_nodes]
    denom_t = math.log1p(TEMPORAL_REF_SECONDS)
    x_rows = []
    for n, dt_sec in zip(kept_nodes, delays_k):
        nid = str(n["node_id"])
        if ce_model is not None and ce_tok is not None:
            enc_n = ce_tok([str(n["text"])], padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                emb_768 = ce_model.encoder(input_ids=enc_n["input_ids"], attention_mask=enc_n["attention_mask"]).last_hidden_state[:, 0, :].squeeze(0)
        else:
            emb_768 = _deterministic_text_projection(str(n["text"]), 768)
        t_norm = min(1.0, math.log1p(dt_sec) / denom_t)
        d_norm = min(1.0, float(depth_map.get(nid, 0)) / float(max(1, k_horizon)))
        x_rows.append(torch.cat([emb_768.cpu(), torch.tensor([t_norm, d_norm], dtype=torch.float32)], dim=0))
    x = torch.stack(x_rows, dim=0)

    td_src, td_dst = [], []
    edge_attr_rows: list[list[float]] = []
    kept_pairs: list[tuple[str, str]] = []
    kept_edge_details: list[dict[str, Any]] = []
    for n in kept_nodes[1:]:
        dst_id = str(n["node_id"])
        src_id = str(n["parent_id"])
        if src_id not in kept_id_to_idx:
            continue
        edge_row = edge_lookup.get((src_id, dst_id))
        eid = str(edge_row["edge_id"]) if edge_row is not None else f"{src_id}->{dst_id}"
        sc = edge_scores[eid]
        td_src.append(kept_id_to_idx[src_id])
        td_dst.append(kept_id_to_idx[dst_id])
        edge_attr_rows.append([sc["s_uv"], sc["h_uv"], sc["r_uv"], sc["c_uv"]])
        kept_pairs.append((src_id, dst_id))
        kept_edge_details.append(
            {
                "edge_id": eid,
                "parent_id": src_id,
                "node_id": dst_id,
                "reply_text": str(n["text"]),
                **sc,
            }
        )

    edge_index_td = torch.tensor([td_src, td_dst], dtype=torch.long) if td_src else torch.empty((2, 0), dtype=torch.long)
    edge_index_bu = edge_index_td[[1, 0], :] if edge_index_td.numel() > 0 else edge_index_td
    edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float32) if edge_attr_rows else torch.empty((0, 4), dtype=torch.float32)
    cascade_stats = compute_early_cascade_burst_stats(
        kept_nodes=kept_nodes,
        delays_k=delays_k,
        depth_map=depth_map,
        edge_attr_rows=edge_attr_rows,
        k_horizon=k_horizon,
    )

    data = CascadeData(
        x=x,
        edge_index=edge_index_td,
        edge_index_td=edge_index_td,
        edge_index_bu=edge_index_bu,
        edge_attr=edge_attr,
        edge_stance=edge_attr[:, 0:1] if edge_attr.numel() > 0 else torch.empty((0, 1)),
        cascade_stats=cascade_stats,
        root_index=torch.tensor([0], dtype=torch.long),
        edge_pairs=kept_pairs,
    )

    # Load GNN checkpoint if present; otherwise build calibrated untuned architecture for structural inspection
    if gnn_ckpt is not None and Path(gnn_ckpt).exists():
        payload = torch.load(Path(gnn_ckpt), map_location="cpu", weights_only=False)
        cfg = payload.get("config", {})
        gnn_model = SBiFAGAT(
            in_dim=int(payload.get("in_dim", TOTAL_NODE_FEATURE_DIM)),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            edge_dim=int(cfg.get("edge_dim", 4)),
            num_rumor_classes=int(payload.get("num_rumor_classes", 4)),
            eps=float(cfg.get("eps", 0.5)),
            dropout=float(cfg.get("dropout", 0.1)),
            use_cascade_stats=bool(cfg.get("use_cascade_stats", True)),
            predict_uncertainty=bool(cfg.get("predict_uncertainty", True)),
        )
        gnn_model.load_state_dict(payload["model_state_dict"])
        checkpoint_loaded = True
    else:
        gnn_model = SBiFAGAT(
            in_dim=TOTAL_NODE_FEATURE_DIM,
            hidden_dim=128,
            edge_dim=4,
            use_cascade_stats=True,
            predict_uncertainty=True,
        )
        checkpoint_loaded = False

    diag = gnn_model.predict_with_diagnostics(data, top_k_edges=5)

    behavior_counts = {lbl: 0 for lbl in DISCOURSE_BEHAVIOR_LABELS}
    for d in kept_edge_details:
        behavior_counts[d["behavior_label"]] = behavior_counts.get(d["behavior_label"], 0) + 1

    # Combine model prediction with interpretable early polarization indicator when checkpoint isn't loaded
    raw_pred_polar = float(diag["y_polar_pred"].view(-1)[0].item())
    raw_pred_replies = float(diag["predicted_total_replies"].view(-1)[0].item())

    actual_total_replies = len(non_roots)
    stance_map = {eid: sc["s_uv"] for eid, sc in edge_scores.items()}
    actual_aligned = _compute_root_aligned_stances(
        root_id=root_id,
        children_adj=children_adj,
        edge_lookup=edge_lookup,
        stance_map=stance_map,
    )
    actual_full_polar = _compute_full_tree_polarization(actual_aligned) if actual_aligned else 0.0

    stats_vec = cascade_stats.view(-1).tolist()
    regimes_polar = (
        "Phân cực xung đột gay gắt"
        if raw_pred_polar >= 0.55 or (behavior_counts["hostile_conflict"] + behavior_counts["irrational_fallacy"]) >= 3
        else ("Tranh luận đa chiều văn minh" if behavior_counts["constructive_oppose"] >= 2 else "Tương đối đồng thuận / ít chia rẽ")
    )

    return {
        "root_title": str(post.get("title") or root_row["text"][:120]),
        "k_observed": len(kept_nodes) - 1,
        "checkpoint_loaded": checkpoint_loaded,
        "early_burst_metrics": {
            "elapsed_span_norm": stats_vec[0],
            "mean_interarrival_norm": stats_vec[1],
            "burst_acceleration": stats_vec[2],
            "root_direct_ratio": stats_vec[3],
            "max_depth_norm": stats_vec[4],
            "unique_user_ratio": stats_vec[5],
            "substantive_text_ratio": stats_vec[6],
            "one_sided_hostility_index": stats_vec[7],
        },
        "behavior_counts": behavior_counts,
        "predicted_total_replies": raw_pred_replies,
        "predicted_polarization_score": raw_pred_polar,
        "predicted_regime": regimes_polar,
        "confidence_score": float(diag["confidence_score"].view(-1)[0].item()),
        "low_signal_warning": bool(diag["low_signal_warning"].view(-1)[0].item()),
        "top_trigger_edges": diag["top_trigger_edges"],
        "flagged_irrational_or_hostile": [
            d for d in kept_edge_details if d["behavior_label"] in {"irrational_fallacy", "hostile_conflict"}
        ],
        "actual_post_hoc_comparison": {
            "has_future_replies": actual_total_replies > (len(kept_nodes) - 1),
            "actual_total_replies": actual_total_replies,
            "actual_full_polarization": actual_full_polar,
        },
    }


def _format_report_vi(result: Mapping[str, Any]) -> str:
    lines = [
        "=" * 78,
        "BAO CAO DU BAO SOM & PHAN TICH HANH VI TRANH LUAN (SBi-FAGAT v2)",
        "=" * 78,
        f"Bai goc      : {result['root_title']}",
        f"Cua so quan sat (V_K): {result['k_observed']} binh luan dau tien",
        f"Trang thai GNN       : {'Da nap checkpoint huan luyen' if result['checkpoint_loaded'] else 'Che do khoi dong nhanh (chua nap checkpoint)'}",
        "",
        "[1] PHAN LOAI 5 NHOM HANH VI TRONG 20 BINH LUAN DAU:",
    ]
    for lbl in DISCOURSE_BEHAVIOR_LABELS:
        cnt = result["behavior_counts"].get(lbl, 0)
        lines.append(f"  - {BEHAVIOR_LABEL_VI.get(lbl, lbl):38s}: {cnt} binh luan")

    bm = result["early_burst_metrics"]
    lines.extend(
        [
            "",
            "[2] DONG HO TOC DO & CAU TRUC NHANH SOM:",
            f"  - Gia toc binh luan (>0.5 la dang tang toc): {bm['burst_acceleration']:.3f}",
            f"  - Ty le cau co noi dung thuc chat          : {bm['substantive_text_ratio']:.1%}",
            f"  - Chi so kich dong 1 chieu                 : {bm['one_sided_hostility_index']:.3f}",
            "",
            "[3] DU BAO TUONG LAI & DO TU TIN:",
            f"  - Quy mo binh luan cuoi ky du kien         : ~{result['predicted_total_replies']:.1f} binh luan",
            f"  - Chi so phan cuc du kien [0..1]           : {result['predicted_polarization_score']:.4f}",
            f"  - Danh gia cuc dien                        : {result['predicted_regime']}",
            f"  - Do tu tin du bao                         : {result['confidence_score']:.1%}",
        ]
    )
    if result["low_signal_warning"]:
        lines.append("  [!] CANH BAO TIN HIEU THAP: Binh luan dau nhat/thieu du kien -> Khuyen nghi cho them den moc K=35-50.")

    flagged = result.get("flagged_irrational_or_hostile") or []
    lines.append("")
    lines.append(f"[4] DIEM MAT BINH LUAN VO LY / CHONG DOI NGUY HIEM NHAT ({len(flagged)} cau):")
    for idx, item in enumerate(flagged[:5], start=1):
        lbl_vi = BEHAVIOR_LABEL_VI.get(item["behavior_label"], item["behavior_label"])
        lines.append(
            f"  {idx}. [{lbl_vi}] (s={item['s_uv']:+.2f}, gay_gat={item['h_uv']:.2f}, vo_ly={item['r_uv']:.2f})"
        )
        lines.append(f"     \"{item['reply_text'][:160]}\"")

    act = result["actual_post_hoc_comparison"]
    lines.extend(
        [
            "",
            "[5] DOI CHIEU THUC TE (KHONG LOT VAO DAU VAO MO HINH):",
            f"  - Tong so binh luan thuc te toan bai       : {act['actual_total_replies']} binh luan",
            f"  - Do phan cuc thuc te toan bai             : {act['actual_full_polarization']:.4f}",
            "=" * 78,
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Live social thread early polarization & virality forecast.")
    parser.add_argument("--reddit-url", type=str, default=None)
    parser.add_argument("--thread-id", type=str, default=None)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--demo-offline", action="store_true")
    parser.add_argument("--cross-encoder-ckpt", type=Path, default=None)
    parser.add_argument("--gnn-ckpt", type=Path, default=None)
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--lang", choices=["vi", "en"], default="vi")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.demo_offline:
        post, comments = _make_demo_offline_thread()
    elif args.input_json is not None:
        post, comments = _load_thread_from_json(args.input_json)
    elif args.reddit_url or args.thread_id:
        pid = args.thread_id or _extract_post_id_from_url(str(args.reddit_url))
        post, comments = _fetch_reddit_thread(pid)
    else:
        raise ValueError("Provide one of --demo-offline, --input-json, --reddit-url, or --thread-id")

    res = predict_live_cascade(
        post,
        comments,
        k_horizon=args.k_horizon,
        cross_encoder_ckpt=args.cross_encoder_ckpt,
        gnn_ckpt=args.gnn_ckpt,
        lang=args.lang,
    )
    print(_format_report_vi(res))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

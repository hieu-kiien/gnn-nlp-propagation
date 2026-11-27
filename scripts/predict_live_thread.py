"""Live Social Thread Predictor & Real-World Check Tool for SBi-FAGAT v2.

Accepts:
  - `--reddit-url https://www.reddit.com/r/vozforums/comments/<post_id>/...`
  - `--thread-id <reddit_post_id>` (via Arctic Shift REST API)
  - `--input-json path/to/thread.json`
  - `--demo-offline` (self-contained realistic 24-reply Vietnamese debate for testing)

Strictly truncates model inputs to the first `K` (default 20) chronological comments,
labels each reply across the unified 2D `[s_uv, h_uv]` behavioral categories
(`support`, `neutral`, `constructive_oppose`, `hostile_conflict`), runs `SBiFAGAT`,
and if the live thread already has `> K` replies, compares the early forecast against
the true future (`t > K`, `future_only`) outcome WITHOUT ever leaking post-K data into the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import requests
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arctic_shift_vn import (
    ARCTIC_SHIFT_API_BASE,
    build_vn_cascade_from_thread,
    hash_author_sha256,
)
from src.data.pyg_dataset import (
    CASCADE_STATS_DIM,
    CascadeData,
    TEMPORAL_REF_SECONDS,
    TOTAL_NODE_FEATURE_DIM,
    _compute_full_tree_polarization,
    _compute_root_aligned_stances,
    compute_causal_structural_node_features,
    compute_early_cascade_burst_stats,
)
from src.models.sbi_fagat import SBiFAGAT
from src.nlp.discourse_labels import _heuristic_behavior_fallback
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    STANCE_LABELS,
    TWITTER_ROBERTA_MODEL_NAME,
    VISOBERT_MODEL_NAME,
    load_tokenizer_safe,
)
from src.nlp.extract_features import (
    compute_masked_mean_pooling,
    encode_edges_with_cross_encoder,
    encode_node_embeddings,
)
from src.nlp.topic_post_analyzer import (
    PHEME_POLAR_Q50,
    PHEME_POLAR_Q75,
    PHEME_POLAR_Q90,
    VN_POLAR_Q50,
    VN_POLAR_Q75,
    VN_POLAR_Q90,
    analyze_large_thread_dynamics,
    classify_post_topics,
    get_polar_quantiles,
)

_REDDIT_POST_ID_RE = re.compile(r"/comments/([^/?#]+)")
_VALID_POST_ID_RE = re.compile(r"[A-Za-z0-9]{4,12}")
_REQUIRED_CE_FILES: tuple[str, ...] = (
    "dual_head_weights.pt",
    "dual_head_meta.json",
    "config.json",
)
_UNIFIED_BEHAVIOR_LABELS: tuple[str, ...] = (
    "support",
    "neutral",
    "constructive_oppose",
    "hostile_conflict",
)
_BEHAVIOR_LABEL_VI_2D: dict[str, str] = {
    "support": "Tán thành / Ủng hộ",
    "neutral": "Trung lập / Hỏi thông tin",
    "constructive_oppose": "Phản biện ôn hòa",
    "hostile_conflict": "Xung đột / Gay gắt cao",
}

VN_QUANTILE_CALIBRATION_NOTE: str = (
    "Regime thresholds calibrated strictly on VN Train split "
    "(N=1,050 cascades, K=20, min_total_replies=25, UTC 2025-06-15 to 2026-06-04); zero Val/Test leakage."
)
PHEME_QUANTILE_CALIBRATION_NOTE: str = (
    "Regime thresholds calibrated strictly on PHEME Train split "
    "(N=1,786 cascades, K=10, min_total_replies=15, 6 Train events); zero Val/Test leakage."
)
TRAINING_MAX_REPLIES_REF: int = 345
OOD_SCALE_THRESHOLD: int = 350


class ScanCancelledError(RuntimeError):
    """Raised when an asynchronous Mega-Thread scan job is cancelled by the user."""


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
            "author_hash": hash_author_sha256(f"user_{idx}"),
        }
        for idx, (cid, pid, dt, body) in enumerate(sample_comments, start=1)
    ]
    return post, comments


def _validate_reddit_post_id(post_id: str, *, source: str = "input") -> str:
    """Validate that a Reddit submission ID contains only safe alphanumeric characters."""
    cleaned = str(post_id or "").strip()
    if not _VALID_POST_ID_RE.fullmatch(cleaned):
        raise ValueError(
            f"Invalid Reddit post_id {cleaned!r} from {source}: expected [A-Za-z0-9]{{4,12}}."
        )
    return cleaned


def _extract_post_id_from_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw or "\\" in raw or any(ch.isspace() for ch in raw):
        raise ValueError(f"Invalid Reddit URL or post ID: {url!r}")

    m = _REDDIT_POST_ID_RE.search(raw)
    if m:
        return _validate_reddit_post_id(m.group(1), source=raw)

    if "/" in raw or "://" in raw or ".." in raw:
        raise ValueError(f"Reddit URL missing valid /comments/<post_id> segment: {url!r}")

    candidate = raw[3:] if raw.startswith("t3_") else raw
    return _validate_reddit_post_id(candidate, source=raw)


def _normalize_arctic_comment(c: Mapping[str, Any], *, post_id: str) -> dict[str, Any]:
    cid = str(c.get("id") or "").removeprefix("t1_")
    raw_author = c.get("author")
    return {
        "fullname": f"t1_{cid}",
        "id": cid,
        "link_id": str(c.get("link_id") or f"t3_{post_id}"),
        "parent_id": str(c.get("parent_id") or f"t3_{post_id}"),
        "body": str(c.get("body") or c.get("text") or ""),
        "text": str(c.get("body") or c.get("text") or ""),
        "created_utc": c.get("created_utc"),
        "author": raw_author,
        "author_is_bot": isinstance(raw_author, str) and raw_author.strip().lower() == "automoderator",
        "author_hash": hash_author_sha256(str(raw_author) if raw_author is not None else None),
    }


def _http_get_json_with_backoff(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str],
    timeout: int = 20,
    max_retries: int = 4,
) -> Any:
    backoff = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if getattr(resp, "status_code", 200) == 429:
                reset_hdr = getattr(resp, "headers", {}).get("x-ratelimit-reset")
                try:
                    wait_s = float(reset_hdr) if reset_hdr is not None else backoff
                except (TypeError, ValueError):
                    wait_s = backoff
                time.sleep(min(max(0.05, wait_s), 16.0))
                backoff *= 2.0
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(min(backoff, 4.0))
                backoff *= 2.0
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed GET {url}")


def _extract_tree_items(items: Sequence[Mapping[str, Any]], *, post_id: str, out: list[dict[str, Any]]) -> None:
    for item in items:
        if not isinstance(item, Mapping):
            continue
        data = item.get("data") if isinstance(item.get("data"), Mapping) else item
        cid = str(data.get("id") or "").removeprefix("t1_")
        if cid:
            out.append(_normalize_arctic_comment(data, post_id=post_id))
        replies = data.get("replies")
        if isinstance(replies, Mapping):
            sub = replies.get("data", {}).get("children") or replies.get("children") or []
            if isinstance(sub, Sequence):
                _extract_tree_items(sub, post_id=post_id, out=out)
        elif isinstance(replies, Sequence) and not isinstance(replies, (str, bytes)):
            _extract_tree_items(replies, post_id=post_id, out=out)


def fetch_reddit_thread_paginated(
    post_id: str,
    *,
    max_comments: int = 10000,
    timeout: int = 20,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fetch submission + up to `max_comments` via paginated Arctic Shift search with backoff and fallbacks."""
    post_id = _validate_reddit_post_id(post_id, source="fetch")
    cap = max(1, min(10000, int(max_comments)))
    headers = {"User-Agent": "SBi-FAGAT-LiveCheck/2.1"}

    post: dict[str, Any] | None = None
    comments: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    try:
        sub_payload = _http_get_json_with_backoff(
            f"{ARCTIC_SHIFT_API_BASE}/posts/search",
            params={"ids": post_id},
            headers=headers,
            timeout=timeout,
            max_retries=3,
        )
        sub_data = (sub_payload or {}).get("data") or []
        if sub_data:
            post = sub_data[0]
    except Exception:
        post = None

    if post is not None:
        cursor_after: int | None = None
        max_pages = max(25, math.ceil(cap / 100) + 5)
        for page_idx in range(max_pages):
            if cancel_event is not None and cancel_event.is_set():
                raise ScanCancelledError("Đã hủy tác vụ tải bình luận.")
            page_limit = min(500, cap - len(comments))
            if page_limit <= 0:
                break
            params: dict[str, Any] = {"link_id": f"t3_{post_id}", "limit": page_limit, "sort": "asc"}
            if cursor_after is not None:
                params["after"] = cursor_after
            try:
                com_payload = _http_get_json_with_backoff(
                    f"{ARCTIC_SHIFT_API_BASE}/comments/search",
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    max_retries=4,
                )
            except Exception:
                break

            raw_comments = (com_payload or {}).get("data") or []
            if not raw_comments:
                break

            added_this_page = 0
            max_ts_seen: float | None = None
            for raw_c in raw_comments:
                if not isinstance(raw_c, Mapping):
                    continue
                norm = _normalize_arctic_comment(raw_c, post_id=post_id)
                cid = norm["id"]
                ts_val = norm.get("created_utc")
                try:
                    ts_f = float(ts_val) if ts_val is not None else None
                except (TypeError, ValueError):
                    ts_f = None
                if ts_f is not None and (max_ts_seen is None or ts_f > max_ts_seen):
                    max_ts_seen = ts_f
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    comments.append(norm)
                    added_this_page += 1
                    if len(comments) >= cap:
                        break

            if progress_cb is not None:
                progress_cb(
                    {
                        "stage": "fetching_comments",
                        "page": page_idx + 1,
                        "processed": len(comments),
                        "total": cap,
                        "percent": min(99.0, round(100.0 * len(comments) / max(1, cap), 1)),
                    }
                )

            if len(comments) >= cap or len(raw_comments) < page_limit:
                break
            if max_ts_seen is None:
                break
            next_after = int(max_ts_seen) if added_this_page > 0 else int(max_ts_seen) + 1
            if cursor_after is not None and next_after <= cursor_after:
                next_after = cursor_after + 1
            cursor_after = next_after

        if not comments:
            try:
                tree_payload = _http_get_json_with_backoff(
                    f"{ARCTIC_SHIFT_API_BASE}/comments/tree",
                    params={"link_id": f"t3_{post_id}", "limit": 9999},
                    headers=headers,
                    timeout=max(timeout, 30),
                    max_retries=3,
                )
                tree_out: list[dict[str, Any]] = []
                _extract_tree_items((tree_payload or {}).get("data") or [], post_id=post_id, out=tree_out)
                for row in tree_out:
                    cid = row["id"]
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        comments.append(row)
                        if len(comments) >= cap:
                            break
            except Exception:
                pass

        if comments:
            return post, comments[:cap]

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
            raw_author = d.get("author")
            flat_comments.append(
                {
                    "fullname": f"t1_{cid}",
                    "id": cid,
                    "link_id": f"t3_{post_id}",
                    "parent_id": str(d.get("parent_id") or f"t3_{post_id}"),
                    "body": str(d.get("body") or ""),
                    "text": str(d.get("body") or ""),
                    "created_utc": d.get("created_utc"),
                    "author": raw_author,
                    "author_is_bot": isinstance(raw_author, str)
                    and raw_author.strip().lower() == "automoderator",
                    "author_hash": hash_author_sha256(
                        str(raw_author) if raw_author is not None else None
                    ),
                }
            )
            replies = d.get("replies")
            if isinstance(replies, Mapping):
                _walk_listing(replies.get("data", {}).get("children", []))

    _walk_listing(payload[1]["data"]["children"])
    return post_raw, flat_comments[:cap]


def _fetch_reddit_thread(
    post_id: str,
    timeout: int = 20,
    max_comments: int = 500,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fetch submission + comments via Arctic Shift API with Reddit public JSON fallback."""
    if int(max_comments) > 500:
        return fetch_reddit_thread_paginated(post_id, max_comments=int(max_comments), timeout=timeout)

    post_id = _validate_reddit_post_id(post_id, source="fetch")
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
            comments = [_normalize_arctic_comment(c, post_id=post_id) for c in raw_comments]
            return post, comments
    except Exception:
        pass

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
            raw_author = d.get("author")
            flat_comments.append(
                {
                    "fullname": f"t1_{cid}",
                    "id": cid,
                    "link_id": f"t3_{post_id}",
                    "parent_id": str(d.get("parent_id") or f"t3_{post_id}"),
                    "body": str(d.get("body") or ""),
                    "text": str(d.get("body") or ""),
                    "created_utc": d.get("created_utc"),
                    "author": raw_author,
                    "author_is_bot": isinstance(raw_author, str)
                    and raw_author.strip().lower() == "automoderator",
                    "author_hash": hash_author_sha256(
                        str(raw_author) if raw_author is not None else None
                    ),
                }
            )
            replies = d.get("replies")
            if isinstance(replies, Mapping):
                _walk_listing(replies.get("data", {}).get("children", []))

    _walk_listing(payload[1]["data"]["children"])
    return post_raw, flat_comments


def load_thread_from_json_payload(
    raw: Any,
    *,
    max_comments: int = 10000,
    source_name: str = "payload",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse either Reddit listing JSON or SBi-FAGAT object JSON up to `max_comments`."""
    cap = max(1, min(10000, int(max_comments)))
    if isinstance(raw, list) and len(raw) >= 2 and isinstance(raw[0], Mapping) and "data" in raw[0]:
        post = raw[0]["data"]["children"][0]["data"]
        flat: list[dict[str, Any]] = []
        post_id = str(post.get("id") or "local_thread")

        def _walk(children: Sequence[Mapping[str, Any]]) -> None:
            for ch in children:
                if len(flat) >= cap:
                    return
                if ch.get("kind") != "t1":
                    continue
                d = ch.get("data", {})
                cid = str(d.get("id") or "")
                raw_author = d.get("author")
                flat.append(
                    {
                        "fullname": f"t1_{cid}",
                        "id": cid,
                        "link_id": f"t3_{post_id}",
                        "parent_id": str(d.get("parent_id") or f"t3_{post_id}"),
                        "text": str(d.get("body") or ""),
                        "body": str(d.get("body") or ""),
                        "created_utc": d.get("created_utc"),
                        "author": raw_author,
                        "author_is_bot": isinstance(raw_author, str)
                        and raw_author.strip().lower() == "automoderator",
                        "author_hash": hash_author_sha256(
                            str(raw_author) if raw_author is not None else None
                        ),
                    }
                )
                rep = d.get("replies")
                if isinstance(rep, Mapping):
                    _walk(rep.get("data", {}).get("children", []))

        _walk(raw[1]["data"]["children"])
        return post, flat[:cap]

    if isinstance(raw, Mapping):
        post_id = str(raw.get("id") or "custom_live").removeprefix("t3_")
        base_ts = float(raw.get("created_utc") or 1763200000.0)
        post = {
            "id": post_id,
            "title": str(raw.get("title") or ""),
            "selftext": str(raw.get("text") or raw.get("selftext") or raw.get("root_text") or ""),
            "subreddit": str(raw.get("subreddit") or "custom"),
            "author": str(raw.get("author") or "op"),
            "created_utc": base_ts,
        }
        comments: list[dict[str, Any]] = []
        for idx, c in enumerate(raw.get("comments", []), start=1):
            if not isinstance(c, Mapping):
                continue
            cid = str(c.get("id") or f"c{idx}").removeprefix("t1_")
            pid = str(c.get("parent_id") or f"t3_{post_id}").strip()
            if pid.lower() in {"root", post_id, f"t3_{post_id}"}:
                pid = f"t3_{post_id}"
            elif not pid.startswith(("t3_", "t1_")):
                pid = f"t1_{pid}"
            raw_author = c["author"] if "author" in c else f"u_{idx}"
            author_str = str(raw_author) if raw_author is not None else None
            text_val = str(c.get("text") or c.get("body") or "")
            raw_delay = c.get("time_delay_sec")
            ts_c = c.get("created_utc")
            if ts_c is None:
                ts_val = base_ts + (float(raw_delay) if raw_delay is not None else idx * 45.0)
            else:
                ts_val = float(ts_c)
            comments.append(
                {
                    "fullname": f"t1_{cid}",
                    "id": cid,
                    "link_id": f"t3_{post_id}",
                    "parent_id": pid,
                    "body": text_val,
                    "text": text_val,
                    "created_utc": ts_val,
                    "author": author_str,
                    "author_is_bot": bool(c.get("author_is_bot"))
                    or (isinstance(author_str, str) and author_str.strip().lower() == "automoderator"),
                    "author_hash": hash_author_sha256(author_str),
                }
            )
            if len(comments) >= cap:
                break
        return post, comments

    raise ValueError(f"Unsupported thread JSON structure in {source_name}")


def load_thread_from_file(path: Path, *, max_comments: int = 10000) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a local JSON or JSONL social discussion file (up to `max_comments`)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if p.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not records:
            raise ValueError(f"Empty JSONL file: {p}")
        first = records[0]
        if isinstance(first, Mapping) and "comments" in first:
            return load_thread_from_json_payload(first, max_comments=max_comments, source_name=str(p))
        root_row = dict(first) if isinstance(first, Mapping) else {}
        root_row["comments"] = [r for r in records[1:] if isinstance(r, Mapping)]
        return load_thread_from_json_payload(root_row, max_comments=max_comments, source_name=str(p))
    raw = json.loads(text)
    return load_thread_from_json_payload(raw, max_comments=max_comments, source_name=str(p))


def _load_thread_from_json(path: Path, *, max_comments: int = 10000) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return load_thread_from_file(path, max_comments=max_comments)


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


def _resolve_local_hf_snapshot(repo_or_path: str) -> str | None:
    """Resolve a local directory or cached Hugging Face snapshot without network calls."""
    if Path(repo_or_path).is_dir():
        return str(repo_or_path)
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_or_path, local_files_only=True)
    except Exception:
        return None


def _load_local_model_from_pretrained(backbone_id: str, *, allow_network: bool) -> Any | None:
    """Load an AutoModel locally without triggering unexpected downloads or Hub threads."""
    local_dir = _resolve_local_hf_snapshot(backbone_id)
    if local_dir is not None:
        try:
            return AutoModel.from_pretrained(local_dir, local_files_only=True)
        except TypeError:
            try:
                return AutoModel.from_pretrained(local_dir)
            except Exception:
                return None
        except Exception:
            return None

    is_real_hf = getattr(AutoModel.from_pretrained, "__module__", "").startswith("transformers.")
    if allow_network or not is_real_hf:
        try:
            return AutoModel.from_pretrained(backbone_id)
        except Exception:
            return None
    return None


def _load_fast_tokenizer_json(target_dir: Path) -> Any | None:
    """Load `tokenizer.json` directly via PreTrainedTokenizerFast when AutoTokenizer hits SentencePiece/Protobuf errors."""
    tok_json = target_dir / "tokenizer.json"
    if not tok_json.is_file():
        return None
    cfg_path = target_dir / "tokenizer_config.json"
    cfg: dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    try:
        return PreTrainedTokenizerFast(
            tokenizer_file=str(tok_json),
            bos_token=str(cfg.get("bos_token") or "<s>"),
            eos_token=str(cfg.get("eos_token") or "</s>"),
            unk_token=str(cfg.get("unk_token") or "<unk>"),
            pad_token=str(cfg.get("pad_token") or "<pad>"),
            cls_token=str(cfg.get("cls_token") or "<s>"),
            sep_token=str(cfg.get("sep_token") or "</s>"),
            mask_token=str(cfg.get("mask_token") or "<mask>"),
        )
    except Exception:
        return None


def _load_local_tokenizer(
    backbone_id: str,
    ce_dir: Path | None,
    *,
    lang: str = "vi",
    node_backbone_overridden: bool,
    allow_network: bool,
) -> Any | None:
    """Load tokenizer from local cache or CE directory with SentencePiece/FastJSON fallback."""
    local_dir = _resolve_local_hf_snapshot(backbone_id)
    targets: list[Path] = [Path(local_dir)] if local_dir is not None else []
    if ce_dir is not None and ce_dir.is_dir() and not node_backbone_overridden:
        targets.append(ce_dir)
    if not node_backbone_overridden:
        default_dirs = (
            (Path("checkpoints/dual_ce_vn"), Path("checkpoints/visobert_vihsd"))
            if lang == "vi"
            else (Path("checkpoints/dual_ce_pheme"),)
        )
        for d in default_dirs:
            if d.is_dir() and d not in targets:
                targets.append(d)

    for target in targets:
        has_tok = any(
            (target / fname).is_file()
            for fname in ("tokenizer_config.json", "tokenizer.json", "vocab.json", "sentencepiece.bpe.model")
        )
        if not has_tok:
            continue
        for use_fast in (True, False):
            kw: dict[str, Any] = {"local_files_only": True}
            if not use_fast:
                kw["use_fast"] = False
            try:
                return AutoTokenizer.from_pretrained(str(target), **kw)
            except TypeError:
                try:
                    return AutoTokenizer.from_pretrained(str(target))
                except Exception:
                    pass
            except Exception:
                pass
        fast_tok = _load_fast_tokenizer_json(target)
        if fast_tok is not None:
            return fast_tok

    is_real_hf = getattr(AutoTokenizer.from_pretrained, "__module__", "").startswith("transformers.")
    if allow_network or not is_real_hf:
        try:
            return load_tokenizer_safe(backbone_id)
        except Exception:
            return None
    return None


def _resolve_nlp_backends(
    *,
    cross_encoder_ckpt: Path | None,
    node_backbone: str | None,
    lang: str,
    is_demo: bool,
    preloaded_ce: tuple[DualHeadCrossEncoder, Any] | None = None,
) -> tuple[DualHeadCrossEncoder | None, Any | None, Any | None, Any | None, bool, str]:
    """Load available Cross-Encoder & SpanPool backends without unwanted offline downloads."""
    ce_dir = Path(cross_encoder_ckpt) if cross_encoder_ckpt is not None else None
    ce_model: DualHeadCrossEncoder | None = None
    ce_tok: Any | None = None

    if preloaded_ce is not None:
        ce_model, ce_tok = preloaded_ce
        if ce_model is not None:
            ce_model.eval()
    else:
        has_all_ce_files = bool(
            ce_dir is not None
            and ce_dir.is_dir()
            and all((ce_dir / fname).is_file() for fname in _REQUIRED_CE_FILES)
        )
        if has_all_ce_files and ce_dir is not None:
            try:
                ce_model, ce_tok = DualHeadCrossEncoder.load_checkpoint(ce_dir, map_location="cpu")
                ce_model.eval()
            except Exception:
                ce_model = None
                ce_tok = None

    backbone_id = node_backbone or (
        VISOBERT_MODEL_NAME if lang == "vi" else TWITTER_ROBERTA_MODEL_NAME
    )
    # Only allow network backbone download when explicitly requested via --node-backbone in live mode;
    # otherwise prefer local HF cache and fall back seamlessly to local ce_model.encoder.
    allow_net = bool(ce_model is not None and not is_demo and node_backbone is not None)
    frozen_node_encoder = _load_local_model_from_pretrained(backbone_id, allow_network=allow_net)
    frozen_node_tok: Any | None = None

    if frozen_node_encoder is not None:
        frozen_node_tok = _load_local_tokenizer(
            backbone_id,
            ce_dir,
            lang=lang,
            node_backbone_overridden=node_backbone is not None,
            allow_network=allow_net,
        )
        if frozen_node_tok is None and ce_tok is not None and node_backbone is None:
            frozen_node_tok = ce_tok
        if frozen_node_tok is None:
            frozen_node_encoder = None

    if frozen_node_encoder is None and ce_model is not None and ce_tok is not None:
        frozen_node_encoder = ce_model.encoder
        frozen_node_tok = ce_tok

    if frozen_node_encoder is not None:
        frozen_node_encoder.eval()
        for p in frozen_node_encoder.parameters():
            p.requires_grad_(False)

    spanpool_ready = frozen_node_encoder is not None and frozen_node_tok is not None
    ce_ready = ce_model is not None and ce_tok is not None and spanpool_ready
    if ce_ready:
        return ce_model, ce_tok, frozen_node_encoder, frozen_node_tok, True, "full_2d_cross_encoder"
    if spanpool_ready:
        return None, None, frozen_node_encoder, frozen_node_tok, False, "visobert_spanpool_heuristic_edges"
    return None, None, None, None, False, "heuristic_hash_sandbox"


def _classify_2d_behavior_label(s_uv: float, h_uv: float) -> str:
    """Map unified 2D `[s_uv, h_uv]` edge scores to the 4 clean behavioral categories."""
    if s_uv > 0.1:
        return "support"
    if h_uv > 0.6:
        return "hostile_conflict"
    if s_uv < -0.1:
        return "constructive_oppose"
    return "neutral"


def _classify_calibrated_regime(
    pred_polar: float,
    behavior_counts: Mapping[str, int] | None = None,
    *,
    lang: str = "vi",
) -> str:
    """Classify cascade polarization regime calibrated strictly against Train quantiles."""
    del behavior_counts  # Retained for signature compatibility; counts never override quantiles.
    lang_norm: Literal["vi", "en"] = "en" if str(lang).lower() == "en" else "vi"
    q50, q75, q90 = get_polar_quantiles(lang_norm)
    tag = "PHEME Train" if lang_norm == "en" else "VN Train"
    if pred_polar >= q90:
        return f"Phân cực xung đột gay gắt (>= Q90 {tag})"
    if pred_polar >= q75:
        return f"Phân hóa khá cao (Q75-Q90 {tag})"
    if pred_polar >= q50:
        return f"Phân hóa trung bình (Q50-Q75 {tag})"
    return f"Tương đối đồng thuận / ít chia rẽ (< Q50 {tag})"


def _select_stratified_edges(
    *,
    root_id: str,
    gnn_nodes: Sequence[Mapping[str, Any]],
    candidate_non_roots: Sequence[Mapping[str, Any]],
    edge_records: Sequence[Mapping[str, Any]],
    depth_map: Mapping[str, int],
    sample_budget: int,
) -> set[str]:
    """Deterministically sample edges across 10 chronological deciles + deep branches with ancestor closure."""
    budget = max(1, int(sample_budget))
    if len(edge_records) <= budget:
        return {str(e["edge_id"]) for e in edge_records}

    edge_by_dst: dict[str, Mapping[str, Any]] = {}
    parent_by_dst: dict[str, str] = {}
    for e in edge_records:
        dst = str(e["target_id"])
        edge_by_dst[dst] = e
        parent_by_dst[dst] = str(e["source_id"])

    selected_eids: set[str] = set()
    gnn_ids = {str(n["node_id"]) for n in gnn_nodes}
    for e in edge_records:
        if str(e["target_id"]) in gnn_ids:
            selected_eids.add(str(e["edge_id"]))

    remaining_nodes = [
        n for n in candidate_non_roots
        if str(n["node_id"]) not in gnn_ids and str(n["node_id"]) in edge_by_dst
    ]
    if not remaining_nodes:
        return selected_eids

    n_deciles = 10
    buckets: list[list[Mapping[str, Any]]] = [[] for _ in range(n_deciles)]
    total_rem = len(remaining_nodes)
    for idx_r, node in enumerate(remaining_nodes):
        b_idx = min(n_deciles - 1, (idx_r * n_deciles) // max(1, total_rem))
        buckets[b_idx].append(node)

    # Inside each decile, prioritize deeper branches first while remaining completely deterministic
    for b_idx in range(n_deciles):
        buckets[b_idx].sort(
            key=lambda n: (-int(depth_map.get(str(n["node_id"]), 1)), str(n["node_id"]))
        )

    def _add_with_ancestors(target_id: str, hard_cap: int) -> bool:
        chain_eids: list[str] = []
        curr = target_id
        seen_c: set[str] = set()
        while curr and curr != root_id and curr not in seen_c:
            seen_c.add(curr)
            e_row = edge_by_dst.get(curr)
            if e_row is None:
                break
            eid = str(e_row["edge_id"])
            if eid not in selected_eids:
                chain_eids.append(eid)
            curr = parent_by_dst.get(curr, "")
        if not chain_eids:
            return False
        if len(selected_eids) + len(chain_eids) <= hard_cap:
            selected_eids.update(chain_eids)
            return True
        return False

    hard_cap = max(budget, len(selected_eids) + int(budget * 0.15))
    cursor_in_bucket = [0] * n_deciles
    made_progress = True
    while len(selected_eids) < budget and made_progress:
        made_progress = False
        for b_idx in range(n_deciles):
            if len(selected_eids) >= budget:
                break
            b_list = buckets[b_idx]
            while cursor_in_bucket[b_idx] < len(b_list):
                cand = b_list[cursor_in_bucket[b_idx]]
                cursor_in_bucket[b_idx] += 1
                if _add_with_ancestors(str(cand["node_id"]), hard_cap):
                    made_progress = True
                    break
                # Single-edge fallback if ancestor chain slightly exceeds soft cap when still under budget
                e_direct = edge_by_dst.get(str(cand["node_id"]))
                if e_direct is not None and len(selected_eids) < budget:
                    eid_d = str(e_direct["edge_id"])
                    if eid_d not in selected_eids:
                        selected_eids.add(eid_d)
                        made_progress = True
                        break

    return selected_eids


def _bootstrap_polar_ci_95(
    aligned_stances: Sequence[float],
    *,
    n_resamples: int = 200,
    seed: int = 42,
) -> list[float] | None:
    """Compute deterministic 95% bootstrap confidence interval for `y_polar` on stratified samples."""
    arr = np.asarray(list(aligned_stances), dtype=np.float64)
    if arr.size < 4:
        return None
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = arr.size
    for _ in range(n_resamples):
        sample = arr[rng.integers(0, n, size=n)]
        vals.append(float(_compute_full_tree_polarization(sample.tolist(), lambda_polar=0.5)))
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return [round(lo, 4), round(hi, 4)]


def predict_live_cascade(
    post: Mapping[str, Any],
    comments: Sequence[Mapping[str, Any]],
    *,
    k_horizon: int = 20,
    analysis_mode: Literal["early", "full"] = "early",
    max_full_comments: int = 500,
    scan_strategy: Literal["full", "stratified"] = "full",
    sample_budget: int = 600,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: threading.Event | None = None,
    custom_topics: str | Sequence[str] | None = None,
    topic_backend: Literal["auto", "nli", "embedding", "lexical"] = "auto",
    allow_nli_topic_download: bool = False,
    cross_encoder_ckpt: Path | None = None,
    preloaded_ce: tuple[DualHeadCrossEncoder, Any] | None = None,
    gnn_ckpt: Path | None = None,
    node_backbone: str | None = None,
    lang: str = "vi",
    junk_mode: Literal["prune", "contract"] = "prune",
    allow_untrained: bool = False,
    is_demo: bool = False,
) -> dict[str, Any]:
    """Build cascade graph, score behavior & topics, run SBiFAGAT (early K) + scalable post-hoc scan (up to 10k)."""
    t_scan_start = time.perf_counter()
    if isinstance(k_horizon, bool) or int(k_horizon) < 1:
        raise ValueError(f"k_horizon must be >= 1, got {k_horizon!r}")
    if analysis_mode not in {"early", "full"}:
        raise ValueError(f"analysis_mode must be 'early' or 'full', got {analysis_mode!r}")
    if isinstance(max_full_comments, bool) or int(max_full_comments) < 1:
        raise ValueError(f"max_full_comments must be >= 1, got {max_full_comments!r}")
    if scan_strategy not in {"full", "stratified"}:
        raise ValueError(f"scan_strategy must be 'full' or 'stratified', got {scan_strategy!r}")
    if isinstance(sample_budget, bool) or int(sample_budget) < 1:
        raise ValueError(f"sample_budget must be >= 1, got {sample_budget!r}")
    if cancel_event is not None and cancel_event.is_set():
        raise ScanCancelledError("Tác vụ quét đã bị hủy trước khi khởi chạy.")

    k_horizon = int(k_horizon)
    max_full_comments = int(max_full_comments)
    sample_budget = int(sample_budget)
    raw_comment_count = len(comments)
    junk_stats: dict[str, Any] = {}
    built = build_vn_cascade_from_thread(
        post,
        comments,
        min_comments=1,
        min_depth=1,
        min_vi_ratio=0.0,
        junk_mode=junk_mode,
        stats_out=junk_stats,
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
    ood_scale_warning = len(non_roots) > OOD_SCALE_THRESHOLD

    # Keep GNN early-warning inference strictly in-distribution on V_K (k_horizon),
    # while full mode expands visualization + large-thread discourse analysis up to max_full_comments.
    gnn_nodes = [root_row] + non_roots[:k_horizon]
    display_limit = min(len(non_roots), max_full_comments) if analysis_mode == "full" else k_horizon
    display_nodes = [root_row] + non_roots[:display_limit]
    horizon_for_features = k_horizon
    gnn_id_to_idx = {str(n["node_id"]): idx for idx, n in enumerate(gnn_nodes)}
    display_id_to_idx = {str(n["node_id"]): idx for idx, n in enumerate(display_nodes)}

    # Resolve NLP backends once
    ce_model, ce_tok, frozen_node_encoder, frozen_node_tok, cross_encoder_loaded, nlp_backend = (
        _resolve_nlp_backends(
            cross_encoder_ckpt=cross_encoder_ckpt,
            node_backbone=node_backbone,
            lang=lang,
            is_demo=is_demo,
            preloaded_ce=preloaded_ce,
        )
    )

    use_root_ctx = bool(getattr(ce_model, "include_root_context", False)) if ce_model is not None else False
    root_text_str = str(root_row["text"])
    live_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select edges to score (100% in full mode; stratified sampling when budget < edges in stratified mode)
    if analysis_mode == "full":
        display_id_set = {str(n["node_id"]) for n in display_nodes}
        scope_edges = [
            e for e in edge_records
            if str(e["source_id"]) in display_id_set and str(e["target_id"]) in display_id_set
        ]
    else:
        scope_edges = list(edge_records)

    is_stratified_subsampled = bool(scan_strategy == "stratified" and len(scope_edges) > sample_budget)
    if is_stratified_subsampled:
        scored_eid_set = _select_stratified_edges(
            root_id=root_id,
            gnn_nodes=gnn_nodes,
            candidate_non_roots=display_nodes[1:] if analysis_mode == "full" else non_roots,
            edge_records=scope_edges,
            depth_map=depth_map,
            sample_budget=sample_budget,
        )
        edges_to_score = [e for e in scope_edges if str(e["edge_id"]) in scored_eid_set]
    else:
        edges_to_score = scope_edges
        scored_eid_set = {str(e["edge_id"]) for e in edges_to_score}

    edge_scores: dict[str, dict[str, Any]] = {}
    score_edge_ids = [str(e["edge_id"]) for e in edges_to_score]
    score_parent_texts = [str(node_by_id[str(e["source_id"])]["text"]) for e in edges_to_score]
    score_reply_texts = [str(node_by_id[str(e["target_id"])]["text"]) for e in edges_to_score]

    total_to_score = len(score_edge_ids)
    batch_sz = 64 if live_device.type == "cuda" else 32
    t_score_start = time.perf_counter()

    if progress_cb is not None:
        progress_cb(
            {
                "stage": "scoring_edges",
                "processed": 0,
                "total": total_to_score,
                "percent": 0.0 if total_to_score > 0 else 100.0,
                "edges_per_sec": 0.0,
                "eta_sec": None,
            }
        )

    if total_to_score > 0 and ce_model is not None and ce_tok is not None:
        for b_start in range(0, total_to_score, batch_sz):
            if cancel_event is not None and cancel_event.is_set():
                raise ScanCancelledError("Tác vụ quét đã bị hủy trong quá trình chấm cạnh.")
            b_end = min(total_to_score, b_start + batch_sz)
            ce_out = encode_edges_with_cross_encoder(
                score_parent_texts[b_start:b_end],
                score_reply_texts[b_start:b_end],
                cross_encoder=ce_model,
                tokenizer=ce_tok,
                root_texts=[root_text_str] * (b_end - b_start),
                include_root_context=use_root_ctx,
                lang=lang,
                batch_size=batch_sz,
                device=live_device,
            )
            for rel_pos, eid in enumerate(score_edge_ids[b_start:b_end]):
                s_v = float(ce_out["s_uv"][rel_pos])
                h_v = float(ce_out["h_uv"][rel_pos])
                raw_blbl = str(ce_out["behavior_label"][rel_pos]) if "behavior_label" in ce_out else ""
                blbl = raw_blbl if raw_blbl in _UNIFIED_BEHAVIOR_LABELS else _classify_2d_behavior_label(s_v, h_v)
                edge_scores[eid] = {"s_uv": s_v, "h_uv": h_v, "behavior_label": blbl, "is_scored": True}

            if progress_cb is not None:
                elapsed_s = max(1e-6, time.perf_counter() - t_score_start)
                eps = float(b_end) / elapsed_s
                eta_s = float(total_to_score - b_end) / max(eps, 1e-6) if b_end < total_to_score else 0.0
                progress_cb(
                    {
                        "stage": "scoring_edges",
                        "processed": b_end,
                        "total": total_to_score,
                        "percent": round(100.0 * b_end / float(total_to_score), 1),
                        "edges_per_sec": round(eps, 1),
                        "eta_sec": round(eta_s, 1),
                    }
                )
    else:
        for idx_s, (eid, r_txt) in enumerate(zip(score_edge_ids, score_reply_texts), start=1):
            if cancel_event is not None and (idx_s % 128 == 0) and cancel_event.is_set():
                raise ScanCancelledError("Tác vụ quét đã bị hủy trong quá trình chấm cạnh.")
            heur = _heuristic_behavior_fallback(r_txt)
            s_v = float(heur["s_uv"])
            h_v = float(heur["h_uv"])
            edge_scores[eid] = {
                "s_uv": s_v,
                "h_uv": h_v,
                "behavior_label": _classify_2d_behavior_label(s_v, h_v),
                "is_scored": True,
            }
            if progress_cb is not None and (idx_s % 256 == 0 or idx_s == total_to_score):
                elapsed_s = max(1e-6, time.perf_counter() - t_score_start)
                eps = float(idx_s) / elapsed_s
                progress_cb(
                    {
                        "stage": "scoring_edges",
                        "processed": idx_s,
                        "total": total_to_score,
                        "percent": round(100.0 * idx_s / float(max(1, total_to_score)), 1),
                        "edges_per_sec": round(eps, 1),
                        "eta_sec": 0.0 if idx_s == total_to_score else round(float(total_to_score - idx_s) / max(eps, 1e-6), 1),
                    }
                )

    for e in edge_records:
        eid_all = str(e["edge_id"])
        if eid_all not in edge_scores:
            edge_scores[eid_all] = {"s_uv": 0.0, "h_uv": 0.0, "behavior_label": "neutral", "is_scored": False}

    # Build early PyG CascadeData restricted strictly to V_K (in-distribution for SBi-FAGAT)
    delays_gnn = [max(0.0, (n["created_at_utc"] - root_ts).total_seconds()) for n in gnn_nodes]
    delays_display = [max(0.0, (n["created_at_utc"] - root_ts).total_seconds()) for n in display_nodes]
    denom_t = math.log1p(TEMPORAL_REF_SECONDS)
    x_rows = []
    if frozen_node_encoder is not None and frozen_node_tok is not None:
        gnn_texts = [str(n["text"]) for n in gnn_nodes]
        gnn_parents = [
            None
            if (bool(n["is_root"]) or n.get("parent_id") is None)
            else str(node_by_id.get(str(n["parent_id"]), {}).get("text", ""))
            for n in gnn_nodes
        ]
        gnn_roots = [
            None
            if (bool(n["is_root"]) or n.get("parent_id") is None)
            else root_text_str
            for n in gnn_nodes
        ]
        emb_mat = encode_node_embeddings(
            gnn_texts,
            encoder=frozen_node_encoder,
            tokenizer=frozen_node_tok,
            parent_texts=gnn_parents,
            root_texts=gnn_roots,
            lang=lang,
            batch_size=32,
            device=live_device,
        )
        for idx_n, (n, dt_sec) in enumerate(zip(gnn_nodes, delays_gnn)):
            nid = str(n["node_id"])
            t_norm = min(1.0, math.log1p(dt_sec) / denom_t)
            d_norm = min(1.0, float(depth_map.get(nid, 0)) / float(max(1, horizon_for_features)))
            x_rows.append(torch.cat([emb_mat[idx_n].cpu(), torch.tensor([t_norm, d_norm], dtype=torch.float32)], dim=0))
    else:
        for n, dt_sec in zip(gnn_nodes, delays_gnn):
            nid = str(n["node_id"])
            emb_768 = _deterministic_text_projection(str(n["text"]), 768)
            t_norm = min(1.0, math.log1p(dt_sec) / denom_t)
            d_norm = min(1.0, float(depth_map.get(nid, 0)) / float(max(1, horizon_for_features)))
            x_rows.append(torch.cat([emb_768.cpu(), torch.tensor([t_norm, d_norm], dtype=torch.float32)], dim=0))
    x = torch.stack(x_rows, dim=0)

    td_src, td_dst = [], []
    edge_attr_rows: list[list[float]] = []
    gnn_pairs: list[tuple[str, str]] = []
    gnn_edge_details: list[dict[str, Any]] = []
    for n in gnn_nodes[1:]:
        dst_id = str(n["node_id"])
        src_id = str(n["parent_id"])
        if src_id not in gnn_id_to_idx:
            continue
        edge_row = edge_lookup.get((src_id, dst_id))
        eid = str(edge_row["edge_id"]) if edge_row is not None else f"{src_id}->{dst_id}"
        sc = edge_scores[eid]
        td_src.append(gnn_id_to_idx[src_id])
        td_dst.append(gnn_id_to_idx[dst_id])
        edge_attr_rows.append([sc["s_uv"], sc["h_uv"]])
        gnn_pairs.append((src_id, dst_id))
        gnn_edge_details.append(
            {
                "edge_id": eid,
                "parent_id": src_id,
                "node_id": dst_id,
                "reply_text": str(n["text"]),
                **sc,
            }
        )

    display_edge_details: list[dict[str, Any]] = []
    for n in display_nodes[1:]:
        dst_id = str(n["node_id"])
        src_id = str(n["parent_id"])
        if src_id not in display_id_to_idx:
            continue
        edge_row = edge_lookup.get((src_id, dst_id))
        eid = str(edge_row["edge_id"]) if edge_row is not None else f"{src_id}->{dst_id}"
        sc = edge_scores[eid]
        display_edge_details.append(
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
    edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float32) if edge_attr_rows else torch.empty((0, 2), dtype=torch.float32)
    struct_x = compute_causal_structural_node_features(
        kept_nodes=gnn_nodes,
        delays_k=delays_gnn,
        depth_map=depth_map,
        k_horizon=horizon_for_features,
    )
    cascade_stats = compute_early_cascade_burst_stats(
        kept_nodes=gnn_nodes,
        delays_k=delays_gnn,
        depth_map=depth_map,
        edge_attr_rows=edge_attr_rows,
        k_horizon=horizon_for_features,
    )

    early_children_adj: dict[str, list[str]] = {}
    early_edge_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    early_stance_map: dict[str, float] = {}
    for d_e in gnn_edge_details:
        pu = str(d_e["parent_id"])
        cv = str(d_e["node_id"])
        eid_e = str(d_e["edge_id"])
        early_children_adj.setdefault(pu, []).append(cv)
        early_edge_lookup[(pu, cv)] = {"edge_id": eid_e}
        early_stance_map[eid_e] = float(d_e["s_uv"])
    early_aligned = _compute_root_aligned_stances(
        root_id=root_id,
        children_adj=early_children_adj,
        edge_lookup=early_edge_lookup,
        stance_map=early_stance_map,
    )
    early_y_polar_val = _compute_full_tree_polarization(early_aligned, lambda_polar=0.5) if early_aligned else 0.0

    data = CascadeData(
        x=x,
        struct_x=struct_x,
        edge_index=edge_index_td,
        edge_index_td=edge_index_td,
        edge_index_bu=edge_index_bu,
        edge_attr=edge_attr,
        edge_stance=edge_attr[:, 0:1] if edge_attr.numel() > 0 else torch.empty((0, 1)),
        early_y_polar=torch.tensor([early_y_polar_val], dtype=torch.float32),
        cascade_stats=cascade_stats,
        root_index=torch.tensor([0], dtype=torch.long),
        edge_pairs=gnn_pairs,
    )

    # Load GNN checkpoint if present; otherwise check authorization before untuned inspection
    if gnn_ckpt is not None:
        gnn_path = Path(gnn_ckpt)
        if not gnn_path.is_file():
            raise FileNotFoundError(f"Checkpoint GNN không tồn tại tại đường dẫn: {gnn_path}")
        try:
            payload = torch.load(gnn_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(f"Tệp checkpoint GNN tại '{gnn_path}' bị lỗi hoặc không thể đọc: {exc}") from exc
        if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
            raise ValueError(f"Checkpoint GNN tại '{gnn_path}' thiếu trường 'model_state_dict'.")
        cfg = payload.get("config", {})
        state_dict = payload["model_state_dict"]
        target_edge_dim = int(cfg.get("edge_dim", 2))
        has_persist = (
            bool(cfg["use_polar_persistence_anchor"])
            if "use_polar_persistence_anchor" in cfg
            else ("head.persist_weight" in state_dict)
        )
        gnn_model = SBiFAGAT(
            in_dim=int(payload.get("in_dim", TOTAL_NODE_FEATURE_DIM)),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            edge_dim=target_edge_dim,
            num_rumor_classes=int(payload.get("num_rumor_classes", 4)),
            eps=float(cfg.get("eps", 0.5)),
            dropout=float(cfg.get("dropout", 0.1)),
            use_cascade_stats=bool(cfg.get("use_cascade_stats", True)),
            predict_uncertainty=bool(cfg.get("predict_uncertainty", False)),
            use_polar_persistence_anchor=has_persist,
        )
        gnn_model.load_state_dict(state_dict, strict=True)
        checkpoint_loaded = True
    else:
        is_demo_post = bool(is_demo or str(post.get("id", "")).startswith("demo_"))
        if not allow_untrained and not is_demo_post:
            raise RuntimeError(
                "Không tìm thấy checkpoint mô hình GNN đã huấn luyện (--gnn-ckpt / --checkpoint). "
                "Để ngăn ngừa kết quả ngẫu nhiên chưa huấn luyện bị hiểu nhầm là dự báo khoa học thực tế, "
                "hãy cung cấp đường dẫn checkpoint hợp lệ qua --gnn-ckpt, hoặc truyền cờ --allow-untrained "
                "(hoặc dùng --demo-offline) nếu bạn chỉ muốn chạy thử nghiệm kiểm tra cấu trúc pipeline."
            )
        target_edge_dim = int(ce_model.edge_dim) if ce_model is not None else 2
        gnn_model = SBiFAGAT(
            in_dim=TOTAL_NODE_FEATURE_DIM,
            hidden_dim=64,
            edge_dim=target_edge_dim,
            use_cascade_stats=True,
            predict_uncertainty=False,
        )
        checkpoint_loaded = False

    if data.edge_attr.numel() > 0:
        data.edge_attr = data.edge_attr[:, :target_edge_dim]
    else:
        data.edge_attr = torch.empty((0, target_edge_dim), dtype=torch.float32)

    diag = gnn_model.predict_with_diagnostics(data, top_k_edges=5)

    behavior_counts = {lbl: 0 for lbl in _UNIFIED_BEHAVIOR_LABELS}
    for d in gnn_edge_details:
        lbl_d = d["behavior_label"]
        if lbl_d in behavior_counts:
            behavior_counts[lbl_d] += 1
        else:
            mapped_lbl = _classify_2d_behavior_label(float(d["s_uv"]), float(d["h_uv"]))
            behavior_counts[mapped_lbl] += 1

    total_early_edges = max(1, len(gnn_edge_details))
    early_behavior_summary = {
        lbl: {
            "count": int(behavior_counts[lbl]),
            "ratio": round(float(behavior_counts[lbl]) / float(total_early_edges), 4),
        }
        for lbl in _UNIFIED_BEHAVIOR_LABELS
    }

    # Combine model prediction with interpretable early polarization indicator when checkpoint isn't loaded
    raw_pred_polar = float(diag["y_polar_pred"].view(-1)[0].item())
    raw_pred_replies = float(diag["predicted_total_replies"].view(-1)[0].item())
    raw_conf = diag.get("confidence_score")
    conf_score: float | None = float(raw_conf.view(-1)[0].item()) if raw_conf is not None else None
    uncertainty_available = bool(diag.get("uncertainty_available", conf_score is not None))

    actual_total_replies = len(non_roots)
    has_future = actual_total_replies > (len(gnn_nodes) - 1)
    gnn_pair_set = {(str(d_e["parent_id"]), str(d_e["node_id"])) for d_e in gnn_edge_details}
    scored_edges_only = [e for e in edges_to_score if bool(edge_scores.get(str(e["edge_id"]), {}).get("is_scored", True))]
    stance_map = {eid: sc["s_uv"] for eid, sc in edge_scores.items()}
    if has_future:
        if is_stratified_subsampled:
            sc_adj: dict[str, list[str]] = {}
            sc_lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
            for e_sc in scored_edges_only:
                sc_src, sc_dst = str(e_sc["source_id"]), str(e_sc["target_id"])
                sc_adj.setdefault(sc_src, []).append(sc_dst)
                sc_lookup[(sc_src, sc_dst)] = e_sc
            actual_aligned = _compute_root_aligned_stances(
                root_id=root_id,
                children_adj=sc_adj,
                edge_lookup=sc_lookup,
                stance_map=stance_map,
                excluded_pairs=gnn_pair_set,
            )
        else:
            actual_aligned = _compute_root_aligned_stances(
                root_id=root_id,
                children_adj=children_adj,
                edge_lookup=edge_lookup,
                stance_map=stance_map,
                excluded_pairs=gnn_pair_set,
            )
        actual_full_polar: float | None = (
            _compute_full_tree_polarization(actual_aligned, lambda_polar=0.5)
            if actual_aligned
            else 0.0
        )
        actual_polar_mode = "stratified_sample_future_only" if is_stratified_subsampled else "future_only"
    else:
        actual_aligned = []
        actual_full_polar = None
        actual_polar_mode = "no_future_replies"
    bootstrap_ci_95 = _bootstrap_polar_ci_95(actual_aligned) if is_stratified_subsampled and actual_aligned else None

    stats_vec = cascade_stats.view(-1).tolist()
    lang_lit: Literal["vi", "en"] = "en" if str(lang).lower() == "en" else "vi"
    regimes_polar = _classify_calibrated_regime(raw_pred_polar, behavior_counts, lang=lang_lit)
    q50_ref, q75_ref, q90_ref = get_polar_quantiles(lang_lit)
    calib_note = PHEME_QUANTILE_CALIBRATION_NOTE if lang_lit == "en" else VN_QUANTILE_CALIBRATION_NOTE

    relinked_pair_set: set[tuple[str, str]] = set()
    for item in junk_stats.get("relinked_edge_ids", []):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            relinked_pair_set.add((str(item[0]), str(item[1])))

    serialized_kept_nodes = [
        {
            "node_id": str(n["node_id"]),
            "parent_id": str(n["parent_id"]) if n.get("parent_id") is not None else None,
            "author_hash": str(n["user_id_hash"]) if n.get("user_id_hash") is not None else None,
            "depth": int(depth_map.get(str(n["node_id"]), 0)),
            "time_delay_sec": float(dt_sec),
            "text": str(n["text"]),
        }
        for n, dt_sec in zip(display_nodes, delays_display)
    ]
    serialized_kept_edges = []
    for d_e in display_edge_details:
        s_val = float(d_e["s_uv"])
        stance_cls = "support" if s_val > 0.1 else ("oppose" if s_val < -0.1 else "neutral")
        src_s = str(d_e["parent_id"])
        dst_s = str(d_e["node_id"])
        serialized_kept_edges.append(
            {
                "source_id": src_s,
                "target_id": dst_s,
                "s_uv": s_val,
                "h_uv": float(d_e["h_uv"]),
                "stance_class": stance_cls,
                "behavior_label": str(d_e["behavior_label"]),
                "relinked": (src_s, dst_s) in relinked_pair_set,
                "is_scored": bool(d_e.get("is_scored", True)),
            }
        )

    imp_by_target: dict[str, float] = {}
    for trig in diag["top_trigger_edges"]:
        pair = trig.get("pair")
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            imp_by_target[str(pair[1])] = float(trig.get("importance", 0.0))

    topic_snippets = [
        str(n.get("text") or "")
        for n in (display_nodes[1:12] if analysis_mode == "full" else gnn_nodes[1:6])
    ]
    topic_analysis = classify_post_topics(
        str(post.get("title") or ""),
        str(post.get("selftext") or post.get("body") or root_row.get("text") or ""),
        reply_snippets=topic_snippets,
        custom_topics=custom_topics,
        lang=lang_lit,
        backend=topic_backend,
        node_encoder=frozen_node_encoder,
        node_tokenizer=frozen_node_tok,
        allow_nli_download=allow_nli_topic_download,
    )

    if analysis_mode == "full":
        display_ids_set = {str(n["node_id"]) for n in display_nodes}
        if is_stratified_subsampled:
            full_scope_edges = [
                e for e in scored_edges_only
                if str(e["source_id"]) in display_ids_set and str(e["target_id"]) in display_ids_set
            ]
            participating_ids = {root_id} | {str(e["source_id"]) for e in full_scope_edges} | {str(e["target_id"]) for e in full_scope_edges}
            analysis_nodes = [n for n in display_nodes if str(n["node_id"]) in participating_ids]
        else:
            analysis_nodes = display_nodes
            full_scope_edges = [
                e for e in edge_records
                if str(e["source_id"]) in display_ids_set and str(e["target_id"]) in display_ids_set
            ]
        large_thread_analysis = analyze_large_thread_dynamics(
            root_node=root_row,
            all_nodes=analysis_nodes,
            all_edges=full_scope_edges,
            edge_scores=edge_scores,
            depth_map=depth_map,
            topic_analysis=topic_analysis,
            trigger_importance_by_target=imp_by_target,
            k_early_ref=k_horizon,
            lang=lang_lit,
            scope="full_analysis",
        )
    else:
        large_thread_analysis = analyze_large_thread_dynamics(
            root_node=root_row,
            all_nodes=node_records,
            all_edges=edges_to_score if is_stratified_subsampled else edge_records,
            edge_scores=edge_scores,
            depth_map=depth_map,
            topic_analysis=topic_analysis,
            trigger_importance_by_target=imp_by_target,
            k_early_ref=k_horizon,
            lang=lang_lit,
            scope="early_post_hoc",
        )

    if is_stratified_subsampled:
        large_thread_analysis["is_stratified_estimate"] = True
        large_thread_analysis["bootstrap_ci_95"] = bootstrap_ci_95
        large_thread_analysis["total_replies_kept_tree"] = len(display_nodes) - 1

    total_scope_edges = max(len(scope_edges), len(edges_to_score))
    coverage_ratio = round(float(len(edges_to_score)) / float(max(1, total_scope_edges)), 4)
    elapsed_ms = round((time.perf_counter() - t_scan_start) * 1000.0, 2)
    mega_scan_summary = {
        "scan_strategy": scan_strategy,
        "max_full_comments_cap": max_full_comments,
        "sample_budget": sample_budget if scan_strategy == "stratified" else None,
        "total_raw_comments": raw_comment_count,
        "kept_comments": len(non_roots),
        "analyzed_display_comments": len(display_nodes) - 1,
        "scored_edges": len(edges_to_score),
        "total_edges": total_scope_edges,
        "coverage_ratio": coverage_ratio,
        "bootstrap_ci_95": bootstrap_ci_95,
        "ood_scale_warning": ood_scale_warning,
        "training_max_replies_ref": TRAINING_MAX_REPLIES_REF,
        "elapsed_ms": elapsed_ms,
        "hotspot_branches": list(large_thread_analysis.get("hotspot_branches") or []),
    }

    return {
        "root_title": str(post.get("title") or root_row["text"][:120]),
        "lang": lang_lit,
        "analysis_mode": analysis_mode,
        "scan_strategy": scan_strategy,
        "gnn_k_horizon": len(gnn_nodes) - 1,
        "forecast_scope": "early_k_in_distribution",
        "k_observed": len(display_nodes) - 1,
        "edge_dim": int(target_edge_dim),
        "checkpoint_loaded": checkpoint_loaded,
        "cross_encoder_loaded": cross_encoder_loaded,
        "nlp_backend": nlp_backend,
        "junk_mode": junk_mode,
        "junk_filter_stats": dict(junk_stats),
        "ood_scale_warning": ood_scale_warning,
        "training_max_replies_ref": TRAINING_MAX_REPLIES_REF,
        "kept_nodes": serialized_kept_nodes,
        "kept_edges": serialized_kept_edges,
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
        "early_behavior_summary": early_behavior_summary,
        "early_y_polar": float(early_y_polar_val),
        "predicted_total_replies": raw_pred_replies,
        "predicted_polarization_score": raw_pred_polar,
        "predicted_regime": regimes_polar,
        "regime_thresholds": {
            "q50": q50_ref,
            "q75": q75_ref,
            "q90": q90_ref,
        },
        "note": calib_note,
        "confidence_score": conf_score,
        "uncertainty_available": uncertainty_available,
        "low_signal_warning": bool(diag["low_signal_warning"].view(-1)[0].item()),
        "top_trigger_edges": diag["top_trigger_edges"],
        "flagged_hostile": [
            d for d in gnn_edge_details if d["behavior_label"] == "hostile_conflict"
        ],
        "topic_analysis": topic_analysis,
        "large_thread_analysis": large_thread_analysis,
        "mega_scan_summary": mega_scan_summary,
        "actual_post_hoc_comparison": {
            "has_future_replies": has_future,
            "actual_total_replies": actual_total_replies,
            "actual_full_polarization": actual_full_polar,
            "actual_full_polarization_mode": actual_polar_mode,
        },
    }


def _format_report_vi(result: Mapping[str, Any]) -> str:
    lines = [
        "=" * 78,
        "BAO CAO DU BAO SOM & PHAN TICH HANH VI TRANH LUAN (SBi-FAGAT Unified 2D)",
        "=" * 78,
    ]
    if not result["checkpoint_loaded"]:
        lines.append(
            "[CHẾ ĐỘ DEMO OFFLINE - TRỌNG SỐ KHỞI TẠO NGẪU NHIÊN, CHỈ KIỂM TRA ĐƯỜNG ỐNG]"
        )
    lines.extend(
        [
            f"Bai goc      : {result['root_title']}",
            f"Cua so GNN du bao som (V_K): {result.get('gnn_k_horizon', result['k_observed'])} binh luan dau (in-distribution)",
            f"Pham vi hien thi & phan tich: {result['k_observed']} binh luan ({result.get('analysis_mode', 'early')})",
            f"Trang thai GNN       : {'Da nap checkpoint huan luyen (2D [s_uv, h_uv])' if result['checkpoint_loaded'] else 'CHE DO KIEM THU CAU TRUC (CHUA NAP CHECKPOINT - KHONG CO GIA TRI DU BAO)'}",
            f"Backend NLP          : {result.get('nlp_backend', 'unknown')} (cross_encoder_loaded={result.get('cross_encoder_loaded', False)})",
            f"Che do loc rac       : {result.get('junk_mode', 'prune')}",
            "",
            "[1] PHAN BO LAP TRUONG & GAY GAT 2D [s_uv, h_uv] TRONG 20 BINH LUAN DAU:",
        ]
    )
    bc = result["behavior_counts"]
    lines.append(f"  - {_BEHAVIOR_LABEL_VI_2D['support']} (s > 0.1)           : {bc.get('support', 0)} binh luan")
    lines.append(f"  - {_BEHAVIOR_LABEL_VI_2D['neutral']}                    : {bc.get('neutral', 0)} binh luan")
    lines.append(f"  - {_BEHAVIOR_LABEL_VI_2D['constructive_oppose']} (s < -0.1, h <= 0.6)  : {bc.get('constructive_oppose', 0)} binh luan")
    lines.append(f"  - {_BEHAVIOR_LABEL_VI_2D['hostile_conflict']} (h > 0.6)       : {bc.get('hostile_conflict', 0)} binh luan")

    bm = result["early_burst_metrics"]
    sec3_header = (
        "[3] DU BAO TUONG LAI & DO TU TIN:"
        if result["checkpoint_loaded"]
        else "[3] DU BAO TUONG LAI: [CANH BAO: CHUA NAP CHECKPOINT - CAC CHI SO BEN DUOI LA NGAU NHIEN]"
    )
    conf_val = result.get("confidence_score")
    conf_str = (
        f"{float(conf_val):.1%}"
        if conf_val is not None
        else "N/A (uncertainty disabled)"
    )
    lines.extend(
        [
            "",
            "[2] DONG HO TOC DO & CAU TRUC NHANH SOM:",
            f"  - Gia toc binh luan (>0.5 la dang tang toc): {bm['burst_acceleration']:.3f}",
            f"  - Ty le cau co noi dung thuc chat          : {bm['substantive_text_ratio']:.1%}",
            f"  - Chi so kich dong 1 chieu                 : {bm['one_sided_hostility_index']:.3f}",
            "",
            sec3_header,
            f"  - Quy mo binh luan cuoi ky du kien         : ~{result['predicted_total_replies']:.1f} binh luan",
            f"  - Chi so phan cuc du kien [0..1]           : {result['predicted_polarization_score']:.4f}",
            f"  - Danh gia cuc dien                        : {result['predicted_regime']}",
            f"  - Do tu tin du bao                         : {conf_str}",
        ]
    )
    if result["low_signal_warning"]:
        lines.append("  [!] CANH BAO TIN HIEU THAP: Binh luan dau nhat/thieu du kien -> Khuyen nghi cho them den moc K=35-50.")

    flagged = result.get("flagged_hostile") or []
    lines.append("")
    lines.append(f"[4] DIEM MAT BINH LUAN GAY GAT / XUNG DOT CAO ({len(flagged)} cau):")
    for idx, item in enumerate(flagged[:5], start=1):
        lines.append(
            f"  {idx}. [Xung dot gay gat] (s_uv={item['s_uv']:+.2f}, h_uv={item['h_uv']:.2f})"
        )
        lines.append(f"     \"{item['reply_text'][:160]}\"")

    act = result["actual_post_hoc_comparison"]
    early_pol = result.get("early_y_polar")
    early_pol_str = f"{float(early_pol):.4f}" if early_pol is not None else "N/A"
    act_pol = act.get("actual_full_polarization")
    if act.get("has_future_replies") and act_pol is not None:
        act_pol_str = f"{float(act_pol):.4f} (future_only, t > K)"
    else:
        act_pol_str = "N/A (chua co binh luan t > K)"
    lines.extend(
        [
            "",
            "[5] DOI CHIEU THUC TE (KHONG LOT VAO DAU VAO MO HINH):",
            f"  - Tong so binh luan thuc te toan bai       : {act['actual_total_replies']} binh luan",
            f"  - Do phan cuc quan sat som (1..K)          : {early_pol_str}",
            f"  - Do phan cuc tuong lai thuc te (t > K)    : {act_pol_str}",
        ]
    )
    topic_info = result.get("topic_analysis") or {}
    large_info = result.get("large_thread_analysis") or {}
    sem_info = large_info.get("semantic_understanding") or {}
    if topic_info:
        sec_str = ", ".join(topic_info.get("secondary_topics") or ()) or "Không"
        lines.extend(
            [
                "",
                "[6] PHAN LOAI CHU DE & HIEU BAI LON (AI):",
                f"  - Chu de chinh ({topic_info.get('topic_backend', 'lexical')}) : "
                f"{topic_info.get('primary_topic', 'N/A')} ({float(topic_info.get('primary_confidence', 0.0)):.1%})",
                f"  - Chu de phu                             : {sec_str}",
            ]
        )
    if large_info:
        lines.extend(
            [
                f"  - Quy mo & Pham vi ({large_info.get('scope', 'full')})   : "
                f"{large_info.get('scale_label_vi', '')} ({large_info.get('total_replies_analyzed', 0)} binh luan, {large_info.get('unique_authors_total', 0)} tac gia)",
                f"  - Tinh chat thao luan                    : {large_info.get('post_archetype_label_vi', '')}",
                f"  - Xu huong tien hoa                      : {large_info.get('evolution_trend_label_vi', '')}",
                f"  - Tong hop trich xuat (Extractive)       : {sem_info.get('summary_vi', '')}",
            ]
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Live social thread early polarization & virality forecast.")
    parser.add_argument("--reddit-url", "--url", dest="reddit_url", type=str, default=None)
    parser.add_argument("--thread-id", type=str, default=None)
    parser.add_argument("--input-json", "--input-file", dest="input_json", type=Path, default=None)
    parser.add_argument("--demo-offline", action="store_true")
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Cho phep chay voi trong so ngau nhien chua nap checkpoint (chi phuc vu kiem tra cau truc ky thuat).",
    )
    parser.add_argument("--cross-encoder-ckpt", type=Path, default=None)
    parser.add_argument("--gnn-ckpt", "--checkpoint", dest="gnn_ckpt", type=Path, default=None)
    parser.add_argument("--node-backbone", type=str, default=None)
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--analysis-mode", choices=["early", "full"], default="early")
    parser.add_argument("--max-full-comments", type=int, default=500)
    parser.add_argument("--scan-strategy", choices=["full", "stratified"], default="full")
    parser.add_argument("--sample-budget", type=int, default=600)
    parser.add_argument("--topics", type=str, default=None, help="Danh sach chu de tuy chinh (ngan cach bang dau phay).")
    parser.add_argument("--topic-backend", choices=["auto", "nli", "embedding", "lexical"], default="auto")
    parser.add_argument(
        "--allow-nli-topic-download",
        action="store_true",
        help="Cho phep tai mDeBERTa-v3-xnli tu HuggingFace neu chua co trong cache cuc bo.",
    )
    parser.add_argument("--lang", choices=["vi", "en"], default="vi")
    parser.add_argument("--junk-mode", choices=["prune", "contract"], default="prune")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.k_horizon < 1:
        raise ValueError(f"k_horizon must be >= 1, got {args.k_horizon}")
    if args.max_full_comments < 1 or args.max_full_comments > 10000:
        raise ValueError(f"max_full_comments must be in [1, 10000], got {args.max_full_comments}")
    if args.sample_budget < 1:
        raise ValueError(f"sample_budget must be >= 1, got {args.sample_budget}")
    if (args.reddit_url or args.thread_id) and args.gnn_ckpt is None and not args.allow_untrained:
        raise RuntimeError(
            "Không tìm thấy checkpoint mô hình GNN (--gnn-ckpt). "
            "Không chạy suy luận trên bài đăng thật bằng trọng số khởi tạo ngẫu nhiên; "
            "hãy truyền --gnn-ckpt hoặc --allow-untrained."
        )

    if args.demo_offline:
        post, comments = _make_demo_offline_thread()
    elif args.input_json is not None:
        post, comments = load_thread_from_file(args.input_json, max_comments=args.max_full_comments)
    elif args.reddit_url or args.thread_id:
        pid = (
            _validate_reddit_post_id(args.thread_id, source="--thread-id")
            if args.thread_id
            else _extract_post_id_from_url(str(args.reddit_url))
        )
        post, comments = (
            fetch_reddit_thread_paginated(pid, max_comments=args.max_full_comments)
            if args.max_full_comments > 500
            else _fetch_reddit_thread(pid)
        )
    else:
        raise ValueError("Provide one of --demo-offline, --input-json, --reddit-url, or --thread-id")

    res = predict_live_cascade(
        post,
        comments,
        k_horizon=args.k_horizon,
        analysis_mode=args.analysis_mode,
        max_full_comments=args.max_full_comments,
        scan_strategy=args.scan_strategy,
        sample_budget=args.sample_budget,
        custom_topics=args.topics,
        topic_backend=args.topic_backend,
        allow_nli_topic_download=args.allow_nli_topic_download,
        cross_encoder_ckpt=args.cross_encoder_ckpt,
        gnn_ckpt=args.gnn_ckpt,
        node_backbone=args.node_backbone,
        lang=args.lang,
        junk_mode=args.junk_mode,
        allow_untrained=args.allow_untrained,
        is_demo=args.demo_offline,
    )
    print(_format_report_vi(res))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

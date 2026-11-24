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
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import requests
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

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
from src.nlp.discourse_labels import (
    BEHAVIOR_LABEL_VI,
    DISCOURSE_BEHAVIOR_LABELS,
    _heuristic_behavior_fallback,
)
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

_REDDIT_POST_ID_RE = re.compile(r"/comments/([^/?#]+)")
_VALID_POST_ID_RE = re.compile(r"[A-Za-z0-9]{4,12}")
_REQUIRED_CE_FILES: tuple[str, ...] = (
    "dual_head_weights.pt",
    "dual_head_meta.json",
    "config.json",
)

# Empirical future_only y_polar quantiles computed strictly on the VN Train split
# (N = 1,050 cascades, K = 20, min_total_replies = 25; data/processed/vn/nodes.parquet
# UTC cutoff 2025-06-15 to 2026-06-04; topic_splits.json train split only — zero Val/Test leakage).
VN_POLAR_Q50: float = 0.0927623026072979
VN_POLAR_Q75: float = 0.12019770219922066
VN_POLAR_Q90: float = 0.15035123974084857
VN_QUANTILE_CALIBRATION_NOTE: str = (
    "Regime thresholds calibrated strictly on VN Train split "
    "(N=1,050 cascades, K=20, min_total_replies=25, UTC 2025-06-15 to 2026-06-04); zero Val/Test leakage."
)


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


def _fetch_reddit_thread(post_id: str, timeout: int = 20) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fetch submission + comments via Arctic Shift API with Reddit public JSON fallback."""
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
            comments = []
            for c in raw_comments:
                cid = str(c.get("id") or "").removeprefix("t1_")
                raw_author = c.get("author")
                comments.append(
                    {
                        "fullname": f"t1_{cid}",
                        "id": cid,
                        "link_id": str(c.get("link_id") or f"t3_{post_id}"),
                        "parent_id": str(c.get("parent_id") or f"t3_{post_id}"),
                        "body": str(c.get("body") or c.get("text") or ""),
                        "text": str(c.get("body") or c.get("text") or ""),
                        "created_utc": c.get("created_utc"),
                        "author": raw_author,
                        "author_is_bot": isinstance(raw_author, str)
                        and raw_author.strip().lower() == "automoderator",
                        "author_hash": hash_author_sha256(
                            str(raw_author) if raw_author is not None else None
                        ),
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
                raw_author = d.get("author")
                flat.append(
                    {
                        "fullname": f"t1_{cid}",
                        "id": cid,
                        "link_id": f"t3_{post_id}",
                        "parent_id": str(d.get("parent_id") or f"t3_{post_id}"),
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
            raw_author = c["author"] if "author" in c else f"u_{idx}"
            author_str = str(raw_author) if raw_author is not None else None
            comments.append(
                {
                    "fullname": f"t1_{cid.removeprefix('t1_')}",
                    "id": cid.removeprefix("t1_"),
                    "link_id": f"t3_{post_id}",
                    "parent_id": pid,
                    "text": str(c.get("text") or c.get("body") or ""),
                    "created_utc": float(c.get("created_utc") or (base_ts + idx * 45.0)),
                    "author": author_str,
                    "author_is_bot": isinstance(author_str, str)
                    and author_str.strip().lower() == "automoderator",
                    "author_hash": hash_author_sha256(author_str),
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
) -> tuple[DualHeadCrossEncoder | None, Any | None, Any | None, Any | None, bool, str]:
    """Load available Cross-Encoder & SpanPool backends without unwanted offline downloads."""
    ce_dir = Path(cross_encoder_ckpt) if cross_encoder_ckpt is not None else None
    ce_model: DualHeadCrossEncoder | None = None
    ce_tok: Any | None = None

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
    allow_net = bool(ce_model is not None and not is_demo)
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


def _classify_calibrated_regime(
    pred_polar: float,
    behavior_counts: Mapping[str, int],
) -> str:
    """Classify cascade polarization regime calibrated strictly against VN Train quantiles."""
    hostile_or_irr = int(behavior_counts.get("hostile_conflict", 0)) + int(
        behavior_counts.get("irrational_fallacy", 0)
    )
    if pred_polar >= VN_POLAR_Q90 or hostile_or_irr >= 3:
        return "Phân cực xung đột gay gắt (>= Q90 VN Train)"
    if pred_polar >= VN_POLAR_Q75 or int(behavior_counts.get("constructive_oppose", 0)) >= 2:
        return "Tranh luận đa chiều văn minh (Q75-Q90 VN Train)"
    if pred_polar >= VN_POLAR_Q50:
        return "Phân hóa trung bình (Q50-Q75 VN Train)"
    return "Tương đối đồng thuận / ít chia rẽ (< Q50 VN Train)"


def predict_live_cascade(
    post: Mapping[str, Any],
    comments: Sequence[Mapping[str, Any]],
    *,
    k_horizon: int = 20,
    cross_encoder_ckpt: Path | None = None,
    gnn_ckpt: Path | None = None,
    node_backbone: str | None = None,
    lang: str = "vi",
    junk_mode: Literal["prune", "contract"] = "prune",
    allow_untrained: bool = False,
    is_demo: bool = False,
) -> dict[str, Any]:
    """Build cascade graph, score early behavior, run SBiFAGAT, and emit live forecast report."""
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

    kept_nodes = [root_row] + non_roots[:k_horizon]
    kept_id_to_idx = {str(n["node_id"]): idx for idx, n in enumerate(kept_nodes)}

    # Score edges (all edges for post-hoc check; strictly isolated from model features beyond K)
    ce_model, ce_tok, frozen_node_encoder, frozen_node_tok, cross_encoder_loaded, nlp_backend = (
        _resolve_nlp_backends(
            cross_encoder_ckpt=cross_encoder_ckpt,
            node_backbone=node_backbone,
            lang=lang,
            is_demo=is_demo,
        )
    )

    use_root_ctx = bool(getattr(ce_model, "include_root_context", False)) if ce_model is not None else False
    root_text_str = str(root_row["text"])
    edge_scores: dict[str, dict[str, Any]] = {}
    all_edge_ids = [str(e["edge_id"]) for e in edge_records]
    all_parent_texts = [str(node_by_id[str(e["source_id"])]["text"]) for e in edge_records]
    all_reply_texts = [str(node_by_id[str(e["target_id"])]["text"]) for e in edge_records]

    if all_edge_ids and ce_model is not None and ce_tok is not None:
        ce_out = encode_edges_with_cross_encoder(
            all_parent_texts,
            all_reply_texts,
            cross_encoder=ce_model,
            tokenizer=ce_tok,
            root_texts=[root_text_str] * len(edge_records),
            include_root_context=use_root_ctx,
            lang=lang,
            batch_size=32,
            device=torch.device("cpu"),
        )
        for pos, eid in enumerate(all_edge_ids):
            s_v = float(ce_out["s_uv"][pos])
            h_v = float(ce_out["h_uv"][pos])
            r_v = float(ce_out["r_uv"][pos]) if "r_uv" in ce_out else 0.10
            c_v = float(ce_out["c_uv"][pos]) if "c_uv" in ce_out else 0.50
            if "behavior_label" in ce_out:
                blbl = str(ce_out["behavior_label"][pos])
            else:
                blbl = "support" if s_v > 0.1 else ("hostile_conflict" if h_v > 0.6 else ("constructive_oppose" if s_v < -0.1 else "neutral"))
            edge_scores[eid] = {"s_uv": s_v, "h_uv": h_v, "r_uv": r_v, "c_uv": c_v, "behavior_label": blbl}
    else:
        for eid, r_txt in zip(all_edge_ids, all_reply_texts):
            heur = _heuristic_behavior_fallback(r_txt)
            edge_scores[eid] = {
                "s_uv": float(heur["s_uv"]),
                "h_uv": float(heur["h_uv"]),
                "r_uv": float(heur["r_uv"]),
                "c_uv": float(heur["c_uv"]),
                "behavior_label": str(heur["label"]),
            }

    # Build early PyG CascadeData restricted strictly to V_K
    delays_k = [max(0.0, (n["created_at_utc"] - root_ts).total_seconds()) for n in kept_nodes]
    denom_t = math.log1p(TEMPORAL_REF_SECONDS)
    x_rows = []
    if frozen_node_encoder is not None and frozen_node_tok is not None:
        kept_texts = [str(n["text"]) for n in kept_nodes]
        kept_parents = [
            None
            if (bool(n["is_root"]) or n.get("parent_id") is None)
            else str(node_by_id.get(str(n["parent_id"]), {}).get("text", ""))
            for n in kept_nodes
        ]
        kept_roots = [
            None
            if (bool(n["is_root"]) or n.get("parent_id") is None)
            else root_text_str
            for n in kept_nodes
        ]
        emb_mat = encode_node_embeddings(
            kept_texts,
            encoder=frozen_node_encoder,
            tokenizer=frozen_node_tok,
            parent_texts=kept_parents,
            root_texts=kept_roots,
            lang=lang,
            batch_size=32,
            device=torch.device("cpu"),
        )
        for idx_n, (n, dt_sec) in enumerate(zip(kept_nodes, delays_k)):
            nid = str(n["node_id"])
            t_norm = min(1.0, math.log1p(dt_sec) / denom_t)
            d_norm = min(1.0, float(depth_map.get(nid, 0)) / float(max(1, k_horizon)))
            x_rows.append(torch.cat([emb_mat[idx_n].cpu(), torch.tensor([t_norm, d_norm], dtype=torch.float32)], dim=0))
    else:
        for n, dt_sec in zip(kept_nodes, delays_k):
            nid = str(n["node_id"])
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
    struct_x = compute_causal_structural_node_features(
        kept_nodes=kept_nodes,
        delays_k=delays_k,
        depth_map=depth_map,
        k_horizon=k_horizon,
    )
    cascade_stats = compute_early_cascade_burst_stats(
        kept_nodes=kept_nodes,
        delays_k=delays_k,
        depth_map=depth_map,
        edge_attr_rows=edge_attr_rows,
        k_horizon=k_horizon,
    )

    early_children_adj: dict[str, list[str]] = {}
    early_edge_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    early_stance_map: dict[str, float] = {}
    for d_e in kept_edge_details:
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
    early_y_polar_val = _compute_full_tree_polarization(early_aligned) if early_aligned else 0.0

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
        edge_pairs=kept_pairs,
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
    regimes_polar = _classify_calibrated_regime(raw_pred_polar, behavior_counts)

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
        for n, dt_sec in zip(kept_nodes, delays_k)
    ]
    serialized_kept_edges = []
    for d_e in kept_edge_details:
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
            }
        )

    return {
        "root_title": str(post.get("title") or root_row["text"][:120]),
        "k_observed": len(kept_nodes) - 1,
        "edge_dim": int(target_edge_dim),
        "checkpoint_loaded": checkpoint_loaded,
        "cross_encoder_loaded": cross_encoder_loaded,
        "nlp_backend": nlp_backend,
        "junk_mode": junk_mode,
        "junk_filter_stats": dict(junk_stats),
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
        "predicted_total_replies": raw_pred_replies,
        "predicted_polarization_score": raw_pred_polar,
        "predicted_regime": regimes_polar,
        "regime_thresholds": {
            "q50": VN_POLAR_Q50,
            "q75": VN_POLAR_Q75,
            "q90": VN_POLAR_Q90,
        },
        "note": VN_QUANTILE_CALIBRATION_NOTE,
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
    is_2d = int(result.get("edge_dim", 2)) == 2 and bool(result.get("checkpoint_loaded"))
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
            f"Cua so quan sat (V_K): {result['k_observed']} binh luan dau tien",
            f"Trang thai GNN       : {'Da nap checkpoint huan luyen (2D [s_uv, h_uv])' if is_2d else ('Da nap checkpoint huan luyen' if result['checkpoint_loaded'] else 'CHE DO KIEM THU CAU TRUC (CHUA NAP CHECKPOINT - KHONG CO GIA TRI DU BAO)')}",
            f"Backend NLP          : {result.get('nlp_backend', 'unknown')} (cross_encoder_loaded={result.get('cross_encoder_loaded', False)})",
            f"Che do loc rac       : {result.get('junk_mode', 'prune')}",
            "",
            (
                "[1] PHAN BO LAP TRUONG & GAY GAT 2D [s_uv, h_uv] TRONG 20 BINH LUAN DAU:"
                if is_2d
                else "[1] PHAN LOAI 5 NHOM HANH VI TRONG 20 BINH LUAN DAU:"
            ),
        ]
    )
    if is_2d:
        bc = result["behavior_counts"]
        lines.append(f"  - Ung ho / dong tinh (s > 0.1)           : {bc.get('support', 0)} binh luan")
        lines.append(f"  - Trung lap / hoi dap                    : {bc.get('neutral', 0)} binh luan")
        lines.append(f"  - Phan bien on hoa (s < -0.1, h <= 0.6)  : {bc.get('constructive_oppose', 0)} binh luan")
        lines.append(f"  - Xung dot / gay gat cao (h > 0.6)       : {bc.get('hostile_conflict', 0)} binh luan")
    else:
        for lbl in DISCOURSE_BEHAVIOR_LABELS:
            cnt = result["behavior_counts"].get(lbl, 0)
            lines.append(f"  - {BEHAVIOR_LABEL_VI.get(lbl, lbl):38s}: {cnt} binh luan")

    bm = result["early_burst_metrics"]
    sec3_header = (
        "[3] DU BAO TUONG LAI & DO TU TIN:"
        if result["checkpoint_loaded"]
        else "[3] DU BAO TUONG LAI: [CANH BAO: CHUA NAP CHECKPOINT - CAC CHI SO BEN DUOI LA NGAU NHIEN]"
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
            f"  - Do tu tin du bao                         : {result['confidence_score']:.1%}",
        ]
    )
    if result["low_signal_warning"]:
        lines.append("  [!] CANH BAO TIN HIEU THAP: Binh luan dau nhat/thieu du kien -> Khuyen nghi cho them den moc K=35-50.")

    flagged = result.get("flagged_irrational_or_hostile") or []
    lines.append("")
    if is_2d:
        lines.append(f"[4] DIEM MAT BINH LUAN GAY GAT / XUNG DOT CAO ({len(flagged)} cau):")
        for idx, item in enumerate(flagged[:5], start=1):
            lines.append(
                f"  {idx}. [Xung dot gay gat] (s_uv={item['s_uv']:+.2f}, h_uv={item['h_uv']:.2f})"
            )
            lines.append(f"     \"{item['reply_text'][:160]}\"")
    else:
        lines.append(f"[4] DIEM MAT BINH LUAN VO LY / CHONG DOI NGUY HIEM NHAT ({len(flagged)} cau):")
        for idx, item in enumerate(flagged[:5], start=1):
            lbl_vi = BEHAVIOR_LABEL_VI.get(item["behavior_label"], item["behavior_label"])
            lines.append(
                f"  {idx}. [{lbl_vi}] (s={item['s_uv']:+.2f}, gay_gat={item['h_uv']:.2f}, vo_ly={item['r_uv']:.2f})"
            )
            lines.append(f"     \"{item['reply_text'][:160]}\"")

    act = result["actual_post_hoc_comparison"]
    polar_note = (
        "(toan bo cay chua nhieu binh luan hon K)"
        if act.get("has_future_replies")
        else "(do so binh luan thuc te <= K nen E[y_polar | stances quan sat])"
    )
    lines.extend(
        [
            "",
            "[5] DOI CHIEU THUC TE (KHONG LOT VAO DAU VAO MO HINH):",
            f"  - Tong so binh luan thuc te toan bai       : {act['actual_total_replies']} binh luan",
            f"  - Do phan cuc quan sat/cuoi ky thuc te     : {act['actual_full_polarization']:.4f} {polar_note}",
            "=" * 78,
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Live social thread early polarization & virality forecast.")
    parser.add_argument("--reddit-url", "--url", dest="reddit_url", type=str, default=None)
    parser.add_argument("--thread-id", type=str, default=None)
    parser.add_argument("--input-json", type=Path, default=None)
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
    parser.add_argument("--lang", choices=["vi", "en"], default="vi")
    parser.add_argument("--junk-mode", choices=["prune", "contract"], default="prune")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.demo_offline:
        post, comments = _make_demo_offline_thread()
    elif args.input_json is not None:
        post, comments = _load_thread_from_json(args.input_json)
    elif args.reddit_url or args.thread_id:
        pid = (
            _validate_reddit_post_id(args.thread_id, source="--thread-id")
            if args.thread_id
            else _extract_post_id_from_url(str(args.reddit_url))
        )
        post, comments = _fetch_reddit_thread(pid)
    else:
        raise ValueError("Provide one of --demo-offline, --input-json, --reddit-url, or --thread-id")

    res = predict_live_cascade(
        post,
        comments,
        k_horizon=args.k_horizon,
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

"""Vietnamese social discussion crawler and tree builder via Arctic Shift REST API.

Collects public Reddit discussion threads from `r/vozforums` and `r/VietNam`
without requiring Reddit API keys, reconstructs multi-level reply trees via
`link_id` (`t3_`) and `parent_id` (`t1_` / `t3_`), hashes author identities with
SHA-256 upon ingestion, normalizes Unicode to NFC while preserving emojis and
slang, prunes deleted/removed/spam nodes safely, enforces UTC temporal causality,
and exports validated `nodes.parquet` and `edges.parquet` tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pyarrow.parquet as pq
import requests

from src.data.schema import (
    build_edges_table,
    build_nodes_table,
    save_cascade_parquet,
)
from src.utils.text import normalize_nfc

ARCTIC_SHIFT_API_BASE = "https://arctic-shift.photon-reddit.com/api"
TARGET_SUBREDDITS: tuple[str, ...] = ("vozforums", "VietNam")
DEFAULT_USER_AGENT = "SBi-FAGAT-Research/1.0 (Academic Cascade Analysis)"

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_SPAM_DOMAIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:shope\.ee|s\.shopee\.vn|bit\.ly|tinyurl\.com|t\.me)/\S*",
    re.IGNORECASE,
)
_CRYPTO_SPAM_RE = re.compile(r"\blmgx\b", re.IGNORECASE)
_VN_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?84|0)(?:3|5|7|8|9)\d(?:[\.\-\s]?\d){7,8}(?!\d)"
)
_BANK_ACCT_RE = re.compile(
    r"(\b(?:stk|số\s*tài\s*khoản|so\s*tai\s*khoan|tài\s*khoản\s*ngân\s*hàng|"
    r"vietcombank|vcb|techcombank|tcb|mb\s?bank|bidv|vietinbank|acb|tpbank|"
    r"vpbank|agribank|sacombank|momo|zalopay)\b[^0-9\n]{0,40}?)(\b\d{8,18}\b)",
    re.IGNORECASE,
)
_CCCD_RE = re.compile(
    r"(\b(?:cccd|cmnd|căn\s*cước(?:\s*công\s*dân)?|chứng\s*minh(?:\s*nhân\s*dân)?)"
    r"\b[^0-9\n]{0,25}?)(\b\d{9,12}\b)",
    re.IGNORECASE,
)
_FB_PROFILE_RE = re.compile(
    r"https?://(?:www\.|m\.|mbasic\.)?(?:facebook|fb)\.com/"
    r"(?!groups/|watch/|events/|share/|photo|reel|pages/)\S+",
    re.IGNORECASE,
)
_DOXXING_INTENT_RE = re.compile(
    r"(?:truy\s+tìm|tìm\s+đối\s+tượng|tìm\s+thông\s+tin\s+người\s+nhà|"
    r"bóc\s+phốt.{0,30}(?:sđt|stk|địa\s+chỉ|cccd)|chia\s+sẻ\s+thông\s+tin\s+cá\s+nhân)",
    re.IGNORECASE,
)
_VN_DIACRITIC_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
    r"ùúủũụưừứửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    r"ÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]"
)


_SALT_WARNING_EMITTED = False


def hash_author_sha256(author: str | None, salt: str | None = None) -> str | None:
    """Pseudonymize an author handle with SHA-256 (returns None for deleted/bots).

    Note: SHA-256 without a secret salt provides pseudonymization (reversible via
    dictionary / rainbow-table attacks for known usernames) rather than full anonymization.
    If `salt` is not provided, reads `os.environ.get('SBIF_HASH_SALT')` or
    `data/interim/.author_hash_salt.secret` if present.
    """
    if author is None:
        return None
    cleaned = author.strip()
    if not cleaned or cleaned.lower() in {"[deleted]", "[removed]", "automoderator"}:
        return None

    effective_salt = salt
    if effective_salt is None:
        effective_salt = os.environ.get("SBIF_HASH_SALT")
    if effective_salt is None:
        secret_file = Path("data/interim/.author_hash_salt.secret")
        if secret_file.is_file():
            try:
                effective_salt = secret_file.read_text(encoding="utf-8").strip()
            except Exception:
                effective_salt = None

    if not effective_salt:
        global _SALT_WARNING_EMITTED
        if not _SALT_WARNING_EMITTED:
            warnings.warn(
                "SBIF_HASH_SALT is unset or empty: Author IDs are pseudonymized using unsalted SHA-256, "
                "which is vulnerable to dictionary / rainbow-table re-identification. "
                "Set SBIF_HASH_SALT in your environment or data/interim/.author_hash_salt.secret for salted pseudonymization.",
                UserWarning,
                stacklevel=2,
            )
            _SALT_WARNING_EMITTED = True
        payload = cleaned
    else:
        payload = f"{effective_salt}{cleaned}"

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_social_text_nfc(text: str) -> str:
    """Normalize text to Unicode NFC while preserving emojis and social slang."""
    return normalize_nfc(text)


def redact_pii_text(text: str) -> str:
    """Redact Vietnamese phone numbers, bank accounts, national IDs, and personal FB profile URLs."""
    cleaned = str(text or "")
    if not cleaned:
        return ""
    cleaned = _FB_PROFILE_RE.sub("[REDACTED_PII]", cleaned)
    cleaned = _BANK_ACCT_RE.sub(r"\1[REDACTED_PII]", cleaned)
    cleaned = _CCCD_RE.sub(r"\1[REDACTED_PII]", cleaned)
    cleaned = _VN_PHONE_RE.sub("[REDACTED_PII]", cleaned)
    return cleaned


def is_doxxing_post(text: str) -> bool:
    """Return True if *text* exhibits doxxing intent alongside personal identifiers (PII)."""
    norm = normalize_social_text_nfc(text)
    if not norm:
        return False
    if not _DOXXING_INTENT_RE.search(norm):
        return False
    return bool(
        _VN_PHONE_RE.search(norm)
        or _BANK_ACCT_RE.search(norm)
        or _CCCD_RE.search(norm)
        or _FB_PROFILE_RE.search(norm)
        or re.search(r"địa\s+chỉ|sinh\s+năm|sđt|stk", norm, re.IGNORECASE)
    )


def is_invalid_or_spam_text(text: str, author_raw: str | None = None) -> bool:
    """Return True for empty, deleted/removed, AutoModerator, doxxing, or link/crypto-spam content."""
    if isinstance(author_raw, str) and author_raw.strip().lower() == "automoderator":
        return True

    norm = normalize_social_text_nfc(text)
    if not norm or norm.lower() in {"[deleted]", "[removed]"}:
        return True

    if _SPAM_DOMAIN_RE.search(norm) or _CRYPTO_SPAM_RE.search(norm):
        return True

    if is_doxxing_post(norm):
        return True

    urls = _URL_RE.findall(norm)
    if urls:
        text_without_urls = _URL_RE.sub(" ", norm)
        alnum_remaining = re.sub(r"[^\w]", "", text_without_urls)
        if len(alnum_remaining) < 2:
            return True
        if len(urls) >= 3 and len(alnum_remaining) < 20:
            return True

    return False


_EXPLICIT_FILLER_PHRASES: frozenset[str] = frozenset(
    {
        "hong",
        "hóng",
        "cham",
        "chấm",
        "up",
        "bump",
        "cmt",
        "ke",
        "ké",
        "xin slot",
        "cham hong",
        "chấm hóng",
        "hong ke",
        "hóng ké",
        "lot dep",
        "lót dép",
        "lot dep hong",
        "lót dép hóng",
        "dau chan",
        "dấu chân",
        "dat gach",
        "đặt gạch",
        "ok",
        "oke",
    }
)


def is_filler_junk_text(text: str, author_raw: str | None = None) -> bool:
    """Return True for invalid/spam comments, punctuation/emoji-only noise, or explicit fillers.

    Preserves meaningful short Vietnamese stances (e.g., ``Đồng ý``, ``Phản đối``,
    ``Không nên``, ``Chuẩn bác``, ``Đúng rồi``, ``Sai rồi``, ``+1``, ``Đồng tình``,
    ``Ủng hộ``, ``Tào lao``) instead of applying a blind token-length cutoff.
    """
    if is_invalid_or_spam_text(text, author_raw=author_raw):
        return True

    norm = normalize_social_text_nfc(text).strip()
    if not norm:
        return True
    if norm in {"+1", "-1"}:
        return False
    if not any(ch.isalnum() for ch in norm):
        return True

    stripped = re.sub(r"[^\w\s]", " ", norm.lower(), flags=re.UNICODE)
    collapsed = " ".join(stripped.split())
    if not collapsed:
        return True
    return collapsed in _EXPLICIT_FILLER_PHRASES


def resolve_contracted_parent_id(
    node_id: str,
    raw_parent_map: Mapping[str, str],
    kept_ids: set[str],
    root_id: str,
    *,
    traversed_junk_out: set[str] | None = None,
    traversed_chain_out: list[str] | None = None,
    truncated_orphans_out: set[str] | None = None,
    drop_truncated_orphans: bool = False,
) -> str | None:
    """Climb ``raw_parent_map`` from ``node_id`` to nearest kept ancestor or ``root_id`` with cycle & orphan guards.

    Distinguishes deleted/filler intermediates (present in ``raw_parent_map``) from
    truncated-API orphans (ancestor ``curr != root_id`` absent from ``raw_parent_map``).
    When a truncated ancestor is encountered, adds ``node_id`` to ``truncated_orphans_out``
    and returns ``None`` if ``drop_truncated_orphans=True`` (otherwise falls back to ``root_id``).
    """
    seen: set[str] = {node_id}
    traversed_junk: list[str] = []
    curr = str(raw_parent_map.get(node_id) or "").strip()
    while curr:
        if curr == root_id or curr in seen:
            if traversed_junk_out is not None:
                traversed_junk_out.update(traversed_junk)
            if traversed_chain_out is not None:
                traversed_chain_out.extend(traversed_junk)
            return root_id
        if curr in kept_ids:
            probe = curr
            probe_seen: set[str] = set(seen)
            has_cycle = False
            hit_missing = False
            while probe and probe != root_id:
                if probe in probe_seen:
                    has_cycle = True
                    break
                if probe not in raw_parent_map:
                    hit_missing = True
                    break
                probe_seen.add(probe)
                probe = str(raw_parent_map.get(probe) or "").strip()
            if hit_missing:
                if drop_truncated_orphans:
                    if truncated_orphans_out is not None:
                        truncated_orphans_out.add(node_id)
                    return None
                if traversed_junk_out is not None:
                    traversed_junk_out.update(traversed_junk)
                if traversed_chain_out is not None:
                    traversed_chain_out.extend(traversed_junk)
                return root_id if has_cycle else curr
            if traversed_junk_out is not None:
                traversed_junk_out.update(traversed_junk)
            if traversed_chain_out is not None:
                traversed_chain_out.extend(traversed_junk)
            return root_id if has_cycle else curr
        if curr not in raw_parent_map:
            if truncated_orphans_out is not None:
                truncated_orphans_out.add(node_id)
            if traversed_junk_out is not None:
                traversed_junk_out.update(traversed_junk)
            if traversed_chain_out is not None:
                traversed_chain_out.extend(traversed_junk)
            return None if drop_truncated_orphans else root_id
        seen.add(curr)
        traversed_junk.append(curr)
        curr = str(raw_parent_map.get(curr) or "").strip()

    if traversed_junk_out is not None:
        traversed_junk_out.update(traversed_junk)
    if traversed_chain_out is not None:
        traversed_chain_out.extend(traversed_junk)
    return root_id


def compose_contracted_edge_stance(
    chain_stances: Sequence[float | None],
    child_stance: float,
) -> tuple[float, int, bool]:
    """Compose signed stance across contracted intermediates: ``(effective_stance, parity, inversion_risk)``.

    Neutral fillers (``|s| <= 0.1``) preserve parity (``+1``); opposing intermediates
    (``s < -0.1``) flip parity (``-1``); any opaque/unknown intermediate (``None``, e.g.
    ``[deleted]`` without prior stance) returns ``(float(child_stance), 0, True)``.
    """
    parity = 1
    for s in chain_stances:
        if s is None:
            return float(child_stance), 0, True
        s_val = float(s)
        if abs(s_val) <= 0.1:
            parity *= 1
        else:
            parity *= 1 if s_val > 0.0 else -1
    return float(parity) * float(child_stance), parity, False


def contains_vietnamese_chars(text: str) -> bool:
    """Return True if text contains Vietnamese diacritic vowels or `đ/Đ`."""
    return bool(_VN_DIACRITIC_RE.search(normalize_social_text_nfc(str(text or ""))))


def _ensure_prefixed_id(raw_id: Any, prefix: str) -> str:
    val = str(raw_id).strip()
    if not val:
        return ""
    if val.startswith(("t1_", "t3_")):
        return val
    return f"{prefix}{val}"


def _strip_id_prefix(raw_id: Any) -> str:
    val = str(raw_id).strip()
    if val.startswith(("t1_", "t3_")):
        return val[3:]
    return val


def _to_float_utc(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    return ts if ts >= 0.0 else None


def extract_comments_from_arctic_payload(
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten nested `/api/comments/tree` or flat `/api/comments/search` payloads.

    Applies `hash_author_sha256` and `normalize_social_text_nfc` immediately upon
    ingestion so raw author usernames are never stored in memory downstream.
    """
    extracted: list[dict[str, Any]] = []

    def _visit(node: Mapping[str, Any], inherited_parent: str | None = None) -> None:
        if "kind" in node and "data" in node:
            kind = str(node.get("kind") or "")
            if kind == "more":
                return
            data = node.get("data")
            if not isinstance(data, Mapping):
                return
            if kind == "t1":
                cid_raw = data.get("id") or data.get("name")
                cid = _strip_id_prefix(cid_raw) if cid_raw else ""
                if cid:
                    fullname = f"t1_{cid}"
                    parent_raw = data.get("parent_id") or inherited_parent
                    link_raw = data.get("link_id")
                    raw_author = data.get("author")
                    body_nfc = redact_pii_text(normalize_social_text_nfc(str(data.get("body") or "")))
                    created_ts = _to_float_utc(data.get("created_utc"))
                    extracted.append(
                        {
                            "id": cid,
                            "fullname": fullname,
                            "link_id": _ensure_prefixed_id(link_raw, "t3_") if link_raw else "",
                            "parent_id": str(parent_raw).strip() if parent_raw else "",
                            "author_hash": hash_author_sha256(
                                str(raw_author) if raw_author is not None else None
                            ),
                            "author_is_bot": isinstance(raw_author, str)
                            and raw_author.strip().lower() == "automoderator",
                            "text": body_nfc,
                            "created_utc": created_ts,
                        }
                    )
                    replies = data.get("replies")
                    if isinstance(replies, Mapping):
                        children = replies.get("data", {}).get("children", [])
                        if isinstance(children, Sequence):
                            for child in children:
                                if isinstance(child, Mapping):
                                    _visit(child, inherited_parent=fullname)
            elif isinstance(data.get("children"), Sequence):
                for child in data["children"]:
                    if isinstance(child, Mapping):
                        _visit(child, inherited_parent=inherited_parent)
            return

        # Flat comment dict from /api/comments/search
        cid_raw = node.get("id") or node.get("name")
        cid = _strip_id_prefix(cid_raw) if cid_raw else ""
        if not cid:
            return
        raw_author = node.get("author")
        extracted.append(
            {
                "id": cid,
                "fullname": f"t1_{cid}",
                "link_id": _ensure_prefixed_id(node.get("link_id"), "t3_")
                if node.get("link_id")
                else "",
                "parent_id": str(node.get("parent_id") or "").strip(),
                "author_hash": hash_author_sha256(
                    str(raw_author) if raw_author is not None else None
                ),
                "author_is_bot": isinstance(raw_author, str)
                and raw_author.strip().lower() == "automoderator",
                "text": redact_pii_text(normalize_social_text_nfc(str(node.get("body") or ""))),
                "created_utc": _to_float_utc(node.get("created_utc")),
            }
        )

    for item in items:
        if isinstance(item, Mapping):
            _visit(item)

    return extracted


_TOPIC_STOPWORDS: frozenset[str] = frozenset(
    {
        "các", "của", "cho", "với", "này", "kia", "đó", "đây", "khi", "như", "nên",
        "thì", "mà", "lại", "được", "không", "chưa", "phải", "cũng", "đang", "trong",
        "ngoài", "trên", "dưới", "cùng", "những", "một", "hai", "anh", "em", "bác",
        "bạn", "mọi", "người", "hỏi", "tâm", "sự", "xin", "nhờ", "review",
        "chia", "sẻ", "thảo", "luận", "nghĩ", "sao", "gì", "nào", "đâu", "voz",
        "vietnam", "reddit", "the", "and", "for", "with", "that", "this", "from",
        "are", "was", "were", "what", "how", "why", "who", "you", "your", "about",
        "have", "has", "just", "will", "not", "but", "can", "out",
    }
)


def _compute_topic_id(post: Mapping[str, Any], norm_title: str) -> str:
    is_self = bool(post.get("is_self", True))
    url = str(post.get("url") or "").strip()
    if not is_self and url and "reddit.com" not in url.lower():
        key = url.lower().split("#", 1)[0]
    else:
        raw_ts = _to_float_utc(post.get("created_utc"))
        week_tag = (
            datetime.fromtimestamp(raw_ts, tz=timezone.utc).strftime("%Y-W%V")
            if raw_ts is not None
            else "unknown-week"
        )
        cleaned = re.sub(r"\s+", " ", norm_title.lower()).strip()
        tokens = [
            tok
            for tok in re.findall(r"[^\W\d_]+", cleaned, flags=re.UNICODE)
            if len(tok) >= 3 and tok not in _TOPIC_STOPWORDS
        ]
        unique_sorted = sorted(set(tokens))[:5]
        if len(unique_sorted) >= 2:
            key = f"{week_tag}|{' '.join(unique_sorted)}"
        else:
            key = f"{week_tag}|{cleaned}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return f"vn_topic_{digest}"


def cluster_vn_cascade_topics(
    node_rows: list[dict[str, Any]],
    *,
    time_window_days: float = 30.0,
    sim_threshold: float = 0.30,
) -> list[dict[str, Any]]:
    """Merge semantically related Vietnamese cascades within a sliding time window into shared topics.

    Uses TF-IDF (1-2 gram) cosine similarity across opening root text (`[:300]`) combined
    with a temporal window (`time_window_days`) to assign a single connected-component
    `topic_id` across recurring events, reducing cross-split topical overlap.
    """
    roots = [r for r in node_rows if bool(r.get("is_root"))]
    if len(roots) < 2:
        return node_rows

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [normalize_nfc(str(r.get("text") or ""))[:300].lower() for r in roots]
    if not any(texts):
        return node_rows

    eff_min_df = 2 if len(roots) >= 5 else 1
    eff_max_df = 0.85 if len(roots) >= 5 else 1.0
    try:
        stop_words = sorted({tok for tok in _TOPIC_STOPWORDS if tok})
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=eff_min_df,
            max_df=eff_max_df,
            stop_words=stop_words,
        )
        mat = vec.fit_transform(texts)
    except ValueError:
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
            mat = vec.fit_transform(texts)
        except ValueError:
            return node_rows

    sim_mat = cosine_similarity(mat)
    timestamps: list[float] = []
    for r in roots:
        ts_val = r.get("created_at_utc")
        if isinstance(ts_val, datetime):
            timestamps.append(ts_val.timestamp())
        else:
            timestamps.append(0.0)

    max_dt_sec = float(time_window_days) * 86400.0
    n = len(roots)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Group first by identical existing topic_id, then merge high-similarity pairs in window
    existing_groups: dict[str, int] = {}
    for idx, r in enumerate(roots):
        tid = str(r.get("topic_id") or "")
        if tid in existing_groups:
            _union(existing_groups[tid], idx)
        else:
            existing_groups[tid] = idx

    for i in range(n):
        for j in range(i + 1, n):
            if abs(timestamps[i] - timestamps[j]) <= max_dt_sec and sim_mat[i, j] >= sim_threshold:
                _union(i, j)

    comp_topics: dict[int, str] = {}
    for idx, r in enumerate(roots):
        rep = _find(idx)
        if rep not in comp_topics:
            comp_topics[rep] = str(r.get("topic_id") or f"vn_topic_{rep:06d}")

    cascade_to_topic: dict[str, str] = {}
    for idx, r in enumerate(roots):
        cid = str(r["cascade_id"])
        cascade_to_topic[cid] = comp_topics[_find(idx)]

    for row in node_rows:
        cid = str(row["cascade_id"])
        if cid in cascade_to_topic:
            row["topic_id"] = cascade_to_topic[cid]

    return node_rows


def build_vn_cascade_from_thread(
    post: Mapping[str, Any],
    comments: Sequence[Mapping[str, Any]],
    *,
    min_comments: int = 25,
    min_depth: int = 3,
    min_vi_ratio: float = 0.65,
    junk_mode: Literal["prune", "contract"] = "prune",
    orphan_mode: Literal["prune", "flag_root"] = "prune",
    preserve_opaque_deleted_bridges: bool = False,
    prior_edge_stances: Mapping[str, float] | None = None,
    stats_out: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None:
    """Validate and convert one Reddit thread into SBi-FAGAT node/edge record lists."""
    if junk_mode not in {"prune", "contract"}:
        raise ValueError(f"Unsupported junk_mode {junk_mode!r}; expected 'prune' or 'contract'")
    if orphan_mode not in {"prune", "flag_root"}:
        raise ValueError(f"Unsupported orphan_mode {orphan_mode!r}; expected 'prune' or 'flag_root'")

    post_id = _strip_id_prefix(post.get("id") or post.get("name") or "")
    if not post_id:
        return None
    root_fullname = f"t3_{post_id}"

    root_raw_ts = _to_float_utc(post.get("created_utc"))
    if root_raw_ts is None:
        return None

    title_nfc = normalize_social_text_nfc(str(post.get("title") or ""))
    selftext_raw = str(post.get("selftext") or "")
    selftext_nfc = normalize_social_text_nfc(selftext_raw)
    if selftext_nfc.lower() in {"[deleted]", "[removed]"}:
        selftext_nfc = ""

    root_text = f"{title_nfc}\n\n{selftext_nfc}".strip() if selftext_nfc else title_nfc
    raw_post_author = post.get("author")
    if is_invalid_or_spam_text(
        root_text,
        author_raw=str(raw_post_author) if raw_post_author is not None else None,
    ):
        return None
    root_text = redact_pii_text(root_text)

    root_author_hash = hash_author_sha256(
        str(raw_post_author) if raw_post_author is not None else None
    )
    subreddit = str(post.get("subreddit") or "unknown").strip()
    topic_id = _compute_topic_id(post, title_nfc or root_text)

    # Pass 1: Map every raw comment's node_id -> raw parent_id before filtering junk.
    raw_comments_map: dict[str, Mapping[str, Any]] = {}
    raw_parent_map: dict[str, str] = {}
    for c in comments:
        fullname = str(c.get("fullname") or "")
        if not fullname.startswith("t1_"):
            continue
        link_id = str(c.get("link_id") or "")
        if link_id and link_id != root_fullname:
            continue
        parent_id = str(c.get("parent_id") or "").strip()
        if not parent_id.startswith(("t3_", "t1_")) or parent_id == fullname:
            continue
        if fullname in raw_comments_map:
            continue
        raw_comments_map[fullname] = c
        raw_parent_map[fullname] = parent_id

    valid_comments: dict[str, dict[str, Any]] = {}
    junk_ids: set[str] = set()
    junk_reasons: dict[str, str] = {}
    raw_prior_stances: dict[str, float] = {}
    for fullname, c in raw_comments_map.items():
        if prior_edge_stances is not None and fullname in prior_edge_stances:
            raw_prior_stances[fullname] = float(prior_edge_stances[fullname])
        elif c.get("prior_s_uv") is not None:
            raw_prior_stances[fullname] = float(c["prior_s_uv"])

        raw_author = c.get("author_raw") if c.get("author_raw") is not None else c.get("author")
        author_str = str(raw_author) if raw_author is not None else None
        if bool(c.get("author_is_bot")) or (
            isinstance(author_str, str) and author_str.strip().lower() == "automoderator"
        ):
            junk_ids.add(fullname)
            junk_reasons[fullname] = "bot"
            continue

        c_text = normalize_social_text_nfc(str(c.get("text") or c.get("body") or ""))
        low_c = c_text.strip().lower()
        if not low_c or low_c in {"[deleted]", "[removed]"}:
            junk_ids.add(fullname)
            junk_reasons[fullname] = "deleted_or_removed"
            continue
        if is_invalid_or_spam_text(c_text, author_raw=author_str):
            junk_ids.add(fullname)
            junk_reasons[fullname] = "spam"
            continue
        if junk_mode == "contract" and is_filler_junk_text(c_text, author_raw=author_str):
            junk_ids.add(fullname)
            junk_reasons[fullname] = "filler"
            continue

        c_ts = _to_float_utc(c.get("created_utc"))
        if c_ts is None:
            junk_ids.add(fullname)
            junk_reasons[fullname] = "invalid_ts"
            continue
        valid_comments[fullname] = {
            "fullname": fullname,
            "parent_id": raw_parent_map[fullname],
            "text": redact_pii_text(c_text),
            "created_utc": c_ts,
            "author_hash": c.get("author_hash"),
        }

    if preserve_opaque_deleted_bridges and junk_mode == "contract":
        bridge_ancestors: set[str] = set()
        for fn in valid_comments:
            seen_b: set[str] = {fn}
            curr_b = raw_parent_map.get(fn, "")
            while curr_b and curr_b != root_fullname and curr_b not in seen_b:
                seen_b.add(curr_b)
                if junk_reasons.get(curr_b) == "deleted_or_removed":
                    bridge_ancestors.add(curr_b)
                curr_b = raw_parent_map.get(curr_b, "")
        for b_fn in sorted(bridge_ancestors):
            b_raw = raw_comments_map[b_fn]
            b_ts = _to_float_utc(b_raw.get("created_utc"))
            valid_comments[b_fn] = {
                "fullname": b_fn,
                "parent_id": raw_parent_map[b_fn],
                "text": "[deleted_bridge]",
                "created_utc": b_ts if b_ts is not None else root_raw_ts,
                "author_hash": None,
            }
            junk_ids.discard(b_fn)
            junk_reasons.pop(b_fn, None)

    # Detect truncated-API orphans (ancestor chain hits a missing parent != root_fullname)
    truncated_orphan_ids: set[str] = set()
    for fn in list(valid_comments.keys()):
        seen_o: set[str] = {fn}
        curr_o = str(raw_parent_map.get(fn) or "").strip()
        while curr_o:
            if curr_o == root_fullname or curr_o in seen_o:
                break
            if curr_o not in raw_parent_map:
                truncated_orphan_ids.add(fn)
                break
            seen_o.add(curr_o)
            curr_o = str(raw_parent_map.get(curr_o) or "").strip()

    # Pass 2: Build adjacency (pruning subtrees in "prune" mode; contracting junk in "contract" mode).
    children_adj: dict[str, list[str]] = {root_fullname: []}
    contracted_junk_ids: set[str] = set()
    relinked_chains: dict[str, list[str]] = {}
    contract_orphans: set[str] = set()
    if junk_mode == "contract":
        if orphan_mode == "prune" and truncated_orphan_ids:
            for fn in truncated_orphan_ids:
                valid_comments.pop(fn, None)
        kept_ids_set = set(valid_comments.keys())
        for fn in list(valid_comments.keys()):
            c_obj = valid_comments[fn]
            chain: list[str] = []
            resolved_parent = resolve_contracted_parent_id(
                fn,
                raw_parent_map,
                kept_ids_set,
                root_fullname,
                traversed_junk_out=contracted_junk_ids,
                traversed_chain_out=chain,
                truncated_orphans_out=contract_orphans,
                drop_truncated_orphans=(orphan_mode == "prune"),
            )
            if resolved_parent is None:
                valid_comments.pop(fn, None)
                continue
            c_obj["parent_id"] = resolved_parent
            if chain:
                relinked_chains[fn] = chain
            children_adj.setdefault(resolved_parent, []).append(fn)
        truncated_orphan_ids.update(contract_orphans)
    else:
        for fn, c_obj in valid_comments.items():
            if orphan_mode == "prune" and fn in truncated_orphan_ids:
                continue
            if (
                orphan_mode == "flag_root"
                and c_obj["parent_id"] != root_fullname
                and c_obj["parent_id"] not in raw_parent_map
            ):
                c_obj["parent_id"] = root_fullname
                contract_orphans.add(fn)
            children_adj.setdefault(c_obj["parent_id"], []).append(fn)

    # Sort children chronologically for deterministic BFS traversal
    for parent_key, c_list in children_adj.items():
        c_list.sort(
            key=lambda fn: (valid_comments[fn]["created_utc"], fn)
            if fn in valid_comments
            else (0.0, fn)
        )

    # BFS from root_fullname computes topological depths, monotonic UTC timestamps, and edges
    root_dt = datetime.fromtimestamp(root_raw_ts, tz=timezone.utc)
    cascade_id = f"vn_{post_id}"
    source_dataset = f"arctic_shift_{subreddit.lower()}"

    visited: set[str] = {root_fullname}
    depths: dict[str, int] = {root_fullname: 0}
    effective_dt: dict[str, datetime] = {root_fullname: root_dt}

    node_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    relinked_edge_ids: list[list[str]] = []
    relinked_edge_metadata: list[dict[str, Any]] = []
    queue: deque[str] = deque([root_fullname])
    vi_node_count = 0

    while queue:
        curr_fn = queue.popleft()
        is_root = curr_fn == root_fullname

        if is_root:
            curr_text = root_text
            curr_author_hash = root_author_hash
            parent_fn: str | None = None
        else:
            c_obj = valid_comments[curr_fn]
            curr_text = c_obj["text"]
            curr_author_hash = c_obj["author_hash"]
            parent_fn = c_obj["parent_id"]

        if contains_vietnamese_chars(curr_text):
            vi_node_count += 1

        node_id = f"{cascade_id}_{curr_fn}"
        parent_node_id = None if is_root else f"{cascade_id}_{parent_fn}"
        node_records.append(
            {
                "node_id": node_id,
                "cascade_id": cascade_id,
                "topic_id": topic_id,
                "parent_id": parent_node_id,
                "is_root": is_root,
                "text": curr_text,
                "created_at_utc": effective_dt[curr_fn],
                "lang": "vi",
                "source_dataset": source_dataset,
                "user_id_hash": curr_author_hash,
                "rumor_label": None,
            }
        )

        if not is_root and parent_fn is not None and parent_node_id is not None:
            delay_sec = max(
                0.0,
                (effective_dt[curr_fn] - effective_dt[parent_fn]).total_seconds(),
            )
            edge_records.append(
                {
                    "edge_id": f"{node_id}<-{parent_fn}",
                    "cascade_id": cascade_id,
                    "source_id": parent_node_id,
                    "target_id": node_id,
                    "created_at_utc": effective_dt[curr_fn],
                    "time_delay_sec": float(delay_sec),
                    "edge_type": "reply",
                }
            )
            if junk_mode == "contract" and parent_fn != raw_parent_map.get(curr_fn):
                relinked_edge_ids.append([parent_node_id, node_id])
                chain_fns = relinked_chains.get(curr_fn, [])
                chain_nids = [f"{cascade_id}_{x}" for x in chain_fns]
                reasons = [junk_reasons.get(x, "unknown") for x in chain_fns]
                if curr_fn in contract_orphans:
                    parity, risk = 0, True
                else:
                    chain_stances: list[float | None] = []
                    for x in chain_fns:
                        if x in raw_prior_stances:
                            chain_stances.append(raw_prior_stances[x])
                        elif junk_reasons.get(x) in {"filler", "bot"}:
                            chain_stances.append(0.0)
                        else:
                            chain_stances.append(None)
                    _, parity, risk = compose_contracted_edge_stance(chain_stances, 1.0)
                relinked_edge_metadata.append(
                    {
                        "source_id": parent_node_id,
                        "target_id": node_id,
                        "contracted_chain": chain_nids,
                        "chain_reasons": reasons,
                        "contracted_sign_parity": int(parity),
                        "stance_inversion_risk": bool(risk),
                    }
                )

        for child_fn in children_adj.get(curr_fn, []):
            if child_fn in visited or child_fn not in valid_comments:
                continue
            visited.add(child_fn)
            depths[child_fn] = depths[curr_fn] + 1
            child_raw_dt = datetime.fromtimestamp(
                valid_comments[child_fn]["created_utc"],
                tz=timezone.utc,
            )
            effective_dt[child_fn] = max(child_raw_dt, effective_dt[curr_fn])
            queue.append(child_fn)

    valid_reply_count = len(node_records) - 1
    max_depth = max(depths.values()) if depths else 0
    vi_ratio = vi_node_count / max(1, len(node_records))

    if junk_mode == "contract":
        contracted_nodes = len(contracted_junk_ids)
        dropped_set = junk_ids - contracted_junk_ids
        if orphan_mode == "prune":
            dropped_set = dropped_set | truncated_orphan_ids
        dropped_leaves = len(dropped_set)
    else:
        contracted_nodes = 0
        dropped_leaves = len(raw_comments_map) - valid_reply_count

    junk_filter_stats: dict[str, Any] = {
        "mode": junk_mode,
        "raw_comments": len(raw_comments_map),
        "kept_comments": valid_reply_count,
        "dropped_leaves": dropped_leaves,
        "contracted_nodes": contracted_nodes,
        "relinked_edges": len(relinked_edge_ids),
        "relinked_edge_ids": relinked_edge_ids,
        "truncated_orphans": len(truncated_orphan_ids),
        "truncated_orphan_node_ids": sorted(
            f"{cascade_id}_{fn}" for fn in truncated_orphan_ids
        ),
        "stance_inversion_risk_edges": sum(
            1 for m in relinked_edge_metadata if m["stance_inversion_risk"]
        ),
        "relinked_edge_metadata": relinked_edge_metadata,
    }
    if stats_out is not None:
        stats_out.clear()
        stats_out.update(junk_filter_stats)

    if valid_reply_count < min_comments:
        return None
    if max_depth < min_depth:
        return None
    if vi_ratio < min_vi_ratio:
        return None

    stats = {
        "cascade_id": cascade_id,
        "subreddit": subreddit,
        "num_replies": valid_reply_count,
        "max_depth": max_depth,
        "vi_ratio": vi_ratio,
        "junk_filter_stats": junk_filter_stats,
    }
    return node_records, edge_records, stats


def _request_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: Mapping[str, Any],
    timeout: int = 30,
    max_retries: int = 4,
) -> Mapping[str, Any]:
    backoff = 1.5
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait_s = float(resp.headers.get("x-ratelimit-reset", backoff))
                time.sleep(min(max(1.0, wait_s), 30.0))
                backoff *= 2.0
                continue
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, Mapping) and payload.get("error"):
                raise RuntimeError(f"Arctic Shift API error on {url}: {payload['error']}")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2.0
    raise RuntimeError(f"Failed request to {url} after {max_retries} attempts: {last_exc}")


def fetch_comment_tree(
    session: requests.Session,
    post_id: str,
    *,
    api_base: str = ARCTIC_SHIFT_API_BASE,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Fetch complete nested comment tree for `t3_<post_id>` via `/api/comments/tree`."""
    payload = _request_with_retry(
        session,
        f"{api_base.rstrip('/')}/comments/tree",
        params={"link_id": f"t3_{_strip_id_prefix(post_id)}", "limit": 9999},
        timeout=timeout,
    )
    items = payload.get("data") or []
    return extract_comments_from_arctic_payload(items)


def _deserialize_checkpoint_record(record: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    post_id = str(record["post_id"])
    nodes: list[dict[str, Any]] = []
    for row in record.get("nodes", []):
        copied = dict(row)
        ts = copied.get("created_at_utc")
        if isinstance(ts, str):
            copied["created_at_utc"] = datetime.fromisoformat(ts)
        nodes.append(copied)
    edges: list[dict[str, Any]] = []
    for row in record.get("edges", []):
        copied_e = dict(row)
        ts_e = copied_e.get("created_at_utc")
        if isinstance(ts_e, str):
            copied_e["created_at_utc"] = datetime.fromisoformat(ts_e)
        edges.append(copied_e)
    stats = dict(record.get("stats", {}))
    return post_id, nodes, edges, stats


def _save_crawl_state(
    state_path: Path,
    *,
    cursors: Mapping[str, str | int | None],
    exhausted: Mapping[str, bool],
    pages_fetched: Mapping[str, int],
    rejected_post_ids: set[str],
) -> None:
    payload = {
        "cursors": dict(cursors),
        "exhausted": {k: bool(v) for k, v in exhausted.items()},
        "pages_fetched": {k: int(v) for k, v in pages_fetched.items()},
        "rejected_post_ids": sorted(rejected_post_ids),
    }
    tmp = state_path.with_name(f"{state_path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(state_path)


def crawl_arctic_shift_discussions(
    output_dir: Path = Path("data/interim/vn"),
    *,
    subreddits: Sequence[str] = TARGET_SUBREDDITS,
    target_count: int = 1500,
    min_comments: int = 25,
    min_depth: int = 3,
    min_vi_ratio: float = 0.65,
    initial_before: str | int = "7d",
    max_pages_per_sub: int = 300,
    request_delay: float = 0.15,
    timeout: int = 30,
    checkpoint_jsonl: Path | None = None,
    allow_partial: bool = False,
) -> tuple[Path, Path]:
    """Crawl public Vietnamese discussions from Arctic Shift and write Parquet tables."""
    if not subreddits:
        raise ValueError("At least one subreddit must be provided")

    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

    cursors: dict[str, str | int | None] = {sub: initial_before for sub in subreddits}
    # Permanently exhausted subreddits (API genuinely returned empty/no older posts).
    exhausted: dict[str, bool] = {sub: False for sub in subreddits}
    # In-memory transient halt flags (network error or max_pages_per_sub cap) — never persisted.
    session_stopped: dict[str, bool] = {sub: False for sub in subreddits}
    had_network_error = False
    pages_fetched: dict[str, int] = {sub: 0 for sub in subreddits}
    seen_post_ids: set[str] = set()
    rejected_post_ids: set[str] = set()
    state_path: Path | None = None

    all_nodes: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    collected_stats: list[dict[str, Any]] = []

    if checkpoint_jsonl is not None:
        checkpoint_jsonl = Path(checkpoint_jsonl)
        checkpoint_jsonl.parent.mkdir(parents=True, exist_ok=True)
        state_path = checkpoint_jsonl.with_name(f"{checkpoint_jsonl.name}.state.json")
        if checkpoint_jsonl.exists():
            with checkpoint_jsonl.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        pid, nodes, edges, stats = _deserialize_checkpoint_record(
                            json.loads(stripped)
                        )
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
                    if pid and pid not in seen_post_ids:
                        seen_post_ids.add(pid)
                        all_nodes.extend(nodes)
                        all_edges.extend(edges)
                        collected_stats.append(stats)
        if state_path.exists():
            try:
                saved_state = json.loads(state_path.read_text(encoding="utf-8"))
                saved_cursors = saved_state.get("cursors", {})
                saved_exhausted = saved_state.get("exhausted", {})
                saved_pages = saved_state.get("pages_fetched", {})
                for sub in subreddits:
                    if sub in saved_cursors:
                        cursors[sub] = saved_cursors[sub]
                    if sub in saved_exhausted:
                        exhausted[sub] = bool(saved_exhausted[sub])
                    if sub in saved_pages:
                        pages_fetched[sub] = int(saved_pages[sub])
                for rpid in saved_state.get("rejected_post_ids", []):
                    rpid_s = str(rpid).strip()
                    if rpid_s:
                        rejected_post_ids.add(rpid_s)
                        seen_post_ids.add(rpid_s)
            except (OSError, json.JSONDecodeError, ValueError):
                pass

    try:
        while len(collected_stats) < target_count and not all(
            exhausted[s] or session_stopped[s] for s in subreddits
        ):
            progressed = False
            for sub in subreddits:
                if len(collected_stats) >= target_count:
                    break
                if exhausted[sub] or session_stopped[sub]:
                    continue
                if pages_fetched[sub] >= max_pages_per_sub:
                    session_stopped[sub] = True
                    continue

                params: dict[str, Any] = {"subreddit": sub, "limit": 100}
                if cursors[sub] is not None:
                    params["before"] = cursors[sub]

                try:
                    payload = _request_with_retry(
                        session,
                        f"{ARCTIC_SHIFT_API_BASE}/posts/search",
                        params=params,
                        timeout=timeout,
                    )
                except RuntimeError as exc:
                    print(
                        f"[Warning] Transient search error on r/{sub} at cursor={cursors[sub]}: {exc}",
                        file=sys.stderr,
                    )
                    had_network_error = True
                    session_stopped[sub] = True
                    if state_path is not None:
                        _save_crawl_state(
                            state_path,
                            cursors=cursors,
                            exhausted=exhausted,
                            pages_fetched=pages_fetched,
                            rejected_post_ids=rejected_post_ids,
                        )
                    continue
                posts = payload.get("data") or []
                pages_fetched[sub] += 1
                if not posts:
                    exhausted[sub] = True
                    continue

                progressed = True
                timestamps = [
                    int(p["created_utc"]) for p in posts if p.get("created_utc") is not None
                ]
                if not timestamps:
                    exhausted[sub] = True
                    continue
                cursors[sub] = min(timestamps) - 1

                for post in posts:
                    if len(collected_stats) >= target_count:
                        break
                    pid = _strip_id_prefix(post.get("id") or "")
                    if not pid or pid in seen_post_ids:
                        continue
                    if int(post.get("num_comments") or 0) < min_comments:
                        rejected_post_ids.add(pid)
                        seen_post_ids.add(pid)
                        continue
                    seen_post_ids.add(pid)

                    time.sleep(request_delay)
                    try:
                        comments = fetch_comment_tree(session, pid, timeout=timeout)
                    except Exception as exc:
                        print(
                            f"[Warning] Skipping post {pid} after comment-tree fetch error: {exc}",
                            file=sys.stderr,
                        )
                        rejected_post_ids.add(pid)
                        continue
                    built = build_vn_cascade_from_thread(
                        post,
                        comments,
                        min_comments=min_comments,
                        min_depth=min_depth,
                        min_vi_ratio=min_vi_ratio,
                    )
                    if built is None:
                        rejected_post_ids.add(pid)
                        continue
                    nodes, edges, stats = built
                    all_nodes.extend(nodes)
                    all_edges.extend(edges)
                    collected_stats.append(stats)
                    if checkpoint_jsonl is not None:
                        serial_nodes = [
                            {
                                **n,
                                "created_at_utc": (
                                    n["created_at_utc"].isoformat()
                                    if isinstance(n.get("created_at_utc"), datetime)
                                    else n.get("created_at_utc")
                                ),
                            }
                            for n in nodes
                        ]
                        serial_edges = [
                            {
                                **e,
                                "created_at_utc": (
                                    e["created_at_utc"].isoformat()
                                    if isinstance(e.get("created_at_utc"), datetime)
                                    else e.get("created_at_utc")
                                ),
                            }
                            for e in edges
                        ]
                        with checkpoint_jsonl.open("a", encoding="utf-8") as fh:
                            fh.write(
                                json.dumps(
                                    {
                                        "post_id": pid,
                                        "nodes": serial_nodes,
                                        "edges": serial_edges,
                                        "stats": stats,
                                    },
                                    ensure_ascii=False,
                                    default=str,
                                )
                                + "\n"
                            )

                if state_path is not None:
                    _save_crawl_state(
                        state_path,
                        cursors=cursors,
                        exhausted=exhausted,
                        pages_fetched=pages_fetched,
                        rejected_post_ids=rejected_post_ids,
                    )

            if not progressed:
                break
    finally:
        session.close()

    min_usable_partial = min(30, max(1, int(target_count)))
    if len(collected_stats) < target_count:
        if had_network_error and len(collected_stats) < min_usable_partial:
            raise RuntimeError(
                f"Crawl halted by transient network errors with only "
                f"{len(collected_stats)}/{target_count} cascades (below safe partial floor "
                f"{min_usable_partial}); refusing to overwrite Parquet so resume can retry."
            )
        if allow_partial and len(collected_stats) > 0:
            print(
                f"[Warning] Only collected {len(collected_stats)}/{target_count} valid cascades "
                f"across {list(subreddits)}; saving partial dataset because allow_partial=True.",
                file=sys.stderr,
            )
        else:
            raise RuntimeError(
                f"Only collected {len(collected_stats)}/{target_count} valid cascades "
                f"across {list(subreddits)}."
            )

    all_nodes = cluster_vn_cascade_topics(all_nodes)
    nodes_table = build_nodes_table(all_nodes)
    edges_table = build_edges_table(all_edges)
    return save_cascade_parquet(nodes_table, edges_table, output_dir=output_dir, validate=True)


def compare_prune_vs_contract_on_threads(
    threads: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]] | Mapping[str, Any]],
    *,
    min_comments: int = 25,
    min_depth: int = 3,
    min_vi_ratio: float = 0.65,
    k_horizon: int = 20,
) -> dict[str, Any]:
    """Compare ``junk_mode='prune'`` vs ``junk_mode='contract'`` across raw Reddit threads."""
    kept_in_prune = 0
    kept_in_contract = 0
    rescued_by_contract = 0
    replies_prune: list[int] = []
    replies_contract: list[int] = []
    depths_prune: list[int] = []
    depths_contract: list[int] = []

    total_contracted_nodes = 0
    total_relinked_edges = 0
    total_truncated_orphans = 0
    total_stance_inversion_risk_edges = 0

    jaccards: list[float] = []
    shared_nodes_k = 0
    mismatched_parents_k = 0

    for item in threads:
        if isinstance(item, Mapping):
            post = item["post"]
            comments = item.get("comments", ())
        else:
            post, comments = item

        p_stats_out: dict[str, Any] = {}
        c_stats_out: dict[str, Any] = {}
        res_p = build_vn_cascade_from_thread(
            post,
            comments,
            min_comments=min_comments,
            min_depth=min_depth,
            min_vi_ratio=min_vi_ratio,
            junk_mode="prune",
            stats_out=p_stats_out,
        )
        res_c = build_vn_cascade_from_thread(
            post,
            comments,
            min_comments=min_comments,
            min_depth=min_depth,
            min_vi_ratio=min_vi_ratio,
            junk_mode="contract",
            stats_out=c_stats_out,
        )

        total_contracted_nodes += int(c_stats_out.get("contracted_nodes", 0))
        total_relinked_edges += int(c_stats_out.get("relinked_edges", 0))
        total_truncated_orphans += int(c_stats_out.get("truncated_orphans", 0))
        total_stance_inversion_risk_edges += int(
            c_stats_out.get("stance_inversion_risk_edges", 0)
        )

        if res_p is not None:
            kept_in_prune += 1
            replies_prune.append(int(res_p[2]["num_replies"]))
            depths_prune.append(int(res_p[2]["max_depth"]))
        if res_c is not None:
            kept_in_contract += 1
            replies_contract.append(int(res_c[2]["num_replies"]))
            depths_contract.append(int(res_c[2]["max_depth"]))
            if res_p is None:
                rescued_by_contract += 1

        if res_p is not None and res_c is not None:
            p_nodes = sorted(
                (r for r in res_p[0] if not bool(r.get("is_root"))),
                key=lambda r: (r["created_at_utc"], str(r["node_id"])),
            )[:k_horizon]
            c_nodes = sorted(
                (r for r in res_c[0] if not bool(r.get("is_root"))),
                key=lambda r: (r["created_at_utc"], str(r["node_id"])),
            )[:k_horizon]
            p_map = {str(r["node_id"]): r.get("parent_id") for r in p_nodes}
            c_map = {str(r["node_id"]): r.get("parent_id") for r in c_nodes}
            p_set, c_set = set(p_map.keys()), set(c_map.keys())
            union = p_set | c_set
            inter = p_set & c_set
            if union:
                jaccards.append(float(len(inter)) / float(len(union)))
            for nid in inter:
                shared_nodes_k += 1
                if p_map[nid] != c_map[nid]:
                    mismatched_parents_k += 1

    return {
        "threads_evaluated": len(threads),
        "kept_in_prune": kept_in_prune,
        "kept_in_contract": kept_in_contract,
        "rescued_by_contract": rescued_by_contract,
        "mean_replies_prune": (
            float(sum(replies_prune)) / float(len(replies_prune)) if replies_prune else 0.0
        ),
        "mean_replies_contract": (
            float(sum(replies_contract)) / float(len(replies_contract))
            if replies_contract
            else 0.0
        ),
        "mean_depth_prune": (
            float(sum(depths_prune)) / float(len(depths_prune)) if depths_prune else 0.0
        ),
        "mean_depth_contract": (
            float(sum(depths_contract)) / float(len(depths_contract)) if depths_contract else 0.0
        ),
        "total_contracted_nodes": total_contracted_nodes,
        "total_relinked_edges": total_relinked_edges,
        "total_truncated_orphans": total_truncated_orphans,
        "total_stance_inversion_risk_edges": total_stance_inversion_risk_edges,
        "opaque_deleted_relink_rate": (
            float(total_stance_inversion_risk_edges) / float(total_relinked_edges)
            if total_relinked_edges > 0
            else 0.0
        ),
        "early_k_jaccard_node_overlap": (
            float(sum(jaccards)) / float(len(jaccards)) if jaccards else 0.0
        ),
        "early_k_parent_mismatch_rate": (
            float(mismatched_parents_k) / float(shared_nodes_k) if shared_nodes_k > 0 else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl Vietnamese discussion trees from Arctic Shift public REST API."
    )
    parser.add_argument("--subreddits", nargs="+", default=list(TARGET_SUBREDDITS))
    parser.add_argument("--target-count", type=int, default=1500)
    parser.add_argument("--min-comments", type=int, default=25)
    parser.add_argument("--min-depth", type=int, default=3)
    parser.add_argument("--min-vi-ratio", type=float, default=0.65)
    parser.add_argument("--before", type=str, default="7d")
    parser.add_argument(
        "--before-epoch",
        type=int,
        default=None,
        help="Optional explicit UTC Unix epoch cursor for reproducible snapshots (overrides --before).",
    )
    parser.add_argument("--max-pages-per-sub", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/vn"))
    parser.add_argument("--checkpoint-jsonl", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    before_cursor: str | int = args.before_epoch if args.before_epoch is not None else args.before
    nodes_path, edges_path = crawl_arctic_shift_discussions(
        output_dir=args.output_dir,
        subreddits=args.subreddits,
        target_count=args.target_count,
        min_comments=args.min_comments,
        min_depth=args.min_depth,
        min_vi_ratio=args.min_vi_ratio,
        initial_before=before_cursor,
        max_pages_per_sub=args.max_pages_per_sub,
        checkpoint_jsonl=args.checkpoint_jsonl,
        allow_partial=args.allow_partial,
    )
    n_nodes = pq.read_metadata(nodes_path).num_rows
    n_edges = pq.read_metadata(edges_path).num_rows
    print(
        f"Saved validated Vietnamese discussion trees ({n_nodes} nodes, {n_edges} edges -> "
        f"target {args.target_count}): {nodes_path}, {edges_path}"
    )


if __name__ == "__main__":
    main()

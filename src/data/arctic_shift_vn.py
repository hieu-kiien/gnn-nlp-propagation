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
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import requests

from src.data.schema import (
    build_edges_table,
    build_nodes_table,
    save_cascade_parquet,
)

ARCTIC_SHIFT_API_BASE = "https://arctic-shift.photon-reddit.com/api"
TARGET_SUBREDDITS: tuple[str, ...] = ("vozforums", "VietNam")
DEFAULT_USER_AGENT = "SBi-FAGAT-Research/1.0 (Academic Cascade Analysis)"

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_SPAM_DOMAIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:shope\.ee|s\.shopee\.vn|bit\.ly|tinyurl\.com|t\.me)/\S*",
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
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text).replace("\x00", "").strip()


def is_invalid_or_spam_text(text: str, author_raw: str | None = None) -> bool:
    """Return True for empty, deleted/removed, AutoModerator, or link-spam content."""
    if isinstance(author_raw, str) and author_raw.strip().lower() == "automoderator":
        return True

    norm = normalize_social_text_nfc(text)
    if not norm or norm.lower() in {"[deleted]", "[removed]"}:
        return True

    if _SPAM_DOMAIN_RE.search(norm):
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


def contains_vietnamese_chars(text: str) -> bool:
    """Return True if text contains Vietnamese diacritic vowels or `đ/Đ`."""
    return bool(_VN_DIACRITIC_RE.search(text))


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
                    body_nfc = normalize_social_text_nfc(str(data.get("body") or ""))
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
                "text": normalize_social_text_nfc(str(node.get("body") or "")),
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
        "các bạn", "mọi", "người", "hỏi", "tâm", "sự", "xin", "nhờ", "review",
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


def build_vn_cascade_from_thread(
    post: Mapping[str, Any],
    comments: Sequence[Mapping[str, Any]],
    *,
    min_comments: int = 40,
    min_depth: int = 3,
    min_vi_ratio: float = 0.25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None:
    """Validate and convert one Reddit thread into SBi-FAGAT node/edge record lists."""
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

    root_author_hash = hash_author_sha256(
        str(raw_post_author) if raw_post_author is not None else None
    )
    subreddit = str(post.get("subreddit") or "unknown").strip()
    topic_id = _compute_topic_id(post, title_nfc or root_text)

    # Filter candidate comments (pre-cleaned text, valid link_id & parent_id, non-spam)
    valid_comments: dict[str, dict[str, Any]] = {}
    children_adj: dict[str, list[str]] = {root_fullname: []}

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
        if bool(c.get("author_is_bot")):
            continue
        c_text = normalize_social_text_nfc(str(c.get("text") or ""))
        if is_invalid_or_spam_text(c_text):
            continue
        c_ts = _to_float_utc(c.get("created_utc"))
        if c_ts is None:
            continue

        if fullname not in valid_comments:
            valid_comments[fullname] = {
                "fullname": fullname,
                "parent_id": parent_id,
                "text": c_text,
                "created_utc": c_ts,
                "author_hash": c.get("author_hash"),
            }
            children_adj.setdefault(parent_id, []).append(fullname)

    # Sort children chronologically for deterministic BFS traversal
    for parent_key, c_list in children_adj.items():
        c_list.sort(
            key=lambda fn: (valid_comments[fn]["created_utc"], fn)
            if fn in valid_comments
            else (0.0, fn)
        )

    # BFS from root_fullname automatically prunes subtrees beneath deleted/spam parents
    root_dt = datetime.fromtimestamp(root_raw_ts, tz=timezone.utc)
    cascade_id = f"vn_{post_id}"
    source_dataset = f"arctic_shift_{subreddit.lower()}"

    visited: set[str] = {root_fullname}
    depths: dict[str, int] = {root_fullname: 0}
    effective_dt: dict[str, datetime] = {root_fullname: root_dt}

    node_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
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

        if not is_root and parent_fn is not None:
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
    edges = [dict(row) for row in record.get("edges", [])]
    stats = dict(record.get("stats", {}))
    return post_id, nodes, edges, stats


def crawl_arctic_shift_discussions(
    output_dir: Path = Path("data/interim/vn"),
    *,
    subreddits: Sequence[str] = TARGET_SUBREDDITS,
    target_count: int = 500,
    min_comments: int = 40,
    min_depth: int = 3,
    min_vi_ratio: float = 0.25,
    initial_before: str | int = "7d",
    max_pages_per_sub: int = 100,
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
    exhausted: dict[str, bool] = {sub: False for sub in subreddits}
    pages_fetched: dict[str, int] = {sub: 0 for sub in subreddits}
    seen_post_ids: set[str] = set()

    all_nodes: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    collected_stats: list[dict[str, Any]] = []

    if checkpoint_jsonl is not None:
        checkpoint_jsonl = Path(checkpoint_jsonl)
        checkpoint_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_jsonl.exists():
            with checkpoint_jsonl.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    pid, nodes, edges, stats = _deserialize_checkpoint_record(json.loads(stripped))
                    if pid and pid not in seen_post_ids:
                        seen_post_ids.add(pid)
                        all_nodes.extend(nodes)
                        all_edges.extend(edges)
                        collected_stats.append(stats)

    try:
        while len(collected_stats) < target_count and not all(exhausted.values()):
            progressed = False
            for sub in subreddits:
                if len(collected_stats) >= target_count:
                    break
                if exhausted[sub] or pages_fetched[sub] >= max_pages_per_sub:
                    exhausted[sub] = True
                    continue

                params: dict[str, Any] = {"subreddit": sub, "limit": 100}
                if cursors[sub] is not None:
                    params["before"] = cursors[sub]

                payload = _request_with_retry(
                    session,
                    f"{ARCTIC_SHIFT_API_BASE}/posts/search",
                    params=params,
                    timeout=timeout,
                )
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
                        continue
                    seen_post_ids.add(pid)

                    time.sleep(request_delay)
                    comments = fetch_comment_tree(session, pid, timeout=timeout)
                    built = build_vn_cascade_from_thread(
                        post,
                        comments,
                        min_comments=min_comments,
                        min_depth=min_depth,
                        min_vi_ratio=min_vi_ratio,
                    )
                    if built is None:
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
                        with checkpoint_jsonl.open("a", encoding="utf-8") as fh:
                            fh.write(
                                json.dumps(
                                    {
                                        "post_id": pid,
                                        "nodes": serial_nodes,
                                        "edges": edges,
                                        "stats": stats,
                                    },
                                    ensure_ascii=False,
                                    default=str,
                                )
                                + "\n"
                            )

            if not progressed:
                break
    finally:
        session.close()

    if len(collected_stats) < target_count:
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

    nodes_table = build_nodes_table(all_nodes)
    edges_table = build_edges_table(all_edges)
    return save_cascade_parquet(nodes_table, edges_table, output_dir=output_dir, validate=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl Vietnamese discussion trees from Arctic Shift public REST API."
    )
    parser.add_argument("--subreddits", nargs="+", default=list(TARGET_SUBREDDITS))
    parser.add_argument("--target-count", type=int, default=500)
    parser.add_argument("--min-comments", type=int, default=40)
    parser.add_argument("--min-depth", type=int, default=3)
    parser.add_argument("--min-vi-ratio", type=float, default=0.25)
    parser.add_argument("--before", type=str, default="7d")
    parser.add_argument(
        "--before-epoch",
        type=int,
        default=None,
        help="Optional explicit UTC Unix epoch cursor for reproducible snapshots (overrides --before).",
    )
    parser.add_argument("--max-pages-per-sub", type=int, default=100)
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

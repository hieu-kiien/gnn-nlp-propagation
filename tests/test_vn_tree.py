"""Unit tests for Arctic Shift Vietnamese crawler, anonymization, and tree integrity."""

from __future__ import annotations

import hashlib
import unicodedata
from collections import deque
from pathlib import Path

from src.data.arctic_shift_vn import (
    build_vn_cascade_from_thread,
    contains_vietnamese_chars,
    extract_comments_from_arctic_payload,
    hash_author_sha256,
    is_invalid_or_spam_text,
    normalize_social_text_nfc,
)
from src.data.schema import (
    build_edges_table,
    build_nodes_table,
    load_cascade_parquet,
    save_cascade_parquet,
    validate_cascade_contract,
)


def _make_t1_node(
    cid: str,
    *,
    link_id: str,
    parent_id: str,
    author: str,
    body: str,
    created_utc: float,
    children: list[dict] | None = None,
) -> dict:
    replies_payload: dict | str = (
        {"kind": "Listing", "data": {"children": children}} if children else ""
    )
    return {
        "kind": "t1",
        "data": {
            "id": cid,
            "name": f"t1_{cid}",
            "link_id": link_id,
            "parent_id": parent_id,
            "author": author,
            "body": body,
            "created_utc": created_utc,
            "replies": replies_payload,
        },
    }


def test_author_hashing_and_nfc_slang_emoji_preservation() -> None:
    digest = hash_author_sha256("vozer_chinh_hieu")
    assert digest == hashlib.sha256(b"vozer_chinh_hieu").hexdigest()
    assert len(digest) == 64
    assert "vozer_chinh_hieu" not in digest

    assert hash_author_sha256("[deleted]") is None
    assert hash_author_sha256("[removed]") is None
    assert hash_author_sha256("AutoModerator") is None
    assert hash_author_sha256(None) is None

    # Decomposed NFD Vietnamese + emoji + slang -> NFC while preserving emoji and slang
    nfd_sample = unicodedata.normalize("NFD", "Thật ko vậy bạn ơi =))) 🤣❤️")
    nfc_norm = normalize_social_text_nfc(nfd_sample)
    assert unicodedata.is_normalized("NFC", nfc_norm)
    assert "🤣" in nfc_norm and "❤️" in nfc_norm and "=)))" in nfc_norm
    assert contains_vietnamese_chars(nfc_norm)
    assert not contains_vietnamese_chars("Pure English discussion only.")


def test_spam_and_deleted_comment_filtering() -> None:
    assert is_invalid_or_spam_text("[deleted]")
    assert is_invalid_or_spam_text("[removed]")
    assert is_invalid_or_spam_text("Chào bạn, bài viết đã được kiểm duyệt.", author_raw="AutoModerator")
    assert is_invalid_or_spam_text("Săn sale ngay https://shope.ee/xyz123")
    assert is_invalid_or_spam_text("https://example.com/spam")
    assert not is_invalid_or_spam_text("Chuẩn luôn bác ơi =))) 👍")
    assert not is_invalid_or_spam_text("Bài báo gốc nói rất rõ về thuế thu nhập nè https://vnexpress.net/abc")


def test_build_vn_cascade_nested_tree_depth_utc_and_pruning(tmp_path: Path) -> None:
    post_id = "1xtest9"
    base_ts = 1794000000.0
    post = {
        "id": post_id,
        "subreddit": "vozforums",
        "title": "Thảo luận về chi phí sinh hoạt ở thành phố lớn năm nay",
        "selftext": unicodedata.normalize("NFD", "Mọi người thấy sao ạ? 🤔"),
        "author": "op_user_123",
        "created_utc": base_ts,
        "is_self": True,
        "url": "",
        "num_comments": 48,
    }

    # Construct 4-level valid branch + 40+ valid replies + deleted branch + clock-skewed child
    deep_leaf = _make_t1_node(
        "c_d4",
        link_id=f"t3_{post_id}",
        parent_id="t1_c_d3",
        author="user_d4",
        body="Tầng 4: Đúng rồi bác ơi, tiền trọ tăng chóng mặt luôn 😅",
        created_utc=base_ts + 250.0,
    )
    d3_node = _make_t1_node(
        "c_d3",
        link_id=f"t3_{post_id}",
        parent_id="t1_c_d2",
        author="user_d3",
        body="Tầng 3: Mình phản đối ý kiến này, ở ngoại thành vẫn ổn mà!",
        created_utc=base_ts + 180.0,  # earlier than d2's 190 -> tests monotonic UTC clamp
        children=[deep_leaf],
    )
    d2_node = _make_t1_node(
        "c_d2",
        link_id=f"t3_{post_id}",
        parent_id="t1_c_d1",
        author="user_d2",
        body="Tầng 2: Chuẩn ko cần chỉnh =)))",
        created_utc=base_ts + 190.0,
        children=[d3_node],
    )
    d1_node = _make_t1_node(
        "c_d1",
        link_id=f"t3_{post_id}",
        parent_id=f"t3_{post_id}",
        author="user_d1",
        body="Tầng 1: Ở nội thành chi phí đội lên nhiều lắm.",
        created_utc=base_ts + 100.0,
        children=[d2_node],
    )

    # Deleted parent + its child (must both be safely excluded without broken edges)
    child_under_deleted = _make_t1_node(
        "c_under_del",
        link_id=f"t3_{post_id}",
        parent_id="t1_c_del",
        author="user_orphan",
        body="Ủa sao bình luận trên bị xóa rồi?",
        created_utc=base_ts + 95.0,
    )
    deleted_node = _make_t1_node(
        "c_del",
        link_id=f"t3_{post_id}",
        parent_id=f"t3_{post_id}",
        author="[deleted]",
        body="[deleted]",
        created_utc=base_ts + 90.0,
        children=[child_under_deleted],
    )

    # Spam node + more placeholder node
    spam_node = _make_t1_node(
        "c_spam",
        link_id=f"t3_{post_id}",
        parent_id=f"t3_{post_id}",
        author="spammer",
        body="Click vào đây nhận quà https://shope.ee/deal999",
        created_utc=base_ts + 92.0,
    )
    more_stub = {"kind": "more", "data": {"count": 2, "children": ["x1", "x2"]}}

    # Add 38 additional valid top-level and level-2 Vietnamese replies (total valid = 4 + 38 = 42 >= 40)
    extra_nodes = []
    for idx in range(38):
        cid = f"c_extra_{idx:02d}"
        parent_ref = f"t3_{post_id}" if idx < 25 else f"t1_c_extra_{idx - 25:02d}"
        extra_nodes.append(
            _make_t1_node(
                cid,
                link_id=f"t3_{post_id}",
                parent_id=parent_ref,
                author=f"vozer_{idx}",
                body=f"Ý kiến thảo luận thứ {idx}: đồng ý một phần nhưng còn tùy khu vực.",
                created_utc=base_ts + 300.0 + idx * 5.0,
            )
        )

    payload = [d1_node, deleted_node, spam_node, more_stub, *extra_nodes]
    extracted = extract_comments_from_arctic_payload(payload)
    # Raw author names must never survive extraction
    assert all("user_d1" not in str(item) and "op_user_123" not in str(item) for item in extracted)

    built = build_vn_cascade_from_thread(post, extracted, min_comments=40, min_depth=3)
    assert built is not None
    node_rows, edge_rows, stats = built

    assert stats["num_replies"] == 42
    assert stats["max_depth"] == 4
    assert len(node_rows) == 43
    assert len(edge_rows) == 42

    # Ensure deleted, orphan-under-deleted, and spam comments are excluded
    kept_ids = {r["node_id"] for r in node_rows}
    assert f"vn_{post_id}_t1_c_del" not in kept_ids
    assert f"vn_{post_id}_t1_c_under_del" not in kept_ids
    assert f"vn_{post_id}_t1_c_spam" not in kept_ids

    # Ensure NFD->NFC, emoji, and SHA-256 hash
    root_row = next(r for r in node_rows if r["is_root"])
    assert unicodedata.is_normalized("NFC", root_row["text"])
    assert "🤔" in root_row["text"]
    assert root_row["user_id_hash"] == hashlib.sha256(b"op_user_123").hexdigest()
    assert root_row["lang"] == "vi"

    # Persist to Parquet and verify against Data Contract + BFS depth/causality
    nodes_tbl = build_nodes_table(node_rows)
    edges_tbl = build_edges_table(edge_rows)
    save_cascade_parquet(nodes_tbl, edges_tbl, tmp_path / "vn_out", validate=True)

    reloaded_n, reloaded_e = load_cascade_parquet(tmp_path / "vn_out", validate=True)
    assert validate_cascade_contract(reloaded_n, reloaded_e, raise_on_error=False) == []

    # Verify strict temporal monotonicity and BFS depth >= 3
    node_map = {r["node_id"]: r for r in reloaded_n.to_pylist()}
    children: dict[str, list[str]] = {nid: [] for nid in node_map}
    for e in reloaded_e.to_pylist():
        children[e["source_id"]].append(e["target_id"])
        assert node_map[e["target_id"]]["created_at_utc"] >= node_map[e["source_id"]]["created_at_utc"]

    depths = {root_row["node_id"]: 0}
    q = deque([root_row["node_id"]])
    while q:
        curr = q.popleft()
        for nxt in children[curr]:
            depths[nxt] = depths[curr] + 1
            q.append(nxt)
    assert set(depths.keys()) == set(node_map.keys())
    assert max(depths.values()) >= 3

    # Verify threshold enforcement rejects shallow or small trees
    assert build_vn_cascade_from_thread(post, extracted, min_comments=100, min_depth=3) is None
    assert build_vn_cascade_from_thread(post, extracted, min_comments=40, min_depth=10) is None

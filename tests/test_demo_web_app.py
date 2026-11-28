"""Tests for the offline-first local SBi-FAGAT Demo Web MVP."""

from __future__ import annotations

import http.client
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from scripts.demo_web_app import create_server

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "web_demo"
PAPER_DIR = REPO_ROOT / "results"
DATA_VN_DIR = REPO_ROOT / "data" / "processed" / "vn"
CKPT_VN_PATH = REPO_ROOT / "checkpoints" / "best_sbifagat_vn.pt"
CE_VN_DIR = REPO_ROOT / "checkpoints" / "dual_ce_vn"


def _start_server(
    *,
    data_dir: Path,
    ckpt_path: Path,
    ce_dir: Path,
    paper_dir: Path = PAPER_DIR,
    web_dir: Path = WEB_DIR,
):
    server = create_server(
        host="127.0.0.1",
        port=0,
        data_dir=data_dir,
        ckpt_path=ckpt_path,
        ce_dir=ce_dir,
        paper_dir=paper_dir,
        web_dir=web_dir,
        preload=False,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _raw_http_get(port: int, path: str) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        headers = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, headers, body
    finally:
        conn.close()


def _http_json_post(port: int, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, str], dict[str, Any]]:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request(
            "POST",
            path,
            body=raw,
            headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
        )
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
        headers = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, headers, body
    finally:
        conn.close()


def test_loopback_bind_guard_rejects_non_local_hosts(tmp_path: Path) -> None:
    """Server must refuse 0.0.0.0 and external interfaces to protect local data."""
    for bad_host in ("0.0.0.0", "192.168.1.10", "example.com"):
        with pytest.raises(ValueError, match="loopback"):
            create_server(
                host=bad_host,
                port=0,
                data_dir=tmp_path / "absent",
                ckpt_path=tmp_path / "absent.pt",
                ce_dir=tmp_path / "absent_ce",
                paper_dir=PAPER_DIR,
                web_dir=WEB_DIR,
            )


def test_static_allowlist_traversal_and_security_headers(tmp_path: Path) -> None:
    """Allowlisted routes serve 200; traversal and disallowed paths return 403/404 with hardened headers."""
    server, thread = _start_server(
        data_dir=tmp_path / "no_data",
        ckpt_path=tmp_path / "no_ckpt.pt",
        ce_dir=tmp_path / "no_ce",
    )
    port = server.server_address[1]
    try:
        for route, expected_ct in (
            ("/", "text/html"),
            ("/styles.css", "text/css"),
            ("/app.js", "application/javascript"),
            ("/assets/fig5_vn.svg", "image/svg+xml"),
            ("/assets/fig5_pheme.svg", "image/svg+xml"),
        ):
            status, hdrs, body = _raw_http_get(port, route)
            assert status == 200, f"Failed static route {route}: {status}"
            assert expected_ct in hdrs["content-type"]
            assert hdrs.get("cache-control") == "no-store"
            assert hdrs.get("x-content-type-options") == "nosniff"
            assert "default-src 'self'" in hdrs.get("content-security-policy", "")
            assert len(body) > 20

        for blocked_path in (
            "/../paper/main.tex",
            "/../archive/docs_v1/paper/main.tex",
            "/assets/../../.env",
            "/%2e%2e/src/models/sbi_fagat.py",
            "/assets/other.svg",
            "/paper/results_macro_network_vn.json",
            "/results/results_macro_network_vn.json",
            "/archive/docs_v1/paper/results_macro_network_vn.json",
        ):
            status, hdrs, _ = _raw_http_get(port, blocked_path)
            assert status in (403, 404), f"Expected 403/404 on {blocked_path}, got {status}"
            assert status == 403
            assert hdrs.get("cache-control") == "no-store"
            assert hdrs.get("x-content-type-options") == "nosniff"

        status_404, hdrs_404, _ = _raw_http_get(port, "/does_not_exist")
        assert status_404 == 404
        assert hdrs_404.get("cache-control") == "no-store"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_ci_synthetic_fallback_when_data_and_checkpoint_absent(tmp_path: Path) -> None:
    """When local VN parquet/checkpoint files are missing, serve 2 clearly labeled synthetic presets."""
    server, thread = _start_server(
        data_dir=tmp_path / "missing_vn",
        ckpt_path=tmp_path / "missing_ckpt.pt",
        ce_dir=tmp_path / "missing_ce",
    )
    port = server.server_address[1]
    try:
        st_h, _, raw_h = _raw_http_get(port, "/api/health")
        assert st_h == 200
        health = json.loads(raw_h.decode("utf-8"))
        assert health["status"] == "ok"
        assert health["real_vn_data_available"] is False
        assert health["gnn_checkpoint_loaded"] is False
        assert health["cross_encoder_loaded"] is False
        assert set(health["regime_thresholds"].keys()) == {"q50", "q75", "q90"}

        st_l, _, raw_l = _raw_http_get(port, "/api/tier1/threads")
        assert st_l == 200
        listing = json.loads(raw_l.decode("utf-8"))
        assert listing["is_synthetic_fallback"] is True
        assert listing["count"] == 2
        assert len(listing["threads"]) == 2
        for item in listing["threads"]:
            assert "[Mẫu tổng hợp]" in item["title_snippet"]

        first_id = listing["threads"][0]["cascade_id"]
        st_d, _, raw_d = _raw_http_get(port, f"/api/tier1/thread?id={first_id}")
        assert st_d == 200
        detail = json.loads(raw_d.decode("utf-8"))
        assert detail["meta"]["is_synthetic_fallback"] is True
        assert detail["meta"]["cross_encoder_loaded"] is False
        assert detail["forecast"]["confidence_score"] is None
        assert detail["forecast"]["uncertainty_available"] is False
        assert "early_y_polar" in detail["forecast"]
        assert "irrational_fallacy" not in detail["forecast"]["behavior_counts"]
        assert detail["tree"]["nodes"][0]["stance_class"] == "root"
        assert len(detail["tree"]["nodes"]) >= 10
        assert len(detail["user_graph"]["nodes"]) >= 3
        assert len(detail["user_graph"]["edges"]) >= 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_sandbox_path_contraction_and_custom_input(tmp_path: Path) -> None:
    """Sandbox mode demonstrates Path Contraction, preserves short stances, and flags cross_encoder_loaded=False."""
    server, thread = _start_server(
        data_dir=tmp_path / "missing_vn",
        ckpt_path=CKPT_VN_PATH if CKPT_VN_PATH.is_file() else (tmp_path / "missing_ckpt.pt"),
        ce_dir=tmp_path / "missing_ce",
    )
    port = server.server_address[1]
    try:
        st, _, body = _http_json_post(
            port,
            "/api/tier1/sandbox",
            {"mode": "preset_junk_1", "junk_mode": "contract"},
        )
        assert st == 200
        stats = body["junk_filter_stats"]
        assert stats["mode"] == "contract"
        assert stats["contracted_nodes"] >= 2
        assert stats["dropped_leaves"] >= 1
        assert stats["relinked_edges"] >= 2
        assert body["meta"]["cross_encoder_loaded"] is False
        kept_texts = [n["text"] for n in body["kept_nodes"]]
        assert "Đồng ý" in kept_texts
        assert "Phản đối" in kept_texts
        assert any(e["relinked"] is True for e in body["tree"]["edges"])
        assert len(body["user_graph"]["nodes"]) >= 4
        assert len(body["user_graph"]["edges"]) >= 3

        # Compare against prune mode: Path Contraction preserves descendants of [deleted]
        st_p, _, body_p = _http_json_post(
            port,
            "/api/tier1/sandbox",
            {"mode": "preset_junk_1", "junk_mode": "prune"},
        )
        assert st_p == 200
        assert body_p["junk_filter_stats"]["mode"] == "prune"
        assert body_p["junk_filter_stats"]["contracted_nodes"] == 0
        assert stats["kept_comments"] >= body_p["junk_filter_stats"]["kept_comments"]

        # Preset 2 also runs cleanly and contracts middle-chain junk
        st_2, _, body_2 = _http_json_post(
            port,
            "/api/tier1/sandbox",
            {"mode": "preset_junk_2", "junk_mode": "contract"},
        )
        assert st_2 == 200
        assert body_2["junk_filter_stats"]["contracted_nodes"] >= 2
        assert body_2["junk_filter_stats"]["relinked_edges"] >= 2

        # Custom multi-line pasted text with middle-chain junk
        raw_lines = (
            "c1 (trả lời root): Ủng hộ thử nghiệm từng bước.\n"
            "c2 (trả lời c1): [deleted]\n"
            "c3 (trả lời c2): Đồng ý\n"
            "c4 (trả lời root): Không đồng ý vì thiếu hạ tầng.\n"
            "c5 (trả lời c4): Phản đối\n"
            "c6 (trả lời c1): hóng\n"
        )
        st_c, _, body_c = _http_json_post(
            port,
            "/api/tier1/sandbox",
            {
                "mode": "custom",
                "title": "Thử nghiệm cây hội thoại nhập tay",
                "root_text": "Nội dung bài đăng thử nghiệm",
                "raw_indented_text": raw_lines,
                "junk_mode": "contract",
            },
        )
        assert st_c == 200
        assert body_c["junk_filter_stats"]["contracted_nodes"] >= 1
        assert body_c["junk_filter_stats"]["relinked_edges"] >= 1
        assert body_c["meta"]["cross_encoder_loaded"] is False
        assert "Đồng ý" in [n["text"] for n in body_c["kept_nodes"]]

        # Malformed Reddit URL returns 400 cleanly
        st_bad, _, body_bad = _http_json_post(
            port,
            "/api/tier1/sandbox",
            {"mode": "reddit_url", "reddit_url": "https://reddit.com/../bad", "junk_mode": "contract"},
        )
        assert st_bad == 400
        assert "error" in body_bad

        # Custom structured comments with author_raw="AutoModerator" must be dropped by M6 bot filter
        st_bot, _, body_bot = _http_json_post(
            port,
            "/api/tier1/sandbox",
            {
                "mode": "custom",
                "title": "Kiểm tra lọc bot AutoModerator qua author_raw",
                "root_text": "Thảo luận chính sách công",
                "comments": [
                    {"id": "c1", "parent_id": "root", "text": "Bình luận hợp lệ ủng hộ.", "author": "user_ok"},
                    {"id": "c_bot", "parent_id": "root", "text": "Bình luận tự động của bot cần lọc.", "author_raw": "AutoModerator"},
                    {"id": "c2", "parent_id": "c1", "text": "Phản đối phương án này.", "author_raw": "user_2"},
                ],
                "junk_mode": "contract",
            },
        )
        assert st_bot == 200
        kept_bot_texts = [n["text"] for n in body_bot["kept_nodes"]]
        assert not any("bot cần lọc" in t for t in kept_bot_texts)
        assert any("Bình luận hợp lệ ủng hộ." in t for t in kept_bot_texts)
        assert body_bot["forecast"]["confidence_score"] is None
        assert body_bot["forecast"]["uncertainty_available"] is False
        assert body_bot["forecast"]["has_future_replies"] is False
        assert body_bot["forecast"]["actual_y_polar"] is None
        assert "early_y_polar" in body_bot["forecast"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_real_vn_test_threads_when_local_artifacts_present() -> None:
    """Verify real local VN Test-split loading when data/processed/vn & checkpoint exist."""
    if not (
        (DATA_VN_DIR / "nodes.parquet").is_file()
        and (DATA_VN_DIR / "edges.parquet").is_file()
        and (DATA_VN_DIR / "nlp_features.pt").is_file()
        and (DATA_VN_DIR / "topic_splits.json").is_file()
        and CKPT_VN_PATH.is_file()
    ):
        # Fresh clone / CI without gitignored real parquet: verify helper still passed fallback contract above
        assert not (DATA_VN_DIR / "nlp_features.pt").is_file()
        return

    server, thread = _start_server(
        data_dir=DATA_VN_DIR,
        ckpt_path=CKPT_VN_PATH,
        ce_dir=CE_VN_DIR,
    )
    port = server.server_address[1]
    try:
        st_h, _, raw_h = _raw_http_get(port, "/api/health")
        assert st_h == 200
        health = json.loads(raw_h.decode("utf-8"))
        assert health["real_vn_data_available"] is True
        assert health["gnn_checkpoint_loaded"] is True
        assert health["preextracted_2d_ce_available"] is True

        st_l, _, raw_l = _raw_http_get(port, "/api/tier1/threads")
        assert st_l == 200
        listing = json.loads(raw_l.decode("utf-8"))
        assert listing["is_synthetic_fallback"] is False
        assert listing["count"] == 10
        for item in listing["threads"]:
            assert "[REDACTED" not in item["title_snippet"]
        polars = [t["actual_y_polar"] for t in listing["threads"]]
        assert max(polars) > min(polars)

        first_id = listing["threads"][0]["cascade_id"]
        st_d, _, raw_d = _raw_http_get(port, f"/api/tier1/thread?id={first_id}")
        assert st_d == 200
        detail = json.loads(raw_d.decode("utf-8"))
        assert detail["meta"]["is_synthetic_fallback"] is False
        assert detail["meta"]["cross_encoder_loaded"] is True
        assert detail["meta"]["nlp_backend"] == "preextracted_full_2d_ce"
        assert detail["forecast"]["confidence_score"] is None
        assert detail["forecast"]["uncertainty_available"] is False
        assert "early_y_polar" in detail["forecast"]
        assert "irrational_fallacy" not in detail["forecast"]["behavior_counts"]
        assert detail["tree"]["nodes"][0]["stance_class"] == "root"
        assert len(detail["tree"]["nodes"]) == 21
        assert len(detail["tree"]["edges"]) == 20
        assert len(detail["top_trigger_edges"]) == 5
        assert len(detail["user_graph"]["nodes"]) >= 2
        for node in detail["tree"]["nodes"]:
            assert len(node["author_hash"]) <= 16
            assert "[REDACTED" not in node["text"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_tier2_macro_metrics_match_paper_jsons(tmp_path: Path) -> None:
    """GET /api/tier2/macro serves exact verified results/ VN and PHEME metrics and caveats."""
    server, thread = _start_server(
        data_dir=tmp_path / "no_data",
        ckpt_path=tmp_path / "no_ckpt.pt",
        ce_dir=tmp_path / "no_ce",
    )
    port = server.server_address[1]
    try:
        st, _, raw = _raw_http_get(port, "/api/tier2/macro")
        assert st == 200
        data = json.loads(raw.decode("utf-8"))
        vn = data["corpora"]["vn"]
        pheme = data["corpora"]["pheme"]

        assert round(vn["macro"]["polarization"]["signed_modularity"], 4) == pytest.approx(0.1988, abs=1e-4)
        assert round(pheme["macro"]["polarization"]["signed_modularity"], 4) == pytest.approx(0.4000, abs=1e-4)
        assert vn["macro"]["structural_balance"]["balance_z_score"] == pytest.approx(-15.1737, abs=1e-3)
        assert pheme["macro"]["structural_balance"]["balance_z_score"] == pytest.approx(-4.2971, abs=1e-3)

        vn_rows = {r["model"]: r for r in vn["signed_link"]["rows"]}
        pheme_rows = {r["model"]: r for r in pheme["signed_link"]["rows"]}
        assert vn_rows["PriorPairSignPersistence"]["trans_roc_auc"]["mean"] == pytest.approx(0.6296, abs=1e-4)
        assert vn_rows["SGCN"]["trans_roc_auc"]["mean"] == pytest.approx(0.6101, abs=1e-4)
        assert vn_rows["SBiFAGAT-Link"]["trans_roc_auc"]["mean"] == pytest.approx(0.5954, abs=1e-4)
        assert pheme_rows["SiGAT"]["trans_roc_auc"]["mean"] == pytest.approx(0.9178, abs=1e-4)
        assert pheme_rows["SBiFAGAT-Link"]["trans_roc_auc"]["mean"] == pytest.approx(0.9106, abs=1e-4)
        assert any("212" in c for c in data["caveats"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_static_web_assets_are_offline_and_clean() -> None:
    """Verify zero CDN links, zero emojis, accessible ARIA tabs, and default-off profanity toggle."""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    css = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    for content in (html, css, js):
        assert "fonts.googleapis.com" not in content
        assert "unpkg.com" not in content
        assert "cdn.jsdelivr.net" not in content
        for emoji in ("🚀", "✨", "🎨", "⚙️", "🔥"):
            assert emoji not in content

    assert 'role="tablist"' in html
    assert html.count('role="tabpanel"') == 2
    assert 'id="toggle-profanity-badge">' in html
    assert 'id="kpi-early-polar"' in html
    assert "innerHTML" not in js
    assert "insertAdjacentHTML" not in js
    assert "eval(" not in js
    assert "Ngụy biện / Bẻ lái" not in js
    assert "irrational_fallacy" not in js
    assert "active_graph_users" in js
    assert "largest_signed_wcc_ratio_active" in js
    assert "N/A (disabled)" in js
    assert 'createNode("div", "topic-bar-row")' not in js
    assert 'createNode("li", "topic-bar-row")' in js
    assert 'state.tier1Mode === "sandbox"' in js
    assert 'id="mega-scan-panel"' in html
    assert 'id="mega-comments-section"' in html
    assert 'id="hotspot-branch-bar"' in html
    assert 'id="btn-mega-clear-file"' in html
    assert 'value="unscored"' in html
    assert "MAX_SVG_NODES = 120" in js
    assert "megaCommentsReqSeq" in js


def test_parse_facebook_and_threads_copy_paste_cleanly() -> None:
    """Verify Facebook and Threads UI chrome is stripped while author names hash cleanly and stances survive."""
    from scripts.demo_web_app import (
        _detect_unsupported_social_url,
        _split_root_and_comments_from_raw,
        parse_sandbox_raw_text,
    )

    fb_raw = (
        "Đề xuất lắp camera AI phạt nguội xe máy\n"
        "Nguyễn Minh Tuấn\n"
        "Ủng hộ mạnh, chứ vượt đèn đỏ nguy hiểm quá.\n"
        "Thích · Phản hồi · 2 giờ\n"
        "Xem tất cả 3 phản hồi\n"
        "Trần Thu Hà\n"
        "Đồng ý\n"
        "Thích · Phản hồi · 1 giờ\n"
        "Lê Quốc Bảo\n"
        "hóng\n"
        "12\n"
        "Thích · Phản hồi · 30 phút\n"
    )
    title, root_text, remaining = _split_root_and_comments_from_raw(fb_raw)
    assert "camera AI" in title
    assert "camera AI" in root_text
    fb_comments = parse_sandbox_raw_text(remaining, post_id="fb_test", base_ts=1763400000.0)
    bodies = [c["body"] for c in fb_comments]
    assert bodies == ["Ủng hộ mạnh, chứ vượt đèn đỏ nguy hiểm quá.", "Đồng ý", "hóng"]
    assert all("Thích" not in b and "Nguyễn Minh Tuấn" not in b for b in bodies)
    assert len({c["author_hash"] for c in fb_comments}) == 3

    threads_raw = (
        "minhanh.99 · 3 giờ\n"
        "Không đồng ý tăng học phí đột ngột khi chưa nâng chất lượng.\n"
        "12 lượt thích · 4 phản hồi\n"
        "tuan_dev · 2 giờ\n"
        "Phản đối\n"
        "Thích · Trả lời · 1 giờ\n"
    )
    th_comments = parse_sandbox_raw_text(threads_raw, post_id="th_test", base_ts=1763400000.0)
    assert [c["body"] for c in th_comments] == [
        "Không đồng ý tăng học phí đột ngột khi chưa nâng chất lượng.",
        "Phản đối",
    ]
    assert th_comments[0]["author_hash"] != th_comments[1]["author_hash"]

    assert _detect_unsupported_social_url("https://www.facebook.com/groups/123/posts/456") == "Facebook"
    assert _detect_unsupported_social_url("https://www.threads.net/@user/post/C9abc") == "Threads"
    assert _detect_unsupported_social_url("https://www.reddit.com/r/vozforums/comments/1o02m18") is None


def test_sandbox_and_scan_jobs_social_url_guard_and_auto_split(tmp_path: Path) -> None:
    """FB/Threads URLs fail with honest 400 copy-paste guidance; auto_split_root runs end-to-end."""
    server, thread = _start_server(
        data_dir=tmp_path / "no_data",
        ckpt_path=tmp_path / "no_ckpt.pt",
        ce_dir=tmp_path / "no_ce",
    )
    port = server.server_address[1]
    try:
        for url in (
            "https://www.facebook.com/groups/demo/posts/12345",
            "https://www.threads.com/@demo/post/ABC123",
        ):
            st1, _, b1 = _http_json_post(
                port,
                "/api/tier1/sandbox",
                {"mode": "reddit_url", "reddit_url": url, "junk_mode": "contract"},
            )
            assert st1 == 400
            assert "Ctrl+C" in b1["error"]

            st2, _, b2 = _http_json_post(
                port,
                "/api/tier1/scan_jobs",
                {"mode": "reddit_url", "reddit_url": url, "junk_mode": "contract"},
            )
            assert st2 == 400
            assert "Ctrl+C" in b2["error"]

        fb_paste = (
            "Thử nghiệm bỏ làm việc từ xa để quay lại văn phòng toàn thời gian\n"
            "Nguyễn Văn An\n"
            "Đồng ý quay lại văn phòng để trao đổi nhanh hơn.\n"
            "Thích · Phản hồi · 2 giờ\n"
            "Trần Thị Bình\n"
            "hóng\n"
            "Thích · Phản hồi · 1 giờ\n"
            "Lê Văn Cường\n"
            "Phản đối, tốn thời gian kẹt xe lắm.\n"
            "Thích · Phản hồi · 30 phút\n"
        )
        st_auto, _, body_auto = _http_json_post(
            port,
            "/api/tier1/sandbox",
            {
                "mode": "custom",
                "title": "",
                "root_text": "",
                "raw_indented_text": fb_paste,
                "auto_split_root": True,
                "junk_mode": "contract",
            },
        )
        assert st_auto == 200
        assert "quay lại văn phòng" in body_auto["meta"]["root_title"]
        kept = [n["text"] for n in body_auto["kept_nodes"]]
        assert "Đồng ý quay lại văn phòng để trao đổi nhanh hơn." in kept
        assert "Phản đối, tốn thời gian kẹt xe lắm." in kept
        assert not any("Thích · Phản hồi" in t for t in kept)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_reddit_shortlink_and_arctic_posts_ids_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify redd.it shortlinks parse cleanly and Arctic Shift fetches via /posts/ids first."""
    from scripts import predict_live_thread as plt_mod

    assert plt_mod._extract_post_id_from_url("https://redd.it/1ndfgzf") == "1ndfgzf"
    called_urls: list[str] = []

    class _Resp:
        status_code = 200
        headers: dict[str, str] = {}

        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    def _mock_get(url: str, params: Any = None, headers: Any = None, timeout: int = 20):
        del params, headers, timeout
        called_urls.append(url)
        if url.endswith("/posts/ids"):
            return _Resp({"data": [{"id": "1ndfgzf", "title": "Bai VN lon", "selftext": "Noi dung", "author": "op", "created_utc": 1760000000}]})
        if url.endswith("/comments/search"):
            return _Resp({"data": [{"id": "c1", "parent_id": "t3_1ndfgzf", "body": "Dong y", "author": "u1", "created_utc": 1760000010}]})
        return _Resp({"data": []})

    monkeypatch.setattr(plt_mod.requests, "get", _mock_get)
    post, comments = plt_mod.fetch_reddit_thread_paginated("1ndfgzf", max_comments=200)
    assert any(u.endswith("/posts/ids") for u in called_urls)
    assert not any(u.endswith("/posts/search") for u in called_urls)
    assert post["title"] == "Bai VN lon"
    assert len(comments) == 1


def test_vnexpress_live_adapter_reddit_share_and_gnn_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify VnExpress multi-level reply parser, Reddit /s/ redirect + SSRF guard, @mention recovery, and GNN probe."""
    import torch
    from scripts import predict_live_thread as plt_mod
    from scripts.demo_web_app import parse_sandbox_raw_text
    from src.models.sbi_fagat import SBiFAGAT

    assert plt_mod.is_vnexpress_url("https://vnexpress.net/dau-tu-chung-khoan-5149672.html") is True
    assert plt_mod.is_vnexpress_url("https://www.reddit.com/r/vozforums/comments/1ndfgzf/") is False

    # 1. Mock Reddit /s/ redirect resolver (302 -> /comments/1ndfgzf/ + SSRF rejection)
    class _RedirectResp:
        def __init__(self, status_code: int, headers: dict[str, str] | None = None, text: str = "") -> None:
            self.status_code = status_code
            self.headers = headers or {}
            self.text = text

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {}

    def _mock_share_ok(url: str, allow_redirects: bool = False, timeout: int = 8, headers: Any = None):
        del allow_redirects, timeout, headers
        if "/s/htBgMuGWSv" in url:
            return _RedirectResp(302, {"location": "/r/vozforums/comments/1ndfgzf/title_slug/"})
        return _RedirectResp(200)

    monkeypatch.setattr(plt_mod.requests, "get", _mock_share_ok)
    assert plt_mod._extract_post_id_from_url("https://www.reddit.com/r/vozforums/s/htBgMuGWSv") == "1ndfgzf"

    def _mock_share_ssrf(url: str, allow_redirects: bool = False, timeout: int = 8, headers: Any = None):
        del url, allow_redirects, timeout, headers
        return _RedirectResp(302, {"location": "http://127.0.0.1:8765/internal"})

    monkeypatch.setattr(plt_mod.requests, "get", _mock_share_ssrf)
    with pytest.raises(ValueError):
        plt_mod._extract_post_id_from_url("https://www.reddit.com/r/vozforums/s/htBgMuGWSv")

    # 2. Mock VnExpress nested article + get + getreplay
    def _mock_vne_get(url: str, params: Any = None, headers: Any = None, timeout: int = 15, allow_redirects: bool = True):
        del headers, timeout, allow_redirects

        class _VneResp:
            status_code = 200
            headers: dict[str, str] = {}

            def __init__(self, text: str = "", json_data: Any = None) -> None:
                self.text = text
                self._json = json_data or {}

            def raise_for_status(self) -> None:
                return None

            def json(self) -> Any:
                return self._json

        if "index/getreplay" in url:
            return _VneResp(
                json_data={
                    "data": {
                        "items": [
                            {
                                "comment_id": 202,
                                "parent_id": 101,
                                "content": "Phản đối ý kiến trên vì chưa hợp lý.",
                                "userid": 9902,
                                "creation_time": 1760000200,
                            }
                        ]
                    }
                }
            )
        if "usi-saas.vnexpress.net/index/get" in url:
            return _VneResp(
                json_data={
                    "data": {
                        "total": 1,
                        "totalitem": 2,
                        "items": [
                            {
                                "comment_id": 101,
                                "parent_id": 101,
                                "content": "Đồng ý nên thử nghiệm trước.",
                                "userid": 9901,
                                "creation_time": 1760000100,
                                "replys": {"total": 1, "items": []},
                            }
                        ],
                    }
                }
            )
        return _VneResp(text='<title>Tranh luận chính sách mới - VnExpress</title><meta name="description" content="Mô tả bài viết thử nghiệm."/>')

    monkeypatch.setattr(plt_mod.requests, "get", _mock_vne_get)
    vne_post, vne_cmts = plt_mod.fetch_url_thread("https://vnexpress.net/tranh-luan-chinh-sach-moi-5149672.html")
    assert vne_post["title"] == "Tranh luận chính sách mới"
    assert len(vne_cmts) == 2
    assert vne_cmts[0]["parent_id"] == "t3_vne_5149672"
    assert vne_cmts[1]["parent_id"] == "t1_101"

    # 3. Deterministic explicit @mention reply recovery in parse_sandbox_raw_text
    raw_fb = (
        "minhanh.99 · 3 giờ\n"
        "Không đồng ý tăng phí gửi xe.\n"
        "tuan_dev · 2 giờ\n"
        "@minhanh.99 Đồng ý với bạn ở điểm này.\n"
        "hong_ha · 1 giờ\n"
        "Tôi thấy bình thường.\n"
    )
    parsed = parse_sandbox_raw_text(raw_fb, post_id="p1", base_ts=1763400000.0)
    assert len(parsed) == 3
    assert parsed[0]["parent_id"] == "t3_p1"
    assert parsed[1]["parent_id"] == parsed[0]["fullname"]
    assert parsed[2]["parent_id"] == "t3_p1"

    # 4. GNN counterfactual structural probe on real tensor
    from torch_geometric.data import Data

    d_probe = Data(
        x=torch.zeros((3, 778), dtype=torch.float32),
        edge_index_td=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_index_bu=torch.tensor([[1, 2], [0, 1]], dtype=torch.long),
        edge_attr=torch.tensor([[0.6, 0.1], [-0.8, 0.2]], dtype=torch.float32),
        cascade_stats=torch.zeros((1, 8), dtype=torch.float32),
    )
    gnn = SBiFAGAT(in_dim=778, hidden_dim=16, edge_dim=2, use_cascade_stats=True, predict_uncertainty=False)
    probe_res = plt_mod.compute_gnn_structural_probe(gnn, d_probe, checkpoint_loaded=True)
    assert probe_res["available"] is True
    assert probe_res["has_multi_hop"] is True
    assert probe_res["multi_hop_edges"] == 1
    assert "delta_polar_structure" in probe_res and "delta_polar_stance" in probe_res
    assert plt_mod.compute_gnn_structural_probe(gnn, d_probe, checkpoint_loaded=False)["available"] is False


def test_vn_news_adapters_stance_consistency_and_full_comments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify Dan Tri / Tuoi Tre / VnExpress pagination, SSRF generic meta, probe leakage fix, quote consistency & default all-comments."""
    import scripts.predict_live_thread as plt_mod
    from src.nlp.topic_post_analyzer import analyze_large_thread_dynamics, extract_hotspot_branches

    class _MockResp:
        def __init__(self, *, text: str = "", json_data: Any = None, status_code: int = 200) -> None:
            self.text = text
            self._json = json_data
            self.status_code = status_code
            self.encoding = "utf-8"

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(f"HTTP {self.status_code}")

        def json(self) -> Any:
            return self._json

        def iter_content(self, chunk_size: int = 32768):
            del chunk_size
            yield self.text.encode("utf-8")

    # 1. Dan Tri nested adapter + 0-comment friendly error
    def _mock_dt_get(url: str, params: dict[str, Any] | None = None, headers: Any = None, timeout: int = 15, **kwargs: Any) -> _MockResp:
        del headers, timeout, kwargs
        if "listcomment" in url:
            if (params or {}).get("isReply") == "true":
                return _MockResp(
                    json_data={
                        "count": 1,
                        "items": [
                            {
                                "commentId": "9002",
                                "parentId": "9001",
                                "commentContent": "Đúng vậy, đồng tình với ý kiến này.",
                                "userId": "u2",
                                "createdDate": "2026-11-28T08:34:57",
                            }
                        ],
                    }
                )
            if str((params or {}).get("objectId")) == "20261128152439984":
                return _MockResp(json_data={"count": 0, "items": []})
            return _MockResp(
                json_data={
                    "count": 1,
                    "items": [
                        {
                            "commentId": "9001",
                            "parentId": None,
                            "commentContent": "Bình luận gốc trên Dân Trí.",
                            "userId": "u1",
                            "replyCount": 1,
                            "createdDate": "2026-11-28T08:05:28",
                        }
                    ],
                }
            )
        return _MockResp(
            text='<title>Tin thử nghiệm | Báo Dân trí</title><meta name="description" content="Mô tả Dân Trí."/><meta property="article:published_time" content="2026-11-28T07:00:00+07:00"/>'
        )

    monkeypatch.setattr(plt_mod.requests, "get", _mock_dt_get)
    dt_post, dt_cmts = plt_mod.fetch_url_thread("https://dantri.com.vn/xa-hoi/bai-thu-nghiem-20261127105443160.htm")
    assert dt_post["source_platform"] == "dantri"
    assert dt_post["has_real_publish_time"] is True
    assert len(dt_cmts) == 2
    assert dt_cmts[1]["parent_id"] == "t1_9001"
    with pytest.raises(ValueError, match="0 bình luận"):
        plt_mod.fetch_url_thread("https://dantri.com.vn/noi-vu/bai-0-cmt-20261128152439984.htm")

    # 2. Tuoi Tre recursive child_comments adapter
    def _mock_tt_get(url: str, params: dict[str, Any] | None = None, headers: Any = None, timeout: int = 15, **kwargs: Any) -> _MockResp:
        del headers, timeout, kwargs
        if "getlist-comment.api" in url:
            if int((params or {}).get("pageindex") or 1) > 1:
                return _MockResp(json_data={"Success": True, "Data": "[]"})
            items = [
                {
                    "id": "501",
                    "parent_id": "0",
                    "content": "Ý kiến mở đầu Tuổi Trẻ.",
                    "user_id": "tt_u1",
                    "published_date": "2026-11-28T09:00:00",
                    "child_comments": [
                        {
                            "id": "502",
                            "parent_id": "501",
                            "content": "Phản hồi tầng 2 Tuổi Trẻ.",
                            "user_id": "tt_u2",
                            "published_date": "2026-11-28T09:10:00",
                            "child_comments": [],
                        }
                    ],
                }
            ]
            return _MockResp(json_data={"Success": True, "Data": json.dumps(items, ensure_ascii=False)})
        return _MockResp(text='<title>Bài báo mẫu - Tuổi Trẻ Online</title><meta name="description" content="Mô tả Tuổi Trẻ."/>')

    monkeypatch.setattr(plt_mod.requests, "get", _mock_tt_get)
    tt_post, tt_cmts = plt_mod.fetch_url_thread("https://tuoitre.vn/bai-bao-mau-100261128095247352.htm")
    assert tt_post["source_platform"] == "tuoitre"
    assert len(tt_cmts) == 2 and tt_cmts[1]["parent_id"] == "t1_501"

    # 3. VnExpress continues past server-capped <50 page and uses article:published_time
    def _mock_vne_paged(url: str, params: dict[str, Any] | None = None, headers: Any = None, timeout: int = 15, **kwargs: Any) -> _MockResp:
        del headers, timeout, kwargs
        if "index/getreplay" in url:
            return _MockResp(json_data={"data": {"items": []}})
        if "usi-saas.vnexpress.net/index/get" in url:
            off = int((params or {}).get("offset") or 0)
            if off == 0:
                return _MockResp(
                    json_data={
                        "data": {
                            "total": 25,
                            "totalitem": 25,
                            "items": [
                                {"comment_id": str(i), "content": f"Bình luận VNE {i}", "userid": f"u{i}", "creation_time": 1760000100 + i}
                                for i in range(1, 21)
                            ],
                        }
                    }
                )
            return _MockResp(
                json_data={
                    "data": {
                        "total": 25,
                        "totalitem": 25,
                        "items": [
                            {"comment_id": str(i), "content": f"Bình luận VNE {i}", "userid": f"u{i}", "creation_time": 1760000100 + i}
                            for i in range(21, 26)
                        ],
                    }
                }
            )
        return _MockResp(
            text='<title>Tin VnExpress nhiều trang - VnExpress</title><meta property="article:published_time" content="2025-10-09T09:00:00+07:00"/>'
        )

    monkeypatch.setattr(plt_mod.requests, "get", _mock_vne_paged)
    vne_p2, vne_c2 = plt_mod.fetch_url_thread("https://vnexpress.net/tin-nhieu-trang-5149999.html")
    assert len(vne_c2) == 25
    assert vne_p2["has_real_publish_time"] is True

    # 4. Generic SSRF block + No depth-2 double-negation fake pro quotes + hotspot oppose matches s_uv
    with pytest.raises(ValueError):
        plt_mod.fetch_generic_article_meta("http://127.0.0.1:8765/api/health")

    root_n = {"node_id": "r0", "author_hash": "op", "text": "Bài viết thử nghiệm."}
    n1 = {"node_id": "c1", "parent_id": "r0", "author_hash": "u1", "text": "Tôi không đồng tình chút nào với bài viết này.", "created_at_utc": 10.0}
    n2 = {"node_id": "c2", "parent_id": "c1", "author_hash": "u2", "text": "Tôi cũng phản đối mạnh mẽ bình luận trên vì vô lý.", "created_at_utc": 20.0}
    e1 = {"edge_id": "r0->c1", "source_id": "r0", "target_id": "c1", "s_uv": -0.35, "h_uv": 0.1}
    e2 = {"edge_id": "c1->c2", "source_id": "c1", "target_id": "c2", "s_uv": -0.40, "h_uv": 0.2}
    dyn = analyze_large_thread_dynamics(
        root_node=root_n,
        all_nodes=[root_n, n1, n2],
        all_edges=[e1, e2],
        edge_scores={"r0->c1": {"s_uv": -0.35, "h_uv": 0.1}, "c1->c2": {"s_uv": -0.40, "h_uv": 0.2}},
        depth_map={"r0": 0, "c1": 1, "c2": 2},
        lang="vi",
    )
    assert dyn["semantic_understanding"]["top_pro_arguments"] == []
    hbs = extract_hotspot_branches(
        root_id="r0",
        all_nodes=[root_n, n1, n2],
        all_edges=[e1, e2],
        edge_scores={"r0->c1": {"s_uv": -0.35, "h_uv": 0.1}, "c1->c2": {"s_uv": -0.40, "h_uv": 0.2}},
        depth_map={"r0": 0, "c1": 1, "c2": 2},
    )
    assert hbs[0]["oppose_ratio"] == pytest.approx(1.0)

    # 5. Server returns full comment_rows for sandbox and defaults scan_jobs branch to _all_top
    server, thread = _start_server(data_dir=tmp_path / "missing_vn", ckpt_path=tmp_path / "missing.pt", ce_dir=tmp_path / "missing_ce")
    port = server.server_address[1]
    try:
        st_l, _, raw_l = _raw_http_get(port, "/api/tier1/threads")
        assert st_l == 200
        first_cid = json.loads(raw_l.decode("utf-8"))["threads"][0]["cascade_id"]
        st_tc, _, raw_tc = _raw_http_get(port, f"/api/tier1/thread/comments?id={first_cid}&page=1")
        assert st_tc == 200
        tc_body = json.loads(raw_tc.decode("utf-8"))
        assert tc_body["total_items"] >= 10
        assert "stance_totals" in tc_body

        st_sb, _, sb_body = _http_json_post(port, "/api/tier1/sandbox", {"mode": "preset_junk_1", "junk_mode": "contract"})
        assert st_sb == 200
        assert len(sb_body.get("comment_rows") or []) >= 5
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

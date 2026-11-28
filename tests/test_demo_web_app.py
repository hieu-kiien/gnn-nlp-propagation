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

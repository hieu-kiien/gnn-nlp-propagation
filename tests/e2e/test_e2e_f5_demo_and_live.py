"""Opaque-box E2E tests for Feature Area F5: Live Predictor, Web Demo API & E2E Latency."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts.benchmark_efficiency import run_efficiency_benchmark
from scripts.demo_web_app import create_server
from scripts.predict_live_thread import (
    VN_POLAR_Q50,
    VN_POLAR_Q75,
    VN_POLAR_Q90,
    _classify_calibrated_regime,
    _extract_post_id_from_url,
    _make_demo_offline_thread,
    predict_live_cascade,
)
from src.data.pyg_dataset import TOTAL_NODE_FEATURE_DIM
from src.models.sbi_fagat import SBiFAGAT
from tests.e2e.conftest import (
    PAPER_DIR,
    WEB_DIR,
    http_get,
    http_post_json,
    http_post_raw,
    start_demo_server,
)


def _make_25_reply_thread(*, future_oppose: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_ts = 1_700_000_000.0
    post = {
        "id": "live25",
        "title": "Thảo luận chính sách giao thông đô thị mới",
        "selftext": "Mọi người đánh giá tác động thực tế của đề xuất này như thế nào?",
        "author": "op_user",
        "subreddit": "vozforums",
        "created_utc": base_ts,
    }
    comments: list[dict[str, Any]] = []
    for idx in range(1, 26):
        is_future = idx > 20
        if is_future and future_oppose:
            body = "Phản đối hoàn toàn phương án này, bất hợp lý và gây kẹt xe thêm!" if idx % 2 == 0 else "Ủng hộ tuyệt đối, chuẩn luôn!"
        else:
            body = "Đồng ý với quan điểm phân tích phía trên, rất hợp lý và rõ ràng."
        comments.append(
            {
                "fullname": f"t1_c{idx:02d}",
                "parent_id": "t3_live25" if idx <= 5 else f"t1_c{(idx - 1) // 2 + 1:02d}",
                "link_id": "t3_live25",
                "author": f"user_{idx:02d}",
                "text": body,
                "created_utc": base_ts + 60.0 * idx,
            }
        )
    return post, comments


def _save_deterministic_gnn_ckpt(ckpt_path: Path, *, hidden_dim: int = 32, predict_uncertainty: bool = False) -> Path:
    torch.manual_seed(42)
    model = SBiFAGAT(
        in_dim=TOTAL_NODE_FEATURE_DIM,
        hidden_dim=hidden_dim,
        edge_dim=2,
        num_rumor_classes=4,
        use_cascade_stats=True,
        predict_uncertainty=predict_uncertainty,
        use_polar_persistence_anchor=True,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim": TOTAL_NODE_FEATURE_DIM,
            "num_rumor_classes": 4,
            "config": {
                "hidden_dim": hidden_dim,
                "edge_dim": 2,
                "eps": 0.5,
                "dropout": 0.1,
                "use_cascade_stats": True,
                "predict_uncertainty": predict_uncertainty,
                "use_polar_persistence_anchor": True,
            },
        },
        ckpt_path,
    )
    return ckpt_path


# ---------------------------------------------------------------------------
# Tier 1: Isolated Feature Coverage
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_f5_t1_untrained_guard_blocks_real_thread_and_allows_demo() -> None:
    """Live Predictor blocks uncheckpointed inference on real threads unless allow_untrained/is_demo."""
    real_post, real_comments = _make_25_reply_thread(future_oppose=False)
    real_post = dict(real_post, id="real999")
    real_comments = [
        dict(
            c,
            link_id="t3_real999",
            parent_id=("t3_real999" if c["parent_id"] == "t3_live25" else c["parent_id"]),
        )
        for c in real_comments
    ]

    with pytest.raises(RuntimeError):
        predict_live_cascade(real_post, real_comments, allow_untrained=False, is_demo=False)

    res_allowed = predict_live_cascade(real_post, real_comments, allow_untrained=True, is_demo=False)
    assert res_allowed["checkpoint_loaded"] is False
    assert "predicted_polarization_score" in res_allowed
    assert 0.0 <= float(res_allowed["predicted_polarization_score"]) <= 1.0

    demo_post, demo_comments = _make_demo_offline_thread()
    res_demo = predict_live_cascade(demo_post, demo_comments, allow_untrained=False, is_demo=True)
    assert res_demo["checkpoint_loaded"] is False


@pytest.mark.tier1
def test_f5_t1_live_cascade_returns_early_y_polar_and_future_only_actual(tmp_path: Path) -> None:
    """B8 contract: predict_live_cascade returns explicit early_y_polar and future_only actual polarization."""
    ckpt = _save_deterministic_gnn_ckpt(tmp_path / "gnn.pt", predict_uncertainty=False)
    post, comments = _make_25_reply_thread(future_oppose=True)
    res = predict_live_cascade(post, comments, k_horizon=20, gnn_ckpt=ckpt, junk_mode="contract")

    assert "early_y_polar" in res and isinstance(res["early_y_polar"], float)
    assert 0.0 <= res["early_y_polar"] <= 1.0
    post_hoc = res.get("actual_post_hoc_comparison", res)
    assert post_hoc["has_future_replies"] is True
    assert post_hoc["actual_full_polarization"] is not None
    assert 0.0 <= float(post_hoc["actual_full_polarization"]) <= 1.0
    # Early replies (1..20) were all agreeing; future replies (21..25) are polarized oppose/support ->
    # under future_only (t > K), actual_full_polarization is noticeably higher than early_y_polar.
    assert float(post_hoc["actual_full_polarization"]) > float(res["early_y_polar"])

    # Verify future_only isolation: changing t > K replies leaves early_y_polar identical while shifting future actual
    post_ag, comments_ag = _make_25_reply_thread(future_oppose=False)
    res_ag = predict_live_cascade(post_ag, comments_ag, k_horizon=20, gnn_ckpt=ckpt, junk_mode="contract")
    post_hoc_ag = res_ag.get("actual_post_hoc_comparison", res_ag)
    assert res_ag["early_y_polar"] == pytest.approx(res["early_y_polar"], abs=1e-6)
    assert float(post_hoc["actual_full_polarization"]) > float(post_hoc_ag["actual_full_polarization"])


@pytest.mark.tier1
def test_f5_t1_calibrated_regime_respects_q50_q75_q90_without_or_override() -> None:
    """B8 contract: _classify_calibrated_regime strictly respects Q50/Q75/Q90 without heuristic OR override."""
    heavy_conflict_counts = {"support": 1, "neutral": 0, "constructive_oppose": 5, "hostile_conflict": 5}

    low_label = _classify_calibrated_regime(VN_POLAR_Q50 - 0.02, heavy_conflict_counts)
    assert "Q75" not in low_label and "Q90" not in low_label

    mid_label = _classify_calibrated_regime((VN_POLAR_Q50 + VN_POLAR_Q75) * 0.5, heavy_conflict_counts)
    assert "Q50" in mid_label and "Q90" not in mid_label

    q75_label = _classify_calibrated_regime((VN_POLAR_Q75 + VN_POLAR_Q90) * 0.5, {"support": 10})
    assert "Q75" in q75_label

    q90_label = _classify_calibrated_regime(VN_POLAR_Q90 + 0.05, {"support": 10})
    assert "Q90" in q90_label


@pytest.mark.tier1
def test_f5_t1_live_predictor_confidence_none_when_uncertainty_false(tmp_path: Path) -> None:
    """B8 contract: predict_live_cascade returns confidence_score=None when predict_uncertainty=False (never 0.75)."""
    ckpt_det = _save_deterministic_gnn_ckpt(tmp_path / "gnn_det.pt", predict_uncertainty=False)
    post, comments = _make_demo_offline_thread()
    res = predict_live_cascade(post, comments, k_horizon=20, gnn_ckpt=ckpt_det)
    assert res["confidence_score"] is None
    assert res.get("uncertainty_available") in (False, None)


@pytest.mark.tier1
def test_f5_t1_live_predictor_no_4d_irrational_fallacy_leftovers(tmp_path: Path) -> None:
    """M9 contract: live predictor emits clean 2D/3-stance behaviors without legacy 4D irrational_fallacy bloat."""
    ckpt = _save_deterministic_gnn_ckpt(tmp_path / "gnn_clean.pt", predict_uncertainty=False)
    post, comments = _make_demo_offline_thread()
    res = predict_live_cascade(post, comments, k_horizon=20, gnn_ckpt=ckpt)
    assert "irrational_fallacy" not in res["behavior_counts"]
    assert "vo_ly=" not in json.dumps(res, ensure_ascii=False)


_LATENCY_KEY_RE = re.compile(r"(?:^|[^a-z0-9])(?:ms|latency|elapsed|time)(?:[^a-z0-9]|$)")
_E2E_TAGS = ("e2e", "raw_text", "raw-text", "end_to_end")


def _is_latency_key(key: Any) -> bool:
    return bool(_LATENCY_KEY_RE.search(str(key).lower()))


def _extract_e2e_latency_ms(payload: Any) -> list[float]:
    """Extract latency (ms) values belonging to E2E raw-text profiling."""
    found: list[float] = []

    def _collect_ms(obj: Any, *, allow_bare_scalar: bool = False) -> None:
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            if allow_bare_scalar:
                found.append(float(obj))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _collect_ms(v, allow_bare_scalar=_is_latency_key(k))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _collect_ms(item, allow_bare_scalar=allow_bare_scalar)

    if isinstance(payload, dict):
        for k, v in payload.items():
            kl = str(k).lower()
            if any(tag in kl for tag in _E2E_TAGS):
                _collect_ms(v, allow_bare_scalar=_is_latency_key(kl))
            elif k in ("models", "rows") and isinstance(v, list):
                found.extend(_extract_e2e_latency_ms(v))
    elif isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            model_name = str(row.get("model", "")).lower()
            if any(tag in model_name for tag in _E2E_TAGS):
                _collect_ms(row, allow_bare_scalar=False)
            else:
                for k, v in row.items():
                    kl = str(k).lower()
                    if any(tag in kl for tag in _E2E_TAGS):
                        _collect_ms(v, allow_bare_scalar=_is_latency_key(kl))
    elif isinstance(payload, tuple):
        for item in payload:
            found.extend(_extract_e2e_latency_ms(item))
    return found


@pytest.mark.tier1
def test_f5_t1_benchmark_efficiency_returns_required_models_and_e2e_profile() -> None:
    """M7 contract: benchmark_efficiency includes BiEdgeGATv2, SBiFAGAT, and E2E raw-text latency profiling."""
    import inspect
    import scripts.benchmark_efficiency as bench_mod

    rows = run_efficiency_benchmark(
        num_cascades=4,
        k_horizon=10,
        hidden_dim=32,
        edge_dim=2,
        warmup=1,
        repeats=1,
        device=torch.device("cpu"),
    )
    row_list = rows if isinstance(rows, list) else rows.get("models", rows.get("rows", []))
    by_name = {r["model"]: r for r in row_list}
    assert "SBiFAGAT" in by_name
    assert "BiEdgeGATv2" in by_name, f"BiEdgeGATv2 missing from benchmark_efficiency models: {list(by_name.keys())}"
    for r in row_list:
        assert r["params"] > 0
        assert r["infer_ms_per_cascade"] > 0.0
        assert r["train_ms_per_batch"] > 0.0

    # Verify M4/M7 E2E raw-text latency profiler is actually executed and produces positive ms/cascade
    e2e_ms_vals = _extract_e2e_latency_ms(rows)
    has_e2e_arg = "include_e2e" in inspect.signature(run_efficiency_benchmark).parameters
    if has_e2e_arg:
        e2e_res = run_efficiency_benchmark(
            num_cascades=2,
            k_horizon=10,
            hidden_dim=32,
            edge_dim=2,
            warmup=1,
            repeats=1,
            include_e2e=True,
            device=torch.device("cpu"),
        )
        e2e_ms_vals = _extract_e2e_latency_ms(e2e_res)
        assert any(v > 0.0 for v in e2e_ms_vals), (
            f"run_efficiency_benchmark(..., include_e2e=True) did not produce positive E2E raw-text latency ms: {e2e_res}"
        )
    else:
        e2e_fns = [
            getattr(bench_mod, attr)
            for attr in dir(bench_mod)
            if callable(getattr(bench_mod, attr))
            and ("e2e" in attr.lower() or "raw_text" in attr.lower())
        ]
        assert e2e_fns or any(v > 0.0 for v in e2e_ms_vals), (
            "scripts.benchmark_efficiency missing E2E raw-text latency profiler (M4/M7)"
        )
        if e2e_fns:
            fn = e2e_fns[0]
            sig = inspect.signature(fn)
            call_kwargs: dict[str, Any] = {}
            if "num_cascades" in sig.parameters:
                call_kwargs["num_cascades"] = 2
            if "repeats" in sig.parameters:
                call_kwargs["repeats"] = 1
            if "warmup" in sig.parameters:
                call_kwargs["warmup"] = 1
            if "device" in sig.parameters:
                call_kwargs["device"] = torch.device("cpu")
            e2e_res = fn(**call_kwargs)
            assert e2e_res is not None
            fn_ms_vals = _extract_e2e_latency_ms(e2e_res)
            if not fn_ms_vals and isinstance(e2e_res, dict):
                fn_ms_vals = [
                    float(v)
                    for k, v in e2e_res.items()
                    if _is_latency_key(k)
                    and isinstance(v, (int, float))
                    and not isinstance(v, bool)
                ]
            assert any(v > 0.0 for v in fn_ms_vals), f"Expected positive E2E latency ms in {e2e_res}"


# ---------------------------------------------------------------------------
# Tier 2: Boundary, Corner & Negative Rejection Cases
# ---------------------------------------------------------------------------


@pytest.mark.tier2
def test_f5_t2_http_loopback_guard_and_security_headers(tmp_path: Path) -> None:
    """HTTP server refuses non-loopback bind hosts, enforces CSP/nosniff/no-store, and blocks traversal."""
    for bad_host in ("0.0.0.0", "192.168.1.5"):
        with pytest.raises(ValueError, match="loopback"):
            create_server(
                host=bad_host,
                port=0,
                data_dir=tmp_path / "no_data",
                ckpt_path=tmp_path / "no_ckpt.pt",
                ce_dir=tmp_path / "no_ce",
                paper_dir=PAPER_DIR,
                web_dir=WEB_DIR,
            )

    server, thread, port = start_demo_server(
        data_dir=tmp_path / "no_data",
        ckpt_path=tmp_path / "no_ckpt.pt",
        ce_dir=tmp_path / "no_ce",
    )
    try:
        st, hdrs, _ = http_get(port, "/")
        assert st == 200
        assert hdrs.get("cache-control") == "no-store"
        assert hdrs.get("x-content-type-options") == "nosniff"
        assert "default-src 'self'" in hdrs.get("content-security-policy", "")

        for blocked in (
            "/../paper/main.tex",
            "/../archive/docs_v1/paper/main.tex",
            "/results/results_macro_network_vn.json",
            "/%2e%2e/src/models/sbi_fagat.py",
            "/assets/forbidden.svg",
        ):
            st_b, _, _ = http_get(port, blocked)
            assert st_b == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.tier2
def test_f5_t2_http_fallback_threads_and_thread_detail_schema(tmp_path: Path) -> None:
    """Offline fallback mode exposes 2 presets, full thread JSON with early_y_polar, and 400/404 errors."""
    server, thread, port = start_demo_server(
        data_dir=tmp_path / "no_data",
        ckpt_path=tmp_path / "no_ckpt.pt",
        ce_dir=tmp_path / "no_ce",
    )
    try:
        st_h, _, raw_h = http_get(port, "/api/health")
        assert st_h == 200
        health = json.loads(raw_h.decode("utf-8"))
        assert health["real_vn_data_available"] is False

        st_l, _, raw_l = http_get(port, "/api/tier1/threads")
        assert st_l == 200
        listing = json.loads(raw_l.decode("utf-8"))
        assert listing["is_synthetic_fallback"] is True
        assert listing["count"] == 2

        cid = listing["threads"][0]["cascade_id"]
        st_d, _, raw_d = http_get(port, f"/api/tier1/thread?id={cid}")
        assert st_d == 200
        detail = json.loads(raw_d.decode("utf-8"))
        assert "early_y_polar" in detail["forecast"]
        assert "predicted_polarization_score" in detail["forecast"]
        assert 0.0 <= float(detail["forecast"]["predicted_polarization_score"]) <= 1.0
        assert len(detail["tree"]["nodes"]) >= 10

        st_400, _, _ = http_get(port, "/api/tier1/thread")
        assert st_400 == 400
        st_404, _, _ = http_get(port, "/api/tier1/thread?id=unknown_id_xyz")
        assert st_404 == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.tier2
def test_f5_t2_http_sandbox_contract_vs_prune_and_word_boundary(tmp_path: Path) -> None:
    """Sandbox contract mode preserves short stances and uses fixed B5 word-boundary cues."""
    server, thread, port = start_demo_server(
        data_dir=tmp_path / "no_data",
        ckpt_path=tmp_path / "no_ckpt.pt",
        ce_dir=tmp_path / "no_ce",
    )
    try:
        raw_lines = (
            "c1 (trả lời root): Phạt nguội qua ứng dụng rất minh bạch.\n"
            "c2 (trả lời c1): [deleted]\n"
            "c3 (trả lời c2): Đồng ý\n"
            "c4 (trả lời root): Chiếc xe đó đã biến mất khỏi camera.\n"
            "c5 (trả lời c4): Phản đối\n"
        )
        st_c, _, body_c = http_post_json(
            port,
            "/api/tier1/sandbox",
            {
                "mode": "custom",
                "title": "Kiểm tra word-boundary và Path Contraction",
                "root_text": "Thảo luận quy định xử phạt giao thông mới",
                "raw_indented_text": raw_lines,
                "junk_mode": "contract",
            },
        )
        assert st_c == 200
        assert body_c["junk_filter_stats"]["contracted_nodes"] >= 1
        assert any(e.get("relinked") is True for e in body_c["tree"]["edges"])

        nodes_by_text = {n["text"]: n for n in body_c["tree"]["nodes"]}
        assert "Đồng ý" in nodes_by_text
        assert "Phản đối" in nodes_by_text
        # Word-boundary fix: 'phạt nguội' and 'biến mất' must NOT trigger hostile_conflict
        for text_key, node in nodes_by_text.items():
            if "Phạt nguội" in text_key or "biến mất" in text_key:
                assert "behavior_label" in node and node["behavior_label"] != "hostile_conflict", (
                    f"False hostile classification on {text_key!r}: {node}"
                )
        # Opposition cue fix: 'Phản đối' must be classified as oppose
        oppose_node = nodes_by_text["Phản đối"]
        assert oppose_node.get("stance_class") == "oppose" or float(oppose_node.get("s_uv", 0.0)) < 0.0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.tier2
def test_f5_t2_http_sandbox_rejects_bad_inputs_and_oversize_payload(tmp_path: Path) -> None:
    """Sandbox rejects invalid JSON, unknown mode, bad Reddit URL, empty custom input, and >256KB body."""
    server, thread, port = start_demo_server(
        data_dir=tmp_path / "no_data",
        ckpt_path=tmp_path / "no_ckpt.pt",
        ce_dir=tmp_path / "no_ce",
    )
    try:
        st_json, _, _ = http_post_raw(port, "/api/tier1/sandbox", b"{not-valid-json")
        assert st_json == 400

        st_mode, _, _ = http_post_json(port, "/api/tier1/sandbox", {"mode": "nonexistent_mode"})
        assert st_mode == 400

        st_url, _, _ = http_post_json(
            port, "/api/tier1/sandbox", {"mode": "reddit_url", "reddit_url": "https://reddit.com/../bad"}
        )
        assert st_url == 400

        st_empty, _, _ = http_post_json(
            port, "/api/tier1/sandbox", {"mode": "custom", "title": "", "root_text": "", "raw_indented_text": ""}
        )
        assert st_empty == 400

        oversize = b'{"mode":"custom","root_text":"' + (b"a" * 270_000) + b'"}'
        try:
            st_big, _, _ = http_post_raw(port, "/api/tier1/sandbox", oversize)
            assert st_big in (400, 413)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # Windows socket RST when server rejects Content-Length > _MAX_POST_BYTES immediately
            pass

        # Server must remain alive and healthy after oversize POST rejection
        st_alive, _, raw_alive = http_get(port, "/api/health")
        assert st_alive == 200
        assert json.loads(raw_alive.decode("utf-8"))["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.tier2
def test_f5_t2_predict_live_corrupt_ckpt_bad_url_and_empty_comments(tmp_path: Path) -> None:
    """Corrupt checkpoint, invalid Reddit URLs, and empty comment trees fail cleanly with ValueError."""
    bad_ckpt = tmp_path / "corrupt.pt"
    torch.save({"not_model_state": 1}, bad_ckpt)
    post, comments = _make_demo_offline_thread()

    with pytest.raises(ValueError, match="model_state_dict"):
        predict_live_cascade(post, comments, gnn_ckpt=bad_ckpt)

    with pytest.raises(ValueError):
        predict_live_cascade(post, [], allow_untrained=True)

    for bad_url in ("", "https://reddit.com/../etc/passwd", "https://reddit.com/r/vozforums/comments/"):
        with pytest.raises(ValueError):
            _extract_post_id_from_url(bad_url)


@pytest.mark.tier2
def test_f5_t2_web_demo_static_assets_offline_and_clean_labels() -> None:
    """M5-UI / M9 contract: web_demo static files are offline-safe and purged of 4D 'Ngụy biện / Bẻ lái' labels."""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    css = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    for content in (html, css, js):
        assert "fonts.googleapis.com" not in content
        assert "unpkg.com" not in content
        assert "cdn.jsdelivr.net" not in content
    assert "innerHTML" not in js
    assert "insertAdjacentHTML" not in js
    assert "eval(" not in js
    assert "Ngụy biện / Bẻ lái" not in js
    assert "irrational_fallacy" not in js
    assert "active_graph_users" in js


@pytest.mark.tier2
def test_f5_t2_http_excludes_redacted_doxxing_pii_threads_offline(tmp_path: Path) -> None:
    """M7 contract: DemoBackend / /api/tier1/threads must exclude [REDACTED_DOXXING_PII] threads."""
    import pyarrow as pa
    from src.data.schema import EDGES_ARROW_SCHEMA, NODES_ARROW_SCHEMA, save_cascade_parquet
    from tests.e2e.conftest import build_synthetic_processed_corpus

    ds_dir = tmp_path / "vn_redacted_ds"
    info = build_synthetic_processed_corpus(
        ds_dir,
        num_topics=12,
        cascades_per_topic=1,
        replies_per_cascade=26,
        min_total_replies_extracted=25,
    )
    test_topics = sorted(info["splits"]["test"])
    assert len(test_topics) >= 2
    redacted_topic = test_topics[0]
    nodes_rows = [dict(r) for r in info["nodes_rows"]]
    redacted_cid = ""
    for r in nodes_rows:
        if r["topic_id"] == redacted_topic and r["is_root"]:
            r["text"] = "[REDACTED_DOXXING_PII] Bài đăng chứa thông tin cá nhân bị ẩn"
            redacted_cid = str(r["cascade_id"])
            break
    assert redacted_cid
    save_cascade_parquet(
        pa.Table.from_pylist(nodes_rows, schema=NODES_ARROW_SCHEMA),
        pa.Table.from_pylist(info["edges_rows"], schema=EDGES_ARROW_SCHEMA),
        ds_dir,
        validate=True,
    )

    ckpt_path = _save_deterministic_gnn_ckpt(tmp_path / "vn_ckpt.pt", hidden_dim=32, predict_uncertainty=False)
    server, thread, port = start_demo_server(
        data_dir=ds_dir,
        ckpt_path=ckpt_path,
        ce_dir=tmp_path / "no_ce",
    )
    try:
        st_l, _, raw_l = http_get(port, "/api/tier1/threads")
        assert st_l == 200
        listing = json.loads(raw_l.decode("utf-8"))
        assert listing["is_synthetic_fallback"] is False
        assert listing["count"] >= 1
        returned_cids = {t["cascade_id"] for t in listing["threads"]}
        assert redacted_cid not in returned_cids, (
            f"DemoBackend exposed [REDACTED_DOXXING_PII] cascade {redacted_cid}"
        )
        for item in listing["threads"]:
            assert "[REDACTED" not in item["title_snippet"]
            st_d, _, raw_d = http_get(port, f"/api/tier1/thread?id={item['cascade_id']}")
            assert st_d == 200
            detail = json.loads(raw_d.decode("utf-8"))
            assert detail["forecast"]["confidence_score"] is None
            for node in detail["tree"]["nodes"]:
                assert "[REDACTED" not in node["text"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

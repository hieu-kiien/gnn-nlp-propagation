"""Unit and integration tests for the isolated Mega-Thread 10k Scanner."""

from __future__ import annotations

import http.client
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import pytest
import requests

from scripts.demo_web_app import (
    _MAX_TREE_RENDER_NODES,
    build_megathread_stress_preset,
    create_server,
)
from scripts.predict_live_thread import (
    OOD_SCALE_THRESHOLD,
    TRAINING_MAX_REPLIES_REF,
    fetch_reddit_thread_paginated,
    load_thread_from_file,
    predict_live_cascade,
)
from src.data.arctic_shift_vn import build_vn_cascade_from_thread
from src.nlp.topic_post_analyzer import (
    analyze_large_thread_dynamics,
    extract_hotspot_branches,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "web_demo"
PAPER_DIR = REPO_ROOT / "results"


def _http_get(port: int, path: str, timeout: float = 20.0) -> tuple[int, dict[str, Any]]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        return resp.status, json.loads(raw)
    finally:
        conn.close()


def _http_post(port: int, path: str, payload: dict[str, Any], timeout: float = 20.0) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        return resp.status, json.loads(raw)
    finally:
        conn.close()


def test_build_tree_path_contraction_and_trajectory_10k_comments() -> None:
    """10,000-comment tree construction, Path Contraction, trajectory milestones, and hotspot cap run in O(N)."""
    t0 = time.perf_counter()
    post, comments = build_megathread_stress_preset(10000)
    stats: dict[str, Any] = {}
    built = build_vn_cascade_from_thread(
        post,
        comments,
        min_comments=1,
        min_depth=1,
        min_vi_ratio=0.0,
        junk_mode="contract",
        stats_out=stats,
    )
    assert built is not None
    nodes, edges, _ = built
    assert stats["raw_comments"] == 10000
    assert stats["kept_comments"] > 9000
    assert stats["contracted_nodes"] > 0 or stats["dropped_leaves"] > 0

    root = next(n for n in nodes if bool(n["is_root"]))
    root_id = str(root["node_id"])
    adj: dict[str, list[str]] = {str(n["node_id"]): [] for n in nodes}
    for e in edges:
        adj.setdefault(str(e["source_id"]), []).append(str(e["target_id"]))
    depth_map: dict[str, int] = {root_id: 0}
    q: deque[str] = deque([root_id])
    while q:
        curr = q.popleft()
        for ch in adj.get(curr, []):
            assert ch not in depth_map, "Cycle detected in 10k tree"
            depth_map[ch] = depth_map[curr] + 1
            q.append(ch)
    assert len(depth_map) == len(nodes)

    edge_scores = {
        str(e["edge_id"]): {
            "s_uv": -0.6 if idx % 2 == 0 else 0.5,
            "h_uv": 0.7 if idx % 5 == 0 else 0.1,
        }
        for idx, e in enumerate(edges)
    }
    dyn = analyze_large_thread_dynamics(
        root_node=root,
        all_nodes=nodes,
        all_edges=edges,
        edge_scores=edge_scores,
        depth_map=depth_map,
        scope="full_analysis",
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 8.0, f"10k graph + trajectory took too long: {elapsed:.2f}s"

    ks = [pt["k"] for pt in dyn["polarization_trajectory"]]
    for expected_m in (500, 1000, 2000, 5000, 7500):
        assert expected_m in ks
    assert ks[-1] == len(nodes) - 1

    hotspots = dyn["hotspot_branches"]
    assert 1 <= len(hotspots) <= 5
    for hb in hotspots:
        assert len(hb["node_ids"]) <= 80
        assert hb["node_ids"][0] == root_id


def test_extract_hotspot_branches_connected_and_capped() -> None:
    """Extracted hotspot branch subtrees remain parent-connected and respect max_nodes_per_branch."""
    post, comments = build_megathread_stress_preset(400)
    built = build_vn_cascade_from_thread(post, comments, min_comments=1, min_depth=1, min_vi_ratio=0.0, junk_mode="contract")
    assert built is not None
    nodes, edges, _ = built
    root = next(n for n in nodes if bool(n["is_root"]))
    root_id = str(root["node_id"])
    depth_map = {root_id: 0}
    parent_of = {str(e["target_id"]): str(e["source_id"]) for e in edges}
    for n in nodes:
        nid = str(n["node_id"])
        d = 0
        c = nid
        while c in parent_of and d < 100:
            d += 1
            c = parent_of[c]
        depth_map[nid] = d
    scores = {str(e["edge_id"]): {"s_uv": -0.4, "h_uv": 0.6} for e in edges}
    branches = extract_hotspot_branches(
        root_id=root_id,
        all_nodes=nodes,
        all_edges=edges,
        edge_scores=scores,
        depth_map=depth_map,
        top_k=3,
        max_nodes_per_branch=25,
    )
    assert len(branches) == 3
    for b in branches:
        nset = set(b["node_ids"])
        assert len(nset) <= 25
        for nid in nset:
            if nid != root_id:
                assert parent_of[nid] in nset


def test_fetch_reddit_thread_paginated_backoff_and_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Paginated fetcher retries HTTP 429, deduplicates overlap IDs, and collects > 500 comments."""

    class _FakeResp:
        def __init__(self, status_code: int, payload: Any, headers: dict[str, str] | None = None) -> None:
            self.status_code = status_code
            self._payload = payload
            self.headers = headers or {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(f"HTTP {self.status_code}")

        def json(self) -> Any:
            return self._payload

    calls: list[dict[str, Any]] = []
    rate_limited_once = False

    def _fake_get(url: str, params: dict[str, Any] | None = None, headers: Any = None, timeout: int = 20) -> _FakeResp:
        nonlocal rate_limited_once
        del headers, timeout
        calls.append({"url": url, "params": dict(params or {})})
        if url.endswith("/posts/search"):
            return _FakeResp(
                200,
                {
                    "data": [
                        {
                            "id": "abc1234",
                            "title": "Mega post",
                            "selftext": "Noi dung bai goc.",
                            "author": "op_user",
                            "created_utc": 1763500000,
                        }
                    ]
                },
            )
        if url.endswith("/comments/search"):
            after = (params or {}).get("after")
            if after is None:
                if not rate_limited_once:
                    rate_limited_once = True
                    return _FakeResp(429, {"error": "rate_limit"}, headers={"x-ratelimit-reset": "0.05"})
                items = [
                    {
                        "id": f"c{i}",
                        "parent_id": "t3_abc1234",
                        "body": f"Binh luan so {i} ung ho",
                        "author": f"u{i}",
                        "created_utc": 1763500000 + i,
                    }
                    for i in range(1, 501)
                ]
                return _FakeResp(200, {"data": items})
            # Page 2 includes 5 overlapping duplicate IDs (c496..c500) + 250 fresh comments
            items = [
                {
                    "id": f"c{i}",
                    "parent_id": "t3_abc1234",
                    "body": f"Binh luan so {i} phan doi",
                    "author": f"u{i}",
                    "created_utc": 1763500000 + i,
                }
                for i in range(496, 751)
            ]
            return _FakeResp(200, {"data": items})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", _fake_get)
    progress_events: list[dict[str, Any]] = []
    post, comments = fetch_reddit_thread_paginated(
        "abc1234",
        max_comments=1200,
        progress_cb=progress_events.append,
    )
    assert post["id"] == "abc1234"
    assert len(comments) == 750
    assert len({c["id"] for c in comments}) == 750
    assert rate_limited_once is True
    assert len(progress_events) == 2


def test_load_thread_from_json_and_jsonl_files(tmp_path: Path) -> None:
    """Local JSON and JSONL loader respects max_comments up to 10,000."""
    json_p = tmp_path / "thread.json"
    json_p.write_text(
        json.dumps(
            {
                "id": "loc1",
                "title": "Chu de cuc bo",
                "text": "Noi dung bai viet",
                "comments": [
                    {"id": f"c{i}", "parent_id": "root", "text": f"Binh luan {i}", "author": f"u{i}"}
                    for i in range(1, 25)
                ],
            }
        ),
        encoding="utf-8",
    )
    post, cmts = load_thread_from_file(json_p, max_comments=10)
    assert post["id"] == "loc1"
    assert len(cmts) == 10

    jsonl_p = tmp_path / "thread.jsonl"
    lines = [
        json.dumps({"id": "loc2", "title": "Bai JSONL", "text": "Noi dung"}),
        *[json.dumps({"id": f"x{i}", "parent_id": "root", "text": f"Dong {i}"}) for i in range(1, 15)],
    ]
    jsonl_p.write_text("\n".join(lines), encoding="utf-8")
    post2, cmts2 = load_thread_from_file(jsonl_p, max_comments=12)
    assert post2["id"] == "loc2"
    assert len(cmts2) == 12


def test_predict_live_cascade_stratified_10k_keeps_early_k20() -> None:
    """Stratified 10k scan preserves GNN K=20 isolation, flags OOD > 350, and computes bootstrap CI."""
    post, comments = build_megathread_stress_preset(10000)
    assert len(comments) == 10000
    progress_seen: list[dict[str, Any]] = []
    res = predict_live_cascade(
        post,
        comments,
        k_horizon=20,
        analysis_mode="full",
        max_full_comments=10000,
        scan_strategy="stratified",
        sample_budget=600,
        progress_cb=progress_seen.append,
        topic_backend="lexical",
        allow_untrained=True,
        is_demo=True,
    )
    assert res["gnn_k_horizon"] == 20
    assert res["ood_scale_warning"] is True
    assert OOD_SCALE_THRESHOLD == 350
    assert res["training_max_replies_ref"] == TRAINING_MAX_REPLIES_REF == 345
    ms = res["mega_scan_summary"]
    assert ms["scan_strategy"] == "stratified"
    assert ms["total_raw_comments"] == 10000
    assert ms["kept_comments"] > 9000
    assert 400 <= ms["scored_edges"] <= 900
    assert 0.0 < ms["coverage_ratio"] < 0.25
    ci = ms["bootstrap_ci_95"]
    assert isinstance(ci, list) and len(ci) == 2 and 0.0 <= ci[0] <= ci[1] <= 1.0
    assert len(ms["hotspot_branches"]) >= 1
    assert progress_seen and progress_seen[-1]["processed"] == progress_seen[-1]["total"]


def test_async_scan_jobs_lifecycle_pagination_branches_and_cancel(tmp_path: Path) -> None:
    """Verify full async /api/tier1/scan_jobs HTTP lifecycle, filtering, branch switch, cancel & frontend guards."""
    server = create_server(
        host="127.0.0.1",
        port=0,
        data_dir=tmp_path / "missing_vn",
        ckpt_path=tmp_path / "missing_ckpt.pt",
        ce_dir=tmp_path / "missing_ce",
        paper_dir=PAPER_DIR,
        web_dir=WEB_DIR,
        preload=False,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        # 1. Validation guards
        st_bad, _ = _http_post(
            port,
            "/api/tier1/scan_jobs",
            {"mode": "preset_mega_stress", "max_full_comments": 50000},
        )
        assert st_bad == 400

        st_bad_url, _ = _http_post(
            port,
            "/api/tier1/scan_jobs",
            {"mode": "reddit_url", "reddit_url": "https://reddit.com/../bad"},
        )
        assert st_bad_url == 400

        # 2. Launch 1,200-comment stratified stress job
        st_cr, created = _http_post(
            port,
            "/api/tier1/scan_jobs",
            {
                "mode": "preset_mega_stress",
                "stress_n_comments": 1200,
                "max_full_comments": 1200,
                "scan_strategy": "stratified",
                "sample_budget": 300,
                "junk_mode": "contract",
            },
        )
        assert st_cr == 202
        job_id = created["job_id"]

        final_job: dict[str, Any] = {}
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            st_p, job_state = _http_get(port, f"/api/tier1/scan_jobs/{job_id}")
            assert st_p == 200
            if job_state["status"] in {"completed", "failed", "cancelled"}:
                final_job = job_state
                break
            time.sleep(0.15)
        assert final_job.get("status") == "completed", f"Job did not complete: {final_job}"

        res = final_job["result"]
        assert res["forecast"]["gnn_k_horizon"] == 20
        assert res["forecast"]["ood_scale_warning"] is True
        assert len(res["tree"]["nodes"]) <= _MAX_TREE_RENDER_NODES
        hotspots = res["large_thread_analysis"]["hotspot_branches"]
        assert len(hotspots) >= 1

        # 3. Server-side paginated & filtered comment retrieval
        st_pg, page1 = _http_get(port, f"/api/tier1/scan_jobs/{job_id}/comments?page=1&page_size=50&stance=all")
        assert st_pg == 200
        assert len(page1["items"]) == 50
        assert page1["total_items"] > 1000
        assert page1["total_pages"] >= 20

        st_flt, filt = _http_get(port, f"/api/tier1/scan_jobs/{job_id}/comments?page=1&page_size=50&q=metro")
        assert st_flt == 200
        assert all("metro" in r["text"].lower() for r in filt["items"])

        # 4. Hotspot branch subtree fetch <= 80 nodes
        bid = hotspots[0]["branch_id"]
        st_br, br_data = _http_get(port, f"/api/tier1/scan_jobs/{job_id}/branch/{bid}")
        assert st_br == 200
        assert len(br_data["tree"]["nodes"]) <= 80

        # 5. Cancellation endpoint works cleanly
        st_cr2, created2 = _http_post(
            port,
            "/api/tier1/scan_jobs",
            {
                "mode": "preset_mega_stress",
                "stress_n_comments": 2000,
                "max_full_comments": 2000,
                "scan_strategy": "full",
            },
        )
        assert st_cr2 == 202
        st_can, can_resp = _http_post(port, f"/api/tier1/scan_jobs/{created2['job_id']}/cancel", {})
        assert st_can == 200
        assert can_resp["cancel_requested"] is True
        # Wait briefly so background worker drains cleanly before server shutdown
        for _ in range(40):
            _, st_check = _http_get(port, f"/api/tier1/scan_jobs/{created2['job_id']}")
            if st_check["status"] in {"cancelled", "completed", "failed"}:
                break
            time.sleep(0.1)

        # 6. Frontend static invariants preserved + Mega-Thread controls present
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
        assert 'id="mega-scan-panel"' in html
        assert 'id="mega-comments-section"' in html
        assert 'id="hotspot-branch-bar"' in html
        assert "MAX_SVG_NODES = 120" in js
        assert "innerHTML" not in js
        assert "insertAdjacentHTML" not in js
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

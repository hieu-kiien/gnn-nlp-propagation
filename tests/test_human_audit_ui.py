"""Tests for local blind Vietnamese Gate 1 Human Audit UI server."""

from __future__ import annotations

import csv
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from scripts.audit_vn_50_trees import evaluate_thread_level_audit_csv
from scripts.compute_kappa import evaluate_audit_csv
from scripts.human_audit_ui import create_server


def _write_pair_csv(path: Path) -> None:
    fieldnames = [
        "sample_id",
        "cascade_id",
        "edge_id",
        "root_text",
        "parent_text",
        "reply_text",
        "annotator_1_stance",
        "annotator_2_stance",
        "annotator_1_hostility",
        "annotator_2_hostility",
        "notes",
        "gold_adjudicated_stance",
        "gold_adjudicated_hostility",
    ]
    rows = [
        {
            "sample_id": "audit_001",
            "cascade_id": "vn_1",
            "edge_id": "vn_1_a<-root",
            "root_text": "Bài đăng gốc tiếng Việt (Ế Ứ Ổ Ặ Ằ)",
            "parent_text": "Bài đăng gốc tiếng Việt (Ế Ứ Ổ Ặ Ằ)",
            "reply_text": "Đồng ý hoàn toàn với bạn!",
            "annotator_1_stance": "support",
            "annotator_2_stance": "oppose",
            "annotator_1_hostility": "clean",
            "annotator_2_hostility": "clean",
            "notes": "",
            "gold_adjudicated_stance": "",
            "gold_adjudicated_hostility": "",
        },
        {
            "sample_id": "audit_002",
            "cascade_id": "vn_2",
            "edge_id": "vn_2_b<-a",
            "root_text": "Chủ đề 2",
            "parent_text": "Ý kiến cha",
            "reply_text": "Hỏi thông tin thêm?",
            "annotator_1_stance": "",
            "annotator_2_stance": "neutral",
            "annotator_1_hostility": "",
            "annotator_2_hostility": "clean",
            "notes": "existing note",
            "gold_adjudicated_stance": "",
            "gold_adjudicated_hostility": "",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _write_tree_csv(path: Path) -> None:
    fieldnames = [
        "sample_id",
        "cascade_id",
        "topic_id",
        "target_window",
        "num_replies",
        "num_target_replies",
        "root_text",
        "sample_replies",
        "annotator_1_polar_grade",
        "annotator_2_polar_grade",
        "gold_adjudicated_polar_grade",
        "notes",
    ]
    rows = [
        {
            "sample_id": "tree_001",
            "cascade_id": "vn_t1",
            "topic_id": "t_1",
            "target_window": "future_only (replies #21..#25)",
            "num_replies": "25",
            "num_target_replies": "5",
            "root_text": "Thảo luận cấp cây",
            "sample_replies": "[#1 -> Root] Chuẩn || [#2 -> Parent(\"Sai\")] Cãi nhau to",
            "annotator_1_polar_grade": "1",
            "annotator_2_polar_grade": "3",
            "gold_adjudicated_polar_grade": "",
            "notes": "",
        },
        {
            "sample_id": "tree_002",
            "cascade_id": "vn_t2",
            "topic_id": "t_2",
            "target_window": "future_only (replies #21..#30)",
            "num_replies": "30",
            "num_target_replies": "10",
            "root_text": "Thảo luận 2",
            "sample_replies": "[#1 -> Root] Đồng thuận",
            "annotator_1_polar_grade": "0",
            "annotator_2_polar_grade": "0",
            "gold_adjudicated_polar_grade": "",
            "notes": "",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _http_req(url: str, *, method: str = "GET", payload: dict | None = None) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8")


@pytest.fixture()
def audit_server(tmp_path: Path):
    data_dir = tmp_path / "interim"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_pair_csv(data_dir / "human_audit_vi_300.csv")
    _write_tree_csv(data_dir / "human_audit_vn_50_trees.csv")
    (data_dir / "human_audit_vi_300_machine_key.csv").write_text("secret,1\n", encoding="utf-8")
    (data_dir / ".author_hash_salt.secret").write_text("salt\n", encoding="utf-8")

    server = create_server(data_dir, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base, data_dir
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_refuse_non_local_bind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-local"):
        create_server(tmp_path, host="0.0.0.0", port=0)


def test_allowlist_and_403_security(audit_server) -> None:
    base, _ = audit_server
    for bad_name in (
        "human_audit_vi_300_machine_key.csv",
        ".author_hash_salt.secret",
        "../secret.csv",
        "arbitrary.csv",
    ):
        status, body = _http_req(f"{base}/api/dataset?file={urllib.parse.quote(bad_name)}&role=annotator_1")
        assert status == 403, f"Expected 403 for {bad_name}, got {status}: {body}"

        post_status, _ = _http_req(
            f"{base}/api/save",
            method="POST",
            payload={"file": bad_name, "role": "annotator_1", "sample_id": "audit_001"},
        )
        assert post_status == 403


def test_strict_dual_annotator_blinding_and_gold_disagreements(audit_server) -> None:
    base, _ = audit_server

    s1, raw1 = _http_req(f"{base}/api/dataset?file=human_audit_vi_300.csv&role=annotator_1")
    assert s1 == 200
    d1 = json.loads(raw1)
    assert d1["summary"]["done_for_role"] == 1
    row1 = d1["rows"][0]
    assert row1["stance"] == "support"
    assert row1["hostility"] == "clean"
    assert not any("annotator_2" in k or "gold_" in k for k in row1)

    s2, raw2 = _http_req(f"{base}/api/dataset?file=human_audit_vi_300.csv&role=annotator_2")
    assert s2 == 200
    d2 = json.loads(raw2)
    row2 = d2["rows"][0]
    assert row2["stance"] == "oppose"
    assert not any("annotator_1" in k or "gold_" in k for k in row2)

    sg, rawg = _http_req(f"{base}/api/dataset?file=human_audit_vi_300.csv&role=gold")
    assert sg == 200
    dg = json.loads(rawg)
    assert dg["summary"]["disagreements_count"] == 1
    assert dg["rows"][0]["disagreement"] is True
    assert dg["rows"][1]["disagreement"] is False

    st, rawt = _http_req(f"{base}/api/dataset?file=human_audit_vn_50_trees.csv&role=gold")
    assert st == 200
    dt = json.loads(rawt)
    assert dt["summary"]["disagreements_count"] == 1
    assert dt["summary"]["hard_disagreements_count"] == 1
    assert dt["rows"][0]["disagreement"] is True
    assert dt["rows"][0]["hard_disagreement"] is True
    assert dt["rows"][1]["disagreement"] is False


def test_atomic_save_validation_and_compute_kappa_compatibility(audit_server) -> None:
    base, data_dir = audit_server

    # Reject invalid labels & unknown sample_id with 400
    bad_s, _ = _http_req(
        f"{base}/api/save",
        method="POST",
        payload={
            "file": "human_audit_vi_300.csv",
            "role": "annotator_1",
            "sample_id": "audit_002",
            "stance": "invalid_stance",
            "hostility": "clean",
        },
    )
    assert bad_s == 400

    bad_h, _ = _http_req(
        f"{base}/api/save",
        method="POST",
        payload={
            "file": "human_audit_vi_300.csv",
            "role": "annotator_1",
            "sample_id": "audit_002",
            "stance": "neutral",
            "hostility": "mega_hate",
        },
    )
    assert bad_h == 400

    bad_g, _ = _http_req(
        f"{base}/api/save",
        method="POST",
        payload={
            "file": "human_audit_vn_50_trees.csv",
            "role": "annotator_1",
            "sample_id": "tree_001",
            "polar_grade": "9",
        },
    )
    assert bad_g == 400

    bad_id, _ = _http_req(
        f"{base}/api/save",
        method="POST",
        payload={
            "file": "human_audit_vi_300.csv",
            "role": "annotator_1",
            "sample_id": "missing_999",
            "stance": "support",
            "hostility": "clean",
        },
    )
    assert bad_id == 400

    # Valid save for annotator_1 on audit_002 preserves annotator_2 columns & writes .bak
    ok_p, raw_ok_p = _http_req(
        f"{base}/api/save",
        method="POST",
        payload={
            "file": "human_audit_vi_300.csv",
            "role": "annotator_1",
            "sample_id": "audit_002",
            "stance": "neutral",
            "hostility": "clean",
            "notes": "đã kiểm tra",
        },
    )
    assert ok_p == 200
    assert json.loads(raw_ok_p)["summary"]["done_a1"] == 2
    assert (data_dir / "human_audit_vi_300.csv.bak").is_file()

    # Gold adjudication unlocks strict_adjudication compatibility with compute_kappa.py
    ok_g, _ = _http_req(
        f"{base}/api/save",
        method="POST",
        payload={
            "file": "human_audit_vi_300.csv",
            "role": "gold",
            "sample_id": "audit_001",
            "stance": "support",
            "hostility": "clean",
        },
    )
    assert ok_g == 200

    # Verify utf-8-sig BOM preserved & compute_kappa works cleanly
    pair_csv = data_dir / "human_audit_vi_300.csv"
    assert pair_csv.read_bytes().startswith(b"\xef\xbb\xbf")
    pair_report = evaluate_audit_csv(pair_csv, strict_adjudication=True)
    assert pair_report["stance"]["num_samples"] == 2
    assert pair_report["hostility"]["num_samples"] == 2

    # Tree gold adjudication + audit_vn_50_trees compatibility
    ok_tg, _ = _http_req(
        f"{base}/api/save",
        method="POST",
        payload={
            "file": "human_audit_vn_50_trees.csv",
            "role": "gold",
            "sample_id": "tree_001",
            "polar_grade": "2",
        },
    )
    assert ok_tg == 200
    tree_report = evaluate_thread_level_audit_csv(
        data_dir / "human_audit_vn_50_trees.csv",
        strict_adjudication=True,
    )
    assert tree_report["inter_annotator"]["num_threads"] == 2


def test_html_modern_web_invariants(audit_server) -> None:
    base, _ = audit_server
    status, html = _http_req(f"{base}/")
    assert status == 200
    assert '<dialog id="guide-dialog" closedby="any"' in html
    assert 'role="dialog"' not in html
    assert "aria-modal" not in html
    assert "overscroll-behavior: contain" in html
    assert "dialog[open]" in html
    assert "guideDialog.returnValue = \"\"" in html or "guideDialog.returnValue = ''" in html
    assert "HTMLDialogElement.prototype" in html
    assert "[hidden] {\n      display: none !important;" in html
    assert "event.isComposing" in html
    assert "event.keyCode === 229" in html

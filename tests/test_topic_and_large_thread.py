from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.demo_web_app import create_server
from scripts.predict_live_thread import _format_report_vi, predict_live_cascade
from src.nlp.topic_post_analyzer import (
    DEFAULT_TOPICS_EN,
    DEFAULT_TOPICS_VI,
    analyze_large_thread_dynamics,
    classify_post_topics,
    parse_custom_topics,
)


class _DummyTokenizer:
    """Offline tokenizer mapping keywords to orthogonal slices so embedding similarity works."""

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 256,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        del padding, truncation, max_length, return_tensors
        feats = torch.zeros((len(texts), 3), dtype=torch.float32)
        for i, txt in enumerate(texts):
            low = txt.lower()
            if any(k in low for k in ("ai", "mô hình", "trí tuệ", "thuật toán", "llm")):
                feats[i, 0] = 1.0
            elif any(k in low for k in ("kẹt xe", "giao thông", "metro", "bus")):
                feats[i, 1] = 1.0
            else:
                feats[i, 2] = 1.0
        # Repeat token sequence length=2 so mean-pooling averages cleanly.
        input_ids = feats.unsqueeze(1).repeat(1, 2, 1)
        attention_mask = torch.ones((len(texts), 2), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _DummyEncoder(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(last_hidden_state=input_ids)


def _start_isolated_server(tmp_path: Path):
    server = create_server(
        host="127.0.0.1",
        port=0,
        data_dir=tmp_path / "no_data",
        ckpt_path=tmp_path / "no_gnn.pt",
        ce_dir=tmp_path / "no_ce",
        paper_dir=tmp_path / "no_results",
        web_dir=Path(__file__).resolve().parents[1] / "web_demo",
        preload=False,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def _http_get_text(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return int(resp.status), resp.read().decode("utf-8")


def _http_post_json(url: str, payload: dict) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return int(resp.status), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return int(err.code), json.loads(err.read().decode("utf-8"))


def test_parse_custom_topics_handles_delimiters_and_duplicates():
    assert parse_custom_topics(None) == []
    assert parse_custom_topics("   ; , \n ") == []

    parsed_str = parse_custom_topics(
        "Giao thông đô thị, Giáo dục & AI; Kinh tế số\nGiao thông đô thị"
    )
    labels = [item["label"] for item in parsed_str]
    assert labels == ["Giao thông đô thị", "Giáo dục & AI", "Kinh tế số"]
    assert len({item["id"] for item in parsed_str}) == 3

    parsed_seq = parse_custom_topics(["Y tế công cộng", "Y tế công cộng", "Năng lượng xanh"])
    assert [item["label"] for item in parsed_seq] == ["Y tế công cộng", "Năng lượng xanh"]


def test_classify_post_topics_default_custom_and_encoder_backends():
    res_lex = classify_post_topics(
        "Tranh luận về trí tuệ nhân tạo và sinh viên công nghệ thông tin",
        "Các mô hình AI lớn đang thay đổi cách lập trình viên làm việc mỗi ngày.",
        reply_snippets=("Cần học nền tảng thuật toán thay vì phụ thuộc chatbot.",),
        lang="vi",
        backend="lexical",
        allow_nli_download=False,
    )
    assert res_lex["topic_backend"] == "lexical_ngram_fallback"
    assert res_lex["transformer_topic_loaded"] is False
    assert res_lex["is_custom_taxonomy"] is False
    assert len(res_lex["topics"]) == len(DEFAULT_TOPICS_VI)
    assert res_lex["primary_topic"] in {"Công nghệ & AI", "Giáo dục & Nghề nghiệp"}

    res_auto = classify_post_topics(
        "Tranh luận về trí tuệ nhân tạo và phần mềm AI",
        "Mô hình ngôn ngữ lớn và lập trình phần mềm.",
        lang="vi",
        backend="auto",
        allow_nli_download=False,
    )
    assert res_auto["topic_backend"] in {
        "mdeberta_zero_shot_nli",
        "transformer_cosine_embedding",
        "lexical_ngram_fallback",
    }
    assert res_auto["transformer_topic_loaded"] == (
        res_auto["topic_backend"] != "lexical_ngram_fallback"
    )

    res_en = classify_post_topics(
        "Federal central bank policy debate on inflation and housing costs",
        "Economists disagree on interest rate hikes versus public infrastructure spending.",
        lang="en",
        backend="lexical",
        allow_nli_download=False,
    )
    assert len(res_en["topics"]) == len(DEFAULT_TOPICS_EN)
    assert res_en["primary_topic"] in {"Economy & Finance", "Policy, Law & Governance"}

    custom_res = classify_post_topics(
        "Đề xuất mở rộng tuyến metro và xe buýt nhanh giảm ùn tắc giờ cao điểm",
        "Người dân cần phương tiện công cộng thuận tiện thay vì chỉ cấm xe máy.",
        custom_topics="Bóng đá thể thao, Y tế bệnh viện, Quy hoạch giao thông metro",
        lang="vi",
        backend="lexical",
        allow_nli_download=False,
    )
    assert custom_res["is_custom_taxonomy"] is True
    assert custom_res["primary_topic"] == "Quy hoạch giao thông metro"
    assert len(custom_res["topics"]) == 3

    emb_res = classify_post_topics(
        "Mô hình AI đa phương thức mới",
        "Đánh giá thuật toán xử lý ngôn ngữ.",
        custom_topics=["Trí tuệ nhân tạo AI", "Giao thông metro", "Ẩm thực"],
        lang="vi",
        backend="embedding",
        node_encoder=_DummyEncoder(),
        node_tokenizer=_DummyTokenizer(),
        allow_nli_download=False,
    )
    assert emb_res["topic_backend"] == "transformer_cosine_embedding"
    assert emb_res["transformer_topic_loaded"] is True
    assert emb_res["primary_topic"] == "Trí tuệ nhân tạo AI"


def test_analyze_large_thread_dynamics_120_comments_and_empty_case():
    root = {
        "node_id": "root",
        "author_hash": "author_root",
        "text": "Đề xuất triển khai AI chấm bài tự luận trong trường đại học",
        "created_at_utc": 1_700_000_000.0,
    }
    all_nodes = [root]
    all_edges = []
    edge_scores: dict[str, dict[str, float]] = {}
    depth_map = {"root": 0}
    imp_map = {}

    # Build 120 chronological replies across 3 levels; early mild/supportive, late polarized/hostile.
    for idx in range(1, 121):
        nid = f"c_{idx:03d}"
        if idx <= 20:
            parent = "root"
            depth = 1
            s_uv, h_uv = 0.65, 0.05
            text = f"Đồng ý phương án thử nghiệm trợ giảng AI minh bạch ở bước thứ {idx} trong lớp học."
        elif idx <= 75:
            parent = f"c_{(idx % 20) + 1:03d}"
            depth = 2
            s_uv = 0.72 if idx % 2 == 0 else -0.78
            h_uv = 0.25 if idx % 3 == 0 else 0.10
            text = (
                f"Phản đối mạnh việc giao toàn quyền chấm điểm cho thuật toán thiếu kiểm chứng #{idx}."
                if s_uv < 0
                else f"Hoàn toàn ủng hộ chuẩn hóa rubric để giảm tải cho giảng viên #{idx}."
            )
        else:
            parent = f"c_{21 + (idx % 40):03d}"
            depth = 3
            if idx % 10 == 0:
                text = "hóng"  # Short junk-like comment should be skipped in argument quotes
                s_uv, h_uv = 0.0, 0.1
            else:
                s_uv = -0.88 if idx % 2 == 1 else 0.85
                h_uv = 0.68 if idx % 4 == 0 else 0.22
                text = (
                    f"Quan điểm quá cực đoan và thiếu dữ liệu thực nghiệm ở bình luận phản biện #{idx}."
                    if s_uv < 0
                    else f"Ủng hộ có kiểm soát kết hợp rà soát ngẫu nhiên của hội đồng chuyên môn #{idx}."
                )

        eid = f"{parent}->{nid}"
        all_nodes.append(
            {
                "node_id": nid,
                "parent_id": parent,
                "author_hash": f"user_{idx % 48}",
                "text": text,
                "depth": depth,
                "created_at_utc": float(1_700_000_000 + idx * 60),
            }
        )
        all_edges.append({"edge_id": eid, "source_id": parent, "target_id": nid})
        edge_scores[eid] = {"s_uv": s_uv, "h_uv": h_uv}
        depth_map[nid] = depth
        if idx in (25, 80, 96):
            imp_map[nid] = 0.9

    topic_info = classify_post_topics(
        "AI trong giáo dục đại học",
        root["text"],
        lang="vi",
        backend="lexical",
        allow_nli_download=False,
    )
    large = analyze_large_thread_dynamics(
        root_node=root,
        all_nodes=all_nodes,
        all_edges=all_edges,
        edge_scores=edge_scores,
        depth_map=depth_map,
        topic_analysis=topic_info,
        trigger_importance_by_target=imp_map,
        k_early_ref=20,
        lang="vi",
        scope="full_analysis",
    )
    assert large["scope"] == "full_analysis"
    assert large["total_replies_analyzed"] == 120
    assert large["scale_tier"] == "large"
    assert large["max_depth"] == 3
    assert large["unique_authors_total"] >= 40
    assert 0.0 <= large["full_y_polar"] <= 1.0
    assert large["evolution_trend"] == "escalating"
    assert large["polar_delta"] > 0.05
    ks = [pt["k"] for pt in large["polarization_trajectory"]]
    assert ks == sorted(ks) and ks[-1] == 120 and 20 in ks
    sem = large["semantic_understanding"]
    assert "AI" in sem["summary_vi"] or "Giáo dục" in sem["summary_vi"]
    assert len(sem["top_pro_arguments"]) >= 1
    assert len(sem["top_anti_arguments"]) >= 1
    assert all(arg["text"] != "hóng" for arg in sem["top_pro_arguments"] + sem["top_anti_arguments"])
    assert len(sem["flashpoint_comments"]) >= 1

    empty = analyze_large_thread_dynamics(
        root_node=root,
        all_nodes=[root],
        all_edges=[],
        edge_scores={},
        depth_map={"root": 0},
        topic_analysis=topic_info,
        scope="full_analysis",
    )
    assert empty["total_replies_analyzed"] == 0
    assert empty["evolution_trend"] == "single_window"
    assert empty["polarization_trajectory"] == []


def test_lexical_custom_topic_ai_no_false_substring_match_and_en_quantiles():
    """Step 1 regressions: 'AI' must not match 'bài'/'phải'/'sai'; EN uses PHEME quantiles; prefix orientations stay causal."""
    res_sub = classify_post_topics(
        "Bài viết này phải xem lại vì có nhiều lỗi sai về tài chính ngân hàng",
        "Hỏi về lãi suất vay mua nhà và quy trình giải ngân.",
        custom_topics="AI, Tài chính ngân hàng",
        lang="vi",
        backend="lexical",
        allow_nli_download=False,
    )
    scores_by_lbl = {t["label"]: t["score"] for t in res_sub["topics"]}
    assert res_sub["primary_topic"] == "Tài chính ngân hàng"
    assert scores_by_lbl["Tài chính ngân hàng"] > scores_by_lbl["AI"]

    # English thread uses PHEME Train quantiles and produces summary_en
    root_en = {
        "node_id": "root",
        "author_hash": "u0",
        "text": "Breaking report on downtown security lockdown",
        "created_at_utc": 1_700_000_000.0,
    }
    nodes_en = [root_en]
    edges_en = []
    scores_en: dict[str, dict[str, float]] = {}
    depths_en = {"root": 0}
    for i in range(1, 26):
        nid = f"e_{i:02d}"
        # c1..c20: only child c1 opposes root; c2..c20 oppose c1 -> causal orientation under root->c1->c_i
        pid = "root" if i == 1 else ("e_01" if i <= 20 else "root")
        s_uv = -0.8 if i <= 20 else 0.7
        nodes_en.append(
            {
                "node_id": nid,
                "parent_id": pid,
                "author_hash": f"eu_{i}",
                "text": f"Comment body with enough words number {i} for analysis.",
                "created_at_utc": float(1_700_000_000 + i * 30),
            }
        )
        eid = f"{pid}->{nid}"
        edges_en.append({"edge_id": eid, "source_id": pid, "target_id": nid})
        scores_en[eid] = {"s_uv": s_uv, "h_uv": 0.05}
        depths_en[nid] = 1 if pid == "root" else 2

    dyn_en = analyze_large_thread_dynamics(
        root_node=root_en,
        all_nodes=nodes_en,
        all_edges=edges_en,
        edge_scores=scores_en,
        depth_map=depths_en,
        topic_analysis={"primary_topic": "Breaking News & Rumors", "primary_confidence": 0.8},
        k_early_ref=15,
        lang="en",
        scope="full_analysis",
    )
    assert dyn_en["lang"] == "en"
    assert dyn_en["early_ref_k"] == 20
    assert "at K=20" in dyn_en["semantic_understanding"]["summary_en"]
    assert "ở K=20" in dyn_en["semantic_understanding"]["summary_vi"]
    assert dyn_en["semantic_understanding"]["summary"] == dyn_en["semantic_understanding"]["summary_en"]


def test_predict_live_cascade_early_vs_full_mode_and_report(tmp_path: Path):
    post = {
        "id": "post_large_test",
        "title": "Kế hoạch đầu tư đường sắt đô thị và vé tháng cho người đi làm",
        "selftext": "Thảo luận về chi phí và hiệu quả của tuyến metro nội đô mới.",
        "author": "u_root",
        "created_utc": 1_700_000_000,
    }
    comments = []
    for i in range(1, 36):
        comments.append(
            {
                "fullname": f"t1_r{i}",
                "link_id": "t3_post_large_test",
                "parent_id": "t3_post_large_test" if i <= 10 else f"t1_r{(i % 10) + 1}",
                "author": f"u_{i}",
                "body": (
                    f"Ủng hộ đầu tư giao thông công cộng giảm kẹt xe lâu dài ở tuyến số {i}."
                    if i % 2 == 0
                    else f"Phản đối vì lo ngại đội vốn kéo dài tiến độ thi công đoạn thứ {i}."
                ),
                "created_utc": 1_700_000_000 + i * 60,
            }
        )

    torch.manual_seed(1234)
    early_res = predict_live_cascade(
        post,
        comments,
        k_horizon=20,
        analysis_mode="early",
        custom_topics="Giao thông công cộng, Giải trí Showbiz, Y tế",
        topic_backend="lexical",
        cross_encoder_ckpt=None,
        gnn_ckpt=None,
        allow_untrained=True,
        is_demo=True,
    )
    assert early_res["gnn_k_horizon"] == 20
    assert early_res["forecast_scope"] == "early_k_in_distribution"
    assert early_res["k_observed"] == 20
    assert len(early_res["kept_nodes"]) == 21
    assert early_res["large_thread_analysis"]["scope"] == "early_post_hoc"
    assert early_res["large_thread_analysis"]["total_replies_analyzed"] == 35
    assert early_res["topic_analysis"]["primary_topic"] == "Giao thông công cộng"

    torch.manual_seed(1234)
    full_res = predict_live_cascade(
        post,
        comments,
        k_horizon=20,
        analysis_mode="full",
        max_full_comments=100,
        custom_topics="Giao thông công cộng, Giải trí Showbiz, Y tế",
        topic_backend="lexical",
        cross_encoder_ckpt=None,
        gnn_ckpt=None,
        allow_untrained=True,
        is_demo=True,
    )
    assert full_res["gnn_k_horizon"] == 20
    assert full_res["forecast_scope"] == "early_k_in_distribution"
    assert full_res["k_observed"] == 35
    assert len(full_res["kept_nodes"]) == 36
    assert full_res["actual_post_hoc_comparison"]["has_future_replies"] is True
    assert full_res["predicted_total_replies"] == pytest.approx(
        early_res["predicted_total_replies"], abs=1e-4
    )
    assert full_res["predicted_polarization_score"] == pytest.approx(
        early_res["predicted_polarization_score"], abs=1e-4
    )
    assert full_res["large_thread_analysis"]["scope"] == "full_analysis"
    assert full_res["large_thread_analysis"]["total_replies_analyzed"] == 35
    assert len(full_res["large_thread_analysis"]["polarization_trajectory"]) >= 2

    report_vi = _format_report_vi(full_res)
    assert "[6] PHAN LOAI CHU DE & HIEU BAI LON (AI):" in report_vi
    assert "Giao thông công cộng" in report_vi


def test_demo_web_server_large_preset_and_classify_topic_endpoint(tmp_path: Path):
    server, thread, base_url = _start_isolated_server(tmp_path)
    try:
        status_html, html_body = _http_get_text(f"{base_url}/")
        assert status_html == 200
        for marker in (
            'id="btn-preset-large"',
            'name="analysis_mode"',
            'id="sb-custom-topics"',
            'id="ai-understanding-card"',
            'id="topic-reclassify-form"',
            'id="trajectory-svg"',
        ):
            assert marker in html_body

        st_sb, sb_data = _http_post_json(
            f"{base_url}/api/tier1/sandbox",
            {
                "mode": "preset_large_1",
                "junk_mode": "contract",
                "analysis_mode": "full",
                "k_horizon": 20,
                "max_full_comments": 500,
                "custom_topics": "Giáo dục đại học, Bóng đá, Ẩm thực",
            },
        )
        assert st_sb == 200
        assert sb_data["forecast"]["gnn_k_horizon"] == 20
        assert sb_data["forecast"]["forecast_scope"] == "early_k_in_distribution"
        assert sb_data["forecast"]["has_future_replies"] is True
        assert sb_data["junk_filter_stats"]["kept_comments"] > 20
        assert sb_data["topic_analysis"]["is_custom_taxonomy"] is True
        assert sb_data["topic_analysis"]["primary_topic"] == "Giáo dục đại học"
        assert sb_data["large_thread_analysis"]["scope"] == "full_analysis"
        assert len(sb_data["large_thread_analysis"]["polarization_trajectory"]) >= 2

        st_topic, topic_resp = _http_post_json(
            f"{base_url}/api/tier1/classify_topic",
            {
                "title": "Xây dựng tuyến metro số 2 và giảm xe cá nhân",
                "body": "Đề xuất mở rộng hạ tầng giao thông công cộng.",
                "reply_snippets": ["Tăng chuyến xe buýt kết nối ga metro."],
                "custom_topics": "Giao thông đô thị, Thời trang, Phim ảnh",
                "lang": "vi",
            },
        )
        assert st_topic == 200
        topic_obj = topic_resp.get("topic_analysis") or topic_resp
        assert topic_obj["is_custom_taxonomy"] is True
        assert topic_obj["primary_topic"] == "Giao thông đô thị"
        assert len(topic_obj["topics"]) == 3

        st_body_only, body_only_resp = _http_post_json(
            f"{base_url}/api/tier1/classify_topic",
            {"body": "Đề xuất mở rộng tuyến metro giảm ùn tắc", "lang": "vi"},
        )
        assert st_body_only == 200
        body_topic_obj = body_only_resp.get("topic_analysis") or body_only_resp
        assert body_topic_obj["primary_topic"]

        st_bad, _ = _http_post_json(
            f"{base_url}/api/tier1/classify_topic",
            {"title": "", "body": ""},
        )
        assert st_bad == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_get_cached_nli_model_respects_hf_home(tmp_path: Path, monkeypatch):
    import sys
    import types
    import src.nlp.topic_post_analyzer as tpa

    model_id = "org/fake-nli-model"
    snap = tmp_path / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots" / "rev1"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(tpa, "_NLI_CACHE", {})

    seen_calls: list[tuple[str, bool]] = []

    class _DummyCfg:
        label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

    class _DummyModel:
        config = _DummyCfg()

        def eval(self):
            return self

        def parameters(self):
            return []

    class _FakeTok:
        @classmethod
        def from_pretrained(cls, name, local_files_only=False):
            seen_calls.append(("tok", bool(local_files_only)))
            return object()

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, name, local_files_only=False):
            seen_calls.append(("model", bool(local_files_only)))
            return _DummyModel()

    fake_tf = types.SimpleNamespace(
        AutoTokenizer=_FakeTok,
        AutoModelForSequenceClassification=_FakeModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    bundle = tpa._get_cached_nli_model(model_id, allow_download=False)
    assert bundle is not None
    assert seen_calls == [("tok", True), ("model", True)]
    assert bundle[2:] == (0, 2)


def test_score_topics_with_nli_uses_concise_short_label_hypotheses():
    import src.nlp.topic_post_analyzer as tpa

    captured: dict[str, Any] = {}

    class _FakeTok:
        def __call__(self, premises, hypotheses, **kwargs):
            captured["premises"] = list(premises)
            captured["hypotheses"] = list(hypotheses)
            captured["max_length"] = kwargs.get("max_length")
            n = len(hypotheses)
            return {"input_ids": torch.ones((n, 4), dtype=torch.long)}

    class _FakeModel:
        def to(self, dev):
            return self

        def __call__(self, **enc):
            n = enc["input_ids"].shape[0]
            logits = torch.tensor([[3.0, 0.0, -2.0] if i == 0 else [-2.0, 0.0, 2.0] for i in range(n)])
            return type("Out", (), {"logits": logits})()

    cands = [
        {"id": "t1", "label": "Giao thông đô thị", "description": "mạng xã hội ùn tắc đường sắt dài dòng"},
        {"id": "t2", "label": "Tài chính", "description": "lãi suất ngân hàng dài dòng"},
    ]
    sc_vi, _ = tpa._score_topics_with_nli(
        "Đề xuất mở rộng tuyến metro.",
        cands,
        nli_bundle=(_FakeModel(), _FakeTok(), 0, 2),
        lang="vi",
        device=torch.device("cpu"),
    )
    assert sc_vi[0] > sc_vi[1]
    assert captured["hypotheses"] == [
        "Văn bản này nói về Giao thông đô thị.",
        "Văn bản này nói về Tài chính.",
    ]
    assert captured["max_length"] == 192

    cands_en = [
        {"id": "e1", "label": "Paris Satirical Newspaper Attack", "description": "long keyword bag"},
        {"id": "e2", "label": "Australian Cafe Hostage Standoff", "description": "another bag"},
    ]
    tpa._score_topics_with_nli(
        "Gunman attack in Paris.",
        cands_en,
        nli_bundle=(_FakeModel(), _FakeTok(), 0, 2),
        lang="en",
        device=torch.device("cpu"),
    )
    assert captured["hypotheses"] == [
        "This text is about Paris Satirical Newspaper Attack.",
        "This text is about Australian Cafe Hostage Standoff.",
    ]

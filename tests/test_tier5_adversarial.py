"""Tier 5 white-box adversarial regression suite covering Challenger R1 & R2 findings."""

from __future__ import annotations

import http.client
import json
import math
import subprocess
import sys
import threading
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import scripts.predict_live_thread as plt_mod
from scripts.analyze_mechanisms import (
    _clone_cascade_with_edge_attr,
    _resolve_edge_depths,
)
from scripts.demo_web_app import DemoBackend, create_server
from scripts.export_dataset_stats import compute_early_k_persistence_metrics
from scripts.predict_live_thread import _make_demo_offline_thread, predict_live_cascade
from scripts.run_macro_signed_network import (
    LINK_PRED_METRICS,
    build_link_markdown_table,
    export_link_latex_table,
)
from src.data.arctic_shift_vn import (
    build_vn_cascade_from_thread,
    contains_vietnamese_chars,
    resolve_contracted_parent_id,
)
from src.data.pyg_dataset import CascadeData
from src.data.schema import EDGES_ARROW_SCHEMA, NODES_ARROW_SCHEMA, save_cascade_parquet
from src.data.user_network import build_temporal_signed_link_dataset
from src.models.sota_cascades import (
    _get_early_polar_scalar,
    extract_cascade_summary_features,
)
from src.nlp.discourse_labels import _heuristic_behavior_fallback
from src.nlp.extract_features import validate_feature_tensors
from src.training.auto_loop import run_self_auditing_validation_loop
from src.training.metrics import _safe_r2, _safe_spearman, evaluate_predictions
from src.training.reporting import (
    MAIN_METRICS,
    _best_model_per_metric,
    build_markdown_table,
    export_latex_table,
    format_mean_std,
    serialize_benchmark_json,
    significance_marker,
)
from src.training.trainer import (
    _boot_pvalue_from_diffs,
    _compute_sample_level_bootstrap_pvalues,
    compute_cluster_paired_tests,
    compute_paired_bootstrap_pvalue,
    compute_paired_ttest,
    compute_paired_wilcoxon,
    holm_bonferroni_adjust,
    select_best_comparator,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_tied_signed_corpus(data_dir: Path, timestamps: list[float]) -> None:
    """Create minimal valid parquet + nlp_features.pt corpus with controlled edge timestamps."""
    data_dir.mkdir(parents=True, exist_ok=True)
    nodes_rows: list[dict[str, Any]] = []
    edges_rows: list[dict[str, Any]] = []
    node_id_to_idx: dict[str, int] = {}
    edge_stance: dict[str, float] = {}
    edge_hostility: dict[str, float] = {}
    teacher_edge_stance: dict[str, float] = {}
    teacher_edge_hostility: dict[str, float] = {}

    users = [f"u_{i}" for i in range(6)]
    node_idx = 0
    for idx, ts in enumerate(timestamps):
        cid = f"cas_{idx}"
        tid = f"topic_{idx % 2}"
        r_nid = f"{cid}_r"
        c_nid = f"{cid}_c"
        eid = f"{cid}_e"
        u_src = users[idx % len(users)]
        u_dst = users[(idx + 1) % len(users)]
        dt_r = datetime.fromtimestamp(ts - 1.0, tz=timezone.utc)
        dt_c = datetime.fromtimestamp(ts, tz=timezone.utc)
        nodes_rows.extend(
            [
                {
                    "node_id": r_nid,
                    "cascade_id": cid,
                    "topic_id": tid,
                    "parent_id": None,
                    "is_root": True,
                    "text": f"Root {idx}",
                    "created_at_utc": dt_r,
                    "lang": "vi",
                    "source_dataset": "tier5_syn",
                    "user_id_hash": u_dst,
                    "rumor_label": None,
                },
                {
                    "node_id": c_nid,
                    "cascade_id": cid,
                    "topic_id": tid,
                    "parent_id": r_nid,
                    "is_root": False,
                    "text": f"Reply {idx}",
                    "created_at_utc": dt_c,
                    "lang": "vi",
                    "source_dataset": "tier5_syn",
                    "user_id_hash": u_src,
                    "rumor_label": None,
                },
            ]
        )
        node_id_to_idx[r_nid] = node_idx
        node_id_to_idx[c_nid] = node_idx + 1
        node_idx += 2

        edges_rows.append(
            {
                "edge_id": eid,
                "cascade_id": cid,
                "source_id": r_nid,
                "target_id": c_nid,
                "created_at_utc": dt_c,
                "time_delay_sec": 1.0,
                "edge_type": "reply",
            }
        )
        s_val = 0.8 if idx % 2 == 0 else -0.8
        h_val = 0.05 if idx % 2 == 0 else 0.75
        edge_stance[eid] = s_val
        edge_hostility[eid] = h_val
        teacher_edge_stance[eid] = s_val
        teacher_edge_hostility[eid] = h_val

    nodes_tbl = pa.Table.from_pylist(nodes_rows, schema=NODES_ARROW_SCHEMA)
    edges_tbl = pa.Table.from_pylist(edges_rows, schema=EDGES_ARROW_SCHEMA)
    save_cascade_parquet(nodes_tbl, edges_tbl, data_dir, validate=False)

    torch.save(
        {
            "node_embeddings": torch.ones((node_idx, 16), dtype=torch.float32),
            "node_id_to_idx": node_id_to_idx,
            "edge_stance": edge_stance,
            "edge_hostility": edge_hostility,
            "teacher_edge_stance": teacher_edge_stance,
            "teacher_edge_hostility": teacher_edge_hostility,
        },
        data_dir / "nlp_features.pt",
    )


def test_tier5_temporal_link_tied_timestamps_nonempty_val(tmp_path: Path) -> None:
    """Fix #1: Tied timestamps must not collapse val split to 0 and must warn when strict causality breaks."""
    ts_partial_tie = [1000.0] * 20 + [2020.0, 2021.0, 2022.0, 2023.0]
    dir_partial = tmp_path / "partial_tie"
    _write_tied_signed_corpus(dir_partial, ts_partial_tie)

    ds_partial = build_temporal_signed_link_dataset(
        dir_partial, train_ratio=0.70, val_ratio=0.15
    )
    n_tr = int(ds_partial["train"]["pairs"].size(1))
    n_va = int(ds_partial["val"]["pairs"].size(1))
    n_te = int(ds_partial["test"]["pairs"].size(1))
    assert n_va >= 1
    assert n_te >= 1
    assert n_tr + n_va + n_te == 24
    assert float(ds_partial["train"]["ts"].max().item()) < float(ds_partial["val"]["ts"].min().item())
    assert float(ds_partial["val"]["ts"].max().item()) < float(ds_partial["test"]["ts"].min().item())

    ts_all_tie = [1000.0] * 24
    dir_all = tmp_path / "all_tie"
    _write_tied_signed_corpus(dir_all, ts_all_tie)
    with pytest.warns(UserWarning, match="Tied edge timestamps"):
        ds_all = build_temporal_signed_link_dataset(
            dir_all, train_ratio=0.70, val_ratio=0.15
        )
    assert int(ds_all["train"]["pairs"].size(1)) == 22
    assert int(ds_all["val"]["pairs"].size(1)) == 1
    assert int(ds_all["test"]["pairs"].size(1)) == 1


def test_tier5_orphan_flag_root_prune_and_contract_preserve_subtree() -> None:
    """Fix #2: orphan_mode='flag_root' keeps orphans in prune/contract and preserves intact c3 -> c2 subtrees."""
    post = {
        "id": "p1",
        "title": "Thảo luận giao thông",
        "selftext": "Mọi người thấy sao về quy định mới?",
        "author": "u_root",
        "created_utc": 100.0,
        "subreddit": "vozforums",
    }
    comments = [
        {
            "fullname": "t1_c1",
            "link_id": "t3_p1",
            "parent_id": "t3_p1",
            "author": "u1",
            "text": "Tôi đồng ý với quy định mới này.",
            "created_utc": 110.0,
        },
        {
            "fullname": "t1_c2",
            "link_id": "t3_p1",
            "parent_id": "t1_missing",
            "author": "u2",
            "text": "Tuy nhiên vẫn cần xem xét kỹ hơn.",
            "created_utc": 120.0,
        },
        {
            "fullname": "t1_c3",
            "link_id": "t3_p1",
            "parent_id": "t1_c2",
            "author": "u3",
            "text": "Đúng vậy, ý kiến của bạn rất hợp lý.",
            "created_utc": 130.0,
        },
    ]

    for mode in ("prune", "contract"):
        stats_out: dict[str, Any] = {}
        built = build_vn_cascade_from_thread(
            post,
            comments,
            min_comments=1,
            min_depth=1,
            min_vi_ratio=0.0,
            junk_mode=mode,
            orphan_mode="flag_root",
            stats_out=stats_out,
        )
        assert built is not None, f"Cascade dropped unexpectedly in junk_mode={mode}"
        nodes, edges, _ = built
        by_id = {r["node_id"]: r for r in nodes}
        assert set(by_id.keys()) == {"vn_p1_t3_p1", "vn_p1_t1_c1", "vn_p1_t1_c2", "vn_p1_t1_c3"}
        assert by_id["vn_p1_t1_c2"]["parent_id"] == "vn_p1_t3_p1"
        assert by_id["vn_p1_t1_c3"]["parent_id"] == "vn_p1_t1_c2"
        assert any(
            e["source_id"] == "vn_p1_t1_c2" and e["target_id"] == "vn_p1_t1_c3"
            for e in edges
        )
        if mode == "contract":
            risk_targets = {
                m["target_id"]
                for m in stats_out["relinked_edge_metadata"]
                if m["stance_inversion_risk"]
            }
            assert "vn_p1_t1_c3" not in risk_targets

    # Direct helper check: drop_truncated_orphans=False preserves c3 -> c2, True drops both
    orphans_keep: set[str] = set()
    assert (
        resolve_contracted_parent_id(
            "t1_c3",
            {"t1_c2": "t1_missing", "t1_c3": "t1_c2"},
            {"t1_c2", "t1_c3"},
            "t3_p1",
            truncated_orphans_out=orphans_keep,
            drop_truncated_orphans=False,
        )
        == "t1_c2"
    )
    orphans_drop: set[str] = set()
    assert (
        resolve_contracted_parent_id(
            "t1_c3",
            {"t1_c2": "t1_missing", "t1_c3": "t1_c2"},
            {"t1_c2", "t1_c3"},
            "t3_p1",
            truncated_orphans_out=orphans_drop,
            drop_truncated_orphans=True,
        )
        is None
    )


def test_tier5_heuristic_negation_substring_and_punctuation_traps() -> None:
    """Fix #3: Negated VN/EN support, 'chuẩn bị' substring, and pure punctuation do not misclassify."""
    for opp in (
        "Không hợp lý",
        "Không chính xác",
        "Chưa chuẩn",
        "Không chuẩn",
        "I don't agree",
        "I do not agree with your logic",
        "That is not correct",
    ):
        res = _heuristic_behavior_fallback(opp)
        assert res["label"] != "support", f"Negated phrase misclassified as support: {opp!r} -> {res}"
        assert res["s_uv"] <= -0.1, f"Expected negative stance for {opp!r}, got {res}"

    for neutral_phrase in ("Chuẩn bị đi làm", "Mình đang chuẩn bị", "??? !!! ...", "!!!"):
        res = _heuristic_behavior_fallback(neutral_phrase)
        assert res["label"] == "neutral", f"Expected neutral for {neutral_phrase!r}, got {res}"

    assert _heuristic_behavior_fallback("Chuẩn bác")["label"] == "support"
    assert _heuristic_behavior_fallback("I agree")["label"] == "support"
    assert _heuristic_behavior_fallback("Nói vớ vẩn!!!")["label"] == "hostile_conflict"


def test_tier5_paired_ttest_zero_variance_nonzero_diff_not_fake_zero() -> None:
    """Fix #4: Zero-variance non-zero diffs must not emit RuntimeWarning or return fake p=0.0."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p1 = compute_paired_ttest([0.5, 0.5, 0.5, 0.5, 0.5], [0.4, 0.4, 0.4, 0.4, 0.4])
        p2 = compute_paired_ttest([0.5, float("nan"), 0.6], [0.4, 0.5, 0.5])
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert math.isnan(p1) or p1 >= 0.05
    assert math.isnan(p2) or p2 >= 0.05
    assert p1 != 0.0 and p2 != 0.0
    assert compute_paired_ttest([0.5, 0.5], [0.5, 0.5]) == 1.0


def test_tier5_auto_loop_predict_uncertainty_uses_nonzero_lambda(tmp_path: Path) -> None:
    """Fix #5: Default candidates must not zero out lambda_uncertainty when predict_uncertainty=True."""
    graphs: list[CascadeData] = []
    for i in range(6):
        x = torch.randn((4, 16), dtype=torch.float32)
        struct_x = torch.zeros((4, 10), dtype=torch.float32)
        ei_td = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long)
        ei_bu = ei_td[[1, 0], :]
        ea = torch.tensor([[0.5, 0.1], [-0.4, 0.2], [0.3, 0.1]], dtype=torch.float32)
        graphs.append(
            CascadeData(
                x=x,
                struct_x=struct_x,
                edge_index=ei_td,
                edge_index_td=ei_td,
                edge_index_bu=ei_bu,
                edge_attr=ea,
                cascade_stats=torch.zeros((1, 8), dtype=torch.float32),
                y_size=torch.tensor([1.5 + 0.1 * i], dtype=torch.float32),
                y_polar=torch.tensor([0.2 + 0.05 * i], dtype=torch.float32),
                y_rumor=torch.tensor([i % 4], dtype=torch.long),
                early_y_polar=torch.tensor([0.3], dtype=torch.float32),
                root_index=torch.tensor([0], dtype=torch.long),
                cascade_id=f"c_{i}",
                topic_id=f"t_{i % 2}",
                node_texts=["root", "r1", "r2", "r3"],
            )
        )
    tr_loader = DataLoader(graphs[:4], batch_size=2, shuffle=True)
    va_loader = DataLoader(graphs[4:], batch_size=2, shuffle=False)

    res_unc = run_self_auditing_validation_loop(
        tr_loader,
        va_loader,
        in_dim=16,
        hidden_dim=16,
        edge_dim=2,
        predict_uncertainty=True,
        candidates=None,
        epochs=1,
        patience=1,
        seed=42,
        output_dir=tmp_path / "loop_unc",
        device="cpu",
    )
    assert res_unc["best_config"]["lambda_uncertainty"] > 0.0
    assert res_unc["regime_diagnostics"]["overall_val"]["mean_confidence"] is not None

    res_det = run_self_auditing_validation_loop(
        tr_loader,
        va_loader,
        in_dim=16,
        hidden_dim=16,
        edge_dim=2,
        predict_uncertainty=False,
        candidates=None,
        epochs=1,
        patience=1,
        seed=42,
        output_dir=tmp_path / "loop_det",
        device="cpu",
    )
    assert res_det["best_config"]["lambda_uncertainty"] == 0.0


def test_tier5_live_predictor_rejects_nonpositive_k_and_untrained_url_preflight() -> None:
    """Fix #6: k_horizon < 1 raises ValueError; CLI --url without checkpoint fails before HTTP fetch."""
    post, comments = _make_demo_offline_thread()
    for bad_k in (0, -1, -5):
        with pytest.raises(ValueError, match="k_horizon"):
            predict_live_cascade(post, comments, k_horizon=bad_k, is_demo=True)

    with (
        patch.object(plt_mod, "_fetch_reddit_thread", side_effect=AssertionError("Network called before preflight")),
        patch.object(
            sys,
            "argv",
            ["predict_live_thread.py", "--url", "https://reddit.com/r/vozforums/comments/abc123/test"],
        ),
        pytest.raises(RuntimeError, match="checkpoint"),
    ):
        plt_mod.main()

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "predict_live_thread.py"),
            "--demo-offline",
            "--k-horizon",
            "-1",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode != 0


def test_tier5_validate_feature_tensors_checks_teacher_dicts() -> None:
    """Fix #7: validate_feature_tensors checks finiteness and bounds on teacher_* feature dicts."""
    base_payload = {
        "node_embeddings": torch.ones((2, 768), dtype=torch.float32),
        "edge_attr": torch.tensor([[0.2, 0.1]], dtype=torch.float32),
        "edge_stance": {"e1": 0.2},
        "edge_hostility": {"e1": 0.1},
        "node_root_stance": {"n2": 0.2},
        "teacher_edge_stance": {"e1": 0.3},
        "teacher_edge_hostility": {"e1": 0.2},
        "teacher_node_root_stance": {"n2": 0.3},
    }
    validate_feature_tensors(base_payload)

    for bad_patch in (
        {"teacher_edge_stance": {"e1": float("nan")}},
        {"teacher_edge_stance": {"e1": 1.5}},
        {"teacher_edge_hostility": {"e1": 5.0}},
        {"teacher_edge_irrationality": {"e1": -0.2}},
        {"teacher_edge_constructive": {"e1": float("inf")}},
        {"teacher_node_root_stance": {"n1": float("inf")}},
        {"teacher_node_root_stance": {"n1": -1.5}},
    ):
        bad = dict(base_payload, **bad_patch)
        with pytest.raises(ValueError):
            validate_feature_tensors(bad)


def test_tier5_demo_sandbox_rejects_nan_and_overflow_delay_with_400(tmp_path: Path) -> None:
    """Fix #8: Custom sandbox rejects NaN, Inf, negative, and overflow time_delay_sec with HTTP 400."""
    server = create_server(
        host="127.0.0.1",
        port=0,
        data_dir=tmp_path / "absent_data",
        ckpt_path=tmp_path / "absent.pt",
        ce_dir=tmp_path / "absent_ce",
        paper_dir=REPO_ROOT / "results",
        web_dir=REPO_ROOT / "web_demo",
        preload=False,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    def _post_raw(raw_json: str) -> int:
        raw = raw_json.encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
        try:
            conn.request(
                "POST",
                "/api/tier1/sandbox",
                body=raw,
                headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
            )
            resp = conn.getresponse()
            resp.read()
            return resp.status
        finally:
            conn.close()

    try:
        for bad_delay in ("1e20", "NaN", "Infinity", "-5.0", "true"):
            raw_body = (
                '{"mode":"custom","junk_mode":"contract","root_title":"Test","root_text":"Nội dung",'
                f'"comments":[{{"id":"c1","parent_id":"root","text":"Tôi đồng ý","time_delay_sec":{bad_delay}}}]}}'
            )
            assert _post_raw(raw_body) == 400, f"Expected 400 for time_delay_sec={bad_delay}"

        ok_body = json.dumps(
            {
                "mode": "custom",
                "junk_mode": "contract",
                "root_title": "Test",
                "root_text": "Nội dung",
                "comments": [
                    {"id": "c1", "parent_id": "root", "text": "Tôi đồng ý", "time_delay_sec": 45.0}
                ],
            }
        )
        assert _post_raw(ok_body) == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_tier5_nfd_vn_chars_and_body_fallback() -> None:
    """Fix #9 & #10: NFD Vietnamese chars are detected and build_vn_cascade_from_thread accepts 'body' fallback."""
    nfd_text = unicodedata.normalize("NFD", "Việt Nam cộng hòa")
    assert contains_vietnamese_chars(nfd_text) is True

    post = {
        "id": "p_body",
        "title": "Thảo luận mới",
        "selftext": "Nội dung bài viết tiếng Việt",
        "author": "u0",
        "created_utc": 100.0,
        "subreddit": "vozforums",
    }
    comments = [
        {
            "fullname": "t1_c1",
            "parent_id": "t3_p_body",
            "author": "u1",
            "body": "Ý kiến này rất hợp lý và đáng cân nhắc.",
            "created_utc": 105.0,
        }
    ]
    built = build_vn_cascade_from_thread(
        post, comments, min_comments=1, min_depth=1, min_vi_ratio=0.0
    )
    assert built is not None
    nodes, _, _ = built
    assert len(nodes) == 2
    assert any("rất hợp lý" in r["text"] for r in nodes)


def test_tier5_web_demo_app_js_signed_delta_pp() -> None:
    """Fix #11: web_demo/app.js must not unconditionally prepend '+' to negative deltaPp."""
    app_js = (REPO_ROOT / "web_demo" / "app.js").read_text(encoding="utf-8")
    assert "`+${deltaPp.toFixed(1)} pp`" not in app_js
    assert 'deltaPp >= 0 ? "+" : ""' in app_js


def test_tier5_r2_g1_paired_tests_filter_inf_without_warnings() -> None:
    """G1: Paired statistical tests filter +/-inf with np.isfinite and emit no RuntimeWarning."""
    a = [0.8, 0.9, float("inf")]
    b = [0.4, 0.5, 0.4]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p_t = compute_paired_ttest(a, b)
        p_w = compute_paired_wilcoxon(a, b)
        p_b = compute_paired_bootstrap_pvalue(a, b)
        p_d = _boot_pvalue_from_diffs(np.array([0.4, float("inf"), 0.4, float("-inf")]))

        p_t_lt2 = compute_paired_ttest([float("inf"), float("inf")], [0.5, 0.6])
        p_w_lt2 = compute_paired_wilcoxon([0.5, float("inf")], [0.4, 0.5])
        p_b_lt2 = compute_paired_bootstrap_pvalue([0.5, float("-inf")], [0.4, 0.5])
        p_d_lt2 = _boot_pvalue_from_diffs(np.array([0.4, float("inf")]))

    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
    for p_val in (p_t, p_w, p_b, p_d):
        assert math.isfinite(p_val) and 0.0 <= p_val <= 1.0
    for p_lt2 in (p_t_lt2, p_w_lt2, p_b_lt2, p_d_lt2):
        assert math.isnan(p_lt2)


def test_tier5_r2_g2_metrics_guard_nan_and_inf_predictions() -> None:
    """G2: _safe_r2, _safe_spearman, and evaluate_predictions return NaN cleanly on NaN/Inf inputs."""
    y_true = np.array([1.0, 1.5, 2.0])
    for bad_pred in (
        np.array([1.0, float("nan"), 2.0]),
        np.array([1.0, float("inf"), 2.0]),
        np.array([1.0, float("-inf"), 2.0]),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert math.isnan(_safe_r2(y_true, bad_pred))
            assert math.isnan(_safe_spearman(y_true, bad_pred))
            ev = evaluate_predictions(
                y_true,
                bad_pred,
                np.array([0.1, 0.2, 0.3]),
                bad_pred,
                y_rumor_true=np.array([0, 1, 0]),
                rumor_logits=np.array([[1.0, 0.0], [float("nan"), 1.0], [0.0, 1.0]]),
            )
        assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
        for key in (
            "msle",
            "mae",
            "rmse_size",
            "r2_size",
            "spearman_size",
            "rmse",
            "mae_polar",
            "r2_polar",
            "pearson_r",
            "spearman_polar",
            "accuracy",
            "macro_f1",
            "auc_roc",
        ):
            assert math.isnan(ev[key]), f"Expected NaN for {key}, got {ev[key]}"


def test_tier5_r2_g3_reporting_renders_inf_as_dash_without_bold_or_worse_marker(
    tmp_path: Path,
) -> None:
    """G3: reporting.py treats +/-inf as non-finite '--' and never crowns or compares against inf."""
    assert significance_marker(float("inf"), direction="better", latex=True) == ""
    assert format_mean_std(float("inf"), 0.0, latex=True, bold=True) == "--"
    assert format_mean_std(float("-inf"), 0.0, latex=False, bold=True) == "--"

    summary_stats = {
        "SBiFAGAT": {"msle": (0.30, 0.02), "r2_size": (0.50, 0.05)},
        "BrokenBaseline": {"msle": (0.40, 0.02), "r2_size": (float("inf"), 0.0)},
    }
    best = _best_model_per_metric(summary_stats, MAIN_METRICS[:3])
    assert best["r2_size"] == "SBiFAGAT"

    benchmark = {
        "reference_model": "SBiFAGAT",
        "comparator_model": "BrokenBaseline",
        "summary_stats": summary_stats,
        "p_values": {"msle": 0.02, "r2_size": float("inf")},
    }
    md = build_markdown_table(benchmark, MAIN_METRICS[:3])
    tex_path = export_latex_table(
        benchmark,
        tmp_path / "nonfinite_report.tex",
        caption="Non-Finite Guard",
        label="tab:nonfinite_guard",
        metrics=MAIN_METRICS[:3],
    )
    tex = tex_path.read_text(encoding="utf-8")

    assert "inf" not in md.lower()
    assert "inf" not in tex.lower()
    assert "\\textbf{inf" not in tex
    assert "$^{-*}$" not in tex
    assert "(-)*" not in md
    assert "BrokenBaseline & 0.4000 $\\pm$ 0.0200 & -- & -- \\\\" in tex


def test_tier5_r2_g4_get_early_polar_scalar_excludes_root_index() -> None:
    """G4: _get_early_polar_scalar excludes d.root_index instead of hardcoding index 0."""
    d_tensor_root = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        early_node_root_stance=torch.tensor([0.5, -1.0, 0.5], dtype=torch.float32),
        root_index=torch.tensor([1], dtype=torch.long),
    )
    assert _get_early_polar_scalar(d_tensor_root) == pytest.approx(0.0, abs=1e-6)

    d_int_root = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        early_node_root_stance=torch.tensor([0.5, -1.0, 0.5], dtype=torch.float32),
        root_index=1,
    )
    assert _get_early_polar_scalar(d_int_root) == pytest.approx(0.0, abs=1e-6)

    d_opposing_replies = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        early_node_root_stance=torch.tensor([0.8, 0.0, -0.8], dtype=torch.float32),
        root_index=torch.tensor([1], dtype=torch.long),
    )
    assert _get_early_polar_scalar(d_opposing_replies) > 0.5

    d_bad_root = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        early_node_root_stance=torch.tensor([-1.0, 0.5, 0.5], dtype=torch.float32),
        root_index=99,
    )
    assert _get_early_polar_scalar(d_bad_root) == pytest.approx(0.0, abs=1e-6)


def test_tier5_r2_g5_discourse_cha_and_benign_exclamations() -> None:
    """G5: 'chả' triggers constructive_oppose; benign short acks/numbers with '!!!' stay neutral."""
    cha_res = _heuristic_behavior_fallback("Chả hợp lý chút nào")
    assert cha_res["label"] == "constructive_oppose"
    assert cha_res["s_uv"] < 0.0

    for benign in ("ok!!!", "123!!!", "haha!!!", "ok nha!!!"):
        res = _heuristic_behavior_fallback(benign)
        assert res["label"] != "hostile_conflict", f"False hostile on {benign!r}: {res}"

    for hostile in ("Ngu thế!!!", "Biến đi!!!", "Nói vớ vẩn!!!"):
        res = _heuristic_behavior_fallback(hostile)
        assert res["label"] == "hostile_conflict", f"Missed hostile on {hostile!r}: {res}"


def test_tier5_r2_g6_validate_feature_tensors_teacher_attr_bool_and_idx_bounds() -> None:
    """G6: validate_feature_tensors checks teacher_edge_attr, rejects bools, and bounds *_id_to_idx."""
    base_payload = {
        "node_embeddings": torch.ones((2, 768), dtype=torch.float32),
        "edge_attr": torch.tensor([[0.2, 0.1]], dtype=torch.float32),
        "teacher_edge_attr": torch.tensor([[0.3, 0.2]], dtype=torch.float32),
        "node_id_to_idx": {"n0": 0, "n1": 1},
        "edge_id_to_idx": {"e0": 0},
        "edge_stance": {"e0": 0.2},
        "teacher_edge_stance": {"e0": 0.3},
        "teacher_node_root_stance": {"n1": 0.3},
    }
    validate_feature_tensors(base_payload)

    bad_patches: list[dict[str, Any]] = [
        {"teacher_edge_attr": torch.tensor([[float("nan"), 0.1]], dtype=torch.float32)},
        {"teacher_edge_attr": torch.tensor([[1.8, 0.1]], dtype=torch.float32)},
        {"teacher_edge_attr": torch.tensor([[0.2, 0.1, 0.3]], dtype=torch.float32)},
        {"teacher_edge_stance": {"e0": True}},
        {"edge_hostility": {"e0": False}},
        {"teacher_node_root_stance": {"n1": True}},
        {"node_id_to_idx": {"n0": 999}},
        {"node_id_to_idx": {"n0": -1}},
        {"node_id_to_idx": {"n0": True}},
        {"edge_id_to_idx": {"e0": 5}},
        {"edge_id_to_idx": {"e0": False}},
    ]
    for patch_dict in bad_patches:
        bad = dict(base_payload, **patch_dict)
        with pytest.raises(ValueError):
            validate_feature_tensors(bad)


def test_tier5_r3_g1_json_roundtrip_none_safe_across_reporting_trainer_demo_and_link(
    tmp_path: Path,
) -> None:
    """R3-G1: JSON-roundtripped None values never raise TypeError across reporting, trainer, demo, or Tier 2."""
    assert significance_marker(None) == ""
    assert format_mean_std(None, 0.1) == "--"
    assert format_mean_std(0.5, None) == "0.5000 ± --"

    bench_raw: dict[str, Any] = {
        "seeds": [42, 100],
        "reference_model": "SBiFAGAT",
        "comparator_model": "BiGCN",
        "nonparametric_test_label": "Cluster-Insufficient (<6 events)",
        "summary_stats": {
            "Early-K Persistence": {
                "msle": (0.90, 0.05),
                "mae": (0.70, 0.04),
                "r2_size": (0.10, 0.02),
                "rmse": (0.35, 0.02),
                "r2_polar": (0.05, 0.01),
                "pearson_r": (0.20, 0.03),
                "spearman_polar": (0.18, 0.02),
                "macro_f1": (float("nan"), float("nan")),
                "accuracy": (float("nan"), float("nan")),
            },
            "SBiFAGAT": {
                "msle": (0.40, 0.02),
                "mae": (0.50, 0.01),
                "r2_size": (0.55, 0.03),
                "rmse": (0.22, 0.01),
                "r2_polar": (0.35, 0.02),
                "pearson_r": (0.60, 0.03),
                "spearman_polar": (0.58, 0.02),
                "macro_f1": (0.72, 0.02),
                "accuracy": (0.75, 0.01),
            },
            "BiGCN": {
                "msle": (0.55, 0.03),
                "mae": (0.60, 0.02),
                "r2_size": (0.40, 0.04),
                "rmse": (0.28, 0.02),
                "r2_polar": (0.20, 0.03),
                "pearson_r": (0.45, 0.04),
                "spearman_polar": (0.44, 0.03),
                "macro_f1": (float("inf"), 0.0),
                "accuracy": (0.65, float("nan")),
            },
        },
        "p_values": {"msle": 0.01, "macro_f1": float("nan")},
        "adjusted_p_values": {"msle": 0.02, "macro_f1": float("nan")},
        "bootstrap_p_values": {"msle": float("nan"), "macro_f1": float("inf")},
        "adjusted_bootstrap_p_values": {"msle": float("nan"), "macro_f1": float("nan")},
        "wilcoxon_p_values": {"msle": 0.06, "macro_f1": float("nan")},
        "adjusted_wilcoxon_p_values": {"msle": 0.12, "macro_f1": float("nan")},
        "raw_results": {},
    }
    json_path = tmp_path / "bench_roundtrip.json"
    serialize_benchmark_json(bench_raw, json_path)
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))

    assert reloaded["summary_stats"]["Early-K Persistence"]["macro_f1"] == [None, None]
    assert reloaded["bootstrap_p_values"]["msle"] is None
    assert select_best_comparator(reloaded["summary_stats"], reference_model="SBiFAGAT") == "BiGCN"
    adj = holm_bonferroni_adjust({"msle": 0.01, "macro_f1": None, "accuracy": float("nan")})
    assert adj["msle"] == pytest.approx(0.01)
    assert math.isnan(adj["macro_f1"]) and math.isnan(adj["accuracy"])

    md = build_markdown_table(reloaded, MAIN_METRICS)
    assert "--" in md
    tex_p = export_latex_table(
        reloaded,
        tmp_path / "bench_roundtrip.tex",
        caption="Roundtrip",
        label="tab:rt",
        metrics=MAIN_METRICS,
    )
    assert tex_p.is_file() and "--" in tex_p.read_text(encoding="utf-8")
    serialize_benchmark_json(reloaded, tmp_path / "bench_roundtrip_2.json")

    for real_rel in (
        "results/tier1_tabular_inductive_pheme.json",
        "results/tier1_tabular_loo_pheme.json",
        "results/tier1_tabular_canonical_vn.json",
    ):
        real_path = REPO_ROOT / real_rel
        if real_path.is_file():
            real_obj = json.loads(real_path.read_text(encoding="utf-8"))
            assert isinstance(build_markdown_table(real_obj, MAIN_METRICS), str)
            export_latex_table(
                real_obj,
                tmp_path / f"{real_path.stem}.tex",
                caption="Real RT",
                label="tab:real_rt",
                metrics=MAIN_METRICS,
            )
            serialize_benchmark_json(real_obj, tmp_path / f"{real_path.stem}_rt.json")

    link_bench = {
        "reference_model": "SBiFAGAT-Link",
        "comparator_model": "BiEdgeGATv2-Link",
        "summary_stats": {
            "SBiFAGAT-Link": {
                "roc_auc": [0.75, 0.02],
                "pr_auc": [None, None],
                "macro_f1": [0.60, None],
                "trans_roc_auc": [0.70, 0.03],
                "trans_pr_auc": [0.68, 0.02],
                "trans_macro_f1": [0.55, 0.02],
            },
            "BiEdgeGATv2-Link": {
                "roc_auc": [0.70, 0.03],
                "pr_auc": [0.65, 0.02],
                "macro_f1": [None, None],
                "trans_roc_auc": [0.68, 0.04],
                "trans_pr_auc": [0.64, 0.03],
                "trans_macro_f1": [0.52, 0.02],
            },
        },
        "p_values": {"roc_auc": 0.01, "pr_auc": None},
        "adjusted_p_values": {"roc_auc": 0.02, "pr_auc": None},
    }
    link_md = build_link_markdown_table(link_bench, LINK_PRED_METRICS[:6])
    assert "--" in link_md
    link_tex = export_link_latex_table(
        link_bench,
        tmp_path / "link_rt.tex",
        caption="Link RT",
        label="tab:link_rt",
        metrics=LINK_PRED_METRICS[:6],
    )
    assert link_tex.is_file()

    paper_mock = tmp_path / "paper_mock"
    paper_mock.mkdir(parents=True, exist_ok=True)
    for fn in ("results_macro_network_vn.json", "results_macro_network_pheme.json"):
        (paper_mock / fn).write_text("{}", encoding="utf-8")
    for fn in ("results_signed_link_vn.json", "results_signed_link_pheme.json"):
        (paper_mock / fn).write_text(json.dumps(link_bench), encoding="utf-8")

    backend = DemoBackend(
        data_dir=tmp_path / "absent_data",
        ckpt_path=tmp_path / "absent.pt",
        ce_dir=tmp_path / "absent_ce",
        paper_dir=paper_mock,
        web_dir=REPO_ROOT / "web_demo",
        preload=False,
    )
    tier2_payload = backend.get_tier2_macro()
    vn_rows = {r["model"]: r for r in tier2_payload["corpora"]["vn"]["signed_link"]["rows"]}
    assert vn_rows["SBiFAGAT-Link"]["pr_auc"] == {"mean": None, "std": None}
    assert vn_rows["SBiFAGAT-Link"]["macro_f1"] == {"mean": 0.6, "std": None}


def test_tier5_r3_g2_consistent_root_index_in_sota_stats_and_mechanisms() -> None:
    """R3-G2: sota_cascades, export_dataset_stats, and analyze_mechanisms handle int/Tensor/bool/OOB root_index."""
    cascades_tensor: list[Data] = []
    cascades_int: list[Data] = []
    for idx in range(3):
        torch.manual_seed(10 + idx)
        x = torch.randn((3, 16), dtype=torch.float32)
        struct_x = torch.rand((3, 10), dtype=torch.float32)
        ei = torch.tensor([[1, 0], [0, 2]], dtype=torch.long)
        ea = torch.tensor([[0.6, 0.1], [-0.5, 0.7]], dtype=torch.float32)
        common = {
            "x": x,
            "struct_x": struct_x,
            "edge_index": ei,
            "edge_index_td": ei,
            "edge_attr": ea,
            "cascade_stats": torch.ones((1, 8), dtype=torch.float32) * (idx + 1),
            "y_size": torch.tensor([1.5], dtype=torch.float32),
            "y_polar": torch.tensor([0.3], dtype=torch.float32),
            "early_node_root_stance": torch.tensor([0.5, -1.0, 0.5], dtype=torch.float32),
            "early_y_polar": torch.tensor([0.2], dtype=torch.float32),
        }
        cascades_tensor.append(Data(**common, root_index=torch.tensor([1], dtype=torch.long)))
        cascades_int.append(Data(**common, root_index=1))

    feats_tensor, _ = extract_cascade_summary_features(cascades_tensor, svd_dim=2)
    feats_int, _ = extract_cascade_summary_features(cascades_int, svd_dim=2)
    assert np.allclose(feats_tensor, feats_int, atol=1e-6)

    d_t = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        y_size=torch.tensor([1.5], dtype=torch.float32),
        y_polar=torch.tensor([0.2], dtype=torch.float32),
        early_node_root_stance=torch.tensor([0.5, -1.0, 0.5], dtype=torch.float32),
        root_index=torch.tensor([1], dtype=torch.long),
    )
    d_i = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        y_size=torch.tensor([1.5], dtype=torch.float32),
        y_polar=torch.tensor([0.2], dtype=torch.float32),
        early_node_root_stance=torch.tensor([0.5, -1.0, 0.5], dtype=torch.float32),
        root_index=1,
    )
    assert compute_early_k_persistence_metrics([d_t])["mean_early_polar"] == pytest.approx(0.0, abs=1e-6)
    assert compute_early_k_persistence_metrics([d_i])["mean_early_polar"] == pytest.approx(0.0, abs=1e-6)

    d_edge_only = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        y_size=torch.tensor([1.5], dtype=torch.float32),
        y_polar=torch.tensor([0.2], dtype=torch.float32),
        edge_index_td=torch.tensor([[1, 0], [0, 2]], dtype=torch.long),
        edge_attr=torch.tensor([[0.6, 0.1], [-0.5, 0.7]], dtype=torch.float32),
        cascade_stats=torch.zeros((1, 8), dtype=torch.float32),
        early_y_polar=torch.tensor([0.2], dtype=torch.float32),
        root_index=1,
    )
    d_edge_only_t = d_edge_only.clone()
    d_edge_only_t.root_index = torch.tensor([1], dtype=torch.long)

    depths_i = _resolve_edge_depths(d_edge_only)
    depths_t = _resolve_edge_depths(d_edge_only_t)
    assert np.array_equal(depths_i, np.array([1, 2], dtype=np.int64))
    assert np.array_equal(depths_i, depths_t)

    cloned_i = _clone_cascade_with_edge_attr(d_edge_only, d_edge_only.edge_attr)
    cloned_t = _clone_cascade_with_edge_attr(d_edge_only_t, d_edge_only_t.edge_attr)
    assert float(cloned_i.early_y_polar.item()) == pytest.approx(float(cloned_t.early_y_polar.item()), abs=1e-6)


def test_tier5_r3_g3_bootstrap_pvalues_nan_inf_no_warnings_and_no_bias() -> None:
    """R3-G3: _compute_sample_level_bootstrap_pvalues returns NaN cleanly on NaN/Inf without RuntimeWarnings."""
    y_size_true = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    y_polar_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    y_rumor_true = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    topic_ids = [f"t_{i}" for i in range(8)]

    clean_ref = {
        "y_size_true": y_size_true,
        "y_polar_true": y_polar_true,
        "y_rumor_true": y_rumor_true,
        "y_size_pred": np.array([1.1, 1.4, 2.1, 2.4, 3.1, 3.4, 4.1, 4.4]),
        "y_polar_pred": np.array([0.12, 0.19, 0.31, 0.39, 0.52, 0.59, 0.71, 0.79]),
        "rumor_logits": np.array([[2.0, 0.1], [0.1, 2.0]] * 4, dtype=np.float64),
        "topic_ids": topic_ids,
    }
    bad_cmp = {
        "y_size_true": y_size_true,
        "y_polar_true": y_polar_true,
        "y_rumor_true": y_rumor_true,
        "y_size_pred": np.array([1.0, float("nan"), 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]),
        "y_polar_pred": np.array([0.1, 0.2, float("inf"), 0.4, 0.5, 0.6, 0.7, 0.8]),
        "rumor_logits": np.array([[float("nan"), 0.1], [0.1, 2.0]] * 4, dtype=np.float64),
        "topic_ids": topic_ids,
    }
    keys = (
        "msle",
        "rmse_size",
        "mae",
        "r2_size",
        "spearman_size",
        "rmse",
        "mae_polar",
        "r2_polar",
        "pearson_r",
        "spearman_polar",
        "accuracy",
        "macro_f1",
        "auc_roc",
    )

    for use_clusters in (False, True):
        ref_in = dict(clean_ref)
        cmp_in = dict(bad_cmp)
        if not use_clusters:
            ref_in["topic_ids"] = None
            cmp_in["topic_ids"] = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pvals = _compute_sample_level_bootstrap_pvalues(
                [ref_in],
                [cmp_in],
                metric_keys=keys,
                n_resamples=30,
                n_stat_resamples=20,
            )
            cl_pvals = compute_cluster_paired_tests([ref_in], [cmp_in], metric_keys=keys)
        assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
        for k in keys:
            assert math.isnan(pvals[k]), f"Expected NaN for {k} (clusters={use_clusters}), got {pvals[k]}"
            assert math.isnan(cl_pvals["ttest"][k])
            assert math.isnan(cl_pvals["wilcoxon"][k])

    # Partial corruption: only y_size_pred has Inf -> size p-values are NaN while polar p-values stay finite
    size_only_bad = dict(clean_ref, y_size_pred=np.array([1.0, float("-inf"), 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]))
    with warnings.catch_warnings(record=True) as caught_part:
        warnings.simplefilter("always")
        part_pvals = _compute_sample_level_bootstrap_pvalues(
            [clean_ref],
            [size_only_bad],
            metric_keys=keys,
            n_resamples=30,
            n_stat_resamples=20,
        )
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught_part)
    for k in ("msle", "rmse_size", "mae", "r2_size", "spearman_size"):
        assert math.isnan(part_pvals[k])
    for k in ("rmse", "mae_polar", "r2_polar", "pearson_r", "spearman_polar", "accuracy", "macro_f1", "auc_roc"):
        assert math.isfinite(part_pvals[k])

"""Tier 4: Real-world application workflow E2E tests against disk corpora, checkpoints & live HTTP server."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from scripts.compute_kappa import evaluate_audit_csv, evaluate_thread_level_audit_csv
from src.data.pyg_dataset import (
    EarlyCascadeDataset,
    TOTAL_NODE_FEATURE_DIM,
    compute_embedding_cluster_splits,
    load_topic_splits_json,
)
from src.data.schema import load_cascade_parquet
from src.data.user_network import (
    build_temporal_signed_link_dataset,
    build_user_interaction_records,
    compute_macro_signed_network_stats,
)
from src.models.sbi_fagat import SBiFAGAT
from src.nlp.dual_edge_scorer import DualHeadCrossEncoder
from src.training.trainer import _compute_sample_level_bootstrap_pvalues
from tests.e2e.conftest import (
    CE_PHEME_DIR,
    CE_VN_DIR,
    CKPT_PHEME_PATH,
    CKPT_VN_PATH,
    DATA_PHEME_DIR,
    DATA_VN_DIR,
    HAS_DUAL_CE_PHEME,
    HAS_DUAL_CE_VN,
    HAS_HUMAN_AUDIT_VN,
    HAS_PAPER_TIER2_JSONS,
    HAS_PHEME_CKPT,
    HAS_REAL_PHEME,
    HAS_REAL_VN,
    HAS_VN_CKPT,
    INTERIM_DIR,
    REPO_ROOT,
    http_get,
    http_post_json,
    start_demo_server,
)


@pytest.mark.tier4
@pytest.mark.tier4_realdata
@pytest.mark.skipif(not HAS_REAL_VN, reason="Real VN processed dataset not present on disk")
def test_t4_vn_processed_corpus_integrity_and_embedding_clusters() -> None:
    """Scenario 1: VN processed corpus enforces zero-leakage future_only, disjoint splits, and 0 zero-rows."""
    ds_tr = EarlyCascadeDataset(root_dir=DATA_VN_DIR, k_horizon=20, split="train", min_total_replies=25)
    ds_va = EarlyCascadeDataset(root_dir=DATA_VN_DIR, k_horizon=20, split="val", min_total_replies=25)
    ds_te = EarlyCascadeDataset(root_dir=DATA_VN_DIR, k_horizon=20, split="test", min_total_replies=25)

    assert ds_tr.polar_target_mode == "future_only"
    assert len(ds_tr) > 100 and len(ds_va) > 50 and len(ds_te) > 50

    tr_topics = {d.topic_id for d in ds_tr}
    va_topics = {d.topic_id for d in ds_va}
    te_topics = {d.topic_id for d in ds_te}
    assert tr_topics.isdisjoint(va_topics) and tr_topics.isdisjoint(te_topics) and va_topics.isdisjoint(te_topics)

    for sample in (ds_tr[0], ds_va[0], ds_te[0]):
        assert torch.isfinite(sample.x).all()
        assert torch.isfinite(sample.edge_attr).all()
        assert float(sample.x[:, :768].abs().sum(dim=1).min().item()) > 0.0

    # Verify embedding-cluster split helper runs cleanly on real VN corpus
    all_items = list(ds_tr) + list(ds_va) + list(ds_te)
    emb_splits = compute_embedding_cluster_splits(all_items, n_clusters=24, seed=42)
    assert emb_splits["train"].isdisjoint(emb_splits["val"])
    assert emb_splits["train"].isdisjoint(emb_splits["test"])
    assert emb_splits["val"].isdisjoint(emb_splits["test"])


@pytest.mark.tier4
@pytest.mark.tier4_realdata
@pytest.mark.skipif(
    not (HAS_REAL_PHEME and HAS_PHEME_CKPT),
    reason="Real PHEME processed corpus or checkpoint not present on disk",
)
def test_t4_pheme_corpus_loo_guard_and_real_checkpoint_inference() -> None:
    """Scenario 2: PHEME 2-event test split triggers <6-event non-IID guard; real checkpoint omits fake 0.75 conf."""
    splits = load_topic_splits_json(DATA_PHEME_DIR / "topic_splits.json")
    assert len(splits["test"]) < 6  # ottawashooting, prince-toronto

    ds_te = EarlyCascadeDataset(root_dir=DATA_PHEME_DIR, k_horizon=10, split="test", min_total_replies=15)
    assert ds_te.polar_target_mode == "future_only"
    assert len(ds_te) > 10

    ckpt = torch.load(CKPT_PHEME_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = SBiFAGAT(
        in_dim=int(ckpt.get("in_dim", TOTAL_NODE_FEATURE_DIM)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        edge_dim=int(cfg.get("edge_dim", 2)),
        num_rumor_classes=int(ckpt.get("num_rumor_classes", 4)),
        use_cascade_stats=bool(cfg.get("use_cascade_stats", True)),
        predict_uncertainty=bool(cfg.get("predict_uncertainty", False)),
        use_polar_persistence_anchor=bool(cfg.get("use_polar_persistence_anchor", True)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    batch = Batch.from_data_list([ds_te[i] for i in range(min(32, len(ds_te)))])
    diag = model.predict_with_diagnostics(batch)
    if not bool(cfg.get("predict_uncertainty", False)):
        assert diag["confidence_score"] is None
        assert diag["uncertainty_available"] is False

    # Verify <6-event guard on real PHEME test topic_ids
    with torch.no_grad():
        out = model(batch)
    tids = [ds_te[i].topic_id for i in range(min(32, len(ds_te)))]
    pred_dict = {
        "y_size_true": batch.y_size.numpy(),
        "y_size_pred": out["y_size_pred"].numpy(),
        "y_polar_true": batch.y_polar.numpy(),
        "y_polar_pred": out["y_polar_pred"].numpy(),
        "y_rumor_true": batch.y_rumor.numpy(),
        "rumor_logits": out["rumor_logits"].numpy(),
        "topic_ids": tids,
    }
    pvals = _compute_sample_level_bootstrap_pvalues(
        [pred_dict],
        [pred_dict],
        metric_keys=("msle", "r2_polar", "macro_f1"),
        seed=42,
        n_resamples=50,
        n_stat_resamples=50,
    )
    assert all(math.isnan(float(v)) for v in pvals.values())


@pytest.mark.tier4
@pytest.mark.tier4_realdata
@pytest.mark.skipif(
    not (HAS_REAL_VN and HAS_REAL_PHEME),
    reason="Real VN and PHEME processed corpora not present on disk",
)
def test_t4_tier2_real_vn_and_pheme_macro_audit() -> None:
    """Scenario 3: Real VN & PHEME Tier 2 audit verifies Gómez Q_signed, Davis Weak Balance & 0 PHEME zero-rows."""
    for name, ddir in (("vn", DATA_VN_DIR), ("pheme", DATA_PHEME_DIR)):
        stats = compute_macro_signed_network_stats(ddir, dataset_name=name, num_null_perms=5, seed=42)
        pol = stats["polarization"]
        sb = stats["structural_balance"]
        assert -1.0 <= float(pol["rwc"]) <= 1.0
        assert -1.0 <= float(pol["signed_modularity"]) <= 1.0
        assert "q_signed_z_score" in pol and math.isfinite(float(pol["q_signed_z_score"]))
        assert "weak_balance_ratio" in sb and 0.0 <= float(sb["weak_balance_ratio"]) <= 1.0
        assert float(sb["weak_balance_ratio"]) >= float(sb["balanced_ratio"])

    # PHEME B4 zero-row filtering check
    rec_pheme = build_user_interaction_records(DATA_PHEME_DIR, prefer_teacher=True)
    embs_pheme = rec_pheme["node_embeddings"]
    for u, posts in rec_pheme["user_posts"].items():
        for _ts, emb_idx in posts:
            assert float(embs_pheme[emb_idx].abs().sum().item()) > 0.0, (
                f"Zero-vector PHEME row {emb_idx} entered user_posts for {u}"
            )
    link_pheme = build_temporal_signed_link_dataset(DATA_PHEME_DIR, dataset_name="pheme")
    assert "unseen_pair_mask" in link_pheme["val"] and "unseen_pair_mask" in link_pheme["test"]


@pytest.mark.tier4
@pytest.mark.tier4_realdata
@pytest.mark.skipif(not HAS_HUMAN_AUDIT_VN, reason="Real VN human audit CSVs not present on disk")
def test_t4_real_vn_human_audit_120_pairs_and_30_trees() -> None:
    """Scenario 4: Real VN human audits (120 pairs + 30 trees) evaluate cleanly with finite metrics."""
    pair_rep = evaluate_audit_csv(
        INTERIM_DIR / "human_audit_vi_120.csv",
        machine_key_csv=INTERIM_DIR / "human_audit_vi_120_machine_key.csv",
        allow_single_gold=True,
    )
    assert pair_rep["machine_vs_gold"]["stance"]["num_samples"] == 120
    assert math.isfinite(float(pair_rep["machine_vs_gold"]["stance"]["macro_f1"]))

    tree_rep = evaluate_thread_level_audit_csv(
        INTERIM_DIR / "human_audit_vn_30_trees.csv",
        machine_key_csv=INTERIM_DIR / "human_audit_vn_30_trees_machine_key.csv",
        allow_single_gold=True,
    )
    assert tree_rep["machine_vs_gold"]["num_threads"] == 30
    assert math.isfinite(float(tree_rep["machine_vs_gold"]["spearman_r"]))


@pytest.mark.tier4
@pytest.mark.tier4_realdata
@pytest.mark.skipif(
    not (HAS_REAL_VN and HAS_VN_CKPT),
    reason="Real VN processed dataset or checkpoint not present on disk",
)
def test_t4_real_vn_web_demo_http_user_journey() -> None:
    """Scenario 5: Live Web Demo over real VN corpus serves 10 clean non-REDACTED threads & honest confidence."""
    server, thread, port = start_demo_server(
        data_dir=DATA_VN_DIR,
        ckpt_path=CKPT_VN_PATH,
        ce_dir=CE_VN_DIR,
    )
    try:
        st_h, _, raw_h = http_get(port, "/api/health")
        assert st_h == 200
        health = json.loads(raw_h.decode("utf-8"))
        assert health["real_vn_data_available"] is True
        assert health["gnn_checkpoint_loaded"] is True

        st_l, _, raw_l = http_get(port, "/api/tier1/threads")
        assert st_l == 200
        listing = json.loads(raw_l.decode("utf-8"))
        assert listing["is_synthetic_fallback"] is False
        assert listing["count"] == 10

        for item in listing["threads"]:
            assert "[REDACTED" not in item["title_snippet"], (
                f"Demo thread {item['cascade_id']} leaked [REDACTED in title_snippet"
            )
            st_d, _, raw_d = http_get(port, f"/api/tier1/thread?id={item['cascade_id']}")
            assert st_d == 200
            detail = json.loads(raw_d.decode("utf-8"))
            assert "early_y_polar" in detail["forecast"]
            assert detail["forecast"]["confidence_score"] is None
            for node in detail["tree"]["nodes"]:
                assert "[REDACTED" not in node["text"], (
                    f"Demo thread {item['cascade_id']} node {node['node_id']} contains [REDACTED"
                )

        if HAS_PAPER_TIER2_JSONS:
            st_m, _, raw_m = http_get(port, "/api/tier2/macro")
            assert st_m == 200
            macro = json.loads(raw_m.decode("utf-8"))
            assert "vn" in macro["corpora"] and "pheme" in macro["corpora"]

        st_s, _, sb_body = http_post_json(
            port, "/api/tier1/sandbox", {"mode": "preset_junk_1", "junk_mode": "contract"}
        )
        assert st_s == 200
        assert sb_body["junk_filter_stats"]["mode"] == "contract"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.tier4
@pytest.mark.tier4_realdata
@pytest.mark.skipif(not HAS_VN_CKPT, reason="Real VN GNN checkpoint not present on disk")
def test_t4_real_cli_live_predictor_and_efficiency_workflow(tmp_path: Path) -> None:
    """Scenario 6: Subprocess CLI predict_live_thread with real VN checkpoint + benchmark_efficiency."""
    # Export a clean real-style thread JSON to test predict_live_thread --input-json --gnn-ckpt
    base_ts = 1_700_000_000.0
    thread_json = tmp_path / "clean_thread.json"
    thread_json.write_text(
        json.dumps(
            {
                "id": "vn_clean_e2e",
                "title": "Thảo luận cải thiện hạ tầng giao thông công cộng",
                "selftext": "Các tuyến xe buýt điện đang giúp giảm ùn tắc giờ cao điểm.",
                "author": "citizen_01",
                "subreddit": "vozforums",
                "created_utc": base_ts,
                "comments": [
                    {
                        "id": f"c{i:02d}",
                        "parent_id": "t3_vn_clean_e2e" if i <= 5 else f"t1_c{(i - 1) // 2 + 1:02d}",
                        "author": f"u_{i:02d}",
                        "body": (
                            "Phản đối phương án tăng vé đột ngột."
                            if i % 4 == 0
                            else "Đồng ý với đề xuất mở rộng tuyến xe buýt mới."
                        ),
                        "created_utc": base_ts + 60.0 * i,
                    }
                    for i in range(1, 25)
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out_live = tmp_path / "live_real_ckpt.json"
    proc_live = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.predict_live_thread",
            "--input-json",
            str(thread_json),
            "--gnn-ckpt",
            str(CKPT_VN_PATH),
            "--junk-mode",
            "contract",
            "--output-json",
            str(out_live),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert proc_live.returncode == 0, f"predict_live_thread failed:\n{proc_live.stdout}\n{proc_live.stderr}"
    live_res = json.loads(out_live.read_text(encoding="utf-8"))
    assert live_res["checkpoint_loaded"] is True
    assert "early_y_polar" in live_res
    assert live_res["confidence_score"] is None

    out_eff = tmp_path / "eff_t4.json"
    proc_eff = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.benchmark_efficiency",
            "--num-cascades",
            "8",
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--output-json",
            str(out_eff),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert proc_eff.returncode == 0, proc_eff.stderr
    assert out_eff.is_file()


@pytest.mark.tier4
@pytest.mark.tier4_realdata
@pytest.mark.skipif(
    not (HAS_DUAL_CE_VN and HAS_DUAL_CE_PHEME),
    reason="Trained DualHeadCrossEncoder binary weights not yet present on disk",
)
def test_t4_real_dual_ce_checkpoint_scoring_journey() -> None:
    """Scenario 7: Trained DualHeadCrossEncoder VN & PHEME checkpoints load and score pairs cleanly."""
    for ce_dir, pair in (
        (CE_VN_DIR, ("Thảo luận chính sách giao thông mới", "Phản đối phương án này hoàn toàn")),
        (CE_PHEME_DIR, ("Breaking news update from downtown", "This rumor is false and completely fabricated")),
    ):
        ce_model, tok = DualHeadCrossEncoder.load_checkpoint(ce_dir, map_location="cpu")
        ce_model.eval()
        assert tok is not None
        enc = tok([pair[0]], [pair[1]], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = ce_model(enc["input_ids"], enc["attention_mask"])
        assert out["e_uv"].shape == (1, 2)
        assert -1.0 <= float(out["s_uv"].item()) <= 1.0
        assert 0.0 <= float(out["h_uv"].item()) <= 1.0

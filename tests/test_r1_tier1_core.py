"""Comprehensive M3 Tier 1 regression & feature tests for SBi-FAGAT."""

from __future__ import annotations

import hashlib
import inspect
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import scripts.run_experiments as run_exp_mod
from src.data.arctic_shift_vn import (
    build_vn_cascade_from_thread,
    compare_prune_vs_contract_on_threads,
    compose_contracted_edge_stance,
    is_filler_junk_text,
    resolve_contracted_parent_id,
)
from src.data.pyg_dataset import (
    TIER2_PRIOR_DIM,
    TOTAL_NODE_FEATURE_DIM,
    CascadeData,
    EarlyCascadeDataset,
    compute_early_cascade_burst_stats,
)
from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet
from src.models.baselines import (
    BiEdgeGATv2,
    GlobalAttentionReadout as BaselineAttentionReadout,
    _BaselineAttentionReadout,
)
from src.models.sbi_fagat import (
    GlobalAttentionReadout as SBiAttentionReadout,
    SBiFAGAT,
    StanceGuidedFAConv,
)
from src.models.sota_cascades import (
    CalibratedEarlyKPersistence,
    CasCN,
    CasFlow,
    FriedkinJohnsenGNN,
    LightGBMBaseline,
    RidgeBaseline,
)
from src.training.auto_loop import run_self_auditing_validation_loop
from src.training.metrics import compute_multitask_loss
from src.training.reporting import build_markdown_table, export_latex_table
from src.training.trainer import (
    _compute_sample_level_bootstrap_pvalues,
    _estimate_train_target_stats,
    compute_cluster_paired_tests,
    compute_paired_wilcoxon,
    run_multi_seed_benchmark,
)
from src.utils.seed import seed_everything


def _make_node_edge_corpus(
    *,
    num_cascades: int = 8,
    replies_per_cascade: int = 26,
    with_rumor_labels: bool = True,
    with_legacy_4d: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_ts = datetime(2026, 3, 1, 8, 0, 0, tzinfo=timezone.utc)
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    node_embs: dict[str, torch.Tensor] = {}
    e_s: dict[str, float] = {}
    e_h: dict[str, float] = {}
    e_r: dict[str, float] = {}
    e_c: dict[str, float] = {}

    # Shared recurring authors across chronological cascades so Tier 2 -> 1 priors are non-trivial
    user_pool = [hashlib.sha256(f"user_{u}".encode()).hexdigest() for u in range(10)]
    rumors = ["non-rumor", "true", "false", "unverified"]

    for c_idx in range(num_cascades):
        cid = f"c_{c_idx:02d}"
        tid = f"topic_{c_idx}"
        r_id = f"{cid}_r"
        r_ts = base_ts + timedelta(hours=6 * c_idx)
        op_hash = user_pool[c_idx % 3]
        node_rows.append(
            {
                "node_id": r_id,
                "cascade_id": cid,
                "topic_id": tid,
                "parent_id": None,
                "is_root": True,
                "text": f"Chủ đề thảo luận số {c_idx} về chính sách giao thông",
                "created_at_utc": r_ts,
                "lang": "vi",
                "source_dataset": "arctic_shift_vozforums",
                "user_id_hash": op_hash,
                "rumor_label": rumors[c_idx % 4] if with_rumor_labels else None,
            }
        )
        torch.manual_seed(100 + c_idx)
        node_embs[r_id] = torch.randn(768, dtype=torch.float32) * 0.3

        for k in range(1, replies_per_cascade + 1):
            nid = f"{cid}_n{k}"
            pid = r_id if k <= 6 else f"{cid}_n{(k - 1) // 3 + 1}"
            n_ts = r_ts + timedelta(seconds=30 * k)
            u_hash = user_pool[(c_idx + k) % len(user_pool)]
            node_rows.append(
                {
                    "node_id": nid,
                    "cascade_id": cid,
                    "topic_id": tid,
                    "parent_id": pid,
                    "is_root": False,
                    "text": f"Bình luận chi tiết thứ {k} trong chuỗi {c_idx} phân tích rất rõ",
                    "created_at_utc": n_ts,
                    "lang": "vi",
                    "source_dataset": "arctic_shift_vozforums",
                    "user_id_hash": u_hash,
                    "rumor_label": None,
                }
            )
            node_embs[nid] = torch.randn(768, dtype=torch.float32) * 0.3
            eid = f"{pid}->{nid}"
            edge_rows.append(
                {
                    "edge_id": eid,
                    "cascade_id": cid,
                    "source_id": pid,
                    "target_id": nid,
                    "created_at_utc": n_ts,
                    "time_delay_sec": float(30 * k),
                    "edge_type": "reply",
                }
            )
            if c_idx % 2 == 0:
                s_val = 0.85 if k % 4 != 0 else -0.25
                h_val = 0.70 if s_val < 0 else 0.10
            else:
                s_val = 0.80
                h_val = 0.05
            e_s[eid] = s_val
            e_h[eid] = h_val
            if with_legacy_4d:
                e_r[eid] = 0.4
                e_c[eid] = 0.6

    nlp_feats: dict[str, Any] = {
        "node_embeddings": node_embs,
        "edge_stance": e_s,
        "edge_hostility": e_h,
    }
    if with_legacy_4d:
        nlp_feats["edge_dim"] = 4
        nlp_feats["edge_irrationality"] = e_r
        nlp_feats["edge_constructive"] = e_c
    return node_rows, edge_rows, nlp_feats


# ---------------------------------------------------------------------------
# 1. arctic_shift_vn.py: M6 Path Contraction, Truncated Orphans & Bot Detection
# ---------------------------------------------------------------------------


def test_arctic_shift_m6_bot_orphans_parity_and_comparison() -> None:
    assert is_filler_junk_text("Chào bạn, bài viết đã được kiểm duyệt.", author_raw="AutoModerator") is True
    assert is_filler_junk_text("Đồng ý quan điểm này.", author_raw="nguoi_dung_binh_thuong") is False

    base_ts = 1_710_000_000.0
    post = {
        "id": "m6post",
        "title": "Thảo luận quy hoạch giao thông nội đô",
        "selftext": "Mọi người cho ý kiến về việc hạn chế xe cá nhân.",
        "author": "op_user",
        "subreddit": "vozforums",
        "created_utc": base_ts,
    }
    comments = [
        # Bot comment WITHOUT author_is_bot key (only author)
        {
            "fullname": "t1_bot1",
            "link_id": "t3_m6post",
            "parent_id": "t3_m6post",
            "author": "AutoModerator",
            "text": "Chào bạn, bài viết đã được kiểm duyệt.",
            "created_utc": base_ts + 5.0,
        },
        # Deleted intermediate node with a valid child
        {
            "fullname": "t1_del1",
            "link_id": "t3_m6post",
            "parent_id": "t3_m6post",
            "author": "[deleted]",
            "text": "[deleted]",
            "created_utc": base_ts + 10.0,
        },
        {
            "fullname": "t1_under_del",
            "link_id": "t3_m6post",
            "parent_id": "t1_del1",
            "author": "u_child",
            "text": "Đồng ý hoàn toàn với phân tích vừa rồi của bác.",
            "created_utc": base_ts + 20.0,
        },
        # Truncated-API orphan pointing to missing parent NOT in comments
        {
            "fullname": "t1_orphan1",
            "link_id": "t3_m6post",
            "parent_id": "t1_missing_parent_from_api",
            "author": "u_orphan",
            "text": "Ý kiến phản đối mạnh mẽ phương án thu phí này.",
            "created_utc": base_ts + 25.0,
        },
        # Filler intermediate with known prior opposing stance
        {
            "fullname": "t1_keep_gp",
            "link_id": "t3_m6post",
            "parent_id": "t3_m6post",
            "author": "u_gp",
            "text": "Tôi ủng hộ giải pháp giao thông công cộng trước.",
            "created_utc": base_ts + 30.0,
        },
        {
            "fullname": "t1_filler_opp",
            "link_id": "t3_m6post",
            "parent_id": "t1_keep_gp",
            "author": "u_mid",
            "text": "hóng",
            "created_utc": base_ts + 35.0,
        },
        {
            "fullname": "t1_under_filler",
            "link_id": "t3_m6post",
            "parent_id": "t1_filler_opp",
            "author": "u_leaf",
            "text": "Phản đối việc tăng giá vé đột ngột gây khó khăn.",
            "created_utc": base_ts + 40.0,
        },
    ]

    # Cycle + truncated orphan check on resolve_contracted_parent_id
    orphans: set[str] = set()
    res_drop = resolve_contracted_parent_id(
        "t1_orphan1",
        {"t1_orphan1": "t1_missing_parent_from_api"},
        {"t1_orphan1"},
        "t3_m6post",
        truncated_orphans_out=orphans,
        drop_truncated_orphans=True,
    )
    assert res_drop is None
    assert "t1_orphan1" in orphans

    # Probe-path missing parent check
    orphans2: set[str] = set()
    res_probe_drop = resolve_contracted_parent_id(
        "t1_d",
        {"t1_d": "t1_k", "t1_k": "t1_missing"},
        {"t1_d", "t1_k"},
        "t3_m6post",
        truncated_orphans_out=orphans2,
        drop_truncated_orphans=True,
    )
    assert res_probe_drop is None and "t1_d" in orphans2

    # Stance composition helper
    eff_neg, par_neg, risk_neg = compose_contracted_edge_stance([-0.8], 0.7)
    assert par_neg == -1 and risk_neg is False and eff_neg == pytest.approx(-0.7)
    eff_risk, par_risk, risk_flag = compose_contracted_edge_stance([None], 0.7)
    assert par_risk == 0 and risk_flag is True and eff_risk == pytest.approx(0.7)

    # Default orphan_mode="prune": bot filtered, truncated orphan pruned (not attached to root),
    # [deleted] bridge flagged with stance_inversion_risk=True, known opposing chain gets parity=-1
    stats_out: dict[str, Any] = {}
    built = build_vn_cascade_from_thread(
        post,
        comments,
        min_comments=1,
        min_depth=1,
        min_vi_ratio=0.0,
        junk_mode="contract",
        orphan_mode="prune",
        prior_edge_stances={"t1_filler_opp": -0.8},
        stats_out=stats_out,
    )
    assert built is not None
    nodes, edges, stats = built
    nids = {r["node_id"] for r in nodes}
    assert "vn_m6post_t1_bot1" not in nids
    assert "vn_m6post_t1_orphan1" not in nids
    assert "vn_m6post_t1_under_del" in nids
    jstats = stats["junk_filter_stats"]
    assert jstats["truncated_orphans"] == 1
    assert jstats["truncated_orphan_node_ids"] == ["vn_m6post_t1_orphan1"]
    assert jstats["stance_inversion_risk_edges"] == 1
    meta_by_target = {m["target_id"]: m for m in jstats["relinked_edge_metadata"]}
    assert meta_by_target["vn_m6post_t1_under_del"]["stance_inversion_risk"] is True
    assert meta_by_target["vn_m6post_t1_under_del"]["contracted_sign_parity"] == 0
    assert meta_by_target["vn_m6post_t1_under_filler"]["stance_inversion_risk"] is False
    assert meta_by_target["vn_m6post_t1_under_filler"]["contracted_sign_parity"] == -1

    # preserve_opaque_deleted_bridges=True keeps neutral structural bridge for t1_del1
    built_bridge = build_vn_cascade_from_thread(
        post,
        comments,
        min_comments=1,
        min_depth=1,
        min_vi_ratio=0.0,
        junk_mode="contract",
        preserve_opaque_deleted_bridges=True,
    )
    assert built_bridge is not None
    bridge_nids = {r["node_id"]: r for r in built_bridge[0]}
    assert "vn_m6post_t1_del1" in bridge_nids
    assert bridge_nids["vn_m6post_t1_del1"]["text"] == "[deleted_bridge]"

    # compare_prune_vs_contract_on_threads
    cmp_res = compare_prune_vs_contract_on_threads(
        [(post, comments)],
        min_comments=1,
        min_depth=1,
        min_vi_ratio=0.0,
        k_horizon=20,
    )
    assert cmp_res["threads_evaluated"] == 1
    assert cmp_res["kept_in_contract"] == 1
    assert cmp_res["total_contracted_nodes"] >= 1
    assert cmp_res["total_truncated_orphans"] == 1


# ---------------------------------------------------------------------------
# 2. pyg_dataset.py: N3/M9 Fixes & Zero-Leakage Causal Tier 2 -> 1 Bridge
# ---------------------------------------------------------------------------


def test_pyg_dataset_n3_m9_and_causal_tier2_prior(tmp_path: Path) -> None:
    assert TIER2_PRIOR_DIM == 7
    sig = inspect.signature(EarlyCascadeDataset.__init__)
    assert sig.parameters["polar_target_mode"].default == "future_only"

    # unique_user_ratio excludes deleted/removed/automoderator/bot in denominator
    base_ts = datetime(2026, 3, 1, 8, 0, 0, tzinfo=timezone.utc)
    kept_nodes = [
        {"node_id": "r", "parent_id": None, "text": "root", "user_id_hash": "op_hash"},
        {"node_id": "n1", "parent_id": "r", "text": "mot hai ba bon nam sau bay tam", "user_id_hash": "u1"},
        {"node_id": "n2", "parent_id": "r", "text": "mot hai ba bon nam sau bay tam", "user_id_hash": "u2"},
        {"node_id": "n3", "parent_id": "r", "text": "ngan", "user_id_hash": "[removed]"},
        {"node_id": "n4", "parent_id": "r", "text": "ngan", "user_id_hash": "AutoModerator"},
        {"node_id": "n5", "parent_id": "r", "text": "ngan", "user_id_hash": "bot"},
    ]
    burst = compute_early_cascade_burst_stats(
        kept_nodes=kept_nodes,
        delays_k=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        depth_map={"r": 0, "n1": 1, "n2": 1, "n3": 1, "n4": 1, "n5": 1},
        edge_attr_rows=[[0.5, 0.1]] * 5,
        k_horizon=20,
    )
    # 2 unique valid users out of 2 valid users -> 1.0 (NOT 2/5 = 0.4)
    assert float(burst[0, 5].item()) == pytest.approx(1.0)

    # Build corpus with legacy 4D keys -> EarlyCascadeDataset must emit strictly 2D edge_attr
    node_rows, edge_rows, nlp_feats = _make_node_edge_corpus(
        num_cascades=8, replies_per_cascade=26, with_legacy_4d=True
    )
    n_tbl = build_nodes_table(node_rows)
    e_tbl = build_edges_table(edge_rows)
    save_cascade_parquet(n_tbl, e_tbl, output_dir=tmp_path, validate=True)

    ds_l0 = EarlyCascadeDataset(
        root_dir=tmp_path,
        k_horizon=10,
        split="all",
        lambda_polar=0.0,
        nodes_table=n_tbl,
        edges_table=e_tbl,
        nlp_features=nlp_feats,
        min_total_replies=25,
    )
    ds_l1 = EarlyCascadeDataset(
        root_dir=tmp_path,
        k_horizon=10,
        split="all",
        lambda_polar=1.0,
        nodes_table=n_tbl,
        edges_table=e_tbl,
        nlp_features=nlp_feats,
        min_total_replies=25,
    )
    assert ds_l0[0].edge_attr.shape[1] == 2
    assert not hasattr(ds_l0[0], "tier2_user_prior")
    # early_y_polar respects lambda_polar when stances vary
    assert not torch.allclose(ds_l0[0].early_y_polar, ds_l1[0].early_y_polar)

    # Causal Tier 2 -> 1 Prior Bridge: shape, Batch collation, future-mutation invariance, causal split isolation
    ds_all_prior = EarlyCascadeDataset(
        root_dir=tmp_path,
        k_horizon=10,
        split="all",
        nodes_table=n_tbl,
        edges_table=e_tbl,
        nlp_features=nlp_feats,
        min_total_replies=25,
        include_tier2_prior=True,
        tier2_prior_scope="causal_all",
    )
    assert ds_all_prior[0].tier2_user_prior.shape == (11, TIER2_PRIOR_DIM)
    # Earliest cascade c_00 has zero prior history -> all zeros
    assert torch.allclose(ds_all_prior[0].tier2_user_prior, torch.zeros((11, TIER2_PRIOR_DIM)))
    # Later cascade c_07 shares recurring users -> non-zero causal prior
    assert float(ds_all_prior[-1].tier2_user_prior.abs().sum().item()) > 0.0

    batched = Batch.from_data_list([ds_all_prior[0], ds_all_prior[-1]])
    assert batched.tier2_user_prior.shape == (22, TIER2_PRIOR_DIM)

    # Future mutation invariance: flip stances & add future nodes AFTER c_02's root_ts -> c_00..c_02 priors unchanged
    c2_root_ts = base_ts + timedelta(hours=12)
    mut_node_rows = [dict(r) for r in node_rows]
    mut_edge_rows = [dict(r) for r in edge_rows]
    mut_feats = {
        "node_embeddings": dict(nlp_feats["node_embeddings"]),
        "edge_stance": dict(nlp_feats["edge_stance"]),
        "edge_hostility": dict(nlp_feats["edge_hostility"]),
    }
    for e in mut_edge_rows:
        if e["created_at_utc"] >= c2_root_ts:
            mut_feats["edge_stance"][e["edge_id"]] = -1.0
            mut_feats["edge_hostility"][e["edge_id"]] = 1.0
    ds_mut_prior = EarlyCascadeDataset(
        root_dir=tmp_path,
        k_horizon=10,
        split="all",
        nodes_table=build_nodes_table(mut_node_rows),
        edges_table=build_edges_table(mut_edge_rows),
        nlp_features=mut_feats,
        min_total_replies=25,
        include_tier2_prior=True,
        tier2_prior_scope="causal_all",
    )
    for idx in range(3):
        assert torch.allclose(ds_all_prior[idx].tier2_user_prior, ds_mut_prior[idx].tier2_user_prior)

    # causal_split isolation: split="train" only indexes train topics even if a val/test topic happened earlier!
    custom_splits = {
        "train": {"topic_1", "topic_2", "topic_3", "topic_4", "topic_5"},
        "val": {"topic_0"},  # topic_0 is earliest!
        "test": {"topic_6", "topic_7"},
    }
    ds_tr_split = EarlyCascadeDataset(
        root_dir=tmp_path,
        k_horizon=10,
        split="train",
        nodes_table=n_tbl,
        edges_table=e_tbl,
        nlp_features=nlp_feats,
        min_total_replies=25,
        custom_topic_splits=custom_splits,
        include_tier2_prior=True,
        tier2_prior_scope="causal_split",
    )
    # Since topic_0 is in val, topic_1 (the earliest train cascade) must see zero prior history!
    c_topic1 = next(d for d in ds_tr_split if d.topic_id == "topic_1")
    assert torch.allclose(c_topic1.tier2_user_prior, torch.zeros_like(c_topic1.tier2_user_prior))


# ---------------------------------------------------------------------------
# 3. baselines.py & sbi_fagat.py: Unconfounding, Camp Dipole Root & Tier2Prior
# ---------------------------------------------------------------------------


def test_sbifagat_unconfounded_ablations_camp_dipole_root_honest_conf_and_tier2_prior() -> None:
    assert BaselineAttentionReadout is SBiAttentionReadout
    assert _BaselineAttentionReadout is BaselineAttentionReadout

    bi_egat = BiEdgeGATv2(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2)
    sd_keys = set(bi_egat.state_dict().keys())
    assert any(k.startswith("td_readout.") for k in sd_keys)
    assert any(k.startswith("bu_readout.") for k in sd_keys)

    # Unbiased symmetric att init
    conv = StanceGuidedFAConv(hidden_dim=16, edge_dim=2)
    assert not torch.allclose(conv.att[0, 32:].detach(), torch.tensor([0.6, -0.4]))

    # Orthogonal ablation flags
    m_base = SBiFAGAT(in_dim=16, hidden_dim=16, edge_dim=2, use_cascade_stats=True, use_tier2_prior=False)
    m_stance = SBiFAGAT(in_dim=16, hidden_dim=16, edge_dim=2, ablate_stance=True, use_cascade_stats=True)
    m_host = SBiFAGAT(in_dim=16, hidden_dim=16, edge_dim=2, ablate_hostility=True, use_cascade_stats=True)
    m_sem = SBiFAGAT(in_dim=16, hidden_dim=16, edge_dim=2, mask_semantic_stats=True, use_cascade_stats=True)
    m_topo = SBiFAGAT(in_dim=16, hidden_dim=16, edge_dim=2, mask_topology_stats=True, use_cascade_stats=True)
    assert m_stance.head.mask_semantic_stats is False
    assert m_host.head.mask_semantic_stats is False
    assert m_stance.use_polar_persistence_anchor is True
    assert m_sem.head.mask_semantic_stats is True
    assert m_topo.head.mask_topology_stats is True

    # No extra parameters when use_tier2_prior=False
    assert m_base.tier2_prior_proj is None and m_base.tier2_prior_gate is None

    # Camp Dipole includes root_index in mu_pos even when ALL replies oppose root
    ei = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long)
    g_all_opp = CascadeData(
        x=torch.randn(4, 16),
        struct_x=torch.rand(4, 10),
        edge_index=ei,
        edge_index_td=ei,
        edge_index_bu=ei[[1, 0]],
        edge_attr=torch.tensor([[-0.95, 0.8], [-0.9, 0.7], [0.8, 0.1]], dtype=torch.float32),
        cascade_stats=torch.rand(1, 8),
        early_y_polar=torch.tensor([0.5]),
        root_index=torch.tensor([0]),
    )
    m_dip = SBiFAGAT(in_dim=16, hidden_dim=16, edge_dim=2, use_camp_dipole=True, dropout=0.0)
    m_nodip = SBiFAGAT(in_dim=16, hidden_dim=16, edge_dim=2, use_camp_dipole=False, dropout=0.0)
    m_nodip.load_state_dict(
        {k: v for k, v in m_dip.state_dict().items() if k in m_nodip.state_dict()}, strict=True
    )
    m_dip.eval()
    m_nodip.eval()
    with torch.no_grad():
        # Force negative edge gates on all direct root->reply edges so sum_pos_edges == 0
        for conv in list(m_dip.td_convs) + (list(m_dip.bu_convs) if m_dip.bu_convs else []):
            conv.att.zero_()
            conv.cos_weight.zero_()
            conv.edge_mlp[2].weight.zero_()
            conv.edge_mlp[2].bias.fill_(-5.0)
        for conv in list(m_nodip.td_convs) + (list(m_nodip.bu_convs) if m_nodip.bu_convs else []):
            conv.load_state_dict(m_dip.td_convs[0].state_dict(), strict=False)
        out_dip = m_dip(g_all_opp)
        out_nodip = m_nodip(g_all_opp)
    # Root (+1) + opposing replies (-1) form both camps, so Camp Dipole shifts graph_emb!
    assert not torch.allclose(out_dip["graph_emb"], out_nodip["graph_emb"], atol=1e-5)

    # Honest confidence_score=None when predict_uncertainty=False
    diag = m_base.predict_with_diagnostics(g_all_opp)
    assert diag["confidence_score"] is None
    assert diag["uncertainty_available"] is False

    # SBiFAGAT + Tier2Prior consumes tier2_user_prior & degrades cleanly when missing
    m_prior = SBiFAGAT(
        in_dim=16,
        hidden_dim=16,
        edge_dim=2,
        use_cascade_stats=True,
        use_tier2_prior=True,
        tier2_prior_dim=TIER2_PRIOR_DIM,
    )
    g_with_prior = g_all_opp.clone()
    g_with_prior.tier2_user_prior = torch.randn(4, TIER2_PRIOR_DIM)
    out_p1 = m_prior(g_with_prior)
    out_p2 = m_prior(g_all_opp)  # missing tier2_user_prior -> zero-safe
    assert out_p1["y_polar_pred"].shape == (1,)
    assert out_p2["y_polar_pred"].shape == (1,)
    assert not torch.allclose(out_p1["graph_emb"], out_p2["graph_emb"])
    (out_p1["y_size_pred"].sum() + out_p1["y_polar_pred"].sum() + out_p1["rumor_logits"].sum()).backward()
    assert m_prior.tier2_prior_gate is not None and m_prior.tier2_prior_gate.grad is not None


# ---------------------------------------------------------------------------
# 4. sota_cascades.py & trainer.py tabular integration
# ---------------------------------------------------------------------------


def test_sota_cascades_tabular_and_neural_models_and_tiny_cohort(tmp_path: Path) -> None:
    seed_everything(42)
    node_rows, edge_rows, nlp_feats = _make_node_edge_corpus(num_cascades=10, replies_per_cascade=26)
    n_tbl = build_nodes_table(node_rows)
    e_tbl = build_edges_table(edge_rows)
    ds = EarlyCascadeDataset(
        root_dir=tmp_path,
        k_horizon=15,
        split="all",
        nodes_table=n_tbl,
        edges_table=e_tbl,
        nlp_features=nlp_feats,
        min_total_replies=25,
        include_tier2_prior=True,
    )
    cascades = list(ds)
    train_c, val_c, test_c = cascades[:6], cascades[6:8], cascades[8:10]

    # 1. CalibratedEarlyKPersistence outperforms constant shift on affine-transformed future targets
    affine_train = []
    affine_test = []
    for c in train_c:
        cc = c.clone()
        cc.y_polar = (0.5 * cc.early_y_polar + 0.35).clamp(0.0, 1.0)
        affine_train.append(cc)
    for c in test_c:
        cc = c.clone()
        cc.y_polar = (0.5 * cc.early_y_polar + 0.35).clamp(0.0, 1.0)
        affine_test.append(cc)
    cal = CalibratedEarlyKPersistence(num_rumor_classes=4).fit_dataset(affine_train, val_c, seed=42)
    cal_preds = cal.predict_dataset(affine_test)
    cal_rmse = float(np.sqrt(np.mean((cal_preds["y_polar_pred"] - np.array([float(c.y_polar.item()) for c in affine_test])) ** 2)))
    raw_rmse = float(np.sqrt(np.mean((np.array([float(c.early_y_polar.item()) for c in affine_test]) - np.array([float(c.y_polar.item()) for c in affine_test])) ** 2)))
    assert cal_rmse < raw_rmse + 1e-6

    # 2. RidgeBaseline & LightGBMBaseline on normal + tiny n=2 smoke cohort without rumor labels
    for cls_tab in (RidgeBaseline, LightGBMBaseline):
        m_tab = cls_tab(num_rumor_classes=4).fit_dataset(train_c, val_c, seed=42)
        p_tab = m_tab.predict_dataset(test_c)
        assert p_tab["y_size_pred"].shape == (2,)
        assert p_tab["y_polar_pred"].shape == (2,)
        assert p_tab["rumor_logits"].shape == (2, 4)
        assert np.all(np.isfinite(p_tab["y_polar_pred"]))

        # Tiny smoke cohort with y_rumor == -1
        tiny_tr = []
        for c in train_c[:2]:
            ct = c.clone()
            ct.y_rumor = torch.tensor([-1], dtype=torch.long)
            tiny_tr.append(ct)
        m_tiny = cls_tab(num_rumor_classes=4).fit_dataset(tiny_tr, tiny_tr, seed=42)
        p_tiny = m_tiny.predict_dataset(test_c[:1])
        assert np.all(np.isfinite(p_tiny["rumor_logits"]))

    # 3. PyTorch SOTA models: FriedkinJohnsenGNN, CasCN, CasFlow (including 0-edge graph batch)
    zero_edge_g = cascades[-1].clone()
    zero_edge_g.x = torch.randn(1, TOTAL_NODE_FEATURE_DIM)
    zero_edge_g.struct_x = torch.zeros(1, 10)
    zero_edge_g.edge_index = torch.empty((2, 0), dtype=torch.long)
    zero_edge_g.edge_index_td = torch.empty((2, 0), dtype=torch.long)
    zero_edge_g.edge_index_bu = torch.empty((2, 0), dtype=torch.long)
    zero_edge_g.edge_attr = torch.empty((0, 2))
    zero_edge_g.edge_stance = torch.empty((0, 1))
    zero_edge_g.early_node_root_stance = torch.zeros(1)
    zero_edge_g.tier2_user_prior = torch.zeros(1, TIER2_PRIOR_DIM)
    zero_edge_g.cascade_stats = torch.zeros(1, 8)
    zero_edge_g.early_y_polar = torch.tensor([0.0])
    zero_edge_g.root_index = torch.tensor([0])
    batch = Batch.from_data_list([cascades[-1], zero_edge_g])
    for mdl in (
        FriedkinJohnsenGNN(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, use_cascade_stats=True),
        CasCN(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, use_cascade_stats=True),
        CasFlow(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, use_cascade_stats=True, tier2_prior_dim=TIER2_PRIOR_DIM),
    ):
        out = mdl(batch)
        assert out["y_size_pred"].shape == (2,)
        assert out["y_polar_pred"].shape == (2,)
        assert out["rumor_logits"].shape == (2, 4)
        assert out["graph_emb"].shape == (2, 64)
        assert out["node_emb"].shape == (batch.num_nodes, 32)
        (out["y_size_pred"].sum() + out["y_polar_pred"].sum() + out["rumor_logits"].sum()).backward()

    # 4. Integrated run_multi_seed_benchmark mixing tabular + PyTorch models
    tr_l = DataLoader(train_c, batch_size=3, shuffle=True)
    va_l = DataLoader(val_c, batch_size=2, shuffle=False)
    te_l = DataLoader(test_c, batch_size=2, shuffle=False)
    bench = run_multi_seed_benchmark(
        {
            "BiEdgeGATv2": lambda: BiEdgeGATv2(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=16, use_cascade_stats=True),
            "Calibrated Early-K Persistence": lambda: CalibratedEarlyKPersistence(num_rumor_classes=4),
            "Ridge": lambda: RidgeBaseline(num_rumor_classes=4),
            "LightGBM": lambda: LightGBMBaseline(num_rumor_classes=4),
            "FriedkinJohnsen": lambda: FriedkinJohnsenGNN(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=16),
            "CasCN": lambda: CasCN(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=16),
            "CasFlow": lambda: CasFlow(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=16),
            "SBiFAGAT": lambda: SBiFAGAT(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=16, use_tier2_prior=True),
        },
        tr_l,
        va_l,
        te_l,
        seeds=[42, 100],
        epochs=2,
        patience=2,
        reference_model="SBiFAGAT",
        comparator_model="auto",
        device="cpu",
    )
    assert "Calibrated Early-K Persistence" in bench["summary_stats"]
    assert "CasFlow" in bench["summary_stats"]
    assert "Ridge" in bench["val_summary_stats"]
    # Test set only has 2 topics -> Cluster-Insufficient (<6 events), bootstrap p_values NaN!
    assert bench["nonparametric_test_label"] == "Cluster-Insufficient (<6 events)"
    assert all(math.isnan(v) for v in bench["bootstrap_p_values"].values())


# ---------------------------------------------------------------------------
# 5. metrics.py, trainer.py, reporting.py, auto_loop.py & scripts
# ---------------------------------------------------------------------------


def test_rumor_loss_scale_cluster_stats_directional_reporting_and_autoloop(tmp_path: Path) -> None:
    # 1. rumor_loss_scale eliminates 2.8x PHEME CE dominance
    graphs = []
    for i in range(12):
        graphs.append(
            CascadeData(
                x=torch.randn(4, 16),
                edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long),
                edge_index_td=torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long),
                edge_index_bu=torch.tensor([[1, 2, 3], [0, 0, 1]], dtype=torch.long),
                edge_attr=torch.tensor([[0.5, 0.1]] * 3, dtype=torch.float32),
                cascade_stats=torch.rand(1, 8),
                root_index=torch.tensor([0]),
                y_size=torch.tensor([2.0 + 0.3 * i]),
                y_polar=torch.tensor([0.1 + 0.07 * i]),
                y_rumor=torch.tensor([i % 4], dtype=torch.long),
                topic_id=f"t_{i // 2}",
            )
        )
    tr_l = DataLoader(graphs[:8], batch_size=4, shuffle=False)
    stats = _estimate_train_target_stats(tr_l, num_rumor_classes=4)
    assert stats["rumor_loss_scale"] < 0.6
    assert stats["base_rumor_loss"] * stats["rumor_loss_scale"] == pytest.approx(0.5, abs=0.05)

    # 2. Strict ban on IID bootstrap fallback when topic_ids has < 6 events (2 or 5 events)
    yt = np.linspace(0.1, 0.9, 20)
    for n_events in (2, 5):
        tids = [f"ev_{i % n_events}" for i in range(20)]
        p_ref = [{"y_size_true": yt, "y_size_pred": yt + 0.01, "y_polar_true": yt, "y_polar_pred": yt, "topic_ids": tids}]
        p_cmp = [{"y_size_true": yt, "y_size_pred": yt + 2.0, "y_polar_true": yt, "y_polar_pred": yt + 0.5, "topic_ids": tids}]
        pvals_small = _compute_sample_level_bootstrap_pvalues(
            p_ref, p_cmp, metric_keys=("msle", "rmse"), n_resamples=200
        )
        assert math.isnan(pvals_small["msle"]) and math.isnan(pvals_small["rmse"])

    # 3. compute_paired_wilcoxon & compute_cluster_paired_tests (>= 6 topics)
    w_p = compute_paired_wilcoxon([0.1, 0.12, 0.11, 0.13, 0.10, 0.14], [0.4, 0.42, 0.41, 0.43, 0.40, 0.44])
    assert 0.0 < w_p < 0.05

    tids_8 = [f"ev_{i // 3}" for i in range(24)]
    yt24 = np.linspace(1.0, 4.0, 24)
    yp24 = np.linspace(0.1, 0.9, 24)
    c_res = compute_cluster_paired_tests(
        [{"y_size_true": yt24, "y_size_pred": yt24 + 0.02, "y_polar_true": yp24, "y_polar_pred": yp24, "topic_ids": tids_8}],
        [{"y_size_true": yt24, "y_size_pred": yt24 + 0.80, "y_polar_true": yp24, "y_polar_pred": yp24 * 0.2, "topic_ids": tids_8}],
        metric_keys=("msle", "rmse"),
    )
    assert c_res["ttest"]["msle"] < 0.01
    assert c_res["wilcoxon"]["msle"] < 0.05

    # 4. Directional significance markers (+)** and (-)** in reporting.py
    mixed_bench = {
        "reference_model": "SBiFAGAT",
        "comparator_model": "BiGCN",
        "summary_stats": {
            "SBiFAGAT": {"msle": (0.10, 0.01), "pearson_r": (0.20, 0.01)},
            "BiGCN": {"msle": (0.50, 0.01), "pearson_r": (0.80, 0.01)},
        },
        "p_values": {"msle": 0.0001, "pearson_r": 0.0001},
        "adjusted_bootstrap_p_values": {"msle": 0.0005, "pearson_r": 0.0005},
    }
    sub_m = (("msle", "MSLE $\\downarrow$", True), ("pearson_r", "Pearson $r$ $\\uparrow$", False))
    md_mixed = build_markdown_table(mixed_bench, sub_m)
    row_sbi = md_mixed.splitlines()[2]
    assert "(+)**" in row_sbi and "(-)**" in row_sbi
    tex_mixed = export_latex_table(
        mixed_bench, tmp_path / "mixed.tex", caption="Mixed", label="tab:mixed", metrics=sub_m
    ).read_text(encoding="utf-8")
    assert "$^{+**}$" in tex_mixed and "$^{-**}$" in tex_mixed

    # 5. auto_loop handles confidence_score=None cleanly when predict_uncertainty=False
    va_l = DataLoader(graphs[8:], batch_size=2, shuffle=False)
    al_res = run_self_auditing_validation_loop(
        tr_l,
        va_l,
        in_dim=16,
        hidden_dim=16,
        edge_dim=2,
        predict_uncertainty=False,
        candidates=[{"lr": 1e-3, "eps": 0.5, "dropout": 0.1}],
        epochs=1,
        patience=1,
        output_dir=tmp_path / "autoloop",
        device="cpu",
    )
    assert al_res["regime_diagnostics"]["overall_val"]["mean_confidence"] is None

    # 6. run_experiments._build_loaders defaults to polar_target_mode="future_only"
    assert inspect.signature(run_exp_mod._build_loaders).parameters["polar_target_mode"].default == "future_only"

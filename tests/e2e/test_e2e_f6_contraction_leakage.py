"""Opaque-box E2E tests for Feature Area F6: Path Contraction, Schema & Zero-Leakage."""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import torch

from src.data.arctic_shift_vn import (
    build_vn_cascade_from_thread,
    hash_author_sha256,
    is_filler_junk_text,
    resolve_contracted_parent_id,
)
from src.data.pyg_dataset import (
    EarlyCascadeDataset,
    compute_embedding_cluster_splits,
    compute_temporal_topic_splits,
    inductive_topic_split,
)
from src.data.schema import (
    EDGES_ARROW_SCHEMA,
    NODES_ARROW_SCHEMA,
    SchemaValidationError,
    validate_cascade_contract,
    validate_nodes_table,
)
from tests.e2e.conftest import build_synthetic_processed_corpus


# ---------------------------------------------------------------------------
# Tier 1: Isolated Feature Coverage
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_f6_t1_dataset_defaults_future_only_and_k_truncation_invariance(tmp_path: Path) -> None:
    """Iron Law 2 & M3 contract: EarlyCascadeDataset defaults to future_only and isolates t > K from inputs."""
    ds_dir = tmp_path / "vn_syn"
    info = build_synthetic_processed_corpus(
        ds_dir, num_topics=3, cascades_per_topic=1, replies_per_cascade=28, min_total_replies_extracted=25
    )

    ds_base = EarlyCascadeDataset(root_dir=ds_dir, k_horizon=20, split="all", min_total_replies=25)
    assert ds_base.polar_target_mode == "future_only"
    base_item = ds_base[0]

    # Perturb replies strictly after K=20 (both text embeddings and edge_attr/teacher stances) + add extra future replies
    feat_p = dict(torch.load(ds_dir / "nlp_features.pt", map_location="cpu", weights_only=False))
    embs_p = feat_p["node_embeddings"].clone()
    node_map = feat_p["node_id_to_idx"]
    stance_p = dict(feat_p["edge_stance"])
    host_p = dict(feat_p["edge_hostility"])
    t_stance_p = dict(feat_p["teacher_edge_stance"])
    t_host_p = dict(feat_p["teacher_edge_hostility"])

    # Compute base root orientations on cas_0_0 so we can force unanimous +0.95 root-aligned stances for t > K
    from collections import deque

    base_nodes_cas0 = [r for r in info["nodes_rows"] if r["cascade_id"] == "cas_0_0"]
    base_edges_cas0 = [e for e in info["edges_rows"] if e["cascade_id"] == "cas_0_0"]
    root_id_cas0 = next(r["node_id"] for r in base_nodes_cas0 if r["is_root"])

    def _analytical_future_y_polar(
        c_nodes: list[dict[str, Any]],
        c_edges: list[dict[str, Any]],
        s_map: dict[str, float],
        k_horizon: int = 20,
    ) -> float:
        import math
        import numpy as np

        adj: dict[str, list[str]] = {str(n["node_id"]): [] for n in c_nodes}
        elookup: dict[tuple[str, str], str] = {}
        for e in c_edges:
            adj.setdefault(str(e["source_id"]), []).append(str(e["target_id"]))
            elookup[(str(e["source_id"]), str(e["target_id"]))] = str(e["edge_id"])
        depths = {root_id_cas0: 0}
        q = deque([root_id_cas0])
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in depths:
                    depths[v] = depths[u] + 1
                    q.append(v)
        non_roots = sorted(
            [n for n in c_nodes if not n["is_root"]],
            key=lambda n: (n["created_at_utc"], depths.get(str(n["node_id"]), 10**6), str(n["node_id"])),
        )
        kept_pairs = {
            (str(n["parent_id"]), str(n["node_id"]))
            for n in non_roots[:k_horizon]
        }
        orient = {root_id_cas0: 1.0}
        aligned_future: list[float] = []
        bfs = deque([root_id_cas0])
        while bfs:
            u = bfs.popleft()
            p_sgn = 1.0 if orient[u] >= 0.1 else (-1.0 if orient[u] <= -0.1 else 0.0)
            for v in adj.get(u, []):
                s_uv = float(s_map[elookup[(u, v)]])
                al = max(-1.0, min(1.0, p_sgn * s_uv))
                if v not in orient:
                    orient[v] = al
                    bfs.append(v)
                if (u, v) not in kept_pairs:
                    aligned_future.append(al)
        arr = np.asarray(aligned_future, dtype=np.float64)
        n = len(arr)
        std_s = math.sqrt(min(1.0, float(np.var(arr, ddof=1))))
        m_pos = float(np.mean(np.maximum(arr, 0.0)))
        m_neg = float(np.mean(np.maximum(-arr, 0.0)))
        bip = min(1.0, 2.0 * math.sqrt(max(0.0, (n / float(n - 1)) * m_pos * m_neg)))
        return float(0.5 * std_s + 0.5 * bip)

    expected_base_y_polar = _analytical_future_y_polar(base_nodes_cas0, base_edges_cas0, t_stance_p, k_horizon=20)

    # Determine early parent orientations to make all perturbed future stances (21..28) identically +0.95 root-aligned
    parent_by_nid = {str(e["target_id"]): (str(e["source_id"]), str(e["edge_id"])) for e in base_edges_cas0}
    early_orient = {root_id_cas0: 1.0}
    adj0: dict[str, list[str]] = {str(n["node_id"]): [] for n in base_nodes_cas0}
    for e in base_edges_cas0:
        adj0.setdefault(str(e["source_id"]), []).append(str(e["target_id"]))
    q0 = deque([root_id_cas0])
    while q0:
        u = q0.popleft()
        p_sgn = 1.0 if early_orient[u] >= 0.1 else (-1.0 if early_orient[u] <= -0.1 else 0.0)
        for v in adj0.get(u, []):
            _, eid_uv = parent_by_nid[v]
            early_orient[v] = p_sgn * float(t_stance_p[eid_uv])
            q0.append(v)

    for k in range(21, 29):
        nid = f"cas_0_0_c{k:02d}"
        eid = f"cas_0_0_e{k:02d}"
        embs_p[node_map[nid]] += 9.0
        par_id, _ = parent_by_nid[nid]
        par_sgn = 1.0 if early_orient[par_id] >= 0.1 else -1.0
        new_s = 0.95 * par_sgn
        stance_p[eid] = new_s
        t_stance_p[eid] = new_s
        host_p[eid] = 0.05
        t_host_p[eid] = 0.05
    feat_p["node_embeddings"] = embs_p
    feat_p["edge_stance"] = stance_p
    feat_p["edge_hostility"] = host_p
    feat_p["teacher_edge_stance"] = t_stance_p
    feat_p["teacher_edge_hostility"] = t_host_p

    # Also extend cascade 0 with 4 more future replies (root-attached, unanimous +0.95) so y_size changes too
    nodes_rows = list(info["nodes_rows"])
    edges_rows = list(info["edges_rows"])
    last_ts = max(r["created_at_utc"] for r in nodes_rows if r["cascade_id"] == "cas_0_0")
    root_ts = [r["created_at_utc"] for r in nodes_rows if r["cascade_id"] == "cas_0_0" and r["is_root"]][0]
    for extra in range(29, 33):
        ts = last_ts + timedelta(seconds=60 * extra)
        nid = f"cas_0_0_c{extra:02d}"
        eid = f"cas_0_0_e{extra:02d}"
        nodes_rows.append(
            {
                "node_id": nid,
                "cascade_id": "cas_0_0",
                "topic_id": "topic_00",
                "parent_id": "cas_0_0_r",
                "is_root": False,
                "text": "Extra future reply",
                "created_at_utc": ts,
                "lang": "vi",
                "source_dataset": "synthetic_e2e",
                "user_id_hash": "u_01",
                "rumor_label": None,
            }
        )
        edges_rows.append(
            {
                "edge_id": eid,
                "cascade_id": "cas_0_0",
                "source_id": "cas_0_0_r",
                "target_id": nid,
                "created_at_utc": ts,
                "time_delay_sec": float((ts - root_ts).total_seconds()),
                "edge_type": "reply",
            }
        )
        embs_p = torch.cat([embs_p, torch.ones((1, embs_p.size(1)))], dim=0)
        node_map[nid] = embs_p.size(0) - 1
        stance_p[eid] = 0.95
        t_stance_p[eid] = 0.95
        host_p[eid] = 0.05
        t_host_p[eid] = 0.05
        feat_p["edge_attr"] = torch.cat([feat_p["edge_attr"], torch.tensor([[0.95, 0.05]])], dim=0)
        feat_p["edge_id_to_idx"][eid] = feat_p["edge_attr"].size(0) - 1
    feat_p["node_embeddings"] = embs_p
    feat_p["node_id_to_idx"] = node_map

    pert_nodes_cas0 = [r for r in nodes_rows if r["cascade_id"] == "cas_0_0"]
    pert_edges_cas0 = [e for e in edges_rows if e["cascade_id"] == "cas_0_0"]
    expected_pert_y_polar = _analytical_future_y_polar(pert_nodes_cas0, pert_edges_cas0, t_stance_p, k_horizon=20)

    ds_pert = EarlyCascadeDataset(
        root_dir=ds_dir,
        k_horizon=20,
        split="all",
        min_total_replies=25,
        nodes_table=pa.Table.from_pylist(nodes_rows, schema=NODES_ARROW_SCHEMA),
        edges_table=pa.Table.from_pylist(edges_rows, schema=EDGES_ARROW_SCHEMA),
        nlp_features=feat_p,
    )
    pert_item = ds_pert[0]

    assert torch.equal(base_item.x, pert_item.x)
    assert torch.equal(base_item.struct_x, pert_item.struct_x)
    assert torch.equal(base_item.cascade_stats, pert_item.cascade_stats)
    assert torch.equal(base_item.edge_index_td, pert_item.edge_index_td)
    assert torch.equal(base_item.edge_index_bu, pert_item.edge_index_bu)
    assert torch.equal(base_item.edge_attr, pert_item.edge_attr)
    assert torch.equal(base_item.early_y_polar, pert_item.early_y_polar)
    assert float(pert_item.y_size.item()) > float(base_item.y_size.item())
    assert not torch.isclose(base_item.y_polar, pert_item.y_polar, atol=1e-4)
    assert not torch.isclose(pert_item.y_polar, pert_item.early_y_polar, atol=1e-4)
    assert float(base_item.y_polar.item()) == pytest.approx(expected_base_y_polar, abs=1e-5)
    assert float(pert_item.y_polar.item()) == pytest.approx(expected_pert_y_polar, abs=1e-5)


@pytest.mark.tier1
def test_f6_t1_lambda_polar_propagates_to_early_y_polar(tmp_path: Path) -> None:
    """M3 contract: EarlyCascadeDataset.lambda_polar controls both y_polar and early_y_polar."""
    ds_dir = tmp_path / "lam_ds"
    info = build_synthetic_processed_corpus(
        ds_dir, num_topics=2, cascades_per_topic=1, replies_per_cascade=26, min_total_replies_extracted=25
    )
    # Set one-sided varying positive stances (+0.1 / +0.9) so std_s > 0.3 while bipolar_balance == 0.0
    feat_p = dict(torch.load(ds_dir / "nlp_features.pt", map_location="cpu", weights_only=False))
    stance_p = {
        e["edge_id"]: (0.10 if idx % 2 == 0 else 0.90) for idx, e in enumerate(info["edges_rows"])
    }
    feat_p["edge_stance"] = stance_p
    feat_p["teacher_edge_stance"] = stance_p

    ds_0 = EarlyCascadeDataset(
        root_dir=ds_dir, k_horizon=20, split="all", lambda_polar=0.0, min_total_replies=25, nlp_features=feat_p
    )
    ds_1 = EarlyCascadeDataset(
        root_dir=ds_dir, k_horizon=20, split="all", lambda_polar=1.0, min_total_replies=25, nlp_features=feat_p
    )

    diffs = [
        abs(float(ds_0[i].early_y_polar.item()) - float(ds_1[i].early_y_polar.item()))
        for i in range(len(ds_0))
    ]
    assert any(d > 1e-5 for d in diffs), "early_y_polar ignored self.lambda_polar"


@pytest.mark.tier1
def test_f6_t1_causal_tier2_prior_strictly_pre_root_and_excludes_current(tmp_path: Path) -> None:
    """R2 contract: include_tier2_prior=True computes features strictly from ts < root_ts (never >= root_ts)."""
    ds_dir = tmp_path / "causal_bridge_ds"
    info = build_synthetic_processed_corpus(
        ds_dir, num_topics=4, cascades_per_topic=2, replies_per_cascade=26, min_total_replies_extracted=25
    )
    ds_base = EarlyCascadeDataset(
        root_dir=ds_dir, k_horizon=20, split="all", min_total_replies=25, include_tier2_prior=True
    )
    # Pick a mid/late cascade (index 3) so it has both prior (< root_ts) and future (>= root_ts) interactions
    target_idx = 3
    base_item = ds_base[target_idx]
    assert hasattr(base_item, "tier2_user_prior") and base_item.tier2_user_prior is not None
    assert base_item.tier2_user_prior.size(0) == base_item.x.size(0)
    assert torch.isfinite(base_item.tier2_user_prior).all()
    target_root_ts = float(base_item.root_ts)

    # Flip ALL edge stances on interactions occurring at ts >= target_root_ts (including current cascade!)
    feat_p = dict(torch.load(ds_dir / "nlp_features.pt", map_location="cpu", weights_only=False))
    stance_p = dict(feat_p["edge_stance"])
    host_p = dict(feat_p["edge_hostility"])
    for e_row in info["edges_rows"]:
        if e_row["created_at_utc"].timestamp() >= target_root_ts - 1e-6:
            stance_p[e_row["edge_id"]] = -stance_p[e_row["edge_id"]]
            host_p[e_row["edge_id"]] = 0.99
    feat_p["edge_stance"] = stance_p
    feat_p["edge_hostility"] = host_p
    if "teacher_edge_stance" in feat_p:
        feat_p["teacher_edge_stance"] = stance_p
    if "teacher_edge_hostility" in feat_p:
        feat_p["teacher_edge_hostility"] = host_p

    ds_pert = EarlyCascadeDataset(
        root_dir=ds_dir,
        k_horizon=20,
        split="all",
        min_total_replies=25,
        include_tier2_prior=True,
        nlp_features=feat_p,
    )
    pert_item = ds_pert[target_idx]
    assert torch.equal(base_item.tier2_user_prior, pert_item.tier2_user_prior), (
        "tier2_user_prior leaked future/current-cascade interactions (>= root_ts)"
    )


@pytest.mark.tier1
def test_f6_t1_path_contraction_preserves_short_stances_and_deleted_parent() -> None:
    """M4 contract: build_vn_cascade_from_thread(..., junk_mode='contract') preserves short stances & links safely."""
    base_ts = 1_700_000_000.0
    post = {
        "id": "vn_contract",
        "title": "Thảo luận về chính sách quy hoạch đô thị mới tại thành phố",
        "selftext": "Mọi người chia sẻ góc nhìn thực tế để cùng thảo luận.",
        "author": "root_author",
        "subreddit": "vozforums",
        "created_utc": base_ts,
    }
    comments: list[dict[str, Any]] = [
        {
            "fullname": "t1_c1",
            "parent_id": "t3_vn_contract",
            "link_id": "t3_vn_contract",
            "author": "u1",
            "text": "Đề xuất này có nhiều điểm cần phân tích kỹ hơn về hạ tầng.",
            "created_at_utc": base_ts + 10,
            "created_utc": base_ts + 10,
        },
        {
            "fullname": "t1_c2",
            "parent_id": "t1_c1",
            "link_id": "t3_vn_contract",
            "author": "[deleted]",
            "text": "[deleted]",
            "created_utc": base_ts + 20,
        },
        {
            "fullname": "t1_c3",
            "parent_id": "t1_c2",
            "link_id": "t3_vn_contract",
            "author": "u3",
            "text": "Đồng ý",
            "created_utc": base_ts + 30,
        },
        {
            "fullname": "t1_c4",
            "parent_id": "t1_c3",
            "link_id": "t3_vn_contract",
            "author": "u4",
            "text": "Phản đối",
            "created_utc": base_ts + 40,
        },
    ]
    for i in range(5, 29):
        comments.append(
            {
                "fullname": f"t1_c{i}",
                "parent_id": "t3_vn_contract" if i % 3 == 0 else f"t1_c{i - 1}",
                "link_id": "t3_vn_contract",
                "author": f"u{i}",
                "text": f"Bình luận tiếng Việt hợp lệ thứ {i} về giao thông và đô thị.",
                "created_utc": base_ts + 50 + i,
            }
        )

    res = build_vn_cascade_from_thread(post, comments, min_comments=20, min_depth=3, junk_mode="contract")
    assert res is not None
    nodes_rows, edges_rows, _ = res
    texts = {r["text"] for r in nodes_rows}
    assert "Đồng ý" in texts
    assert "Phản đối" in texts
    root_id = next(r["node_id"] for r in nodes_rows if r["is_root"])
    c1_id = next(r["node_id"] for r in nodes_rows if "hạ tầng" in r["text"])
    c3_id = next(r["node_id"] for r in nodes_rows if r["text"] == "Đồng ý")
    edge_pairs = {(e["source_id"], e["target_id"]) for e in edges_rows}
    assert (c1_id, c3_id) in edge_pairs, (
        f"Expected contracted relink ({c1_id}, {c3_id}) across [deleted] c2, got {edge_pairs}"
    )
    assert (root_id, c3_id) not in edge_pairs, (
        f"Contracted child {c3_id} wrongly collapsed directly to root {root_id}"
    )
    issues = validate_cascade_contract(
        pa.Table.from_pylist(nodes_rows, schema=NODES_ARROW_SCHEMA),
        pa.Table.from_pylist(edges_rows, schema=EDGES_ARROW_SCHEMA),
        raise_on_error=True,
    )
    assert issues == []


@pytest.mark.tier1
def test_f6_t1_topic_splits_inductive_temporal_and_embedding_disjoint(tmp_path: Path) -> None:
    """Inductive, temporal, and embedding_cluster topic splits produce pairwise-disjoint topic sets."""
    ds_dir = tmp_path / "split_ds"
    build_synthetic_processed_corpus(
        ds_dir, num_topics=9, cascades_per_topic=2, replies_per_cascade=12, min_total_replies_extracted=0
    )
    ds_all = EarlyCascadeDataset(root_dir=ds_dir, k_horizon=10, split="all", min_total_replies=0)
    items = list(ds_all)

    ind_idx = inductive_topic_split(items, seed=42)
    tr_t = {items[i].topic_id for i in ind_idx["train"]}
    va_t = {items[i].topic_id for i in ind_idx["val"]}
    te_t = {items[i].topic_id for i in ind_idx["test"]}
    assert tr_t and va_t and te_t
    assert tr_t.isdisjoint(va_t) and tr_t.isdisjoint(te_t) and va_t.isdisjoint(te_t)

    for split_dict in (
        compute_temporal_topic_splits(items),
        compute_embedding_cluster_splits(items, n_clusters=4, seed=42),
    ):
        tr, va, te = split_dict["train"], split_dict["val"], split_dict["test"]
        assert tr and va and te
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)


# ---------------------------------------------------------------------------
# Tier 2: Boundary, Corner & Negative Rejection Cases
# ---------------------------------------------------------------------------


@pytest.mark.tier2
def test_f6_t2_schema_rejects_leakage_columns_and_invalid_timestamps() -> None:
    """Iron Law 2 schema contract: forbidden leakage columns, naive/non-UTC timestamps, and future-parent cycles fail."""
    t_utc = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    bad_tbl = pa.table(
        {
            "node_id": ["r0"],
            "cascade_id": ["c0"],
            "topic_id": ["t0"],
            "parent_id": [None],
            "is_root": [True],
            "text": ["root"],
            "created_at_utc": pa.array([t_utc], type=pa.timestamp("us", tz="UTC")),
            "lang": ["vi"],
            "source_dataset": ["e2e"],
            "user_id_hash": ["u0"],
            "rumor_label": [None],
            "y_polar": [0.5],  # FORBIDDEN LEAKAGE COLUMN
        }
    )
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_nodes_table(bad_tbl)
        raise SchemaValidationError(validate_nodes_table(bad_tbl))
    assert any("FORBIDDEN" in str(iss.code) or "LEAKAGE" in str(iss.code) for iss in exc_info.value.issues)

    # Future-dated parent edge (child timestamp earlier than parent)
    nodes_ok = pa.Table.from_pylist(
        [
            {
                "node_id": "r0",
                "cascade_id": "c0",
                "topic_id": "t0",
                "parent_id": None,
                "is_root": True,
                "text": "root",
                "created_at_utc": t_utc,
                "lang": "vi",
                "source_dataset": "e2e",
                "user_id_hash": "u0",
                "rumor_label": None,
            },
            {
                "node_id": "c1",
                "cascade_id": "c0",
                "topic_id": "t0",
                "parent_id": "r0",
                "is_root": False,
                "text": "child before root",
                "created_at_utc": t_utc - timedelta(seconds=30),
                "lang": "vi",
                "source_dataset": "e2e",
                "user_id_hash": "u1",
                "rumor_label": None,
            },
        ],
        schema=NODES_ARROW_SCHEMA,
    )
    edges_bad = pa.Table.from_pylist(
        [
            {
                "edge_id": "e1",
                "cascade_id": "c0",
                "source_id": "r0",
                "target_id": "c1",
                "created_at_utc": t_utc - timedelta(seconds=30),
                "time_delay_sec": -30.0,
                "edge_type": "reply",
            }
        ],
        schema=EDGES_ARROW_SCHEMA,
    )
    with pytest.raises(SchemaValidationError):
        validate_cascade_contract(nodes_ok, edges_bad, raise_on_error=True)


@pytest.mark.tier2
def test_f6_t2_min_total_replies_mismatch_and_missing_teacher_edges(tmp_path: Path) -> None:
    """EarlyCascadeDataset rejects min_total_replies < min_total_replies_extracted and missing teacher edges."""
    ds_dir = tmp_path / "mismatch_ds"
    build_synthetic_processed_corpus(
        ds_dir, num_topics=3, cascades_per_topic=1, replies_per_cascade=26, min_total_replies_extracted=25
    )
    with pytest.raises(ValueError, match="min_total_replies"):
        EarlyCascadeDataset(root_dir=ds_dir, k_horizon=20, split="all", min_total_replies=5)

    feat = dict(torch.load(ds_dir / "nlp_features.pt", map_location="cpu", weights_only=False))
    feat.pop("teacher_edge_stance", None)
    feat.pop("teacher_edge_attr", None)
    with pytest.raises(ValueError, match="teacher"):
        EarlyCascadeDataset(
            root_dir=ds_dir,
            k_horizon=20,
            split="all",
            min_total_replies=25,
            nlp_features=feat,
            edge_source="teacher",
        )


@pytest.mark.tier2
def test_f6_t2_path_contraction_truncation_orphans_and_bot_filter() -> None:
    """M4/M6 contract: API-truncated orphans are pruned by default (not attached to root) and bots filtered."""
    assert is_filler_junk_text("I am a bot performing this action automatically", author_raw="AutoModerator")

    base_ts = 1_700_000_000.0
    post = {
        "id": "vn_orphan",
        "title": "Thảo luận về việc cắt tỉa bình luận mồ côi khi API bị giới hạn",
        "selftext": "Kiểm tra hành vi xử lý cây bình luận bị cắt.",
        "author": "root_u",
        "subreddit": "vozforums",
        "created_utc": base_ts,
    }
    comments: list[dict[str, Any]] = []
    for i in range(1, 28):
        comments.append(
            {
                "fullname": f"t1_ok{i}",
                "parent_id": "t3_vn_orphan" if i <= 4 else f"t1_ok{i - 1}",
                "link_id": "t3_vn_orphan",
                "author": f"u_{i}",
                "text": f"Bình luận tiếng Việt hợp lệ số {i} có đầy đủ nội dung.",
                "created_utc": base_ts + 10 * i,
            }
        )
    # Add an API-truncated orphan whose parent_id t1_missing_parent is NOT in comments
    comments.append(
        {
            "fullname": "t1_orphan_child",
            "parent_id": "t1_missing_parent",
            "link_id": "t3_vn_orphan",
            "author": "u_orphan",
            "text": "Bình luận mồ côi có cha bị cắt mất khỏi API",
            "created_utc": base_ts + 999,
        }
    )
    # Add an AutoModerator comment that must be dropped
    comments.append(
        {
            "fullname": "t1_bot_msg",
            "parent_id": "t3_vn_orphan",
            "link_id": "t3_vn_orphan",
            "author": "AutoModerator",
            "text": "Đây là thông báo tự động tiếng Việt của hệ thống kiểm duyệt.",
            "created_utc": base_ts + 5,
        }
    )

    res = build_vn_cascade_from_thread(post, comments, min_comments=20, min_depth=3, junk_mode="contract")
    assert res is not None
    nodes_rows, _, _ = res
    kept_ids = {r["node_id"] for r in nodes_rows}
    assert "t1_orphan_child" not in kept_ids
    assert "t1_bot_msg" not in kept_ids


@pytest.mark.tier2
def test_f6_t2_unique_user_ratio_excludes_deleted_authors(tmp_path: Path) -> None:
    """M6 contract: unique_user_ratio divides by valid hashed reply authors rather than deleted/None rows."""
    ds_dir = tmp_path / "uniq_ratio_ds"
    info = build_synthetic_processed_corpus(
        ds_dir, num_topics=1, cascades_per_topic=1, replies_per_cascade=25, min_total_replies_extracted=25
    )
    # Mark half of the early replies as [deleted] author with distinct valid users on the remaining half
    nodes_rows = list(info["nodes_rows"])
    reply_idx = 0
    for r in nodes_rows:
        if not r["is_root"]:
            reply_idx += 1
            if reply_idx <= 10:
                r["user_id_hash"] = None
            else:
                r["user_id_hash"] = f"distinct_user_{reply_idx:03d}"

    ds = EarlyCascadeDataset(
        root_dir=ds_dir,
        k_horizon=20,
        split="all",
        min_total_replies=25,
        nodes_table=pa.Table.from_pylist(nodes_rows, schema=NODES_ARROW_SCHEMA),
        edges_table=pa.Table.from_pylist(info["edges_rows"], schema=EDGES_ARROW_SCHEMA),
    )
    # In V_K (20 replies): 10 have None author, 10 have 10 completely distinct valid users -> unique_user_ratio == 1.0
    unique_user_ratio = float(ds[0].cascade_stats[0, 5].item())
    assert unique_user_ratio == pytest.approx(1.0, abs=1e-4)


@pytest.mark.tier2
def test_f6_t2_hash_author_pseudonymization_salted_and_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Author pseudonymization filters [deleted]/None, uses deterministic SHA-256 with salt, warns when unsalted."""
    import hashlib
    import src.data.arctic_shift_vn as arctic_shift_vn

    assert hash_author_sha256(None, salt="s1") is None
    assert hash_author_sha256("[deleted]", salt="s1") is None
    assert hash_author_sha256("[removed]", salt="s1") is None

    h1 = hash_author_sha256("alice", salt="secret_salt_1")
    h2 = hash_author_sha256("alice", salt="secret_salt_2")
    assert h1 is not None and h2 is not None and h1 != h2
    assert len(h1) == 64

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SBIF_HASH_SALT", raising=False)
    monkeypatch.setattr(arctic_shift_vn, "_SALT_WARNING_EMITTED", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h_unsalted = hash_author_sha256("user_a", salt="")
    assert h_unsalted == hashlib.sha256(b"user_a").hexdigest()
    assert len(h_unsalted) == 64
    assert any(issubclass(w.category, UserWarning) for w in caught), (
        f"Expected UserWarning for unsalted SHA-256 pseudonymization, got {caught!r}"
    )

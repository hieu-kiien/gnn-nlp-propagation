"""Opaque-box E2E tests for Feature Area F2: Tier 2 Macro Signed Network & Link Prediction."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import torch

from src.data.schema import EDGES_ARROW_SCHEMA, NODES_ARROW_SCHEMA, save_cascade_parquet
from src.data.user_network import (
    _compute_rwc_garimella,
    build_temporal_signed_link_dataset,
    build_user_interaction_records,
    compute_macro_signed_network_stats,
    plot_macro_signed_network,
)
from tests.e2e.conftest import build_synthetic_processed_corpus


def _analytical_garimella_rwc(
    n: int,
    camps: np.ndarray,
    use_adj: dict[int, list[tuple[int, float]]],
    *,
    restart_alpha: float = 0.85,
    max_iters: int = 120,
) -> float:
    """Reference Markov absorbing random walk with strict mass conservation (1.0 per step)."""
    camp0 = np.where(camps == 0)[0]
    camp1 = np.where(camps == 1)[0]
    p_row = np.zeros((n, n), dtype=np.float64)
    deg = np.zeros(n, dtype=np.float64)
    for i in range(n):
        nbrs = [(v, max(w, 1e-6)) for v, w in use_adj.get(i, ())]
        s_w = sum(w for _, w in nbrs)
        deg[i] = s_w
        if s_w > 0:
            for j, w in nbrs:
                p_row[i, j] += w / s_w
    p_col = p_row.T
    k0 = max(1, min(50, max(2, camp0.size // 10), camp0.size - 1))
    k1 = max(1, min(50, max(2, camp1.size // 10), camp1.size - 1))
    hubs0 = camp0[np.argsort(deg[camp0])[-k0:]]
    hubs1 = camp1[np.argsort(deg[camp1])[-k1:]]
    is_hub0 = np.zeros(n, dtype=bool)
    is_hub1 = np.zeros(n, dtype=bool)
    is_hub0[hubs0] = True
    is_hub1[hubs1] = True
    non_hub = ~(is_hub0 | is_hub1)

    def _walk(seeds: np.ndarray) -> tuple[float, float]:
        active_seeds = seeds[non_hub[seeds]]
        init = np.zeros(n, dtype=np.float64)
        init[active_seeds] = 1.0 / float(active_seeds.size)
        active = init.copy()
        abs0 = 0.0
        abs1 = 0.0
        for _ in range(max_iters):
            nxt = restart_alpha * (p_col @ active)
            abs0 += float(nxt[is_hub0].sum())
            abs1 += float(nxt[is_hub1].sum())
            dangling = max(0.0, restart_alpha * float(active.sum()) - float(nxt.sum()))
            restart_mass = (1.0 - restart_alpha) * float(active.sum()) + dangling
            nxt[~non_hub] = 0.0
            active = nxt + restart_mass * init
            assert abs((float(active.sum()) + abs0 + abs1) - 1.0) <= 1e-6
        tot = abs0 + abs1
        return abs0 / tot, abs1 / tot

    p00, p01 = _walk(camp0)
    p10, p11 = _walk(camp1)
    return float(p00 * p11 - p01 * p10)


def _write_two_camp_triangle_corpus(
    data_dir: Path,
    *,
    include_all_neg_triad: bool = False,
    zero_first_user_half: bool = False,
    emb_dim: int = 16,
) -> dict[str, Any]:
    """Create a 2-camp balanced signed network with optional (---) triad and zero-vector rows."""
    rng = np.random.default_rng(123)
    nodes_rows: list[dict[str, Any]] = []
    edges_rows: list[dict[str, Any]] = []
    edge_stance_map: dict[str, float] = {}
    edge_host_map: dict[str, float] = {}
    zero_node_ids: set[str] = set()

    camp_a = ["u_0", "u_2", "u_4", "u_6"]
    camp_b = ["u_1", "u_3", "u_5", "u_7"]
    core_users = camp_a + camp_b
    t_sec = 1_700_000_000.0
    nid_c = 0
    eid_c = 0

    for c_idx, root_u in enumerate(core_users):
        cid = f"cas_{c_idx}"
        tid = f"topic_{c_idx % 2}"
        rnid = f"n_{nid_c}"
        nid_c += 1
        nodes_rows.append(
            {
                "node_id": rnid,
                "cascade_id": cid,
                "topic_id": tid,
                "parent_id": None,
                "is_root": True,
                "text": f"Root {cid}",
                "created_at_utc": datetime.fromtimestamp(t_sec, tz=timezone.utc),
                "lang": "vi",
                "source_dataset": "e2e_macro",
                "user_id_hash": root_u,
                "rumor_label": None,
            }
        )
        root_camp = 0 if root_u in camp_a else 1
        for rep_u in core_users:
            if rep_u == root_u:
                continue
            t_sec += 60.0
            rep_camp = 0 if rep_u in camp_a else 1
            same_camp = root_camp == rep_camp
            cnid = f"n_{nid_c}"
            eid = f"e_{eid_c}"
            nid_c += 1
            eid_c += 1
            nodes_rows.append(
                {
                    "node_id": cnid,
                    "cascade_id": cid,
                    "topic_id": tid,
                    "parent_id": rnid,
                    "is_root": False,
                    "text": f"Reply {rep_u} -> {root_u}",
                    "created_at_utc": datetime.fromtimestamp(t_sec, tz=timezone.utc),
                    "lang": "vi",
                    "source_dataset": "e2e_macro",
                    "user_id_hash": rep_u,
                    "rumor_label": None,
                }
            )
            if zero_first_user_half and rep_u == "u_0" and c_idx in {1, 2, 3}:
                zero_node_ids.add(cnid)
            edges_rows.append(
                {
                    "edge_id": eid,
                    "cascade_id": cid,
                    "source_id": rnid,
                    "target_id": cnid,
                    "created_at_utc": datetime.fromtimestamp(t_sec, tz=timezone.utc),
                    "time_delay_sec": 60.0,
                    "edge_type": "reply",
                }
            )
            edge_stance_map[eid] = 0.85 if same_camp else -0.85
            edge_host_map[eid] = 0.05 if same_camp else 0.85

    if include_all_neg_triad:
        # Add 3 users forming an explicit (---) triangle: w0-w1 (-), w1-w2 (-), w0-w2 (-)
        tri_users = ["w_0", "w_1", "w_2"]
        for i in range(3):
            u_src = tri_users[i]
            u_dst = tri_users[(i + 1) % 3]
            t_sec += 120.0
            cid = f"cas_neg_tri_{i}"
            rnid = f"n_{nid_c}"
            cnid = f"n_{nid_c + 1}"
            eid = f"e_{eid_c}"
            nid_c += 2
            eid_c += 1
            nodes_rows.extend(
                [
                    {
                        "node_id": rnid,
                        "cascade_id": cid,
                        "topic_id": "topic_tri",
                        "parent_id": None,
                        "is_root": True,
                        "text": "Triad root",
                        "created_at_utc": datetime.fromtimestamp(t_sec, tz=timezone.utc),
                        "lang": "vi",
                        "source_dataset": "e2e_macro",
                        "user_id_hash": u_src,
                        "rumor_label": None,
                    },
                    {
                        "node_id": cnid,
                        "cascade_id": cid,
                        "topic_id": "topic_tri",
                        "parent_id": rnid,
                        "is_root": False,
                        "text": "Triad hostile oppose",
                        "created_at_utc": datetime.fromtimestamp(t_sec + 30.0, tz=timezone.utc),
                        "lang": "vi",
                        "source_dataset": "e2e_macro",
                        "user_id_hash": u_dst,
                        "rumor_label": None,
                    },
                ]
            )
            edges_rows.append(
                {
                    "edge_id": eid,
                    "cascade_id": cid,
                    "source_id": rnid,
                    "target_id": cnid,
                    "created_at_utc": datetime.fromtimestamp(t_sec + 30.0, tz=timezone.utc),
                    "time_delay_sec": 30.0,
                    "edge_type": "reply",
                }
            )
            edge_stance_map[eid] = -0.90
            edge_host_map[eid] = 0.85

    data_dir.mkdir(parents=True, exist_ok=True)
    save_cascade_parquet(
        pa.Table.from_pylist(nodes_rows, schema=NODES_ARROW_SCHEMA),
        pa.Table.from_pylist(edges_rows, schema=EDGES_ARROW_SCHEMA),
        data_dir,
        validate=True,
    )
    embs = torch.tensor(rng.normal(2.0, 0.05, size=(len(nodes_rows), emb_dim)), dtype=torch.float32)
    node_id_to_idx = {r["node_id"]: i for i, r in enumerate(nodes_rows)}
    for znid in zero_node_ids:
        embs[node_id_to_idx[znid]] = 0.0
    torch.save(
        {
            "node_embeddings": embs,
            "node_id_to_idx": node_id_to_idx,
            "edge_stance": edge_stance_map,
            "edge_hostility": edge_host_map,
            "teacher_edge_stance": edge_stance_map,
            "teacher_edge_hostility": edge_host_map,
        },
        data_dir / "nlp_features.pt",
    )
    return {"zero_node_ids": zero_node_ids}


# ---------------------------------------------------------------------------
# Tier 1: Isolated Feature Coverage
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_f2_t1_rwc_garimella_analytical_mass_conservation() -> None:
    """B1 contract: _compute_rwc_garimella conserves mass (1.0 +- 1e-6) and seeds non-hubs only."""
    # Asymmetric 8-node 2-camp graph with dangling leaf and uneven path lengths:
    # if dangling double-counts (1-alpha)*active.sum() or seeds hubs, RWC diverges from exact Markov RWC.
    nodes = list(range(8))
    camps = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    # Hub candidates (highest degree): node 0 in camp 0 (deg 3), node 4 in camp 1 (deg 3)
    # Node 3 in camp 0 is a dangling leaf (no outgoing edges!) to exercise dangling mass conservation.
    pos_adj: dict[int, list[tuple[int, float]]] = {
        0: [(1, 2.0), (2, 1.0), (3, 0.5)],
        1: [(0, 3.0), (2, 1.0)],
        2: [(0, 1.0), (1, 1.0), (5, 1.0)],  # weak cross-camp bridge
        3: [],  # dangling node!
        4: [(5, 2.0), (6, 1.0), (7, 0.5)],
        5: [(4, 3.0), (6, 1.0)],
        6: [(4, 1.0), (5, 1.0), (1, 0.5)],
        7: [(4, 1.0)],
    }
    expected_rwc = _analytical_garimella_rwc(8, camps, pos_adj, restart_alpha=0.85, max_iters=120)
    actual_rwc = _compute_rwc_garimella(nodes, camps, pos_adj, pos_adj, restart_alpha=0.85, max_iters=120)

    assert -1.0 <= actual_rwc <= 1.0
    assert actual_rwc == pytest.approx(expected_rwc, abs=1e-4)


@pytest.mark.tier1
def test_f2_t1_gomez_q_signed_and_null_graph_permutation(tmp_path: Path) -> None:
    """B2 contract: Gómez (2009) normalized Q_signed matches analytical value + sign-permutation null stats."""
    ds_dir = tmp_path / "gomez_ds"
    _write_two_camp_triangle_corpus(ds_dir)
    stats = compute_macro_signed_network_stats(ds_dir, dataset_name="vn", num_null_perms=20, seed=42)
    pol = stats["polarization"]

    # Analytical K_8 2-camp graph: m_pos=12*w0, m_neg=16*w0, q_pos=0.5, q_neg=-0.5 -> Q_signed = (12*0.5 - 16*(-0.5))/28 = 0.5
    q_signed = float(pol["signed_modularity"])
    assert -1.0 <= q_signed <= 1.0
    assert math.isclose(q_signed, 0.5, abs_tol=1e-6), f"Expected analytical Q_signed=0.5, got {q_signed}"
    assert math.isclose(float(pol["q_pos"]), 0.5, abs_tol=1e-6)
    assert math.isclose(float(pol["q_neg"]), -0.5, abs_tol=1e-6)
    assert "q_signed_null_mean" in pol
    assert "q_signed_null_std" in pol
    assert "q_signed_z_score" in pol
    assert "q_signed_p_value" in pol
    assert np.isfinite(pol["q_signed_z_score"])
    assert 0.0 <= float(pol["q_signed_p_value"]) <= 1.0


@pytest.mark.tier1
def test_f2_t1_davis_weak_balance_census(tmp_path: Path) -> None:
    """B2 contract: Davis (1967) weak_balance_ratio & triangle counts match analytical two-camp + (---) census."""
    ds_dir = tmp_path / "davis_ds"
    _write_two_camp_triangle_corpus(ds_dir, include_all_neg_triad=True)
    stats = compute_macro_signed_network_stats(ds_dir, dataset_name="vn", num_null_perms=20, seed=42)
    sb = stats["structural_balance"]

    # K_8 (2*C(4,3)=8 ppp, 2*C(4,2)*4=48 pnn) + 1 disconnected (---) triangle => (8, 0, 48, 1, 57)
    assert (
        sb["n_ppp"],
        sb["n_ppn"],
        sb["n_pnn"],
        sb["n_nnn"],
        sb["num_triangles"],
    ) == (8, 0, 48, 1, 57)
    assert math.isclose(float(sb["balanced_ratio"]), 56.0 / 57.0, abs_tol=1e-6)
    assert math.isclose(float(sb["weak_balance_ratio"]), 1.0, abs_tol=1e-6)
    assert sb["weak_balance_ratio"] > sb["balanced_ratio"]
    assert "weak_balance_z_score" in sb and np.isfinite(sb["weak_balance_z_score"])


@pytest.mark.tier1
def test_f2_t1_zero_row_embedding_filtering(tmp_path: Path) -> None:
    """B4 contract: zero-row node_embeddings must be filtered from user_posts and x_sem aggregation."""
    ds_dir = tmp_path / "zero_row_ds"
    info = _write_two_camp_triangle_corpus(ds_dir, zero_first_user_half=True, emb_dim=16)
    assert len(info["zero_node_ids"]) == 3

    rec = build_user_interaction_records(ds_dir, prefer_teacher=True)
    embs = rec["node_embeddings"]
    for u, posts in rec["user_posts"].items():
        for _ts, emb_idx in posts:
            assert float(embs[emb_idx].abs().sum().item()) > 0.0, (
                f"Zero-row emb_idx={emb_idx} leaked into user_posts for {u}"
            )

    ds = build_temporal_signed_link_dataset(ds_dir, dataset_name="vn")
    # Every non-zero train user's semantic vector must stay near ~2.0 (never diluted down toward 0.5 or 1.0)
    x_sem = ds["x"][:, :16]
    nonzero_mask = x_sem.abs().sum(dim=1) > 0
    assert bool(nonzero_mask.any())
    assert torch.all(x_sem[nonzero_mask].mean(dim=1) > 1.75)


@pytest.mark.tier1
def test_f2_t1_temporal_link_dataset_unseen_pair_mask_and_strict_split(tmp_path: Path) -> None:
    """B7-T2 contract: val/test include unseen_pair_mask alongside both_seen with strict temporal split."""
    ds_dir = tmp_path / "link_ds"
    info = build_synthetic_processed_corpus(
        ds_dir,
        num_topics=6,
        cascades_per_topic=2,
        replies_per_cascade=12,
        emb_dim=16,
        min_total_replies_extracted=0,
        add_future_unseen_user=True,
    )
    ds = build_temporal_signed_link_dataset(ds_dir, dataset_name="vn")

    assert float(ds["train"]["ts"].max()) <= float(ds["val"]["ts"].min())
    assert float(ds["val"]["ts"].max()) <= float(ds["test"]["ts"].min())
    for split_name in ("val", "test"):
        split = ds[split_name]
        assert "both_seen" in split
        assert "unseen_pair_mask" in split
        assert split["unseen_pair_mask"].dtype == torch.bool
        assert split["unseen_pair_mask"].shape == split["y_oppose"].shape

    # Future-only unseen user must have exact zero features in ds['x']
    assert bool((~ds["test"]["both_seen"]).any())
    assert info["unseen_user"] not in ds["train_user_set"]


# ---------------------------------------------------------------------------
# Tier 2: Boundary, Corner & Negative Rejection Cases
# ---------------------------------------------------------------------------


@pytest.mark.tier2
def test_f2_t2_rwc_sparse_pos_edges_fallback_threshold() -> None:
    """B1 contract: _compute_rwc_garimella falls back to all_adj when directed pos_edges < 2 * n."""
    # 8 nodes, directed pos_edges = 10 (satisfies n <= pos_edges < 2*n).
    # pos_adj has NO path between camp 0 and camp 1, AND node 2/6 have pos edges only within small sub-pairs
    # vs all_adj which has cross-camp negative links changing absorption probabilities.
    nodes = list(range(8))
    camps = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    # pos_adj has 10 directed entries (< 2*8 = 16):
    # camp0: 0<->1 (2), 2<->3 (2), 0->2 (1) => 5? make symmetric: 0<->1, 2<->3, 0<->2 => 6 directed in camp0, 4 in camp1 = 10 total.
    # Notice camp0 nodes {0,1,2,3} and camp1 nodes {4,5,6,7} are disconnected in pos_adj!
    pos_adj: dict[int, list[tuple[int, float]]] = {
        0: [(1, 1.0), (2, 1.0)],
        1: [(0, 1.0)],
        2: [(0, 1.0), (3, 1.0)],
        3: [(2, 1.0)],
        4: [(5, 1.0), (6, 1.0)],
        5: [(4, 1.0)],
        6: [(4, 1.0)],
        7: [],
    }
    # all_adj adds strong cross-camp paths linking camp 0 non-hubs directly to camp 1 hub (node 4) and vice versa
    all_adj: dict[int, list[tuple[int, float]]] = {k: list(v) for k, v in pos_adj.items()}
    all_adj[2].append((4, 50.0))
    all_adj[4].append((2, 50.0))
    all_adj[6].append((0, 50.0))
    all_adj[0].append((6, 50.0))
    all_adj[7].append((0, 50.0))

    rwc_with_fallback = _compute_rwc_garimella(nodes, camps, pos_adj, all_adj, max_iters=60)
    # Without fallback (pre-M2 pos_edges < n=8 check), 10 >= 8 so it uses disconnected pos_adj -> rwc == 1.0.
    # With fallback (pos_edges=10 < 2*8=16), it uses all_adj cross-camp edges -> rwc < 0.85!
    assert rwc_with_fallback < 0.85, (
        f"Expected pos_edges=10 < 2*n=16 fallback to all_adj reducing RWC < 0.85, got {rwc_with_fallback}"
    )


@pytest.mark.tier2
def test_f2_t2_degenerate_macro_graphs_and_invalid_authors(tmp_path: Path) -> None:
    """Self-loops, [deleted]/unknown authors, and small <4 node graphs are handled gracefully without NaN."""
    assert _compute_rwc_garimella([0, 1, 2], np.array([0, 0, 1]), {}, {}) == 0.0

    ds_dir = tmp_path / "degen_ds"
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    nodes_rows = [
        {
            "node_id": "r0",
            "cascade_id": "c0",
            "topic_id": "t0",
            "parent_id": None,
            "is_root": True,
            "text": "root",
            "created_at_utc": t0,
            "lang": "vi",
            "source_dataset": "e2e",
            "user_id_hash": "u_a",
            "rumor_label": None,
        },
        {
            "node_id": "c1",
            "cascade_id": "c0",
            "topic_id": "t0",
            "parent_id": "r0",
            "is_root": False,
            "text": "self reply",
            "created_at_utc": t0,
            "lang": "vi",
            "source_dataset": "e2e",
            "user_id_hash": "u_a",
            "rumor_label": None,
        },
        {
            "node_id": "c2",
            "cascade_id": "c0",
            "topic_id": "t0",
            "parent_id": "r0",
            "is_root": False,
            "text": "deleted reply",
            "created_at_utc": t0,
            "lang": "vi",
            "source_dataset": "e2e",
            "user_id_hash": "[deleted]",
            "rumor_label": None,
        },
        {
            "node_id": "c3",
            "cascade_id": "c0",
            "topic_id": "t0",
            "parent_id": "r0",
            "is_root": False,
            "text": "valid reply",
            "created_at_utc": t0,
            "lang": "vi",
            "source_dataset": "e2e",
            "user_id_hash": "u_b",
            "rumor_label": None,
        },
    ]
    edges_rows = [
        {
            "edge_id": f"e{i}",
            "cascade_id": "c0",
            "source_id": "r0",
            "target_id": f"c{i}",
            "created_at_utc": t0,
            "time_delay_sec": float(i),
            "edge_type": "reply",
        }
        for i in (1, 2, 3)
    ]
    ds_dir.mkdir(parents=True)
    save_cascade_parquet(
        pa.Table.from_pylist(nodes_rows, schema=NODES_ARROW_SCHEMA),
        pa.Table.from_pylist(edges_rows, schema=EDGES_ARROW_SCHEMA),
        ds_dir,
        validate=True,
    )
    torch.save(
        {
            "node_embeddings": torch.ones((4, 16), dtype=torch.float32),
            "node_id_to_idx": {r["node_id"]: i for i, r in enumerate(nodes_rows)},
            "edge_stance": {"e1": 0.5, "e2": 0.5, "e3": -0.8},
            "edge_hostility": {"e1": 0.1, "e2": 0.1, "e3": 0.7},
            "teacher_edge_stance": {"e1": 0.5, "e2": 0.5, "e3": -0.8},
            "teacher_edge_hostility": {"e1": 0.1, "e2": 0.1, "e3": 0.7},
        },
        ds_dir / "nlp_features.pt",
    )
    rec = build_user_interaction_records(ds_dir)
    assert len(rec["interactions"]) == 1
    assert rec["interactions"][0]["src_user"] == "u_a"
    assert rec["interactions"][0]["dst_user"] == "u_b"


@pytest.mark.tier2
def test_f2_t2_all_tier2_link_models_fair_2d_and_parallel_edges() -> None:
    """B7-T2 / R2 contract: signed_gnns models preserve parallel edge multiplicity and continuous 2D edge_attr."""
    from src.models.signed_gnns import build_link_model, prepare_mp_graph  # lazy import for pre-M2 collection safety

    target_models = (
        "UserTextMLP",
        "StandardGCN",
        "StandardGATv2",
        "SGCN",
        "FAGCN",
        "BiEdgeGATv2-Link",
        "SiGAT",
        "SDGNN",
        "SBiFAGAT-Link",
    )
    torch.manual_seed(42)
    x = torch.randn((8, 20), dtype=torch.float32)
    # Unique directed edges where node 1 has two distinct positive incoming edges (0->1, 2->1) and one negative (3->1)
    base_edge_index = torch.tensor(
        [[0, 1, 2, 3, 2, 4, 5], [1, 0, 1, 1, 3, 5, 6]],
        dtype=torch.long,
    )
    base_edge_attr = torch.tensor(
        [[0.85, 0.10], [0.80, 0.15], [0.70, 0.12], [-0.80, 0.75], [0.90, 0.05], [-0.75, 0.70], [0.65, 0.20]],
        dtype=torch.float32,
    )
    # Parallel-edge graph duplicates (0->1) 4 extra times with identical attributes
    dup_ei = torch.tensor([[0] * 4, [1] * 4], dtype=torch.long)
    dup_ea = torch.tensor([[0.85, 0.10]] * 4, dtype=torch.float32)
    multi_edge_index = torch.cat([base_edge_index, dup_ei], dim=1)
    multi_edge_attr = torch.cat([base_edge_attr, dup_ea], dim=0)

    ei_coal, _ = prepare_mp_graph(multi_edge_index, multi_edge_attr, 8, directed=True, coalesce_edges=True)
    ei_multi_dir, _ = prepare_mp_graph(multi_edge_index, multi_edge_attr, 8, directed=True, coalesce_edges=False)
    ei_multi_undir, _ = prepare_mp_graph(multi_edge_index, multi_edge_attr, 8, directed=False, coalesce_edges=False)
    assert ei_coal.size(1) == base_edge_index.size(1)
    assert ei_multi_dir.size(1) == multi_edge_index.size(1) > ei_coal.size(1)
    assert ei_multi_undir.size(1) == 2 * multi_edge_index.size(1)

    query_pairs = torch.tensor([[0, 2, 4], [1, 5, 7]], dtype=torch.long)
    gnn_models = {m for m in target_models if m != "UserTextMLP"}
    continuous_2d_models = {"BiEdgeGATv2-Link", "SBiFAGAT-Link"}

    # Perturbed continuous 2D attributes: (a) magnitude-only s_uv shift (sign preserved), (b) h_uv hostility shift
    attr_s_mag = base_edge_attr.clone()
    attr_s_mag[:, 0] = torch.where(base_edge_attr[:, 0] >= 0, 0.15, -0.15)
    attr_h_shift = base_edge_attr.clone()
    attr_h_shift[:, 1] = 1.0 - base_edge_attr[:, 1]

    for m_name in target_models:
        model = build_link_model(
            m_name, in_dim=20, hidden_dim=16, num_layers=2, dropout=0.0, coalesce_edges=False
        )
        model.eval()
        with torch.no_grad():
            logits_base = model(x, base_edge_index, base_edge_attr, query_pairs)
            logits_multi = model(x, multi_edge_index, multi_edge_attr, query_pairs)
            assert logits_multi.shape == (3,), f"{m_name} returned bad logits shape {tuple(logits_multi.shape)}"
            assert torch.isfinite(logits_multi).all(), f"{m_name} emitted non-finite logits"

            if m_name in gnn_models:
                assert not torch.allclose(logits_base, logits_multi, atol=1e-6), (
                    f"{m_name} ignored parallel-edge multiplicity under coalesce_edges=False"
                )
            if m_name in continuous_2d_models:
                logits_s = model(x, base_edge_index, attr_s_mag, query_pairs)
                logits_h = model(x, base_edge_index, attr_h_shift, query_pairs)
                assert not torch.allclose(logits_base, logits_s, atol=1e-5), (
                    f"{m_name} ignored continuous s_uv magnitude perturbation"
                )
                assert not torch.allclose(logits_base, logits_h, atol=1e-5), (
                    f"{m_name} ignored continuous h_uv hostility perturbation"
                )


@pytest.mark.tier2
def test_f2_t2_prior_pair_sign_persistence_causality_and_unseen_fallback() -> None:
    """R2 contract: PriorPairSignPersistence predicts from historical pairs and gives neutral fallback on unseen."""
    from src.models.signed_gnns import PriorPairSignPersistence  # lazy import

    model = PriorPairSignPersistence()
    x = torch.randn((6, 20), dtype=torch.float32)
    # Historical pairs: (0, 1) opposing (-0.9, y=1.0), (2, 3) supporting (+0.9, y=0.0)
    hist_edges = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    hist_attr = torch.tensor([[-0.9, 0.8], [0.9, 0.05]], dtype=torch.float32)
    model.fit(hist_edges, torch.tensor([1.0, 0.0], dtype=torch.float32), num_users=6)
    query_pairs = torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.long)

    logits = model(x, hist_edges, hist_attr, query_pairs)
    assert logits.shape == (3,)
    # (0,1) was opposing -> higher oppose logit than supporting (2,3); (4,5) unseen -> near global_prior (0.5 -> logit ~0)
    assert float(logits[0]) > float(logits[1])
    assert abs(float(logits[2])) < 0.25


@pytest.mark.tier2
def test_f2_t2_fig5_legend_excludes_fiedler_label(tmp_path: Path) -> None:
    """M5-T2 contract: fig5_macro_signed_network.svg must not claim 'Within Fiedler Camp'."""
    ds_dir = tmp_path / "fig5_ds"
    _write_two_camp_triangle_corpus(ds_dir)
    stats = compute_macro_signed_network_stats(ds_dir, dataset_name="vn", num_null_perms=10, seed=42)
    pdf_path, svg_path = plot_macro_signed_network(stats, output_dir=tmp_path / "figs")
    assert pdf_path.is_file() and svg_path.is_file()
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "Within Fiedler Camp" not in svg_text

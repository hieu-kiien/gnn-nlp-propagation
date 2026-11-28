"""Regression tests for R1 Tier 2 mathematical, data, and provenance fixes (B1, B2, B4, B3/M5, B7)."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import torch

from scripts.assemble_paper_artifacts import find_unescaped_underscores_in_latex
from src.data.schema import EDGES_ARROW_SCHEMA, NODES_ARROW_SCHEMA, save_cascade_parquet
from src.data.user_network import (
    _compute_q_signed_from_arrays,
    _compute_rwc_garimella,
    build_temporal_signed_link_dataset,
    build_user_interaction_records,
    compute_macro_signed_network_stats,
    format_macro_network_latex,
    plot_macro_signed_network,
)
from tests.test_macro_signed_network import _write_synthetic_polarized_corpus


def test_rwc_mass_conservation_and_fallback_2n() -> None:
    # 6-node 2-camp graph with dangling node (node 2 has no pos_adj edge, only in camp 0)
    lcc_nodes = [0, 1, 2, 3, 4, 5]
    camps = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    # 5 undirected positive edges -> pos_edges = 10 < 2*6 = 12 -> triggers all_adj fallback
    pos_adj_sparse: dict[int, list[tuple[int, float]]] = {
        0: [(1, 1.0), (2, 1.0)],
        1: [(0, 1.0)],
        2: [(0, 1.0)],
        3: [(4, 1.0), (5, 1.0)],
        4: [(3, 1.0)],
        5: [(3, 1.0), (4, 1.0)],
    }
    # Give node 1 & 4 cross-camp edges in all_adj, plus keep node 2 dangling in sparse case
    all_adj_sparse: dict[int, list[tuple[int, float]]] = {
        0: [(1, 1.0), (2, 1.0), (3, 1.0)],
        1: [(0, 1.0)],
        2: [(0, 1.0)],
        3: [(4, 1.0), (0, 1.0)],
        4: [(3, 1.0), (5, 1.0)],
        5: [(4, 1.0)],
    }
    # Force a true dangling node in all_adj by omitting outgoing from node 2 while keeping incoming;
    # 4 symmetric undirected positive edges among {0, 1, 3, 4, 5} -> pos_edges = 8 < 2*6 = 12 -> triggers all_adj fallback
    pos_adj_dangling: dict[int, list[tuple[int, float]]] = {
        0: [(1, 1.0)],
        1: [(0, 1.0)],
        2: [],
        3: [(4, 1.0), (5, 1.0)],
        4: [(3, 1.0), (5, 1.0)],
        5: [(3, 1.0), (4, 1.0)],
    }
    all_adj_dangling: dict[int, list[tuple[int, float]]] = {
        0: [(1, 1.0), (2, 1.0), (3, 1.0)],
        1: [(0, 1.0)],
        2: [],
        3: [(4, 1.0), (0, 1.0), (5, 1.0)],
        4: [(3, 1.0), (5, 1.0)],
        5: [(3, 1.0), (4, 1.0)],
    }

    rwc_scalar = _compute_rwc_garimella(lcc_nodes, camps, pos_adj_dangling, all_adj_dangling)
    assert isinstance(rwc_scalar, float)

    rwc_val, details = _compute_rwc_garimella(
        lcc_nodes,
        camps,
        pos_adj_dangling,
        all_adj_dangling,
        restart_alpha=0.85,
        max_iters=40,
        return_mass_trace=True,
    )
    assert rwc_val == pytest.approx(rwc_scalar, abs=1e-12)
    assert details["rwc_used_all_adj_fallback"] is True
    assert details["rwc_pos_undirected_edges"] == 4
    assert details["init_non_hub_only"] is True
    assert len(details["mass_trace_camp0"]) == 40
    assert len(details["mass_trace_camp1"]) == 40
    for m0, m1 in zip(details["mass_trace_camp0"], details["mass_trace_camp1"]):
        assert abs(m0 - 1.0) <= 1e-6
        assert abs(m1 - 1.0) <= 1e-6

    # Now provide 6 undirected positive edges -> pos_edges = 12 >= 2*6 -> stays on pos_adj
    pos_adj_dense: dict[int, list[tuple[int, float]]] = {
        0: [(1, 1.0), (2, 1.0)],
        1: [(0, 1.0), (2, 1.0)],
        2: [(0, 1.0), (1, 1.0)],
        3: [(4, 1.0), (5, 1.0)],
        4: [(3, 1.0), (5, 1.0)],
        5: [(3, 1.0), (4, 1.0)],
    }
    rwc_dense, dense_details = _compute_rwc_garimella(
        lcc_nodes,
        camps,
        pos_adj_dense,
        all_adj_sparse,
        restart_alpha=0.85,
        max_iters=30,
        return_mass_trace=True,
    )
    assert dense_details["rwc_used_all_adj_fallback"] is False
    assert dense_details["rwc_pos_undirected_edges"] == 6
    assert rwc_dense > 0.5
    for m0, m1 in zip(dense_details["mass_trace_camp0"], dense_details["mass_trace_camp1"]):
        assert abs(m0 - 1.0) <= 1e-6
        assert abs(m1 - 1.0) <= 1e-6


def test_gomez_2009_q_signed_bounds_and_null_graph(tmp_path: Path) -> None:
    # 1. Exact formula check on 4 nodes: camp 0={0,1}, camp 1={2,3}
    # Positive within-camp edges: (0,1) w=2.0, (2,3) w=2.0 -> q_pos = 0.5
    # Negative cross-camp edges: (0,2) w=1.0, (1,3) w=1.0 -> within_neg=0, expected=(2^2+2^2)/(4*2)=1.0 -> q_neg=-0.5
    # Gómez Q_signed = (4.0 * 0.5 - 2.0 * (-0.5)) / 6.0 = 3.0 / 6.0 = 0.5 (NOT 1.0!)
    camps = np.array([0, 0, 1, 1], dtype=np.int64)
    eu = np.array([0, 2, 0, 1], dtype=np.int64)
    ev = np.array([1, 3, 2, 3], dtype=np.int64)
    es = np.array([1, 1, -1, -1], dtype=np.int8)
    ew = np.array([2.0, 2.0, 1.0, 1.0], dtype=np.float64)
    q_s, q_p, q_n = _compute_q_signed_from_arrays(camps, eu, ev, es, ew)
    assert q_p == pytest.approx(0.5, abs=1e-9)
    assert q_n == pytest.approx(-0.5, abs=1e-9)
    assert q_s == pytest.approx(0.5, abs=1e-9)
    assert -1.0 <= q_s <= 1.0

    # All-negative cross-camp graph: Q_signed = -q_neg = +0.5 (bounded in [-1, 1])
    es_all_neg = np.array([-1, -1], dtype=np.int8)
    q_neg_only, qp0, qn0 = _compute_q_signed_from_arrays(
        camps, np.array([0, 1]), np.array([2, 3]), es_all_neg, np.array([1.0, 1.0])
    )
    assert qp0 == 0.0
    assert qn0 == pytest.approx(-0.5, abs=1e-9)
    assert q_neg_only == pytest.approx(0.5, abs=1e-9)
    assert -1.0 <= q_neg_only <= 1.0

    # Empty / zero-weight graph -> 0.0
    q_empty, _, _ = _compute_q_signed_from_arrays(
        camps, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8), np.empty(0)
    )
    assert q_empty == 0.0

    # 2. Null-graph sign-permutation evaluation on strongly polarized synthetic corpus
    ds_dir = tmp_path / "pol_q"
    _write_synthetic_polarized_corpus(ds_dir, add_unseen_future_user=False)
    stats_a = compute_macro_signed_network_stats(ds_dir, num_null_perms=20, num_q_null_perms=15, seed=42)
    stats_b = compute_macro_signed_network_stats(ds_dir, num_null_perms=20, num_q_null_perms=15, seed=42)
    pol = stats_a["polarization"]
    assert -1.0 <= pol["signed_modularity"] <= 1.0
    assert pol["q_signed_null_perms"] == 15
    assert np.isfinite(pol["q_signed_null_mean"])
    assert np.isfinite(pol["q_signed_null_std"])
    assert pol["q_signed_z_score"] > 0.0
    assert 0.0 < pol["q_signed_p_value"] <= 1.0
    assert pol["q_signed_z_score"] == pytest.approx(stats_b["polarization"]["q_signed_z_score"], abs=1e-12)


def test_davis_1967_weak_balance(tmp_path: Path) -> None:
    # Build K4 (4 users, all 6 pairs negative -> 4 (-,-,-) triangles, 0 Heider-balanced, 4 Davis-weak-balanced)
    ds_dir = tmp_path / "k4_neg"
    users = ["u0", "u1", "u2", "u3"]
    nodes_rows: list[dict[str, Any]] = []
    edges_rows: list[dict[str, Any]] = []
    stance_map: dict[str, float] = {}
    host_map: dict[str, float] = {}
    t_sec = 1_700_000_000.0
    nid_c = 0
    eid_c = 0
    for i in range(4):
        for j in range(i + 1, 4):
            t_sec += 60.0
            cid = f"c_{i}_{j}"
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
                        "topic_id": "t0",
                        "parent_id": None,
                        "is_root": True,
                        "text": "root",
                        "created_at_utc": datetime.fromtimestamp(t_sec, tz=timezone.utc),
                        "lang": "en",
                        "source_dataset": "pheme",
                        "user_id_hash": users[i],
                        "rumor_label": None,
                    },
                    {
                        "node_id": cnid,
                        "cascade_id": cid,
                        "topic_id": "t0",
                        "parent_id": rnid,
                        "is_root": False,
                        "text": "reply",
                        "created_at_utc": datetime.fromtimestamp(t_sec + 10.0, tz=timezone.utc),
                        "lang": "en",
                        "source_dataset": "pheme",
                        "user_id_hash": users[j],
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
                    "created_at_utc": datetime.fromtimestamp(t_sec + 10.0, tz=timezone.utc),
                    "time_delay_sec": 10.0,
                    "edge_type": "reply",
                }
            )
            stance_map[eid] = -0.9
            host_map[eid] = 0.8

    save_cascade_parquet(
        pa.Table.from_pylist(nodes_rows, schema=NODES_ARROW_SCHEMA),
        pa.Table.from_pylist(edges_rows, schema=EDGES_ARROW_SCHEMA),
        ds_dir,
        validate=True,
    )
    torch.save(
        {
            "node_embeddings": torch.ones((len(nodes_rows), 8), dtype=torch.float32),
            "node_id_to_idx": {r["node_id"]: idx for idx, r in enumerate(nodes_rows)},
            "teacher_edge_stance": stance_map,
            "teacher_edge_hostility": host_map,
        },
        ds_dir / "nlp_features.pt",
    )

    stats = compute_macro_signed_network_stats(ds_dir, num_null_perms=20, num_q_null_perms=5, seed=42)
    sb = stats["structural_balance"]
    assert sb["num_triangles"] == 4
    assert sb["n_nnn"] == 4
    assert sb["balanced_ratio"] == 0.0
    assert sb["weak_balance_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert np.isfinite(sb["weak_balance_null_mean"])
    assert np.isfinite(sb["weak_balance_null_std"])
    assert 0.0 < sb["weak_balance_p_value"] <= 1.0


def test_zero_vector_embedding_filtering_pheme_b4(tmp_path: Path) -> None:
    ds_dir = tmp_path / "b4_zero"
    nodes_rows: list[dict[str, Any]] = []
    edges_rows: list[dict[str, Any]] = []
    stance_map: dict[str, float] = {}
    host_map: dict[str, float] = {}
    embs_list: list[list[float]] = []

    t_sec = 1_700_000_000.0
    # Create 22 chronological interactions so temporal split has enough train/val/test edges.
    # User A posts 1 valid [2.0]*8 embedding + 3 zero [0.0]*8 embeddings in Train;
    # User B posts ONLY zero [0.0]*8 embeddings in Train.
    for idx in range(22):
        t_sec += 100.0
        cid = f"c_{idx}"
        rnid = f"r_{idx}"
        cnid = f"cnode_{idx}"
        eid = f"e_{idx}"
        src_u = "u_A" if idx < 4 else (f"u_{idx % 5}")
        dst_u = "u_B" if idx < 3 else (f"u_{(idx + 1) % 5}")
        if src_u == dst_u:
            dst_u = "u_C"

        nodes_rows.append(
            {
                "node_id": rnid,
                "cascade_id": cid,
                "topic_id": "t0",
                "parent_id": None,
                "is_root": True,
                "text": "root",
                "created_at_utc": datetime.fromtimestamp(t_sec, tz=timezone.utc),
                "lang": "en",
                "source_dataset": "pheme",
                "user_id_hash": src_u,
                "rumor_label": None,
            }
        )
        nodes_rows.append(
            {
                "node_id": cnid,
                "cascade_id": cid,
                "topic_id": "t0",
                "parent_id": rnid,
                "is_root": False,
                "text": "reply",
                "created_at_utc": datetime.fromtimestamp(t_sec + 10.0, tz=timezone.utc),
                "lang": "en",
                "source_dataset": "pheme",
                "user_id_hash": dst_u,
                "rumor_label": None,
            }
        )
        edges_rows.append(
            {
                "edge_id": eid,
                "cascade_id": cid,
                "source_id": rnid,
                "target_id": cnid,
                "created_at_utc": datetime.fromtimestamp(t_sec + 10.0, tz=timezone.utc),
                "time_delay_sec": 10.0,
                "edge_type": "reply",
            }
        )
        stance_map[eid] = 0.8 if idx % 2 == 0 else -0.8
        host_map[eid] = 0.1 if idx % 2 == 0 else 0.7

        # Embeddings: u_A gets [2.0]*8 on idx==0, [0.0]*8 on idx in {1,2,3}
        # u_B gets [0.0]*8 everywhere; others get [1.0]*8
        embs_list.append([2.0] * 8 if (src_u == "u_A" and idx == 0) else ([0.0] * 8 if src_u in {"u_A", "u_B"} else [1.0] * 8))
        embs_list.append([0.0] * 8 if dst_u == "u_B" else [1.0] * 8)

    save_cascade_parquet(
        pa.Table.from_pylist(nodes_rows, schema=NODES_ARROW_SCHEMA),
        pa.Table.from_pylist(edges_rows, schema=EDGES_ARROW_SCHEMA),
        ds_dir,
        validate=True,
    )
    torch.save(
        {
            "node_embeddings": torch.tensor(embs_list, dtype=torch.float32),
            "node_id_to_idx": {r["node_id"]: i for i, r in enumerate(nodes_rows)},
            "teacher_edge_stance": stance_map,
            "teacher_edge_hostility": host_map,
        },
        ds_dir / "nlp_features.pt",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rec = build_user_interaction_records(ds_dir, prefer_teacher=True)
    assert any(issubclass(w.category, UserWarning) for w in caught)
    assert rec["filtered_zero_embedding_posts"] >= 6

    ds = build_temporal_signed_link_dataset(ds_dir, dataset_name="pheme")
    assert ds["filtered_zero_embedding_posts"] >= 6
    assert torch.isfinite(ds["x"]).all()

    all_users = sorted({u for r in rec["interactions"] for u in (r["src_user"], r["dst_user"])})
    idx_a = all_users.index("u_A")
    idx_b = all_users.index("u_B")
    assert bool(ds["has_sem"][idx_a].item()) is True
    assert bool(ds["has_sem"][idx_b].item()) is False
    # User A's semantic embedding must NOT be diluted from [2.0]*8 to [0.5]*8
    assert torch.allclose(ds["x"][idx_a, :8], torch.full((8,), 2.0, dtype=torch.float32), atol=1e-5)
    assert torch.allclose(ds["x"][idx_b, :8], torch.zeros(8, dtype=torch.float32), atol=1e-6)


def test_unseen_pair_mask_and_provenance_legend_denominators(tmp_path: Path) -> None:
    ds_dir = tmp_path / "b7_prov"
    # Build 24 chronological interactions across 5 users (u_0..u_3 in train, u_4 unseen until test):
    # Train (first 17): pairs among {u_0, u_1, u_2, u_3} EXCLUDING {u_2, u_3}
    # Test (last 4):
    #   - (u_0, u_1) recurring same-direction
    #   - (u_1, u_0) recurring reversed-direction (while (u_0, u_1) was in Train)
    #   - (u_2, u_3) unseen pair among seen users -> unseen_pair_transductive == True
    #   - (u_0, u_4) cold-start user u_4 -> unseen_pair_mask == True, unseen_pair_transductive == False
    nodes_rows: list[dict[str, Any]] = []
    edges_rows: list[dict[str, Any]] = []
    stance_map: dict[str, float] = {}
    host_map: dict[str, float] = {}

    train_seq = [
        ("u_0", "u_1", 0.8),
        ("u_0", "u_2", -0.8),
        ("u_1", "u_2", 0.7),
        ("u_0", "u_3", -0.7),
        ("u_1", "u_3", 0.6),
    ] * 3 + [("u_0", "u_1", 0.8), ("u_0", "u_2", -0.8)]  # 17 train edges
    val_seq = [("u_0", "u_2", -0.8), ("u_1", "u_2", 0.7), ("u_0", "u_3", -0.7)]  # 3 val edges
    test_seq = [
        ("u_0", "u_1", 0.8),   # recurring same-direction
        ("u_1", "u_0", 0.8),   # recurring reversed-direction
        ("u_2", "u_3", -0.8),  # unseen pair, both seen
        ("u_0", "u_4", -0.9),  # cold-start user u_4
    ]
    all_seq = train_seq + val_seq + test_seq

    t_sec = 1_700_000_000.0
    for idx, (su, du, s_val) in enumerate(all_seq):
        t_sec += 100.0
        cid = f"c_{idx}"
        rnid = f"r_{idx}"
        cnid = f"cnode_{idx}"
        eid = f"e_{idx}"
        nodes_rows.extend(
            [
                {
                    "node_id": rnid,
                    "cascade_id": cid,
                    "topic_id": "t0",
                    "parent_id": None,
                    "is_root": True,
                    "text": "root",
                    "created_at_utc": datetime.fromtimestamp(t_sec, tz=timezone.utc),
                    "lang": "vi",
                    "source_dataset": "vn",
                    "user_id_hash": su,
                    "rumor_label": None,
                },
                {
                    "node_id": cnid,
                    "cascade_id": cid,
                    "topic_id": "t0",
                    "parent_id": rnid,
                    "is_root": False,
                    "text": "reply",
                    "created_at_utc": datetime.fromtimestamp(t_sec + 10.0, tz=timezone.utc),
                    "lang": "vi",
                    "source_dataset": "vn",
                    "user_id_hash": du,
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
                "created_at_utc": datetime.fromtimestamp(t_sec + 10.0, tz=timezone.utc),
                "time_delay_sec": 10.0,
                "edge_type": "reply",
            }
        )
        stance_map[eid] = s_val
        host_map[eid] = 0.1 if s_val > 0 else 0.8

    save_cascade_parquet(
        pa.Table.from_pylist(nodes_rows, schema=NODES_ARROW_SCHEMA),
        pa.Table.from_pylist(edges_rows, schema=EDGES_ARROW_SCHEMA),
        ds_dir,
        validate=True,
    )
    torch.save(
        {
            "node_embeddings": torch.ones((len(nodes_rows), 8), dtype=torch.float32),
            "node_id_to_idx": {r["node_id"]: i for i, r in enumerate(nodes_rows)},
            "teacher_edge_stance": stance_map,
            "teacher_edge_hostility": host_map,
        },
        ds_dir / "nlp_features.pt",
    )

    ds = build_temporal_signed_link_dataset(ds_dir, dataset_name="vn")
    for split_name in ("train", "val", "test"):
        for val in ds[split_name].values():
            assert isinstance(val, torch.Tensor)
            _ = val.to("cpu")

    test_sp = ds["test"]
    assert test_sp["unseen_pair_mask"].tolist() == [False, False, True, True]
    assert test_sp["unseen_pair_transductive"].tolist() == [False, False, True, False]
    assert test_sp["both_seen"].tolist() == [True, True, True, False]
    assert ds["edge_source"] == "teacher"
    assert ds["label_provenance"]["is_silver_nli_derived"] is True

    stats = compute_macro_signed_network_stats(ds_dir, dataset_name="vn", num_null_perms=10, num_q_null_perms=5)
    assert stats["edge_source"] == "teacher"
    assert stats["total_corpus_users"] == stats["num_users"]
    assert "largest_signed_wcc_ratio_active" in stats
    assert "largest_signed_wcc_ratio_corpus" in stats
    assert "denominator_notes" in stats

    tex = format_macro_network_latex([stats])
    assert find_unescaped_underscores_in_latex(tex) == []
    assert f"{stats['active_graph_users']:,} / {stats['num_users']:,}" in tex

    _, svg_path = plot_macro_signed_network(stats, output_dir=tmp_path / "figs")
    svg_txt = svg_path.read_text(encoding="utf-8")
    assert "Within Fiedler Camp" not in svg_txt
    assert "Within Signed Spectral Camp" in svg_txt

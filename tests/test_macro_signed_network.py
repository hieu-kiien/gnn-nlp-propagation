"""Tests for Tier 2 Macro Signed Social Network analytics and Temporal Signed Link GNNs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import torch

from scripts.assemble_paper_artifacts import find_unescaped_underscores_in_latex
from scripts.run_macro_signed_network import run_smoke_test
from src.data.schema import EDGES_ARROW_SCHEMA, NODES_ARROW_SCHEMA, save_cascade_parquet
from src.data.user_network import (
    build_temporal_signed_link_dataset,
    build_user_interaction_records,
    compute_macro_signed_network_stats,
    format_macro_network_latex,
    plot_macro_signed_network,
)


def _write_synthetic_polarized_corpus(
    data_dir: Path,
    *,
    emb_dim: int = 16,
    add_unseen_future_user: bool = False,
) -> dict[str, Any]:
    """Create a schema-valid 2-camp balanced triangle social network + nlp_features.pt."""
    rng = np.random.default_rng(123)
    nodes_rows: list[dict[str, Any]] = []
    edges_rows: list[dict[str, Any]] = []
    edge_stance_map: dict[str, float] = {}
    edge_host_map: dict[str, float] = {}

    # 8 core users partitioned cleanly into Camp A (even idx) and Camp B (odd idx).
    # Create explicit cascades rooted by each user so triangle triads (a, b, c) form cleanly:
    # same-camp pairs have +0.85 stance / 0.05 hostility; cross-camp have -0.85 stance / 0.85 hostility.
    camp_a = ["u_0", "u_2", "u_4", "u_6"]
    camp_b = ["u_1", "u_3", "u_5", "u_7"]
    core_users = camp_a + camp_b

    t_sec = 1_700_000_000.0
    nid_counter = 0
    eid_counter = 0

    for c_idx, root_u in enumerate(core_users):
        cid = f"cas_{c_idx}"
        tid = f"topic_{c_idx % 2}"
        root_nid = f"n_{nid_counter}"
        nid_counter += 1
        nodes_rows.append(
            {
                "node_id": root_nid,
                "cascade_id": cid,
                "topic_id": tid,
                "parent_id": None,
                "is_root": True,
                "text": f"Root discussion {cid}",
                "created_at_utc": datetime.fromtimestamp(t_sec, tz=timezone.utc),
                "lang": "vi",
                "source_dataset": "test_macro",
                "user_id_hash": root_u,
                "rumor_label": None,
            }
        )
        root_camp = 0 if root_u in camp_a else 1

        # Add replies from every other core user -> forms dense complete bipartite-signed graph + within-camp cliques
        for rep_u in core_users:
            if rep_u == root_u:
                # Include one self-loop reply on cascade 0 and one [deleted] author to verify filtering
                if c_idx == 0:
                    for bad_tag, bad_user in (("self", root_u), ("del", "[deleted]"), ("null", None), ("unk", "unknown")):
                        t_sec += 10.0
                        bad_nid = f"n_{nid_counter}"
                        bad_eid = f"e_{eid_counter}"
                        nid_counter += 1
                        eid_counter += 1
                        nodes_rows.append(
                            {
                                "node_id": bad_nid,
                                "cascade_id": cid,
                                "topic_id": tid,
                                "parent_id": root_nid,
                                "is_root": False,
                                "text": f"Filtered {bad_tag}",
                                "created_at_utc": datetime.fromtimestamp(t_sec, tz=timezone.utc),
                                "lang": "vi",
                                "source_dataset": "test_macro",
                                "user_id_hash": bad_user,
                                "rumor_label": None,
                            }
                        )
                        edges_rows.append(
                            {
                                "edge_id": bad_eid,
                                "cascade_id": cid,
                                "source_id": root_nid,
                                "target_id": bad_nid,
                                "created_at_utc": datetime.fromtimestamp(t_sec, tz=timezone.utc),
                                "time_delay_sec": 10.0,
                                "edge_type": "reply",
                            }
                        )
                        edge_stance_map[bad_eid] = 0.7
                        edge_host_map[bad_eid] = 0.05
                continue

            t_sec += 60.0
            rep_camp = 0 if rep_u in camp_a else 1
            same_camp = root_camp == rep_camp
            child_nid = f"n_{nid_counter}"
            eid = f"e_{eid_counter}"
            nid_counter += 1
            eid_counter += 1
            nodes_rows.append(
                {
                    "node_id": child_nid,
                    "cascade_id": cid,
                    "topic_id": tid,
                    "parent_id": root_nid,
                    "is_root": False,
                    "text": f"Reply by {rep_u} to {root_u}",
                    "created_at_utc": datetime.fromtimestamp(t_sec, tz=timezone.utc),
                    "lang": "vi",
                    "source_dataset": "test_macro",
                    "user_id_hash": rep_u,
                    "rumor_label": None,
                }
            )
            edges_rows.append(
                {
                    "edge_id": eid,
                    "cascade_id": cid,
                    "source_id": root_nid,
                    "target_id": child_nid,
                    "created_at_utc": datetime.fromtimestamp(t_sec, tz=timezone.utc),
                    "time_delay_sec": 60.0 * (eid_counter % 10 + 1),
                    "edge_type": "reply",
                }
            )
            edge_stance_map[eid] = 0.85 if same_camp else -0.85
            edge_host_map[eid] = 0.05 if same_camp else 0.85
        t_sec += 300.0

    unseen_user_id = "u_future_only"
    if add_unseen_future_user:
        # Final late cascade at the tail (> 85% quantile) involving an unseen test user
        for extra_i in range(4):
            t_sec += 1000.0
            c_id = f"cas_future_{extra_i}"
            rnid = f"n_{nid_counter}"
            nid_counter += 1
            cnid = f"n_{nid_counter}"
            nid_counter += 1
            eid = f"e_{eid_counter}"
            eid_counter += 1
            nodes_rows.append(
                {
                    "node_id": rnid,
                    "cascade_id": c_id,
                    "topic_id": "topic_future",
                    "parent_id": None,
                    "is_root": True,
                    "text": "Late future thread root",
                    "created_at_utc": datetime.fromtimestamp(t_sec, tz=timezone.utc),
                    "lang": "vi",
                    "source_dataset": "test_macro",
                    "user_id_hash": "u_0",
                    "rumor_label": None,
                }
            )
            nodes_rows.append(
                {
                    "node_id": cnid,
                    "cascade_id": c_id,
                    "topic_id": "topic_future",
                    "parent_id": rnid,
                    "is_root": False,
                    "text": "Late future user reply",
                    "created_at_utc": datetime.fromtimestamp(t_sec + 60.0, tz=timezone.utc),
                    "lang": "vi",
                    "source_dataset": "test_macro",
                    "user_id_hash": unseen_user_id,
                    "rumor_label": None,
                }
            )
            edges_rows.append(
                {
                    "edge_id": eid,
                    "cascade_id": c_id,
                    "source_id": rnid,
                    "target_id": cnid,
                    "created_at_utc": datetime.fromtimestamp(t_sec + 60.0, tz=timezone.utc),
                    "time_delay_sec": 60.0,
                    "edge_type": "reply",
                }
            )
            edge_stance_map[eid] = -0.90
            edge_host_map[eid] = 0.90

    nodes_tbl = pa.Table.from_pylist(nodes_rows, schema=NODES_ARROW_SCHEMA)
    edges_tbl = pa.Table.from_pylist(edges_rows, schema=EDGES_ARROW_SCHEMA)
    save_cascade_parquet(nodes_tbl, edges_tbl, data_dir, validate=True)

    embs = torch.tensor(rng.normal(0.0, 0.1, size=(len(nodes_rows), emb_dim)), dtype=torch.float32) + 2.0
    torch.save(
        {
            "node_embeddings": embs,
            "node_id_to_idx": {r["node_id"]: i for i, r in enumerate(nodes_rows)},
            "edge_stance": edge_stance_map,
            "edge_hostility": edge_host_map,
            "teacher_edge_stance": edge_stance_map,
            "teacher_edge_hostility": edge_host_map,
        },
        data_dir / "nlp_features.pt",
    )
    return {"unseen_user_id": unseen_user_id}


def test_user_interaction_filtering_and_polarization_stats(tmp_path: Path) -> None:
    ds_dir = tmp_path / "vn"
    _write_synthetic_polarized_corpus(ds_dir, add_unseen_future_user=False)

    rec = build_user_interaction_records(ds_dir, prefer_teacher=True)
    # 8 * 7 = 56 valid core user-user interactions; self-loop and deleted/None/unknown are filtered
    assert rec["edge_label_source"] == "teacher"
    assert len(rec["interactions"]) == 56
    for inter in rec["interactions"]:
        assert inter["src_user"] != inter["dst_user"]
        assert inter["src_user"] not in {"[deleted]", "unknown"}
        assert inter["dst_user"] not in {"[deleted]", "unknown"}

    stats = compute_macro_signed_network_stats(ds_dir, dataset_name="vn", num_null_perms=30, seed=42)
    assert stats["dataset"] == "vn"
    assert stats["num_users"] == 8
    sb = stats["structural_balance"]
    assert sb["num_triangles"] > 0
    assert sb["balanced_ratio"] == 1.0
    assert sb["balance_z_score"] > 0.0

    pol = stats["polarization"]
    assert pol["signed_modularity"] > 0.0
    assert pol["rwc"] > 0.0
    assert pol["cross_camp_neg_rate"] > pol["within_camp_neg_rate"]
    assert pol["cross_camp_uncivil_rate"] > pol["within_camp_uncivil_rate"]

    tex = format_macro_network_latex([stats])
    assert "\\begin{table" in tex
    assert find_unescaped_underscores_in_latex(tex) == []

    pdf_path, svg_path = plot_macro_signed_network(stats, output_dir=tmp_path / "figs")
    assert pdf_path.is_file() and pdf_path.stat().st_size > 0
    assert svg_path.is_file() and svg_path.stat().st_size > 0


def test_temporal_signed_link_dataset_zero_leakage(tmp_path: Path) -> None:
    ds_dir = tmp_path / "custom_folder"
    _write_synthetic_polarized_corpus(ds_dir, emb_dim=16, add_unseen_future_user=True)

    ds = build_temporal_signed_link_dataset(ds_dir, dataset_name="vn")
    assert ds["dataset"] == "vn"
    assert ds["in_dim"] == 20  # 16-D SpanPool + 4-D unsigned activity (zero degree-1 sign leak)
    assert ds["x"].shape == (ds["num_users"], 20)

    tr_ts = ds["train"]["ts"]
    va_ts = ds["val"]["ts"]
    te_ts = ds["test"]["ts"]
    assert float(tr_ts.max().item()) <= float(va_ts.min().item())
    assert float(va_ts.max().item()) <= float(te_ts.min().item())

    # Unseen late user in Test must appear in ~both_seen and have exact zero Train features
    both_seen = ds["test"]["both_seen"]
    assert bool((~both_seen).any())
    unseen_pairs = ds["test"]["pairs"][:, ~both_seen]
    train_users = ds["train_user_set"]
    unseen_u = [int(u) for u in unseen_pairs.reshape(-1).tolist() if int(u) not in train_users]
    assert len(unseen_u) > 0
    for u_idx in set(unseen_u):
        assert torch.allclose(ds["x"][u_idx], torch.zeros(20, dtype=torch.float32))


def test_macro_signed_network_cli_smoke_test() -> None:
    run_smoke_test()

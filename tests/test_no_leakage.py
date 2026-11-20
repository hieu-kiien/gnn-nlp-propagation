"""Leakage prevention, K-horizon truncation, and inductive topic split tests."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.data.pyg_dataset import (
    EarlyCascadeDataset,
    SUPPORTED_HORIZONS,
    UPFDAdapterTransform,
    inductive_topic_split,
)
from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet


def _build_synthetic_processed_dir(
    root_dir: Path,
    *,
    num_topics: int = 8,
    cascades_per_topic: int = 2,
    replies_per_cascade: int = 60,
    extra_tail_replies: int = 0,
    mutate_tail_features: bool = False,
) -> Path:
    tail_rng = torch.Generator().manual_seed(9999)
    base_time = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    rumor_labels = ["non-rumor", "true", "false", "unverified"]

    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    node_embeddings: dict[str, torch.Tensor] = {}
    edge_stance: dict[str, float] = {}
    edge_hostility: dict[str, float] = {}

    cascade_counter = 0
    for t_idx in range(num_topics):
        topic_id = f"topic_{t_idx:02d}"
        for c_sub in range(cascades_per_topic):
            cid = f"casc_{cascade_counter:03d}"
            label = rumor_labels[cascade_counter % len(rumor_labels)]
            cascade_counter += 1

            root_id = f"{cid}_n000"
            node_rows.append(
                {
                    "node_id": root_id,
                    "cascade_id": cid,
                    "topic_id": topic_id,
                    "parent_id": None,
                    "is_root": True,
                    "text": f"Root claim for {cid} on {topic_id}",
                    "created_at_utc": base_time,
                    "lang": "en" if t_idx % 2 == 0 else "vi",
                    "source_dataset": "pheme" if t_idx % 2 == 0 else "vn_social",
                    "user_id_hash": f"u_r_{cid}",
                    "rumor_label": label,
                }
            )
            node_embeddings[root_id] = torch.randn(
                768, generator=torch.Generator().manual_seed(100000 + cascade_counter * 1000)
            )

            total_replies = replies_per_cascade + extra_tail_replies
            for r_idx in range(1, total_replies + 1):
                nid = f"{cid}_n{r_idx:03d}"
                # Attach cleanly to earlier nodes to build multi-depth tree
                parent_num = (r_idx - 1) // 3
                pid = f"{cid}_n{parent_num:03d}"
                is_tail = r_idx > 20

                delay_sec = float(r_idx * 60)
                if mutate_tail_features and is_tail:
                    delay_sec += 86400.0 * 10.0

                ts = base_time + timedelta(seconds=delay_sec)
                node_rows.append(
                    {
                        "node_id": nid,
                        "cascade_id": cid,
                        "topic_id": topic_id,
                        "parent_id": pid,
                        "is_root": False,
                        "text": (
                            f"Altered future reply {r_idx}"
                            if (mutate_tail_features and is_tail)
                            else f"Reply {r_idx} to {pid}"
                        ),
                        "created_at_utc": ts,
                        "lang": "en" if t_idx % 2 == 0 else "vi",
                        "source_dataset": "pheme" if t_idx % 2 == 0 else "vn_social",
                        "user_id_hash": f"u_{cid}_{r_idx}",
                        "rumor_label": label,
                    }
                )

                if r_idx <= replies_per_cascade:
                    emb = torch.randn(
                        768,
                        generator=torch.Generator().manual_seed(
                            100000 + cascade_counter * 1000 + r_idx
                        ),
                    )
                else:
                    emb = torch.randn(768, generator=tail_rng)
                if mutate_tail_features and is_tail:
                    emb = -emb * 5.0
                node_embeddings[nid] = emb

                eid = f"e_{cid}_{r_idx:03d}"
                edge_rows.append(
                    {
                        "edge_id": eid,
                        "cascade_id": cid,
                        "source_id": pid,
                        "target_id": nid,
                        "created_at_utc": ts,
                        "time_delay_sec": delay_sec,
                        "edge_type": "reply",
                    }
                )
                base_stance = 0.7 if r_idx % 2 == 0 else -0.75
                if mutate_tail_features and is_tail:
                    base_stance = -0.95
                edge_stance[eid] = base_stance
                edge_hostility[eid] = 0.85 if base_stance < 0 else 0.10

    nodes_table = build_nodes_table(node_rows)
    edges_table = build_edges_table(edge_rows)
    save_cascade_parquet(nodes_table, edges_table, root_dir, validate=True)

    torch.save(
        {
            "node_embeddings": node_embeddings,
            "edge_stance": edge_stance,
            "edge_hostility": edge_hostility,
        },
        root_dir / "nlp_features.pt",
    )
    return root_dir


@pytest.mark.parametrize("k", list(SUPPORTED_HORIZONS))
def test_early_truncation_and_bidirectional_edges(tmp_path: Path, k: int) -> None:
    data_dir = _build_synthetic_processed_dir(tmp_path / f"k_{k}", replies_per_cascade=60)
    ds = EarlyCascadeDataset(root_dir=data_dir, k_horizon=k, split="all")
    assert len(ds) == 16

    sample = ds[0]
    assert sample.num_nodes == k + 1
    assert sample.x.shape == (k + 1, 770)
    assert sample.edge_index_td.shape == (2, k)
    assert sample.edge_index_bu.shape == (2, k)
    assert torch.equal(sample.edge_index_bu, sample.edge_index_td[[1, 0], :])
    assert sample.edge_attr.shape == (k, 2)
    assert sample.edge_stance.shape == (k, 1)
    assert sample.root_index.item() == 0
    assert math.isclose(float(sample.y_size.item()), math.log1p(60), rel_tol=1e-5)
    assert 0.0 <= float(sample.y_polar.item()) <= 1.0


def test_inductive_topic_split_zero_overlap(tmp_path: Path) -> None:
    data_dir = _build_synthetic_processed_dir(tmp_path / "split_data", num_topics=10, cascades_per_topic=2)

    train_ds = EarlyCascadeDataset(root_dir=data_dir, k_horizon=20, split="train", split_seed=42)
    val_ds = EarlyCascadeDataset(root_dir=data_dir, k_horizon=20, split="val", split_seed=42)
    test_ds = EarlyCascadeDataset(root_dir=data_dir, k_horizon=20, split="test", split_seed=42)

    assert len(train_ds) > 0 and len(val_ds) > 0 and len(test_ds) > 0
    assert len(train_ds) + len(val_ds) + len(test_ds) == 20

    # Strict zero intersection of topic_id and cascade_id across Train / Val / Test
    assert train_ds.topic_ids.isdisjoint(val_ds.topic_ids)
    assert train_ds.topic_ids.isdisjoint(test_ds.topic_ids)
    assert val_ds.topic_ids.isdisjoint(test_ds.topic_ids)

    assert train_ds.cascade_ids.isdisjoint(val_ds.cascade_ids)
    assert train_ds.cascade_ids.isdisjoint(test_ds.cascade_ids)
    assert val_ds.cascade_ids.isdisjoint(test_ds.cascade_ids)

    # Also test direct helper error on < 3 topics
    with pytest.raises(ValueError):
        inductive_topic_split([{"topic_id": "t0"}, {"topic_id": "t1"}])


def test_zero_data_leakage_future_perturbation_invariant(tmp_path: Path) -> None:
    dir_clean = _build_synthetic_processed_dir(
        tmp_path / "clean",
        num_topics=4,
        cascades_per_topic=1,
        replies_per_cascade=60,
        extra_tail_replies=0,
        mutate_tail_features=False,
    )
    dir_future_altered = _build_synthetic_processed_dir(
        tmp_path / "altered",
        num_topics=4,
        cascades_per_topic=1,
        replies_per_cascade=60,
        extra_tail_replies=40,
        mutate_tail_features=True,
    )

    ds_clean = EarlyCascadeDataset(root_dir=dir_clean, k_horizon=20, split="all")
    ds_altered = EarlyCascadeDataset(root_dir=dir_future_altered, k_horizon=20, split="all")

    for g_clean, g_alt in zip(ds_clean, ds_altered):
        # Targets MUST reflect post-hoc G_full differences
        assert not torch.isclose(g_clean.y_size, g_alt.y_size)
        assert not torch.isclose(g_clean.y_polar, g_alt.y_polar)

        # Input features & topology up to K=20 MUST be bitwise identical (Zero Leakage!)
        assert torch.equal(g_clean.x, g_alt.x)
        assert torch.equal(g_clean.struct_x, g_alt.struct_x)
        assert torch.equal(g_clean.cascade_stats, g_alt.cascade_stats)
        assert torch.equal(g_clean.edge_index_td, g_alt.edge_index_td)
        assert torch.equal(g_clean.edge_index_bu, g_alt.edge_index_bu)
        assert torch.equal(g_clean.edge_attr, g_alt.edge_attr)
        assert torch.equal(g_clean.edge_stance, g_alt.edge_stance)
        assert torch.equal(g_clean.root_index, g_alt.root_index)


def test_cascade_data_batching_and_upfd_adapter(tmp_path: Path) -> None:
    data_dir = _build_synthetic_processed_dir(tmp_path / "batch_test", num_topics=4, cascades_per_topic=2)
    ds = EarlyCascadeDataset(root_dir=data_dir, k_horizon=20, split="all")
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    assert torch.equal(batch.root_index, batch.ptr[:-1])
    assert batch.edge_index_td.max().item() < batch.num_nodes
    assert batch.edge_index_bu.max().item() < batch.num_nodes

    raw_upfd = Data(
        x=torch.randn(6, 768),
        edge_index=torch.tensor([[0, 0, 1, 2, 2], [1, 2, 3, 4, 5]], dtype=torch.long),
        y=torch.tensor([1], dtype=torch.long),
    )
    adapted = UPFDAdapterTransform()(raw_upfd)
    assert adapted.x.shape == (6, 768)
    assert adapted.root_index.item() == 0
    assert adapted.y_rumor.item() == 1
    assert torch.equal(adapted.edge_index_bu, adapted.edge_index_td[[1, 0], :])

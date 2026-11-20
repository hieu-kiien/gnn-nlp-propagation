"""CPU gate tests for Step 1 GNN core redesign: 20-cascade overfit, 50/50 dipole cancellation, author ping-pong vs crowd separation, fair 2d head parity, and uncertainty detach."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from scripts.export_dataset_stats import compute_dataset_statistics
from scripts.predict_live_thread import _load_thread_from_json
from src.data.arctic_shift_vn import build_vn_cascade_from_thread, hash_author_sha256
from src.data.pyg_dataset import (
    CASCADE_STATS_DIM,
    STRUCTURAL_NODE_DIM,
    CascadeData,
    EarlyCascadeDataset,
    compute_causal_structural_node_features,
)
from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet
from src.models.baselines import (
    BiGCN,
    EdgeGATv2,
    FAGCNBaseline,
    MultiTaskHead,
    StandardGATv2,
    StandardGCN,
    TemporalStanceMLP,
    TextMLP,
)
from src.models.sbi_fagat import SBiFAGAT, _propagate_root_aligned_edge_signal
from src.training.metrics import compute_multitask_loss, evaluate_predictions
from src.utils.seed import seed_everything


def _build_synthetic_20_cascade_dir(root_dir: Path, num_cascades: int = 20) -> Path:
    base_time = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
    rumor_labels = ["non-rumor", "true", "false", "unverified"]

    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    node_embeddings: dict[str, torch.Tensor] = {}
    edge_stance: dict[str, float] = {}
    edge_hostility: dict[str, float] = {}

    for c_idx in range(num_cascades):
        cid = f"overfit20_c{c_idx:02d}"
        tid = f"topic_{c_idx:02d}"
        lbl = rumor_labels[c_idx % len(rumor_labels)]
        # Vary total replies widely (25..100) so y_size has clear dynamic range
        n_replies = 25 + (c_idx * 4)
        # Distinct topic prototype per cascade & regime
        g_gen = torch.Generator().manual_seed(5000 + c_idx * 97)
        base_emb = torch.randn(768, generator=g_gen)

        rid = f"{cid}_r"
        node_rows.append(
            {
                "node_id": rid,
                "cascade_id": cid,
                "topic_id": tid,
                "parent_id": None,
                "is_root": True,
                "text": f"Root discussion {cid}",
                "created_at_utc": base_time + timedelta(hours=c_idx),
                "lang": "vi" if c_idx % 2 == 0 else "en",
                "source_dataset": "vn_social" if c_idx % 2 == 0 else "pheme",
                "user_id_hash": f"op_{cid}",
                "rumor_label": lbl,
            }
        )
        node_embeddings[rid] = base_emb.clone()

        # Assign distinct polarization regime per cascade
        regime = c_idx % 4
        for r_idx in range(1, n_replies + 1):
            nid = f"{cid}_n{r_idx:03d}"
            parent_num = (r_idx - 1) // 3
            pid = rid if parent_num == 0 else f"{cid}_n{parent_num:03d}"
            delay_sec = float(r_idx * (15 + (c_idx % 5) * 40))
            ts = base_time + timedelta(hours=c_idx, seconds=delay_sec)
            # Some cascades are dyadic ping-pong, others are multi-user crowds
            if c_idx % 2 == 0:
                uid = f"u_{cid}_{r_idx % 2}"
            else:
                uid = f"u_{cid}_{r_idx}"

            node_rows.append(
                {
                    "node_id": nid,
                    "cascade_id": cid,
                    "topic_id": tid,
                    "parent_id": pid,
                    "is_root": False,
                    "text": f"Reply {r_idx} on {cid}",
                    "created_at_utc": ts,
                    "lang": "vi" if c_idx % 2 == 0 else "en",
                    "source_dataset": "vn_social" if c_idx % 2 == 0 else "pheme",
                    "user_id_hash": uid,
                    "rumor_label": None,
                }
            )
            noise = 0.25 * torch.randn(768, generator=torch.Generator().manual_seed(70000 + c_idx * 200 + r_idx))
            sign = 1.0 if (r_idx % 2 == 0) else -1.0
            node_embeddings[nid] = (0.5 * sign * base_emb) + noise if regime == 0 else (base_emb + noise)

            eid = f"{pid}->{nid}"
            edge_rows.append(
                {
                    "edge_id": eid,
                    "cascade_id": cid,
                    "source_id": pid,
                    "target_id": nid,
                    "created_at_utc": ts,
                    "time_delay_sec": float(delay_sec if parent_num == 0 else 30.0),
                    "edge_type": "reply",
                }
            )
            if regime == 0:
                # Polarized 50/50 split on direct root-aligned edges
                s_uv = 0.85 if (parent_num == 0 and r_idx % 2 == 0) else (-0.85 if parent_num == 0 else 0.75)
                h_uv = 0.70 if s_uv < 0 else 0.08
            elif regime == 1:
                # Unanimous support
                s_uv = 0.85
                h_uv = 0.05
            elif regime == 2:
                # Mild mixed conflict
                s_uv = -0.75 if r_idx % 3 == 0 else 0.75
                h_uv = 0.35
            else:
                # Neutral chatter
                s_uv = 0.02
                h_uv = 0.04
            edge_stance[eid] = s_uv
            edge_hostility[eid] = h_uv

    save_cascade_parquet(build_nodes_table(node_rows), build_edges_table(edge_rows), root_dir, validate=True)
    torch.save(
        {
            "node_embeddings": node_embeddings,
            "edge_stance": edge_stance,
            "edge_hostility": edge_hostility,
        },
        root_dir / "nlp_features.pt",
    )
    return root_dir


def test_sbifagat_overfits_20_cascades_joint_tasks(tmp_path: Path) -> None:
    seed_everything(42)
    data_dir = _build_synthetic_20_cascade_dir(tmp_path / "overfit20", num_cascades=20)
    ds = EarlyCascadeDataset(
        root_dir=data_dir,
        k_horizon=20,
        split="all",
        min_total_replies=25,
        polar_target_mode="future_only",
    )
    assert len(ds) == 20
    assert ds[0].struct_x.shape == (21, STRUCTURAL_NODE_DIM)

    batch = next(iter(DataLoader(ds, batch_size=20, shuffle=False)))
    model = SBiFAGAT(
        in_dim=770,
        hidden_dim=64,
        num_layers=2,
        edge_dim=2,
        num_rumor_classes=4,
        dropout=0.0,
        use_cascade_stats=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=0.0)

    model.train()
    for step in range(160):
        optimizer.zero_grad()
        outputs = model(batch)
        loss = compute_multitask_loss(
            outputs,
            batch,
            alpha_size=1.0,
            alpha_polar=4.0,
            alpha_rumor=1.0,
        )
        loss.backward()
        if step == 0:
            for name, param in model.named_parameters():
                assert param.requires_grad
                assert param.grad is not None, f"Missing gradient on {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient on {name}"
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_out = model(batch)
    metrics = evaluate_predictions(
        y_size_true=batch.y_size.numpy(),
        y_size_pred=final_out["y_size_pred"].numpy(),
        y_polar_true=batch.y_polar.numpy(),
        y_polar_pred=final_out["y_polar_pred"].numpy(),
        y_rumor_true=batch.y_rumor.numpy(),
        rumor_logits=final_out["rumor_logits"].numpy(),
    )
    assert metrics["r2_size"] > 0.95
    assert metrics["r2_polar"] > 0.95
    assert metrics["macro_f1"] == 1.0


def test_camp_dipole_resolves_50_50_polarization_cancellation() -> None:
    seed_everything(7)
    torch.manual_seed(7)
    num_replies = 20
    num_nodes = num_replies + 1
    src = [0] * num_replies
    dst = list(range(1, num_nodes))
    ei_td = torch.tensor([src, dst], dtype=torch.long)
    ei_bu = ei_td[[1, 0], :]

    x_root = torch.randn(1, 770) * 0.25
    v_camp = torch.zeros(1, 770)
    v_camp[0, :384] = 1.2
    v_camp[0, 384:768] = -0.8
    v_neutral = torch.zeros(1, 770)

    def _make_tree(child_x: torch.Tensor, stances: list[float], host: list[float], y_pol: float, y_sz: float, y_rm: int) -> Any:
        from src.data.pyg_dataset import CascadeData

        x_full = torch.cat([x_root, child_x], dim=0)
        x_full[:, 768] = torch.linspace(0.0, 0.3, steps=num_nodes)
        x_full[:, 769] = torch.tensor([0.0] + [0.05] * num_replies)
        ea = torch.tensor(list(zip(stances, host)), dtype=torch.float32)
        return CascadeData(
            x=x_full,
            edge_index=ei_td,
            edge_index_td=ei_td,
            edge_index_bu=ei_bu,
            edge_attr=ea,
            edge_stance=ea[:, 0:1],
            root_index=torch.tensor([0], dtype=torch.long),
            y_size=torch.tensor([y_sz], dtype=torch.float32),
            y_polar=torch.tensor([y_pol], dtype=torch.float32),
            y_rumor=torch.tensor([y_rm], dtype=torch.long),
        )

    # C1: 50/50 split (+v_camp vs -v_camp)
    c1_x = torch.cat([v_camp.expand(10, -1), (-v_camp).expand(10, -1)], dim=0)
    c1 = _make_tree(c1_x, [0.9] * 10 + [-0.9] * 10, [0.05] * 10 + [0.80] * 10, 1.0, 4.0, 0)
    # C2: 100% unanimous support
    c2 = _make_tree(v_camp.expand(num_replies, -1), [0.9] * num_replies, [0.05] * num_replies, 0.0, 3.5, 1)
    # C3: 100% unanimous refute
    c3 = _make_tree((-v_camp).expand(num_replies, -1), [-0.9] * num_replies, [0.20] * num_replies, 0.0, 3.5, 2)
    # C4: 100% neutral
    c4 = _make_tree(v_neutral.expand(num_replies, -1), [0.0] * num_replies, [0.02] * num_replies, 0.0, 3.3, 3)

    batch = next(iter(DataLoader([c1, c2, c3, c4], batch_size=4, shuffle=False)))
    model = SBiFAGAT(in_dim=770, hidden_dim=64, num_layers=2, edge_dim=2, dropout=0.0)

    model.eval()
    with torch.no_grad():
        init_out = model(batch)
    dipole_dist = torch.norm(init_out["graph_emb"][0] - init_out["graph_emb"][3], p=2).item()
    assert dipole_dist > 0.1, f"50/50 polarized graph collapsed onto neutral graph: dist={dipole_dist}"

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    model.train()
    for _ in range(55):
        optimizer.zero_grad()
        out = model(batch)
        loss = compute_multitask_loss(out, batch, alpha_size=1.0, alpha_polar=8.0, alpha_rumor=2.0)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_out = model(batch)
    y_pol_pred = final_out["y_polar_pred"]
    rumor_pred = final_out["rumor_logits"].argmax(dim=-1)
    assert y_pol_pred[0].item() > 0.85
    assert y_pol_pred[1].item() < 0.12
    assert y_pol_pred[2].item() < 0.12
    assert y_pol_pred[3].item() < 0.12
    assert int(rumor_pred[1].item()) == 1
    assert int(rumor_pred[2].item()) == 2


def test_struct_x_separates_dyadic_ping_pong_from_20_user_crowd(tmp_path: Path) -> None:
    seed_everything(11)
    base_time = datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc)
    shared_root_emb = torch.randn(768)
    shared_reply_embs = [torch.randn(768) for _ in range(25)]

    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    node_embeddings: dict[str, torch.Tensor] = {}
    edge_stance: dict[str, float] = {}
    edge_hostility: dict[str, float] = {}

    specs = [
        ("casc_pingpong", "top_pp", 25, 0),
        ("casc_crowd", "top_cr", 150, 1),
    ]
    for cid, tid, total_rep, is_crowd in specs:
        rid = f"{cid}_r"
        node_rows.append(
            {
                "node_id": rid,
                "cascade_id": cid,
                "topic_id": tid,
                "parent_id": None,
                "is_root": True,
                "text": "Identical claim",
                "created_at_utc": base_time,
                "lang": "vi",
                "source_dataset": "vn_social",
                "user_id_hash": "op_user_hash",
                "rumor_label": "true",
            }
        )
        node_embeddings[rid] = shared_root_emb.clone()
        for r_idx in range(1, total_rep + 1):
            nid = f"{cid}_n{r_idx:03d}"
            pid = rid if r_idx == 1 else f"{cid}_n{r_idx - 1:03d}"
            ts = base_time + timedelta(seconds=20 * r_idx)
            uid = f"crowd_user_{r_idx}" if is_crowd else ("op_user_hash" if r_idx % 2 == 0 else "troll_user")
            node_rows.append(
                {
                    "node_id": nid,
                    "cascade_id": cid,
                    "topic_id": tid,
                    "parent_id": pid,
                    "is_root": False,
                    "text": f"Identical reply {r_idx}",
                    "created_at_utc": ts,
                    "lang": "vi",
                    "source_dataset": "vn_social",
                    "user_id_hash": uid,
                    "rumor_label": None,
                }
            )
            node_embeddings[nid] = shared_reply_embs[r_idx % len(shared_reply_embs)].clone()
            eid = f"{pid}->{nid}"
            edge_rows.append(
                {
                    "edge_id": eid,
                    "cascade_id": cid,
                    "source_id": pid,
                    "target_id": nid,
                    "created_at_utc": ts,
                    "time_delay_sec": 20.0,
                    "edge_type": "reply",
                }
            )
            edge_stance[eid] = -0.8
            edge_hostility[eid] = 0.75

    data_dir = tmp_path / "pingpong_vs_crowd"
    save_cascade_parquet(build_nodes_table(node_rows), build_edges_table(edge_rows), data_dir, validate=True)
    torch.save(
        {
            "node_embeddings": node_embeddings,
            "edge_stance": edge_stance,
            "edge_hostility": edge_hostility,
        },
        data_dir / "nlp_features.pt",
    )

    ds = EarlyCascadeDataset(root_dir=data_dir, k_horizon=20, split="all", min_total_replies=25)
    by_cid = {d.cascade_id: d for d in ds}
    g_pp = by_cid["casc_pingpong"]
    g_cr = by_cid["casc_crowd"]
    assert torch.equal(g_pp.x, g_cr.x)
    assert float(g_pp.struct_x[2:, 9].mean().item()) > 0.20
    assert float(g_cr.struct_x[1:, 9].max().item()) == 0.0
    assert float(g_pp.struct_x[1:, 8].mean().item()) > 0.30
    assert float(g_cr.struct_x[1:, 8].max().item()) == 0.0

    batch = next(iter(DataLoader([g_pp, g_cr], batch_size=2, shuffle=False)))
    model = SBiFAGAT(in_dim=770, hidden_dim=64, num_layers=2, edge_dim=2, dropout=0.0)
    model.eval()
    with torch.no_grad():
        init_g = model(batch)["graph_emb"]
    assert torch.norm(init_g[1] - init_g[0], p=2).item() > 0.1

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(80):
        opt.zero_grad()
        out = model(batch)
        loss = compute_multitask_loss(out, batch, alpha_size=2.0, alpha_polar=1.0, alpha_rumor=0.0)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_size = model(batch)["y_size_pred"]
    assert pred_size[1].item() - pred_size[0].item() > 1.2


def test_fair_head_parity_semantic_stats_mask_detach_and_eda_with_nlp_pt(tmp_path: Path) -> None:
    seed_everything(21)
    data_dir = _build_synthetic_20_cascade_dir(tmp_path / "parity_eda", num_cascades=8)
    ds = EarlyCascadeDataset(root_dir=data_dir, k_horizon=20, split="all", min_total_replies=25)
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))

    edge_blind_models = [
        TextMLP(in_dim=770, hidden_dim=64, use_cascade_stats=True, dropout=0.0),
        StandardGCN(in_dim=770, hidden_dim=64, use_cascade_stats=True, dropout=0.0),
        StandardGATv2(in_dim=770, hidden_dim=64, use_cascade_stats=True, dropout=0.0),
        BiGCN(in_dim=770, hidden_dim=64, use_cascade_stats=True, dropout=0.0),
        FAGCNBaseline(in_dim=770, hidden_dim=64, use_cascade_stats=True, dropout=0.0),
    ]
    edge_aware_models = [
        TemporalStanceMLP(in_dim=770, hidden_dim=64, edge_dim=2, use_cascade_stats=True, dropout=0.0),
        EdgeGATv2(in_dim=770, hidden_dim=64, edge_dim=2, use_cascade_stats=True, dropout=0.0),
        SBiFAGAT(in_dim=770, hidden_dim=64, edge_dim=2, use_cascade_stats=True, dropout=0.0),
    ]

    for mdl in edge_blind_models + edge_aware_models:
        mdl.eval()
        out = mdl(batch)
        assert out["graph_emb"].shape == (4, 128)
        assert out["node_emb"].shape == (batch.num_nodes, 64)

    batch_alt_col7 = batch.clone()
    batch_alt_col7.cascade_stats = batch_alt_col7.cascade_stats.clone()
    batch_alt_col7.cascade_stats[:, 7] = 1.0 - batch_alt_col7.cascade_stats[:, 7]
    for mdl in edge_blind_models:
        with torch.no_grad():
            o1 = mdl(batch)
            o2 = mdl(batch_alt_col7)
        assert torch.allclose(o1["y_size_pred"], o2["y_size_pred"])
        assert torch.allclose(o1["y_polar_pred"], o2["y_polar_pred"])
        assert torch.allclose(o1["rumor_logits"], o2["rumor_logits"])

    # Uncertainty head must not leak gradients back into graph_emb
    head = MultiTaskHead(
        in_dim=128,
        hidden_dim=64,
        num_rumor_classes=4,
        dropout=0.0,
        cascade_feat_dim=CASCADE_STATS_DIM,
        predict_uncertainty=True,
    )
    g_emb = torch.randn((4, 128), requires_grad=True)
    n_emb = torch.randn((12, 64))
    c_stats = torch.rand((4, CASCADE_STATS_DIM))
    h_out = head(g_emb, n_emb, cascade_stats=c_stats)
    (h_out["size_logvar"].sum() + h_out["polar_logvar"].sum()).backward()
    assert g_emb.grad is None or torch.allclose(g_emb.grad, torch.zeros_like(g_emb))

    # compute_dataset_statistics works end-to-end when nlp_features.pt is present
    stats = compute_dataset_statistics(data_dir, name="test_overfit20", min_total_replies=25, edge_dim=2)
    assert stats["eligible_cascades"] == 8
    assert stats["target_stats"] is not None
    assert stats["early_k_persistence"] is not None


def test_depth3_root_sign_alignment_and_neutral_camp_balance() -> None:
    seed_everything(31)
    # 1. Verify BFS root-alignment sign propagation and autograd on depth=3 chain: 0 -> 1 -> 2 -> 3
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    root_idx = torch.tensor([0], dtype=torch.long)
    edge_sig = torch.tensor([-0.8, 0.85, -0.9], dtype=torch.float32, requires_grad=True)
    aligned = _propagate_root_aligned_edge_signal(ei, edge_sig, root_idx, num_nodes=4, neutral_eps=0.1)
    assert torch.allclose(aligned, torch.tensor([-0.8, -0.85, 0.90]), atol=1e-5)
    aligned.sum().backward()
    assert edge_sig.grad is not None
    assert torch.allclose(edge_sig.grad, torch.tensor([1.0, -1.0, -1.0]))

    # 2. Neutral gate (camp_sig = 0.0) must keep camp_balance < 0.02
    neutral_sig = torch.zeros((20,), dtype=torch.float32)
    w_pos = torch.sigmoid(12.0 * (neutral_sig - 0.25))
    w_neg = torch.sigmoid(12.0 * (-neutral_sig - 0.25))
    camp_balance = 4.0 * w_pos.mean() * w_neg.mean()
    assert camp_balance.item() < 0.02

    # 3. Separate unanimous depth-3 opposition (y_polar=0) vs deep 50/50 split (y_polar=1)
    # 4 chains of depth 3 from root 0 -> 13 nodes total
    src_edges: list[int] = []
    dst_edges: list[int] = []
    depths = [0] + [1, 2, 3] * 4
    for ch in range(4):
        n1, n2, n3 = 1 + 3 * ch, 2 + 3 * ch, 3 + 3 * ch
        src_edges.extend([0, n1, n2])
        dst_edges.extend([n1, n2, n3])
    ei_td = torch.tensor([src_edges, dst_edges], dtype=torch.long)
    ei_bu = ei_td[[1, 0], :]

    x_root = torch.randn(1, 770) * 0.2
    v_opp = torch.zeros(1, 770)
    v_opp[0, :384] = -1.0
    v_opp[0, 384:768] = 0.9
    v_pro = -v_opp

    def _build_depth3_graph(edge_pairs_sh: list[tuple[float, float]], node_signs: list[int], y_pol: float) -> CascadeData:
        rows = [x_root] + [v_pro if s > 0 else v_opp for s in node_signs]
        x_t = torch.cat(rows, dim=0)
        x_t[:, 768] = torch.linspace(0.0, 0.35, steps=13)
        x_t[:, 769] = torch.tensor([d / 20.0 for d in depths], dtype=torch.float32)
        ea_t = torch.tensor(edge_pairs_sh, dtype=torch.float32)
        return CascadeData(
            x=x_t,
            edge_index=ei_td,
            edge_index_td=ei_td,
            edge_index_bu=ei_bu,
            edge_attr=ea_t,
            edge_stance=ea_t[:, 0:1],
            root_index=torch.tensor([0], dtype=torch.long),
            y_size=torch.tensor([3.6], dtype=torch.float32),
            y_polar=torch.tensor([y_pol], dtype=torch.float32),
            y_rumor=torch.tensor([2 if y_pol == 0.0 else 1], dtype=torch.long),
        )

    # Graph A: all 12 replies oppose root across depth 1..3 (local parent edges look +0.9 at depth 2..3!)
    ea_a = [(-0.9, 0.70), (0.9, 0.05), (0.9, 0.05)] * 4
    g_a = _build_depth3_graph(ea_a, [-1] * 12, 0.0)

    # Graph B: 6 edges align negative, 6 edges align positive (50/50 at deep branches)
    ea_b = [
        (-0.9, 0.70), (-0.9, 0.65), (0.9, 0.05),
        (-0.9, 0.70), (-0.9, 0.65), (0.9, 0.05),
        (-0.9, 0.70), (-0.9, 0.65), (-0.9, 0.65),
        (-0.9, 0.70), (-0.9, 0.65), (-0.9, 0.65),
    ]
    signs_b = [-1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1]
    g_b = _build_depth3_graph(ea_b, signs_b, 1.0)

    batch = Batch.from_data_list([g_a, g_b])
    model = SBiFAGAT(in_dim=770, hidden_dim=64, num_layers=2, edge_dim=2, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    model.train()
    for _ in range(55):
        opt.zero_grad()
        out = model(batch)
        loss = compute_multitask_loss(out, batch, alpha_size=1.0, alpha_polar=8.0, alpha_rumor=1.0)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(batch)["y_polar_pred"]
    assert pred[0].item() < 0.12
    assert pred[1].item() > 0.85


def test_ablation_integrity_sigmoid_mask_and_graph_agnostic_textmlp(tmp_path: Path) -> None:
    seed_everything(41)
    data_dir = _build_synthetic_20_cascade_dir(tmp_path / "ablation_integrity", num_cascades=4)
    ds = EarlyCascadeDataset(root_dir=data_dir, k_horizon=20, split="all", min_total_replies=25)
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))

    # 1. gate_mode='sigmoid' must use gamma_td_last, never peek directly at raw edge_attr[:, 0] in readout
    sig_model = SBiFAGAT(
        in_dim=770,
        hidden_dim=64,
        edge_dim=2,
        gate_mode="sigmoid",
        use_cascade_stats=True,
        dropout=0.0,
    )
    with torch.no_grad():
        for conv in list(sig_model.td_convs) + (list(sig_model.bu_convs) if sig_model.bu_convs else []):
            conv.att[:, 2 * conv.hidden_dim :].zero_()
            conv.edge_mlp[0].weight.zero_()
            conv.edge_mlp[0].bias.zero_()
            conv.edge_mlp[2].weight.zero_()
            conv.edge_mlp[2].bias.zero_()
    sig_model.eval()
    batch_flipped_s = batch.clone()
    batch_flipped_s.edge_attr = batch_flipped_s.edge_attr.clone()
    batch_flipped_s.edge_attr[:, 0] = -batch_flipped_s.edge_attr[:, 0]
    with torch.no_grad():
        o_sig_1 = sig_model(batch)
        o_sig_2 = sig_model(batch_flipped_s)
    assert torch.allclose(o_sig_1["y_polar_pred"], o_sig_2["y_polar_pred"])

    # 2. ablate_stance=True or ablate_hostility=True must mask cascade_stats[:, 7]
    for abl_kwargs in ({"ablate_stance": True}, {"ablate_hostility": True}):
        abl_model = SBiFAGAT(in_dim=770, hidden_dim=64, use_cascade_stats=True, dropout=0.0, **abl_kwargs)
        assert abl_model.head.mask_semantic_stats is True
        abl_model.eval()
        batch_alt = batch.clone()
        batch_alt.cascade_stats = batch_alt.cascade_stats.clone()
        batch_alt.cascade_stats[:, 7] = 1.0 - batch_alt.cascade_stats[:, 7]
        with torch.no_grad():
            o_a1 = abl_model(batch)
            o_a2 = abl_model(batch_alt)
        assert torch.allclose(o_a1["y_size_pred"], o_a2["y_size_pred"])
        assert torch.allclose(o_a1["y_polar_pred"], o_a2["y_polar_pred"])

    # 3. TextMLP must remain invariant to rewiring edge_index_td and perturbing tree-topology cols [1, 4, 6, 7]
    mlp = TextMLP(in_dim=770, hidden_dim=64, use_cascade_stats=True, dropout=0.0)
    mlp.eval()
    batch_topo_mut = batch.clone()
    batch_topo_mut.struct_x = batch_topo_mut.struct_x.clone()
    batch_topo_mut.struct_x[:, [1, 4, 6, 7]] = torch.rand_like(batch_topo_mut.struct_x[:, [1, 4, 6, 7]])
    # Rewire every node's parent to a linear chain within graph 0
    chain_src = torch.arange(batch_topo_mut.num_nodes - 1, dtype=torch.long)
    chain_dst = torch.arange(1, batch_topo_mut.num_nodes, dtype=torch.long)
    batch_topo_mut.edge_index_td = torch.stack([chain_src, chain_dst], dim=0)
    batch_topo_mut.edge_index = batch_topo_mut.edge_index_td
    with torch.no_grad():
        m1 = mlp(batch)
        m2 = mlp(batch_topo_mut)
    assert torch.allclose(m1["y_size_pred"], m2["y_size_pred"])
    assert torch.allclose(m1["y_polar_pred"], m2["y_polar_pred"])
    assert torch.allclose(m1["rumor_logits"], m2["rumor_logits"])


def test_live_predictor_salted_hash_deleted_and_batched_diagnostics(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("SBIF_HASH_SALT", "audit_salt_test_v1")

    # 1. Default hidden_dim parity across models
    assert SBiFAGAT().hidden_dim == 64
    assert TextMLP().head.size_head[0].in_features == 128
    assert StandardGCN().head.size_head[0].in_features == 128

    # 2. JSON loader & build_vn_cascade_from_thread share salted hash and keep [deleted]/[removed] as None
    base_ts = 1763200000.0
    thread_json = {
        "id": "salt_demo_01",
        "title": "Thảo luận mở rộng tuyến buýt nhanh nội đô",
        "selftext": "Mọi người thấy phương án mới có khả thi không?",
        "subreddit": "vozforums",
        "author": "op_author",
        "created_utc": base_ts,
        "comments": [
            {"id": "c1", "parent_id": "t3_salt_demo_01", "text": "Bình luận đầu đã bị xóa tài khoản", "created_utc": base_ts + 30, "author": "[deleted]"},
            {"id": "c2", "parent_id": "c1", "text": "Bình luận thứ hai cũng deleted", "created_utc": base_ts + 60, "author": "[deleted]"},
            {"id": "c3", "parent_id": "c2", "text": "Tác giả gốc quay lại giải thích thêm", "created_utc": base_ts + 90, "author": "op_author"},
            {"id": "c4", "parent_id": "c3", "text": "Bình luận bị removed", "created_utc": base_ts + 120, "author": "[removed]"},
        ],
    }
    json_path = tmp_path / "thread.json"
    json_path.write_text(json.dumps(thread_json), encoding="utf-8")

    post, comments = _load_thread_from_json(json_path)
    assert comments[0]["author_hash"] is None
    assert comments[1]["author_hash"] is None
    assert comments[3]["author_hash"] is None
    assert comments[2]["author_hash"] == hash_author_sha256("op_author")

    rec = build_vn_cascade_from_thread(post, comments, min_comments=1, min_depth=1, min_vi_ratio=0.0)
    assert rec is not None
    nodes, _edges, _meta = rec
    delays = [max(0.0, (n["created_at_utc"] - nodes[0]["created_at_utc"]).total_seconds()) for n in nodes]
    depth_map = {
        "vn_salt_demo_01_t3_salt_demo_01": 0,
        "vn_salt_demo_01_t1_c1": 1,
        "vn_salt_demo_01_t1_c2": 2,
        "vn_salt_demo_01_t1_c3": 3,
        "vn_salt_demo_01_t1_c4": 4,
    }
    struct_x = compute_causal_structural_node_features(nodes, delays, depth_map, k_horizon=20)
    by_nid = {n["node_id"]: i for i, n in enumerate(nodes)}
    # [deleted] comments must NOT increment author_repeat_norm (col 9) or match root (col 8)
    assert struct_x[by_nid["vn_salt_demo_01_t1_c1"], 9].item() == 0.0
    assert struct_x[by_nid["vn_salt_demo_01_t1_c2"], 9].item() == 0.0
    assert struct_x[by_nid["vn_salt_demo_01_t1_c4"], 9].item() == 0.0
    # op_author re-entry comment must match salted root user_id_hash -> is_op_reentry == 1.0
    assert struct_x[by_nid["vn_salt_demo_01_t1_c3"], 8].item() == 1.0

    # 3. predict_with_diagnostics flattens edge_pairs cleanly on a PyG Batch of 2 graphs
    ei1 = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    ei2 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    g1 = CascadeData(
        x=torch.randn(3, 770),
        edge_index=ei1,
        edge_index_td=ei1,
        edge_index_bu=ei1[[1, 0]],
        edge_attr=torch.tensor([[0.8, 0.1], [-0.7, 0.6]], dtype=torch.float32),
        root_index=torch.tensor([0], dtype=torch.long),
        edge_pairs=[("g1_r", "g1_c1"), ("g1_r", "g1_c2")],
    )
    g2 = CascadeData(
        x=torch.randn(3, 770),
        edge_index=ei2,
        edge_index_td=ei2,
        edge_index_bu=ei2[[1, 0]],
        edge_attr=torch.tensor([[-0.9, 0.8], [0.5, 0.2]], dtype=torch.float32),
        root_index=torch.tensor([0], dtype=torch.long),
        edge_pairs=[("g2_r", "g2_c1"), ("g2_c1", "g2_c2")],
    )
    batch = Batch.from_data_list([g1, g2])
    expected_pairs = {("g1_r", "g1_c1"), ("g1_r", "g1_c2"), ("g2_r", "g2_c1"), ("g2_c1", "g2_c2")}
    diag = SBiFAGAT().predict_with_diagnostics(batch, top_k_edges=4)
    assert len(diag["top_trigger_edges"]) == 4
    for entry in diag["top_trigger_edges"]:
        assert isinstance(entry.get("pair"), tuple)
        assert entry["pair"] in expected_pairs


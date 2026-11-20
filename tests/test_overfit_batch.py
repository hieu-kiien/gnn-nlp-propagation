"""Overfit-on-8-graphs gradient flow test & architectural checks for SBiFAGAT and baselines."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.loader import DataLoader

from src.data.pyg_dataset import CascadeData
from src.models.baselines import BiGCN, StandardGATv2, StandardGCN, TextMLP
from src.models.sbi_fagat import SBiFAGAT
from src.training.metrics import compute_multitask_loss
from src.utils.seed import seed_everything


def _make_8_graph_batch(in_dim: int = 770) -> tuple[list[CascadeData], CascadeData]:
    seed_everything(42)
    graphs: list[CascadeData] = []
    for i in range(8):
        num_nodes = 6 + (i % 4) * 3
        x = torch.randn(num_nodes, in_dim)
        src = [max(0, (n - 1) // 2) for n in range(1, num_nodes)]
        dst = list(range(1, num_nodes))
        edge_index_td = torch.tensor([src, dst], dtype=torch.long)
        edge_index_bu = edge_index_td[[1, 0], :]
        stances = torch.linspace(-0.9, 0.9, steps=len(src)).unsqueeze(-1)
        hostilities = (stances < 0).float() * 0.8 + 0.1
        edge_attr = torch.cat([stances, hostilities], dim=-1)

        graphs.append(
            CascadeData(
                x=x,
                edge_index=edge_index_td,
                edge_index_td=edge_index_td,
                edge_index_bu=edge_index_bu,
                edge_attr=edge_attr,
                edge_stance=stances,
                root_index=torch.tensor([0], dtype=torch.long),
                y_size=torch.tensor([1.0 + 0.35 * i], dtype=torch.float32),
                y_polar=torch.tensor([0.1 + 0.1 * i], dtype=torch.float32),
                y_rumor=torch.tensor([i % 4], dtype=torch.long),
                cascade_id=f"mock_batch_{i}",
                topic_id=f"mock_topic_{i}",
            )
        )

    loader = DataLoader(graphs, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    return graphs, batch


def test_sbifagat_overfits_8_graph_batch_in_100_steps() -> None:
    seed_everything(42)
    _, batch = _make_8_graph_batch(in_dim=770)

    model = SBiFAGAT(in_dim=770, hidden_dim=128, num_layers=2, edge_dim=2, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)

    losses: list[float] = []
    model.train()
    for step in range(100):
        optimizer.zero_grad()
        outputs = model(batch)
        loss = compute_multitask_loss(outputs, batch)
        loss.backward()

        if step == 0:
            # Verify gradients flow through all parameters including TD/BU conflict gate `att`
            for name, param in model.named_parameters():
                assert param.requires_grad
                assert param.grad is not None, f"Missing gradient on {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient on {name}"
            assert model.td_convs[0].att.grad.abs().sum().item() > 0.0
            assert model.bu_convs[0].att.grad.abs().sum().item() > 0.0

        optimizer.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0] * 0.05, f"Loss did not converge: start={losses[0]}, end={losses[-1]}"
    assert losses[-1] < 0.02, f"Final loss too high: {losses[-1]}"


def test_sbifagat_gate_extraction_ablation_and_upfd_latent_mlp() -> None:
    _, batch = _make_8_graph_batch(in_dim=770)

    full_model = SBiFAGAT(in_dim=770, hidden_dim=128, edge_dim=2, dropout=0.0)
    gates = full_model.extract_edge_gates(batch)
    num_edges = batch.edge_index_td.size(1)
    assert gates["gamma_td"].shape == (num_edges,)
    assert gates["gamma_bu"].shape == (num_edges,)
    assert gates["edge_importance"].shape == (num_edges,)
    assert torch.all(gates["gamma_td"].abs() <= 1.0 + 1e-6)
    assert torch.all(gates["edge_importance"] >= 0.0)

    # Sigmoid conflict-gate ablation must produce strictly positive weights
    sigmoid_model = SBiFAGAT(in_dim=770, hidden_dim=128, edge_dim=2, gate_mode="sigmoid")
    sig_gates = sigmoid_model.extract_edge_gates(batch)
    assert torch.all(sig_gates["gamma_td"] > 0.0)
    assert torch.all(sig_gates["gamma_bu"] > 0.0)

    # Stance ablation ignores s_uv perturbations; hostility ablation ignores h_uv perturbations
    batch_flip_s = batch.clone()
    batch_flip_s.edge_attr = batch_flip_s.edge_attr.clone()
    batch_flip_s.edge_attr[:, 0] = -batch_flip_s.edge_attr[:, 0]

    no_stance_model = SBiFAGAT(in_dim=770, hidden_dim=128, edge_dim=2, ablate_stance=True, dropout=0.0)
    no_stance_model.eval()
    assert torch.allclose(
        no_stance_model(batch)["y_polar_pred"],
        no_stance_model(batch_flip_s)["y_polar_pred"],
    )

    batch_flip_h = batch.clone()
    batch_flip_h.edge_attr = batch_flip_h.edge_attr.clone()
    batch_flip_h.edge_attr[:, 1] = 0.0
    no_host_model = SBiFAGAT(in_dim=770, hidden_dim=128, edge_dim=2, ablate_hostility=True, dropout=0.0)
    no_host_model.eval()
    assert torch.allclose(
        no_host_model(batch)["y_polar_pred"],
        no_host_model(batch_flip_h)["y_polar_pred"],
    )

    # Bottom-up ablation runs cleanly
    no_bu_model = SBiFAGAT(in_dim=770, hidden_dim=128, edge_dim=2, use_bottom_up=False)
    out_no_bu = no_bu_model(batch)
    assert out_no_bu["graph_emb"].shape == (8, 256)

    # UPFD latent edge MLP mode (768-d BERT input without edge_attr)
    _, upfd_batch = _make_8_graph_batch(in_dim=768)
    upfd_batch.edge_attr = torch.empty((upfd_batch.edge_index_td.size(1), 2))
    upfd_model = SBiFAGAT(
        in_dim=768,
        hidden_dim=128,
        edge_dim=2,
        num_rumor_classes=2,
        use_latent_edge_mlp=True,
        dropout=0.0,
    )
    upfd_batch.y_rumor = upfd_batch.y_rumor % 2
    out_upfd = upfd_model(upfd_batch)
    loss_upfd = compute_multitask_loss(out_upfd, upfd_batch, classification_only=True)
    loss_upfd.backward()
    assert upfd_model.latent_edge_mlp is not None
    for p in upfd_model.latent_edge_mlp.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


@pytest.mark.parametrize("model_cls", [TextMLP, StandardGCN, StandardGATv2, BiGCN])
def test_all_four_baselines_forward_and_backward(model_cls: type) -> None:
    _, batch = _make_8_graph_batch(in_dim=770)
    model = model_cls(in_dim=770, hidden_dim=128)
    out = model(batch)
    assert out["y_size_pred"].shape == (8,)
    assert out["y_polar_pred"].shape == (8,)
    assert out["rumor_logits"].shape == (8, 4)
    assert out["graph_emb"].shape == (8, 256)
    assert out["node_emb"].shape == (batch.num_nodes, 128)

    loss = compute_multitask_loss(out, batch)
    loss.backward()
    assert torch.isfinite(loss)

    # Verify degenerate 1-node (0-edge) graph works without crashing
    single_node = CascadeData(
        x=torch.randn(1, 770),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_index_td=torch.empty((2, 0), dtype=torch.long),
        edge_index_bu=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 2), dtype=torch.float32),
        root_index=torch.tensor([0], dtype=torch.long),
    )
    out_single = model(single_node)
    assert out_single["y_size_pred"].shape == (1,)

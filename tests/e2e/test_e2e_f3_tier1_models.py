"""Opaque-box E2E tests for Feature Area F3: Tier 1 Models, Baselines, SOTA & Causal Bridge."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch_geometric.data import Batch

from src.data.pyg_dataset import CascadeData, TOTAL_NODE_FEATURE_DIM
from src.models.baselines import (
    BiEdgeGATv2,
    BiGCN,
    EdgeGATv2,
    FAGCNBaseline,
    StandardGATv2,
    StandardGCN,
    TemporalStanceMLP,
    TextMLP,
)
from src.models.sbi_fagat import SBiFAGAT
from tests.e2e.conftest import make_synthetic_cascade_data_list


def _load_m3_sota_gnn_factories(in_dim: int, hidden_dim: int, num_classes: int) -> dict[str, Any]:
    """Lazily import M3 SOTA PyTorch models inside test execution scope."""
    from src.models.sota_cascades import CasCN, CasFlow, FriedkinJohnsenGNN

    return {
        "FriedkinJohnsenGNN": lambda: FriedkinJohnsenGNN(
            in_dim=in_dim, hidden_dim=hidden_dim, edge_dim=2, num_rumor_classes=num_classes
        ),
        "CasCN": lambda: CasCN(
            in_dim=in_dim, hidden_dim=hidden_dim, num_rumor_classes=num_classes
        ),
        "CasFlow": lambda: CasFlow(
            in_dim=in_dim, hidden_dim=hidden_dim, edge_dim=2, num_rumor_classes=num_classes
        ),
    }


def _load_m3_tabular_baselines(num_classes: int) -> list[Any]:
    """Lazily import M3 tabular baselines inside test execution scope."""
    from src.models.sota_cascades import (
        CalibratedEarlyKPersistence,
        LightGBMBaseline,
        RidgeBaseline,
    )

    return [
        CalibratedEarlyKPersistence(num_rumor_classes=num_classes),
        RidgeBaseline(num_rumor_classes=num_classes),
        LightGBMBaseline(num_rumor_classes=num_classes),
    ]


# ---------------------------------------------------------------------------
# Tier 1: Isolated Feature Coverage
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_f3_t1_all_tier1_models_unified_dict_contract() -> None:
    """Every Tier 1 model (existing + M3 baselines & SOTA) obeys the CascadeData -> dict contract."""
    graphs = make_synthetic_cascade_data_list(num_graphs=4, in_dim=TOTAL_NODE_FEATURE_DIM, num_nodes_per_graph=6)
    batch = Batch.from_data_list(graphs)
    B = len(graphs)
    N = batch.x.size(0)
    H = 32
    C = 4

    factories = {
        "TextMLP": lambda: TextMLP(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=H, num_rumor_classes=C),
        "TemporalStanceMLP": lambda: TemporalStanceMLP(
            in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=H, edge_dim=2, num_rumor_classes=C
        ),
        "StandardGCN": lambda: StandardGCN(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=H, num_rumor_classes=C),
        "StandardGATv2": lambda: StandardGATv2(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=H, num_rumor_classes=C),
        "BiGCN": lambda: BiGCN(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=H, num_rumor_classes=C),
        "EdgeGATv2": lambda: EdgeGATv2(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=H, edge_dim=2, num_rumor_classes=C),
        "BiEdgeGATv2": lambda: BiEdgeGATv2(
            in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=H, edge_dim=2, num_rumor_classes=C
        ),
        "FAGCNBaseline": lambda: FAGCNBaseline(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=H, num_rumor_classes=C),
        "SBiFAGAT": lambda: SBiFAGAT(
            in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=H, edge_dim=2, num_rumor_classes=C, use_cascade_stats=True
        ),
    }
    factories.update(_load_m3_sota_gnn_factories(TOTAL_NODE_FEATURE_DIM, H, C))

    for name, factory in factories.items():
        model = factory()
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert isinstance(out, dict), f"{name} did not return a dict"
        for key in ("y_size_pred", "y_polar_pred", "rumor_logits", "graph_emb", "node_emb"):
            assert key in out, f"{name} missing key {key!r}"
            assert torch.isfinite(out[key]).all(), f"{name} produced non-finite {key!r}"
        assert out["y_size_pred"].shape == (B,), f"{name} y_size_pred shape mismatch"
        assert out["y_polar_pred"].shape == (B,), f"{name} y_polar_pred shape mismatch"
        assert out["rumor_logits"].shape == (B, C), f"{name} rumor_logits shape mismatch"
        assert out["graph_emb"].shape == (B, 2 * H), f"{name} graph_emb shape mismatch"
        assert out["node_emb"].shape == (N, H), f"{name} node_emb shape mismatch"
        assert bool(((out["y_polar_pred"] >= 0.0) & (out["y_polar_pred"] <= 1.0)).all())

    for tab_model in _load_m3_tabular_baselines(C):
        tab_model.fit_dataset(graphs, graphs[:2], seed=42)
        t_out = tab_model.predict_dataset(graphs)
        assert t_out["y_size_pred"].shape == (B,)
        assert t_out["y_polar_pred"].shape == (B,)
        assert t_out["rumor_logits"].shape == (B, C)
        assert bool(((t_out["y_polar_pred"] >= 0.0) & (t_out["y_polar_pred"] <= 1.0)).all())


@pytest.mark.tier1
def test_f3_t1_predict_with_diagnostics_no_hardcoded_confidence() -> None:
    """B8 contract: predict_uncertainty=False returns confidence_score=None & uncertainty_available=False (never 0.75)."""
    graphs = make_synthetic_cascade_data_list(num_graphs=2, in_dim=TOTAL_NODE_FEATURE_DIM)
    batch = Batch.from_data_list(graphs)

    model_det = SBiFAGAT(
        in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2, predict_uncertainty=False, use_cascade_stats=True
    )
    diag_det = model_det.predict_with_diagnostics(batch)
    assert diag_det["confidence_score"] is None
    assert diag_det["uncertainty_available"] is False

    model_unc = SBiFAGAT(
        in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2, predict_uncertainty=True, use_cascade_stats=True
    )
    diag_unc = model_unc.predict_with_diagnostics(batch)
    assert diag_unc["confidence_score"] is not None
    assert diag_unc["uncertainty_available"] is True
    assert torch.isfinite(diag_unc["confidence_score"]).all()
    assert bool(((diag_unc["confidence_score"] >= 0.0) & (diag_unc["confidence_score"] <= 1.0)).all())


@pytest.mark.tier1
def test_f3_t1_ablate_stance_decoupled_from_persistence_and_stats() -> None:
    """M5-T1 contract: ablate_stance=True must NOT silently turn off persistence anchor or cascade_stats[:, 7]."""
    torch.manual_seed(42)
    model = SBiFAGAT(
        in_dim=TOTAL_NODE_FEATURE_DIM,
        hidden_dim=32,
        edge_dim=2,
        ablate_stance=True,
        use_cascade_stats=True,
        use_polar_persistence_anchor=True,
    )
    assert model.use_polar_persistence_anchor is True

    # Run 2 training steps so persist_weight moves away from its zero init if early_y_polar is wired
    train_graphs = make_synthetic_cascade_data_list(num_graphs=4, in_dim=TOTAL_NODE_FEATURE_DIM)
    for idx_g, tg in enumerate(train_graphs):
        val = 0.15 + 0.2 * idx_g
        tg.early_y_polar = torch.tensor([val], dtype=torch.float32)
        tg.y_polar = torch.tensor([val], dtype=torch.float32)
    train_batch = Batch.from_data_list(train_graphs)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    model.train()
    for _ in range(3):
        opt.zero_grad()
        tr_out = model(train_batch)
        loss = torch.nn.functional.mse_loss(tr_out["y_polar_pred"], train_batch.y_polar)
        loss.backward()
        opt.step()

    model.eval()
    g1 = train_graphs[0].clone()
    g2 = train_graphs[0].clone()
    g1.early_y_polar = torch.tensor([0.05], dtype=torch.float32)
    g2.early_y_polar = torch.tensor([0.92], dtype=torch.float32)

    with torch.no_grad():
        out1 = model(Batch.from_data_list([g1]))
        out2 = model(Batch.from_data_list([g2]))
    assert not torch.allclose(out1["y_polar_pred"], out2["y_polar_pred"], atol=1e-6), (
        "ablate_stance=True appears to have disabled persistence anchor sensitivity"
    )


@pytest.mark.tier1
def test_f3_t1_camp_dipole_includes_root_and_handles_all_negative_replies() -> None:
    """M1 contract: Camp Dipole includes root_index in mu_pos so all-negative-reply graphs retain dipole signal."""
    torch.manual_seed(7)
    g = make_synthetic_cascade_data_list(num_graphs=1, in_dim=TOTAL_NODE_FEATURE_DIM, num_nodes_per_graph=5)[0]
    # Force ALL reply edges to be strictly negative (cross-camp) -> without root in mu_pos, has_pos==False!
    g.edge_attr[:, 0] = -0.90
    g.edge_stance[:, 0] = -0.90

    m_dipole = SBiFAGAT(
        in_dim=TOTAL_NODE_FEATURE_DIM,
        hidden_dim=32,
        edge_dim=2,
        use_camp_dipole=True,
        use_polar_persistence_anchor=False,
    )
    m_no_dipole = SBiFAGAT(
        in_dim=TOTAL_NODE_FEATURE_DIM,
        hidden_dim=32,
        edge_dim=2,
        use_camp_dipole=False,
        use_polar_persistence_anchor=False,
    )
    m_no_dipole.load_state_dict(m_dipole.state_dict(), strict=False)
    m_dipole.eval()
    m_no_dipole.eval()
    batch_g = Batch.from_data_list([g])
    with torch.no_grad():
        out = m_dipole(batch_g)
        out_no_dipole = m_no_dipole(batch_g)
    assert torch.isfinite(out["graph_emb"]).all()
    assert torch.isfinite(out["y_polar_pred"]).all()
    assert not torch.allclose(out["graph_emb"], out_no_dipole["graph_emb"], atol=1e-5), (
        "Camp Dipole produced identical graph_emb to use_camp_dipole=False on all-negative-reply cascade"
    )


@pytest.mark.tier1
def test_f3_t1_sota_cascades_gradient_flow_and_determinism() -> None:
    """R2 contract: FriedkinJohnsenGNN, CasCN, CasFlow support deterministic forward + clean backprop."""
    from src.models.sota_cascades import CasCN, CasFlow, FriedkinJohnsenGNN

    graphs = make_synthetic_cascade_data_list(num_graphs=4, in_dim=TOTAL_NODE_FEATURE_DIM)
    batch = Batch.from_data_list(graphs)

    for cls in (FriedkinJohnsenGNN, CasCN, CasFlow):
        torch.manual_seed(42)
        kwargs: dict[str, Any] = {"in_dim": TOTAL_NODE_FEATURE_DIM, "hidden_dim": 32, "num_rumor_classes": 4}
        if cls is not CasCN:
            kwargs["edge_dim"] = 2
        m1 = cls(**kwargs)
        m1.eval()
        with torch.no_grad():
            o1 = m1(batch)
            o2 = m1(batch)
        assert torch.allclose(o1["y_size_pred"], o2["y_size_pred"], atol=1e-6)
        assert torch.allclose(o1["y_polar_pred"], o2["y_polar_pred"], atol=1e-6)

        m1.train()
        m1.zero_grad()
        out = m1(batch)
        loss = out["y_size_pred"].sum() + out["y_polar_pred"].sum() + out["rumor_logits"].sum()
        loss.backward()
        grad_norms = [p.grad.abs().sum().item() for p in m1.parameters() if p.requires_grad and p.grad is not None]
        assert any(g > 0.0 for g in grad_norms), f"{cls.__name__} had zero gradients"


@pytest.mark.tier1
def test_f3_t1_causal_tier2_prior_bridge_consumed_when_enabled() -> None:
    """R2 contract: SBiFAGAT / SOTA accept optional causal tier2_user_prior [N_K, D_prior] when enabled."""
    graphs = make_synthetic_cascade_data_list(
        num_graphs=3,
        in_dim=TOTAL_NODE_FEATURE_DIM,
        include_tier2_prior=True,
        tier2_prior_dim=4,
    )
    batch = Batch.from_data_list(graphs)

    model = SBiFAGAT(
        in_dim=TOTAL_NODE_FEATURE_DIM,
        hidden_dim=32,
        edge_dim=2,
        use_tier2_prior=True,
        tier2_prior_dim=4,
    )
    model.eval()
    with torch.no_grad():
        out_with_prior = model(batch)
        batch_zero = batch.clone()
        batch_zero.tier2_user_prior = torch.ones_like(batch_zero.tier2_user_prior) * 3.0
        out_shifted = model(batch_zero)

    assert torch.isfinite(out_with_prior["y_polar_pred"]).all()
    assert not torch.allclose(out_with_prior["graph_emb"], out_shifted["graph_emb"], atol=1e-6)


# ---------------------------------------------------------------------------
# Tier 2: Boundary, Corner & Negative Rejection Cases
# ---------------------------------------------------------------------------


@pytest.mark.tier2
def test_f3_t2_single_node_zero_edge_cascade() -> None:
    """Single-node cascade (root only, E_K == 0) runs cleanly without NaN or scatter crash."""
    single = CascadeData(
        x=torch.randn((1, TOTAL_NODE_FEATURE_DIM), dtype=torch.float32),
        struct_x=torch.zeros((1, 10), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_index_td=torch.empty((2, 0), dtype=torch.long),
        edge_index_bu=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 2), dtype=torch.float32),
        edge_stance=torch.empty((0, 1), dtype=torch.float32),
        cascade_stats=torch.zeros((1, 8), dtype=torch.float32),
        root_index=torch.tensor([0], dtype=torch.long),
        y_size=torch.tensor([0.0], dtype=torch.float32),
        y_polar=torch.tensor([0.0], dtype=torch.float32),
        early_y_polar=torch.tensor([0.0], dtype=torch.float32),
        y_rumor=torch.tensor([0], dtype=torch.long),
    )
    batch = Batch.from_data_list([single])
    for model in (
        SBiFAGAT(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2, use_cascade_stats=True),
        BiEdgeGATv2(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2, use_cascade_stats=True),
    ):
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert torch.isfinite(out["y_size_pred"]).all()
        assert torch.isfinite(out["y_polar_pred"]).all()


@pytest.mark.tier2
def test_f3_t2_heterogeneous_batch_sizes_and_missing_optional_attrs() -> None:
    """Batch mixing N_K in {2, 5, 11, 21} with missing cascade_stats & early_y_polar works cleanly."""
    graphs = []
    for i, n_k in enumerate((2, 5, 11, 21)):
        g = make_synthetic_cascade_data_list(num_graphs=1, num_nodes_per_graph=n_k, seed=100 + i)[0]
        del g.cascade_stats
        del g.early_y_polar
        graphs.append(g)
    batch = Batch.from_data_list(graphs)
    model = SBiFAGAT(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2, use_cascade_stats=False)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out["y_size_pred"].shape == (4,)
    assert out["node_emb"].shape == (sum((2, 5, 11, 21)), 32)
    assert torch.isfinite(out["y_polar_pred"]).all()


@pytest.mark.tier2
def test_f3_t2_ablation_zero_edge_attr_and_latent_edge_mlp() -> None:
    """SBiFAGAT with all-zero edge_attr and use_latent_edge_mlp=True maintains finite gradients."""
    graphs = make_synthetic_cascade_data_list(num_graphs=3)
    for g in graphs:
        g.edge_attr.zero_()
        g.edge_stance.zero_()
    batch = Batch.from_data_list(graphs)

    model = SBiFAGAT(
        in_dim=TOTAL_NODE_FEATURE_DIM,
        hidden_dim=32,
        edge_dim=2,
        use_latent_edge_mlp=True,
    )
    out = model(batch)
    loss = out["y_size_pred"].mean() + out["y_polar_pred"].mean()
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.tier2
def test_f3_t2_tabular_baselines_constant_targets_and_small_n() -> None:
    """CalibratedEarlyKPersistence, RidgeBaseline, and LightGBMBaseline handle constant targets without NaN."""
    import numpy as np
    from src.models.sota_cascades import (
        CalibratedEarlyKPersistence,
        LightGBMBaseline,
        RidgeBaseline,
    )

    graphs = make_synthetic_cascade_data_list(num_graphs=5, in_dim=TOTAL_NODE_FEATURE_DIM)
    for g in graphs:
        g.y_polar = torch.tensor([0.25], dtype=torch.float32)
        g.y_size = torch.tensor([3.2], dtype=torch.float32)

    for cls in (CalibratedEarlyKPersistence, RidgeBaseline, LightGBMBaseline):
        m = cls(num_rumor_classes=4)
        m.fit_dataset(graphs, graphs[:2], seed=42)
        out = m.predict_dataset(graphs)
        assert np.isfinite(out["y_size_pred"]).all()
        assert np.isfinite(out["y_polar_pred"]).all()


@pytest.mark.tier2
def test_f3_t2_bidedge_gatv2_sensitive_to_edge_attr() -> None:
    """BiEdgeGATv2 responds to edge_attr stance flips and preserves TD+BU message passing."""
    g1 = make_synthetic_cascade_data_list(num_graphs=1)[0]
    g2 = g1.clone()
    g2.edge_attr = -g2.edge_attr

    model = BiEdgeGATv2(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, heads=4, edge_dim=2)
    model.eval()
    with torch.no_grad():
        o1 = model(Batch.from_data_list([g1]))
        o2 = model(Batch.from_data_list([g2]))
    assert not torch.allclose(o1["graph_emb"], o2["graph_emb"], atol=1e-6)

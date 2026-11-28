"""Tests for Tier 2 Macro Signed Social Network analytics and Temporal Signed Link GNNs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import torch

from torch_geometric.utils import to_undirected

from scripts.assemble_paper_artifacts import find_unescaped_underscores_in_latex
from scripts.run_macro_signed_network import (
    LINK_PRED_METRICS,
    build_link_markdown_table,
    compute_wilcoxon_signed_rank,
    evaluate_link_split,
    export_link_latex_table,
    parse_args,
    run_signed_link_benchmark,
    run_smoke_test,
)
from src.data.schema import EDGES_ARROW_SCHEMA, NODES_ARROW_SCHEMA, save_cascade_parquet
from src.data.user_network import (
    build_temporal_signed_link_dataset,
    build_user_interaction_records,
    compute_macro_signed_network_stats,
    format_macro_network_latex,
    plot_macro_signed_network,
)
from src.models.signed_gnns import (
    LINK_PRED_MODEL_NAMES,
    BiEdgeGATv2Link,
    PriorPairSignPersistence,
    SBiFAGATLink,
    SDGNN,
    SGCNLink,
    build_link_model,
)
from src.utils.seed import OFFICIAL_SEEDS


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
    assert list(parse_args([]).seeds) == list(OFFICIAL_SEEDS)
    assert parse_args([]).mp_graph_mode == "coalesced"
    run_smoke_test()


_NEURAL_MODELS = tuple(m for m in LINK_PRED_MODEL_NAMES if m != "PriorPairSignPersistence")
_GNN_MODELS = tuple(m for m in _NEURAL_MODELS if m != "UserTextMLP")


@pytest.mark.parametrize("model_name", _NEURAL_MODELS)
@pytest.mark.parametrize("hidden_dim", [16, 64])
def test_all_signed_gnns_shapes_grads_and_edge_cases(model_name: str, hidden_dim: int) -> None:
    torch.manual_seed(42)
    in_dim = 20
    num_nodes = 10
    x = torch.randn(num_nodes, in_dim)
    # Leave node 9 completely isolated
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 0, 4, 5, 3, 7, 8]],
        dtype=torch.long,
    )
    edge_attr = torch.tensor(
        [
            [0.8, 0.1],
            [-0.7, 0.8],
            [0.6, 0.2],
            [-0.9, 0.9],
            [0.5, 0.1],
            [-0.6, 0.5],
            [0.4, 0.2],
            [-0.5, 0.6],
        ],
        dtype=torch.float32,
    )
    pairs = torch.tensor([[0, 2, 4, 9], [1, 0, 5, 8]], dtype=torch.long)

    model = build_link_model(model_name, in_dim=in_dim, hidden_dim=hidden_dim, num_layers=2, dropout=0.0)
    model.train()
    h = model.encode(x, edge_index, edge_attr)
    assert h.shape == (num_nodes, hidden_dim)
    assert torch.isfinite(h).all()

    logits = model(x, edge_index, edge_attr, pairs)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()

    loss = logits.pow(2).mean()
    if isinstance(model, SDGNN):
        aux = model.compute_aux_loss(h, edge_index, edge_attr)
        assert aux.ndim == 0 and torch.isfinite(aux)
        loss = loss + 0.2 * aux
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0.0 for p in model.parameters() if p.requires_grad)
    assert has_grad

    # Edge cases: empty edges, edge_attr=None, empty pairs
    model.eval()
    empty_ei = torch.empty((2, 0), dtype=torch.long)
    empty_ea = torch.empty((0, 2), dtype=torch.float32)
    empty_pairs = torch.empty((2, 0), dtype=torch.long)

    h_empty = model.encode(x, empty_ei, empty_ea)
    assert h_empty.shape == (num_nodes, hidden_dim)
    h_none = model.encode(x, edge_index, None)
    assert h_none.shape == (num_nodes, hidden_dim)
    logits_empty = model(x, empty_ei, empty_ea, empty_pairs)
    assert logits_empty.shape == (0,)
    if isinstance(model, SDGNN):
        assert float(model.compute_aux_loss(h_empty, empty_ei, empty_ea).item()) == 0.0


def test_prior_pair_sign_persistence() -> None:
    model = PriorPairSignPersistence(alpha=2.0, user_alpha=2.0, undirected_fallback=True)
    assert sum(p.numel() for p in model.parameters()) == 0

    # Train pairs: (0->1) opposed 3/3 times; (2->3) supported 2/2 times; (4->5) opposed 1/1
    train_pairs = torch.tensor([[0, 0, 0, 2, 2, 4], [1, 1, 1, 3, 3, 5]], dtype=torch.long)
    train_y = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    model.fit(train_pairs, train_y, num_users=8)

    # Query: (0,1) seen directed; (1,0) reversed undirected fallback; (0,3) seen users unseen pair; (6,7) unseen users
    q_pairs = torch.tensor([[0, 1, 0, 6], [1, 0, 3, 7]], dtype=torch.long)
    probs = model.predict_proba(q_pairs)
    logits = model.predict_logits(q_pairs)
    assert probs.shape == (4,)
    assert logits.shape == (4,)
    # (0,1) opposed > (0,3) mixed user prior > (6,7) global > (2,3) supported
    assert probs[0] > probs[2]
    assert probs[1] > 0.7  # undirected fallback from (0,1)
    assert probs[3] == pytest.approx(model.global_prior, abs=1e-5)

    p_23 = model.predict_proba(torch.tensor([[2], [3]], dtype=torch.long))[0]
    assert p_23 < 0.35

    empty_probs = model.predict_proba(torch.empty((2, 0), dtype=torch.long))
    assert empty_probs.shape == (0,)
    fwd = model(torch.zeros((8, 4)), torch.empty((2, 0), dtype=torch.long), None, q_pairs)
    assert torch.allclose(fwd, logits)


@pytest.mark.parametrize("model_name", _GNN_MODELS)
def test_unified_graph_coalescing_parity_and_edge_attr_sensitivity(model_name: str) -> None:
    torch.manual_seed(7)
    x = torch.randn(5, 20)
    ei_1x = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    ea_1x = torch.tensor([[0.8, 0.1], [-0.7, 0.8], [0.6, 0.2], [-0.5, 0.6]], dtype=torch.float32)

    # Duplicate (0 -> 1) 10 times with identical attr
    ei_10x = torch.cat([ei_1x, torch.tensor([[0] * 9, [1] * 9], dtype=torch.long)], dim=1)
    ea_10x = torch.cat([ea_1x, ea_1x[0:1].repeat(9, 1)], dim=0)

    model = build_link_model(model_name, in_dim=20, hidden_dim=16, num_layers=2, dropout=0.0, coalesce_edges=True)
    model.eval()
    with torch.no_grad():
        h_1x = model.encode(x, ei_1x, ea_1x)
        h_10x = model.encode(x, ei_10x, ea_10x)
    assert torch.allclose(h_1x, h_10x, atol=1e-5)

    if model_name in {"BiEdgeGATv2-Link", "SBiFAGAT-Link"}:
        ea_flipped = ea_1x.clone()
        ea_flipped[:, 0] = -ea_flipped[:, 0]
        ea_flipped[:, 1] = 1.0 - ea_flipped[:, 1]
        with torch.no_grad():
            h_flip = model.encode(x, ei_1x, ea_flipped)
        assert not torch.allclose(h_1x, h_flip, atol=1e-4)


def test_sgcn_and_sbifagat_distortion_fixes() -> None:
    torch.manual_seed(11)
    x = torch.randn(6, 20)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    ea = torch.tensor([[0.9, 0.1], [-0.9, 0.9], [0.8, 0.2], [-0.8, 0.8]], dtype=torch.float32)

    # SGCN: 1-layer and 2-layer must NOT add raw Layer-0 stem output onto [balanced || unbalanced]
    for n_layers in (1, 2):
        sgcn = SGCNLink(in_dim=20, hidden_dim=16, num_layers=n_layers, dropout=0.0)
        sgcn.eval()
        with torch.no_grad():
            h0 = sgcn.stem(x)
            h_out = sgcn.encode(x, ei, ea)
            # Manually trace Layer 0 without residual vs Layer 1 with residual
            pos_ref = to_undirected(ei[:, [0, 2]], num_nodes=6)
            neg_ref = to_undirected(ei[:, [1, 3]], num_nodes=6)
            out0 = sgcn.norms[0](torch.tanh(sgcn.convs[0](h0, pos_ref, neg_ref)))
            assert h_out.shape == (6, 16)
            if n_layers == 1:
                assert torch.allclose(h_out, out0, atol=1e-5)
            else:
                assert not torch.allclose(h_out, h0 + out0, atol=1e-4)

    # SBiFAGATLink: lin_root weight and bias must be strictly zeroed & frozen
    sbif = SBiFAGATLink(in_dim=20, hidden_dim=16, num_layers=2, dropout=0.0)
    for conv in (*sbif.convs_td, *sbif.convs_bu):
        assert conv.lin_root.weight.requires_grad is False
        assert conv.lin_root.bias.requires_grad is False
        assert torch.allclose(conv.lin_root.weight, torch.zeros_like(conv.lin_root.weight))
        assert torch.allclose(conv.lin_root.bias, torch.zeros_like(conv.lin_root.bias))

    logits = sbif(x, ei, ea, torch.tensor([[0, 2], [1, 3]], dtype=torch.long))
    logits.sum().backward()
    for conv in (*sbif.convs_td, *sbif.convs_bu):
        assert conv.lin_root.weight.grad is None
        assert conv.lin_root.bias.grad is None


def test_unseen_pair_transductive_and_unseen_evaluation(tmp_path: Path) -> None:
    ds_dir = tmp_path / "eval_subsets"
    _write_synthetic_polarized_corpus(ds_dir, emb_dim=16, add_unseen_future_user=True)
    ds = build_temporal_signed_link_dataset(ds_dir, dataset_name="vn")
    model = BiEdgeGATv2Link(in_dim=ds["in_dim"], hidden_dim=16, num_layers=1, heads=1, dropout=0.0)
    metrics, used_th = evaluate_link_split(
        model,
        ds["x"],
        ds["train"]["pairs"],
        ds["train"]["edge_attr"],
        ds["test"],
        tune_threshold=True,
    )
    assert 0.29 <= used_th <= 0.71
    for key, _, _ in LINK_PRED_METRICS:
        assert key in metrics

    # Empty unseen_pair_transductive mask yields clean NaN rather than raising
    empty_mask_split = dict(ds["test"])
    empty_mask_split["unseen_pair_mask"] = torch.zeros_like(ds["test"]["both_seen"])
    metrics_nan, _ = evaluate_link_split(
        model,
        ds["x"],
        ds["train"]["pairs"],
        ds["train"]["edge_attr"],
        empty_mask_split,
        threshold=0.5,
    )
    assert np.isnan(metrics_nan["unseen_pair_trans_roc_auc"])
    assert np.isnan(metrics_nan["unseen_pair_trans_pr_auc"])
    assert np.isnan(metrics_nan["unseen_pair_trans_macro_f1"])


def test_five_seed_stats_and_no_one_sided_star_hiding(tmp_path: Path) -> None:
    ds_dir = tmp_path / "five_seed_ds"
    _write_synthetic_polarized_corpus(ds_dir, emb_dim=16, add_unseen_future_user=True)

    bench = run_signed_link_benchmark(
        ds_dir,
        models=("PriorPairSignPersistence", "BiEdgeGATv2-Link", "SDGNN", "SBiFAGAT-Link"),
        seeds=OFFICIAL_SEEDS,
        hidden_dim=16,
        num_layers=1,
        epochs=2,
        patience=2,
        batch_size=32,
    )
    assert bench["seeds"] == list(OFFICIAL_SEEDS)
    assert len(bench["raw_results"]["SBiFAGAT-Link"]) == 5
    assert "wilcoxon_p_values" in bench
    assert "adjusted_wilcoxon_p_values" in bench
    assert bench["comparator_model"] != "PriorPairSignPersistence"

    # Construct synthetic result where comparator significantly BEATS reference_model
    # and assert stars are NOT hidden (both LaTeX and Markdown show '(-)' markers!)
    fake_summary = {
        "BiEdgeGATv2-Link": {k: (0.90, 0.01) for k, _, _ in LINK_PRED_METRICS},
        "SBiFAGAT-Link": {k: (0.60, 0.01) for k, _, _ in LINK_PRED_METRICS},
    }
    fake_bench = {
        "reference_model": "SBiFAGAT-Link",
        "comparator_model": "BiEdgeGATv2-Link",
        "summary_stats": fake_summary,
        "p_values": {k: 1e-4 for k, _, _ in LINK_PRED_METRICS},
        "adjusted_p_values": {k: 1e-3 for k, _, _ in LINK_PRED_METRICS},
        "bootstrap_p_values": {k: 0.0625 for k, _, _ in LINK_PRED_METRICS},
        "adjusted_bootstrap_p_values": {k: 0.125 for k, _, _ in LINK_PRED_METRICS},
        "wilcoxon_p_values": {k: 0.0625 for k, _, _ in LINK_PRED_METRICS},
        "adjusted_wilcoxon_p_values": {k: 0.125 for k, _, _ in LINK_PRED_METRICS},
        "nonparametric_test_label": "Exact Perm",
    }
    md = build_link_markdown_table(fake_bench)
    assert "**(-)" in md
    tex_path = export_link_latex_table(
        fake_bench,
        tmp_path / "no_hide.tex",
        caption="Two-sided test table.",
        label="tab:no_hide",
    )
    tex = tex_path.read_text(encoding="utf-8")
    assert "$^{**(-)}$" in tex
    assert "Wilcoxon" in tex
    assert find_unescaped_underscores_in_latex(tex) == []

    # Direct check of two-sided Wilcoxon helper
    w_p = compute_wilcoxon_signed_rank([0.9, 0.91, 0.89, 0.92, 0.90], [0.6, 0.61, 0.59, 0.62, 0.60])
    assert w_p == pytest.approx(0.0625, abs=1e-6)
    assert compute_wilcoxon_signed_rank([0.5, 0.5], [0.5, 0.5]) == 1.0

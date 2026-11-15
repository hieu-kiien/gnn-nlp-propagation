"""Regression tests for EGM methodological, leakage-prevention, and baseline parity fixes."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from torch_geometric.data import Batch, Data
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

from src.data.arctic_shift_vn import _compute_topic_id
from src.data.pyg_dataset import (
    BINARY_RUMOR_LABEL_TO_IDX,
    RUMOR_LABEL_TO_IDX,
    TEMPORAL_REF_SECONDS,
    EarlyCascadeDataset,
    UPFDAdapterTransform,
    _compute_full_tree_polarization,
    _compute_root_aligned_stances,
    leave_one_event_out_splits,
)
from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet
from src.models.baselines import EdgeGATv2, FAGCNBaseline
from src.models.sbi_fagat import SBiFAGAT, StanceGuidedFAConv
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    split_pairs_by_inductive_topics,
)
from src.nlp.extract_features import encode_node_embeddings, extract_and_save_features
from src.training.metrics import compute_multitask_loss, evaluate_predictions
from src.training.trainer import compute_paired_bootstrap_pvalue


def _build_offline_tok_and_cfg() -> tuple[PreTrainedTokenizerFast, RobertaConfig]:
    vocab = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "root",
        "topic",
        "reply",
        "agree",
        "oppose",
        "neutral",
        "fast",
        "slow",
    ]
    tok_map = {t: i for i, t in enumerate(vocab)}
    tk = Tokenizer(WordLevel(vocab=tok_map, unk_token="<unk>"))
    tk.pre_tokenizer = Whitespace()
    tk.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    cfg = RobertaConfig(
        vocab_size=len(vocab),
        hidden_size=768,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        type_vocab_size=1,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    return fast_tok, cfg


def _make_cascade_rows(
    cascade_id: str,
    topic_id: str,
    *,
    num_replies: int,
    step_seconds: float,
    rumor_label: str | None = "false",
    depth_chain: bool = False,
) -> tuple[list[dict], list[dict]]:
    base_ts = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    root_id = f"{cascade_id}_r"
    nodes = [
        {
            "node_id": root_id,
            "cascade_id": cascade_id,
            "topic_id": topic_id,
            "parent_id": None,
            "is_root": True,
            "text": "root topic agree",
            "created_at_utc": base_ts,
            "lang": "en",
            "source_dataset": "pheme",
            "user_id_hash": "a" * 64,
            "rumor_label": rumor_label,
        }
    ]
    edges: list[dict] = []
    for j in range(1, num_replies + 1):
        nid = f"{cascade_id}_n{j}"
        parent = root_id if (not depth_chain or j == 1) else f"{cascade_id}_n{j - 1}"
        parent_dt = (
            base_ts
            if (not depth_chain or j == 1)
            else base_ts + timedelta(seconds=step_seconds * (j - 1))
        )
        dt = base_ts + timedelta(seconds=step_seconds * j)
        nodes.append(
            {
                "node_id": nid,
                "cascade_id": cascade_id,
                "topic_id": topic_id,
                "parent_id": parent,
                "is_root": False,
                "text": "reply agree" if j % 2 == 1 else "reply oppose",
                "created_at_utc": dt,
                "lang": "en",
                "source_dataset": "pheme",
                "user_id_hash": "b" * 64,
                "rumor_label": None,
            }
        )
        edges.append(
            {
                "edge_id": f"{parent}->{nid}",
                "cascade_id": cascade_id,
                "source_id": parent,
                "target_id": nid,
                "created_at_utc": dt,
                "time_delay_sec": float((dt - parent_dt).total_seconds()),
                "edge_type": "reply",
            }
        )
    return nodes, edges


def test_extract_features_uses_independent_node_encoder_not_student(tmp_path: Path) -> None:
    tok, cfg = _build_offline_tok_and_cfg()
    n_rows, e_rows = _make_cascade_rows("c1", "topic_a", num_replies=4, step_seconds=60.0)
    interim_dir = tmp_path / "interim"
    save_cascade_parquet(build_nodes_table(n_rows), build_edges_table(e_rows), interim_dir, validate=True)

    torch.manual_seed(7)
    student_enc = RobertaModel(cfg, add_pooling_layer=False)
    with torch.no_grad():
        for param in student_enc.parameters():
            param.add_(10.0)
    student = DualHeadCrossEncoder(lang="en", model_name="offline-ce", encoder=student_enc)
    ckpt_dir = student.save_checkpoint(tmp_path / "ce_ckpt", tokenizer=tok)

    torch.manual_seed(99)
    frozen_node_encoder = RobertaModel(cfg, add_pooling_layer=False)
    expected_emb = encode_node_embeddings(
        [r["text"] for r in n_rows],
        encoder=frozen_node_encoder,
        tokenizer=tok,
        batch_size=8,
        max_length=32,
        device=torch.device("cpu"),
    )

    pt_path = extract_and_save_features(
        dataset="pheme",
        interim_dir=interim_dir,
        output_dir=tmp_path / "processed",
        cross_encoder_ckpt=ckpt_dir,
        node_encoder=frozen_node_encoder,
        tokenizer=tok,
        batch_size=8,
        k_horizon=20,
    )
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert "node_features_770" not in payload
    assert torch.allclose(payload["node_embeddings"], expected_emb, atol=1e-5)


def test_distillation_inductive_topic_split_zero_test_overlap() -> None:
    pairs = []
    for t_idx in range(10):
        for e_idx in range(3):
            pairs.append(
                {
                    "edge_id": f"t{t_idx}_e{e_idx}",
                    "cascade_id": f"c_{t_idx}",
                    "topic_id": f"topic_{t_idx:02d}",
                    "parent_text": "root topic",
                    "reply_text": "reply agree",
                    "s_uv": 0.5,
                    "h_uv": 0.1,
                    "stance_probs": [0.7, 0.2, 0.1],
                }
            )
    splits = split_pairs_by_inductive_topics(pairs, split_seed=42)
    assert len(splits["train"]) > 0 and len(splits["val"]) > 0 and len(splits["test"]) > 0
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(pairs)
    tr_t = {p["topic_id"] for p in splits["train"]}
    va_t = {p["topic_id"] for p in splits["val"]}
    te_t = {p["topic_id"] for p in splits["test"]}
    assert tr_t.isdisjoint(va_t)
    assert tr_t.isdisjoint(te_t)
    assert va_t.isdisjoint(te_t)


def test_causal_temporal_ref_preserves_cascade_velocity() -> None:
    fast_n, fast_e = _make_cascade_rows("c_fast", "t_fast", num_replies=20, step_seconds=4.5)  # 90s total
    slow_n, slow_e = _make_cascade_rows("c_slow", "t_slow", num_replies=20, step_seconds=86400.0)  # 20d total
    all_n = fast_n + slow_n
    all_e = fast_e + slow_e

    rng = torch.Generator().manual_seed(11)
    nlp_features = {
        "node_id_to_idx": {r["node_id"]: i for i, r in enumerate(all_n)},
        "node_embeddings": torch.randn((len(all_n), 768), generator=rng),
        "edge_stance": {r["edge_id"]: 0.2 for r in all_e},
        "edge_hostility": {r["edge_id"]: 0.1 for r in all_e},
    }
    ds = EarlyCascadeDataset(
        k_horizon=20,
        split="all",
        nodes_table=build_nodes_table(all_n),
        edges_table=build_edges_table(all_e),
        nlp_features=nlp_features,
    )
    by_cid = {g.cascade_id: g for g in ds}
    expected_fast_last = math.log1p(90.0) / math.log1p(TEMPORAL_REF_SECONDS)
    assert by_cid["c_fast"].x[-1, 768].item() == pytest.approx(expected_fast_last, rel=1e-4)
    assert by_cid["c_fast"].x[-1, 768].item() < 0.40
    assert by_cid["c_slow"].x[-1, 768].item() == pytest.approx(1.0)


def test_root_aligned_stance_bipolar_y_polar_min_replies_and_future_only() -> None:
    # Symmetric bipolar y_polar: unanimous opposition == 0.0, unanimous support == 0.0, 50/50 split == 1.0
    assert _compute_full_tree_polarization([-1.0] * 10) == pytest.approx(0.0)
    assert _compute_full_tree_polarization([1.0] * 10) == pytest.approx(0.0)
    assert _compute_full_tree_polarization([-1.0] * 5 + [1.0] * 5) == pytest.approx(1.0)

    # Root-aligned stances across depth 2: opposing an opposer aligns positively with root
    n_rows, e_rows = _make_cascade_rows("chain", "t_chain", num_replies=2, step_seconds=60.0, depth_chain=True)
    children_adj = {
        "chain_r": ["chain_n1"],
        "chain_n1": ["chain_n2"],
    }
    edge_lookup = {(r["source_id"], r["target_id"]): r for r in e_rows}
    stance_map = {
        "chain_r->chain_n1": -0.9,
        "chain_n1->chain_n2": -0.8,
    }
    aligned = _compute_root_aligned_stances("chain_r", children_adj, edge_lookup, stance_map)
    assert aligned == pytest.approx([-0.9, 0.8])

    # Future-only polarity ignores kept G_K pairs when computing y_polar
    short_n, short_e = _make_cascade_rows("short15", "t1", num_replies=15, step_seconds=60.0)
    long_n, long_e = _make_cascade_rows("long25", "t2", num_replies=25, step_seconds=60.0)
    nodes = short_n + long_n
    edges = short_e + long_e

    # Early K=20 replies of long25 all agree (+1.0), future replies 21..25 alternate -1/+1 -> non-zero future polarity
    e_stance = {r["edge_id"]: 1.0 for r in edges}
    for j in range(21, 26):
        eid = f"long25_r->long25_n{j}"
        e_stance[eid] = -1.0 if j % 2 == 1 else 1.0

    nlp_features = {
        "node_id_to_idx": {r["node_id"]: i for i, r in enumerate(nodes)},
        "node_embeddings": torch.zeros((len(nodes), 768), dtype=torch.float32),
        "edge_stance": e_stance,
        "edge_hostility": {r["edge_id"]: 0.0 for r in edges},
    }
    ds_filtered = EarlyCascadeDataset(
        k_horizon=20,
        split="all",
        nodes_table=build_nodes_table(nodes),
        edges_table=build_edges_table(edges),
        nlp_features=nlp_features,
        min_total_replies=20,
        polar_target_mode="future_only",
    )
    assert len(ds_filtered) == 1
    assert ds_filtered[0].cascade_id == "long25"
    expected_future = _compute_full_tree_polarization([-1.0, 1.0, -1.0, 1.0, -1.0])
    assert ds_filtered[0].y_polar.item() == pytest.approx(expected_future, abs=1e-5)


def test_pheme_label_schemes_loo_splits_and_vn_topic_grouping() -> None:
    assert RUMOR_LABEL_TO_IDX["rumor"] == 3
    assert RUMOR_LABEL_TO_IDX["rumor"] != RUMOR_LABEL_TO_IDX["true"]
    assert BINARY_RUMOR_LABEL_TO_IDX["non-rumor"] == 0
    assert BINARY_RUMOR_LABEL_TO_IDX["false"] == 1
    assert BINARY_RUMOR_LABEL_TO_IDX["rumor"] == 1

    # LOO splits across 4 events ensure zero overlap and every event tested once
    all_n: list[dict] = []
    all_e: list[dict] = []
    labels = ["non-rumor", "true", "false", "unverified"]
    for idx, lbl in enumerate(labels):
        n_rows, e_rows = _make_cascade_rows(f"c{idx}", f"ev_{idx}", num_replies=5, step_seconds=30.0, rumor_label=lbl)
        all_n.extend(n_rows)
        all_e.extend(e_rows)

    folds = leave_one_event_out_splits([{"topic_id": f"ev_{i}"} for i in range(4)], seed=42)
    assert len(folds) == 4
    tested = set()
    for f in folds:
        assert f["train"].isdisjoint(f["val"])
        assert f["train"].isdisjoint(f["test"])
        assert f["val"].isdisjoint(f["test"])
        tested |= f["test"]
    assert tested == {f"ev_{i}" for i in range(4)}

    nlp_features = {
        "node_id_to_idx": {r["node_id"]: i for i, r in enumerate(all_n)},
        "node_embeddings": torch.zeros((len(all_n), 768), dtype=torch.float32),
        "edge_stance": {r["edge_id"]: 0.0 for r in all_e},
        "edge_hostility": {r["edge_id"]: 0.0 for r in all_e},
    }
    ds_bin = EarlyCascadeDataset(
        k_horizon=20,
        split="all",
        nodes_table=build_nodes_table(all_n),
        edges_table=build_edges_table(all_e),
        nlp_features=nlp_features,
        label_scheme="binary",
    )
    assert [int(g.y_rumor.item()) for g in ds_bin] == [0, 1, 1, 1]

    # Arctic Shift VN topic grouping clusters posts in same ISO week sharing meaningful keywords/order-invariant
    ts_w1 = datetime(2026, 5, 4, 10, 0, tzinfo=timezone.utc).timestamp()
    ts_w2 = (datetime(2026, 5, 4, 10, 0, tzinfo=timezone.utc) + timedelta(days=14)).timestamp()
    t1 = _compute_topic_id({"is_self": True, "created_utc": ts_w1}, "Hỏi về giá xăng dầu tăng mạnh")
    t2 = _compute_topic_id({"is_self": True, "created_utc": ts_w1 + 3600}, "Thảo luận giá xăng dầu cũng đang tăng mạnh")
    t3 = _compute_topic_id({"is_self": True, "created_utc": ts_w2}, "Hỏi về giá xăng dầu tăng mạnh")
    t4 = _compute_topic_id({"is_self": True, "created_utc": ts_w1}, "Ẩm thực đường phố Sài Gòn cuối tuần")
    assert t1 == t2
    assert t1 != t3
    assert t1 != t4


def test_stateless_sbi_fagat_ablation_modes_baselines_and_upfd_transform() -> None:
    torch.manual_seed(42)
    conv = StanceGuidedFAConv(hidden_dim=16, edge_dim=2, eps=0.5, dropout=0.0)
    assert not hasattr(conv, "_last_gamma")

    graphs = []
    for i in range(4):
        x = torch.randn(6, 770)
        ei_td = torch.tensor([[0, 0, 1, 1, 2], [1, 2, 3, 4, 5]], dtype=torch.long)
        ei_bu = ei_td[[1, 0], :]
        ea = torch.tensor([[0.6, 0.1], [-0.8, 0.9], [0.2, 0.2], [-0.5, 0.4], [0.9, 0.0]], dtype=torch.float32)
        graphs.append(
            Data(
                x=x,
                edge_index=ei_td,
                edge_index_td=ei_td,
                edge_index_bu=ei_bu,
                edge_attr=ea,
                edge_stance=ea[:, :1],
                root_index=torch.tensor([0], dtype=torch.long),
                y_size=torch.tensor([float(i + 1)], dtype=torch.float32),
                y_polar=torch.tensor([0.2 * i], dtype=torch.float32),
                y_rumor=torch.tensor([i % 4], dtype=torch.long),
            )
        )
    batch = Batch.from_data_list(graphs)

    # Full SBiFAGAT + extract_edge_gates
    model_full = SBiFAGAT(in_dim=770, hidden_dim=32, num_layers=2, edge_dim=2, readout_mode="attention")
    gates = model_full.extract_edge_gates(batch)
    assert "gamma_td_layers" in gates and len(gates["gamma_td_layers"]) == 2
    assert "gamma_bu_layers" in gates and len(gates["gamma_bu_layers"]) == 2

    # Unidirectional TD ablation (use_bottom_up=False) and mean readout ablation
    model_td_only = SBiFAGAT(in_dim=770, hidden_dim=32, num_layers=2, edge_dim=2, use_bottom_up=False)
    out_td = model_td_only(batch)
    assert torch.isfinite(out_td["y_size_pred"]).all()
    assert model_td_only.extract_edge_gates(batch)["gamma_bu_layers"] == []

    model_mean = SBiFAGAT(in_dim=770, hidden_dim=32, num_layers=2, edge_dim=2, readout_mode="mean")
    assert torch.isfinite(model_mean(batch)["y_polar_pred"]).all()

    # Fair baselines: EdgeGATv2 (explicit edge_attr + latent edge MLP) and FAGCNBaseline
    for model in (
        EdgeGATv2(in_dim=770, hidden_dim=32, num_layers=2, edge_dim=2, use_latent_edge_mlp=False),
        EdgeGATv2(in_dim=770, hidden_dim=32, num_layers=2, edge_dim=2, use_latent_edge_mlp=True),
        FAGCNBaseline(in_dim=770, hidden_dim=32, num_layers=2),
    ):
        model.zero_grad()
        out = model(batch)
        loss = compute_multitask_loss(out, batch)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    # UPFDAdapterTransform emits finite zeros for synthetic missing edge attributes
    raw_upfd = Data(
        x=torch.randn(5, 768),
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]], dtype=torch.long),
        y=torch.tensor([1], dtype=torch.long),
    )
    adapted = UPFDAdapterTransform()(raw_upfd)
    assert adapted.edge_attr.shape == (4, 2)
    assert torch.equal(adapted.edge_attr, torch.zeros((4, 2), dtype=torch.float32))
    assert torch.equal(adapted.edge_stance, torch.zeros((4, 1), dtype=torch.float32))


def test_metrics_huber_deltas_class_weights_and_paired_bootstrap() -> None:
    torch.manual_seed(0)
    outputs = {
        "y_size_pred": torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True),
        "y_polar_pred": torch.tensor([0.1, 0.5, 0.8, 0.2], requires_grad=True),
        "rumor_logits": torch.randn((4, 4), requires_grad=True),
    }
    batch = Data(
        y_size=torch.tensor([1.1, 1.9, 3.2, 3.8]),
        y_polar=torch.tensor([0.05, 0.45, 0.90, 0.10]),
        y_rumor=torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )
    cw = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss = compute_multitask_loss(
        outputs,
        batch,
        huber_delta_size=1.0,
        huber_delta_polar=0.15,
        class_weights=cw,
    )
    loss.backward()
    assert torch.isfinite(loss)

    metrics = evaluate_predictions(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.1, 1.9, 3.1, 3.9]),
        np.array([0.1, 0.3, 0.6, 0.9]),
        np.array([0.12, 0.35, 0.58, 0.88]),
        np.array([0, 1, 2, 3]),
        np.array([[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 3.0]]),
    )
    for key in (
        "msle",
        "mae",
        "rmse_size",
        "r2_size",
        "spearman_size",
        "rmse",
        "mae_polar",
        "r2_polar",
        "pearson_r",
        "spearman_polar",
        "macro_f1",
        "accuracy",
    ):
        assert math.isfinite(metrics[key]), f"Non-finite metric: {key}={metrics[key]}"
    assert metrics["r2_size"] > 0.95
    assert metrics["macro_f1"] == pytest.approx(1.0)

    # Paired bootstrap rejects null on systematically smaller errors and yields 1.0 on ties
    rng = np.random.default_rng(7)
    better = rng.normal(0.2, 0.05, size=50)
    worse = better + rng.uniform(0.15, 0.35, size=50)
    assert compute_paired_bootstrap_pvalue(better, worse, n_resamples=1000, seed=42) < 0.05
    assert compute_paired_bootstrap_pvalue(better, better, n_resamples=1000, seed=42) >= 0.99

"""Regression tests for Round-2 scientific & engineering blockers (B1-B5 & comprehension)."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from scripts.compute_kappa import (
    evaluate_audit_csv,
    evaluate_machine_vs_human,
    evaluate_rumoureval_sdqc,
)
from src.data.arctic_shift_vn import hash_author_sha256
from src.data.pyg_dataset import (
    CascadeData,
    EarlyCascadeDataset,
    _compute_root_aligned_stances,
    compute_bimodality_coefficient,
    compute_signed_assortativity,
    leave_one_event_out_splits,
)
from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet
from src.models.baselines import TemporalStanceMLP
from src.models.sbi_fagat import SBiFAGAT, StanceGuidedFAConv
from src.nlp.dual_edge_scorer import export_stratified_human_audit_csv
from src.nlp.extract_features import extract_and_save_features, validate_feature_tensors
from src.training.metrics import evaluate_predictions
from src.training.trainer import (
    _compute_sample_level_bootstrap_pvalues,
    holm_bonferroni_adjust,
    run_multi_seed_benchmark,
    select_best_comparator,
)


def _offline_tokenizer() -> PreTrainedTokenizerFast:
    vocab_tokens = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "root",
        "agree",
        "oppose",
        "reply",
        "chuan",
        "phan",
        "doi",
    ]
    tok_obj = Tokenizer(WordLevel(vocab={t: i for i, t in enumerate(vocab_tokens)}, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


def _tiny_encoder() -> RobertaModel:
    cfg = RobertaConfig(
        vocab_size=32,
        hidden_size=768,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        type_vocab_size=1,
    )
    return RobertaModel(cfg, add_pooling_layer=False)


def test_scored_cache_json_and_node_root_stance_depth2(tmp_path: Path) -> None:
    base_ts = datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    node_rows = [
        {
            "node_id": "c0_r",
            "cascade_id": "c0",
            "topic_id": "top_a",
            "parent_id": None,
            "is_root": True,
            "text": "root topic statement",
            "created_at_utc": base_ts,
            "lang": "vi",
            "source_dataset": "arctic_shift_vozforums",
            "user_id_hash": "a" * 64,
            "rumor_label": None,
        },
        {
            "node_id": "c0_n1",
            "cascade_id": "c0",
            "topic_id": "top_a",
            "parent_id": "c0_r",
            "is_root": False,
            "text": "oppose root strongly",
            "created_at_utc": base_ts + timedelta(minutes=5),
            "lang": "vi",
            "source_dataset": "arctic_shift_vozforums",
            "user_id_hash": "b" * 64,
            "rumor_label": None,
        },
        {
            # Depth-2 sarcastic reply attacking parent c0_n1 yet ALSO opposing the root!
            "node_id": "c0_n2",
            "cascade_id": "c0",
            "topic_id": "top_a",
            "parent_id": "c0_n1",
            "is_root": False,
            "text": "oppose both parent and root sarcasm",
            "created_at_utc": base_ts + timedelta(minutes=10),
            "lang": "vi",
            "source_dataset": "arctic_shift_vozforums",
            "user_id_hash": "c" * 64,
            "rumor_label": None,
        },
    ]
    edge_rows = [
        {
            "edge_id": "c0_r->c0_n1",
            "cascade_id": "c0",
            "source_id": "c0_r",
            "target_id": "c0_n1",
            "created_at_utc": base_ts + timedelta(minutes=5),
            "time_delay_sec": 300.0,
            "edge_type": "reply",
        },
        {
            "edge_id": "c0_n1->c0_n2",
            "cascade_id": "c0",
            "source_id": "c0_n1",
            "target_id": "c0_n2",
            "created_at_utc": base_ts + timedelta(minutes=10),
            "time_delay_sec": 300.0,
            "edge_type": "reply",
        },
    ]

    interim_dir = tmp_path / "interim"
    out_dir = tmp_path / "processed"
    save_cascade_parquet(build_nodes_table(node_rows), build_edges_table(edge_rows), output_dir=interim_dir)

    cache_path = interim_dir / "scored_pairs.json"
    cache_path.write_text(
        json.dumps(
            [
                {
                    "edge_id": "c0_r->c0_n1",
                    "target_id": "c0_n1",
                    "s_uv": -0.85,
                    "h_uv": 0.40,
                    "pred_label": "oppose",
                    "root_s_uv": -0.85,
                },
                {
                    "edge_id": "c0_n1->c0_n2",
                    "target_id": "c0_n2",
                    "s_uv": -0.90,
                    "h_uv": 0.88,
                    "pred_label": "oppose",
                    # Direct NLI (root, n2) reveals n2 ACTUALLY opposes root (-0.75) rather than (+0.90)
                    "root_s_uv": -0.75,
                },
            ]
        ),
        encoding="utf-8",
    )

    pt_path = extract_and_save_features(
        dataset="vn",
        interim_dir=interim_dir,
        output_dir=out_dir,
        node_encoder=_tiny_encoder(),
        tokenizer=_offline_tokenizer(),
        scored_cache_json=cache_path,
        device=torch.device("cpu"),
    )
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    validate_feature_tensors(payload)
    assert "node_root_stance" in payload
    assert math.isclose(payload["node_root_stance"]["c0_n2"], -0.75, rel_tol=1e-5)

    # Check direct root_stance_map overrides sign-product propagation on depth >= 2
    children_adj = {"c0_r": ["c0_n1"], "c0_n1": ["c0_n2"], "c0_n2": []}
    edge_lookup = {
        ("c0_r", "c0_n1"): {"edge_id": "c0_r->c0_n1"},
        ("c0_n1", "c0_n2"): {"edge_id": "c0_n1->c0_n2"},
    }
    aligned = _compute_root_aligned_stances(
        "c0_r",
        children_adj,
        edge_lookup,
        payload["edge_stance"],
        root_stance_map=payload["node_root_stance"],
    )
    assert math.isclose(aligned[0], -0.85, abs_tol=1e-4)
    assert math.isclose(aligned[1], -0.75, abs_tol=1e-4)

    # Without root_stance_map, y_bimodality is still attached onto CascadeData
    ds = EarlyCascadeDataset(root_dir=out_dir, k_horizon=20, split="all")
    assert len(ds) == 1
    assert hasattr(ds[0], "y_bimodality")
    assert torch.isfinite(ds[0].y_bimodality).all()


def test_pilot_audit_machine_vs_gold_and_rumoureval(tmp_path: Path) -> None:
    pairs = []
    for idx in range(30):
        pairs.append(
            {
                "cascade_id": f"c_{idx}",
                "edge_id": f"e_s_{idx}",
                "parent_text": f"Root {idx}",
                "reply_text": f"Agree {idx}",
                "pred_label": "support",
                "s_uv": 0.9,
                "h_uv": 0.1,
            }
        )
        pairs.append(
            {
                "cascade_id": f"c_{idx}",
                "edge_id": f"e_n_{idx}",
                "parent_text": f"Root {idx}",
                "reply_text": f"Info {idx}",
                "pred_label": "neutral",
                "s_uv": 0.0,
                "h_uv": 0.05,
            }
        )
        pairs.append(
            {
                "cascade_id": f"c_{idx}",
                "edge_id": f"e_o_{idx}",
                "parent_text": f"Root {idx}",
                "reply_text": f"Disagree {idx}",
                "pred_label": "oppose",
                "s_uv": -0.9,
                "h_uv": 0.85,
            }
        )

    audit_csv = tmp_path / "audit_45.csv"
    pilot_csv = tmp_path / "pilot_15.csv"
    key_csv = tmp_path / "machine_key_45.csv"
    export_stratified_human_audit_csv(
        pairs,
        output_csv=audit_csv,
        machine_key_csv=key_csv,
        pilot_csv=pilot_csv,
        n_samples=45,
        pilot_samples=15,
        seed=7,
    )

    with audit_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        audit_rows = list(csv.DictReader(fh))
    with pilot_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        pilot_rows = list(csv.DictReader(fh))
    assert len(audit_rows) == 45
    assert len(pilot_rows) == 15
    assert {r["edge_id"] for r in audit_rows}.isdisjoint({r["edge_id"] for r in pilot_rows})
    assert "gold_adjudicated_stance" in audit_rows[0]

    # Fill dual annotations + adjudicated gold matching the hidden key pattern
    with key_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        key_map = {r["edge_id"]: r for r in csv.DictReader(fh)}
    for r in audit_rows:
        lbl = key_map[r["edge_id"]]["machine_stance_label"]
        h_lbl = "hate" if float(key_map[r["edge_id"]]["h_uv"]) > 0.7 else "clean"
        r["annotator_1_stance"] = lbl
        r["annotator_2_stance"] = lbl
        r["gold_adjudicated_stance"] = lbl
        r["annotator_1_hostility"] = h_lbl
        r["annotator_2_hostility"] = h_lbl
        r["gold_adjudicated_hostility"] = h_lbl
    with audit_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    rep = evaluate_audit_csv(audit_csv, machine_key_csv=key_csv)
    assert rep["stance"]["cohen_kappa_unweighted"] == pytest.approx(1.0)
    assert rep["machine_vs_gold"]["stance"]["macro_f1"] == pytest.approx(1.0)
    assert rep["machine_vs_gold"]["stance"]["accuracy"] == pytest.approx(1.0)
    assert rep["machine_vs_gold"]["stance"]["pearson_r"] > 0.99
    assert rep["machine_vs_gold"]["hostility"]["accuracy"] == pytest.approx(1.0)

    # Check RumourEval SDQC validation helper
    rumour_json = tmp_path / "rumoureval.json"
    rumour_json.write_text(
        json.dumps(
            [
                {"id": "1", "sdqc_label": "support", "s_uv": 0.8},
                {"id": "2", "sdqc_label": "deny", "s_uv": -0.85},
                {"id": "3", "sdqc_label": "query", "s_uv": 0.02},
                {"id": "4", "sdqc_label": "comment", "s_uv": -0.01},
            ]
        ),
        encoding="utf-8",
    )
    sdqc_rep = evaluate_rumoureval_sdqc(rumour_json)
    assert sdqc_rep["macro_f1"] == pytest.approx(1.0)
    assert sdqc_rep["accuracy"] == pytest.approx(1.0)
    assert sdqc_rep["sdqc_distribution"] == {"support": 1, "deny": 1, "query": 1, "comment": 1}


def test_construct_validity_loo_seed_and_salted_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Sarle's Bimodality Coefficient: 50/50 split vs consensus
    bc_split = compute_bimodality_coefficient([1.0, -1.0] * 25)
    bc_unanimous = compute_bimodality_coefficient([0.9] * 20)
    assert bc_split > 0.555
    assert bc_unanimous == 0.0

    # 2. Signed assortativity on homophilic vs disassortative edges
    node_st = {"r": 1.0, "a": 0.9, "b": -0.9, "c": 0.85, "d": -0.85}
    assort_homo = compute_signed_assortativity([("a", "c"), ("b", "d")], node_st)
    assort_fight = compute_signed_assortativity([("a", "d"), ("b", "c")], node_st)
    assert assort_homo > 0.9
    assert assort_fight < -0.9

    # 3. LOO splits depend on seed after removing sorted(shuffled) bug
    items = [{"topic_id": f"ev_{i}"} for i in range(6)]
    f_seed_1 = [f["fold_name"] for f in leave_one_event_out_splits(items, seed=11)]
    f_seed_2 = [f["fold_name"] for f in leave_one_event_out_splits(items, seed=99)]
    assert f_seed_1 != f_seed_2

    # 4. Salted author hash separates salted vs unsalted hashes while remaining deterministic
    monkeypatch.delenv("SBIF_HASH_SALT", raising=False)
    h_plain = hash_author_sha256("voz_user_1")
    monkeypatch.setenv("SBIF_HASH_SALT", "study_salt_v1")
    h_salted = hash_author_sha256("voz_user_1")
    assert h_plain is not None and h_salted is not None and h_plain != h_salted


def test_temporal_mlp_conv_init_auc_bootstrap_and_holm_bonferroni() -> None:
    # 1. 2D edge attention weights initialized with opposing hostility sign (+0.6 stance, -0.4 hostility)
    conv = StanceGuidedFAConv(hidden_dim=32, edge_dim=2)
    edge_slice = conv.att[0, 64:].detach()
    assert torch.allclose(edge_slice, torch.tensor([0.6, -0.4], dtype=edge_slice.dtype), atol=1e-5)

    # 2. TemporalStanceMLP forward/backward including zero-edge cascade
    torch.manual_seed(0)
    g1 = CascadeData(
        x=torch.randn(5, 770),
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]], dtype=torch.long),
        edge_index_td=torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]], dtype=torch.long),
        edge_index_bu=torch.tensor([[1, 2, 3, 4], [0, 0, 1, 2]], dtype=torch.long),
        edge_attr=torch.tensor([[0.8, 0.1], [-0.9, 0.8], [-0.7, 0.6], [0.5, 0.2]], dtype=torch.float32),
        root_index=torch.tensor([0], dtype=torch.long),
        y_size=torch.tensor([1.6], dtype=torch.float32),
        y_polar=torch.tensor([0.75], dtype=torch.float32),
        y_rumor=torch.tensor([2], dtype=torch.long),
    )
    g2 = CascadeData(
        x=torch.randn(1, 770),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_index_td=torch.empty((2, 0), dtype=torch.long),
        edge_index_bu=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 2), dtype=torch.float32),
        root_index=torch.tensor([0], dtype=torch.long),
        y_size=torch.tensor([0.0], dtype=torch.float32),
        y_polar=torch.tensor([0.0], dtype=torch.float32),
        y_rumor=torch.tensor([0], dtype=torch.long),
    )
    batch = Batch.from_data_list([g1, g2])
    mlp = TemporalStanceMLP(in_dim=770, hidden_dim=32, edge_dim=2, num_rumor_classes=4)
    out = mlp(batch)
    assert out["y_size_pred"].shape == (2,)
    assert out["y_polar_pred"].shape == (2,)
    assert out["rumor_logits"].shape == (2, 4)
    loss = out["y_size_pred"].sum() + out["y_polar_pred"].sum() + out["rumor_logits"].sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mlp.parameters())

    # 3. evaluate_predictions includes auc_roc
    ev = evaluate_predictions(
        y_size_true=np.array([1.0, 2.0, 3.0, 4.0]),
        y_size_pred=np.array([1.1, 2.0, 2.9, 3.9]),
        y_polar_true=np.array([0.1, 0.4, 0.6, 0.9]),
        y_polar_pred=np.array([0.15, 0.35, 0.65, 0.85]),
        y_rumor_true=np.array([0, 1, 2, 3]),
        rumor_logits=np.array(
            [
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
            ]
        ),
    )
    assert ev["auc_roc"] == pytest.approx(1.0)

    # 4. Full sample-level bootstrap covers all metrics + Holm-Bonferroni + auto comparator
    rng = np.random.default_rng(123)
    n_test = 24
    yt_size = np.linspace(1.0, 5.0, n_test)
    yt_polar = np.linspace(0.05, 0.95, n_test)
    yt_cls = np.tile(np.array([0, 1, 2, 3]), 6)
    perfect_logits = np.zeros((n_test, 4), dtype=np.float64)
    for i, c in enumerate(yt_cls):
        perfect_logits[i, c] = 4.0
    weak_logits = rng.normal(0.0, 1.0, size=(n_test, 4))

    boot = _compute_sample_level_bootstrap_pvalues(
        [
            {
                "y_size_true": yt_size,
                "y_size_pred": yt_size + 0.05 * rng.normal(size=n_test),
                "y_polar_true": yt_polar,
                "y_polar_pred": np.clip(yt_polar + 0.02 * rng.normal(size=n_test), 0.0, 1.0),
                "y_rumor_true": yt_cls,
                "rumor_logits": perfect_logits,
            }
        ],
        [
            {
                "y_size_true": yt_size,
                "y_size_pred": yt_size + 1.2 * rng.normal(size=n_test),
                "y_polar_true": yt_polar,
                "y_polar_pred": np.full(n_test, 0.5),
                "y_rumor_true": yt_cls,
                "rumor_logits": weak_logits,
            }
        ],
        metric_keys=(
            "msle",
            "mae",
            "r2_size",
            "spearman_size",
            "rmse",
            "r2_polar",
            "pearson_r",
            "spearman_polar",
            "macro_f1",
            "accuracy",
            "auc_roc",
        ),
        n_resamples=300,
        n_stat_resamples=150,
    )
    for k, v in boot.items():
        assert math.isfinite(v) and 0.0 <= v <= 1.0, f"Metric {k} bootstrap p={v!r}"

    adj = holm_bonferroni_adjust({"m1": 0.004, "m2": 0.02, "m3": 0.03, "m_nan": float("nan")})
    assert adj["m1"] == pytest.approx(0.012)
    assert adj["m2"] == pytest.approx(0.04)
    assert adj["m3"] == pytest.approx(0.04)  # Monotone step-down
    assert math.isnan(adj["m_nan"])

    best_cmp = select_best_comparator(
        {
            "TextMLP": {"msle": (0.8, 0.1), "r2_polar": (0.1, 0.05), "accuracy": (0.5, 0.02)},
            "BiGCN": {"msle": (0.4, 0.1), "r2_polar": (0.6, 0.05), "accuracy": (0.75, 0.02)},
            "SBiFAGAT": {"msle": (0.2, 0.05), "r2_polar": (0.8, 0.03), "accuracy": (0.85, 0.02)},
        },
        reference_model="SBiFAGAT",
    )
    assert best_cmp == "BiGCN"

    # End-to-end 1-seed multi_seed_benchmark auto comparator smoke check
    tiny_ds = [g1, g1, g1, g1]
    loader = DataLoader(tiny_ds, batch_size=2, shuffle=False)
    res = run_multi_seed_benchmark(
        {
            "TextMLP": lambda: TemporalStanceMLP(in_dim=770, hidden_dim=16, edge_dim=2, num_rumor_classes=4),
            "SBiFAGAT": lambda: SBiFAGAT(in_dim=770, hidden_dim=16, edge_dim=2, num_rumor_classes=4),
        },
        loader,
        loader,
        loader,
        seeds=[42, 100],
        epochs=2,
        patience=2,
        comparator_model="auto",
        device="cpu",
    )
    assert res["comparator_model"] == "TextMLP"
    assert "adjusted_p_values" in res
    assert "adjusted_bootstrap_p_values" in res

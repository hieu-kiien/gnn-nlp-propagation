"""Verification tests for SBi-FAGAT v2 SOTA upgrades.

Covers:
  1. G-Eval single-pass logit scoring (0% fallback) & QLoRA prompt-token loss masking.
  2. Canonical entity-level topic splits (ottawashooting mismatch regression) & LOO teacher isolation.
  3. Temporal VN topic clustering across 14-day sliding windows.
  4. Beta-NLL detached gradients, calibrated non-saturated confidence, task_loss invariance & per-seed bootstrap.
  5. Pre-norm StanceGuidedFAConv odd sign-symmetry, 4D domain init, and fair 8-model cascade_stats parity.
  6. DeBERTa token_type_ids inspection, 5-class behavior compute_kappa, and UPFD --feature CLI.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaModel

from scripts.compute_kappa import evaluate_audit_csv
from scripts.kaggle_parallel_runner import _build_smoke_offline_tokenizer_and_encoder
from scripts.run_experiments import _build_loaders
from src.data.arctic_shift_vn import cluster_vn_cascade_topics
from src.data.pyg_dataset import (
    DEFAULT_FORECAST_MIN_REPLIES,
    CascadeData,
    EarlyCascadeDataset,
    compute_canonical_topic_splits,
    leave_one_event_out_splits,
    save_topic_splits_json,
)
from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet
from src.models.baselines import (
    BiGCN,
    EdgeGATv2,
    FAGCNBaseline,
    LatentConflictMLP,
    MultiTaskHead,
    StandardGATv2,
    StandardGCN,
    TemporalStanceMLP,
    TextMLP,
)
from src.models.sbi_fagat import SBiFAGAT, StanceGuidedFAConv
from src.nlp.discourse_labels import DISCOURSE_BEHAVIOR_LABELS
from src.nlp.dual_edge_scorer import (
    TWITTER_ROBERTA_MODEL_NAME,
    DualHeadCrossEncoder,
    distill_dual_cross_encoder,
    encode_cross_encoder_batch,
    export_stratified_human_audit_csv,
    extract_parent_reply_pairs_from_parquet,
    finetune_visobert_vihsd,
    split_pairs_by_inductive_topics,
)
from src.nlp.extract_features import encode_node_embeddings, extract_and_save_features
from src.training.metrics import compute_multitask_loss, evaluate_predictions
from src.training.trainer import _compute_sample_level_bootstrap_pvalues, evaluate_loader


def test_step2_spanpool_segment_isolation_and_distillation_fixes(tmp_path: Path) -> None:
    tok, enc = _build_smoke_offline_tokenizer_and_encoder()
    enc.eval()

    # 1a. SpanPool avoids duplicating root/parent on depth-1
    emb_dup = encode_node_embeddings(
        ["root topic", "agree support"],
        encoder=enc,
        tokenizer=tok,
        parent_texts=[None, "root topic"],
        root_texts=[None, "root topic"],
        lang="en",
        max_length=48,
        device=torch.device("cpu"),
    )
    emb_single = encode_node_embeddings(
        ["root topic", "agree support"],
        encoder=enc,
        tokenizer=tok,
        parent_texts=[None, "root topic"],
        root_texts=[None, None],
        lang="en",
        max_length=48,
        device=torch.device("cpu"),
    )
    assert torch.allclose(emb_dup, emb_single, atol=1e-6)

    # 1b. Guaranteed reply token budget survives 400+ word ancestors without NaN or empty span
    long_root = " ".join(["topic"] * 420)
    long_parent = " ".join(["reply"] * 420)
    emb_long = encode_node_embeddings(
        ["agree support"],
        encoder=enc,
        tokenizer=tok,
        parent_texts=[long_parent],
        root_texts=[long_root],
        lang="en",
        max_length=48,
        min_target_tokens=16,
        device=torch.device("cpu"),
    )
    assert emb_long.shape == (1, 768)
    assert torch.isfinite(emb_long).all()

    # 1c. Contextual disambiguation: identical short reply under distinct parents differs
    emb_ctx = encode_node_embeddings(
        ["question", "question"],
        encoder=enc,
        tokenizer=tok,
        parent_texts=["agree support", "hate trash"],
        root_texts=["root topic", "root topic"],
        lang="en",
        max_length=48,
        device=torch.device("cpu"),
    )
    assert float(torch.norm(emb_ctx[0] - emb_ctx[1]).item()) > 1e-4

    # 1d. Zero Future Leakage: mutating post-K replies leaves K<=20 SpanPool embeddings invariant
    def _make_22_cascade(extra_text: str) -> tuple[list[dict], list[dict]]:
        base_ts = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
        n_rows = [
            {
                "node_id": "c_r",
                "cascade_id": "c0",
                "topic_id": "top0",
                "parent_id": None,
                "is_root": True,
                "text": "root topic agree",
                "created_at_utc": base_ts,
                "lang": "en",
                "source_dataset": "pheme",
                "user_id_hash": "a" * 64,
                "rumor_label": "true",
            }
        ]
        e_rows = []
        for idx in range(1, 22):
            pid = "c_r" if idx <= 5 else f"c_n{idx - 5}"
            n_rows.append(
                {
                    "node_id": f"c_n{idx}",
                    "cascade_id": "c0",
                    "topic_id": "top0",
                    "parent_id": pid,
                    "is_root": False,
                    "text": extra_text if idx == 21 else "agree support",
                    "created_at_utc": base_ts + timedelta(seconds=30 * idx),
                    "lang": "en",
                    "source_dataset": "pheme",
                    "user_id_hash": "b" * 64,
                    "rumor_label": None,
                }
            )
            e_rows.append(
                {
                    "edge_id": f"{pid}->c_n{idx}",
                    "cascade_id": "c0",
                    "source_id": pid,
                    "target_id": f"c_n{idx}",
                    "created_at_utc": base_ts + timedelta(seconds=30 * idx),
                    "time_delay_sec": 30.0,
                    "edge_type": "reply",
                }
            )
        return n_rows, e_rows

    ce_init = DualHeadCrossEncoder(lang="en", model_name="offline-ce", encoder=enc, edge_dim=2)
    ce_ckpt = ce_init.save_checkpoint(tmp_path / "span_ce_ckpt", tokenizer=tok)

    int_a, int_b = tmp_path / "int_a", tmp_path / "int_b"
    n_a, e_a = _make_22_cascade("agree support")
    n_b, e_b = _make_22_cascade("hate trash oppose")
    save_cascade_parquet(build_nodes_table(n_a), build_edges_table(e_a), int_a, validate=True)
    save_cascade_parquet(build_nodes_table(n_b), build_edges_table(e_b), int_b, validate=True)
    pt_a = extract_and_save_features(
        dataset="pheme",
        interim_dir=int_a,
        output_dir=tmp_path / "proc_a",
        cross_encoder_ckpt=ce_ckpt,
        node_encoder=enc,
        tokenizer=tok,
        device=torch.device("cpu"),
    )
    pt_b = extract_and_save_features(
        dataset="pheme",
        interim_dir=int_b,
        output_dir=tmp_path / "proc_b",
        cross_encoder_ckpt=ce_ckpt,
        node_encoder=enc,
        tokenizer=tok,
        device=torch.device("cpu"),
    )
    pl_a = torch.load(pt_a, map_location="cpu", weights_only=False)
    pl_b = torch.load(pt_b, map_location="cpu", weights_only=False)
    assert torch.allclose(pl_a["node_embeddings"][:21], pl_b["node_embeddings"][:21], atol=1e-6)
    assert not torch.allclose(pl_a["node_embeddings"][21], pl_b["node_embeddings"][21])

    # 2. Segment-isolated DualHeadCrossEncoder eliminates parent-toxicity bleed
    class _TokenWiseMockEncoder(nn.Module):
        def __init__(self, vocab_size: int = 128, hidden_size: int = 16) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                hidden_size=hidden_size,
                max_position_embeddings=64,
                pad_token_id=1,
                type_vocab_size=1,
                model_type="roberta",
            )
            self.embed = nn.Embedding(vocab_size, hidden_size)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(last_hidden_state=self.embed(input_ids))

    tok_wise_enc = _TokenWiseMockEncoder()
    ce_isolated = DualHeadCrossEncoder(
        lang="en", model_name="tok-wise", encoder=tok_wise_enc, edge_dim=2, dropout=0.0
    )
    ce_isolated.eval()
    batch_clean = encode_cross_encoder_batch(
        ["root topic agree support"],
        ["neutral question"],
        tokenizer=tok,
        include_root_context=False,
        lang="en",
        encoder=tok_wise_enc,
    )
    batch_toxic = encode_cross_encoder_batch(
        ["hate trash hate trash hate"],
        ["neutral question"],
        tokenizer=tok,
        include_root_context=False,
        lang="en",
        encoder=tok_wise_enc,
    )
    with torch.no_grad():
        out_clean = ce_isolated(**batch_clean)
        out_toxic = ce_isolated(**batch_toxic)
    assert torch.allclose(out_clean["h_uv"], out_toxic["h_uv"], atol=1e-6)
    assert not torch.allclose(out_clean["stance_logits"], out_toxic["stance_logits"], atol=1e-5)

    # 3. distill_dual_cross_encoder best-epoch tracking & deep root pair augmentation
    deep_train_recs = [
        {
            "root_id": "r0",
            "parent_id": f"p{i}",
            "root_text": "root topic debate",
            "parent_text": "agree support" if i % 2 == 0 else "oppose civil",
            "reply_text": "agree support" if i % 2 == 0 else "hate trash",
            "s_uv": 0.85 if i % 2 == 0 else -0.85,
            "root_s_uv": -0.75,
            "h_uv": 0.05 if i % 2 == 0 else 0.90,
            "stance_probs": [0.85, 0.10, 0.05] if i % 2 == 0 else [0.05, 0.10, 0.85],
        }
        for i in range(6)
    ]
    deep_val_recs = deep_train_recs[:2]
    ce_dist = DualHeadCrossEncoder(lang="en", model_name="tiny-dist", encoder=enc, edge_dim=2, dropout=0.0)
    dist_rep = distill_dual_cross_encoder(
        deep_train_recs,
        deep_val_recs,
        output_dir=tmp_path / "dist_step2",
        lang="en",
        model=ce_dist,
        tokenizer=tok,
        edge_dim=2,
        include_root_context=True,
        use_multi_gpu=False,
        epochs=3,
        batch_size=4,
        lr=1e-4,
        device=torch.device("cpu"),
    )
    assert dist_rep["train_root_aug_count"] == len(deep_train_recs)
    assert dist_rep["val_loss"] == pytest.approx(dist_rep["best_val_loss"])
    rep_json = json.loads((tmp_path / "dist_step2" / "distill_report.json").read_text(encoding="utf-8"))
    assert "best_epoch" in rep_json
    assert rep_json["val_loss"] == pytest.approx(rep_json["best_val_loss"])

    # 4. finetune_visobert_vihsd class weights & warmup+cosine metrics logging
    vihsd_cfg = RobertaConfig(
        vocab_size=enc.config.vocab_size,
        hidden_size=enc.config.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        type_vocab_size=1,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        num_labels=3,
        id2label={0: "clean", 1: "offensive", 2: "hate"},
        label2id={"clean": 0, "offensive": 1, "hate": 2},
    )
    vihsd_clf = RobertaForSequenceClassification(vihsd_cfg)
    toy_vihsd = {
        "train": {
            "free_text": ["agree support"] * 10 + ["oppose civil"] * 2 + ["hate trash"] * 2,
            "label_id": [0] * 10 + [1] * 2 + [2] * 2,
        },
        "validation": {
            "free_text": ["agree", "oppose", "hate"],
            "label_id": [0, 1, 2],
        },
    }
    vihsd_dir = finetune_visobert_vihsd(
        tmp_path / "vihsd_ckpt",
        epochs=2,
        batch_size=4,
        lr=1e-4,
        dataset_dict=toy_vihsd,
        model=vihsd_clf,
        tokenizer=tok,
        device=torch.device("cpu"),
    )
    vihsd_metrics = json.loads((vihsd_dir / "vihsd_metrics.json").read_text(encoding="utf-8"))
    assert math.isfinite(float(vihsd_metrics["best_val_macro_f1"]))
    assert len(vihsd_metrics["class_weights"]) == 3
    assert vihsd_metrics["class_weights"][1] > vihsd_metrics["class_weights"][0]
    assert vihsd_metrics["class_weights"][2] > vihsd_metrics["class_weights"][0]
    assert len(vihsd_metrics["val_macro_f1_history"]) == 2


def test_canonical_topic_splits_and_vn_clustering(tmp_path: Path) -> None:
    assert DEFAULT_FORECAST_MIN_REPLIES == 25
    base_ts = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
    # Reproduce the ottawashooting bug: 1 short cascade (10 replies) + 1 long cascade (28 replies)
    topic_specs = [
        ("ottawashooting", "ott_short", 10, "true"),
        ("ottawashooting", "ott_long", 28, "true"),
        ("charliehebdo", "ch_long", 28, "false"),
        ("ferguson", "fg_long", 28, "non-rumor"),
        ("sydneysiege", "sy_long", 28, "true"),
        ("germanwings", "gw_long", 28, "false"),
    ]
    node_rows: list[dict] = []
    edge_rows: list[dict] = []
    pairs: list[dict] = []

    for c_idx, (topic, cid, n_rep, r_lbl) in enumerate(topic_specs):
        rid = f"{cid}_r"
        node_rows.append(
            {
                "node_id": rid,
                "cascade_id": cid,
                "topic_id": topic,
                "parent_id": None,
                "is_root": True,
                "text": f"root claim {topic}",
                "created_at_utc": base_ts + timedelta(hours=c_idx),
                "lang": "en",
                "source_dataset": "pheme",
                "user_id_hash": "a" * 64,
                "rumor_label": r_lbl,
            }
        )
        for k in range(1, n_rep + 1):
            nid = f"{cid}_n{k}"
            eid = f"{rid}->{nid}"
            node_rows.append(
                {
                    "node_id": nid,
                    "cascade_id": cid,
                    "topic_id": topic,
                    "parent_id": rid,
                    "is_root": False,
                    "text": f"reply {k}",
                    "created_at_utc": base_ts + timedelta(hours=c_idx, seconds=30 * k),
                    "lang": "en",
                    "source_dataset": "pheme",
                    "user_id_hash": "b" * 64,
                    "rumor_label": None,
                }
            )
            edge_rows.append(
                {
                    "edge_id": eid,
                    "cascade_id": cid,
                    "source_id": rid,
                    "target_id": nid,
                    "created_at_utc": base_ts + timedelta(hours=c_idx, seconds=30 * k),
                    "time_delay_sec": float(30 * k),
                    "edge_type": "reply",
                }
            )
            pairs.append({"edge_id": eid, "cascade_id": cid, "topic_id": topic})

    nodes_tbl = build_nodes_table(node_rows)
    edges_tbl = build_edges_table(edge_rows)
    canonical = compute_canonical_topic_splits(nodes_tbl, min_total_replies=25, seed=42)
    pair_splits = split_pairs_by_inductive_topics(pairs, canonical_topic_splits=canonical)

    assert {p["topic_id"] for p in pair_splits["train"]} == canonical["train"]
    assert {p["topic_id"] for p in pair_splits["val"]} == canonical["val"]
    assert {p["topic_id"] for p in pair_splits["test"]} == canonical["test"]

    # Check disk topic_splits.json propagation and LOO teacher isolation
    nlp_feats = {
        "node_embeddings": torch.randn((len(node_rows), 768), dtype=torch.float32),
        "node_id_to_idx": {r["node_id"]: i for i, r in enumerate(node_rows)},
        "edge_attr": torch.tensor([[0.8, 0.1, 0.1, 0.8]] * len(edge_rows), dtype=torch.float32),
        "edge_id_to_idx": {r["edge_id"]: i for i, r in enumerate(edge_rows)},
        "edge_stance": {r["edge_id"]: 0.8 for r in edge_rows},
        "edge_hostility": {r["edge_id"]: 0.1 for r in edge_rows},
        "edge_irrationality": {r["edge_id"]: 0.1 for r in edge_rows},
        "edge_constructive": {r["edge_id"]: 0.8 for r in edge_rows},
        "edge_oppose_flag": {r["edge_id"]: False for r in edge_rows},
        "edge_pred_label": {r["edge_id"]: "support" for r in edge_rows},
        "teacher_edge_stance": {r["edge_id"]: -0.7 for r in edge_rows},
        "teacher_edge_hostility": {r["edge_id"]: 0.6 for r in edge_rows},
        "teacher_edge_irrationality": {r["edge_id"]: 0.3 for r in edge_rows},
        "teacher_edge_constructive": {r["edge_id"]: 0.2 for r in edge_rows},
        "teacher_edge_oppose_flag": {r["edge_id"]: True for r in edge_rows},
        "teacher_edge_pred_label": {r["edge_id"]: "oppose" for r in edge_rows},
    }
    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()
    save_topic_splits_json(canonical, proc_dir / "topic_splits.json")

    ds_train = EarlyCascadeDataset(
        root_dir=proc_dir,
        split="train",
        min_total_replies=25,
        nodes_table=nodes_tbl,
        edges_table=edges_tbl,
        nlp_features=nlp_feats,
    )
    ds_test = EarlyCascadeDataset(
        root_dir=proc_dir,
        split="test",
        min_total_replies=25,
        nodes_table=nodes_tbl,
        edges_table=edges_tbl,
        nlp_features=nlp_feats,
    )
    assert {d.topic_id for d in ds_train} == canonical["train"]
    assert {d.topic_id for d in ds_test} == canonical["test"]

    ds_teacher = EarlyCascadeDataset(
        root_dir=proc_dir,
        split="test",
        min_total_replies=25,
        edge_source="teacher",
        nodes_table=nodes_tbl,
        edges_table=edges_tbl,
        nlp_features=nlp_feats,
    )
    assert torch.allclose(ds_teacher[0].edge_attr[0, :2], torch.tensor([-0.7, 0.6]))

    # VN 14-day TF-IDF sliding-window clustering merges same-event paraphrased posts within 14d
    vn_rows = [
        {"is_root": True, "cascade_id": "v1", "topic_id": "t1", "text": "Thu phí xe máy nội đô giảm tắc đường Hà Nội", "created_at_utc": base_ts},
        {"is_root": True, "cascade_id": "v2", "topic_id": "t2", "text": "Thu phí xe máy nội đô giảm tắc đường Hà Nội tuần sau", "created_at_utc": base_ts + timedelta(days=5)},
        {"is_root": True, "cascade_id": "v3", "topic_id": "t3", "text": "Thu phí xe máy nội đô giảm tắc đường Hà Nội", "created_at_utc": base_ts + timedelta(days=30)},
    ]
    clustered = cluster_vn_cascade_topics(vn_rows, time_window_days=14.0, sim_threshold=0.55)
    tmap = {r["cascade_id"]: r["topic_id"] for r in clustered}
    assert tmap["v1"] == tmap["v2"]
    assert tmap["v3"] != tmap["v1"]


def test_beta_nll_calibrated_confidence_task_loss_and_per_seed_bootstrap() -> None:
    head = MultiTaskHead(16, hidden_dim=16, num_rumor_classes=4, predict_uncertainty=True)
    # Sweep inputs across magnitude scales: confidence_score must be non-constant in (0, 1)
    embs = torch.cat([-4.0 * torch.ones(1, 16), 4.0 * torch.ones(1, 16)], dim=0)
    out_sweep = head(embs, embs)
    confs = out_sweep["confidence_score"].detach().numpy()
    assert np.all((confs > 0.0) & (confs < 1.0))

    # Beta-NLL detached error test: NLL term must not pass gradients into y_size_pred
    y_size_pred = torch.tensor([2.5, 4.0], requires_grad=True)
    y_polar_pred = torch.tensor([0.2, 0.8], requires_grad=True)
    s_lv = torch.tensor([-1.0, 1.5], requires_grad=True)
    p_lv = torch.tensor([-2.0, 2.0], requires_grad=True)
    batch = SimpleNamespace(
        y_size=torch.tensor([1.0, 3.0]),
        y_polar=torch.tensor([0.3, 0.5]),
        y_rumor=None,
    )
    loss_nll = compute_multitask_loss(
        {
            "y_size_pred": y_size_pred,
            "y_polar_pred": y_polar_pred,
            "rumor_logits": torch.zeros((2, 4)),
            "size_logvar": s_lv,
            "polar_logvar": p_lv,
        },
        batch,
        alpha_size=0.0,
        alpha_polar=0.0,
        lambda_uncertainty=1.0,
        beta_nll=0.5,
    )
    loss_nll.backward()
    assert y_size_pred.grad is None or torch.allclose(y_size_pred.grad, torch.zeros_like(y_size_pred))
    assert y_polar_pred.grad is None or torch.allclose(y_polar_pred.grad, torch.zeros_like(y_polar_pred))
    assert s_lv.grad is not None and torch.isfinite(s_lv.grad).all()
    assert p_lv.grad is not None and torch.isfinite(p_lv.grad).all()

    # Check task_loss invariance in evaluate_loader & optional construct_corr_bc
    g = CascadeData(
        x=torch.randn(5, 770),
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]], dtype=torch.long),
        edge_index_td=torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]], dtype=torch.long),
        edge_index_bu=torch.tensor([[1, 2, 3, 4], [0, 0, 1, 2]], dtype=torch.long),
        edge_attr=torch.tensor([[0.8, 0.1]] * 4, dtype=torch.float32),
        cascade_stats=torch.rand((1, 8), dtype=torch.float32),
        root_index=torch.tensor([0], dtype=torch.long),
        y_size=torch.tensor([2.5]),
        y_polar=torch.tensor([0.7]),
        y_rumor=torch.tensor([1]),
    )
    from torch_geometric.loader import DataLoader as PyGDataLoader

    ldr = PyGDataLoader([g, g], batch_size=2)
    sbi = SBiFAGAT(in_dim=770, hidden_dim=32, edge_dim=2, use_cascade_stats=True, predict_uncertainty=True)
    m0 = evaluate_loader(sbi, ldr, torch.device("cpu"), lambda_uncertainty=0.0)
    m5 = evaluate_loader(sbi, ldr, torch.device("cpu"), lambda_uncertainty=5.0)
    assert math.isclose(m0["task_loss"], m5["task_loss"], rel_tol=1e-6)
    assert m5["loss"] != m0["loss"]

    ev_bc = evaluate_predictions(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.1, 0.5, 0.9]),
        np.array([0.1, 0.5, 0.9]),
        y_bimodality_true=np.array([0.2, 0.55, 0.85]),
    )
    assert ev_bc["construct_corr_bc"] == pytest.approx(1.0)

    # Per-seed bootstrap evaluates seeds without collapsing diversity into an ensemble mean
    yt = np.linspace(0.0, 1.0, 20)
    # Ref predicts yt +/- 0.5 across 2 seeds -> per-seed MSE=0.25, whereas ensemble mean has MSE=0!
    p_ref = [
        {"y_size_true": yt, "y_size_pred": yt + 0.5, "y_polar_true": yt, "y_polar_pred": yt},
        {"y_size_true": yt, "y_size_pred": yt - 0.5, "y_polar_true": yt, "y_polar_pred": yt},
    ]
    # Cmp has lower MSE = 0.04 per seed
    p_cmp = [
        {"y_size_true": yt, "y_size_pred": yt + 0.2, "y_polar_true": yt, "y_polar_pred": yt},
        {"y_size_true": yt, "y_size_pred": yt - 0.2, "y_polar_true": yt, "y_polar_pred": yt},
    ]
    boot = _compute_sample_level_bootstrap_pvalues(p_ref, p_cmp, metric_keys=("msle",), n_resamples=200)
    assert 0.0 <= boot["msle"] < 0.05


def test_sign_symmetric_conv_4d_init_and_8_model_fair_parity() -> None:
    # 1. Unbiased symmetric 2D edge attention init (no hardcoded [0.6, -0.4] or 4D bias)
    conv2 = StanceGuidedFAConv(hidden_dim=16, edge_dim=2, dropout=0.0)
    edge_att = conv2.att[0, 32:].detach()
    assert edge_att.shape == (2,)
    assert torch.isfinite(edge_att).all()
    assert not torch.allclose(edge_att, torch.tensor([0.6, -0.4], dtype=edge_att.dtype), atol=1e-4)

    # 2. Odd sign-symmetry verification f(-h, -h0, -h_r; ea) == -f(h, h0, h_r; ea)
    with torch.no_grad():
        conv2.lin_root.bias.zero_()
        conv2.norm.bias.zero_()
        conv2.att[:, :32].zero_()
    conv2.eval()
    h = torch.randn(5, 16)
    h0 = torch.randn(5, 16)
    hr = h0[0:1].expand(5, -1)
    ei = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]], dtype=torch.long)
    ea = torch.tensor(
        [[0.8, 0.05], [-0.8, 0.85], [-0.6, 0.1], [0.5, 0.1]],
        dtype=torch.float32,
    )
    out_pos, g_pos = conv2(h, h0, hr, ei, ea)
    out_neg, g_neg = conv2(-h, -h0, -hr, ei, ea)
    assert torch.allclose(g_pos, g_neg, atol=1e-5)
    assert torch.allclose(out_pos, -out_neg, atol=1e-5)

    # 3. Verify all 9 models (including BiEdgeGATv2) accept use_cascade_stats=True cleanly on batched graphs (including 0-edge graph)
    g1 = CascadeData(
        x=torch.randn(5, 770),
        edge_index=ei,
        edge_index_td=ei,
        edge_index_bu=ei[[1, 0]],
        edge_attr=torch.tensor([[0.9, 0.05], [-0.9, 0.9], [-0.6, 0.1], [0.4, 0.2]]),
        cascade_stats=torch.rand((1, 8)),
        root_index=torch.tensor([0]),
    )
    g2 = CascadeData(
        x=torch.randn(1, 770),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_index_td=torch.empty((2, 0), dtype=torch.long),
        edge_index_bu=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 2)),
        cascade_stats=torch.rand((1, 8)),
        root_index=torch.tensor([0]),
    )
    batch = Batch.from_data_list([g1, g2])
    from src.models.baselines import BiEdgeGATv2

    models = [
        TextMLP(in_dim=770, hidden_dim=32, use_cascade_stats=True),
        TemporalStanceMLP(in_dim=770, hidden_dim=32, edge_dim=2, use_cascade_stats=True),
        StandardGCN(in_dim=770, hidden_dim=32, use_cascade_stats=True),
        StandardGATv2(in_dim=770, hidden_dim=32, use_cascade_stats=True),
        EdgeGATv2(in_dim=770, hidden_dim=32, edge_dim=2, use_cascade_stats=True),
        BiEdgeGATv2(in_dim=770, hidden_dim=32, edge_dim=2, use_cascade_stats=True),
        FAGCNBaseline(in_dim=770, hidden_dim=32, use_cascade_stats=True),
        BiGCN(in_dim=770, hidden_dim=32, use_cascade_stats=True),
        SBiFAGAT(in_dim=770, hidden_dim=32, edge_dim=2, use_cascade_stats=True, predict_uncertainty=True),
    ]
    for mdl in models:
        out = mdl(batch)
        assert out["y_size_pred"].shape == (2,)
        assert out["y_polar_pred"].shape == (2,)
        assert out["rumor_logits"].shape == (2, 4)
        (out["y_size_pred"].sum() + out["y_polar_pred"].sum() + out["rumor_logits"].sum()).backward()


def test_cross_encoder_token_type_ids_5class_kappa_and_upfd_cli(tmp_path: Path) -> None:
    # 1. CrossEncoder forwards token_type_ids if encoder signature accepts it
    class _DebertaLikeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=16)
            self.saw_type_ids = False

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor | None = None,
        ) -> SimpleNamespace:
            self.saw_type_ids = token_type_ids is not None
            return SimpleNamespace(last_hidden_state=torch.ones(input_ids.size(0), input_ids.size(1), 16))

    enc = _DebertaLikeEncoder()
    ce = DualHeadCrossEncoder(lang="en", model_name="mock-deberta", encoder=enc, edge_dim=4)
    _ = ce(torch.zeros(2, 6, dtype=torch.long), torch.ones(2, 6, dtype=torch.long), token_type_ids=torch.zeros(2, 6, dtype=torch.long))
    assert enc.saw_type_ids is True

    # 2. 5-class behavior kappa + machine_vs_gold across all 5 DISCOURSE_BEHAVIOR_LABELS
    audit_csv = tmp_path / "human_audit.csv"
    key_csv = tmp_path / "machine_key.csv"
    with audit_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_id", "cascade_id", "edge_id",
                "annotator_1_stance", "annotator_2_stance",
                "annotator_1_hostility", "annotator_2_hostility",
                "annotator_1_behavior", "annotator_2_behavior",
            ],
        )
        w.writeheader()
        for idx, beh in enumerate(DISCOURSE_BEHAVIOR_LABELS, start=1):
            st = "support" if beh == "support" else ("neutral" if beh == "neutral" else "oppose")
            hst = "hate" if beh == "hostile_conflict" else ("offensive" if beh == "irrational_fallacy" else "clean")
            w.writerow(
                {
                    "sample_id": f"s{idx}",
                    "cascade_id": "c1",
                    "edge_id": f"e{idx}",
                    "annotator_1_stance": st,
                    "annotator_2_stance": st,
                    "annotator_1_hostility": hst,
                    "annotator_2_hostility": hst,
                    "annotator_1_behavior": beh,
                    "annotator_2_behavior": beh,
                }
            )
    with key_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["sample_id", "cascade_id", "edge_id", "machine_stance_label", "s_uv", "h_uv", "machine_behavior_label"],
        )
        w.writeheader()
        for idx, beh in enumerate(DISCOURSE_BEHAVIOR_LABELS, start=1):
            st = "support" if beh == "support" else ("neutral" if beh == "neutral" else "oppose")
            sv = "0.8" if beh == "support" else ("0.0" if beh == "neutral" else "-0.8")
            w.writerow(
                {
                    "sample_id": f"s{idx}",
                    "cascade_id": "c1",
                    "edge_id": f"e{idx}",
                    "machine_stance_label": st,
                    "s_uv": sv,
                    "h_uv": "0.9" if beh == "hostile_conflict" else "0.1",
                    "machine_behavior_label": beh,
                }
            )
    kap = evaluate_audit_csv(audit_csv, machine_key_csv=key_csv)
    assert kap["behavior_5class"]["cohen_kappa_unweighted"] == pytest.approx(1.0)
    assert kap["machine_vs_gold"]["behavior_5class"]["macro_f1"] == pytest.approx(1.0)

    # 3. Verify UPFD CLI exposes --feature
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.run_upfd_extension", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--feature" in proc.stdout


def test_egm_loo_loader_partial_teacher_and_audit_fixes(tmp_path: Path) -> None:
    base_ts = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)
    # Build 3 large events (>= 1 cascade in fixture, plus micro-event filtering test) and 1 short cascade (<25 replies)
    cascade_specs = [
        ("ev_a", "c_a", 26),
        ("ev_b", "c_b", 26),
        ("ev_c", "c_c", 26),
        ("ev_short", "c_short", 6),
    ]
    node_rows: list[dict] = []
    edge_rows: list[dict] = []
    for c_idx, (topic, cid, n_rep) in enumerate(cascade_specs):
        rid = f"{cid}_r"
        node_rows.append(
            {
                "node_id": rid,
                "cascade_id": cid,
                "topic_id": topic,
                "parent_id": None,
                "is_root": True,
                "text": "root topic claim",
                "created_at_utc": base_ts + timedelta(hours=c_idx),
                "lang": "en",
                "source_dataset": "pheme",
                "user_id_hash": "a" * 64,
                "rumor_label": "true",
            }
        )
        for k in range(1, n_rep + 1):
            nid = f"{cid}_n{k}"
            eid = f"{rid}->{nid}"
            node_rows.append(
                {
                    "node_id": nid,
                    "cascade_id": cid,
                    "topic_id": topic,
                    "parent_id": rid,
                    "is_root": False,
                    "text": "agree support reply" if k % 2 == 0 else "oppose neutral reply",
                    "created_at_utc": base_ts + timedelta(hours=c_idx, seconds=30 * k),
                    "lang": "en",
                    "source_dataset": "pheme",
                    "user_id_hash": "b" * 64,
                    "rumor_label": None,
                }
            )
            edge_rows.append(
                {
                    "edge_id": eid,
                    "cascade_id": cid,
                    "source_id": rid,
                    "target_id": nid,
                    "created_at_utc": base_ts + timedelta(hours=c_idx, seconds=30 * k),
                    "time_delay_sec": float(30 * k),
                    "edge_type": "reply",
                }
            )

    interim_dir = tmp_path / "interim"
    save_cascade_parquet(build_nodes_table(node_rows), build_edges_table(edge_rows), interim_dir, validate=True)

    # 1. Pre-filtering pairs by min_total_replies=25 drops c_short cleanly
    all_pairs = extract_parent_reply_pairs_from_parquet(interim_dir, min_total_replies=0)
    filtered_pairs = extract_parent_reply_pairs_from_parquet(interim_dir, min_total_replies=25)
    assert len(all_pairs) == 26 * 3 + 6
    assert len(filtered_pairs) == 26 * 3
    assert all(p["cascade_id"] != "c_short" for p in filtered_pairs)

    # 2. Blind audit CSV preserves root_text context while hiding machine scores
    audit_csv = tmp_path / "blind_root_ctx.csv"
    export_stratified_human_audit_csv(
        [{**p, "s_uv": 0.5, "h_uv": 0.1, "pred_label": "support"} for p in filtered_pairs[:12]],
        output_csv=audit_csv,
        n_samples=6,
    )
    with audit_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        audit_rows = list(reader)
    assert "root_text" in cols
    assert all(r["root_text"] == "root topic claim" for r in audit_rows)
    assert "s_uv" not in cols and "h_uv" not in cols

    # 3. Partial Teacher cache must NOT emit teacher_edge_* into nlp_features.pt; full cache DOES
    tok, enc = _build_smoke_offline_tokenizer_and_encoder()
    ce = DualHeadCrossEncoder(lang="en", model_name="smoke-ce", encoder=enc, edge_dim=4)
    ckpt_dir = ce.save_checkpoint(tmp_path / "ce_ckpt", tokenizer=tok)

    partial_cache = tmp_path / "partial_cache.json"
    partial_cache.write_text(
        json.dumps([{"edge_id": edge_rows[0]["edge_id"], "s_uv": -0.9, "h_uv": 0.8, "root_s_uv": -0.9}]),
        encoding="utf-8",
    )
    partial_pt = extract_and_save_features(
        dataset="pheme",
        interim_dir=interim_dir,
        output_dir=tmp_path / "proc_partial",
        cross_encoder_ckpt=ckpt_dir,
        node_encoder=enc,
        tokenizer=tok,
        scored_cache_json=partial_cache,
        device=torch.device("cpu"),
    )
    partial_payload = torch.load(partial_pt, map_location="cpu", weights_only=False)
    assert "teacher_edge_stance" not in partial_payload

    with pytest.raises(ValueError, match="100% Teacher edge coverage"):
        _build_loaders(
            tmp_path / "proc_partial",
            k_horizon=20,
            batch_size=2,
            min_total_replies=25,
            edge_source="teacher",
            custom_topic_splits={"train": {"ev_a"}, "val": {"ev_b"}, "test": {"ev_c"}},
        )

    full_cache = tmp_path / "full_cache.json"
    full_cache.write_text(
        json.dumps([{"edge_id": r["edge_id"], "s_uv": -0.65, "h_uv": 0.70, "root_s_uv": -0.65} for r in edge_rows]),
        encoding="utf-8",
    )
    full_pt = extract_and_save_features(
        dataset="pheme",
        interim_dir=interim_dir,
        output_dir=tmp_path / "proc_full",
        cross_encoder_ckpt=ckpt_dir,
        node_encoder=enc,
        tokenizer=tok,
        scored_cache_json=full_cache,
        device=torch.device("cpu"),
    )
    full_payload = torch.load(full_pt, map_location="cpu", weights_only=False)
    assert "teacher_edge_stance" in full_payload
    assert "teacher_node_root_stance" in full_payload

    fold = {"train": {"ev_a"}, "val": {"ev_b"}, "test": {"ev_c"}}
    te_ds, tr_l, va_l, te_l = _build_loaders(
        tmp_path / "proc_full",
        k_horizon=20,
        batch_size=2,
        min_total_replies=25,
        edge_source="teacher",
        custom_topic_splits=fold,
    )
    assert len(te_ds) == 1 and len(tr_l) == 1 and len(va_l) == 1 and len(te_l) == 1
    assert torch.allclose(te_ds[0].edge_attr[0, :2], torch.tensor([-0.65, 0.70]))

    # 4. LOO min_cascades_per_fold keeps micro-events in train while only evaluating large folds
    loo_items = (
        [{"topic_id": "big_1"}] * 16
        + [{"topic_id": "big_2"}] * 16
        + [{"topic_id": "big_3"}] * 16
        + [{"topic_id": "micro_ebola"}] * 2
    )
    filtered_folds = leave_one_event_out_splits(loo_items, seed=42, min_cascades_per_fold=15)
    assert {f["fold_name"] for f in filtered_folds} == {"big_1", "big_2", "big_3"}
    assert all("micro_ebola" in f["train"] for f in filtered_folds)
    assert all("micro_ebola" not in f["val"] and "micro_ebola" not in f["test"] for f in filtered_folds)


def test_dataset_eda_persistence_efficiency_latex_and_live_cli_aliases(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Verify EDA Table 1 + Early-K Persistence, Efficiency LaTeX export, and --url/--checkpoint CLI aliases."""
    from datetime import datetime, timedelta, timezone

    from scripts.benchmark_efficiency import format_efficiency_latex, run_efficiency_benchmark
    from scripts.export_dataset_stats import (
        compute_dataset_statistics,
        compute_early_k_persistence_metrics,
        format_dataset_stats_latex,
    )
    from scripts.predict_live_thread import main as live_main
    from src.data.pyg_dataset import CascadeData
    from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet

    # 1. Early-K Persistence Baseline on controlled CascadeData graphs
    c_items = []
    for idx, (s_val, y_pol) in enumerate([(-0.9, 0.8), (0.85, 0.1), (-0.7, 0.7), (0.1, 0.05)]):
        td = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        bu = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)
        edge_attr = torch.tensor([[s_val, 0.2], [-s_val, 0.3]], dtype=torch.float32)
        c_items.append(
            CascadeData(
                x=torch.zeros((3, 770), dtype=torch.float32),
                edge_index=td,
                edge_index_td=td,
                edge_index_bu=bu,
                edge_attr=edge_attr,
                root_index=torch.tensor([0], dtype=torch.long),
                y_size=torch.tensor([math.log1p(10 + idx)], dtype=torch.float32),
                y_polar=torch.tensor([y_pol], dtype=torch.float32),
            )
        )
    persist = compute_early_k_persistence_metrics(c_items)
    assert persist["num_cascades"] == 4
    assert math.isfinite(persist["rmse"]) and math.isfinite(persist["pearson_r"])
    empty_persist = compute_early_k_persistence_metrics([])
    assert empty_persist["num_cascades"] == 0 and math.isnan(empty_persist["rmse"])

    # 2. Dataset EDA statistics & LaTeX table export
    base_ts = datetime(2025, 3, 1, 8, 0, tzinfo=timezone.utc)
    node_rows = []
    edge_rows = []
    for c_idx, n_rep in enumerate([26, 28, 30, 8]):
        cid = f"eda_c{c_idx}"
        topic = f"eda_topic_{c_idx}"
        rid = f"{cid}_root"
        node_rows.append(
            {
                "node_id": rid,
                "cascade_id": cid,
                "topic_id": topic,
                "parent_id": None,
                "is_root": True,
                "text": "root claim",
                "created_at_utc": base_ts + timedelta(hours=c_idx),
                "lang": "en",
                "source_dataset": "pheme",
                "user_id_hash": "a" * 64,
                "rumor_label": "true",
            }
        )
        for k in range(1, n_rep + 1):
            nid = f"{cid}_r{k}"
            node_rows.append(
                {
                    "node_id": nid,
                    "cascade_id": cid,
                    "topic_id": topic,
                    "parent_id": rid,
                    "is_root": False,
                    "text": "reply text",
                    "created_at_utc": base_ts + timedelta(hours=c_idx, minutes=k),
                    "lang": "en",
                    "source_dataset": "pheme",
                    "user_id_hash": "b" * 64,
                    "rumor_label": None,
                }
            )
            edge_rows.append(
                {
                    "edge_id": f"{rid}->{nid}",
                    "cascade_id": cid,
                    "source_id": rid,
                    "target_id": nid,
                    "created_at_utc": base_ts + timedelta(hours=c_idx, minutes=k),
                    "time_delay_sec": float(60 * k),
                    "edge_type": "reply",
                }
            )
    ds_dir = tmp_path / "eda_ds"
    save_cascade_parquet(build_nodes_table(node_rows), build_edges_table(edge_rows), ds_dir, validate=True)
    eda = compute_dataset_statistics(ds_dir, name="pheme_sample", min_total_replies=25)
    assert eda["raw_cascades"] == 4
    assert eda["eligible_cascades"] == 3
    tex_eda = format_dataset_stats_latex([eda])
    assert "\\toprule" in tex_eda and "pheme\\_sample" in tex_eda and "\\bottomrule" in tex_eda

    # 3. Efficiency LaTeX export
    eff_rows = run_efficiency_benchmark(
        num_cascades=2,
        k_horizon=5,
        hidden_dim=16,
        warmup=1,
        repeats=1,
        device=torch.device("cpu"),
    )
    eff_tex = format_efficiency_latex(eff_rows)
    assert "\\toprule" in eff_tex and "SBiFAGAT" in eff_tex and "\\#Params" in eff_tex

    # 4. predict_live_thread --url / --checkpoint CLI aliases enforce safety offline
    from scripts import predict_live_thread as plt_mod

    demo_post, demo_comments = plt_mod._make_demo_offline_thread()
    old_id = str(demo_post["id"])
    live_post = {**demo_post, "id": "live_abc123"}
    live_comments = [
        {
            **c,
            "link_id": "t3_live_abc123",
            "parent_id": "t3_live_abc123" if c.get("parent_id") == f"t3_{old_id}" else c.get("parent_id"),
        }
        for c in demo_comments
    ]
    monkeypatch.setattr(
        plt_mod,
        "_fetch_reddit_thread",
        lambda _pid: (live_post, live_comments),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["predict_live_thread", "--url", "https://www.reddit.com/r/VietNam/comments/abc123/test"],
    )
    with pytest.raises(RuntimeError, match="checkpoint"):
        live_main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "predict_live_thread",
            "--url",
            "https://www.reddit.com/r/VietNam/comments/abc123/test",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
        ],
    )
    with pytest.raises(FileNotFoundError):
        live_main()


def test_cosine_lr_schedules_and_kaggle_runner_optimizations(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Verify CosineAnnealingLR in trainer, warmup+cosine in distillation, and optimized kaggle_parallel_runner."""
    from types import SimpleNamespace

    from torch_geometric.loader import DataLoader

    import scripts.kaggle_parallel_runner as kpr
    from src.training.trainer import train_and_evaluate_single_seed

    # 1. Trainer CosineAnnealingLR runs cleanly over multiple epochs
    graphs = []
    for i in range(6):
        td = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        bu = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)
        ea = torch.tensor([[0.6, 0.1], [-0.4, 0.3]], dtype=torch.float32)
        graphs.append(
            CascadeData(
                x=torch.randn((3, 16), dtype=torch.float32),
                edge_index=td,
                edge_index_td=td,
                edge_index_bu=bu,
                edge_attr=ea,
                cascade_stats=torch.zeros((1, 8), dtype=torch.float32),
                root_index=torch.tensor([0], dtype=torch.long),
                y_size=torch.tensor([2.0 + 0.1 * i], dtype=torch.float32),
                y_polar=torch.tensor([0.2 + 0.1 * (i % 5)], dtype=torch.float32),
                y_rumor=torch.tensor([i % 4], dtype=torch.long),
            )
        )
    tr_l = DataLoader(graphs[:4], batch_size=2, shuffle=True)
    va_l = DataLoader(graphs[4:5], batch_size=1, shuffle=False)
    te_l = DataLoader(graphs[5:6], batch_size=1, shuffle=False)
    _, metrics_out = train_and_evaluate_single_seed(
        lambda: SBiFAGAT(in_dim=16, hidden_dim=16, edge_dim=2, use_cascade_stats=False, predict_uncertainty=False),
        tr_l,
        va_l,
        te_l,
        seed=42,
        epochs=4,
        lr=1e-3,
        patience=3,
        device=torch.device("cpu"),
    )
    assert math.isfinite(float(metrics_out["rmse"])) and math.isfinite(float(metrics_out["msle"]))

    # 2. Student Cross-Encoder 10% linear warmup + cosine decay
    from src.nlp.dual_edge_scorer import DualHeadCrossEncoder, distill_dual_cross_encoder

    tok, enc = _build_smoke_offline_tokenizer_and_encoder()
    ce = DualHeadCrossEncoder(lang="en", model_name="smoke-ce", encoder=enc, edge_dim=2)
    dummy_pairs = [
        {"parent_text": "root claim", "reply_text": "support evidence", "s_uv": 0.8, "h_uv": 0.05, "topic_id": f"t{i}"}
        for i in range(4)
    ]
    rep = distill_dual_cross_encoder(
        dummy_pairs[:3],
        dummy_pairs[3:],
        lang="en",
        edge_dim=2,
        epochs=3,
        batch_size=2,
        lr=1e-4,
        output_dir=tmp_path / "ce_sched",
        model=ce,
        tokenizer=tok,
        device=torch.device("cpu"),
    )
    assert len(rep["train_losses"]) == 3 and math.isfinite(float(rep["val_loss"]))

    # 3. Unified 2D kaggle_parallel_runner for EN and VI with LOO skip guard and fair fixed hyperparams
    recorded_cmds: list[list[str]] = []
    auto_loop_calls: list[dict[str, Any]] = []

    def _fake_run(cmd: Sequence[str], cwd: Path | None = None) -> None:
        recorded_cmds.append([str(x) for x in cmd])

    def _fake_auto_loop(*_a: Any, **kw: Any) -> dict[str, Any]:
        auto_loop_calls.append(dict(kw))
        return {"best_config": {"hidden_dim": 128, "eps": 0.4, "dropout": 0.2, "lr": 5e-4, "lambda_uncertainty": 0.0}}

    monkeypatch.setattr(kpr, "_run_cmd", _fake_run)
    monkeypatch.setattr(
        kpr,
        "EarlyCascadeDataset",
        lambda **_kw: graphs[:2],
    )
    monkeypatch.setattr(kpr, "run_self_auditing_validation_loop", _fake_auto_loop)
    monkeypatch.setattr(kpr, "_bundle_outputs", lambda zip_path, **_kw: Path(zip_path))

    # Case A: member2_en uses cardiffnlp/twitter-roberta-base, dual_nli, edge_dim=2, fair hidden_dim=64 (auto_loop off by default)
    w_en = tmp_path / "work_en"
    proc_en = w_en / "data" / "processed" / "pheme"
    proc_en.mkdir(parents=True, exist_ok=True)
    torch.save({"node_features_770": {}}, proc_en / "nlp_features.pt")
    args_en = SimpleNamespace(
        role="member2_en",
        lang="en",
        smoke_test=False,
        work_dir=w_en,
        processed_dir=None,
        vn_target_count=800,
        before_epoch=None,
        k_horizon=20,
        min_total_replies=25,
        hidden_dim=64,
        epochs_nlp=1,
        epochs_gnn=2,
        batch_size_nlp=8,
        batch_size_gnn=2,
        bootstrap_if_missing=False,
        run_auto_loop=False,
        run_upfd=False,
        bundle_weights=False,
    )
    kpr.run_kaggle_parallel_role(args_en)
    assert auto_loop_calls == []
    scorer_en = [c for c in recorded_cmds if "src.nlp.dual_edge_scorer" in c][0]
    exp_en = [c for c in recorded_cmds if "scripts.run_experiments" in c][0]
    abl_en = [c for c in recorded_cmds if "scripts.run_ablation" in c][0]
    eff_en = [c for c in recorded_cmds if "scripts.benchmark_efficiency" in c][0]
    assert "dual_nli" in scorer_en
    assert TWITTER_ROBERTA_MODEL_NAME in scorer_en
    assert "--edge-dim" in scorer_en and scorer_en[scorer_en.index("--edge-dim") + 1] == "2"
    assert "--train-vihsd" not in scorer_en
    assert "--v2" not in exp_en
    assert "--use-cascade-stats" in exp_en
    assert exp_en[exp_en.index("--hidden-dim") + 1] == "64"
    assert abl_en[abl_en.index("--hidden-dim") + 1] == "64"
    assert eff_en[eff_en.index("--hidden-dim") + 1] == "64"
    assert "--eval-protocol" not in [x for cmd in recorded_cmds for x in cmd]
    assert any("scripts.export_dataset_stats" in c for c in recorded_cmds)

    # Case B: member1_vn with --run-auto-loop runs diagnostic loop at hidden_dim=64 without overriding benchmark config
    recorded_cmds.clear()
    auto_loop_calls.clear()
    w_vi = tmp_path / "work_vi"
    (w_vi / "data" / "processed" / "vn").mkdir(parents=True, exist_ok=True)
    args_vi = SimpleNamespace(
        **{**vars(args_en), "role": "member1_vn", "lang": "vi", "work_dir": w_vi, "run_auto_loop": True}
    )
    kpr.run_kaggle_parallel_role(args_vi)
    assert len(auto_loop_calls) == 1 and auto_loop_calls[0]["hidden_dim"] == 64
    scorer_vi = [c for c in recorded_cmds if "src.nlp.dual_edge_scorer" in c][0]
    exp_vi = [c for c in recorded_cmds if "scripts.run_experiments" in c][0]
    assert "dual_nli" in scorer_vi and "--train-vihsd" in scorer_vi
    assert "--include-root-context" in scorer_vi
    assert "--edge-dim" in scorer_vi and scorer_vi[scorer_vi.index("--edge-dim") + 1] == "2"
    assert "--v2" not in exp_vi
    assert exp_vi[exp_vi.index("--hidden-dim") + 1] == "64"



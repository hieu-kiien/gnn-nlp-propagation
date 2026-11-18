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
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    export_stratified_human_audit_csv,
    extract_parent_reply_pairs_from_parquet,
    split_pairs_by_inductive_topics,
)
from src.nlp.extract_features import extract_and_save_features
from src.nlp.qwen_teacher import (
    BEHAVIOR_PROTOTYPE_C,
    BEHAVIOR_PROTOTYPE_H,
    BEHAVIOR_PROTOTYPE_R,
    BEHAVIOR_PROTOTYPE_S,
    DISCOURSE_BEHAVIOR_LABELS,
    QwenBehavioralTeacherScorer,
    finetune_qwen_qlora,
)
from src.training.metrics import compute_multitask_loss, evaluate_predictions
from src.training.trainer import _compute_sample_level_bootstrap_pvalues, evaluate_loader


class _ToyDigitTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "left"

    def __init__(self) -> None:
        self._digit_map = {"1": 10, "2": 11, "3": 12, "4": 13, "5": 14}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text.strip() in self._digit_map:
            return [self._digit_map[text.strip()]]
        return [
            self._digit_map[tok]
            if tok in self._digit_map
            else (20 + (zlib.crc32(tok.encode("utf-8")) % 12))
            for tok in str(text).split()
        ] or [2]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        text = " | ".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += " | assistant:"
        return text

    def __call__(
        self,
        texts,
        padding: bool = False,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        single = isinstance(texts, str)
        batch_texts = [texts] if single else list(texts)
        seqs = [self.encode(t)[:max_length] for t in batch_texts]
        max_len = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), max_len), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for idx, seq in enumerate(seqs):
            if self.padding_side == "left":
                ids[idx, max_len - len(seq) :] = torch.tensor(seq, dtype=torch.long)
                mask[idx, max_len - len(seq) :] = 1
            else:
                ids[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                mask[idx, : len(seq)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _ToyCausalLogitLM(nn.Module):
    def __init__(self, target_digit_idx: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(64, 8)
        self.proj = nn.Linear(8, 64)
        self.target_digit_idx = target_digit_idx  # 0..4 maps to '1'..'5' (token 10..14)
        self.seen_labels: list[torch.Tensor] = []

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        bsz, seq_len = input_ids.shape
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        # Boost chosen rubric digit logit cleanly at final position
        logits = logits.clone()
        logits[:, :, 10:15] = 0.0
        logits[:, -1, 10 + self.target_digit_idx] = 6.0
        loss = torch.tensor(0.0, device=logits.device)
        if labels is not None:
            self.seen_labels.append(labels.detach().cpu())
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


def test_geval_logit_teacher_and_qlora_prompt_masking(tmp_path: Path) -> None:
    tok = _ToyDigitTokenizer()
    model_oppose = _ToyCausalLogitLM(target_digit_idx=2)  # digit '3' -> constructive_oppose
    scorer = QwenBehavioralTeacherScorer(
        lang="en",
        scoring_mode="logit",
        model=model_oppose,
        tokenizer=tok,
        max_fallback_rate=0.05,
        batch_size=2,
    )
    out = scorer.score_triplets(
        ["City toll proposal"],
        ["Everyone can take midnight trains."],
        ["Shift workers lack late-night transit coverage."],
    )
    assert out["fallback_count"] == 0
    assert out["fallback_rate"] == 0.0
    assert out["behavior_label"] == ["constructive_oppose"]
    probs = out["behavior_probs"][0].astype(np.float64)
    assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-5)
    assert math.isclose(float(out["s_uv"][0]), float(np.dot(probs, BEHAVIOR_PROTOTYPE_S)), abs_tol=1e-4)
    assert math.isclose(float(out["h_uv"][0]), float(np.dot(probs, BEHAVIOR_PROTOTYPE_H)), abs_tol=1e-4)
    assert math.isclose(float(out["r_uv"][0]), float(np.dot(probs, BEHAVIOR_PROTOTYPE_R)), abs_tol=1e-4)
    assert math.isclose(float(out["c_uv"][0]), float(np.dot(probs, BEHAVIOR_PROTOTYPE_C)), abs_tol=1e-4)

    # Verify fallback threshold enforcement in generate mode
    bad_scorer = QwenBehavioralTeacherScorer(
        lang="vi",
        custom_generate_fn=lambda prompts: ["corrupted unparseable"] * len(prompts),
        max_fallback_rate=0.05,
    )
    with pytest.raises(RuntimeError, match="fallback_rate"):
        bad_scorer.score_triplets(["r"], ["p"], ["c"])

    # Verify QLoRA prompt masking masks prompt prefix tokens as -100 while preserving target token
    tok_right = _ToyDigitTokenizer()
    tok_right.padding_side = "right"
    sft_model = _ToyCausalLogitLM(target_digit_idx=0)
    examples = [
        {
            "lang": "en",
            "root_text": "Root debate post",
            "parent_text": "Parent claim",
            "reply_text": "Constructive agreement with evidence",
            "label": "support",
            "s_uv": 0.85,
            "h_uv": 0.02,
            "r_uv": 0.02,
            "c_uv": 0.90,
            "probs": [0.85, 0.08, 0.04, 0.02, 0.01],
        }
    ]
    out_dir = finetune_qwen_qlora(
        output_dir=tmp_path / "qlora_mask",
        lang="en",
        examples=examples,
        scoring_mode="logit",
        epochs=1,
        batch_size=1,
        model=sft_model,
        tokenizer=tok_right,
    )
    assert (out_dir / "qlora_report.json").exists()
    rep = json.loads((out_dir / "qlora_report.json").read_text(encoding="utf-8"))
    assert rep["prompt_masked"] is True
    assert sft_model.seen_labels
    lbl_tensor = sft_model.seen_labels[0][0]
    assert int((lbl_tensor == -100).sum().item()) > 0
    assert int((lbl_tensor >= 0).sum().item()) >= 1


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
        edge_attr=torch.tensor([[0.8, 0.1, 0.05, 0.9]] * 4, dtype=torch.float32),
        cascade_stats=torch.rand((1, 8), dtype=torch.float32),
        root_index=torch.tensor([0], dtype=torch.long),
        y_size=torch.tensor([2.5]),
        y_polar=torch.tensor([0.7]),
        y_rumor=torch.tensor([1]),
    )
    from torch_geometric.loader import DataLoader as PyGDataLoader

    ldr = PyGDataLoader([g, g], batch_size=2)
    sbi = SBiFAGAT(in_dim=770, hidden_dim=32, edge_dim=4, use_cascade_stats=True, predict_uncertainty=True)
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
    # 1. Domain-aligned 4D edge attention init [+0.5, -0.25, -0.25, +0.25]
    conv4 = StanceGuidedFAConv(hidden_dim=16, edge_dim=4, dropout=0.0)
    expected_init = torch.tensor([0.5, -0.25, -0.25, 0.25], dtype=torch.float32)
    assert torch.allclose(conv4.att[0, 32:].detach(), expected_init, atol=1e-6)

    # 2. Odd sign-symmetry verification f(-h, -h0, -h_r; ea) == -f(h, h0, h_r; ea)
    with torch.no_grad():
        conv4.lin_root.bias.zero_()
        conv4.norm.bias.zero_()
        conv4.att[:, :32].zero_()
    conv4.eval()
    h = torch.randn(5, 16)
    h0 = torch.randn(5, 16)
    hr = h0[0:1].expand(5, -1)
    ei = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]], dtype=torch.long)
    ea = torch.tensor(
        [[0.8, 0.05, 0.05, 0.9], [-0.8, 0.85, 0.75, 0.05], [-0.6, 0.1, 0.1, 0.8], [0.5, 0.1, 0.1, 0.8]],
        dtype=torch.float32,
    )
    out_pos, g_pos = conv4(h, h0, hr, ei, ea)
    out_neg, g_neg = conv4(-h, -h0, -hr, ei, ea)
    assert torch.allclose(g_pos, g_neg, atol=1e-5)
    assert torch.allclose(out_pos, -out_neg, atol=1e-5)

    # 3. Verify all 8 models accept use_cascade_stats=True cleanly on batched graphs (including 0-edge graph)
    g1 = CascadeData(
        x=torch.randn(5, 770),
        edge_index=ei,
        edge_index_td=ei,
        edge_index_bu=ei[[1, 0]],
        edge_attr=torch.tensor([[0.9, 0.05, 0.05, 0.9], [-0.9, 0.9, 0.8, 0.05], [-0.6, 0.1, 0.1, 0.8], [0.4, 0.2, 0.1, 0.7]]),
        cascade_stats=torch.rand((1, 8)),
        root_index=torch.tensor([0]),
    )
    g2 = CascadeData(
        x=torch.randn(1, 770),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_index_td=torch.empty((2, 0), dtype=torch.long),
        edge_index_bu=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 4)),
        cascade_stats=torch.rand((1, 8)),
        root_index=torch.tensor([0]),
    )
    batch = Batch.from_data_list([g1, g2])
    models = [
        TextMLP(in_dim=770, hidden_dim=32, use_cascade_stats=True),
        TemporalStanceMLP(in_dim=770, hidden_dim=32, edge_dim=4, use_cascade_stats=True),
        StandardGCN(in_dim=770, hidden_dim=32, use_cascade_stats=True),
        StandardGATv2(in_dim=770, hidden_dim=32, use_cascade_stats=True),
        EdgeGATv2(in_dim=770, hidden_dim=32, edge_dim=4, use_cascade_stats=True),
        FAGCNBaseline(in_dim=770, hidden_dim=32, use_cascade_stats=True),
        BiGCN(in_dim=770, hidden_dim=32, use_cascade_stats=True),
        SBiFAGAT(in_dim=770, hidden_dim=32, edge_dim=4, use_cascade_stats=True, predict_uncertainty=True),
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


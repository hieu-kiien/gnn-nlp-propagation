"""Unit & integration tests for Phase 2 dual open-source NLP pipeline & clean `.pt` export."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaModel,
)

from scripts.compute_kappa import evaluate_audit_csv, interpret_kappa_landis_koch
from src.data.pyg_dataset import EarlyCascadeDataset
from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet
from src.models.sbi_fagat import SBiFAGAT
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    HostilityTeacherScorer,
    NLIStanceTeacherScorer,
    distill_dual_cross_encoder,
    export_stratified_human_audit_csv,
)
from src.nlp.extract_features import (
    extract_and_save_features,
    validate_feature_tensors,
)


def _build_offline_roberta_tokenizer() -> PreTrainedTokenizerFast:
    vocab_tokens = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "root",
        "topic",
        "reply",
        "agree",
        "support",
        "good",
        "yes",
        "oppose",
        "hate",
        "trash",
        "wrong",
        "neutral",
        "question",
        "maybe",
        "chuan",
        "ung",
        "ho",
        "phan",
        "doi",
        "tuyet",
        "te",
    ]
    vocab = {tok: idx for idx, tok in enumerate(vocab_tokens)}
    tok_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Whitespace()
    tok_obj.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


def _tiny_roberta_config(num_labels: int = 2, id2label: dict[int, str] | None = None) -> RobertaConfig:
    cfg = RobertaConfig(
        vocab_size=256,
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=128,
        type_vocab_size=1,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        num_labels=num_labels,
    )
    if id2label is not None:
        cfg.id2label = id2label
        cfg.label2id = {v: k for k, v in id2label.items()}
    return cfg


def test_hostility_and_nli_stance_scoring_formulas() -> None:
    # Exact probability algebra checks
    probs_vi = torch.tensor([[0.70, 0.20, 0.10], [0.05, 0.45, 0.50]], dtype=torch.float32)
    h_vi = HostilityTeacherScorer.hostility_from_probs(probs_vi, lang="vi")
    assert torch.allclose(h_vi, torch.tensor([0.30, 0.95]), atol=1e-6)

    probs_en = torch.tensor([[0.85, 0.15], [0.10, 0.90]], dtype=torch.float32)
    h_en = HostilityTeacherScorer.hostility_from_probs(
        probs_en, lang="en", id2label={0: "non-offensive", 1: "offensive"}
    )
    assert torch.allclose(h_en, torch.tensor([0.15, 0.90]), atol=1e-6)

    probs_nli = torch.tensor([[0.80, 0.15, 0.05], [0.10, 0.20, 0.70]], dtype=torch.float32)
    s_uv = NLIStanceTeacherScorer.stance_from_canonical_probs(probs_nli)
    assert torch.allclose(s_uv, torch.tensor([0.75, -0.60]), atol=1e-6)

    # End-to-end scorers with offline tiny transformers models + swapped NLI id2label mapping
    tok = _build_offline_roberta_tokenizer()
    host_model = RobertaForSequenceClassification(
        _tiny_roberta_config(3, {0: "clean", 1: "offensive", 2: "hate"})
    )
    host_scorer = HostilityTeacherScorer(lang="vi", model=host_model, tokenizer=tok)
    h_out = host_scorer.score_texts(["agree good", "oppose hate trash"], batch_size=2)
    assert h_out.shape == (2,)
    assert np.all((h_out >= 0.0) & (h_out <= 1.0))

    nli_model = RobertaForSequenceClassification(
        _tiny_roberta_config(3, {0: "contradiction", 1: "neutral", 2: "entailment"})
    )
    nli_scorer = NLIStanceTeacherScorer(model=nli_model, tokenizer=tok)
    assert (nli_scorer.entail_idx, nli_scorer.neutral_idx, nli_scorer.contra_idx) == (2, 1, 0)
    nli_out = nli_scorer.score_pairs(["root topic"], ["oppose hate"], batch_size=2)
    assert nli_out["s_uv"].shape == (1,)
    assert -1.0 <= float(nli_out["s_uv"][0]) <= 1.0
    assert nli_out["pred_label"][0] in {"support", "neutral", "oppose"}


def test_dual_head_cross_encoder_forward_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    tok = _build_offline_roberta_tokenizer()
    encoder = RobertaModel(_tiny_roberta_config(), add_pooling_layer=False)
    model = DualHeadCrossEncoder(lang="vi", model_name="offline-tiny-visobert", encoder=encoder)
    model.eval()

    batch = tok(
        ["root topic", "root topic"],
        ["agree good support", "oppose hate trash"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        out_before = model(batch["input_ids"], batch["attention_mask"])

    assert out_before["e_uv"].shape == (2, 2)
    assert torch.all((out_before["s_uv"] >= -1.0) & (out_before["s_uv"] <= 1.0))
    assert torch.all((out_before["h_uv"] >= 0.0) & (out_before["h_uv"] <= 1.0))

    ckpt_dir = tmp_path / "ce_ckpt"
    model.save_checkpoint(ckpt_dir, tokenizer=tok)

    reloaded, reloaded_tok = DualHeadCrossEncoder.load_checkpoint(ckpt_dir)
    reloaded.eval()
    assert reloaded_tok is not None
    with torch.no_grad():
        out_after = reloaded(batch["input_ids"], batch["attention_mask"])

    assert torch.allclose(out_before["e_uv"], out_after["e_uv"], atol=1e-6)


def test_stratified_human_audit_csv_blind_and_kappa(tmp_path: Path) -> None:
    pairs = []
    for idx in range(120):
        pairs.append(
            {
                "cascade_id": f"vn_{idx}",
                "edge_id": f"e_sup_{idx}",
                "parent_text": f"Bài gốc {idx}",
                "reply_text": f"Đồng tình hoàn toàn {idx}",
                "pred_label": "support",
                "s_uv": 0.85,
                "h_uv": 0.05,
            }
        )
        pairs.append(
            {
                "cascade_id": f"vn_{idx}",
                "edge_id": f"e_neu_{idx}",
                "parent_text": f"Bài gốc {idx}",
                "reply_text": f"Cho mình hỏi thêm {idx}",
                "pred_label": "neutral",
                "s_uv": 0.0,
                "h_uv": 0.02,
            }
        )
        pairs.append(
            {
                "cascade_id": f"vn_{idx}",
                "edge_id": f"e_opp_{idx}",
                "parent_text": f"Bài gốc {idx}",
                "reply_text": f"Phản đối gay gắt {idx}",
                "pred_label": "oppose",
                "s_uv": -0.80,
                "h_uv": 0.90,
            }
        )

    audit_csv = tmp_path / "human_audit_300.csv"
    key_csv = tmp_path / "machine_key_300.csv"
    export_stratified_human_audit_csv(
        pairs,
        output_csv=audit_csv,
        machine_key_csv=key_csv,
        n_samples=300,
        seed=42,
    )

    with audit_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        rows = list(reader)

    assert len(rows) == 300
    # Machine scores & labels must be strictly hidden from the human audit sheet
    for forbidden in ("s_uv", "h_uv", "pred_label", "machine_stance_label"):
        assert forbidden not in cols
    assert all(r["annotator_1_stance"] == "" and r["annotator_2_stance"] == "" for r in rows)

    with key_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        key_rows = list(csv.DictReader(fh))
    strata_counts = {lbl: sum(1 for r in key_rows if r["machine_stance_label"] == lbl) for lbl in ("support", "neutral", "oppose")}
    assert strata_counts == {"support": 100, "neutral": 100, "oppose": 100}

    # Test Cohen's Kappa computation: perfect agreement == 1.0, partial disagreement in (0, 1)
    perfect_csv = tmp_path / "audit_perfect.csv"
    with perfect_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r, k in zip(rows, key_rows):
            r_copy = dict(r)
            r_copy["annotator_1_stance"] = k["machine_stance_label"]
            r_copy["annotator_2_stance"] = k["machine_stance_label"]
            r_copy["annotator_1_hostility"] = "hate" if float(k["h_uv"]) > 0.5 else "clean"
            r_copy["annotator_2_hostility"] = r_copy["annotator_1_hostility"]
            writer.writerow(r_copy)

    rep_perfect = evaluate_audit_csv(perfect_csv)
    assert rep_perfect["stance"]["cohen_kappa_unweighted"] == pytest.approx(1.0)
    assert rep_perfect["hostility"]["cohen_kappa_unweighted"] == pytest.approx(1.0)
    assert "Almost Perfect" in interpret_kappa_landis_koch(0.85)


def test_distill_dual_cross_encoder_and_sanity_check_shuffle(tmp_path: Path) -> None:
    tok = _build_offline_roberta_tokenizer()

    def _make_records(n_per_class: int) -> list[dict]:
        recs = []
        for i in range(n_per_class):
            recs.append(
                {
                    "parent_text": "root topic good",
                    "reply_text": "agree support good yes",
                    "s_uv": 0.90,
                    "h_uv": 0.05,
                    "stance_probs": [0.92, 0.06, 0.02],
                }
            )
            recs.append(
                {
                    "parent_text": "root topic good",
                    "reply_text": "oppose hate trash wrong",
                    "s_uv": -0.90,
                    "h_uv": 0.92,
                    "stance_probs": [0.02, 0.06, 0.92],
                }
            )
        return recs

    train_recs = _make_records(12)
    val_recs = _make_records(6)

    torch.manual_seed(42)
    normal_encoder = RobertaModel(_tiny_roberta_config(), add_pooling_layer=False)
    normal_student = DualHeadCrossEncoder(
        lang="vi", model_name="offline-tiny-visobert", encoder=normal_encoder, dropout=0.0
    )
    rep_normal = distill_dual_cross_encoder(
        train_recs,
        val_recs,
        output_dir=tmp_path / "distill_normal",
        model=normal_student,
        tokenizer=tok,
        epochs=8,
        batch_size=8,
        lr=3e-4,
        seed=42,
        sanity_check_shuffle=False,
    )

    torch.manual_seed(42)
    shuffled_encoder = RobertaModel(_tiny_roberta_config(), add_pooling_layer=False)
    shuffled_student = DualHeadCrossEncoder(
        lang="vi", model_name="offline-tiny-visobert", encoder=shuffled_encoder, dropout=0.0
    )
    rep_shuffle = distill_dual_cross_encoder(
        train_recs,
        val_recs,
        output_dir=tmp_path / "distill_shuffled",
        model=shuffled_student,
        tokenizer=tok,
        epochs=8,
        batch_size=8,
        lr=3e-4,
        seed=42,
        sanity_check_shuffle=True,
    )

    assert rep_normal["train_losses"][-1] < rep_normal["train_losses"][0]
    assert rep_normal["val_pearson_s"] > 0.70
    assert rep_normal["val_pearson_h"] > 0.70
    assert rep_normal["val_loss"] < rep_shuffle["val_loss"]
    assert rep_shuffle["sanity_check_shuffle"] is True


def test_extract_features_clean_pt_zero_nan_and_pyg_end_to_end(tmp_path: Path) -> None:
    # Build 6 valid multi-topic cascades under interim_dir
    base_ts = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    node_rows: list[dict] = []
    edge_rows: list[dict] = []

    for c_idx in range(6):
        cid = f"vn_casc_{c_idx:02d}"
        topic = f"vn_topic_{c_idx:02d}"
        r_id = f"{cid}_r"
        node_rows.append(
            {
                "node_id": r_id,
                "cascade_id": cid,
                "topic_id": topic,
                "parent_id": None,
                "is_root": True,
                "text": "root topic good",
                "created_at_utc": base_ts + timedelta(hours=c_idx),
                "lang": "vi",
                "source_dataset": "arctic_shift_vozforums",
                "user_id_hash": "a" * 64,
                "rumor_label": None,
            }
        )
        for j in range(1, 5):
            nid = f"{cid}_n{j}"
            parent = r_id if j <= 2 else f"{cid}_n{j - 2}"
            parent_dt = (
                base_ts + timedelta(hours=c_idx)
                if j <= 2
                else base_ts + timedelta(hours=c_idx, minutes=10 * (j - 2))
            )
            dt = base_ts + timedelta(hours=c_idx, minutes=10 * j)
            node_rows.append(
                {
                    "node_id": nid,
                    "cascade_id": cid,
                    "topic_id": topic,
                    "parent_id": parent,
                    "is_root": False,
                    "text": "agree support good" if j % 2 == 1 else "oppose hate trash",
                    "created_at_utc": dt,
                    "lang": "vi",
                    "source_dataset": "arctic_shift_vozforums",
                    "user_id_hash": "b" * 64,
                    "rumor_label": None,
                }
            )
            edge_rows.append(
                {
                    "edge_id": f"{parent}->{nid}",
                    "cascade_id": cid,
                    "source_id": parent,
                    "target_id": nid,
                    "created_at_utc": dt,
                    "time_delay_sec": float((dt - parent_dt).total_seconds()),
                    "edge_type": "reply",
                }
            )

    interim_dir = tmp_path / "interim_vn"
    save_cascade_parquet(
        build_nodes_table(node_rows),
        build_edges_table(edge_rows),
        interim_dir,
        validate=True,
    )

    tok = _build_offline_roberta_tokenizer()
    encoder = RobertaModel(_tiny_roberta_config(), add_pooling_layer=False)
    student = DualHeadCrossEncoder(lang="vi", model_name="offline-tiny-visobert", encoder=encoder)
    ckpt_dir = student.save_checkpoint(tmp_path / "ce_ckpt", tokenizer=tok)
    frozen_node_encoder = RobertaModel(_tiny_roberta_config(), add_pooling_layer=False)

    with pytest.raises(FileNotFoundError):
        HostilityTeacherScorer(lang="vi", model_name_or_path=None, device=torch.device("cpu"))
    with pytest.raises(FileNotFoundError):
        HostilityTeacherScorer(
            lang="vi",
            model_name_or_path=tmp_path / "missing_vihsd",
            device=torch.device("cpu"),
        )

    processed_dir = tmp_path / "processed_vn"
    pt_path = extract_and_save_features(
        dataset="vn",
        interim_dir=interim_dir,
        output_dir=processed_dir,
        cross_encoder_ckpt=ckpt_dir,
        node_encoder=frozen_node_encoder,
        tokenizer=tok,
        batch_size=8,
        k_horizon=20,
    )
    assert pt_path.is_file()

    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    validate_feature_tensors(payload)
    assert payload["node_embeddings"].shape == (len(node_rows), 768)
    assert "node_features_770" not in payload
    assert payload["edge_attr"].shape == (len(edge_rows), 2)

    pt_path_770 = extract_and_save_features(
        dataset="vn",
        interim_dir=interim_dir,
        output_dir=tmp_path / "processed_vn_770",
        cross_encoder_ckpt=ckpt_dir,
        node_encoder=frozen_node_encoder,
        tokenizer=tok,
        batch_size=8,
        k_horizon=20,
        include_node_features_770=True,
    )
    payload_770 = torch.load(pt_path_770, map_location="cpu", weights_only=False)
    validate_feature_tensors(payload_770)
    assert payload_770["node_features_770"].shape == (len(node_rows), 770)

    # Ensure validate_feature_tensors rejects NaN or Inf
    bad_payload = dict(payload)
    bad_emb = payload["node_embeddings"].clone()
    bad_emb[0, 0] = float("nan")
    bad_payload["node_embeddings"] = bad_emb
    with pytest.raises(ValueError, match="non-finite"):
        validate_feature_tensors(bad_payload)

    # End-to-end integration with Phase 3 EarlyCascadeDataset + SBiFAGAT
    train_ds = EarlyCascadeDataset(root_dir=processed_dir, k_horizon=20, split="train")
    val_ds = EarlyCascadeDataset(root_dir=processed_dir, k_horizon=20, split="val")
    test_ds = EarlyCascadeDataset(root_dir=processed_dir, k_horizon=20, split="test")
    assert len(train_ds) + len(val_ds) + len(test_ds) == 6
    assert train_ds.topic_ids.isdisjoint(val_ds.topic_ids)
    assert train_ds.topic_ids.isdisjoint(test_ds.topic_ids)

    sample = train_ds[0]
    sample.batch = torch.zeros(sample.num_nodes, dtype=torch.long)
    assert torch.isfinite(sample.x).all()
    assert sample.edge_attr.size(1) == 2
    gnn = SBiFAGAT(in_dim=770, hidden_dim=64, edge_dim=2)
    out = gnn(sample)
    assert torch.isfinite(out["y_size_pred"]).all()
    assert torch.isfinite(out["y_polar_pred"]).all()

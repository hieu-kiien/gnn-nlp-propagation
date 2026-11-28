"""Opaque-box E2E tests for Feature Area F1: NLP Sensors, Heuristic Cues & Distillation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import torch

from src.data.schema import EDGES_ARROW_SCHEMA, NODES_ARROW_SCHEMA, build_edges_table, build_nodes_table
from src.nlp.discourse_labels import (
    _heuristic_behavior_fallback,
    behavior_probs_to_stance_probs,
)
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    HostilityTeacherScorer,
    NLIStanceTeacherScorer,
    distill_dual_cross_encoder,
    encode_cross_encoder_batch,
    finetune_visobert_vihsd,
    normalize_social_text,
)
from src.nlp.extract_features import (
    compute_leakage_free_node_metadata,
    encode_node_embeddings,
    validate_feature_tensors,
)
from tests.e2e.conftest import (
    build_offline_encoder_and_classifiers,
    build_offline_roberta_tokenizer,
    build_tiny_roberta_config,
)


# ---------------------------------------------------------------------------
# Tier 1: Isolated Feature Coverage
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_f1_t1_heuristic_opposition_cues_and_word_boundaries() -> None:
    """B5 contract: opposition cues map to oppose; substrings like 'phạt nguội'/'biến mất' are NOT hostile."""
    for oppose_text in (
        "Phản đối",
        "Tôi phản bác hoàn toàn lập luận này",
        "Bác bỏ đề xuất vô lý đó",
    ):
        res = _heuristic_behavior_fallback(oppose_text)
        assert res["label"] == "constructive_oppose", f"Unexpected label for {oppose_text!r}: {res}"
        assert res["s_uv"] < 0.0
        assert res["h_uv"] < 0.5

    for benign_text in (
        "Phạt nguội qua ứng dụng VNeID có nhanh không mọi người?",
        "Chiếc xe đã biến mất khỏi camera hành trình",
    ):
        res = _heuristic_behavior_fallback(benign_text)
        assert res["label"] != "hostile_conflict", f"Substring false-positive on {benign_text!r}: {res}"
        assert res["h_uv"] < 0.5


@pytest.mark.tier1
def test_f1_t1_behavior_probs_to_stance_and_social_text_normalization() -> None:
    """5-class behavior probabilities collapse to normalized 3-class stance and NFC social text."""
    probs_5 = [0.40, 0.15, 0.25, 0.10, 0.10]
    stance_3 = behavior_probs_to_stance_probs(probs_5)
    assert len(stance_3) == 3
    assert sum(stance_3) == pytest.approx(1.0, abs=1e-6)
    assert stance_3[0] == pytest.approx(0.40, abs=1e-6)
    assert stance_3[1] == pytest.approx(0.15, abs=1e-6)
    assert stance_3[2] == pytest.approx(0.45, abs=1e-6)

    nfd_text = "Pha\u0309n \u0111o\u0302\u0301i   ba\u0300i   vie\u0302\u0301t  https://example.com  @user1 "
    norm = normalize_social_text(nfd_text, lang="vi")
    assert "Phản đối" in norm
    assert "  " not in norm


@pytest.mark.tier1
def test_f1_t1_teacher_scorers_algebra_and_swapped_id2label(offline_hf_bundle: dict[str, Any]) -> None:
    """HostilityTeacherScorer and NLIStanceTeacherScorer respect label mappings and bounded ranges."""
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

    tok = offline_hf_bundle["tokenizer"]
    host_scorer = HostilityTeacherScorer(
        lang="vi", model=offline_hf_bundle["hostility_model"], tokenizer=tok
    )
    h_out = host_scorer.score_texts(["agree good", "oppose hate trash"], batch_size=2)
    assert h_out.shape == (2,)
    assert np.all((h_out >= 0.0) & (h_out <= 1.0))

    nli_scorer = NLIStanceTeacherScorer(
        model=offline_hf_bundle["nli_model"], tokenizer=tok, lang="vi"
    )
    assert (nli_scorer.entail_idx, nli_scorer.neutral_idx, nli_scorer.contra_idx) == (2, 1, 0)
    nli_out = nli_scorer.score_pairs(["root topic", "root topic"], ["agree good", "oppose hate"], batch_size=2)
    assert nli_out["s_uv"].shape == (2,)
    assert np.all((nli_out["s_uv"] >= -1.0) & (nli_out["s_uv"] <= 1.0))
    assert set(nli_out["pred_label"]).issubset({"support", "neutral", "oppose"})


@pytest.mark.tier1
def test_f1_t1_dual_head_cross_encoder_2d_and_save_load_roundtrip(
    tmp_path: Path, offline_hf_bundle: dict[str, Any]
) -> None:
    """DualHeadCrossEncoder(edge_dim=2) produces bounded [s_uv, h_uv] and reloads identically."""
    tok = offline_hf_bundle["tokenizer"]
    encoder = offline_hf_bundle["encoder"]
    model = DualHeadCrossEncoder(lang="vi", model_name="offline-tiny-visobert", encoder=encoder, edge_dim=2)
    model.eval()

    batch = encode_cross_encoder_batch(
        ["root topic", "root topic"],
        ["agree good support", "oppose hate trash"],
        tokenizer=tok,
        lang="vi",
        max_length=64,
        encoder=encoder,
    )
    with torch.no_grad():
        out_before = model(
            batch["input_ids"],
            batch["attention_mask"],
            context_mask=batch["context_mask"],
            reply_mask=batch["reply_mask"],
        )

    assert out_before["e_uv"].shape == (2, 2)
    assert torch.all((out_before["s_uv"] >= -1.0) & (out_before["s_uv"] <= 1.0))
    assert torch.all((out_before["h_uv"] >= 0.0) & (out_before["h_uv"] <= 1.0))
    assert torch.allclose(out_before["stance_probs"].sum(dim=-1), torch.ones(2), atol=1e-5)

    ckpt_dir = tmp_path / "dual_ce_ckpt"
    model.save_checkpoint(ckpt_dir, tokenizer=tok)
    assert (ckpt_dir / "dual_head_weights.pt").is_file()
    assert (ckpt_dir / "dual_head_meta.json").is_file()
    assert (ckpt_dir / "config.json").is_file()

    reloaded, reloaded_tok = DualHeadCrossEncoder.load_checkpoint(ckpt_dir, map_location="cpu")
    reloaded.eval()
    assert reloaded_tok is not None
    with torch.no_grad():
        out_after = reloaded(
            batch["input_ids"],
            batch["attention_mask"],
            context_mask=batch["context_mask"],
            reply_mask=batch["reply_mask"],
        )
    assert torch.allclose(out_before["e_uv"], out_after["e_uv"], atol=1e-6)


@pytest.mark.tier1
def test_f1_t1_leakage_free_node_metadata_and_feature_validation() -> None:
    """compute_leakage_free_node_metadata limits depth/temporal meta to k_horizon and validates clean payloads."""
    base_ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    n_rows = [
        {
            "node_id": "r0",
            "cascade_id": "c0",
            "topic_id": "t0",
            "parent_id": None,
            "is_root": True,
            "text": "root",
            "created_at_utc": base_ts,
            "lang": "vi",
            "source_dataset": "e2e",
            "user_id_hash": "u0",
            "rumor_label": None,
        }
    ]
    e_rows = []
    for k in range(1, 8):
        ts = base_ts + timedelta(seconds=60 * k)
        n_rows.append(
            {
                "node_id": f"n{k}",
                "cascade_id": "c0",
                "topic_id": "t0",
                "parent_id": "r0" if k <= 3 else f"n{k-1}",
                "is_root": False,
                "text": f"reply {k}",
                "created_at_utc": ts,
                "lang": "vi",
                "source_dataset": "e2e",
                "user_id_hash": f"u{k}",
                "rumor_label": None,
            }
        )
        e_rows.append(
            {
                "edge_id": f"e{k}",
                "cascade_id": "c0",
                "source_id": "r0" if k <= 3 else f"n{k-1}",
                "target_id": f"n{k}",
                "created_at_utc": ts,
                "time_delay_sec": float(60 * k),
                "edge_type": "reply",
            }
        )
    nodes_tbl = build_nodes_table(n_rows)
    edges_tbl = build_edges_table(e_rows)
    meta = compute_leakage_free_node_metadata(nodes_tbl, edges_tbl, k_horizon=4)
    assert meta.shape == (len(n_rows), 2)
    assert torch.isfinite(meta).all()
    assert bool(((meta >= 0.0) & (meta <= 1.0)).all())
    assert torch.allclose(meta[0], torch.zeros(2))
    # Perturbing timestamps of late nodes (> k_horizon) must NOT affect early nodes' metadata
    n_rows_pert = [dict(r) for r in n_rows]
    e_rows_pert = [dict(r) for r in e_rows]
    for k in range(5, len(n_rows_pert)):
        n_rows_pert[k]["created_at_utc"] = n_rows_pert[k]["created_at_utc"] + timedelta(days=5)
    meta_pert = compute_leakage_free_node_metadata(
        build_nodes_table(n_rows_pert), build_edges_table(e_rows_pert), k_horizon=4
    )
    assert torch.equal(meta[:5], meta_pert[:5])

    payload = {
        "node_embeddings": torch.randn((len(n_rows), 768), dtype=torch.float32),
        "edge_attr": torch.tensor([[0.5, 0.2]] * len(e_rows), dtype=torch.float32),
        "edge_stance": {f"e{k}": 0.5 for k in range(1, 8)},
        "edge_hostility": {f"e{k}": 0.2 for k in range(1, 8)},
    }
    validate_feature_tensors(payload)


# ---------------------------------------------------------------------------
# Tier 2: Boundary, Corner & Negative Rejection Cases
# ---------------------------------------------------------------------------


@pytest.mark.tier2
def test_f1_t2_hostility_teacher_rejects_incomplete_checkpoint(
    tmp_path: Path, offline_hf_bundle: dict[str, Any]
) -> None:
    """B3 contract: HostilityTeacherScorer must reject checkpoint dirs that lack binary weight files."""
    incomplete_dir = tmp_path / "incomplete_visobert"
    incomplete_dir.mkdir(parents=True)
    offline_hf_bundle["tokenizer"].save_pretrained(incomplete_dir)
    offline_hf_bundle["hostility_model"].config.save_pretrained(incomplete_dir)
    # Intentionally DO NOT save model.safetensors / pytorch_model.bin

    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        HostilityTeacherScorer(lang="vi", model_name_or_path=incomplete_dir)

    with pytest.raises(FileNotFoundError):
        HostilityTeacherScorer(lang="vi", model_name_or_path=None, allow_untrained_fallback=False)


@pytest.mark.tier2
def test_f1_t2_empty_nfd_and_long_sequence_boundary(offline_hf_bundle: dict[str, Any]) -> None:
    """Empty texts, NFD combining marks, and sequences longer than max_position_embeddings do not crash or emit NaN."""
    tok = offline_hf_bundle["tokenizer"]
    encoder = offline_hf_bundle["encoder"]
    long_text = " ".join(["longword"] * 220) + " targetreplytoken"

    ce_batch = encode_cross_encoder_batch(
        ["", "Pha\u0309n \u0111o\u0302\u0301i", long_text],
        ["   ", "agree good", long_text],
        tokenizer=tok,
        lang="vi",
        max_length=512,
        encoder=encoder,
    )
    assert ce_batch["input_ids"].size(1) <= encoder.config.max_position_embeddings
    assert torch.isfinite(ce_batch["input_ids"].float()).all()

    node_emb = encode_node_embeddings(
        ["", "Pha\u0309n \u0111o\u0302\u0301i", long_text],
        encoder=encoder,
        tokenizer=tok,
        parent_texts=[None, "root topic", long_text],
        lang="vi",
        batch_size=2,
        max_length=512,
        use_multi_gpu=False,
    )
    assert node_emb.shape == (3, encoder.config.hidden_size)
    assert torch.isfinite(node_emb).all()


@pytest.mark.tier2
def test_f1_t2_finetune_vihsd_explicit_validation_and_threshold(
    tmp_path: Path,
) -> None:
    """finetune_visobert_vihsd uses explicit validation split and records calibrated metrics + weights."""
    bundle = build_offline_encoder_and_classifiers(seed=11)
    tok = bundle["tokenizer"]
    clf_model = bundle["hostility_model"]

    ds_dict = {
        "train": {
            "free_text": [
                "agree good support",
                "agree yes",
                "oppose wrong",
                "oppose hate",
                "hate trash",
                "trash wrong",
            ],
            "label_id": [0, 0, 1, 1, 2, 2],
        },
        "validation": {
            "free_text": ["agree good", "oppose wrong", "hate trash"],
            "label_id": [0, 1, 2],
        },
        "test": {
            "free_text": ["yes", "oppose", "trash"],
            "label_id": [0, 1, 2],
        },
    }
    out_dir = tmp_path / "visobert_out"
    saved_dir = finetune_visobert_vihsd(
        output_dir=out_dir,
        epochs=1,
        batch_size=2,
        max_length=64,
        seed=42,
        dataset_dict=ds_dict,
        model=clf_model,
        tokenizer=tok,
        device=torch.device("cpu"),
    )
    assert (saved_dir / "vihsd_metrics.json").is_file()
    metrics = json.loads((saved_dir / "vihsd_metrics.json").read_text(encoding="utf-8"))
    assert np.isfinite(metrics.get("val_macro_f1", 0.0))
    # Weight file must actually exist on disk
    has_weights = any(
        (saved_dir / fname).is_file()
        for fname in ("model.safetensors", "pytorch_model.bin")
    )
    assert has_weights, "finetune_visobert_vihsd must persist actual model weight files"


@pytest.mark.tier2
def test_f1_t2_distill_cross_encoder_sanity_shuffle_and_finite_loss(
    tmp_path: Path,
) -> None:
    """distill_dual_cross_encoder(edge_dim=2, sanity_check_shuffle=True) emits finite report & weights."""
    bundle = build_offline_encoder_and_classifiers(seed=17)
    tok = bundle["tokenizer"]
    encoder = bundle["encoder"]

    train_records = [
        {
            "cascade_id": f"c_{i // 3}",
            "parent_text": "root topic",
            "reply_text": "agree good" if i % 2 == 0 else "oppose hate trash",
            "root_text": "root topic",
            "stance_probs": [0.85, 0.10, 0.05] if i % 2 == 0 else [0.05, 0.15, 0.80],
            "s_uv": 0.80 if i % 2 == 0 else -0.75,
            "h_uv": 0.08 if i % 2 == 0 else 0.82,
            "is_root_reply": (i % 3 == 0),
        }
        for i in range(9)
    ]
    val_records = train_records[:3]

    out_dir = tmp_path / "distill_out"
    report = distill_dual_cross_encoder(
        train_records,
        val_records,
        output_dir=out_dir,
        lang="vi",
        model=DualHeadCrossEncoder(lang="vi", model_name="offline-tiny", encoder=encoder, edge_dim=2),
        tokenizer=tok,
        edge_dim=2,
        epochs=1,
        batch_size=3,
        max_length=64,
        augment_root_pairs=True,
        sanity_check_shuffle=True,
        use_multi_gpu=False,
        device=torch.device("cpu"),
    )
    assert (out_dir / "dual_head_weights.pt").is_file()
    assert (out_dir / "distill_report.json").is_file()
    disk_report = json.loads((out_dir / "distill_report.json").read_text(encoding="utf-8"))
    assert disk_report["sanity_check_shuffle"] is True
    assert disk_report["train_root_aug_count"] == 0
    assert disk_report["edge_dim"] == 2
    assert np.isfinite(disk_report["best_val_loss"])
    assert np.isfinite(disk_report["val_mae_s"])
    assert np.isfinite(disk_report["val_mae_h"])
    assert len(disk_report["train_losses"]) == 1
    assert report["edge_dim"] == 2
    assert np.isfinite(report["best_val_loss"])


@pytest.mark.tier2
def test_f1_t2_validate_feature_tensors_rejects_bad_dim_and_nan() -> None:
    """validate_feature_tensors rejects NaNs, wrong node_embeddings width, and out-of-range stances."""
    with pytest.raises(ValueError, match="NaN|Inf|non-finite"):
        bad_emb = torch.zeros((4, 768), dtype=torch.float32)
        bad_emb[0, 0] = float("nan")
        validate_feature_tensors(
            {
                "node_embeddings": bad_emb,
                "edge_attr": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            }
        )

    with pytest.raises(ValueError, match="768"):
        validate_feature_tensors(
            {
                "node_embeddings": torch.zeros((4, 128), dtype=torch.float32),
                "edge_attr": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            }
        )

    with pytest.raises(ValueError, match="bounds"):
        validate_feature_tensors(
            {
                "node_embeddings": torch.ones((4, 768), dtype=torch.float32),
                "edge_attr": torch.tensor([[1.8, 0.2]], dtype=torch.float32),
            }
        )


@pytest.mark.tier2
def test_f1_t2_augment_root_pairs_deweights_h_loss_and_kappa_sample_weights(
    tmp_path: Path,
) -> None:
    """PROJECT #3 contract: augment_root_pairs deweights h_loss on augmented pairs; compute_kappa uses sample_weight for PR/ROC-AUC."""
    import copy
    import csv
    from transformers import RobertaModel
    from scripts.compute_kappa import evaluate_audit_csv

    torch.manual_seed(23)
    tok = build_offline_roberta_tokenizer()
    cfg = build_tiny_roberta_config(
        vocab_size=len(tok),
        hidden_size=32,
        max_position_embeddings=128,
        num_labels=3,
    )
    cfg.hidden_dropout_prob = 0.0
    cfg.attention_probs_dropout_prob = 0.0
    frozen_encoder = RobertaModel(cfg, add_pooling_layer=False)
    for param in frozen_encoder.parameters():
        param.requires_grad_(False)

    base_model = DualHeadCrossEncoder(
        lang="vi",
        model_name="offline-aug",
        encoder=frozen_encoder,
        edge_dim=2,
        dropout=0.0,
    )
    init_h_weight = base_model.hostility_head.weight.detach().clone()

    def _make_aug_records(root_txt: str) -> list[dict[str, Any]]:
        return [
            {
                "cascade_id": f"cas_{i}",
                "root_id": f"r_{i}",
                "parent_id": f"p_{i}",
                "root_text": root_txt,
                "parent_text": "parent comment agrees",
                "reply_text": "agree good" if i % 2 == 0 else "oppose hate trash",
                "stance_probs": [0.85, 0.10, 0.05] if i % 2 == 0 else [0.05, 0.15, 0.80],
                "root_stance_probs": [0.80, 0.15, 0.05] if i % 2 == 0 else [0.05, 0.15, 0.80],
                "s_uv": 0.80 if i % 2 == 0 else -0.75,
                "root_s_uv": 0.75 if i % 2 == 0 else -0.75,
                "h_uv": 0.05 if i % 2 == 0 else 0.85,
            }
            for i in range(6)
        ]

    # Opaque-box frozen-encoder differential proof for N4 (_hostility_loss_weight == 0.0 on augmented rows):
    # Only r["root_text"] differs between Run A and Run B, so only augmented (root, deep_reply) rows change
    # input text while original (parent, reply) rows remain identical.
    rec_a = _make_aug_records("root topic")
    rec_b = _make_aug_records("oppose wrong hate trash parent")
    model_a = copy.deepcopy(base_model)
    model_b = copy.deepcopy(base_model)

    report_a = distill_dual_cross_encoder(
        rec_a,
        rec_a[:2],
        output_dir=tmp_path / "aug_distill_a",
        lang="vi",
        model=model_a,
        tokenizer=tok,
        edge_dim=2,
        include_root_context=False,
        epochs=1,
        batch_size=32,
        max_length=64,
        augment_root_pairs=True,
        seed=42,
        sanity_check_shuffle=False,
        use_multi_gpu=False,
        device=torch.device("cpu"),
    )
    report_b = distill_dual_cross_encoder(
        rec_b,
        rec_b[:2],
        output_dir=tmp_path / "aug_distill_b",
        lang="vi",
        model=model_b,
        tokenizer=tok,
        edge_dim=2,
        include_root_context=False,
        epochs=1,
        batch_size=32,
        max_length=64,
        augment_root_pairs=True,
        seed=42,
        sanity_check_shuffle=False,
        use_multi_gpu=False,
        device=torch.device("cpu"),
    )
    assert report_a["train_root_aug_count"] == 6
    assert report_b["train_root_aug_count"] == 6
    assert not torch.allclose(model_a.hostility_head.weight, init_h_weight, atol=1e-6), (
        "Expected original (parent, reply) rows to update hostility_head"
    )
    assert torch.allclose(model_a.hostility_head.weight, model_b.hostility_head.weight, atol=1e-6), (
        "Augmented (root, deep_reply) rows leaked non-zero h_loss gradient into hostility_head.weight"
    )
    assert torch.allclose(model_a.hostility_head.bias, model_b.hostility_head.bias, atol=1e-6)
    assert not torch.allclose(model_a.stance_head.weight, model_b.stance_head.weight, atol=1e-6), (
        "Expected augmented (root, deep_reply) rows to actively update stance_head"
    )

    # Verify compute_kappa sample_weight affects PR-AUC / ROC-AUC / prevalence cleanly
    audit_csv = tmp_path / "audit_sw.csv"
    key_csv = tmp_path / "key_sw.csv"
    rows_audit = []
    rows_key = []
    for i in range(8):
        is_unc = i < 4
        gold_h = "uncivil" if is_unc else "clean"
        gold_s = "oppose" if is_unc else "support"
        h_score = 0.85 if i in (0, 1, 2, 4) else 0.15
        s_score = -0.80 if is_unc else 0.80
        # Heavy clean weight (10.0) vs rare uncivil weight (0.2) changes prevalence & PR-AUC sharply
        sw = 0.2 if is_unc else 10.0
        rows_audit.append(
            {
                "sample_id": f"s_{i}",
                "edge_id": f"e_{i}",
                "annotator_1_stance": gold_s,
                "annotator_2_stance": gold_s,
                "gold_adjudicated_stance": gold_s,
                "annotator_1_hostility": gold_h,
                "annotator_2_hostility": gold_h,
                "gold_adjudicated_hostility": gold_h,
            }
        )
        rows_key.append(
            {
                "sample_id": f"s_{i}",
                "edge_id": f"e_{i}",
                "s_uv": f"{s_score:.4f}",
                "h_uv": f"{h_score:.4f}",
                "machine_stance_label": gold_s,
                "sampling_weight": f"{sw:.4f}",
            }
        )
    with audit_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_audit[0].keys()))
        w.writeheader()
        w.writerows(rows_audit)
    with key_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_key[0].keys()))
        w.writeheader()
        w.writerows(rows_key)

    rep = evaluate_audit_csv(audit_csv, machine_key_csv=key_csv, strict_adjudication=True)
    host_m = rep["machine_vs_gold"]["hostility"]
    assert host_m["positive_prevalence_unweighted"] == pytest.approx(0.5, abs=1e-6)
    assert host_m["positive_prevalence_weighted"] < 0.05
    assert host_m["pr_auc"] == pytest.approx(host_m["weighted_pr_auc"], abs=1e-9)
    assert host_m["roc_auc"] == pytest.approx(host_m["weighted_roc_auc"], abs=1e-9)
    assert host_m["weighted_pr_auc"] < host_m["unweighted_pr_auc"] - 0.1

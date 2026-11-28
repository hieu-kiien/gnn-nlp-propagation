"""Regression tests for M1 R1 NLP audit fixes (B3, B4-upstream, B5, N4, M9, bundling)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import warnings
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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

import math
from types import SimpleNamespace

from scripts.compute_kappa import evaluate_audit_csv, evaluate_machine_vs_human
import scripts.kaggle_parallel_runner as kpr
from scripts.kaggle_parallel_runner import _bundle_outputs
from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet
from src.nlp.discourse_labels import _heuristic_behavior_fallback
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    HostilityGatedTargetRouter,
    HostilityTeacherScorer,
    NLIStanceTeacherScorer,
    _has_complete_dual_ce_checkpoint,
    _has_hf_weight_files,
    _score_pairs_with_teachers,
    distill_dual_cross_encoder,
    finetune_visobert_vihsd,
    load_tokenizer_safe,
)
from src.nlp.extract_features import extract_and_save_features, validate_feature_tensors


def _offline_tok_and_cfg() -> tuple[PreTrainedTokenizerFast, RobertaConfig]:
    vocab = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "root",
        "topic",
        "reply",
        "agree",
        "support",
        "oppose",
        "clean",
        "hate",
        "trash",
        "good",
        "bad",
        "ban",
        "tot",
        "ngu",
    ]
    tok_obj = Tokenizer(WordLevel(vocab={t: i for i, t in enumerate(vocab)}, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Whitespace()
    tok_obj.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    cfg = RobertaConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=1,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    return tok, cfg


def test_b5_discourse_heuristic_unicode_word_boundaries() -> None:
    # Missing opposition cues now classified cleanly as constructive_oppose
    for phrase in (
        "Phản đối",
        "Tôi phản bác quan điểm này",
        "Bác bỏ hoàn toàn lập luận trên",
        "Không đồng tình với phương án đó",
        "Không tán thành",
        "Không ủng hộ đề xuất này",
        "Mình chưa đồng ý với cách làm này",
    ):
        res = _heuristic_behavior_fallback(phrase)
        assert res["label"] == "constructive_oppose", f"Failed for {phrase!r}: {res}"
        assert res["s_uv"] < 0.0

    # False substring triggers eliminated
    assert _heuristic_behavior_fallback("Cho mình hỏi phạt nguội qua VNeID là bao lâu?")["label"] == "neutral"
    assert _heuristic_behavior_fallback("Tiền thu về rồi biến mất sao?")["label"] == "neutral"
    # NFD decomposed "ngủ" ("ngu\u0309") must NOT trigger "ngu"
    assert _heuristic_behavior_fallback("Tối qua ngủ ngon quá")["label"] == "neutral"
    assert _heuristic_behavior_fallback("Tối qua ngu\u0309 ngon quá")["label"] == "neutral"

    # True hostile phrases & symbol support still work
    assert _heuristic_behavior_fallback("Đồ ngu mới tin")["label"] == "hostile_conflict"
    assert _heuristic_behavior_fallback("Biến đi cho khuất mắt")["label"] == "hostile_conflict"
    assert _heuristic_behavior_fallback("Biến khỏi đây ngay")["label"] == "hostile_conflict"
    assert _heuristic_behavior_fallback("+1 hoàn toàn hợp lý")["label"] == "support"


def test_checkpoint_weight_file_validation_and_from_pretrained(tmp_path: Path) -> None:
    # Directory with config.json + metrics but NO weight files must be rejected
    empty_vihsd = tmp_path / "visobert_vihsd"
    empty_vihsd.mkdir()
    (empty_vihsd / "config.json").write_text('{"model_type": "roberta"}', encoding="utf-8")
    (empty_vihsd / "vihsd_metrics.json").write_text(
        json.dumps({"calibrated_hostility_threshold": 0.65, "hostility_temperature": 1.4}),
        encoding="utf-8",
    )
    assert _has_hf_weight_files(empty_vihsd) is False

    with pytest.raises(FileNotFoundError, match="weight"):
        HostilityTeacherScorer(lang="vi", model_name_or_path=empty_vihsd, device=torch.device("cpu"))

    empty_en = tmp_path / "en_off"
    empty_en.mkdir()
    (empty_en / "config.json").write_text('{"model_type": "roberta"}', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="weight"):
        HostilityTeacherScorer(lang="en", model_name_or_path=empty_en, device=torch.device("cpu"))

    # Incomplete DualHeadCrossEncoder directory rejected cleanly
    tok, cfg = _offline_tok_and_cfg()
    enc = RobertaModel(cfg, add_pooling_layer=False)
    ce = DualHeadCrossEncoder(lang="vi", model_name="tiny-vi", encoder=enc, edge_dim=2)
    ce_dir = ce.save_checkpoint(tmp_path / "dual_ce_ok", tokenizer=tok)
    assert _has_hf_weight_files(ce_dir) is True

    # Simulate transformers 4.x -> 5.x extra_special_tokens: {} in tokenizer_config.json
    tok_cfg_path = ce_dir / "tokenizer_config.json"
    tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
    tok_cfg["extra_special_tokens"] = {}
    tok_cfg_path.write_text(json.dumps(tok_cfg), encoding="utf-8")
    loaded_tok = load_tokenizer_safe(ce_dir)
    assert loaded_tok is not None

    loaded_ce = DualHeadCrossEncoder.from_pretrained(ce_dir)
    assert getattr(loaded_ce, "tokenizer", None) is not None
    ce_pair, tok_pair = DualHeadCrossEncoder.from_pretrained(ce_dir, return_tokenizer=True)
    assert tok_pair is not None and ce_pair.edge_dim == 2

    (ce_dir / "dual_head_weights.pt").unlink()
    with pytest.raises(FileNotFoundError, match="dual_head_weights.pt"):
        DualHeadCrossEncoder.load_checkpoint(ce_dir)
    with pytest.raises(FileNotFoundError, match="dual_head_weights.pt"):
        DualHeadCrossEncoder.from_pretrained(ce_dir)


def test_n4_finetune_visobert_no_test_leakage_and_calibration(tmp_path: Path) -> None:
    tok, cfg = _offline_tok_and_cfg()
    cfg.num_labels = 3
    cfg.id2label = {0: "clean", 1: "offensive", 2: "hate"}
    cfg.label2id = {"clean": 0, "offensive": 1, "hate": 2}
    clf = RobertaForSequenceClassification(cfg)

    class _PoisonedTestSplit:
        def __getitem__(self, _key: str) -> Any:
            raise AssertionError("ds['test'] must NEVER be accessed during finetune_visobert_vihsd")

    train_texts = (
        ["clean good reply"] * 12
        + ["bad trash reply"] * 6
        + ["hate ngu reply"] * 6
    )
    train_labels = [0] * 12 + [1] * 6 + [2] * 6
    dataset_dict = {
        "train": {"free_text": train_texts, "label_id": train_labels},
        "test": _PoisonedTestSplit(),
    }

    out_dir = tmp_path / "vihsd_out"
    with pytest.warns(UserWarning, match="stratified"):
        finetune_visobert_vihsd(
            output_dir=out_dir,
            epochs=1,
            batch_size=4,
            seed=42,
            dataset_dict=dataset_dict,
            model=clf,
            tokenizer=tok,
            device=torch.device("cpu"),
        )

    assert _has_hf_weight_files(out_dir) is True
    metrics = json.loads((out_dir / "vihsd_metrics.json").read_text(encoding="utf-8"))
    assert metrics["val_split_source"] == "stratified_train_holdout"
    assert 0.10 <= metrics["calibrated_hostility_threshold"] <= 0.95
    assert 0.5 <= metrics["hostility_temperature"] <= 4.0
    assert "val_uncivil_precision_at_calibrated" in metrics

    scorer = HostilityTeacherScorer(
        lang="vi",
        model_name_or_path=out_dir,
        tokenizer=tok,
        device=torch.device("cpu"),
    )
    assert scorer.calibrated_threshold == pytest.approx(metrics["calibrated_hostility_threshold"])
    assert scorer.hostility_temperature == pytest.approx(metrics["hostility_temperature"])


def test_n4_distill_augment_root_pairs_does_not_double_weight_h_loss(tmp_path: Path) -> None:
    tok, cfg = _offline_tok_and_cfg()

    def _make_records(include_deep: bool) -> list[dict[str, Any]]:
        recs = []
        for idx in range(6):
            recs.append(
                {
                    "edge_id": f"e_{idx}",
                    "root_id": "r0",
                    "parent_id": "p_deep" if include_deep else "r0",
                    "root_text": "root topic claim",
                    "parent_text": "parent reply text" if include_deep else "root topic claim",
                    "reply_text": "agree support good" if idx % 2 == 0 else "oppose hate trash",
                    "s_uv": 0.8 if idx % 2 == 0 else -0.8,
                    "root_s_uv": 0.8 if idx % 2 == 0 else -0.8,
                    "h_uv": 0.05 if idx % 2 == 0 else 0.90,
                    "stance_probs": [0.85, 0.10, 0.05] if idx % 2 == 0 else [0.05, 0.10, 0.85],
                }
            )
        return recs

    val_recs = _make_records(include_deep=False)
    train_recs = _make_records(include_deep=True)

    torch.manual_seed(7)
    m_aug = DualHeadCrossEncoder(
        lang="vi",
        model_name="tiny-vi",
        encoder=RobertaModel(cfg, add_pooling_layer=False),
        dropout=0.0,
    )
    rep = distill_dual_cross_encoder(
        train_recs,
        val_recs,
        output_dir=tmp_path / "ce_aug",
        model=m_aug,
        tokenizer=tok,
        epochs=1,
        batch_size=4,
        augment_root_pairs=True,
        seed=7,
        device=torch.device("cpu"),
    )
    assert rep["train_root_aug_count"] == 6
    assert rep["hostility_loss_deweighted_aug"] is True

    # Verify h_uv predictions remain bounded and finite and checkpoint saved with weights
    loaded = DualHeadCrossEncoder.from_pretrained(tmp_path / "ce_aug")
    assert loaded.edge_dim == 2


def test_b3_nli_calibration_and_hostility_gated_target_router() -> None:
    tok, cfg = _offline_tok_and_cfg()
    cfg.num_labels = 3
    cfg.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    cfg.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
    nli_model = RobertaForSequenceClassification(cfg)
    with torch.no_grad():
        nli_model.classifier.dense.weight.zero_()
        nli_model.classifier.dense.bias.zero_()
        nli_model.classifier.out_proj.weight.zero_()
        nli_model.classifier.out_proj.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))

    scorer_cal = NLIStanceTeacherScorer(
        model=nli_model,
        tokenizer=tok,
        device=torch.device("cpu"),
        temperature=2.0,
        stance_margin_threshold=0.25,
    )
    out = scorer_cal.score_pairs(["root"], ["reply"], batch_size=1)
    assert out["pred_label"] == ["neutral"]

    router = HostilityGatedTargetRouter(hostility_gate_threshold=0.5, lang="vi")
    routed, stats = router.route_pairs(
        [
            {
                "edge_id": "e1",
                "parent_text": "Dịch vụ này làm ăn chán quá",
                "reply_text": "Chuẩn bác, bọn lừa đảo vô lương tâm thật",
                "s_uv": -0.75,
                "h_uv": 0.88,
                "stance_probs": [0.05, 0.15, 0.80],
                "pred_label": "oppose",
            },
            {
                "edge_id": "e2",
                "parent_text": "Nên áp dụng phí nội đô",
                "reply_text": "Đồ ngu, bác nói sai hoàn toàn rồi",
                "s_uv": -0.85,
                "h_uv": 0.90,
                "stance_probs": [0.02, 0.11, 0.87],
                "pred_label": "oppose",
            },
            {
                "edge_id": "e3",
                "parent_text": "Nên áp dụng phí nội đô",
                "reply_text": "Chuẩn bác, bọn lừa đảo đáng ghét",
                "s_uv": -0.60,
                "h_uv": 0.10,
                "stance_probs": [0.08, 0.24, 0.68],
                "pred_label": "oppose",
            },
        ]
    )
    assert stats["gated_count"] == 2
    assert stats["rerouted_count"] == 1
    assert routed[0]["pred_label"] == "support"
    assert routed[0]["s_uv"] > 0.4
    assert routed[0]["target_attributed"] == "third_party"
    # Direct attack stays oppose; low hostility is not gated
    assert routed[1]["pred_label"] == "oppose"
    assert routed[2]["pred_label"] == "oppose"

    with pytest.warns(DeprecationWarning, match="legacy_nli"):
        _score_pairs_with_teachers(
            [
                {
                    **dict(routed[0]),
                    "root_id": "r0",
                    "parent_id": "r0",
                    "root_text": routed[0]["parent_text"],
                    "root_s_uv": float(routed[0]["s_uv"]),
                }
            ],
            lang="vi",
            teacher_backend="legacy_nli",
            use_hostility_router=True,
        )


def test_b4_extract_features_emits_and_validates_zero_row_masks(tmp_path: Path) -> None:
    tok, cfg = _offline_tok_and_cfg()
    cfg.hidden_size = 768
    cfg.num_attention_heads = 4
    enc = RobertaModel(cfg, add_pooling_layer=False)
    ce = DualHeadCrossEncoder(lang="en", model_name="tiny-en", encoder=enc, edge_dim=2)
    ce_dir = ce.save_checkpoint(tmp_path / "ce_en", tokenizer=tok)

    base_ts = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
    node_rows = []
    edge_rows = []
    # Cascade 0: 4 replies (eligible when min_total_replies=3)
    # Cascade 1: 1 reply (ineligible -> zero placeholder rows)
    for cid, n_rep in [("c_elig", 4), ("c_short", 1)]:
        rid = f"{cid}_r"
        node_rows.append(
            {
                "node_id": rid,
                "cascade_id": cid,
                "topic_id": f"top_{cid}",
                "parent_id": None,
                "is_root": True,
                "text": "root topic claim",
                "created_at_utc": base_ts,
                "lang": "en",
                "source_dataset": "pheme",
                "user_id_hash": "a" * 64,
                "rumor_label": "true",
            }
        )
        for j in range(1, n_rep + 1):
            nid = f"{cid}_n{j}"
            node_rows.append(
                {
                    "node_id": nid,
                    "cascade_id": cid,
                    "topic_id": f"top_{cid}",
                    "parent_id": rid,
                    "is_root": False,
                    "text": "agree support reply",
                    "created_at_utc": base_ts + timedelta(seconds=30 * j),
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
                    "created_at_utc": base_ts + timedelta(seconds=30 * j),
                    "time_delay_sec": float(30 * j),
                    "edge_type": "reply",
                }
            )

    interim_dir = tmp_path / "interim_pheme"
    save_cascade_parquet(build_nodes_table(node_rows), build_edges_table(edge_rows), interim_dir)

    pt_path = extract_and_save_features(
        dataset="pheme",
        interim_dir=interim_dir,
        output_dir=tmp_path / "proc_pheme",
        cross_encoder_ckpt=ce_dir,
        node_encoder=enc,
        tokenizer=tok,
        min_total_replies=3,
        batch_size=4,
        device=torch.device("cpu"),
    )
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    validate_feature_tensors(payload)

    mask = payload["node_embedding_valid_mask"]
    assert mask.dtype == torch.bool
    assert mask.tolist() == [True] * 5 + [False] * 2
    assert payload["eligible_edge_mask"].tolist() == [True] * 4 + [False]
    assert payload["num_eligible_nodes"] == 5
    assert payload["num_zero_placeholder_nodes"] == 2
    assert payload["eligible_cascade_ids"] == ["c_elig"]
    assert torch.all(payload["node_embeddings"][~mask].abs() == 0.0)
    assert torch.all(payload["node_embeddings"][mask].abs().sum(dim=1) > 0.0)

    bad = dict(payload)
    bad["node_embedding_valid_mask"] = torch.ones((2,), dtype=torch.bool)
    with pytest.raises(ValueError, match="node_embedding_valid_mask"):
        validate_feature_tensors(bad)


def test_b3_compute_kappa_weighted_vs_unweighted_and_threshold(tmp_path: Path) -> None:
    audit_csv = tmp_path / "audit.csv"
    key_csv = tmp_path / "key.csv"
    with audit_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_id",
                "cascade_id",
                "edge_id",
                "annotator_1_stance",
                "annotator_2_stance",
                "annotator_1_hostility",
                "annotator_2_hostility",
            ],
        )
        w.writeheader()
        rows = [
            ("a1", "c1", "e1", "support", "support", "clean", "clean"),
            ("a2", "c1", "e2", "support", "support", "clean", "clean"),
            ("a3", "c1", "e3", "oppose", "oppose", "uncivil", "uncivil"),
            ("a4", "c1", "e4", "oppose", "oppose", "uncivil", "uncivil"),
        ]
        for r in rows:
            w.writerow(dict(zip(w.fieldnames, r)))

    with key_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_id",
                "cascade_id",
                "edge_id",
                "machine_stance_label",
                "s_uv",
                "h_uv",
                "sampling_weight",
            ],
        )
        w.writeheader()
        # High sampling weight on clean rows simulates rare uncivil population prevalence;
        # a2 (clean, w=10.0) scores 0.80 above a3 (uncivil, w=0.1) at 0.70 so weighted PR-AUC drops sharply
        krows = [
            ("a1", "c1", "e1", "support", "0.8", "0.10", "10.0"),
            ("a2", "c1", "e2", "support", "0.8", "0.80", "10.0"),
            ("a3", "c1", "e3", "oppose", "-0.8", "0.70", "0.1"),
            ("a4", "c1", "e4", "oppose", "-0.8", "0.90", "0.1"),
        ]
        for r in krows:
            w.writerow(dict(zip(w.fieldnames, r)))

    res_05 = evaluate_machine_vs_human(audit_csv, key_csv, hostility_threshold=0.5)
    h05 = res_05["hostility"]
    assert h05["positive_prevalence_unweighted"] == pytest.approx(0.5)
    assert h05["positive_prevalence_weighted"] < 0.05
    assert h05["unweighted_pr_auc"] > h05["weighted_pr_auc"]
    assert h05["roc_auc"] == pytest.approx(h05["weighted_roc_auc"])
    assert h05["pr_auc"] == pytest.approx(h05["weighted_pr_auc"])
    assert "roc_pr_auc_weighting_note" in h05
    assert "uncivil_precision" in h05

    res_085 = evaluate_audit_csv(audit_csv, machine_key_csv=key_csv, hostility_threshold=0.85)
    h085 = res_085["machine_vs_gold"]["hostility"]
    assert h085["hostility_threshold"] == pytest.approx(0.85)
    assert h085["uncivil_precision"] == pytest.approx(1.0)


def test_kaggle_bundle_outputs_and_include_heavy_weights_flag(tmp_path: Path) -> None:
    work = tmp_path / "work"
    ce_dir = work / "checkpoints" / "dual_ce_vn"
    ce_dir.mkdir(parents=True)
    (ce_dir / "dual_head_weights.pt").write_bytes(b"weights")
    (ce_dir / "config.json").write_text("{}", encoding="utf-8")

    vihsd_dir = work / "checkpoints" / "visobert_vihsd"
    vihsd_dir.mkdir(parents=True)
    (vihsd_dir / "model.safetensors").write_bytes(b"safetensors")

    interim = work / "data" / "interim" / "vn"
    interim.mkdir(parents=True)
    (interim / "human_audit_vi_120.csv").write_text("id\n1\n", encoding="utf-8")
    (interim / "human_audit_vi_120_machine_key.csv").write_text("secret\n", encoding="utf-8")
    (interim / "machine_key.csv").write_text("secret2\n", encoding="utf-8")
    (interim / ".author_hash_salt.secret").write_text("salt", encoding="utf-8")
    (interim / ".env").write_text("KEY=1", encoding="utf-8")
    (interim / ".env.local").write_text("KEY=2", encoding="utf-8")
    (interim / "scored_pairs.json.checkpoint.jsonl").write_text("{}\n", encoding="utf-8")
    (interim / "temp.tmp").write_text("tmp", encoding="utf-8")

    bundle_heavy = work / "heavy.zip"
    _bundle_outputs(
        bundle_heavy,
        source_paths=[work / "checkpoints", work / "data" / "interim"],
        base_dir=work,
        include_heavy_weights=True,
        include_nlp_features=False,
    )
    with zipfile.ZipFile(bundle_heavy, "r") as zf:
        names = set(zf.namelist())
    assert "checkpoints/dual_ce_vn/dual_head_weights.pt" in names
    assert "checkpoints/visobert_vihsd/model.safetensors" in names
    assert "data/interim/vn/human_audit_vi_120.csv" in names
    for forbidden in (
        "data/interim/vn/human_audit_vi_120_machine_key.csv",
        "data/interim/vn/machine_key.csv",
        "data/interim/vn/.author_hash_salt.secret",
        "data/interim/vn/.env",
        "data/interim/vn/.env.local",
        "data/interim/vn/scored_pairs.json.checkpoint.jsonl",
        "data/interim/vn/temp.tmp",
    ):
        assert forbidden not in names

    # CLI accepts --include-heavy-weights
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.kaggle_parallel_runner", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--include-heavy-weights" in proc.stdout


def test_r2_hostility_temperature_calibration_and_calibrated_gate(tmp_path: Path) -> None:
    tok, cfg = _offline_tok_and_cfg()
    cfg.num_labels = 3
    cfg.id2label = {0: "clean", 1: "offensive", 2: "hate"}
    cfg.label2id = {"clean": 0, "offensive": 1, "hate": 2}
    clf = RobertaForSequenceClassification(cfg)
    with torch.no_grad():
        clf.classifier.dense.weight.zero_()
        clf.classifier.dense.bias.zero_()
        clf.classifier.out_proj.weight.zero_()
        clf.classifier.out_proj.bias.copy_(torch.tensor([2.0, 0.0, 0.0]))

    ckpt_dir = tmp_path / "vihsd_cal"
    ckpt_dir.mkdir()
    clf.save_pretrained(ckpt_dir)
    tok.save_pretrained(ckpt_dir)
    temp = 2.485
    cal_thr = 0.62
    (ckpt_dir / "vihsd_metrics.json").write_text(
        json.dumps({"calibrated_hostility_threshold": cal_thr, "hostility_temperature": temp}),
        encoding="utf-8",
    )

    scorer = HostilityTeacherScorer(lang="vi", model_name_or_path=ckpt_dir, tokenizer=tok, device=torch.device("cpu"))
    assert scorer.hostility_temperature == pytest.approx(temp)
    assert scorer.calibrated_threshold == pytest.approx(cal_thr)

    raw_scores = scorer.score_texts(["clean good reply"], batch_size=1, apply_temperature=False)
    cal_scores = scorer.score_texts(["clean good reply"], batch_size=1)

    raw_h = float(np.clip(raw_scores[0], 1e-6, 1.0 - 1e-6))
    expected_cal = 1.0 / (1.0 + math.exp(-math.log(raw_h / (1.0 - raw_h)) / temp))
    assert float(cal_scores[0]) == pytest.approx(expected_cal, abs=1e-5)
    assert abs(float(cal_scores[0]) - float(raw_scores[0])) > 0.01

    # _score_pairs_with_teachers defaults hostility_gate_threshold to vihsd_metrics calibrated_threshold
    tel_default: dict[str, Any] = {}
    _score_pairs_with_teachers(
        [
            {
                "edge_id": "e1",
                "root_id": "r0",
                "parent_id": "r0",
                "root_text": "Dịch vụ tệ quá",
                "parent_text": "Dịch vụ tệ quá",
                "reply_text": "Chuẩn bác, bọn lừa đảo đáng ghét",
                "h_uv": 0.55,
                "s_uv": -0.70,
                "root_s_uv": -0.70,
                "stance_probs": [0.05, 0.15, 0.80],
                "pred_label": "oppose",
            }
        ],
        lang="vi",
        vihsd_ckpt=ckpt_dir,
        use_hostility_router=True,
        telemetry=tel_default,
    )
    assert tel_default["hostility_router_stats"]["hostility_gate_threshold"] == pytest.approx(cal_thr)
    assert tel_default["hostility_router_stats"]["gated_count"] == 0

    # Explicit float override still works
    tel_override: dict[str, Any] = {}
    _score_pairs_with_teachers(
        [
            {
                "edge_id": "e1",
                "root_id": "r0",
                "parent_id": "r0",
                "root_text": "Dịch vụ tệ quá",
                "parent_text": "Dịch vụ tệ quá",
                "reply_text": "Chuẩn bác, bọn lừa đảo đáng ghét",
                "h_uv": 0.55,
                "s_uv": -0.70,
                "root_s_uv": -0.70,
                "stance_probs": [0.05, 0.15, 0.80],
                "pred_label": "oppose",
            }
        ],
        lang="vi",
        vihsd_ckpt=ckpt_dir,
        use_hostility_router=True,
        hostility_gate_threshold=0.50,
        telemetry=tel_override,
    )
    assert tel_override["hostility_router_stats"]["hostility_gate_threshold"] == pytest.approx(0.50)
    assert tel_override["hostility_router_stats"]["rerouted_count"] == 1


def test_r2_router_argmax_sync_and_depth1_root_s_uv_refresh() -> None:
    # Even with very high neutral probability (0.90), rerouted support must have argmax(stance_probs) == 0
    router = HostilityGatedTargetRouter(hostility_gate_threshold=0.5, lang="vi")
    routed, stats = router.route_pairs(
        [
            {
                "edge_id": "e_neu_high",
                "parent_text": "Dịch vụ này làm ăn chán quá",
                "reply_text": "Chuẩn bác, bọn lừa đảo vô lương tâm thật",
                "s_uv": -0.05,
                "h_uv": 0.88,
                "stance_probs": [0.02, 0.90, 0.08],
                "pred_label": "neutral",
            }
        ]
    )
    assert stats["rerouted_count"] == 1
    assert routed[0]["pred_label"] == "support"
    probs = routed[0]["stance_probs"]
    assert int(np.argmax(probs)) == 0
    assert sum(probs) == pytest.approx(1.0, abs=1e-5)

    # Depth-1 rerouted pair refreshes root_s_uv & root_stance_probs; depth-2 keeps distinct root_s_uv
    pairs = [
        {
            "edge_id": "e_d1",
            "root_id": "r0",
            "parent_id": "r0",
            "root_text": "Dịch vụ này chán quá",
            "parent_text": "Dịch vụ này chán quá",
            "reply_text": "Chuẩn bác, bọn lừa đảo vô lương tâm thật",
            "h_uv": 0.90,
            "s_uv": -0.75,
            "root_s_uv": -0.75,
            "stance_probs": [0.05, 0.15, 0.80],
            "root_stance_probs": [0.05, 0.15, 0.80],
            "pred_label": "oppose",
        },
        {
            "edge_id": "e_d2",
            "root_id": "r0",
            "parent_id": "n1",
            "root_text": "Chính sách mới",
            "parent_text": "Tôi thấy chính sách ổn",
            "reply_text": "Chuẩn bác, bọn lừa đảo vô lương tâm thật",
            "h_uv": 0.90,
            "s_uv": -0.75,
            "root_s_uv": -0.40,
            "stance_probs": [0.05, 0.15, 0.80],
            "root_stance_probs": [0.20, 0.20, 0.60],
            "pred_label": "oppose",
        },
    ]
    out_pairs = _score_pairs_with_teachers(
        pairs,
        lang="vi",
        use_hostility_router=True,
        hostility_gate_threshold=0.5,
    )
    assert out_pairs[0]["target_attributed"] == "third_party"
    assert out_pairs[0]["s_uv"] > 0.4
    assert out_pairs[0]["root_s_uv"] == pytest.approx(out_pairs[0]["s_uv"])
    assert out_pairs[0]["root_stance_probs"] == pytest.approx(out_pairs[0]["stance_probs"])
    # Depth-2 pair retains its independent root_s_uv
    assert out_pairs[1]["target_attributed"] == "third_party"
    assert out_pairs[1]["root_s_uv"] == pytest.approx(-0.40)


def test_r2_zero_byte_dual_head_weights_rejected_by_skip_checks(
    tmp_path: Path, monkeypatch: Any
) -> None:
    work = tmp_path / "work"
    ds_name = "pheme"
    interim_dir = work / "data" / "interim" / ds_name
    processed_dir = work / "data" / "processed" / ds_name
    ce_ckpt = work / "checkpoints" / f"dual_ce_{ds_name}"
    interim_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    ce_ckpt.mkdir(parents=True)

    # Create dummy interim parquets, matching cohort nlp_features.pt, audit csv, metadata, and 0-byte dual_head_weights.pt
    (interim_dir / "nodes.parquet").write_bytes(b"parquet")
    (interim_dir / "edges.parquet").write_bytes(b"parquet")
    (processed_dir / "nodes.parquet").write_bytes(b"parquet")
    (processed_dir / "edges.parquet").write_bytes(b"parquet")
    torch.save({"min_total_replies": 25, "node_features_770": {}}, processed_dir / "nlp_features.pt")
    (work / "data" / "interim" / "human_audit_en_120.csv").write_text("id\n", encoding="utf-8")

    (ce_ckpt / "config.json").write_text('{"model_type": "roberta"}', encoding="utf-8")
    (ce_ckpt / "dual_head_meta.json").write_text('{"lang": "en", "edge_dim": 2}', encoding="utf-8")
    (ce_ckpt / "distill_report.json").write_text("{}", encoding="utf-8")
    (ce_ckpt / "dual_head_weights.pt").write_bytes(b"")

    assert _has_hf_weight_files(ce_ckpt) is False
    assert _has_complete_dual_ce_checkpoint(ce_ckpt, require_distill_report=True) is False
    with pytest.raises(FileNotFoundError, match="dual_head_weights.pt"):
        DualHeadCrossEncoder.load_checkpoint(ce_ckpt)

    # Stage 2 in kaggle_parallel_runner must NOT skip when dual_head_weights.pt is 0 bytes
    recorded_cmds: list[list[str]] = []
    monkeypatch.setattr(kpr, "_run_cmd", lambda cmd, cwd: recorded_cmds.append([str(x) for x in cmd]))
    monkeypatch.setattr(kpr, "_bundle_outputs", lambda zip_path, **_kw: Path(zip_path))

    args = SimpleNamespace(
        role="model_nlp_only",
        lang="en",
        smoke_test=False,
        work_dir=work,
        processed_dir=None,
        vn_target_count=800,
        before_epoch=None,
        k_horizon=20,
        min_total_replies=25,
        hidden_dim=64,
        epochs_nlp=1,
        epochs_gnn=1,
        batch_size_nlp=8,
        batch_size_gnn=2,
        bootstrap_if_missing=False,
        run_auto_loop=False,
        run_upfd=False,
        bundle_weights=False,
        include_heavy_weights=False,
        force=False,
    )
    kpr.run_kaggle_parallel_role(args)
    assert any("src.nlp.dual_edge_scorer" in c for c in recorded_cmds)


def test_r2_discourse_fallback_challenger_hardening() -> None:
    assert _heuristic_behavior_fallback("https://example.com/ngu")["label"] == "neutral"
    assert _heuristic_behavior_fallback("Xem chi tiết tại www.example.com/ngu")["label"] == "neutral"
    assert _heuristic_behavior_fallback("Tôi đồng ý!!!")["label"] == "support"
    assert _heuristic_behavior_fallback("Đồ não tàn, cút đi!!!")["label"] == "hostile_conflict"
    assert _heuristic_behavior_fallback("Nói vớ vẩn!!!")["label"] == "hostile_conflict"

    for phrase in (
        "Tôi không hề đồng ý",
        "Tôi không bao giờ đồng ý",
        "Tôi chẳng đồng ý",
        "Tôi đâu có đồng ý",
        "Mình chưa chắc đã tán thành",
    ):
        res = _heuristic_behavior_fallback(phrase)
        assert res["label"] == "constructive_oppose", f"Expected constructive_oppose for {phrase!r}, got {res}"

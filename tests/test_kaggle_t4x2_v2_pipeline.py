"""Unit & integration tests for unified 2D NLP + GNN pipeline, burst stats, auto-loop, live predictor & Kaggle notebooks."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from torch_geometric.loader import DataLoader
from transformers import (
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaModel,
)

from scripts.kaggle_parallel_runner import _run_smoke_pipeline
from scripts.predict_live_thread import _make_demo_offline_thread, predict_live_cascade
from src.data.pyg_dataset import (
    CASCADE_STATS_DIM,
    EarlyCascadeDataset,
    TOTAL_NODE_FEATURE_DIM,
)
from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet
from src.models.sbi_fagat import SBiFAGAT
from src.nlp.discourse_labels import (
    DISCOURSE_BEHAVIOR_LABELS,
    _heuristic_behavior_fallback,
    behavior_probs_to_stance_probs,
)
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    NLIStanceTeacherScorer,
    distill_dual_cross_encoder,
    export_stratified_human_audit_csv,
    normalize_social_text,
    split_pairs_by_inductive_topics,
)
from src.nlp.extract_features import extract_and_save_features, validate_feature_tensors
from src.training.auto_loop import run_self_auditing_validation_loop
from src.training.metrics import compute_multitask_loss


def _offline_tok_and_enc() -> tuple[PreTrainedTokenizerFast, RobertaModel]:
    vocab = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "root",
        "topic",
        "agree",
        "support",
        "oppose",
        "civil",
        "hate",
        "trash",
        "fallacy",
        "neutral",
        "question",
        "reply",
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
    enc = RobertaModel(cfg, add_pooling_layer=False)
    return tok, enc


def test_discourse_labels_and_social_text_normalization() -> None:
    fb_hostile = _heuristic_behavior_fallback("Đồ não tàn, cút đi!!!")
    assert fb_hostile["fallback_used"] is True
    assert fb_hostile["label"] == "hostile_conflict"
    assert fb_hostile["h_uv"] > 0.7
    assert math_close(sum(fb_hostile["probs"]), 1.0)
    assert len(fb_hostile["stance_probs"]) == 3

    collapsed = behavior_probs_to_stance_probs([0.02, 0.03, 0.45, 0.30, 0.20])
    assert math_close(collapsed[0], 0.02)
    assert math_close(collapsed[1], 0.03)
    assert math_close(collapsed[2], 0.95)

    assert (
        normalize_social_text("Hey @Alice_99 check https://t.co/abc now!", lang="en")
        == "Hey @user check http now!"
    )
    vi_raw = "Bạn @minh xem link   https://example.com   nhé"
    assert normalize_social_text(vi_raw, lang="vi") == "Bạn @minh xem link http nhé"


def math_close(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol


def test_nli_stance_verbalizer_simplex_and_2class_rejection() -> None:
    tok, enc = _offline_tok_and_enc()

    # 1. Reject 2-class not_entailment NLI models loudly
    cfg_2cls = RobertaConfig(
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
        num_labels=2,
        id2label={0: "entailment", 1: "not_entailment"},
        label2id={"entailment": 0, "not_entailment": 1},
    )
    model_2cls = RobertaForSequenceClassification(cfg_2cls)
    with pytest.raises(ValueError, match="not_entailment|num_labels=2|Expected 3-class"):
        NLIStanceTeacherScorer(model=model_2cls, tokenizer=tok, device=torch.device("cpu"))

    # 2. Symmetric 2-hypothesis verbalizer on 3-class NLI produces exact simplex sums & zero bias
    cfg_3cls = RobertaConfig(
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
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
    )
    model_3cls = RobertaForSequenceClassification(cfg_3cls)
    with torch.no_grad():
        model_3cls.classifier.dense.weight.zero_()
        model_3cls.classifier.dense.bias.zero_()
        model_3cls.classifier.out_proj.weight.zero_()
        model_3cls.classifier.out_proj.bias.zero_()

    scorer = NLIStanceTeacherScorer(
        model=model_3cls,
        tokenizer=tok,
        device=torch.device("cpu"),
        lang="en",
        use_stance_verbalizer=True,
    )
    out_uniform = scorer.score_pairs(["root topic"], ["agree support"], batch_size=2, lang="en")
    assert np.allclose(out_uniform["stance_probs"].sum(axis=-1), 1.0, atol=1e-5)
    assert abs(float(out_uniform["s_uv"][0])) < 1e-5

    out_vi = scorer.score_pairs(["root topic"], ["oppose civil"], batch_size=2, lang="vi")
    assert np.allclose(out_vi["stance_probs"].sum(axis=-1), 1.0, atol=1e-5)
    assert abs(float(out_vi["s_uv"][0])) < 1e-5

    # Bias entailment class on classifier head -> support pass boosts p_sup, oppose pass boosts p_opp symmetrically,
    # while distinct pass probs with neutral bias zeroed remain on exact simplex
    with torch.no_grad():
        model_3cls.classifier.out_proj.bias[0] = 2.0
    out_bias = scorer.score_pairs(["root"], ["reply"], batch_size=1)
    assert np.allclose(out_bias["stance_probs"].sum(axis=-1), 1.0, atol=1e-5)
    assert -1.0 <= float(out_bias["s_uv"][0]) <= 1.0


def test_dual_head_4d_distill_audit_and_extract(tmp_path: Path) -> None:
    tok, enc = _offline_tok_and_enc()
    base_ts = datetime(2026, 7, 1, 9, 0, 0, tzinfo=timezone.utc)

    node_rows = []
    edge_rows = []
    pairs_cache = []
    for t_idx, topic in enumerate(["top_a", "top_b", "top_c"]):
        cid = f"c{t_idx}"
        node_rows.append(
            {
                "node_id": f"{cid}_r",
                "cascade_id": cid,
                "topic_id": topic,
                "parent_id": None,
                "is_root": True,
                "text": "root topic statement",
                "created_at_utc": base_ts + timedelta(hours=t_idx),
                "lang": "vi",
                "source_dataset": "arctic_shift_vozforums",
                "user_id_hash": "a" * 64,
                "rumor_label": None,
            }
        )
        for r_idx, lbl in enumerate(DISCOURSE_BEHAVIOR_LABELS, start=1):
            nid = f"{cid}_n{r_idx}"
            pid = f"{cid}_r" if r_idx <= 3 else f"{cid}_n1"
            eid = f"{pid}->{nid}"
            node_rows.append(
                {
                    "node_id": nid,
                    "cascade_id": cid,
                    "topic_id": topic,
                    "parent_id": pid,
                    "is_root": False,
                    "text": f"reply {lbl} question",
                    "created_at_utc": base_ts + timedelta(hours=t_idx, minutes=r_idx),
                    "lang": "vi",
                    "source_dataset": "arctic_shift_vozforums",
                    "user_id_hash": hashlib.sha256(f"{cid}_{r_idx}".encode()).hexdigest(),
                    "rumor_label": None,
                }
            )
            edge_rows.append(
                {
                    "edge_id": eid,
                    "cascade_id": cid,
                    "source_id": pid,
                    "target_id": nid,
                    "created_at_utc": base_ts + timedelta(hours=t_idx, minutes=r_idx),
                    "time_delay_sec": float(r_idx * 60),
                    "edge_type": "reply",
                }
            )
            bp = [0.05, 0.05, 0.05, 0.05, 0.05]
            bp[DISCOURSE_BEHAVIOR_LABELS.index(lbl)] = 0.80
            tot = sum(bp)
            bp = [v / tot for v in bp]
            pairs_cache.append(
                {
                    "edge_id": eid,
                    "cascade_id": cid,
                    "topic_id": topic,
                    "root_id": f"{cid}_r",
                    "parent_id": pid,
                    "target_id": nid,
                    "root_text": "root topic statement",
                    "parent_text": "root topic statement",
                    "reply_text": f"reply {lbl} question",
                    "s_uv": 0.75 if lbl == "support" else (0.0 if lbl == "neutral" else -0.75),
                    "h_uv": 0.90 if lbl == "hostile_conflict" else 0.10,
                    "r_uv": 0.88 if lbl == "irrational_fallacy" else 0.08,
                    "c_uv": 0.90 if lbl == "constructive_oppose" else 0.30,
                    "behavior_label": lbl,
                    "behavior_probs": bp,
                    "stance_probs": [bp[0], bp[1], bp[2] + bp[3] + bp[4]],
                    "pred_label": "support" if lbl == "support" else ("neutral" if lbl == "neutral" else "oppose"),
                    "root_s_uv": 0.75 if lbl == "support" else (0.0 if lbl == "neutral" else -0.75),
                }
            )

    interim_dir = tmp_path / "interim"
    save_cascade_parquet(build_nodes_table(node_rows), build_edges_table(edge_rows), output_dir=interim_dir)
    cache_path = interim_dir / "scored_pairs.json"
    cache_path.write_text(json.dumps(pairs_cache), encoding="utf-8")

    # 5-class audit CSV stratification
    audit_csv = interim_dir / "audit_5cls.csv"
    key_csv = interim_dir / "audit_5cls_key.csv"
    export_stratified_human_audit_csv(pairs_cache, output_csv=audit_csv, machine_key_csv=key_csv, n_samples=15)
    with key_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        k_rows = list(csv.DictReader(fh))
    assert {r["machine_behavior_label"] for r in k_rows} == set(DISCOURSE_BEHAVIOR_LABELS)

    # 4D Student distillation + checkpoint round-trip
    splits = split_pairs_by_inductive_topics(pairs_cache, split_seed=42)
    ce_4d = DualHeadCrossEncoder(lang="vi", model_name="tiny-4d", encoder=enc, edge_dim=4)
    ckpt_dir = tmp_path / "ce_4d"
    rep = distill_dual_cross_encoder(
        splits["train"],
        splits["val"],
        output_dir=ckpt_dir,
        lang="vi",
        model=ce_4d,
        tokenizer=tok,
        edge_dim=4,
        include_root_context=True,
        use_multi_gpu=False,
        epochs=1,
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert rep["edge_dim"] == 4
    reloaded, _ = DualHeadCrossEncoder.load_checkpoint(ckpt_dir)
    assert reloaded.edge_dim == 4

    # Feature extraction using checkpoint + partial scored_cache_json (simulating --max-teacher-pairs budget):
    # Must NOT fail on unscored test edges and must still compute all 4D edge_attr via Student CE
    partial_cache_path = interim_dir / "partial_scored_pairs.json"
    partial_cache_path.write_text(json.dumps(pairs_cache[:6]), encoding="utf-8")

    # Cache-only WITHOUT Cross-Encoder still enforces strict completeness check
    with pytest.raises(ValueError, match="missing"):
        extract_and_save_features(
            dataset="vn",
            interim_dir=interim_dir,
            output_dir=tmp_path / "should_fail",
            cross_encoder_ckpt=None,
            scored_cache_json=partial_cache_path,
            node_encoder=enc,
            tokenizer=tok,
            batch_size=8,
            device=torch.device("cpu"),
        )

    proc_dir = tmp_path / "processed"
    pt_path = extract_and_save_features(
        dataset="vn",
        interim_dir=interim_dir,
        output_dir=proc_dir,
        cross_encoder_ckpt=ckpt_dir,
        scored_cache_json=partial_cache_path,
        node_encoder=enc,
        tokenizer=tok,
        edge_dim=4,
        include_root_context=True,
        batch_size=8,
        device=torch.device("cpu"),
    )
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    validate_feature_tensors(payload)
    assert payload["edge_attr"].shape == (len(edge_rows), 4)
    assert "edge_irrationality" in payload
    assert "edge_constructive" in payload


def test_cascade_stats_zero_leakage_and_sbi_fagat_v2(tmp_path: Path) -> None:
    def _build_copy(root_dir: Path, future_delay_mult: float) -> EarlyCascadeDataset:
        base_ts = datetime(2026, 7, 2, 10, 0, 0, tzinfo=timezone.utc)
        node_rows, edge_rows = [], []
        n_emb = {}
        e_s, e_h, e_r, e_c = {}, {}, {}, {}

        for t_i, top in enumerate(["t0", "t1", "t2"]):
            cid = f"c{t_i}"
            rid = f"{cid}_r"
            node_rows.append(
                {
                    "node_id": rid,
                    "cascade_id": cid,
                    "topic_id": top,
                    "parent_id": None,
                    "is_root": True,
                    "text": "root post text for debate",
                    "created_at_utc": base_ts + timedelta(hours=t_i),
                    "lang": "vi",
                    "source_dataset": "arctic_shift_vozforums",
                    "user_id_hash": "r" * 64,
                    "rumor_label": None,
                }
            )
            n_emb[rid] = torch.full((768,), 0.1 * (t_i + 1))
            for k in range(1, 26):
                nid = f"{cid}_n{k}"
                pid = rid if k <= 10 else f"{cid}_n{k - 5}"
                # Mutate ONLY replies > 20 in second copy to verify Zero Data Leakage
                dt = float(k * 20) if k <= 20 else float(k * 20 * future_delay_mult)
                txt = (
                    "chửi thề công kích quá đáng thật sự chẳng hiểu nổi"
                    if t_i == 0
                    else f"bình luận thứ {k} trong thảo luận mạng xã hội đầy đủ"
                )
                if k > 20 and future_delay_mult > 1.0:
                    txt = "CHANGED FUTURE TEXT BEYOND K " + txt
                node_rows.append(
                    {
                        "node_id": nid,
                        "cascade_id": cid,
                        "topic_id": top,
                        "parent_id": pid,
                        "is_root": False,
                        "text": txt,
                        "created_at_utc": base_ts + timedelta(hours=t_i, seconds=dt),
                        "lang": "vi",
                        "source_dataset": "arctic_shift_vozforums",
                        "user_id_hash": hashlib.sha256(f"{cid}_{k}".encode()).hexdigest(),
                        "rumor_label": None,
                    }
                )
                n_emb[nid] = torch.full((768,), -0.05 * k if k <= 20 else -0.5 * future_delay_mult)
                eid = f"{pid}->{nid}"
                edge_rows.append(
                    {
                        "edge_id": eid,
                        "cascade_id": cid,
                        "source_id": pid,
                        "target_id": nid,
                        "created_at_utc": base_ts + timedelta(hours=t_i, seconds=dt),
                        "time_delay_sec": dt,
                        "edge_type": "reply",
                    }
                )
                # c0 has one-sided hostile replies
                e_s[eid] = -0.85 if t_i == 0 else (0.7 if k % 2 == 0 else -0.7)
                e_h[eid] = 0.85 if t_i == 0 else 0.10
                e_r[eid] = 0.70 if t_i == 0 else 0.10
                e_c[eid] = 0.05 if t_i == 0 else 0.85

        n_tbl = build_nodes_table(node_rows)
        e_tbl = build_edges_table(edge_rows)
        save_cascade_parquet(n_tbl, e_tbl, output_dir=root_dir)
        feats = {
            "edge_dim": 4,
            "node_embeddings": n_emb,
            "edge_stance": e_s,
            "edge_hostility": e_h,
            "edge_irrationality": e_r,
            "edge_constructive": e_c,
        }
        return EarlyCascadeDataset(
            root_dir=root_dir,
            k_horizon=20,
            split="all",
            nodes_table=n_tbl,
            edges_table=e_tbl,
            nlp_features=feats,
            min_total_replies=21,
            polar_target_mode="future_only",
        )

    ds_a = _build_copy(tmp_path / "copy_a", future_delay_mult=1.0)
    ds_b = _build_copy(tmp_path / "copy_b", future_delay_mult=50.0)
    assert len(ds_a) == len(ds_b) == 3

    for da, db in zip(ds_a, ds_b):
        assert torch.allclose(da.x, db.x, atol=1e-7)
        assert torch.allclose(da.edge_attr, db.edge_attr, atol=1e-7)
        assert da.edge_attr.shape[1] == 4
        assert da.cascade_stats.shape == (1, CASCADE_STATS_DIM)
        assert torch.allclose(da.cascade_stats, db.cascade_stats, atol=1e-7)

    # First cascade c0 had unanimous negative stance + high hostility -> elevated one_sided_hostility_index
    assert float(ds_a[0].cascade_stats[0, 7].item()) > 0.5

    # Test SBiFAGAT v2 forward + uncertainty NLL + ablations + diagnostics
    loader = DataLoader(ds_a, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    assert batch.cascade_stats.shape == (3, CASCADE_STATS_DIM)

    model_v2 = SBiFAGAT(
        in_dim=TOTAL_NODE_FEATURE_DIM,
        hidden_dim=256,
        edge_dim=4,
        use_cascade_stats=True,
        predict_uncertainty=True,
    )
    out = model_v2(batch)
    assert out["y_size_pred"].shape == (3,)
    assert out["confidence_score"].shape == (3,)
    assert torch.all((out["confidence_score"] >= 0.0) & (out["confidence_score"] <= 1.0))

    loss = compute_multitask_loss(out, batch, alpha_polar=4.0, lambda_uncertainty=0.05)
    assert torch.isfinite(loss)
    loss.backward()

    # Verify ablate_irrationality & ablate_constructive zero exact edge_attr columns
    model_abl = SBiFAGAT(
        in_dim=TOTAL_NODE_FEATURE_DIM,
        hidden_dim=64,
        edge_dim=4,
        ablate_irrationality=True,
        ablate_constructive=True,
    )
    ea_abl = model_abl._prepare_edge_attr(batch.x, batch.edge_index_td, batch)
    assert torch.allclose(ea_abl[:, 2:], torch.zeros_like(ea_abl[:, 2:]))

    diag = model_v2.predict_with_diagnostics(ds_a[0], top_k_edges=3)
    assert len(diag["top_trigger_edges"]) == 3
    assert "predicted_total_replies" in diag


def test_auto_loop_live_predictor_smoke_runner_and_notebooks(tmp_path: Path) -> None:
    # 1. Offline smoke runner builds data, 4D CE, nlp_features.pt, auto_loop, benchmark & zip
    smoke_res = _run_smoke_pipeline("member1_vn", "vi", tmp_path / "smoke_work")
    assert Path(smoke_res["bundle_zip"]).exists()
    assert Path(smoke_res["auto_loop"]["diagnostics_json"]).exists()
    assert Path(smoke_res["auto_loop"]["worst_cases_json"]).exists()
    assert Path(smoke_res["results_main_json"]).exists()

    diag_data = json.loads(Path(smoke_res["auto_loop"]["diagnostics_json"]).read_text(encoding="utf-8"))
    assert "regime_diagnostics" in diag_data
    assert "one_sided_echo_chambers" in diag_data["regime_diagnostics"]

    # 2. Live thread predictor demo offline
    post, comments = _make_demo_offline_thread()
    live_res = predict_live_cascade(post, comments, k_horizon=20)
    assert live_res["k_observed"] == 20
    assert live_res["actual_post_hoc_comparison"]["actual_total_replies"] == 24
    assert live_res["behavior_counts"]["irrational_fallacy"] >= 1
    assert live_res["behavior_counts"]["hostile_conflict"] >= 1

    # 3. Auto-loop direct API test with temporary loaders (unified 2D)
    proc_dir = tmp_path / "smoke_work" / "data" / "processed" / "vn"
    tr_ds = EarlyCascadeDataset(root_dir=proc_dir, k_horizon=20, split="train", min_total_replies=21)
    va_ds = EarlyCascadeDataset(root_dir=proc_dir, k_horizon=20, split="val", min_total_replies=21)
    loop_out = run_self_auditing_validation_loop(
        DataLoader(tr_ds, batch_size=2, shuffle=True),
        DataLoader(va_ds, batch_size=2, shuffle=False),
        edge_dim=2,
        candidates=[{"hidden_dim": 32, "lr": 1e-3, "eps": 0.5, "dropout": 0.1, "lambda_uncertainty": 0.0}],
        epochs=1,
        patience=1,
        output_dir=tmp_path / "loop_direct",
        device=torch.device("cpu"),
    )
    assert Path(loop_out["best_checkpoint_path"]).exists()

    # 4. Validate both Kaggle notebooks are valid nbformat 4 JSON synced to unified 2D pipeline
    repo_root = Path(__file__).resolve().parents[1]
    for nb_name in ("kaggle_member1_nlp_and_vn_t4x2.ipynb", "kaggle_member2_gnn_and_en_t4x2.ipynb"):
        nb_path = repo_root / "notebooks" / nb_name
        assert nb_path.exists()
        nb_json = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb_json["nbformat"] == 4
        assert len(nb_json["cells"]) >= 5
        joined_src = "\n".join("".join(c.get("source", [])) for c in nb_json["cells"])
        assert "requirements.txt" in joined_src
        assert "--hidden-dim 64" in joined_src
        assert "scripts.kaggle_parallel_runner" in joined_src
        assert "Qwen3.8-27B" not in joined_src
        assert "requirements-kaggle-t4x2.txt" not in joined_src

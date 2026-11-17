"""Unit & integration tests for SBi-FAGAT v2: 5-class discourse, Qwen3.8-27B QLoRA, 8D burst stats, auto-loop, live predictor & Kaggle T4x2 notebooks."""

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
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

from scripts.kaggle_parallel_runner import _run_smoke_pipeline
from scripts.predict_live_thread import _make_demo_offline_thread, predict_live_cascade
from src.data.pyg_dataset import (
    CASCADE_STATS_DIM,
    EarlyCascadeDataset,
    TOTAL_NODE_FEATURE_DIM,
)
from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet
from src.models.sbi_fagat import SBiFAGAT
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    distill_dual_cross_encoder,
    export_stratified_human_audit_csv,
    split_pairs_by_inductive_topics,
)
from src.nlp.extract_features import extract_and_save_features, validate_feature_tensors
from src.nlp.qwen_teacher import (
    DISCOURSE_BEHAVIOR_LABELS,
    QWEN_KAGGLE_4BIT_MODEL_ID,
    QWEN_TEACHER_MODEL_ID,
    QwenBehavioralTeacherScorer,
    _freeze_vision_modules_for_text_qlora,
    _resolve_text_lora_target_modules,
    build_qlora_gold_examples,
    build_t4x2_loading_config,
    parse_behavioral_teacher_json,
)
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


def test_parse_behavioral_teacher_json_and_qlora_configs(tmp_path: Path) -> None:
    raw = (
        "<think>User used sarcasm and illogical whataboutism</think>\n"
        "```json\n"
        '{"label":"irrational_fallacy","s_uv":-1.4,"h_uv":0.42,"r_uv":0.91,"c_uv":0.08,'
        '"probs":[0.02,0.03,0.10,0.75,0.10]}\n'
        "```"
    )
    parsed = parse_behavioral_teacher_json(raw)
    assert parsed["fallback_used"] is False
    assert parsed["label"] == "irrational_fallacy"
    assert parsed["s_uv"] == -1.0  # Clamped to [-1, 1]
    assert math_close(sum(parsed["probs"]), 1.0)
    assert len(parsed["stance_probs"]) == 3

    bad_raw = "corrupted non-json text"
    fb_hostile = parse_behavioral_teacher_json(bad_raw, reply_text="Đồ não tàn, cút đi!!!")
    assert fb_hostile["fallback_used"] is True
    assert fb_hostile["label"] == "hostile_conflict"
    assert fb_hostile["h_uv"] > 0.7

    t4_cfg = build_t4x2_loading_config(num_gpus=2, max_per_gpu_gib=14)
    assert t4_cfg["torch_dtype"] == torch.float16
    assert t4_cfg["device_map"] == "auto"
    assert t4_cfg["max_memory"][0] == "14GiB"
    assert t4_cfg["max_memory"][1] == "14GiB"

    pilot_csv = tmp_path / "pilot.csv"
    with pilot_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["parent_text", "reply_text", "gold_adjudicated_stance", "gold_adjudicated_hostility"],
        )
        w.writeheader()
        w.writerow(
            {
                "parent_text": "Cần kiểm tra khí thải xe máy cũ.",
                "reply_text": "Nói ngu như bò, cút hết đi!",
                "gold_adjudicated_stance": "hostile_conflict",
                "gold_adjudicated_hostility": "hate",
            }
        )
    examples = build_qlora_gold_examples(lang="vi", pilot_gold_csv=pilot_csv, max_samples=50)
    labels_seen = {ex["label"] for ex in examples}
    assert set(DISCOURSE_BEHAVIOR_LABELS).issubset(labels_seen)


def math_close(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol


def test_qwen_scorer_mock_and_vision_tower_freeze() -> None:
    class _ToyVLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = torch.nn.Linear(4, 4)
            self.q_proj = torch.nn.Linear(4, 4)
            self.in_proj_qkv = torch.nn.Linear(4, 12)
            self.out_proj = torch.nn.Linear(4, 4)

    vlm = _ToyVLM()
    frozen = _freeze_vision_modules_for_text_qlora(vlm)
    assert any("visual" in n for n in frozen)
    assert all(not p.requires_grad for p in vlm.visual.parameters())
    assert all(p.requires_grad for p in vlm.q_proj.parameters())
    targets = _resolve_text_lora_target_modules(vlm)
    assert "q_proj" in targets and "in_proj_qkv" in targets and "out_proj" in targets

    def _mock_gen(prompts):
        return [
            json.dumps(
                {
                    "label": "constructive_oppose" if "data" in p.lower() else "support",
                    "s_uv": -0.75 if "data" in p.lower() else 0.80,
                    "h_uv": 0.08,
                    "r_uv": 0.05,
                    "c_uv": 0.90,
                    "probs": [0.02, 0.05, 0.88, 0.03, 0.02]
                    if "data" in p.lower()
                    else [0.85, 0.08, 0.04, 0.02, 0.01],
                }
            )
            for p in prompts
        ]

    scorer = QwenBehavioralTeacherScorer(lang="vi", custom_generate_fn=_mock_gen)
    out = scorer.score_triplets(
        ["root post", "root post"],
        ["parent a", "parent b"],
        ["oppose based on data evidence", "agree strongly"],
        batch_size=2,
    )
    assert out["behavior_label"] == ["constructive_oppose", "support"]
    assert out["s_uv"].shape == (2,)
    assert out["r_uv"].shape == (2,)
    assert out["c_uv"].shape == (2,)
    assert out["behavior_probs"].shape == (2, 5)
    assert out["stance_probs"].shape == (2, 3)


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

    # 3. Auto-loop direct API test with temporary loaders
    proc_dir = tmp_path / "smoke_work" / "data" / "processed" / "vn"
    tr_ds = EarlyCascadeDataset(root_dir=proc_dir, k_horizon=20, split="train", min_total_replies=21)
    va_ds = EarlyCascadeDataset(root_dir=proc_dir, k_horizon=20, split="val", min_total_replies=21)
    loop_out = run_self_auditing_validation_loop(
        DataLoader(tr_ds, batch_size=2, shuffle=True),
        DataLoader(va_ds, batch_size=2, shuffle=False),
        edge_dim=4,
        candidates=[{"hidden_dim": 32, "lr": 1e-3, "eps": 0.5, "dropout": 0.1, "lambda_uncertainty": 0.05}],
        epochs=1,
        patience=1,
        output_dir=tmp_path / "loop_direct",
        device=torch.device("cpu"),
    )
    assert Path(loop_out["best_checkpoint_path"]).exists()

    # 4. Validate both Kaggle T4 x 2 notebooks are valid nbformat 4 JSON referencing Qwen3.8-27B
    repo_root = Path(__file__).resolve().parents[1]
    for nb_name in ("kaggle_member1_nlp_and_vn_t4x2.ipynb", "kaggle_member2_gnn_and_en_t4x2.ipynb"):
        nb_path = repo_root / "notebooks" / nb_name
        assert nb_path.exists()
        nb_json = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb_json["nbformat"] == 4
        assert len(nb_json["cells"]) >= 5
        joined_src = "\n".join("".join(c.get("source", [])) for c in nb_json["cells"])
        assert QWEN_TEACHER_MODEL_ID in joined_src
        assert QWEN_KAGGLE_4BIT_MODEL_ID in joined_src
        assert "scripts.kaggle_parallel_runner" in joined_src

"""Step 3 gate tests: multi-task KL-BCE optimization, EMA/best-val seed selection, and 4 audit fixes."""

from __future__ import annotations

import csv
import inspect
import json
import math
import warnings
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import torch
from torch_geometric.loader import DataLoader
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaModel

from scripts.benchmark_efficiency import _resolve_efficiency_table_label
from scripts.compute_kappa import evaluate_machine_vs_human
from scripts.export_dataset_stats import _resolve_eda_table_label, compute_early_k_persistence_metrics
from scripts.kaggle_parallel_runner import _bundle_outputs, _bundle_sealed_machine_keys, _run_smoke_pipeline
from scripts.run_ablation import _resolve_ablation_table_label
from scripts.run_experiments import _resolve_main_table_label, attach_early_k_persistence_to_result
from src.data.pyg_dataset import (
    CascadeData,
    EarlyCascadeDataset,
    save_topic_splits_json,
)
from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet
from src.models.baselines import MultiTaskHead
from src.models.sbi_fagat import SBiFAGAT, _propagate_root_aligned_edge_signal
from src.nlp import extract_features as ef_mod
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    NLIStanceTeacherScorer,
    normalize_social_text,
)
from src.training.auto_loop import DEFAULT_V2_CANDIDATES, run_self_auditing_validation_loop
from src.training.metrics import compute_multitask_loss
from src.training.trainer import (
    EarlyStopping,
    run_multi_seed_benchmark,
    select_best_comparator,
    train_and_evaluate_single_seed,
)
from src.utils.seed import seed_everything


def _build_offline_tok_and_cls_model(
    max_pos: int = 512,
    *,
    hidden_size: int = 16,
    num_heads: int = 4,
) -> tuple[Any, Any, Any]:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "Parent": 4,
        "comment:": 5,
        "Reply:": 6,
        "longword": 7,
        "targetreplytoken": 8,
        "The": 9,
        "reply": 10,
        "agrees": 11,
        "with": 12,
        "or": 13,
        "supports": 14,
        "the": 15,
        "parent": 16,
        "comment.": 17,
        "disagrees": 18,
        "criticizes,": 19,
        "opposes": 20,
    }
    tok_raw = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_raw.pre_tokenizer = Whitespace()
    tok_raw.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_raw,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    cfg = RobertaConfig(
        vocab_size=len(vocab),
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=num_heads,
        intermediate_size=max(32, hidden_size),
        max_position_embeddings=max_pos,
        type_vocab_size=1,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        num_labels=3,
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
    )
    cls_model = RobertaForSequenceClassification(cfg)
    enc = RobertaModel(cfg)
    return tok, cls_model, enc


def _make_cascade_graphs(num_graphs: int = 8, in_dim: int = 16) -> list[CascadeData]:
    seed_everything(42)
    graphs: list[CascadeData] = []
    for i in range(num_graphs):
        td = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long)
        bu = td[[1, 0], :]
        ea = torch.tensor(
            [[0.8, 0.05], [-0.75, 0.25], [0.1, 0.1]],
            dtype=torch.float32,
        )
        graphs.append(
            CascadeData(
                x=torch.randn((4, in_dim), dtype=torch.float32),
                struct_x=torch.rand((4, 10), dtype=torch.float32) * 0.2,
                edge_index=td,
                edge_index_td=td,
                edge_index_bu=bu,
                edge_attr=ea,
                edge_stance=ea[:, 0:1],
                cascade_stats=torch.rand((1, 8), dtype=torch.float32),
                root_index=torch.tensor([0], dtype=torch.long),
                y_size=torch.tensor([2.5 + 0.4 * i], dtype=torch.float32),
                y_polar=torch.tensor([0.15 + 0.1 * (i % 7)], dtype=torch.float32),
                y_rumor=torch.tensor([i % 4], dtype=torch.long),
                cascade_id=f"c_{i}",
                topic_id=f"t_{i}",
            )
        )
    return graphs


def test_kl_calibrated_bce_zero_minimum_and_nonvanishing_tail_gradient() -> None:
    head = MultiTaskHead(in_dim=16, hidden_dim=16, num_rumor_classes=4)
    out = head(torch.randn(3, 16), torch.randn(6, 16))
    assert "y_polar_logit" in out and out["y_polar_logit"].shape == (3,)
    assert torch.allclose(torch.sigmoid(out["y_polar_logit"]), out["y_polar_pred"], atol=1e-6)

    # Exact optimum sigma(z) == y_polar must produce ~0.0 regression loss (no positive entropy floor)
    y_polar = torch.tensor([0.1, 0.5, 0.85], dtype=torch.float32)
    y_size = torch.tensor([3.2, 4.1, 2.7], dtype=torch.float32)
    opt_logits = torch.log(y_polar / (1.0 - y_polar))
    batch_opt = type("B", (), {"y_size": y_size, "y_polar": y_polar, "y_rumor": None})()
    opt_out = {
        "y_size_pred": y_size.clone(),
        "y_polar_logit": opt_logits,
        "y_polar_pred": torch.sigmoid(opt_logits),
        "rumor_logits": torch.zeros((3, 4), dtype=torch.float32),
    }
    loss_opt = compute_multitask_loss(opt_out, batch_opt)
    assert float(loss_opt.item()) < 1e-6

    # Far tail logit=-3.0 with target y_polar=0.95 must retain strong gradient via BCEWithLogits
    z_tail = torch.tensor([-3.0], dtype=torch.float32, requires_grad=True)
    y_tail = torch.tensor([0.95], dtype=torch.float32)
    batch_tail = type(
        "B",
        (),
        {"y_size": torch.tensor([1.0]), "y_polar": y_tail, "y_rumor": None},
    )()
    tail_out = {
        "y_size_pred": torch.tensor([1.0]),
        "y_polar_logit": z_tail,
        "y_polar_pred": torch.sigmoid(z_tail),
        "rumor_logits": torch.zeros((1, 4)),
    }
    loss_tail = compute_multitask_loss(tail_out, batch_tail)
    loss_tail.backward()
    assert z_tail.grad is not None and abs(float(z_tail.grad.item())) > 0.8


def test_trainer_bias_init_ema_best_val_seed_selection_and_comparator() -> None:
    graphs = _make_cascade_graphs(num_graphs=8, in_dim=16)
    tr_l = DataLoader(graphs[:4], batch_size=2, shuffle=True)
    va_l = DataLoader(graphs[4:6], batch_size=2, shuffle=False)
    te_l = DataLoader(graphs[6:8], batch_size=2, shuffle=False)

    model, metrics = train_and_evaluate_single_seed(
        lambda: SBiFAGAT(in_dim=16, hidden_dim=16, edge_dim=2, use_cascade_stats=True),
        tr_l,
        va_l,
        te_l,
        seed=42,
        epochs=12,
        lr=2e-3,
        patience=5,
        device=torch.device("cpu"),
    )
    assert "val_task_loss" in metrics and math.isfinite(float(metrics["val_task_loss"]))
    assert model is not None

    # Multi-seed selection must return best-val seed checkpoint, not blindly the last seed
    class _SeedTagModel(torch.nn.Module):
        def __init__(self, tag: int) -> None:
            super().__init__()
            self.tag = tag
            self.p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

        def forward(self, batch: Any) -> dict[str, torch.Tensor]:
            n = int(batch.num_graphs)
            offset = 0.05 if self.tag == 42 else 2.5
            return {
                "y_size_pred": batch.y_size.float() + offset + 0.0 * self.p,
                "y_polar_logit": torch.zeros((n,), dtype=torch.float32) + offset + 0.0 * self.p,
                "y_polar_pred": torch.sigmoid(torch.zeros((n,), dtype=torch.float32) + offset) + 0.0 * self.p,
                "rumor_logits": torch.zeros((n, 4), dtype=torch.float32) + 0.0 * self.p,
                "graph_emb": torch.zeros((n, 32), dtype=torch.float32),
                "node_emb": torch.zeros((batch.x.size(0), 16), dtype=torch.float32),
            }

    call_order: list[int] = []

    def _tag_factory() -> torch.nn.Module:
        seed_idx = len(call_order)
        call_order.append(seed_idx)
        return _SeedTagModel(42 if seed_idx == 0 else 999)

    bench = run_multi_seed_benchmark(
        {"BiGCN": _tag_factory},
        tr_l,
        va_l,
        te_l,
        seeds=[42, 999],
        epochs=2,
        patience=2,
        reference_model="BiGCN",
        compute_bootstrap=False,
        device=torch.device("cpu"),
    )
    assert bench["best_seed_by_model"]["BiGCN"] == 42
    assert getattr(bench["models"]["BiGCN"], "tag") == 42

    # Non-learned Early-K Persistence must never be chosen as statistical comparator
    stats_mock = {
        "Early-K Persistence": {m: (0.0, 0.0) for m in ("msle", "mae", "r2_size", "rmse", "pearson_r", "spearman_polar")},
        "BiGCN": {"msle": (0.2, 0.01), "mae": (0.3, 0.01), "r2_size": (0.5, 0.01), "rmse": (0.2, 0.01), "pearson_r": (0.6, 0.01), "spearman_polar": (0.6, 0.01)},
        "StandardGCN": {"msle": (0.4, 0.01), "mae": (0.5, 0.01), "r2_size": (0.2, 0.01), "rmse": (0.4, 0.01), "pearson_r": (0.2, 0.01), "spearman_polar": (0.2, 0.01)},
        "SBiFAGAT": {"msle": (0.1, 0.01), "mae": (0.2, 0.01), "r2_size": (0.7, 0.01), "rmse": (0.1, 0.01), "pearson_r": (0.8, 0.01), "spearman_polar": (0.8, 0.01)},
    }
    assert select_best_comparator(stats_mock, reference_model="SBiFAGAT") == "BiGCN"


def test_auto_loop_hidden64_and_bundle_security_exclusions(tmp_path: Path) -> None:
    assert all(cfg["hidden_dim"] == 64 and cfg["lambda_uncertainty"] == 0.0 for cfg in DEFAULT_V2_CANDIDATES)
    graphs = _make_cascade_graphs(num_graphs=4, in_dim=16)
    tr_l = DataLoader(graphs[:2], batch_size=2, shuffle=True)
    va_l = DataLoader(graphs[2:4], batch_size=2, shuffle=False)
    loop_out = run_self_auditing_validation_loop(
        tr_l,
        va_l,
        in_dim=16,
        hidden_dim=64,
        edge_dim=2,
        candidates=[{"lr": 1e-3, "eps": 0.5, "dropout": 0.1}],
        epochs=1,
        patience=1,
        output_dir=tmp_path / "al",
        device=torch.device("cpu"),
    )
    assert loop_out["best_config"]["hidden_dim"] == 64

    # _bundle_outputs MUST strip *_machine_key.csv, *.secret, and .env even if include_heavy_weights=True
    interim_dir = tmp_path / "data" / "interim" / "vn"
    interim_dir.mkdir(parents=True, exist_ok=True)
    blind_csv = interim_dir / "human_audit_vi_300.csv"
    key_csv = interim_dir / "human_audit_vi_300_machine_key.csv"
    salt_file = interim_dir / ".author_hash_salt.secret"
    env_file = tmp_path / ".env"
    blind_csv.write_text("sample_id,text\ns1,ok\n", encoding="utf-8")
    key_csv.write_text("sample_id,s_uv\ns1,0.9\n", encoding="utf-8")
    salt_file.write_text("super-secret-salt", encoding="utf-8")
    env_file.write_text("HF_TOKEN=hf_secret", encoding="utf-8")

    bundle_path = _bundle_outputs(
        tmp_path / "out_bundle.zip",
        source_paths=[interim_dir, env_file],
        base_dir=tmp_path,
        include_heavy_weights=True,
    )
    with zipfile.ZipFile(bundle_path, "r") as zf:
        archived = {Path(name).name for name in zf.namelist()}
    assert "human_audit_vi_300.csv" in archived
    assert "human_audit_vi_300_machine_key.csv" not in archived
    assert ".author_hash_salt.secret" not in archived
    assert ".env" not in archived


def test_canonical_topic_splits_enforced_with_in_memory_tables_and_k50_subset(tmp_path: Path) -> None:
    base_ts = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    node_embs: dict[str, torch.Tensor] = {}
    e_s: dict[str, float] = {}
    e_h: dict[str, float] = {}

    # 3 train topics (>=51 replies only on topic_train_big), 1 val topic, 1 test topic (>=51 replies)
    specs = [
        ("cid_tr1", "topic_tr_1", 52),
        ("cid_tr2", "topic_tr_2", 30),
        ("cid_tr3", "topic_tr_3", 30),
        ("cid_va1", "topic_va_1", 52),
        ("cid_te1", "topic_te_1", 52),
    ]
    for c_idx, (cid, tid, n_rep) in enumerate(specs):
        rid = f"{cid}_r"
        node_rows.append(
            {
                "node_id": rid,
                "cascade_id": cid,
                "topic_id": tid,
                "parent_id": None,
                "is_root": True,
                "text": "root",
                "created_at_utc": base_ts + timedelta(hours=c_idx),
                "lang": "en",
                "source_dataset": "pheme",
                "user_id_hash": "a" * 64,
                "rumor_label": "true",
            }
        )
        node_embs[rid] = torch.zeros(768, dtype=torch.float32)
        for k in range(1, n_rep + 1):
            nid = f"{cid}_n{k}"
            node_rows.append(
                {
                    "node_id": nid,
                    "cascade_id": cid,
                    "topic_id": tid,
                    "parent_id": rid,
                    "is_root": False,
                    "text": "reply",
                    "created_at_utc": base_ts + timedelta(hours=c_idx, seconds=10 * k),
                    "lang": "en",
                    "source_dataset": "pheme",
                    "user_id_hash": "b" * 64,
                    "rumor_label": None,
                }
            )
            node_embs[nid] = torch.zeros(768, dtype=torch.float32)
            eid = f"{rid}->{nid}"
            edge_rows.append(
                {
                    "edge_id": eid,
                    "cascade_id": cid,
                    "source_id": rid,
                    "target_id": nid,
                    "created_at_utc": base_ts + timedelta(hours=c_idx, seconds=10 * k),
                    "time_delay_sec": float(10 * k),
                    "edge_type": "reply",
                }
            )
            e_s[eid] = 0.5
            e_h[eid] = 0.1

    ds_dir = tmp_path / "proc"
    n_tbl = build_nodes_table(node_rows)
    e_tbl = build_edges_table(edge_rows)
    save_cascade_parquet(n_tbl, e_tbl, ds_dir, validate=True)
    nlp_feats = {"node_embeddings": node_embs, "edge_stance": e_s, "edge_hostility": e_h}
    torch.save(nlp_feats, ds_dir / "nlp_features.pt")
    canonical = {
        "train": ["topic_tr_1", "topic_tr_2", "topic_tr_3"],
        "val": ["topic_va_1"],
        "test": ["topic_te_1"],
    }
    save_topic_splits_json(canonical, ds_dir / "topic_splits.json")

    # Even with in-memory tables passed (used_disk_tables=False), topic_splits.json must be strictly honored!
    tr_mem = EarlyCascadeDataset(
        root_dir=ds_dir,
        k_horizon=20,
        split="train",
        min_total_replies=51,
        nodes_table=n_tbl,
        edges_table=e_tbl,
        nlp_features=nlp_feats,
    )
    te_mem = EarlyCascadeDataset(
        root_dir=ds_dir,
        k_horizon=50,
        split="test",
        min_total_replies=51,
        nodes_table=n_tbl,
        edges_table=e_tbl,
        nlp_features=nlp_feats,
    )
    assert tr_mem.topic_ids == {"topic_tr_1"}
    assert te_mem.topic_ids == {"topic_te_1"}
    assert "topic_tr_1" not in te_mem.topic_ids


def test_vi_url_normalization_nli_reply_preservation_and_40hop_bfs() -> None:
    # 1. normalize_social_text cleans URLs and extra spaces for VI while keeping @handle
    assert (
        normalize_social_text("Xem   https://vnexpress.net/abc   với @tuan_01 nha", lang="vi")
        == "Xem http với @tuan_01 nha"
    )

    # 2. NLIStanceTeacherScorer default max_length is 384 and reply tokens survive huge parent text
    sig = inspect.signature(NLIStanceTeacherScorer.score_pairs)
    assert sig.parameters["max_length"].default == 384

    tok, cls_model, _ = _build_offline_tok_and_cls_model(max_pos=512)
    captured_premises: list[str] = []
    scorer = NLIStanceTeacherScorer(
        model=cls_model,
        tokenizer=tok,
        lang="en",
        device=torch.device("cpu"),
    )
    orig_score = scorer._score_premise_hypothesis

    def _spy_score(premises: Any, hypotheses: Any, *, effective_max_len: int) -> torch.Tensor:
        captured_premises.extend(list(premises))
        return orig_score(premises, hypotheses, effective_max_len=effective_max_len)

    scorer._score_premise_hypothesis = _spy_score  # type: ignore[method-assign]
    huge_parent = " ".join(["longword"] * 600)
    scorer.score_pairs([huge_parent], ["targetreplytoken"], max_length=128)
    assert captured_premises and "targetreplytoken" in captured_premises[0]

    # 3. Root-aligned BFS propagates alternating signs over a 40-hop chain (beyond old limit 25)
    num_nodes = 41
    src = torch.arange(0, 40, dtype=torch.long)
    dst = torch.arange(1, 41, dtype=torch.long)
    ei = torch.stack([src, dst], dim=0)
    # Every hop opposes its parent (-0.9): odd depths should end negative, even depths positive (+0.9)
    local_sig = torch.full((40,), -0.9, dtype=torch.float32)
    aligned = _propagate_root_aligned_edge_signal(
        ei, local_sig, root_idx=torch.tensor([0], dtype=torch.long), num_nodes=num_nodes
    )
    assert float(aligned[0].item()) < -0.5   # hop 0->1: oppose root
    assert float(aligned[1].item()) > 0.5    # hop 1->2: oppose(oppose) = support root
    assert float(aligned[38].item()) < -0.5  # hop 38->39: odd depth -> negative
    assert float(aligned[39].item()) > 0.5   # hop 39->40: even depth -> positive


def test_extract_features_skips_redundant_deep_pass_and_kappa_coverage_and_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tok, _, enc = _build_offline_tok_and_cls_model(max_pos=128, hidden_size=768, num_heads=12)
    base_ts = datetime(2026, 5, 2, 0, 0, 0, tzinfo=timezone.utc)
    interim_dir = tmp_path / "interim"
    out_dir = tmp_path / "processed"

    ce = DualHeadCrossEncoder(lang="en", model_name="smoke-ce", encoder=enc, edge_dim=2)

    nodes_tbl = build_nodes_table(
        [
            {
                "node_id": "r0",
                "cascade_id": "c1",
                "topic_id": "t1",
                "parent_id": None,
                "is_root": True,
                "text": "Parent comment:",
                "created_at_utc": base_ts,
                "lang": "en",
                "source_dataset": "pheme",
                "user_id_hash": "a" * 64,
                "rumor_label": "true",
            },
            {
                "node_id": "r1",
                "cascade_id": "c1",
                "topic_id": "t1",
                "parent_id": "r0",
                "is_root": False,
                "text": "Reply:",
                "created_at_utc": base_ts + timedelta(seconds=10),
                "lang": "en",
                "source_dataset": "pheme",
                "user_id_hash": "b" * 64,
                "rumor_label": None,
            },
            {
                "node_id": "r2",
                "cascade_id": "c1",
                "topic_id": "t1",
                "parent_id": "r1",
                "is_root": False,
                "text": "targetreplytoken",
                "created_at_utc": base_ts + timedelta(seconds=20),
                "lang": "en",
                "source_dataset": "pheme",
                "user_id_hash": "c" * 64,
                "rumor_label": None,
            },
        ]
    )
    edges_tbl = build_edges_table(
        [
            {
                "edge_id": "r0->r1",
                "cascade_id": "c1",
                "source_id": "r0",
                "target_id": "r1",
                "created_at_utc": base_ts + timedelta(seconds=10),
                "time_delay_sec": 10.0,
                "edge_type": "reply",
            },
            {
                "edge_id": "r1->r2",
                "cascade_id": "c1",
                "source_id": "r1",
                "target_id": "r2",
                "created_at_utc": base_ts + timedelta(seconds=20),
                "time_delay_sec": 10.0,
                "edge_type": "reply",
            },
        ]
    )
    save_cascade_parquet(nodes_tbl, edges_tbl, interim_dir, validate=True)

    cache_path = interim_dir / "scored_cache.json"
    cache_path.write_text(
        json.dumps(
            [
                {"edge_id": "r0->r1", "s_uv": 0.6, "h_uv": 0.1, "root_s_uv": 0.6},
                {"edge_id": "r1->r2", "s_uv": -0.5, "h_uv": 0.4, "root_s_uv": -0.42},
            ]
        ),
        encoding="utf-8",
    )

    ce_call_count = 0
    orig_ce_fn = ef_mod.encode_edges_with_cross_encoder

    def _counting_ce(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal ce_call_count
        ce_call_count += 1
        return orig_ce_fn(*args, **kwargs)

    monkeypatch.setattr(ef_mod, "encode_edges_with_cross_encoder", _counting_ce)
    pt_path = ef_mod.extract_and_save_features(
        dataset="pheme",
        interim_dir=interim_dir,
        output_dir=out_dir,
        cross_encoder=ce,
        node_encoder=enc,
        tokenizer=tok,
        scored_cache_json=cache_path,
        edge_dim=2,
        include_root_context=False,
        device=torch.device("cpu"),
    )
    # Only the 1 parent->reply CE call runs; redundant deep_pred CE call is skipped when full root_s_uv exists!
    assert ce_call_count == 1
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert math.isclose(payload["node_root_stance"]["r2"], -0.42, abs_tol=1e-5)

    # compute_kappa must warn & track unadjudicated disagreements, and block in strict mode
    audit_csv = tmp_path / "audit.csv"
    key_csv = tmp_path / "audit_machine_key.csv"
    with audit_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_id",
                "edge_id",
                "annotator_1_stance",
                "annotator_2_stance",
                "gold_adjudicated_stance",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "sample_id": "s1",
                "edge_id": "r0->r1",
                "annotator_1_stance": "support",
                "annotator_2_stance": "support",
                "gold_adjudicated_stance": "",
            }
        )
        w.writerow(
            {
                "sample_id": "s2",
                "edge_id": "r1->r2",
                "annotator_1_stance": "support",
                "annotator_2_stance": "oppose",
                "gold_adjudicated_stance": "",
            }
        )
    with key_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "edge_id", "s_uv", "machine_stance_label"])
        w.writeheader()
        w.writerow({"sample_id": "s1", "edge_id": "r0->r1", "s_uv": "0.7", "machine_stance_label": "support"})
        w.writerow({"sample_id": "s2", "edge_id": "r1->r2", "s_uv": "-0.6", "machine_stance_label": "oppose"})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep = evaluate_machine_vs_human(audit_csv, key_csv)
    assert any(issubclass(w.category, UserWarning) for w in caught)
    assert rep["stance"]["num_samples"] == 1
    assert rep["stance"]["total_matched_rows"] == 2
    assert rep["stance"]["unadjudicated_disagreements"] == 1
    assert math.isclose(rep["stance"]["adjudication_coverage"], 0.5, abs_tol=1e-6)
    with pytest.raises(ValueError, match="disagreement"):
        evaluate_machine_vs_human(audit_csv, key_csv, strict_adjudication=True)

    # Early-K Persistence helper includes size + polar keys and integrates into main summary
    graphs = _make_cascade_graphs(num_graphs=4, in_dim=16)
    persist = compute_early_k_persistence_metrics(graphs)
    for k in ("mae", "rmse_size", "r2_size", "spearman_size", "rmse", "spearman_polar"):
        assert k in persist and math.isfinite(float(persist[k]))

    fake_res = {
        "summary_stats": {
            "SBiFAGAT": {"msle": (0.1, 0.01), "rmse": (0.1, 0.01)},
        }
    }
    attached = attach_early_k_persistence_to_result(fake_res, graphs)
    assert list(attached["summary_stats"].keys())[0] == "Early-K Persistence"
    assert math.isnan(attached["summary_stats"]["Early-K Persistence"]["macro_f1"][0])


def test_early_stopping_nan_guard_sealed_keys_dynamic_labels_and_notebooks(tmp_path: Path) -> None:
    # 1. EarlyStopping rejects NaN/+inf/-inf both initially and after a valid finite score
    stopper = EarlyStopping(patience=2, mode="min")
    tiny = torch.nn.Linear(2, 1)
    with torch.no_grad():
        tiny.weight.fill_(1.0)
    assert not stopper._is_improved(float("nan"))
    assert not stopper._is_improved(float("inf"))
    assert not stopper._is_improved(float("-inf"))
    stopper.step(float("nan"), tiny)
    assert stopper.best_score is None and stopper.best_state is None

    assert stopper._is_improved(0.5)
    stopper.step(0.5, tiny)
    saved_w = float(stopper.best_state["weight"][0, 0].item())
    with torch.no_grad():
        tiny.weight.fill_(9.0)
    stopper.step(float("nan"), tiny)
    stopper.step(float("-inf"), tiny)
    assert stopper.best_score == 0.5
    assert float(stopper.best_state["weight"][0, 0].item()) == saved_w
    assert stopper.should_stop is True

    # 2. _bundle_sealed_machine_keys and _run_smoke_pipeline produce separate sealed key archive
    smoke_res = _run_smoke_pipeline("member1_vn", "vi", tmp_path / "smoke_sealed")
    main_zip = Path(smoke_res["bundle_zip"])
    sealed_zip = Path(smoke_res["sealed_machine_keys_zip"])
    assert main_zip.exists() and sealed_zip.exists()
    with zipfile.ZipFile(main_zip, "r") as zf:
        main_names = {Path(n).name for n in zf.namelist()}
    with zipfile.ZipFile(sealed_zip, "r") as zf:
        sealed_names = {Path(n).name for n in zf.namelist()}
    assert "human_audit_vi_300.csv" in main_names
    assert not any(n.endswith("_machine_key.csv") for n in main_names)
    assert "human_audit_vi_300_machine_key.csv" in sealed_names
    assert _bundle_sealed_machine_keys(
        tmp_path / "empty_sealed.zip", search_roots=[tmp_path / "missing_dir"], base_dir=tmp_path
    ) is None

    # 3. Dynamic LaTeX \label resolvers deduplicate labels across PHEME, LOO, and VN
    assert _resolve_main_table_label(Path("data/processed/pheme"), "inductive", Path("paper/tables_main_pheme.tex")) == "tab:main_pheme"
    assert _resolve_main_table_label(Path("data/processed/pheme"), "loo", Path("paper/tables_loo_pheme.tex")) == "tab:loo_pheme"
    assert _resolve_main_table_label(Path("data/processed/vn"), "inductive", Path("paper/tables_main_vn.tex")) == "tab:main_vn"
    assert _resolve_ablation_table_label(Path("data/processed/pheme"), Path("paper/tables_ablation_pheme.tex")) == "tab:ablation_pheme"
    assert _resolve_ablation_table_label(Path("data/processed/vn"), Path("paper/tables_ablation_vn.tex")) == "tab:ablation_vn"
    assert _resolve_eda_table_label(Path("paper/tables_dataset_eda_pheme.tex"), [{"name": "pheme"}]) == "tab:dataset_eda_pheme"
    assert _resolve_eda_table_label(Path("paper/tables_dataset_eda_vn.tex"), [{"name": "vn"}]) == "tab:dataset_eda_vn"
    assert _resolve_eda_table_label(Path("paper/tables_dataset_eda.tex"), [{"name": "pheme"}, {"name": "vn"}]) == "tab:dataset_eda"
    assert _resolve_efficiency_table_label(Path("paper/tables_efficiency_pheme.tex")) == "tab:efficiency_pheme"
    assert _resolve_efficiency_table_label(Path("paper/tables_efficiency_vn.tex")) == "tab:efficiency_vn"
    assert _resolve_efficiency_table_label(Path("paper/tables_efficiency.tex")) == "tab:efficiency"

    # 4. Notebooks reference checkpoints/best_sbifagat_vn.pt and sealed machine keys archive
    repo_root = Path(__file__).resolve().parents[1]
    nb1 = (repo_root / "notebooks" / "kaggle_member1_nlp_and_vn_t4x2.ipynb").read_text(encoding="utf-8")
    nb2 = (repo_root / "notebooks" / "kaggle_member2_gnn_and_en_t4x2.ipynb").read_text(encoding="utf-8")
    assert "--gnn-ckpt checkpoints/best_sbifagat_vn.pt" in nb1
    assert "--gnn-ckpt outputs/auto_loop_vn/best_validated_sbifagat.pt" not in nb1
    assert "sbifagat_member1_vn_sealed_machine_keys.zip" in nb1
    assert "sbifagat_member2_en_sealed_machine_keys.zip" in nb2


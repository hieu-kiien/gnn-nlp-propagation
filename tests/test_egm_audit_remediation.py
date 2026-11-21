"""Comprehensive regression test suite for EGM 5-Goldfish audit remediations (B1-B10, M1-M8)."""

from __future__ import annotations

import csv
import json
import math
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from transformers import RobertaConfig, RobertaModel

from scripts.assemble_paper_artifacts import (
    assemble_paper_artifacts,
    find_unescaped_underscores_in_latex,
)
from scripts.audit_vn_50_trees import export_vn_50_tree_audit
from scripts.benchmark_efficiency import format_efficiency_latex
from scripts.compute_kappa import evaluate_thread_level_audit_csv, format_human_audit_latex
from scripts.export_dataset_stats import compute_early_k_persistence_metrics
import scripts.kaggle_parallel_runner as kpr
from src.data import arctic_shift_vn as asvn
from src.data.pheme_loader import parse_pheme_thread
from src.data.pyg_dataset import (
    CASCADE_STATS_DIM,
    CascadeData,
    EarlyCascadeDataset,
    STRUCTURAL_NODE_DIM,
    TOTAL_NODE_FEATURE_DIM,
    _compute_full_tree_polarization,
    _compute_root_aligned_stances,
)
from src.data.schema import build_edges_table, build_nodes_table, load_cascade_parquet, save_cascade_parquet
from src.models.baselines import BiEdgeGATv2, TextMLP
from src.models.sbi_fagat import SBiFAGAT
import src.nlp.dual_edge_scorer as des
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    NLIStanceTeacherScorer,
    _score_pairs_with_teachers,
    export_stratified_human_audit_csv,
)
from src.nlp.extract_features import extract_and_save_features
from src.training.metrics import evaluate_predictions
from src.training.reporting import (
    MAIN_METRICS,
    UPFD_METRICS,
    build_markdown_table,
    export_latex_table,
    serialize_benchmark_json,
)
from src.training.trainer import (
    _compute_sample_level_bootstrap_pvalues,
    compute_paired_bootstrap_pvalue,
)


def _tiny_tok_and_enc():
    return kpr._build_smoke_offline_tokenizer_and_encoder()


def _make_cascade_rows(
    cid: str,
    topic_id: str,
    num_replies: int,
    *,
    lang: str = "en",
    rumor_label: str | None = "true",
    base_ts: datetime | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    t0 = base_ts or datetime(2026, 7, 1, 8, 0, 0, tzinfo=timezone.utc)
    root_id = f"{cid}_r"
    nodes: list[dict[str, Any]] = [
        {
            "node_id": root_id,
            "cascade_id": cid,
            "topic_id": topic_id,
            "parent_id": None,
            "is_root": True,
            "text": f"root topic {cid}",
            "created_at_utc": t0,
            "lang": lang,
            "source_dataset": "pheme" if lang == "en" else "arctic_shift_vozforums",
            "user_id_hash": "a" * 64,
            "rumor_label": rumor_label,
        }
    ]
    edges: list[dict[str, Any]] = []
    for i in range(1, num_replies + 1):
        nid = f"{cid}_n{i}"
        pid = root_id if i <= 5 else f"{cid}_n{i - 5}"
        dt = float(i * 45)
        nodes.append(
            {
                "node_id": nid,
                "cascade_id": cid,
                "topic_id": topic_id,
                "parent_id": pid,
                "is_root": False,
                "text": "agree support reply" if i % 2 == 0 else "oppose hate trash reply",
                "created_at_utc": t0 + timedelta(seconds=dt),
                "lang": lang,
                "source_dataset": "pheme" if lang == "en" else "arctic_shift_vozforums",
                "user_id_hash": f"{i:064x}",
                "rumor_label": None,
            }
        )
        edges.append(
            {
                "edge_id": f"{pid}->{nid}",
                "cascade_id": cid,
                "source_id": pid,
                "target_id": nid,
                "created_at_utc": t0 + timedelta(seconds=dt),
                "time_delay_sec": dt,
                "edge_type": "reply",
            }
        )
    return nodes, edges


def test_b1_extract_features_min_total_replies_preserves_teacher_and_raw_parquet(tmp_path: Path) -> None:
    """B1: extract_features(--min-total-replies 25) populates teacher_edge_* for LOO while preserving raw parquet."""
    interim_dir = tmp_path / "interim"
    out_dir = tmp_path / "processed"
    n_short, e_short = _make_cascade_rows("c_short", "ev_a", num_replies=5)
    n_elig1, e_elig1 = _make_cascade_rows("c_elig1", "ev_a", num_replies=26)
    n_elig2, e_elig2 = _make_cascade_rows("c_elig2", "ev_b", num_replies=27)
    n_elig3, e_elig3 = _make_cascade_rows("c_elig3", "ev_c", num_replies=28)

    all_nodes = n_short + n_elig1 + n_elig2 + n_elig3
    all_edges = e_short + e_elig1 + e_elig2 + e_elig3
    save_cascade_parquet(
        build_nodes_table(all_nodes),
        build_edges_table(all_edges),
        output_dir=interim_dir,
        validate=True,
    )

    # Scored cache contains ONLY eligible (>=25) edges — exact real-world PHEME condition!
    elig_edges = e_elig1 + e_elig2 + e_elig3
    scored_cache = [
        {
            "edge_id": e["edge_id"],
            "s_uv": -0.75 if idx % 2 == 0 else 0.80,
            "h_uv": 0.65 if idx % 2 == 0 else 0.05,
            "root_s_uv": -0.70 if idx % 2 == 0 else 0.75,
            "pred_label": "oppose" if idx % 2 == 0 else "support",
        }
        for idx, e in enumerate(elig_edges)
    ]
    cache_path = interim_dir / "scored_pairs.json"
    cache_path.write_text(json.dumps(scored_cache), encoding="utf-8")

    tok, enc = _tiny_tok_and_enc()
    ce = DualHeadCrossEncoder(lang="en", model_name="tiny-ce", encoder=enc, edge_dim=2)

    pt_path = extract_and_save_features(
        dataset="pheme",
        interim_dir=interim_dir,
        output_dir=out_dir,
        cross_encoder=ce,
        node_encoder=enc,
        tokenizer=tok,
        scored_cache_json=cache_path,
        edge_dim=2,
        min_total_replies=25,
        batch_size=16,
        device=torch.device("cpu"),
    )
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert payload["min_total_replies_extracted"] == 25
    assert "teacher_edge_stance" in payload
    assert "teacher_node_root_stance" in payload
    assert len(payload["teacher_edge_stance"]) == len(elig_edges)

    # Full raw parquet preserved for Table 1 EDA (raw_cascades=4, eligible=3)
    saved_nodes, _ = load_cascade_parquet(out_dir, validate=True)
    saved_cids = {str(r["cascade_id"]) for r in saved_nodes.to_pylist()}
    assert saved_cids == {"c_short", "c_elig1", "c_elig2", "c_elig3"}

    # PHEME LOO with edge_source="teacher" now loads cleanly without ValueError
    loo_ds = EarlyCascadeDataset(
        root_dir=out_dir,
        k_horizon=20,
        split="test",
        min_total_replies=25,
        custom_topic_splits={"train": {"ev_a"}, "val": {"ev_b"}, "test": {"ev_c"}},
        edge_source="teacher",
    )
    assert len(loo_ds) == 1
    assert loo_ds[0].cascade_id == "c_elig3"


def test_b2_edge_source_auto_uses_student_even_with_custom_topic_splits() -> None:
    """B2: edge_source='auto' must select Student edges even when custom_topic_splits is passed."""
    n1, e1 = _make_cascade_rows("c1", "t1", num_replies=26)
    n2, e2 = _make_cascade_rows("c2", "t2", num_replies=26)
    n3, e3 = _make_cascade_rows("c3", "t3", num_replies=26)
    nodes = n1 + n2 + n3
    edges = e1 + e2 + e3
    nlp_feats = {
        "node_id_to_idx": {r["node_id"]: i for i, r in enumerate(nodes)},
        "node_embeddings": torch.zeros((len(nodes), 768), dtype=torch.float32),
        "edge_stance": {e["edge_id"]: 0.25 for e in edges},
        "edge_hostility": {e["edge_id"]: 0.10 for e in edges},
        "teacher_edge_stance": {e["edge_id"]: -0.85 for e in edges},
        "teacher_edge_hostility": {e["edge_id"]: 0.90 for e in edges},
    }
    splits = {"train": {"t1"}, "val": {"t2"}, "test": {"t3"}}
    ds_auto = EarlyCascadeDataset(
        k_horizon=20,
        split="test",
        nodes_table=build_nodes_table(nodes),
        edges_table=build_edges_table(edges),
        nlp_features=nlp_feats,
        min_total_replies=25,
        custom_topic_splits=splits,
        edge_source="auto",
    )
    ds_teacher = EarlyCascadeDataset(
        k_horizon=20,
        split="test",
        nodes_table=build_nodes_table(nodes),
        edges_table=build_edges_table(edges),
        nlp_features=nlp_feats,
        min_total_replies=25,
        custom_topic_splits=splits,
        edge_source="teacher",
    )
    assert float(ds_auto[0].edge_attr[:, 0].mean().item()) == pytest.approx(0.25, abs=1e-5)
    assert float(ds_teacher[0].edge_attr[:, 0].mean().item()) == pytest.approx(-0.85, abs=1e-5)


def test_b3_incremental_jsonl_teacher_checkpoint_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B3: Teacher scoring appends compact score-only JSONL deltas and resumes without rescoring."""
    pairs = [
        {
            "edge_id": f"e_{i}",
            "root_id": "r0",
            "parent_id": "r0",
            "root_text": "root text",
            "parent_text": "root text",
            "reply_text": f"reply text {i}",
        }
        for i in range(6)
    ]
    cache_path = tmp_path / "scored_pairs.json"
    ckpt_jsonl = tmp_path / "scored_pairs.json.checkpoint.jsonl"

    host_calls: list[int] = []
    nli_calls: list[int] = []
    fail_once = {"armed": True}

    class _FakeHost:
        def __init__(self, **_kw: Any) -> None:
            pass

        def score_texts(self, texts: list[str], **_kw: Any) -> np.ndarray:
            host_calls.append(len(texts))
            return np.full((len(texts),), 0.25, dtype=np.float32)

    class _FakeNLI:
        def __init__(self, **_kw: Any) -> None:
            pass

        def score_pairs(self, parents: list[str], replies: list[str], **_kw: Any) -> dict[str, Any]:
            nli_calls.append(len(replies))
            if fail_once["armed"] and len(nli_calls) == 2:
                fail_once["armed"] = False
                raise RuntimeError("Simulated mid-chunk preemption")
            n = len(replies)
            return {
                "s_uv": np.full((n,), 0.6, dtype=np.float32),
                "stance_probs": np.tile(np.array([[0.7, 0.2, 0.1]], dtype=np.float32), (n, 1)),
                "pred_label": ["support"] * n,
            }

    monkeypatch.setattr(des, "HostilityTeacherScorer", _FakeHost)
    monkeypatch.setattr(des, "NLIStanceTeacherScorer", _FakeNLI)

    # First run fails on NLI chunk 2 after flushing hostility + NLI chunk 1 to sidecar JSONL
    with pytest.raises(RuntimeError, match="Simulated mid-chunk preemption"):
        _score_pairs_with_teachers(
            [dict(p) for p in pairs],
            lang="en",
            cache_path=cache_path,
            checkpoint_every=2,
        )
    assert ckpt_jsonl.exists()
    assert not cache_path.exists()
    # Sidecar delta lines MUST NOT duplicate parent_text/reply_text
    raw_delta = ckpt_jsonl.read_text(encoding="utf-8")
    assert "reply text" not in raw_delta

    # Second run resumes cleanly from .checkpoint.jsonl without calling hostility again
    host_calls.clear()
    nli_calls.clear()
    resumed = _score_pairs_with_teachers(
        [dict(p) for p in pairs],
        lang="en",
        cache_path=cache_path,
        checkpoint_every=2,
    )
    assert host_calls == []  # All 6 hostility scores recovered from JSONL
    assert sum(nli_calls) == 4  # Only remaining 4 pairs scored by NLI
    assert cache_path.exists()
    assert not ckpt_jsonl.exists()  # Sidecar cleaned up after atomic final consolidation
    assert all("s_uv" in p and "h_uv" in p and "root_s_uv" in p for p in resumed)


def test_b4_crawl_state_resume_fault_isolation_and_salted_pheme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4: Crawl isolates failing threads, persists .state.json cursor/rejects; PHEME uses salted SHA-256."""
    base_ts = 1763164800
    posts = [
        {
            "id": "bad_post",
            "title": "Bài lỗi API",
            "selftext": "Thảo luận tiếng Việt dài đầy đủ",
            "subreddit": "vozforums",
            "author": "op1",
            "created_utc": base_ts - 100,
            "num_comments": 40,
        },
        {
            "id": "good_post",
            "title": "Thảo luận giao thông Hà Nội và tàu điện đô thị",
            "selftext": "Mọi người nghĩ sao về phương án mở rộng tuyến tàu điện và xe buýt điện?",
            "subreddit": "vozforums",
            "author": "op2",
            "created_utc": base_ts - 200,
            "num_comments": 40,
        },
    ]

    def _fake_req(_session: Any, _url: str, **_kw: Any) -> dict[str, Any]:
        return {"data": posts}

    tree_calls: list[str] = []

    def _fake_tree(_session: Any, post_id: str, **_kw: Any) -> list[dict[str, Any]]:
        tree_calls.append(post_id)
        if post_id == "bad_post":
            raise RuntimeError("502 Bad Gateway")
        return [
            {
                "fullname": f"t1_c{i}",
                "id": f"c{i}",
                "link_id": "t3_good_post",
                "parent_id": "t3_good_post" if i <= 3 else f"t1_c{i - 3}",
                "text": f"Bình luận tiếng Việt hoàn chỉnh có dấu số {i} về giao thông đô thị.",
                "created_utc": base_ts - 190 + i * 10,
                "author_hash": asvn.hash_author_sha256(f"user_{i}"),
                "author_is_bot": False,
            }
            for i in range(1, 28)
        ]

    monkeypatch.setattr(asvn, "_request_with_retry", _fake_req)
    monkeypatch.setattr(asvn, "fetch_comment_tree", _fake_tree)
    monkeypatch.setattr(asvn.time, "sleep", lambda _s: None)

    ckpt = tmp_path / "crawl_checkpoint.jsonl"
    out_dir = tmp_path / "vn_out"
    asvn.crawl_arctic_shift_discussions(
        output_dir=out_dir,
        subreddits=["vozforums"],
        target_count=5,
        min_comments=25,
        min_vi_ratio=0.65,
        checkpoint_jsonl=ckpt,
        allow_partial=True,
        max_pages_per_sub=1,
    )
    state_path = ckpt.with_name(f"{ckpt.name}.state.json")
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "bad_post" in state["rejected_post_ids"]
    nodes_tbl, _ = load_cascade_parquet(out_dir, validate=True)
    assert len({r["cascade_id"] for r in nodes_tbl.to_pylist()}) == 1

    # Resume must skip previously rejected bad_post and already-collected good_post
    tree_calls.clear()
    asvn.crawl_arctic_shift_discussions(
        output_dir=out_dir,
        subreddits=["vozforums"],
        target_count=5,
        min_comments=25,
        min_vi_ratio=0.65,
        checkpoint_jsonl=ckpt,
        allow_partial=True,
        max_pages_per_sub=2,
    )
    assert "bad_post" not in tree_calls and "good_post" not in tree_calls

    # Salted SHA-256 in PHEME loader
    pheme_thread = tmp_path / "pheme_thread"
    (pheme_thread / "source-tweets").mkdir(parents=True)
    (pheme_thread / "reactions").mkdir(parents=True)
    (pheme_thread / "source-tweets" / "100.json").write_text(
        json.dumps(
            {
                "id_str": "100",
                "text": "Breaking rumor claim",
                "created_at": "Wed Jan 07 11:00:00 +0000 2015",
                "user": {"id_str": "u_root"},
            }
        ),
        encoding="utf-8",
    )
    (pheme_thread / "structure.json").write_text(json.dumps({"100": {}}), encoding="utf-8")

    monkeypatch.delenv("SBIF_HASH_SALT", raising=False)
    res_unsalted = parse_pheme_thread(pheme_thread, topic_id="charliehebdo", is_rumour=True)
    monkeypatch.setenv("SBIF_HASH_SALT", "unit-test-secret-salt")
    res_salted = parse_pheme_thread(pheme_thread, topic_id="charliehebdo", is_rumour=True)
    assert res_unsalted is not None and res_salted is not None
    assert res_unsalted[0][0]["user_id_hash"] != res_salted[0][0]["user_id_hash"]


def test_b6_latex_underscore_escaping_wide_table_and_rfc_json(tmp_path: Path) -> None:
    """B6 & M8: Escapes '_' in comparator_model, uses table* for wide tables, emits strict RFC JSON."""
    bench = {
        "seeds": [42, 100],
        "reference_model": "SBiFAGAT (Full)",
        "comparator_model": "w/o Stance Score (s_uv=0)",
        "summary_stats": {
            "SBiFAGAT (Full)": {
                "msle": (0.20, 0.01),
                "mae": (0.30, 0.02),
                "r2_size": (0.60, 0.03),
                "rmse": (0.12, 0.01),
                "pearson_r": (0.70, 0.02),
                "spearman_polar": (0.68, 0.02),
                "macro_f1": (0.75, 0.02),
                "accuracy": (0.80, 0.02),
            },
            "w/o Stance Score (s_uv=0)": {
                "msle": (0.35, 0.02),
                "mae": (0.45, 0.02),
                "r2_size": (0.40, 0.03),
                "rmse": (0.22, 0.02),
                "pearson_r": (0.45, 0.03),
                "spearman_polar": (0.42, 0.03),
                "macro_f1": (0.62, 0.02),
                "accuracy": (0.68, 0.02),
            },
        },
        "p_values": {"msle": 0.002, "rmse": 0.004, "pearson_r": 0.003, "accuracy": 0.005},
        "adjusted_p_values": {"msle": 0.008, "rmse": 0.012, "pearson_r": 0.009, "accuracy": 0.015},
        "bootstrap_p_values": {"msle": 0.001, "rmse": 0.002, "pearson_r": 0.001, "accuracy": 0.002},
        "adjusted_bootstrap_p_values": {"msle": 0.004, "rmse": 0.006, "pearson_r": 0.004, "accuracy": 0.006},
        "raw_results": {
            "SBiFAGAT (Full)": [{"msle": 0.2, "macro_f1": float("nan")}],
            "w/o Stance Score (s_uv=0)": [{"msle": 0.35, "macro_f1": float("inf")}],
        },
    }
    wide_tex = export_latex_table(
        bench,
        tmp_path / "tables_ablation.tex",
        caption="Ablation Table",
        label="tab:ablation_test",
        metrics=MAIN_METRICS,
    )
    wide_txt = wide_tex.read_text(encoding="utf-8")
    assert "\\begin{table*}[t]" in wide_txt
    assert "\\resizebox{\\textwidth}{!}" in wide_txt
    assert "s\\_uv=0" in wide_txt
    assert find_unescaped_underscores_in_latex(wide_txt) == []

    narrow_tex = export_latex_table(
        bench,
        tmp_path / "tables_upfd.tex",
        caption="UPFD Table",
        label="tab:upfd_test",
        metrics=UPFD_METRICS,
    )
    narrow_txt = narrow_tex.read_text(encoding="utf-8")
    assert "\\begin{table}[t]" in narrow_txt
    assert "\\resizebox{\\columnwidth}{!}" in narrow_txt
    assert find_unescaped_underscores_in_latex(narrow_txt) == []

    json_path = serialize_benchmark_json(
        bench, tmp_path / "results.json", extra={"bad_float": float("nan")}
    )
    raw_json = json_path.read_text(encoding="utf-8")
    assert "NaN" not in raw_json and "Infinity" not in raw_json
    parsed = json.loads(raw_json)
    assert parsed["raw_results"]["SBiFAGAT (Full)"][0]["macro_f1"] is None
    assert parsed["extra"]["bad_float"] is None


def test_b7_nli_double_contradiction_neutral_joint_audit_and_vn_50_tree_audit(tmp_path: Path) -> None:
    """B7: NLI double-contradiction -> neutral; 2D joint stratification; 50-tree VN blind audit."""
    class _DoubleContraNLIModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
                num_labels=3,
                max_position_embeddings=128,
            )

        def forward(self, input_ids: torch.Tensor, **_kw: Any) -> Any:
            bsz = input_ids.size(0)
            # Both support and oppose hypotheses are strongly contradicted -> off-topic/question
            logits = torch.tensor([[-4.0, -4.0, 4.0]], dtype=torch.float32).repeat(bsz, 1)
            return SimpleNamespace(logits=logits)

    tok, _ = _tiny_tok_and_enc()
    scorer = NLIStanceTeacherScorer(
        model=_DoubleContraNLIModel(),
        tokenizer=tok,
        device=torch.device("cpu"),
        lang="vi",
        use_stance_verbalizer=True,
    )
    out = scorer.score_pairs(["Bài gốc về giao thông"], ["Bạn mua áo ở đâu vậy?"])
    assert out["pred_label"][0] == "neutral"
    assert float(out["stance_probs"][0, 1]) > 0.95
    assert abs(float(out["s_uv"][0])) < 1e-4

    # Joint stance x hostility stratification in 2D audit CSV
    pairs_2d = []
    for stance_idx, lbl in enumerate(("support", "neutral", "oppose")):
        for j in range(20):
            pairs_2d.append(
                {
                    "cascade_id": f"c_{stance_idx}_{j}",
                    "edge_id": f"e_{stance_idx}_{j}",
                    "parent_text": "parent",
                    "reply_text": "reply",
                    "pred_label": lbl,
                    "s_uv": 0.8 if lbl == "support" else (-0.8 if lbl == "oppose" else 0.0),
                    "h_uv": 0.85 if j < 6 else 0.05,
                }
            )
    audit_csv = tmp_path / "human_audit_vi_300.csv"
    key_csv = tmp_path / "human_audit_vi_300_machine_key.csv"
    export_stratified_human_audit_csv(
        pairs_2d,
        output_csv=audit_csv,
        machine_key_csv=key_csv,
        n_samples=30,
        seed=42,
    )
    with key_csv.open("r", encoding="utf-8-sig") as fh:
        key_rows = list(csv.DictReader(fh))
    hostile_selected = sum(1 for r in key_rows if float(r["h_uv"]) >= 0.4)
    assert hostile_selected >= 12  # Stratified instead of underrepresented

    # 50-tree VN blind audit export & evaluation
    vn_dir = tmp_path / "processed_vn"
    nodes, edges = [], []
    for t_i in range(6):
        n_r, e_r = _make_cascade_rows(
            f"vn_c{t_i}", f"vn_t{t_i}", num_replies=26, lang="vi", rumor_label=None
        )
        nodes.extend(n_r)
        edges.extend(e_r)
    save_cascade_parquet(build_nodes_table(nodes), build_edges_table(edges), output_dir=vn_dir, validate=True)
    e_stance = {}
    for e in edges:
        cid = e["cascade_id"]
        idx = int(cid.replace("vn_c", ""))
        num = int(e["target_id"].split("_n")[-1])
        e_stance[e["edge_id"]] = (1.0 if (num % 2 == 0) else -1.0) if idx % 2 == 0 else 0.9
    torch.save(
        {
            "node_id_to_idx": {r["node_id"]: i for i, r in enumerate(nodes)},
            "node_embeddings": torch.zeros((len(nodes), 768), dtype=torch.float32),
            "edge_stance": e_stance,
            "edge_hostility": {e["edge_id"]: 0.1 for e in edges},
        },
        vn_dir / "nlp_features.pt",
    )
    (vn_dir / "topic_splits.json").write_text(
        json.dumps({"train": ["vn_t0", "vn_t1"], "val": ["vn_t2"], "test": ["vn_t3", "vn_t4", "vn_t5"]}),
        encoding="utf-8",
    )

    tree_csv, tree_key = export_vn_50_tree_audit(
        data_dir=vn_dir,
        output_csv=tmp_path / "human_audit_vn_50_trees.csv",
        n_samples=50,
        min_total_replies=25,
    )
    assert "y_polar" not in tree_csv.read_text(encoding="utf-8-sig")
    assert "y_polar" in tree_key.read_text(encoding="utf-8-sig")

    # Simulate dual human grading matching the sealed machine key
    with tree_key.open("r", encoding="utf-8-sig") as fk:
        kmap = {r["cascade_id"]: int(r["y_polar_grade"]) for r in csv.DictReader(fk)}
    with tree_csv.open("r", encoding="utf-8-sig") as fb:
        rows = list(csv.DictReader(fb))
    for r in rows:
        grade = kmap[r["cascade_id"]]
        r["annotator_1_polar_grade"] = str(grade)
        r["annotator_2_polar_grade"] = str(grade)
        r["gold_adjudicated_polar_grade"] = str(grade)
    with tree_csv.open("w", encoding="utf-8-sig", newline="") as fb:
        w = csv.DictWriter(fb, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    thread_rep = evaluate_thread_level_audit_csv(tree_csv, tree_key, strict_adjudication=True)
    assert thread_rep["inter_annotator"]["cohen_kappa_quadratic"] == pytest.approx(1.0)
    assert thread_rep["machine_vs_gold"]["cohen_kappa_quadratic"] == pytest.approx(1.0)
    audit_tex = format_human_audit_latex(thread_rep)
    assert "\\toprule" in audit_tex and find_unescaped_underscores_in_latex(audit_tex) == []


def test_b8_direction_aware_stars_exact_loo_permutation_and_topic_cluster_bootstrap(tmp_path: Path) -> None:
    """B8: Zero stars when ref is worse; exact F=4 permutation p=0.125; topic-cluster bootstrap works."""
    # 1. Direction-aware significance markers
    worse_bench = {
        "reference_model": "SBiFAGAT",
        "comparator_model": "BiGCN",
        "summary_stats": {
            "SBiFAGAT": {"msle": (0.90, 0.01), "pearson_r": (0.20, 0.01)},
            "BiGCN": {"msle": (0.30, 0.01), "pearson_r": (0.75, 0.01)},
        },
        "p_values": {"msle": 0.0001, "pearson_r": 0.0001},
        "adjusted_bootstrap_p_values": {"msle": 0.0005, "pearson_r": 0.0005},
    }
    subset_metrics = (("msle", "MSLE $\\downarrow$", True), ("pearson_r", "Pearson $r$ $\\uparrow$", False))
    md_worse = build_markdown_table(worse_bench, subset_metrics)
    assert "**" not in md_worse.splitlines()[2].replace("**0.9000", "")  # No significance star on worse row
    assert "*" not in md_worse.splitlines()[2]
    tex_worse = export_latex_table(
        worse_bench, tmp_path / "worse.tex", caption="Worse", label="tab:w", metrics=subset_metrics
    ).read_text(encoding="utf-8")
    assert "$^{**}$" not in tex_worse and "$^{*}$" not in tex_worse

    # 2. Exact small-sample LOO permutation (F=4 all-positive diffs -> exact 2/16 = 0.125, NOT 0.0!)
    p_loo4 = compute_paired_bootstrap_pvalue([0.80, 0.85, 0.90, 0.95], [0.50, 0.55, 0.60, 0.65])
    assert p_loo4 == pytest.approx(0.125, abs=1e-9)

    # 3. Topic-cluster bootstrap when >= 6 distinct test topics exist
    rng = np.random.default_rng(12)
    n = 36
    topic_ids = [f"topic_{i // 3}" for i in range(n)]  # 12 topics
    yt_s = rng.uniform(3.0, 5.0, size=n)
    yt_p = rng.uniform(0.1, 0.9, size=n)
    preds_ref = [
        {
            "y_size_true": yt_s,
            "y_size_pred": yt_s + rng.normal(0.0, 0.05, size=n),
            "y_polar_true": yt_p,
            "y_polar_pred": np.clip(yt_p + rng.normal(0.0, 0.03, size=n), 0.0, 1.0),
            "topic_ids": topic_ids,
        }
    ]
    preds_cmp = [
        {
            "y_size_true": yt_s,
            "y_size_pred": yt_s + rng.normal(0.0, 0.45, size=n),
            "y_polar_true": yt_p,
            "y_polar_pred": np.clip(yt_p + rng.normal(0.0, 0.30, size=n), 0.0, 1.0),
            "topic_ids": topic_ids,
        }
    ]
    boot_p = _compute_sample_level_bootstrap_pvalues(
        preds_ref, preds_cmp, metric_keys=("msle", "mae", "rmse", "pearson_r"), n_resamples=300, n_stat_resamples=200
    )
    assert 0.0 < boot_p["msle"] < 0.05
    assert 0.0 < boot_p["rmse"] < 0.05


def test_b9_continuous_ypolar_noise_suppression_and_fixed_k_future_window() -> None:
    """B9: Weak noise yields ~0.013 (not 0.506), and polar_future_start_k aligns target across K."""
    weak_noise = [0.11] * 10 + [-0.11] * 10
    genuine_split = [0.9] * 6 + [0.0] * 8 + [-0.9] * 6
    assert _compute_full_tree_polarization(weak_noise) < 0.02
    assert _compute_full_tree_polarization(genuine_split) > 0.35
    assert _compute_full_tree_polarization([1.0] * 5 + [-1.0] * 5) == pytest.approx(1.0)

    n_rows, e_rows = _make_cascade_rows("kwin", "tk", num_replies=35)
    e_stance = {
        e["edge_id"]: (0.95 if idx < 15 else (-0.95 if idx % 2 == 0 else 0.95))
        for idx, e in enumerate(e_rows)
    }
    nlp_feats = {
        "node_id_to_idx": {r["node_id"]: i for i, r in enumerate(n_rows)},
        "node_embeddings": torch.zeros((len(n_rows), 768), dtype=torch.float32),
        "edge_stance": e_stance,
        "edge_hostility": {e["edge_id"]: 0.1 for e in e_rows},
    }
    ds_k10 = EarlyCascadeDataset(
        k_horizon=10,
        split="all",
        nodes_table=build_nodes_table(n_rows),
        edges_table=build_edges_table(e_rows),
        nlp_features=nlp_feats,
        min_total_replies=30,
        polar_target_mode="future_only",
        polar_future_start_k=20,
    )
    ds_k20 = EarlyCascadeDataset(
        k_horizon=20,
        split="all",
        nodes_table=build_nodes_table(n_rows),
        edges_table=build_edges_table(e_rows),
        nlp_features=nlp_feats,
        min_total_replies=30,
        polar_target_mode="future_only",
        polar_future_start_k=20,
    )
    assert ds_k10[0].num_nodes == 11 and ds_k20[0].num_nodes == 21
    assert float(ds_k10[0].y_polar.item()) == pytest.approx(float(ds_k20[0].y_polar.item()), abs=1e-6)


def test_b10_sigmoid_camp_readout_ablations_biedgegatv2_textmlp_mask_and_assortativity() -> None:
    """B10 & M5/M7: Ablations, BiEdgeGATv2, TextMLP topology mask, neutral parent attenuation, assortativity."""
    # Neutral parent attenuation (u_ori = 0.0 -> child aligned = 0.0, NOT +s_uv)
    children_adj = {"r": ["n1"], "n1": ["n2"], "n2": []}
    edge_lookup = {
        ("r", "n1"): {"edge_id": "r->n1"},
        ("n1", "n2"): {"edge_id": "n1->n2"},
    }
    aligned = _compute_root_aligned_stances(
        "r", children_adj, edge_lookup, {"r->n1": 0.0, "n1->n2": -0.9}, neutral_eps=0.1
    )
    assert aligned[0] == pytest.approx(0.0)
    assert aligned[1] == pytest.approx(0.0)

    # Synthetic CascadeData for architecture forward/backward & mask verification
    num_nodes = 12
    src = [0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6]
    dst = list(range(1, num_nodes))
    ei_td = torch.tensor([src, dst], dtype=torch.long)
    ei_bu = ei_td[[1, 0], :]
    data = CascadeData(
        x=torch.randn((num_nodes, TOTAL_NODE_FEATURE_DIM)),
        struct_x=torch.rand((num_nodes, STRUCTURAL_NODE_DIM)),
        edge_index=ei_td,
        edge_index_td=ei_td,
        edge_index_bu=ei_bu,
        edge_attr=torch.tensor([[0.8 if i % 2 == 0 else -0.8, 0.2 if i % 2 == 0 else 0.8] for i in range(len(dst))]),
        edge_stance=torch.tensor([[0.8 if i % 2 == 0 else -0.8] for i in range(len(dst))]),
        early_node_root_stance=torch.tensor([1.0] + [0.8 if i % 2 == 0 else -0.8 for i in range(len(dst))]),
        cascade_stats=torch.tensor([[0.3, 0.2, 0.6, 0.8, 0.5, 0.9, 0.7, 0.4]], dtype=torch.float32),
        root_index=torch.tensor([0], dtype=torch.long),
        y_size=torch.tensor([3.5]),
        y_polar=torch.tensor([0.65]),
        y_bimodality=torch.tensor([0.72]),
        y_assortativity=torch.tensor([-0.45]),
        y_rumor=torch.tensor([1], dtype=torch.long),
    )

    for model in (
        SBiFAGAT(hidden_dim=32, gate_mode="sigmoid", use_cascade_stats=True),
        SBiFAGAT(hidden_dim=32, use_camp_dipole=False, use_cascade_stats=True),
        SBiFAGAT(hidden_dim=32, use_dir_mix=False, use_cascade_stats=True),
        SBiFAGAT(hidden_dim=32, use_dual_channel=False, use_cascade_stats=True),
        BiEdgeGATv2(hidden_dim=32, edge_dim=2, use_cascade_stats=True),
    ):
        out = model(data)
        loss = out["y_size_pred"].sum() + out["y_polar_pred"].sum() + out["rumor_logits"].sum()
        loss.backward()
        assert torch.isfinite(loss)

    # TextMLP strictly invariant to cascade_stats[:, [3, 4, 7]] (topology + semantic stats)
    mlp = TextMLP(hidden_dim=32, use_cascade_stats=True)
    mlp.eval()
    data_alt = data.clone()
    data_alt.cascade_stats = data.cascade_stats.clone()
    data_alt.cascade_stats[0, 3] = 0.0
    data_alt.cascade_stats[0, 4] = 1.0
    data_alt.cascade_stats[0, 7] = 0.99
    with torch.no_grad():
        out1 = mlp(data)
        out2 = mlp(data_alt)
    assert torch.allclose(out1["y_size_pred"], out2["y_size_pred"], atol=1e-6)
    assert torch.allclose(out1["y_polar_pred"], out2["y_polar_pred"], atol=1e-6)

    # Train-mean Early-K Persistence & construct_corr_assort
    persist = compute_early_k_persistence_metrics([data], train_cascades=[data])
    assert persist["msle"] == pytest.approx(0.0, abs=1e-6)
    eval_m = evaluate_predictions(
        np.array([3.0, 4.0, 5.0]),
        np.array([3.1, 3.9, 5.1]),
        np.array([0.1, 0.5, 0.9]),
        np.array([0.15, 0.45, 0.85]),
        y_bimodality_true=np.array([0.2, 0.55, 0.85]),
        y_assortativity_true=np.array([0.8, 0.0, -0.7]),
    )
    assert eval_m["construct_corr_bc"] > 0.9
    assert eval_m["construct_corr_assort"] < -0.9


def test_m5_m8_assemble_paper_artifacts_and_kaggle_runner_resilience(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M5/M8 & B5: Paper assembler merges EDA/efficiency & validates LaTeX; bundle skips caches & UPFD is nonfatal."""
    pdir = tmp_path / "paper"
    pdir.mkdir()
    (pdir / "results_dataset_eda_pheme.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "pheme",
                        "raw_cascades": 6000,
                        "eligible_cascades": 400,
                        "eligible_topics": 9,
                        "split_cascade_counts": {"train": 280, "val": 60, "test": 60},
                        "median_replies": 32.0,
                        "mean_replies": 45.0,
                        "median_duration_hours": 1.5,
                        "early_k_persistence": {"num_cascades": 60, "rmse": 0.25, "pearson_r": 0.40},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (pdir / "results_dataset_eda_vn.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "vn",
                        "raw_cascades": 1500,
                        "eligible_cascades": 1400,
                        "eligible_topics": 90,
                        "split_cascade_counts": {"train": 980, "val": 210, "test": 210},
                        "median_replies": 55.0,
                        "mean_replies": 78.0,
                        "median_duration_hours": 8.0,
                        "early_k_persistence": {"num_cascades": 210, "rmse": 0.22, "pearson_r": 0.45},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    # Real benchmark_efficiency.py JSON output is a top-level list of model rows
    (pdir / "results_efficiency_pheme.json").write_text(
        json.dumps(
            [
                {
                    "model": "SBiFAGAT",
                    "params": 120000,
                    "infer_ms_per_cascade": 0.42,
                    "train_ms_per_batch": 14.5,
                    "peak_mem_mb": 98.0,
                    "mem_label": "Peak VRAM (MB)",
                }
            ]
        ),
        encoding="utf-8",
    )
    (pdir / "results_human_audit_pairs.json").write_text(
        json.dumps(
            {
                "stance": {"num_samples": 300, "cohen_kappa_unweighted": 0.78, "cohen_kappa_quadratic": 0.84},
                "machine_vs_gold": {"stance": {"macro_f1": 0.81, "spearman_r": 0.79}},
            }
        ),
        encoding="utf-8",
    )
    (pdir / "results_human_audit_vn_50_trees.json").write_text(
        json.dumps(
            {
                "inter_annotator": {"num_threads": 50, "cohen_kappa_unweighted": 0.72, "cohen_kappa_quadratic": 0.83},
                "machine_vs_gold": {"spearman_r": 0.76},
            }
        ),
        encoding="utf-8",
    )
    manifest = assemble_paper_artifacts(pdir, strict_latex_check=True)
    assert (pdir / "tables_dataset_eda.tex").is_file()
    assert (pdir / "tables_efficiency.tex").is_file()
    assert (pdir / "tables_human_audit.tex").is_file()
    audit_tex = (pdir / "tables_human_audit.tex").read_text(encoding="utf-8")
    assert "Pair Stance" in audit_tex and "Thread $y_{\\text{polar}}$" in audit_tex
    assert manifest["latex_underscore_issues"] == {}
    assert "pheme" in (pdir / "tables_dataset_eda.tex").read_text(encoding="utf-8")
    assert "vn" in (pdir / "tables_dataset_eda.tex").read_text(encoding="utf-8")

    # Inject broken unescaped underscore and verify assembler catches it
    (pdir / "tables_bad.tex").write_text("SBiFAGAT vs s_uv=0 \\\\\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unescaped '_'"):
        assemble_paper_artifacts(pdir, strict_latex_check=True)

    # _bundle_outputs deduplication & exclusion of scored_pairs.json / *.jsonl / nlp_features.pt
    work = tmp_path / "bundle_work"
    (work / "data" / "interim" / "vn").mkdir(parents=True)
    (work / "data" / "processed" / "vn").mkdir(parents=True)
    (work / "data" / "interim" / "vn" / "scored_pairs.json").write_text("[]", encoding="utf-8")
    (work / "data" / "interim" / "vn" / "crawl_checkpoint.jsonl").write_text("{}\n", encoding="utf-8")
    (work / "data" / "interim" / "vn" / "human_audit_vi_300.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (work / "data" / "processed" / "vn" / "nlp_features.pt").write_bytes(b"ptdata")
    zpath = kpr._bundle_outputs(
        work / "compact.zip",
        source_paths=[work / "data" / "interim", work / "data" / "interim" / "vn", work / "data" / "processed" / "vn"],
        base_dir=work,
        include_heavy_weights=False,
        include_nlp_features=False,
    )
    with zipfile.ZipFile(zpath, "r") as zf:
        names = zf.namelist()
    assert len(names) == len(set(names))  # Zero duplicate zip entries
    basenames = {Path(n).name for n in names}
    assert "human_audit_vi_300.csv" in basenames
    assert "scored_pairs.json" not in basenames
    assert "crawl_checkpoint.jsonl" not in basenames
    assert "nlp_features.pt" not in basenames

    # Stage skip-if-exists & nonfatal UPFD failure in run_kaggle_parallel_role
    w = tmp_path / "resilient_en"
    interim_en = w / "data" / "interim" / "pheme"
    proc_en = w / "data" / "processed" / "pheme"
    interim_en.mkdir(parents=True)
    proc_en.mkdir(parents=True)
    (interim_en / "nodes.parquet").write_bytes(b"ok")
    (interim_en / "edges.parquet").write_bytes(b"ok")
    (proc_en / "nodes.parquet").write_bytes(b"ok")
    (proc_en / "edges.parquet").write_bytes(b"ok")
    torch.save({"teacher_edge_stance": {"e1": 0.5}}, proc_en / "nlp_features.pt")

    called: list[list[str]] = []

    def _fake_cmd(cmd: list[str], *, cwd: Path) -> None:
        called.append(list(cmd))
        if "scripts.run_upfd_extension" in cmd:
            raise RuntimeError("Simulated GDrive rate limit")

    fake_graph = CascadeData(
        x=torch.zeros((2, TOTAL_NODE_FEATURE_DIM)),
        edge_index=torch.tensor([[0], [1]]),
        edge_index_td=torch.tensor([[0], [1]]),
        edge_index_bu=torch.tensor([[1], [0]]),
        edge_attr=torch.tensor([[0.5, 0.1]]),
    )
    monkeypatch.setattr(kpr, "_run_cmd", _fake_cmd)
    monkeypatch.setattr(kpr, "EarlyCascadeDataset", lambda **_kw: [fake_graph])

    args = SimpleNamespace(
        role="member2_en",
        lang="en",
        smoke_test=False,
        work_dir=w,
        processed_dir=None,
        vn_target_count=1500,
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
        run_upfd=True,
        bundle_weights=False,
        force=False,
    )
    res = kpr.run_kaggle_parallel_role(args)
    assert not any("src.data.pheme_loader" in c for c in called)  # Stage 1 skipped
    assert not any("src.nlp.extract_features" in c for c in called)  # Stage 3 skipped
    assert any("--eval-protocol" in c and "loo" in c for c in called)  # LOO ran with teacher_edge_stance
    assert Path(res["bundle_zip"]).exists()  # Primary bundle preserved despite UPFD failure
    assert "failed_nonfatal" in res["upfd_status"]


def test_round2_deep_chain_bfs_no_gradient_explosion_and_deberta_amp_policy() -> None:
    """Round-2 P0: detached BFS parent orientation prevents 10^d gradient explosion on depth-30 chains."""
    from src.models.sbi_fagat import SBiFAGAT, _propagate_root_aligned_edge_signal
    from src.nlp.dual_edge_scorer import _AutoCastLogitsWrapper, _resolve_teacher_amp_policy

    depth = 30
    src = list(range(depth))
    dst = list(range(1, depth + 1))
    ei_td = torch.tensor([src, dst], dtype=torch.long)
    # Neutral root edge (0.0) followed by strong +-0.9 edges (worst case for 1/0.1 = 10x slope)
    edge_sig = torch.tensor([0.0] + [0.9] * (depth - 1), dtype=torch.float32, requires_grad=True)
    aligned = _propagate_root_aligned_edge_signal(
        ei_td, edge_sig, root_idx=torch.tensor([0], dtype=torch.long), num_nodes=depth + 1
    )
    aligned.sum().backward()
    assert edge_sig.grad is not None
    assert torch.isfinite(edge_sig.grad).all()
    assert float(edge_sig.grad.abs().max().item()) <= 1.0 + 1e-6

    # Also test full SBiFAGAT forward + backward on a depth-30 chain with zero stance
    data = CascadeData(
        x=torch.randn((depth + 1, TOTAL_NODE_FEATURE_DIM)),
        edge_index=ei_td,
        edge_index_td=ei_td,
        edge_index_bu=ei_td[[1, 0]],
        edge_attr=torch.zeros((depth, 2), dtype=torch.float32),
        root_index=torch.tensor([0], dtype=torch.long),
        cascade_stats=torch.zeros((1, 8), dtype=torch.float32),
        y_size=torch.tensor([3.5]),
        y_polar=torch.tensor([0.5]),
        y_rumor=torch.tensor([0], dtype=torch.long),
    )
    model = SBiFAGAT(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2, use_cascade_stats=True)
    out = model(data)
    loss = out["y_size_pred"].sum() + out["y_polar_pred"].sum()
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            assert float(p.grad.abs().max().item()) < 1e4

    # _AutoCastLogitsWrapper returns float32 logits cleanly on CPU
    class _MiniSeqCls(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="deberta-v2")
            self.lin = torch.nn.Linear(4, 3)

        def forward(self, x: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(logits=self.lin(x))

    inner = _MiniSeqCls()
    wrapped = _AutoCastLogitsWrapper(inner, use_amp=False, amp_dtype=torch.float32)
    out_logits = wrapped(x=torch.randn(2, 4))
    assert out_logits.shape == (2, 3) and out_logits.dtype == torch.float32
    use_amp_cpu, dt_cpu = _resolve_teacher_amp_policy(inner, torch.device("cpu"))
    assert use_amp_cpu is False and dt_cpu == torch.float32


def test_round2_bootstrap_p001_achievable_upfd_comparator_and_exact_perm_label() -> None:
    """Round-2 P0/P1: p_adj < 0.01 achievable under Holm-Bonferroni, UPFD classification_only comparator clean."""
    from src.training.reporting import export_latex_table
    from src.training.trainer import (
        _compute_sample_level_bootstrap_pvalues,
        holm_bonferroni_adjust,
        select_best_comparator,
    )

    # 1. Topic-cluster bootstrap with 8 topics & clear separation achieves Holm-Bonf p < 0.01
    n_topics = 8
    n_per = 10
    n_tot = n_topics * n_per
    rng = np.random.default_rng(7)
    y_size = rng.uniform(3.0, 5.0, size=n_tot)
    y_polar = rng.uniform(0.1, 0.9, size=n_tot)
    y_rumor = np.array([i % 2 for i in range(n_tot)], dtype=np.int64)
    topic_ids = [f"topic_{i // n_per}" for i in range(n_tot)]

    # Strong separation: ref is near-exact, cmp has large systematic error
    ref_logits = np.stack([np.where(y_rumor == 0, 4.0, -4.0), np.where(y_rumor == 1, 4.0, -4.0)], axis=1)
    cmp_logits = np.zeros_like(ref_logits)
    preds_ref = [
        {
            "y_size_true": y_size,
            "y_size_pred": y_size + 0.01,
            "y_polar_true": y_polar,
            "y_polar_pred": np.clip(y_polar + 0.01, 0.0, 1.0),
            "y_rumor_true": y_rumor,
            "rumor_logits": ref_logits,
            "topic_ids": topic_ids,
        }
    ]
    preds_cmp = [
        {
            "y_size_true": y_size,
            "y_size_pred": y_size + 1.5,
            "y_polar_true": y_polar,
            "y_polar_pred": np.full_like(y_polar, 0.5),
            "y_rumor_true": y_rumor,
            "rumor_logits": cmp_logits,
            "topic_ids": topic_ids,
        }
    ]
    keys = ("msle", "mae", "r2_size", "rmse", "pearson_r", "spearman_polar", "macro_f1", "accuracy", "auc_roc")
    pvals = _compute_sample_level_bootstrap_pvalues(
        preds_ref, preds_cmp, metric_keys=keys, n_resamples=2500, n_stat_resamples=2000
    )
    adj = holm_bonferroni_adjust({k: pvals[k] for k in keys})
    assert adj["msle"] < 0.01
    assert adj["rmse"] < 0.01
    assert adj["accuracy"] < 0.01
    assert adj["macro_f1"] < 0.01

    # 2. classification_only empty y_size_true still runs classification bootstrap!
    preds_cls_ref = [
        {
            "y_size_true": np.array([]),
            "y_size_pred": np.array([]),
            "y_polar_true": np.array([]),
            "y_polar_pred": np.array([]),
            "y_rumor_true": y_rumor,
            "rumor_logits": ref_logits,
            "topic_ids": None,
        }
    ]
    preds_cls_cmp = [
        {
            "y_size_true": np.array([]),
            "y_size_pred": np.array([]),
            "y_polar_true": np.array([]),
            "y_polar_pred": np.array([]),
            "y_rumor_true": y_rumor,
            "rumor_logits": cmp_logits,
            "topic_ids": None,
        }
    ]
    cls_pvals = _compute_sample_level_bootstrap_pvalues(
        preds_cls_ref,
        preds_cls_cmp,
        metric_keys=("accuracy", "macro_f1", "auc_roc"),
        n_resamples=1000,
        n_stat_resamples=500,
    )
    assert math.isfinite(cls_pvals["accuracy"]) and cls_pvals["accuracy"] < 0.01
    assert math.isfinite(cls_pvals["macro_f1"]) and cls_pvals["macro_f1"] < 0.01

    # 3. select_best_comparator respects metric_keys (ignores untrained regression metrics on UPFD)
    summary_upfd = {
        "SBiFAGAT": {"msle": (5.0, 0.0), "accuracy": (0.90, 0.01), "macro_f1": (0.89, 0.01)},
        "BadClsGoodReg": {"msle": (0.01, 0.0), "accuracy": (0.50, 0.01), "macro_f1": (0.48, 0.01)},
        "StrongCls": {"msle": (9.0, 0.0), "accuracy": (0.85, 0.01), "macro_f1": (0.84, 0.01)},
    }
    assert (
        select_best_comparator(
            summary_upfd,
            reference_model="SBiFAGAT",
            metric_keys=("accuracy", "macro_f1", "auc_roc"),
        )
        == "StrongCls"
    )

    # 4. Exact Perm. label renders properly in LaTeX reporting
    tex_out = export_latex_table(
        {
            "summary_stats": {
                "SBiFAGAT": {"msle": (0.10, 0.01)},
                "BiGCN": {"msle": (0.20, 0.02)},
            },
            "p_values": {"msle": 0.04},
            "bootstrap_p_values": {"msle": 0.125},
            "adjusted_bootstrap_p_values": {"msle": 0.125},
            "reference_model": "SBiFAGAT",
            "comparator_model": "BiGCN",
            "nonparametric_test_label": "Exact Perm.",
        },
        Path("outputs/_tmp_exact_perm_test.tex"),
        caption="Test",
        label="tab:test_perm",
        metrics=(("msle", "MSLE", True),),
    )
    try:
        txt = tex_out.read_text(encoding="utf-8")
        assert "Exact Perm. $p$" in txt and "Holm-Bonf Exact Perm. $p$" in txt
    finally:
        tex_out.unlink(missing_ok=True)


def test_round2_dataset_guard_crawler_resilience_and_50tree_window(tmp_path: Path, monkeypatch: Any) -> None:
    """Round-2 P0/P1: min_total_replies_extracted guard, truncated JSONL & transient error recovery, 50-tree window."""
    import scripts.audit_vn_50_trees as a50
    import scripts.export_dataset_stats as eds
    import src.data.arctic_shift_vn as asvn

    # 1. EarlyCascadeDataset rejects min_total_replies < min_total_replies_extracted
    d = tmp_path / "min_guard"
    all_nodes: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    node_embs: dict[str, torch.Tensor] = {}
    edge_st: dict[str, float] = {}
    edge_ho: dict[str, float] = {}
    for t_i in range(6):
        ns, es = _make_cascade_rows(f"mg_c{t_i}", f"mg_t{t_i}", 30, lang="vi", rumor_label=None)
        all_nodes.extend(ns)
        all_edges.extend(es)
        for n in ns:
            node_embs[str(n["node_id"])] = torch.zeros((768,), dtype=torch.float32)
        for e in es:
            edge_st[str(e["edge_id"])] = 0.6 if (hash(e["edge_id"]) % 2 == 0) else -0.6
            edge_ho[str(e["edge_id"])] = 0.2
    save_cascade_parquet(build_nodes_table(all_nodes), build_edges_table(all_edges), output_dir=d, validate=True)
    torch.save(
        {
            "node_embeddings": node_embs,
            "edge_stance": edge_st,
            "edge_hostility": edge_ho,
            "min_total_replies_extracted": 25,
        },
        d / "nlp_features.pt",
    )
    with pytest.raises(ValueError, match="min_total_replies_extracted=25"):
        EarlyCascadeDataset(root_dir=d, k_horizon=20, split="all", min_total_replies=0)

    # 2. 50-tree VN audit strictly samples replies[k_horizon:] when future_only and attaches parent tags
    out_csv, _ = a50.export_vn_50_tree_audit(
        data_dir=d,
        output_csv=tmp_path / "audit_50.csv",
        n_samples=5,
        k_horizon=20,
        min_total_replies=25,
        polar_target_mode="future_only",
    )
    with out_csv.open("r", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 1
    assert rows[0]["target_window"].startswith("future_only (replies #21")
    assert int(rows[0]["num_target_replies"]) == 10
    assert "-> Root" in rows[0]["sample_replies"] or "-> Parent(" in rows[0]["sample_replies"]

    # 3. export_dataset_stats handles None persistence metrics cleanly (null from JSON)
    latex_null = eds.format_dataset_stats_latex(
        [
            {
                "name": "vn",
                "raw_cascades": 100,
                "eligible_cascades": 80,
                "eligible_topics": 10,
                "split_cascade_counts": {"train": 56, "val": 12, "test": 12},
                "median_replies": 30.0,
                "mean_replies": 40.0,
                "median_duration_hours": 3.5,
                "early_k_persistence": {"num_cascades": 12, "rmse": None, "pearson_r": None},
            }
        ]
    )
    assert "-- & --" in latex_null

    # 4. Arctic Shift crawler tolerates truncated JSONL line & never poisons exhausted=True on transient error
    ckpt = tmp_path / "crawl_checkpoint.jsonl"
    ckpt.write_text('{"post_id": "trunc_broken\n', encoding="utf-8")

    def _fail_search(*_a: Any, **_kw: Any) -> dict[str, Any]:
        raise RuntimeError("HTTP 502 Bad Gateway")

    monkeypatch.setattr(asvn, "_request_with_retry", _fail_search)
    with pytest.raises(RuntimeError, match="transient network errors|Only collected"):
        asvn.crawl_arctic_shift_discussions(
            output_dir=tmp_path / "vn_err",
            subreddits=("vozforums",),
            target_count=50,
            checkpoint_jsonl=ckpt,
            allow_partial=True,
        )
    state_file = ckpt.with_name(f"{ckpt.name}.state.json")
    assert state_file.exists()
    state = json.loads(state_file.read_text(encoding="utf-8"))
    # Must NOT poison exhausted["vozforums"] = True on transient 502!
    assert state["exhausted"]["vozforums"] is False


"""Pha 2 hardening regression tests: SSRF/path guards, NLP backend tiers, VN quantiles,
audit overwrite guards, pure-stdlib LaTeX verifier, and CI/script hygiene.
"""

from __future__ import annotations

import builtins
import csv
import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts import audit_vn_50_trees, predict_live_thread as plt_mod
from scripts.assemble_paper_artifacts import (
    assemble_paper_artifacts,
    find_unescaped_underscores_in_latex,
)
from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet
from src.nlp import dual_edge_scorer as des
from src.nlp.dual_edge_scorer import DualHeadCrossEncoder
from tests.test_step3_training_and_audit_fixes import _build_offline_tok_and_cls_model


def test_extract_post_id_from_url_rejects_traversal_and_accepts_valid_ids() -> None:
    """Rejects path traversal, slashes, underscores, and short IDs while accepting valid Reddit IDs."""
    for bad in (
        "../abc",
        "../abcd",
        "abc/def",
        "ab",
        "",
        "https://www.reddit.com/r/vozforums/comments/../etc",
        "https://www.reddit.com/r/vozforums/comments/ab/",
        "https://www.reddit.com/r/vozforums/comments/abc_123/title",
        "https://www.reddit.com/r/vozforums/comments/toolong1234567/title",
        "https://evil.com/path/to/1abc2d",
    ):
        with pytest.raises(ValueError):
            plt_mod._extract_post_id_from_url(bad)

    assert plt_mod._extract_post_id_from_url("1abc2d") == "1abc2d"
    assert plt_mod._extract_post_id_from_url("t3_1abc2d") == "1abc2d"
    assert (
        plt_mod._extract_post_id_from_url("https://www.reddit.com/r/vozforums/comments/1abc2d/some_title/")
        == "1abc2d"
    )
    assert plt_mod._extract_post_id_from_url("https://www.reddit.com/comments/abc123") == "abc123"


def test_vn_empirical_regime_thresholds_calibrated_and_ordered() -> None:
    """VN regime quantiles must be strictly ordered and well below legacy uncalibrated 0.55."""
    assert 0.0 < plt_mod.VN_POLAR_Q50 < plt_mod.VN_POLAR_Q75 < plt_mod.VN_POLAR_Q90 < 0.55
    assert plt_mod.VN_POLAR_Q50 == pytest.approx(0.092762, abs=1e-4)
    assert plt_mod.VN_POLAR_Q75 == pytest.approx(0.120198, abs=1e-4)
    assert plt_mod.VN_POLAR_Q90 == pytest.approx(0.150351, abs=1e-4)

    empty_counts = {
        "support": 0,
        "neutral": 5,
        "constructive_oppose": 0,
        "irrational_fallacy": 0,
        "hostile_conflict": 0,
    }
    heavy_counts = {
        "support": 0,
        "neutral": 0,
        "constructive_oppose": 6,
        "hostile_conflict": 6,
    }
    assert "< Q50" in plt_mod._classify_calibrated_regime(plt_mod.VN_POLAR_Q50 - 0.01, empty_counts)
    assert "< Q50" in plt_mod._classify_calibrated_regime(plt_mod.VN_POLAR_Q50 - 0.01, heavy_counts)
    assert "Q50-Q75" in plt_mod._classify_calibrated_regime(plt_mod.VN_POLAR_Q50 + 0.005, empty_counts)
    assert "Q50-Q75" in plt_mod._classify_calibrated_regime(plt_mod.VN_POLAR_Q50 + 0.005, heavy_counts)
    assert "Q75-Q90" in plt_mod._classify_calibrated_regime(plt_mod.VN_POLAR_Q75 + 0.005, empty_counts)
    assert ">= Q90" in plt_mod._classify_calibrated_regime(plt_mod.VN_POLAR_Q90 + 0.005, empty_counts)


def test_predict_live_cascade_nlp_backends_and_contract_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify 3-tier NLP backend reporting and JSON-serializable kept_nodes/kept_edges under contract mode."""
    base_ts = 1763200000.0
    post = {
        "id": "1abc2d",
        "title": "Thảo luận đề xuất thu phí xe cá nhân vào giờ cao điểm",
        "selftext": "Mọi người thấy phương án này hợp lý không?",
        "subreddit": "vozforums",
        "author": "op_user",
        "created_utc": base_ts,
    }
    comments = [
        {
            "fullname": "t1_c1",
            "id": "c1",
            "link_id": "t3_1abc2d",
            "parent_id": "t3_1abc2d",
            "text": "hóng",
            "created_utc": base_ts + 30.0,
            "author_hash": "a" * 64,
        },
        {
            "fullname": "t1_c2",
            "id": "c2",
            "link_id": "t3_1abc2d",
            "parent_id": "t1_c1",
            "text": "Tôi phản đối vì chưa có xe buýt thay thế đủ dày.",
            "created_utc": base_ts + 60.0,
            "author_hash": "b" * 64,
        },
        {
            "fullname": "t1_c3",
            "id": "c3",
            "link_id": "t3_1abc2d",
            "parent_id": "t3_1abc2d",
            "text": "Đồng ý",
            "created_utc": base_ts + 90.0,
            "author_hash": "c" * 64,
        },
    ]

    # 1. Incomplete CE directory (missing dual_head_weights.pt) -> cross_encoder_loaded is False
    incomplete_ce = tmp_path / "dual_ce_missing_weights"
    incomplete_ce.mkdir()
    (incomplete_ce / "config.json").write_text("{}", encoding="utf-8")
    (incomplete_ce / "dual_head_meta.json").write_text('{"lang":"vi"}', encoding="utf-8")

    # Force offline sandbox (no local ViSoBERT snapshot) -> heuristic_hash_sandbox
    monkeypatch.setattr(plt_mod, "_resolve_local_hf_snapshot", lambda _repo: None)
    monkeypatch.setattr(
        plt_mod.AutoModel,
        "from_pretrained",
        staticmethod(lambda *_a, **_kw: (_ for _ in ()).throw(OSError("offline"))),
    )
    res_sandbox = plt_mod.predict_live_cascade(
        post,
        comments,
        cross_encoder_ckpt=incomplete_ce,
        junk_mode="contract",
        allow_untrained=True,
    )
    assert res_sandbox["cross_encoder_loaded"] is False
    assert res_sandbox["nlp_backend"] == "heuristic_hash_sandbox"
    assert res_sandbox["junk_mode"] == "contract"
    assert res_sandbox["junk_filter_stats"]["contracted_nodes"] == 1
    assert res_sandbox["junk_filter_stats"]["relinked_edges"] == 1
    assert res_sandbox["k_observed"] == 2
    assert len(res_sandbox["kept_nodes"]) == 3
    assert len(res_sandbox["kept_edges"]) == 2
    for kn in res_sandbox["kept_nodes"]:
        assert set(kn.keys()) == {"node_id", "parent_id", "author_hash", "depth", "time_delay_sec", "text"}
    relinked_edges = [ke for ke in res_sandbox["kept_edges"] if ke["relinked"]]
    assert len(relinked_edges) == 1
    assert relinked_edges[0]["target_id"] == "vn_1abc2d_t1_c2"
    assert relinked_edges[0]["stance_class"] in {"support", "oppose", "neutral"}
    json.dumps(res_sandbox, ensure_ascii=False)  # Strictly JSON serializable

    # 2. If real checkpoints/dual_ce_vn exists without dual_head_weights.pt, must also stay False
    real_ce_vn = Path("checkpoints/dual_ce_vn")
    if real_ce_vn.is_dir() and not (real_ce_vn / "dual_head_weights.pt").is_file():
        res_real_ce = plt_mod.predict_live_cascade(
            post, comments, cross_encoder_ckpt=real_ce_vn, allow_untrained=True
        )
        assert res_real_ce["cross_encoder_loaded"] is False
        assert res_real_ce["nlp_backend"] != "full_2d_cross_encoder"

    # 3. ViSoBERT SpanPool available without CE weights -> visobert_spanpool_heuristic_edges
    tok, _, enc = _build_offline_tok_and_cls_model(max_pos=128, hidden_size=768, num_heads=12)
    monkeypatch.setattr(
        plt_mod,
        "_load_local_model_from_pretrained",
        lambda *_a, **_kw: enc,
    )
    monkeypatch.setattr(
        plt_mod,
        "_load_local_tokenizer",
        lambda *_a, **_kw: tok,
    )
    res_spanpool = plt_mod.predict_live_cascade(
        post, comments, cross_encoder_ckpt=incomplete_ce, allow_untrained=True
    )
    assert res_spanpool["cross_encoder_loaded"] is False
    assert res_spanpool["nlp_backend"] == "visobert_spanpool_heuristic_edges"

    # 4. Complete 3-file CE checkpoint + SpanPool -> full_2d_cross_encoder
    full_ce_dir = tmp_path / "full_ce"
    ce = DualHeadCrossEncoder(lang="vi", model_name="offline-ce", encoder=enc, edge_dim=2)
    ce.save_checkpoint(full_ce_dir, tokenizer=tok)
    res_full = plt_mod.predict_live_cascade(
        post, comments, cross_encoder_ckpt=full_ce_dir, allow_untrained=True
    )
    assert res_full["cross_encoder_loaded"] is True
    assert res_full["nlp_backend"] == "full_2d_cross_encoder"


def test_audit_overwrite_guards_without_machine_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protect existing audit_csv when machine_key_csv is absent unless --force-export-audit is set."""
    # Stage tiny valid Parquet + nlp_features + topic_splits for both scripts
    interim_dir = tmp_path / "interim"
    proc_dir = tmp_path / "proc"
    base_ts = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    embs: dict[str, torch.Tensor] = {}
    s_map: dict[str, float] = {}
    h_map: dict[str, float] = {}

    for t_i, sp_tag in enumerate(("tr", "va", "te")):
        cid = f"c_{t_i}"
        tid = f"top_{sp_tag}"
        rid = f"{cid}_r"
        nodes.append(
            {
                "node_id": rid,
                "cascade_id": cid,
                "topic_id": tid,
                "parent_id": None,
                "is_root": True,
                "text": "Bài gốc thảo luận về chính sách giao thông.",
                "created_at_utc": base_ts + timedelta(hours=t_i),
                "lang": "vi",
                "source_dataset": "arctic_shift_vozforums",
                "user_id_hash": "a" * 64,
                "rumor_label": None,
            }
        )
        embs[rid] = torch.zeros(768, dtype=torch.float32)
        for k in range(1, 27):
            nid = f"{cid}_n{k}"
            nodes.append(
                {
                    "node_id": nid,
                    "cascade_id": cid,
                    "topic_id": tid,
                    "parent_id": rid,
                    "is_root": False,
                    "text": f"Bình luận thảo luận số {k} về chính sách giao thông.",
                    "created_at_utc": base_ts + timedelta(hours=t_i, seconds=10 * k),
                    "lang": "vi",
                    "source_dataset": "arctic_shift_vozforums",
                    "user_id_hash": f"{k:064x}",
                    "rumor_label": None,
                }
            )
            embs[nid] = torch.zeros(768, dtype=torch.float32)
            eid = f"{rid}->{nid}"
            edges.append(
                {
                    "edge_id": eid,
                    "cascade_id": cid,
                    "source_id": rid,
                    "target_id": nid,
                    "created_at_utc": base_ts + timedelta(hours=t_i, seconds=10 * k),
                    "time_delay_sec": float(10 * k),
                    "edge_type": "reply",
                }
            )
            s_map[eid] = 0.7 if k % 2 == 0 else -0.7
            h_map[eid] = 0.1

    n_tbl = build_nodes_table(nodes)
    e_tbl = build_edges_table(edges)
    save_cascade_parquet(n_tbl, e_tbl, interim_dir, validate=True)
    save_cascade_parquet(n_tbl, e_tbl, proc_dir, validate=True)
    torch.save(
        {"node_embeddings": embs, "edge_stance": s_map, "edge_hostility": h_map},
        proc_dir / "nlp_features.pt",
    )
    (proc_dir / "topic_splits.json").write_text(
        json.dumps({"train": ["top_tr"], "val": ["top_va"], "test": ["top_te"]}),
        encoding="utf-8",
    )

    # 1. dual_edge_scorer.py overwrite guard when machine_key_csv does not exist
    pair_audit = tmp_path / "human_audit_vi_120.csv"
    pair_key = tmp_path / "human_audit_vi_120_machine_key.csv"
    sentinel_pair = "sample_id,gold_adjudicated_stance\ns1,support\n"
    pair_audit.write_text(sentinel_pair, encoding="utf-8")
    assert not pair_key.exists()

    monkeypatch.setattr(
        des,
        "_score_pairs_with_teachers",
        lambda pairs, **_kw: [{**p, "s_uv": 0.6, "h_uv": 0.1, "root_s_uv": 0.6} for p in pairs],
    )
    base_des_argv = [
        "dual_edge_scorer.py",
        "--lang",
        "vi",
        "--interim-dir",
        str(interim_dir),
        "--audit-csv",
        str(pair_audit),
        "--machine-key-csv",
        str(pair_key),
        "--topic-splits-json",
        str(tmp_path / "des_splits.json"),
        "--audit-n-samples",
        "6",
        "--export-audit",
    ]

    monkeypatch.setattr(sys, "argv", [*base_des_argv, "--skip-if-exists"])
    des.main()
    assert pair_audit.read_text(encoding="utf-8") == sentinel_pair
    assert not pair_key.exists()

    monkeypatch.setattr(sys, "argv", base_des_argv)
    with pytest.raises(FileExistsError):
        des.main()
    assert pair_audit.read_text(encoding="utf-8") == sentinel_pair

    monkeypatch.setattr(sys, "argv", [*base_des_argv, "--force-export-audit"])
    des.main()
    assert pair_audit.read_text(encoding="utf-8") != sentinel_pair
    assert pair_key.exists()

    # 2. audit_vn_50_trees.py overwrite guard when machine_key_csv does not exist
    tree_audit = tmp_path / "human_audit_vn_30_trees.csv"
    tree_key = tmp_path / "human_audit_vn_30_trees_machine_key.csv"
    sentinel_tree = "sample_id,gold_adjudicated_polar_grade\ntree_001,3\n"
    tree_audit.write_text(sentinel_tree, encoding="utf-8")
    assert not tree_key.exists()

    base_tree_argv = [
        "audit_vn_50_trees.py",
        "--data-dir",
        str(proc_dir),
        "--audit-csv",
        str(tree_audit),
        "--machine-key-csv",
        str(tree_key),
        "--n-samples",
        "1",
        "--export",
    ]

    monkeypatch.setattr(sys, "argv", [*base_tree_argv, "--skip-if-exists"])
    audit_vn_50_trees.main()
    assert tree_audit.read_text(encoding="utf-8") == sentinel_tree
    assert not tree_key.exists()

    monkeypatch.setattr(sys, "argv", base_tree_argv)
    with pytest.raises(FileExistsError):
        audit_vn_50_trees.main()
    assert tree_audit.read_text(encoding="utf-8") == sentinel_tree

    monkeypatch.setattr(sys, "argv", [*base_tree_argv, "--force-export-audit"])
    audit_vn_50_trees.main()
    assert tree_audit.read_text(encoding="utf-8") != sentinel_tree
    assert tree_key.exists()
    with tree_audit.open("r", encoding="utf-8-sig") as fh:
        assert len(list(csv.DictReader(fh))) == 1


def test_assemble_paper_artifacts_pure_stdlib_import_and_latex_checker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Import assemble_paper_artifacts without torch/scipy/sklearn and test multi-line LaTeX rules."""
    # Temporarily block heavy third-party imports in sys.modules + builtins.__import__
    blocked = ("torch", "scipy", "sklearn", "numpy")
    orig_import = builtins.__import__
    saved_modules: dict[str, Any] = {}
    for mod_name in list(sys.modules):
        if mod_name == "scripts.assemble_paper_artifacts" or any(
            mod_name == b or mod_name.startswith(f"{b}.") for b in blocked
        ):
            saved_modules[mod_name] = sys.modules.pop(mod_name)

    def _guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        root_pkg = name.split(".", 1)[0]
        if root_pkg in blocked:
            raise ModuleNotFoundError(f"Blocked heavy dependency in stdlib test: {name}")
        return orig_import(name, *args, **kwargs)

    try:
        monkeypatch.setattr(builtins, "__import__", _guarded_import)
        pure_mod = importlib.import_module("scripts.assemble_paper_artifacts")
        assert pure_mod.find_unescaped_underscores_in_latex("clean prose $a_b$") == []
    finally:
        monkeypatch.undo()
        sys.modules.update(saved_modules)

    # Multi-line math environments, macros, comments, and positive violation detection
    clean_tex = "\n".join(
        [
            "% comment with unescaped_underscore must be ignored",
            r"Line break before comment \\ % another raw_underscore",
            r"\IfFileExists{tables_dataset_eda_vn.tex}{\input{tables_dataset_eda_vn}}{}",
            r"\includegraphics[width=0.5\textwidth]{figures/vn/fig_a_b.pdf}",
            r"\include{chap_1_intro}",
            r"\bibliography{my_refs}",
            r"\label{tab:main_vn} \ref{fig:umap_vn} \eqref{eq:loss_1} \pageref{sec:m_1}",
            r"\cite{ref_a} \citet{ref_b} \citep[p.~1]{ref_c} \url{https://x.com/a_b}",
            r"\href{https://x.com/a_b}{Click Here}",
            r"Inline math $y_{\text{polar}} + x_1$ and \(z_2 = a_b\).",
            r"\[",
            r"  a_b + c_d = 0",
            r"\]",
            r"\begin{multline}",
            r"  y_{\text{size}} = \ln(1 + N_{\text{total}}), \\",
            r"  y_{\text{polar}} = s_{\tilde{s}, \text{ddof}=1}",
            r"\end{multline}",
            r"\begin{align*}",
            r"  e_{uv} &= [s_{uv}, h_{uv}]^\top",
            r"\end{align*}",
            r"Escaped text\_token and \texttt{safe\_name} are fine.",
        ]
    )
    assert find_unescaped_underscores_in_latex(clean_tex) == []

    dirty_tex = "\n".join(
        [
            r"Prose with unescaped foo_bar here.",
            r"Inline code \texttt{foo_bar} must be caught.",
            r"\href{https://x.com/safe_url}{bad_link_text}",
            r"\IfFileExists{safe_file_a_b.tex}{bad_true_branch}{}",
            r"Escaped dollar \$100 and safe math $a_b$ is clean.",
            r"$\text{bad_inside_math}$",
            r"After line break \\ bad_after_break",
        ]
    )
    issues = find_unescaped_underscores_in_latex(dirty_tex)
    flagged_lines = [ln for ln, _ in issues]
    assert flagged_lines == [1, 2, 3, 4, 6, 7]

    # optional_figures manifest tracking in assemble_paper_artifacts
    pdir = tmp_path / "paper"
    (pdir / "figures" / "vn").mkdir(parents=True)
    (pdir / "figures" / "vn" / "fig5_macro_signed_network.pdf").write_bytes(b"%PDF-1.4\n")
    manifest = assemble_paper_artifacts(pdir, strict_latex_check=True)
    assert manifest["optional_figures_present"] == ["figures/vn/fig5_macro_signed_network.pdf"]
    assert manifest["optional_figures_missing"] == ["figures/pheme/fig5_macro_signed_network.pdf"]


def test_run_experiments_loo_edge_source_and_macro_wording(tmp_path: Path) -> None:
    """Verify LOO records edge_source='teacher' and macro network captions/help match implementation."""
    repo_root = Path(__file__).resolve().parents[1]
    macro_src = (repo_root / "scripts" / "run_macro_signed_network.py").read_text(encoding="utf-8")
    assert "degree-preserving" not in macro_src.lower()
    assert "unseen users have zeroed features" not in macro_src
    assert "uniform edge-sign permutations" in macro_src.lower()
    assert "user features are aggregated strictly from pre-cutoff Train timestamps" in macro_src

    exp_src = (repo_root / "scripts" / "run_experiments.py").read_text(encoding="utf-8")
    assert 'args.eval_protocol == "loo"' in exp_src and '"teacher"' in exp_src


def test_benchmark_efficiency_biedgegatv2_and_honest_e2e_profiler(tmp_path: Path) -> None:
    """M7 regression: BiEdgeGATv2 is profiled and include_e2e=True honestly reports missing DualCE weights."""
    from scripts.benchmark_efficiency import run_efficiency_benchmark

    missing_ce = tmp_path / "missing_ce"
    missing_ce.mkdir()
    (missing_ce / "config.json").write_text("{}", encoding="utf-8")
    (missing_ce / "dual_head_meta.json").write_text('{"lang":"vi"}', encoding="utf-8")

    out = run_efficiency_benchmark(
        num_cascades=2,
        k_horizon=10,
        hidden_dim=32,
        edge_dim=2,
        warmup=1,
        repeats=1,
        include_e2e=True,
        ce_ckpt=missing_ce,
        gnn_ckpt=tmp_path / "missing_gnn.pt",
        device=torch.device("cpu"),
    )
    assert isinstance(out, dict)
    models = {r["model"]: r for r in out["models"]}
    assert "BiEdgeGATv2" in models
    assert "SBiFAGAT" in models
    e2e = out["e2e_raw_text_profile"]
    assert e2e["num_nodes"] == 11
    assert e2e["cross_encoder_loaded"] is False
    assert e2e["cross_encoder_ms"] is None
    assert e2e["is_partial_without_dual_ce"] is True
    assert e2e["e2e_total_ms"] > 0.0
    assert e2e["graph_and_stats_ms"] > 0.0
    assert e2e["gnn_forward_ms"] > 0.0

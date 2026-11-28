"""Tier 3: Pairwise cross-feature integration & CLI subprocess E2E tests."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from scripts.assemble_paper_artifacts import find_unescaped_underscores_in_latex
from src.data.pyg_dataset import EarlyCascadeDataset, TOTAL_NODE_FEATURE_DIM
from src.models.sbi_fagat import SBiFAGAT
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    HostilityTeacherScorer,
    NLIStanceTeacherScorer,
    distill_dual_cross_encoder,
    export_stratified_human_audit_csv,
    finetune_visobert_vihsd,
)
from src.nlp.extract_features import extract_and_save_features
from src.training.trainer import train_and_evaluate_single_seed
from src.utils.seed import OFFICIAL_SEEDS
from tests.e2e.conftest import (
    REPO_ROOT,
    build_offline_encoder_and_classifiers,
    build_synthetic_processed_corpus,
)


def _run_cli(args: list[str], *, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


@pytest.mark.tier3
def test_t3_f1_f6_offline_nlp_finetune_distill_extract_to_dataset(tmp_path: Path) -> None:
    """F1 x F6: Offline ViHSD finetune -> Teacher -> Distill (2D) -> Extract -> EarlyCascadeDataset."""
    bundle = build_offline_encoder_and_classifiers(hidden_size=768, seed=101)
    tok = bundle["tokenizer"]

    # 1) Fine-tune hostility teacher
    vihsd_dir = tmp_path / "visobert_vihsd"
    ds_dict = {
        "train": {
            "free_text": ["agree good", "support yes", "oppose wrong", "oppose", "hate trash", "trash"],
            "label_id": [0, 0, 1, 1, 2, 2],
        },
        "validation": {"free_text": ["agree", "oppose", "trash"], "label_id": [0, 1, 2]},
    }
    finetune_visobert_vihsd(
        output_dir=vihsd_dir,
        epochs=1,
        batch_size=2,
        max_length=64,
        dataset_dict=ds_dict,
        model=bundle["hostility_model"],
        tokenizer=tok,
        device=torch.device("cpu"),
    )

    # 2) Build interim Parquet corpus & score with Teachers
    interim_dir = tmp_path / "interim"
    build_synthetic_processed_corpus(
        interim_dir,
        num_topics=4,
        cascades_per_topic=1,
        replies_per_cascade=26,
        min_total_replies_extracted=25,
    )
    host_scorer = HostilityTeacherScorer(lang="vi", model=bundle["hostility_model"], tokenizer=tok)
    nli_scorer = NLIStanceTeacherScorer(lang="vi", model=bundle["nli_model"], tokenizer=tok)

    # 3) Distill DualHeadCrossEncoder (edge_dim=2)
    ce_dir = tmp_path / "dual_ce"
    train_rec = [
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
        for i in range(6)
    ]
    distill_dual_cross_encoder(
        train_rec,
        train_rec[:2],
        output_dir=ce_dir,
        lang="vi",
        model=DualHeadCrossEncoder(lang="vi", model_name="offline-768", encoder=bundle["encoder"], edge_dim=2),
        tokenizer=tok,
        edge_dim=2,
        epochs=1,
        batch_size=3,
        max_length=64,
        use_multi_gpu=False,
        device=torch.device("cpu"),
    )

    # 4) Extract features & load EarlyCascadeDataset
    proc_dir = tmp_path / "processed"
    reloaded_ce, reloaded_tok = DualHeadCrossEncoder.load_checkpoint(ce_dir, map_location="cpu")
    feat_path = extract_and_save_features(
        dataset="vn",
        interim_dir=interim_dir,
        output_dir=proc_dir,
        cross_encoder=reloaded_ce,
        node_encoder=bundle["encoder"],
        tokenizer=reloaded_tok,
        hostility_scorer=host_scorer,
        stance_scorer=nli_scorer,
        edge_dim=2,
        k_horizon=20,
        min_total_replies=25,
        device=torch.device("cpu"),
    )
    assert feat_path.is_file()

    ds_train = EarlyCascadeDataset(root_dir=proc_dir, k_horizon=20, split="train", min_total_replies=25)
    ds_test = EarlyCascadeDataset(root_dir=proc_dir, k_horizon=20, split="test", min_total_replies=25)
    assert ds_train.polar_target_mode == "future_only"
    assert len(ds_train) >= 1 and len(ds_test) >= 1
    assert set(d.topic_id for d in ds_train).isdisjoint(set(d.topic_id for d in ds_test))
    assert ds_train[0].x.size(1) == TOTAL_NODE_FEATURE_DIM
    assert ds_train[0].edge_attr.size(1) == 2


@pytest.mark.tier3
def test_t3_f1_f4_human_audit_export_to_compute_kappa_cli(tmp_path: Path) -> None:
    """F1 x F4: Stratified human audit CSV export -> scripts.compute_kappa CLI strict vs single-gold."""
    scored_pairs = [
        {
            "cascade_id": f"cas_{i // 3}",
            "edge_id": f"e_{i}",
            "root_text": f"Root discussion {i // 3}",
            "parent_text": f"Parent comment {i}",
            "reply_text": f"Reply stance text {i}",
            "s_uv": 0.85 if i % 3 == 0 else (-0.80 if i % 3 == 1 else 0.05),
            "h_uv": 0.85 if i % 4 == 0 else 0.10,
            "pred_label": "support" if i % 3 == 0 else ("oppose" if i % 3 == 1 else "neutral"),
        }
        for i in range(18)
    ]
    audit_csv = tmp_path / "audit.csv"
    key_csv = tmp_path / "audit_key.csv"
    export_stratified_human_audit_csv(
        scored_pairs,
        output_csv=audit_csv,
        machine_key_csv=key_csv,
        n_samples=12,
        pilot_samples=0,
        seed=42,
    )

    with audit_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0].keys())
    for i, r in enumerate(rows):
        st1 = "support" if i % 3 == 0 else ("oppose" if i % 3 == 1 else "neutral")
        st2 = "oppose" if i == 0 else st1  # unresolved tie on row 0
        h1 = "uncivil" if i % 4 == 0 else "clean"
        r["annotator_1_stance"] = st1
        r["annotator_2_stance"] = st2
        r["annotator_1_hostility"] = h1
        r["annotator_2_hostility"] = h1
        if i > 0:
            r["gold_adjudicated_stance"] = st1
            r["gold_adjudicated_hostility"] = h1
    with audit_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    proc_strict = _run_cli(
        [
            "-m",
            "scripts.compute_kappa",
            "--input",
            str(audit_csv),
            "--machine-key",
            str(key_csv),
            "--strict-adjudication",
        ]
    )
    assert proc_strict.returncode != 0, "Expected --strict-adjudication to fail on unresolved disagreement"

    # Now populate row 0 single gold and run --allow-single-gold
    rows[0]["gold_adjudicated_stance"] = rows[0]["annotator_1_stance"]
    rows[0]["gold_adjudicated_hostility"] = rows[0]["annotator_1_hostility"]
    with audit_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    out_json = tmp_path / "kappa.json"
    out_tex = tmp_path / "kappa.tex"
    proc_ok = _run_cli(
        [
            "-m",
            "scripts.compute_kappa",
            "--input",
            str(audit_csv),
            "--machine-key",
            str(key_csv),
            "--allow-single-gold",
            "--output-json",
            str(out_json),
            "--output-table",
            str(out_tex),
        ]
    )
    assert proc_ok.returncode == 0, proc_ok.stderr
    parsed = json.loads(out_json.read_text(encoding="utf-8"))
    assert parsed["stance"]["num_samples"] == len(rows)
    assert find_unescaped_underscores_in_latex(out_tex.read_text(encoding="utf-8")) == []


@pytest.mark.tier3
def test_t3_f3_f4_f6_run_experiments_cli_e2e(tmp_path: Path) -> None:
    """F3 x F4 x F6: Subprocess CLI scripts.run_experiments executes 5-seed benchmark + outputs valid JSON/TeX."""
    ds_dir = tmp_path / "vn_exp"
    build_synthetic_processed_corpus(
        ds_dir, num_topics=9, cascades_per_topic=1, replies_per_cascade=27, min_total_replies_extracted=25
    )
    out_json = tmp_path / "exp_results.json"
    out_tex = tmp_path / "exp_table.tex"
    ckpt_out = tmp_path / "best.pt"

    proc = _run_cli(
        [
            "-m",
            "scripts.run_experiments",
            "--data-dir",
            str(ds_dir),
            "--k-horizon",
            "20",
            "--hidden-dim",
            "16",
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--min-total-replies",
            "25",
            "--polar-target-mode",
            "future_only",
            "--skip-figures",
            "--output-json",
            str(out_json),
            "--output-table",
            str(out_tex),
            "--ckpt-out",
            str(ckpt_out),
        ],
        timeout=240,
    )
    assert proc.returncode == 0, f"run_experiments failed:\n{proc.stdout}\n{proc.stderr}"
    assert out_json.is_file() and out_tex.is_file()

    def _reject_nan(tok: str) -> Any:
        raise ValueError(tok)

    data = json.loads(out_json.read_text(encoding="utf-8"), parse_constant=_reject_nan)
    assert tuple(data["seeds"]) == OFFICIAL_SEEDS
    summary = data["summary_stats"]
    for expected_model in (
        "SBiFAGAT",
        "BiEdgeGATv2",
        "Early-K Persistence",
        "Calibrated Early-K Persistence",
        "Ridge",
        "LightGBM",
        "FriedkinJohnsen",
        "CasCN",
        "CasFlow",
    ):
        assert expected_model in summary, (
            f"Missing {expected_model!r} in run_experiments summary_stats: {list(summary.keys())}"
        )
    assert find_unescaped_underscores_in_latex(out_tex.read_text(encoding="utf-8")) == []


@pytest.mark.tier3
def test_t3_f3_f4_run_ablation_cli_e2e(tmp_path: Path) -> None:
    """F3 x F4: Subprocess CLI scripts.run_ablation runs horizon & component ablations cleanly."""
    ds_dir = tmp_path / "vn_abl"
    build_synthetic_processed_corpus(
        ds_dir, num_topics=9, cascades_per_topic=1, replies_per_cascade=27, min_total_replies_extracted=25
    )
    out_json = tmp_path / "abl_results.json"
    out_tex = tmp_path / "abl_table.tex"

    proc = _run_cli(
        [
            "-m",
            "scripts.run_ablation",
            "--data-dir",
            str(ds_dir),
            "--horizons",
            "10",
            "20",
            "--hidden-dim",
            "16",
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--min-total-replies",
            "25",
            "--figures-dir",
            str(tmp_path / "abl_figs"),
            "--output-json",
            str(out_json),
            "--output-table",
            str(out_tex),
        ],
        timeout=240,
    )
    assert proc.returncode == 0, f"run_ablation failed:\n{proc.stdout}\n{proc.stderr}"
    assert out_json.is_file() and out_tex.is_file()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    abl_summary = data.get("summary_stats", {})
    assert "SBiFAGAT (Full)" in abl_summary
    assert any("w/o Stance Score" in k for k in abl_summary), f"Missing stance ablation in {list(abl_summary)}"
    assert any("w/o Hostility Score" in k for k in abl_summary), f"Missing hostility ablation in {list(abl_summary)}"
    k_curve = data.get("extra", {}).get("k_curve", data.get("k_curve", {}))
    assert "10" in k_curve and "20" in k_curve, f"Expected horizons 10 and 20 in k_curve, got {list(k_curve)}"
    assert find_unescaped_underscores_in_latex(out_tex.read_text(encoding="utf-8")) == []


@pytest.mark.tier3
def test_t3_f2_f4_run_macro_signed_network_cli_smoke_and_analysis(tmp_path: Path) -> None:
    """F2 x F4: CLI scripts.run_macro_signed_network --smoke-test and --analysis-only export valid artifacts."""
    proc_smoke = _run_cli(["-m", "scripts.run_macro_signed_network", "--smoke-test"], timeout=120)
    assert proc_smoke.returncode == 0, f"smoke-test failed:\n{proc_smoke.stdout}\n{proc_smoke.stderr}"

    ds_dir = tmp_path / "macro_cli_ds"
    build_synthetic_processed_corpus(
        ds_dir, num_topics=6, cascades_per_topic=2, replies_per_cascade=10, min_total_replies_extracted=0
    )
    paper_dir = tmp_path / "paper"
    figs_dir = tmp_path / "figs"
    proc_an = _run_cli(
        [
            "-m",
            "scripts.run_macro_signed_network",
            "--data-dir",
            str(ds_dir),
            "--dataset-name",
            "vn",
            "--analysis-only",
            "--num-null-perms",
            "10",
            "--paper-dir",
            str(paper_dir),
            "--figures-dir",
            str(figs_dir),
        ],
        timeout=120,
    )
    assert proc_an.returncode == 0, f"analysis-only failed:\n{proc_an.stdout}\n{proc_an.stderr}"
    macro_json = paper_dir / "results_macro_network_vn.json"
    assert macro_json.is_file()
    macro_data = json.loads(macro_json.read_text(encoding="utf-8"))
    assert -1.0 <= float(macro_data["polarization"]["signed_modularity"]) <= 1.0
    svg_path = figs_dir / "fig5_macro_signed_network.svg"
    assert svg_path.is_file()
    assert "Within Fiedler Camp" not in svg_path.read_text(encoding="utf-8")


@pytest.mark.tier3
def test_t3_f2_f3_f6_tier2_to_tier1_causal_bridge_training(tmp_path: Path) -> None:
    """F2 x F3 x F6: Causal pre-t_root tier2_user_prior flows from EarlyCascadeDataset into SBiFAGAT training."""
    ds_dir = tmp_path / "bridge_train_ds"
    build_synthetic_processed_corpus(
        ds_dir, num_topics=9, cascades_per_topic=1, replies_per_cascade=26, min_total_replies_extracted=25
    )
    ds_tr = EarlyCascadeDataset(
        root_dir=ds_dir, k_horizon=20, split="train", min_total_replies=25, include_tier2_prior=True
    )
    ds_va = EarlyCascadeDataset(
        root_dir=ds_dir, k_horizon=20, split="val", min_total_replies=25, include_tier2_prior=True
    )
    ds_te = EarlyCascadeDataset(
        root_dir=ds_dir, k_horizon=20, split="test", min_total_replies=25, include_tier2_prior=True
    )
    prior_dim = int(ds_tr[0].tier2_user_prior.size(1))
    _, metrics = train_and_evaluate_single_seed(
        lambda: SBiFAGAT(
            in_dim=TOTAL_NODE_FEATURE_DIM,
            hidden_dim=16,
            edge_dim=2,
            use_cascade_stats=True,
            use_tier2_prior=True,
            tier2_prior_dim=prior_dim,
        ),
        DataLoader(list(ds_tr), batch_size=2, shuffle=True),
        DataLoader(list(ds_va), batch_size=2, shuffle=False),
        DataLoader(list(ds_te), batch_size=2, shuffle=False),
        seed=42,
        epochs=2,
        patience=2,
        device=torch.device("cpu"),
    )
    assert np.isfinite(metrics["msle"])
    assert np.isfinite(metrics["rmse"])


@pytest.mark.tier3
def test_t3_f1_f5_demo_offline_cli_and_json_contract(tmp_path: Path) -> None:
    """F1 x F5: CLI scripts.predict_live_thread --demo-offline emits clean JSON with early_y_polar."""
    out_json = tmp_path / "live_demo.json"
    proc = _run_cli(
        [
            "-m",
            "scripts.predict_live_thread",
            "--demo-offline",
            "--junk-mode",
            "contract",
            "--output-json",
            str(out_json),
        ],
        timeout=120,
    )
    assert proc.returncode == 0, f"predict_live_thread failed:\n{proc.stdout}\n{proc.stderr}"
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "early_y_polar" in payload
    assert payload["confidence_score"] is None
    assert "irrational_fallacy" not in payload.get("behavior_counts", {})
    assert "vo_ly=" not in proc.stdout


_LATENCY_KEY_RE = re.compile(r"(?:^|[^a-z0-9])(?:ms|latency|elapsed|time)(?:[^a-z0-9]|$)")
_E2E_TAGS = ("e2e", "raw_text", "raw-text", "end_to_end")


def _is_latency_key(key: Any) -> bool:
    return bool(_LATENCY_KEY_RE.search(str(key).lower()))


def _extract_e2e_latency_ms(payload: Any) -> list[float]:
    """Extract latency (ms) values belonging to E2E raw-text profiling."""
    found: list[float] = []

    def _collect_ms(obj: Any, *, allow_bare_scalar: bool = False) -> None:
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            if allow_bare_scalar:
                found.append(float(obj))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _collect_ms(v, allow_bare_scalar=_is_latency_key(k))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _collect_ms(item, allow_bare_scalar=allow_bare_scalar)

    if isinstance(payload, dict):
        for k, v in payload.items():
            kl = str(k).lower()
            if any(tag in kl for tag in _E2E_TAGS):
                _collect_ms(v, allow_bare_scalar=_is_latency_key(kl))
            elif k in ("models", "rows") and isinstance(v, list):
                found.extend(_extract_e2e_latency_ms(v))
    elif isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            model_name = str(row.get("model", "")).lower()
            if any(tag in model_name for tag in _E2E_TAGS):
                _collect_ms(row, allow_bare_scalar=False)
            else:
                for k, v in row.items():
                    kl = str(k).lower()
                    if any(tag in kl for tag in _E2E_TAGS):
                        _collect_ms(v, allow_bare_scalar=_is_latency_key(kl))
    elif isinstance(payload, tuple):
        for item in payload:
            found.extend(_extract_e2e_latency_ms(item))
    return found


@pytest.mark.tier3
def test_t3_f3_f5_benchmark_efficiency_cli_e2e(tmp_path: Path) -> None:
    """F3 x F5: CLI scripts.benchmark_efficiency writes valid latency JSON and Markdown table."""

    help_proc = _run_cli(["-m", "scripts.benchmark_efficiency", "--help"], timeout=60)
    assert help_proc.returncode == 0
    e2e_flags = re.findall(r"(?<!\w)(--(?:include[-_]e2e|e2e[-_]raw[-_]text|e2e))\b", help_proc.stdout)

    out_json = tmp_path / "eff.json"
    out_md = tmp_path / "eff.md"
    cli_args = [
        "-m",
        "scripts.benchmark_efficiency",
        "--num-cascades",
        "6",
        "--k-horizon",
        "10",
        "--hidden-dim",
        "32",
        "--warmup",
        "1",
        "--repeats",
        "1",
        "--output-json",
        str(out_json),
        "--output-table",
        str(out_md),
    ]
    if e2e_flags:
        cli_args.append(e2e_flags[0])

    proc = _run_cli(cli_args, timeout=120)
    assert proc.returncode == 0, f"benchmark_efficiency failed:\n{proc.stdout}\n{proc.stderr}"
    assert out_json.is_file() and out_md.is_file()
    eff = json.loads(out_json.read_text(encoding="utf-8"))
    rows = eff if isinstance(eff, list) else eff.get("models", eff.get("rows", []))
    names = {r["model"] for r in rows}
    assert "SBiFAGAT" in names
    assert "BiEdgeGATv2" in names, f"BiEdgeGATv2 missing from benchmark_efficiency CLI output: {sorted(names)}"

    e2e_ms_vals = _extract_e2e_latency_ms(eff)
    assert any(v > 0.0 for v in e2e_ms_vals), (
        f"scripts.benchmark_efficiency CLI JSON output missing positive E2E raw-text latency ms: {eff}"
    )


@pytest.mark.tier3
def test_t3_f2_f4_run_macro_signed_network_link_prediction_cli_e2e(tmp_path: Path) -> None:
    """F2 x F4: Subprocess CLI scripts.run_macro_signed_network executes link-prediction mode & exports artifacts."""
    ds_dir = tmp_path / "macro_link_cli_ds"
    build_synthetic_processed_corpus(
        ds_dir,
        num_topics=6,
        cascades_per_topic=2,
        replies_per_cascade=12,
        emb_dim=16,
        min_total_replies_extracted=0,
        add_future_unseen_user=True,
    )
    paper_dir = tmp_path / "paper_link"
    figs_dir = tmp_path / "figs_link"
    selected_models = [
        "PriorPairSignPersistence",
        "BiEdgeGATv2-Link",
        "SiGAT",
        "SDGNN",
        "SBiFAGAT-Link",
    ]
    proc = _run_cli(
        [
            "-m",
            "scripts.run_macro_signed_network",
            "--data-dir",
            str(ds_dir),
            "--dataset-name",
            "vn",
            "--models",
            *selected_models,
            "--seeds",
            "42",
            "100",
            "--epochs",
            "2",
            "--hidden-dim",
            "16",
            "--num-layers",
            "1",
            "--batch-size",
            "32",
            "--num-null-perms",
            "5",
            "--num-q-null-perms",
            "5",
            "--paper-dir",
            str(paper_dir),
            "--figures-dir",
            str(figs_dir),
        ],
        timeout=180,
    )
    assert proc.returncode == 0, f"run_macro_signed_network link mode failed:\n{proc.stdout}\n{proc.stderr}"

    link_json = paper_dir / "results_signed_link_vn.json"
    link_tex = paper_dir / "tables_signed_link_vn.tex"
    link_md = paper_dir / "summary_signed_link_vn.md"
    assert link_json.is_file() and link_tex.is_file() and link_md.is_file()

    def _reject_nan(tok: str) -> Any:
        raise ValueError(tok)

    payload = json.loads(link_json.read_text(encoding="utf-8"), parse_constant=_reject_nan)
    assert set(selected_models).issubset(set(payload["summary_stats"].keys()))
    assert "unseen_pair_trans_roc_auc" in payload["summary_stats"]["SBiFAGAT-Link"]
    assert "unseen_pair_trans_pr_auc" in payload["summary_stats"]["SBiFAGAT-Link"]
    extra = payload.get("extra", payload)
    assert extra["split_counts"]["test_unseen_pair_transductive_edges"] > 0
    assert "wilcoxon_p_values" in payload and "adjusted_wilcoxon_p_values" in payload
    assert find_unescaped_underscores_in_latex(link_tex.read_text(encoding="utf-8")) == []

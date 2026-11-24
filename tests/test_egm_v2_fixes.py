"""Regression tests for EGM v2/v3 audit fixes across evaluation, downloaders, and notebooks."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from scripts import (
    audit_vn_50_trees,
    compute_kappa,
    eval_rumoureval_sdqc,
    kaggle_parallel_runner,
    run_experiments,
    run_upfd_extension,
)
from scripts.assemble_paper_artifacts import find_unescaped_underscores_in_latex
from scripts.export_dataset_stats import format_dataset_stats_latex, format_dataset_stats_markdown
from scripts.human_audit_ui import ALLOWLISTED_SPECS, create_server
from src.data import pheme_loader


def _write_single_gold_pair_files(tmp_path: Path) -> tuple[Path, Path]:
    audit_csv = tmp_path / "human_audit_vi_120.csv"
    key_csv = tmp_path / "human_audit_vi_120_machine_key.csv"
    audit_fields = [
        "sample_id",
        "cascade_id",
        "edge_id",
        "annotator_1_stance",
        "annotator_2_stance",
        "gold_adjudicated_stance",
        "annotator_1_hostility",
        "annotator_2_hostility",
        "gold_adjudicated_hostility",
    ]
    key_fields = ["sample_id", "edge_id", "s_uv", "h_uv"]
    rows = [
        ("s1", "e1", "support", "clean", 0.8, 0.1),
        ("s2", "e2", "oppose", "uncivil", -0.8, 0.8),
        ("s3", "e3", "neutral", "clean", 0.0, 0.1),
        ("s4", "e4", "support", "clean", 0.6, 0.2),
    ]
    with audit_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=audit_fields)
        w.writeheader()
        for sid, eid, st, ho, _s, _h in rows:
            w.writerow(
                {
                    "sample_id": sid,
                    "cascade_id": "c1",
                    "edge_id": eid,
                    "annotator_1_stance": "",
                    "annotator_2_stance": "",
                    "gold_adjudicated_stance": st,
                    "annotator_1_hostility": "",
                    "annotator_2_hostility": "",
                    "gold_adjudicated_hostility": ho,
                }
            )
    with key_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=key_fields)
        w.writeheader()
        for sid, eid, _st, _ho, s_val, h_val in rows:
            w.writerow({"sample_id": sid, "edge_id": eid, "s_uv": s_val, "h_uv": h_val})
    return audit_csv, key_csv


def _write_single_gold_tree_files(tmp_path: Path) -> tuple[Path, Path]:
    audit_csv = tmp_path / "human_audit_vn_30_trees.csv"
    key_csv = tmp_path / "human_audit_vn_30_trees_machine_key.csv"
    audit_fields = [
        "sample_id",
        "cascade_id",
        "annotator_1_polar_grade",
        "annotator_2_polar_grade",
        "gold_adjudicated_polar_grade",
    ]
    key_fields = ["sample_id", "cascade_id", "y_polar", "y_polar_quintile_grade"]
    rows = [
        ("t1", "c1", 0, 0.1, 0),
        ("t2", "c2", 1, 0.25, 1),
        ("t3", "c3", 2, 0.5, 2),
        ("t4", "c4", 3, 0.75, 3),
        ("t5", "c5", 4, 0.9, 4),
    ]
    with audit_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=audit_fields)
        w.writeheader()
        for sid, cid, grade, _yp, _qg in rows:
            w.writerow(
                {
                    "sample_id": sid,
                    "cascade_id": cid,
                    "annotator_1_polar_grade": "",
                    "annotator_2_polar_grade": "",
                    "gold_adjudicated_polar_grade": str(grade),
                }
            )
    with key_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=key_fields)
        w.writeheader()
        for sid, cid, _grade, yp, qg in rows:
            w.writerow({"sample_id": sid, "cascade_id": cid, "y_polar": yp, "y_polar_quintile_grade": qg})
    return audit_csv, key_csv


def test_single_verified_gold_pair_and_thread_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pair_csv, pair_key = _write_single_gold_pair_files(tmp_path)
    with pytest.raises(ValueError, match="allow-single-gold"):
        compute_kappa.evaluate_audit_csv(pair_csv, machine_key_csv=pair_key, allow_single_gold=False)

    pair_rep = compute_kappa.evaluate_audit_csv(pair_csv, machine_key_csv=pair_key, allow_single_gold=True)
    assert pair_rep["annotation_protocol"] == "single_verified_gold"
    assert pair_rep["machine_vs_gold"]["stance"]["accuracy"] == 1.0
    assert pair_rep["machine_vs_gold"]["hostility"]["accuracy"] == 1.0

    tree_csv, tree_key = _write_single_gold_tree_files(tmp_path)
    with pytest.raises(ValueError, match="allow-single-gold"):
        compute_kappa.evaluate_thread_level_audit_csv(tree_csv, machine_key_csv=tree_key, allow_single_gold=False)

    tree_rep = compute_kappa.evaluate_thread_level_audit_csv(tree_csv, machine_key_csv=tree_key, allow_single_gold=True)
    assert tree_rep["annotation_protocol"] == "single_verified_gold"
    assert tree_rep["machine_vs_gold"]["pearson_r"] > 0.9

    # CLI forwarding via audit_vn_50_trees.py --evaluate --allow-single-gold
    out_tree_json = tmp_path / "tree_out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_vn_50_trees.py",
            "--evaluate",
            "--allow-single-gold",
            "--audit-csv",
            str(tree_csv),
            "--machine-key",
            str(tree_key),
            "--output-json",
            str(out_tree_json),
        ],
    )
    audit_vn_50_trees.main()
    assert out_tree_json.is_file()

    # Verified Gold LaTeX formatting keeps IAA column cleanly empty (--) with no unescaped underscores
    tex = compute_kappa.format_human_audit_latex(
        {"corpora": [{"prefix": "VN ", "report": pair_rep}, {"prefix": "VN ", "report": tree_rep}]}
    )
    assert "Verified Gold" in tex
    assert "-- &" in tex
    assert find_unescaped_underscores_in_latex(tex) == []


def test_mixed_cohort_dataset_stats_formatting() -> None:
    pheme_s = {
        "name": "pheme",
        "k_horizon": 10,
        "min_total_replies": 15,
        "raw_cascades": 6425,
        "eligible_cascades": 1353,
        "eligible_topics": 9,
        "split_cascade_counts": {"train": 900, "val": 200, "test": 253},
        "median_replies": 22.0,
        "mean_replies": 25.4,
        "median_duration_hours": 1.25,
        "early_k_persistence": {"num_cascades": 253, "rmse": 0.21, "pearson_r": 0.65},
    }
    vn_s = {
        "name": "vn",
        "k_horizon": 20,
        "min_total_replies": 25,
        "raw_cascades": 4000,
        "eligible_cascades": 1821,
        "eligible_topics": 110,
        "split_cascade_counts": {"train": 1270, "val": 270, "test": 281},
        "median_replies": 45.0,
        "mean_replies": 68.2,
        "median_duration_hours": 5.5,
        "early_k_persistence": {"num_cascades": 281, "rmse": 0.28, "pearson_r": 0.48},
    }
    tex = format_dataset_stats_latex([pheme_s, vn_s])
    assert "Forecast Eligible" in tex
    assert "K{=}10" in tex
    assert "K{=}20" in tex
    assert find_unescaped_underscores_in_latex(tex) == []

    md = format_dataset_stats_markdown([pheme_s, vn_s])
    assert "Forecast Eligible" in md
    assert "pheme (K=10, >=15)" in md
    assert "vn (K=20, >=25)" in md


def test_loo_output_paths_and_rumoureval_root_default(tmp_path: Path) -> None:
    tbl, jsn, _ = run_experiments._resolve_output_paths(
        Path("data/processed/pheme"),
        None,
        None,
        None,
        split_mode="canonical",
        eval_protocol="loo",
    )
    assert tbl.name == "tables_loo_pheme.tex"
    assert jsn.name == "results_loo_pheme.json"

    class _DummyScorer:
        def __init__(self) -> None:
            self.seen_u: list[str] = []

        def score_pairs(self, u_texts: list[str], v_texts: list[str], **_kwargs: object) -> dict[str, list[Any]]:
            self.seen_u = list(u_texts)
            return {"s_uv": [0.8 for _ in v_texts], "pred_label": ["support" for _ in v_texts]}

    scorer = _DummyScorer()
    res = eval_rumoureval_sdqc.evaluate_nli_on_rumoureval(
        [
            {
                "id": "r1",
                "edge_id": "r1",
                "thread_id": "th1",
                "platform": "reddit",
                "sdqc_label": "support",
                "root_text": "ROOT_CLAIM",
                "parent_text": "NESTED_PARENT",
                "reply_text": "I agree with root",
            }
        ],
        scorer=scorer,
        output_json=tmp_path / "rumour.json",
    )
    assert scorer.seen_u == ["ROOT_CLAIM"]
    assert res["meta"]["context_mode"] == "root"

    good_file = tmp_path / "archive.bin"
    good_file.write_bytes(b"rumoureval-payload")
    good_sha = hashlib.sha256(b"rumoureval-payload").hexdigest()
    assert eval_rumoureval_sdqc._verify_archive_sha256(good_file, good_sha) == good_sha
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        eval_rumoureval_sdqc._verify_archive_sha256(good_file, "0" * 64)


def test_pheme_loader_curl_fast_path_mocked(tmp_path: Path) -> None:
    payload = b"P" * 1_100_000
    expected_md5 = hashlib.md5(payload).hexdigest()

    def _fake_subprocess_run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_bytes(payload)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    with (
        patch.object(pheme_loader, "_resolve_figshare_file_metadata", return_value=("https://ndownloader.figshare.com/files/test", expected_md5)),
        patch.object(pheme_loader, "_discover_local_pheme_source", return_value=(None, None)),
        patch.object(pheme_loader.subprocess, "run", side_effect=_fake_subprocess_run) as mock_run,
    ):
        archive = pheme_loader.download_pheme_from_figshare(raw_dir=tmp_path / "raw_pheme")
        assert mock_run.called
        assert archive.is_file()
        assert (tmp_path / "raw_pheme" / f"{pheme_loader.PHEME_ARCHIVE_FILENAME}.sha256").is_file()


def test_human_audit_ui_en_allowlist_and_notebooks() -> None:
    allow_names = {spec.filename for spec in ALLOWLISTED_SPECS}
    assert "pilot_calibration_en_20.csv" in allow_names
    assert "human_audit_en_120.csv" in allow_names

    notebooks_expected = {
        Path("notebooks/colab_gpu_v3_gnn_benchmarks.ipynb"): ("min_total_replies_extracted", "tables_loo_pheme.tex"),
        Path("notebooks/kaggle_member1_nlp_and_vn_t4x2.ipynb"): ("SBIF_BRANCH", "member1_vn"),
        Path("notebooks/kaggle_member2_gnn_and_en_t4x2.ipynb"): ("--context-mode", "root"),
        Path("notebooks/kaggle_macro_signed_gnn_t4.ipynb"): ("run_macro_signed_network", "--dataset-name"),
    }
    for nb_path, required_tokens in notebooks_expected.items():
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        joined_src = []
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            lines = cell["source"] if isinstance(cell["source"], list) else cell["source"].splitlines(keepends=True)
            joined_src.append("".join(lines))
            if lines and lines[0].lstrip().startswith("%%"):
                continue
            py_lines: list[str] = []
            in_bang_cont = False
            for ln in lines:
                stripped = ln.lstrip()
                if in_bang_cont:
                    in_bang_cont = ln.rstrip().endswith("\\")
                    continue
                if stripped.startswith(("!", "%")):
                    indent = " " * (len(ln) - len(stripped))
                    py_lines.append(f"{indent}pass\n")
                    in_bang_cont = ln.rstrip().endswith("\\")
                    continue
                py_lines.append(ln)
            ast.parse("".join(py_lines))
        full_src = "\n".join(joined_src)
        for tok in required_tokens:
            assert tok in full_src, f"Missing {tok!r} in {nb_path}"


def test_kaggle_runner_cohort_guard_and_lazy_umap(tmp_path: Path) -> None:
    pt_path = tmp_path / "nlp_features.pt"
    assert not kaggle_parallel_runner._nlp_features_match_cohort(pt_path, 15)

    torch.save({"min_total_replies_extracted": 25}, pt_path)
    assert not kaggle_parallel_runner._nlp_features_match_cohort(pt_path, 15)
    assert kaggle_parallel_runner._nlp_features_match_cohort(pt_path, 25)

    torch.save({"teacher_edge_stance": {"e1": 0.5}}, pt_path)
    assert kaggle_parallel_runner._nlp_features_match_cohort(pt_path, 25)
    assert not kaggle_parallel_runner._nlp_features_match_cohort(pt_path, 15)

    torch.save({"_meta": {"min_total_replies_extracted": 15}}, pt_path)
    assert kaggle_parallel_runner._nlp_features_match_cohort(pt_path, 15)

    # Importing src.utils.visualize in a clean subprocess must not eagerly import umap
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, src.utils.visualize; assert 'umap' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"


def test_human_audit_ui_http_vi120_vn30_en120(tmp_path: Path) -> None:
    interim = tmp_path / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    _write_single_gold_pair_files(interim)  # creates human_audit_vi_120.csv
    _write_single_gold_tree_files(interim)  # creates human_audit_vn_30_trees.csv

    # Also stage human_audit_en_120.csv with empty annotations
    en_csv = interim / "human_audit_en_120.csv"
    with en_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_id",
                "cascade_id",
                "edge_id",
                "root_text",
                "parent_text",
                "reply_text",
                "annotator_1_stance",
                "annotator_2_stance",
                "annotator_1_hostility",
                "annotator_2_hostility",
                "gold_adjudicated_stance",
                "gold_adjudicated_hostility",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "sample_id": "en_001",
                "cascade_id": "p_1",
                "edge_id": "e_1",
                "root_text": "Root tweet",
                "parent_text": "Root tweet",
                "reply_text": "Reply tweet",
                "annotator_1_stance": "",
                "annotator_2_stance": "",
                "annotator_1_hostility": "",
                "annotator_2_hostility": "",
                "gold_adjudicated_stance": "",
                "gold_adjudicated_hostility": "",
            }
        )

    server = create_server(interim, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for fname in ("human_audit_vi_120.csv", "human_audit_vn_30_trees.csv", "human_audit_en_120.csv"):
            with urllib.request.urlopen(f"{base}/api/dataset?file={fname}&role=annotator_1", timeout=5) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert data["rows"]

        for payload in (
            {
                "file": "human_audit_vi_120.csv",
                "role": "annotator_1",
                "sample_id": "s1",
                "stance": "support",
                "hostility": "clean",
            },
            {
                "file": "human_audit_vn_30_trees.csv",
                "role": "annotator_1",
                "sample_id": "t1",
                "polar_grade": "2",
            },
            {
                "file": "human_audit_en_120.csv",
                "role": "annotator_1",
                "sample_id": "en_001",
                "stance": "oppose",
                "hostility": "uncivil",
            },
        ):
            req = urllib.request.Request(
                f"{base}/api/save",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert resp.status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_upfd_retry_helper_and_auc_metric() -> None:
    keys = [k for k, _, _ in run_upfd_extension.UPFD_REPORT_METRICS]
    assert keys == ["accuracy", "macro_f1", "auc_roc"]

    attempts = {"n": 0}

    def _flaky_upfd(**kwargs: object) -> str:
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("transient GDrive error")
        return f"loaded-{kwargs['split']}"

    with (
        patch.object(run_upfd_extension, "UPFD", side_effect=_flaky_upfd),
        patch.object(run_upfd_extension.time, "sleep") as mock_sleep,
    ):
        res = run_upfd_extension._load_upfd_split(Path("upfd"), "gossipcop", "bert", "train", lambda x: x, max_retries=3)
        assert res == "loaded-train"
        assert attempts["n"] == 2
        assert mock_sleep.called

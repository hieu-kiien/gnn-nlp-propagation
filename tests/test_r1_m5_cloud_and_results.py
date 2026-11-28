"""Tests for M5 cloud runner synchronization, 4 notebooks, and real-data `results/*.json` invariants."""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
import torch

from src.data.pyg_dataset import CascadeData, TOTAL_NODE_FEATURE_DIM
from scripts import kaggle_parallel_runner as kpr
from scripts.run_m5_local_audits import _assert_finite_json_obj


NOTEBOOK_PATHS = (
    Path("notebooks/colab_gpu_v3_gnn_benchmarks.ipynb"),
    Path("notebooks/kaggle_member1_nlp_and_vn_t4x2.ipynb"),
    Path("notebooks/kaggle_member2_gnn_and_en_t4x2.ipynb"),
    Path("notebooks/kaggle_macro_signed_gnn_t4.ipynb"),
)


def test_kaggle_runner_cli_exposes_m5_flags() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.kaggle_parallel_runner", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for flag in (
        "--results-dir",
        "--distill-from-features",
        "--hostility-target-router",
        "--hostility-gate-threshold",
        "--polar-target-mode",
        "--include-tier2-prior",
        "--run-embedding-cluster",
        "--skip-sota",
        "--run-tier2",
        "--mp-graph-mode",
        "--skip-e2e-efficiency",
        "--include-heavy-weights",
    ):
        assert flag in proc.stdout, f"Missing CLI flag {flag} in kaggle_parallel_runner --help"


def test_kaggle_runner_defaults_to_results_and_supports_m5_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    proc_vn = work / "data" / "processed" / "vn"
    interim_vn = work / "data" / "interim" / "vn"
    ce_dir = work / "checkpoints" / "dual_ce_vn"
    vihsd_dir = work / "checkpoints" / "visobert_vihsd"
    proc_vn.mkdir(parents=True)
    interim_vn.mkdir(parents=True)
    ce_dir.mkdir(parents=True)
    vihsd_dir.mkdir(parents=True)
    (interim_vn / "nodes.parquet").write_bytes(b"ok")
    (interim_vn / "edges.parquet").write_bytes(b"ok")
    (proc_vn / "nodes.parquet").write_bytes(b"ok")
    (proc_vn / "edges.parquet").write_bytes(b"ok")
    torch.save({"min_total_replies_extracted": 25, "teacher_edge_stance": {"e1": 0.5}}, proc_vn / "nlp_features.pt")

    fake_graph = CascadeData(
        x=torch.zeros((2, TOTAL_NODE_FEATURE_DIM)),
        edge_index=torch.tensor([[0], [1]]),
        edge_index_td=torch.tensor([[0], [1]]),
        edge_index_bu=torch.tensor([[1], [0]]),
        edge_attr=torch.tensor([[0.5, 0.1]]),
    )
    recorded_cmds: list[list[str]] = []
    monkeypatch.setattr(kpr, "_run_cmd", lambda cmd, cwd: recorded_cmds.append([str(x) for x in cmd]))
    monkeypatch.setattr(kpr, "EarlyCascadeDataset", lambda **_kw: [fake_graph])

    # Minimal SimpleNamespace omitting all new M5 fields -> getattr defaults must apply cleanly
    args_min = SimpleNamespace(
        role="member1_vn",
        lang="vi",
        smoke_test=False,
        work_dir=work,
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
        run_upfd=False,
        bundle_weights=False,
    )
    res_min = kpr.run_kaggle_parallel_role(args_min)
    assert Path(res_min["results_dir"]).name == "results"
    assert Path(res_min["bundle_zip"]).exists()
    exp_cmds = [c for c in recorded_cmds if "scripts.run_experiments" in c]
    assert exp_cmds[0][exp_cmds[0].index("--output-json") + 1] == "results/results_main_vn.json"
    assert all("paper/" not in tok for cmd in recorded_cmds for tok in cmd)
    eff_cmd = [c for c in recorded_cmds if "scripts.benchmark_efficiency" in c][0]
    assert "--include-e2e" in eff_cmd

    # Full M5 flags active
    recorded_cmds.clear()
    args_m5 = SimpleNamespace(
        **vars(args_min),
        results_dir=Path("custom_results"),
        hostility_target_router=True,
        hostility_gate_threshold=0.55,
        polar_target_mode="future_only",
        include_tier2_prior=True,
        run_embedding_cluster=True,
        skip_sota=False,
        run_tier2=True,
        mp_graph_mode="multiedge",
        skip_e2e_efficiency=True,
        include_heavy_weights=True,
        force=False,
    )
    res_m5 = kpr.run_kaggle_parallel_role(args_m5)
    assert Path(res_m5["results_dir"]).name == "custom_results"
    scorer_cmd = [c for c in recorded_cmds if "src.nlp.dual_edge_scorer" in c][0]
    assert "--hostility-target-router" in scorer_cmd
    assert scorer_cmd[scorer_cmd.index("--hostility-gate-threshold") + 1] == "0.55"

    exp_cmds_m5 = [c for c in recorded_cmds if "scripts.run_experiments" in c]
    assert "--include-tier2-prior" in exp_cmds_m5[0]
    assert exp_cmds_m5[0][exp_cmds_m5[0].index("--output-json") + 1] == "custom_results/results_main_vn.json"
    assert any("--split-mode" in c and "embedding_cluster" in c for c in exp_cmds_m5)
    assert any("--split-mode" in c and "temporal" in c for c in exp_cmds_m5)

    tier2_cmds = [c for c in recorded_cmds if "scripts.run_macro_signed_network" in c]
    assert len(tier2_cmds) == 1
    assert tier2_cmds[0][tier2_cmds[0].index("--mp-graph-mode") + 1] == "multiedge"
    assert tier2_cmds[0][tier2_cmds[0].index("--paper-dir") + 1] == "custom_results"

    eff_cmd_skip = [c for c in recorded_cmds if "scripts.benchmark_efficiency" in c][0]
    assert "--include-e2e" not in eff_cmd_skip


def test_kaggle_bundle_includes_results_and_dual_head_weights(tmp_path: Path) -> None:
    work = tmp_path / "work"
    res_dir = work / "results"
    ce_dir = work / "checkpoints" / "dual_ce_vn"
    res_dir.mkdir(parents=True)
    ce_dir.mkdir(parents=True)
    (res_dir / "results_main_vn.json").write_text('{"ok": true}', encoding="utf-8")
    (ce_dir / "dual_head_weights.pt").write_bytes(b"weights_bytes")

    zpath = kpr._bundle_outputs(
        work / "bundle.zip",
        source_paths=[res_dir, work / "checkpoints"],
        base_dir=work,
        include_heavy_weights=True,
        include_nlp_features=False,
    )
    with zipfile.ZipFile(zpath, "r") as zf:
        names = set(zf.namelist())
    assert "results/results_main_vn.json" in names
    assert "checkpoints/dual_ce_vn/dual_head_weights.pt" in names


def test_all_four_notebooks_valid_json_ast_and_r4_isolated() -> None:
    for nb_path in NOTEBOOK_PATHS:
        assert nb_path.is_file(), f"Missing notebook: {nb_path}"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        cells = nb["cells"]
        assert len(cells) >= 5
        all_code = ""
        for cell in cells:
            src = "".join(cell.get("source", []))
            if cell.get("cell_type") != "code":
                continue
            all_code += "\n" + src
            # Never target or delete inside paper/
            assert "paper/" not in src, f"R4 violation in {nb_path}: references 'paper/'"
            assert '"paper"' not in src and "'paper'" not in src, (
                f"R4 violation in {nb_path}: targets 'paper' dir"
            )
            if src.lstrip().startswith("%%"):
                continue
            py_lines: list[str] = []
            in_bang_cont = False
            for line in src.splitlines():
                stripped = line.lstrip()
                if in_bang_cont:
                    in_bang_cont = line.rstrip().endswith("\\")
                    continue
                if stripped.startswith(("!", "%")):
                    indent = " " * (len(line) - len(stripped))
                    py_lines.append(f"{indent}pass")
                    in_bang_cont = line.rstrip().endswith("\\")
                    continue
                py_lines.append(line)
            ast.parse("\n".join(py_lines), filename=str(nb_path))
        assert "results" in all_code


def test_real_results_dir_json_invariants_if_present() -> None:
    results_dir = Path("results")
    if not results_dir.is_dir() or not list(results_dir.glob("*.json")):
        pytest.skip("No results/*.json generated yet")

    for jf in sorted(results_dir.rglob("*.json")):
        payload = json.loads(jf.read_text(encoding="utf-8"))
        _assert_finite_json_obj(payload, path_prefix=jf.name)

    for ds in ("vn", "pheme"):
        macro_p = results_dir / f"results_macro_network_{ds}.json"
        if macro_p.is_file():
            m = json.loads(macro_p.read_text(encoding="utf-8"))
            rwc_val = float(m["polarization"]["rwc"])
            assert -1.0 - 1e-6 <= rwc_val <= 1.0 + 1e-6
            assert math.isfinite(float(m["polarization"]["q_signed_z_score"]))
            assert math.isfinite(float(m["polarization"]["q_signed_p_value"]))
            assert math.isfinite(float(m["structural_balance"]["weak_balance_ratio"]))
            assert math.isfinite(float(m["structural_balance"]["weak_balance_z_score"]))
            if ds == "pheme":
                assert int(m["filtered_zero_embedding_posts"]) == 24354

    contraction_p = results_dir / "contraction_eval_vn.json"
    if contraction_p.is_file():
        c = json.loads(contraction_p.read_text(encoding="utf-8"))
        assert int(c["threads_evaluated"]) == 1500
        assert isinstance(c.get("post_crawl_parquet_caveat"), str) and len(c["post_crawl_parquet_caveat"]) > 20

    eff_p = results_dir / "efficiency_e2e.json"
    if eff_p.is_file():
        e = json.loads(eff_p.read_text(encoding="utf-8"))
        assert float(e["e2e_raw_text_profile"]["e2e_total_ms"]) > 0.0

    for probe_p in sorted(results_dir.glob("probe_signed_link_*_seed42.json")):
        p = json.loads(probe_p.read_text(encoding="utf-8"))
        assert p.get("is_single_seed_probe") is True
        assert p.get("seeds") == [42]

    for tab_p in sorted(results_dir.glob("tier1_tabular_*.json")):
        t = json.loads(tab_p.read_text(encoding="utf-8"))
        assert t["seeds"] == [42, 100, 2024, 777, 999]
        assert t.get("extra", {}).get("is_real_data") is True
        for m_name in ("Early-K Persistence", "Calibrated Early-K Persistence", "Ridge", "LightGBM"):
            assert m_name in t["summary_stats"]

    for link_p in sorted(results_dir.glob("results_signed_link_*.json")):
        sl = json.loads(link_p.read_text(encoding="utf-8"))
        assert sl["seeds"] == [42, 100, 2024, 777, 999]
        assert len(sl["summary_stats"]) == 10
        assert "test_unseen_pair_transductive_edges" in sl.get("extra", {}).get("split_counts", {})
        assert "unseen_pair_transductive" in sl["summary_stats"]["SBiFAGAT-Link"] or "unseen_pair_trans_roc_auc" in sl["summary_stats"]["SBiFAGAT-Link"]

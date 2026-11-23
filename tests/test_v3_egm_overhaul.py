"""Regression tests for the v3 EGM overhaul: PII/spam hygiene, linear y_polar,
causal Persistence-Residual anchor, masked Camp Dipole, Validation-only comparator
selection, binary clean/uncivil Gate 1 audit, --lock-role UI isolation, and
non-overwriting multi-corpus LaTeX assembly.
"""

from __future__ import annotations

import csv
import json
import math
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from scripts.analyze_mechanisms import _clone_cascade_with_edge_attr
from scripts.assemble_paper_artifacts import assemble_paper_artifacts, find_unescaped_underscores_in_latex
from scripts.audit_vn_50_trees import export_vn_50_tree_audit
from scripts.compute_kappa import (
    HOSTILITY_ORDER,
    evaluate_audit_csv,
    evaluate_machine_vs_human,
    evaluate_thread_level_audit_csv,
    format_human_audit_latex,
)
from scripts.human_audit_ui import AuditStore, create_server
from src.data.arctic_shift_vn import is_doxxing_post, is_invalid_or_spam_text, redact_pii_text
from src.data.pyg_dataset import (
    CascadeData,
    EarlyCascadeDataset,
    STRUCTURAL_NODE_DIM,
    TOTAL_NODE_FEATURE_DIM,
    _compute_full_tree_polarization,
    compute_embedding_cluster_splits,
    compute_temporal_topic_splits,
)
from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet
from src.models.baselines import (
    BiEdgeGATv2,
    BiGCN,
    EdgeGATv2,
    FAGCNBaseline,
    StandardGATv2,
    StandardGCN,
    TemporalStanceMLP,
    TextMLP,
)
from src.models.sbi_fagat import SBiFAGAT
from src.nlp.dual_edge_scorer import export_stratified_human_audit_csv
from src.training.trainer import run_multi_seed_benchmark


def _http_json(url: str, *, method: str = "GET", body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return int(resp.status), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return int(exc.code), json.loads(exc.read().decode("utf-8"))


def test_v3_pii_doxxing_and_crypto_spam_filtering() -> None:
    dox = "Truy tìm đối tượng tên A sđt 0912345678 stk VCB 1234567890"
    assert is_doxxing_post(dox) is True
    assert is_invalid_or_spam_text(dox) is True

    phone_txt = "Liên hệ mình qua 0987654321 hoặc facebook.com/profile_user123 nhé"
    red = redact_pii_text(f"https://{phone_txt.split('hoặc ')[1]} và {phone_txt.split(' hoặc')[0]}")
    assert "0987654321" not in red
    assert "profile_user123" not in red
    assert "[REDACTED_PII]" in red

    # Public FB share link is allowed by _FB_PROFILE_RE
    pub_txt = "Xem bài báo tại https://www.facebook.com/share/p/123abc/"
    assert redact_pii_text(pub_txt) == pub_txt

    # LMGX crypto spam caught
    assert is_invalid_or_spam_text("Mọi người đầu tư LMGX ngay đi nhé") is True


def test_v3_linear_ypolar_early_ypolar_and_splits(tmp_path: Path) -> None:
    # Linear polarization properties
    assert _compute_full_tree_polarization([1.0] * 5 + [-1.0] * 5) == pytest.approx(1.0)
    weak = _compute_full_tree_polarization([0.11] * 10 + [-0.11] * 10)
    strong = _compute_full_tree_polarization([0.8] * 10 + [-0.8] * 10)
    assert 0.08 < weak < 0.15
    assert strong > 0.65

    # Temporal + embedding cluster splits
    base_ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_embs: dict[str, torch.Tensor] = {}
    edge_st: dict[str, float] = {}
    edge_ho: dict[str, float] = {}
    rng = np.random.default_rng(7)
    for t_i in range(10):
        cid = f"c_{t_i:02d}"
        tid = f"topic_{t_i:02d}"
        ts0 = base_ts + timedelta(days=t_i * 5)
        rid = f"{cid}_root"
        rows.append(
            {
                "cascade_id": cid,
                "topic_id": tid,
                "node_id": rid,
                "parent_id": None,
                "is_root": True,
                "user_id_hash": "auth_0",
                "created_at_utc": ts0,
                "text": f"Bài gốc chủ đề {tid}",
                "lang": "vi",
                "source_dataset": "vn_reddit",
                "rumor_label": None,
            }
        )
        base_vec = torch.tensor(rng.normal(loc=float(t_i), scale=0.2, size=(768,)), dtype=torch.float32)
        node_embs[rid] = base_vec
        for r_i in range(1, 28):
            nid = f"{cid}_r{r_i}"
            rows.append(
                {
                    "cascade_id": cid,
                    "topic_id": tid,
                    "node_id": nid,
                    "parent_id": rid if r_i % 2 == 1 else f"{cid}_r{r_i - 1}",
                    "is_root": False,
                    "user_id_hash": f"auth_{r_i}",
                    "created_at_utc": ts0 + timedelta(minutes=r_i),
                    "text": f"Bình luận số {r_i} trong {tid}",
                    "lang": "vi",
                    "source_dataset": "vn_reddit",
                    "rumor_label": None,
                }
            )
            pid = rid if r_i % 2 == 1 else f"{cid}_r{r_i - 1}"
            eid = f"{pid}->{nid}"
            c_ts = ts0 + timedelta(minutes=r_i)
            p_min = 0 if pid == rid else (r_i - 1)
            edges.append(
                {
                    "cascade_id": cid,
                    "edge_id": eid,
                    "source_id": pid,
                    "target_id": nid,
                    "created_at_utc": c_ts,
                    "time_delay_sec": float((r_i - p_min) * 60.0),
                    "edge_type": "reply",
                }
            )
            node_embs[nid] = base_vec.clone()
            edge_st[eid] = 0.7 if r_i % 2 == 1 else -0.7
            edge_ho[eid] = 0.1

    ds_dir = tmp_path / "vn_ds"
    save_cascade_parquet(build_nodes_table(rows), build_edges_table(edges), output_dir=ds_dir, validate=True)
    torch.save(
        {
            "node_embeddings": node_embs,
            "edge_stance": edge_st,
            "edge_hostility": edge_ho,
            "teacher_edge_stance": edge_st,
            "teacher_edge_hostility": edge_ho,
            "min_total_replies_extracted": 25,
        },
        ds_dir / "nlp_features.pt",
    )
    ds_all = EarlyCascadeDataset(root_dir=ds_dir, k_horizon=20, split="all", min_total_replies=25)
    assert len(ds_all) == 10
    for d in ds_all:
        assert hasattr(d, "early_y_polar") and 0.0 <= float(d.early_y_polar.item()) <= 1.0
        assert hasattr(d, "root_ts") and float(d.root_ts) > 0.0

    t_sp = compute_temporal_topic_splits(ds_all._all_cascades)
    assert t_sp["train"] and t_sp["val"] and t_sp["test"]
    assert t_sp["train"].isdisjoint(t_sp["val"]) and t_sp["val"].isdisjoint(t_sp["test"])
    max_tr = max(float(d.root_ts) for d in ds_all if d.topic_id in t_sp["train"])
    min_te = min(float(d.root_ts) for d in ds_all if d.topic_id in t_sp["test"])
    assert max_tr < min_te

    c_sp = compute_embedding_cluster_splits(ds_all._all_cascades, seed=42)
    assert c_sp["train"] and c_sp["val"] and c_sp["test"]
    assert c_sp["train"].isdisjoint(c_sp["test"])


def test_v3_models_persistence_single_camp_dipole_and_val_selection() -> None:
    torch.manual_seed(0)
    graphs: list[CascadeData] = []
    for i in range(6):
        n_nodes = 8
        src = [0, 0, 1, 1, 2, 3, 4]
        dst = [1, 2, 3, 4, 5, 6, 7]
        e_td = torch.tensor([src, dst], dtype=torch.long)
        e_bu = e_td[[1, 0], :]
        s_uv = torch.tensor([[0.8], [0.8], [0.8], [0.8], [0.8], [0.8], [0.8]], dtype=torch.float32)
        h_uv = torch.full((7, 1), 0.15, dtype=torch.float32)
        graphs.append(
            CascadeData(
                x=torch.randn((n_nodes, TOTAL_NODE_FEATURE_DIM)),
                struct_x=torch.rand((n_nodes, STRUCTURAL_NODE_DIM)),
                edge_index=e_td,
                edge_index_td=e_td,
                edge_index_bu=e_bu,
                edge_attr=torch.cat([s_uv, h_uv], dim=-1),
                edge_stance=s_uv,
                cascade_stats=torch.rand((1, 8)),
                early_y_polar=torch.tensor([0.12 + 0.05 * i], dtype=torch.float32),
                root_index=torch.tensor([0], dtype=torch.long),
                y_size=torch.tensor([2.5 + 0.2 * i], dtype=torch.float32),
                y_polar=torch.tensor([0.15 + 0.05 * i], dtype=torch.float32),
                y_rumor=torch.tensor([i % 4], dtype=torch.long),
                cascade_id=f"c{i}",
                topic_id=f"t{i}",
            )
        )

    loader = DataLoader(graphs, batch_size=3, shuffle=False)
    models = [
        TextMLP(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, predict_uncertainty=True),
        TemporalStanceMLP(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2, predict_uncertainty=True),
        StandardGCN(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, predict_uncertainty=True),
        StandardGATv2(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, predict_uncertainty=True),
        EdgeGATv2(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2, predict_uncertainty=True),
        FAGCNBaseline(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, predict_uncertainty=True),
        BiGCN(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, predict_uncertainty=True),
        SBiFAGAT(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2, predict_uncertainty=True),
    ]
    batch = next(iter(loader))
    for m in models:
        out = m(batch)
        assert "polar_logvar" in out and "confidence_score" in out
        assert torch.isfinite(out["y_polar_pred"]).all()

    # Latent-edge and ablate-stance auto-disable persistence anchor
    assert EdgeGATv2(use_latent_edge_mlp=True).use_polar_persistence_anchor is False
    assert BiEdgeGATv2(use_latent_edge_mlp=True).use_polar_persistence_anchor is False
    assert SBiFAGAT(ablate_stance=True).use_polar_persistence_anchor is False
    assert SBiFAGAT(use_latent_edge_mlp=True).use_polar_persistence_anchor is False
    assert getattr(TextMLP().head, "persist_ep_logit_ref", None) is None
    anchor_m = SBiFAGAT(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2, predict_uncertainty=False)
    anchor_m.eval()
    assert hasattr(anchor_m.head, "persist_ep_logit_ref")
    b0 = batch.clone()
    b0.early_y_polar = torch.zeros_like(b0.early_y_polar)
    b1 = batch.clone()
    b1.early_y_polar = torch.full_like(b1.early_y_polar, 0.95)
    with torch.no_grad():
        out_zero_0 = anchor_m(b0)["y_polar_logit"]
        out_zero_1 = anchor_m(b1)["y_polar_logit"]
    assert torch.allclose(out_zero_0, out_zero_1, atol=1e-6)
    with torch.no_grad():
        anchor_m.head.persist_weight.fill_(1.0)
        out_one_0 = anchor_m(b0)["y_polar_logit"]
        out_one_1 = anchor_m(b1)["y_polar_logit"]
    assert torch.isfinite(out_one_0).all() and torch.isfinite(out_one_1).all()
    assert bool((out_one_1 > out_one_0).all())
    assert float((out_one_1 - out_one_0).max().item()) < 8.0

    # Mechanism clone keeps early_y_polar consistent under opposition neutralization
    opp_g = graphs[0].clone()
    opp_ea = opp_g.edge_attr.clone()
    opp_ea[0, 0] = -0.9
    opp_ea[1, 0] = 0.9
    opp_clone = _clone_cascade_with_edge_attr(opp_g, opp_ea)
    neut_ea = opp_ea.clone()
    neut_ea[:, 0] = 0.0
    neut_clone = _clone_cascade_with_edge_attr(opp_clone, neut_ea)
    assert float(opp_clone.early_y_polar.item()) > 0.3
    assert float(neut_clone.early_y_polar.item()) < 1e-4

    res = run_multi_seed_benchmark(
        {
            "BiGCN": lambda: BiGCN(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=16, use_cascade_stats=True, predict_uncertainty=True),
            "SBiFAGAT": lambda: SBiFAGAT(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=16, edge_dim=2, use_cascade_stats=True, predict_uncertainty=True),
        },
        DataLoader(graphs[:2], batch_size=2, shuffle=True),
        DataLoader(graphs[2:4], batch_size=2, shuffle=False),
        DataLoader(graphs[4:], batch_size=2, shuffle=False),
        seeds=[42, 100],
        epochs=2,
        patience=2,
        lambda_uncertainty=0.05,
        reference_model="SBiFAGAT",
        comparator_model="auto",
        device="cpu",
    )
    assert "val_summary_stats" in res and "BiGCN" in res["val_summary_stats"]
    assert res["comparator_model"] == "BiGCN"
    best_sbi = res["models"]["SBiFAGAT"]
    ref_val = float(best_sbi.head.persist_ep_logit_ref.view(-1)[0].item())
    assert math.isfinite(ref_val) and ref_val != 0.0


def test_v3_audit_binary_incivility_lock_role_and_assembly(tmp_path: Path) -> None:
    assert HOSTILITY_ORDER == ("clean", "uncivil")

    # 1. Export balanced 20 + 10 audit filtering out LMGX spam & [REDACTED_PII]
    pairs: list[dict[str, Any]] = [
        {
            "edge_id": "spam_e",
            "cascade_id": "c_test_0",
            "root_text": "Bài thảo luận hay về giao thông công cộng",
            "grandparent_text": "",
            "parent_text": "Nên đầu tư xe buýt điện đồng bộ hơn",
            "reply_text": "Tham gia LMGX kiếm tiền ngay đi",
            "s_uv": -0.7,
            "h_uv": 0.85,
            "pred_label": "oppose",
        },
        {
            "edge_id": "red_e",
            "cascade_id": "c_test_0",
            "root_text": "Bài thảo luận hay",
            "grandparent_text": "",
            "parent_text": "Bình luận bình thường",
            "reply_text": "[REDACTED_DOXXING_PII]",
            "s_uv": -0.7,
            "h_uv": 0.9,
            "pred_label": "oppose",
        },
    ]
    stances = ["support", "neutral", "oppose"]
    for i in range(60):
        pairs.append(
            {
                "edge_id": f"ok_{i:03d}",
                "cascade_id": "c_test_0" if i < 40 else "c_train_ignore",
                "root_text": "Bài gốc thảo luận cộng đồng về hạ tầng",
                "grandparent_text": f"Bình luận ông bà {i}",
                "parent_text": f"Bình luận cha phân tích chi tiết số {i}",
                "reply_text": f"Mình đồng tình hoặc phản biện rõ ràng ý số {i} trong bài viết này",
                "s_uv": 0.7 if i % 3 == 0 else (-0.7 if i % 3 == 2 else 0.0),
                "h_uv": 0.8 if i % 2 == 0 else 0.1,
                "pred_label": stances[i % 3],
            }
        )

    out_csv = tmp_path / "human_audit_vi_120.csv"
    out_key = tmp_path / "human_audit_vi_120_machine_key.csv"
    pilot_csv = tmp_path / "pilot_calibration_vi_20.csv"
    pilot_key = tmp_path / "pilot_calibration_vi_20_machine_key.csv"
    export_stratified_human_audit_csv(
        pairs,
        output_csv=out_csv,
        machine_key_csv=out_key,
        pilot_csv=pilot_csv,
        pilot_machine_key_csv=pilot_key,
        n_samples=20,
        pilot_samples=10,
        test_cascade_ids={"c_test_0"},
        seed=42,
    )
    with out_csv.open("r", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 20
    assert all(r["cascade_id"] == "c_test_0" for r in rows)
    assert all(r["edge_id"] not in {"spam_e", "red_e"} for r in rows)
    assert "grandparent_text" in rows[0] and "annotator_1_notes" in rows[0]

    # Unadjudicated disagreement triggers strict_adjudication even WITHOUT machine_key_csv
    rows[0]["annotator_1_stance"] = "support"
    rows[0]["annotator_2_stance"] = "oppose"
    rows[0]["annotator_1_hostility"] = "clean"
    rows[0]["annotator_2_hostility"] = "uncivil"
    for r in rows[1:]:
        r["annotator_1_stance"] = "support"
        r["annotator_2_stance"] = "support"
        r["annotator_1_hostility"] = "clean"
        r["annotator_2_hostility"] = "clean"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with pytest.raises(ValueError, match="disagreement"):
        evaluate_audit_csv(out_csv, machine_key_csv=None, strict_adjudication=True)

    # Adjudicate row 0 + row 1 uncivil to verify ROC-AUC / PR-AUC / weighted metrics
    rows[0]["gold_adjudicated_stance"] = "oppose"
    rows[0]["gold_adjudicated_hostility"] = "uncivil"
    rows[1]["annotator_1_hostility"] = "hate"  # legacy alias normalizes to uncivil
    rows[1]["annotator_2_hostility"] = "uncivil"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rep_eval = evaluate_audit_csv(out_csv, machine_key_csv=out_key, strict_adjudication=True)
    assert rep_eval["hostility"]["labels"] == ["clean", "uncivil"]
    assert "roc_auc" in rep_eval["machine_vs_gold"]["hostility"]
    assert "weighted_macro_f1" in rep_eval["machine_vs_gold"]["hostility"]

    # 2. UI --lock-role blocks other roles & isolates notes + uncertain
    ui_dir = tmp_path / "ui_data"
    ui_dir.mkdir()
    (ui_dir / "pilot_calibration_vi_20.csv").write_text(pilot_csv.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
    srv = create_server(ui_dir, host="127.0.0.1", port=0, locked_role="annotator_1")
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        st_bad, _ = _http_json(f"{base}/api/dataset?file=pilot_calibration_vi_20.csv&role=annotator_2")
        assert st_bad == 403
        st_gold, _ = _http_json(f"{base}/api/dataset?file=pilot_calibration_vi_20.csv&role=gold")
        assert st_gold == 403
        st_ok, ds_ok = _http_json(f"{base}/api/dataset?file=pilot_calibration_vi_20.csv&role=annotator_1")
        assert st_ok == 200 and ds_ok["locked_role"] == "annotator_1"

        sid = ds_ok["rows"][0]["sample_id"]
        st_sv, sv_res = _http_json(
            f"{base}/api/save",
            method="POST",
            body={
                "file": "pilot_calibration_vi_20.csv",
                "role": "annotator_1",
                "sample_id": sid,
                "stance": "support",
                "hostility": "uncivil",
                "notes": "ghi chú riêng 1",
                "uncertain": True,
            },
        )
        assert st_sv == 200 and sv_res["row"]["notes"] == "ghi chú riêng 1" and sv_res["row"]["uncertain"] is True
        # Ensure annotator_2 gets empty notes even on direct store read
        store_unlocked = AuditStore(ui_dir)
        d2 = store_unlocked.get_dataset("pilot_calibration_vi_20.csv", "annotator_2")
        assert d2["rows"][0]["notes"] == ""
    finally:
        srv.shutdown()
        srv.server_close()

    # 3. Thread audit computes kendall_tau_b + roc_auc_ge2
    t_csv = tmp_path / "trees.csv"
    t_key = tmp_path / "trees_machine_key.csv"
    with t_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "cascade_id", "annotator_1_polar_grade", "annotator_2_polar_grade", "gold_adjudicated_polar_grade"])
        w.writeheader()
        for idx in range(5):
            w.writerow({"sample_id": f"tree_{idx+1:03d}", "cascade_id": f"c_{idx}", "annotator_1_polar_grade": idx, "annotator_2_polar_grade": idx, "gold_adjudicated_polar_grade": idx})
    with t_key.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "cascade_id", "y_polar", "y_polar_quintile_grade"])
        w.writeheader()
        for idx in range(5):
            w.writerow({"sample_id": f"tree_{idx+1:03d}", "cascade_id": f"c_{idx}", "y_polar": 0.05 + 0.05 * idx, "y_polar_quintile_grade": idx})
    t_rep = evaluate_thread_level_audit_csv(t_csv, t_key, strict_adjudication=True)
    assert t_rep["machine_vs_gold"]["kendall_tau_b"] == pytest.approx(1.0)
    assert t_rep["machine_vs_gold"]["roc_auc_ge2"] == pytest.approx(1.0)

    # 4. assemble_paper_artifacts preserves BOTH VN and EN rows without overwrite
    pdir = tmp_path / "paper_multi"
    pdir.mkdir()
    (pdir / "results_human_audit_vi.json").write_text(
        json.dumps({"stance": {"num_samples": 120, "cohen_kappa_unweighted": 0.72, "cohen_kappa_quadratic": 0.78}, "machine_vs_gold": {"stance": {"macro_f1": 0.75, "spearman_r": 0.71}}}),
        encoding="utf-8",
    )
    (pdir / "results_human_audit_en.json").write_text(
        json.dumps({"stance": {"num_samples": 300, "cohen_kappa_unweighted": 0.81, "cohen_kappa_quadratic": 0.85}, "machine_vs_gold": {"stance": {"macro_f1": 0.82, "spearman_r": 0.80}}}),
        encoding="utf-8",
    )
    manifest = assemble_paper_artifacts(pdir, strict_latex_check=True)
    audit_tex = (pdir / "tables_human_audit.tex").read_text(encoding="utf-8")
    assert "VN Pair Stance (3-cls) & 120" in audit_tex
    assert "EN Pair Stance (3-cls) & 300" in audit_tex
    assert find_unescaped_underscores_in_latex(audit_tex) == []
    assert manifest["latex_underscore_issues"] == {}

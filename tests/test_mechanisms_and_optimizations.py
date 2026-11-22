"""Tests for lossless NLP length-bucketing/DP wrappers and mechanistic counterfactual analysis."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader
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

from scripts.analyze_mechanisms import (
    format_mechanisms_latex,
    load_sbifagat_checkpoint,
    run_mechanism_analysis,
)
from scripts.assemble_paper_artifacts import find_unescaped_underscores_in_latex
from src.data.pyg_dataset import CascadeData, STRUCTURAL_NODE_DIM, TOTAL_NODE_FEATURE_DIM
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
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    HostilityTeacherScorer,
    NLIStanceTeacherScorer,
)
from src.nlp.extract_features import (
    encode_edges_with_cross_encoder,
    encode_node_embeddings,
)
from src.training.reporting import export_latex_table
from src.training.trainer import _estimate_train_target_stats, _init_multitask_head_biases
from src.utils.seed import seed_everything
from src.utils.visualize import plot_counterfactual_interventions


def _build_offline_tokenizer() -> PreTrainedTokenizerFast:
    vocab_tokens = [
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
        "neutral",
        "question",
        "reply",
        "evidence",
        "clear",
        "agrees",
        "supports",
        "disagrees",
        "opposes",
        "comment",
    ]
    tok_obj = Tokenizer(
        WordLevel(vocab={tok: i for i, tok in enumerate(vocab_tokens)}, unk_token="<unk>")
    )
    tok_obj.pre_tokenizer = Whitespace()
    tok_obj.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


def _tiny_roberta_config(
    num_labels: int = 3, id2label: dict[int, str] | None = None
) -> RobertaConfig:
    cfg = RobertaConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=1,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        num_labels=num_labels,
    )
    if id2label is not None:
        cfg.id2label = id2label
    return cfg


def _diversify_tiny_cls_model(model: RobertaForSequenceClassification) -> None:
    """Keep logits non-saturated and token-sensitive across distinct sequence lengths."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "word_embeddings.weight" in name:
                torch.nn.init.normal_(param, std=1.0)
            elif "value.weight" in name or "classifier.dense.weight" in name or "out_proj.weight" in name:
                torch.nn.init.normal_(param, std=0.4)
            elif "classifier" in name and name.endswith(".bias"):
                torch.nn.init.zeros_(param)


def test_length_bucketed_nlp_order_invariance_and_parity() -> None:
    """Deliberately unordered mixed short/long inputs must yield identical scores at smaller batch sizes."""
    seed_everything(42)
    tok = _build_offline_tokenizer()

    mixed_replies = [
        "oppose hate trash oppose hate trash oppose hate trash oppose hate trash",
        "agree",
        "neutral question reply evidence clear",
        "oppose civil evidence clear support agree question neutral reply topic",
        "trash",
        "support agree clear evidence support agree clear evidence",
        "neutral",
        "oppose hate trash civil evidence reply",
        "yes",
    ]
    mixed_parents = [
        "root topic",
        "root topic evidence clear support agree oppose neutral question reply",
        "topic",
        "root agree",
        "root topic evidence clear support agree oppose neutral",
        "agree",
        "root topic clear evidence support",
        "oppose",
        "root topic evidence clear",
    ]
    mixed_roots = [
        "root topic evidence clear",
        "root topic",
        None,
        "root topic clear evidence support agree",
        "root",
        "root topic neutral",
        "root",
        "root topic oppose hate",
        None,
    ]

    # 1. HostilityTeacherScorer.score_texts
    host_model = RobertaForSequenceClassification(
        _tiny_roberta_config(3, {0: "clean", 1: "offensive", 2: "hate"})
    )
    _diversify_tiny_cls_model(host_model)
    host_scorer = HostilityTeacherScorer(
        lang="vi", model=host_model, tokenizer=tok, device=torch.device("cpu")
    )
    h_bucketed = host_scorer.score_texts(mixed_replies, batch_size=2)
    h_single = host_scorer.score_texts(mixed_replies, batch_size=len(mixed_replies))
    assert h_bucketed.shape == (len(mixed_replies),)
    assert np.allclose(h_bucketed, h_single, atol=1e-5)
    assert float(np.ptp(h_bucketed)) > 1e-2
    assert not np.allclose(h_bucketed, h_single[::-1], atol=1e-5)

    # 2. NLIStanceTeacherScorer.score_pairs (verbalizer=True and False)
    nli_model = RobertaForSequenceClassification(
        _tiny_roberta_config(3, {0: "contradiction", 1: "neutral", 2: "entailment"})
    )
    _diversify_tiny_cls_model(nli_model)
    for use_verb in (True, False):
        nli_scorer = NLIStanceTeacherScorer(
            model=nli_model,
            tokenizer=tok,
            device=torch.device("cpu"),
            lang="en",
            use_stance_verbalizer=use_verb,
        )
        out_bucketed = nli_scorer.score_pairs(
            mixed_parents, mixed_replies, batch_size=2, lang="en"
        )
        out_single = nli_scorer.score_pairs(
            mixed_parents, mixed_replies, batch_size=len(mixed_replies), lang="en"
        )
        assert np.allclose(out_bucketed["s_uv"], out_single["s_uv"], atol=1e-5)
        assert np.allclose(out_bucketed["stance_probs"], out_single["stance_probs"], atol=1e-5)
        assert out_bucketed["pred_label"] == out_single["pred_label"]
        assert float(np.ptp(out_bucketed["s_uv"])) > 1e-3
        assert not np.allclose(out_bucketed["s_uv"], out_single["s_uv"][::-1], atol=1e-5)

    # 3. encode_node_embeddings
    enc = RobertaModel(_tiny_roberta_config(), add_pooling_layer=False)
    emb_bucketed = encode_node_embeddings(
        mixed_replies,
        encoder=enc,
        tokenizer=tok,
        parent_texts=mixed_parents,
        root_texts=mixed_roots,
        batch_size=2,
        device=torch.device("cpu"),
    )
    emb_single = encode_node_embeddings(
        mixed_replies,
        encoder=enc,
        tokenizer=tok,
        parent_texts=mixed_parents,
        root_texts=mixed_roots,
        batch_size=len(mixed_replies),
        device=torch.device("cpu"),
    )
    assert emb_bucketed.shape == (len(mixed_replies), 32)
    assert torch.allclose(emb_bucketed, emb_single, atol=1e-4)

    # 4. encode_edges_with_cross_encoder (edge_dim=2 and edge_dim=4)
    for edim in (2, 4):
        ce = DualHeadCrossEncoder(
            lang="vi",
            model_name="offline-tiny",
            encoder=RobertaModel(_tiny_roberta_config(), add_pooling_layer=False),
            edge_dim=edim,
            include_root_context=True,
        )
        ce_b = encode_edges_with_cross_encoder(
            mixed_parents,
            mixed_replies,
            cross_encoder=ce,
            tokenizer=tok,
            root_texts=mixed_roots,
            include_root_context=True,
            batch_size=2,
            device=torch.device("cpu"),
        )
        ce_s = encode_edges_with_cross_encoder(
            mixed_parents,
            mixed_replies,
            cross_encoder=ce,
            tokenizer=tok,
            root_texts=mixed_roots,
            include_root_context=True,
            batch_size=len(mixed_replies),
            device=torch.device("cpu"),
        )
        assert np.allclose(ce_b["s_uv"], ce_s["s_uv"], atol=1e-4)
        assert np.allclose(ce_b["h_uv"], ce_s["h_uv"], atol=1e-4)
        assert ce_b["pred_label"] == ce_s["pred_label"]
        if edim == 4:
            assert np.allclose(ce_b["r_uv"], ce_s["r_uv"], atol=1e-4)
            assert np.allclose(ce_b["c_uv"], ce_s["c_uv"], atol=1e-4)
            assert ce_b["behavior_label"] == ce_s["behavior_label"]


def _make_mechanism_test_cascades(num_cascades: int = 8) -> list[CascadeData]:
    seed_everything(123)
    cascades: list[CascadeData] = []
    num_nodes = 9  # root 0 + 8 replies (depths 1 and 2)
    src = [0, 0, 0, 0, 1, 1, 2, 3]
    dst = [1, 2, 3, 4, 5, 6, 7, 8]
    edge_index_td = torch.tensor([src, dst], dtype=torch.long)
    edge_index_bu = torch.tensor([dst, src], dtype=torch.long)

    for idx in range(num_cascades):
        x = torch.randn((num_nodes, TOTAL_NODE_FEATURE_DIM), dtype=torch.float32) * 0.2
        struct_x = torch.rand((num_nodes, STRUCTURAL_NODE_DIM), dtype=torch.float32)
        if idx % 3 == 0:
            # Low conflict
            s_uv = torch.tensor([0.5, 0.4, 0.2, 0.3, 0.5, 0.1, 0.2, 0.3], dtype=torch.float32)
            h_uv = torch.tensor([0.05, 0.08, 0.04, 0.06, 0.05, 0.07, 0.09, 0.05], dtype=torch.float32)
            y_size = 3.1
            y_polar = 0.18
        elif idx % 3 == 1:
            # Civil disagreement
            s_uv = torch.tensor([0.5, -0.5, -0.4, 0.4, -0.6, 0.3, -0.45, 0.2], dtype=torch.float32)
            h_uv = torch.tensor([0.08, 0.10, 0.12, 0.07, 0.09, 0.11, 0.10, 0.08], dtype=torch.float32)
            y_size = 3.7
            y_polar = 0.55
        else:
            # Hostile conflict
            s_uv = torch.tensor([0.7, -0.8, -0.75, 0.6, -0.85, -0.7, 0.5, -0.65], dtype=torch.float32)
            h_uv = torch.tensor([0.4, 0.85, 0.72, 0.35, 0.90, 0.68, 0.45, 0.75], dtype=torch.float32)
            y_size = 4.4
            y_polar = 0.82

        edge_attr = torch.stack([s_uv, h_uv], dim=-1)
        one_sided = min(1.0, abs(float(s_uv.mean())) * float(h_uv.mean()))
        c_stats = torch.tensor(
            [[0.3, 0.2, 0.6, 0.5, 0.1, 0.85, 0.9, one_sided]],
            dtype=torch.float32,
        )
        cascades.append(
            CascadeData(
                x=x,
                struct_x=struct_x,
                edge_index=edge_index_td,
                edge_index_td=edge_index_td,
                edge_index_bu=edge_index_bu,
                edge_attr=edge_attr,
                edge_stance=s_uv.unsqueeze(-1),
                cascade_stats=c_stats,
                root_index=torch.tensor([0], dtype=torch.long),
                y_size=torch.tensor([y_size], dtype=torch.float32),
                y_polar=torch.tensor([y_polar], dtype=torch.float32),
                y_rumor=torch.tensor([idx % 4], dtype=torch.long),
                cascade_id=f"mech_c{idx}",
                topic_id=f"mech_t{idx // 2}",
            )
        )
    return cascades


def test_mechanism_analysis_non_mutating_checkpoint_latex_and_figures(tmp_path: Path) -> None:
    """Verify SBiFAGAT checkpoint loading, non-mutating counterfactuals, clean LaTeX, and fig4."""
    seed_everything(42)
    cascades = _make_mechanism_test_cascades(8)
    orig_ea_clones = [c.edge_attr.clone() for c in cascades]
    orig_stats_clones = [c.cascade_stats.clone() for c in cascades]
    orig_stance_clones = [c.edge_stance.clone() for c in cascades]

    model = SBiFAGAT(
        in_dim=TOTAL_NODE_FEATURE_DIM,
        hidden_dim=16,
        edge_dim=2,
        num_rumor_classes=4,
        use_cascade_stats=True,
        predict_uncertainty=False,
    )
    ckpt_path = tmp_path / "best_sbifagat_vn.pt"
    torch.save(
        {
            "config": {
                "hidden_dim": 16,
                "edge_dim": 2,
                "eps": 0.5,
                "dropout": 0.1,
                "use_cascade_stats": True,
                "predict_uncertainty": False,
            },
            "model_state_dict": model.state_dict(),
            "in_dim": TOTAL_NODE_FEATURE_DIM,
            "num_rumor_classes": 4,
        },
        ckpt_path,
    )

    loaded_model, payload = load_sbifagat_checkpoint(ckpt_path, device=torch.device("cpu"))
    assert payload["config"]["hidden_dim"] == 16
    model.eval()
    with torch.no_grad():
        assert torch.allclose(
            model(cascades[0])["y_polar_pred"],
            loaded_model(cascades[0])["y_polar_pred"],
            atol=1e-6,
        )

    report = run_mechanism_analysis(
        loaded_model,
        cascades,
        dataset_name="vn",
        device=torch.device("cpu"),
        seed=42,
    )

    # Strict non-mutation guarantee on original cascade tensors
    for i, c in enumerate(cascades):
        assert torch.equal(c.edge_attr, orig_ea_clones[i])
        assert torch.equal(c.cascade_stats, orig_stats_clones[i])
        assert torch.equal(c.edge_stance, orig_stance_clones[i])

    assert report["num_test_cascades"] == 8
    assert set(report["discourse_regimes"].keys()) == {
        "low_conflict",
        "civil_disagreement",
        "hostile_conflict",
    }
    assert sum(v["count"] for v in report["discourse_regimes"].values()) == 8

    trig = report["trigger_depth_profile"]
    assert trig["num_trigger_edges"] == 8 * 3
    assert trig["depth1_fraction"] + trig["depth2plus_fraction"] == pytest.approx(1.0)

    assert "early_hostility_vs_y_size" in report["empirical_coupling"]
    assert "future_y_polar_vs_y_size" in report["empirical_coupling"]

    expected_ints = (
        "cool_top3_hostility",
        "cool_all_hostility",
        "neutralize_top3_conflict",
        "neutralize_all_oppose",
    )
    for k in expected_ints:
        assert k in report["interventions"]
        for cohort in ("all_test", "high_polar_q75"):
            stats = report["interventions"][k][cohort]
            assert stats["num_cascades"] >= 1
            for m_key, ci_key in (
                ("mean_delta_y_size", "ci95_delta_y_size"),
                ("mean_delta_y_polar", "ci95_delta_y_polar"),
            ):
                mv = float(stats[m_key])
                lo, hi = float(stats[ci_key][0]), float(stats[ci_key][1])
                assert np.isfinite(mv) and np.isfinite(lo) and np.isfinite(hi)
                assert lo - 1e-9 <= mv <= hi + 1e-9

    tex = format_mechanisms_latex(report, label="tab:mechanisms_vn")
    assert find_unescaped_underscores_in_latex(tex) == []

    pdf_p, svg_p = plot_counterfactual_interventions(
        report["interventions"],
        output_dir=tmp_path / "figs",
        stem="fig4_counterfactual_interventions",
    )
    assert pdf_p.is_file() and pdf_p.stat().st_size > 0
    assert svg_p.is_file() and svg_p.stat().st_size > 0


def test_multitask_loss_balance_polar_decoupling_and_cleanres_dipole(tmp_path: Path) -> None:
    """Verify 2:1 baseline-normalized loss scaling, cross-model polar_head decoupling, CleanRes Camp Dipole, and LaTeX & escaping."""
    seed_everything(2024)
    torch.manual_seed(2024)

    # 1. Baseline-normalized 2:1 loss balance + 1e-3 polar bias clamp
    graphs: list[CascadeData] = []
    size_targets = [2.5, 3.1, 4.0, 4.8]
    polar_targets = [0.002, 0.008, 0.015, 0.025]
    for ys, yp in zip(size_targets, polar_targets):
        ei_td = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long)
        graphs.append(
            CascadeData(
                x=torch.randn((4, TOTAL_NODE_FEATURE_DIM)),
                struct_x=torch.rand((4, STRUCTURAL_NODE_DIM)),
                edge_index=ei_td,
                edge_index_td=ei_td,
                edge_index_bu=ei_td[[1, 0], :],
                edge_attr=torch.tensor([[0.6, 0.1], [-0.5, 0.7], [0.2, 0.2]], dtype=torch.float32),
                edge_stance=torch.tensor([[0.6], [-0.5], [0.2]], dtype=torch.float32),
                cascade_stats=torch.rand((1, 8), dtype=torch.float32),
                root_index=torch.tensor([0], dtype=torch.long),
                y_size=torch.tensor([ys], dtype=torch.float32),
                y_polar=torch.tensor([yp], dtype=torch.float32),
                y_rumor=torch.tensor([0], dtype=torch.long),
            )
        )

    loader = DataLoader(graphs, batch_size=4, shuffle=False)
    stats = _estimate_train_target_stats(loader)
    assert stats["size_loss_scale"] * stats["base_size_loss"] == pytest.approx(1.0, rel=1e-4)
    assert stats["polar_loss_scale"] * stats["base_polar_loss"] == pytest.approx(0.5, rel=1e-4)
    assert stats["polar_loss_scale"] > 10.0

    ref_model = SBiFAGAT(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2, use_cascade_stats=True)
    _init_multitask_head_biases(ref_model, stats)
    init_polar_prob = torch.sigmoid(ref_model.head.polar_head[-1].bias).item()
    assert init_polar_prob == pytest.approx(float(np.mean(polar_targets)), abs=1e-4)

    # 2. Fair cross-model polar_head decoupling from cascade_stats across all compared models
    batch = next(iter(loader))
    batch_alt = batch.clone()
    batch_alt.cascade_stats = batch.cascade_stats.clone() * 0.1 + 5.0
    all_models = (
        TextMLP(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, use_cascade_stats=True),
        TemporalStanceMLP(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2, use_cascade_stats=True),
        StandardGCN(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, use_cascade_stats=True),
        StandardGATv2(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, use_cascade_stats=True),
        BiGCN(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, use_cascade_stats=True),
        EdgeGATv2(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2, use_cascade_stats=True),
        BiEdgeGATv2(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2, use_cascade_stats=True),
        FAGCNBaseline(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, use_cascade_stats=True),
        ref_model,
    )
    for model in all_models:
        model.eval()
        with torch.no_grad():
            o1 = model(batch)
            o2 = model(batch_alt)
        assert torch.allclose(o1["y_polar_pred"], o2["y_polar_pred"], atol=1e-6)

    # 3. CleanRes Camp Dipole init scales, ablation toggle & full parameter gradient flow
    assert ref_model.td_convs[0].cos_weight.item() == pytest.approx(0.10, abs=1e-6)
    assert ref_model.dir_mix_scale is not None
    assert torch.allclose(ref_model.dir_mix_scale.detach(), torch.full((ref_model.num_layers,), 0.10))
    assert ref_model.dipole_scale is not None
    assert ref_model.dipole_scale.item() == pytest.approx(0.25, abs=1e-6)

    ref_model.train()
    ref_model.zero_grad()
    out_grad = ref_model(batch)
    loss_grad = out_grad["y_size_pred"].sum() + out_grad["y_polar_pred"].sum() + out_grad["rumor_logits"].sum()
    loss_grad.backward()
    for p_name, p in ref_model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Missing/non-finite grad for {p_name}"

    no_dip = SBiFAGAT(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=32, edge_dim=2, use_camp_dipole=False, use_cascade_stats=True)
    assert no_dip.dipole_proj is None and no_dip.dipole_scale is None
    no_dip.train()
    out_nd = no_dip(batch)
    (out_nd["y_size_pred"].sum() + out_nd["y_polar_pred"].sum() + out_nd["rumor_logits"].sum()).backward()
    for p_name, p in no_dip.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Missing grad in no_dip for {p_name}"

    # 4. LaTeX ampersand escaping in export_latex_table
    tex_path = tmp_path / "tab_escape.tex"
    dummy_metric = {"r2_size": (0.4, 0.02), "r2_polar": (0.2, 0.01)}
    export_latex_table(
        {
            "reference_model": "SBiFAGAT",
            "comparator_model": "BiEdgeGATv2 & Base",
            "summary_stats": {
                "SBiFAGAT": dummy_metric,
                "w/o Camp Dipole & Moments": {"r2_size": (0.3, 0.02), "r2_polar": (0.1, 0.01)},
            },
            "p_values": {"r2_size": 0.01, "r2_polar": 0.02},
        },
        tex_path,
        caption="Ablation \\& Comparison.",
        label="tab:escape",
        metrics=(("r2_size", "$R^2$", False), ("r2_polar", "$R^2_p$", False)),
    )
    tex_txt = tex_path.read_text(encoding="utf-8")
    assert "w/o Camp Dipole \\& Moments" in tex_txt
    assert "BiEdgeGATv2 \\& Base" in tex_txt
    assert find_unescaped_underscores_in_latex(tex_txt) == []


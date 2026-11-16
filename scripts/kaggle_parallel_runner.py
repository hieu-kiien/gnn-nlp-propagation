"""Unified Kaggle T4 x 2 (32 GB VRAM) Parallel Training & Self-Auditing Loop Runner.

Supports 4 clean parallel execution roles across two teammates:
  - `member1_vn`: Vietnamese track end-to-end (Arctic Shift 1,500 VN trees -> Qwen3.8-27B
    QLoRA -> 4D ViSoBERT distillation -> nlp_features.pt -> SBi-FAGAT v2 auto-loop +
    5-seed benchmark + ablation + output bundle zip).
  - `member2_en`: International English track end-to-end (PHEME MD5 -> Qwen3.8-27B QLoRA ->
    4D DeBERTa-v3-base distillation -> nlp_features.pt -> SBi-FAGAT v2 auto-loop +
    5-seed Inductive & LOO benchmarks + ablation + optional UPFD + output bundle zip).
  - `model_nlp_only`: Runs strictly Model 1 (Qwen3.8-27B QLoRA) + Model 2 (System-1 4D
    Cross-Encoder distillation & `.pt` feature export) so Member 1 can focus on NLP.
  - `model_gnn_only`: Runs strictly Model 3 (SBi-FAGAT v2 self-auditing loop + 7 baselines x
    5 seeds + ablation + figures) from precomputed or fast-bootstrapped features so Member 2
    can tune/train GNN models immediately in parallel.

Also provides `--smoke-test` verifying the entire 4D + v2 self-auditing pipeline on CPU in <10s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from torch_geometric.loader import DataLoader

from src.data.pyg_dataset import EarlyCascadeDataset, TOTAL_NODE_FEATURE_DIM
from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet
from src.models.baselines import BiGCN, TemporalStanceMLP, TextMLP
from src.models.sbi_fagat import SBiFAGAT
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    distill_dual_cross_encoder,
    export_stratified_human_audit_csv,
    extract_parent_reply_pairs_from_parquet,
    split_pairs_by_inductive_topics,
)
from src.nlp.extract_features import extract_and_save_features
from src.nlp.qwen_teacher import (
    EN_DEBERTA_MODEL_NAME,
    QWEN_TEACHER_MODEL_ID,
    QwenBehavioralTeacherScorer,
)
from src.training.auto_loop import run_self_auditing_validation_loop
from src.training.reporting import serialize_benchmark_json
from src.training.trainer import run_multi_seed_benchmark


def _run_cmd(cmd: Sequence[str], *, cwd: Path) -> None:
    print(f"[KaggleRunner] Running: {' '.join(cmd)}")
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def _bundle_outputs(
    bundle_path: Path,
    *,
    source_paths: Sequence[Path],
    base_dir: Path,
) -> Path:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_paths:
            if not path.exists():
                continue
            if path.is_file():
                arc = path.relative_to(base_dir) if path.is_relative_to(base_dir) else Path(path.name)
                zf.write(path, arcname=str(arc))
            elif path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file():
                        arc = child.relative_to(base_dir) if child.is_relative_to(base_dir) else child.name
                        zf.write(child, arcname=str(arc))
    return bundle_path


def _build_smoke_offline_tokenizer_and_encoder() -> tuple[Any, Any]:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

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
        "fallacy",
        "neutral",
        "question",
        "reply",
    ]
    tok_obj = Tokenizer(WordLevel(vocab={t: i for i, t in enumerate(vocab_tokens)}, unk_token="<unk>"))
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


def _build_smoke_parquet_dataset(interim_dir: Path, *, lang: str) -> None:
    base_ts = datetime(2026, 7, 1, 8, 0, 0, tzinfo=timezone.utc)
    topics = ["topic_alpha", "topic_alpha", "topic_beta", "topic_beta", "topic_gamma", "topic_gamma"]
    rumors = ["true", "false", "non-rumor", "true", "false", "non-rumor"]
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    for c_idx, (topic_id, r_lbl) in enumerate(zip(topics, rumors)):
        cid = f"smoke_c{c_idx}"
        root_id = f"{cid}_r"
        node_rows.append(
            {
                "node_id": root_id,
                "cascade_id": cid,
                "topic_id": topic_id,
                "parent_id": None,
                "is_root": True,
                "text": f"root topic claim {c_idx} debate",
                "created_at_utc": base_ts + timedelta(hours=c_idx),
                "lang": lang,
                "source_dataset": "pheme" if lang == "en" else "arctic_shift_vozforums",
                "user_id_hash": hashlib.sha256(f"u_r_{c_idx}".encode()).hexdigest(),
                "rumor_label": r_lbl if lang == "en" else None,
            }
        )
        n_replies = 23 + (c_idx % 3) * 2
        for r_idx in range(1, n_replies + 1):
            nid = f"{cid}_n{r_idx}"
            pid = root_id if r_idx <= 8 else f"{cid}_n{(r_idx - 1) // 2 + 1}"
            dt_sec = float(r_idx * (35 if c_idx % 2 == 0 else 120))
            if r_idx % 4 == 0:
                txt = "agree support clear evidence for this point in thread"
            elif r_idx % 4 == 1:
                txt = "oppose hate trash idiotic illogical take on this topic"
            elif r_idx % 4 == 2:
                txt = "oppose civil counter argument with data question reply"
            else:
                txt = "neutral question reply clarification"
            node_rows.append(
                {
                    "node_id": nid,
                    "cascade_id": cid,
                    "topic_id": topic_id,
                    "parent_id": pid,
                    "is_root": False,
                    "text": txt,
                    "created_at_utc": base_ts + timedelta(hours=c_idx, seconds=dt_sec),
                    "lang": lang,
                    "source_dataset": "pheme" if lang == "en" else "arctic_shift_vozforums",
                    "user_id_hash": hashlib.sha256(f"u_{c_idx}_{r_idx % 9}".encode()).hexdigest(),
                    "rumor_label": None,
                }
            )
            edge_rows.append(
                {
                    "edge_id": f"{pid}->{nid}",
                    "cascade_id": cid,
                    "source_id": pid,
                    "target_id": nid,
                    "created_at_utc": base_ts + timedelta(hours=c_idx, seconds=dt_sec),
                    "time_delay_sec": dt_sec,
                    "edge_type": "reply",
                }
            )

    save_cascade_parquet(
        build_nodes_table(node_rows),
        build_edges_table(edge_rows),
        output_dir=interim_dir,
        validate=True,
    )


def _mock_qwen_generate(prompts: Sequence[str]) -> list[str]:
    outputs: list[str] = []
    for p in prompts:
        low = p.lower()
        if "hate" in low or "trash" in low:
            obj = {
                "label": "hostile_conflict",
                "s_uv": -0.88,
                "h_uv": 0.90,
                "r_uv": 0.72,
                "c_uv": 0.05,
                "probs": [0.01, 0.02, 0.05, 0.14, 0.78],
            }
        elif "civil" in low:
            obj = {
                "label": "constructive_oppose",
                "s_uv": -0.72,
                "h_uv": 0.08,
                "r_uv": 0.06,
                "c_uv": 0.88,
                "probs": [0.03, 0.08, 0.82, 0.05, 0.02],
            }
        elif "agree" in low or "support" in low:
            obj = {
                "label": "support",
                "s_uv": 0.80,
                "h_uv": 0.04,
                "r_uv": 0.03,
                "c_uv": 0.85,
                "probs": [0.84, 0.08, 0.04, 0.03, 0.01],
            }
        else:
            obj = {
                "label": "neutral",
                "s_uv": 0.0,
                "h_uv": 0.05,
                "r_uv": 0.05,
                "c_uv": 0.55,
                "probs": [0.06, 0.82, 0.06, 0.04, 0.02],
            }
        outputs.append(f"<think>short reasoning check</think>\n{json.dumps(obj)}")
    return outputs


def _run_smoke_pipeline(role: str, lang: str, work_dir: Path) -> dict[str, Any]:
    ds_name = "pheme" if lang == "en" else "vn"
    interim_dir = work_dir / "data" / "interim" / ds_name
    processed_dir = work_dir / "data" / "processed" / ds_name
    ckpt_dir = work_dir / "checkpoints" / f"dual_ce_{ds_name}_v2"
    out_dir = work_dir / "outputs" / role

    _build_smoke_parquet_dataset(interim_dir, lang=lang)
    pairs = extract_parent_reply_pairs_from_parquet(interim_dir)

    scorer = QwenBehavioralTeacherScorer(lang=lang, custom_generate_fn=_mock_qwen_generate)
    out = scorer.score_triplets(
        [p["root_text"] for p in pairs],
        [p["parent_text"] for p in pairs],
        [p["reply_text"] for p in pairs],
        batch_size=32,
        lang=lang,
    )
    for idx, p in enumerate(pairs):
        p["s_uv"] = float(out["s_uv"][idx])
        p["h_uv"] = float(out["h_uv"][idx])
        p["r_uv"] = float(out["r_uv"][idx])
        p["c_uv"] = float(out["c_uv"][idx])
        p["behavior_probs"] = out["behavior_probs"][idx].tolist()
        p["behavior_label"] = out["behavior_label"][idx]
        p["stance_probs"] = out["stance_probs"][idx].tolist()
        p["pred_label"] = out["pred_label"][idx]
        p["root_s_uv"] = float(out["s_uv"][idx])

    cache_path = interim_dir / "scored_pairs.json"
    cache_path.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")

    audit_csv = interim_dir / f"human_audit_{lang}_300.csv"
    export_stratified_human_audit_csv(
        pairs,
        output_csv=audit_csv,
        machine_key_csv=interim_dir / f"human_audit_{lang}_300_machine_key.csv",
        pilot_csv=interim_dir / f"pilot_calibration_{lang}_30.csv",
        n_samples=25,
        pilot_samples=10,
    )

    tok, enc = _build_smoke_offline_tokenizer_and_encoder()
    topic_splits = split_pairs_by_inductive_topics(pairs, split_seed=42)
    ce_model = DualHeadCrossEncoder(lang=lang, model_name="smoke-tiny-ce", encoder=enc, edge_dim=4)
    distill_report = distill_dual_cross_encoder(
        topic_splits["train"],
        topic_splits["val"],
        output_dir=ckpt_dir,
        lang=lang,
        model=ce_model,
        tokenizer=tok,
        edge_dim=4,
        include_root_context=True,
        use_multi_gpu=False,
        epochs=1,
        batch_size=8,
        device=torch.device("cpu"),
    )

    pt_path = extract_and_save_features(
        dataset=ds_name,
        interim_dir=interim_dir,
        output_dir=processed_dir,
        cross_encoder_ckpt=ckpt_dir,
        node_encoder=enc,
        tokenizer=tok,
        scored_cache_json=cache_path,
        edge_dim=4,
        include_root_context=True,
        batch_size=16,
        device=torch.device("cpu"),
    )

    summary: dict[str, Any] = {
        "mode": "smoke_test",
        "role": role,
        "lang": lang,
        "distill_report": distill_report,
        "nlp_features_pt": str(pt_path),
    }

    if role != "model_nlp_only":
        train_ds = EarlyCascadeDataset(
            root_dir=processed_dir, k_horizon=20, split="train", min_total_replies=21, polar_target_mode="future_only"
        )
        val_ds = EarlyCascadeDataset(
            root_dir=processed_dir, k_horizon=20, split="val", min_total_replies=21, polar_target_mode="future_only"
        )
        test_ds = EarlyCascadeDataset(
            root_dir=processed_dir, k_horizon=20, split="test", min_total_replies=21, polar_target_mode="future_only"
        )
        tr_l = DataLoader(train_ds, batch_size=2, shuffle=True)
        va_l = DataLoader(val_ds, batch_size=2, shuffle=False)
        te_l = DataLoader(test_ds, batch_size=2, shuffle=False)

        num_cls = 4
        loop_res = run_self_auditing_validation_loop(
            tr_l,
            va_l,
            in_dim=TOTAL_NODE_FEATURE_DIM,
            edge_dim=4,
            num_rumor_classes=num_cls,
            use_cascade_stats=True,
            predict_uncertainty=True,
            candidates=[
                {"hidden_dim": 64, "lr": 1e-3, "eps": 0.5, "dropout": 0.1, "lambda_uncertainty": 0.05},
                {"hidden_dim": 32, "lr": 5e-4, "eps": 0.4, "dropout": 0.1, "lambda_uncertainty": 0.05},
            ],
            epochs=2,
            patience=2,
            output_dir=out_dir / "auto_loop",
            device=torch.device("cpu"),
        )
        best_cfg = loop_res["best_config"]
        h_dim = int(best_cfg["hidden_dim"])
        bench_res = run_multi_seed_benchmark(
            {
                "TextMLP": lambda: TextMLP(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=h_dim, num_rumor_classes=num_cls),
                "TemporalStanceMLP": lambda: TemporalStanceMLP(
                    in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=h_dim, edge_dim=4, num_rumor_classes=num_cls
                ),
                "BiGCN": lambda: BiGCN(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=h_dim, num_rumor_classes=num_cls),
                "SBiFAGAT": lambda: SBiFAGAT(
                    in_dim=TOTAL_NODE_FEATURE_DIM,
                    hidden_dim=h_dim,
                    edge_dim=4,
                    num_rumor_classes=num_cls,
                    eps=float(best_cfg["eps"]),
                    dropout=float(best_cfg["dropout"]),
                    use_cascade_stats=True,
                    predict_uncertainty=True,
                ),
            },
            tr_l,
            va_l,
            te_l,
            seeds=[42, 100],
            epochs=2,
            patience=2,
            lambda_uncertainty=float(best_cfg["lambda_uncertainty"]),
            device=torch.device("cpu"),
        )
        res_json = out_dir / "results_main_smoke.json"
        serialize_benchmark_json(bench_res, res_json, extra={"auto_loop_best_config": best_cfg})
        summary["auto_loop"] = {
            "best_config": best_cfg,
            "diagnostics_json": loop_res["diagnostics_json"],
            "worst_cases_json": loop_res["worst_cases_json"],
        }
        summary["results_main_json"] = str(res_json)

    summary_path = work_dir / f"runner_summary_{role}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    bundle_path = _bundle_outputs(
        work_dir / f"sbifagat_{role}_bundle.zip",
        source_paths=[summary_path, processed_dir, out_dir, ckpt_dir, audit_csv],
        base_dir=work_dir,
    )
    summary["bundle_zip"] = str(bundle_path)
    return summary


def run_kaggle_parallel_role(args: argparse.Namespace) -> dict[str, Any]:
    lang = args.lang or ("en" if args.role == "member2_en" else "vi")
    ds_name = "pheme" if lang == "en" else "vn"

    if args.smoke_test:
        if args.work_dir is not None:
            args.work_dir.mkdir(parents=True, exist_ok=True)
            return _run_smoke_pipeline(args.role, lang, args.work_dir)
        with tempfile.TemporaryDirectory(prefix="sbifagat_smoke_") as tmp:
            return _run_smoke_pipeline(args.role, lang, Path(tmp))

    work_dir = Path(args.work_dir or ".").resolve()
    py = sys.executable

    interim_dir = work_dir / "data" / "interim" / ds_name
    processed_dir = Path(args.processed_dir) if args.processed_dir else (work_dir / "data" / "processed" / ds_name)
    scored_cache = interim_dir / "scored_pairs.json"
    ce_ckpt = work_dir / "checkpoints" / f"dual_ce_{ds_name}_v2"
    qlora_dir = work_dir / "checkpoints" / f"qwen38_27b_qlora_{lang}"
    audit_csv = work_dir / "data" / "interim" / f"human_audit_{lang}_300.csv"
    pilot_csv = work_dir / "data" / "interim" / f"pilot_calibration_{lang}_30.csv"
    student_backbone = "uitnlp/visobert" if lang == "vi" else EN_DEBERTA_MODEL_NAME

    run_nlp = args.role in {"member1_vn", "member2_en", "model_nlp_only"}
    run_gnn = args.role in {"member1_vn", "member2_en", "model_gnn_only"}

    if args.role == "model_gnn_only" and not (processed_dir / "nlp_features.pt").exists():
        if not args.bootstrap_if_missing:
            raise FileNotFoundError(
                f"Missing '{processed_dir / 'nlp_features.pt'}' for model_gnn_only. "
                "Provide --processed-dir from Member 1 or pass --bootstrap-if-missing."
            )
        run_nlp = True

    if run_nlp:
        # Stage 1: Real dataset acquisition
        if lang == "en":
            _run_cmd([py, "-m", "src.data.pheme_loader", "--output-dir", str(interim_dir)], cwd=work_dir)
        else:
            crawl_cmd = [
                py,
                "-m",
                "src.data.arctic_shift_vn",
                "--output-dir",
                str(interim_dir),
                "--target-count",
                str(args.vn_target_count),
                "--checkpoint-jsonl",
                str(interim_dir / "crawl_checkpoint.jsonl"),
                "--allow-partial",
            ]
            if args.before_epoch is not None:
                crawl_cmd.extend(["--before-epoch", str(args.before_epoch)])
            _run_cmd(crawl_cmd, cwd=work_dir)

        # Stage 2: Qwen3.8-27B QLoRA System-2 -> Student System-1 (zero test-topic leakage)
        use_legacy_bootstrap = bool(args.role == "model_gnn_only" and args.bootstrap_if_missing and args.skip_qlora)
        scorer_cmd = [
            py,
            "-m",
            "src.nlp.dual_edge_scorer",
            "--lang",
            lang,
            "--interim-dir",
            str(interim_dir),
            "--output-dir",
            str(ce_ckpt),
            "--scored-cache-json",
            str(scored_cache),
            "--audit-csv",
            str(audit_csv),
            "--pilot-csv",
            str(pilot_csv),
            "--teacher-backend",
            "legacy_nli" if use_legacy_bootstrap else "qwen_qlora",
            "--teacher-model",
            args.teacher_model,
            "--qlora-adapter-dir",
            str(qlora_dir),
            "--edge-dim",
            "4" if not use_legacy_bootstrap else "2",
            "--student-backbone",
            student_backbone,
            "--epochs",
            str(args.epochs_nlp),
            "--batch-size",
            str(args.batch_size_nlp),
            "--include-root-context",
            "--score-only",
            "--export-audit",
            "--distill",
        ]
        if lang == "vi":
            scorer_cmd.append("--train-vihsd")
        if not args.skip_qlora and not use_legacy_bootstrap:
            scorer_cmd.append("--train-qwen-qlora")
        _run_cmd(scorer_cmd, cwd=work_dir)

        # Stage 3: Frozen 768-d node embeddings + 4D edge attributes
        extract_cmd = [
            py,
            "-m",
            "src.nlp.extract_features",
            "--dataset",
            ds_name,
            "--interim-dir",
            str(interim_dir),
            "--output-dir",
            str(processed_dir),
            "--scored-cache-json",
            str(scored_cache),
            "--cross-encoder-ckpt",
            str(ce_ckpt),
            "--edge-dim",
            "4" if not use_legacy_bootstrap else "2",
            "--node-backbone",
            student_backbone,
            "--k-horizon",
            str(args.k_horizon),
            "--batch-size",
            str(args.batch_size_nlp * 2),
            "--include-root-context",
        ]
        _run_cmd(extract_cmd, cwd=work_dir)

    best_cfg: dict[str, Any] = {
        "hidden_dim": args.hidden_dim,
        "eps": 0.5,
        "dropout": 0.1,
        "lambda_uncertainty": 0.05,
        "lr": 1e-3,
    }
    loop_res: dict[str, Any] | None = None

    if run_gnn:
        train_ds = EarlyCascadeDataset(
            root_dir=processed_dir,
            k_horizon=args.k_horizon,
            split="train",
            min_total_replies=args.k_horizon + 1,
            polar_target_mode="future_only",
        )
        val_ds = EarlyCascadeDataset(
            root_dir=processed_dir,
            k_horizon=args.k_horizon,
            split="val",
            min_total_replies=args.k_horizon + 1,
            polar_target_mode="future_only",
        )
        inferred_edge_dim = (
            int(train_ds[0].edge_attr.size(1))
            if len(train_ds) > 0 and train_ds[0].edge_attr.ndim == 2
            else 4
        )
        num_cls = 4
        loop_res = run_self_auditing_validation_loop(
            DataLoader(train_ds, batch_size=args.batch_size_gnn, shuffle=True),
            DataLoader(val_ds, batch_size=args.batch_size_gnn, shuffle=False),
            in_dim=TOTAL_NODE_FEATURE_DIM,
            edge_dim=inferred_edge_dim,
            num_rumor_classes=num_cls,
            use_cascade_stats=True,
            predict_uncertainty=True,
            epochs=min(60, args.epochs_gnn),
            patience=10,
            output_dir=work_dir / "outputs" / f"auto_loop_{ds_name}",
        )
        best_cfg = loop_res["best_config"]

        # Main 5-seed benchmark (Inductive)
        exp_cmd = [
            py,
            "-m",
            "scripts.run_experiments",
            "--data-dir",
            str(processed_dir),
            "--k-horizon",
            str(args.k_horizon),
            "--hidden-dim",
            str(best_cfg["hidden_dim"]),
            "--edge-dim",
            str(inferred_edge_dim),
            "--eps",
            str(best_cfg["eps"]),
            "--dropout",
            str(best_cfg["dropout"]),
            "--lr",
            str(best_cfg["lr"]),
            "--epochs",
            str(args.epochs_gnn),
            "--batch-size",
            str(args.batch_size_gnn),
            "--v2",
        ]
        _run_cmd(exp_cmd, cwd=work_dir)

        if lang == "en":
            _run_cmd(exp_cmd + ["--eval-protocol", "loo", "--output-table", f"paper/tables_loo_{ds_name}.tex", "--output-json", f"paper/results_loo_{ds_name}.json", "--skip-figures"], cwd=work_dir)

        # Full component & K-horizon ablations
        abl_cmd = [
            py,
            "-m",
            "scripts.run_ablation",
            "--data-dir",
            str(processed_dir),
            "--k-horizon",
            str(args.k_horizon),
            "--hidden-dim",
            str(best_cfg["hidden_dim"]),
            "--edge-dim",
            str(inferred_edge_dim),
            "--epochs",
            str(args.epochs_gnn),
            "--batch-size",
            str(args.batch_size_gnn),
            "--v2",
        ]
        _run_cmd(abl_cmd, cwd=work_dir)

    bundle_sources = [
        processed_dir,
        work_dir / "paper",
        work_dir / "outputs",
        work_dir / "data" / "interim",
        ce_ckpt,
        qlora_dir,
    ]
    bundle_zip = _bundle_outputs(
        work_dir / f"sbifagat_{args.role}_bundle.zip",
        source_paths=bundle_sources,
        base_dir=work_dir,
    )
    return {
        "mode": "kaggle_t4x2",
        "role": args.role,
        "lang": lang,
        "best_config": best_cfg,
        "auto_loop": loop_res,
        "bundle_zip": str(bundle_zip),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle T4 x 2 Parallel Runner & Loop Orchestrator.")
    parser.add_argument(
        "--role",
        choices=["member1_vn", "member2_en", "model_nlp_only", "model_gnn_only"],
        required=True,
    )
    parser.add_argument("--lang", choices=["vi", "en"], default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--vn-target-count", type=int, default=1500)
    parser.add_argument("--before-epoch", type=int, default=None)
    parser.add_argument("--teacher-model", type=str, default=QWEN_TEACHER_MODEL_ID)
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs-qlora", type=int, default=1)
    parser.add_argument("--epochs-nlp", type=int, default=3)
    parser.add_argument("--epochs-gnn", type=int, default=100)
    parser.add_argument("--batch-size-nlp", type=int, default=16)
    parser.add_argument("--batch-size-gnn", type=int, default=32)
    parser.add_argument("--skip-qlora", action="store_true")
    parser.add_argument("--bootstrap-if-missing", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    res = run_kaggle_parallel_role(args)
    print(json.dumps({k: v for k, v in res.items() if k != "auto_loop"}, indent=2, default=str))


if __name__ == "__main__":
    main()

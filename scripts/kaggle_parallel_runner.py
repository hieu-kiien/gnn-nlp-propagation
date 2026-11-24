"""Unified Kaggle T4 x 2 (32 GB VRAM) Parallel Training & Self-Auditing Loop Runner.

Supports 4 clean parallel execution roles across two teammates using the unified 2D NLP + GNN pipeline:
  - `member1_vn`: Vietnamese track end-to-end (Arctic Shift 1,500 VN trees -> ViSoBERT-ViHSD +
    mDeBERTa Stance Verbalizer -> 2D ViSoBERT Cross-Encoder distillation -> Ancestor SpanPool
    nlp_features.pt -> SBi-FAGAT auto-loop + 5-seed benchmark + ablation + output bundle zip).
  - `member2_en`: International English track end-to-end (PHEME MD5 -> Twitter-RoBERTa-Offensive +
    mDeBERTa Stance Verbalizer -> 2D Twitter-RoBERTa Cross-Encoder distillation -> Ancestor SpanPool
    nlp_features.pt -> SBi-FAGAT auto-loop + 5-seed Inductive & LOO benchmarks + ablation + UPFD).
  - `model_nlp_only`: Runs strictly Stage 1-3 NLP teacher scoring, Cross-Encoder distillation, and
    `.pt` feature extraction so Member 1 can focus on NLP.
  - `model_gnn_only`: Runs strictly SBi-FAGAT auto-loop + 7 baselines x 5 seeds + ablation + figures
    from precomputed or bootstrapped features so Member 2 can tune/train GNN models in parallel.

Also provides `--smoke-test` verifying the entire 2D + SpanPool + self-auditing pipeline on CPU in <10s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch_geometric.loader import DataLoader

from src.data.pyg_dataset import (
    DEFAULT_FORECAST_MIN_REPLIES,
    EarlyCascadeDataset,
    TOTAL_NODE_FEATURE_DIM,
    compute_canonical_topic_splits,
)
from src.data.schema import build_edges_table, build_nodes_table, save_cascade_parquet
from src.models.baselines import BiGCN, TemporalStanceMLP, TextMLP
from src.models.sbi_fagat import SBiFAGAT
from src.nlp.dual_edge_scorer import (
    DualHeadCrossEncoder,
    TWITTER_ROBERTA_MODEL_NAME,
    VISOBERT_MODEL_NAME,
    distill_dual_cross_encoder,
    export_stratified_human_audit_csv,
    extract_parent_reply_pairs_from_parquet,
    split_pairs_by_inductive_topics,
)
from src.nlp.extract_features import extract_and_save_features
from src.training.auto_loop import run_self_auditing_validation_loop
from src.training.reporting import serialize_benchmark_json
from src.training.trainer import run_multi_seed_benchmark


def _run_cmd(cmd: Sequence[str], *, cwd: Path) -> None:
    print(f"[KaggleRunner] Running: {' '.join(cmd)}", flush=True)
    repo_root = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    existing_pp = env.get("PYTHONPATH", "")
    parts = [p for p in existing_pp.split(os.pathsep) if p] if existing_pp else []
    if repo_root not in parts:
        parts.insert(0, repo_root)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    subprocess.run(list(cmd), cwd=str(cwd), env=env, check=True)


def _bundle_outputs(
    bundle_path: Path,
    *,
    source_paths: Sequence[Path],
    base_dir: Path,
    include_heavy_weights: bool = False,
    include_nlp_features: bool = False,
) -> Path:
    """Package results into a compact archive, skipping multi-GB caches/weights unless requested."""
    heavy_suffixes = {".safetensors", ".bin", ".jsonl", ".tmp"}
    heavy_dir_tags = ("dual_ce", "visobert", "roberta", "deberta")

    def _should_skip(file_path: Path) -> bool:
        low_name = file_path.name.lower()
        # Unconditionally exclude blind-audit machine answer keys and secrets
        if (
            low_name.endswith("_machine_key.csv")
            or low_name.endswith(".secret")
            or low_name == ".env"
        ):
            return True
        if include_heavy_weights:
            return False
        if (
            low_name == "scored_pairs.json"
            or low_name.endswith(".state.json")
            or file_path.suffix.lower() in heavy_suffixes
        ):
            return True
        if low_name == "nlp_features.pt" and not include_nlp_features:
            return True
        if file_path.suffix.lower() == ".pt" and "sbifagat" not in low_name and "nlp_features" not in low_name:
            try:
                if file_path.stat().st_size > 20 * 1024 * 1024:
                    return True
            except OSError:
                pass
        try:
            rel_parts = [p.lower() for p in file_path.relative_to(base_dir).parts]
        except ValueError:
            rel_parts = [p.lower() for p in file_path.parts]
        if any(any(tag in part for tag in heavy_dir_tags) for part in rel_parts):
            if file_path.suffix.lower() not in {".json", ".csv", ".txt", ".yaml", ".yml"}:
                return True
        return False

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    seen_arcs: set[str] = set()
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_paths:
            if not path.exists():
                continue
            if path.is_file() and not _should_skip(path):
                arc = str(path.relative_to(base_dir) if path.is_relative_to(base_dir) else Path(path.name))
                if arc not in seen_arcs:
                    seen_arcs.add(arc)
                    zf.write(path, arcname=arc)
            elif path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file() and not _should_skip(child):
                        arc = str(child.relative_to(base_dir) if child.is_relative_to(base_dir) else Path(child.name))
                        if arc not in seen_arcs:
                            seen_arcs.add(arc)
                            zf.write(child, arcname=arc)
    return bundle_path


def _bundle_sealed_machine_keys(
    sealed_zip_path: Path,
    *,
    search_roots: Sequence[Path],
    base_dir: Path,
) -> Path | None:
    """Package `*_machine_key.csv` answer-key files into a separate sealed zip archive."""
    seen: set[Path] = set()
    key_files: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else sorted(root.rglob("*_machine_key.csv"))
        for child in candidates:
            if not child.is_file() or not child.name.lower().endswith("_machine_key.csv"):
                continue
            resolved = child.resolve()
            if resolved not in seen:
                seen.add(resolved)
                key_files.append(child)

    if not key_files:
        return None

    sealed_zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(sealed_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for child in key_files:
            arc = child.relative_to(base_dir) if child.is_relative_to(base_dir) else Path(child.name)
            zf.write(child, arcname=str(arc))
    return sealed_zip_path


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
        n_replies = 25 + (c_idx % 3) * 2
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


def _score_smoke_pairs_offline(pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically score smoke-test edges in 2D (`[s_uv, h_uv]`) with zero network calls."""
    scored: list[dict[str, Any]] = []
    for p in pairs:
        low = str(p.get("reply_text") or "").lower()
        if "hate" in low or "trash" in low:
            s_val, h_val, lbl, probs = -0.88, 0.90, "oppose", [0.02, 0.08, 0.90]
        elif "civil" in low or "oppose" in low:
            s_val, h_val, lbl, probs = -0.72, 0.08, "oppose", [0.04, 0.14, 0.82]
        elif "agree" in low or "support" in low:
            s_val, h_val, lbl, probs = 0.80, 0.04, "support", [0.86, 0.08, 0.06]
        else:
            s_val, h_val, lbl, probs = 0.0, 0.05, "neutral", [0.08, 0.84, 0.08]
        item = dict(p)
        item["s_uv"] = float(s_val)
        item["h_uv"] = float(h_val)
        item["stance_probs"] = list(probs)
        item["pred_label"] = lbl
        item["root_s_uv"] = float(s_val)
        item["root_stance_probs"] = list(probs)
        scored.append(item)
    return scored


def _run_smoke_pipeline(role: str, lang: str, work_dir: Path) -> dict[str, Any]:
    ds_name = "pheme" if lang == "en" else "vn"
    interim_dir = work_dir / "data" / "interim" / ds_name
    processed_dir = work_dir / "data" / "processed" / ds_name
    ckpt_dir = work_dir / "checkpoints" / f"dual_ce_{ds_name}"
    out_dir = work_dir / "outputs" / role

    _build_smoke_parquet_dataset(interim_dir, lang=lang)
    pairs = _score_smoke_pairs_offline(extract_parent_reply_pairs_from_parquet(interim_dir))

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
    ce_model = DualHeadCrossEncoder(lang=lang, model_name="smoke-tiny-ce", encoder=enc, edge_dim=2)
    distill_report = distill_dual_cross_encoder(
        topic_splits["train"],
        topic_splits["val"],
        output_dir=ckpt_dir,
        lang=lang,
        model=ce_model,
        tokenizer=tok,
        edge_dim=2,
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
        edge_dim=2,
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
            root_dir=processed_dir,
            k_horizon=20,
            split="train",
            min_total_replies=DEFAULT_FORECAST_MIN_REPLIES,
            polar_target_mode="future_only",
        )
        val_ds = EarlyCascadeDataset(
            root_dir=processed_dir,
            k_horizon=20,
            split="val",
            min_total_replies=DEFAULT_FORECAST_MIN_REPLIES,
            polar_target_mode="future_only",
        )
        test_ds = EarlyCascadeDataset(
            root_dir=processed_dir,
            k_horizon=20,
            split="test",
            min_total_replies=DEFAULT_FORECAST_MIN_REPLIES,
            polar_target_mode="future_only",
        )
        tr_l = DataLoader(train_ds, batch_size=2, shuffle=True)
        va_l = DataLoader(val_ds, batch_size=2, shuffle=False)
        te_l = DataLoader(test_ds, batch_size=2, shuffle=False)

        num_cls = 4
        loop_res = run_self_auditing_validation_loop(
            tr_l,
            va_l,
            in_dim=TOTAL_NODE_FEATURE_DIM,
            edge_dim=2,
            num_rumor_classes=num_cls,
            use_cascade_stats=True,
            predict_uncertainty=False,
            candidates=[
                {"hidden_dim": 64, "lr": 1e-3, "eps": 0.5, "dropout": 0.1, "lambda_uncertainty": 0.0},
                {"hidden_dim": 32, "lr": 5e-4, "eps": 0.4, "dropout": 0.1, "lambda_uncertainty": 0.0},
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
                "TextMLP": lambda: TextMLP(
                    in_dim=TOTAL_NODE_FEATURE_DIM,
                    hidden_dim=h_dim,
                    num_rumor_classes=num_cls,
                    use_cascade_stats=True,
                ),
                "TemporalStanceMLP": lambda: TemporalStanceMLP(
                    in_dim=TOTAL_NODE_FEATURE_DIM,
                    hidden_dim=h_dim,
                    edge_dim=2,
                    num_rumor_classes=num_cls,
                    use_cascade_stats=True,
                ),
                "BiGCN": lambda: BiGCN(
                    in_dim=TOTAL_NODE_FEATURE_DIM,
                    hidden_dim=h_dim,
                    num_rumor_classes=num_cls,
                    use_cascade_stats=True,
                ),
                "SBiFAGAT": lambda: SBiFAGAT(
                    in_dim=TOTAL_NODE_FEATURE_DIM,
                    hidden_dim=h_dim,
                    edge_dim=2,
                    num_rumor_classes=num_cls,
                    eps=float(best_cfg["eps"]),
                    dropout=float(best_cfg["dropout"]),
                    use_cascade_stats=True,
                    predict_uncertainty=False,
                ),
            },
            tr_l,
            va_l,
            te_l,
            seeds=[42, 100],
            epochs=2,
            patience=2,
            lambda_uncertainty=0.0,
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
    sealed_zip = _bundle_sealed_machine_keys(
        work_dir / f"sbifagat_{role}_sealed_machine_keys.zip",
        search_roots=[work_dir / "data" / "interim", interim_dir],
        base_dir=work_dir,
    )
    summary["bundle_zip"] = str(bundle_path)
    summary["sealed_machine_keys_zip"] = str(sealed_zip) if sealed_zip is not None else None
    return summary


def _nlp_features_match_cohort(feats_path: Path, expected_min_replies: int) -> bool:
    """Return True only if `nlp_features.pt` exists and matches `expected_min_replies`."""
    if not feats_path.is_file():
        return False
    try:
        payload = torch.load(feats_path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not isinstance(payload, Mapping):
        return False
    ext_min = payload.get("min_total_replies_extracted")
    if ext_min is None and isinstance(payload.get("_meta"), Mapping):
        ext_min = payload["_meta"].get("min_total_replies_extracted")
    if ext_min is None:
        return int(expected_min_replies) >= int(DEFAULT_FORECAST_MIN_REPLIES)
    try:
        return int(ext_min) == int(expected_min_replies)
    except (TypeError, ValueError):
        return False


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

    salt_file = work_dir / "data" / "interim" / ".author_hash_salt.secret"
    if not os.environ.get("SBIF_HASH_SALT") and not salt_file.is_file():
        import secrets

        salt_file.parent.mkdir(parents=True, exist_ok=True)
        salt_file.write_text(secrets.token_hex(16), encoding="utf-8")

    interim_dir = work_dir / "data" / "interim" / ds_name
    processed_dir = Path(args.processed_dir) if args.processed_dir else (work_dir / "data" / "processed" / ds_name)
    scored_cache = interim_dir / "scored_pairs.json"
    ce_ckpt = work_dir / "checkpoints" / f"dual_ce_{ds_name}"
    vihsd_dir = work_dir / "checkpoints" / "visobert_vihsd"
    audit_csv = work_dir / "data" / "interim" / f"human_audit_{lang}_120.csv"
    pilot_csv = work_dir / "data" / "interim" / f"pilot_calibration_{lang}_20.csv"
    student_backbone = VISOBERT_MODEL_NAME if lang == "vi" else TWITTER_ROBERTA_MODEL_NAME

    run_nlp = args.role in {"member1_vn", "member2_en", "model_nlp_only"}
    run_gnn = args.role in {"member1_vn", "member2_en", "model_gnn_only"}

    if args.role == "model_gnn_only" and not _nlp_features_match_cohort(
        processed_dir / "nlp_features.pt", int(args.min_total_replies)
    ):
        if not args.bootstrap_if_missing:
            raise FileNotFoundError(
                f"Missing or cohort-mismatched '{processed_dir / 'nlp_features.pt'}' "
                f"(expected min_total_replies={int(args.min_total_replies)}) for model_gnn_only. "
                "Provide matching --processed-dir from Member 1 or pass --bootstrap-if-missing."
            )
        run_nlp = True

    force = bool(getattr(args, "force", False))

    if run_nlp:
        # Stage 1: Real dataset acquisition (skip cleanly if parquet already built and not --force)
        if not force and (interim_dir / "nodes.parquet").exists() and (interim_dir / "edges.parquet").exists():
            print(f"[KaggleRunner] [SKIP Stage 1] Validated interim parquet already exists: {interim_dir}")
        elif lang == "en":
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
                "--min-comments",
                str(args.min_total_replies),
                "--min-vi-ratio",
                "0.65",
                "--max-pages-per-sub",
                "300",
                "--checkpoint-jsonl",
                str(interim_dir / "crawl_checkpoint.jsonl"),
                "--allow-partial",
            ]
            if args.before_epoch is not None:
                crawl_cmd.extend(["--before-epoch", str(args.before_epoch)])
            _run_cmd(crawl_cmd, cwd=work_dir)

        # Stage 2: Unified 2D Teacher scoring (Hostility + mDeBERTa Stance Verbalizer) -> Cross-Encoder
        if (
            not force
            and _nlp_features_match_cohort(processed_dir / "nlp_features.pt", int(args.min_total_replies))
            and (ce_ckpt / "dual_head_weights.pt").exists()
            and audit_csv.exists()
        ):
            print(f"[KaggleRunner] [SKIP Stage 2] Teacher/Student/Audit artifacts already complete: {ce_ckpt}")
        else:
            scorer_cmd = [
                py,
                "-m",
                "src.nlp.dual_edge_scorer",
                "--lang",
                lang,
                "--interim-dir",
                str(interim_dir),
                "--vihsd-dir",
                str(vihsd_dir),
                "--output-dir",
                str(ce_ckpt),
                "--scored-cache-json",
                str(scored_cache),
                "--audit-csv",
                str(audit_csv),
                "--pilot-csv",
                str(pilot_csv),
                "--audit-n-samples",
                "120",
                "--pilot-samples",
                "20",
                "--teacher-backend",
                "dual_nli",
                "--min-total-replies",
                str(args.min_total_replies),
                "--edge-dim",
                "2",
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
            if not force:
                scorer_cmd.append("--skip-if-exists")
            if lang == "vi":
                scorer_cmd.append("--train-vihsd")
            _run_cmd(scorer_cmd, cwd=work_dir)

        # Stage 3: Frozen 768-d Ancestor SpanPool node embeddings + 2D edge attributes
        if (
            not force
            and _nlp_features_match_cohort(processed_dir / "nlp_features.pt", int(args.min_total_replies))
            and (processed_dir / "nodes.parquet").exists()
            and (processed_dir / "edges.parquet").exists()
        ):
            print(f"[KaggleRunner] [SKIP Stage 3] Processed nlp_features.pt already exists: {processed_dir}")
        else:
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
                "--vihsd-dir",
                str(vihsd_dir),
                "--edge-dim",
                "2",
                "--node-backbone",
                student_backbone,
                "--k-horizon",
                str(args.k_horizon),
                "--min-total-replies",
                str(args.min_total_replies),
                "--batch-size",
                str(args.batch_size_nlp * 2),
                "--include-root-context",
            ]
            _run_cmd(extract_cmd, cwd=work_dir)

    best_cfg: dict[str, Any] = {
        "hidden_dim": args.hidden_dim,
        "eps": 0.5,
        "dropout": 0.1,
        "lambda_uncertainty": 0.0,
        "lr": 1e-3,
    }
    loop_res: dict[str, Any] | None = None

    if run_gnn:
        train_ds = EarlyCascadeDataset(
            root_dir=processed_dir,
            k_horizon=args.k_horizon,
            split="train",
            min_total_replies=args.min_total_replies,
            polar_target_mode="future_only",
        )
        inferred_edge_dim = (
            int(train_ds[0].edge_attr.size(1))
            if len(train_ds) > 0 and train_ds[0].edge_attr.ndim == 2
            else 2
        )
        num_cls = 4
        if bool(getattr(args, "run_auto_loop", False)):
            val_ds = EarlyCascadeDataset(
                root_dir=processed_dir,
                k_horizon=args.k_horizon,
                split="val",
                min_total_replies=args.min_total_replies,
                polar_target_mode="future_only",
            )
            loop_res = run_self_auditing_validation_loop(
                DataLoader(train_ds, batch_size=args.batch_size_gnn, shuffle=True),
                DataLoader(val_ds, batch_size=args.batch_size_gnn, shuffle=False),
                in_dim=TOTAL_NODE_FEATURE_DIM,
                hidden_dim=int(args.hidden_dim),
                edge_dim=inferred_edge_dim,
                num_rumor_classes=num_cls,
                use_cascade_stats=True,
                predict_uncertainty=False,
                epochs=min(60, args.epochs_gnn),
                patience=10,
                output_dir=work_dir / "outputs" / f"auto_loop_{ds_name}",
            )

        # Main 5-seed benchmark (Inductive) — explicit ds_name suffixes preserve naming under model_gnn_only
        exp_base_cmd = [
            py,
            "-m",
            "scripts.run_experiments",
            "--data-dir",
            str(processed_dir),
            "--k-horizon",
            str(args.k_horizon),
            "--min-total-replies",
            str(args.min_total_replies),
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
            "--lambda-uncertainty",
            "0.0",
            "--use-cascade-stats",
            "--epochs",
            str(args.epochs_gnn),
            "--batch-size",
            str(args.batch_size_gnn),
        ]
        main_json = work_dir / "paper" / f"results_main_{ds_name}.json"
        main_tex = work_dir / "paper" / f"tables_main_{ds_name}.tex"
        if not force and main_json.is_file() and main_tex.is_file():
            print(f"[KaggleRunner] [SKIP Main] Already complete: {main_json}")
        else:
            _run_cmd(
                exp_base_cmd
                + [
                    "--output-table",
                    f"paper/tables_main_{ds_name}.tex",
                    "--output-json",
                    f"paper/results_main_{ds_name}.json",
                    "--figures-dir",
                    f"paper/figures/{ds_name}",
                    "--ckpt-out",
                    f"checkpoints/best_sbifagat_{ds_name}.pt",
                ],
                cwd=work_dir,
            )

        mech_ckpt = work_dir / "checkpoints" / f"best_sbifagat_{ds_name}.pt"
        mech_json = work_dir / "paper" / f"results_mechanisms_{ds_name}.json"
        mech_tex = work_dir / "paper" / f"tables_mechanisms_{ds_name}.tex"
        if mech_ckpt.is_file() and (force or not mech_json.is_file() or not mech_tex.is_file()):
            try:
                _run_cmd(
                    [
                        py,
                        "-m",
                        "scripts.analyze_mechanisms",
                        "--data-dir",
                        str(processed_dir),
                        "--ckpt",
                        str(mech_ckpt),
                        "--k-horizon",
                        str(args.k_horizon),
                        "--min-total-replies",
                        str(args.min_total_replies),
                        "--output-json",
                        f"paper/results_mechanisms_{ds_name}.json",
                        "--output-table",
                        f"paper/tables_mechanisms_{ds_name}.tex",
                        "--figures-dir",
                        f"paper/figures/{ds_name}",
                    ],
                    cwd=work_dir,
                )
            except Exception as exc:
                print(
                    f"[WARN] Optional mechanistic analysis failed ({exc}); "
                    "primary benchmark artifacts remain preserved."
                )

        if lang == "en":
            loo_json = work_dir / "paper" / f"results_loo_{ds_name}.json"
            loo_tex = work_dir / "paper" / f"tables_loo_{ds_name}.tex"
            if not force and loo_json.is_file() and loo_tex.is_file():
                print(f"[KaggleRunner] [SKIP LOO] Already complete: {loo_json}")
            else:
                pt_path = processed_dir / "nlp_features.pt"
                pt_payload = torch.load(pt_path, map_location="cpu", weights_only=False) if pt_path.exists() else {}
                has_teacher_cov = "teacher_edge_stance" in pt_payload
                del pt_payload
                if has_teacher_cov:
                    _run_cmd(
                        exp_base_cmd
                        + [
                            "--eval-protocol",
                            "loo",
                            "--output-table",
                            f"paper/tables_loo_{ds_name}.tex",
                            "--output-json",
                            f"paper/results_loo_{ds_name}.json",
                            "--ckpt-out",
                            f"checkpoints/best_sbifagat_{ds_name}_loo.pt",
                            "--skip-figures",
                        ],
                        cwd=work_dir,
                    )
                else:
                    print(
                        "[SKIP LOO] nlp_features.pt lacks full teacher_edge_stance coverage "
                        "(ensure dual_edge_scorer scored 100% of PHEME edges before extract_features)."
                    )
        else:
            # Export 30-tree blind thread-level human audit on VN Test cascades if data exists
            vn_tree_audit = work_dir / "data" / "interim" / "human_audit_vn_30_trees.csv"
            if (processed_dir / "nlp_features.pt").exists() and (processed_dir / "nodes.parquet").exists():
                if force or not vn_tree_audit.exists():
                    try:
                        _run_cmd(
                            [
                                py,
                                "-m",
                                "scripts.audit_vn_50_trees",
                                "--data-dir",
                                str(processed_dir),
                                "--audit-csv",
                                str(vn_tree_audit),
                                "--n-samples",
                                "30",
                                "--k-horizon",
                                str(args.k_horizon),
                                "--min-total-replies",
                                str(args.min_total_replies),
                                "--export",
                            ],
                            cwd=work_dir,
                        )
                    except Exception as exc:
                        print(f"[WARN] Could not export 30-tree VN blind audit CSV: {exc}")

            # Temporal holdout check with isolated teacher edges (zero Student leakage)
            temp_json = work_dir / "paper" / f"results_temporal_{ds_name}.json"
            temp_tex = work_dir / "paper" / f"tables_temporal_{ds_name}.tex"
            if force or not temp_json.is_file() or not temp_tex.is_file():
                _run_cmd(
                    exp_base_cmd
                    + [
                        "--split-mode",
                        "temporal",
                        "--edge-source",
                        "teacher",
                        "--output-table",
                        f"paper/tables_temporal_{ds_name}.tex",
                        "--output-json",
                        f"paper/results_temporal_{ds_name}.json",
                        "--skip-figures",
                    ],
                    cwd=work_dir,
                )

        # Full component & K-horizon ablations
        abl_json = work_dir / "paper" / f"results_ablation_{ds_name}.json"
        abl_tex = work_dir / "paper" / f"tables_ablation_{ds_name}.tex"
        if not force and abl_json.is_file() and abl_tex.is_file():
            print(f"[KaggleRunner] [SKIP Ablation] Already complete: {abl_json}")
        else:
            abl_cmd = [
                py,
                "-m",
                "scripts.run_ablation",
                "--data-dir",
                str(processed_dir),
                "--k-horizon",
                str(args.k_horizon),
                "--min-total-replies",
                str(args.min_total_replies),
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
                "--lambda-uncertainty",
                "0.0",
                "--use-cascade-stats",
                "--epochs",
                str(args.epochs_gnn),
                "--batch-size",
                str(args.batch_size_gnn),
                "--output-table",
                f"paper/tables_ablation_{ds_name}.tex",
                "--output-json",
                f"paper/results_ablation_{ds_name}.json",
                "--figures-dir",
                f"paper/figures/{ds_name}",
            ]
            _run_cmd(abl_cmd, cwd=work_dir)

        # Export Table 1 Dataset EDA + Early-K Persistence Baseline & Efficiency LaTeX Table
        eda_json = work_dir / "paper" / f"results_dataset_eda_{ds_name}.json"
        eda_tex = work_dir / "paper" / f"tables_dataset_eda_{ds_name}.tex"
        if not force and eda_json.is_file() and eda_tex.is_file():
            print(f"[KaggleRunner] [SKIP EDA] Already complete: {eda_json}")
        else:
            _run_cmd(
                [
                    py,
                    "-m",
                    "scripts.export_dataset_stats",
                    "--data-dir",
                    str(processed_dir),
                    "--names",
                    ds_name,
                    "--k-horizon",
                    str(args.k_horizon),
                    "--min-total-replies",
                    str(args.min_total_replies),
                    "--edge-dim",
                    str(inferred_edge_dim),
                    "--output-table",
                    f"paper/tables_dataset_eda_{ds_name}.tex",
                    "--output-json",
                    f"paper/results_dataset_eda_{ds_name}.json",
                ],
                cwd=work_dir,
            )
        eff_json = work_dir / "paper" / f"results_efficiency_{ds_name}.json"
        eff_tex = work_dir / "paper" / f"tables_efficiency_{ds_name}.tex"
        if not force and eff_json.is_file() and eff_tex.is_file():
            print(f"[KaggleRunner] [SKIP Efficiency] Already complete: {eff_json}")
        else:
            eff_cmd = [
                py,
                "-m",
                "scripts.benchmark_efficiency",
                "--k-horizon",
                str(args.k_horizon),
                "--hidden-dim",
                str(best_cfg["hidden_dim"]),
                "--edge-dim",
                str(inferred_edge_dim),
                "--use-cascade-stats",
                "--output-table",
                f"paper/tables_efficiency_{ds_name}.tex",
                "--output-json",
                f"paper/results_efficiency_{ds_name}.json",
            ]
            _run_cmd(eff_cmd, cwd=work_dir)

    bundle_sources = [
        processed_dir,
        work_dir / "paper",
        work_dir / "outputs",
        work_dir / "data" / "interim",
        work_dir / "checkpoints",
        ce_ckpt,
    ]
    bundle_nlp_pt = bool(args.bundle_weights or args.role == "model_nlp_only")
    bundle_zip = _bundle_outputs(
        work_dir / f"sbifagat_{args.role}_bundle.zip",
        source_paths=bundle_sources,
        base_dir=work_dir,
        include_heavy_weights=bool(args.bundle_weights),
        include_nlp_features=bundle_nlp_pt,
    )
    sealed_zip = _bundle_sealed_machine_keys(
        work_dir / f"sbifagat_{args.role}_sealed_machine_keys.zip",
        search_roots=[work_dir / "data" / "interim", interim_dir],
        base_dir=work_dir,
    )

    # Run optional UPFD transfer AFTER primary PHEME/VN outputs are safely bundled
    upfd_status = "skipped"
    if run_gnn and args.run_upfd and lang == "en":
        try:
            _run_cmd(
                [
                    py,
                    "-m",
                    "scripts.run_upfd_extension",
                    "--hidden-dim",
                    str(best_cfg["hidden_dim"]),
                    "--epochs",
                    str(args.epochs_gnn),
                ],
                cwd=work_dir,
            )
            upfd_status = "completed"
            bundle_zip = _bundle_outputs(
                work_dir / f"sbifagat_{args.role}_bundle.zip",
                source_paths=bundle_sources,
                base_dir=work_dir,
                include_heavy_weights=bool(args.bundle_weights),
                include_nlp_features=bundle_nlp_pt,
            )
        except Exception as exc:
            upfd_status = f"failed_nonfatal: {exc}"
            print(
                f"[WARN] Optional UPFD extension failed ({exc}); "
                f"primary PHEME bundle is already safely saved at '{bundle_zip}'."
            )

    return {
        "mode": "unified_2d",
        "role": args.role,
        "lang": lang,
        "best_config": best_cfg,
        "auto_loop": loop_res,
        "upfd_status": upfd_status,
        "bundle_zip": str(bundle_zip),
        "sealed_machine_keys_zip": str(sealed_zip) if sealed_zip is not None else None,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Kaggle Parallel Runner & Loop Orchestrator (Unified 2D).")
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
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--min-total-replies", type=int, default=DEFAULT_FORECAST_MIN_REPLIES)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs-nlp", type=int, default=3)
    parser.add_argument("--epochs-gnn", type=int, default=100)
    parser.add_argument("--batch-size-nlp", type=int, default=16)
    parser.add_argument("--batch-size-gnn", type=int, default=32)
    parser.add_argument("--bootstrap-if-missing", action="store_true")
    parser.add_argument("--run-auto-loop", action="store_true")
    parser.add_argument("--run-upfd", action="store_true")
    parser.add_argument("--bundle-weights", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run all stages even if outputs already exist.")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    res = run_kaggle_parallel_role(args)
    print(json.dumps({k: v for k, v in res.items() if k != "auto_loop"}, indent=2, default=str))


if __name__ == "__main__":
    main()

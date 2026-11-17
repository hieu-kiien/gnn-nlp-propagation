"""Dual-branch open-source NLP edge scorer and student distillation (`s_uv`, `h_uv`).

Operates exclusively with open Hugging Face Hub models (no paid APIs):
  1. Hostility Teacher (`h_uv in [0, 1]`):
       - Vietnamese: `uitnlp/visobert` fine-tuned on `ura-hcmut/UIT-ViHSD`
         with 3 labels {0: Clean, 1: Offensive, 2: Hate} -> `h_uv = P(Offensive) + P(Hate)`.
       - English (PHEME): `cardiffnlp/twitter-roberta-base-offensive`
         -> `h_uv = P(offensive)`.
  2. Stance NLI Teacher (`s_uv in [-1, 1]`):
       - `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
         reading `(parent_text u, reply_text v)` ->
         `s_uv = P(entailment | u, v) - P(contradiction | u, v)`.
  3. Blind Stratified Human Audit CSV exporter (`data/interim/human_audit_300.csv`).
  4. Unified 2-head Student `DualHeadCrossEncoder` (`uitnlp/visobert` for `vi`,
     `roberta-base` for `en`) distilled from teacher soft labels with an
     anti-cheating `--sanity-check-shuffle` flag.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import random
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from contextlib import nullcontext
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Any, **_kwargs: Any) -> Any:
        return iterable

from src.data.pyg_dataset import inductive_topic_split
from src.data.schema import load_cascade_parquet
from src.nlp.qwen_teacher import (
    DISCOURSE_BEHAVIOR_LABELS,
    EN_DEBERTA_MODEL_NAME,
    QWEN_TEACHER_MODEL_ID,
    QwenBehavioralTeacherScorer,
    behavior_probs_to_stance_probs,
    finetune_qwen_qlora,
)
from src.utils.seed import seed_everything

VISOBERT_MODEL_NAME = "uitnlp/visobert"
UIT_VIHSD_DATASET_NAME = "ura-hcmut/UIT-ViHSD"
EN_OFFENSIVE_MODEL_NAME = "cardiffnlp/twitter-roberta-base-offensive"
MDEBERTA_NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
ROBERTA_BASE_MODEL_NAME = "roberta-base"

STANCE_LABELS: tuple[str, ...] = ("support", "neutral", "oppose")


def _make_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _cuda_autocast(enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=torch.float16, enabled=True)
    return torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)


def _normalize_nfc(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text).replace("\x00", "").strip()


def load_tokenizer_safe(model_name_or_path: str | Path) -> Any:
    """Load an AutoTokenizer with automatic `use_fast=False` fallback for SentencePiece models."""
    target = str(model_name_or_path)
    try:
        return AutoTokenizer.from_pretrained(target)
    except Exception:
        return AutoTokenizer.from_pretrained(target, use_fast=False)


def load_uit_vihsd_dataset(cache_dir: str | None = None) -> Any:
    """Load `ura-hcmut/UIT-ViHSD` directly from Hugging Face Hub and normalize valid rows."""
    from datasets import load_dataset

    ds = load_dataset(UIT_VIHSD_DATASET_NAME, cache_dir=cache_dir)

    def _is_valid_row(example: Mapping[str, Any]) -> bool:
        txt = _normalize_nfc(example.get("free_text"))
        lbl = example.get("label_id")
        return bool(txt) and lbl is not None and int(lbl) in {0, 1, 2}

    def _clean_row(example: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "free_text": _normalize_nfc(example["free_text"]),
            "label_id": int(example["label_id"]),
        }

    cleaned = ds.filter(_is_valid_row)
    return cleaned.map(_clean_row)


class _TextClassificationDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        self.texts = list(texts)
        self.labels = [int(x) for x in labels]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], self.labels[idx]


def finetune_visobert_vihsd(
    output_dir: Path = Path("checkpoints/visobert_vihsd"),
    *,
    model_name: str = VISOBERT_MODEL_NAME,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 256,
    max_train_samples: int | None = None,
    seed: int = 42,
    dataset_dict: Any | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: torch.device | None = None,
) -> Path:
    """Fine-tune ViSoBERT (FP16 AMP on CUDA) for 3-class UIT-ViHSD hostility detection."""
    seed_everything(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = target_device.type == "cuda"

    ds = dataset_dict if dataset_dict is not None else load_uit_vihsd_dataset()
    train_split = ds["train"]
    val_split = ds["validation"] if "validation" in ds else ds["test"]

    train_texts = [str(x) for x in train_split["free_text"]]
    train_labels = [int(x) for x in train_split["label_id"]]
    if max_train_samples is not None and len(train_texts) > max_train_samples:
        train_texts = train_texts[:max_train_samples]
        train_labels = train_labels[:max_train_samples]

    val_texts = [str(x) for x in val_split["free_text"]]
    val_labels = [int(x) for x in val_split["label_id"]]

    tok = tokenizer if tokenizer is not None else load_tokenizer_safe(model_name)
    if model is None:
        clf = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label={0: "clean", 1: "offensive", 2: "hate"},
            label2id={"clean": 0, "offensive": 1, "hate": 2},
        )
    else:
        clf = model
    clf = clf.to(target_device)

    train_loader = DataLoader(
        _TextClassificationDataset(train_texts, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        _TextClassificationDataset(val_texts, val_labels),
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=0.01)
    scaler = _make_grad_scaler(enabled=use_amp)

    best_f1 = -1.0
    best_state: dict[str, Tensor] = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}

    for _epoch in range(max(1, epochs)):
        clf.train()
        for batch_texts, batch_labels in train_loader:
            enc = tok(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(target_device) for k, v in enc.items()}
            labels_t = torch.as_tensor(batch_labels, dtype=torch.long, device=target_device)

            optimizer.zero_grad(set_to_none=True)
            with _cuda_autocast(enabled=use_amp):
                out = clf(**enc, labels=labels_t)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        clf.eval()
        preds: list[int] = []
        trues: list[int] = []
        with torch.no_grad():
            for batch_texts, batch_labels in val_loader:
                enc = tok(
                    list(batch_texts),
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(target_device) for k, v in enc.items()}
                with _cuda_autocast(enabled=use_amp):
                    logits = clf(**enc).logits
                preds.extend(logits.argmax(dim=-1).cpu().tolist())
                trues.extend(torch.as_tensor(batch_labels).tolist())

        macro_f1 = float(f1_score(trues, preds, average="macro", zero_division=0))
        if macro_f1 >= best_f1:
            best_f1 = macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}

    clf.load_state_dict(best_state)
    clf.save_pretrained(out_path)
    if hasattr(tok, "save_pretrained"):
        tok.save_pretrained(out_path)
    (out_path / "vihsd_metrics.json").write_text(
        json.dumps({"best_val_macro_f1": best_f1}, indent=2),
        encoding="utf-8",
    )
    return out_path


class HostilityTeacherScorer:
    """Compute hostility intensity `h_uv in [0, 1]` for reply texts."""

    def __init__(
        self,
        *,
        lang: str = "vi",
        model_name_or_path: str | Path | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: torch.device | None = None,
        allow_untrained_fallback: bool = False,
    ) -> None:
        if lang not in {"vi", "en"}:
            raise ValueError(f"Unsupported lang {lang!r}; expected 'vi' or 'en'")
        self.lang = lang
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        default_name = VISOBERT_MODEL_NAME if lang == "vi" else EN_OFFENSIVE_MODEL_NAME

        if model is None and lang == "vi":
            if model_name_or_path is None:
                if not allow_untrained_fallback:
                    raise FileNotFoundError(
                        "Fine-tuned UIT-ViHSD checkpoint is required for lang='vi' hostility "
                        "scoring. Run with --train-vihsd first or pass allow_untrained_fallback=True."
                    )
            else:
                ckpt_path = Path(model_name_or_path)
                if str(model_name_or_path) != VISOBERT_MODEL_NAME and not ckpt_path.exists():
                    raise FileNotFoundError(
                        f"Missing fine-tuned ViHSD checkpoint directory: {ckpt_path}"
                    )

        target = str(model_name_or_path or default_name)

        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer_safe(target)
        if model is not None:
            self.model = model
        else:
            kwargs = {"num_labels": 3} if (lang == "vi" and model_name_or_path is None) else {}
            self.model = AutoModelForSequenceClassification.from_pretrained(target, **kwargs)
        self.model = self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def hostility_from_probs(
        probs: Tensor,
        *,
        lang: str,
        id2label: Mapping[int | str, str] | None = None,
    ) -> Tensor:
        """Compute `h_uv = P(Offensive) + P(Hate)` (vi) or `P(offensive)` (en)."""
        if lang == "vi":
            if probs.size(-1) < 3:
                raise ValueError(f"Expected 3 classes for UIT-ViHSD, got {probs.size(-1)}")
            h = probs[:, 1] + probs[:, 2]
            return h.clamp(0.0, 1.0)

        off_idx = 1 if probs.size(-1) >= 2 else 0
        if id2label:
            for k, name in id2label.items():
                low = str(name).lower()
                if "offensive" in low and "non" not in low:
                    off_idx = int(k)
                    break
        return probs[:, off_idx].clamp(0.0, 1.0)

    @torch.no_grad()
    def score_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        max_length: int = 256,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0,), dtype=np.float32)

        use_amp = self.device.type == "cuda"
        id2label = getattr(getattr(self.model, "config", None), "id2label", None)
        scores: list[Tensor] = []

        for start in range(0, len(texts), batch_size):
            batch = [_normalize_nfc(t) for t in texts[start : start + batch_size]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with _cuda_autocast(enabled=use_amp):
                logits = self.model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1)
                h = self.hostility_from_probs(probs, lang=self.lang, id2label=id2label)
            scores.append(h.cpu())

        return torch.cat(scores, dim=0).numpy().astype(np.float32)


class NLIStanceTeacherScorer:
    """Compute pairwise NLI stance signal `s_uv = P(entailment | u, v) - P(contradiction | u, v)`."""

    def __init__(
        self,
        *,
        model_name_or_path: str | Path = MDEBERTA_NLI_MODEL_NAME,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target = str(model_name_or_path)
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer_safe(target)
        self.model = (
            model
            if model is not None
            else AutoModelForSequenceClassification.from_pretrained(target)
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.entail_idx, self.neutral_idx, self.contra_idx = self._resolve_nli_indices()

    def _resolve_nli_indices(self) -> tuple[int, int, int]:
        id2label = getattr(getattr(self.model, "config", None), "id2label", None) or {}
        entail, neutral, contra = 0, 1, 2
        for k, name in id2label.items():
            low = str(name).lower()
            idx = int(k)
            if "entail" in low:
                entail = idx
            elif "neutral" in low:
                neutral = idx
            elif "contradict" in low:
                contra = idx
        return entail, neutral, contra

    @staticmethod
    def stance_from_canonical_probs(probs_3: Tensor) -> Tensor:
        """Compute `s_uv = P(entailment) - P(contradiction)` from `[P_ent, P_neu, P_con]`."""
        return (probs_3[:, 0] - probs_3[:, 2]).clamp(-1.0, 1.0)

    @torch.no_grad()
    def score_pairs(
        self,
        parent_texts: Sequence[str],
        reply_texts: Sequence[str],
        *,
        batch_size: int = 16,
        max_length: int = 256,
    ) -> dict[str, Any]:
        if len(parent_texts) != len(reply_texts):
            raise ValueError("parent_texts and reply_texts must have equal length")
        if not parent_texts:
            return {
                "s_uv": np.zeros((0,), dtype=np.float32),
                "stance_probs": np.zeros((0, 3), dtype=np.float32),
                "pred_label": [],
            }

        use_amp = self.device.type == "cuda"
        all_probs: list[Tensor] = []
        all_s: list[Tensor] = []

        for start in range(0, len(parent_texts), batch_size):
            u_batch = [_normalize_nfc(t) for t in parent_texts[start : start + batch_size]]
            v_batch = [_normalize_nfc(t) for t in reply_texts[start : start + batch_size]]
            enc = self.tokenizer(
                u_batch,
                v_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with _cuda_autocast(enabled=use_amp):
                logits = self.model(**enc).logits.float()
            if not torch.isfinite(logits).all():
                with _cuda_autocast(enabled=False):
                    logits = self.model(**enc).logits.float()
            raw_probs = torch.softmax(logits, dim=-1)
            canonical = torch.stack(
                [
                    raw_probs[:, self.entail_idx],
                    raw_probs[:, self.neutral_idx],
                    raw_probs[:, self.contra_idx],
                ],
                dim=-1,
            )
            s = self.stance_from_canonical_probs(canonical)
            all_probs.append(canonical.cpu())
            all_s.append(s.cpu())

        probs_np = torch.cat(all_probs, dim=0).numpy().astype(np.float32)
        s_np = torch.cat(all_s, dim=0).numpy().astype(np.float32)
        labels = [STANCE_LABELS[int(idx)] for idx in probs_np.argmax(axis=-1)]
        return {
            "s_uv": s_np,
            "stance_probs": probs_np,
            "pred_label": labels,
        }


def _infer_stance_stratum(pair: Mapping[str, Any]) -> str:
    lbl = str(pair.get("pred_label") or "").strip().lower()
    if lbl in {"support", "entailment"}:
        return "support"
    if lbl in {"oppose", "contradiction", "constructive_oppose", "irrational_fallacy", "hostile_conflict"}:
        return "oppose"
    if lbl == "neutral":
        return "neutral"
    s_val = float(pair.get("s_uv", 0.0))
    if s_val > 0.1:
        return "support"
    if s_val < -0.1:
        return "oppose"
    return "neutral"


def _infer_behavior_stratum(pair: Mapping[str, Any]) -> str:
    blbl = str(pair.get("behavior_label") or "").strip().lower()
    if blbl in DISCOURSE_BEHAVIOR_LABELS:
        return blbl
    stance = _infer_stance_stratum(pair)
    if stance == "support":
        return "support"
    if stance == "neutral":
        return "neutral"
    h_val = float(pair.get("h_uv", 0.0))
    r_val = float(pair.get("r_uv", 0.0))
    if h_val >= 0.65:
        return "hostile_conflict"
    if r_val >= 0.55:
        return "irrational_fallacy"
    return "constructive_oppose"


def export_stratified_human_audit_csv(
    scored_pairs: Sequence[Mapping[str, Any]],
    output_csv: Path = Path("data/interim/human_audit_300.csv"),
    *,
    machine_key_csv: Path | None = None,
    pilot_csv: Path | None = None,
    pilot_samples: int = 30,
    n_samples: int = 300,
    seed: int = 42,
) -> Path:
    """Export balanced stratified blind pairs to CSV (strictly hiding machine predictions)."""
    if not scored_pairs:
        raise ValueError("Cannot export human audit CSV from empty scored_pairs")

    has_behavior = any(p.get("behavior_label") in DISCOURSE_BEHAVIOR_LABELS for p in scored_pairs)
    strata_order = DISCOURSE_BEHAVIOR_LABELS if has_behavior else STANCE_LABELS

    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {lbl: [] for lbl in strata_order}
    for item in scored_pairs:
        row_copy = dict(item)
        stance_stratum = _infer_stance_stratum(item)
        row_copy["_stance_stratum"] = stance_stratum
        stratum = _infer_behavior_stratum(item) if has_behavior else stance_stratum
        row_copy["_stratum"] = stratum
        buckets[stratum].append(row_copy)

    for lbl in strata_order:
        rng.shuffle(buckets[lbl])

    target_total = min(n_samples, len(scored_pairs))
    selected: list[dict[str, Any]] = []
    base_quota = target_total // len(strata_order)
    offsets: dict[str, int] = {lbl: 0 for lbl in strata_order}

    for lbl in strata_order:
        take = min(base_quota, len(buckets[lbl]))
        selected.extend(buckets[lbl][:take])
        offsets[lbl] = take

    # Redistribute remaining slots across non-exhausted strata in round-robin order
    while len(selected) < target_total:
        added_any = False
        for lbl in strata_order:
            if len(selected) >= target_total:
                break
            idx = offsets[lbl]
            if idx < len(buckets[lbl]):
                selected.append(buckets[lbl][idx])
                offsets[lbl] += 1
                added_any = True
        if not added_any:
            break

    rng.shuffle(selected)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blind_fieldnames = [
        "sample_id",
        "cascade_id",
        "edge_id",
        "parent_text",
        "reply_text",
        "annotator_1_stance",
        "annotator_2_stance",
        "annotator_1_hostility",
        "annotator_2_hostility",
        "notes",
        "gold_adjudicated_stance",
        "gold_adjudicated_hostility",
    ]
    if has_behavior:
        blind_fieldnames.extend(
            ["annotator_1_behavior", "annotator_2_behavior", "gold_adjudicated_behavior"]
        )

    def _write_blind_csv(csv_dest: Path, rows_to_write: Sequence[Mapping[str, Any]], prefix: str) -> None:
        csv_dest.parent.mkdir(parents=True, exist_ok=True)
        with csv_dest.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=blind_fieldnames)
            writer.writeheader()
            for idx, row in enumerate(rows_to_write):
                payload_row = {
                    "sample_id": f"{prefix}_{idx + 1:03d}",
                    "cascade_id": str(row.get("cascade_id") or ""),
                    "edge_id": str(row.get("edge_id") or ""),
                    "parent_text": str(row.get("parent_text") or ""),
                    "reply_text": str(row.get("reply_text") or ""),
                    "annotator_1_stance": "",
                    "annotator_2_stance": "",
                    "annotator_1_hostility": "",
                    "annotator_2_hostility": "",
                    "notes": "",
                    "gold_adjudicated_stance": "",
                    "gold_adjudicated_hostility": "",
                }
                if has_behavior:
                    payload_row["annotator_1_behavior"] = ""
                    payload_row["annotator_2_behavior"] = ""
                    payload_row["gold_adjudicated_behavior"] = ""
                writer.writerow(payload_row)

    _write_blind_csv(out_path, selected, prefix="audit")

    if pilot_csv is not None and pilot_samples > 0:
        pilot_selected: list[dict[str, Any]] = []
        while len(pilot_selected) < pilot_samples:
            added_pilot = False
            for lbl in strata_order:
                if len(pilot_selected) >= pilot_samples:
                    break
                idx = offsets[lbl]
                if idx < len(buckets[lbl]):
                    pilot_selected.append(buckets[lbl][idx])
                    offsets[lbl] += 1
                    added_pilot = True
            if not added_pilot:
                break
        if pilot_selected:
            rng.shuffle(pilot_selected)
            _write_blind_csv(Path(pilot_csv), pilot_selected, prefix="pilot")

    if machine_key_csv is not None:
        key_path = Path(machine_key_csv)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_fields = [
            "sample_id",
            "cascade_id",
            "edge_id",
            "machine_stance_label",
            "s_uv",
            "h_uv",
        ]
        if has_behavior:
            key_fields.extend(["machine_behavior_label", "r_uv", "c_uv"])
        with key_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=key_fields)
            writer.writeheader()
            for idx, row in enumerate(selected):
                k_row = {
                    "sample_id": f"audit_{idx + 1:03d}",
                    "cascade_id": str(row.get("cascade_id") or ""),
                    "edge_id": str(row.get("edge_id") or ""),
                    "machine_stance_label": row["_stance_stratum"],
                    "s_uv": f"{float(row.get('s_uv', 0.0)):.6f}",
                    "h_uv": f"{float(row.get('h_uv', 0.0)):.6f}",
                }
                if has_behavior:
                    k_row["machine_behavior_label"] = row["_stratum"]
                    k_row["r_uv"] = f"{float(row.get('r_uv', 0.0)):.6f}"
                    k_row["c_uv"] = f"{float(row.get('c_uv', 0.5)):.6f}"
                writer.writerow(k_row)

    return out_path


class DualHeadCrossEncoder(nn.Module):
    """Unified multi-head Cross-Encoder predicting 2D `[s_uv, h_uv]` or 4D `[s_uv, h_uv, r_uv, c_uv]`."""

    def __init__(
        self,
        *,
        lang: str = "vi",
        model_name: str | None = None,
        encoder: Any | None = None,
        edge_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if lang not in {"vi", "en"}:
            raise ValueError(f"Unsupported lang {lang!r}")
        if edge_dim not in {2, 4}:
            raise ValueError(f"edge_dim must be 2 or 4, got {edge_dim}")
        self.lang = lang
        self.edge_dim = int(edge_dim)
        self.model_name = model_name or (VISOBERT_MODEL_NAME if lang == "vi" else ROBERTA_BASE_MODEL_NAME)
        self.dropout_p = float(dropout)
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(self.model_name)
        hidden_size = int(self.encoder.config.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.stance_head = nn.Linear(hidden_size, 3)  # [support, neutral, oppose]
        self.hostility_head = nn.Linear(hidden_size, 1)
        if self.edge_dim == 4:
            self.behavior_head = nn.Linear(hidden_size, len(DISCOURSE_BEHAVIOR_LABELS))
            self.irrational_head = nn.Linear(hidden_size, 1)
            self.constructive_head = nn.Linear(hidden_size, 1)
        else:
            self.behavior_head = None
            self.irrational_head = None
            self.constructive_head = None

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        kwargs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        cls_repr = self.dropout(outputs.last_hidden_state[:, 0, :])

        stance_logits = self.stance_head(cls_repr)
        stance_probs = torch.softmax(stance_logits.float(), dim=-1)
        s_uv = (stance_probs[:, 0] - stance_probs[:, 2]).clamp(-1.0, 1.0)

        hostility_logit = self.hostility_head(cls_repr).squeeze(-1)
        h_uv = torch.sigmoid(hostility_logit.float()).clamp(0.0, 1.0)

        if self.edge_dim == 4 and self.behavior_head is not None:
            behavior_logits = self.behavior_head(cls_repr)
            behavior_probs = torch.softmax(behavior_logits.float(), dim=-1)
            irrational_logit = self.irrational_head(cls_repr).squeeze(-1)
            constructive_logit = self.constructive_head(cls_repr).squeeze(-1)
            r_uv = torch.sigmoid(irrational_logit.float()).clamp(0.0, 1.0)
            c_uv = torch.sigmoid(constructive_logit.float()).clamp(0.0, 1.0)
            e_uv = torch.stack([s_uv, h_uv, r_uv, c_uv], dim=-1)
            return {
                "stance_logits": stance_logits,
                "stance_probs": stance_probs,
                "hostility_logit": hostility_logit,
                "behavior_logits": behavior_logits,
                "behavior_probs": behavior_probs,
                "s_uv": s_uv,
                "h_uv": h_uv,
                "r_uv": r_uv,
                "c_uv": c_uv,
                "e_uv": e_uv,
            }

        e_uv = torch.stack([s_uv, h_uv], dim=-1)
        return {
            "stance_logits": stance_logits,
            "stance_probs": stance_probs,
            "hostility_logit": hostility_logit,
            "s_uv": s_uv,
            "h_uv": h_uv,
            "e_uv": e_uv,
        }

    def save_checkpoint(self, output_dir: Path, tokenizer: Any | None = None) -> Path:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self.encoder.config.save_pretrained(out_path)
        torch.save(self.state_dict(), out_path / "dual_head_weights.pt")
        meta = {
            "lang": self.lang,
            "model_name": self.model_name,
            "edge_dim": self.edge_dim,
            "dropout": self.dropout_p,
            "has_pooler": getattr(self.encoder, "pooler", None) is not None,
        }
        (out_path / "dual_head_meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out_path)
        return out_path

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple[DualHeadCrossEncoder, Any | None]:
        ckpt_path = Path(checkpoint_dir)
        meta = json.loads((ckpt_path / "dual_head_meta.json").read_text(encoding="utf-8"))
        config = AutoConfig.from_pretrained(ckpt_path)
        encoder = AutoModel.from_config(config)
        if not meta.get("has_pooler", True) and hasattr(encoder, "pooler"):
            encoder.pooler = None
        instance = cls(
            lang=meta["lang"],
            model_name=meta.get("model_name"),
            encoder=encoder,
            edge_dim=int(meta.get("edge_dim", 2)),
            dropout=float(meta.get("dropout", 0.1)),
        )
        state = torch.load(ckpt_path / "dual_head_weights.pt", map_location=map_location, weights_only=True)
        instance.load_state_dict(state)

        tokenizer: Any | None = None
        if any(
            (ckpt_path / fname).exists()
            for fname in ("tokenizer_config.json", "tokenizer.json", "vocab.json", "sentencepiece.bpe.model")
        ):
            tokenizer = load_tokenizer_safe(ckpt_path)
        return instance, tokenizer


class _PairDistillDataset(Dataset):
    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        return self.records[idx]


def _safe_pearsonr(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size < 2 or np.std(pred) < 1e-12 or np.std(target) < 1e-12:
        return 0.0
    corr = float(np.corrcoef(pred, target)[0, 1])
    return 0.0 if math.isnan(corr) else corr


def _build_default_stance_probs(s_uv: float) -> list[float]:
    s = max(-1.0, min(1.0, float(s_uv)))
    if s >= 0.0:
        p_sup = 0.2 + 0.75 * s
        p_opp = 0.05 * (1.0 - s)
    else:
        p_opp = 0.2 + 0.75 * (-s)
        p_sup = 0.05 * (1.0 + s)
    p_neu = max(0.05, 1.0 - p_sup - p_opp)
    total = p_sup + p_neu + p_opp
    return [p_sup / total, p_neu / total, p_opp / total]


def _build_default_behavior_probs(item: Mapping[str, Any]) -> list[float]:
    raw_bp = item.get("behavior_probs")
    if isinstance(raw_bp, Sequence) and not isinstance(raw_bp, (str, bytes)) and len(raw_bp) == 5:
        arr = [max(1e-6, float(x)) for x in raw_bp]
        tot = sum(arr)
        return [x / tot for x in arr]
    s_v = float(item.get("s_uv", 0.0))
    h_v = float(item.get("h_uv", 0.0))
    r_v = float(item.get("r_uv", 0.1))
    probs = [0.05, 0.05, 0.05, 0.05, 0.05]
    if s_v > 0.15:
        probs[0] = 0.75
    elif s_v < -0.15:
        if h_v >= 0.65:
            probs[4] = 0.75
        elif r_v >= 0.55:
            probs[3] = 0.75
        else:
            probs[2] = 0.75
    else:
        probs[1] = 0.75
    tot = sum(probs)
    return [x / tot for x in probs]


def distill_dual_cross_encoder(
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
    output_dir: Path = Path("checkpoints/dual_cross_encoder"),
    *,
    lang: str = "vi",
    model_name: str | None = None,
    model: DualHeadCrossEncoder | None = None,
    tokenizer: Any | None = None,
    edge_dim: int = 2,
    include_root_context: bool = False,
    use_multi_gpu: bool = True,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 256,
    seed: int = 42,
    sanity_check_shuffle: bool = False,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Distill unified `[s_uv, h_uv]` or 4D `[s_uv, h_uv, r_uv, c_uv]` Cross-Encoder."""
    if not train_records or not val_records:
        raise ValueError("Both train_records and val_records must be non-empty")

    seed_everything(seed)
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = target_device.type == "cuda"

    train_data = [dict(r) for r in train_records]
    val_data = [dict(r) for r in val_records]

    effective_edge_dim = int(model.edge_dim) if model is not None else int(edge_dim)
    if any("r_uv" in r or "behavior_probs" in r for r in train_data[:10]) and model is None and edge_dim == 2:
        # Respect explicit edge_dim=2 unless caller instantiated a 4D model or requested edge_dim=4
        pass

    if sanity_check_shuffle:
        # Shuffle teacher target labels on TRAIN ONLY while preserving input pairs & VAL targets
        rng = random.Random(seed + 101)
        perm = list(range(len(train_data)))
        rng.shuffle(perm)
        orig_targets = [
            (
                float(train_data[i]["s_uv"]),
                float(train_data[i]["h_uv"]),
                float(train_data[i].get("r_uv", 0.0)),
                float(train_data[i].get("c_uv", 0.5)),
                train_data[i].get("stance_probs"),
                train_data[i].get("behavior_probs"),
            )
            for i in range(len(train_data))
        ]
        for new_idx, old_idx in enumerate(perm):
            s_t, h_t, r_t, c_t, sp_t, bp_t = orig_targets[old_idx]
            train_data[new_idx]["s_uv"] = s_t
            train_data[new_idx]["h_uv"] = h_t
            if "r_uv" in train_data[new_idx]:
                train_data[new_idx]["r_uv"] = r_t
            if "c_uv" in train_data[new_idx]:
                train_data[new_idx]["c_uv"] = c_t
            if sp_t is not None:
                train_data[new_idx]["stance_probs"] = sp_t
            if bp_t is not None:
                train_data[new_idx]["behavior_probs"] = bp_t

    student = (
        model
        if model is not None
        else DualHeadCrossEncoder(lang=lang, model_name=model_name, edge_dim=effective_edge_dim)
    )
    student = student.to(target_device)
    tok = tokenizer if tokenizer is not None else load_tokenizer_safe(student.model_name)

    active_student: nn.Module = student
    if use_multi_gpu and target_device.type == "cuda" and torch.cuda.device_count() > 1:
        active_student = nn.DataParallel(student)

    def _format_u(item: Mapping[str, Any]) -> str:
        parent_txt = _normalize_nfc(item["parent_text"])
        if not include_root_context:
            return parent_txt
        root_txt = _normalize_nfc(item.get("root_text") or "")
        if root_txt and root_txt != parent_txt:
            return f"{root_txt[:220]} | {parent_txt}"
        return parent_txt

    def _collate(
        batch: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        u_texts = [_format_u(item) for item in batch]
        v_texts = [_normalize_nfc(item["reply_text"]) for item in batch]
        enc = tok(
            u_texts,
            v_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        s_tgt = torch.tensor([float(item["s_uv"]) for item in batch], dtype=torch.float32)
        h_tgt = torch.tensor([float(item["h_uv"]) for item in batch], dtype=torch.float32)
        r_tgt = torch.tensor([float(item.get("r_uv", 0.1)) for item in batch], dtype=torch.float32)
        c_tgt = torch.tensor([float(item.get("c_uv", 0.5)) for item in batch], dtype=torch.float32)
        probs_list = [
            list(item["stance_probs"])
            if item.get("stance_probs") is not None
            else _build_default_stance_probs(float(item["s_uv"]))
            for item in batch
        ]
        p_tgt = torch.tensor(probs_list, dtype=torch.float32).clamp(min=1e-6)
        p_tgt = p_tgt / p_tgt.sum(dim=-1, keepdim=True)

        bp_list = [_build_default_behavior_probs(item) for item in batch]
        bp_tgt = torch.tensor(bp_list, dtype=torch.float32).clamp(min=1e-6)
        bp_tgt = bp_tgt / bp_tgt.sum(dim=-1, keepdim=True)
        return enc, s_tgt, h_tgt, r_tgt, c_tgt, p_tgt, bp_tgt

    train_loader = DataLoader(
        _PairDistillDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        _PairDistillDataset(val_data),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scaler = _make_grad_scaler(enabled=use_amp)

    train_losses: list[float] = []
    best_val_loss = float("inf")
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in student.state_dict().items()})
    last_val_metrics: dict[str, float] = {}

    def _batch_loss(
        out: Mapping[str, Tensor],
        s_tgt_d: Tensor,
        h_tgt_d: Tensor,
        r_tgt_d: Tensor,
        c_tgt_d: Tensor,
        p_tgt_d: Tensor,
        bp_tgt_d: Tensor,
    ) -> Tensor:
        kl_loss = F.kl_div(
            torch.log_softmax(out["stance_logits"].float(), dim=-1),
            p_tgt_d,
            reduction="batchmean",
        )
        s_loss = F.smooth_l1_loss(out["s_uv"], s_tgt_d)
        h_loss = F.smooth_l1_loss(out["h_uv"], h_tgt_d)
        loss = 0.5 * kl_loss + s_loss + h_loss
        if student.edge_dim == 4 and "behavior_logits" in out:
            kl_beh = F.kl_div(
                torch.log_softmax(out["behavior_logits"].float(), dim=-1),
                bp_tgt_d,
                reduction="batchmean",
            )
            r_loss = F.smooth_l1_loss(out["r_uv"], r_tgt_d)
            c_loss = F.smooth_l1_loss(out["c_uv"], c_tgt_d)
            loss = loss + 0.5 * kl_beh + r_loss + c_loss
        return loss

    for _epoch in range(max(1, epochs)):
        active_student.train()
        running_loss = 0.0
        step_count = 0
        for enc, s_tgt, h_tgt, r_tgt, c_tgt, p_tgt, bp_tgt in train_loader:
            enc_dev = {k: v.to(target_device) for k, v in enc.items()}
            s_tgt_d = s_tgt.to(target_device)
            h_tgt_d = h_tgt.to(target_device)
            r_tgt_d = r_tgt.to(target_device)
            c_tgt_d = c_tgt.to(target_device)
            p_tgt_d = p_tgt.to(target_device)
            bp_tgt_d = bp_tgt.to(target_device)

            optimizer.zero_grad(set_to_none=True)
            with _cuda_autocast(enabled=use_amp):
                out = active_student(**enc_dev)
                loss = _batch_loss(out, s_tgt_d, h_tgt_d, r_tgt_d, c_tgt_d, p_tgt_d, bp_tgt_d)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu().item())
            step_count += 1

        train_losses.append(running_loss / max(1, step_count))

        active_student.eval()
        val_loss_sum = 0.0
        val_steps = 0
        s_preds, s_trues, h_preds, h_trues = [], [], [], []
        r_preds, r_trues, c_preds, c_trues = [], [], [], []
        with torch.no_grad():
            for enc, s_tgt, h_tgt, r_tgt, c_tgt, p_tgt, bp_tgt in val_loader:
                enc_dev = {k: v.to(target_device) for k, v in enc.items()}
                s_tgt_d = s_tgt.to(target_device)
                h_tgt_d = h_tgt.to(target_device)
                r_tgt_d = r_tgt.to(target_device)
                c_tgt_d = c_tgt.to(target_device)
                p_tgt_d = p_tgt.to(target_device)
                bp_tgt_d = bp_tgt.to(target_device)
                with _cuda_autocast(enabled=use_amp):
                    out = active_student(**enc_dev)
                    vloss = _batch_loss(out, s_tgt_d, h_tgt_d, r_tgt_d, c_tgt_d, p_tgt_d, bp_tgt_d)

                val_loss_sum += float(vloss.cpu().item())
                val_steps += 1
                s_preds.extend(out["s_uv"].cpu().tolist())
                s_trues.extend(s_tgt.tolist())
                h_preds.extend(out["h_uv"].cpu().tolist())
                h_trues.extend(h_tgt.tolist())
                if student.edge_dim == 4 and "r_uv" in out:
                    r_preds.extend(out["r_uv"].cpu().tolist())
                    r_trues.extend(r_tgt.tolist())
                    c_preds.extend(out["c_uv"].cpu().tolist())
                    c_trues.extend(c_tgt.tolist())

        val_loss = val_loss_sum / max(1, val_steps)
        s_p_arr = np.asarray(s_preds, dtype=np.float32)
        s_t_arr = np.asarray(s_trues, dtype=np.float32)
        h_p_arr = np.asarray(h_preds, dtype=np.float32)
        h_t_arr = np.asarray(h_trues, dtype=np.float32)
        last_val_metrics = {
            "val_loss": val_loss,
            "val_mae_s": float(np.mean(np.abs(s_p_arr - s_t_arr))),
            "val_mae_h": float(np.mean(np.abs(h_p_arr - h_t_arr))),
            "val_pearson_s": _safe_pearsonr(s_p_arr, s_t_arr),
            "val_pearson_h": _safe_pearsonr(h_p_arr, h_t_arr),
        }
        if r_preds:
            r_p_arr = np.asarray(r_preds, dtype=np.float32)
            r_t_arr = np.asarray(r_trues, dtype=np.float32)
            c_p_arr = np.asarray(c_preds, dtype=np.float32)
            c_t_arr = np.asarray(c_trues, dtype=np.float32)
            last_val_metrics["val_mae_r"] = float(np.mean(np.abs(r_p_arr - r_t_arr)))
            last_val_metrics["val_mae_c"] = float(np.mean(np.abs(c_p_arr - c_t_arr)))
            last_val_metrics["val_pearson_r"] = _safe_pearsonr(r_p_arr, r_t_arr)
            last_val_metrics["val_pearson_c"] = _safe_pearsonr(c_p_arr, c_t_arr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    student.load_state_dict(best_state)
    ckpt_dir = student.save_checkpoint(Path(output_dir), tokenizer=tok)

    report = {
        "lang": lang,
        "model_name": student.model_name,
        "edge_dim": student.edge_dim,
        "sanity_check_shuffle": bool(sanity_check_shuffle),
        "train_losses": train_losses,
        "best_val_loss": best_val_loss,
        **last_val_metrics,
        "checkpoint_dir": str(ckpt_dir),
    }
    (Path(output_dir) / "distill_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def extract_parent_reply_pairs_from_parquet(interim_dir: Path) -> list[dict[str, Any]]:
    """Read validated `nodes.parquet` and `edges.parquet` into `(u, v)` text pair records."""
    nodes_tbl, edges_tbl = load_cascade_parquet(interim_dir, validate=True)
    node_rows = nodes_tbl.to_pylist()
    node_text_by_id = {str(row["node_id"]): str(row["text"]) for row in node_rows}
    node_topic_by_id = {str(row["node_id"]): str(row["topic_id"]) for row in node_rows}
    cascade_root_id: dict[str, str] = {}
    for row in node_rows:
        if bool(row.get("is_root")):
            cascade_root_id[str(row["cascade_id"])] = str(row["node_id"])

    pairs: list[dict[str, Any]] = []
    for edge in edges_tbl.to_pylist():
        cid = str(edge["cascade_id"])
        src = str(edge["source_id"])
        tgt = str(edge["target_id"])
        root_id = cascade_root_id.get(cid, src)
        pairs.append(
            {
                "edge_id": str(edge["edge_id"]),
                "cascade_id": cid,
                "topic_id": node_topic_by_id[src],
                "root_id": root_id,
                "parent_id": src,
                "target_id": tgt,
                "root_text": node_text_by_id.get(root_id, node_text_by_id[src]),
                "parent_text": node_text_by_id[src],
                "reply_text": node_text_by_id[tgt],
            }
        )
    return pairs


def split_pairs_by_inductive_topics(
    pairs: Sequence[Mapping[str, Any]],
    *,
    split_seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    """Partition `(u, v)` edge pairs strictly by `topic_id` so test topics never leak into distillation."""
    if not pairs:
        raise ValueError("Cannot split empty pair sequence")
    normalized = [
        {**dict(p), "topic_id": str(p.get("topic_id") or p.get("cascade_id") or "default")}
        for p in pairs
    ]
    split_indices = inductive_topic_split(
        normalized,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )
    train_pairs = [normalized[i] for i in split_indices["train"]]
    val_pairs = [normalized[i] for i in split_indices["val"]]
    test_pairs = [normalized[i] for i in split_indices["test"]]

    tr_topics = {str(p["topic_id"]) for p in train_pairs}
    va_topics = {str(p["topic_id"]) for p in val_pairs}
    te_topics = {str(p["topic_id"]) for p in test_pairs}
    assert tr_topics.isdisjoint(te_topics) and va_topics.isdisjoint(te_topics), (
        "Topic leakage detected between Cross-Encoder distillation and held-out test set"
    )
    return {"train": train_pairs, "val": val_pairs, "test": test_pairs}


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _score_pairs_with_teachers(
    pairs: list[dict[str, Any]],
    *,
    lang: str,
    vihsd_ckpt: Path | None = None,
    batch_size: int = 16,
    allow_untrained_fallback: bool = False,
    cache_path: Path | None = None,
    checkpoint_every: int = 250,
    teacher_backend: str = "legacy_nli",
    teacher_model: str = QWEN_TEACHER_MODEL_ID,
    qlora_adapter_dir: Path | None = None,
    qwen_scorer: QwenBehavioralTeacherScorer | None = None,
) -> list[dict[str, Any]]:
    """Run teacher scorers (`legacy_nli` or `qwen_qlora`) with atomic shard checkpointing."""
    if not pairs:
        return pairs

    chunk_size = max(1, int(checkpoint_every))
    cached_by_eid: dict[str, dict[str, Any]] = {}
    if cache_path is not None and Path(cache_path).exists():
        try:
            loaded_cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            if isinstance(loaded_cache, list):
                for item in loaded_cache:
                    eid = str(item.get("edge_id") or "")
                    if eid:
                        cached_by_eid[eid] = item
        except (OSError, json.JSONDecodeError):
            cached_by_eid = {}

    cache_fields = (
        "h_uv",
        "s_uv",
        "r_uv",
        "c_uv",
        "behavior_probs",
        "behavior_label",
        "stance_probs",
        "pred_label",
        "root_s_uv",
    )
    for p in pairs:
        eid = str(p.get("edge_id") or "")
        cached = cached_by_eid.get(eid)
        if cached is not None:
            for field in cache_fields:
                if field in cached:
                    p[field] = cached[field]

    if teacher_backend == "qwen_qlora":
        need_idx = [
            i
            for i, p in enumerate(pairs)
            if any(k not in p for k in ("s_uv", "h_uv", "r_uv", "c_uv", "behavior_label", "root_s_uv"))
        ]
        if need_idx:
            owns_scorer = qwen_scorer is None
            scorer = qwen_scorer or QwenBehavioralTeacherScorer(
                model_name_or_path=teacher_model,
                adapter_dir=qlora_adapter_dir,
                lang=lang,
                batch_size=max(1, min(int(batch_size), 8)),
            )
            chunk_starts = range(0, len(need_idx), chunk_size)
            for start in tqdm(chunk_starts, desc=f"Qwen3.8-27B QLoRA teacher ({lang})"):
                idx_slice = need_idx[start : start + chunk_size]
                need_edge = [
                    i
                    for i in idx_slice
                    if any(k not in pairs[i] for k in ("s_uv", "h_uv", "r_uv", "c_uv", "behavior_label"))
                ]
                if need_edge:
                    out = scorer.score_triplets(
                        [str(pairs[i].get("root_text") or pairs[i]["parent_text"]) for i in need_edge],
                        [str(pairs[i]["parent_text"]) for i in need_edge],
                        [str(pairs[i]["reply_text"]) for i in need_edge],
                        lang=lang,
                    )
                    for pos, i in enumerate(need_edge):
                        pairs[i]["s_uv"] = float(out["s_uv"][pos])
                        pairs[i]["h_uv"] = float(out["h_uv"][pos])
                        pairs[i]["r_uv"] = float(out["r_uv"][pos])
                        pairs[i]["c_uv"] = float(out["c_uv"][pos])
                        pairs[i]["behavior_probs"] = out["behavior_probs"][pos].tolist()
                        pairs[i]["behavior_label"] = out["behavior_label"][pos]
                        pairs[i]["stance_probs"] = out["stance_probs"][pos].tolist()
                        pairs[i]["pred_label"] = out["pred_label"][pos]

                deep_indices: list[int] = []
                for i in idx_slice:
                    if "root_s_uv" in pairs[i]:
                        continue
                    root_txt = pairs[i].get("root_text")
                    root_id = str(pairs[i].get("root_id") or "")
                    parent_id = str(pairs[i].get("parent_id") or "")
                    is_deep = bool(root_txt) and (
                        (root_id and parent_id and root_id != parent_id)
                        or str(root_txt) != str(pairs[i].get("parent_text") or "")
                    )
                    if is_deep:
                        deep_indices.append(i)
                    else:
                        pairs[i]["root_s_uv"] = float(pairs[i]["s_uv"])

                if deep_indices:
                    root_out = scorer.score_triplets(
                        [str(pairs[i]["root_text"]) for i in deep_indices],
                        [str(pairs[i]["root_text"]) for i in deep_indices],
                        [str(pairs[i]["reply_text"]) for i in deep_indices],
                        lang=lang,
                    )
                    for pos, i in enumerate(deep_indices):
                        pairs[i]["root_s_uv"] = float(root_out["s_uv"][pos])

                if cache_path is not None:
                    _atomic_write_json(Path(cache_path), pairs)

            if owns_scorer:
                del scorer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return pairs

    # Legacy dual teacher path (Hostility RoBERTa/ViSoBERT + mDeBERTa NLI)
    # Stage 1: Hostility intensity h_uv
    need_host_idx = [i for i, p in enumerate(pairs) if "h_uv" not in p]
    if need_host_idx:
        host_scorer = HostilityTeacherScorer(
            lang=lang,
            model_name_or_path=vihsd_ckpt if lang == "vi" else None,
            allow_untrained_fallback=allow_untrained_fallback,
        )
        chunk_starts = range(0, len(need_host_idx), chunk_size)
        for start in tqdm(chunk_starts, desc=f"Hostility teacher ({lang})"):
            idx_slice = need_host_idx[start : start + chunk_size]
            h_scores = host_scorer.score_texts(
                [pairs[i]["reply_text"] for i in idx_slice],
                batch_size=batch_size * 2,
            )
            for pos, i in enumerate(idx_slice):
                pairs[i]["h_uv"] = float(h_scores[pos])
            if cache_path is not None:
                _atomic_write_json(Path(cache_path), pairs)
        del host_scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stage 2: NLI stance score s_uv + direct (root -> reply) root_s_uv for depth >= 2
    need_nli_idx = [
        i for i, p in enumerate(pairs) if "s_uv" not in p or "root_s_uv" not in p
    ]
    if need_nli_idx:
        nli_scorer = NLIStanceTeacherScorer()
        chunk_starts = range(0, len(need_nli_idx), chunk_size)
        for start in tqdm(chunk_starts, desc=f"NLI stance teacher ({lang})"):
            idx_slice = need_nli_idx[start : start + chunk_size]
            need_edge_pos = [i for i in idx_slice if "s_uv" not in pairs[i]]
            if need_edge_pos:
                nli_out = nli_scorer.score_pairs(
                    [pairs[i]["parent_text"] for i in need_edge_pos],
                    [pairs[i]["reply_text"] for i in need_edge_pos],
                    batch_size=batch_size,
                )
                for pos, i in enumerate(need_edge_pos):
                    pairs[i]["s_uv"] = float(nli_out["s_uv"][pos])
                    pairs[i]["stance_probs"] = nli_out["stance_probs"][pos].tolist()
                    pairs[i]["pred_label"] = nli_out["pred_label"][pos]

            deep_indices = []
            for i in idx_slice:
                if "root_s_uv" in pairs[i]:
                    continue
                root_txt = pairs[i].get("root_text")
                root_id = str(pairs[i].get("root_id") or "")
                parent_id = str(pairs[i].get("parent_id") or "")
                is_deep = bool(root_txt) and (
                    (root_id and parent_id and root_id != parent_id)
                    or str(root_txt) != str(pairs[i].get("parent_text") or "")
                )
                if is_deep:
                    deep_indices.append(i)
                else:
                    pairs[i]["root_s_uv"] = float(pairs[i]["s_uv"])

            if deep_indices:
                root_out = nli_scorer.score_pairs(
                    [str(pairs[i]["root_text"]) for i in deep_indices],
                    [str(pairs[i]["reply_text"]) for i in deep_indices],
                    batch_size=batch_size,
                )
                for pos, i in enumerate(deep_indices):
                    pairs[i]["root_s_uv"] = float(root_out["s_uv"][pos])

            if cache_path is not None:
                _atomic_write_json(Path(cache_path), pairs)
        del nli_scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual open-source NLP edge scoring & distillation.")
    parser.add_argument("--lang", choices=["vi", "en"], default="vi")
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim/vn"))
    parser.add_argument("--vihsd-dir", type=Path, default=Path("checkpoints/visobert_vihsd"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument("--pilot-csv", type=Path, default=None)
    parser.add_argument("--machine-key-csv", type=Path, default=None)
    parser.add_argument("--scored-cache-json", type=Path, default=None)
    parser.add_argument("--train-vihsd", action="store_true")
    parser.add_argument("--train-qwen-qlora", action="store_true")
    parser.add_argument(
        "--teacher-backend",
        choices=["legacy_nli", "qwen_qlora"],
        default="legacy_nli",
    )
    parser.add_argument("--teacher-model", type=str, default=QWEN_TEACHER_MODEL_ID)
    parser.add_argument(
        "--qlora-adapter-dir",
        type=Path,
        default=Path("checkpoints/qwen38_27b_qlora"),
    )
    parser.add_argument("--edge-dim", type=int, choices=[2, 4], default=2)
    parser.add_argument("--student-backbone", type=str, default=None)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--export-audit", action="store_true")
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--include-root-context", action="store_true")
    parser.add_argument("--sanity-check-shuffle", action="store_true")
    parser.add_argument("--allow-untrained-vihsd", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-teacher-pairs", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    out_dir = args.output_dir or Path(
        f"checkpoints/dual_ce_{args.lang}{'_sanity_shuffle' if args.sanity_check_shuffle else ''}"
    )
    audit_csv = args.audit_csv or Path(f"data/interim/human_audit_{args.lang}_300.csv")
    machine_key_csv = args.machine_key_csv or audit_csv.with_name(
        f"{audit_csv.stem}_machine_key.csv"
    )

    if args.train_vihsd:
        finetune_visobert_vihsd(
            output_dir=args.vihsd_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    if args.train_qwen_qlora:
        finetune_qwen_qlora(
            output_dir=args.qlora_adapter_dir,
            model_name_or_path=args.teacher_model,
            lang=args.lang,
            pilot_gold_csv=args.pilot_csv if (args.pilot_csv is not None and args.pilot_csv.exists()) else None,
            epochs=1,
            batch_size=2,
        )

    if args.score_only or args.export_audit or args.distill:
        pairs = extract_parent_reply_pairs_from_parquet(args.interim_dir)
        if args.max_pairs is not None:
            pairs = pairs[: args.max_pairs]

        presplit_train_count: int | None = None
        if (
            args.teacher_backend == "qwen_qlora"
            and args.max_teacher_pairs is not None
            and len(pairs) > args.max_teacher_pairs
        ):
            # Budget Teacher scoring strictly inside train + val topics (ZERO test-topic leakage)
            topic_splits = split_pairs_by_inductive_topics(pairs, split_seed=args.split_seed)
            rng_budget = random.Random(args.split_seed)
            tr_pool = list(topic_splits["train"])
            va_pool = list(topic_splits["val"])
            rng_budget.shuffle(tr_pool)
            rng_budget.shuffle(va_pool)
            tr_quota = min(len(tr_pool), max(1, int(round(0.82 * args.max_teacher_pairs))))
            va_quota = min(len(va_pool), max(1, int(args.max_teacher_pairs) - tr_quota))
            pairs = tr_pool[:tr_quota] + va_pool[:va_quota]
            presplit_train_count = tr_quota

        scored = _score_pairs_with_teachers(
            pairs,
            lang=args.lang,
            vihsd_ckpt=args.vihsd_dir,
            batch_size=args.batch_size,
            allow_untrained_fallback=args.allow_untrained_vihsd,
            cache_path=args.scored_cache_json,
            checkpoint_every=args.checkpoint_every,
            teacher_backend=args.teacher_backend,
            teacher_model=args.teacher_model,
            qlora_adapter_dir=args.qlora_adapter_dir,
        )
        if args.scored_cache_json is not None:
            _atomic_write_json(args.scored_cache_json, scored)

        if args.export_audit:
            csv_path = export_stratified_human_audit_csv(
                scored,
                output_csv=audit_csv,
                machine_key_csv=machine_key_csv,
                pilot_csv=args.pilot_csv,
            )
            print(f"Exported blind stratified human audit CSV: {csv_path} (key: {machine_key_csv})")

        if args.distill:
            if presplit_train_count is not None:
                train_distill = scored[:presplit_train_count]
                val_distill = scored[presplit_train_count:]
            else:
                topic_splits = split_pairs_by_inductive_topics(
                    scored, split_seed=args.split_seed
                )
                train_distill = topic_splits["train"]
                val_distill = topic_splits["val"]
            student_backbone = args.student_backbone or (
                VISOBERT_MODEL_NAME if args.lang == "vi" else EN_DEBERTA_MODEL_NAME
            )
            report = distill_dual_cross_encoder(
                train_distill,
                val_distill,
                output_dir=out_dir,
                lang=args.lang,
                model_name=student_backbone,
                edge_dim=args.edge_dim,
                include_root_context=bool(args.include_root_context),
                epochs=args.epochs,
                batch_size=args.batch_size,
                sanity_check_shuffle=args.sanity_check_shuffle,
            )
            print("Distillation summary:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

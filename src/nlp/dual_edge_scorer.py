"""Dual-branch open-source NLP edge scorer and student distillation (`s_uv`, `h_uv`).

Operates exclusively with open Hugging Face Hub models (no paid APIs):
  1. Hostility Teacher (`h_uv in [0, 1]`):
       - Vietnamese: `uitnlp/visobert` fine-tuned on `ura-hcmut/UIT-ViHSD`
         with 3 labels {0: Clean, 1: Offensive, 2: Hate} -> `h_uv = P(Offensive) + P(Hate)`.
       - English (PHEME): `cardiffnlp/twitter-roberta-base-offensive`
         -> `h_uv = P(offensive)`.
  2. Stance NLI Teacher (`s_uv in [-1, 1]`):
       - `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
         using a symmetric 2-hypothesis Stance Verbalizer over
         `(parent_text u, reply_text v)` -> `s_uv = P(support) - P(oppose)`.
  3. Blind Stratified Human Audit CSV exporter (`data/interim/human_audit_300.csv`).
  4. Unified segment-isolated Student `DualHeadCrossEncoder` (`uitnlp/visobert`
     for `vi`, `cardiffnlp/twitter-roberta-base` for `en`) distilled with
     temperature-scaled KL, `BCEWithLogitsLoss`, and `--sanity-check-shuffle`.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import inspect
import json
import math
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
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

from src.data.pyg_dataset import (
    DEFAULT_FORECAST_MIN_REPLIES,
    compute_canonical_topic_splits,
    inductive_topic_split,
    save_topic_splits_json,
)
from src.data.schema import load_cascade_parquet
from src.nlp.discourse_labels import DISCOURSE_BEHAVIOR_LABELS, behavior_probs_to_stance_probs
from src.utils.seed import seed_everything
from src.utils.text import normalize_nfc

VISOBERT_MODEL_NAME = "uitnlp/visobert"
UIT_VIHSD_DATASET_NAME = "ura-hcmut/UIT-ViHSD"
EN_OFFENSIVE_MODEL_NAME = "cardiffnlp/twitter-roberta-base-offensive"
MDEBERTA_NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
TWITTER_ROBERTA_MODEL_NAME = "cardiffnlp/twitter-roberta-base"
DEFAULT_EN_BACKBONE = TWITTER_ROBERTA_MODEL_NAME

STANCE_LABELS: tuple[str, ...] = ("support", "neutral", "oppose")

_EN_MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
_EN_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MULTI_WS_RE = re.compile(r"\s+")


def _make_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _cuda_autocast(enabled: bool, *, dtype: torch.dtype = torch.float16) -> Any:
    if not torch.cuda.is_available():
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=dtype, enabled=bool(enabled))
    return torch.cuda.amp.autocast(dtype=dtype, enabled=bool(enabled))


def _should_use_fp16_amp(device: torch.device) -> bool:
    """Use FP16 AMP on CUDA Tensor-Core GPUs; skip on GTX 16xx/10xx where FP16 emulation is ~5x slower."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = str(torch.cuda.get_device_name(idx) or "").lower()
        if "gtx 16" in name or "gtx 10" in name or "gtx 9" in name:
            return False
    except Exception:
        pass
    return True


def _can_use_dataparallel(device: torch.device, use_multi_gpu: bool) -> bool:
    return bool(
        use_multi_gpu
        and device.type == "cuda"
        and (device.index is None or device.index == 0)
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    )


def _resolve_teacher_amp_policy(model: Any, device: torch.device) -> tuple[bool, torch.dtype]:
    """Select numerically safe AMP dtype for inference (avoids mDeBERTa-v3 FP16 overflow on T4)."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return False, torch.float32
    cfg = getattr(model, "config", None)
    model_type = str(getattr(cfg, "model_type", "") or "").lower()
    cls_name = type(model).__name__.lower()
    is_deberta = "deberta" in model_type or "deberta" in cls_name
    if is_deberta:
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
        # Tesla T4 (Turing) lacks native BF16 and DeBERTa-v2/v3 overflows FP16 attention logits.
        return False, torch.float32
    return True, torch.float16


def _truncate_head_tail(token_ids: Sequence[int], budget: int, head_ratio: float = 0.6) -> list[int]:
    """Preserve opening claim (60%) and closing verdict/TL;DR (40%) when over token budget."""
    b = max(0, int(budget))
    if b == 0:
        return []
    if len(token_ids) <= b or b < 4:
        return list(token_ids[:b])
    head_n = max(1, int(round(b * head_ratio)))
    tail_n = max(0, b - head_n)
    return list(token_ids[:head_n]) + (list(token_ids[-tail_n:]) if tail_n > 0 else [])


def _normalize_nfc(text: Any) -> str:
    return normalize_nfc(text)


def normalize_social_text(text: Any, *, lang: str = "vi") -> str:
    """Normalize social-media text (`@handle -> @user` EN-only, URLs -> `http` & whitespace collapse for EN/VI)."""
    norm = _normalize_nfc(text)
    if norm:
        if lang == "en":
            norm = _EN_MENTION_RE.sub("@user", norm)
        norm = _EN_URL_RE.sub("http", norm)
        norm = _MULTI_WS_RE.sub(" ", norm).strip()
    return norm


def _effective_max_length(
    max_length: int,
    *,
    encoder: Any | None = None,
    tokenizer: Any | None = None,
) -> int:
    """Clamp sequence max_length safely to encoder position-embedding capacity."""
    eff = max(8, int(max_length))
    cfg = getattr(encoder, "config", None) if encoder is not None else None
    max_pos = getattr(cfg, "max_position_embeddings", None)
    pad_id = getattr(cfg, "pad_token_id", None)
    if pad_id is None and tokenizer is not None:
        pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        mtype = str(getattr(cfg, "model_type", "")).lower()
        pad_id = 1 if ("roberta" in mtype or "xlm" in mtype) else 0
    if isinstance(max_pos, int) and max_pos > 8:
        eff = min(eff, max(8, max_pos - int(pad_id) - 1))
    return eff


_HF_WEIGHT_FILENAMES: tuple[str, ...] = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "dual_head_weights.pt",
)


def _has_hf_weight_files(ckpt_dir: str | Path) -> bool:
    """Return True iff *ckpt_dir* exists and contains at least one non-empty weight file."""
    path = Path(ckpt_dir)
    if not path.is_dir():
        return False
    for fname in _HF_WEIGHT_FILENAMES:
        cand = path / fname
        try:
            if cand.is_file() and cand.stat().st_size > 0:
                return True
        except OSError:
            continue
    for pattern in ("*.safetensors", "*.bin"):
        for cand in path.glob(pattern):
            try:
                if cand.is_file() and cand.stat().st_size > 0:
                    return True
            except OSError:
                continue
    return False


def _looks_like_local_path(model_name_or_path: str | Path) -> bool:
    raw = str(model_name_or_path)
    p = Path(raw)
    return p.exists() or p.is_absolute() or raw.startswith((".", "/", "\\", "checkpoints")) or "\\" in raw


def _extract_special_tok_str(val: Any, default: str) -> str:
    if isinstance(val, str) and val:
        return val
    if isinstance(val, Mapping):
        c = val.get("content")
        if isinstance(c, str) and c:
            return c
    return default


def _load_fast_tokenizer_from_dir(target_dir: Path) -> Any | None:
    """Load `tokenizer.json` via PreTrainedTokenizerFast when AutoTokenizer hits transformers 5.x/SentencePiece errors."""
    tok_json = target_dir / "tokenizer.json"
    if not tok_json.is_file():
        return None
    cfg: dict[str, Any] = {}
    for cfg_name in ("tokenizer_config.json", "special_tokens_map.json"):
        cfg_path = target_dir / cfg_name
        if cfg_path.is_file():
            try:
                loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    cfg.update(loaded)
            except Exception:
                pass
    try:
        from transformers import PreTrainedTokenizerFast

        return PreTrainedTokenizerFast(
            tokenizer_file=str(tok_json),
            bos_token=_extract_special_tok_str(cfg.get("bos_token"), "<s>"),
            eos_token=_extract_special_tok_str(cfg.get("eos_token"), "</s>"),
            unk_token=_extract_special_tok_str(cfg.get("unk_token"), "<unk>"),
            pad_token=_extract_special_tok_str(cfg.get("pad_token"), "<pad>"),
            cls_token=_extract_special_tok_str(cfg.get("cls_token"), "<s>"),
            sep_token=_extract_special_tok_str(cfg.get("sep_token"), "</s>"),
            mask_token=_extract_special_tok_str(cfg.get("mask_token"), "<mask>"),
        )
    except Exception:
        return None


def load_tokenizer_safe(model_name_or_path: str | Path) -> Any:
    """Load an AutoTokenizer with `use_fast=False` and `tokenizer.json` fast fallback."""
    target = str(model_name_or_path)
    last_exc: Exception | None = None
    try:
        return AutoTokenizer.from_pretrained(target)
    except Exception as exc:
        last_exc = exc
    try:
        return AutoTokenizer.from_pretrained(target, use_fast=False)
    except Exception as exc:
        last_exc = exc
    target_path = Path(target)
    if target_path.is_dir():
        fast_tok = _load_fast_tokenizer_from_dir(target_path)
        if fast_tok is not None:
            return fast_tok
    if last_exc is not None:
        raise last_exc
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
    """Fine-tune ViSoBERT (FP16 AMP, class-weighted CE, warmup+cosine LR) for UIT-ViHSD."""
    seed_everything(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = _should_use_fp16_amp(target_device)

    ds = dataset_dict if dataset_dict is not None else load_uit_vihsd_dataset()
    train_split = ds["train"]
    if "validation" in ds:
        val_split = ds["validation"]
        val_split_source = "dataset_validation"
        train_texts = [normalize_social_text(x, lang="vi") for x in train_split["free_text"]]
        train_labels = [int(x) for x in train_split["label_id"]]
        val_texts = [normalize_social_text(x, lang="vi") for x in val_split["free_text"]]
        val_labels = [int(x) for x in val_split["label_id"]]
    elif "val" in ds:
        val_split = ds["val"]
        val_split_source = "dataset_val"
        train_texts = [normalize_social_text(x, lang="vi") for x in train_split["free_text"]]
        train_labels = [int(x) for x in train_split["label_id"]]
        val_texts = [normalize_social_text(x, lang="vi") for x in val_split["free_text"]]
        val_labels = [int(x) for x in val_split["label_id"]]
    elif "dev" in ds:
        val_split = ds["dev"]
        val_split_source = "dataset_dev"
        train_texts = [normalize_social_text(x, lang="vi") for x in train_split["free_text"]]
        train_labels = [int(x) for x in train_split["label_id"]]
        val_texts = [normalize_social_text(x, lang="vi") for x in val_split["free_text"]]
        val_labels = [int(x) for x in val_split["label_id"]]
    else:
        warnings.warn(
            "UIT-ViHSD dataset has no 'validation'/'val'/'dev' split; carving a deterministic "
            "stratified 15% validation split from 'train' (zero 'test' leakage).",
            UserWarning,
            stacklevel=2,
        )
        val_split_source = "stratified_train_holdout"
        all_tr_texts = [normalize_social_text(x, lang="vi") for x in train_split["free_text"]]
        all_tr_labels = [int(x) for x in train_split["label_id"]]
        n_all = len(all_tr_texts)
        if n_all < 2:
            raise ValueError("Cannot carve validation split from fewer than 2 training samples")
        indices = np.arange(n_all)
        counts_tr = np.bincount(np.asarray(all_tr_labels, dtype=np.int64))
        nonzero_counts = counts_tr[counts_tr > 0]
        val_n = max(1, int(round(0.15 * n_all)))
        can_stratify = (
            nonzero_counts.size >= 1
            and int(nonzero_counts.min()) >= 2
            and val_n >= nonzero_counts.size
            and (n_all - val_n) >= nonzero_counts.size
        )
        try:
            tr_idx, va_idx = train_test_split(
                indices,
                test_size=val_n,
                random_state=seed,
                shuffle=True,
                stratify=all_tr_labels if can_stratify else None,
            )
        except ValueError:
            tr_idx, va_idx = train_test_split(
                indices,
                test_size=val_n,
                random_state=seed,
                shuffle=True,
                stratify=None,
            )
        train_texts = [all_tr_texts[int(i)] for i in tr_idx]
        train_labels = [all_tr_labels[int(i)] for i in tr_idx]
        val_texts = [all_tr_texts[int(i)] for i in va_idx]
        val_labels = [all_tr_labels[int(i)] for i in va_idx]

    if max_train_samples is not None and len(train_texts) > max_train_samples:
        train_texts = train_texts[:max_train_samples]
        train_labels = train_labels[:max_train_samples]

    # Dampened inverse-frequency class weights computed strictly on train_labels
    counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=3)
    total = float(counts.sum())
    weights = np.ones(3, dtype=np.float32)
    nonzero_mask = counts > 0
    if total > 0.0 and bool(nonzero_mask.any()):
        for c in range(3):
            if counts[c] > 0:
                weights[c] = math.sqrt(total / (3.0 * float(counts[c])))
        mean_nz = float(weights[nonzero_mask].mean())
        if mean_nz > 0.0:
            weights = weights / mean_nz
            weights[~nonzero_mask] = 1.0
    class_weights_t = torch.tensor(weights, dtype=torch.float32, device=target_device)

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
    eff_max_len = _effective_max_length(max_length, encoder=clf, tokenizer=tok)

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

    no_decay_tags = ("bias", "layernorm", "layer_norm")
    decay_params, no_decay_params = [], []
    for pname, param in clf.named_parameters():
        if not param.requires_grad:
            continue
        low_p = pname.lower()
        if low_p.endswith("bias") or any(tag in low_p for tag in no_decay_tags):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    opt_groups = []
    if decay_params:
        opt_groups.append({"params": decay_params, "lr": lr, "weight_decay": 0.01})
    if no_decay_params:
        opt_groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(opt_groups or clf.parameters(), lr=lr)
    scaler = _make_grad_scaler(enabled=use_amp)

    total_steps = max(1, max(1, epochs) * max(1, len(train_loader)))
    warmup_steps = max(1, int(round(0.10 * total_steps)))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return 0.1 + 0.9 * (float(step + 1) / float(warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    best_f1 = -1.0
    f1_history: list[float] = []
    best_state: dict[str, Tensor] = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}

    for _epoch in range(max(1, epochs)):
        clf.train()
        for batch_texts, batch_labels in train_loader:
            enc = tok(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=eff_max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(target_device) for k, v in enc.items()}
            labels_t = torch.as_tensor(batch_labels, dtype=torch.long, device=target_device)

            optimizer.zero_grad(set_to_none=True)
            with _cuda_autocast(enabled=use_amp):
                logits = clf(**enc).logits
            loss = F.cross_entropy(logits.float(), labels_t, weight=class_weights_t)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale() if use_amp else 1.0
            scaler.step(optimizer)
            scaler.update()
            if not use_amp or scaler.get_scale() >= scale_before:
                scheduler.step()

        clf.eval()
        preds: list[int] = []
        trues: list[int] = []
        with torch.no_grad():
            for batch_texts, batch_labels in val_loader:
                enc = tok(
                    list(batch_texts),
                    padding=True,
                    truncation=True,
                    max_length=eff_max_len,
                    return_tensors="pt",
                )
                enc = {k: v.to(target_device) for k, v in enc.items()}
                with _cuda_autocast(enabled=use_amp):
                    logits = clf(**enc).logits
                preds.extend(logits.float().argmax(dim=-1).cpu().tolist())
                trues.extend(torch.as_tensor(batch_labels).tolist())

        macro_f1 = float(f1_score(trues, preds, average="macro", zero_division=0))
        f1_history.append(macro_f1)
        if macro_f1 >= best_f1:
            best_f1 = macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}

    clf.load_state_dict(best_state)
    clf.eval()

    # Post-hoc scalar temperature & uncivil decision threshold calibration strictly on val_split
    val_logits_list: list[Tensor] = []
    with torch.no_grad():
        for batch_texts, _batch_labels in val_loader:
            enc = tok(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=eff_max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(target_device) for k, v in enc.items()}
            with _cuda_autocast(enabled=use_amp):
                logits = clf(**enc).logits
            val_logits_list.append(logits.float().cpu())

    if val_logits_list:
        val_logits_t = torch.cat(val_logits_list, dim=0)
        val_probs_np = torch.softmax(val_logits_t, dim=-1).numpy()
        raw_h = np.clip(val_probs_np[:, 1] + val_probs_np[:, 2], 1e-6, 1.0 - 1e-6)
    else:
        raw_h = np.zeros((0,), dtype=np.float64)

    y_val_bin = (np.asarray(val_labels, dtype=np.int64) > 0).astype(np.int64)
    hostility_temperature = 1.0
    calibrated_threshold = 0.5
    if raw_h.size > 0 and y_val_bin.size == raw_h.size:
        raw_bin_logit = np.log(raw_h / (1.0 - raw_h))
        if np.unique(y_val_bin).size >= 2:
            best_nll = float("inf")
            for t_cand in np.linspace(0.5, 4.0, 71):
                p_cand = 1.0 / (1.0 + np.exp(-np.clip(raw_bin_logit / float(t_cand), -30.0, 30.0)))
                p_cand = np.clip(p_cand, 1e-6, 1.0 - 1e-6)
                nll = float(
                    -np.mean(y_val_bin * np.log(p_cand) + (1.0 - y_val_bin) * np.log(1.0 - p_cand))
                )
                if nll < best_nll:
                    best_nll = nll
                    hostility_temperature = float(t_cand)
            lo = max(0.5, hostility_temperature - 0.05)
            hi = min(4.0, hostility_temperature + 0.05)
            for t_cand in np.linspace(lo, hi, 21):
                p_cand = 1.0 / (1.0 + np.exp(-np.clip(raw_bin_logit / float(t_cand), -30.0, 30.0)))
                p_cand = np.clip(p_cand, 1e-6, 1.0 - 1e-6)
                nll = float(
                    -np.mean(y_val_bin * np.log(p_cand) + (1.0 - y_val_bin) * np.log(1.0 - p_cand))
                )
                if nll < best_nll:
                    best_nll = nll
                    hostility_temperature = float(t_cand)

        cal_h = 1.0 / (
            1.0 + np.exp(-np.clip(raw_bin_logit / max(1e-3, hostility_temperature), -30.0, 30.0))
        )
        if int(y_val_bin.sum()) > 0:
            best_tau_f1 = -1.0
            for tau in np.linspace(0.10, 0.95, 86):
                pred_tau = (cal_h >= float(tau)).astype(np.int64)
                f1_tau = float(f1_score(y_val_bin, pred_tau, pos_label=1, zero_division=0))
                if f1_tau > best_tau_f1 + 1e-12 or (
                    abs(f1_tau - best_tau_f1) <= 1e-12
                    and abs(float(tau) - 0.5) < abs(calibrated_threshold - 0.5)
                ):
                    best_tau_f1 = f1_tau
                    calibrated_threshold = round(float(tau), 4)

        p05, r05, f05, _ = precision_recall_fscore_support(
            y_val_bin, (raw_h >= 0.5).astype(np.int64), pos_label=1, average="binary", zero_division=0
        )
        pc, rc, fc, _ = precision_recall_fscore_support(
            y_val_bin,
            (cal_h >= calibrated_threshold).astype(np.int64),
            pos_label=1,
            average="binary",
            zero_division=0,
        )
    else:
        p05 = r05 = f05 = pc = rc = fc = 0.0

    clf.save_pretrained(out_path)
    if hasattr(tok, "save_pretrained"):
        tok.save_pretrained(out_path)
    (out_path / "vihsd_metrics.json").write_text(
        json.dumps(
            {
                "best_val_macro_f1": best_f1,
                "class_weights": [float(w) for w in weights.tolist()],
                "val_macro_f1_history": f1_history,
                "val_split_source": val_split_source,
                "calibrated_hostility_threshold": float(calibrated_threshold),
                "hostility_temperature": float(hostility_temperature),
                "val_uncivil_precision_at_05": float(p05),
                "val_uncivil_recall_at_05": float(r05),
                "val_uncivil_f1_at_05": float(f05),
                "val_uncivil_precision_at_calibrated": float(pc),
                "val_uncivil_recall_at_calibrated": float(rc),
                "val_uncivil_f1_at_calibrated": float(fc),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


class _AutoCastLogitsWrapper(nn.Module):
    """Wrap HF SequenceClassification models so `nn.DataParallel` runs AMP autocast per GPU thread."""

    def __init__(
        self,
        inner: nn.Module,
        *,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.use_amp = bool(use_amp)
        self.amp_dtype = amp_dtype

    def forward(self, **kwargs: Tensor) -> Tensor:
        with _cuda_autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            out = self.inner(**kwargs)
            return out.logits.float()


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
        use_multi_gpu: bool = True,
    ) -> None:
        if lang not in {"vi", "en"}:
            raise ValueError(f"Unsupported lang {lang!r}; expected 'vi' or 'en'")
        self.lang = lang
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        default_name = VISOBERT_MODEL_NAME if lang == "vi" else EN_OFFENSIVE_MODEL_NAME
        self.calibrated_threshold: float = 0.5
        self.hostility_temperature: float = 1.0

        if model is None:
            if lang == "vi":
                if model_name_or_path is None:
                    if not allow_untrained_fallback:
                        raise FileNotFoundError(
                            "Fine-tuned UIT-ViHSD checkpoint is required for lang='vi' hostility "
                            "scoring. Run with --train-vihsd first or pass allow_untrained_fallback=True."
                        )
                elif str(model_name_or_path) != VISOBERT_MODEL_NAME:
                    ckpt_path = Path(model_name_or_path)
                    if not ckpt_path.exists():
                        raise FileNotFoundError(
                            f"Missing fine-tuned ViHSD checkpoint directory: {ckpt_path}"
                        )
                    if not _has_hf_weight_files(ckpt_path):
                        raise FileNotFoundError(
                            f"Missing model weight files (model.safetensors / pytorch_model.bin) "
                            f"in ViHSD checkpoint directory: {ckpt_path}"
                        )
            elif (
                model_name_or_path is not None
                and str(model_name_or_path) != EN_OFFENSIVE_MODEL_NAME
                and _looks_like_local_path(model_name_or_path)
            ):
                ckpt_path = Path(model_name_or_path)
                if not _has_hf_weight_files(ckpt_path):
                    raise FileNotFoundError(
                        f"Missing model weight files (model.safetensors / pytorch_model.bin) "
                        f"in EN hostility checkpoint directory: {ckpt_path}"
                    )

        if model_name_or_path is not None:
            metrics_file = Path(model_name_or_path) / "vihsd_metrics.json"
            if metrics_file.is_file():
                try:
                    m_data = json.loads(metrics_file.read_text(encoding="utf-8"))
                    if "calibrated_hostility_threshold" in m_data:
                        self.calibrated_threshold = float(m_data["calibrated_hostility_threshold"])
                    if "hostility_temperature" in m_data:
                        self.hostility_temperature = max(1e-3, float(m_data["hostility_temperature"]))
                except Exception:
                    pass

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

        self._use_amp, self._amp_dtype = _resolve_teacher_amp_policy(self.model, self.device)
        dp_ok = _can_use_dataparallel(self.device, use_multi_gpu) and isinstance(self.model, nn.Module)
        self._num_gpus = torch.cuda.device_count() if dp_ok else 1
        self._dp_model: nn.Module | None = (
            nn.DataParallel(
                _AutoCastLogitsWrapper(
                    self.model,
                    use_amp=self._use_amp,
                    amp_dtype=self._amp_dtype,
                )
            )
            if dp_ok
            else None
        )

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
        apply_temperature: bool = True,
    ) -> np.ndarray:
        n_texts = len(texts)
        if n_texts == 0:
            return np.zeros((0,), dtype=np.float32)

        eff_batch = max(1, int(batch_size) * max(1, int(self._num_gpus)))
        eff_max_len = _effective_max_length(max_length, encoder=self.model, tokenizer=self.tokenizer)
        id2label = getattr(getattr(self.model, "config", None), "id2label", None)

        norm_texts = [normalize_social_text(t, lang=self.lang) for t in texts]
        # Length-bucket batches to minimize padding while restoring exact input order.
        order = (
            sorted(range(n_texts), key=lambda idx: len(norm_texts[idx]))
            if n_texts > eff_batch
            else list(range(n_texts))
        )
        scores: list[Tensor] = []

        for start in range(0, n_texts, eff_batch):
            batch = [norm_texts[idx] for idx in order[start : start + eff_batch]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=eff_max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            if self._dp_model is not None and len(batch) >= self._num_gpus:
                logits = self._dp_model(**enc).float()
            else:
                with _cuda_autocast(enabled=self._use_amp, dtype=self._amp_dtype):
                    logits = self.model(**enc).logits.float()
            if not torch.isfinite(logits).all():
                self._use_amp = False
                self._amp_dtype = torch.float32
                if self._dp_model is not None:
                    self._dp_model.module.use_amp = False
                    self._dp_model.module.amp_dtype = torch.float32
                    if len(batch) >= self._num_gpus:
                        logits = self._dp_model(**enc).float()
                    else:
                        with _cuda_autocast(enabled=False):
                            logits = self.model(**enc).logits.float()
                else:
                    with _cuda_autocast(enabled=False):
                        logits = self.model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1)
            h = self.hostility_from_probs(probs, lang=self.lang, id2label=id2label)
            host_temp = float(getattr(self, "hostility_temperature", 1.0))
            if apply_temperature and abs(host_temp - 1.0) > 1e-6:
                raw_h = h.clamp(1e-6, 1.0 - 1e-6)
                h = torch.sigmoid(torch.logit(raw_h) / host_temp)
            scores.append(h.cpu())

        ordered_np = torch.cat(scores, dim=0).numpy().astype(np.float32)
        if n_texts <= eff_batch:
            return ordered_np
        out_np = np.empty_like(ordered_np)
        for pos, orig_idx in enumerate(order):
            out_np[orig_idx] = ordered_np[pos]
        return out_np


class NLIStanceTeacherScorer:
    """Compute pairwise NLI stance signal `s_uv` via symmetric 2-hypothesis Stance Verbalizer."""

    def __init__(
        self,
        *,
        model_name_or_path: str | Path = MDEBERTA_NLI_MODEL_NAME,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: torch.device | None = None,
        lang: str = "vi",
        use_stance_verbalizer: bool = True,
        use_multi_gpu: bool = True,
        temperature: float = 1.0,
        stance_margin_threshold: float | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lang = lang
        self.use_stance_verbalizer = bool(use_stance_verbalizer)
        self.temperature = max(1e-3, float(temperature))
        self.stance_margin_threshold = (
            float(stance_margin_threshold) if stance_margin_threshold is not None else None
        )
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

        self._use_amp, self._amp_dtype = _resolve_teacher_amp_policy(self.model, self.device)
        dp_ok = _can_use_dataparallel(self.device, use_multi_gpu) and isinstance(self.model, nn.Module)
        self._num_gpus = torch.cuda.device_count() if dp_ok else 1
        self._dp_model: nn.Module | None = (
            nn.DataParallel(
                _AutoCastLogitsWrapper(
                    self.model,
                    use_amp=self._use_amp,
                    amp_dtype=self._amp_dtype,
                )
            )
            if dp_ok
            else None
        )

        self.entail_idx, self.neutral_idx, self.contra_idx = self._resolve_nli_indices()

    def _resolve_nli_indices(self) -> tuple[int, int, int]:
        cfg = getattr(self.model, "config", None)
        id2label = getattr(cfg, "id2label", None) or {}
        num_labels = getattr(cfg, "num_labels", None)
        if num_labels is not None and int(num_labels) < 3:
            raise ValueError(
                f"Expected 3-class NLI model (entailment/neutral/contradiction), got num_labels={num_labels}"
            )
        if id2label and len(id2label) < 3:
            raise ValueError(
                f"Expected 3-class NLI id2label mapping, got {len(id2label)} labels: {id2label!r}"
            )

        entail: int | None = None
        neutral: int | None = None
        contra: int | None = None
        saw_not_entail = False
        for k, name in id2label.items():
            low = str(name).lower().strip()
            idx = int(k)
            if "not_entail" in low or "not entail" in low:
                saw_not_entail = True
            elif "entail" in low and "not" not in low:
                entail = idx
            elif "neutral" in low:
                neutral = idx
            elif "contradict" in low:
                contra = idx

        if saw_not_entail and contra is None:
            raise ValueError(
                f"2-class NLI label 'not_entailment' is unsupported; expected 3-class entailment/neutral/contradiction ({id2label!r})"
            )

        if entail is None and neutral is None and contra is None:
            generic = all(str(v).lower().startswith("label_") for v in id2label.values())
            if generic and (num_labels is None or int(num_labels) >= 3):
                return 0, 1, 2
            raise ValueError(f"Could not resolve 3-class NLI indices from id2label={id2label!r}")

        if entail is None or neutral is None or contra is None or len({entail, neutral, contra}) != 3:
            raise ValueError(
                f"Incomplete 3-class NLI mapping resolved from id2label={id2label!r}: "
                f"entail={entail}, neutral={neutral}, contra={contra}"
            )
        return entail, neutral, contra

    @staticmethod
    def stance_from_canonical_probs(probs_3: Tensor) -> Tensor:
        """Compute `s_uv = P(support) - P(oppose)` from `[P_sup, P_neu, P_opp]`."""
        return (probs_3[:, 0] - probs_3[:, 2]).clamp(-1.0, 1.0)

    def _score_premise_hypothesis(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        *,
        effective_max_len: int,
    ) -> Tensor:
        try:
            enc = self.tokenizer(
                list(premises),
                list(hypotheses),
                padding=True,
                truncation="only_first",
                max_length=effective_max_len,
                return_tensors="pt",
            )
        except Exception:
            enc = self.tokenizer(
                list(premises),
                list(hypotheses),
                padding=True,
                truncation=True,
                max_length=effective_max_len,
                return_tensors="pt",
            )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        if self._dp_model is not None and len(premises) >= self._num_gpus:
            logits = self._dp_model(**enc).float()
        else:
            with _cuda_autocast(enabled=self._use_amp, dtype=self._amp_dtype):
                logits = self.model(**enc).logits.float()
        if not torch.isfinite(logits).all():
            # Permanently disable AMP for the remainder of this session and rerun across
            # DataParallel (avoiding single-GPU 2x batch OOM & repeated failed FP16 forwards).
            self._use_amp = False
            self._amp_dtype = torch.float32
            if self._dp_model is not None:
                self._dp_model.module.use_amp = False
                self._dp_model.module.amp_dtype = torch.float32
                if len(premises) >= self._num_gpus:
                    logits = self._dp_model(**enc).float()
                else:
                    with _cuda_autocast(enabled=False):
                        logits = self.model(**enc).logits.float()
            else:
                with _cuda_autocast(enabled=False):
                    logits = self.model(**enc).logits.float()
        raw_probs = torch.softmax(logits / float(getattr(self, "temperature", 1.0)), dim=-1)
        return torch.stack(
            [
                raw_probs[:, self.entail_idx],
                raw_probs[:, self.neutral_idx],
                raw_probs[:, self.contra_idx],
            ],
            dim=-1,
        )

    @torch.no_grad()
    def score_pairs(
        self,
        parent_texts: Sequence[str],
        reply_texts: Sequence[str],
        *,
        batch_size: int = 16,
        max_length: int = 384,
        lang: str | None = None,
        use_stance_verbalizer: bool | None = None,
    ) -> dict[str, Any]:
        if len(parent_texts) != len(reply_texts):
            raise ValueError("parent_texts and reply_texts must have equal length")
        if not parent_texts:
            return {
                "s_uv": np.zeros((0,), dtype=np.float32),
                "stance_probs": np.zeros((0, 3), dtype=np.float32),
                "pred_label": [],
            }

        eff_lang = str(lang or self.lang or "vi")
        active_verbalizer = (
            self.use_stance_verbalizer
            if use_stance_verbalizer is None
            else bool(use_stance_verbalizer)
        )
        eff_batch = max(1, int(batch_size) * max(1, int(self._num_gpus)))
        eff_max_len = _effective_max_length(
            max_length, encoder=self.model, tokenizer=self.tokenizer
        )
        all_probs: list[Tensor] = []
        all_s: list[Tensor] = []

        if eff_lang == "en":
            h_sup_tmpl = "The reply agrees with or supports the parent comment."
            h_opp_tmpl = "The reply disagrees with, criticizes, or opposes the parent comment."
            tpl_prefix = "Parent comment: \nReply: "
        else:
            h_sup_tmpl = "Phản hồi này đồng ý, ủng hộ hoặc xác nhận bình luận trước."
            h_opp_tmpl = "Phản hồi này phản đối, bác bỏ hoặc chỉ trích bình luận trước."
            tpl_prefix = "Bình luận trước: \nPhản hồi: "

        has_tok_encode = hasattr(self.tokenizer, "encode") and hasattr(self.tokenizer, "decode")
        if active_verbalizer and has_tok_encode:
            hyp_tok_len = max(
                len(self.tokenizer.encode(h_sup_tmpl, add_special_tokens=False)),
                len(self.tokenizer.encode(h_opp_tmpl, add_special_tokens=False)),
            )
            tpl_overhead = len(self.tokenizer.encode(tpl_prefix, add_special_tokens=False)) + 10
            avail_premise_tok = max(16, eff_max_len - hyp_tok_len - tpl_overhead)
        else:
            avail_premise_tok = max(64, eff_max_len - 48)

        n_pairs = len(parent_texts)
        if active_verbalizer:
            all_premises: list[str] = []
            for u, v in zip(parent_texts, reply_texts):
                u_norm = normalize_social_text(u, lang=eff_lang)
                v_norm = normalize_social_text(v, lang=eff_lang)
                if has_tok_encode:
                    u_ids = self.tokenizer.encode(u_norm, add_special_tokens=False)
                    v_ids = self.tokenizer.encode(v_norm, add_special_tokens=False)
                    v_reserve = min(len(v_ids), max(8, avail_premise_tok // 2))
                    u_max_tok = max(8, avail_premise_tok - v_reserve)
                    u_keep = _truncate_head_tail(u_ids, u_max_tok, head_ratio=0.6)
                    v_max_tok = max(8, avail_premise_tok - len(u_keep))
                    v_keep = v_ids[:v_max_tok]
                    u_trim = (
                        self.tokenizer.decode(u_keep, skip_special_tokens=True).strip()
                        if len(u_ids) > len(u_keep)
                        else u_norm
                    )
                    v_trim = (
                        self.tokenizer.decode(v_keep, skip_special_tokens=True).strip()
                        if len(v_ids) > len(v_keep)
                        else v_norm
                    )
                else:
                    u_trim = u_norm[:300]
                    v_trim = v_norm[:500]
                if eff_lang == "en":
                    all_premises.append(f"Parent comment: {u_trim}\nReply: {v_trim}")
                else:
                    all_premises.append(f"Bình luận trước: {u_trim}\nPhản hồi: {v_trim}")
            order = (
                sorted(range(n_pairs), key=lambda idx: len(all_premises[idx]))
                if n_pairs > eff_batch
                else list(range(n_pairs))
            )
            for start in range(0, n_pairs, eff_batch):
                idx_slice = order[start : start + eff_batch]
                premises = [all_premises[idx] for idx in idx_slice]
                p_sup_pass = self._score_premise_hypothesis(
                    premises, [h_sup_tmpl] * len(premises), effective_max_len=eff_max_len
                )
                p_opp_pass = self._score_premise_hypothesis(
                    premises, [h_opp_tmpl] * len(premises), effective_max_len=eff_max_len
                )
                # Symmetric simplex combination with double-contradiction neutral attribution:
                # contradicting BOTH "agrees" and "disagrees" assigns min(c_sup, c_opp) to neutral.
                e_sup, n_sup, c_sup = p_sup_pass[:, 0], p_sup_pass[:, 1], p_sup_pass[:, 2]
                e_opp, n_opp, c_opp = p_opp_pass[:, 0], p_opp_pass[:, 1], p_opp_pass[:, 2]
                double_contra = torch.minimum(c_sup, c_opp)
                p_sup = 0.5 * (e_sup + (c_opp - double_contra).clamp_min(0.0))
                p_opp = 0.5 * (e_opp + (c_sup - double_contra).clamp_min(0.0))
                p_neu = 0.5 * (n_sup + n_opp) + double_contra
                canonical = torch.stack([p_sup, p_neu, p_opp], dim=-1)
                s = self.stance_from_canonical_probs(canonical)
                all_probs.append(canonical.cpu())
                all_s.append(s.cpu())
        else:
            u_all = [normalize_social_text(t, lang=eff_lang) for t in parent_texts]
            v_all = [normalize_social_text(t, lang=eff_lang) for t in reply_texts]
            order = (
                sorted(range(n_pairs), key=lambda idx: len(u_all[idx]) + len(v_all[idx]))
                if n_pairs > eff_batch
                else list(range(n_pairs))
            )
            for start in range(0, n_pairs, eff_batch):
                idx_slice = order[start : start + eff_batch]
                u_batch = [u_all[idx] for idx in idx_slice]
                v_batch = [v_all[idx] for idx in idx_slice]
                canonical = self._score_premise_hypothesis(
                    u_batch, v_batch, effective_max_len=eff_max_len
                )
                s = self.stance_from_canonical_probs(canonical)
                all_probs.append(canonical.cpu())
                all_s.append(s.cpu())

        probs_ordered = torch.cat(all_probs, dim=0).numpy().astype(np.float32)
        s_ordered = torch.cat(all_s, dim=0).numpy().astype(np.float32)
        if n_pairs <= eff_batch:
            probs_np = probs_ordered
            s_np = s_ordered
        else:
            probs_np = np.empty_like(probs_ordered)
            s_np = np.empty_like(s_ordered)
            for pos, orig_idx in enumerate(order):
                probs_np[orig_idx] = probs_ordered[pos]
                s_np[orig_idx] = s_ordered[pos]
        labels: list[str] = []
        margin_thr = getattr(self, "stance_margin_threshold", None)
        for row_idx in range(probs_np.shape[0]):
            if margin_thr is not None and abs(float(probs_np[row_idx, 0] - probs_np[row_idx, 2])) < margin_thr:
                labels.append("neutral")
            else:
                labels.append(STANCE_LABELS[int(probs_np[row_idx].argmax())])
        return {
            "s_uv": s_np,
            "stance_probs": probs_np,
            "pred_label": labels,
        }


_PARENT_AGREE_RE = re.compile(
    r"(?<![\w\u0300-\u036f])("
    r"đồng ý|đồng tình|chuẩn|đúng rồi|chính xác|tán thành|công nhận|"
    r"bác nói đúng|bạn nói đúng|agree|exactly|you'?re right|true that"
    r")(?![\w\u0300-\u036f])",
    flags=re.IGNORECASE | re.UNICODE,
)
_THIRD_PARTY_TARGET_RE = re.compile(
    r"(?<![\w\u0300-\u036f])("
    r"bọn nó|tụi nó|chúng nó|lũ|bọn kia|tụi kia|bọn|tụi|mấy thằng|mấy con|"
    r"nhà nước|công ty|chủ đầu tư|chính quyền|ban quản lý|"
    r"they|them|those people|those guys|corrupt|scammers"
    r")(?![\w\u0300-\u036f])",
    flags=re.IGNORECASE | re.UNICODE,
)


class HostilityGatedTargetRouter:
    """Disambiguate third-party outrage on the hostile (`h_uv >= threshold`) subset."""

    def __init__(
        self,
        *,
        hostility_gate_threshold: float = 0.5,
        lang: str = "vi",
        nli_scorer: NLIStanceTeacherScorer | None = None,
        llm_reranker: Any | None = None,
    ) -> None:
        self.hostility_gate_threshold = float(hostility_gate_threshold)
        self.lang = str(lang)
        self.nli_scorer = nli_scorer
        self.llm_reranker = llm_reranker

    def route_pairs(
        self,
        pairs: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        routed: list[dict[str, Any]] = []
        gated_count = 0
        rerouted_count = 0

        for pair in pairs:
            item = dict(pair)
            h_val = float(item.get("h_uv", 0.0))
            if h_val < self.hostility_gate_threshold:
                routed.append(item)
                continue

            gated_count += 1
            if self.llm_reranker is not None:
                upd = self.llm_reranker(item)
                if isinstance(upd, Mapping):
                    item.update(upd)
                    if upd.get("target_attributed") == "third_party":
                        rerouted_count += 1
                routed.append(item)
                continue

            u_txt = normalize_social_text(item.get("parent_text") or "", lang=self.lang)
            v_txt = normalize_social_text(item.get("reply_text") or "", lang=self.lang)
            s_val = float(item.get("s_uv", 0.0))
            is_third_party_agree = bool(
                _PARENT_AGREE_RE.search(v_txt) and _THIRD_PARTY_TARGET_RE.search(v_txt)
            )

            if (
                not is_third_party_agree
                and self.nli_scorer is not None
                and hasattr(self.nli_scorer, "_score_premise_hypothesis")
            ):
                u_clip = u_txt[:600]
                v_clip = v_txt[:600]
                if self.lang == "en":
                    premise = f"Parent comment: {u_clip}\nReply: {v_clip}"
                    h_3p = "The reply agrees with the parent commenter while criticizing a third party."
                    h_direct = "The reply directly attacks or disagrees with the parent commenter."
                else:
                    premise = f"Bình luận trước: {u_clip}\nPhản hồi: {v_clip}"
                    h_3p = "Phản hồi này đồng tình với người bình luận trước nhưng chỉ trích một bên thứ ba."
                    h_direct = "Phản hồi này phản đối hoặc công kích trực tiếp người bình luận trước."
                eff_len = _effective_max_length(
                    384,
                    encoder=getattr(self.nli_scorer, "model", None),
                    tokenizer=getattr(self.nli_scorer, "tokenizer", None),
                )
                try:
                    p_3p = self.nli_scorer._score_premise_hypothesis(
                        [premise], [h_3p], effective_max_len=eff_len
                    )[0]
                    p_dir = self.nli_scorer._score_premise_hypothesis(
                        [premise], [h_direct], effective_max_len=eff_len
                    )[0]
                    if float(p_3p[0].item() - p_dir[0].item()) > 0.15:
                        is_third_party_agree = True
                except Exception:
                    pass

            if is_third_party_agree and s_val <= 0.10:
                new_s = max(0.45, min(0.90, abs(s_val) if abs(s_val) >= 0.45 else 0.65))
                old_probs = item.get("stance_probs")
                if isinstance(old_probs, Sequence) and not isinstance(old_probs, (str, bytes)) and len(old_probs) == 3:
                    p_sup_new = max(float(old_probs[2]), 0.60)
                    p_neu_new = max(0.0, min(float(old_probs[1]), p_sup_new - 1e-3))
                    p_opp_new = max(0.0, min(float(old_probs[0]), 0.15, p_sup_new - 1e-3))
                    tot = max(1e-6, p_sup_new + p_neu_new + p_opp_new)
                    item["stance_probs"] = [p_sup_new / tot, p_neu_new / tot, p_opp_new / tot]
                    new_s = max(-1.0, min(1.0, item["stance_probs"][0] - item["stance_probs"][2]))
                else:
                    item["stance_probs"] = _build_default_stance_probs(new_s)
                item["s_uv"] = float(new_s)
                item["pred_label"] = "support"
                item["target_attributed"] = "third_party"
                rerouted_count += 1
            else:
                item.setdefault("target_attributed", "parent")

            routed.append(item)

        stats = {
            "hostility_gate_threshold": self.hostility_gate_threshold,
            "total_pairs": len(pairs),
            "gated_count": gated_count,
            "rerouted_count": rerouted_count,
        }
        return routed, stats


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


_MD_LINK_ONLY_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)]+)\)", re.IGNORECASE)


def _is_unrateable_audit_pair(item: Mapping[str, Any], *, min_words: int = 3) -> tuple[bool, bool]:
    """Return `(hard_unrateable, short_reply)` flags for audit pool hygiene."""
    from src.data.arctic_shift_vn import is_invalid_or_spam_text

    root_txt = str(item.get("root_text") or "")
    parent_txt = str(item.get("parent_text") or "")
    reply_txt = str(item.get("reply_text") or "").strip()
    if "[REDACTED" in root_txt or "[REDACTED" in parent_txt or "[REDACTED" in reply_txt:
        return True, True
    if not reply_txt or is_invalid_or_spam_text(reply_txt) or is_invalid_or_spam_text(parent_txt):
        return True, True
    stripped = _MD_LINK_ONLY_RE.sub(" ", reply_txt)
    stripped = _EN_URL_RE.sub(" ", stripped).strip()
    alnum = re.sub(r"[^\w]", "", stripped, flags=re.UNICODE)
    if len(alnum) < 2:
        return True, True
    word_count = len([w for w in stripped.split() if re.search(r"[^\W_]", w, flags=re.UNICODE)])
    return False, (word_count < int(min_words))


def _take_balanced_from_subbuckets(
    hostile_list: Sequence[dict[str, Any]],
    clean_list: Sequence[dict[str, Any]],
    *,
    h_start: int,
    c_start: int,
    quota: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Sample up to *quota* items balanced 50/50 between hostile/uncivil and clean sub-buckets."""
    avail_h = max(0, len(hostile_list) - h_start)
    avail_c = max(0, len(clean_list) - c_start)
    if quota <= 0 or (avail_h == 0 and avail_c == 0):
        return [], h_start, c_start

    want_h = quota // 2
    want_c = quota - want_h
    take_h = min(avail_h, want_h)
    take_c = min(avail_c, want_c)
    rem = quota - (take_h + take_c)
    if rem > 0:
        extra_h = min(rem, avail_h - take_h)
        take_h += extra_h
        rem -= extra_h
    if rem > 0:
        extra_c = min(rem, avail_c - take_c)
        take_c += extra_c
    chosen = list(hostile_list[h_start : h_start + take_h]) + list(
        clean_list[c_start : c_start + take_c]
    )
    return chosen, h_start + take_h, c_start + take_c


def export_stratified_human_audit_csv(
    scored_pairs: Sequence[Mapping[str, Any]],
    output_csv: Path = Path("data/interim/human_audit_vi_120.csv"),
    *,
    machine_key_csv: Path | None = None,
    pilot_csv: Path | None = None,
    pilot_machine_key_csv: Path | None = None,
    pilot_samples: int = 20,
    n_samples: int = 120,
    hostile_threshold: float = 0.5,
    test_cascade_ids: set[str] | Sequence[str] | None = None,
    exclude_unrateable: bool = True,
    seed: int = 42,
) -> Path:
    """Export balanced stratified blind pairs to CSV (strictly hiding machine predictions)."""
    if not scored_pairs:
        raise ValueError("Cannot export human audit CSV from empty scored_pairs")

    pool = [dict(p) for p in scored_pairs]
    if test_cascade_ids is not None:
        allowed_cids = {str(c) for c in test_cascade_ids}
        test_only = [p for p in pool if str(p.get("cascade_id") or "") in allowed_cids]
        if test_only:
            pool = test_only

    if exclude_unrateable:
        non_redacted = [p for p in pool if not _is_unrateable_audit_pair(p)[0]]
        if non_redacted:
            pool = non_redacted
        rateable = [p for p in pool if not _is_unrateable_audit_pair(p)[1]]
        need_total = min(len(pool), int(n_samples) + (int(pilot_samples) if pilot_csv is not None else 0))
        if len(rateable) >= max(1, need_total):
            pool = rateable

    has_behavior = any(p.get("behavior_label") in DISCOURSE_BEHAVIOR_LABELS for p in pool)
    strata_order = DISCOURSE_BEHAVIOR_LABELS if has_behavior else STANCE_LABELS

    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {lbl: [] for lbl in strata_order}
    hostile_sub: dict[str, list[dict[str, Any]]] = {lbl: [] for lbl in strata_order}
    clean_sub: dict[str, list[dict[str, Any]]] = {lbl: [] for lbl in strata_order}
    h_offsets: dict[str, int] = {lbl: 0 for lbl in strata_order}
    c_offsets: dict[str, int] = {lbl: 0 for lbl in strata_order}
    pop_joint_counts: dict[tuple[str, bool], int] = {}

    for item in pool:
        row_copy = dict(item)
        stance_stratum = _infer_stance_stratum(item)
        row_copy["_stance_stratum"] = stance_stratum
        stratum = _infer_behavior_stratum(item) if has_behavior else stance_stratum
        row_copy["_stratum"] = stratum
        is_uncivil = float(row_copy.get("h_uv", 0.0)) >= float(hostile_threshold)
        row_copy["_is_uncivil"] = is_uncivil
        pop_joint_counts[(stratum, is_uncivil)] = pop_joint_counts.get((stratum, is_uncivil), 0) + 1
        buckets[stratum].append(row_copy)
        if is_uncivil:
            hostile_sub[stratum].append(row_copy)
        else:
            clean_sub[stratum].append(row_copy)

    for lbl in strata_order:
        rng.shuffle(buckets[lbl])
        rng.shuffle(hostile_sub[lbl])
        rng.shuffle(clean_sub[lbl])

    target_total = min(int(n_samples), len(pool))
    selected: list[dict[str, Any]] = []
    base_quota = target_total // len(strata_order)
    offsets: dict[str, int] = {lbl: 0 for lbl in strata_order}

    if not has_behavior:
        for lbl in strata_order:
            picked, nh, nc = _take_balanced_from_subbuckets(
                hostile_sub[lbl], clean_sub[lbl], h_start=0, c_start=0, quota=base_quota
            )
            selected.extend(picked)
            h_offsets[lbl] = nh
            c_offsets[lbl] = nc

        # Round-robin fill remainder while preserving balanced uncivil/clean where possible
        while len(selected) < target_total:
            added_any = False
            for lbl in strata_order:
                if len(selected) >= target_total:
                    break
                want_h_next = h_offsets[lbl] < len(hostile_sub[lbl])
                want_c_next = c_offsets[lbl] < len(clean_sub[lbl])
                if want_h_next and (not want_c_next or (len(selected) % 2 == 0)):
                    selected.append(hostile_sub[lbl][h_offsets[lbl]])
                    h_offsets[lbl] += 1
                    added_any = True
                elif want_c_next:
                    selected.append(clean_sub[lbl][c_offsets[lbl]])
                    c_offsets[lbl] += 1
                    added_any = True
            if not added_any:
                break
    else:
        for lbl in strata_order:
            take = min(base_quota, len(buckets[lbl]))
            selected.extend(buckets[lbl][:take])
            offsets[lbl] = take
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
        "root_text",
        "grandparent_text",
        "parent_text",
        "reply_text",
        "annotator_1_stance",
        "annotator_2_stance",
        "annotator_1_hostility",
        "annotator_2_hostility",
        "annotator_1_uncertain",
        "annotator_2_uncertain",
        "annotator_1_notes",
        "annotator_2_notes",
        "gold_notes",
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
                    "root_text": str(row.get("root_text") or row.get("parent_text") or ""),
                    "grandparent_text": str(row.get("grandparent_text") or ""),
                    "parent_text": str(row.get("parent_text") or ""),
                    "reply_text": str(row.get("reply_text") or ""),
                    "annotator_1_stance": "",
                    "annotator_2_stance": "",
                    "annotator_1_hostility": "",
                    "annotator_2_hostility": "",
                    "annotator_1_uncertain": "",
                    "annotator_2_uncertain": "",
                    "annotator_1_notes": "",
                    "annotator_2_notes": "",
                    "gold_notes": "",
                    "gold_adjudicated_stance": "",
                    "gold_adjudicated_hostility": "",
                }
                if has_behavior:
                    payload_row["annotator_1_behavior"] = ""
                    payload_row["annotator_2_behavior"] = ""
                    payload_row["gold_adjudicated_behavior"] = ""
                writer.writerow(payload_row)

    def _write_key_csv(
        key_dest: Path, rows_to_write: Sequence[Mapping[str, Any]], prefix: str
    ) -> None:
        key_dest.parent.mkdir(parents=True, exist_ok=True)
        key_fields = [
            "sample_id",
            "cascade_id",
            "edge_id",
            "machine_stance_label",
            "s_uv",
            "h_uv",
            "sampling_weight",
        ]
        if has_behavior:
            key_fields.extend(["machine_behavior_label", "r_uv", "c_uv"])

        sel_counts: dict[tuple[str, bool], int] = {}
        for r in rows_to_write:
            k = (str(r["_stratum"]), bool(r["_is_uncivil"]))
            sel_counts[k] = sel_counts.get(k, 0) + 1
        n_pool = max(1, len(pool))
        n_sel = max(1, len(rows_to_write))

        with key_dest.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=key_fields)
            writer.writeheader()
            for idx, row in enumerate(rows_to_write):
                jk = (str(row["_stratum"]), bool(row["_is_uncivil"]))
                pop_share = pop_joint_counts.get(jk, 0) / float(n_pool)
                sel_share = sel_counts.get(jk, 1) / float(n_sel)
                sw = (pop_share / sel_share) if sel_share > 0 else 1.0
                k_row = {
                    "sample_id": f"{prefix}_{idx + 1:03d}",
                    "cascade_id": str(row.get("cascade_id") or ""),
                    "edge_id": str(row.get("edge_id") or ""),
                    "machine_stance_label": row["_stance_stratum"],
                    "s_uv": f"{float(row.get('s_uv', 0.0)):.6f}",
                    "h_uv": f"{float(row.get('h_uv', 0.0)):.6f}",
                    "sampling_weight": f"{sw:.6f}",
                }
                if has_behavior:
                    k_row["machine_behavior_label"] = row["_stratum"]
                    k_row["r_uv"] = f"{float(row.get('r_uv', 0.0)):.6f}"
                    k_row["c_uv"] = f"{float(row.get('c_uv', 0.5)):.6f}"
                writer.writerow(k_row)

    _write_blind_csv(out_path, selected, prefix="audit")

    pilot_selected: list[dict[str, Any]] = []
    if pilot_csv is not None and pilot_samples > 0:
        pilot_target = min(int(pilot_samples), max(0, len(pool) - len(selected)))
        if not has_behavior:
            pilot_base = pilot_target // len(strata_order)
            for lbl in strata_order:
                picked_p, nh, nc = _take_balanced_from_subbuckets(
                    hostile_sub[lbl],
                    clean_sub[lbl],
                    h_start=h_offsets[lbl],
                    c_start=c_offsets[lbl],
                    quota=pilot_base,
                )
                pilot_selected.extend(picked_p)
                h_offsets[lbl] = nh
                c_offsets[lbl] = nc
            while len(pilot_selected) < pilot_target:
                added_pilot = False
                for lbl in strata_order:
                    if len(pilot_selected) >= pilot_target:
                        break
                    want_h = h_offsets[lbl] < len(hostile_sub[lbl])
                    want_c = c_offsets[lbl] < len(clean_sub[lbl])
                    if want_h and (not want_c or (len(pilot_selected) % 2 == 0)):
                        pilot_selected.append(hostile_sub[lbl][h_offsets[lbl]])
                        h_offsets[lbl] += 1
                        added_pilot = True
                    elif want_c:
                        pilot_selected.append(clean_sub[lbl][c_offsets[lbl]])
                        c_offsets[lbl] += 1
                        added_pilot = True
                if not added_pilot:
                    break
        else:
            while len(pilot_selected) < pilot_target:
                added_pilot = False
                for lbl in strata_order:
                    if len(pilot_selected) >= pilot_target:
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
            if pilot_machine_key_csv is not None:
                _write_key_csv(Path(pilot_machine_key_csv), pilot_selected, prefix="pilot")

    if machine_key_csv is not None:
        _write_key_csv(Path(machine_key_csv), selected, prefix="audit")

    return out_path


def _encode_no_specials(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    if hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            return [int(x) for x in ids]
        except TypeError:
            pass
    out = tokenizer(text, add_special_tokens=False)
    ids = out["input_ids"] if isinstance(out, Mapping) else out.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def _inspect_pair_tokenizer(tokenizer: Any, encoder: Any | None = None) -> dict[str, Any]:
    """Detect special token IDs and RoBERTa (4-special) vs BERT (3-special) pair formatting."""
    cfg = getattr(encoder, "config", None) if encoder is not None else None
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(cfg, "pad_token_id", 0)
    if pad_id is None:
        pad_id = 0

    cls_id = getattr(tokenizer, "cls_token_id", None)
    if cls_id is None:
        cls_id = getattr(tokenizer, "bos_token_id", None)
    if cls_id is None:
        cls_id = 0

    sep_id = getattr(tokenizer, "sep_token_id", None)
    if sep_id is None:
        sep_id = getattr(tokenizer, "eos_token_id", None)
    if sep_id is None:
        sep_id = 2

    num_specials = 3
    use_type_ids = False
    try:
        probe = tokenizer("a", "b", add_special_tokens=True)
        probe_ids = probe["input_ids"] if isinstance(probe, Mapping) else probe.input_ids
        if hasattr(probe_ids, "tolist"):
            probe_ids = probe_ids.tolist()
        if probe_ids and isinstance(probe_ids[0], list):
            probe_ids = probe_ids[0]
        a_ids = _encode_no_specials(tokenizer, "a")
        b_ids = _encode_no_specials(tokenizer, "b")
        diff = len(probe_ids) - len(a_ids) - len(b_ids)
        if diff >= 4:
            num_specials = 4
        elif diff == 3:
            num_specials = 3
        t_ids = probe.get("token_type_ids") if isinstance(probe, Mapping) else None
        type_vocab_size = getattr(cfg, "type_vocab_size", None)
        if t_ids is not None and type_vocab_size is not None and int(type_vocab_size) > 1:
            use_type_ids = True
    except Exception:
        mtype = str(getattr(cfg, "model_type", "")).lower()
        if "roberta" in mtype or "xlm" in mtype:
            num_specials = 4

    return {
        "pad_id": int(pad_id),
        "cls_id": int(cls_id),
        "sep_id": int(sep_id),
        "num_specials": int(num_specials),
        "use_type_ids": bool(use_type_ids),
    }


def encode_cross_encoder_batch(
    parent_texts: Sequence[str],
    reply_texts: Sequence[str],
    *,
    tokenizer: Any,
    root_texts: Sequence[str | None] | None = None,
    include_root_context: bool = False,
    lang: str = "vi",
    max_length: int = 256,
    encoder: Any | None = None,
) -> dict[str, Tensor]:
    """Tokenize `(context, reply)` pairs with guaranteed reply budget and segment masks."""
    n = len(parent_texts)
    if n != len(reply_texts):
        raise ValueError("parent_texts and reply_texts must have equal length")
    if root_texts is not None and len(root_texts) != n:
        raise ValueError("root_texts must match parent_texts length when provided")
    if n == 0:
        empty = torch.empty((0, 0), dtype=torch.long)
        return {
            "input_ids": empty,
            "attention_mask": empty,
            "context_mask": empty,
            "reply_mask": empty,
        }

    spec = _inspect_pair_tokenizer(tokenizer, encoder=encoder)
    pad_id = spec["pad_id"]
    cls_id = spec["cls_id"]
    sep_id = spec["sep_id"]
    num_specials = spec["num_specials"]
    use_type_ids = spec["use_type_ids"]

    eff_max = _effective_max_length(max_length, encoder=encoder, tokenizer=tokenizer)
    content_budget = max(4, eff_max - num_specials)

    seqs: list[list[int]] = []
    ctx_masks: list[list[int]] = []
    rep_masks: list[list[int]] = []
    type_seqs: list[list[int]] = []

    for i in range(n):
        p_norm = normalize_social_text(parent_texts[i], lang=lang)
        v_norm = normalize_social_text(reply_texts[i], lang=lang)
        r_raw = root_texts[i] if root_texts is not None else None
        r_norm = normalize_social_text(r_raw, lang=lang) if r_raw else ""

        parent_ids = _encode_no_specials(tokenizer, p_norm)
        reply_ids = _encode_no_specials(tokenizer, v_norm)

        min_reply_budget = min(len(reply_ids), max(1, content_budget // 2))
        ctx_budget = max(0, content_budget - min_reply_budget)

        if include_root_context and r_norm and r_norm != p_norm and ctx_budget >= 3:
            root_ids = _encode_no_specials(tokenizer, r_norm)
            root_cap = min(len(root_ids), max(1, ctx_budget // 3))
            parent_cap = min(len(parent_ids), max(0, ctx_budget - 1 - root_cap))
            root_budget = min(len(root_ids), max(0, ctx_budget - 1 - parent_cap))
            root_keep = _truncate_head_tail(root_ids, root_budget, head_ratio=0.6)
            parent_keep = parent_ids[:parent_cap]
            if root_keep and parent_keep:
                ctx_ids = root_keep + [sep_id] + parent_keep
            elif root_keep:
                ctx_ids = root_keep
            else:
                ctx_ids = parent_keep
        else:
            # Apply 60/40 Head+Tail truncation when parent IS the root post (depth 1) too
            if include_root_context and r_norm and r_norm == p_norm:
                ctx_ids = _truncate_head_tail(parent_ids, ctx_budget, head_ratio=0.6)
            else:
                ctx_ids = parent_ids[:ctx_budget]

        reply_budget = max(1, content_budget - len(ctx_ids))
        reply_keep = reply_ids[:reply_budget]

        mid_seps = [sep_id, sep_id] if num_specials >= 4 else [sep_id]
        seq = [cls_id] + ctx_ids + mid_seps + reply_keep + [sep_id]
        ctx_mask = [0] + [1] * len(ctx_ids) + [0] * len(mid_seps) + [0] * len(reply_keep) + [0]
        rep_mask = [0] + [0] * len(ctx_ids) + [0] * len(mid_seps) + [1] * len(reply_keep) + [0]

        seqs.append(seq)
        ctx_masks.append(ctx_mask)
        rep_masks.append(rep_mask)
        if use_type_ids:
            seg0_len = 1 + len(ctx_ids) + len(mid_seps)
            seg1_len = len(reply_keep) + 1
            type_seqs.append([0] * seg0_len + [1] * seg1_len)

    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, max_len), dtype=torch.long)
    context_mask = torch.zeros((n, max_len), dtype=torch.long)
    reply_mask = torch.zeros((n, max_len), dtype=torch.long)
    token_type_ids = torch.zeros((n, max_len), dtype=torch.long) if use_type_ids else None

    for i, (seq, cm, rm) in enumerate(zip(seqs, ctx_masks, rep_masks)):
        cur_len = len(seq)
        input_ids[i, :cur_len] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :cur_len] = 1
        context_mask[i, :cur_len] = torch.tensor(cm, dtype=torch.long)
        reply_mask[i, :cur_len] = torch.tensor(rm, dtype=torch.long)
        if token_type_ids is not None:
            token_type_ids[i, :cur_len] = torch.tensor(type_seqs[i], dtype=torch.long)

    out: dict[str, Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "context_mask": context_mask,
        "reply_mask": reply_mask,
    }
    if token_type_ids is not None:
        out["token_type_ids"] = token_type_ids
    return out


class DualHeadCrossEncoder(nn.Module):
    """Segment-isolated multi-head Cross-Encoder predicting 2D `[s_uv, h_uv]` (or optional 4D)."""

    def __init__(
        self,
        *,
        lang: str = "vi",
        model_name: str | None = None,
        encoder: Any | None = None,
        edge_dim: int = 2,
        dropout: float = 0.1,
        include_root_context: bool = False,
    ) -> None:
        super().__init__()
        if lang not in {"vi", "en"}:
            raise ValueError(f"Unsupported lang {lang!r}")
        if edge_dim not in {2, 4}:
            raise ValueError(f"edge_dim must be 2 or 4, got {edge_dim}")
        self.lang = lang
        self.edge_dim = int(edge_dim)
        self.model_name = model_name or (VISOBERT_MODEL_NAME if lang == "vi" else TWITTER_ROBERTA_MODEL_NAME)
        self.dropout_p = float(dropout)
        self.include_root_context = bool(include_root_context)
        self._amp_enabled = False
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(self.model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        rel_proj_dim = min(256, hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.stance_proj = nn.Sequential(
            nn.Linear(5 * hidden_size, rel_proj_dim),
            nn.LayerNorm(rel_proj_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )
        self.stance_head = nn.Linear(rel_proj_dim, 3)  # [support, neutral, oppose]
        self.hostility_norm = nn.LayerNorm(hidden_size)
        self.hostility_head = nn.Linear(hidden_size, 1)
        if self.edge_dim == 4:
            self.behavior_head = nn.Linear(rel_proj_dim, len(DISCOURSE_BEHAVIOR_LABELS))
            self.irrational_head = nn.Linear(rel_proj_dim, 1)
            self.constructive_head = nn.Linear(rel_proj_dim, 1)
        else:
            self.behavior_head = None
            self.irrational_head = None
            self.constructive_head = None

    @staticmethod
    def _masked_pool_or_fallback(
        hidden: Tensor,
        mask: Tensor | None,
        fallback: Tensor,
    ) -> Tensor:
        if mask is None:
            return fallback
        m = (mask > 0).unsqueeze(-1).to(dtype=torch.float32)
        m_sum = m.sum(dim=1)
        pooled = (hidden.to(dtype=torch.float32) * m).sum(dim=1) / m_sum.clamp(min=1.0)
        return torch.where(
            m_sum > 0.0,
            pooled,
            fallback.to(dtype=torch.float32),
        ).to(dtype=hidden.dtype)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
        context_mask: Tensor | None = None,
        reply_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        kwargs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            try:
                sig_params = inspect.signature(self.encoder.forward).parameters
                accepts_type_ids = "token_type_ids" in sig_params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
                )
            except (TypeError, ValueError):
                accepts_type_ids = "token_type_ids" in self.encoder.forward.__code__.co_varnames
            if accepts_type_ids:
                kwargs["token_type_ids"] = token_type_ids
        amp_on = bool(getattr(self, "_amp_enabled", False) and input_ids.is_cuda)
        with _cuda_autocast(enabled=amp_on):
            outputs = self.encoder(**kwargs)
            hidden = outputs.last_hidden_state
            h_cls = hidden[:, 0, :]

            h_ctx = self._masked_pool_or_fallback(hidden, context_mask, h_cls)
            h_reply = self._masked_pool_or_fallback(hidden, reply_mask, h_cls)

            rel_in = torch.cat(
                [h_cls, h_ctx, h_reply, h_ctx * h_reply, (h_ctx - h_reply).abs()],
                dim=-1,
            )
            stance_repr = self.stance_proj(self.dropout(rel_in))
            stance_logits = self.stance_head(stance_repr)

            # Hostility reads strictly reply-only span representation to eliminate parent-toxicity bleed
            host_repr = self.dropout(self.hostility_norm(h_reply))
            hostility_logit = self.hostility_head(host_repr).squeeze(-1)

            if self.edge_dim == 4 and self.behavior_head is not None:
                behavior_logits = self.behavior_head(stance_repr)
                irrational_logit = self.irrational_head(stance_repr).squeeze(-1)
                constructive_logit = self.constructive_head(stance_repr).squeeze(-1)
            else:
                behavior_logits = None
                irrational_logit = None
                constructive_logit = None

        stance_probs = torch.softmax(stance_logits.float(), dim=-1)
        s_uv = (stance_probs[:, 0] - stance_probs[:, 2]).clamp(-1.0, 1.0)
        h_uv = torch.sigmoid(hostility_logit.float()).clamp(0.0, 1.0)

        if self.edge_dim == 4 and behavior_logits is not None and irrational_logit is not None and constructive_logit is not None:
            behavior_probs = torch.softmax(behavior_logits.float(), dim=-1)
            r_uv = torch.sigmoid(irrational_logit.float()).clamp(0.0, 1.0)
            c_uv = torch.sigmoid(constructive_logit.float()).clamp(0.0, 1.0)
            e_uv = torch.stack([s_uv, h_uv, r_uv, c_uv], dim=-1)
            return {
                "stance_logits": stance_logits,
                "stance_probs": stance_probs,
                "hostility_logit": hostility_logit,
                "behavior_logits": behavior_logits,
                "behavior_probs": behavior_probs,
                "irrational_logit": irrational_logit,
                "constructive_logit": constructive_logit,
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
            "include_root_context": bool(getattr(self, "include_root_context", False)),
            "has_pooler": getattr(self.encoder, "pooler", None) is not None,
            "pooling_mode": "segment_isolated",
            "arch_version": 2,
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
        if not ckpt_path.is_dir():
            raise FileNotFoundError(f"Cross-Encoder checkpoint directory not found: {ckpt_path}")
        required_files = ("config.json", "dual_head_meta.json", "dual_head_weights.pt")
        missing = [
            fname
            for fname in required_files
            if not (ckpt_path / fname).is_file() or (ckpt_path / fname).stat().st_size == 0
        ]
        if missing:
            raise FileNotFoundError(
                f"Incomplete Cross-Encoder checkpoint in '{ckpt_path}'; missing or empty: {', '.join(missing)}"
            )
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
            include_root_context=bool(meta.get("include_root_context", False)),
        )
        state = torch.load(ckpt_path / "dual_head_weights.pt", map_location=map_location, weights_only=True)
        instance.load_state_dict(state)

        tokenizer: Any | None = None
        if any(
            (ckpt_path / fname).exists()
            for fname in ("tokenizer_config.json", "tokenizer.json", "vocab.json", "sentencepiece.bpe.model")
        ):
            tokenizer = load_tokenizer_safe(ckpt_path)
        instance.tokenizer = tokenizer
        return instance, tokenizer

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        return_tokenizer: bool = False,
    ) -> DualHeadCrossEncoder | tuple[DualHeadCrossEncoder, Any | None]:
        """Load a saved DualHeadCrossEncoder checkpoint (attaching `.tokenizer`)."""
        instance, tokenizer = cls.load_checkpoint(Path(checkpoint_dir), map_location=map_location)
        instance.tokenizer = tokenizer
        if return_tokenizer:
            return instance, tokenizer
        return instance


class _PairDistillDataset(Dataset):
    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        return self.records[idx]


class _LengthBucketBatchSampler:
    """Stochastic mega-batch length bucketing to minimize padding waste without bias."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        batch_size: int,
        seed: int = 42,
    ) -> None:
        self.lengths = [
            len(str(r.get("parent_text") or ""))
            + len(str(r.get("reply_text") or ""))
            + len(str(r.get("root_text") or ""))
            for r in records
        ]
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Any:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        mega_size = self.batch_size * 50
        batches: list[list[int]] = []
        for start in range(0, len(indices), mega_size):
            chunk = indices[start : start + mega_size]
            chunk.sort(key=lambda i: self.lengths[i])
            for b_start in range(0, len(chunk), self.batch_size):
                batches.append(chunk[b_start : b_start + self.batch_size])
        rng.shuffle(batches)
        yield from batches


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
    distill_temperature: float = 2.0,
    augment_root_pairs: bool = True,
    seed: int = 42,
    sanity_check_shuffle: bool = False,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Distill unified `[s_uv, h_uv]` (or optional 4D) segment-isolated Cross-Encoder."""
    if not train_records or not val_records:
        raise ValueError("Both train_records and val_records must be non-empty")

    seed_everything(seed)
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = _should_use_fp16_amp(target_device)

    train_data = [dict(r) for r in train_records]
    val_data = [dict(r) for r in val_records]

    effective_edge_dim = int(model.edge_dim) if model is not None else int(edge_dim)
    train_root_aug_count = 0

    if augment_root_pairs and not sanity_check_shuffle:
        # Augment Train split with direct (root, deep_reply) pairs matching inference-time root stance queries.
        # Set `_hostility_loss_weight = 0.0` on augmented duplicates so depth >= 2 replies are not double-weighted in h_loss.
        augmented_rows: list[dict[str, Any]] = []
        for r in train_data:
            r.setdefault("_hostility_loss_weight", 1.0)
            if r.get("root_s_uv") is None:
                continue
            root_txt = _normalize_nfc(r.get("root_text") or "")
            parent_txt = _normalize_nfc(r.get("parent_text") or "")
            root_id = str(r.get("root_id") or "")
            parent_id = str(r.get("parent_id") or "")
            is_deep = bool(root_txt) and (
                (root_id and parent_id and root_id != parent_id) or (root_txt != parent_txt)
            )
            if not is_deep:
                continue
            root_s = max(-1.0, min(1.0, float(r["root_s_uv"])))
            aug_row = dict(r)
            aug_row["parent_text"] = r["root_text"]
            aug_row["root_text"] = r["root_text"]
            aug_row["parent_id"] = root_id or parent_id
            aug_row["s_uv"] = root_s
            aug_row["stance_probs"] = r.get("root_stance_probs") or _build_default_stance_probs(root_s)
            aug_row["h_uv"] = float(r["h_uv"])
            aug_row["_hostility_loss_weight"] = 0.0
            augmented_rows.append(aug_row)
        train_root_aug_count = len(augmented_rows)
        if augmented_rows:
            train_data.extend(augmented_rows)

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
        else DualHeadCrossEncoder(
            lang=lang,
            model_name=model_name,
            edge_dim=effective_edge_dim,
            include_root_context=include_root_context,
        )
    )
    student.include_root_context = bool(include_root_context)
    student._amp_enabled = bool(use_amp)
    student = student.to(target_device)
    tok = tokenizer if tokenizer is not None else load_tokenizer_safe(student.model_name)

    active_student: nn.Module = student
    if _can_use_dataparallel(target_device, use_multi_gpu):
        active_student = nn.DataParallel(student)

    def _collate(
        batch: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        u_texts = [str(item["parent_text"]) for item in batch]
        v_texts = [str(item["reply_text"]) for item in batch]
        r_texts = [item.get("root_text") for item in batch]
        enc = encode_cross_encoder_batch(
            u_texts,
            v_texts,
            tokenizer=tok,
            root_texts=r_texts,
            include_root_context=include_root_context,
            lang=lang,
            max_length=max_length,
            encoder=student.encoder,
        )
        s_tgt = torch.tensor([float(item["s_uv"]) for item in batch], dtype=torch.float32)
        h_tgt = torch.tensor([float(item["h_uv"]) for item in batch], dtype=torch.float32)
        h_weight = torch.tensor(
            [float(item.get("_hostility_loss_weight", 1.0)) for item in batch], dtype=torch.float32
        )
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
        return enc, s_tgt, h_tgt, h_weight, r_tgt, c_tgt, p_tgt, bp_tgt

    use_pin_mem = target_device.type == "cuda"
    train_sampler = _LengthBucketBatchSampler(train_data, batch_size=batch_size, seed=seed)
    train_loader = DataLoader(
        _PairDistillDataset(train_data),
        batch_sampler=train_sampler,
        collate_fn=_collate,
        pin_memory=use_pin_mem,
    )
    val_sorted = sorted(
        val_data,
        key=lambda r: (
            len(str(r.get("parent_text") or ""))
            + len(str(r.get("reply_text") or ""))
            + len(str(r.get("root_text") or ""))
        ),
    )
    val_loader = DataLoader(
        _PairDistillDataset(val_sorted),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        pin_memory=use_pin_mem,
    )

    head_lr = lr * 7.5 if lr <= 3e-5 else lr
    no_decay_tags = ("bias", "layernorm", "layer_norm")
    enc_decay, enc_no_decay, head_decay, head_no_decay = [], [], [], []
    for pname, param in student.named_parameters():
        if not param.requires_grad:
            continue
        low_name = pname.lower()
        is_no_decay = low_name.endswith("bias") or any(t in low_name for t in no_decay_tags)
        is_enc = pname.startswith("encoder.")
        if is_enc:
            (enc_no_decay if is_no_decay else enc_decay).append(param)
        else:
            (head_no_decay if is_no_decay else head_decay).append(param)

    opt_groups = []
    if enc_decay:
        opt_groups.append({"params": enc_decay, "lr": lr, "weight_decay": 0.01})
    if enc_no_decay:
        opt_groups.append({"params": enc_no_decay, "lr": lr, "weight_decay": 0.0})
    if head_decay:
        opt_groups.append({"params": head_decay, "lr": head_lr, "weight_decay": 0.01})
    if head_no_decay:
        opt_groups.append({"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(opt_groups or student.parameters(), lr=lr)

    scaler = _make_grad_scaler(enabled=use_amp)
    total_steps = max(1, max(1, epochs) * max(1, len(train_loader)))
    warmup_steps = max(1, int(round(0.10 * total_steps)))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return 0.1 + 0.9 * (float(step + 1) / float(warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    train_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in student.state_dict().items()})
    best_val_metrics: dict[str, float] = {}
    last_val_metrics: dict[str, float] = {}
    temp = max(1e-3, float(distill_temperature))

    def _batch_loss(
        out: Mapping[str, Tensor],
        s_tgt_d: Tensor,
        h_tgt_d: Tensor,
        h_w_d: Tensor,
        r_tgt_d: Tensor,
        c_tgt_d: Tensor,
        p_tgt_d: Tensor,
        bp_tgt_d: Tensor,
    ) -> Tensor:
        p_tgt_temp = torch.softmax(torch.log(p_tgt_d.clamp(min=1e-6)) / temp, dim=-1)
        kl_loss = (temp * temp) * F.kl_div(
            torch.log_softmax(out["stance_logits"].float() / temp, dim=-1),
            p_tgt_temp,
            reduction="batchmean",
        )
        s_loss = F.smooth_l1_loss(out["s_uv"], s_tgt_d, beta=0.1)
        h_bce = F.binary_cross_entropy_with_logits(
            out["hostility_logit"].float(),
            h_tgt_d.clamp(0.0, 1.0),
            reduction="none",
        )
        h_w_sum = h_w_d.sum()
        if float(h_w_sum.item()) > 0.0:
            h_loss = (h_bce * h_w_d).sum() / h_w_sum.clamp(min=1e-6)
        else:
            h_loss = h_bce.new_zeros(())
        loss = kl_loss + 0.5 * s_loss + h_loss
        if student.edge_dim == 4 and "behavior_logits" in out:
            bp_tgt_temp = torch.softmax(torch.log(bp_tgt_d.clamp(min=1e-6)) / temp, dim=-1)
            kl_beh = (temp * temp) * F.kl_div(
                torch.log_softmax(out["behavior_logits"].float() / temp, dim=-1),
                bp_tgt_temp,
                reduction="batchmean",
            )
            r_loss = F.binary_cross_entropy_with_logits(
                out["irrational_logit"].float(),
                r_tgt_d.clamp(0.0, 1.0),
            )
            c_loss = F.binary_cross_entropy_with_logits(
                out["constructive_logit"].float(),
                c_tgt_d.clamp(0.0, 1.0),
            )
            loss = loss + 0.5 * kl_beh + r_loss + c_loss
        return loss

    try:
        for _epoch in range(max(1, epochs)):
            active_student.train()
            running_loss = 0.0
            step_count = 0
            for enc, s_tgt, h_tgt, h_weight, r_tgt, c_tgt, p_tgt, bp_tgt in train_loader:
                enc_dev = {k: v.to(target_device) for k, v in enc.items()}
                s_tgt_d = s_tgt.to(target_device)
                h_tgt_d = h_tgt.to(target_device)
                h_w_d = h_weight.to(target_device)
                r_tgt_d = r_tgt.to(target_device)
                c_tgt_d = c_tgt.to(target_device)
                p_tgt_d = p_tgt.to(target_device)
                bp_tgt_d = bp_tgt.to(target_device)

                optimizer.zero_grad(set_to_none=True)
                with _cuda_autocast(enabled=use_amp):
                    out = active_student(**enc_dev)
                    loss = _batch_loss(out, s_tgt_d, h_tgt_d, h_w_d, r_tgt_d, c_tgt_d, p_tgt_d, bp_tgt_d)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scale_before = scaler.get_scale() if use_amp else 1.0
                scaler.step(optimizer)
                scaler.update()
                if not use_amp or scaler.get_scale() >= scale_before:
                    scheduler.step()

                running_loss += float(loss.detach().cpu().item())
                step_count += 1

            train_losses.append(running_loss / max(1, step_count))

            active_student.eval()
            val_loss_sum = 0.0
            val_steps = 0
            s_preds, s_trues, h_preds, h_trues = [], [], [], []
            r_preds, r_trues, c_preds, c_trues = [], [], [], []
            with torch.no_grad():
                for enc, s_tgt, h_tgt, h_weight, r_tgt, c_tgt, p_tgt, bp_tgt in val_loader:
                    enc_dev = {k: v.to(target_device) for k, v in enc.items()}
                    s_tgt_d = s_tgt.to(target_device)
                    h_tgt_d = h_tgt.to(target_device)
                    h_w_d = h_weight.to(target_device)
                    r_tgt_d = r_tgt.to(target_device)
                    c_tgt_d = c_tgt.to(target_device)
                    p_tgt_d = p_tgt.to(target_device)
                    bp_tgt_d = bp_tgt.to(target_device)
                    with _cuda_autocast(enabled=use_amp):
                        out = active_student(**enc_dev)
                        vloss = _batch_loss(out, s_tgt_d, h_tgt_d, h_w_d, r_tgt_d, c_tgt_d, p_tgt_d, bp_tgt_d)

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
                best_epoch = _epoch
                best_val_metrics = dict(last_val_metrics)
                best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            print(
                f"[Distill:{lang}] Epoch {_epoch + 1}/{max(1, epochs)} | "
                f"train_loss={train_losses[-1]:.4f} | val_loss={val_loss:.4f} | "
                f"val_pearson_s={last_val_metrics['val_pearson_s']:.4f} | "
                f"val_pearson_h={last_val_metrics['val_pearson_h']:.4f}",
                flush=True,
            )
    finally:
        student._amp_enabled = False

    if not best_val_metrics:
        best_val_metrics = dict(last_val_metrics)

    student.load_state_dict(best_state)
    ckpt_dir = student.save_checkpoint(Path(output_dir), tokenizer=tok)

    report = {
        "lang": lang,
        "model_name": student.model_name,
        "edge_dim": student.edge_dim,
        "sanity_check_shuffle": bool(sanity_check_shuffle),
        "distill_temperature": temp,
        "train_root_aug_count": train_root_aug_count,
        "hostility_loss_deweighted_aug": True,
        "train_losses": train_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_loss": last_val_metrics.get("val_loss", best_val_loss),
        **best_val_metrics,
        "checkpoint_dir": str(ckpt_dir),
    }
    (Path(output_dir) / "distill_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def extract_parent_reply_pairs_from_parquet(
    interim_dir: Path,
    *,
    min_total_replies: int = 0,
) -> list[dict[str, Any]]:
    """Read validated `nodes.parquet` and `edges.parquet` into `(u, v)` text pair records."""
    nodes_tbl, edges_tbl = load_cascade_parquet(interim_dir, validate=True)
    node_rows = nodes_tbl.to_pylist()
    allowed_cascades: set[str] | None = None
    if int(min_total_replies) > 0:
        reply_counts: dict[str, int] = {}
        for row in node_rows:
            cid = str(row["cascade_id"])
            if not bool(row.get("is_root")):
                reply_counts[cid] = reply_counts.get(cid, 0) + 1
        allowed_cascades = {
            cid for cid, cnt in reply_counts.items() if cnt >= int(min_total_replies)
        }

    node_text_by_id = {str(row["node_id"]): str(row["text"]) for row in node_rows}
    node_topic_by_id = {str(row["node_id"]): str(row["topic_id"]) for row in node_rows}
    parent_of_node = {
        str(row["node_id"]): (str(row["parent_id"]) if row.get("parent_id") is not None else "")
        for row in node_rows
    }
    cascade_root_id: dict[str, str] = {}
    for row in node_rows:
        if bool(row.get("is_root")):
            cascade_root_id[str(row["cascade_id"])] = str(row["node_id"])

    pairs: list[dict[str, Any]] = []
    for edge in edges_tbl.to_pylist():
        cid = str(edge["cascade_id"])
        if allowed_cascades is not None and cid not in allowed_cascades:
            continue
        src = str(edge["source_id"])
        tgt = str(edge["target_id"])
        root_id = cascade_root_id.get(cid, src)
        gp_id = parent_of_node.get(src, "")
        gp_text = node_text_by_id.get(gp_id, "") if gp_id and gp_id != src else ""
        pairs.append(
            {
                "edge_id": str(edge["edge_id"]),
                "cascade_id": cid,
                "topic_id": node_topic_by_id[src],
                "root_id": root_id,
                "parent_id": src,
                "target_id": tgt,
                "root_text": node_text_by_id.get(root_id, node_text_by_id[src]),
                "grandparent_text": gp_text,
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
    canonical_topic_splits: Mapping[str, set[str] | Sequence[str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Partition `(u, v)` edge pairs strictly by `topic_id` so test topics never leak into distillation."""
    if not pairs:
        raise ValueError("Cannot split empty pair sequence")
    normalized = [
        {**dict(p), "topic_id": str(p.get("topic_id") or p.get("cascade_id") or "default")}
        for p in pairs
    ]
    if canonical_topic_splits is not None:
        tr_topics_cfg = {str(t) for t in canonical_topic_splits.get("train", ())}
        va_topics_cfg = {str(t) for t in canonical_topic_splits.get("val", ())}
        te_topics_cfg = {str(t) for t in canonical_topic_splits.get("test", ())}
        train_pairs: list[dict[str, Any]] = []
        val_pairs: list[dict[str, Any]] = []
        test_pairs: list[dict[str, Any]] = []
        for p in normalized:
            top = str(p["topic_id"])
            if top in tr_topics_cfg:
                train_pairs.append(p)
            elif top in va_topics_cfg:
                val_pairs.append(p)
            elif top in te_topics_cfg:
                test_pairs.append(p)
    else:
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


def _append_score_deltas(
    ckpt_jsonl: Path,
    pairs: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    fields: Sequence[str],
) -> None:
    """Append compact score-only JSONL delta records (`O(chunk_size)` I/O, no text duplication)."""
    ckpt_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with ckpt_jsonl.open("a", encoding="utf-8") as fh:
        for i in indices:
            rec: dict[str, Any] = {"edge_id": str(pairs[i].get("edge_id") or "")}
            for f in fields:
                if f in pairs[i]:
                    rec[f] = pairs[i][f]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()


def _has_complete_dual_ce_checkpoint(
    ckpt_dir: str | Path,
    *,
    require_distill_report: bool = False,
) -> bool:
    ce_path = Path(ckpt_dir)
    weights_pt = ce_path / "dual_head_weights.pt"
    try:
        has_weights_pt = weights_pt.is_file() and weights_pt.stat().st_size > 0
    except OSError:
        has_weights_pt = False
    ok = (
        _has_hf_weight_files(ce_path)
        and has_weights_pt
        and (ce_path / "dual_head_meta.json").is_file()
        and (ce_path / "config.json").is_file()
    )
    if require_distill_report:
        ok = ok and (ce_path / "distill_report.json").is_file()
    return ok


def _is_depth1_pair(pair: Mapping[str, Any]) -> bool:
    if pair.get("depth") is not None:
        return int(pair["depth"]) == 1
    root_txt = pair.get("root_text")
    root_id = str(pair.get("root_id") or "")
    parent_id = str(pair.get("parent_id") or "")
    is_deep = bool(root_txt) and (
        (bool(root_id) and bool(parent_id) and root_id != parent_id)
        or str(root_txt) != str(pair.get("parent_text") or "")
    )
    return not is_deep


def _sync_rerouted_depth1_root_scores(pairs: Sequence[dict[str, Any]]) -> None:
    for p in pairs:
        if p.get("target_attributed") == "third_party" and _is_depth1_pair(p):
            p["root_s_uv"] = float(p["s_uv"])
            if "stance_probs" in p:
                p["root_stance_probs"] = list(p["stance_probs"])


def _score_pairs_with_teachers(
    pairs: list[dict[str, Any]],
    *,
    lang: str,
    vihsd_ckpt: Path | None = None,
    batch_size: int = 16,
    allow_untrained_fallback: bool = False,
    cache_path: Path | None = None,
    checkpoint_every: int = 250,
    teacher_backend: str = "dual_nli",
    use_hostility_router: bool = False,
    hostility_gate_threshold: float | None = None,
    telemetry: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run dual open-source teachers (Hostility + mDeBERTa Stance Verbalizer) with O(N) JSONL delta checkpointing."""
    if not pairs:
        return pairs

    if teacher_backend == "legacy_nli":
        warnings.warn(
            "teacher_backend='legacy_nli' is deprecated and retired; using canonical 'dual_nli'.",
            DeprecationWarning,
            stacklevel=2,
        )
        teacher_backend = "dual_nli"
    elif teacher_backend != "dual_nli":
        raise ValueError(f"Unsupported teacher_backend {teacher_backend!r}; expected 'dual_nli'")

    chunk_size = max(1, int(checkpoint_every))
    cache_file = Path(cache_path) if cache_path is not None else None
    ckpt_jsonl = (
        cache_file.with_name(f"{cache_file.name}.checkpoint.jsonl")
        if cache_file is not None
        else None
    )
    cached_by_eid: dict[str, dict[str, Any]] = {}
    if cache_file is not None and cache_file.exists():
        try:
            loaded_cache = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(loaded_cache, list):
                for item in loaded_cache:
                    eid = str(item.get("edge_id") or "")
                    if eid:
                        cached_by_eid[eid] = dict(item)
        except (OSError, json.JSONDecodeError):
            cached_by_eid = {}

    if ckpt_jsonl is not None and ckpt_jsonl.exists():
        try:
            with ckpt_jsonl.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    eid = str(rec.get("edge_id") or "")
                    if eid:
                        cached_by_eid.setdefault(eid, {}).update(rec)
        except OSError:
            pass

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
        "root_stance_probs",
    )
    for p in pairs:
        eid = str(p.get("edge_id") or "")
        cached = cached_by_eid.get(eid)
        if cached is not None:
            for field in cache_fields:
                if field in cached:
                    p[field] = cached[field]

    did_compute = False
    resolved_gate_threshold: float
    if hostility_gate_threshold is not None:
        resolved_gate_threshold = float(hostility_gate_threshold)
    else:
        resolved_gate_threshold = 0.5
        if lang == "vi" and vihsd_ckpt is not None:
            metrics_file = Path(vihsd_ckpt) / "vihsd_metrics.json"
            if metrics_file.is_file():
                try:
                    m_data = json.loads(metrics_file.read_text(encoding="utf-8"))
                    if "calibrated_hostility_threshold" in m_data:
                        resolved_gate_threshold = float(m_data["calibrated_hostility_threshold"])
                except Exception:
                    pass

    # Stage 1: Hostility intensity h_uv
    need_host_idx = [i for i, p in enumerate(pairs) if "h_uv" not in p]
    if need_host_idx:
        did_compute = True
        host_scorer = HostilityTeacherScorer(
            lang=lang,
            model_name_or_path=vihsd_ckpt if lang == "vi" else None,
            allow_untrained_fallback=allow_untrained_fallback,
        )
        if hostility_gate_threshold is None:
            resolved_gate_threshold = float(
                getattr(host_scorer, "calibrated_threshold", resolved_gate_threshold)
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
            if ckpt_jsonl is not None:
                _append_score_deltas(ckpt_jsonl, pairs, idx_slice, ("h_uv",))
        del host_scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stage 2: NLI Stance Verbalizer score s_uv + direct (root -> reply) root_s_uv for depth >= 2
    need_nli_idx = [
        i for i, p in enumerate(pairs) if "s_uv" not in p or "root_s_uv" not in p
    ]
    router_stats: dict[str, Any] | None = None
    if need_nli_idx:
        did_compute = True
        nli_scorer = NLIStanceTeacherScorer(lang=lang, use_stance_verbalizer=True)
        chunk_starts = range(0, len(need_nli_idx), chunk_size)
        for start in tqdm(chunk_starts, desc=f"NLI stance verbalizer ({lang})"):
            idx_slice = need_nli_idx[start : start + chunk_size]
            need_edge_pos = [i for i in idx_slice if "s_uv" not in pairs[i]]
            if need_edge_pos:
                nli_out = nli_scorer.score_pairs(
                    [pairs[i]["parent_text"] for i in need_edge_pos],
                    [pairs[i]["reply_text"] for i in need_edge_pos],
                    batch_size=batch_size,
                    lang=lang,
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
                    if "stance_probs" in pairs[i]:
                        pairs[i]["root_stance_probs"] = list(pairs[i]["stance_probs"])

            if deep_indices:
                root_out = nli_scorer.score_pairs(
                    [str(pairs[i]["root_text"]) for i in deep_indices],
                    [str(pairs[i]["reply_text"]) for i in deep_indices],
                    batch_size=batch_size,
                    lang=lang,
                )
                for pos, i in enumerate(deep_indices):
                    pairs[i]["root_s_uv"] = float(root_out["s_uv"][pos])
                    pairs[i]["root_stance_probs"] = root_out["stance_probs"][pos].tolist()

            if ckpt_jsonl is not None:
                _append_score_deltas(
                    ckpt_jsonl,
                    pairs,
                    idx_slice,
                    ("s_uv", "stance_probs", "pred_label", "root_s_uv", "root_stance_probs"),
                )
        if use_hostility_router:
            router = HostilityGatedTargetRouter(
                hostility_gate_threshold=resolved_gate_threshold,
                lang=lang,
                nli_scorer=nli_scorer,
            )
            pairs, router_stats = router.route_pairs(pairs)
            _sync_rerouted_depth1_root_scores(pairs)
            if router_stats.get("rerouted_count", 0) > 0:
                did_compute = True
        del nli_scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif use_hostility_router:
        router = HostilityGatedTargetRouter(
            hostility_gate_threshold=resolved_gate_threshold,
            lang=lang,
        )
        pairs, router_stats = router.route_pairs(pairs)
        _sync_rerouted_depth1_root_scores(pairs)
        if router_stats.get("rerouted_count", 0) > 0:
            did_compute = True

    if cache_file is not None and (did_compute or not cache_file.exists() or (ckpt_jsonl is not None and ckpt_jsonl.exists())):
        _atomic_write_json(cache_file, pairs)
        if ckpt_jsonl is not None and ckpt_jsonl.exists():
            try:
                ckpt_jsonl.unlink()
            except OSError:
                pass

    if telemetry is not None:
        telemetry["teacher_backend"] = teacher_backend
        telemetry["stance_model"] = MDEBERTA_NLI_MODEL_NAME
        telemetry["hostility_model"] = (
            str(vihsd_ckpt)
            if lang == "vi" and vihsd_ckpt is not None
            else (VISOBERT_MODEL_NAME if lang == "vi" else EN_OFFENSIVE_MODEL_NAME)
        )
        telemetry["use_stance_verbalizer"] = True
        telemetry["use_hostility_router"] = bool(use_hostility_router)
        if router_stats is not None:
            telemetry["hostility_router_stats"] = router_stats
        telemetry["num_pairs"] = len(pairs)
        telemetry["num_deep_pairs"] = int(
            sum(
                1
                for p in pairs
                if bool(p.get("root_text"))
                and (
                    (
                        str(p.get("root_id") or "")
                        and str(p.get("parent_id") or "")
                        and str(p.get("root_id")) != str(p.get("parent_id"))
                    )
                    or str(p.get("root_text")) != str(p.get("parent_text") or "")
                )
            )
        )

    return pairs


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Dual open-source NLP edge scoring & distillation.")
    parser.add_argument("--lang", choices=["vi", "en"], default="vi")
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim/vn"))
    parser.add_argument("--vihsd-dir", type=Path, default=Path("checkpoints/visobert_vihsd"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument("--pilot-csv", type=Path, default=None)
    parser.add_argument("--machine-key-csv", type=Path, default=None)
    parser.add_argument("--scored-cache-json", type=Path, default=None)
    parser.add_argument("--topic-splits-json", type=Path, default=None)
    parser.add_argument("--train-vihsd", action="store_true")
    parser.add_argument(
        "--teacher-backend",
        choices=["dual_nli", "legacy_nli"],
        default="dual_nli",
        help="Canonical NLI stance teacher ('dual_nli'; 'legacy_nli' is deprecated).",
    )
    parser.add_argument("--hostility-target-router", action="store_true")
    parser.add_argument("--hostility-gate-threshold", type=float, default=None)
    parser.add_argument("--edge-dim", type=int, choices=[2, 4], default=2)
    parser.add_argument("--student-backbone", type=str, default=None)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--export-audit", action="store_true")
    parser.add_argument(
        "--force-export-audit",
        action="store_true",
        help="Overwrite existing audit CSVs even when --skip-if-exists or existing audit files are present.",
    )
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument("--include-root-context", action="store_true")
    parser.add_argument("--sanity-check-shuffle", action="store_true")
    parser.add_argument("--allow-untrained-vihsd", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--min-total-replies",
        type=int,
        default=DEFAULT_FORECAST_MIN_REPLIES,
        help="Canonical minimum replies threshold aligned with EarlyCascadeDataset (default: 25).",
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-n-samples", type=int, default=120)
    parser.add_argument("--pilot-samples", type=int, default=20)
    args = parser.parse_args()

    out_dir = args.output_dir or Path(
        f"checkpoints/dual_ce_{args.lang}{'_sanity_shuffle' if args.sanity_check_shuffle else ''}"
    )
    audit_csv = args.audit_csv or Path(f"data/interim/human_audit_{args.lang}_120.csv")
    machine_key_csv = args.machine_key_csv or audit_csv.with_name(
        f"{audit_csv.stem}_machine_key.csv"
    )
    pilot_key_csv = (
        args.pilot_csv.with_name(f"{args.pilot_csv.stem}_machine_key.csv")
        if args.pilot_csv is not None
        else None
    )
    splits_path = args.topic_splits_json or (args.interim_dir / "topic_splits.json")

    if args.train_vihsd:
        if (
            args.skip_if_exists
            and (args.vihsd_dir / "vihsd_metrics.json").is_file()
            and _has_hf_weight_files(args.vihsd_dir)
        ):
            print(f"[SKIP] Fine-tuned ViHSD checkpoint already exists: {args.vihsd_dir}")
        else:
            finetune_visobert_vihsd(
                output_dir=args.vihsd_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

    if args.score_only or args.export_audit or args.distill:
        nodes_tbl, _ = load_cascade_parquet(args.interim_dir, validate=True)
        canonical_splits = compute_canonical_topic_splits(
            nodes_tbl,
            min_total_replies=args.min_total_replies,
            seed=args.split_seed,
        )
        save_topic_splits_json(
            canonical_splits,
            splits_path,
            meta={
                "lang": args.lang,
                "min_total_replies": int(args.min_total_replies),
                "split_seed": int(args.split_seed),
            },
        )

        pairs = extract_parent_reply_pairs_from_parquet(
            args.interim_dir,
            min_total_replies=args.min_total_replies,
        )
        if args.max_pairs is not None:
            pairs = pairs[: args.max_pairs]

        telemetry: dict[str, Any] = {}
        scored = _score_pairs_with_teachers(
            pairs,
            lang=args.lang,
            vihsd_ckpt=args.vihsd_dir,
            batch_size=args.batch_size,
            allow_untrained_fallback=args.allow_untrained_vihsd,
            cache_path=args.scored_cache_json,
            checkpoint_every=args.checkpoint_every,
            teacher_backend=args.teacher_backend,
            use_hostility_router=bool(args.hostility_target_router),
            hostility_gate_threshold=(
                float(args.hostility_gate_threshold)
                if args.hostility_gate_threshold is not None
                else None
            ),
            telemetry=telemetry,
        )
        if args.scored_cache_json is not None:
            if not args.scored_cache_json.exists():
                _atomic_write_json(args.scored_cache_json, scored)
            if telemetry:
                _atomic_write_json(
                    args.scored_cache_json.with_name("teacher_scoring_report.json"),
                    telemetry,
                )

        if args.export_audit:
            if not args.force_export_audit and audit_csv.exists():
                if args.skip_if_exists:
                    print(f"[SKIP] Blind human audit CSV already exists: {audit_csv}")
                else:
                    raise FileExistsError(
                        f"Refusing to overwrite existing audit CSV '{audit_csv}' without "
                        "--force-export-audit (or pass --skip-if-exists to keep existing file)."
                    )
            elif (
                not args.force_export_audit
                and args.pilot_csv is not None
                and args.pilot_csv.exists()
            ):
                if args.skip_if_exists:
                    print(f"[SKIP] Pilot human audit CSV already exists: {args.pilot_csv}")
                else:
                    raise FileExistsError(
                        f"Refusing to overwrite existing pilot CSV '{args.pilot_csv}' without "
                        "--force-export-audit (or pass --skip-if-exists to keep existing file)."
                    )
            else:
                test_topics = {str(t) for t in canonical_splits.get("test", ())}
                test_cids = {
                    str(r["cascade_id"])
                    for r in nodes_tbl.to_pylist()
                    if str(r.get("topic_id") or "") in test_topics
                }
                csv_path = export_stratified_human_audit_csv(
                    scored,
                    output_csv=audit_csv,
                    machine_key_csv=machine_key_csv,
                    pilot_csv=args.pilot_csv,
                    pilot_machine_key_csv=pilot_key_csv,
                    pilot_samples=args.pilot_samples,
                    n_samples=args.audit_n_samples,
                    test_cascade_ids=test_cids or None,
                    seed=args.split_seed,
                )
                print(f"Exported blind stratified human audit CSV: {csv_path} (key: {machine_key_csv})")

        if args.distill:
            if (
                args.skip_if_exists
                and _has_complete_dual_ce_checkpoint(out_dir, require_distill_report=True)
            ):
                print(f"[SKIP] Distilled Student checkpoint already exists: {out_dir}")
            else:
                topic_splits = split_pairs_by_inductive_topics(
                    scored,
                    split_seed=args.split_seed,
                    canonical_topic_splits=canonical_splits,
                )
                train_distill = topic_splits["train"]
                val_distill = topic_splits["val"]
                student_backbone = args.student_backbone or (
                    VISOBERT_MODEL_NAME if args.lang == "vi" else TWITTER_ROBERTA_MODEL_NAME
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
                report["canonical_splits_path"] = str(splits_path)
                if telemetry:
                    report["teacher_telemetry"] = telemetry
                    (out_dir / "distill_report.json").write_text(
                        json.dumps(report, indent=2),
                        encoding="utf-8",
                    )
                print("Distillation summary:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

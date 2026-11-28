"""Lightweight 5-class discourse category constants and rule-based fallback helpers.

Note on canonical pipeline:
  The production SBi-FAGAT pipeline uses the unified 2D edge representation
  `e_uv = [s_uv, h_uv]` (stance `s_uv in [-1, 1]` + hostility `h_uv in [0, 1]`).
  `DISCOURSE_BEHAVIOR_LABELS` and `_heuristic_behavior_fallback` are preserved
  exclusively for offline checkpoint-free demo fallback and backward-compatible
  5-class / 4D test coverage.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

from src.utils.text import normalize_nfc

DISCOURSE_BEHAVIOR_LABELS: tuple[str, ...] = (
    "support",
    "neutral",
    "constructive_oppose",
    "irrational_fallacy",
    "hostile_conflict",
)

BEHAVIOR_LABEL_TO_IDX: dict[str, int] = {
    lbl: idx for idx, lbl in enumerate(DISCOURSE_BEHAVIOR_LABELS)
}

BEHAVIOR_LABEL_VI: dict[str, str] = {
    "support": "Tán thành / Ủng hộ hợp lý",
    "neutral": "Trung lập / Hỏi thông tin",
    "constructive_oppose": "Phản biện văn minh có lý lẽ",
    "irrational_fallacy": "Vô lý / Ngụy biện / Bẻ lái",
    "hostile_conflict": "Chống đối gay gắt / Công kích",
}

_LABEL_SYNONYMS: dict[str, str] = {
    "support": "support",
    "agree": "support",
    "endorse": "support",
    "tan_thanh": "support",
    "neutral": "neutral",
    "query": "neutral",
    "comment": "neutral",
    "trung_lap": "neutral",
    "constructive_oppose": "constructive_oppose",
    "civil_oppose": "constructive_oppose",
    "oppose": "constructive_oppose",
    "phan_bien": "constructive_oppose",
    "irrational_fallacy": "irrational_fallacy",
    "irrational": "irrational_fallacy",
    "fallacy": "irrational_fallacy",
    "troll": "irrational_fallacy",
    "vo_ly": "irrational_fallacy",
    "hostile_conflict": "hostile_conflict",
    "hostile": "hostile_conflict",
    "attack": "hostile_conflict",
    "conflict": "hostile_conflict",
    "xung_dot": "hostile_conflict",
}

_HOSTILE_CUE_WORDS: tuple[str, ...] = (
    "ngu",
    "óc chó",
    "oc cho",
    "não tàn",
    "nao tan",
    "rác",
    "cút",
    "biến đi",
    "biến khỏi",
    "biến mẹ",
    "cút biến",
    "súc vật",
    "đồ điên",
    "thằng khùng",
    "con điên",
    "idiot",
    "moron",
    "trash",
    "shut up",
    "loser",
    "stupid",
)
_IRRATIONAL_CUE_WORDS: tuple[str, ...] = (
    "biết gì mà nói",
    "đỉnh quá cơ",
    "chuẩn bị đi tù",
    "tất cả tại",
    "thế ai cho tiền",
    "chắc mỗi nhà bạn",
    "vãi cả",
    "cạn lời",
    "ngáo",
    "whatabout",
    "wake up sheeple",
    "clown",
    "lmao yeah right",
)
_OPPOSE_CUE_WORDS: tuple[str, ...] = (
    "phản đối",
    "phản bác",
    "bác bỏ",
    "không đồng tình",
    "không tán thành",
    "không ủng hộ",
    "chưa đồng ý",
    "không đồng ý",
    "chưa hợp lý",
    "không hợp lý",
    "không chính xác",
    "chưa chính xác",
    "nhầm rồi",
    "sai ở chỗ",
    "tuy nhiên",
    "trái lại",
    "cần xem lại",
    "disagree",
    "however",
    "incorrect",
    "not true",
    "false",
)
_SUPPORT_CUE_WORDS: tuple[str, ...] = (
    "đồng ý",
    "chính xác",
    "ủng hộ",
    "hợp lý",
    "quá đúng",
    "+1",
    "agree",
    "exactly",
    "well said",
    "confirmed",
    "true",
)


def _compile_cue_pattern(cue: str) -> re.Pattern[str]:
    """Compile a Unicode-aware token-boundary regex for word or symbol cues."""
    norm_cue = normalize_nfc(cue).lower()
    left = (
        r"(?<![\w\u0300-\u036f])"
        if norm_cue and (norm_cue[0].isalnum() or norm_cue[0] == "_")
        else r"(?:^|(?<=[\s(\[\"'.,:;!?]))"
    )
    right = (
        r"(?![\w\u0300-\u036f])"
        if norm_cue and (norm_cue[-1].isalnum() or norm_cue[-1] == "_")
        else r"(?:$|(?=[\s)\]\}\"'.,:;!?]))"
    )
    return re.compile(f"{left}{re.escape(norm_cue)}{right}", flags=re.UNICODE)


_GAPPED_VI_OPPOSE_RE = re.compile(
    r"(?<![\w\u0300-\u036f])"
    r"(?:không|chẳng|chả|chưa|đâu\s+có)\s+"
    r"(?:[\w\u0300-\u036f]+\s+){0,2}"
    r"(?:đồng\s+ý|đồng\s+tình|tán\s+thành|ủng\s+hộ|hợp\s+lý|chính\s+xác|chuẩn(?!\s+bị)|đúng)"
    r"(?![\w\u0300-\u036f])",
    flags=re.UNICODE,
)
_GAPPED_EN_OPPOSE_RE = re.compile(
    r"(?<![\w\u0300-\u036f])"
    r"(?:don['’]?t|do\s+not|doesn['’]?t|does\s+not|didn['’]?t|did\s+not|"
    r"can['’]?t|cannot|can\s+not|won['’]?t|wouldn['’]?t|never|not)\s+"
    r"(?:[\w\u0300-\u036f]+\s+){0,2}"
    r"(?:agree|agreed|agreeing|correct|true|exactly|well\s+said|confirmed)"
    r"(?![\w\u0300-\u036f])",
    flags=re.UNICODE,
)
_CHUAN_SUPPORT_RE = re.compile(
    r"(?<![\w\u0300-\u036f])chuẩn(?!\s+bị)(?![\w\u0300-\u036f])",
    flags=re.UNICODE,
)
_HOSTILE_CUE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    _compile_cue_pattern(tok) for tok in _HOSTILE_CUE_WORDS
)
_IRRATIONAL_CUE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    _compile_cue_pattern(tok) for tok in _IRRATIONAL_CUE_WORDS
)
_OPPOSE_CUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    *(_compile_cue_pattern(tok) for tok in _OPPOSE_CUE_WORDS),
    _GAPPED_VI_OPPOSE_RE,
    _GAPPED_EN_OPPOSE_RE,
)
_SUPPORT_CUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    *(_compile_cue_pattern(tok) for tok in _SUPPORT_CUE_WORDS),
    _CHUAN_SUPPORT_RE,
)


def behavior_probs_to_stance_probs(probs_5: Sequence[float]) -> list[float]:
    """Collapse 5-class behavior probs `[sup, neu, con_opp, irr, hos]` to 3-class `[sup, neu, opp]`."""
    arr = np.asarray(probs_5, dtype=np.float64).reshape(-1)
    if arr.size != 5:
        return [0.2, 0.6, 0.2]
    arr = np.clip(arr, 0.0, None)
    tot = float(arr.sum())
    if tot <= 1e-12:
        return [0.2, 0.6, 0.2]
    arr = arr / tot
    p_sup = float(arr[0])
    p_neu = float(arr[1])
    p_opp = float(arr[2] + arr[3] + arr[4])
    return [p_sup, p_neu, p_opp]


def _default_probs_from_label(label: str) -> list[float]:
    base = [0.05, 0.05, 0.05, 0.05, 0.05]
    idx = BEHAVIOR_LABEL_TO_IDX.get(label, 1)
    base[idx] = 0.80
    tot = sum(base)
    return [v / tot for v in base]


_BENIGN_EXCLAMATION_WORDS: frozenset[str] = frozenset(
    {
        "ok",
        "oke",
        "okay",
        "okie",
        "yes",
        "yeah",
        "yep",
        "no",
        "wow",
        "cool",
        "nice",
        "good",
        "great",
        "thanks",
        "thank",
        "tks",
        "ty",
        "cảm",
        "ơn",
        "vâng",
        "dạ",
        "ừ",
        "uh",
        "nha",
        "nhé",
        "ạ",
        "haha",
        "hihi",
        "hehe",
        "lol",
    }
)


def _heuristic_behavior_fallback(reply_text: str) -> dict[str, Any]:
    """Heuristic fallback for offline checkpoint-free demo inspection."""
    low = normalize_nfc(reply_text).lower()
    low = re.sub(r"https?://\S+|www\.\S+", " ", low)
    alpha_words = re.findall(r"[^\W\d_]+", low, flags=re.UNICODE)
    has_substantive_exclamation_text = (
        len(alpha_words) >= 2
        and sum(len(w) for w in alpha_words) >= 6
        and not all(w in _BENIGN_EXCLAMATION_WORDS for w in alpha_words)
    )
    if any(pat.search(low) for pat in _HOSTILE_CUE_PATTERNS):
        lbl = "hostile_conflict"
        s_uv, h_uv, r_uv, c_uv = -0.85, 0.88, 0.65, 0.05
    elif any(pat.search(low) for pat in _IRRATIONAL_CUE_PATTERNS):
        lbl = "irrational_fallacy"
        s_uv, h_uv, r_uv, c_uv = -0.60, 0.45, 0.85, 0.10
    elif any(pat.search(low) for pat in _OPPOSE_CUE_PATTERNS):
        lbl = "constructive_oppose"
        s_uv, h_uv, r_uv, c_uv = -0.65, 0.10, 0.12, 0.82
    elif any(pat.search(low) for pat in _SUPPORT_CUE_PATTERNS):
        lbl = "support"
        s_uv, h_uv, r_uv, c_uv = 0.75, 0.05, 0.05, 0.78
    elif low.count("!") >= 3 and has_substantive_exclamation_text:
        lbl = "hostile_conflict"
        s_uv, h_uv, r_uv, c_uv = -0.85, 0.88, 0.65, 0.05
    else:
        lbl = "neutral"
        s_uv, h_uv, r_uv, c_uv = 0.0, 0.05, 0.08, 0.45

    probs = _default_probs_from_label(lbl)
    return {
        "label": lbl,
        "s_uv": s_uv,
        "h_uv": h_uv,
        "r_uv": r_uv,
        "c_uv": c_uv,
        "probs": probs,
        "stance_probs": behavior_probs_to_stance_probs(probs),
        "fallback_used": True,
    }

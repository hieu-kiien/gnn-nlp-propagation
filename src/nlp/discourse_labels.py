"""Lightweight 5-class discourse category constants and rule-based fallback helpers.

Extracted cleanly without any heavy generative-LLM (`transformers`/`peft`/`bitsandbytes`)
dependencies so audit utilities, offline live-thread inspection, and feature extractors
stay lightweight.
"""

from __future__ import annotations

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
    "biến",
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
    "không đồng ý",
    "chưa hợp lý",
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
    "chuẩn",
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


def _heuristic_behavior_fallback(reply_text: str) -> dict[str, Any]:
    """Heuristic fallback for offline checkpoint-free demo inspection."""
    low = normalize_nfc(reply_text).lower()
    if any(tok in low for tok in _HOSTILE_CUE_WORDS) or low.count("!") >= 3:
        lbl = "hostile_conflict"
        s_uv, h_uv, r_uv, c_uv = -0.85, 0.88, 0.65, 0.05
    elif any(tok in low for tok in _IRRATIONAL_CUE_WORDS):
        lbl = "irrational_fallacy"
        s_uv, h_uv, r_uv, c_uv = -0.60, 0.45, 0.85, 0.10
    elif any(tok in low for tok in _OPPOSE_CUE_WORDS):
        lbl = "constructive_oppose"
        s_uv, h_uv, r_uv, c_uv = -0.65, 0.10, 0.12, 0.82
    elif any(tok in low for tok in _SUPPORT_CUE_WORDS):
        lbl = "support"
        s_uv, h_uv, r_uv, c_uv = 0.75, 0.05, 0.05, 0.78
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

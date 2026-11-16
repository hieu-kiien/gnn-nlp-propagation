"""System-2 Teacher (`Qwen/Qwen3.8-27B` + QLoRA) for 5-Class Behavioral Discourse Scoring.

Classifies social media `(root_post, parent_comment, reply_comment)` triplets into
5 fine-grained discourse behavior categories beyond simple agree/disagree:
  0. `support`: Logical agreement / constructive reinforcement
  1. `neutral`: Informational query / neutral observation
  2. `constructive_oppose`: Civil, evidence-based disagreement or counter-argument
  3. `irrational_fallacy`: Unreasonable, whataboutism, strawman, non-sequitur, trolling
  4. `hostile_conflict`: Ad-hominem attacks, hate/harassment, toxic provocation

Outputs calibrated 4D continuous edge attributes `e_uv = [s_uv, h_uv, r_uv, c_uv]`:
  - `s_uv in [-1, 1]`: Stance polarity (`P(support) - P(all_opposing_categories)`)
  - `h_uv in [0, 1]`: Hostile conflict / verbal antagonism intensity
  - `r_uv in [0, 1]`: Irrationality / fallacy / derailment intensity
  - `c_uv in [0, 1]`: Constructive reasoning quality

Optimized for Kaggle dual Tesla T4 (`T4 x 2`, 32 GB total VRAM, Turing `sm_75`):
  - Strictly `torch.float16` compute + 4-bit NF4 double quantization (`device_map="auto"`).
  - Freezes multimodal vision towers (`visual` / `vision_tower`) when attaching LoRA.
  - Zero Data Leakage: QLoRA trains strictly on external gold datasets (`UIT-ViHSD`,
    `RumourEval 2019 SDQC`, curated behavioral rubric anchors, and optional disjoint pilot).
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

QWEN_TEACHER_MODEL_ID: str = "Qwen/Qwen3.8-27B"
EN_DEBERTA_MODEL_NAME: str = "microsoft/deberta-v3-base"

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

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

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


def _normalize_nfc(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text).replace("\x00", "").strip()


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _clamp_stance(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(1.0, max(-1.0, value)))


def build_t4x2_loading_config(
    *,
    num_gpus: int | None = None,
    max_per_gpu_gib: int = 14,
    cpu_offload_gib: int = 24,
) -> dict[str, Any]:
    """Build hardware-safe 4-bit NF4 float16 loading config for Kaggle T4 x 2 (32GB VRAM).

    Tesla T4 uses Turing `sm_75` (no fast native `bfloat16`), so compute dtype is
    pinned strictly to `torch.float16`.
    """
    detected_gpus = num_gpus if num_gpus is not None else torch.cuda.device_count()
    bnb_config: Any = None
    try:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    except Exception:
        bnb_config = None

    if detected_gpus and detected_gpus > 0:
        max_memory: dict[int | str, str] = {
            gpu_idx: f"{int(max_per_gpu_gib)}GiB" for gpu_idx in range(int(detected_gpus))
        }
        max_memory["cpu"] = f"{int(cpu_offload_gib)}GiB"
        device_map: str | None = "auto"
    else:
        max_memory = {"cpu": f"{int(cpu_offload_gib)}GiB"}
        device_map = None

    return {
        "quantization_config": bnb_config,
        "device_map": device_map,
        "max_memory": max_memory,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }


build_t4x2_LoadingConfig = build_t4x2_loading_config


def build_behavioral_prompt(
    root_text: str,
    parent_text: str,
    reply_text: str,
    lang: str = "vi",
) -> list[dict[str, str]]:
    """Construct compact 5-class discourse behavior prompt for `(root, parent, reply)`."""
    r_txt = _normalize_nfc(root_text)[:600]
    p_txt = _normalize_nfc(parent_text)[:600]
    c_txt = _normalize_nfc(reply_text)[:600]

    if lang == "vi":
        system_msg = (
            "Bạn là chuyên gia phân tích tranh luận mạng xã hội Việt Nam/quốc tế. "
            "Hãy phân loại bình luận phản hồi so với bình luận cha và bài gốc vào ĐÚNG 1 trong 5 nhãn:\n"
            "1) support: tán thành, bổ sung tích cực\n"
            "2) neutral: trung lập, hỏi thông tin\n"
            "3) constructive_oppose: phản biện văn minh, có lý lẽ/dẫn chứng\n"
            "4) irrational_fallacy: bình luận vô lý, ngụy biện, nói cùn, bẻ lái, mỉa mai vô căn cứ\n"
            "5) hostile_conflict: chống đối gay gắt, công kích cá nhân, chửi bới, kích động xung đột\n"
            "Chỉ trả về DUY NHẤT 1 dòng JSON hợp lệ theo mẫu:\n"
            '{"label":"...","s_uv":0.0,"h_uv":0.0,"r_uv":0.0,"c_uv":0.0,"probs":[0,0,0,0,0]}'
        )
        user_msg = (
            f"[Bài gốc]: {r_txt}\n"
            f"[Bình luận trước]: {p_txt}\n"
            f"[Bình luận trả lời]: {c_txt}\n"
            "JSON:"
        )
    else:
        system_msg = (
            "You are an expert social-media discourse analyst. Classify the reply relative to the "
            "parent comment and root post into ONE of 5 labels:\n"
            "1) support\n2) neutral\n3) constructive_oppose (civil, reasoned counter-argument)\n"
            "4) irrational_fallacy (illogical, strawman, whataboutism, derailment, baseless sarcasm)\n"
            "5) hostile_conflict (toxic personal attack, harassment, inflammatory provocation)\n"
            "Return STRICTLY one valid JSON object on a single line:\n"
            '{"label":"...","s_uv":0.0,"h_uv":0.0,"r_uv":0.0,"c_uv":0.0,"probs":[0,0,0,0,0]}'
        )
        user_msg = (
            f"[Root Post]: {r_txt}\n"
            f"[Parent Comment]: {p_txt}\n"
            f"[Reply]: {c_txt}\n"
            "JSON:"
        )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


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
    low = _normalize_nfc(reply_text).lower()
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


def _strip_thinking_and_fences(raw_text: str) -> str:
    text = str(raw_text or "")
    text = _THINK_BLOCK_RE.sub("", text)
    if "<think>" in text.lower() and "</think>" not in text.lower():
        # Handle truncated thinking trace if generation stopped early
        parts = re.split(r"<think>", text, flags=re.IGNORECASE)
        text = parts[0]
        if "{" in parts[-1]:
            text = text + "\n" + parts[-1][parts[-1].find("{") :]
    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1)
    return text.strip()


def parse_behavioral_teacher_json(
    raw_text: str,
    *,
    reply_text: str = "",
) -> dict[str, Any]:
    """Parse teacher generation into normalized 5-class probabilities and 4D `[s, h, r, c]` scores."""
    cleaned = _strip_thinking_and_fences(raw_text)
    parsed: dict[str, Any] | None = None

    json_match = _JSON_OBJECT_RE.search(cleaned)
    if json_match:
        candidate = json_match.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                parsed = obj
        except json.JSONDecodeError:
            parsed = None

    if parsed is None:
        return _heuristic_behavior_fallback(reply_text)

    raw_lbl = str(parsed.get("label") or "").strip().lower()
    norm_lbl = _LABEL_SYNONYMS.get(raw_lbl)

    raw_probs = parsed.get("probs")
    probs: list[float] | None = None
    if isinstance(raw_probs, Sequence) and not isinstance(raw_probs, (str, bytes)) and len(raw_probs) == 5:
        try:
            arr = [max(0.0, float(v)) if math.isfinite(float(v)) else 0.0 for v in raw_probs]
            tot = sum(arr)
            if tot > 1e-9:
                probs = [v / tot for v in arr]
        except (TypeError, ValueError):
            probs = None

    if probs is None:
        probs = _default_probs_from_label(norm_lbl or "neutral")
    if norm_lbl is None:
        norm_lbl = DISCOURSE_BEHAVIOR_LABELS[int(np.argmax(probs))]

    stance_probs = behavior_probs_to_stance_probs(probs)
    derived_s = float(probs[0] - (probs[2] + probs[3] + probs[4]))

    if "s_uv" in parsed:
        try:
            s_uv = _clamp_stance(float(parsed["s_uv"]))
        except (TypeError, ValueError):
            s_uv = _clamp_stance(derived_s)
    else:
        s_uv = _clamp_stance(derived_s)

    defaults_by_label = {
        "support": (0.05, 0.05, 0.80),
        "neutral": (0.05, 0.05, 0.50),
        "constructive_oppose": (0.10, 0.10, 0.85),
        "irrational_fallacy": (0.40, 0.85, 0.10),
        "hostile_conflict": (0.90, 0.65, 0.05),
    }
    def_h, def_r, def_c = defaults_by_label[norm_lbl]

    def _read_unit(key: str, default: float) -> float:
        if key not in parsed:
            return default
        try:
            return _clamp01(float(parsed[key]))
        except (TypeError, ValueError):
            return default

    h_uv = _read_unit("h_uv", def_h)
    r_uv = _read_unit("r_uv", def_r)
    c_uv = _read_unit("c_uv", def_c)

    return {
        "label": norm_lbl,
        "s_uv": s_uv,
        "h_uv": h_uv,
        "r_uv": r_uv,
        "c_uv": c_uv,
        "probs": probs,
        "stance_probs": stance_probs,
        "fallback_used": False,
    }


_CURATED_GOLD_SEEDS: tuple[dict[str, Any], ...] = (
    # Vietnamese social debate calibration examples (5 behavioral classes)
    {
        "lang": "vi",
        "root": "Đề xuất hạn chế xe máy xăng trong khu vực trung tâm để giảm ô nhiễm không khí.",
        "parent": "Không khí nội đô nhiều hôm bụi mịn vượt ngưỡng, làm dần theo lộ trình là hợp lý.",
        "reply": "Chuẩn luôn, đồng ý với bác, nên hỗ trợ thêm xe buýt điện trước khi áp dụng.",
        "label": "support",
        "s_uv": 0.88,
        "h_uv": 0.03,
        "r_uv": 0.02,
        "c_uv": 0.85,
        "probs": [0.88, 0.06, 0.03, 0.02, 0.01],
    },
    {
        "lang": "vi",
        "root": "Đề xuất hạn chế xe máy xăng trong khu vực trung tâm để giảm ô nhiễm không khí.",
        "parent": "Dự thảo dự kiến áp dụng thí điểm từ năm sau trên 4 quận lõi.",
        "reply": "Cho mình hỏi xe máy điện có bị giới hạn khung giờ hay vẫn đi bình thường vậy mọi người?",
        "label": "neutral",
        "s_uv": 0.0,
        "h_uv": 0.02,
        "r_uv": 0.03,
        "c_uv": 0.65,
        "probs": [0.05, 0.86, 0.05, 0.03, 0.01],
    },
    {
        "lang": "vi",
        "root": "Đề xuất hạn chế xe máy xăng trong khu vực trung tâm để giảm ô nhiễm không khí.",
        "parent": "Cấm xe xăng là tự nhiên đường phố sạch ngay thôi, có gì khó đâu.",
        "reply": "Tôi chưa đồng ý điểm này vì hạ tầng giao thông công cộng mới đáp ứng khoảng 15% nhu cầu, nên mở rộng tuyến metro trước khi cấm.",
        "label": "constructive_oppose",
        "s_uv": -0.82,
        "h_uv": 0.06,
        "r_uv": 0.05,
        "c_uv": 0.92,
        "probs": [0.03, 0.08, 0.84, 0.04, 0.01],
    },
    {
        "lang": "vi",
        "root": "Đề xuất hạn chế xe máy xăng trong khu vực trung tâm để giảm ô nhiễm không khí.",
        "parent": "Số liệu quan trắc cho thấy khí thải giao thông chiếm tỷ trọng lớn.",
        "reply": "Ôi dào, đỉnh quá cơ =)) hay là cấm luôn người dân hít thở cho đỡ khí thải? Chắc nhà bác đi ô tô máy lạnh quen rồi.",
        "label": "irrational_fallacy",
        "s_uv": -0.78,
        "h_uv": 0.42,
        "r_uv": 0.90,
        "c_uv": 0.06,
        "probs": [0.02, 0.04, 0.10, 0.72, 0.12],
    },
    {
        "lang": "vi",
        "root": "Đề xuất hạn chế xe máy xăng trong khu vực trung tâm để giảm ô nhiễm không khí.",
        "parent": "Mình thấy cần cân nhắc chi phí đổi xe cho người lao động thu nhập thấp.",
        "reply": "Não tàn mới đi bênh mấy thành phần phá hoại môi trường, nghèo mà lười còn bày đặt lý do, cút hết về quê đi!",
        "label": "hostile_conflict",
        "s_uv": -0.94,
        "h_uv": 0.96,
        "r_uv": 0.78,
        "c_uv": 0.02,
        "probs": [0.01, 0.01, 0.03, 0.10, 0.85],
    },
    {
        "lang": "vi",
        "root": "Tranh luận về việc cấm sử dụng điện thoại hoàn toàn trong giờ ra chơi tại trường THPT.",
        "parent": "Cấm hẳn để học sinh giao tiếp trực tiếp với nhau cũng có mặt tích cực.",
        "reply": "Nhưng nếu gia đình có việc gấp cần liên lạc hay học sinh cần tra tài liệu dự án nhóm thì trường có phương án dự phòng chưa?",
        "label": "constructive_oppose",
        "s_uv": -0.68,
        "h_uv": 0.04,
        "r_uv": 0.04,
        "c_uv": 0.90,
        "probs": [0.05, 0.12, 0.79, 0.03, 0.01],
    },
    {
        "lang": "vi",
        "root": "Tranh luận về việc cấm sử dụng điện thoại hoàn toàn trong giờ ra chơi tại trường THPT.",
        "parent": "Học sinh thời nay phụ thuộc mạng xã hội quá nhiều.",
        "reply": "Đồ cổ hủ ngáo quyền lực, thế thời trước không có điện thoại chắc tụi bây giỏi hơn thiên hạ à? Tụi rác rưởi thích ra vẻ!",
        "label": "hostile_conflict",
        "s_uv": -0.91,
        "h_uv": 0.93,
        "r_uv": 0.74,
        "c_uv": 0.02,
        "probs": [0.01, 0.02, 0.04, 0.13, 0.80],
    },
    {
        "lang": "vi",
        "root": "Tranh luận về việc tăng giá điện sinh hoạt theo bậc thang mới.",
        "parent": "Bậc thang mới giúp người dùng ít tiết kiệm hơn.",
        "reply": "Thế sao không tăng lương gấp 10 lần trước đi rồi hãy bàn chuyện điện? Lại bài ca tẩy trắng cho tập đoàn thôi, buồn cười thật sự.",
        "label": "irrational_fallacy",
        "s_uv": -0.76,
        "h_uv": 0.40,
        "r_uv": 0.86,
        "c_uv": 0.08,
        "probs": [0.02, 0.04, 0.12, 0.70, 0.12],
    },
    # English social debate calibration examples
    {
        "lang": "en",
        "root": "City council proposes congestion pricing in downtown business districts.",
        "parent": "Revenue from the toll will directly fund expanded express bus lanes.",
        "reply": "Agreed, London and Stockholm saw measurable drops in gridlock once transit alternatives scaled up.",
        "label": "support",
        "s_uv": 0.86,
        "h_uv": 0.02,
        "r_uv": 0.03,
        "c_uv": 0.90,
        "probs": [0.86, 0.07, 0.04, 0.02, 0.01],
    },
    {
        "lang": "en",
        "root": "City council proposes congestion pricing in downtown business districts.",
        "parent": "The toll fee begins at 7 AM on weekdays.",
        "reply": "Does anyone know if electric delivery vans are exempt during the pilot phase?",
        "label": "neutral",
        "s_uv": 0.0,
        "h_uv": 0.02,
        "r_uv": 0.02,
        "c_uv": 0.70,
        "probs": [0.04, 0.88, 0.05, 0.02, 0.01],
    },
    {
        "lang": "en",
        "root": "City council proposes congestion pricing in downtown business districts.",
        "parent": "Everyone commuting downtown can easily switch to commuter rail.",
        "reply": "That assumes suburban rail lines run past 9 PM; shift workers in hospitality have no train option and would bear a regressive cost.",
        "label": "constructive_oppose",
        "s_uv": -0.80,
        "h_uv": 0.05,
        "r_uv": 0.04,
        "c_uv": 0.93,
        "probs": [0.02, 0.07, 0.86, 0.04, 0.01],
    },
    {
        "lang": "en",
        "root": "City council proposes congestion pricing in downtown business districts.",
        "parent": "Traffic engineers published the peak emission estimates today.",
        "reply": "Yeah right, because billionaires fly private jets so clearly banning cars means they want us locked in 15-minute prisons lol wake up sheeple.",
        "label": "irrational_fallacy",
        "s_uv": -0.82,
        "h_uv": 0.35,
        "r_uv": 0.92,
        "c_uv": 0.04,
        "probs": [0.01, 0.03, 0.08, 0.78, 0.10],
    },
    {
        "lang": "en",
        "root": "City council proposes congestion pricing in downtown business districts.",
        "parent": "Small retail shops worry delivery surcharges could hurt margins.",
        "reply": "Shut up you absolute moron, only braindead losers complain about traffic rules, get lost!",
        "label": "hostile_conflict",
        "s_uv": -0.95,
        "h_uv": 0.97,
        "r_uv": 0.72,
        "c_uv": 0.02,
        "probs": [0.01, 0.01, 0.02, 0.09, 0.87],
    },
)


def _format_example_record(
    *,
    root_text: str,
    parent_text: str,
    reply_text: str,
    lang: str,
    label: str,
    s_uv: float,
    h_uv: float,
    r_uv: float,
    c_uv: float,
    probs: list[float],
) -> dict[str, Any]:
    target_json = json.dumps(
        {
            "label": label,
            "s_uv": round(float(s_uv), 4),
            "h_uv": round(float(h_uv), 4),
            "r_uv": round(float(r_uv), 4),
            "c_uv": round(float(c_uv), 4),
            "probs": [round(float(p), 4) for p in probs],
        },
        ensure_ascii=False,
    )
    messages = build_behavioral_prompt(root_text, parent_text, reply_text, lang=lang)
    return {
        "lang": lang,
        "root_text": root_text,
        "parent_text": parent_text,
        "reply_text": reply_text,
        "label": label,
        "s_uv": float(s_uv),
        "h_uv": float(h_uv),
        "r_uv": float(r_uv),
        "c_uv": float(c_uv),
        "probs": probs,
        "messages": messages + [{"role": "assistant", "content": target_json}],
    }


def build_qlora_gold_examples(
    *,
    lang: str = "vi",
    vihsd_samples: Sequence[Mapping[str, Any]] | None = None,
    rumoureval_samples: Sequence[Mapping[str, Any]] | None = None,
    pilot_gold_csv: Path | None = None,
    max_samples: int = 1200,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build strictly leakage-free QLoRA calibration examples from gold external sources.

    Combines curated 5-class rubric anchors with optional external datasets (`UIT-ViHSD`
    hate/hostility samples, `RumourEval` SDQC stance pairs, and an explicit human-adjudicated
    pilot CSV). Never reads held-out target `val`/`test` cascade splits.
    """
    if lang not in {"vi", "en"}:
        raise ValueError(f"Unsupported lang {lang!r}; expected 'vi' or 'en'")

    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []

    for item in _CURATED_GOLD_SEEDS:
        # Include both languages with slight priority to active language
        if item["lang"] == lang or lang == "vi":
            examples.append(
                _format_example_record(
                    root_text=item["root"],
                    parent_text=item["parent"],
                    reply_text=item["reply"],
                    lang=item["lang"],
                    label=item["label"],
                    s_uv=item["s_uv"],
                    h_uv=item["h_uv"],
                    r_uv=item["r_uv"],
                    c_uv=item["c_uv"],
                    probs=list(item["probs"]),
                )
            )

    if vihsd_samples:
        for row in vihsd_samples:
            text = _normalize_nfc(str(row.get("free_text") or row.get("text") or ""))
            if not text:
                continue
            raw_lbl = int(row.get("label_id", 0))
            if raw_lbl == 2:  # Hate -> hostile_conflict
                lbl, s_v, h_v, r_v, c_v = "hostile_conflict", -0.85, 0.92, 0.65, 0.05
            elif raw_lbl == 1:  # Offensive -> irrational_fallacy / mild hostility
                lbl, s_v, h_v, r_v, c_v = "irrational_fallacy", -0.55, 0.62, 0.68, 0.12
            else:
                lbl, s_v, h_v, r_v, c_v = "neutral", 0.0, 0.05, 0.08, 0.55
            examples.append(
                _format_example_record(
                    root_text="Chủ đề thảo luận cộng đồng mạng xã hội Việt Nam.",
                    parent_text="Ý kiến trao đổi trong chủ đề.",
                    reply_text=text,
                    lang="vi",
                    label=lbl,
                    s_uv=s_v,
                    h_uv=h_v,
                    r_uv=r_v,
                    c_uv=c_v,
                    probs=_default_probs_from_label(lbl),
                )
            )

    if rumoureval_samples:
        sdqc_map = {
            "support": ("support", 0.82, 0.05, 0.05, 0.80),
            "supporting": ("support", 0.82, 0.05, 0.05, 0.80),
            "deny": ("constructive_oppose", -0.78, 0.12, 0.10, 0.82),
            "denying": ("constructive_oppose", -0.78, 0.12, 0.10, 0.82),
            "query": ("neutral", 0.0, 0.04, 0.04, 0.65),
            "questioning": ("neutral", 0.0, 0.04, 0.04, 0.65),
            "comment": ("neutral", 0.0, 0.08, 0.10, 0.45),
            "commenting": ("neutral", 0.0, 0.08, 0.10, 0.45),
        }
        for row in rumoureval_samples:
            parent_txt = _normalize_nfc(str(row.get("parent_text") or ""))
            reply_txt = _normalize_nfc(str(row.get("reply_text") or ""))
            if not parent_txt or not reply_txt:
                continue
            gold_raw = str(row.get("gold_sdqc") or "comment").strip().lower()
            lbl, s_v, h_v, r_v, c_v = sdqc_map.get(
                gold_raw, ("neutral", 0.0, 0.08, 0.10, 0.45)
            )
            examples.append(
                _format_example_record(
                    root_text=_normalize_nfc(str(row.get("root_text") or parent_txt)),
                    parent_text=parent_txt,
                    reply_text=reply_txt,
                    lang="en",
                    label=lbl,
                    s_uv=s_v,
                    h_uv=h_v,
                    r_uv=r_v,
                    c_uv=c_v,
                    probs=_default_probs_from_label(lbl),
                )
            )

    if pilot_gold_csv is not None and Path(pilot_gold_csv).exists():
        with Path(pilot_gold_csv).open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                p_txt = _normalize_nfc(row.get("parent_text", ""))
                c_txt = _normalize_nfc(row.get("reply_text", ""))
                raw_st = str(
                    row.get("gold_adjudicated_stance") or row.get("annotator_1_stance") or ""
                ).strip().lower()
                raw_host = str(
                    row.get("gold_adjudicated_hostility") or row.get("annotator_1_hostility") or ""
                ).strip().lower()
                if not p_txt or not c_txt or not raw_st:
                    continue
                lbl = _LABEL_SYNONYMS.get(raw_st)
                if lbl is None:
                    if raw_host in {"hate", "2"}:
                        lbl = "hostile_conflict"
                    elif raw_st in {"oppose", "-1"} and raw_host in {"offensive", "1"}:
                        lbl = "irrational_fallacy"
                    elif raw_st in {"oppose", "-1"}:
                        lbl = "constructive_oppose"
                    elif raw_st in {"support", "1", "+1"}:
                        lbl = "support"
                    else:
                        lbl = "neutral"
                h_map = {"clean": 0.05, "0": 0.05, "offensive": 0.55, "1": 0.55, "hate": 0.92, "2": 0.92}
                h_val = h_map.get(raw_host, 0.10)
                s_map = {
                    "support": 0.80,
                    "neutral": 0.0,
                    "constructive_oppose": -0.75,
                    "irrational_fallacy": -0.70,
                    "hostile_conflict": -0.90,
                }
                examples.append(
                    _format_example_record(
                        root_text=_normalize_nfc(row.get("root_text") or p_txt),
                        parent_text=p_txt,
                        reply_text=c_txt,
                        lang=lang,
                        label=lbl,
                        s_uv=s_map[lbl],
                        h_uv=h_val,
                        r_uv=0.80 if lbl == "irrational_fallacy" else (0.60 if lbl == "hostile_conflict" else 0.08),
                        c_uv=0.85 if lbl in {"support", "constructive_oppose"} else (0.50 if lbl == "neutral" else 0.05),
                        probs=_default_probs_from_label(lbl),
                    )
                )

    rng.shuffle(examples)
    if max_samples > 0 and len(examples) > max_samples:
        examples = examples[:max_samples]
    return examples


class _CausalSFTDataset(Dataset):
    def __init__(self, tokenized_rows: Sequence[dict[str, torch.Tensor]]) -> None:
        self.rows = list(tokenized_rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.rows[idx]


def _render_chat_text(tokenizer: Any, messages: Sequence[Mapping[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=False,
                )
            )
        except TypeError:
            return str(
                tokenizer.apply_chat_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            )
    parts = [f"<{m['role']}>\n{m['content']}" for m in messages]
    if add_generation_prompt:
        parts.append("<assistant>\n")
    return "\n".join(parts)


def _freeze_vision_modules_for_text_qlora(model: torch.nn.Module) -> list[str]:
    """Explicitly freeze multimodal vision towers so QLoRA updates text projections only."""
    frozen: list[str] = []
    for name, param in model.named_parameters():
        low = name.lower()
        if any(tag in low for tag in ("visual", "vision_tower", "vision_model", "merger")):
            param.requires_grad_(False)
            frozen.append(name)
    return frozen


def _load_external_vihsd_subset(limit: int = 800, seed: int = 42) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset

        ds = load_dataset("ura-hcmut/UIT-ViHSD", split="train")
        indices = list(range(len(ds)))
        random.Random(seed).shuffle(indices)
        rows: list[dict[str, Any]] = []
        for idx in indices[:limit]:
            item = ds[int(idx)]
            rows.append(
                {
                    "free_text": item.get("free_text", ""),
                    "label_id": int(item.get("label_id", 0)),
                }
            )
        return rows
    except Exception:
        return []


def finetune_qwen_qlora(
    *,
    output_dir: Path = Path("checkpoints/qwen38_27b_qlora"),
    model_name_or_path: str = QWEN_TEACHER_MODEL_ID,
    lang: str = "vi",
    examples: Sequence[Mapping[str, Any]] | None = None,
    pilot_gold_csv: Path | None = None,
    include_vihsd: bool = True,
    max_vihsd_samples: int = 600,
    epochs: int = 1,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    lr: float = 1e-4,
    max_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    seed: int = 42,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> Path:
    """Adapt Qwen3.8-27B via text-only 4-bit QLoRA on external gold behavioral examples.

    Keeps 99.5%+ of base weights frozen in NF4 across Kaggle `T4 x 2` and adapts only
    thin LoRA projections on gold examples (never target `val` or `test` cascades).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if examples is None:
        vihsd_rows = _load_external_vihsd_subset(limit=max_vihsd_samples, seed=seed) if (include_vihsd and lang == "vi") else None
        gold_records = build_qlora_gold_examples(
            lang=lang,
            vihsd_samples=vihsd_rows,
            pilot_gold_csv=pilot_gold_csv,
            seed=seed,
        )
    else:
        gold_records = [dict(r) for r in examples]

    if not gold_records:
        raise ValueError("Cannot fine-tune QLoRA on empty calibration examples")

    owns_model = model is None
    tok = tokenizer
    active_model = model

    if tok is None or active_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if tok is None:
            tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        if active_model is None:
            load_cfg = build_t4x2_loading_config()
            active_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                **{k: v for k, v in load_cfg.items() if v is not None},
            )
            if hasattr(active_model, "config") and hasattr(active_model.config, "use_cache"):
                active_model.config.use_cache = False

    frozen_vision = _freeze_vision_modules_for_text_qlora(active_model)

    used_peft = False
    if owns_model:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            active_model = prepare_model_for_kbit_training(active_model, use_gradient_checkpointing=True)
            # Match text attention/MLP linear projections explicitly without touching vision modules
            text_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=text_target_modules,
                exclude_modules=".*(visual|vision_tower|vision_model).*",
            )
            try:
                active_model = get_peft_model(active_model, lora_cfg)
            except TypeError:
                lora_cfg = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=text_target_modules,
                )
                active_model = get_peft_model(active_model, lora_cfg)
            _freeze_vision_modules_for_text_qlora(active_model)
            used_peft = True
        except Exception:
            used_peft = False

    tokenized_rows: list[dict[str, torch.Tensor]] = []
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tok, "eos_token_id", 0) or 0

    for rec in gold_records:
        messages = rec.get("messages")
        if not messages:
            messages = build_behavioral_prompt(
                rec.get("root_text", ""),
                rec.get("parent_text", ""),
                rec.get("reply_text", ""),
                lang=str(rec.get("lang", lang)),
            ) + [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "label": rec.get("label", "neutral"),
                            "s_uv": float(rec.get("s_uv", 0.0)),
                            "h_uv": float(rec.get("h_uv", 0.0)),
                            "r_uv": float(rec.get("r_uv", 0.0)),
                            "c_uv": float(rec.get("c_uv", 0.5)),
                            "probs": rec.get("probs", [0.1, 0.6, 0.1, 0.1, 0.1]),
                        },
                        ensure_ascii=False,
                    ),
                }
            ]
        full_text = _render_chat_text(tok, messages, add_generation_prompt=False)
        enc = tok(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0).long()
        attention_mask = enc["attention_mask"].squeeze(0).long()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        tokenized_rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    loader = DataLoader(_CausalSFTDataset(tokenized_rows), batch_size=max(1, batch_size), shuffle=True)
    trainable_params = [p for p in active_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for Qwen QLoRA adaptation")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    first_param = trainable_params[0]
    param_device = first_param.device
    use_amp = param_device.type == "cuda"

    active_model.train()
    losses: list[float] = []
    accum = max(1, int(grad_accum_steps))

    for _epoch in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        steps = 0
        for step_idx, batch in enumerate(loader):
            batch_dev = {k: v.to(param_device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = active_model(**batch_dev)
                loss = out.loss / float(accum)
            loss.backward()
            epoch_loss += float(loss.detach().cpu().item()) * float(accum)
            steps += 1
            if (step_idx + 1) % accum == 0 or (step_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        losses.append(epoch_loss / max(1, steps))

    if hasattr(active_model, "save_pretrained"):
        active_model.save_pretrained(out_path)
    if hasattr(tok, "save_pretrained"):
        tok.save_pretrained(out_path)

    report = {
        "model_name_or_path": model_name_or_path,
        "lang": lang,
        "num_examples": len(gold_records),
        "used_peft_qlora": used_peft,
        "frozen_vision_tensors": len(frozen_vision),
        "epochs": epochs,
        "epoch_losses": losses,
        "zero_target_leakage": True,
    }
    (out_path / "qlora_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


class QwenBehavioralTeacherScorer:
    """System-2 teacher scoring `(root, parent, reply)` into 5 classes and 4D `[s, h, r, c]`.

    Uses `Qwen/Qwen3.8-27B` (optionally with QLoRA adapter) across Kaggle `T4 x 2`,
    disabling verbose `<think>` tokens and falling back deterministically on malformed JSON.
    """

    def __init__(
        self,
        *,
        model_name_or_path: str = QWEN_TEACHER_MODEL_ID,
        adapter_dir: Path | str | None = None,
        lang: str = "vi",
        model: Any | None = None,
        tokenizer: Any | None = None,
        custom_generate_fn: Callable[[Sequence[str]], Sequence[str]] | None = None,
        max_new_tokens: int = 160,
        batch_size: int = 8,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.adapter_dir = Path(adapter_dir) if adapter_dir is not None else None
        self.lang = lang
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = max(1, int(batch_size))
        self.custom_generate_fn = custom_generate_fn
        self.model = model
        self.tokenizer = tokenizer

        if self.custom_generate_fn is None and self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok_source = (
                str(self.adapter_dir)
                if self.adapter_dir is not None and self.adapter_dir.exists()
                else self.model_name_or_path
            )
            self.tokenizer = self.tokenizer or AutoTokenizer.from_pretrained(
                tok_source, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            load_cfg = build_t4x2_loading_config()
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                **{k: v for k, v in load_cfg.items() if v is not None},
            )
            if self.adapter_dir is not None and self.adapter_dir.exists():
                try:
                    from peft import PeftModel

                    base_model = PeftModel.from_pretrained(base_model, str(self.adapter_dir))
                except Exception:
                    pass
            base_model.eval()
            for p in base_model.parameters():
                p.requires_grad_(False)
            self.model = base_model
        elif self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()

    def _generate_batch(self, prompts: Sequence[str]) -> list[str]:
        if self.custom_generate_fn is not None:
            return [str(x) for x in self.custom_generate_fn(prompts)]

        assert self.model is not None and self.tokenizer is not None
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        enc = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=1536,
            return_tensors="pt",
        )
        first_param = next(self.model.parameters(), None)
        target_dev = first_param.device if first_param is not None else torch.device("cpu")
        enc_dev = {k: v.to(target_dev) for k, v in enc.items()}
        input_len = int(enc_dev["input_ids"].shape[1])

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(target_dev.type == "cuda")):
                gen_ids = self.model.generate(
                    **enc_dev,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                )

        outputs: list[str] = []
        for row_idx in range(gen_ids.shape[0]):
            new_tokens = gen_ids[row_idx, input_len:]
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(str(decoded))
        return outputs

    def score_triplets(
        self,
        root_texts: Sequence[str],
        parent_texts: Sequence[str],
        reply_texts: Sequence[str],
        *,
        batch_size: int | None = None,
        lang: str | None = None,
    ) -> dict[str, Any]:
        if not (len(root_texts) == len(parent_texts) == len(reply_texts)):
            raise ValueError("root_texts, parent_texts, and reply_texts must have equal length")

        n = len(reply_texts)
        if n == 0:
            return {
                "s_uv": np.zeros((0,), dtype=np.float32),
                "h_uv": np.zeros((0,), dtype=np.float32),
                "r_uv": np.zeros((0,), dtype=np.float32),
                "c_uv": np.zeros((0,), dtype=np.float32),
                "behavior_probs": np.zeros((0, 5), dtype=np.float32),
                "behavior_label": [],
                "stance_probs": np.zeros((0, 3), dtype=np.float32),
                "pred_label": [],
            }

        active_lang = lang or self.lang
        bs = max(1, int(batch_size or self.batch_size))

        s_list: list[float] = []
        h_list: list[float] = []
        r_list: list[float] = []
        c_list: list[float] = []
        b_probs_list: list[list[float]] = []
        b_labels: list[str] = []
        st_probs_list: list[list[float]] = []
        st_labels: list[str] = []
        stance_3_names = ("support", "neutral", "oppose")

        for start in range(0, n, bs):
            prompts: list[str] = []
            slice_replies = reply_texts[start : start + bs]
            for idx in range(start, min(n, start + bs)):
                msgs = build_behavioral_prompt(
                    root_texts[idx],
                    parent_texts[idx],
                    reply_texts[idx],
                    lang=active_lang,
                )
                prompts.append(_render_chat_text(self.tokenizer, msgs, add_generation_prompt=True))

            raw_outputs = self._generate_batch(prompts)
            for raw_out, rep_txt in zip(raw_outputs, slice_replies):
                parsed = parse_behavioral_teacher_json(raw_out, reply_text=str(rep_txt))
                s_list.append(float(parsed["s_uv"]))
                h_list.append(float(parsed["h_uv"]))
                r_list.append(float(parsed["r_uv"]))
                c_list.append(float(parsed["c_uv"]))
                b_probs_list.append(list(parsed["probs"]))
                b_labels.append(str(parsed["label"]))
                st_p = list(parsed["stance_probs"])
                st_probs_list.append(st_p)
                st_labels.append(stance_3_names[int(np.argmax(st_p))])

        return {
            "s_uv": np.asarray(s_list, dtype=np.float32),
            "h_uv": np.asarray(h_list, dtype=np.float32),
            "r_uv": np.asarray(r_list, dtype=np.float32),
            "c_uv": np.asarray(c_list, dtype=np.float32),
            "behavior_probs": np.asarray(b_probs_list, dtype=np.float32),
            "behavior_label": b_labels,
            "stance_probs": np.asarray(st_probs_list, dtype=np.float32),
            "pred_label": st_labels,
        }

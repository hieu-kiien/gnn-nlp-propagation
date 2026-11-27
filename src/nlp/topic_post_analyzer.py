"""Zero-shot AI topic classification and full-thread large-post discourse analyzer.

Extends SBi-FAGAT beyond the fixed early-horizon (`K=20` / `K=10`) snapshot so both
research evaluations and the interactive Demo Web MVP can:
  1. Classify arbitrary posts and discussions under either standard Vietnamese/English
     topic taxonomies or user-specified custom topics (`mDeBERTa` zero-shot NLI when
     available, Transformer cosine embedding over `ViSoBERT`/`Twitter-RoBERTa`, or an
     honest deterministic lexical n-gram fallback offline).
  2. Analyze full large discussion trees (`N >> 20`, up to hundreds of replies): track
     how polarization `y_polar` evolves across milestones (`K=5, 10, 20, ..., N`),
     assign quantitative discourse archetypes grounded in VN-Train quantiles, and surface
     key arguments on both sides plus top conflict flashpoints without violating the
     frozen Tier-1 early-horizon evaluation benchmark.
"""

from __future__ import annotations

import math
import re
import threading
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F

from src.data.pyg_dataset import (
    _compute_full_tree_polarization,
    _compute_root_aligned_stances_and_orientations,
)
from src.nlp.dual_edge_scorer import MDEBERTA_NLI_MODEL_NAME
from src.nlp.extract_features import compute_masked_mean_pooling
from src.utils.text import normalize_nfc

# Canonical empirical future_only y_polar quantiles computed strictly on Train splits
# (zero Val/Test leakage).
# VN: N = 1,050 cascades, K = 20, min_total_replies = 25, topic_splits.json train split only.
VN_POLAR_Q50: float = 0.0927623026072979
VN_POLAR_Q75: float = 0.12019770219922066
VN_POLAR_Q90: float = 0.15035123974084857
# PHEME: N = 1,786 cascades, K = 10, min_total_replies = 15, topic_splits.json train split only.
PHEME_POLAR_Q50: float = 0.06384398415684700
PHEME_POLAR_Q75: float = 0.09671241976320744
PHEME_POLAR_Q90: float = 0.13436114042997360


def get_polar_quantiles(lang: Literal["vi", "en"] = "vi") -> tuple[float, float, float]:
    """Return Train-calibrated `(q50, q75, q90)` polarization thresholds for `vi` or `en`."""
    if str(lang).lower() == "en":
        return (PHEME_POLAR_Q50, PHEME_POLAR_Q75, PHEME_POLAR_Q90)
    return (VN_POLAR_Q50, VN_POLAR_Q75, VN_POLAR_Q90)

DEFAULT_TOPICS_VI: tuple[dict[str, str], ...] = (
    {
        "id": "policy_traffic",
        "label": "Chính sách & Giao thông",
        "description": "Chính sách công, pháp luật, quy định nhà nước, thuế phí và giao thông đô thị đường bộ xe cộ phạt nguội",
    },
    {
        "id": "economy_finance",
        "label": "Kinh tế & Tài chính",
        "description": "Kinh tế, tài chính cá nhân, đầu tư, chứng khoán, ngân hàng, lương giá cả và bất động sản",
    },
    {
        "id": "tech_ai",
        "label": "Công nghệ & AI",
        "description": "Công nghệ thông tin, lập trình phần mềm, trí tuệ nhân tạo AI, dữ liệu, bảo mật và thiết bị số",
    },
    {
        "id": "education_career",
        "label": "Giáo dục & Nghề nghiệp",
        "description": "Giáo dục, đại học, sinh viên, tuyển dụng, việc làm IT, môi trường công sở, hybrid WFH và lương thưởng",
    },
    {
        "id": "society_life",
        "label": "Đời sống & Xã hội",
        "description": "Đời sống xã hội, gia đình, tâm lý, ứng xử cộng đồng, văn hóa tranh luận và các mối quan hệ",
    },
    {
        "id": "health_lifestyle",
        "label": "Sức khỏe & Lối sống",
        "description": "Sức khỏe, y tế, bệnh viện, dinh dưỡng, luyện tập thể thao và thói quen sinh hoạt",
    },
    {
        "id": "entertainment_media",
        "label": "Giải trí & Truyền thông",
        "description": "Giải trí, phim ảnh, âm nhạc, trò chơi điện tử, mạng xã hội và văn hóa đại chúng",
    },
    {
        "id": "news_controversy",
        "label": "Thời sự & Tranh cãi",
        "description": "Tin tức thời sự nóng, sự kiện dư luận, tin đồn kiểm chứng và tranh cãi đa chiều trên mạng",
    },
)

DEFAULT_TOPICS_EN: tuple[dict[str, str], ...] = (
    {
        "id": "policy_traffic",
        "label": "Policy, Law & Governance",
        "description": "Public policy, law, government regulation, policing, civic infrastructure and urban transit",
    },
    {
        "id": "economy_finance",
        "label": "Economy & Finance",
        "description": "Economics, personal finance, markets, inflation, taxation, business and housing",
    },
    {
        "id": "tech_ai",
        "label": "Technology & AI",
        "description": "Software engineering, artificial intelligence, computing hardware, cybersecurity and digital platforms",
    },
    {
        "id": "education_career",
        "label": "Education & Career",
        "description": "Universities, learning, hiring, workplace culture, remote work, engineering teams and careers",
    },
    {
        "id": "society_life",
        "label": "Society & Community",
        "description": "Social norms, family, daily life, interpersonal ethics and community discourse",
    },
    {
        "id": "health_lifestyle",
        "label": "Health & Lifestyle",
        "description": "Public health, medicine, nutrition, mental wellbeing and fitness",
    },
    {
        "id": "entertainment_media",
        "label": "Entertainment & Media",
        "description": "Movies, music, gaming, social media trends and pop culture",
    },
    {
        "id": "news_controversy",
        "label": "Breaking News & Rumors",
        "description": "Breaking news, crisis events, rumor verification, misinformation and public controversy",
    },
)

_SCALE_LABELS_VI: dict[str, str] = {
    "small": "Bài nhỏ / Khởi phát",
    "medium": "Bài trung bình",
    "large": "Bài lớn",
    "mega": "Bài rất lớn / Bùng nổ",
}

_SCALE_LABELS_EN: dict[str, str] = {
    "small": "Small / Early Thread",
    "medium": "Medium Thread",
    "large": "Large Thread",
    "mega": "Mega / Viral Thread",
}

_ARCHETYPE_LABELS_VI: dict[str, str] = {
    "hostile_flamewar": "Xung đột gay gắt / Bùng nổ",
    "bipolar_debate": "Phân cực lưỡng cực rõ rệt",
    "consensus_support": "Đồng thuận tích cực",
    "constructive_critique": "Phản biện ôn hòa / Đa chiều",
    "exploratory_neutral": "Thảo luận trung lập / Trao đổi thông tin",
}

_ARCHETYPE_LABELS_EN: dict[str, str] = {
    "hostile_flamewar": "High-Hostility Flamewar",
    "bipolar_debate": "Bipolar Debate",
    "consensus_support": "Supportive Consensus",
    "constructive_critique": "Constructive Critique",
    "exploratory_neutral": "Exploratory / Neutral Exchange",
}

_TREND_LABELS_VI: dict[str, str] = {
    "escalating": "Gia tăng phân cực theo thời gian",
    "de_escalating": "Hạ nhiệt / Hội tụ dần",
    "stable": "Ổn định quanh ngưỡng ban đầu",
    "single_window": "Chưa đủ bình luận hậu K để đo xu hướng",
}

_TREND_LABELS_EN: dict[str, str] = {
    "escalating": "Escalating polarization over time",
    "de_escalating": "De-escalating / Converging",
    "stable": "Stable around early baseline",
    "single_window": "Insufficient post-K replies for trend",
}

_SPLIT_CUSTOM_TOPICS_RE = re.compile(r"[,;\r\n]+")
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_JUNK_LIKE_RE = re.compile(r"^[\W_0-9]+$", re.UNICODE)
_VI_STOPWORDS: frozenset[str] = frozenset(
    {
        "và",
        "là",
        "của",
        "có",
        "cho",
        "với",
        "trong",
        "trên",
        "này",
        "kia",
        "đó",
        "thì",
        "mà",
        "nhưng",
        "nên",
        "vì",
        "nếu",
        "như",
        "khi",
        "đã",
        "đang",
        "sẽ",
        "bị",
        "được",
        "còn",
        "cũng",
        "các",
        "những",
        "một",
        "không",
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "about",
        "into",
    }
)

_NLI_LOCK = threading.Lock()
_NLI_CACHE: dict[str, tuple[Any, Any, int, int] | None] = {}
_ENC_CAND_LOCK = threading.Lock()
_ENC_CAND_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")


def _slugify_topic_id(label: str, idx: int) -> str:
    ascii_str = _strip_accents(label).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_str).strip("_")
    return slug[:40] if slug else f"custom_topic_{idx}"


def parse_custom_topics(raw_topics: str | Sequence[str] | None) -> list[dict[str, str]]:
    """Parse user-defined custom topics from comma/newline string or list, deduplicated <= 16."""
    if raw_topics is None:
        return []
    if isinstance(raw_topics, str):
        raw_items = _SPLIT_CUSTOM_TOPICS_RE.split(raw_topics)
    elif isinstance(raw_topics, Sequence):
        raw_items = []
        for item in raw_topics:
            if item is not None:
                raw_items.extend(_SPLIT_CUSTOM_TOPICS_RE.split(str(item)))
    else:
        raise ValueError(f"Unsupported custom_topics type: {type(raw_topics).__name__}")

    topics: list[dict[str, str]] = []
    seen_low: set[str] = set()
    used_ids: set[str] = set()
    for item in raw_items:
        cleaned = " ".join(normalize_nfc(str(item)).split()).strip()
        if not cleaned or len(cleaned) > 120:
            continue
        low = cleaned.lower()
        if low in seen_low:
            continue
        seen_low.add(low)
        tid = _slugify_topic_id(cleaned, len(topics) + 1)
        if tid in used_ids:
            tid = f"{tid}_{len(topics) + 1}"
        used_ids.add(tid)
        topics.append({"id": tid, "label": cleaned, "description": cleaned})
        if len(topics) >= 16:
            break
    return topics


def _build_topic_context_text(
    title: str,
    body: str,
    reply_snippets: Sequence[str],
    *,
    max_chars: int = 1200,
) -> str:
    clean_title = " ".join(normalize_nfc(title or "").split())
    clean_body = " ".join(normalize_nfc(body or "").split())
    parts: list[str] = []
    if clean_title:
        parts.append(clean_title)
    if clean_body and clean_body.lower() != clean_title.lower():
        parts.append(clean_body)
    valid_replies = [
        " ".join(normalize_nfc(str(r)).split())
        for r in reply_snippets
        if r and len(" ".join(str(r).split())) >= 8
    ]
    if valid_replies:
        parts.append(" | ".join(valid_replies[:5]))
    joined = ". ".join(parts).strip()
    return joined[:max_chars] if len(joined) > max_chars else joined


def _resolve_nli_indices(config: Any) -> tuple[int, int]:
    """Extract `(entailment_idx, contradiction_idx)` from HF NLI `config.label2id`."""
    label2id = getattr(config, "label2id", None) or {}
    norm_map = {str(k).strip().lower(): int(v) for k, v in label2id.items()}
    ent_idx = norm_map.get("entailment", 0)
    con_idx = norm_map.get("contradiction", 2)
    return int(ent_idx), int(con_idx)


def _get_cached_nli_model(
    model_name: str = MDEBERTA_NLI_MODEL_NAME,
    *,
    allow_download: bool = False,
) -> tuple[Any, Any, int, int] | None:
    """Lazy-load NLI classifier strictly from local cache unless `allow_download=True`."""
    with _NLI_LOCK:
        if model_name in _NLI_CACHE and (_NLI_CACHE[model_name] is not None or not allow_download):
            return _NLI_CACHE[model_name]

        local_only = not allow_download
        if local_only and not Path(model_name).exists():
            import os

            hf_hub_env = os.environ.get("HF_HUB_CACHE")
            hf_home_env = os.environ.get("HF_HOME")
            if hf_hub_env:
                hf_hub = Path(hf_hub_env)
            elif hf_home_env:
                hf_hub = Path(hf_home_env) / "hub"
            else:
                hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
            snap_dir = hf_hub / f"models--{model_name.replace('/', '--')}" / "snapshots"
            if not snap_dir.is_dir() or not any(snap_dir.iterdir()):
                _NLI_CACHE[model_name] = None
                return None
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, local_files_only=local_only
            )
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            ent_idx, con_idx = _resolve_nli_indices(model.config)
            bundle = (model, tok, ent_idx, con_idx)
            _NLI_CACHE[model_name] = bundle
            return bundle
        except Exception:
            if not allow_download:
                _NLI_CACHE[model_name] = None
            return None


def _score_topics_with_nli(
    context_text: str,
    candidates: Sequence[Mapping[str, str]],
    *,
    nli_bundle: tuple[Any, Any, int, int],
    lang: str,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    model, tok, ent_idx, con_idx = nli_bundle
    model = model.to(device)
    if lang == "en":
        hypotheses = [f"This text is about {c['label']}." for c in candidates]
    else:
        hypotheses = [f"Văn bản này nói về {c['label']}." for c in candidates]

    enc = tok(
        [context_text] * len(candidates),
        hypotheses,
        padding=True,
        truncation=True,
        max_length=192,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits.float()
    diff = logits[:, ent_idx] - logits[:, con_idx]
    relevances = torch.sigmoid(diff).detach().cpu().tolist()
    scores = F.softmax(diff / 1.5, dim=0).detach().cpu().tolist()
    return [float(s) for s in scores], [float(r) for r in relevances]


def _score_topics_with_encoder(
    context_text: str,
    candidates: Sequence[Mapping[str, str]],
    *,
    node_encoder: Any,
    node_tokenizer: Any,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    try:
        target_dev = next(node_encoder.parameters()).device
    except StopIteration:
        target_dev = device

    cand_texts = [f"{c['label']}: {c['description']}" for c in candidates]
    cache_key = (id(node_encoder), str(target_dev), tuple(cand_texts))
    with _ENC_CAND_LOCK:
        cached_cand_vecs = _ENC_CAND_CACHE.get(cache_key)

    node_encoder.eval()
    if cached_cand_vecs is not None:
        enc_ctx = node_tokenizer(
            [context_text],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc_ctx = {k: v.to(target_dev) for k, v in enc_ctx.items()}
        with torch.no_grad():
            out = node_encoder(**enc_ctx)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            ctx_vec = F.normalize(compute_masked_mean_pooling(hidden, enc_ctx["attention_mask"]), p=2, dim=-1)
        cand_vecs = cached_cand_vecs.to(target_dev)
    else:
        all_texts = [context_text] + cand_texts
        enc = node_tokenizer(
            all_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(target_dev) for k, v in enc.items()}
        with torch.no_grad():
            out = node_encoder(**enc)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            pooled = compute_masked_mean_pooling(hidden, enc["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=-1)
        ctx_vec = pooled[0:1]
        cand_vecs = pooled[1:]
        with _ENC_CAND_LOCK:
            if len(_ENC_CAND_CACHE) >= 64:
                _ENC_CAND_CACHE.clear()
            _ENC_CAND_CACHE[cache_key] = cand_vecs.detach().cpu()

    sims = torch.matmul(cand_vecs, ctx_vec.squeeze(0))
    relevances = ((sims + 1.0) * 0.5).clamp(0.0, 1.0).detach().cpu().tolist()
    scores = F.softmax(sims / 0.08, dim=0).detach().cpu().tolist()
    return [float(s) for s in scores], [float(r) for r in relevances]


def _lexical_feature_weights(text: str) -> Counter[str]:
    norm = normalize_nfc(text).lower()
    raw_tokens = [t for t in _WORD_RE.findall(norm) if len(t) >= 2 and t not in _VI_STOPWORDS]
    ascii_tokens = [_strip_accents(t) for t in raw_tokens]

    weights: Counter[str] = Counter()
    for t in raw_tokens:
        weights[f"w:{t}"] += 2.2
    for t in ascii_tokens:
        weights[f"a:{t}"] += 1.4
    for a, b in zip(ascii_tokens, ascii_tokens[1:]):
        weights[f"bi:{a}_{b}"] += 2.5

    compact = " ".join(ascii_tokens)
    for n in (3, 4):
        if len(compact) >= n:
            for i in range(len(compact) - n + 1):
                gram = compact[i : i + n]
                if " " not in gram:
                    weights[f"c{n}:{gram}"] += 0.35
    return weights


def _score_topics_lexical(
    context_text: str,
    candidates: Sequence[Mapping[str, str]],
) -> tuple[list[float], list[float]]:
    ctx_w = _lexical_feature_weights(context_text)
    ctx_norm = math.sqrt(sum(v * v for v in ctx_w.values()))
    ctx_ascii = _strip_accents(context_text.lower())
    sims: list[float] = []
    for c in candidates:
        c_desc = f"{c['label']} {c['description']}"
        cand_w = _lexical_feature_weights(c_desc)
        cand_norm = math.sqrt(sum(v * v for v in cand_w.values()))
        if ctx_norm <= 1e-8 or cand_norm <= 1e-8:
            sim = 0.0
        else:
            dot = sum(val * cand_w[k] for k, val in ctx_w.items() if k in cand_w)
            sim = dot / (ctx_norm * cand_norm)
        label_ascii = _strip_accents(c["label"].lower()).strip()
        if label_ascii and len(label_ascii) >= 2:
            if re.search(rf"(?<![a-z0-9]){re.escape(label_ascii)}(?![a-z0-9])", ctx_ascii):
                sim += 0.35
        sims.append(float(sim))

    max_sim = max(sims) if sims else 0.0
    logits = [10.0 * (s - max_sim) for s in sims]
    exps = [math.exp(max(-30.0, min(0.0, lg))) for lg in logits]
    total = sum(exps) or 1.0
    scores = [e / total for e in exps]
    relevances = [min(1.0, max(0.0, s / 0.45)) for s in sims]
    return scores, relevances


def classify_post_topics(
    title: str,
    body: str = "",
    *,
    reply_snippets: Sequence[str] = (),
    custom_topics: str | Sequence[str] | None = None,
    lang: Literal["vi", "en"] = "vi",
    backend: Literal["auto", "nli", "embedding", "lexical"] = "auto",
    node_encoder: Any = None,
    node_tokenizer: Any = None,
    allow_nli_download: bool = False,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Classify post & early replies across default or arbitrary user-defined topics.

    Explicitly records `topic_backend` and `transformer_topic_loaded` so callers and
    the UI never misrepresent offline lexical fallback as Transformer inference.
    """
    if lang not in {"vi", "en"}:
        raise ValueError(f"Unsupported lang={lang!r}; expected 'vi' or 'en'")
    if backend not in {"auto", "nli", "embedding", "lexical"}:
        raise ValueError(f"Unsupported topic backend={backend!r}")

    parsed_custom = parse_custom_topics(custom_topics)
    is_custom = len(parsed_custom) > 0
    candidates: list[dict[str, str]] = (
        parsed_custom
        if is_custom
        else [dict(t) for t in (DEFAULT_TOPICS_VI if lang == "vi" else DEFAULT_TOPICS_EN)]
    )

    context_text = _build_topic_context_text(title, body, reply_snippets)
    target_device = device or torch.device("cpu")

    used_backend = "lexical_ngram_fallback"
    scores: list[float] | None = None
    relevances: list[float] | None = None

    if context_text and backend in {"auto", "nli"}:
        nli_bundle = _get_cached_nli_model(
            MDEBERTA_NLI_MODEL_NAME,
            allow_download=bool(allow_nli_download and backend in {"auto", "nli"}),
        )
        if nli_bundle is not None:
            try:
                scores, relevances = _score_topics_with_nli(
                    context_text,
                    candidates,
                    nli_bundle=nli_bundle,
                    lang=lang,
                    device=target_device,
                )
                used_backend = "mdeberta_zero_shot_nli"
            except Exception:
                scores = None

    if scores is None and context_text and backend in {"auto", "embedding"}:
        if node_encoder is not None and node_tokenizer is not None:
            try:
                scores, relevances = _score_topics_with_encoder(
                    context_text,
                    candidates,
                    node_encoder=node_encoder,
                    node_tokenizer=node_tokenizer,
                    device=target_device,
                )
                used_backend = "transformer_cosine_embedding"
            except Exception:
                scores = None

    if scores is None or relevances is None:
        if context_text:
            scores, relevances = _score_topics_lexical(context_text, candidates)
        else:
            uniform = 1.0 / float(len(candidates))
            scores = [uniform] * len(candidates)
            relevances = [0.0] * len(candidates)
        used_backend = "lexical_ngram_fallback"

    ranked: list[dict[str, Any]] = []
    for cand, sc, rel in zip(candidates, scores, relevances):
        ranked.append(
            {
                "id": cand["id"],
                "label": cand["label"],
                "description": cand["description"],
                "score": round(float(sc), 4),
                "relevance": round(float(rel), 4),
            }
        )
    ranked.sort(key=lambda item: (-item["score"], -item["relevance"], item["label"]))

    top = ranked[0]
    secondary = [item["label"] for item in ranked[1:3] if item["score"] >= 0.15]

    return {
        "primary_topic": top["label"],
        "primary_topic_id": top["id"],
        "primary_confidence": round(float(top["score"]), 4),
        "secondary_topics": secondary,
        "is_custom_taxonomy": is_custom,
        "topic_backend": used_backend,
        "transformer_topic_loaded": used_backend != "lexical_ngram_fallback",
        "topics": ranked,
    }


def _classify_scale_tier(total_replies: int) -> tuple[str, str]:
    if total_replies >= 300:
        return "mega", _SCALE_LABELS_VI["mega"]
    if total_replies >= 100:
        return "large", _SCALE_LABELS_VI["large"]
    if total_replies >= 25:
        return "medium", _SCALE_LABELS_VI["medium"]
    return "small", _SCALE_LABELS_VI["small"]


def _classify_post_archetype(
    full_polar: float,
    *,
    support_ratio: float,
    oppose_ratio: float,
    hostile_ratio: float,
    total_replies: int,
    lang: Literal["vi", "en"] = "vi",
) -> tuple[str, str, str]:
    _, q75, q90 = get_polar_quantiles(lang)
    q50, _, _ = get_polar_quantiles(lang)
    if total_replies == 0:
        key = "exploratory_neutral"
    elif hostile_ratio >= 0.20 or (full_polar >= q90 and hostile_ratio >= 0.12):
        key = "hostile_flamewar"
    elif full_polar >= q75 or (support_ratio >= 0.20 and oppose_ratio >= 0.20):
        key = "bipolar_debate"
    elif support_ratio >= 0.45 and oppose_ratio <= 0.15 and hostile_ratio < 0.10:
        key = "consensus_support"
    elif oppose_ratio >= 0.25 or full_polar >= q50:
        key = "constructive_critique"
    else:
        key = "exploratory_neutral"
    return key, _ARCHETYPE_LABELS_VI[key], _ARCHETYPE_LABELS_EN[key]


def extract_hotspot_branches(
    *,
    root_id: str,
    all_nodes: Sequence[Mapping[str, Any]],
    all_edges: Sequence[Mapping[str, Any]],
    edge_scores: Mapping[str, Mapping[str, Any]],
    depth_map: Mapping[str, int],
    node_orientations: Mapping[str, float] | None = None,
    top_k: int = 5,
    max_nodes_per_branch: int = 80,
) -> list[dict[str, Any]]:
    """Rank top-k depth-1 discussion branches by heated debate volume and cap subtree size for lag-free SVG."""
    if top_k < 1 or max_nodes_per_branch < 2:
        return []

    node_by_id: dict[str, Mapping[str, Any]] = {str(n["node_id"]): n for n in all_nodes}
    children_adj: dict[str, list[str]] = {str(n["node_id"]): [] for n in all_nodes}
    parent_of: dict[str, str] = {}
    edge_score_by_target: dict[str, tuple[float, float]] = {}

    for e in all_edges:
        src = str(e["source_id"])
        dst = str(e["target_id"])
        eid = str(e.get("edge_id") or f"{src}->{dst}")
        children_adj.setdefault(src, []).append(dst)
        children_adj.setdefault(dst, [])
        parent_of[dst] = src
        sc = edge_scores.get(eid) or {}
        s_uv = float(sc.get("s_uv", e.get("s_uv", 0.0)))
        h_uv = float(sc.get("h_uv", e.get("h_uv", 0.0)))
        edge_score_by_target[dst] = (s_uv, h_uv)

    root_children = list(children_adj.get(root_id, ()))
    if not root_children:
        return []

    ori_map = node_orientations or {}
    from collections import deque

    branches: list[dict[str, Any]] = []
    for lead_id in root_children:
        if lead_id not in node_by_id:
            continue
        subtree_ids: list[str] = []
        seen_b: set[str] = {root_id, lead_id}
        q: deque[str] = deque([lead_id])
        max_b_depth = int(depth_map.get(lead_id, 1))
        while q:
            curr = q.popleft()
            subtree_ids.append(curr)
            d_val = int(depth_map.get(curr, 1))
            if d_val > max_b_depth:
                max_b_depth = d_val
            for ch in children_adj.get(curr, ()):
                if ch not in seen_b:
                    seen_b.add(ch)
                    q.append(ch)

        size = len(subtree_ids)
        if size == 0:
            continue

        sup_c = opp_c = neu_c = hos_c = 0
        for nid in subtree_ids:
            s_uv, h_uv = edge_score_by_target.get(nid, (0.0, 0.0))
            ori = float(ori_map.get(nid, s_uv))
            if ori > 0.1:
                sup_c += 1
            elif ori < -0.1:
                opp_c += 1
            else:
                neu_c += 1
            if h_uv >= 0.5:
                hos_c += 1

        opp_ratio = opp_c / float(size)
        hos_ratio = hos_c / float(size)
        heat_score = float(size) * (0.5 + opp_ratio + hos_ratio)

        # Select connected representative nodes <= max_nodes_per_branch:
        # prioritize shallow BFS backbone + highest intensity nodes with ancestor closure.
        target_cap = max(2, int(max_nodes_per_branch))
        selected_set: set[str] = {root_id, lead_id}
        selected_order: list[str] = [root_id, lead_id]

        scored_desc = sorted(
            [nid for nid in subtree_ids if nid != lead_id],
            key=lambda nid: (
                int(depth_map.get(nid, 999)) > 3,
                -(edge_score_by_target.get(nid, (0.0, 0.0))[1] + abs(edge_score_by_target.get(nid, (0.0, 0.0))[0])),
                int(depth_map.get(nid, 999)),
                nid,
            ),
        )
        for nid in scored_desc:
            if len(selected_order) >= target_cap:
                break
            chain: list[str] = []
            curr = nid
            visited_chain: set[str] = set()
            while curr and curr not in selected_set and curr not in visited_chain:
                visited_chain.add(curr)
                chain.append(curr)
                curr = parent_of.get(curr, "")
            if len(selected_order) + len(chain) <= target_cap:
                for anc in reversed(chain):
                    if anc not in selected_set:
                        selected_set.add(anc)
                        selected_order.append(anc)

        lead_txt = " ".join(str(node_by_id[lead_id].get("text") or "").split())
        if len(lead_txt) > 140:
            lead_txt = lead_txt[:139].rstrip() + "…"

        branches.append(
            {
                "branch_id": lead_id,
                "lead_text": lead_txt,
                "size": size,
                "support_count": sup_c,
                "oppose_count": opp_c,
                "neutral_count": neu_c,
                "hostile_count": hos_c,
                "oppose_ratio": round(opp_ratio, 4),
                "hostile_ratio": round(hos_ratio, 4),
                "max_depth": max_b_depth,
                "heat_score": round(heat_score, 3),
                "node_ids": selected_order,
                "truncated_for_view": size > max(0, target_cap - 1),
            }
        )

    branches.sort(key=lambda b: (-float(b["heat_score"]), -int(b["size"]), str(b["branch_id"])))
    return branches[:top_k]


def analyze_large_thread_dynamics(
    *,
    root_node: Mapping[str, Any],
    all_nodes: Sequence[Mapping[str, Any]],
    all_edges: Sequence[Mapping[str, Any]],
    edge_scores: Mapping[str, Mapping[str, Any]],
    depth_map: Mapping[str, int],
    topic_analysis: Mapping[str, Any] | None = None,
    trigger_importance_by_target: Mapping[str, float] | None = None,
    k_early_ref: int = 20,
    lang: Literal["vi", "en"] = "vi",
    scope: Literal["early_post_hoc", "full_analysis"] = "full_analysis",
) -> dict[str, Any]:
    """Compute multi-milestone polarization trajectory and extractive discourse summary for a full post."""
    lang_norm: Literal["vi", "en"] = "en" if str(lang).lower() == "en" else "vi"
    root_id = str(root_node["node_id"])
    non_roots = [n for n in all_nodes if str(n["node_id"]) != root_id]

    def _sort_key(node: Mapping[str, Any]) -> tuple[float, int, str]:
        ts = node.get("created_at_utc")
        if hasattr(ts, "timestamp"):
            ts_val = float(ts.timestamp())
        else:
            try:
                ts_val = float(ts) if ts is not None else 0.0
            except (TypeError, ValueError):
                ts_val = 0.0
        nid = str(node["node_id"])
        return (ts_val, int(depth_map.get(nid, 999)), nid)

    non_roots.sort(key=_sort_key)
    total_replies = len(non_roots)
    scale_tier, scale_label_vi = _classify_scale_tier(total_replies)
    scale_label_en = _SCALE_LABELS_EN[scale_tier]

    children_adj: dict[str, list[str]] = {root_id: []}
    edge_lookup: dict[tuple[str, str], dict[str, str]] = {}
    stance_map: dict[str, float] = {}
    hostility_by_target: dict[str, float] = {}
    edge_score_by_target: dict[str, tuple[float, float]] = {}

    for e in all_edges:
        src = str(e["source_id"])
        dst = str(e["target_id"])
        eid = str(e.get("edge_id") or f"{src}->{dst}")
        children_adj.setdefault(src, []).append(dst)
        children_adj.setdefault(dst, [])
        edge_lookup[(src, dst)] = {"edge_id": eid}
        sc = edge_scores.get(eid) or {}
        s_uv = float(sc.get("s_uv", e.get("s_uv", 0.0)))
        h_uv = float(sc.get("h_uv", e.get("h_uv", 0.0)))
        stance_map[eid] = s_uv
        hostility_by_target[dst] = h_uv
        edge_score_by_target[dst] = (s_uv, h_uv)

    full_aligned, node_orientations = _compute_root_aligned_stances_and_orientations(
        root_id=root_id,
        children_adj=children_adj,
        edge_lookup=edge_lookup,
        stance_map=stance_map,
    )
    full_y_polar = (
        float(_compute_full_tree_polarization(full_aligned, lambda_polar=0.5))
        if len(full_aligned) >= 2
        else 0.0
    )

    milestone_grid = (5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 3500, 5000, 7500, 10000)
    milestones = [m for m in milestone_grid if m < total_replies]
    if total_replies > 0:
        milestones.append(total_replies)

    # Bucket edges by max chronological node rank for fast O(N) incremental milestone expansion.
    node_rank: dict[str, int] = {root_id: 0}
    for idx_r, n_r in enumerate(non_roots, start=1):
        node_rank[str(n_r["node_id"])] = idx_r
    edges_by_rank: dict[int, list[tuple[str, str, dict[str, str]]]] = {}
    for (src, dst), e_row in edge_lookup.items():
        r_src = node_rank.get(src)
        r_dst = node_rank.get(dst)
        if r_src is None or r_dst is None:
            continue
        r_max = r_src if r_src >= r_dst else r_dst
        edges_by_rank.setdefault(r_max, []).append((src, dst, e_row))

    pref_adj: dict[str, list[str]] = {}
    pref_lookup: dict[tuple[str, str], dict[str, str]] = {}
    cur_rank_added = -1
    if 0 in edges_by_rank:
        for src, dst, e_row in edges_by_rank[0]:
            pref_adj.setdefault(src, []).append(dst)
            pref_lookup[(src, dst)] = e_row
        cur_rank_added = 0

    trajectory: list[dict[str, Any]] = []
    for k_step in milestones:
        for r_idx in range(cur_rank_added + 1, k_step + 1):
            for src, dst, e_row in edges_by_rank.get(r_idx, ()):
                pref_adj.setdefault(src, []).append(dst)
                pref_lookup[(src, dst)] = e_row
        cur_rank_added = k_step

        prefix_nodes = non_roots[:k_step]
        pref_aligned, pref_orientations = _compute_root_aligned_stances_and_orientations(
            root_id=root_id,
            children_adj=pref_adj,
            edge_lookup=pref_lookup,
            stance_map=stance_map,
        )
        y_step = (
            float(_compute_full_tree_polarization(pref_aligned, lambda_polar=0.5))
            if len(pref_aligned) >= 2
            else 0.0
        )

        sup_c = opp_c = neu_c = hos_c = 0
        authors_k: set[str] = set()
        for idx_n, n in enumerate(prefix_nodes):
            nid = str(n["node_id"])
            ori = float(pref_orientations.get(nid, 0.0))
            _, h_uv = edge_score_by_target.get(nid, (0.0, 0.0))
            if ori > 0.1:
                sup_c += 1
            elif ori < -0.1:
                opp_c += 1
            else:
                neu_c += 1
            if h_uv >= 0.5:
                hos_c += 1
            ah = n.get("author_hash") or n.get("user_id_hash") or f"anon_{idx_n}"
            authors_k.add(str(ah))

        denom = float(max(1, k_step))
        trajectory.append(
            {
                "k": int(k_step),
                "y_polar": round(y_step, 4),
                "support_ratio": round(sup_c / denom, 4),
                "oppose_ratio": round(opp_c / denom, 4),
                "neutral_ratio": round(neu_c / denom, 4),
                "hostile_ratio": round(hos_c / denom, 4),
                "unique_authors": len(authors_k),
            }
        )

    final_pt = (
        trajectory[-1]
        if trajectory
        else {
            "support_ratio": 0.0,
            "oppose_ratio": 0.0,
            "neutral_ratio": 0.0,
            "hostile_ratio": 0.0,
            "unique_authors": 0,
        }
    )

    if total_replies <= k_early_ref or len(trajectory) < 2:
        evolution_trend = "single_window"
        ref_k = int(trajectory[0]["k"]) if trajectory else int(total_replies)
        early_ref_polar = float(trajectory[0]["y_polar"]) if trajectory else full_y_polar
        polar_delta = 0.0
    else:
        ref_pt = next((pt for pt in trajectory if pt["k"] >= k_early_ref), trajectory[0])
        ref_k = int(ref_pt["k"])
        early_ref_polar = float(ref_pt["y_polar"])
        polar_delta = full_y_polar - early_ref_polar
        if polar_delta >= 0.025:
            evolution_trend = "escalating"
        elif polar_delta <= -0.025:
            evolution_trend = "de_escalating"
        else:
            evolution_trend = "stable"

    archetype, archetype_label_vi, archetype_label_en = _classify_post_archetype(
        full_y_polar,
        support_ratio=float(final_pt["support_ratio"]),
        oppose_ratio=float(final_pt["oppose_ratio"]),
        hostile_ratio=float(final_pt["hostile_ratio"]),
        total_replies=total_replies,
        lang=lang_norm,
    )

    pro_candidates: list[tuple[float, dict[str, Any]]] = []
    anti_candidates: list[tuple[float, dict[str, Any]]] = []
    flash_candidates: list[tuple[float, dict[str, Any]]] = []
    imp_map = trigger_importance_by_target or {}
    all_authors: set[str] = {
        str(root_node.get("author_hash") or root_node.get("user_id_hash") or "root")
    }

    for idx_n, n in enumerate(non_roots):
        nid = str(n["node_id"])
        txt = " ".join(str(n.get("text") or "").split())
        ah = n.get("author_hash") or n.get("user_id_hash") or f"anon_{idx_n}"
        all_authors.add(str(ah))
        if not txt or _JUNK_LIKE_RE.fullmatch(txt):
            continue
        ori = float(node_orientations.get(nid, 0.0))
        s_uv, h_uv = edge_score_by_target.get(nid, (0.0, 0.0))
        word_cnt = len(txt.split())
        if word_cnt < 4:
            continue
        len_factor = min(1.8, math.log1p(word_cnt) / 3.0)
        imp = float(imp_map.get(nid, 0.0))
        subtree_deg = len(children_adj.get(nid, ()))
        entry = {
            "node_id": nid,
            "aligned_stance": round(ori, 4),
            "s_uv": round(s_uv, 4),
            "h_uv": round(h_uv, 4),
            "depth": int(depth_map.get(nid, 1)),
            "child_replies": subtree_deg,
            "text": txt[:240] + ("…" if len(txt) > 240 else ""),
        }
        if ori > 0.1:
            pro_candidates.append((ori * len_factor * (1.0 + imp) * (1.0 - 0.35 * h_uv), entry))
        elif ori < -0.1:
            anti_candidates.append((abs(ori) * len_factor * (1.0 + imp) * (1.0 - 0.25 * h_uv), entry))

        flash_score = 0.5 * h_uv + 0.3 * abs(s_uv) + 0.2 * min(1.0, math.log1p(subtree_deg) / 2.0)
        if h_uv >= 0.45 or (abs(s_uv) >= 0.35 and subtree_deg >= 1):
            flash_candidates.append((flash_score, entry))

    pro_candidates.sort(key=lambda pair: (-pair[0], pair[1]["depth"], pair[1]["node_id"]))
    anti_candidates.sort(key=lambda pair: (-pair[0], pair[1]["depth"], pair[1]["node_id"]))
    flash_candidates.sort(key=lambda pair: (-pair[0], pair[1]["depth"], pair[1]["node_id"]))

    primary_topic = str((topic_analysis or {}).get("primary_topic") or "Chưa phân loại")
    primary_conf = float((topic_analysis or {}).get("primary_confidence") or 0.0)
    max_depth = max((int(depth_map.get(str(n["node_id"]), 0)) for n in non_roots), default=0)

    if total_replies == 0:
        summary_vi = (
            f"Tổng hợp trích xuất: Bài viết thuộc chủ đề \"{primary_topic}\" hiện chưa có bình luận phản hồi hợp lệ."
        )
        summary_en = (
            f"Extractive synthesis: Post under topic \"{primary_topic}\" currently has no valid replies."
        )
    else:
        trend_phrase_vi = (
            f"xu hướng {(_TREND_LABELS_VI.get(evolution_trend, evolution_trend)).lower()} "
            f"(y_polar từ {early_ref_polar:.3f} ở K={ref_k} "
            f"đến {full_y_polar:.3f} toàn cây)"
            if evolution_trend != "single_window"
            else f"độ phân cực quan sát y_polar = {full_y_polar:.3f}"
        )
        summary_vi = (
            f"Tổng hợp trích xuất — Chủ đề chính: {primary_topic} (độ tin cậy {primary_conf:.1%}). "
            f"Cấu trúc quy mô \"{scale_label_vi}\" gồm {total_replies} bình luận từ {len(all_authors)} tác giả "
            f"(độ sâu tối đa {max_depth} tầng), hình thái \"{archetype_label_vi}\" "
            f"(Tán thành {final_pt['support_ratio']:.0%}, Phản đối {final_pt['oppose_ratio']:.0%}, "
            f"Gay gắt {final_pt['hostile_ratio']:.0%}) với {trend_phrase_vi}."
        )
        trend_phrase_en = (
            f"{_TREND_LABELS_EN.get(evolution_trend, evolution_trend).lower()} "
            f"(y_polar from {early_ref_polar:.3f} at K={ref_k} "
            f"to {full_y_polar:.3f} full-tree)"
            if evolution_trend != "single_window"
            else f"observed y_polar = {full_y_polar:.3f}"
        )
        summary_en = (
            f"Extractive synthesis — Primary topic: {primary_topic} ({primary_conf:.1%} confidence). "
            f"\"{scale_label_en}\" with {total_replies} replies from {len(all_authors)} authors "
            f"(max depth {max_depth}), pattern \"{archetype_label_en}\" "
            f"(Support {final_pt['support_ratio']:.0%}, Oppose {final_pt['oppose_ratio']:.0%}, "
            f"Hostile {final_pt['hostile_ratio']:.0%}), {trend_phrase_en}."
        )

    hotspot_branches = extract_hotspot_branches(
        root_id=root_id,
        all_nodes=all_nodes,
        all_edges=all_edges,
        edge_scores=edge_scores,
        depth_map=depth_map,
        node_orientations=node_orientations,
        top_k=5,
        max_nodes_per_branch=80,
    )

    return {
        "lang": lang_norm,
        "scope": scope,
        "total_replies_analyzed": total_replies,
        "unique_authors_total": len(all_authors),
        "max_depth": max_depth,
        "scale_tier": scale_tier,
        "scale_label_vi": scale_label_vi,
        "scale_label_en": scale_label_en,
        "full_y_polar": round(full_y_polar, 4),
        "early_ref_k": ref_k,
        "early_ref_y_polar": round(float(early_ref_polar), 4),
        "polar_delta": round(float(polar_delta), 4),
        "evolution_trend": evolution_trend,
        "evolution_trend_label_vi": _TREND_LABELS_VI[evolution_trend],
        "evolution_trend_label_en": _TREND_LABELS_EN[evolution_trend],
        "post_archetype": archetype,
        "post_archetype_label_vi": archetype_label_vi,
        "post_archetype_label_en": archetype_label_en,
        "stance_ratios_full": {
            "support": float(final_pt["support_ratio"]),
            "oppose": float(final_pt["oppose_ratio"]),
            "neutral": float(final_pt["neutral_ratio"]),
            "hostile": float(final_pt["hostile_ratio"]),
        },
        "polarization_trajectory": trajectory,
        "hotspot_branches": hotspot_branches,
        "semantic_understanding": {
            "synthesis_method": "extractive_quantitative",
            "summary": summary_en if lang_norm == "en" else summary_vi,
            "summary_vi": summary_vi,
            "summary_en": summary_en,
            "top_pro_arguments": [item[1] for item in pro_candidates[:3]],
            "top_anti_arguments": [item[1] for item in anti_candidates[:3]],
            "flashpoint_comments": [item[1] for item in flash_candidates[:3]],
        },
    }

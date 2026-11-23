"""Compute inter-annotator agreement (Cohen's Kappa) & Machine-vs-Gold validity metrics.

Supports stance annotations (`support` / `neutral` / `oppose`, or `1` / `0` / `-1`),
binary reply incivility (`clean` / `uncivil`, accepting legacy `offensive`/`hate`/`hostile`),
5-class discourse behavior labels, RumourEval SDQC external benchmark validation, and
thread-level polarization Likert grading (`0..4`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.nlp.discourse_labels import DISCOURSE_BEHAVIOR_LABELS, _LABEL_SYNONYMS

_STANCE_SYNONYMS: dict[str, str] = {
    "support": "support",
    "entailment": "support",
    "agree": "support",
    "1": "support",
    "+1": "support",
    "neutral": "neutral",
    "0": "neutral",
    "oppose": "oppose",
    "contradiction": "oppose",
    "disagree": "oppose",
    "-1": "oppose",
}

_HOSTILITY_SYNONYMS: dict[str, str] = {
    "clean": "clean",
    "non-offensive": "clean",
    "0": "clean",
    "uncivil": "uncivil",
    "hostile": "uncivil",
    "offensive": "uncivil",
    "1": "uncivil",
    "hate": "uncivil",
    "2": "uncivil",
}

STANCE_ORDER: tuple[str, ...] = ("support", "neutral", "oppose")
HOSTILITY_ORDER: tuple[str, ...] = ("clean", "uncivil")

_SDQC_TO_STANCE: dict[str, str] = {
    "support": "support",
    "supporting": "support",
    "agree": "support",
    "deny": "oppose",
    "denying": "oppose",
    "oppose": "oppose",
    "disagree": "oppose",
    "query": "neutral",
    "questioning": "neutral",
    "comment": "neutral",
    "commenting": "neutral",
    "neutral": "neutral",
}

_STANCE_TO_NUM: dict[str, float] = {"oppose": -1.0, "neutral": 0.0, "support": 1.0}
_HOSTILITY_TO_NUM: dict[str, float] = {"clean": 0.0, "uncivil": 1.0}


def interpret_kappa_landis_koch(kappa: float) -> str:
    """Classify Cohen's Kappa according to Landis & Koch (1977) thresholds."""
    if kappa < 0.0:
        return "Poor (< 0.00)"
    if kappa <= 0.20:
        return "Slight (0.00 - 0.20)"
    if kappa <= 0.40:
        return "Fair (0.21 - 0.40)"
    if kappa <= 0.60:
        return "Moderate (0.41 - 0.60)"
    if kappa <= 0.80:
        return "Substantial (0.61 - 0.80)"
    return "Almost Perfect (0.81 - 1.00)"


def _normalize_label(raw: Any, synonym_map: dict[str, str], field_name: str) -> str:
    key = str(raw or "").strip().lower()
    if key not in synonym_map:
        raise ValueError(
            f"Unrecognized label {raw!r} for '{field_name}'. "
            f"Allowed values: {sorted(synonym_map.keys())}"
        )
    return synonym_map[key]


def _compute_pair_agreement(
    y1: Sequence[str],
    y2: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    if not y1 or len(y1) != len(y2):
        raise ValueError("Annotation lists must be non-empty and of equal length")

    n = len(y1)
    agreement_rate = sum(int(a == b) for a, b in zip(y1, y2)) / float(n)

    unweighted = float(cohen_kappa_score(y1, y2, labels=list(labels)))
    if math.isnan(unweighted):
        unweighted = 1.0 if agreement_rate == 1.0 else 0.0

    label_to_num = {lbl: idx for idx, lbl in enumerate(labels)}
    n1 = [label_to_num[a] for a in y1]
    n2 = [label_to_num[b] for b in y2]
    weighted = float(
        cohen_kappa_score(n1, n2, labels=list(range(len(labels))), weights="quadratic")
    )
    if math.isnan(weighted):
        weighted = unweighted

    cm = confusion_matrix(y1, y2, labels=list(labels)).tolist()

    return {
        "num_samples": n,
        "labels": list(labels),
        "agreement_rate": agreement_rate,
        "cohen_kappa_unweighted": unweighted,
        "cohen_kappa_quadratic": weighted,
        "landis_koch_band": interpret_kappa_landis_koch(unweighted),
        "confusion_matrix": cm,
    }


def _safe_corr_pair(y_true: Sequence[float], y_pred: Sequence[float]) -> tuple[float, float]:
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.size < 2 or float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return 0.0, 0.0
    pr = float(pearsonr(a, b).statistic)
    sr = float(spearmanr(a, b).statistic)
    return (0.0 if math.isnan(pr) else pr, 0.0 if math.isnan(sr) else sr)


def evaluate_machine_vs_human(
    audit_csv: Path,
    machine_key_csv: Path,
    *,
    strict_adjudication: bool = False,
) -> dict[str, Any]:
    """Compare hidden teacher predictions (`machine_key_csv`) against adjudicated human gold."""
    audit_path = Path(audit_csv)
    key_path = Path(machine_key_csv)
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit CSV not found: {audit_path}")
    if not key_path.is_file():
        raise FileNotFoundError(f"Machine key CSV not found: {key_path}")

    key_by_eid: dict[str, dict[str, str]] = {}
    key_by_sid: dict[str, dict[str, str]] = {}
    with key_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            eid = str(row.get("edge_id") or "").strip()
            sid = str(row.get("sample_id") or "").strip()
            if eid:
                key_by_eid[eid] = row
            if sid:
                key_by_sid[sid] = row

    stance_gold: list[str] = []
    stance_pred: list[str] = []
    stance_gold_num: list[float] = []
    stance_pred_s: list[float] = []

    host_gold_bin: list[str] = []
    host_pred_bin: list[str] = []
    host_gold_num: list[float] = []
    host_pred_h: list[float] = []
    stance_weights: list[float] = []
    host_weights: list[float] = []

    behavior_gold: list[str] = []
    behavior_pred: list[str] = []
    behavior_order = list(DISCOURSE_BEHAVIOR_LABELS)

    matched_counts = {"stance": 0, "hostility": 0, "behavior_5class": 0}
    unadj_counts = {"stance": 0, "hostility": 0, "behavior_5class": 0}

    with audit_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            eid = str(row.get("edge_id") or "").strip()
            sid = str(row.get("sample_id") or "").strip()
            key_row = key_by_eid.get(eid) or key_by_sid.get(sid)
            if key_row is None:
                continue

            # Stance gold: adjudicated column or unanimous dual-annotator consensus
            raw_gs = str(row.get("gold_adjudicated_stance") or "").strip()
            s1 = str(row.get("annotator_1_stance") or "").strip()
            s2 = str(row.get("annotator_2_stance") or "").strip()
            if raw_gs or (s1 and s2):
                matched_counts["stance"] += 1
            if not raw_gs and s1 and s2:
                n1 = _normalize_label(s1, _STANCE_SYNONYMS, "stance")
                n2 = _normalize_label(s2, _STANCE_SYNONYMS, "stance")
                if n1 == n2:
                    raw_gs = n1
                else:
                    unadj_counts["stance"] += 1
            if raw_gs:
                g_lbl = _normalize_label(raw_gs, _STANCE_SYNONYMS, "stance")
                s_uv = float(key_row.get("s_uv", 0.0))
                raw_ps = str(key_row.get("machine_stance_label") or "").strip()
                if raw_ps:
                    p_lbl = _normalize_label(raw_ps, _STANCE_SYNONYMS, "machine_stance_label")
                else:
                    p_lbl = "support" if s_uv >= 0.1 else ("oppose" if s_uv <= -0.1 else "neutral")
                stance_gold.append(g_lbl)
                stance_pred.append(p_lbl)
                stance_gold_num.append(_STANCE_TO_NUM[g_lbl])
                stance_pred_s.append(s_uv)
                try:
                    sw = float(row.get("sampling_weight") or key_row.get("sampling_weight") or 1.0)
                    stance_weights.append(sw if math.isfinite(sw) and sw > 0.0 else 1.0)
                except ValueError:
                    stance_weights.append(1.0)

            # Incivility / hostility gold: adjudicated column or unanimous dual-annotator consensus
            raw_gh = str(row.get("gold_adjudicated_hostility") or "").strip()
            h1 = str(row.get("annotator_1_hostility") or "").strip()
            h2 = str(row.get("annotator_2_hostility") or "").strip()
            if raw_gh or (h1 and h2):
                matched_counts["hostility"] += 1
            if not raw_gh and h1 and h2:
                hn1 = _normalize_label(h1, _HOSTILITY_SYNONYMS, "hostility")
                hn2 = _normalize_label(h2, _HOSTILITY_SYNONYMS, "hostility")
                if hn1 == hn2:
                    raw_gh = hn1
                else:
                    unadj_counts["hostility"] += 1
            if raw_gh:
                gh_lbl = _normalize_label(raw_gh, _HOSTILITY_SYNONYMS, "hostility")
                h_uv = float(key_row.get("h_uv", 0.0))
                host_gold_bin.append("clean" if gh_lbl == "clean" else "uncivil")
                host_pred_bin.append("uncivil" if h_uv >= 0.5 else "clean")
                host_gold_num.append(_HOSTILITY_TO_NUM[gh_lbl])
                host_pred_h.append(h_uv)
                try:
                    hw = float(row.get("sampling_weight") or key_row.get("sampling_weight") or 1.0)
                    host_weights.append(hw if math.isfinite(hw) and hw > 0.0 else 1.0)
                except ValueError:
                    host_weights.append(1.0)

            # 5-class behavior gold vs machine behavior label
            raw_gb = str(row.get("gold_adjudicated_behavior") or "").strip().lower()
            b1 = str(row.get("annotator_1_behavior") or "").strip().lower()
            b2 = str(row.get("annotator_2_behavior") or "").strip().lower()
            nb1 = _LABEL_SYNONYMS.get(b1)
            nb2 = _LABEL_SYNONYMS.get(b2)
            pb_raw = str(key_row.get("machine_behavior_label") or "").strip().lower()
            pb_norm = _LABEL_SYNONYMS.get(pb_raw)
            if (raw_gb in _LABEL_SYNONYMS or (nb1 is not None and nb2 is not None)) and pb_norm is not None:
                matched_counts["behavior_5class"] += 1
            if not raw_gb and nb1 is not None and nb2 is not None:
                if nb1 == nb2:
                    raw_gb = nb1
                elif pb_norm is not None:
                    unadj_counts["behavior_5class"] += 1
            gb_norm = _LABEL_SYNONYMS.get(raw_gb)
            if gb_norm is not None and pb_norm is not None:
                behavior_gold.append(gb_norm)
                behavior_pred.append(pb_norm)

    total_unadj = sum(unadj_counts.values())
    if total_unadj > 0:
        msg = (
            f"Detected {total_unadj} dual-annotator disagreement rows without gold_adjudicated_* labels "
            f"(stance={unadj_counts['stance']}, hostility={unadj_counts['hostility']}, "
            f"behavior_5class={unadj_counts['behavior_5class']}). Adjudicate ties to prevent selection bias."
        )
        if strict_adjudication:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    result: dict[str, Any] = {
        "adjudication_summary": {
            task: {
                "total_matched_rows": matched_counts[task],
                "unadjudicated_disagreements": unadj_counts[task],
                "adjudication_coverage": float(
                    (matched_counts[task] - unadj_counts[task]) / max(1, matched_counts[task])
                ),
            }
            for task in ("stance", "hostility", "behavior_5class")
            if matched_counts[task] > 0
        }
    }
    if stance_gold:
        kappa = float(cohen_kappa_score(stance_gold, stance_pred, labels=list(STANCE_ORDER)))
        if math.isnan(kappa):
            kappa = 1.0 if stance_gold == stance_pred else 0.0
        pr, sr = _safe_corr_pair(stance_gold_num, stance_pred_s)
        sw_arr = np.asarray(stance_weights, dtype=np.float64) if len(stance_weights) == len(stance_gold) else None
        result["stance"] = {
            "num_samples": len(stance_gold),
            "total_matched_rows": matched_counts["stance"],
            "unadjudicated_disagreements": unadj_counts["stance"],
            "adjudication_coverage": float(len(stance_gold) / max(1, matched_counts["stance"])),
            "macro_f1": float(
                f1_score(stance_gold, stance_pred, labels=list(STANCE_ORDER), average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(stance_gold, stance_pred)),
            "weighted_macro_f1": float(
                f1_score(
                    stance_gold,
                    stance_pred,
                    labels=list(STANCE_ORDER),
                    average="macro",
                    sample_weight=sw_arr,
                    zero_division=0,
                )
            ),
            "weighted_accuracy": float(accuracy_score(stance_gold, stance_pred, sample_weight=sw_arr)),
            "cohen_kappa_unweighted": kappa,
            "pearson_r": pr,
            "spearman_r": sr,
            "confusion_matrix": confusion_matrix(
                stance_gold, stance_pred, labels=list(STANCE_ORDER)
            ).tolist(),
        }
    if host_gold_bin:
        h_labels = list(HOSTILITY_ORDER)
        h_kappa = float(cohen_kappa_score(host_gold_bin, host_pred_bin, labels=h_labels))
        if math.isnan(h_kappa):
            h_kappa = 1.0 if host_gold_bin == host_pred_bin else 0.0
        pr_h, sr_h = _safe_corr_pair(host_gold_num, host_pred_h)
        hw_arr = np.asarray(host_weights, dtype=np.float64) if len(host_weights) == len(host_gold_bin) else None
        y_bin_true = np.asarray([1 if lbl == "uncivil" else 0 for lbl in host_gold_bin], dtype=np.int64)
        y_bin_score = np.asarray(host_pred_h, dtype=np.float64)
        roc_auc_v = (
            float(roc_auc_score(y_bin_true, y_bin_score, sample_weight=hw_arr))
            if np.unique(y_bin_true).size >= 2
            else float("nan")
        )
        pr_auc_v = (
            float(average_precision_score(y_bin_true, y_bin_score, sample_weight=hw_arr))
            if int(y_bin_true.sum()) >= 1
            else float("nan")
        )
        result["hostility"] = {
            "num_samples": len(host_gold_bin),
            "total_matched_rows": matched_counts["hostility"],
            "unadjudicated_disagreements": unadj_counts["hostility"],
            "adjudication_coverage": float(len(host_gold_bin) / max(1, matched_counts["hostility"])),
            "macro_f1": float(
                f1_score(host_gold_bin, host_pred_bin, labels=h_labels, average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(host_gold_bin, host_pred_bin)),
            "weighted_macro_f1": float(
                f1_score(
                    host_gold_bin,
                    host_pred_bin,
                    labels=h_labels,
                    average="macro",
                    sample_weight=hw_arr,
                    zero_division=0,
                )
            ),
            "weighted_accuracy": float(accuracy_score(host_gold_bin, host_pred_bin, sample_weight=hw_arr)),
            "roc_auc": roc_auc_v if math.isfinite(roc_auc_v) else None,
            "pr_auc": pr_auc_v if math.isfinite(pr_auc_v) else None,
            "cohen_kappa_unweighted": h_kappa,
            "pearson_r": pr_h,
            "spearman_r": sr_h,
        }
    if behavior_gold:
        b_kappa = float(cohen_kappa_score(behavior_gold, behavior_pred, labels=behavior_order))
        if math.isnan(b_kappa):
            b_kappa = 1.0 if behavior_gold == behavior_pred else 0.0
        result["behavior_5class"] = {
            "num_samples": len(behavior_gold),
            "total_matched_rows": matched_counts["behavior_5class"],
            "unadjudicated_disagreements": unadj_counts["behavior_5class"],
            "adjudication_coverage": float(len(behavior_gold) / max(1, matched_counts["behavior_5class"])),
            "labels": behavior_order,
            "macro_f1": float(
                f1_score(behavior_gold, behavior_pred, labels=behavior_order, average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(behavior_gold, behavior_pred)),
            "cohen_kappa_unweighted": b_kappa,
            "confusion_matrix": confusion_matrix(
                behavior_gold, behavior_pred, labels=behavior_order
            ).tolist(),
        }
    return result


def evaluate_rumoureval_sdqc(
    rumoureval_json: Path,
    machine_predictions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate NLI stance mapping against RumourEval SDQC benchmark JSON."""
    path = Path(rumoureval_json)
    if not path.is_file():
        raise FileNotFoundError(f"RumourEval SDQC JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("items") or payload.get("data") or []
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"Unsupported RumourEval JSON structure in {path}")
    if not items:
        raise ValueError(f"No records found in RumourEval JSON: {path}")

    gold_3cls: list[str] = []
    pred_3cls: list[str] = []
    gold_num: list[float] = []
    pred_num: list[float] = []
    sdqc_counts: dict[str, int] = {"support": 0, "deny": 0, "query": 0, "comment": 0}

    for item in items:
        raw_sdqc = str(item.get("sdqc_label") or item.get("gold_label") or item.get("label") or "").strip().lower()
        if raw_sdqc in sdqc_counts:
            sdqc_counts[raw_sdqc] += 1
        if raw_sdqc not in _SDQC_TO_STANCE:
            continue
        g3 = _SDQC_TO_STANCE[raw_sdqc]
        item_id = str(item.get("edge_id") or item.get("id") or "")
        ext_pred = machine_predictions.get(item_id) if machine_predictions is not None else None
        raw_pred = item.get("pred_label") if ext_pred is None else ext_pred
        s_uv_val = item.get("s_uv")
        if raw_pred is not None and isinstance(raw_pred, (int, float)):
            s_uv_val = float(raw_pred)
            raw_pred = None
        if raw_pred is not None:
            raw_pred_s = str(raw_pred).strip().lower()
            p3 = _SDQC_TO_STANCE.get(raw_pred_s) or _normalize_label(raw_pred_s, _STANCE_SYNONYMS, "pred_label")
        elif s_uv_val is not None:
            sv = float(s_uv_val)
            p3 = "support" if sv >= 0.1 else ("oppose" if sv <= -0.1 else "neutral")
        else:
            continue
        gold_3cls.append(g3)
        pred_3cls.append(p3)
        gold_num.append(_STANCE_TO_NUM[g3])
        pred_num.append(float(s_uv_val) if s_uv_val is not None else _STANCE_TO_NUM[p3])

    if not gold_3cls:
        raise ValueError(f"No comparable SDQC predictions found in {path}")
    kappa = float(cohen_kappa_score(gold_3cls, pred_3cls, labels=list(STANCE_ORDER)))
    if math.isnan(kappa):
        kappa = 1.0 if gold_3cls == pred_3cls else 0.0
    pr, sr = _safe_corr_pair(gold_num, pred_num)
    return {
        "rumoureval_json": str(path),
        "num_samples": len(gold_3cls),
        "sdqc_distribution": sdqc_counts,
        "labels": list(STANCE_ORDER),
        "macro_f1": float(
            f1_score(gold_3cls, pred_3cls, labels=list(STANCE_ORDER), average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(gold_3cls, pred_3cls)),
        "cohen_kappa_unweighted": kappa,
        "pearson_r": pr,
        "spearman_r": sr,
        "confusion_matrix": confusion_matrix(
            gold_3cls, pred_3cls, labels=list(STANCE_ORDER)
        ).tolist(),
    }


def evaluate_audit_csv(
    csv_path: Path,
    *,
    machine_key_csv: Path | None = None,
    strict_adjudication: bool = False,
) -> dict[str, Any]:
    """Read dual-annotated audit CSV and compute Cohen's Kappa (+ optional machine-vs-gold)."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audit CSV not found: {path}")

    stance_1: list[str] = []
    stance_2: list[str] = []
    hostility_1: list[str] = []
    hostility_2: list[str] = []
    behavior_1: list[str] = []
    behavior_2: list[str] = []
    behavior_order = list(DISCOURSE_BEHAVIOR_LABELS)
    unadj_iaa = {"stance": 0, "hostility": 0, "behavior_5class": 0}

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            s1 = str(row.get("annotator_1_stance") or "").strip()
            s2 = str(row.get("annotator_2_stance") or "").strip()
            gs = str(row.get("gold_adjudicated_stance") or "").strip()
            if s1 and s2:
                ns1 = _normalize_label(s1, _STANCE_SYNONYMS, "stance")
                ns2 = _normalize_label(s2, _STANCE_SYNONYMS, "stance")
                stance_1.append(ns1)
                stance_2.append(ns2)
                if ns1 != ns2 and not gs:
                    unadj_iaa["stance"] += 1

            h1 = str(row.get("annotator_1_hostility") or "").strip()
            h2 = str(row.get("annotator_2_hostility") or "").strip()
            gh = str(row.get("gold_adjudicated_hostility") or "").strip()
            if h1 and h2:
                nh1 = _normalize_label(h1, _HOSTILITY_SYNONYMS, "hostility")
                nh2 = _normalize_label(h2, _HOSTILITY_SYNONYMS, "hostility")
                hostility_1.append(nh1)
                hostility_2.append(nh2)
                if nh1 != nh2 and not gh:
                    unadj_iaa["hostility"] += 1

            b1 = _LABEL_SYNONYMS.get(str(row.get("annotator_1_behavior") or "").strip().lower())
            b2 = _LABEL_SYNONYMS.get(str(row.get("annotator_2_behavior") or "").strip().lower())
            gb = str(row.get("gold_adjudicated_behavior") or "").strip().lower()
            if b1 is not None and b2 is not None:
                behavior_1.append(b1)
                behavior_2.append(b2)
                if b1 != b2 and gb not in _LABEL_SYNONYMS:
                    unadj_iaa["behavior_5class"] += 1

    total_unadj_iaa = sum(unadj_iaa.values())
    if total_unadj_iaa > 0:
        msg = (
            f"Detected {total_unadj_iaa} dual-annotator disagreement rows without gold_adjudicated_* labels "
            f"(stance={unadj_iaa['stance']}, hostility={unadj_iaa['hostility']}, "
            f"behavior_5class={unadj_iaa['behavior_5class']}). Adjudicate ties before final reporting."
        )
        if strict_adjudication:
            raise ValueError(msg)
        if machine_key_csv is None:
            warnings.warn(msg, UserWarning, stacklevel=2)

    report: dict[str, Any] = {
        "input_csv": str(path),
        "unadjudicated_disagreements": unadj_iaa,
    }
    if stance_1:
        report["stance"] = _compute_pair_agreement(stance_1, stance_2, STANCE_ORDER)
    if hostility_1:
        report["hostility"] = _compute_pair_agreement(hostility_1, hostility_2, HOSTILITY_ORDER)
    if behavior_1:
        report["behavior_5class"] = _compute_pair_agreement(behavior_1, behavior_2, behavior_order)

    if "stance" not in report and "hostility" not in report and "behavior_5class" not in report:
        raise ValueError(
            f"No completed dual annotations found in '{path}'. "
            "Fill annotator_1_* and annotator_2_* columns before running compute_kappa.py."
        )
    if machine_key_csv is not None:
        report["machine_vs_gold"] = evaluate_machine_vs_human(
            path,
            machine_key_csv,
            strict_adjudication=strict_adjudication,
        )
    return report


_POLAR_GRADE_SYNONYMS: dict[str, int] = {
    "0": 0,
    "consensus": 0,
    "dong_thuan": 0,
    "1": 1,
    "mild": 1,
    "nhe": 1,
    "2": 2,
    "moderate": 2,
    "trung_binh": 2,
    "3": 3,
    "strong": 3,
    "manh": 3,
    "4": 4,
    "extreme": 4,
    "rat_manh": 4,
}


def _parse_polar_grade(raw: Any) -> int:
    key = str(raw or "").strip().lower()
    if key in _POLAR_GRADE_SYNONYMS:
        return _POLAR_GRADE_SYNONYMS[key]
    try:
        val = int(round(float(key)))
        if 0 <= val <= 4:
            return val
    except ValueError:
        pass
    raise ValueError(
        f"Unrecognized thread polarization grade {raw!r}; expected integer in 0..4."
    )


def evaluate_thread_level_audit_csv(
    audit_csv: Path,
    machine_key_csv: Path | None = None,
    *,
    strict_adjudication: bool = False,
) -> dict[str, Any]:
    """Evaluate 50-tree thread-level polarization blind human audit (0-4 Likert scale)."""
    audit_path = Path(audit_csv)
    if not audit_path.is_file():
        raise FileNotFoundError(f"Thread-level audit CSV not found: {audit_path}")

    g1: list[int] = []
    g2: list[int] = []
    gold_by_id: dict[str, float] = {}
    matched_rows = 0
    unadj_rows = 0

    with audit_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            cid = str(row.get("cascade_id") or "").strip()
            sid = str(row.get("sample_id") or "").strip()
            r1 = str(row.get("annotator_1_polar_grade") or "").strip()
            r2 = str(row.get("annotator_2_polar_grade") or "").strip()
            rg = str(row.get("gold_adjudicated_polar_grade") or "").strip()

            v1 = _parse_polar_grade(r1) if r1 else None
            v2 = _parse_polar_grade(r2) if r2 else None
            vg = _parse_polar_grade(rg) if rg else None

            if v1 is not None and v2 is not None:
                g1.append(v1)
                g2.append(v2)
            if vg is not None or (v1 is not None and v2 is not None):
                matched_rows += 1

            resolved_gold: float | None = None
            if vg is not None:
                resolved_gold = float(vg)
            elif v1 is not None and v2 is not None:
                if abs(v1 - v2) <= 1:
                    resolved_gold = 0.5 * float(v1 + v2)
                else:
                    unadj_rows += 1

            if resolved_gold is not None:
                if cid:
                    gold_by_id[cid] = resolved_gold
                if sid:
                    gold_by_id[sid] = resolved_gold

    if unadj_rows > 0:
        msg = (
            f"Detected {unadj_rows} thread-level audit rows with |annotator_1 - annotator_2| >= 2 "
            "without gold_adjudicated_polar_grade. Adjudicate disagreements before final reporting."
        )
        if strict_adjudication:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    if not g1 and not gold_by_id:
        raise ValueError(
            f"No completed thread-level annotations found in '{audit_path}'. "
            "Fill annotator_1_polar_grade / annotator_2_polar_grade (0-4) first."
        )

    report: dict[str, Any] = {
        "input_csv": str(audit_path),
        "total_matched_rows": matched_rows,
        "unadjudicated_disagreements": unadj_rows,
    }
    if g1:
        labels_5 = [0, 1, 2, 3, 4]
        unw = float(cohen_kappa_score(g1, g2, labels=labels_5))
        if math.isnan(unw):
            unw = 1.0 if g1 == g2 else 0.0
        quad = float(cohen_kappa_score(g1, g2, labels=labels_5, weights="quadratic"))
        if math.isnan(quad):
            quad = unw
        pr, sr = _safe_corr_pair(g1, g2)
        report["inter_annotator"] = {
            "num_threads": len(g1),
            "exact_agreement": float(sum(int(a == b) for a, b in zip(g1, g2)) / len(g1)),
            "within_one_agreement": float(sum(int(abs(a - b) <= 1) for a, b in zip(g1, g2)) / len(g1)),
            "cohen_kappa_unweighted": unw,
            "cohen_kappa_quadratic": quad,
            "landis_koch_band": interpret_kappa_landis_koch(quad),
            "pearson_r": pr,
            "spearman_r": sr,
            "mae_grade": float(np.mean(np.abs(np.asarray(g1, dtype=np.float64) - np.asarray(g2, dtype=np.float64)))),
        }

    if machine_key_csv is not None:
        key_path = Path(machine_key_csv)
        if not key_path.is_file():
            raise FileNotFoundError(f"Thread-level machine key CSV not found: {key_path}")
        gold_cont: list[float] = []
        gold_int: list[int] = []
        mach_ypolar: list[float] = []
        mach_grade: list[int] = []
        matched_key_rows: list[tuple[float, dict[str, str]]] = []
        with key_path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                cid = str(row.get("cascade_id") or "").strip()
                sid = str(row.get("sample_id") or "").strip()
                g_val = gold_by_id.get(cid)
                if g_val is None:
                    g_val = gold_by_id.get(sid)
                if g_val is None:
                    continue
                matched_key_rows.append((g_val, row))

        n_m = len(matched_key_rows)
        quintile_grades: list[int] = []
        if n_m > 0:
            order = sorted(
                range(n_m),
                key=lambda i: float(matched_key_rows[i][1].get("y_polar", 0.0)),
            )
            quintile_grades = [0] * n_m
            n_q = min(5, n_m)
            for rank, orig_i in enumerate(order):
                quintile_grades[orig_i] = min(4, int(rank * n_q / n_m)) if n_q > 1 else 0

        for i, (g_val, row) in enumerate(matched_key_rows):
            yp = float(row.get("y_polar", 0.0))
            raw_q = str(row.get("y_polar_quintile_grade") or "").strip()
            raw_g = str(row.get("y_polar_grade") or "").strip()
            if raw_q:
                mg = min(4, max(0, int(math.floor(float(raw_q) + 0.5))))
            elif raw_g:
                mg = min(4, max(0, int(math.floor(float(raw_g) + 0.5))))
            elif n_m >= 5:
                mg = quintile_grades[i]
            else:
                mg = min(4, max(0, int(math.floor(yp * 4.0 + 0.5))))
            gold_cont.append(g_val / 4.0)
            gold_int.append(min(4, max(0, int(math.floor(g_val + 0.5)))))
            mach_ypolar.append(yp)
            mach_grade.append(mg)

        if gold_cont:
            pr_m, sr_m = _safe_corr_pair(gold_cont, mach_ypolar)
            if len(gold_cont) >= 2 and np.std(gold_cont) > 1e-12 and np.std(mach_ypolar) > 1e-12:
                kt = float(kendalltau(gold_cont, mach_ypolar, variant="b").statistic)
                tau_b = 0.0 if math.isnan(kt) else kt
            else:
                tau_b = 0.0
            gold_ge2 = np.asarray([1 if g >= 2 else 0 for g in gold_int], dtype=np.int64)
            roc_auc_ge2: float | None = (
                float(roc_auc_score(gold_ge2, np.asarray(mach_ypolar, dtype=np.float64)))
                if np.unique(gold_ge2).size >= 2
                else None
            )
            quad_m = float(
                cohen_kappa_score(gold_int, mach_grade, labels=[0, 1, 2, 3, 4], weights="quadratic")
            )
            if math.isnan(quad_m):
                quad_m = 1.0 if gold_int == mach_grade else 0.0
            report["machine_vs_gold"] = {
                "num_threads": len(gold_cont),
                "pearson_r": pr_m,
                "spearman_r": sr_m,
                "kendall_tau_b": tau_b,
                "roc_auc_ge2": roc_auc_ge2,
                "cohen_kappa_quadratic": quad_m,
                "landis_koch_band": interpret_kappa_landis_koch(quad_m),
                "mae_norm": float(
                    np.mean(
                        np.abs(np.asarray(gold_cont, dtype=np.float64) - np.asarray(mach_ypolar, dtype=np.float64))
                    )
                ),
            }
    return report


def _append_human_audit_rows(rows: list[str], report: Mapping[str, Any], *, prefix: str = "") -> None:
    p = f"{prefix} " if prefix else ""
    mvg = report.get("machine_vs_gold") if isinstance(report.get("machine_vs_gold"), Mapping) else {}

    if isinstance(report.get("stance"), Mapping):
        st = report["stance"]
        mg_s = mvg.get("stance") if isinstance(mvg.get("stance"), Mapping) else {}
        f1_str = f"{float(mg_s['macro_f1']):.4f}" if "macro_f1" in mg_s else "--"
        r_str = f"{float(mg_s['spearman_r']):.4f}" if "spearman_r" in mg_s else "--"
        rows.append(
            f"{p}Pair Stance (3-cls) & {int(st['num_samples'])} & "
            f"{float(st['cohen_kappa_unweighted']):.4f} & {float(st['cohen_kappa_quadratic']):.4f} & "
            f"{f1_str} & {r_str} \\\\"
        )
    if isinstance(report.get("hostility"), Mapping):
        ho = report["hostility"]
        mg_h = mvg.get("hostility") if isinstance(mvg.get("hostility"), Mapping) else {}
        f1_str = f"{float(mg_h['macro_f1']):.4f}" if "macro_f1" in mg_h else "--"
        r_str = f"{float(mg_h['spearman_r']):.4f}" if "spearman_r" in mg_h else "--"
        rows.append(
            f"{p}Pair Incivility (2-cls) & {int(ho['num_samples'])} & "
            f"{float(ho['cohen_kappa_unweighted']):.4f} & {float(ho['cohen_kappa_quadratic']):.4f} & "
            f"{f1_str} & {r_str} \\\\"
        )
    if isinstance(report.get("rumoureval_sdqc"), Mapping):
        re_m = report["rumoureval_sdqc"]
        rows.append(
            f"RumourEval SDQC & {int(re_m['num_samples'])} & -- & -- & "
            f"{float(re_m['macro_f1']):.4f} & {float(re_m['spearman_r']):.4f} \\\\"
        )
    if isinstance(report.get("inter_annotator"), Mapping):
        ia = report["inter_annotator"]
        r_str = f"{float(mvg['spearman_r']):.4f}" if "spearman_r" in mvg else "--"
        rows.append(
            f"{p}Thread $y_{{\\text{{polar}}}}$ (0--4) & {int(ia['num_threads'])} & "
            f"{float(ia['cohen_kappa_unweighted']):.4f} & {float(ia['cohen_kappa_quadratic']):.4f} & "
            f"-- & {r_str} \\\\"
        )


def format_human_audit_latex(
    report: Mapping[str, Any],
    *,
    caption: str = (
        "Human Reliability Audit and Silver-Label Validity: Inter-Annotator Agreement "
        "(Cohen's $\\kappa$ / Quadratic $\\kappa_w$) and Machine-vs.-Gold Alignment."
    ),
    label: str = "tab:human_audit",
) -> str:
    """Format single or multi-corpus human audit report into an IEEEtran `booktabs` LaTeX table."""
    rows: list[str] = []
    corpora = report.get("corpora")
    if isinstance(corpora, Sequence) and not isinstance(corpora, (str, bytes)):
        for entry in corpora:
            if isinstance(entry, Mapping) and isinstance(entry.get("report"), Mapping):
                _append_human_audit_rows(
                    rows,
                    entry["report"],
                    prefix=str(entry.get("prefix") or ""),
                )
    else:
        _append_human_audit_rows(rows, report)

    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\resizebox{\\columnwidth}{!}{",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Audit Task & $N$ & IAA $\\kappa$ & IAA $\\kappa_w$ & Mach. F1 & Mach. $\\rho$ \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table}",
            "",
        ]
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Compute Cohen's Kappa & Machine-vs-Gold metrics.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--machine-key", type=Path, default=None)
    parser.add_argument("--rumoureval-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-table", type=Path, default=None)
    parser.add_argument("--strict-adjudication", action="store_true")
    args = parser.parse_args()

    resolved_input = args.input
    if resolved_input is None:
        for candidate in (
            Path("data/interim/human_audit_vi_120.csv"),
            Path("data/interim/human_audit_vi_300.csv"),
            Path("data/interim/human_audit_en_300.csv"),
            Path("data/interim/human_audit_300.csv"),
        ):
            if candidate.is_file():
                resolved_input = candidate
                break
        if resolved_input is None:
            resolved_input = Path("data/interim/human_audit_300.csv")

    resolved_key = args.machine_key
    if resolved_key is None:
        default_key = resolved_input.with_name(f"{resolved_input.stem}_machine_key.csv")
        if default_key.is_file():
            resolved_key = default_key

    report: dict[str, Any] = {}
    if args.rumoureval_json is not None and not resolved_input.is_file():
        report["rumoureval_sdqc"] = evaluate_rumoureval_sdqc(args.rumoureval_json)
    else:
        report.update(
            evaluate_audit_csv(
                resolved_input,
                machine_key_csv=resolved_key,
                strict_adjudication=bool(args.strict_adjudication),
            )
        )
        if args.rumoureval_json is not None:
            report["rumoureval_sdqc"] = evaluate_rumoureval_sdqc(args.rumoureval_json)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.output_table is not None:
        args.output_table.parent.mkdir(parents=True, exist_ok=True)
        args.output_table.write_text(
            format_human_audit_latex(report),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

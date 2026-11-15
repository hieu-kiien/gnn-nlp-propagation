"""Compute inter-annotator agreement (Cohen's Kappa) on `human_audit_300.csv`.

Supports stance annotations (`support` / `neutral` / `oppose`, or `1` / `0` / `-1`)
and optional hostility annotations (`clean` / `offensive` / `hate`, or `0` / `1` / `2`),
reporting unweighted and quadratic-weighted Cohen's Kappa, raw agreement rate,
confusion matrices, and Landis & Koch reliability bands.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

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
    "offensive": "offensive",
    "1": "offensive",
    "hate": "hate",
    "2": "hate",
}

STANCE_ORDER: tuple[str, ...] = ("support", "neutral", "oppose")
HOSTILITY_ORDER: tuple[str, ...] = ("clean", "offensive", "hate")

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
_HOSTILITY_TO_NUM: dict[str, float] = {"clean": 0.0, "offensive": 0.5, "hate": 1.0}


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

    with audit_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            eid = str(row.get("edge_id") or "").strip()
            sid = str(row.get("sample_id") or "").strip()
            key_row = key_by_eid.get(eid) or key_by_sid.get(sid)
            if key_row is None:
                continue

            # Stance gold: adjudicated column or unanimous dual-annotator consensus
            raw_gs = str(row.get("gold_adjudicated_stance") or "").strip()
            if not raw_gs:
                s1 = str(row.get("annotator_1_stance") or "").strip()
                s2 = str(row.get("annotator_2_stance") or "").strip()
                if s1 and s2:
                    n1 = _normalize_label(s1, _STANCE_SYNONYMS, "stance")
                    n2 = _normalize_label(s2, _STANCE_SYNONYMS, "stance")
                    if n1 == n2:
                        raw_gs = n1
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

            # Hostility gold: adjudicated column or unanimous dual-annotator consensus
            raw_gh = str(row.get("gold_adjudicated_hostility") or "").strip()
            if not raw_gh:
                h1 = str(row.get("annotator_1_hostility") or "").strip()
                h2 = str(row.get("annotator_2_hostility") or "").strip()
                if h1 and h2:
                    hn1 = _normalize_label(h1, _HOSTILITY_SYNONYMS, "hostility")
                    hn2 = _normalize_label(h2, _HOSTILITY_SYNONYMS, "hostility")
                    if hn1 == hn2:
                        raw_gh = hn1
            if raw_gh:
                gh_lbl = _normalize_label(raw_gh, _HOSTILITY_SYNONYMS, "hostility")
                h_uv = float(key_row.get("h_uv", 0.0))
                host_gold_bin.append("clean" if gh_lbl == "clean" else "hostile")
                host_pred_bin.append("hostile" if h_uv >= 0.5 else "clean")
                host_gold_num.append(_HOSTILITY_TO_NUM[gh_lbl])
                host_pred_h.append(h_uv)

    result: dict[str, Any] = {}
    if stance_gold:
        kappa = float(cohen_kappa_score(stance_gold, stance_pred, labels=list(STANCE_ORDER)))
        if math.isnan(kappa):
            kappa = 1.0 if stance_gold == stance_pred else 0.0
        pr, sr = _safe_corr_pair(stance_gold_num, stance_pred_s)
        result["stance"] = {
            "num_samples": len(stance_gold),
            "macro_f1": float(
                f1_score(stance_gold, stance_pred, labels=list(STANCE_ORDER), average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(stance_gold, stance_pred)),
            "cohen_kappa_unweighted": kappa,
            "pearson_r": pr,
            "spearman_r": sr,
            "confusion_matrix": confusion_matrix(
                stance_gold, stance_pred, labels=list(STANCE_ORDER)
            ).tolist(),
        }
    if host_gold_bin:
        h_kappa = float(cohen_kappa_score(host_gold_bin, host_pred_bin, labels=["clean", "hostile"]))
        if math.isnan(h_kappa):
            h_kappa = 1.0 if host_gold_bin == host_pred_bin else 0.0
        pr_h, sr_h = _safe_corr_pair(host_gold_num, host_pred_h)
        result["hostility"] = {
            "num_samples": len(host_gold_bin),
            "macro_f1": float(
                f1_score(host_gold_bin, host_pred_bin, labels=["clean", "hostile"], average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(host_gold_bin, host_pred_bin)),
            "cohen_kappa_unweighted": h_kappa,
            "pearson_r": pr_h,
            "spearman_r": sr_h,
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
) -> dict[str, Any]:
    """Read dual-annotated audit CSV and compute Cohen's Kappa (+ optional machine-vs-gold)."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audit CSV not found: {path}")

    stance_1: list[str] = []
    stance_2: list[str] = []
    hostility_1: list[str] = []
    hostility_2: list[str] = []

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            s1 = str(row.get("annotator_1_stance") or "").strip()
            s2 = str(row.get("annotator_2_stance") or "").strip()
            if s1 and s2:
                stance_1.append(_normalize_label(s1, _STANCE_SYNONYMS, "stance"))
                stance_2.append(_normalize_label(s2, _STANCE_SYNONYMS, "stance"))

            h1 = str(row.get("annotator_1_hostility") or "").strip()
            h2 = str(row.get("annotator_2_hostility") or "").strip()
            if h1 and h2:
                hostility_1.append(_normalize_label(h1, _HOSTILITY_SYNONYMS, "hostility"))
                hostility_2.append(_normalize_label(h2, _HOSTILITY_SYNONYMS, "hostility"))

    report: dict[str, Any] = {"input_csv": str(path)}
    if stance_1:
        report["stance"] = _compute_pair_agreement(stance_1, stance_2, STANCE_ORDER)
    if hostility_1:
        report["hostility"] = _compute_pair_agreement(hostility_1, hostility_2, HOSTILITY_ORDER)

    if "stance" not in report and "hostility" not in report:
        raise ValueError(
            f"No completed dual annotations found in '{path}'. "
            "Fill annotator_1_* and annotator_2_* columns before running compute_kappa.py."
        )
    if machine_key_csv is not None:
        report["machine_vs_gold"] = evaluate_machine_vs_human(path, machine_key_csv)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Cohen's Kappa & Machine-vs-Gold metrics.")
    parser.add_argument("--input", type=Path, default=Path("data/interim/human_audit_300.csv"))
    parser.add_argument("--machine-key", type=Path, default=None)
    parser.add_argument("--rumoureval-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    report: dict[str, Any] = {}
    if args.rumoureval_json is not None and not args.input.is_file():
        report["rumoureval_sdqc"] = evaluate_rumoureval_sdqc(args.rumoureval_json)
    else:
        report.update(evaluate_audit_csv(args.input, machine_key_csv=args.machine_key))
        if args.rumoureval_json is not None:
            report["rumoureval_sdqc"] = evaluate_rumoureval_sdqc(args.rumoureval_json)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

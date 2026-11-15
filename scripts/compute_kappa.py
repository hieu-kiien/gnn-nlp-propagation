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
from typing import Any, Sequence

from sklearn.metrics import cohen_kappa_score, confusion_matrix

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


def evaluate_audit_csv(csv_path: Path) -> dict[str, Any]:
    """Read dual-annotated audit CSV and compute Cohen's Kappa report."""
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
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Cohen's Kappa on human audit CSV.")
    parser.add_argument("--input", type=Path, default=Path("data/interim/human_audit_300.csv"))
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    report = evaluate_audit_csv(args.input)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

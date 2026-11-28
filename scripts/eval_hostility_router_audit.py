"""Proof-of-concept benchmark of `HostilityGatedTargetRouter` on audited VN pairs.

Evaluates Before (raw symmetric NLI `s_uv`) vs. After (`HostilityGatedTargetRouter`
disambiguating third-party outrage on `h_uv >= threshold`) across:
  1. The combined N=140 VN audited pairs (`pilot_calibration_vi_20` + `human_audit_vi_120`),
     isolating `support+uncivil` (Target-Misattributed Profanity) and `oppose+uncivil` quadrants.
  2. The canonical N=120 VN Test audit (`human_audit_vi_120`), reporting unweighted and
     inverse-propensity `sampling_weight` Accuracy, Macro-F1, Cohen's kappa, and correlations.

Exports strictly aggregate, zero-PII outputs (never raw texts or machine-key IDs):
  - `results/results_hostility_router_audit.json`
  - `results/tables_hostility_router_audit.tex`
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.dual_edge_scorer import (  # noqa: E402
    MDEBERTA_NLI_MODEL_NAME,
    HostilityGatedTargetRouter,
    NLIStanceTeacherScorer,
)

STANCE_ORDER: tuple[str, ...] = ("support", "neutral", "oppose")
STANCE_TO_NUM: dict[str, float] = {"support": 1.0, "neutral": 0.0, "oppose": -1.0}


def _resolve_local_hf_snapshot(model_name: str = MDEBERTA_NLI_MODEL_NAME) -> Path | None:
    if Path(model_name).exists():
        return Path(model_name)
    hf_hub_env = os.environ.get("HF_HUB_CACHE")
    hf_home_env = os.environ.get("HF_HOME")
    if hf_hub_env:
        hf_hub = Path(hf_hub_env)
    elif hf_home_env:
        hf_hub = Path(hf_home_env) / "hub"
    else:
        hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    snap_dir = hf_hub / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if not snap_dir.is_dir():
        return None
    snaps = sorted([p for p in snap_dir.iterdir() if p.is_dir()])
    return snaps[-1] if snaps else None


def _load_joined_audit_pairs(audit_csv: Path, key_csv: Path) -> list[dict[str, Any]]:
    keys_by_sid: dict[str, dict[str, str]] = {}
    keys_by_eid: dict[str, dict[str, str]] = {}
    with key_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            sid = str(row.get("sample_id") or "").strip()
            eid = str(row.get("edge_id") or "").strip()
            if sid:
                keys_by_sid[sid] = row
            if eid:
                keys_by_eid[eid] = row

    joined: list[dict[str, Any]] = []
    with audit_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            sid = str(row.get("sample_id") or "").strip()
            eid = str(row.get("edge_id") or "").strip()
            krow = keys_by_sid.get(sid) or keys_by_eid.get(eid)
            if krow is None:
                continue
            gs = str(row.get("gold_adjudicated_stance") or "").strip().lower()
            gh = str(row.get("gold_adjudicated_hostility") or "").strip().lower()
            if gs not in STANCE_TO_NUM:
                continue
            s_uv = float(krow.get("s_uv", 0.0))
            h_uv = float(krow.get("h_uv", 0.0))
            sw_raw = row.get("sampling_weight") or krow.get("sampling_weight") or "1.0"
            try:
                sw = float(sw_raw)
            except ValueError:
                sw = 1.0
            if not math.isfinite(sw) or sw <= 0.0:
                sw = 1.0
            raw_pred = str(krow.get("machine_stance_label") or "").strip().lower()
            if raw_pred not in STANCE_TO_NUM:
                raw_pred = "support" if s_uv >= 0.1 else ("oppose" if s_uv <= -0.1 else "neutral")
            joined.append(
                {
                    "sample_id": sid,
                    "parent_text": str(row.get("parent_text") or ""),
                    "reply_text": str(row.get("reply_text") or ""),
                    "s_uv": s_uv,
                    "h_uv": h_uv,
                    "pred_label": raw_pred,
                    "gold_stance": gs,
                    "gold_hostility": "uncivil" if gh != "clean" else "clean",
                    "sampling_weight": sw,
                }
            )
    return joined


def _safe_corr(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0, 0.0
    pr = float(pearsonr(x, y).statistic)
    sr = float(spearmanr(x, y).statistic)
    return (0.0 if math.isnan(pr) else pr), (0.0 if math.isnan(sr) else sr)


def _compute_stance_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gold = [str(r["gold_stance"]) for r in rows]
    pred = [str(r["pred_label"]) for r in rows]
    gold_num = [STANCE_TO_NUM[g] for g in gold]
    pred_s = [float(r["s_uv"]) for r in rows]
    weights = np.asarray([float(r.get("sampling_weight", 1.0)) for r in rows], dtype=np.float64)

    kappa = float(cohen_kappa_score(gold, pred, labels=list(STANCE_ORDER)))
    if math.isnan(kappa):
        kappa = 1.0 if gold == pred else 0.0
    pr, sr = _safe_corr(gold_num, pred_s)
    return {
        "num_samples": len(rows),
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(
            f1_score(gold, pred, labels=list(STANCE_ORDER), average="macro", zero_division=0)
        ),
        "weighted_accuracy": float(accuracy_score(gold, pred, sample_weight=weights)),
        "weighted_macro_f1": float(
            f1_score(
                gold,
                pred,
                labels=list(STANCE_ORDER),
                average="macro",
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "cohen_kappa_unweighted": kappa,
        "pearson_r": pr,
        "spearman_r": sr,
        "confusion_matrix": confusion_matrix(gold, pred, labels=list(STANCE_ORDER)).tolist(),
    }


def _compute_hostile_quadrant_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sup_unc = [r for r in rows if r["gold_stance"] == "support" and r["gold_hostility"] == "uncivil"]
    opp_unc = [r for r in rows if r["gold_stance"] == "oppose" and r["gold_hostility"] == "uncivil"]

    def _summarize(sub: Sequence[Mapping[str, Any]], expected: str) -> dict[str, Any]:
        n = len(sub)
        flipped_opp = sum(1 for r in sub if r["pred_label"] == "oppose")
        correct = sum(1 for r in sub if r["pred_label"] == expected)
        mean_s = float(np.mean([float(r["s_uv"]) for r in sub])) if n > 0 else 0.0
        return {
            "count": n,
            "correct_count": correct,
            "accuracy": float(correct / n) if n > 0 else 0.0,
            "oppose_count": flipped_opp,
            "oppose_rate": float(flipped_opp / n) if n > 0 else 0.0,
            "mean_s_uv": mean_s,
        }

    return {
        "total_pairs": len(rows),
        "support_uncivil": _summarize(sup_unc, "support"),
        "oppose_uncivil": _summarize(opp_unc, "oppose"),
        "all_uncivil_count": sum(1 for r in rows if r["gold_hostility"] == "uncivil"),
        "rerouted_third_party_count": sum(
            1 for r in rows if r.get("target_attributed") == "third_party"
        ),
    }


def evaluate_hostility_router(
    *,
    audit_120_csv: Path = Path("data/interim/human_audit_vi_120.csv"),
    key_120_csv: Path = Path("data/interim/human_audit_vi_120_machine_key.csv"),
    pilot_20_csv: Path = Path("data/interim/pilot_calibration_vi_20.csv"),
    key_20_csv: Path = Path("data/interim/pilot_calibration_vi_20_machine_key.csv"),
    hostility_gate_threshold: float = 0.5,
    nli_scorer: Any | None = None,
    output_json: Path | None = Path("results/results_hostility_router_audit.json"),
    output_tex: Path | None = Path("results/tables_hostility_router_audit.tex"),
) -> dict[str, Any]:
    rows_20 = _load_joined_audit_pairs(pilot_20_csv, key_20_csv)
    rows_120 = _load_joined_audit_pairs(audit_120_csv, key_120_csv)
    rows_140 = rows_20 + rows_120

    if nli_scorer is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        snap = _resolve_local_hf_snapshot(MDEBERTA_NLI_MODEL_NAME)
        if snap is None:
            raise FileNotFoundError(
                f"Local HF snapshot not found for {MDEBERTA_NLI_MODEL_NAME}"
            )
        tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(str(snap), local_files_only=True)
        nli_scorer = NLIStanceTeacherScorer(
            model_name_or_path=str(snap),
            model=mdl,
            tokenizer=tok,
            device=device,
            lang="vi",
            use_multi_gpu=False,
        )

    router = HostilityGatedTargetRouter(
        hostility_gate_threshold=hostility_gate_threshold,
        lang="vi",
        nli_scorer=nli_scorer,
    )
    routed_120, stats_120 = router.route_pairs(rows_120)
    routed_140, stats_140 = router.route_pairs(rows_140)

    payload = {
        "is_real_data": True,
        "provenance_note": (
            "Evaluated strictly on audited VN pairs (pilot_20 + test_120 = 140 pairs; "
            "no raw text or machine-key rows exported)."
        ),
        "hostility_gate_threshold": float(hostility_gate_threshold),
        "vn_140_router_stats": stats_140,
        "vn_120_router_stats": stats_120,
        "vn_140_hostile_quadrants": {
            "before": _compute_hostile_quadrant_summary(rows_140),
            "after": _compute_hostile_quadrant_summary(routed_140),
        },
        "vn_120_test_overall": {
            "before": _compute_stance_metrics(rows_120),
            "after": _compute_stance_metrics(routed_120),
        },
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if output_tex is not None:
        qb = payload["vn_140_hostile_quadrants"]["before"]
        qa = payload["vn_140_hostile_quadrants"]["after"]
        tb = payload["vn_120_test_overall"]["before"]
        ta = payload["vn_120_test_overall"]["after"]
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Diagnostic Audit of \texttt{HostilityGatedTargetRouter} ($\tau_h = 0.50$) on Audited Vietnamese Pairs ($N=140$ Hostile Quadrants; $N=120$ Test Overall).}",
            r"\label{tab:hostility_router_audit}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lccccccc}",
            r"\toprule",
            r"Setting & \texttt{Sup+Unc} Acc ($n=7$) & \texttt{Sup+Unc} Oppose Flip & \texttt{Sup+Unc} $\bar{s}_{uv}$ & \texttt{Opp+Unc} Acc ($n=4$) & Test-120 Acc & Test-120 W-Acc & Test-120 $\kappa$ \\",
            r"\midrule",
            (
                f"Baseline (Raw NLI) & {qb['support_uncivil']['correct_count']}/{qb['support_uncivil']['count']} ({100*qb['support_uncivil']['accuracy']:.1f}\\%) "
                f"& {qb['support_uncivil']['oppose_count']}/{qb['support_uncivil']['count']} ({100*qb['support_uncivil']['oppose_rate']:.1f}\\%) "
                f"& {qb['support_uncivil']['mean_s_uv']:+.4f} "
                f"& {qb['oppose_uncivil']['correct_count']}/{qb['oppose_uncivil']['count']} ({100*qb['oppose_uncivil']['accuracy']:.1f}\\%) "
                f"& {100*tb['accuracy']:.2f}\\% & {100*tb['weighted_accuracy']:.2f}\\% & {tb['cohen_kappa_unweighted']:.4f} \\\\"
            ),
            (
                f"+ HostilityTargetRouter & {qa['support_uncivil']['correct_count']}/{qa['support_uncivil']['count']} ({100*qa['support_uncivil']['accuracy']:.1f}\\%) "
                f"& {qa['support_uncivil']['oppose_count']}/{qa['support_uncivil']['count']} ({100*qa['support_uncivil']['oppose_rate']:.1f}\\%) "
                f"& {qa['support_uncivil']['mean_s_uv']:+.4f} "
                f"& {qa['oppose_uncivil']['correct_count']}/{qa['oppose_uncivil']['count']} ({100*qa['oppose_uncivil']['accuracy']:.1f}\\%) "
                f"& {100*ta['accuracy']:.2f}\\% & {100*ta['weighted_accuracy']:.2f}\\% & {ta['cohen_kappa_unweighted']:.4f} \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
        output_tex.parent.mkdir(parents=True, exist_ok=True)
        output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HostilityGatedTargetRouter on audited VN pairs.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path, default=Path("results/results_hostility_router_audit.json"))
    parser.add_argument("--output-tex", type=Path, default=Path("results/tables_hostility_router_audit.tex"))
    args = parser.parse_args()
    res = evaluate_hostility_router(
        hostility_gate_threshold=args.threshold,
        output_json=args.output_json,
        output_tex=args.output_tex,
    )
    print("[HostilityRouterAudit] Done:", json.dumps(res["vn_140_hostile_quadrants"], indent=2))


if __name__ == "__main__":
    main()

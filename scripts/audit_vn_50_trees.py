"""Thread-level 30-tree blind human audit exporter & evaluator for VN Test cascades.

Validates the silver-standard thread polarization label `y_polar` against independent
human expert grading (0 = Consensus, 1 = Mild disagreement, 2 = Moderate split,
3 = Strong polarization, 4 = Extreme bipolar conflict) strictly on held-out Test trees.

Exports:
  - Strictly blind CSV (`data/interim/human_audit_vn_30_trees.csv`): hides `y_polar`,
    `y_bimodality`, and `y_assortativity`.
  - Sealed machine key CSV (`data/interim/human_audit_vn_30_trees_machine_key.csv`):
    automatically excluded from primary bundles and sealed separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

from scripts.compute_kappa import (
    evaluate_thread_level_audit_csv,
    format_human_audit_latex,
)
from src.data.pyg_dataset import DEFAULT_FORECAST_MIN_REPLIES, EarlyCascadeDataset
from src.data.schema import load_cascade_parquet


def _is_visible_reply(row: dict[str, Any]) -> bool:
    txt = str(row.get("text") or "").strip()
    if not txt or txt in {"[deleted]", "[removed]"}:
        return False
    return "[REDACTED" not in txt


def export_vn_50_tree_audit(
    data_dir: Path = Path("data/processed/vn"),
    *,
    output_csv: Path = Path("data/interim/human_audit_vn_30_trees.csv"),
    machine_key_csv: Path | None = None,
    n_samples: int = 30,
    k_horizon: int = 20,
    min_total_replies: int = DEFAULT_FORECAST_MIN_REPLIES,
    polar_target_mode: str = "future_only",
    max_reply_snippets: int = 12,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Export stratified 30-tree blind audit CSV from the held-out Test split."""
    test_ds = EarlyCascadeDataset(
        root_dir=data_dir,
        k_horizon=k_horizon,
        split="test",
        min_total_replies=min_total_replies,
        polar_target_mode=polar_target_mode,
    )
    if len(test_ds) == 0:
        raise ValueError(f"No Test cascades found in '{data_dir}' with min_total_replies={min_total_replies}")

    nodes_tbl, _ = load_cascade_parquet(data_dir, validate=True)
    nodes_by_cid: dict[str, list[dict[str, Any]]] = {}
    for row in nodes_tbl.to_pylist():
        nodes_by_cid.setdefault(str(row["cascade_id"]), []).append(row)

    # Stratify across 5 y_polar quintiles to ensure full dynamic range coverage
    items = list(test_ds)
    items.sort(key=lambda d: (float(d.y_polar.view(-1)[0].item()), str(d.cascade_id)))
    n_bins = min(5, len(items))
    bins: list[list[Any]] = [[] for _ in range(n_bins)]
    for idx, item in enumerate(items):
        b_idx = min(n_bins - 1, int(idx * n_bins / len(items)))
        bins[b_idx].append(item)

    rng = random.Random(seed)
    for b in bins:
        rng.shuffle(b)

    target_n = min(int(n_samples), len(items))
    selected: list[Any] = []
    offsets = [0] * n_bins
    base_quota = target_n // max(1, n_bins)
    for b_i in range(n_bins):
        take = min(base_quota, len(bins[b_i]))
        selected.extend(bins[b_i][:take])
        offsets[b_i] = take

    while len(selected) < target_n:
        added = False
        for b_i in range(n_bins):
            if len(selected) >= target_n:
                break
            if offsets[b_i] < len(bins[b_i]):
                selected.append(bins[b_i][offsets[b_i]])
                offsets[b_i] += 1
                added = True
        if not added:
            break

    rng.shuffle(selected)

    out_path = Path(output_csv)
    key_path = Path(machine_key_csv) if machine_key_csv is not None else out_path.with_name(
        f"{out_path.stem}_machine_key.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    blind_fields = [
        "sample_id",
        "cascade_id",
        "topic_id",
        "target_window",
        "num_replies",
        "num_target_replies",
        "root_text",
        "sample_replies",
        "sample_replies_json",
        "annotator_1_polar_grade",
        "annotator_2_polar_grade",
        "gold_adjudicated_polar_grade",
        "annotator_1_uncertain",
        "annotator_2_uncertain",
        "annotator_1_notes",
        "annotator_2_notes",
        "gold_notes",
    ]
    key_fields = [
        "sample_id",
        "cascade_id",
        "topic_id",
        "y_polar",
        "y_polar_grade",
        "y_polar_quintile_grade",
        "y_polar_linear_grade",
        "y_bimodality",
        "y_assortativity",
    ]

    # Compute dynamic 5-quintile grades over the selected sample without leaking into blind CSV
    sel_order = sorted(
        range(len(selected)),
        key=lambda i: float(selected[i].y_polar.view(-1)[0].item()),
    )
    sel_quintiles = [0] * len(selected)
    if selected:
        n_q = min(5, len(selected))
        for rank, orig_i in enumerate(sel_order):
            sel_quintiles[orig_i] = min(4, int(rank * n_q / len(selected))) if n_q > 1 else 0

    with (
        out_path.open("w", encoding="utf-8-sig", newline="") as f_blind,
        key_path.open("w", encoding="utf-8-sig", newline="") as f_key,
    ):
        w_blind = csv.DictWriter(f_blind, fieldnames=blind_fields)
        w_key = csv.DictWriter(f_key, fieldnames=key_fields)
        w_blind.writeheader()
        w_key.writeheader()

        for idx, item in enumerate(selected, start=1):
            q_grade = sel_quintiles[idx - 1]
            sid = f"tree_{idx:03d}"
            cid = str(item.cascade_id)
            tid = str(item.topic_id)
            c_rows = nodes_by_cid.get(cid, [])
            by_nid = {str(r["node_id"]): r for r in c_rows}
            root_row = next((r for r in c_rows if bool(r.get("is_root"))), c_rows[0] if c_rows else None)
            root_nid = str(root_row["node_id"]) if root_row is not None else ""
            root_txt = str(root_row["text"]) if root_row is not None else ""

            children_adj: dict[str, list[str]] = {}
            for r in c_rows:
                if not bool(r.get("is_root")):
                    pid_s = str(r.get("parent_id") or root_nid)
                    children_adj.setdefault(pid_s, []).append(str(r["node_id"]))
            depth_map: dict[str, int] = {root_nid: 0} if root_nid else {}
            bfs = deque([root_nid] if root_nid else [])
            while bfs:
                cur = bfs.popleft()
                for ch in children_adj.get(cur, ()):
                    if ch not in depth_map:
                        depth_map[ch] = depth_map[cur] + 1
                        bfs.append(ch)

            replies = [r for r in c_rows if not bool(r.get("is_root"))]
            replies.sort(
                key=lambda r: (
                    r["created_at_utc"],
                    depth_map.get(str(r["node_id"]), 999),
                    str(r["node_id"]),
                )
            )

            # Align sampled human-inspection replies strictly with the y_polar target window
            if polar_target_mode == "future_only" and len(replies) > k_horizon:
                target_replies = replies[k_horizon:]
                window_desc = f"future_only (replies #{k_horizon + 1}..#{len(replies)})"
            else:
                target_replies = replies
                window_desc = f"full_tree (replies #1..#{len(replies)})"

            visible_targets = [r for r in target_replies if _is_visible_reply(r)]
            pool_replies = visible_targets if visible_targets else target_replies

            if len(pool_replies) <= max_reply_snippets:
                chosen_replies = pool_replies
            else:
                step = (len(pool_replies) - 1) / float(max(1, max_reply_snippets - 1))
                chosen_replies = [
                    pool_replies[int(round(i * step))] for i in range(max_reply_snippets)
                ]

            formatted_snippets: list[str] = []
            structured_snippets: list[dict[str, Any]] = []
            for pos, r in enumerate(chosen_replies, start=1):
                nid_s = str(r["node_id"])
                depth_v = int(depth_map.get(nid_s, 1))
                pid_raw = r.get("parent_id")
                pid_s = str(pid_raw) if pid_raw is not None else ""
                if not pid_s or pid_s == root_nid:
                    target_kind = "root"
                    parent_snip = ""
                    parent_tag = "-> Root"
                else:
                    target_kind = "parent"
                    p_row = by_nid.get(pid_s)
                    parent_snip = (
                        str(p_row["text"]).replace("\n", " ").strip()[:80]
                        if p_row is not None
                        else pid_s
                    )
                    parent_tag = f"-> Parent(\"{parent_snip}\")"
                rep_txt = str(r["text"]).replace("\n", " ").strip()[:200]
                formatted_snippets.append(f"[#{pos} {parent_tag}] {rep_txt}")
                structured_snippets.append(
                    {
                        "pos": pos,
                        "depth": depth_v,
                        "target_kind": target_kind,
                        "parent_snippet": parent_snip,
                        "text": rep_txt,
                    }
                )
            snippet_str = " || ".join(formatted_snippets)

            w_blind.writerow(
                {
                    "sample_id": sid,
                    "cascade_id": cid,
                    "topic_id": tid,
                    "target_window": window_desc,
                    "num_replies": len(replies),
                    "num_target_replies": len(target_replies),
                    "root_text": root_txt[:600],
                    "sample_replies": snippet_str,
                    "sample_replies_json": json.dumps(structured_snippets, ensure_ascii=False),
                    "annotator_1_polar_grade": "",
                    "annotator_2_polar_grade": "",
                    "gold_adjudicated_polar_grade": "",
                    "annotator_1_uncertain": "",
                    "annotator_2_uncertain": "",
                    "annotator_1_notes": "",
                    "annotator_2_notes": "",
                    "gold_notes": "",
                }
            )
            yp = float(item.y_polar.view(-1)[0].item())
            ybc = float(item.y_bimodality.view(-1)[0].item()) if hasattr(item, "y_bimodality") else 0.0
            yas = float(item.y_assortativity.view(-1)[0].item()) if hasattr(item, "y_assortativity") else 0.0
            w_key.writerow(
                {
                    "sample_id": sid,
                    "cascade_id": cid,
                    "topic_id": tid,
                    "y_polar": f"{yp:.6f}",
                    "y_polar_grade": q_grade,
                    "y_polar_quintile_grade": q_grade,
                    "y_polar_linear_grade": min(4, max(0, int(math.floor(yp * 4.0 + 0.5)))),
                    "y_bimodality": f"{ybc:.6f}",
                    "y_assortativity": f"{yas:.6f}",
                }
            )

    return out_path, key_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Export/evaluate 30-tree blind human audit on VN Test cascades."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/vn"))
    parser.add_argument("--audit-csv", type=Path, default=Path("data/interim/human_audit_vn_30_trees.csv"))
    parser.add_argument("--machine-key-csv", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--k-horizon", type=int, default=20)
    parser.add_argument("--min-total-replies", type=int, default=DEFAULT_FORECAST_MIN_REPLIES)
    parser.add_argument("--polar-target-mode", choices=["future_only", "full"], default="future_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--strict-adjudication", action="store_true")
    parser.add_argument("--allow-single-gold", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-table", type=Path, default=None)
    args = parser.parse_args()

    if not args.export and not args.evaluate:
        raise ValueError("Specify --export and/or --evaluate")

    key_csv = args.machine_key_csv or args.audit_csv.with_name(f"{args.audit_csv.stem}_machine_key.csv")

    if args.export:
        out_csv, out_key = export_vn_50_tree_audit(
            data_dir=args.data_dir,
            output_csv=args.audit_csv,
            machine_key_csv=key_csv,
            n_samples=args.n_samples,
            k_horizon=args.k_horizon,
            min_total_replies=args.min_total_replies,
            polar_target_mode=args.polar_target_mode,
            seed=args.seed,
        )
        print(f"Exported blind {args.n_samples}-tree VN audit CSV: {out_csv} (sealed key: {out_key})")

    if args.evaluate:
        report = evaluate_thread_level_audit_csv(
            args.audit_csv,
            machine_key_csv=key_csv if key_csv.is_file() else None,
            strict_adjudication=bool(args.strict_adjudication),
            allow_single_gold=bool(args.allow_single_gold),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.output_table is not None:
            args.output_table.parent.mkdir(parents=True, exist_ok=True)
            args.output_table.write_text(
                format_human_audit_latex(
                    report,
                    caption="Thread-Level Polarization Human Audit on Held-Out VN Test Cascades (0--4 Likert Scale).",
                    label="tab:human_audit_vn_trees",
                ),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()

"""Local blind Vietnamese Human Audit Studio for Gate 1 CSV annotation.

Runs a strictly local (127.0.0.1) stdlib HTTP server so human annotators can
annotate stance, hostility, and thread-level polarization blind CSVs without
manually editing raw spreadsheets. Enforce strict dual-annotator blinding
(Annotator 1 never sees Annotator 2 or gold columns, and vice versa), rejects
access to any ``*_machine_key.csv`` or ``*.secret`` files, validates labels
against ``scripts/compute_kappa.py``, and writes CSV updates atomically.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import threading
import webbrowser
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

VALID_PAIR_STANCE: frozenset[str] = frozenset({"", "support", "neutral", "oppose"})
VALID_PAIR_HOSTILITY: frozenset[str] = frozenset({"", "clean", "uncivil", "offensive", "hate"})
VALID_TREE_GRADE: frozenset[str] = frozenset({"", "0", "1", "2", "3", "4"})
VALID_ROLES: frozenset[str] = frozenset({"annotator_1", "annotator_2", "gold"})


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata describing an allowlisted blind audit CSV."""

    filename: str
    mode: str  # "pair" | "tree"
    title: str
    subtitle: str
    kappa_command: str


ALLOWLISTED_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        filename="pilot_calibration_vi_20.csv",
        mode="pair",
        title="1. Pilot hiệu chuẩn VN (20 cặp Test)",
        subtitle="Chấm thử 20 cặp tiếng Việt (10 clean / 10 uncivil) để 2 người thống nhất chuẩn trước khi chấm 120 cặp.",
        kappa_command="python -m scripts.compute_kappa --input data/interim/pilot_calibration_vi_20.csv",
    ),
    DatasetSpec(
        filename="human_audit_vi_120.csv",
        mode="pair",
        title="2. Kiểm định chính VN (120 cặp Test)",
        subtitle="Chấm mù độc lập Thái độ (stance) + Khiếm nhã/Công kích nhị phân (clean vs. uncivil) trên tập Test VN.",
        kappa_command=(
            "python -m scripts.compute_kappa --input data/interim/human_audit_vi_120.csv "
            "--strict-adjudication --output-json paper/results_human_audit_vi.json"
        ),
    ),
    DatasetSpec(
        filename="human_audit_vn_30_trees.csv",
        mode="tree",
        title="3. Phân cực cấp cây VN (30 cây Test)",
        subtitle="Chấm mức độ phân cực toàn cửa sổ tương lai theo thang Likert 0–4 trên 30 cây Test phân tầng.",
        kappa_command=(
            "python -m scripts.audit_vn_50_trees --evaluate --strict-adjudication "
            "--audit-csv data/interim/human_audit_vn_30_trees.csv "
            "--output-json paper/results_human_audit_vn_30_trees.json"
        ),
    ),
    DatasetSpec(
        filename="pilot_calibration_vi_30.csv",
        mode="pair",
        title="[Di sản] Pilot VN cũ (30 cặp)",
        subtitle="Bộ pilot cũ 30 cặp (giữ tương thích kiểm thử cũ).",
        kappa_command="python -m scripts.compute_kappa --input data/interim/pilot_calibration_vi_30.csv",
    ),
    DatasetSpec(
        filename="human_audit_vi_300.csv",
        mode="pair",
        title="[Di sản] Kiểm định VN cũ (300 cặp)",
        subtitle="Bộ kiểm định cũ 300 cặp (giữ tương thích kiểm thử cũ).",
        kappa_command=(
            "python -m scripts.compute_kappa --input data/interim/human_audit_vi_300.csv "
            "--strict-adjudication --output-json paper/results_human_audit_vi.json"
        ),
    ),
    DatasetSpec(
        filename="human_audit_vn_50_trees.csv",
        mode="tree",
        title="[Di sản] Phân cực cây VN cũ (50 cây)",
        subtitle="Bộ kiểm định cũ 50 cây (giữ tương thích kiểm thử cũ).",
        kappa_command=(
            "python -m scripts.audit_vn_50_trees --evaluate --strict-adjudication "
            "--audit-csv data/interim/human_audit_vn_50_trees.csv "
            "--output-json paper/results_human_audit_vn_50_trees.json"
        ),
    ),
    DatasetSpec(
        filename="pilot_calibration_en_20.csv",
        mode="pair",
        title="Tùy chọn: Pilot EN PHEME (20 cặp)",
        subtitle="Tùy chọn: 20 cặp tiếng Anh PHEME (EN pair validation ưu tiên dùng RumourEval SDQC làm chuẩn chính).",
        kappa_command="python -m scripts.compute_kappa --input data/interim/pilot_calibration_en_20.csv",
    ),
    DatasetSpec(
        filename="human_audit_en_120.csv",
        mode="pair",
        title="Tùy chọn: Kiểm định EN PHEME (120 cặp)",
        subtitle="Tùy chọn: 120 cặp tiếng Anh PHEME (clean vs. uncivil + 3-class stance).",
        kappa_command=(
            "python -m scripts.compute_kappa --input data/interim/human_audit_en_120.csv "
            "--strict-adjudication --output-json paper/results_human_audit_en.json"
        ),
    ),
    DatasetSpec(
        filename="pilot_calibration_en_30.csv",
        mode="pair",
        title="Tuỳ chọn: Pilot EN PHEME (30 cặp cũ)",
        subtitle="Tuỳ chọn: 30 cặp tiếng Anh PHEME (tương thích ngược).",
        kappa_command="python -m scripts.compute_kappa --input data/interim/pilot_calibration_en_30.csv",
    ),
    DatasetSpec(
        filename="human_audit_en_300.csv",
        mode="pair",
        title="Tuỳ chọn: Kiểm định EN PHEME (300 cặp cũ)",
        subtitle="Tuỳ chọn: 300 cặp tiếng Anh PHEME (tương thích ngược).",
        kappa_command=(
            "python -m scripts.compute_kappa --input data/interim/human_audit_en_300.csv "
            "--strict-adjudication --output-json paper/results_human_audit_en.json"
        ),
    ),
)

ALLOWED_SPEC_MAP: dict[str, DatasetSpec] = {spec.filename: spec for spec in ALLOWLISTED_SPECS}


class ForbiddenDatasetError(PermissionError):
    """Raised when an unallowlisted or sealed-key filename is requested."""


class ValidationError(ValueError):
    """Raised on malformed request payload or disallowed label values."""


def validate_dataset_filename(filename: str) -> DatasetSpec:
    """Strictly validate that *filename* is in the blind CSV allowlist."""
    raw = str(filename or "").strip()
    low = raw.lower()
    if (
        not raw
        or "/" in raw
        or "\\" in raw
        or ".." in raw
        or "machine_key" in low
        or ".secret" in low
        or raw not in ALLOWED_SPEC_MAP
    ):
        raise ForbiddenDatasetError(f"Access denied for dataset {filename!r}.")
    return ALLOWED_SPEC_MAP[raw]


def validate_role(role: str) -> str:
    """Validate annotator role (`annotator_1`, `annotator_2`, or `gold`)."""
    r = str(role or "").strip()
    if r not in VALID_ROLES:
        raise ValidationError(f"Invalid role {role!r}; expected one of {sorted(VALID_ROLES)}.")
    return r


def _is_pair_row_done(row: dict[str, str], prefix: str) -> bool:
    return bool(
        str(row.get(f"{prefix}_stance") or "").strip()
        and str(row.get(f"{prefix}_hostility") or "").strip()
    )


def _pair_disagreement(row: dict[str, str]) -> bool:
    s1 = str(row.get("annotator_1_stance") or "").strip()
    s2 = str(row.get("annotator_2_stance") or "").strip()
    h1 = str(row.get("annotator_1_hostility") or "").strip()
    h2 = str(row.get("annotator_2_hostility") or "").strip()
    stance_diff = bool(s1 and s2 and s1 != s2)
    host_diff = bool(h1 and h2 and h1 != h2)
    return stance_diff or host_diff


def _tree_disagreement(row: dict[str, str]) -> tuple[bool, bool]:
    g1 = str(row.get("annotator_1_polar_grade") or "").strip()
    g2 = str(row.get("annotator_2_polar_grade") or "").strip()
    if not (g1 and g2):
        return False, False
    try:
        diff = abs(int(g1) - int(g2))
    except ValueError:
        return True, True
    return diff >= 1, diff >= 2


def _has_isolated_notes_cols(row: dict[str, str]) -> bool:
    return any(k in row for k in ("annotator_1_notes", "annotator_2_notes", "gold_notes"))


def _read_role_notes(row: dict[str, str], role: str) -> str:
    if _has_isolated_notes_cols(row):
        col = "gold_notes" if role == "gold" else f"{role}_notes"
        return str(row.get(col) or "")
    # Legacy fallback: only expose legacy `notes` to annotator_1/gold, never leak to annotator_2
    if role in {"annotator_1", "gold"}:
        return str(row.get("notes") or "")
    return ""


def _read_role_uncertain(row: dict[str, str], role: str) -> bool:
    col = "gold_uncertain" if role == "gold" else f"{role}_uncertain"
    return str(row.get(col) or "").strip().lower() in {"1", "true", "yes"}


def _sanitize_row_for_role(row: dict[str, str], spec: DatasetSpec, role: str) -> dict[str, Any]:
    """Return a strictly blinded row dictionary for the requested *role*."""
    if spec.mode == "pair":
        base: dict[str, Any] = {
            "sample_id": str(row.get("sample_id") or ""),
            "cascade_id": str(row.get("cascade_id") or ""),
            "edge_id": str(row.get("edge_id") or ""),
            "root_text": str(row.get("root_text") or ""),
            "grandparent_text": str(row.get("grandparent_text") or ""),
            "parent_text": str(row.get("parent_text") or ""),
            "reply_text": str(row.get("reply_text") or ""),
            "notes": _read_role_notes(row, role),
            "uncertain": _read_role_uncertain(row, role),
        }
        if role == "annotator_1":
            base["stance"] = str(row.get("annotator_1_stance") or "").strip()
            base["hostility"] = str(row.get("annotator_1_hostility") or "").strip()
            base["is_done"] = _is_pair_row_done(row, "annotator_1")
        elif role == "annotator_2":
            base["stance"] = str(row.get("annotator_2_stance") or "").strip()
            base["hostility"] = str(row.get("annotator_2_hostility") or "").strip()
            base["is_done"] = _is_pair_row_done(row, "annotator_2")
        else:
            disagree = _pair_disagreement(row)
            base["stance"] = str(row.get("gold_adjudicated_stance") or "").strip()
            base["hostility"] = str(row.get("gold_adjudicated_hostility") or "").strip()
            base["annotator_1_stance"] = str(row.get("annotator_1_stance") or "").strip()
            base["annotator_2_stance"] = str(row.get("annotator_2_stance") or "").strip()
            base["annotator_1_hostility"] = str(row.get("annotator_1_hostility") or "").strip()
            base["annotator_2_hostility"] = str(row.get("annotator_2_hostility") or "").strip()
            base["annotator_1_uncertain"] = _read_role_uncertain(row, "annotator_1")
            base["annotator_2_uncertain"] = _read_role_uncertain(row, "annotator_2")
            base["disagreement"] = disagree
            base["hard_disagreement"] = disagree
            base["is_done"] = bool(base["stance"] and base["hostility"]) or (
                not disagree
                and _is_pair_row_done(row, "annotator_1")
                and _is_pair_row_done(row, "annotator_2")
            )
        return base

    # Thread-level mode
    base = {
        "sample_id": str(row.get("sample_id") or ""),
        "cascade_id": str(row.get("cascade_id") or ""),
        "topic_id": str(row.get("topic_id") or ""),
        "target_window": str(row.get("target_window") or ""),
        "num_replies": str(row.get("num_replies") or ""),
        "num_target_replies": str(row.get("num_target_replies") or ""),
        "root_text": str(row.get("root_text") or ""),
        "sample_replies": str(row.get("sample_replies") or ""),
        "sample_replies_json": str(row.get("sample_replies_json") or ""),
        "notes": _read_role_notes(row, role),
        "uncertain": _read_role_uncertain(row, role),
    }
    if role == "annotator_1":
        val = str(row.get("annotator_1_polar_grade") or "").strip()
        base["polar_grade"] = val
        base["is_done"] = bool(val)
    elif role == "annotator_2":
        val = str(row.get("annotator_2_polar_grade") or "").strip()
        base["polar_grade"] = val
        base["is_done"] = bool(val)
    else:
        disagree, hard = _tree_disagreement(row)
        val = str(row.get("gold_adjudicated_polar_grade") or "").strip()
        a1 = str(row.get("annotator_1_polar_grade") or "").strip()
        a2 = str(row.get("annotator_2_polar_grade") or "").strip()
        base["polar_grade"] = val
        base["annotator_1_polar_grade"] = a1
        base["annotator_2_polar_grade"] = a2
        base["annotator_1_uncertain"] = _read_role_uncertain(row, "annotator_1")
        base["annotator_2_uncertain"] = _read_role_uncertain(row, "annotator_2")
        base["disagreement"] = disagree
        base["hard_disagreement"] = hard
        base["is_done"] = bool(val) or (bool(a1) and bool(a2) and not disagree)
    return base


class AuditStore:
    """Thread-safe atomic CSV store enforcing blind read/write access."""

    def __init__(self, data_dir: Path, *, locked_role: str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.locked_role = validate_role(locked_role) if locked_role is not None else None
        self._lock = threading.Lock()

    def _assert_role_allowed(self, role: str) -> str:
        clean = validate_role(role)
        if self.locked_role is not None and clean != self.locked_role:
            raise ForbiddenDatasetError(
                f"Server is locked to role {self.locked_role!r}; access as {clean!r} is forbidden."
            )
        return clean

    def _read_csv(self, csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(r) for r in reader]
        return fieldnames, rows

    def _write_csv_atomic(
        self,
        csv_path: Path,
        fieldnames: list[str],
        rows: list[dict[str, str]],
    ) -> None:
        bak_path = csv_path.with_name(f"{csv_path.name}.bak")
        if not bak_path.exists():
            shutil.copy2(csv_path, bak_path)

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8-sig",
                newline="",
                dir=csv_path.parent,
                prefix=f".{csv_path.stem}_",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                writer = csv.DictWriter(tmp, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
            os.replace(tmp_path, csv_path)
            tmp_path = None
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _summarize_rows(self, rows: list[dict[str, str]], spec: DatasetSpec) -> dict[str, int]:
        total = len(rows)
        if spec.mode == "pair":
            done_a1 = sum(1 for r in rows if _is_pair_row_done(r, "annotator_1"))
            done_a2 = sum(1 for r in rows if _is_pair_row_done(r, "annotator_2"))
            done_gold = sum(
                1 for r in rows if _is_pair_row_done(r, "gold_adjudicated")
            )
            disagreements = sum(1 for r in rows if _pair_disagreement(r))
            hard_disagreements = disagreements
        else:
            done_a1 = sum(
                1 for r in rows if str(r.get("annotator_1_polar_grade") or "").strip()
            )
            done_a2 = sum(
                1 for r in rows if str(r.get("annotator_2_polar_grade") or "").strip()
            )
            done_gold = sum(
                1 for r in rows if str(r.get("gold_adjudicated_polar_grade") or "").strip()
            )
            disagreements = 0
            hard_disagreements = 0
            for r in rows:
                d, hd = _tree_disagreement(r)
                if d:
                    disagreements += 1
                if hd:
                    hard_disagreements += 1
        return {
            "total": total,
            "done_a1": done_a1,
            "done_a2": done_a2,
            "done_gold": done_gold,
            "disagreements_count": disagreements,
            "hard_disagreements_count": hard_disagreements,
        }

    def list_datasets(self, role: str = "annotator_1") -> list[dict[str, Any]]:
        """List allowlisted datasets with availability and role-specific progress counts."""
        active_role = self.locked_role or validate_role(role)
        out: list[dict[str, Any]] = []
        with self._lock:
            for spec in ALLOWLISTED_SPECS:
                csv_path = self.data_dir / spec.filename
                entry: dict[str, Any] = {
                    "filename": spec.filename,
                    "mode": spec.mode,
                    "title": spec.title,
                    "subtitle": spec.subtitle,
                    "kappa_command": spec.kappa_command,
                    "exists": csv_path.is_file(),
                    "total": 0,
                    "done_a1": 0,
                    "done_a2": 0,
                    "done_gold": 0,
                    "done_for_role": 0,
                    "disagreements_count": 0,
                    "hard_disagreements_count": 0,
                }
                if csv_path.is_file():
                    _, rows = self._read_csv(csv_path)
                    summ = self._summarize_rows(rows, spec)
                    entry.update(summ)
                    if active_role == "annotator_1":
                        entry["done_for_role"] = summ["done_a1"]
                    elif active_role == "annotator_2":
                        entry["done_for_role"] = summ["done_a2"]
                    else:
                        entry["done_for_role"] = summ["done_gold"]
                out.append(entry)
        return out

    def get_dataset(self, filename: str, role: str) -> dict[str, Any]:
        """Load *filename* with strict role-based blinding."""
        spec = validate_dataset_filename(filename)
        clean_role = self._assert_role_allowed(role)
        csv_path = self.data_dir / spec.filename
        if not csv_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        with self._lock:
            _, rows = self._read_csv(csv_path)

        summary = self._summarize_rows(rows, spec)
        sanitized = [_sanitize_row_for_role(r, spec, clean_role) for r in rows]
        if clean_role == "annotator_1":
            done_for_role = summary["done_a1"]
        elif clean_role == "annotator_2":
            done_for_role = summary["done_a2"]
        else:
            done_for_role = summary["done_gold"]

        return {
            "filename": spec.filename,
            "mode": spec.mode,
            "title": spec.title,
            "subtitle": spec.subtitle,
            "kappa_command": spec.kappa_command,
            "role": clean_role,
            "locked_role": self.locked_role,
            "summary": {**summary, "done_for_role": done_for_role},
            "rows": sanitized,
        }

    def save_annotation(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate and atomically persist a single row annotation."""
        if not isinstance(payload, dict):
            raise ValidationError("Request body must be a JSON object.")
        spec = validate_dataset_filename(str(payload.get("file") or ""))
        clean_role = self._assert_role_allowed(str(payload.get("role") or ""))
        sample_id = str(payload.get("sample_id") or "").strip()
        if not sample_id:
            raise ValidationError("Missing sample_id.")

        notes_present = "notes" in payload
        notes_val = str(payload.get("notes") or "") if notes_present else None
        uncertain_present = "uncertain" in payload
        uncertain_val = "1" if bool(payload.get("uncertain")) else ""

        if spec.mode == "pair":
            stance = str(payload.get("stance") or "").strip().lower()
            hostility = str(payload.get("hostility") or "").strip().lower()
            if stance not in VALID_PAIR_STANCE:
                raise ValidationError(
                    f"Invalid stance {stance!r}; allowed: {sorted(VALID_PAIR_STANCE)}."
                )
            if hostility not in VALID_PAIR_HOSTILITY:
                raise ValidationError(
                    f"Invalid hostility {hostility!r}; allowed: {sorted(VALID_PAIR_HOSTILITY)}."
                )
            col_prefix = "gold_adjudicated" if clean_role == "gold" else clean_role
            target_updates = {
                f"{col_prefix}_stance": stance,
                f"{col_prefix}_hostility": hostility,
            }
        else:
            polar_grade = str(payload.get("polar_grade") or "").strip()
            if polar_grade not in VALID_TREE_GRADE:
                raise ValidationError(
                    f"Invalid polar_grade {polar_grade!r}; allowed: {sorted(VALID_TREE_GRADE)}."
                )
            col_name = (
                "gold_adjudicated_polar_grade"
                if clean_role == "gold"
                else f"{clean_role}_polar_grade"
            )
            target_updates = {col_name: polar_grade}

        csv_path = self.data_dir / spec.filename
        if not csv_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        with self._lock:
            fieldnames, rows = self._read_csv(csv_path)
            target_row: dict[str, str] | None = None
            for r in rows:
                if str(r.get("sample_id") or "").strip() == sample_id:
                    target_row = r
                    break
            if target_row is None:
                raise ValidationError(f"Unknown sample_id {sample_id!r} in {spec.filename}.")

            for k, v in target_updates.items():
                if k not in fieldnames:
                    fieldnames.append(k)
                target_row[k] = v
            if notes_present:
                if any(c in fieldnames for c in ("annotator_1_notes", "annotator_2_notes", "gold_notes")):
                    n_col = "gold_notes" if clean_role == "gold" else f"{clean_role}_notes"
                elif clean_role == "annotator_1":
                    n_col = "notes"
                else:
                    n_col = f"{clean_role}_notes"
                if n_col not in fieldnames:
                    fieldnames.append(n_col)
                target_row[n_col] = notes_val or ""
            if uncertain_present:
                u_col = "gold_uncertain" if clean_role == "gold" else f"{clean_role}_uncertain"
                if u_col not in fieldnames:
                    fieldnames.append(u_col)
                target_row[u_col] = uncertain_val

            self._write_csv_atomic(csv_path, fieldnames, rows)
            summary = self._summarize_rows(rows, spec)
            sanitized_row = _sanitize_row_for_role(target_row, spec, clean_role)

        if clean_role == "annotator_1":
            done_for_role = summary["done_a1"]
        elif clean_role == "annotator_2":
            done_for_role = summary["done_a2"]
        else:
            done_for_role = summary["done_gold"]

        return {
            "ok": True,
            "sample_id": sample_id,
            "row": sanitized_row,
            "summary": {**summary, "done_for_role": done_for_role},
        }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SBi-FAGAT — Phòng Chấm Tay Kiểm Định (Gate 1)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      color-scheme: light;
      --bg: #FAFAFA;
      --surface: #FFFFFF;
      --surface-subtle: #F4F4F5;
      --ink: #18181B;
      --ink-soft: #3F3F46;
      --muted: #52525B;
      --border: #E4E4E7;
      --border-strong: #D4D4D8;
      --emerald: #059669;
      --emerald-bg: #ECFDF5;
      --slate: #475569;
      --slate-bg: #F1F5F9;
      --amber: #B45309;
      --amber-bg: #FFFBEB;
      --red: #DC2626;
      --red-bg: #FEF2F2;
      --ring: #18181B;
      --font-sans: 'Be Vietnam Pro', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      --radius: 10px;
    }

    *, *::before, *::after { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: var(--font-sans);
      font-size: 15px;
      line-height: 1.6;
      -webkit-font-smoothing: antialiased;
    }

    .tabular {
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
    }

    .visually-hidden {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }

    [hidden] {
      display: none !important;
    }

    button, select, input {
      font: inherit;
      color: inherit;
    }

    button:focus-visible,
    select:focus-visible,
    input:focus-visible,
    summary:focus-visible {
      outline: 2px solid var(--ring);
      outline-offset: 2px;
    }

    .app-shell {
      max-width: 960px;
      margin: 0 auto;
      padding: 16px 20px 36px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    /* Region 1: Compact Control Bar */
    .topbar {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .topbar-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .brand-mark {
      width: 34px;
      height: 34px;
      border-radius: 8px;
      background: var(--ink);
      color: #FFFFFF;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }

    .brand h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 700;
      letter-spacing: -0.01em;
    }

    .brand-sub {
      font-size: 12.5px;
      color: var(--muted);
    }

    .topbar-controls {
      display: flex;
      flex-wrap: wrap;
      align-items: flex-end;
      gap: 10px;
    }

    .field-stack {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .field-label {
      font-size: 12px;
      font-weight: 600;
      color: var(--muted);
      letter-spacing: 0.01em;
    }

    .select-control {
      min-height: 44px;
      padding: 8px 12px;
      border: 1px solid var(--border-strong);
      border-radius: 8px;
      background: var(--surface);
      cursor: pointer;
      font-weight: 500;
    }

    .btn-outline {
      min-height: 44px;
      padding: 8px 14px;
      border: 1px solid var(--border-strong);
      border-radius: 8px;
      background: var(--surface);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-weight: 600;
      font-size: 13.5px;
      transition: background-color 150ms ease, border-color 150ms ease;
    }

    .btn-outline:hover:not(:disabled) {
      background: var(--surface-subtle);
      border-color: var(--ink-soft);
    }

    .btn-outline:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }

    .progress-strip {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
    }

    .progress-meta {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 12px;
      font-size: 13px;
      color: var(--ink-soft);
    }

    .progress-track {
      flex: 1 1 180px;
      height: 8px;
      border-radius: 999px;
      background: var(--surface-subtle);
      border: 1px solid var(--border);
      overflow: hidden;
    }

    .progress-fill {
      height: 100%;
      width: 0%;
      background: var(--ink);
      transition: width 180ms ease;
    }

    .save-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12.5px;
      font-weight: 600;
      border: 1px solid var(--border);
      background: var(--surface-subtle);
      color: var(--ink-soft);
    }

    .save-pill[data-state="saved"] {
      background: var(--emerald-bg);
      border-color: #A7F3D0;
      color: var(--emerald);
    }

    .save-pill[data-state="saving"] {
      background: var(--amber-bg);
      border-color: #FDE68A;
      color: var(--amber);
    }

    .save-pill[data-state="error"] {
      background: var(--red-bg);
      border-color: #FECACA;
      color: var(--red);
    }

    /* Region 2: Reading Stage */
    .stage {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .stage-header {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }

    .id-badges {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 12.5px;
      font-weight: 600;
      background: var(--surface-subtle);
      border: 1px solid var(--border);
      color: var(--ink-soft);
    }

    .badge-emerald {
      background: var(--emerald-bg);
      border-color: #A7F3D0;
      color: var(--emerald);
    }

    .badge-amber {
      background: var(--amber-bg);
      border-color: #FDE68A;
      color: var(--amber);
    }

    .badge-red {
      background: var(--red-bg);
      border-color: #FECACA;
      color: var(--red);
    }

    .filter-group {
      display: inline-flex;
      gap: 6px;
      align-items: center;
    }

    .chip-btn {
      min-height: 36px;
      padding: 4px 10px;
      border-radius: 6px;
      border: 1px solid var(--border-strong);
      background: var(--surface);
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
    }

    .chip-btn[aria-pressed="true"] {
      background: var(--ink);
      color: #FFFFFF;
      border-color: var(--ink);
    }

    .context-details {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--bg);
      padding: 10px 14px;
    }

    .context-details summary {
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      color: var(--muted);
      list-style: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .context-details summary::-webkit-details-marker { display: none; }

    .text-body {
      margin-top: 8px;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--ink);
      font-size: 15px;
    }

    .context-details .text-body {
      color: var(--ink-soft);
      font-size: 14px;
      max-height: 220px;
      overflow-y: auto;
    }

    .card-box {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
      background: var(--surface-subtle);
    }

    .card-box-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      font-size: 12.5px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
    }

    .reply-focus {
      background: #FFFFFF;
      border: 1.5px solid var(--ink);
      border-left: 5px solid var(--ink);
      border-radius: 8px;
      padding: 16px 18px;
    }

    .reply-focus .text-body {
      font-size: 16.5px;
      font-weight: 500;
      line-height: 1.65;
    }

    .tree-replies-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 360px;
      overflow-y: auto;
      padding-right: 4px;
    }

    .tree-reply-item {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      background: var(--surface);
      font-size: 14px;
    }

    .tree-reply-item[data-target-kind="root"] {
      border-left: 4px solid var(--slate);
    }

    .tree-reply-item[data-target-kind="parent"] {
      border-left: 4px solid var(--amber);
      background: #FFFDF7;
    }

    .tree-reply-tag {
      display: inline-block;
      font-size: 12px;
      font-weight: 600;
      margin-bottom: 4px;
      color: var(--muted);
    }

    .gold-compare {
      border: 1px solid var(--border-strong);
      border-radius: 8px;
      padding: 12px 14px;
      background: var(--amber-bg);
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      justify-content: space-between;
      align-items: center;
      font-size: 13.5px;
    }

    /* Region 3: Decision & Navigation Dock */
    .dock {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .choice-group {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .choice-heading {
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      justify-content: space-between;
      gap: 8px;
    }

    .choice-heading h2 {
      margin: 0;
      font-size: 14.5px;
      font-weight: 700;
    }

    .choice-hint {
      font-size: 12.5px;
      color: var(--muted);
    }

    .choice-grid-3 {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }

    .choice-grid-5 {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
    }

    @media (max-width: 760px) {
      .choice-grid-3, .choice-grid-5 {
        grid-template-columns: 1fr;
      }
    }

    .choice-btn {
      min-height: 60px;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1.5px solid var(--border-strong);
      background: var(--surface);
      text-align: left;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 4px;
      transition: background-color 140ms ease, border-color 140ms ease;
    }

    .choice-btn:hover {
      background: var(--surface-subtle);
      border-color: var(--ink-soft);
    }

    .choice-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      font-weight: 700;
      font-size: 14px;
    }

    .choice-desc {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
    }

    .keycap {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 24px;
      height: 22px;
      padding: 0 6px;
      border-radius: 5px;
      border: 1px solid var(--border-strong);
      background: var(--surface-subtle);
      font-size: 12px;
      font-weight: 700;
      color: var(--ink-soft);
    }

    .choice-btn[aria-pressed="true"][data-tone="emerald"] {
      background: var(--emerald-bg);
      border-color: var(--emerald);
      color: #065F46;
    }

    .choice-btn[aria-pressed="true"][data-tone="slate"] {
      background: var(--slate-bg);
      border-color: var(--slate);
      color: #1E293B;
    }

    .choice-btn[aria-pressed="true"][data-tone="amber"] {
      background: var(--amber-bg);
      border-color: var(--amber);
      color: #78350F;
    }

    .choice-btn[aria-pressed="true"][data-tone="red"] {
      background: var(--red-bg);
      border-color: var(--red);
      color: #7F1D1D;
    }

    .choice-btn[aria-pressed="true"][data-tone="ink"] {
      background: var(--ink);
      border-color: var(--ink);
      color: #FFFFFF;
    }

    .choice-btn[aria-pressed="true"][data-tone="ink"] .choice-desc {
      color: #E4E4E7;
    }

    .notes-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }

    .notes-input {
      flex: 1 1 260px;
      min-height: 44px;
      padding: 8px 12px;
      border: 1px solid var(--border-strong);
      border-radius: 8px;
      background: var(--bg);
    }

    .nav-bar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
    }

    .nav-buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .btn-primary {
      min-height: 44px;
      padding: 8px 16px;
      border: 1px solid var(--ink);
      border-radius: 8px;
      background: var(--ink);
      color: #FFFFFF;
      font-weight: 600;
      font-size: 13.5px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: opacity 140ms ease;
    }

    .btn-primary:hover {
      opacity: 0.9;
    }

    .checkbox-label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 44px;
      cursor: pointer;
      font-size: 13.5px;
      color: var(--ink-soft);
    }

    .checkbox-label input[type="checkbox"] {
      width: 18px;
      height: 18px;
      accent-color: var(--ink);
    }

    .verify-footer {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 12px 16px;
      font-size: 13px;
      color: var(--muted);
    }

    .cmd-box {
      margin-top: 6px;
      padding: 8px 10px;
      border-radius: 6px;
      background: var(--surface-subtle);
      border: 1px solid var(--border);
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--ink);
      overflow-x: auto;
      white-space: nowrap;
    }

    /* Native Modal Dialog following modern-web-guidance */
    dialog {
      max-width: 720px;
      width: calc(100% - 32px);
      max-height: 85vh;
      padding: 0;
      border: 1px solid var(--border-strong);
      border-radius: 12px;
      background: var(--surface);
      color: var(--ink);
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.16);
    }

    dialog[open] {
      display: flex;
      flex-direction: column;
    }

    dialog:modal {
      overscroll-behavior: contain;
    }

    dialog::backdrop {
      overflow: hidden;
      overscroll-behavior: contain;
      background: rgba(9, 9, 11, 0.52);
      backdrop-filter: blur(2px);
    }

    .dialog-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
    }

    .dialog-head h2 {
      margin: 0;
      font-size: 16px;
      font-weight: 700;
    }

    .dialog-body {
      padding: 18px 20px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
      font-size: 14px;
    }

    .guide-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13.5px;
    }

    .guide-table th, .guide-table td {
      border: 1px solid var(--border);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }

    .guide-table th {
      background: var(--surface-subtle);
      font-weight: 700;
    }

    .dialog-foot {
      padding: 12px 20px;
      border-top: 1px solid var(--border);
      display: flex;
      justify-content: flex-end;
    }

    .empty-state {
      padding: 28px 18px;
      text-align: center;
      color: var(--muted);
    }

    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        transition-duration: 0.001ms !important;
        animation-duration: 0.001ms !important;
      }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <!-- Region 1: Header & Progress -->
    <header class="topbar">
      <div class="topbar-row">
        <div class="brand">
          <span class="brand-mark" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="5" r="2.5"></circle>
              <circle cx="5" cy="18" r="2.5"></circle>
              <circle cx="19" cy="18" r="2.5"></circle>
              <line x1="10.7" y1="7.2" x2="6.3" y2="15.8"></line>
              <line x1="13.3" y1="7.2" x2="17.7" y2="15.8"></line>
            </svg>
          </span>
          <div>
            <h1>SBi-FAGAT — Phòng Chấm Tay Kiểm Định (Gate 1)</h1>
            <div class="brand-sub" id="dataset-subtitle">Đang tải cấu hình bộ dữ liệu...</div>
          </div>
        </div>

        <div class="topbar-controls">
          <div class="field-stack">
            <label class="field-label" for="dataset-select">Bộ dữ liệu</label>
            <select id="dataset-select" class="select-control"></select>
          </div>
          <div class="field-stack">
            <label class="field-label" for="role-select">Vai trò (chấm mù)</label>
            <select id="role-select" class="select-control">
              <option value="annotator_1">Người chấm 1</option>
              <option value="annotator_2">Người chấm 2</option>
              <option value="gold">Chốt Gold (chỉ câu lệch)</option>
            </select>
          </div>
          <button type="button" id="open-guide-btn" class="btn-outline">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <circle cx="12" cy="12" r="9"></circle>
              <path d="M9.5 9a2.5 2.5 0 0 1 5 0c0 1.8-2.5 2.2-2.5 4"></path>
              <circle cx="12" cy="17" r="0.6" fill="currentColor"></circle>
            </svg>
            Hướng dẫn &amp; Ví dụ
          </button>
        </div>
      </div>

      <div class="progress-strip">
        <div class="progress-meta">
          <span class="tabular" id="progress-text">0 / 0</span>
          <span class="tabular" id="index-text">Câu 0 / 0</span>
          <span id="disagree-badge" class="badge badge-amber" hidden>Lệch 2 người: 0</span>
        </div>
        <div class="progress-track" aria-hidden="true">
          <div class="progress-fill" id="progress-fill"></div>
        </div>
        <div class="save-pill" id="save-pill" data-state="idle">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
          <span id="save-pill-text">Sẵn sàng</span>
        </div>
      </div>
    </header>

    <!-- Region 2: Reading Stage -->
    <main class="stage" id="reading-stage">
      <div class="stage-header">
        <div class="id-badges">
          <span class="badge tabular" id="badge-sample-id">sample_id</span>
          <span class="badge tabular" id="badge-cascade-id">cascade_id</span>
          <span class="badge badge-emerald" id="badge-row-done" hidden>Đã chấm xong câu này</span>
          <span class="badge badge-amber" id="badge-same-root" hidden>Trả lời trực tiếp Bài gốc</span>
        </div>
        <div class="filter-group" role="group" aria-label="Bộ lọc danh sách câu">
          <button type="button" class="chip-btn" id="filter-all" aria-pressed="true">Tất cả</button>
          <button type="button" class="chip-btn" id="filter-pending" aria-pressed="false">Chưa chấm</button>
          <button type="button" class="chip-btn" id="filter-disagree" aria-pressed="false" hidden>Chỉ câu lệch</button>
        </div>
      </div>

      <div id="stage-content"></div>
    </main>

    <!-- Region 3: Decision & Navigation Dock -->
    <section class="dock" aria-label="Bảng chọn nhãn và điều hướng">
      <div id="gold-banner" class="gold-compare" hidden></div>
      <div id="decision-controls"></div>

      <div class="notes-row">
        <label class="field-label" for="notes-input">Ghi chú (chỉ lưu riêng cho vai trò hiện tại):</label>
        <input
          type="text"
          id="notes-input"
          class="notes-input"
          placeholder="Nhập lý do nếu câu khó/mơ hồ (nhấn Enter sau khi gõ xong tiếng Việt để lưu)..."
        >
        <label class="checkbox-label" style="white-space:nowrap;">
          <input type="checkbox" id="uncertain-checkbox">
          <span>Câu khó / chưa chắc</span>
        </label>
      </div>

      <div class="nav-bar">
        <label class="checkbox-label">
          <input type="checkbox" id="auto-advance-checkbox" checked>
          <span>Tự chuyển câu tiếp theo khi chấm đủ nhãn</span>
        </label>

        <div class="nav-buttons">
          <button type="button" class="btn-outline" id="btn-prev">← Câu trước</button>
          <button type="button" class="btn-outline" id="btn-next-todo">Chưa chấm tiếp theo</button>
          <button type="button" class="btn-primary" id="btn-next">Câu sau →</button>
        </div>
      </div>
    </section>

    <footer class="verify-footer">
      <div><strong>Lệnh kiểm định Kappa sau khi chấm xong tệp hiện tại:</strong> (tệp tự lưu trực tiếp ra CSV và sao lưu `.bak` an toàn)</div>
      <div class="cmd-box" id="kappa-cmd-box">python -m scripts.compute_kappa</div>
    </footer>
  </div>

  <!-- Native Modal Guide Dialog -->
  <dialog id="guide-dialog" closedby="any" aria-labelledby="guide-title">
    <div class="dialog-head">
      <h2 id="guide-title">Quy ước chấm nhanh (ngắn gọn, dễ nhớ)</h2>
      <button type="button" class="btn-outline" id="close-guide-x">Đóng (Esc)</button>
    </div>
    <div class="dialog-body">
      <p style="margin:0;">
        <strong>Nguyên tắc vàng:</strong> Khi chấm cặp bình luận, hãy hỏi xem <em>Bình luận cần chấm</em> đang phản ứng thế nào với <strong>Bình luận cha</strong> ngay phía trên nó (bài gốc chỉ để hiểu ngữ cảnh).
      </p>

      <table class="guide-table">
        <thead>
          <tr>
            <th>Nhóm 1: Thái độ (Stance)</th>
            <th>Ý nghĩa &amp; Ví dụ nhanh</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>[1] Ủng hộ (support)</strong></td>
            <td>Đồng tình, khen ngợi, hùa theo hoặc bổ sung bằng chứng ủng hộ bình luận cha. <em>Ví dụ: "Chuẩn quá b", "100% đúng luôn".</em></td>
          </tr>
          <tr>
            <td><strong>[2] Trung lập / Khác (neutral)</strong></td>
            <td>Hỏi thêm thông tin, gửi link trung lập, bình luận vu vơ hoặc không rõ khen/chê cha. <em>Ví dụ: "Xin nguồn với bạn", "link ở đâu?".</em></td>
          </tr>
          <tr>
            <td><strong>[3] Phản đối (oppose)</strong></td>
            <td>Bác bỏ, không đồng ý, chỉ trích, vặn lại hoặc chê trách ý kiến bình luận cha. <em>Ví dụ: "Vứt vào sọt rác đi", "Nói sai rồi".</em></td>
          </tr>
        </tbody>
      </table>

      <table class="guide-table">
        <thead>
          <tr>
            <th>Nhóm 2: Khiếm nhã / Công kích (Reply Incivility — 2 mức)</th>
            <th>Cây quyết định &amp; Slang mạng VN</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>[Q] Bình thường (clean)</strong></td>
            <td>Khẩu ngữ mạng thông thường, tranh luận gay gắt nhưng nhắm vào <em>luận điểm/sự việc</em>, hoặc chửi thề cảm thán bạn bè không xúc phạm người đối thoại (<em>"vãi", "vl thật", "chuyện hài vcl"</em>).</td>
          </tr>
          <tr>
            <td><strong>[W] Xúc phạm / Công kích (uncivil)</strong></td>
            <td>Mạt sát, xúc phạm nhân phẩm cá nhân (<em>"ngu", "óc chó", "rác rưởi", "đồ súc vật"</em>), mỉa mai ác ý nhắm vào người viết/nhóm người, kỳ thị vùng miền/giới tính hoặc đe dọa.</td>
          </tr>
        </tbody>
      </table>

      <table class="guide-table">
        <thead>
          <tr>
            <th>Thang 30 cây Test (0–4)</th>
            <th>Cách chấm mức độ phân cực toàn đoạn thảo luận</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>0 → 4</strong></td>
            <td><strong>0:</strong> Một chiều/hỏi đáp êm ả · <strong>1:</strong> Lệch ý nhẹ · <strong>2:</strong> Hai luồng ý kiến ôn hòa · <strong>3:</strong> Hai phe tranh cãi rõ nét · <strong>4:</strong> Xung đột lưỡng cực gay gắt, công kích nhau dữ dội.</td>
          </tr>
        </tbody>
      </table>

      <p style="margin:0;color:var(--muted);">
        <strong>Phím tắt bàn phím:</strong> Cặp: <code>1 / 2 / 3</code> chọn Thái độ, <code>Q / W</code> chọn Công kích (nhị phân). Cây: <code>0–4</code> chọn mức. Chuyển câu: <code>←</code> / <code>→</code>.
      </p>
    </div>
    <div class="dialog-foot">
      <form method="dialog">
        <button type="submit" class="btn-primary" value="close" autofocus>Đã hiểu, quay lại chấm</button>
      </form>
    </div>
  </dialog>

  <script>
    const state = {
      datasets: [],
      filename: "pilot_calibration_vi_20.csv",
      role: "annotator_1",
      lockedRole: null,
      mode: "pair",
      rows: [],
      filteredIndices: [],
      filterMode: "all", // "all" | "pending" | "disagree"
      cursor: 0,
      summary: {},
      saving: false,
      saveQueue: Promise.resolve(),
    };

    const datasetSelect = document.getElementById("dataset-select");
    const roleSelect = document.getElementById("role-select");
    const datasetSubtitle = document.getElementById("dataset-subtitle");
    const progressText = document.getElementById("progress-text");
    const indexText = document.getElementById("index-text");
    const progressFill = document.getElementById("progress-fill");
    const disagreeBadge = document.getElementById("disagree-badge");
    const savePill = document.getElementById("save-pill");
    const savePillText = document.getElementById("save-pill-text");
    const stageContent = document.getElementById("stage-content");
    const decisionControls = document.getElementById("decision-controls");
    const goldBanner = document.getElementById("gold-banner");
    const badgeSampleId = document.getElementById("badge-sample-id");
    const badgeCascadeId = document.getElementById("badge-cascade-id");
    const badgeRowDone = document.getElementById("badge-row-done");
    const badgeSameRoot = document.getElementById("badge-same-root");
    const notesInput = document.getElementById("notes-input");
    const uncertainCheckbox = document.getElementById("uncertain-checkbox");
    const autoAdvanceCheckbox = document.getElementById("auto-advance-checkbox");
    const kappaCmdBox = document.getElementById("kappa-cmd-box");
    const filterAllBtn = document.getElementById("filter-all");
    const filterPendingBtn = document.getElementById("filter-pending");
    const filterDisagreeBtn = document.getElementById("filter-disagree");
    const btnPrev = document.getElementById("btn-prev");
    const btnNextTodo = document.getElementById("btn-next-todo");
    const btnNext = document.getElementById("btn-next");
    const guideDialog = document.getElementById("guide-dialog");
    const openGuideBtn = document.getElementById("open-guide-btn");
    const closeGuideX = document.getElementById("close-guide-x");

    function escapeHtml(str) {
      return String(str || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function setSaveStatus(status, label) {
      savePill.dataset.state = status;
      savePillText.textContent = label;
    }

    function computeFilteredIndices() {
      const idxs = [];
      for (let i = 0; i < state.rows.length; i++) {
        const r = state.rows[i];
        if (state.filterMode === "pending" && r.is_done) continue;
        if (state.filterMode === "disagree" && !r.disagreement) continue;
        idxs.push(i);
      }
      state.filteredIndices = idxs;
      if (state.cursor >= idxs.length) {
        state.cursor = Math.max(0, idxs.length - 1);
      }
    }

    function updateFilterButtons() {
      filterDisagreeBtn.hidden = state.role !== "gold";
      if (state.role !== "gold" && state.filterMode === "disagree") {
        state.filterMode = "all";
      }
      filterAllBtn.setAttribute("aria-pressed", state.filterMode === "all" ? "true" : "false");
      filterPendingBtn.setAttribute("aria-pressed", state.filterMode === "pending" ? "true" : "false");
      filterDisagreeBtn.setAttribute("aria-pressed", state.filterMode === "disagree" ? "true" : "false");
    }

    function updateProgressBar() {
      const total = Number(state.summary.total || 0);
      const disCount = Number(state.summary.disagreements_count || 0);
      const isGoldDis = state.role === "gold" && disCount > 0;
      const denom = isGoldDis ? disCount : total;
      const doneRaw = Number(state.summary.done_for_role || 0);
      const done = isGoldDis ? Math.min(doneRaw, denom) : doneRaw;
      const pct = denom > 0 ? Math.min(100, Math.round((done / denom) * 100)) : 0;
      progressText.textContent = isGoldDis
        ? `Chốt Gold: ${done} / ${denom} câu lệch (${pct}%)`
        : `${done} / ${total} (${pct}%)`;
      progressFill.style.width = `${pct}%`;

      const visTotal = state.filteredIndices.length;
      indexText.textContent = visTotal > 0 ? `Câu ${state.cursor + 1} / ${visTotal}` : "0 / 0 câu";

      if (state.role === "gold") {
        disagreeBadge.hidden = false;
        disagreeBadge.textContent = `Lệch 2 người: ${disCount}`;
      } else {
        disagreeBadge.hidden = true;
      }
    }

    function renderTreeReplies(row) {
      if (row && row.sample_replies_json) {
        try {
          const parsed = JSON.parse(row.sample_replies_json);
          if (Array.isArray(parsed) && parsed.length > 0) {
            return parsed.map(item => {
              const kind = item.target_kind === "root" ? "root" : "parent";
              const depthLbl = item.depth ? `Tầng ${item.depth}` : "";
              const targetLbl = kind === "root"
                ? `-> Bài gốc`
                : `-> Cha ("${item.parent_snippet || ""}")`;
              const tag = `#${item.pos || ""} ${depthLbl} ${targetLbl}`.trim();
              return `
                <div class="tree-reply-item" data-target-kind="${kind}">
                  <span class="tree-reply-tag tabular">${escapeHtml(tag)}</span>
                  <div>${escapeHtml(item.text || "")}</div>
                </div>
              `;
            }).join("");
          }
        } catch (_e) { /* fall back to delimited parser */ }
      }
      const parts = String((row && row.sample_replies) || "")
        .split(/\s*\|\|\s*/)
        .map(s => s.trim())
        .filter(Boolean);
      if (!parts.length) {
        return `<div class="empty-state">Không có bình luận mẫu trong cửa sổ này.</div>`;
      }
      return parts.map(part => {
        const tagEnd = part.indexOf("] ");
        const hasTag = part.startsWith("[") && tagEnd > 0;
        const tag = hasTag ? part.slice(0, tagEnd + 1) : "[Bình luận]";
        const body = hasTag ? part.slice(tagEnd + 2) : part;
        const kind = tag.includes("-> Root") ? "root" : "parent";
        return `
          <div class="tree-reply-item" data-target-kind="${kind}">
            <span class="tree-reply-tag tabular">${escapeHtml(tag)}</span>
            <div>${escapeHtml(body)}</div>
          </div>
        `;
      }).join("");
    }

    function currentRow() {
      if (!state.filteredIndices.length) return null;
      return state.rows[state.filteredIndices[state.cursor]] || null;
    }

    function renderCurrentSample() {
      updateFilterButtons();
      updateProgressBar();

      const row = currentRow();
      if (!row) {
        badgeSampleId.textContent = "—";
        badgeCascadeId.textContent = "—";
        badgeRowDone.hidden = true;
        badgeSameRoot.hidden = true;
        goldBanner.hidden = true;
        stageContent.innerHTML = `<div class="empty-state">Không có câu nào khớp bộ lọc hiện tại.</div>`;
        decisionControls.innerHTML = "";
        notesInput.value = "";
        notesInput.disabled = true;
        uncertainCheckbox.checked = false;
        uncertainCheckbox.disabled = true;
        return;
      }

      notesInput.disabled = false;
      uncertainCheckbox.disabled = false;
      badgeSampleId.textContent = row.sample_id;
      badgeCascadeId.textContent = row.cascade_id;
      badgeRowDone.hidden = !row.is_done;
      notesInput.value = row.notes || "";
      uncertainCheckbox.checked = Boolean(row.uncertain);

      if (state.mode === "pair") {
        const rootTrim = String(row.root_text || "").trim();
        const grandTrim = String(row.grandparent_text || "").trim();
        const parentTrim = String(row.parent_text || "").trim();
        const isSameRoot = rootTrim && rootTrim === parentTrim;
        const hasDistinctGrand = grandTrim && grandTrim !== parentTrim && grandTrim !== rootTrim;
        badgeSameRoot.hidden = !isSameRoot;

        stageContent.innerHTML = `
          ${isSameRoot ? "" : `
            <details class="context-details">
              <summary>
                <span>Ngữ cảnh: Bài đăng gốc (bấm để mở/thu gọn)</span>
                <span class="tabular">${rootTrim.length} ký tự</span>
              </summary>
              <div class="text-body">${escapeHtml(row.root_text)}</div>
            </details>
          `}
          ${hasDistinctGrand ? `
            <details class="context-details" open>
              <summary>
                <span>Bình luận ông/bà (ngữ cảnh trước Bình luận cha)</span>
                <span class="tabular">${grandTrim.length} ký tự</span>
              </summary>
              <div class="text-body">${escapeHtml(row.grandparent_text)}</div>
            </details>
          ` : ""}
          <div class="card-box">
            <div class="card-box-title">
              <span>${isSameRoot ? "Bài đăng gốc (= Bình luận cha được trả lời trực tiếp)" : "Bình luận cha đang được trả lời"}</span>
            </div>
            <div class="text-body">${escapeHtml(row.parent_text)}</div>
          </div>
          <div class="reply-focus">
            <div class="card-box-title">
              <span>Bình luận cần chấm (đang trả lời bình luận cha ở trên)</span>
            </div>
            <div class="text-body">${escapeHtml(row.reply_text)}</div>
          </div>
        `;

        if (state.role === "gold") {
          goldBanner.hidden = false;
          const mismatch = row.disagreement ? `<span class="badge badge-red">Cần chốt vì 2 người khác nhau</span>` : `<span class="badge badge-emerald">2 người đã đồng thuận</span>`;
          goldBanner.innerHTML = `
            <div>
              <strong>Người 1:</strong> <code>${escapeHtml(row.annotator_1_stance || "—")}</code> / <code>${escapeHtml(row.annotator_1_hostility || "—")}</code>
              &nbsp;|&nbsp;
              <strong>Người 2:</strong> <code>${escapeHtml(row.annotator_2_stance || "—")}</code> / <code>${escapeHtml(row.annotator_2_hostility || "—")}</code>
            </div>
            ${mismatch}
          `;
        } else {
          goldBanner.hidden = true;
        }

        const curS = row.stance || "";
        const curH = (row.hostility === "offensive" || row.hostility === "hate") ? "uncivil" : (row.hostility || "");
        decisionControls.innerHTML = `
          <div class="choice-group" role="group" aria-label="1. Thái độ đối với bình luận cha">
            <div class="choice-heading">
              <h2>1. Thái độ với bình luận cha (stance)</h2>
              <span class="choice-hint">Chỉ xét phản ứng của bình luận cần chấm đối với bình luận cha</span>
            </div>
            <div class="choice-grid-3">
              <button type="button" class="choice-btn" data-kind="stance" data-value="support" data-tone="emerald" aria-pressed="${curS === "support"}">
                <span class="choice-top"><span>Ủng hộ (support)</span><span class="keycap tabular">1</span></span>
                <span class="choice-desc">Đồng tình, khen ngợi, củng cố ý kiến cha</span>
              </button>
              <button type="button" class="choice-btn" data-kind="stance" data-value="neutral" data-tone="slate" aria-pressed="${curS === "neutral"}">
                <span class="choice-top"><span>Trung lập / Khác (neutral)</span><span class="keycap tabular">2</span></span>
                <span class="choice-desc">Hỏi thêm, đưa tin trung lập hoặc ngoài lề</span>
              </button>
              <button type="button" class="choice-btn" data-kind="stance" data-value="oppose" data-tone="amber" aria-pressed="${curS === "oppose"}">
                <span class="choice-top"><span>Phản đối (oppose)</span><span class="keycap tabular">3</span></span>
                <span class="choice-desc">Không đồng ý, bác bỏ, chê bai hoặc phản biện</span>
              </button>
            </div>
          </div>

          <div class="choice-group" role="group" aria-label="2. Mức độ khiếm nhã công kích">
            <div class="choice-heading">
              <h2>2. Khiếm nhã / Công kích (incivility)</h2>
              <span class="choice-hint">Chửi thề cảm thán/khẩu ngữ không mạt sát ai => Bình thường · Xúc phạm/mỉa mai ác ý => Công kích</span>
            </div>
            <div class="choice-grid-3" style="grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));">
              <button type="button" class="choice-btn" data-kind="hostility" data-value="clean" data-tone="slate" aria-pressed="${curH === "clean"}">
                <span class="choice-top"><span>Bình thường (clean)</span><span class="keycap tabular">Q</span></span>
                <span class="choice-desc">Lịch sự hoặc khẩu ngữ/chửi thề cảm thán, không xúc phạm ai</span>
              </button>
              <button type="button" class="choice-btn" data-kind="hostility" data-value="uncivil" data-tone="red" aria-pressed="${curH === "uncivil"}">
                <span class="choice-top"><span>Xúc phạm / Công kích (uncivil)</span><span class="keycap tabular">W</span></span>
                <span class="choice-desc">Mạt sát cá nhân, mỉa mai cay nghiệt, kỳ thị hoặc đe dọa</span>
              </button>
            </div>
          </div>
        `;
      } else {
        badgeSameRoot.hidden = true;
        stageContent.innerHTML = `
          <div class="card-box">
            <div class="card-box-title">
              <span>Bài đăng gốc (Root)</span>
              <span class="tabular">${escapeHtml(row.target_window)} (${escapeHtml(row.num_target_replies)} replies mục tiêu / ${escapeHtml(row.num_replies)} tổng)</span>
            </div>
            <div class="text-body">${escapeHtml(row.root_text)}</div>
          </div>
          <div class="choice-group">
            <div class="choice-heading">
              <h2>Các bình luận tiêu biểu trong cửa sổ tương lai</h2>
              <span class="choice-hint">Viền cam = trả lời bình luận cha · Viền xám = trả lời bài gốc</span>
            </div>
            <div class="tree-replies-list">${renderTreeReplies(row)}</div>
          </div>
        `;

        if (state.role === "gold") {
          goldBanner.hidden = false;
          const badge = row.hard_disagreement
            ? `<span class="badge badge-red">Lệch mạnh ≥ 2 bậc (bắt buộc chốt)</span>`
            : (row.disagreement ? `<span class="badge badge-amber">Lệch nhẹ 1 bậc</span>` : `<span class="badge badge-emerald">2 người chấm giống nhau</span>`);
          goldBanner.innerHTML = `
            <div>
              <strong>Người 1:</strong> mức <code>${escapeHtml(row.annotator_1_polar_grade || "—")}</code>
              &nbsp;|&nbsp;
              <strong>Người 2:</strong> mức <code>${escapeHtml(row.annotator_2_polar_grade || "—")}</code>
            </div>
            ${badge}
          `;
        } else {
          goldBanner.hidden = true;
        }

        const curG = row.polar_grade || "";
        const levels = [
          ["0", "Đồng thuận", "Một chiều, gần như không xung đột"],
          ["1", "Lệch nhẹ", "Có vài ý khác nhưng ôn hòa"],
          ["2", "Hai chiều vừa", "Có 2 luồng quan điểm rõ"],
          ["3", "Phân cực rõ", "Tranh luận gay gắt giữa 2 phe"],
          ["4", "Cực đoan", "Xung đột lưỡng cực dữ dội"],
        ];
        decisionControls.innerHTML = `
          <div class="choice-group" role="group" aria-label="Mức độ phân cực toàn cây 0 đến 4">
            <div class="choice-heading">
              <h2>Mức độ phân cực của cây thảo luận (0–4)</h2>
              <span class="choice-hint">Đánh giá mức chia rẽ phe đối lập trên đoạn thảo luận phía trên</span>
            </div>
            <div class="choice-grid-5">
              ${levels.map(([val, name, desc]) => `
                <button type="button" class="choice-btn" data-kind="polar_grade" data-value="${val}" data-tone="ink" aria-pressed="${curG === val}">
                  <span class="choice-top"><span>${name}</span><span class="keycap tabular">${val}</span></span>
                  <span class="choice-desc">${desc}</span>
                </button>
              `).join("")}
            </div>
          </div>
        `;
      }

      btnPrev.disabled = state.cursor <= 0;
      btnNext.disabled = state.cursor >= state.filteredIndices.length - 1;
    }

    function persistRow(row, maybeAutoAdvance) {
      const targetFile = state.filename;
      const targetRole = state.role;
      const targetSampleId = row.sample_id;
      const targetMode = state.mode;
      row.notes = notesInput.value;
      row.uncertain = Boolean(uncertainCheckbox.checked);

      const body = {
        file: targetFile,
        role: targetRole,
        sample_id: targetSampleId,
        notes: row.notes,
        uncertain: row.uncertain,
      };
      if (targetMode === "pair") {
        body.stance = row.stance || "";
        body.hostility = row.hostility || "";
      } else {
        body.polar_grade = row.polar_grade || "";
      }

      setSaveStatus("saving", "Đang lưu...");
      state.saving = true;
      state.saveQueue = state.saveQueue.then(async () => {
        try {
          const res = await fetch("/api/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          });
          const data = await res.json();
          if (!res.ok || !data.ok) {
            throw new Error(data.error || `HTTP ${res.status}`);
          }
          if (state.filename !== targetFile || state.role !== targetRole) {
            return;
          }
          const origIdx = state.rows.findIndex(r => r.sample_id === targetSampleId);
          if (origIdx >= 0) {
            state.rows[origIdx] = data.row;
          }
          state.summary = data.summary;
          setSaveStatus("saved", "Đã lưu CSV");

          const becameDone = Boolean(data.row.is_done);
          if (maybeAutoAdvance && becameDone && autoAdvanceCheckbox.checked) {
            if (state.filterMode === "pending") {
              computeFilteredIndices();
            } else if (state.cursor < state.filteredIndices.length - 1) {
              state.cursor += 1;
            }
          }
          renderCurrentSample();
          loadDatasetList(true);
        } catch (err) {
          setSaveStatus("error", `Lỗi lưu: ${err.message}`);
        } finally {
          state.saving = false;
        }
      });
      return state.saveQueue;
    }

    function handleDecisionClick(kind, value) {
      const row = currentRow();
      if (!row) return;
      row[kind] = value;
      const ready = state.mode === "pair"
        ? Boolean(row.stance && row.hostility)
        : Boolean(row.polar_grade);
      persistRow(row, ready);
    }

    function jumpToNextPending() {
      const curOrig = state.filteredIndices[state.cursor] ?? -1;
      for (let i = state.cursor + 1; i < state.filteredIndices.length; i++) {
        if (!state.rows[state.filteredIndices[i]].is_done) {
          state.cursor = i;
          renderCurrentSample();
          return;
        }
      }
      for (let i = 0; i < state.filteredIndices.length; i++) {
        if (!state.rows[state.filteredIndices[i]].is_done && state.filteredIndices[i] !== curOrig) {
          state.cursor = i;
          renderCurrentSample();
          return;
        }
      }
    }

    async function loadDatasetList(preserveOnly = false) {
      const roleParam = encodeURIComponent(state.role);
      const res = await fetch(`/api/datasets?role=${roleParam}`);
      const data = await res.json();
      state.datasets = data.datasets || [];
      if (data.locked_role) {
        state.lockedRole = data.locked_role;
        state.role = data.locked_role;
        roleSelect.value = data.locked_role;
        roleSelect.disabled = true;
      }
      datasetSelect.innerHTML = "";
      for (const ds of state.datasets) {
        const opt = document.createElement("option");
        opt.value = ds.filename;
        const doneN = Number(ds.done_for_role ?? ds.done_a1 ?? 0);
        opt.textContent = ds.exists
          ? `${ds.title} (${doneN}/${ds.total})`
          : `${ds.title} [Chưa có tệp]`;
        opt.disabled = !ds.exists;
        datasetSelect.appendChild(opt);
      }
      if (!preserveOnly) {
        const firstAvail = state.datasets.find(d => d.exists);
        if (firstAvail && (!state.filename || !state.datasets.some(d => d.filename === state.filename && d.exists))) {
          state.filename = firstAvail.filename;
        }
      }
      datasetSelect.value = state.filename;
    }

    async function loadCurrentDataset() {
      setSaveStatus("idle", "Đang tải...");
      const params = new URLSearchParams({ file: state.filename, role: state.role });
      const res = await fetch(`/api/dataset?${params.toString()}`);
      const data = await res.json();
      if (!res.ok) {
        setSaveStatus("error", data.error || `HTTP ${res.status}`);
        stageContent.innerHTML = `<div class="empty-state">${escapeHtml(data.error || "Không tải được dữ liệu.")}</div>`;
        return;
      }
      if (data.locked_role) {
        state.lockedRole = data.locked_role;
        state.role = data.locked_role;
        roleSelect.value = data.locked_role;
        roleSelect.disabled = true;
      }
      state.mode = data.mode;
      state.rows = data.rows || [];
      state.summary = data.summary || {};
      datasetSubtitle.textContent = data.subtitle || "";
      kappaCmdBox.textContent = data.kappa_command || "";
      if (state.role === "gold") {
        state.filterMode = "disagree";
      } else if (state.filterMode === "disagree") {
        state.filterMode = "all";
      }
      computeFilteredIndices();
      const firstTodo = state.filteredIndices.findIndex(idx => !state.rows[idx].is_done);
      state.cursor = firstTodo >= 0 ? firstTodo : 0;
      setSaveStatus("idle", "Sẵn sàng");
      renderCurrentSample();
    }

    decisionControls.addEventListener("click", (event) => {
      const btn = event.target.closest(".choice-btn");
      if (!btn) return;
      handleDecisionClick(btn.dataset.kind, btn.dataset.value);
    });

    datasetSelect.addEventListener("change", () => {
      state.filename = datasetSelect.value;
      loadCurrentDataset();
    });

    roleSelect.addEventListener("change", () => {
      const nextRole = roleSelect.value;
      if (nextRole === "gold" && state.role !== "gold" && !state.lockedRole) {
        const ok = window.confirm(
          "CẢNH BÁO CHỐT GOLD: Chế độ Gold sẽ hiển thị nhãn của cả Người 1 và Người 2. Chỉ chuyển sang chế độ này sau khi hai người đã chấm mù xong. Tiếp tục?"
        );
        if (!ok) {
          roleSelect.value = state.role;
          return;
        }
      }
      state.role = nextRole;
      loadDatasetList(true);
      loadCurrentDataset();
    });

    filterAllBtn.addEventListener("click", () => {
      state.filterMode = "all";
      computeFilteredIndices();
      renderCurrentSample();
    });
    filterPendingBtn.addEventListener("click", () => {
      state.filterMode = "pending";
      computeFilteredIndices();
      renderCurrentSample();
    });
    filterDisagreeBtn.addEventListener("click", () => {
      state.filterMode = "disagree";
      computeFilteredIndices();
      renderCurrentSample();
    });

    btnPrev.addEventListener("click", () => {
      if (state.cursor > 0) {
        state.cursor -= 1;
        renderCurrentSample();
      }
    });
    btnNext.addEventListener("click", () => {
      if (state.cursor < state.filteredIndices.length - 1) {
        state.cursor += 1;
        renderCurrentSample();
      }
    });
    btnNextTodo.addEventListener("click", jumpToNextPending);

    // Mandatory IME-safe Enter handling for notes input (event.isComposing || keyCode === 229)
    notesInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        if (event.isComposing || event.keyCode === 229) {
          return;
        }
        const row = currentRow();
        if (row) persistRow(row, false);
      }
    });
    notesInput.addEventListener("change", () => {
      const row = currentRow();
      if (row) persistRow(row, false);
    });
    uncertainCheckbox.addEventListener("change", () => {
      const row = currentRow();
      if (row) persistRow(row, false);
    });

    document.addEventListener("keydown", (event) => {
      if (guideDialog.open) return;
      const tag = (document.activeElement && document.activeElement.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if (event.isComposing || event.keyCode === 229 || event.ctrlKey || event.metaKey || event.altKey) return;

      const k = event.key.toLowerCase();
      if (state.mode === "pair") {
        if (k === "1") handleDecisionClick("stance", "support");
        else if (k === "2") handleDecisionClick("stance", "neutral");
        else if (k === "3") handleDecisionClick("stance", "oppose");
        else if (k === "q") handleDecisionClick("hostility", "clean");
        else if (k === "w") handleDecisionClick("hostility", "uncivil");
      } else {
        if (["0", "1", "2", "3", "4"].includes(k)) {
          handleDecisionClick("polar_grade", k);
        }
      }
      if (event.key === "ArrowLeft" && state.cursor > 0) {
        state.cursor -= 1;
        renderCurrentSample();
      } else if (event.key === "ArrowRight" && state.cursor < state.filteredIndices.length - 1) {
        state.cursor += 1;
        renderCurrentSample();
      }
    });

    // Native <dialog> modal handling + returnValue reset + mandatory Safari closedBy fallback
    openGuideBtn.addEventListener("click", () => guideDialog.showModal());
    closeGuideX.addEventListener("click", () => guideDialog.close("dismiss"));
    guideDialog.addEventListener("close", () => {
      guideDialog.returnValue = "";
    });
    if (!("closedBy" in HTMLDialogElement.prototype)) {
      let pressedOutside = false;
      const isOutside = (e) => {
        const r = guideDialog.getBoundingClientRect();
        return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
      };
      guideDialog.addEventListener("pointerdown", (e) => {
        pressedOutside = isOutside(e);
      });
      guideDialog.addEventListener("pointerup", (e) => {
        if (pressedOutside && isOutside(e)) {
          guideDialog.close("backdrop");
        }
        pressedOutside = false;
      });
    }

    (async () => {
      await loadDatasetList();
      await loadCurrentDataset();
    })();
  </script>
</body>
</html>
"""


def make_handler(store: AuditStore):
    """Create a request handler class bound to *store*."""

    class AuditRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover
            pass

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _send_html(self, status: int, html: str) -> None:
            raw = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(HTTPStatus.OK, HTML_PAGE)
                return
            if parsed.path == "/api/datasets":
                qs = parse_qs(parsed.query)
                req_role = (qs.get("role") or [store.locked_role or "annotator_1"])[0]
                try:
                    datasets = store.list_datasets(role=req_role)
                except ValidationError:
                    datasets = store.list_datasets()
                self._send_json(
                    HTTPStatus.OK,
                    {"datasets": datasets, "locked_role": store.locked_role},
                )
                return
            if parsed.path == "/api/dataset":
                qs = parse_qs(parsed.query)
                filename = (qs.get("file") or [""])[0]
                role = (qs.get("role") or ["annotator_1"])[0]
                try:
                    data = store.get_dataset(filename, role)
                except ForbiddenDatasetError as exc:
                    self._send_json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
                    return
                except ValidationError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                except FileNotFoundError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.OK, data)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/save":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))
                result = store.save_annotation(payload)
            except ForbiddenDatasetError as exc:
                self._send_json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
                return
            except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, result)

    return AuditRequestHandler


def create_server(
    data_dir: Path = Path("data/interim"),
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    locked_role: str | None = None,
) -> ThreadingHTTPServer:
    """Create a local-only HTTP server for the Gate 1 Human Audit Studio."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(f"Refusing non-local bind host {host!r}; use 127.0.0.1.")
    store = AuditStore(Path(data_dir), locked_role=locked_role)
    handler_cls = make_handler(store)
    return ThreadingHTTPServer((host, int(port)), handler_cls)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local blind Vietnamese UI for Gate 1 human audit.")
    parser.add_argument("--host", default="127.0.0.1", help="Local host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/interim"), help="Directory containing blind CSVs")
    parser.add_argument(
        "--lock-role",
        choices=["annotator_1", "annotator_2", "gold"],
        default=None,
        help="Optional server-side role lock to enforce strict single-annotator blinding.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser tab")
    args = parser.parse_args()

    server = create_server(args.data_dir, host=args.host, port=args.port, locked_role=args.lock_role)
    actual_port = server.server_address[1]
    url = f"http://{args.host}:{actual_port}/"
    print(f"[HumanAuditUI] Running at {url} (data_dir={args.data_dir})", flush=True)
    print("[HumanAuditUI] Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[HumanAuditUI] Shutting down cleanly.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

# SBi-FAGAT: Prioritized Technical Roadmap & Action Plan (P0 – P4)

This document formalizes the prioritized technical roadmap for the SBi-FAGAT research codebase.

---

## Priority Overview

| Priority Level | Category | Focus Area | Status |
| :--- | :--- | :--- | :--- |
| **P0 (Critical)** | Scientific Integrity & Safeguards | Prevent misleading inference, enforce zero-leakage, guard untrained checkpoints | **COMPLETED** |
| **P1 (High)** | Data Governance & Privacy | Pseudonymization standards, salt enforcement, dataset split methodology documentation | **COMPLETED** |
| **P2 (Medium)** | Pipeline Versioning & Tooling | Explicit v1 vs. v2 execution profiles, Windows dependency hardening (`tzdata`) | **COMPLETED** |
| **P3 (Planned)** | Model Training & Benchmark Execution | Run feature extraction, train 8 models across 5 seeds on Kaggle/GPU cluster | **PLANNED / REQUIRES GPU** |
| **P4 (Future)** | Paper Publication & Deliverables | Generate IEEE/Springer LaTeX camera-ready tables, export publication figures | **PLANNED** |

---

## Detailed Task Breakdown

### P0-1: Live Predictor Untrained Model Guard
- **Issue:** `scripts/predict_live_thread.py` allowed running inference on real live URLs using randomly initialized weights without warning or restriction if a checkpoint path was omitted. This violated research integrity standards.
- **Root Cause:** Script defaulted to `SBiFAGAT(...)` initialization without checking if weights were trained.
- **Affected File:** `scripts/predict_live_thread.py`, `tests/test_live_predictor_audit.py`.
- **Resolution:**
  - Added strict check: running on live URLs without a valid checkpoint raises `RuntimeError` unless `--allow-untrained` is explicitly passed.
  - Added `--demo-offline` mode for offline CPU testing.
  - Added corrupt checkpoint validation (`try/except` on `torch.load` and `model_state_dict` inspection).
- **Acceptance Criteria:** `pytest tests/test_live_predictor_audit.py` passes 5/5 tests; unauthorized execution raises descriptive error.

---

### P0-2: Math & Literature Specification Reconciliation
- **Issue:** `docs/math_spec.md` contained a broken absolute local path (`file:///d:/1.%20LapTrinh/...`), and claimed that Random Walk Controversy (RWC) and Modularity were used for polarization computation, whereas the code implements Sarle's Bimodality Coefficient and Signed Newman Assortativity.
- **Root Cause:** Theoretical early design draft was not reconciled after empirical implementation decisions were made.
- **Affected File:** `docs/math_spec.md`.
- **Resolution:**
  - Fixed link to relative `literature_review.md`.
  - Added explicit remark explaining why RWC and Modularity were empirically analyzed and rejected for sparse cascades ($K=20$) in favor of Bimodality Coefficient and Signed Assortativity.
  - Formally defined $y_{\text{polar}}$ as a silver-standard label derived from stance distributions.
  - Clarified Vietnamese inductive split as *Heuristic Temporal-Keyword Grouped Inductive Split*.
- **Acceptance Criteria:** Documentation accurately matches implementation in `src/data/schema.py` and `src/models/sbi_fagat.py`.

---

### P1-1: Author Privacy & Pseudonymization Hardening
- **Issue:** `src/data/arctic_shift_vn.py` referred to author hashing as "anonymization" without enforcing a cryptographic salt. SHA-256 without salt is vulnerable to precomputation and rainbow-table inversion on common usernames.
- **Root Cause:** Initial implementation used `hashlib.sha256(author.encode()).hexdigest()[:16]` directly.
- **Affected File:** `src/data/arctic_shift_vn.py`.
- **Resolution:**
  - Changed terminology from "anonymization" to "pseudonymization".
  - Implemented salt retrieval from `os.environ.get("SBIF_HASH_SALT")` or `data/interim/.author_hash_salt.secret`.
  - Emitted `UserWarning` when unsalted hashing is used, advising the user of re-identification risks.
- **Acceptance Criteria:** `tests/test_schema.py` and `tests/test_pipeline_and_viz.py` pass; warning correctly logged if no salt is present.

---

### P2-1: Pipeline Execution Profiling (v1 vs. v2)
- **Issue:** `run_all.sh` and `scripts/run_phase2.sh` lacked explicit controls for v1 (legacy NLI, 2D edges) vs. v2 (Qwen QLoRA, 4D edges), risking profile mismatch between feature extraction and model training.
- **Root Cause:** v2 features were added without an environment variable or flag gate in bash scripts.
- **Affected Files:** `run_all.sh`, `scripts/run_phase2.sh`.
- **Resolution:**
  - Introduced `PIPELINE_VERSION="${PIPELINE_VERSION:-v1}"` in both shell scripts.
  - In `run_phase2.sh`: default `v1` passes `--teacher-backend legacy_nli --edge-dim 2`; `v2` passes `--teacher-backend qwen_qlora --edge-dim 4`.
  - In `run_all.sh`: default `v1` runs standard pipeline with `--edge-dim 2`; `v2` adds `--v2 --edge-dim 4`.
- **Acceptance Criteria:** Both scripts execute with predictable default behavior and can be toggled seamlessly via `export PIPELINE_VERSION=v2`.

---

### P2-2: Cross-Platform Dependency Hardening (`tzdata`)
- **Issue:** On Windows platforms, `pyarrow` fails when parsing timestamps with `"UTC"` timezones if `tzdata` is not installed.
- **Root Cause:** Python standard library `zoneinfo` relies on system IANA timezone databases, which are not bundled with Windows OS.
- **Affected File:** `requirements.txt`.
- **Resolution:**
  - Added `tzdata>=2024.1; sys_platform == 'win32'` to `requirements.txt`.
  - Installed `tzdata` in `.venv`.
- **Acceptance Criteria:** Unit tests involving PyArrow parquet serialization run on Windows without `ArrowInvalid` errors.

---

### P3: Official Model Training & Benchmark Execution (Upcoming)
- **Objective:** Ingest full PHEME (5,802 cascades) and Reddit VN datasets, extract NLP embeddings via `run_phase2.sh`, and execute official training across 5 random seeds: `[42, 100, 2024, 777, 999]`.
- **Prerequisites:** GPU compute access (Colab T4 or Kaggle T4x2) with user approval.
- **Estimated Resource Cost:** ~6-8 GPU hours on Kaggle dual T4.
- **Deliverables:** `checkpoints/` containing model weights, `results/` containing CSV metrics and bootstrap statistical significance values.

---

### P4: Paper Deliverables & Artifact Compilation (Upcoming)
- **Objective:** Generate IEEE/Springer publication-ready artifacts.
- **Deliverables:**
  - LaTeX tables for Table 1 (Main Benchmark), Table 2 (Ablation on $K$), Table 3 (Inference Latency & Efficiency).
  - High-resolution vector figures (`paper/figures/fagat_attention_dist.pdf`, `paper/figures/cascade_umap.pdf`).
  - Verified BibTeX citations in `paper/references.bib`.

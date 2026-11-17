# SBi-FAGAT: Project Changelog & Audit Modification Record

All notable modifications, hardening measures, and test additions are documented in this file.

---

## [Uncommitted Audit Release] - Technical Lead & Methodology Verification Cycle

### Added
- **`tests/test_live_predictor_audit.py`**:
  - Implemented 5 dedicated unit tests verifying the newly introduced safety guards on `scripts/predict_live_thread.py`.
  - Tests verify: blocking of uncheckpointed live inference, explicit `--allow-untrained` bypass, corrupt checkpoint handling, `--demo-offline` functionality, and empty thread error reporting.
- **`docs/project-management/`**:
  - `PROJECT_CONTEXT.md`: Problem statement, architecture, data contract, v1 vs. v2 profiles.
  - `DEVELOPMENT_HISTORY.md`: Commit log analysis, historical progression from v1 to v2.
  - `PROJECT_STATUS.md`: Module implementation and verification matrix.
  - `TODO_ROADMAP.md`: Prioritized P0–P4 action items with root cause analysis and acceptance criteria.
  - `TECHNICAL_DECISIONS.md`: Architecture Decision Records (ADR-01 through ADR-06).
  - `TEST_REPORT.md`: Comprehensive test report documenting 79/79 passing unit tests and smoke tests.
  - `CHANGELOG.md`: Detailed log of all code and documentation modifications.
- **Environment & Dependencies**:
  - Added `tzdata>=2024.1; sys_platform == 'win32'` to `requirements.txt` to eliminate Windows PyArrow UTC parsing exceptions.
  - Set up isolated virtual environment `.venv` with Python 3.14, PyTorch 2.15, and PyG 2.8.1.

### Changed & Hardened
- **`scripts/predict_live_thread.py`**:
  - Added safety guard blocking live thread prediction when no trained `--checkpoint` is provided (raises `RuntimeError`).
  - Added `--allow-untrained` flag to allow intentional structural or dry-run checks with random weights.
  - Added checkpoint file inspection and corruption handling (`try/except` on `torch.load` and validation of `model_state_dict`).
  - Enhanced terminal reporting banner to prominently state when untuned structural check weights are being used.
- **`src/data/arctic_shift_vn.py`**:
  - Changed author identity handling from "anonymization" to "pseudonymization" across docstrings and code comments.
  - Added support for cryptographic salt via `os.environ.get("SBIF_HASH_SALT")` or `data/interim/.author_hash_salt.secret`.
  - Added `warnings.warn` fallback notifying the user when unsalted hashing is active.
- **`docs/math_spec.md`**:
  - Fixed broken local filesystem link (`file:///d:/1.%20LapTrinh/...` replaced with relative link to `literature_review.md`).
  - Documented empirical rejection of Random Walk Controversy (RWC) and Modularity ($Q$) on sparse early cascades ($K=20$) in favor of Sarle's Bimodality Coefficient and Signed Newman Assortativity.
  - Formally specified $y_{\text{polar}}$ as a silver-standard model-derived label rather than human ground truth.
  - Defined the Reddit Vietnam split as a *Heuristic Temporal-Keyword Grouped Inductive Split*.
- **`scripts/run_phase2.sh`**:
  - Added environment variable `PIPELINE_VERSION="${PIPELINE_VERSION:-v1}"`.
  - Default `v1` passes `--teacher-backend legacy_nli --edge-dim 2`.
  - Profile `v2` passes `--teacher-backend qwen_qlora --edge-dim 4`.
- **`run_all.sh`**:
  - Added environment variable `PIPELINE_VERSION="${PIPELINE_VERSION:-v1}"`.
  - Profile `v1` runs standard pipeline with `--edge-dim 2`.
  - Profile `v2` adds `--v2 --edge-dim 4`.

### Security & Scientific Integrity Notes
- Verified that all unit tests run without accessing external network or modifying public datasets.
- Confirmed that no synthetic data is used to generate publication benchmark figures.
- Preserved existing git working tree without commits or destructive resets.

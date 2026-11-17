# SBi-FAGAT: Comprehensive Test & Verification Report

**Execution Timestamp:** Current Verification Cycle  
**Platform:** Windows 10/11 (x86_64)  
**Python Runtime:** Python 3.14.0 (in isolated virtual environment `d:\dự án SIC\.venv`)  
**Overall Result:** **79 / 79 PASSED (100% Pass Rate)**

---

## 1. Test Environment Specifications

| Component | Package / Tool | Version | Verification Note |
| :--- | :--- | :--- | :--- |
| **OS Shell** | PowerShell 7 / Windows Terminal | Windows NT | Native PowerShell environment |
| **Python** | CPython | 3.14.0 | Isolated in `.venv` |
| **PyTorch** | `torch` | 2.15.0 | CPU mode for testing |
| **PyG** | `torch-geometric` | 2.8.1 | Graph message-passing core |
| **Transformers**| `transformers` | 5.23.0 | HuggingFace NLP models |
| **Data Engine** | `pyarrow`, `pandas` | 26.0.1 / 3.1.1 | Parquet and tabular manipulation |
| **Timezone** | `tzdata` | 2026.6 | Required for PyArrow Windows UTC parsing |
| **Scientific** | `numpy`, `scipy`, `scikit-learn`| 2.5.5 / 1.18.1 / 1.9.1 | Math & evaluation metrics |
| **Visuals** | `matplotlib`, `seaborn`, `umap-learn` | 3.11.2 / 0.13.2 / 0.6.0 | Figure generation & dimensionality reduction |
| **Test Runner** | `pytest` | 9.2.0 | Unit test suite orchestration |

---

## 2. Test Suite Breakdown by Module

Command executed:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

| Test File | Number of Tests | Status | Scope & Tested Functionality |
| :--- | :---: | :---: | :--- |
| `tests/test_schema.py` | 8 | **PASSED** | Pydantic data contract validation (`PostNode`, `EdgeInteraction`, `CascadeTree`), timestamp ordering, root-reply validation, JSON serialization. |
| `tests/test_pheme_loader.py` | 6 | **PASSED** | PHEME directory structure parsing, rumor label mapping, thread reconstruction, and event-based inductive splitting. |
| `tests/test_pyg_dataset.py` | 9 | **PASSED** | Conversion of `CascadeTree` to PyG `Data` objects, graph batching, burst feature extraction, and target label isolation. |
| `tests/test_no_leakage.py` | 5 | **PASSED** | **Critical Zero-Leakage Audit:** Confirms target values ($y_{\text{size}}$, $y_{\text{polar}}$) are completely absent from node feature matrix $X$ and edge attribute tensor $E$. Confirms no interactions past $K=20$ enter graph structure. |
| `tests/test_models.py` | 14 | **PASSED** | Forward pass integrity for SBi-FAGAT and all 7 baselines (GCN, GAT, BiGCN, BiGAT, FAGCN, StanceGCN, StanceGAT), batching behavior, and edge-score modulation. |
| `tests/test_metrics.py` | 10 | **PASSED** | Multi-task metrics: MSE, MAE, Pearson $r$, Spearman $\rho$, Macro-F1, Accuracy, and handling of zero-variance degenerate cases. |
| `tests/test_overfit.py` | 3 | **PASSED** | Optimization sanity check: verifies that SBi-FAGAT and baselines can overfit a small batch of synthetic cascades to near-zero loss. |
| `tests/test_tree_builder.py` | 7 | **PASSED** | Graph reconstruction from flattened parent-child lists, handling of missing parents, orphan nodes, and cyclic reference pruning. |
| `tests/test_pipeline_and_viz.py` | 5 | **PASSED** | Offline NLP feature extraction pipeline, UMAP projection generator, and cascade visualization functions. |
| `tests/test_ablation.py` | 7 | **PASSED** | Component ablation: verifies model behavior when low-pass attention, high-pass attention, bidirectional edges, or stance guidance are ablated. |
| `tests/test_live_predictor_audit.py` | 5 | **PASSED** | **New Audit Test Suite:** Verifies blocking of uncheckpointed live inference, `--allow-untrained` flag, corrupt checkpoint detection, `--demo-offline` mode, and empty thread error handling. |
| **TOTAL** | **79** | **PASSED** | **100% Pass Rate across all 11 test modules.** |

---

## 3. Dedicated Smoke Test Results

### 1. Kaggle Parallel Runner Smoke Test
```powershell
.\.venv\Scripts\python.exe -m scripts.kaggle_parallel_runner --role member1_vn --smoke-test
```
- **Output:** Verified execution of role initialization, environment detection, and argument validation.
- **Exit Code:** `0` (Success).

### 2. Live Thread Predictor Demo-Offline Mode
```powershell
.\.venv\Scripts\python.exe -m scripts.predict_live_thread --demo-offline
```
- **Output:** Successfully constructed mock demo cascade, performed feature extraction, ran model forward pass with untuned structural check notice, and printed multi-task predictions.
- **Exit Code:** `0` (Success).

### 3. Live Thread Predictor Security Guard Verification
```powershell
.\.venv\Scripts\python.exe -m scripts.predict_live_thread --url "https://reddit.com/r/VietNam/comments/test"
```
- **Output:**
  ```text
  RuntimeError: [SAFETY ERROR] No trained model checkpoint was provided via --checkpoint.
  Running inference on real social media threads with randomly initialized weights
  produces scientifically invalid predictions.
  To perform a structural/dry-run check with random weights, explicitly pass --allow-untrained.
  ```
- **Result:** Successfully blocked unauthorized execution.

---

## 4. Test Conclusion & Certification
The SBi-FAGAT codebase meets all verification and testing requirements:
- Zero data leakage is mathematically and programmatically enforced.
- All models instantiate and compute forward/backward passes cleanly.
- Edge scoring functions handle both 2D and 4D configurations.
- Safeguards are active on external entrypoints.

# SBi-FAGAT: Test & Verification Report

**Platform:** Windows (`win32`, CPU verification pass) + GitHub Actions Ubuntu (`Python 3.11`)  
**Suite Scope:** 15 test modules in `tests/` — **114 passed, 0 failed** (unified 2D architecture, multi-task KL-BCE optimization, zero data leakage, EGM Round-1 `B1–B10`/`M1–M8` + Round-2 3-Goldfish adversarial audit remediations: detached BFS depth-30 gradient stability, Holm-Bonferroni `p < 0.01` bootstrap resolution, T4 DeBERTa AMP policy, UPFD classification-only comparator, `min_total_replies_extracted` guard, crawler fault recovery, 50-tree future-window human audit)

---

## 1. Test Environment Specifications

| Component | Package / Tool | Declared Constraint | Verified Local Version |
| :--- | :--- | :--- | :--- |
| **Python** | CPython | `>=3.10` | `3.13.14` |
| **PyTorch** | `torch` | `>=2.3.0` | `2.15.0+cpu` |
| **PyG** | `torch-geometric` | `>=2.5.0` | `2.8.1` |
| **Transformers** | `transformers` | `>=4.38.0,<5.0.0` | `5.23.0` (local) / `4.x` (Kaggle & CI) |
| **Arrow / Tabular** | `pyarrow`, `pandas` | `>=14.0.0` / `>=2.1.0` | `25.0.1` / `3.1.1` |
| **Scientific** | `numpy`, `scipy`, `scikit-learn` | `>=1.24.0,<3.0.0` / `>=1.11.0` / `>=1.3.0` | `2.4.6` / `1.18.1` / `1.9.1` |
| **Test Runner** | `pytest` | `>=8.0.0` | `9.1.0` |

---

## 2. Pytest Suite Coverage (`python -m pytest -v`)

| Test Module | Focus |
| :--- | :--- |
| `tests/test_egm_audit_remediation.py` | Full EGM audit regressions (`B1–B10`, `M1–M8`): `min_total_replies` LOO teacher coverage, `edge_source="auto"` Student consistency, $O(N)$ JSONL teacher delta resume, crawl `.state.json` + salted PHEME, LaTeX `_` escaping & `table*` + RFC JSON, NLI double-contradiction neutral + 50-tree VN blind audit, direction-aware stars + exact $F=4$ LOO permutation + topic-cluster bootstrap, continuous `y_polar` + fixed future $K$-window, Camp Dipole/Dir-GNN/Dual-Channel ablations + `BiEdgeGATv2` + `TextMLP` topology mask + `assemble_paper_artifacts` + UPFD fault isolation |
| `tests/test_egm_fixes.py` | EGM hardening fixes (`F1–F6`), `node_backbone`, fair model parity, `min_total_replies=25` |
| `tests/test_kaggle_t4x2_v2_pipeline.py` | Unified 2D discourse labels, NLI verbalizer simplex, `DualHeadCrossEncoder`, `auto_loop`, smoke runner & notebooks |
| `tests/test_live_predictor_audit.py` | Live predictor untrained guard, `--allow-untrained`, corrupt checkpoints, `--demo-offline` |
| `tests/test_nlp_and_features.py` | Dual teachers, *Ancestor-Conditioned SpanPool*, finite `.pt` bank extraction |
| `tests/test_no_leakage.py` | Strict $K=20$ truncation, `future_only` $y_{\text{polar}}$, zero post-$t_K$ leakage |
| `tests/test_overfit_20.py` | Step 1 20-cascade overfit, 50/50 dipole cancellation, `TwoStreamNodeStem` topology mask, fair `hidden_dim=64` |
| `tests/test_overfit_batch.py` | Overfit convergence across `SBiFAGAT` and baselines |
| `tests/test_pheme_tree.py` | PHEME tree parser & event-grouped inductive splitting |
| `tests/test_pipeline_and_viz.py` | Vector figure generation (`.pdf`/`.svg`), UMAP, top-5 trigger export, reporting |
| `tests/test_round2_research_fixes.py` | Loss balancing, statistical tests, Sarle's BC & Signed Assortativity |
| `tests/test_schema.py` | `NodeRecord`/`EdgeRecord` Pydantic contracts, UTC enforcement, cycle/leakage rejection |
| `tests/test_sota_v2_upgrades.py` | LOO `edge_source="teacher"`, EDA/Persistence & Efficiency LaTeX, CLI aliases, fair runner config |
| `tests/test_step3_training_and_audit_fixes.py` | KL-BCE zero min & tail gradient, EMA/best-val seed selection, `EarlyStopping` NaN/Inf guard, sealed key bundles, canonical split locks, dynamic LaTeX labels, NLI reply preservation |
| `tests/test_vn_tree.py` | Arctic Shift VN tree builder & salted SHA-256 pseudonymization |

---

## 3. Offline CLI & Smoke Verification (`exit code 0`)

1. `python -m scripts.demo_research_walkthrough` — Exited `0` (CPU walkthrough comparing `StandardGCN` vs. `SBiFAGAT`).
2. `python -m scripts.predict_live_thread --demo-offline` — Exited `0` (verified live safety guard + offline diagnostics).
3. `python -m scripts.benchmark_efficiency` — Exited `0` (`hidden_dim=64`, `SBiFAGAT = 163,836` params; with `--use-cascade-stats`: `170,332` params).
4. `python -m scripts.kaggle_parallel_runner --role member1_vn --smoke-test` — Exited `0` (builds `sbifagat_member1_vn_bundle.zip` excluding `*_machine_key.csv` AND `sbifagat_member1_vn_sealed_machine_keys.zip` preserving the blind answer key).
5. `python -m scripts.kaggle_parallel_runner --role member2_en --smoke-test` — Exited `0`.

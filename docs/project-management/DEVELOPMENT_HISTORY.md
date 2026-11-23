# SBi-FAGAT: Development History & Git Commit Progression

## 1. Verified Git Commit Log

| Commit | Summary | Key Architectural & Engineering Impact |
| :--- | :--- | :--- |
| `471823e` | `Initial commit` | Initial repository scaffold. |
| `46c363f` | `feat: initialize SBi-FAGAT codebase, math specs, unit tests, and team guide` | Core modules (`src/data`, `src/nlp`, `src/models`, `src/training`), `docs/math_spec.md`, `docs/literature_review.md`, `SBiFAGAT`, and 7 fair baselines. |
| `71431cc` | `research: resolve round-2 scientific blockers & add SBi-FAGAT learning lab` | Loss calibration, `docs/huong_dan_nghien_cuu_de_hieu.md`, `demo_research_walkthrough.py`, `benchmark_efficiency.py`, bootstrap/Holm reporting. |
| `5950c65` | `feat(v2): 5-class Qwen3.8-27B QLoRA teacher, 8D burst stats, auto-loop, live predictor & Kaggle T4x2 notebooks` | Profile `v2`: 5-class `Qwen/Qwen3.8-27B` QLoRA 1-pass G-Eval teacher (`[s_uv, h_uv, r_uv, c_uv]`), 8D `cascade_stats`, heteroscedastic $\beta$-NLL uncertainty (`beta_nll=0.5`), `auto_loop.py`, `predict_live_thread.py`, and dual-T4 notebooks. |
| `f4be964` | `fix(kaggle-t4x2): harden Qwen3_5ForConditionalGeneration 4-bit loader, Student CE extraction & teacher budget` | Hardened 4-bit NF4 loading, student CE extraction, and Kaggle `T4 x 2` token budget management. |
| `5820e71` | `feat(audit): implement scientific safeguards, profile gating, and project management suite` *(PR #1 by `thuyenevux`)* | Added salted author pseudonymization + `UserWarning`, live-predictor untrained guard + `test_live_predictor_audit.py`, `tzdata` Windows fix, `math_spec.md` cleanup, and initial `docs/project-management/`. |
| `090d777` | `feat(unified-2d): Steps 1–3 unified 2D SBi-FAGAT, SpanPool, KL-BCE & fair 8-model suite` | Retired `Qwen3.8-27B` QLoRA in favor of 100%-coverage Dual-Teacher (`mDeBERTa-v3-xnli` Symmetric Verbalizer + `ViHSD`/`TweetEval-Offensive`) -> reply-span-pooled 2D `DualHeadCrossEncoder`, Ancestor SpanPool, dual-channel Dir-GNN `SBiFAGAT`, `hidden_dim=64` parity, KL-calibrated BCE, and sealed machine-key zip separation. |
| `3d95581` | *EGM Hardening (`B1–B10`, `M1–M8`)* | Fixed PHEME LOO `min_total_replies` teacher coverage (`B1`), eliminated `edge_source` split-brain (`B2`), replaced $O(N^2)$ teacher JSON checkpointing with $O(N)$ JSONL deltas + multi-GPU AMP (`B3`), added crawl cursor `.state.json` resume + salted PHEME hashing (`B4`), hardened Kaggle stage skip & UPFD fault isolation (`B5`), fixed LaTeX `_` escaping + `table*` + 6 paper figure embeds + `assemble_paper_artifacts.py` (`B6`), fixed NLI double-contradiction neutral attribution + added `audit_vn_50_trees.py` (`B7`), added direction-aware stars + exact small-sample LOO permutation + topic-cluster bootstrap (`B8`), upgraded to continuous finite-sample-unbiased `y_polar` + fixed future $K$-curve window (`B9`), and added Camp Dipole/Dir-GNN/Dual-Channel ablations + matched `BiEdgeGATv2` + `TextMLP` topology masking (`B10`). |
| `2575295` | `feat(model,loss,pheme): CleanRes Camp Dipole & decoupled polar_head` | Added CleanRes Camp Dipole readout, decoupled `polar_head`, balanced 2:1 multi-task loss, and shallow `/kaggle/input` PHEME archive discovery (`3215ede` extended BFS depth to 6). |
| `338651b` | `results(vn): official 5-seed VN benchmark & mechanism artifacts` | Integrated Track 1 Vietnamese 5-seed benchmarks, ablations, `analyze_mechanisms.py` counterfactual results, efficiency stats, and VN vector figures. |
| `11637ed` | `perf(pheme): parallelized GCS-FUSE parsing & unbuffered logs` | Sped up Kaggle PHEME directory parsing with `os.scandir` + `ThreadPoolExecutor` and unbuffered progress logging. |
| `5332750` | `feat(paper,ui): Track 2 PHEME+UPFD artifacts & local blind audit UI` | Integrated Track 2 PHEME + UPFD results/figures, assembled unified tables (`0` missing), added `scripts/human_audit_ui.py` + `tests/test_human_audit_ui.py`, and synced `PROJECT_STATUS.md` / `TEST_REPORT.md`. |
| `fix/egm-audit-hardening` *(v3)* | *EGM v3 Overhaul: ReZero Anchor, Binary Gate 1 & Colab GPU Offload* | Scrubbed PII/spam in VN Parquets; linearized `y_polar`; fixed single-camp Camp Dipole; switched Persistence-Residual anchor to ReZero `tanh(w_p)` centered around train logit; added validation-only comparator selection, `r2_polar`, `beta=0.05`, and `--split-mode temporal`; streamlined blind Gate 1 to `20/120/30` binary `clean`/`uncivil` with `--lock-role` UI & multi-corpus `assemble_paper_artifacts.py`; executed v3 canonical VN + sensitivity + EDA + efficiency (`18` test modules) and added `notebooks/colab_gpu_v3_gnn_benchmarks.ipynb` + Notebook 2 Tier 2A ($K=10, N \ge 15$). |

---

## 2. Unified 2D Architecture Pipeline (Current Production Standard)

```mermaid
flowchart TD
    subgraph Stage1 ["Stage 1: Ingestion & Leakage-Free Splits"]
        s1_data["PHEME Figshare (MD5 + Salted SHA-256)<br>Arctic Shift VN (vozforums / VietNam + .state.json)"]
        s1_split["Canonical topic_splits.json<br>(70/15/15, min_total_replies=25)"]
        s1_data --> s1_split
    end

    subgraph Stage2 ["Stage 2: Offline 2D NLP Extraction (T4 x 2 FP16)"]
        s2_teacher["Dual Open-Source Teachers (100% Eligible Coverage)<br>mDeBERTa-v3-xnli Symmetric Verbalizer + ViHSD / TweetEval"]
        s2_student["Reply-Span-Pooled DualHeadCrossEncoder<br>(2D [s_uv, h_uv], O(N) JSONL delta cache)"]
        s2_span["Frozen Ancestor-Conditioned SpanPool<br>(ViSoBERT / Twitter-RoBERTa -> 768-d)"]
        s1_split --> s2_teacher --> s2_student
        s1_split --> s2_span
    end

    subgraph Stage3 ["Stage 3: Dual-Channel Dir-GNN SBi-FAGAT (hidden_dim=64)"]
        s3_stem["TwoStreamNodeStem (768-d Sem + 10-D Causal struct_x)"]
        s3_gnn["TD + BU StanceGuidedFAConv<br>(Signed Polarity || Excitation + Dir-GNN Mix)"]
        s3_read["Camp Dipole Readout (64-hop BFS) + 8-D cascade_stats"]
        s3_out["MultiTaskHead: y_size (Huber), y_polar (KL-BCE), y_rumor (CE)"]
        s2_span --> s3_stem
        s2_student --> s3_gnn
        s3_stem --> s3_gnn --> s3_read --> s3_out
    end
```

# SBi-FAGAT Turnkey Cloud Execution Guide (Google Colab & Kaggle T4 x 2)

> **Local Environment Note:** `kaggle` CLI and `~/.kaggle/kaggle.json` (or `KAGGLE_USERNAME`/`KAGGLE_KEY`) are **not** configured on this local machine, so cloud jobs cannot be pushed unattended from the terminal. All 4 notebooks under `notebooks/` and `scripts/kaggle_parallel_runner.py` have been synchronized with M1–M5 and validated cleanly (`json` + `ast.parse` + `--smoke-test`) for immediate one-click execution.
>
> **R4 Safety Guarantee:** Every cloud notebook and `kaggle_parallel_runner.py` writes outputs strictly to **`results/`** and **`checkpoints/`**—never modifying `paper/`, `reports/`, or `docs/`.

---

## 1. What Is Already Computed Locally vs. Queued for Cloud GPU

| Workload | Local Status (`d:\1. LapTrinh\4_AI\results\`) | Cloud Notebook Target |
| :--- | :--- | :--- |
| Tier 2 Macro Analytics (VN & PHEME: Garimella RWC, Gómez $Q_{\text{signed}}$ + null, Davis Weak Balance, zero-row filter) | **Completed locally** (`results_macro_network_{vn,pheme}.json`) | Included in `kaggle_macro_signed_gnn_t4.ipynb` & `colab_gpu_v3_gnn_benchmarks.ipynb` |
| Tier 2 Signed Link Prediction (10 models x 5 seeds) | **Executed locally if `<15 min` else 1-seed probe + ETA** (see `results/local_timing_benchmarks.json`) | `kaggle_macro_signed_gnn_t4.ipynb` (or `--run-tier2` in Member 1/2) |
| Tier 1 5-Seed Tabular Baselines (`Early-K Persistence`, `Calibrated Early-K Persistence`, `Ridge`, `LightGBM`) across all 5 splits | **Completed locally** (`results/tier1_tabular_*.json`) | Full 15+ model tables include these + neural/SOTA models |
| VN 1,500-Cascade Contraction Eval | **Completed locally** (`results/contraction_eval_vn.json`) | N/A (local parquet audit) |
| Human & Tree Audits (`120` pairs, `20` pilot, `30` trees) | **Completed locally** (`results/*audit*_summary.json`) | Sealed machine keys remain excluded from main bundles |
| Efficiency + E2E Raw-Text Latency | **Completed locally** (`results/efficiency_e2e.json`, `is_partial_without_dual_ce=true` until `dual_head_weights.pt` restored) | Re-run automatically with real CE weights in Colab/Kaggle, or locally in ~15s after unzip |
| Missing `checkpoints/dual_ce_{vn,pheme}/dual_head_weights.pt` Distillation Recovery | **Missing locally** (queued for GPU) | **Colab Cell `colab-v3-nlp-distill-recovery`** (`RUN_NLP_DISTILL_RECOVERY = True`) or `--distill-from-features` |
| Full 15+ Model x 5-Seed Tier 1 (`VN canonical`, `VN embedding_cluster`, `VN temporal`, `PHEME inductive`, `PHEME LOO`, `SBiFAGAT + Tier2Prior`) & Ablation + $K$-Curve | **Probed/timed locally** (`~60–140 min`/split, `~8–10h` PHEME LOO on GTX 1660 SUPER) | **`colab_gpu_v3_gnn_benchmarks.ipynb`** or **`kaggle_member1/2_*.ipynb`** |

---

## 2. Option A — Recommended Single-Session Run on Google Colab (`notebooks/colab_gpu_v3_gnn_benchmarks.ipynb`)

Best if you already have `data/processed/{vn,pheme}/` (`nodes.parquet`, `edges.parquet`, `nlp_features.pt`, `topic_splits.json`) on disk and want to recover `dual_ce_{vn,pheme}/dual_head_weights.pt` + run all heavy 5-seed Tier 1 / Ablation / Tier 2 sweeps in one place.

1. **Stage processed datasets to Google Drive** (from local `d:\1. LapTrinh\4_AI\`):
   - Upload `data/processed/vn/{nodes.parquet,edges.parquet,nlp_features.pt,topic_splits.json}` $\to$ `MyDrive/SBiFAGAT/data/processed/vn/`
   - Upload `data/processed/pheme/{nodes.parquet,edges.parquet,nlp_features.pt,topic_splits.json}` $\to$ `MyDrive/SBiFAGAT/data/processed/pheme/`
   - *(Optional, only if re-running NLI Teacher scoring with `RUN_TEACHER_RESCORE = True`)*: also upload `data/interim/{vn,pheme}` and `checkpoints/visobert_vihsd/`.
2. **Open `notebooks/colab_gpu_v3_gnn_benchmarks.ipynb`** in Google Colab and select **`Runtime -> Change runtime type -> T4 GPU`** (or **L4 / A100** for faster PHEME LOO).
3. If testing an unmerged branch, set `os.environ["SBIF_BRANCH"] = "<your_branch>"` before running Cell 2 (defaults to `main`).
4. **Run All Cells top-to-bottom**:
   - **Cell 3 (`colab-v3-nlp-distill-recovery`):** `RUN_NLP_DISTILL_RECOVERY = True` distills `checkpoints/dual_ce_{vn,pheme}/dual_head_weights.pt` strictly from `topic_splits.json` (`train`/`val` only; zero test leakage).
   - **Cell 4 (`colab-v3-vn-benchmarks`):** Runs VN `export_dataset_stats`, `canonical` (+ `SBiFAGAT + Tier2Prior`), `embedding_cluster`, `temporal`, `run_ablation`, `analyze_mechanisms`, `benchmark_efficiency --include-e2e`, and Tier 2 VN (`10` link models x `5` seeds) into `results/`.
   - **Cell 5 (`colab-v3-pheme-tier2a`):** Verifies `min_total_replies_extracted <= 15` and runs PHEME `export_dataset_stats`, `inductive` (+ `SBiFAGAT + Tier2Prior`), `loo` (5 folds x 5 seeds), `run_ablation`, `analyze_mechanisms`, `benchmark_efficiency --include-e2e`, and Tier 2 PHEME into `results/`.
   - **Cell 6 (`colab-v3-pack`):** Runs `scripts.assemble_paper_artifacts --paper-dir results` and zips `results/` + `checkpoints/` (including `dual_head_weights.pt` and `best_sbifagat_*.pt`, excluding `nlp_features.pt` and secrets) into **`sbifagat_v3_gpu_results.zip`**.

---

## 3. Option B — Parallel Kaggle T4 x 2 Notebooks (When Re-Running Full Pipeline or Splitting Workloads)

1. **`notebooks/kaggle_member1_nlp_and_vn_t4x2.ipynb`** *(Accelerator: GPU T4 x 2, Internet: ON)*:
   - Set `DISTILL_FROM_FEATURES = True` if you attached `data/processed/vn/` as a Kaggle Dataset and only want fast Cross-Encoder distillation + GNN sweeps; keep `False` to run full Arctic Shift + ViSoBERT-ViHSD + mDeBERTa NLI (`--hostility-target-router`) + distillation + SpanPool + Tier 1 (`canonical`, `embedding_cluster`, `temporal`, `SBiFAGAT + Tier2Prior`) + Ablation + E2E + Tier 2 VN.
   - Outputs in `/kaggle/working/`: `sbifagat_member1_vn_bundle.zip` and `sbifagat_member1_vn_sealed_machine_keys.zip`.
2. **`notebooks/kaggle_member2_gnn_and_en_t4x2.ipynb`** *(Accelerator: GPU T4 x 2, Internet: ON)*:
   - Same toggles for PHEME Tier 2A (`K=10, min_total_replies=15`), `inductive` + `loo` + `SBiFAGAT + Tier2Prior`, Ablation, E2E, RumourEval-2019 SDQC (`--context-mode root` & `parent`), and Tier 2 PHEME (`10` link models x `5` seeds).
   - Outputs in `/kaggle/working/`: `sbifagat_member2_en_bundle.zip` and `sbifagat_member2_en_sealed_machine_keys.zip`.
3. **`notebooks/kaggle_macro_signed_gnn_t4.ipynb`** *(Accelerator: GPU T4, Internet: ON)*:
   - Dedicated Tier 2 runner (`10` link models x `5` seeds, Davis Weak Balance, Gómez $Q_{\text{signed}}$ + null-graph, zero-row filter, `--mp-graph-mode coalesced`, optional `RUN_MULTIEDGE_PARITY=1`).
   - Outputs in `/kaggle/working/`: `sbifagat_macro_signed_bundle.zip`.

---

## 4. Equivalent Standalone CLI Commands (Inside Any Cloud Shell)

```bash
# 1. Fast Cross-Encoder distillation recovery from staged processed features
python -u -m scripts.kaggle_parallel_runner --role model_nlp_only --lang vi --distill-from-features --results-dir results --include-heavy-weights
python -u -m scripts.kaggle_parallel_runner --role model_nlp_only --lang en --k-horizon 10 --min-total-replies 15 --distill-from-features --results-dir results --include-heavy-weights

# 2. Tier 1 Full 15+ Model 5-Seed Sweeps + Causal Bridge
python -u -m scripts.run_experiments --data-dir data/processed/vn --k-horizon 20 --hidden-dim 64 --edge-dim 2 --use-cascade-stats --include-tier2-prior --min-total-replies 25 --polar-target-mode future_only --epochs 100 --patience 10 --output-json results/results_main_vn.json --output-table results/tables_main_vn.tex --figures-dir results/figures/vn --ckpt-out checkpoints/best_sbifagat_vn.pt
python -u -m scripts.run_experiments --data-dir data/processed/vn --k-horizon 20 --hidden-dim 64 --edge-dim 2 --use-cascade-stats --include-tier2-prior --min-total-replies 25 --polar-target-mode future_only --split-mode embedding_cluster --edge-source teacher --epochs 100 --patience 10 --output-json results/results_cluster_vn.json --output-table results/tables_cluster_vn.tex --skip-figures
python -u -m scripts.run_experiments --data-dir data/processed/vn --k-horizon 20 --hidden-dim 64 --edge-dim 2 --use-cascade-stats --include-tier2-prior --min-total-replies 25 --polar-target-mode future_only --split-mode temporal --edge-source teacher --epochs 100 --patience 10 --output-json results/results_temporal_vn.json --output-table results/tables_temporal_vn.tex --skip-figures
python -u -m scripts.run_experiments --data-dir data/processed/pheme --k-horizon 10 --hidden-dim 64 --edge-dim 2 --use-cascade-stats --include-tier2-prior --min-total-replies 15 --polar-target-mode future_only --epochs 100 --patience 10 --output-json results/results_main_pheme.json --output-table results/tables_main_pheme.tex --figures-dir results/figures/pheme --ckpt-out checkpoints/best_sbifagat_pheme.pt
python -u -m scripts.run_experiments --data-dir data/processed/pheme --k-horizon 10 --hidden-dim 64 --edge-dim 2 --use-cascade-stats --include-tier2-prior --min-total-replies 15 --polar-target-mode future_only --eval-protocol loo --edge-source teacher --epochs 100 --patience 10 --output-json results/results_loo_pheme.json --output-table results/tables_loo_pheme.tex --ckpt-out checkpoints/best_sbifagat_pheme_loo.pt --skip-figures

# 3. Ablation + K-Horizon Curve
python -u -m scripts.run_ablation --data-dir data/processed/vn --k-horizon 20 --hidden-dim 64 --edge-dim 2 --use-cascade-stats --min-total-replies 25 --polar-target-mode future_only --epochs 100 --patience 10 --output-json results/results_ablation_vn.json --output-table results/tables_ablation_vn.tex --figures-dir results/figures/vn
python -u -m scripts.run_ablation --data-dir data/processed/pheme --k-horizon 10 --hidden-dim 64 --edge-dim 2 --use-cascade-stats --min-total-replies 15 --polar-target-mode future_only --epochs 100 --patience 10 --output-json results/results_ablation_pheme.json --output-table results/tables_ablation_pheme.tex --figures-dir results/figures/pheme

# 4. Tier 2 Macro & 10-Model x 5-Seed Signed Link Prediction
python -u -m scripts.run_macro_signed_network --data-dir data/processed/vn --dataset-name vn --batch-size 2048 --mp-graph-mode coalesced --paper-dir results --figures-dir results/figures/vn
python -u -m scripts.run_macro_signed_network --data-dir data/processed/pheme --dataset-name pheme --batch-size 2048 --mp-graph-mode coalesced --paper-dir results --figures-dir results/figures/pheme
```

---

## 5. Merging Downloaded Bundles Back Locally (PowerShell, R4-Safe)

After downloading the zip bundle(s) into `Downloads\`, extract strictly into `results/` and `checkpoints/` (never touching `paper/`, `reports/`, or `docs/`):

```powershell
cd d:\1. LapTrinh\4_AI

# Extract downloaded zip into a staging folder first so paper/ can never be overwritten
$stage = "$env:TEMP\sbifagat_cloud_stage"
Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
Expand-Archive -Path "$env:USERPROFILE\Downloads\sbifagat_v3_gpu_results.zip" -DestinationPath $stage -Force

if (Test-Path "$stage\results")     { Copy-Item "$stage\results\*"     "results\"     -Recurse -Force }
if (Test-Path "$stage\checkpoints") { Copy-Item "$stage\checkpoints\*" "checkpoints\" -Recurse -Force }

# Verify restored weights and re-run local E2E raw-text latency with real DualHeadCrossEncoder (~15s)
D:\sbifagat_cuda_venv\Scripts\python.exe -u -m scripts.benchmark_efficiency --include-e2e --use-cascade-stats --k-horizon 20 --output-json results/efficiency_e2e.json --output-table results/tables_efficiency_e2e.tex
D:\sbifagat_cuda_venv\Scripts\python.exe -u -m scripts.run_m5_local_audits --validate-results
```

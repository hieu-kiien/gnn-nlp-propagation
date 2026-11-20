#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
UPFD_ROOT="${UPFD_ROOT:-data/raw/upfd}"
MIN_TOTAL_REPLIES="${MIN_TOTAL_REPLIES:-25}"
POLAR_MODE="${POLAR_MODE:-future_only}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"

UNIFIED_FLAGS=("--edge-dim" "2" "--use-cascade-stats" "--hidden-dim" "${HIDDEN_DIM}")
echo ">>> Running Unified 2D SBi-FAGAT pipeline (hidden_dim=${HIDDEN_DIM}, edge_dim=2, min_total_replies=${MIN_TOTAL_REPLIES})"

echo "[1/4] Running full pytest verification suite..."
"${PYTHON_BIN}" -m pytest tests/ -vv -o addopts=""

run_dataset_suite() {
    local ddir="$1"
    local ds_name
    ds_name="$(basename "${ddir}")"
    echo ">>> Benchmarking dataset: ${ddir} (min_total_replies=${MIN_TOTAL_REPLIES}, polar_mode=${POLAR_MODE})"
    "${PYTHON_BIN}" -m scripts.run_experiments \
        --data-dir "${ddir}" \
        --min-total-replies "${MIN_TOTAL_REPLIES}" \
        --polar-target-mode "${POLAR_MODE}" \
        "${UNIFIED_FLAGS[@]}"
    "${PYTHON_BIN}" -m scripts.run_ablation \
        --data-dir "${ddir}" \
        --min-total-replies "${MIN_TOTAL_REPLIES}" \
        --polar-target-mode "${POLAR_MODE}" \
        "${UNIFIED_FLAGS[@]}"
    "${PYTHON_BIN}" -m scripts.export_dataset_stats \
        --data-dir "${ddir}" \
        --names "${ds_name}" \
        --min-total-replies "${MIN_TOTAL_REPLIES}" \
        --edge-dim 2 \
        --output-table "paper/tables_dataset_eda_${ds_name}.tex" \
        --output-json "paper/results_dataset_eda_${ds_name}.json"
    "${PYTHON_BIN}" -m scripts.benchmark_efficiency \
        --hidden-dim "${HIDDEN_DIM}" \
        --edge-dim 2 \
        --use-cascade-stats \
        --output-table "paper/tables_efficiency_${ds_name}.tex" \
        --output-json "paper/results_efficiency_${ds_name}.json"
}

echo "[2/4] Discovering processed datasets..."
if [[ -n "${DATA_DIR:-}" ]]; then
    run_dataset_suite "${DATA_DIR}"
else
    FOUND_DATASET=0
    for candidate in "data/processed/pheme" "data/processed/vn" "data/processed"; do
        if [[ -f "${candidate}/nlp_features.pt" && -f "${candidate}/nodes.parquet" ]]; then
            FOUND_DATASET=1
            run_dataset_suite "${candidate}"
        fi
    done
    if [[ "${FOUND_DATASET}" -eq 0 ]]; then
        echo "[ERROR] No processed dataset found under data/processed/{pheme,vn}. Run bash scripts/run_phase2.sh first." >&2
        exit 1
    fi
fi

echo "[3/4] Running UPFD GossipCop (BERT) 5-seed generalization benchmark..."
"${PYTHON_BIN}" -m scripts.run_upfd_extension --upfd-root "${UPFD_ROOT}" --hidden-dim "${HIDDEN_DIM}"

echo "[4/4] All tests and SBi-FAGAT Phase 3 experimental pipelines completed."

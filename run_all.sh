#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
UPFD_ROOT="${UPFD_ROOT:-data/raw/upfd}"
MIN_TOTAL_REPLIES="${MIN_TOTAL_REPLIES:-21}"
POLAR_MODE="${POLAR_MODE:-future_only}"
PIPELINE_VERSION="${PIPELINE_VERSION:-v1}"

EXTRA_FLAGS=()
if [[ "${PIPELINE_VERSION}" == "v2" ]]; then
    EXTRA_FLAGS+=("--v2" "--edge-dim" "4")
    echo ">>> PIPELINE_VERSION=v2 selected: Enabling 4D edge attributes, 8D burst stats, and uncertainty head."
else
    echo ">>> PIPELINE_VERSION=v1 selected (default): Running 2D edge attributes foundation."
fi

echo "[1/4] Running full pytest verification suite..."
"${PYTHON_BIN}" -m pytest tests/ -vv -o addopts=""

run_dataset_suite() {
    local ddir="$1"
    echo ">>> Benchmarking dataset: ${ddir} (min_total_replies=${MIN_TOTAL_REPLIES}, polar_mode=${POLAR_MODE}, pipeline=${PIPELINE_VERSION})"
    "${PYTHON_BIN}" -m scripts.run_experiments \
        --data-dir "${ddir}" \
        --min-total-replies "${MIN_TOTAL_REPLIES}" \
        --polar-target-mode "${POLAR_MODE}" \
        "${EXTRA_FLAGS[@]}"
    "${PYTHON_BIN}" -m scripts.run_ablation \
        --data-dir "${ddir}" \
        --min-total-replies "${MIN_TOTAL_REPLIES}" \
        --polar-target-mode "${POLAR_MODE}" \
        "${EXTRA_FLAGS[@]}"
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
"${PYTHON_BIN}" -m scripts.run_upfd_extension --upfd-root "${UPFD_ROOT}"

echo "[4/4] All tests and SBi-FAGAT Phase 3 experimental pipelines completed."

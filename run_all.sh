#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
UPFD_ROOT="${UPFD_ROOT:-data/raw/upfd}"
MIN_TOTAL_REPLIES="${MIN_TOTAL_REPLIES:-25}"
PHEME_K_HORIZON="${PHEME_K_HORIZON:-20}"
PHEME_MIN_REPLIES="${PHEME_MIN_REPLIES:-${MIN_TOTAL_REPLIES}}"
VN_K_HORIZON="${VN_K_HORIZON:-20}"
VN_MIN_REPLIES="${VN_MIN_REPLIES:-${MIN_TOTAL_REPLIES}}"
POLAR_MODE="${POLAR_MODE:-future_only}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"

UNIFIED_FLAGS=("--edge-dim" "2" "--use-cascade-stats" "--hidden-dim" "${HIDDEN_DIM}")
echo ">>> Running Unified 2D SBi-FAGAT pipeline (hidden_dim=${HIDDEN_DIM}, edge_dim=2, vn=${VN_K_HORIZON}/>=${VN_MIN_REPLIES}, pheme=${PHEME_K_HORIZON}/>=${PHEME_MIN_REPLIES})"

echo "[1/4] Running full pytest verification suite..."
"${PYTHON_BIN}" -m pytest tests/ -vv -o addopts=""

run_dataset_suite() {
    local ddir="$1"
    local ds_name
    ds_name="$(basename "${ddir}")"
    local k_val="${VN_K_HORIZON}"
    local m_val="${VN_MIN_REPLIES}"
    if [[ "${ds_name}" == "pheme" ]]; then
        k_val="${PHEME_K_HORIZON}"
        m_val="${PHEME_MIN_REPLIES}"
    fi
    echo ">>> Benchmarking dataset: ${ddir} (k_horizon=${k_val}, min_total_replies=${m_val}, polar_mode=${POLAR_MODE})"
    "${PYTHON_BIN}" -m scripts.run_experiments \
        --data-dir "${ddir}" \
        --k-horizon "${k_val}" \
        --min-total-replies "${m_val}" \
        --polar-target-mode "${POLAR_MODE}" \
        "${UNIFIED_FLAGS[@]}"
    if [[ "${ds_name}" == "pheme" ]]; then
        "${PYTHON_BIN}" -m scripts.run_experiments \
            --data-dir "${ddir}" \
            --eval-protocol loo \
            --k-horizon "${k_val}" \
            --min-total-replies "${m_val}" \
            --polar-target-mode "${POLAR_MODE}" \
            --output-table "paper/tables_loo_${ds_name}.tex" \
            --output-json "paper/results_loo_${ds_name}.json" \
            --ckpt-out "checkpoints/best_sbifagat_${ds_name}_loo.pt" \
            --skip-figures \
            "${UNIFIED_FLAGS[@]}"
    fi
    if [[ "${ds_name}" == "vn" ]]; then
        "${PYTHON_BIN}" -m scripts.audit_vn_50_trees \
            --data-dir "${ddir}" \
            --min-total-replies "${m_val}" \
            --export
    fi
    "${PYTHON_BIN}" -m scripts.run_ablation \
        --data-dir "${ddir}" \
        --k-horizon "${k_val}" \
        --min-total-replies "${m_val}" \
        --polar-target-mode "${POLAR_MODE}" \
        "${UNIFIED_FLAGS[@]}"
    "${PYTHON_BIN}" -m scripts.export_dataset_stats \
        --data-dir "${ddir}" \
        --names "${ds_name}" \
        --k-horizon "${k_val}" \
        --min-total-replies "${m_val}" \
        --edge-dim 2 \
        --output-table "paper/tables_dataset_eda_${ds_name}.tex" \
        --output-json "paper/results_dataset_eda_${ds_name}.json"
    "${PYTHON_BIN}" -m scripts.run_macro_signed_network \
        --data-dir "${ddir}" \
        --dataset-name "${ds_name}" \
        --analysis-only
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

echo "[3/4] Running UPFD GossipCop (BERT) 5-seed structural transfer benchmark..."
"${PYTHON_BIN}" -m scripts.run_upfd_extension --upfd-root "${UPFD_ROOT}" --hidden-dim "${HIDDEN_DIM}" || \
    echo "[WARN] Optional UPFD benchmark failed (check network/GDrive mirror); continuing to paper assembly."

echo "[4/4] Assembling unified paper tables & validating LaTeX syntax..."
"${PYTHON_BIN}" -m scripts.assemble_paper_artifacts --paper-dir paper

echo "All tests and SBi-FAGAT Phase 3 experimental pipelines completed."

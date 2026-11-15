#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PHEME="${RUN_PHEME:-1}"
RUN_VN="${RUN_VN:-1}"
VN_TARGET_COUNT="${VN_TARGET_COUNT:-500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-3}"

if [[ "${RUN_PHEME}" -eq 1 ]]; then
    echo "[Phase 2 - PHEME 1/3] Downloading & parsing PHEME cascades..."
    "${PYTHON_BIN}" -m src.data.pheme_loader --output-dir data/interim/pheme

    echo "[Phase 2 - PHEME 2/3] Frozen teacher scoring & blind human audit export (en)..."
    "${PYTHON_BIN}" -m src.nlp.dual_edge_scorer \
        --lang en \
        --interim-dir data/interim/pheme \
        --scored-cache-json data/interim/pheme/scored_pairs.json \
        --audit-csv data/interim/human_audit_en_300.csv \
        --pilot-csv data/interim/pilot_calibration_en_30.csv \
        --batch-size "${BATCH_SIZE}" \
        --score-only \
        --export-audit

    echo "[Phase 2 - PHEME 3/3] Extracting frozen node embeddings & edge attributes from teacher cache..."
    "${PYTHON_BIN}" -m src.nlp.extract_features \
        --dataset pheme \
        --interim-dir data/interim/pheme \
        --output-dir data/processed/pheme \
        --scored-cache-json data/interim/pheme/scored_pairs.json \
        --batch-size "$((BATCH_SIZE * 2))" \
        --k-horizon 20
fi

if [[ "${RUN_VN}" -eq 1 ]]; then
    echo "[Phase 2 - VN 1/3] Crawling Vietnamese discussion trees from Arctic Shift..."
    "${PYTHON_BIN}" -m src.data.arctic_shift_vn \
        --output-dir data/interim/vn \
        --target-count "${VN_TARGET_COUNT}" \
        --checkpoint-jsonl data/interim/vn/crawl_checkpoint.jsonl \
        --allow-partial

    echo "[Phase 2 - VN 2/3] Fine-tuning ViSoBERT-ViHSD, frozen teacher scoring & blind audit export (vi)..."
    "${PYTHON_BIN}" -m src.nlp.dual_edge_scorer \
        --lang vi \
        --interim-dir data/interim/vn \
        --vihsd-dir checkpoints/visobert_vihsd \
        --scored-cache-json data/interim/vn/scored_pairs.json \
        --audit-csv data/interim/human_audit_300.csv \
        --pilot-csv data/interim/pilot_calibration_30.csv \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --train-vihsd \
        --score-only \
        --export-audit

    echo "[Phase 2 - VN 3/3] Extracting frozen node embeddings & edge attributes from teacher cache..."
    "${PYTHON_BIN}" -m src.nlp.extract_features \
        --dataset vn \
        --interim-dir data/interim/vn \
        --output-dir data/processed/vn \
        --scored-cache-json data/interim/vn/scored_pairs.json \
        --vihsd-dir checkpoints/visobert_vihsd \
        --batch-size "$((BATCH_SIZE * 2))" \
        --k-horizon 20
fi

echo "Phase 2 data & offline NLP feature extraction completed."

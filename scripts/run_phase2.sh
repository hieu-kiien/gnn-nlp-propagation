#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PHEME="${RUN_PHEME:-1}"
RUN_VN="${RUN_VN:-1}"
VN_TARGET_COUNT="${VN_TARGET_COUNT:-1500}"
VN_BEFORE_EPOCH="${VN_BEFORE_EPOCH:-1763164800}"
MIN_TOTAL_REPLIES="${MIN_TOTAL_REPLIES:-25}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-3}"

TEACHER_BACKEND="dual_nli"
EDGE_DIM="2"
echo ">>> Unified SBi-FAGAT 2D NLP Pipeline: teacher=${TEACHER_BACKEND}, edge_dim=${EDGE_DIM}, Ancestor SpanPool"

mkdir -p data/interim
if [[ -z "${SBIF_HASH_SALT:-}" && ! -f "data/interim/.author_hash_salt.secret" ]]; then
    "${PYTHON_BIN}" -c "import pathlib, secrets; p = pathlib.Path('data/interim/.author_hash_salt.secret'); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(secrets.token_hex(16), encoding='utf-8')"
    echo ">>> Initialized persistent author pseudonymization salt at data/interim/.author_hash_salt.secret"
fi

if [[ "${RUN_PHEME}" -eq 1 ]]; then
    echo "[Phase 2 - PHEME 1/3] Downloading & parsing PHEME cascades..."
    "${PYTHON_BIN}" -m src.data.pheme_loader --output-dir data/interim/pheme

    echo "[Phase 2 - PHEME 2/3] Dual NLI Verbalizer scoring, blind audit export & Cross-Encoder distillation (en)..."
    "${PYTHON_BIN}" -m src.nlp.dual_edge_scorer \
        --lang en \
        --interim-dir data/interim/pheme \
        --output-dir checkpoints/dual_ce_pheme \
        --scored-cache-json data/interim/pheme/scored_pairs.json \
        --audit-csv data/interim/human_audit_en_120.csv \
        --pilot-csv data/interim/pilot_calibration_en_20.csv \
        --audit-n-samples 120 \
        --pilot-samples 20 \
        --min-total-replies "${MIN_TOTAL_REPLIES}" \
        --teacher-backend "${TEACHER_BACKEND}" \
        --edge-dim "${EDGE_DIM}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --include-root-context \
        --score-only \
        --export-audit \
        --distill

    echo "[Phase 2 - PHEME 3/3] Extracting Ancestor SpanPool embeddings & 2D edge attributes (with isolated teacher cache)..."
    "${PYTHON_BIN}" -m src.nlp.extract_features \
        --dataset pheme \
        --interim-dir data/interim/pheme \
        --output-dir data/processed/pheme \
        --cross-encoder-ckpt checkpoints/dual_ce_pheme \
        --scored-cache-json data/interim/pheme/scored_pairs.json \
        --edge-dim "${EDGE_DIM}" \
        --include-root-context \
        --min-total-replies "${MIN_TOTAL_REPLIES}" \
        --batch-size "$((BATCH_SIZE * 2))" \
        --k-horizon 20
fi

if [[ "${RUN_VN}" -eq 1 ]]; then
    echo "[Phase 2 - VN 1/3] Crawling Vietnamese discussion trees from Arctic Shift (before_epoch=${VN_BEFORE_EPOCH})..."
    CRAWL_ARGS=(
        --output-dir data/interim/vn
        --target-count "${VN_TARGET_COUNT}"
        --min-comments "${MIN_TOTAL_REPLIES}"
        --min-vi-ratio 0.65
        --max-pages-per-sub 300
        --checkpoint-jsonl data/interim/vn/crawl_checkpoint.jsonl
        --allow-partial
    )
    if [[ -n "${VN_BEFORE_EPOCH}" ]]; then
        CRAWL_ARGS+=(--before-epoch "${VN_BEFORE_EPOCH}")
    fi
    "${PYTHON_BIN}" -m src.data.arctic_shift_vn "${CRAWL_ARGS[@]}"

    echo "[Phase 2 - VN 2/3] Fine-tuning ViSoBERT-ViHSD, NLI Verbalizer scoring, audit export & Cross-Encoder distillation (vi)..."
    "${PYTHON_BIN}" -m src.nlp.dual_edge_scorer \
        --lang vi \
        --interim-dir data/interim/vn \
        --vihsd-dir checkpoints/visobert_vihsd \
        --output-dir checkpoints/dual_ce_vn \
        --scored-cache-json data/interim/vn/scored_pairs.json \
        --audit-csv data/interim/human_audit_vi_120.csv \
        --pilot-csv data/interim/pilot_calibration_vi_20.csv \
        --audit-n-samples 120 \
        --pilot-samples 20 \
        --min-total-replies "${MIN_TOTAL_REPLIES}" \
        --teacher-backend "${TEACHER_BACKEND}" \
        --edge-dim "${EDGE_DIM}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --include-root-context \
        --train-vihsd \
        --score-only \
        --export-audit \
        --distill

    echo "[Phase 2 - VN 3/3] Extracting Ancestor SpanPool embeddings & 2D edge attributes..."
    "${PYTHON_BIN}" -m src.nlp.extract_features \
        --dataset vn \
        --interim-dir data/interim/vn \
        --output-dir data/processed/vn \
        --cross-encoder-ckpt checkpoints/dual_ce_vn \
        --scored-cache-json data/interim/vn/scored_pairs.json \
        --vihsd-dir checkpoints/visobert_vihsd \
        --edge-dim "${EDGE_DIM}" \
        --include-root-context \
        --min-total-replies "${MIN_TOTAL_REPLIES}" \
        --batch-size "$((BATCH_SIZE * 2))" \
        --k-horizon 20
fi

echo "Phase 2 data & offline NLP feature extraction completed."

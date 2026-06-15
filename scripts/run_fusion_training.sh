#!/bin/bash
# ============================================================
# Fusion Layer Training Pipeline (Clean Version)
# ============================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
BOLD='\033[1m'

PROJECT_ROOT="/workspace/eeg-genetic-fusion"
cd "$PROJECT_ROOT"

LOG_DIR="models/fusion/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/fusion_training_${TIMESTAMP}.log"

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

> "$LOG_FILE"

log "${BOLD}================================================${NC}"
log "${BOLD}  EEG-Genetic Fusion Training Pipeline${NC}"
log "${BOLD}================================================${NC}"
log ""
log "Start time: $(date)"
log "Log file: $LOG_FILE"
log ""

# ---- Step 1: Check prerequisites ----
log "${YELLOW}Step 1: Checking prerequisites...${NC}"

EEG_MODEL="seizure_prediction/SavedModels/EEGLSTM_CHB-MIT_all_patients_train_Epoch-15_TLoss-0.0501_VLoss-0.7880_20260605-152850.net"
XGB_MODEL="models/xgboost_genetic/xgboost_genetic_model.pkl"
GENETIC_DATA="data/processed/genetic_vectors/genetic_training_cohort.csv"

for f in "$EEG_MODEL" "$XGB_MODEL" "$GENETIC_DATA"; do
    if [ ! -f "$f" ]; then
        log "${RED}ERROR: Missing: $f${NC}"
        exit 1
    fi
done
log "${GREEN}  All files found!${NC}"
log ""

# ---- Step 2: Extract EEG embeddings (if not cached) ----
CACHE_FILE="models/fusion/eeg_embeddings.npz"
if [ -f "$CACHE_FILE" ]; then
    log "${GREEN}Step 2: EEG embeddings already cached - skipping${NC}"
else
    log "${YELLOW}Step 2: Extracting EEG embeddings from BiLSTM...${NC}"
    log "This will take 1-2 hours (one-time extraction)."
    log "Start time: $(date)"
    log ""
    python scripts/extract_eeg_embeddings.py 2>&1 | tee -a "$LOG_FILE"
    log ""
    log "${GREEN}Step 2 complete! End time: $(date)${NC}"
fi
log ""

# ---- Step 3: Train fusion layer ----
log "${YELLOW}Step 3: Training fusion layer...${NC}"
log "Start time: $(date)"
log ""

python scripts/train_fusion_clean.py \
    --epochs 30 \
    --lr 0.001 \
    --batch-size 32 \
    --genetic-csv "$GENETIC_DATA" \
    --xgb-model "$XGB_MODEL" \
    2>&1 | tee -a "$LOG_FILE"

log ""
log "${GREEN}Step 3 complete! End time: $(date)${NC}"
log ""

# ---- Step 4: Summary ----
log "${GREEN}================================================${NC}"
log "${GREEN}  TRAINING COMPLETE!${NC}"
log "${GREEN}================================================${NC}"
log ""
log "End time: $(date)"
log ""
log "Output files:"
log "  - models/fusion/fusion_best.pt"
log "  - models/fusion/fusion_final.pt"
log "  - models/fusion/fusion_metrics.json"
log "  - models/fusion/plots/*.png"
log "  - $LOG_FILE"
log ""

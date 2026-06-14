#!/bin/bash
# ============================================================
# Fusion Layer Training Script
# ============================================================
# This script runs the complete fusion training pipeline on the cloud PC.
# All output is saved to a log file for later review.
#
# Usage:
#   chmod +x scripts/run_fusion_training.sh
#   ./scripts/run_fusion_training.sh
#
# Prerequisites:
#   - Trained EEG model: seizure_prediction/SavedModels/EEGLSTM_CHB-MIT_all_patients_train_Epoch-15_TLoss-0.0501_VLoss-0.7880_20260605-152850.net
#   - Trained XGBoost model: models/xgboost_genetic/xgboost_genetic_model.pkl
#   - Genetic training data: data/processed/genetic_vectors/genetic_training_cohort.csv
#
# Estimated time: ~2-3 hours (depending on GPU availability)
# ============================================================

set -e  # Exit on error

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Project root
PROJECT_ROOT="/workspace/eeg-genetic-fusion"
cd "$PROJECT_ROOT"

# Log file setup
LOG_DIR="models/fusion/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/fusion_training_${TIMESTAMP}.log"

# Function to log and display
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Clear log file
> "$LOG_FILE"

log "${BOLD}================================================${NC}"
log "${BOLD}  EEG-Genetic Fusion Training Pipeline${NC}"
log "${BOLD}================================================${NC}"
log ""
log "Project root: $PROJECT_ROOT"
log "Log file: $LOG_FILE"
log "Start time: $(date)"
log ""

# ---- Step 0: Check prerequisites ----
log "${YELLOW}Step 0: Checking prerequisites...${NC}"

# Check EEG model
EEG_MODEL="seizure_prediction/SavedModels/EEGLSTM_CHB-MIT_all_patients_train_Epoch-15_TLoss-0.0501_VLoss-0.7880_20260605-152850.net"
if [ ! -f "$EEG_MODEL" ]; then
    log "${RED}ERROR: EEG model not found at $EEG_MODEL${NC}"
    log "Please ensure the trained EEG model exists."
    exit 1
fi
log "  EEG model: $EEG_MODEL"

# Check XGBoost model
XGB_MODEL="models/xgboost_genetic/xgboost_genetic_model.pkl"
if [ ! -f "$XGB_MODEL" ]; then
    log "${RED}ERROR: XGBoost model not found at $XGB_MODEL${NC}"
    log "Please ensure the trained XGBoost model exists."
    exit 1
fi
log "  XGBoost model: $XGB_MODEL"

# Check genetic training data
GENETIC_DATA="data/processed/genetic_vectors/genetic_training_cohort.csv"
if [ ! -f "$GENETIC_DATA" ]; then
    log "${RED}ERROR: Genetic training data not found at $GENETIC_DATA${NC}"
    log "Please generate synthetic genetic patients first."
    exit 1
fi
log "  Genetic data: $GENETIC_DATA"

log "${GREEN}  All prerequisites met!${NC}"
log ""

# ---- Step 1: Create output directories ----
log "${YELLOW}Step 1: Creating output directories...${NC}"
mkdir -p models/fusion
mkdir -p models/fusion/plots
mkdir -p data/processed/fusion
log "${GREEN}  Directories created!${NC}"
log ""

# ---- Step 2: Train fusion layer ----
log "${YELLOW}Step 2: Training fusion layer...${NC}"
log "This may take 1-2 hours depending on GPU availability."
log "Start time: $(date)"
log ""

python scripts/train_fusion.py \
    --eeg-model "$EEG_MODEL" \
    --genetic-features "$GENETIC_DATA" \
    --xgb-model "$XGB_MODEL" \
    --epochs 30 \
    --lr 0.001 \
    --batch-size 32 \
    2>&1 | tee -a "$LOG_FILE"

log ""
log "${GREEN}Step 2 complete! End time: $(date)${NC}"
log ""

# ---- Step 3: Evaluate fusion model ----
log "${YELLOW}Step 3: Evaluating fusion model on test set...${NC}"
log "Start time: $(date)"
log ""

python scripts/evaluate_fusion.py \
    --fusion-model models/fusion/fusion_best.pt \
    --data-dir data/processed/fusion \
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
log "  - models/fusion/fusion_best.pt (best model)"
log "  - models/fusion/fusion_final.pt (final model)"
log "  - models/fusion/fusion_metrics.json (training metrics)"
log "  - models/fusion/fusion_evaluation.json (evaluation metrics)"
log "  - models/fusion/plots/*.png (9 publication-quality plots)"
log "  - $LOG_FILE (this log)"
log ""
log "Next steps:"
log "  1. Review plots in models/fusion/plots/"
log "  2. Check fusion_metrics.json for performance"
log "  3. Deploy FastAPI backend (src/api/main.py)"
log "  4. Build React dashboard (frontend/dashboard/)"
log ""

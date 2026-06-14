#!/bin/bash
# ============================================================
# Fusion Layer Training Script
# ============================================================
# This script runs the complete fusion training pipeline on the cloud PC.
#
# Usage:
#   chmod +x scripts/run_fusion_training.sh
#   ./scripts/run_fusion_training.sh
#
# Prerequisites:
#   - Trained EEG model: seizure_prediction/SavedModels/EEGLSTM_*.net
#   - Trained XGBoost model: models/xgboost_genetic/xgboost_genetic_model.pkl
#   - Genetic training data: data/processed/genetic_vectors/genetic_training_cohort.csv
#
# Estimated time: ~2-3 hours (depending on GPU availability)
# ============================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/workspace/eeg-genetic-fusion"
cd "$PROJECT_ROOT"

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  EEG-Genetic Fusion Training Pipeline${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Start time: $(date)"
echo ""

# ---- Step 0: Check prerequisites ----
echo -e "${YELLOW}Step 0: Checking prerequisites...${NC}"

# Check EEG model
EEG_MODEL="seizure_prediction/SavedModels/EEGLSTM_CHB-MIT_all_patients_train_Epoch-15_TLoss-0.0501_VLoss-0.7880_20260605-152850.net"
if [ ! -f "$EEG_MODEL" ]; then
    echo -e "${RED}ERROR: EEG model not found at $EEG_MODEL${NC}"
    echo "Please ensure the trained EEG model exists."
    exit 1
fi
echo "  EEG model: $EEG_MODEL"

# Check XGBoost model
XGB_MODEL="models/xgboost_genetic/xgboost_genetic_model.pkl"
if [ ! -f "$XGB_MODEL" ]; then
    echo -e "${RED}ERROR: XGBoost model not found at $XGB_MODEL${NC}"
    echo "Please ensure the trained XGBoost model exists."
    exit 1
fi
echo "  XGBoost model: $XGB_MODEL"

# Check genetic training data
GENETIC_DATA="data/processed/genetic_vectors/genetic_training_cohort.csv"
if [ ! -f "$GENETIC_DATA" ]; then
    echo -e "${RED}ERROR: Genetic training data not found at $GENETIC_DATA${NC}"
    echo "Please generate synthetic genetic patients first."
    exit 1
fi
echo "  Genetic data: $GENETIC_DATA"

echo -e "${GREEN}  All prerequisites met!${NC}"
echo ""

# ---- Step 1: Create output directories ----
echo -e "${YELLOW}Step 1: Creating output directories...${NC}"
mkdir -p models/fusion
mkdir -p models/fusion/plots
mkdir -p data/processed/fusion
echo -e "${GREEN}  Directories created!${NC}"
echo ""

# ---- Step 2: Extract EEG embeddings ----
echo -e "${YELLOW}Step 2: Extracting EEG embeddings from trained BiLSTM...${NC}"
echo "This may take 1-2 hours depending on GPU availability."
echo "Start time: $(date)"
echo ""

python scripts/train_fusion.py \
    --eeg-model "$EEG_MODEL" \
    --genetic-features "$GENETIC_DATA" \
    --xgb-model "$XGB_MODEL" \
    --epochs 30 \
    --lr 0.001 \
    --batch-size 32

echo ""
echo -e "${GREEN}Step 2 complete! End time: $(date)${NC}"
echo ""

# ---- Step 3: Evaluate fusion model ----
echo -e "${YELLOW}Step 3: Evaluating fusion model on test set...${NC}"
echo "Start time: $(date)"
echo ""

python scripts/evaluate_fusion.py \
    --fusion-model models/fusion/fusion_best.pt \
    --data-dir data/processed/fusion

echo ""
echo -e "${GREEN}Step 3 complete! End time: $(date)${NC}"
echo ""

# ---- Step 4: Summary ----
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  TRAINING COMPLETE!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "End time: $(date)"
echo ""
echo "Output files:"
echo "  - models/fusion/fusion_best.pt (best model)"
echo "  - models/fusion/fusion_final.pt (final model)"
echo "  - models/fusion/fusion_metrics.json (metrics)"
echo "  - models/fusion/fusion_evaluation.json (evaluation)"
echo "  - models/fusion/plots/*.png (9 plots)"
echo ""
echo "Next steps:"
echo "  1. Review plots in models/fusion/plots/"
echo "  2. Check fusion_metrics.json for performance"
echo "  3. Deploy FastAPI backend (src/api/main.py)"
echo "  4. Build React dashboard (frontend/dashboard/)"
echo ""

#!/bin/bash
# ==============================================================================
# BoRAD: Ablation Study — Component Contribution Analysis
# ==============================================================================
# 
# Factorial design over 3 components: Predictor, Dense Loss, Prototype Loss
# 7 cases (2³ - 1 base already counted) on MVTec dataset
#
# Table:
#   A1: CosLoss only, no predictor          (baseline)
#   A2: CosLoss + Dense, no predictor       (+Dense)
#   A3: CosLoss + Proto, no predictor       (+Proto)
#   A4: CosLoss + Dense + Proto, no pred    (+Dense+Proto)
#   A5: CosLoss + Predictor only            (+Predictor)
#   A6: CosLoss + Dense + Predictor         (+Predictor+Dense)
#   A7: Full model (Predictor+Dense+Proto)  (proposed)
#
# Usage:
#   GPU=0 bash scripts/run_ablation_study.sh
#   GPU=0 DATASET=mvtec bash scripts/run_ablation_study.sh
#   GPU=0 DATASET=visa  bash scripts/run_ablation_study.sh
# ==============================================================================

set -e  # Exit on error

# Configuration
GPU=${GPU:-0}
SEED=${SEED:-42}
DATASET=${DATASET:-mvtec}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/ablation_${DATASET}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Map dataset to config path
case $DATASET in
    mvtec)  BASE_CFG="configs/rd/rd_byol_mvtec.py" ;;
    visa)   BASE_CFG="configs/rd/rd_byol_visa.py" ;;
    btad)   BASE_CFG="configs/rd/rd_byol_btad.py" ;;
    realiad) BASE_CFG="configs/rd/rd_byol_realiad.py" ;;
    *)      echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

echo "================================================================"
echo "  BoRAD Ablation Study"
echo "  Dataset: $DATASET | GPU: $GPU | Seed: $SEED"
echo "  Log dir: $LOG_DIR"
echo "================================================================"

# Helper function
run_exp() {
    local name=$1
    local config=$2
    shift 2
    local extra_opts="$@"
    
    echo ""
    echo "--------------------------------------------------------------"
    echo "  [$name] Config: $config"
    echo "  Extra opts: $extra_opts"
    echo "--------------------------------------------------------------"
    
    CUDA_VISIBLE_DEVICES=$GPU python run.py \
        -c "$config" \
        -m train \
        seed=$SEED \
        trainer.logdir_sub="ablation_${name}" \
        $extra_opts \
        2>&1 | tee "$LOG_DIR/${name}.log"
    
    echo "  [$name] Done."
}

# ==============================================================================
# WITHOUT PREDICTOR (A1–A4): Tests loss components in isolation
# ==============================================================================

# A1: Baseline — CosLoss only, no predictor
run_exp "A1_cos_only" \
    "configs/rd/ablation/rd_cos_only_no_pred.py"

# A2: +Dense — CosLoss + BYOLDenseLoss, no predictor
run_exp "A2_cos_dense" \
    "configs/rd/ablation/rd_cos_dense_no_pred.py"

# A3: +Proto — CosLoss + PrototypeInfoNCELoss, no predictor
run_exp "A3_cos_proto" \
    "configs/rd/ablation/rd_cos_proto_no_pred.py"

# A4: +Dense+Proto — All losses, no predictor
run_exp "A4_cos_dense_proto" \
    "configs/rd/ablation/rd_full_no_pred.py"

# ==============================================================================
# WITH PREDICTOR (A5–A7): Tests predictor effect on each combination
# ==============================================================================

# A5: +Predictor only — CosLoss with predictor, no contrastive losses
run_exp "A5_predictor_only" \
    "configs/rd/ablation/rd_predictor_only.py"

# A6: +Predictor+Dense — CosLoss + BYOLDenseLoss with predictor
run_exp "A6_predictor_dense" \
    "configs/rd/ablation/rd_cos_dense.py"

# A7: Full model — Predictor + Dense + Proto (proposed method)
run_exp "A7_full_model" \
    "$BASE_CFG"

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo "================================================================"
echo "  ABLATION STUDY COMPLETED"
echo "================================================================"
echo "  Results saved in: $LOG_DIR"
echo ""
echo "  Experiments:"
echo "    A1: CosLoss only (no predictor)        — baseline"
echo "    A2: + Dense (no predictor)              — dense contribution"
echo "    A3: + Proto (no predictor)              — proto contribution"
echo "    A4: + Dense + Proto (no predictor)      — combined losses"
echo "    A5: + Predictor only                    — predictor contribution"
echo "    A6: + Predictor + Dense                 — predictor + dense"
echo "    A7: Full model (Pred+Dense+Proto)       — proposed method"
echo "================================================================"

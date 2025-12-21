#!/bin/bash
# ==============================================================================
# BorAD: Full Experiment Suite for Rank-A Publication
# ==============================================================================
# This script runs all experiments needed for a comprehensive evaluation:
# 1. Main results on benchmark datasets (MVTec, VisA, BTAD, Real-IAD)
# 2. Ablation studies on loss components
# 3. Hyperparameter sensitivity analysis
# 4. Comparison with different backbones
# ==============================================================================

set -e  # Exit on error

# Configuration
GPU=${GPU:-0}
SEED=${SEED:-42}
LOG_DIR="logs/experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo "=================================================="
echo "BorAD Experiment Suite"
echo "GPU: $GPU"
echo "Seed: $SEED"
echo "Log directory: $LOG_DIR"
echo "=================================================="

# ==============================================================================
# 1. MAIN RESULTS - Benchmark Datasets
# ==============================================================================
echo ""
echo "=============================================="
echo "1. MAIN RESULTS - Benchmark Datasets"
echo "=============================================="

# MVTec AD
echo "[1.1] Training on MVTec AD..."
CUDA_VISIBLE_DEVICES=$GPU python run.py \
    -c configs/rd/rd_byol_mvtec.py \
    -m train \
    seed=$SEED \
    2>&1 | tee "$LOG_DIR/mvtec_train.log"

# VisA
echo "[1.2] Training on VisA..."
CUDA_VISIBLE_DEVICES=$GPU python run.py \
    -c configs/rd/rd_byol_visa.py \
    -m train \
    seed=$SEED \
    2>&1 | tee "$LOG_DIR/visa_train.log"

# BTAD
echo "[1.3] Training on BTAD..."
CUDA_VISIBLE_DEVICES=$GPU python run.py \
    -c configs/rd/rd_byol_btad.py \
    -m train \
    seed=$SEED \
    2>&1 | tee "$LOG_DIR/btad_train.log"

# Real-IAD
echo "[1.4] Training on Real-IAD..."
CUDA_VISIBLE_DEVICES=$GPU python run.py \
    -c configs/rd/rd_byol_realiad.py \
    -m train \
    seed=$SEED \
    2>&1 | tee "$LOG_DIR/realiad_train.log"

echo "Main results completed!"

# ==============================================================================
# 2. ABLATION STUDY - Loss Components
# ==============================================================================
echo ""
echo "=============================================="
echo "2. ABLATION STUDY - Loss Components"
echo "=============================================="

# Baseline: Only reconstruction loss (CosLoss)
echo "[2.1] Ablation: CosLoss only..."
CUDA_VISIBLE_DEVICES=$GPU python run.py \
    -c configs/rd/ablation/rd_cos_only.py \
    -m train \
    seed=$SEED \
    2>&1 | tee "$LOG_DIR/ablation_cos_only.log"

# CosLoss + Dense BYOL (no prototype)
echo "[2.2] Ablation: CosLoss + DenseBYOL..."
CUDA_VISIBLE_DEVICES=$GPU python run.py \
    -c configs/rd/ablation/rd_cos_dense.py \
    -m train \
    seed=$SEED \
    2>&1 | tee "$LOG_DIR/ablation_cos_dense.log"

# CosLoss + Prototype InfoNCE (no dense)
echo "[2.3] Ablation: CosLoss + PrototypeInfoNCE..."
CUDA_VISIBLE_DEVICES=$GPU python run.py \
    -c configs/rd/ablation/rd_cos_proto.py \
    -m train \
    seed=$SEED \
    2>&1 | tee "$LOG_DIR/ablation_cos_proto.log"

# Full model (CosLoss + Dense + Prototype)
echo "[2.4] Ablation: Full model..."
CUDA_VISIBLE_DEVICES=$GPU python run.py \
    -c configs/rd/rd_byol_mvtec.py \
    -m train \
    seed=$SEED \
    2>&1 | tee "$LOG_DIR/ablation_full.log"

echo "Ablation study completed!"

# ==============================================================================
# 3. HYPERPARAMETER SENSITIVITY - Number of Prototypes
# ==============================================================================
echo ""
echo "=============================================="
echo "3. SENSITIVITY - Number of Prototypes"
echo "=============================================="

for N_PROTO in 3 5 7 10 15; do
    echo "[3.x] n_prototypes = $N_PROTO..."
    CUDA_VISIBLE_DEVICES=$GPU python run.py \
        -c configs/rd/rd_byol_mvtec.py \
        -m train \
        seed=$SEED \
        "loss.loss_terms=[dict(type='CosLoss', name='cos', avg=False, lam=1.0), dict(type='BYOLDenseLoss', name='dense', lam=1.0, use_spatial_matching=True), dict(type='PrototypeInfoNCELoss', name='proto', lam=1.0, n_prototypes=$N_PROTO, temperature=0.07)]" \
        trainer.logdir_sub="sensitivity_nproto_$N_PROTO" \
        2>&1 | tee "$LOG_DIR/sensitivity_nproto_$N_PROTO.log"
done

echo "Prototype sensitivity analysis completed!"

# ==============================================================================
# 4. HYPERPARAMETER SENSITIVITY - Temperature
# ==============================================================================
echo ""
echo "=============================================="
echo "4. SENSITIVITY - Temperature"
echo "=============================================="

for TEMP in 0.03 0.05 0.07 0.1 0.2; do
    echo "[4.x] temperature = $TEMP..."
    CUDA_VISIBLE_DEVICES=$GPU python run.py \
        -c configs/rd/rd_byol_mvtec.py \
        -m train \
        seed=$SEED \
        "loss.loss_terms=[dict(type='CosLoss', name='cos', avg=False, lam=1.0), dict(type='BYOLDenseLoss', name='dense', lam=1.0, use_spatial_matching=True), dict(type='PrototypeInfoNCELoss', name='proto', lam=1.0, n_prototypes=5, temperature=$TEMP)]" \
        trainer.logdir_sub="sensitivity_temp_$TEMP" \
        2>&1 | tee "$LOG_DIR/sensitivity_temp_$TEMP.log"
done

echo "Temperature sensitivity analysis completed!"

# ==============================================================================
# 5. HYPERPARAMETER SENSITIVITY - Loss Weights
# ==============================================================================
echo ""
echo "=============================================="
echo "5. SENSITIVITY - Loss Weights"
echo "=============================================="

for LAM_DENSE in 0.5 1.0 2.0; do
    for LAM_PROTO in 0.5 1.0 2.0; do
        echo "[5.x] lam_dense=$LAM_DENSE, lam_proto=$LAM_PROTO..."
        CUDA_VISIBLE_DEVICES=$GPU python run.py \
            -c configs/rd/rd_byol_mvtec.py \
            -m train \
            seed=$SEED \
            "loss.loss_terms=[dict(type='CosLoss', name='cos', avg=False, lam=1.0), dict(type='BYOLDenseLoss', name='dense', lam=$LAM_DENSE, use_spatial_matching=True), dict(type='PrototypeInfoNCELoss', name='proto', lam=$LAM_PROTO, n_prototypes=5, temperature=0.07)]" \
            trainer.logdir_sub="sensitivity_lam_d${LAM_DENSE}_p${LAM_PROTO}" \
            2>&1 | tee "$LOG_DIR/sensitivity_lam_d${LAM_DENSE}_p${LAM_PROTO}.log"
    done
done

echo "Loss weight sensitivity analysis completed!"

# ==============================================================================
# 6. MOMENTUM SCHEDULE COMPARISON
# ==============================================================================
echo ""
echo "=============================================="
echo "6. MOMENTUM SCHEDULE COMPARISON"
echo "=============================================="

for SCHEDULE in constant cosine linear; do
    echo "[6.x] momentum_schedule = $SCHEDULE..."
    CUDA_VISIBLE_DEVICES=$GPU python run.py \
        -c configs/rd/rd_byol_mvtec.py \
        -m train \
        seed=$SEED \
        model.kwargs.momentum_schedule=$SCHEDULE \
        trainer.logdir_sub="momentum_$SCHEDULE" \
        2>&1 | tee "$LOG_DIR/momentum_$SCHEDULE.log"
done

echo "Momentum schedule comparison completed!"

# ==============================================================================
# 7. SPATIAL MATCHING ABLATION
# ==============================================================================
echo ""
echo "=============================================="
echo "7. SPATIAL MATCHING ABLATION"
echo "=============================================="

# Without spatial matching
echo "[7.1] Without spatial matching..."
CUDA_VISIBLE_DEVICES=$GPU python run.py \
    -c configs/rd/rd_byol_mvtec.py \
    -m train \
    seed=$SEED \
    "loss.loss_terms=[dict(type='CosLoss', name='cos', avg=False, lam=1.0), dict(type='BYOLDenseLoss', name='dense', lam=1.0, use_spatial_matching=False), dict(type='PrototypeInfoNCELoss', name='proto', lam=1.0, n_prototypes=5, temperature=0.07)]" \
    trainer.logdir_sub="spatial_matching_off" \
    2>&1 | tee "$LOG_DIR/spatial_matching_off.log"

# With spatial matching (default)
echo "[7.2] With spatial matching..."
CUDA_VISIBLE_DEVICES=$GPU python run.py \
    -c configs/rd/rd_byol_mvtec.py \
    -m train \
    seed=$SEED \
    trainer.logdir_sub="spatial_matching_on" \
    2>&1 | tee "$LOG_DIR/spatial_matching_on.log"

echo "Spatial matching ablation completed!"

# ==============================================================================
# 8. MULTIPLE SEEDS FOR STATISTICAL SIGNIFICANCE
# ==============================================================================
echo ""
echo "=============================================="
echo "8. MULTIPLE SEEDS (Statistical Significance)"
echo "=============================================="

for S in 42 123 456 789 1024; do
    echo "[8.x] seed = $S..."
    CUDA_VISIBLE_DEVICES=$GPU python run.py \
        -c configs/rd/rd_byol_mvtec.py \
        -m train \
        seed=$S \
        trainer.logdir_sub="seed_$S" \
        2>&1 | tee "$LOG_DIR/seed_$S.log"
done

echo "Multi-seed experiments completed!"

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo "=================================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=================================================="
echo "Results saved in: $LOG_DIR"
echo ""
echo "Experiments run:"
echo "  1. Main results: MVTec, VisA, BTAD, Real-IAD"
echo "  2. Ablation: Loss components"
echo "  3. Sensitivity: Number of prototypes (3, 5, 7, 10, 15)"
echo "  4. Sensitivity: Temperature (0.03, 0.05, 0.07, 0.1, 0.2)"
echo "  5. Sensitivity: Loss weights"
echo "  6. Momentum schedule comparison"
echo "  7. Spatial matching ablation"
echo "  8. Multiple seeds for statistical significance"
echo "=================================================="

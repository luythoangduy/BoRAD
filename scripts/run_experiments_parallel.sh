#!/bin/bash
# ==============================================================================
# BorAD: Parallel Experiment Suite (Multi-GPU)
# ==============================================================================
# Run experiments in parallel on multiple GPUs
# Usage: ./run_experiments_parallel.sh [NUM_GPUS]
# ==============================================================================

set -e

NUM_GPUS=${1:-4}
SEED=42
LOG_DIR="logs/experiments_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo "=================================================="
echo "BorAD Parallel Experiment Suite"
echo "Number of GPUs: $NUM_GPUS"
echo "Log directory: $LOG_DIR"
echo "=================================================="

# Function to run experiment on specific GPU
run_exp() {
    local gpu=$1
    local config=$2
    local name=$3
    local extra_args=$4

    echo "[GPU $gpu] Starting: $name"
    CUDA_VISIBLE_DEVICES=$gpu python run.py \
        -c $config \
        -m train \
        seed=$SEED \
        $extra_args \
        2>&1 | tee "$LOG_DIR/${name}.log"
    echo "[GPU $gpu] Completed: $name"
}

# ==============================================================================
# 1. MAIN DATASETS (Parallel on 4 GPUs)
# ==============================================================================
echo ""
echo "=============================================="
echo "1. MAIN DATASETS (Parallel)"
echo "=============================================="

run_exp 0 "configs/rd/rd_byol_mvtec.py" "mvtec" "trainer.logdir_sub=main_mvtec" &
run_exp 1 "configs/rd/rd_byol_visa.py" "visa" "trainer.logdir_sub=main_visa" &
run_exp 2 "configs/rd/rd_byol_btad.py" "btad" "trainer.logdir_sub=main_btad" &
run_exp 3 "configs/rd/rd_byol_realiad.py" "realiad" "trainer.logdir_sub=main_realiad" &

wait
echo "Main datasets completed!"

# ==============================================================================
# 2. ABLATION STUDY (Parallel)
# ==============================================================================
echo ""
echo "=============================================="
echo "2. ABLATION STUDY (Parallel)"
echo "=============================================="

# Ablation experiments
run_exp 0 "configs/rd/rd_byol_mvtec.py" "ablation_cos_only" \
    "loss.loss_terms=[dict(type='CosLoss',name='cos',avg=False,lam=1.0)] trainer.logdir_sub=ablation_cos_only" &

run_exp 1 "configs/rd/rd_byol_mvtec.py" "ablation_cos_dense" \
    "loss.loss_terms=[dict(type='CosLoss',name='cos',avg=False,lam=1.0),dict(type='BYOLDenseLoss',name='dense',lam=1.0,use_spatial_matching=True)] trainer.logdir_sub=ablation_cos_dense" &

run_exp 2 "configs/rd/rd_byol_mvtec.py" "ablation_cos_proto" \
    "loss.loss_terms=[dict(type='CosLoss',name='cos',avg=False,lam=1.0),dict(type='PrototypeInfoNCELoss',name='proto',lam=1.0,n_prototypes=5,temperature=0.07)] trainer.logdir_sub=ablation_cos_proto" &

run_exp 3 "configs/rd/rd_byol_mvtec.py" "ablation_full" \
    "trainer.logdir_sub=ablation_full" &

wait
echo "Ablation study completed!"

# ==============================================================================
# 3. PROTOTYPE SENSITIVITY (Parallel)
# ==============================================================================
echo ""
echo "=============================================="
echo "3. PROTOTYPE SENSITIVITY (Parallel)"
echo "=============================================="

gpu_idx=0
for N_PROTO in 3 5 7 10; do
    run_exp $gpu_idx "configs/rd/rd_byol_mvtec.py" "nproto_$N_PROTO" \
        "loss.loss_terms=[dict(type='CosLoss',name='cos',avg=False,lam=1.0),dict(type='BYOLDenseLoss',name='dense',lam=1.0,use_spatial_matching=True),dict(type='PrototypeInfoNCELoss',name='proto',lam=1.0,n_prototypes=$N_PROTO,temperature=0.07)] trainer.logdir_sub=nproto_$N_PROTO" &
    gpu_idx=$((($gpu_idx + 1) % $NUM_GPUS))
done

wait
echo "Prototype sensitivity completed!"

# ==============================================================================
# 4. TEMPERATURE SENSITIVITY (Parallel)
# ==============================================================================
echo ""
echo "=============================================="
echo "4. TEMPERATURE SENSITIVITY (Parallel)"
echo "=============================================="

gpu_idx=0
for TEMP in 0.03 0.05 0.07 0.1; do
    run_exp $gpu_idx "configs/rd/rd_byol_mvtec.py" "temp_$TEMP" \
        "loss.loss_terms=[dict(type='CosLoss',name='cos',avg=False,lam=1.0),dict(type='BYOLDenseLoss',name='dense',lam=1.0,use_spatial_matching=True),dict(type='PrototypeInfoNCELoss',name='proto',lam=1.0,n_prototypes=5,temperature=$TEMP)] trainer.logdir_sub=temp_$TEMP" &
    gpu_idx=$((($gpu_idx + 1) % $NUM_GPUS))
done

wait
echo "Temperature sensitivity completed!"

# ==============================================================================
# 5. MULTI-SEED (Parallel)
# ==============================================================================
echo ""
echo "=============================================="
echo "5. MULTI-SEED (Statistical Significance)"
echo "=============================================="

gpu_idx=0
for S in 42 123 456 789; do
    CUDA_VISIBLE_DEVICES=$gpu_idx python run.py \
        -c configs/rd/rd_byol_mvtec.py \
        -m train \
        seed=$S \
        trainer.logdir_sub="seed_$S" \
        2>&1 | tee "$LOG_DIR/seed_$S.log" &
    gpu_idx=$((($gpu_idx + 1) % $NUM_GPUS))
done

wait
echo "Multi-seed experiments completed!"

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo "=================================================="
echo "ALL PARALLEL EXPERIMENTS COMPLETED!"
echo "Results saved in: $LOG_DIR"
echo "=================================================="

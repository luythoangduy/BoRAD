#!/bin/bash
# ==============================================================================
# BorAD: Distributed Data Parallel Training
# ==============================================================================
# Usage: ./run_ddp.sh [CONFIG] [NUM_GPUS]
# Example: ./run_ddp.sh configs/rd/rd_byol_mvtec.py 8
# ==============================================================================

CONFIG=${1:-"configs/rd/rd_byol_mvtec.py"}
NUM_GPUS=${2:-8}
MASTER_PORT=${MASTER_PORT:-12355}

echo "=================================================="
echo "BorAD DDP Training"
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"
echo "Master Port: $MASTER_PORT"
echo "=================================================="

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    run.py \
    -c $CONFIG \
    -m train \
    "$@"

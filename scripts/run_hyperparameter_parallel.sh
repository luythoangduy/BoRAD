#!/bin/bash
# ==============================================================================
# BoRAD: Parallel Hyperparameter Sensitivity Study (4 GPUs)
# ==============================================================================
#
# Runs experiments in parallel across 4 GPUs using a job queue.
# Duplicate cases (matching base config defaults) are removed.
#
# Base config defaults (rd_byol_mvtec.py):
#   n_prototypes=5, temperature=0.07, lam_dense=1.0, lam_proto=1.0
#   momentum_schedule='linear', momentum_start=0.99, momentum_end=0.999
#
# After dedup:
#   H1: n_prototypes      → 3, 7, 10           (3 runs, removed 5=default)
#   H2: temperature       → 0.01, 0.05, 0.1, 0.5 (4 runs, removed 0.07=default)
#   H3: λ_dense           → 0.1, 0.5, 2.0, 5.0  (4 runs, removed 1.0=default)
#   H4: λ_proto           → 0.1, 0.5, 2.0, 5.0  (4 runs, removed 1.0=default)
#   H5: momentum sched×val → 8 runs             (removed linear+0.999=default)
#   Total: 23 runs
#
# Usage:
#   bash scripts/run_hyperparameter_parallel.sh
#   NGPUS=2 bash scripts/run_hyperparameter_parallel.sh
#   STUDY=H1 bash scripts/run_hyperparameter_parallel.sh
# ==============================================================================

set -e

NGPUS=${NGPUS:-4}
SEED=${SEED:-42}
STUDY=${STUDY:-all}
BASE_CFG="configs/rd/rd_byol_mvtec.py"

# If RESUME_DIR is provided, use it. Otherwise, create a new TIMESTAMP dir.
if [ -n "$RESUME_DIR" ]; then
    LOG_DIR="$RESUME_DIR"
    echo "================================================================"
    echo "  RESUMING BoRAD Hyperparameter Study"
    echo "  Resuming from: $LOG_DIR"
    echo "================================================================"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="logs/hyperparam_parallel_${TIMESTAMP}"
    mkdir -p "$LOG_DIR"
    echo "================================================================"
    echo "  BoRAD Hyperparameter Study (Parallel, ${NGPUS} GPUs)"
    echo "  Seed: $SEED | Study: $STUDY"
    echo "  Log dir: $LOG_DIR"
    echo "================================================================"
fi

# ──────────────────────────────────────────────────────────────────────
# Job queue: collect all experiments, then dispatch across GPUs
# ──────────────────────────────────────────────────────────────────────
JOBS=()  # Array of "name|opts" strings

add_job() {
    local name=$1
    shift
    local opts="$*"
    JOBS+=("${name}|${opts}")
}

# ══════════════════════════════════════════════════════════════════════
# H1: Number of Prototypes (default=5, skip 5)
# ══════════════════════════════════════════════════════════════════════
if [[ "$STUDY" == "all" || "$STUDY" == "H1" ]]; then
    for N in 3 7 10; do
        add_job "H1_nproto_${N}" loss.loss_terms.2.n_prototypes=${N}
    done
fi

# ══════════════════════════════════════════════════════════════════════
# H2: Temperature (default=0.07, skip 0.07)
# ══════════════════════════════════════════════════════════════════════
if [[ "$STUDY" == "all" || "$STUDY" == "H2" ]]; then
    for T in 0.01 0.05 0.1 0.5; do
        add_job "H2_temp_${T}" loss.loss_terms.2.temperature=${T}
    done
fi

# ══════════════════════════════════════════════════════════════════════
# H3: Masking Ratio (mask_ratio)
# ══════════════════════════════════════════════════════════════════════
if [[ "$STUDY" == "all" || "$STUDY" == "H3" ]]; then
    declare -a MASK_RATIOS=(0.1 0.3 0.5 0.7 0.9)
    for RATIO in "${MASK_RATIOS[@]}"; do
        # We test this with standard settings (lam_proto=1.0)
        add_job "H3_mask_${RATIO}" \
            loss.kwargs_proto.n_prototypes=10 \
            model.kwargs.mask_ratio=${RATIO} \
            loss_terms.proto.lam=1.0
    done
fi

# ══════════════════════════════════════════════════════════════════════
# H4: Proto loss weight (default=1.0, skip 1.0)
# ══════════════════════════════════════════════════════════════════════
if [[ "$STUDY" == "all" || "$STUDY" == "H4" ]]; then
    for L in 0.1 0.5 2.0 5.0; do
        add_job "H4_lam_proto_${L}" loss.loss_terms.2.lam=${L}
    done
fi

# ══════════════════════════════════════════════════════════════════════
# H5: Momentum schedule × value (skip linear+0.999 = default)
# ══════════════════════════════════════════════════════════════════════
if [[ "$STUDY" == "all" || "$STUDY" == "H5" ]]; then
    for SCHED in constant linear cosine; do
        if [[ "$SCHED" == "constant" ]]; then
            for MOM in 0.9 0.99 0.999; do
                add_job "H5_mom_${SCHED}_${MOM}" \
                    model.kwargs.momentum=${MOM} \
                    model.kwargs.momentum_schedule=${SCHED} \
                    model.kwargs.momentum_start=${MOM} \
                    model.kwargs.momentum_end=${MOM}
            done
        else
            # For linear/cosine: test specific start->end ranges
            for MOM_START in 0.9 0.99; do
                MOM_END=0.999
                
                if [[ "$SCHED" == "linear" && "$MOM_START" == "0.99" ]]; then
                    echo "  [SKIP] H5_mom_${SCHED}_${MOM_START}_${MOM_END} (= base config default)"
                    continue
                fi

                add_job "H5_mom_${SCHED}_${MOM_START}_${MOM_END}" \
                    model.kwargs.momentum=${MOM_END} \
                    model.kwargs.momentum_schedule=${SCHED} \
                    model.kwargs.momentum_start=${MOM_START} \
                    model.kwargs.momentum_end=${MOM_END}
            done
        fi
    done
fi

# ══════════════════════════════════════════════════════════════════════
# H6: Spatial Proto Loss Weight (lam_spatial)
# ══════════════════════════════════════════════════════════════════════
if [[ "$STUDY" == "all" || "$STUDY" == "H6" ]]; then
    for L in 0 0.1 0.5 1.0 2.0; do
        add_job "H6_lam_spatial_${L}" loss.loss_terms.1.lam_spatial=${L}
    done
fi

# ══════════════════════════════════════════════════════════════════════
# H7: Global Proto Loss Weight (lam_global)
# ══════════════════════════════════════════════════════════════════════
if [[ "$STUDY" == "all" || "$STUDY" == "H7" ]]; then
    for L in 0 0.1 0.5 1.0 2.0; do
        add_job "H7_lam_global_${L}" loss.loss_terms.1.lam_global=${L}
    done
fi

# ──────────────────────────────────────────────────────────────────────
# Dispatch: Round-robin job assignment across GPUs
# ──────────────────────────────────────────────────────────────────────
TOTAL_JOBS=${#JOBS[@]}
echo ""
echo "Total jobs: $TOTAL_JOBS (across $NGPUS GPUs)"
echo "================================================================"

# Track PIDs per GPU slot
declare -A GPU_PIDS

run_on_gpu() {
    local gpu=$1
    local name=$2
    local opts=$3

    # Skip if job is already done (by checking if log exists and has "Done" or if we just want to skip any existing log)
    # We will simply check if the log file exists and has size > 0. 
    # For more robust check, you could grep for "Training completed" in the log.
    if [ -n "$RESUME_DIR" ] && [ -f "$LOG_DIR/${name}.log" ]; then
        if grep -q "best metric" "$LOG_DIR/${name}.log"; then
           echo "  [GPU $gpu] ⏭ $name (already completed in $LOG_DIR)"
           return
        fi
    fi

    echo "  [GPU $gpu] ▶ $name"
    CUDA_VISIBLE_DEVICES=$gpu python run.py \
        -c "$BASE_CFG" \
        -m train \
        seed=$SEED \
        trainer.logdir_sub="hyperparam_${name}" \
        $opts \
        2>&1 | tee "$LOG_DIR/${name}.log"
    echo "  [GPU $gpu] ✓ $name"
}

# Process jobs in batches of NGPUS
batch=0
while [ $batch -lt $TOTAL_JOBS ]; do
    echo ""
    echo "──── Batch $((batch / NGPUS + 1)) ────"

    # Launch up to NGPUS jobs in parallel
    pids=()
    for ((i=0; i<NGPUS && batch+i<TOTAL_JOBS; i++)); do
        idx=$((batch + i))
        gpu=$i

        # Parse job: "name|opts"
        IFS='|' read -r name opts <<< "${JOBS[$idx]}"

        run_on_gpu $gpu "$name" "$opts" &
        pids+=($!)
    done

    # Wait for all jobs in this batch to finish
    for pid in "${pids[@]}"; do
        wait $pid
    done

    batch=$((batch + NGPUS))
done

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  HYPERPARAMETER STUDY COMPLETED"
echo "================================================================"
echo "  Total runs: $TOTAL_JOBS"
echo "  GPUs used: $NGPUS"
echo "  Results: $LOG_DIR"
echo ""
echo "  Studies (after dedup):"
echo "    H1: n_prototypes   = {3, 7, 10}              (3 runs)"
echo "    H2: temperature    = {0.01, 0.05, 0.1, 0.5}  (4 runs)"
echo "    H3: λ_dense        = {0.1, 0.5, 2.0, 5.0}    (4 runs)"
echo "    H4: λ_proto        = {0.1, 0.5, 2.0, 5.0}    (4 runs)"
echo "    H5: momentum       = 3×3 - 1 default          (8 runs)"
echo "================================================================"

#!/bin/bash
# ==============================================================================
# BoRAD: Hyperparameter Sensitivity Analysis
# ==============================================================================
#
# Studies (all on MVTec with full model config):
#   H1: Number of prototypes       — 3, 5, 7, 10
#   H2: Temperature (Proto)        — 0.01, 0.05, 0.07, 0.1, 0.5
#   H3: Dense loss weight (λ_d)    — 0.1, 0.5, 1.0, 2.0, 5.0
#   H4: Proto loss weight (λ_p)    — 0.1, 0.5, 1.0, 2.0, 5.0
#   H5: Momentum (schedule × val)  — {constant,linear,cosine} × {0.9,0.99,0.999}
#
# Total: 4 + 5 + 5 + 5 + 9 = 28 runs
#
# Base config loss_terms ordering:
#   [0] = CosLoss (name='cos')
#   [1] = BYOLDenseLoss (name='dense')
#   [2] = PrototypeInfoNCELoss (name='proto')
#
# Config overrides use dot-notation with list indexing:
#   loss.loss_terms.2.n_prototypes=10  → changes proto's n_prototypes to 10
#
# Usage:
#   GPU=0 bash scripts/run_hyperparameter_study.sh
#   GPU=0 STUDY=H1 bash scripts/run_hyperparameter_study.sh   # run only H1
#   GPU=0 STUDY=H5 bash scripts/run_hyperparameter_study.sh   # run only H5
# ==============================================================================

set -e

# Configuration
GPU=${GPU:-0}
SEED=${SEED:-42}
STUDY=${STUDY:-all}  # all, H1, H2, H3, H4, H5
BASE_CFG="configs/rd/rd_byol_mvtec.py"

if [ -n "$RESUME_DIR" ]; then
    LOG_DIR="$RESUME_DIR"
    echo "================================================================"
    echo "  RESUMING BoRAD Hyperparameter Sensitivity Analysis"
    echo "  GPU: $GPU | Seed: $SEED | Study: $STUDY"
    echo "  Resuming from: $LOG_DIR"
    echo "================================================================"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="logs/hyperparam_${TIMESTAMP}"
    mkdir -p "$LOG_DIR"

    echo "================================================================"
    echo "  BoRAD Hyperparameter Sensitivity Analysis"
    echo "  GPU: $GPU | Seed: $SEED | Study: $STUDY"
    echo "  Log dir: $LOG_DIR"
    echo "================================================================"
fi

# Helper function
run_exp() {
    local name=$1
    shift
    local extra_opts="$@"

    if [ -n "$RESUME_DIR" ] && [ -f "$LOG_DIR/${name}.log" ]; then
        if grep -q "best metric" "$LOG_DIR/${name}.log"; then
           echo "--------------------------------------------------------------"
           echo "  [$name] ⏭ SKIPPED (already completed in $LOG_DIR)"
           echo "--------------------------------------------------------------"
           return
        fi
    fi

    echo ""
    echo "--------------------------------------------------------------"
    echo "  [$name]"
    echo "  Opts: $extra_opts"
    echo "--------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=$GPU python run.py \
        -c "$BASE_CFG" \
        -m train \
        seed=$SEED \
        trainer.logdir_sub="hyperparam_${name}" \
        $extra_opts \
        2>&1 | tee "$LOG_DIR/${name}.log"

    echo "  [$name] Done."
}

# ==============================================================================
# H1: Number of Prototypes (loss_terms[1] = PrototypeInfoNCELoss)
# ==============================================================================
if [[ "$STUDY" == "all" || "$STUDY" == "H1" ]]; then
    echo ""
    echo "=============================================="
    echo "  H1: Number of Prototypes"
    echo "=============================================="

    for N_PROTO in 3 5 7 10; do
        run_exp "H1_nproto_${N_PROTO}" \
            loss.loss_terms.1.n_prototypes=${N_PROTO}
    done
fi

# ==============================================================================
# H2: Temperature (loss_terms[1] = PrototypeInfoNCELoss)
# ==============================================================================
if [[ "$STUDY" == "all" || "$STUDY" == "H2" ]]; then
    echo ""
    echo "=============================================="
    echo "  H2: Temperature (Prototype InfoNCE)"
    echo "=============================================="

    for TEMP in 0.01 0.05 0.07 0.1 0.5; do
        run_exp "H2_temp_${TEMP}" \
            loss.loss_terms.1.temperature=${TEMP}
    done
fi

# ==============================================================================
# H3: Masking Ratio
# ==============================================================================
if [[ "$STUDY" == "all" || "$STUDY" == "H3" ]]; then
    echo ""
    echo "=============================================="
    echo "  H3: Masking Ratio"
    echo "=============================================="

    for RATIO in 0.1 0.3 0.5 0.7 0.9; do
        run_exp "H3_mask_${RATIO}" \
            model.kwargs.mask_ratio=${RATIO}
    done
fi

# ==============================================================================
# H4: Proto Loss Weight (loss_terms[2] = PrototypeInfoNCELoss)
# ==============================================================================
# if [[ "$STUDY" == "all" || "$STUDY" == "H4" ]]; then
#     echo ""
#     echo "=============================================="
#     echo "  H4: Proto Loss Weight (λ_proto)"
#     echo "=============================================="

#     for LAM_P in 0.1 0.5 1.0 2.0 5.0; do
#         run_exp "H4_lam_proto_${LAM_P}" \
#             loss.loss_terms.1.lam=${LAM_P}
#     done
# fi

# ==============================================================================
# H5: Momentum Schedule × Value (3×3 = 9 runs)
# ==============================================================================
if [[ "$STUDY" == "all" || "$STUDY" == "H5" ]]; then
    echo ""
    echo "=============================================="
    echo "  H5: Momentum (Schedule × Value)"
    echo "=============================================="

    for SCHEDULE in constant linear cosine; do
        for MOM_VAL in 0.9 0.99 0.999; do
            if [[ "$SCHEDULE" == "constant" ]]; then
                # For constant: start = end = MOM_VAL
                run_exp "H5_mom_${SCHEDULE}_${MOM_VAL}" \
                    model.kwargs.momentum=${MOM_VAL} \
                    model.kwargs.momentum_schedule=${SCHEDULE} \
                    model.kwargs.momentum_start=${MOM_VAL} \
                    model.kwargs.momentum_end=${MOM_VAL}
            else
                # For scheduled: start lower, end at MOM_VAL
                MOM_START=$(echo "$MOM_VAL" | awk '{printf "%.4f", $1 * 0.9}')
                run_exp "H5_mom_${SCHEDULE}_${MOM_VAL}" \
                    model.kwargs.momentum=${MOM_VAL} \
                    model.kwargs.momentum_schedule=${SCHEDULE} \
                    model.kwargs.momentum_start=${MOM_START} \
                    model.kwargs.momentum_end=${MOM_VAL}
            fi
        done
    done
fi

# ==============================================================================
# H6: Spatial Proto Loss Weight (lam_spatial)
# ==============================================================================
if [[ "$STUDY" == "all" || "$STUDY" == "H6" ]]; then
    echo ""
    echo "=============================================="
    echo "  H6: Spatial Proto Loss Weight (lam_spatial)"
    echo "=============================================="

    for LAM_S in 0.1 0.5 1.0 2.0 5.0; do
        run_exp "H6_lam_spatial_${LAM_S}" \
            loss.loss_terms.1.lam_spatial=${LAM_S}
    done
fi

# ==============================================================================
# H7: Global Proto Loss Weight (lam_global)
# ==============================================================================
if [[ "$STUDY" == "all" || "$STUDY" == "H7" ]]; then
    echo ""
    echo "=============================================="
    echo "  H7: Global Proto Loss Weight (lam_global)"
    echo "=============================================="

    for LAM_G in 0.1 0.5 1.0 2.0 5.0; do
        run_exp "H7_lam_global_${LAM_G}" \
            loss.loss_terms.1.lam_global=${LAM_G}
    done
fi

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo "================================================================"
echo "  EXPERIMENTS FINISHED"
echo "================================================================"
echo "  Results saved in: $LOG_DIR"
echo ""
echo "  Studies:"
echo "    H1: n_prototypes     = {3, 5, 7, 10}                 (4 runs)"
echo "    H2: temperature      = {0.01, 0.05, 0.07, 0.1, 0.5}  (5 runs)"
echo "    H3: λ_dense          = {0.1, 0.5, 1.0, 2.0, 5.0}     (5 runs)"
echo "    H4: λ_proto          = {0.1, 0.5, 1.0, 2.0, 5.0}     (5 runs)"
echo "    H5: momentum sched×val = 3×3                          (9 runs)"
echo "    Total: 28 runs"
echo "================================================================"

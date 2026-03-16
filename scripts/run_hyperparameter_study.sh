#!/bin/bash
# ==============================================================================
# BoRAD: Hyperparameter Sensitivity Analysis
# ==============================================================================
#
# Studies (all on MVTec with full model config):
#   H1: Number of prototypes       — 3, 5, 7, 10
#   H2: Temperature (Proto)        — 0.01, 0.05, 0.07, 0.1, 0.5
#   H3: Masking Ratio              — 0.1, 0.3, 0.5, 0.7, 0.9
#   H4: Proto loss weight          — 0.1, 0.5, 1.0, 2.0, 5.0
#   H5: Momentum (schedule × val)  — {constant,linear,cosine} × {0.9,0.99,0.999}
#   H6: Spatial Proto Loss Weight  — 0, 0.1, 0.5, 1.0, 2.0
#   H7: Global Proto Loss Weight   — 0, 0.1, 0.5, 1.0, 2.0
#
# Usage:
#   GPU=0 bash scripts/run_hyperparameter_study.sh
#   GPU=0 STUDY=H1 bash scripts/run_hyperparameter_study.sh   # run only H1
# ==============================================================================

set -e

# Configuration
GPU=${GPU:-0}
SEED=${SEED:-42}
STUDY=${STUDY:-all}  # all, H1, H2, H3, H4, H5, H6, H7
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
# H1: Number of Prototypes
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
# H2: Temperature (Prototype InfoNCE)
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
# H4: Proto Loss Weight (λ_proto)
# ==============================================================================
if [[ "$STUDY" == "all" || "$STUDY" == "H4" ]]; then
    echo ""
    echo "=============================================="
    echo "  H4: Proto Loss Weight (λ_proto)"
    echo "=============================================="

    for LAM_P in 0.1 0.5 1.0 2.0 5.0; do
        run_exp "H4_lam_proto_${LAM_P}" \
            loss.loss_terms.1.lam=${LAM_P}
    done
fi

# ==============================================================================
# H5: Momentum Schedule × Value
# ==============================================================================
if [[ "$STUDY" == "all" || "$STUDY" == "H5" ]]; then
    echo ""
    echo "=============================================="
    echo "  H5: Momentum (Schedule × Value)"
    echo "=============================================="

    for SCHEDULE in constant linear cosine; do
        if [[ "$SCHEDULE" == "constant" ]]; then
            # For constant: just test different absolute momentum values
            for MOM_VAL in 0.9 0.99 0.999; do
                run_exp "H5_mom_${SCHEDULE}_${MOM_VAL}" \
                    model.kwargs.momentum=${MOM_VAL} \
                    model.kwargs.momentum_schedule=${SCHEDULE} \
                    model.kwargs.momentum_start=${MOM_VAL} \
                    model.kwargs.momentum_end=${MOM_VAL}
            done
        else
            # For scheduled (linear/cosine), test explicit start -> end ranges
            # 0.9 -> 0.999 and 0.99 -> 0.999
            for MOM_START in 0.9 0.99; do
                MOM_END=0.999
                
                run_exp "H5_mom_${SCHEDULE}_${MOM_START}_${MOM_END}" \
                    model.kwargs.momentum=${MOM_END} \
                    model.kwargs.momentum_schedule=${SCHEDULE} \
                    model.kwargs.momentum_start=${MOM_START} \
                    model.kwargs.momentum_end=${MOM_END}
            done
        fi
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

    for LAM_S in 0 0.1 0.5 1.0 2.0; do
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

    for LAM_G in 0 0.1 0.5 1.0 2.0; do
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

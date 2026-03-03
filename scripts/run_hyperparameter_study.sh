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
#   H5: Momentum (schedule × val)  — {static,linear,cosine} × {0.9,0.99,0.999}
#
# Total: 4 + 5 + 5 + 5 + 9 = 28 runs
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
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/hyperparam_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  BoRAD Hyperparameter Sensitivity Analysis"
echo "  GPU: $GPU | Seed: $SEED | Study: $STUDY"
echo "  Log dir: $LOG_DIR"
echo "================================================================"

# Helper function
run_exp() {
    local name=$1
    shift
    local extra_opts="$@"

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
            "loss.loss_terms=[dict(type='CosLoss',name='cos',avg=False,lam=1.0),dict(type='BYOLDenseLoss',name='dense',lam=1.0,use_spatial_matching=True),dict(type='PrototypeInfoNCELoss',name='proto',lam=1.0,n_prototypes=${N_PROTO},temperature=0.07)]"
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
            "loss.loss_terms=[dict(type='CosLoss',name='cos',avg=False,lam=1.0),dict(type='BYOLDenseLoss',name='dense',lam=1.0,use_spatial_matching=True),dict(type='PrototypeInfoNCELoss',name='proto',lam=1.0,n_prototypes=5,temperature=${TEMP})]"
    done
fi

# ==============================================================================
# H3: Dense Loss Weight (λ_dense)
# ==============================================================================
if [[ "$STUDY" == "all" || "$STUDY" == "H3" ]]; then
    echo ""
    echo "=============================================="
    echo "  H3: Dense Loss Weight (λ_dense)"
    echo "=============================================="

    for LAM_D in 0.1 0.5 1.0 2.0 5.0; do
        run_exp "H3_lam_dense_${LAM_D}" \
            "loss.loss_terms=[dict(type='CosLoss',name='cos',avg=False,lam=1.0),dict(type='BYOLDenseLoss',name='dense',lam=${LAM_D},use_spatial_matching=True),dict(type='PrototypeInfoNCELoss',name='proto',lam=1.0,n_prototypes=5,temperature=0.07)]"
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
            "loss.loss_terms=[dict(type='CosLoss',name='cos',avg=False,lam=1.0),dict(type='BYOLDenseLoss',name='dense',lam=1.0,use_spatial_matching=True),dict(type='PrototypeInfoNCELoss',name='proto',lam=${LAM_P},n_prototypes=5,temperature=0.07)]"
    done
fi

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
            # For 'constant' schedule, momentum value is used directly
            # For 'linear'/'cosine', momentum_start=0.9 or lower, momentum_end=MOM_VAL
            if [[ "$SCHEDULE" == "constant" ]]; then
                run_exp "H5_mom_${SCHEDULE}_${MOM_VAL}" \
                    "model.kwargs.momentum=${MOM_VAL}" \
                    "model.kwargs.momentum_schedule=${SCHEDULE}" \
                    "model.kwargs.momentum_start=${MOM_VAL}" \
                    "model.kwargs.momentum_end=${MOM_VAL}"
            else
                # For scheduled: start lower, end at MOM_VAL
                # Use a reasonable start based on end value
                MOM_START=$(echo "$MOM_VAL" | awk '{printf "%.4f", $1 * 0.9}')
                run_exp "H5_mom_${SCHEDULE}_${MOM_VAL}" \
                    "model.kwargs.momentum=${MOM_VAL}" \
                    "model.kwargs.momentum_schedule=${SCHEDULE}" \
                    "model.kwargs.momentum_start=${MOM_START}" \
                    "model.kwargs.momentum_end=${MOM_VAL}"
            fi
        done
    done
fi

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo "================================================================"
echo "  HYPERPARAMETER STUDY COMPLETED"
echo "================================================================"
echo "  Results saved in: $LOG_DIR"
echo ""
echo "  Studies:"
echo "    H1: n_prototypes     = {3, 5, 7, 10}         (4 runs)"
echo "    H2: temperature      = {0.01, 0.05, 0.07, 0.1, 0.5} (5 runs)"
echo "    H3: λ_dense          = {0.1, 0.5, 1.0, 2.0, 5.0}    (5 runs)"
echo "    H4: λ_proto          = {0.1, 0.5, 1.0, 2.0, 5.0}    (5 runs)"
echo "    H5: momentum sched×val = 3×3                         (9 runs)"
echo "    Total: 28 runs"
echo "================================================================"

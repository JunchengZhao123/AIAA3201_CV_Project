#!/usr/bin/env bash
set -euo pipefail

# ================================================================
#  Part 3: Full Pipeline - Generative Enhancement
# ================================================================
#
#  Runs the complete Part 3 pipeline:
#    Step 1: Generate pseudo-views via Difix3D
#    Step 2: Compute confidence masks
#    Step 3: Hybrid training (3 modes for ablation)
#    Step 4: Evaluate and compare
#
#  Usage:
#    bash scripts/scripts_p3/run_full_pipeline_p3.sh
#    bash scripts/scripts_p3/run_full_pipeline_p3.sh --dataset waymo
#    bash scripts/scripts_p3/run_full_pipeline_p3.sh --skip_pseudo --skip_confidence
#    bash scripts/scripts_p3/run_full_pipeline_p3.sh --skip_difix   # raw renders only
#
#  Environment:
#    conda activate S3PO-GS           Required conda env (from Part 2)
#    S3PO_ROOT=/path/to/S3PO-GS     (default: $PROJ/S3PO-GS)
#    GPU_ID=0                         GPU device
#    DIFIX3D_ROOT=/path/to/Difix3D   (default: $PROJ/Difix3D)
#
# ================================================================

SCRIPTS_P3="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "${CV_PROJ_ROOT:-"$SCRIPTS_P3/../.."}" && pwd)"

S3PO_ROOT="${S3PO_ROOT:-$PROJ/S3PO-GS}"
GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-all}"

# S3PO-GS must be on PYTHONPATH for Gaussian rendering imports
export PYTHONPATH="${S3PO_ROOT}:${PYTHONPATH:-}"
SKIP_PSEUDO=0
SKIP_CONFIDENCE=0
SKIP_TRAIN=0
SKIP_EVAL=0
SKIP_DIFIX=0
N_INTERP="${N_INTERP:-2}"
BLEND_MODE="${BLEND_MODE:-rife_gs}"
TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-15000}"
BETA_MAX="${BETA_MAX:-0.5}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)          DATASET="$2"; shift 2;;
        --skip_pseudo)      SKIP_PSEUDO=1; shift;;
        --skip_confidence)  SKIP_CONFIDENCE=1; shift;;
        --skip_train)       SKIP_TRAIN=1; shift;;
        --skip_eval)        SKIP_EVAL=1; shift;;
        --skip_difix)       SKIP_DIFIX=1; shift;;
        --n_interp)         N_INTERP="$2"; shift 2;;
        --blend_mode)       BLEND_MODE="$2"; shift 2;;
        --iterations)       TOTAL_ITERATIONS="$2"; shift 2;;
        --beta_max)         BETA_MAX="$2"; shift 2;;
        --gpu)              GPU_ID="$2"; shift 2;;
        *)                  echo "Unknown arg: $1"; exit 1;;
    esac
done

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "================================================================"
echo "  Part 3: Generative Enhancement - Full Pipeline"
echo "================================================================"
echo ""
echo "  Project root: $PROJ"
echo "  S3PO-GS:     $S3PO_ROOT"
echo "  GPU:         $GPU_ID"
echo "  Dataset:     $DATASET"
echo "  N interp:    $N_INTERP"
echo "  Blend mode:  $BLEND_MODE"
echo "  Iterations:  $TOTAL_ITERATIONS"
echo "  Skip Difix:  $SKIP_DIFIX"
echo ""

RESULTS_BASE="$PROJ/running_result/5_3"
mkdir -p "$RESULTS_BASE"

CKPT_DIR="$PROJ/checkpoints/difix3d"
DIFIX_FLAG=""
if [ "$SKIP_DIFIX" = "1" ]; then
    DIFIX_FLAG="--skip_difix"
fi

# Build dataset list
DATASETS=()
if [ "$DATASET" = "all" ]; then
    DATASETS=("waymo" "dl3dv" "re10k")
else
    DATASETS=("$DATASET")
fi

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "================================================================"
    echo "  Processing: $DS"
    echo "================================================================"

    DS_NAMES_waymo="waymo_405841"
    DS_NAMES_dl3dv="dl3dv_2"
    DS_NAMES_re10k="re10k_1"
    eval "DS_NAME=\$DS_NAMES_$DS"

    PSEUDO_DIR="$RESULTS_BASE/pseudo_views/$DS_NAME"
    CONF_DIR="$RESULTS_BASE/confidence_masks/$DS_NAME"

    # ============================================================
    # Step 1: Generate pseudo-views
    # ============================================================
    if [ "$SKIP_PSEUDO" = "0" ]; then
        echo ""
        echo "--- [Step 1/4] Generating pseudo-views ($DS) ---"

        python "$SCRIPTS_P3/generate_pseudo_views.py" \
            --dataset_type "$DS" \
            --n_interp "$N_INTERP" \
            --blend_mode "$BLEND_MODE" \
            --proj_root "$PROJ" \
            $DIFIX_FLAG
    else
        echo "--- [Step 1/4] SKIPPED (--skip_pseudo) ---"
    fi

    # ============================================================
    # Step 2: Compute confidence masks
    # ============================================================
    if [ "$SKIP_CONFIDENCE" = "0" ]; then
        echo ""
        echo "--- [Step 2/4] Computing confidence masks ($DS) ---"
        python "$SCRIPTS_P3/compute_confidence_masks.py" \
            --dataset_type "$DS" \
            --proj_root "$PROJ"
    else
        echo "--- [Step 2/4] SKIPPED (--skip_confidence) ---"
    fi

    # ============================================================
    # Step 3: Hybrid training (all 3 modes for ablation)
    # ============================================================
    if [ "$SKIP_TRAIN" = "0" ]; then
        echo ""
        echo "--- [Step 3/4] Hybrid 3DGS Training ($DS) ---"

        for MODE in "sparse_only" "sparse_pseudo" "sparse_pseudo_confidence"; do
            echo ""
            echo "  Mode: $MODE"
            python "$SCRIPTS_P3/hybrid_train_3dgs.py" \
                --dataset_type "$DS" \
                --mode "$MODE" \
                --total_iterations "$TOTAL_ITERATIONS" \
                --beta_max "$BETA_MAX" \
                --proj_root "$PROJ"
        done
    else
        echo "--- [Step 3/4] SKIPPED (--skip_train) ---"
    fi

    # ============================================================
    # Step 4: Evaluate
    # ============================================================
    if [ "$SKIP_EVAL" = "0" ]; then
        echo ""
        echo "--- [Step 4/4] Evaluating ($DS) ---"

        python "$SCRIPTS_P3/evaluate_p3.py" \
            --datasets "$DS" \
            --proj_root "$PROJ"
    else
        echo "--- [Step 4/4] SKIPPED (--skip_eval) ---"
    fi
done

echo ""
echo "================================================================"
echo "  Part 3 Pipeline Complete!"
echo "================================================================"
echo "  Results: $RESULTS_BASE/"
echo ""
echo "  Directory structure:"
echo "    pseudo_views/      - Generated pseudo-views per dataset"
echo "    confidence_masks/  - Confidence masks per dataset"
echo "    hybrid_train/      - Fine-tuned models (3 modes per dataset)"
echo "    evaluation/        - Metrics + comparison visualizations"
echo "================================================================"

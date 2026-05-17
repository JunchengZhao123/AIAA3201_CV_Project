#!/usr/bin/env bash
set -euo pipefail

# ================================================================
#  Part 2: Full Pipeline - Unposed Sparse Reconstruction
# ================================================================
#
#  Runs the complete Part 2 pipeline:
#    Step 1: Sub-sample datasets to sparse frames
#    Step 2: Patch S3PO-GS and link data
#    Step 3: Run S3PO-GS SLAM on all datasets
#    Step 4: Evaluate results
#
#  Usage:
#    bash scripts/scripts_p2/run_full_pipeline.sh
#    bash scripts/scripts_p2/run_full_pipeline.sh --skip_subsample
#    bash scripts/scripts_p2/run_full_pipeline.sh --dataset waymo
#
#  Layout (defaults):
#    Dense data:   $CV_PROJ_ROOT/dataset/  (405841/FRONT, DL3DV-2, Re10k-1)
#    Sparse out:   $DATASET_SPARSE_P2      (default: $CV_PROJ_ROOT/dataset_sparse_p2)
#
#  Environment:
#    S3PO_ROOT=/path/to/S3PO-GS   (default: $PROJ/S3PO-GS)
#    GPU_ID=0                       GPU device
#
# ================================================================

PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
S3PO_ROOT="${S3PO_ROOT:-$PROJ/S3PO-GS}"
GPU_ID="${GPU_ID:-0}"
SKIP_SUBSAMPLE=0
DATASET="all"
WAYMO_SPARSITY="${WAYMO_SPARSITY:-10}"
DOWNSCALE="${DOWNSCALE:-1}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_subsample) SKIP_SUBSAMPLE=1; shift;;
        --dataset) DATASET="$2"; shift 2;;
        --waymo_sparsity) WAYMO_SPARSITY="$2"; shift 2;;
        --downscale) DOWNSCALE="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

echo "================================================================"
echo "  Part 2: Unposed Sparse Reconstruction - Full Pipeline"
echo "================================================================"
echo ""
echo "  Project root:     $PROJ"
echo "  S3PO-GS:         $S3PO_ROOT"
echo "  GPU:             $GPU_ID"
echo "  Dataset:         $DATASET"
echo "  Waymo sparsity:  1/$WAYMO_SPARSITY"
echo "  Downscale:       ${DOWNSCALE}x"
echo ""

# ============================================================
# Step 1: Sub-sample to sparse frames
# ============================================================
if [ "$SKIP_SUBSAMPLE" = "0" ]; then
    echo "--- [Step 1/4] Sub-sampling datasets to sparse frames ---"
    SUBSAMPLE_ARGS="--proj_root $PROJ --waymo_sparsity $WAYMO_SPARSITY --downscale $DOWNSCALE"
    if [ "$DATASET" = "all" ]; then
        python "$PROJ/scripts/scripts_p2/subsample_sparse.py" $SUBSAMPLE_ARGS
    else
        python "$PROJ/scripts/scripts_p2/subsample_sparse.py" $SUBSAMPLE_ARGS --datasets "$DATASET"
    fi
    echo ""
else
    echo "--- [Step 1/4] SKIPPED (--skip_subsample) ---"
fi

# ============================================================
# Step 2: Setup S3PO-GS (patch + link data)
# ============================================================
echo "--- [Step 2/4] Patching S3PO-GS and linking sparse data ---"

# Patch S3PO-GS
python "$PROJ/scripts/scripts_p2/add_re10k_parser.py" --s3po_root "$S3PO_ROOT"

# Link sparse datasets into S3PO-GS
SPARSE_DATA="$PROJ/dataset_sparse_p2"
mkdir -p "$S3PO_ROOT/datasets"
if [ -d "$SPARSE_DATA/waymo" ] && [ ! -e "$S3PO_ROOT/datasets/waymo_sparse" ]; then
    ln -sf "$SPARSE_DATA/waymo" "$S3PO_ROOT/datasets/waymo_sparse"
    echo "  Linked: waymo_sparse"
fi
if [ -d "$SPARSE_DATA/dl3dv" ] && [ ! -e "$S3PO_ROOT/datasets/dl3dv_sparse" ]; then
    ln -sf "$SPARSE_DATA/dl3dv" "$S3PO_ROOT/datasets/dl3dv_sparse"
    echo "  Linked: dl3dv_sparse"
fi
if [ -d "$SPARSE_DATA/re10k" ] && [ ! -e "$S3PO_ROOT/datasets/re10k_sparse" ]; then
    ln -sf "$SPARSE_DATA/re10k" "$S3PO_ROOT/datasets/re10k_sparse"
    echo "  Linked: re10k_sparse"
fi
echo ""

# ============================================================
# Step 3: Run S3PO-GS SLAM
# ============================================================
echo "--- [Step 3/4] Running S3PO-GS SLAM ---"
export S3PO_ROOT GPU_ID
bash "$PROJ/scripts/scripts_p2/run_s3po_all.sh" "$DATASET"
echo ""

# ============================================================
# Step 4: Evaluate
# ============================================================
echo "--- [Step 4/4] Evaluating results ---"
if [ "$DATASET" = "all" ]; then
    python "$PROJ/scripts/scripts_p2/evaluate_p2.py" --proj_root "$PROJ"
else
    python "$PROJ/scripts/scripts_p2/evaluate_p2.py" --proj_root "$PROJ" --datasets "$DATASET"
fi

echo ""
echo "================================================================"
echo "  Part 2 Pipeline Complete!"
echo "  Results: $PROJ/running_result/5_2/S3PO_GS_SLAM/"
echo "================================================================"

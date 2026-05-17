#!/usr/bin/env bash
set -euo pipefail

# ================================================================
#  Part 2: Run S3PO-GS Monocular SLAM on Sparse Frames
# ================================================================
#
#  Usage:
#    bash scripts/scripts_p2/run_s3po_all.sh              # Run all 3 datasets
#    bash scripts/scripts_p2/run_s3po_all.sh waymo        # Waymo only
#    bash scripts/scripts_p2/run_s3po_all.sh dl3dv        # DL3DV only
#    bash scripts/scripts_p2/run_s3po_all.sh re10k        # Re10k only
#
#  Layout (defaults):
#    Raw data:           $CV_PROJ_ROOT/dataset/   (same as subsample_sparse.py expects)
#    Sparse subsampled:  $DATASET_SPARSE_P2       (default: $CV_PROJ_ROOT/dataset_sparse_p2)
#
#  Environment variables:
#    CV_PROJ_ROOT=/path/to/home_or_project    Default: parent of scripts/ (../.. from this dir)
#    DATASET_SPARSE_P2=/path/to/sparse_root   Overrides sparse output location for symlinks
#    S3PO_ROOT=/path/to/S3PO-GS               (default: $CV_PROJ_ROOT/S3PO-GS)
#    GPU_ID=0                                 GPU device id
#    FORCE_RERUN=0                            Set to 1 to overwrite existing results
#
#  Prerequisites:
#    1. conda activate S3PO-GS
#    2. Run subsample_sparse.py to create sparse datasets
#    3. Run add_re10k_parser.py to patch S3PO-GS
#    4. Copy/symlink sparse data into S3PO-GS/datasets/
#
# ================================================================

SCRIPTS_P2="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "${CV_PROJ_ROOT:-"$SCRIPTS_P2/../.."}" && pwd)"

# S3PO-GS location
S3PO_ROOT="${S3PO_ROOT:-$PROJ/S3PO-GS}"
GPU_ID="${GPU_ID:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
DATASET="${1:-all}"

# Output directory for results
RESULTS_DIR="$PROJ/running_result/5_2/S3PO_GS_SLAM"
mkdir -p "$RESULTS_DIR"

# Verify S3PO-GS exists
if [ ! -f "$S3PO_ROOT/slam.py" ]; then
    echo "ERROR: S3PO-GS not found at $S3PO_ROOT"
    echo "Please set S3PO_ROOT or clone S3PO-GS to $S3PO_ROOT"
    echo ""
    echo "Setup instructions:"
    echo "  git clone https://github.com/3DAgentWorld/S3PO-GS.git --recursive $S3PO_ROOT"
    echo "  cd $S3PO_ROOT"
    echo "  conda env create -f environment.yml"
    echo "  conda activate S3PO-GS"
    echo "  pip install submodules/simple-knn"
    echo "  pip install submodules/diff-gaussian-rasterization"
    echo "  cd croco/models/curope/ && python setup.py build_ext --inplace && cd ../../../"
    exit 1
fi

# Check if sparse datasets exist (matches subsample_sparse.py default unless --output_root was used)
SPARSE_DATA="${DATASET_SPARSE_P2:-$PROJ/dataset_sparse_p2}"
if [ ! -d "$SPARSE_DATA" ]; then
    echo "ERROR: Sparse datasets not found at $SPARSE_DATA"
    echo "Run: python scripts/scripts_p2/subsample_sparse.py"
    exit 1
fi

# ============================================================
# Setup: Link sparse data and configs into S3PO-GS
# ============================================================
setup_links() {
    echo "Setting up data and config links in S3PO-GS..."

    # Link sparse datasets
    if [ -d "$SPARSE_DATA/waymo" ] && [ ! -e "$S3PO_ROOT/datasets/waymo_sparse" ]; then
        ln -sf "$SPARSE_DATA/waymo" "$S3PO_ROOT/datasets/waymo_sparse"
        echo "  Linked waymo_sparse"
    fi
    if [ -d "$SPARSE_DATA/dl3dv" ] && [ ! -e "$S3PO_ROOT/datasets/dl3dv_sparse" ]; then
        ln -sf "$SPARSE_DATA/dl3dv" "$S3PO_ROOT/datasets/dl3dv_sparse"
        echo "  Linked dl3dv_sparse"
    fi
    if [ -d "$SPARSE_DATA/re10k" ] && [ ! -e "$S3PO_ROOT/datasets/re10k_sparse" ]; then
        ln -sf "$SPARSE_DATA/re10k" "$S3PO_ROOT/datasets/re10k_sparse"
        echo "  Linked re10k_sparse"
    fi
}

# ============================================================
# Run S3PO-GS on a single dataset
# ============================================================
run_slam() {
    local DATASET_NAME=$1
    local CONFIG=$2
    local RESULT_NAME=$3

    local RESULT_DIR="$RESULTS_DIR/$RESULT_NAME"
    local DONE_FLAG="$RESULT_DIR/.done"

    if [ -f "$DONE_FLAG" ] && [ "$FORCE_RERUN" != "1" ]; then
        echo "  [SKIP] $RESULT_NAME already completed. Set FORCE_RERUN=1 to rerun."
        return 0
    fi

    mkdir -p "$RESULT_DIR"

    echo "  [RUN] $DATASET_NAME -> $RESULT_DIR"
    echo "        Config: $CONFIG"
    echo "        GPU: $GPU_ID"
    echo ""

    cd "$S3PO_ROOT"
    CUDA_VISIBLE_DEVICES=$GPU_ID python slam.py \
        --config "$CONFIG" \
        2>&1 | tee "$RESULT_DIR/slam_log.txt"

    # Copy results from S3PO-GS results dir to our output
    LATEST_RESULT=$(find results/ -maxdepth 2 -name "config.yml" -newer "$RESULT_DIR/slam_log.txt" -printf '%h\n' 2>/dev/null | sort -r | head -1 || true)
    if [ -n "$LATEST_RESULT" ]; then
        echo "  Copying results from $LATEST_RESULT ..."
        cp -r "$LATEST_RESULT"/* "$RESULT_DIR/"
    fi

    touch "$DONE_FLAG"
    echo "  [DONE] $RESULT_NAME"
    cd "$PROJ"
}

# ============================================================
# Main execution
# ============================================================
echo "================================================================"
echo "  Part 2: S3PO-GS Monocular SLAM (Sparse Views)"
echo "================================================================"
echo ""
echo "  S3PO-GS root: $S3PO_ROOT"
echo "  Sparse data:  $SPARSE_DATA"
echo "  Results:      $RESULTS_DIR"
echo "  GPU:          $GPU_ID"
echo ""

setup_links

if [ "$DATASET" = "all" ] || [ "$DATASET" = "waymo" ]; then
    echo ""
    WAYMO_SPARSITY="${WAYMO_SPARSITY:-10}"
    echo "--- Waymo-405841 (1/${WAYMO_SPARSITY} sparsity) ---"
    run_slam "waymo-405841" "configs/mono/waymo_sparse/405841_sparse.yaml" "waymo_405841"
fi

if [ "$DATASET" = "all" ] || [ "$DATASET" = "dl3dv" ]; then
    echo ""
    echo "--- DL3DV-2 (1/30 sparsity) ---"
    run_slam "dl3dv-2" "configs/mono/dl3dv_sparse/2_sparse.yaml" "dl3dv_2"
fi

if [ "$DATASET" = "all" ] || [ "$DATASET" = "re10k" ]; then
    echo ""
    echo "--- Re10k-1 (1/30 sparsity) ---"
    run_slam "re10k-1" "configs/mono/re10k_sparse/1_sparse.yaml" "re10k_1"
fi

echo ""
echo "================================================================"
echo "  All S3PO-GS runs complete!"
echo "  Results at: $RESULTS_DIR"
echo ""
echo "  Next: python scripts/scripts_p2/evaluate_p2.py"
echo "================================================================"
